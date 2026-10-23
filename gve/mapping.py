"""Chart-of-accounts mapping: propose where each of an entity's accounts belongs in a standard chart.

The risky failure is not "wrong suggestion", it is "wrong suggestion applied without anyone looking". So the
output is a proposal plus a status: auto (confident enough to apply) or review (a person decides). Approved
decisions become rules that win on the next run. Accounts with no home in the standard chart should go to
review, not get forced into the nearest bucket.

Two mappers share one interface:
  SimilarityMapper   no model. TF-IDF similarity between account names/descriptions. The floor.
  ClaudeMapper       one model call per entity, then every answer is validated in code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "mapping"
AUTO_SCORE, AUTO_MARGIN, NO_MATCH = 0.45, 0.10, 0.20   # untuned defaults for the similarity mapper


def load_standard() -> list[dict]:
    return list(csv.DictReader(open(DATA / "standard_coa.csv")))


def load_entities() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(DATA / "entity_accounts.csv")):
        out.setdefault(r["entity"], []).append(r)
    return out


def load_key() -> dict:
    return json.loads((DATA / "mapping_key.json").read_text())


def _row(entity: str, a: dict, target: dict | None, status: str, confidence: str, reason: str, method: str) -> dict:
    return {"entity": entity, "source_code": a["code"], "source_name": a["name"], "type": a["type"],
            "target_code": target["code"] if target else None, "target_name": target["name"] if target else None,
            "status": status, "confidence": confidence, "reason": reason, "method": method}


class SimilarityMapper:
    name = "similarity-baseline"

    def __init__(self, standard: list[dict] | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.std = standard or load_standard()
        texts = [f"{s['name']} {s['description']}" for s in self.std]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts)
        self.M = self.vec.transform(texts)

    def map_entity(self, entity: str, accounts: list[dict]) -> list[dict]:
        out = []
        for a in accounts:
            sims = (self.vec.transform([f"{a['name']} {a.get('description', '')}"]) @ self.M.T).toarray()[0]
            ok = sorted(((float(s), i) for i, s in enumerate(sims) if self.std[i]["type"] == a["type"]), reverse=True)
            (s1, i1), s2 = ok[0], (ok[1][0] if len(ok) > 1 else 0.0)
            t = self.std[i1]
            if s1 < NO_MATCH:
                out.append(_row(entity, a, None, "review", "low", f"Nothing in the standard chart looks similar (best {s1:.2f}).", self.name))
            else:
                status = "auto" if s1 >= AUTO_SCORE and s1 - s2 >= AUTO_MARGIN else "review"
                conf = "high" if status == "auto" else ("medium" if s1 >= AUTO_SCORE else "low")
                out.append(_row(entity, a, t, status, conf, f"Name similarity {s1:.2f} to “{t['name']}”; next best {s2:.2f}.", self.name))
        return out


SYSTEM = """You map one company's chart of accounts into a standard chart, for a finance team.

For every source account, choose the single best standard account, or NONE.

Rules:
- Use the account's name, description, type and where it sits among the entity's other accounts. Do NOT assume the source code means the same as the same code in the standard chart; numbering differs between companies.
- The source type (revenue or expense) must match the standard account's type.
- If no standard account is a fair home (for example intercompany charges, suspense or clearing accounts, contra-revenue, equity-based pay, or a vague catch-all like "Misc"), answer NONE. Forcing a bad match is worse than asking a person.
- confidence "high" only when you would be surprised if an accountant disagreed. Use "medium" or "low" when the name is ambiguous or you are translating.
- Give a short reason a reviewer can check (one sentence)."""

TOOL = {
    "name": "propose_mappings",
    "description": "Submit the mapping for every source account in this entity. Call exactly once.",
    "input_schema": {"type": "object", "properties": {"mappings": {"type": "array", "items": {"type": "object", "properties": {
        "source_code": {"type": "string"},
        "target_code": {"type": "string", "description": "A standard account code, or NONE"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"}},
        "required": ["source_code", "target_code", "confidence", "reason"]}}}, "required": ["mappings"]},
}


class ClaudeMapper:
    def __init__(self, llm=None, standard: list[dict] | None = None):
        self.llm, self.std = llm, standard or load_standard()

    @property
    def name(self) -> str:
        from .llm import agent_name
        return agent_name(self.llm).replace("-agent", "-mapper")

    def _prompt(self, entity: str, context: list[dict], batch: list[dict]) -> str:
        std = "\n".join(f"{s['code']} | {s['name']} | {s['type']} | {s['description']}" for s in self.std)
        fmt = lambda xs: "\n".join(f"{a['code']} | {a['name']} | {a['type']} | {a.get('description', '')}" for a in xs)
        return (f"Standard chart (code | name | type | description):\n{std}\n\nEntity: {entity}\n"
                f"All of this entity's accounts, for context (code | name | type | description):\n{fmt(context)}\n\n"
                f"Map ONLY these accounts, one entry each:\n{fmt(batch)}")

    BATCH, TRIES = 12, 3

    def map_entity(self, entity: str, accounts: list[dict]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(accounts), self.BATCH):
            out += self._map_batch(entity, accounts, accounts[i:i + self.BATCH])
        return out

    def _ask(self, entity: str, context: list[dict], batch: list[dict]) -> tuple[dict, str]:
        """Up to TRIES attempts; returns (answers by source code, last text the model said)."""
        from .llm import make_llm
        llm = self.llm or make_llm()
        said = ""
        for _ in range(self.TRIES):
            resp = llm.create(SYSTEM, [{"role": "user", "content": self._prompt(entity, context, batch)}], [TOOL])
            got = next((dict(b.input) for b in resp.content if getattr(b, "type", "") == "tool_use" and b.name == "propose_mappings"), {})
            said = " ".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")[:120]
            by_src = {str(m.get("source_code")): m for m in got.get("mappings", [])}
            if by_src:
                return by_src, said
        return {}, said

    def _map_batch(self, entity: str, context: list[dict], accounts: list[dict]) -> list[dict]:
        by_src, said = self._ask(entity, context, accounts)
        std = {s["code"]: s for s in self.std}
        out = []
        for a in accounts:
            m = by_src.get(a["code"])
            if not m:
                out.append(_row(entity, a, None, "review", "low", f"The model gave no answer for this account. {said}".strip(), self.name)); continue
            code, conf, why = str(m.get("target_code", "")).strip(), m.get("confidence", "low"), str(m.get("reason", "")).strip()
            if code == "NONE":
                out.append(_row(entity, a, None, "review", conf if conf in ("high", "medium", "low") else "low", f"No fair home. {why}", self.name))
            elif code not in std:
                out.append(_row(entity, a, None, "review", "low", f"The model named “{code}”, which is not in the standard chart.", self.name))
            elif std[code]["type"] != a["type"]:
                out.append(_row(entity, a, None, "review", "low", f"Proposed {code} is a {std[code]['type']} account but this is {a['type']}.", self.name))
            else:
                status = "auto" if conf == "high" else "review"
                out.append(_row(entity, a, std[code], status, conf if conf in ("high", "medium", "low") else "low", why, self.name))
        return out


def apply_rules(rows: list[dict], rules: list[dict], standard: list[dict]) -> list[dict]:
    """Approved decisions win over any proposal. A rule is {entity, source_code, target_code}."""
    std = {s["code"]: s for s in standard}
    r = {(x["entity"], x["source_code"]): x["target_code"] for x in rules}
    out = []
    for row in rows:
        t = r.get((row["entity"], row["source_code"]))
        if t in std:
            row = dict(row, target_code=t, target_name=std[t]["name"], status="rule", confidence="approved",
                       reason="Approved by a reviewer.", method="rule")
        out.append(row)
    return out


def evaluate(rows: list[dict], key: dict) -> dict:
    """Score proposals against the answer key. A wrong proposal in review is cheap; a wrong one applied automatically is not."""
    n = len(rows)
    ok = wrong_auto = auto = review = wrong = caught = 0
    tags: dict[str, list[int]] = {}
    for r in rows:
        g = key[f"{r['entity']}|{r['source_code']}"]
        correct = (r["target_code"] or "NONE") == g["target"]
        ok += correct
        t = tags.setdefault(g["tag"], [0, 0]); t[1] += 1; t[0] += correct
        if r["status"] == "auto":
            auto += 1; wrong_auto += (not correct)
        else:
            review += 1
        if not correct:
            wrong += 1; caught += r["status"] != "auto"
    return {"n": n, "correct": ok, "accuracy": ok / n, "auto": auto, "auto_correct": auto - wrong_auto,
            "auto_precision": (auto - wrong_auto) / auto if auto else None, "wrong_auto": wrong_auto,
            "review": review, "wrong": wrong, "wrong_caught": caught,
            "by_tag": {k: {"correct": v[0], "n": v[1]} for k, v in sorted(tags.items())}}


def run_all(mapper, entities: dict[str, list[dict]] | None = None) -> list[dict]:
    rows: list[dict] = []
    for ent, accts in (entities or load_entities()).items():
        rows += mapper.map_entity(ent, accts)
    return rows


def summary(include_claude: bool = False, keep_rows: bool = False) -> dict:
    key, std = load_key(), load_standard()
    base_rows = run_all(SimilarityMapper(std))
    res = [{"mapper": "similarity-baseline", **evaluate(base_rows, key)}]
    if include_claude:
        mapper = ClaudeMapper(standard=std)
        rows = run_all(mapper)
        res.append({"mapper": mapper.name, **evaluate(rows, key), **({"rows": rows} if keep_rows else {})})
    return {"accounts": len(key), "no_home": sum(g["target"] == "NONE" for g in key.values()),
            "entities": len(load_entities()), "standard": len(std), "results": res}


def main():
    import os, sys
    from .llm import provider
    claude = "--claude" in sys.argv or "--llm" in sys.argv
    if claude and not provider():
        raise SystemExit("--llm needs ANTHROPIC_API_KEY or GEMINI_API_KEY set.")
    s = summary(claude, keep_rows="--show" in sys.argv)
    print(f"{s['accounts']} accounts across {s['entities']} entities ({s['no_home']} with no home in the standard chart)\n")
    print("| Mapper | Accuracy | Auto-applied | Auto precision | Wrong & auto-applied | Wrong, caught by review |\n|---|---|---|---|---|---|")
    for r in s["results"]:
        p = "n/a" if r["auto_precision"] is None else f"{r['auto_correct']}/{r['auto']}"
        print(f"| {r['mapper']} | {r['correct']}/{r['n']} | {r['auto']} | {p} | {r['wrong_auto']} | {r['wrong_caught']}/{r['wrong']} |")
    if "--show" in sys.argv and claude:
        key = load_key()
        print("\nWrong proposals from the AI mapper (status, what it said, why):")
        for r in s["results"][-1]["rows"]:
            g = key[f"{r['entity']}|{r['source_code']}"]["target"]
            if (r["target_code"] or "NONE") != g:
                print(f"- {r['entity'].split()[0]} {r['source_code']} {r['source_name']}: said {r['target_code']} (right: {g}), {r['status']}. {r['reason'][:110]}")
    for r in s["results"]:
        print(f"\n{r['mapper']} by kind: " + ", ".join(f"{k} {v['correct']}/{v['n']}" for k, v in r["by_tag"].items()))


if __name__ == "__main__":
    main()
