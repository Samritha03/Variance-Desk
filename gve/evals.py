"""Evaluation harness for the investigating agent.

Scores an investigator against data/corpus_key.json, which says which documents
truly explain each variance and which line has no explanation at all.

Metrics
  right document      of the lines that DO have an explanation, how many got a
                      supported finding citing at least one correct document
  cause stated        stricter: the right document AND the explanation or quoted text
                      actually contains the cause (keywords from the answer key). Citing
                      an email header from the right email does not count.
  citation precision  of the documents cited on those lines, how many were correct
  false attribution   of the lines with NO explanation, how many got a supported
                      finding anyway (the failure that matters most)
  groundedness        of all findings that claimed a cause, how many passed the
                      quote-and-figure check
  tool calls          average per line (cost proxy)

Run:
  python -m gve.evals                    # keyword baseline, date filter on and off
  python -m gve.evals --llm              # also run the LLM agent, hybrid search (ANTHROPIC_API_KEY or GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import ClaudeInvestigator, KeywordBaseline, run_investigation
from .commentary import build_facts
from .loader import load_coa, load_gl
from .retrieval import Retriever
from .tools import load_ledger
from .variance import compute_variances, prepare_for_analysis, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def material_lines(month: str, min_dollars: float = 5_000, min_pct: float = 5.0) -> list[dict]:
    gl, coa = load_gl(DATA / "sample_gl.csv"), load_coa(DATA / "chart_of_accounts.csv")
    clean = prepare_for_analysis(gl, coa)
    var = compute_variances(clean, month, min_dollars, min_pct)
    return build_facts(var, summarize(clean, month), {})["lines"]


def _norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def score(findings: dict, gold_lines: dict) -> dict:
    sup_total = sup_correct = doc_hit = 0
    cited = cited_ok = 0
    ne_total = ne_false = 0
    claimed = grounded = 0
    calls = []
    detail = {}
    for key, f in findings.items():
        g = gold_lines.get(key)
        calls.append(len(f.trace))
        if f.status in ("supported", "unverified") and f.evidence:
            claimed += 1
            grounded += f.status == "supported"
        if g is None:
            continue
        docs = [s for s in f.sources if s.startswith("D")]
        if g["expected"] == "supported":
            sup_total += 1
            hit = f.status == "supported" and any(d in g["gold_docs"] for d in docs)
            text = _norm_text(" ".join([f.explanation] + [e.get("quote", "") for e in f.evidence]))
            says_cause = hit and (not g.get("keywords") or any(k in text for k in g["keywords"]))
            doc_hit += hit
            sup_correct += says_cause
            if f.status == "supported":
                cited += len(docs)
                cited_ok += sum(d in g["gold_docs"] for d in docs)
            detail[key] = ("cause stated" if says_cause else
                           "right document, cause not stated" if hit else f"missed ({f.status})")
        else:
            ne_total += 1
            bad = f.status == "supported"
            ne_false += bad
            detail[key] = "FALSE ATTRIBUTION" if bad else "correctly said no evidence"
    pct = lambda a, b: (a / b if b else float("nan"))
    return {
        "cause_found": pct(sup_correct, sup_total), "cause_found_n": f"{sup_correct}/{sup_total}",
        "doc_hit_n": f"{doc_hit}/{sup_total}",
        "citation_precision": pct(cited_ok, cited), "citation_n": f"{cited_ok}/{cited}",
        "false_attribution": pct(ne_false, ne_total), "false_attribution_n": f"{ne_false}/{ne_total}",
        "groundedness": pct(grounded, claimed), "groundedness_n": f"{grounded}/{claimed}",
        "avg_tool_calls": sum(calls) / max(len(calls), 1),
        "per_line": detail,
    }


# Paraphrased queries: an agent will not phrase a search the way the answer key does.
PARAPHRASES = {
    "Marketing|6200": ["why was event spend high in September", "conference exhibit costs and attendee travel",
                       "who approved the retail show", "booth shipping and exhibitor fees"],
    "Sales|4000": ["revenue shortfall deals delayed", "contracts pushed to next month", "customers postponed signing",
                   "why did bookings miss plan"],
    "Engineering|6500": ["cloud hosting bill increase", "new data platform launch costs", "infrastructure usage went up",
                         "compute charges jumped"],
    "Operations|6300": ["shipping costs higher than planned", "rush deliveries carrier extra charges", "freight overage",
                        "transport bill increase"],
    "G&A|6800": ["external accountant fees", "annual financial statement review billing", "professional services invoice",
                 "auditor charges"],
}


def retrieval_scores(retriever, gold: dict, month: str, k: int = 4, date_filter: bool = True) -> list[dict]:
    """Retrieval alone. For each retrieval mode, over the gold query and several paraphrases per line:
    recall = share of (query, gold document) pairs found in the top k;
    stale hits = wrong-period documents (last year's invoice, an old memo) returned in the top k."""
    from .retrieval import window_for_month
    stale = set(gold.get("out_of_period_docs", []))
    lo, hi = window_for_month(month) if date_filter else (None, None)
    out = []
    for mode in ("bm25", "dense", "hybrid"):
        for qset, name in (("gold", "key query"), ("para", "paraphrased")):
            found = total = stale_hits = 0
            for key, g in gold["lines"].items():
                if not g.get("gold_docs"):
                    continue
                qs = [g.get("query") or " ".join(g.get("keywords", [])[:3])] if qset == "gold" else PARAPHRASES.get(key, [])
                for q in qs:
                    ids = {h["doc_id"] for h in retriever.search(q, k=k, date_from=lo, date_to=hi, mode=mode)}
                    found += len(ids & set(g["gold_docs"]))
                    total += len(g["gold_docs"])
                    stale_hits += len(ids & stale)
            out.append({"config": f"{mode}, {name}", "mode": mode, "queries": name,
                        "recall": f"{found}/{total}", "recall_pct": found / total if total else 0.0, "stale": stale_hits})
    return out


def run_config(name: str, investigator, date_filter: bool, month: str, retriever, ledger, gold) -> dict:
    from .app_helpers import month_label
    lines = material_lines(month)
    findings = run_investigation(lines, month, month_label(month), investigator, retriever, ledger, date_filter)
    s = score(findings, gold["lines"])
    s["config"] = name
    s["findings"] = findings
    return s


def format_table(results: list[dict]) -> str:
    rows = ["| Config | Right document | Cause stated | Citation precision | False attribution | Groundedness | Avg tool calls |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['config']} | {r['doc_hit_n']} | {r['cause_found_n']} | {r['citation_n']} | {r['false_attribution_n']} | "
                    f"{r['groundedness_n']} | {r['avg_tool_calls']:.1f} |")
    out = "\n".join(rows) + "\n"
    for r in results:
        out += f"\n{r['config']}:\n" + "\n".join(f"  {k}: {v}" for k, v in r["per_line"].items())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", "--llm", dest="claude", action="store_true", help="also evaluate the LLM agent (Claude or Gemini, whichever key is set)")
    ap.add_argument("--month", default="2026-09")
    a = ap.parse_args(argv)
    gold = json.loads((DATA / "corpus_key.json").read_text())
    retriever, ledger = Retriever(DATA / "docs"), load_ledger(DATA / "transactions.csv")
    results = []
    import copy
    for mode in ("bm25", "hybrid"):
        r = copy.copy(retriever); r.mode = mode
        for df in (True, False):
            tag = "date filter on" if df else "date filter off"
            results.append(run_config(f"baseline, {mode} search, {tag}", KeywordBaseline(), df, a.month, r, ledger, gold))
    if a.claude:
        from .llm import provider
        if not provider():
            raise SystemExit("--llm needs ANTHROPIC_API_KEY or GEMINI_API_KEY set.")
        for df in (True, False):
            tag = "date filter on" if df else "date filter off"
            r = copy.copy(retriever); r.mode = "hybrid"
            inv = ClaudeInvestigator()
            results.append(run_config(f"{inv.name}, hybrid search, {tag}", inv, df, a.month, r, ledger, gold))
    print(format_table(results))
    print("\nRetrieval alone, top-4, date filter on:")
    print("| Config | Gold docs found | Wrong-period docs in results |\n|---|---|---|")
    for r in retrieval_scores(retriever, gold, a.month):
        print(f"| {r['config']} | {r['recall']} | {r['stale']} |")
    off = [r for r in retrieval_scores(retriever, gold, a.month, date_filter=False) if r["mode"] == "bm25" and r["queries"] == "key query"][0]
    print(f"\nSame keyword query with the date filter OFF: {off['recall']} found, {off['stale']} wrong-period docs in results.")
    return results


if __name__ == "__main__":
    main()
