"""Benchmark: 36 independent cases (see data/bench/generate_bench.py).

Retrieval metrics, per search mode, over each case's key query and three paraphrases:
  recall@k        share of the true documents found in the top k
  MRR             1 / rank of the first true document (0 if none in the top k)
  decoys@k        decoys returned in the top k per query (last year's version, unapproved proposal, quote-only estimate)

Agent metrics, end to end (same definitions as gve.evals.score) plus
  trap cited      findings that cite a decoy document

Run:
  python -m gve.bench                  # retrieval modes + keyword baseline
  python -m gve.bench --llm            # also the LLM agent (ANTHROPIC_API_KEY or GEMINI_API_KEY), hybrid search
  python -m gve.bench --llm --limit 10 # ...on 10 cases only (cheaper, or to stay inside a free tier)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path

from .agent import ClaudeInvestigator, KeywordBaseline, run_investigation
from .app_helpers import month_label
from .evals import score
from .retrieval import Retriever, window_for_month
from .tools import load_ledger

BENCH = Path(__file__).resolve().parents[1] / "data" / "bench"


def load_bench() -> dict:
    return json.loads((BENCH / "bench_key.json").read_text())


def make_retriever(mode: str = "hybrid") -> Retriever:
    return Retriever(BENCH / "docs", mode=mode)


def _traps(case: dict) -> set[str]:
    return set(case.get("decoy_docs", [])) | set(case.get("stale_docs", []))


def retrieval_table(retriever: Retriever, bench: dict, k: int = 4, date_filter: bool = True) -> list[dict]:
    rows = []
    for mode in ("bm25", "dense", "hybrid"):
        for qset in ("key", "paraphrased"):
            found = total = decoys = nq = 0
            rr = []
            for c in bench["cases"]:
                if not c["gold_docs"]:
                    continue
                lo, hi = window_for_month(c["month"]) if date_filter else (None, None)
                qs = c["queries"][:1] if qset == "key" else c["queries"][1:]
                for q in qs:
                    hits = retriever.search(q, k=k, date_from=lo, date_to=hi, mode=mode)
                    ids = [h["doc_id"] for h in hits]
                    gold = set(c["gold_docs"])
                    found += len(set(ids) & gold)
                    total += len(gold)
                    decoys += len(set(ids) & _traps(c))
                    first = next((i for i, d in enumerate(ids, 1) if d in gold), None)
                    rr.append(1 / first if first else 0.0)
                    nq += 1
            rows.append({"mode": mode, "queries": qset, "n_queries": nq, "recall": found / total,
                         "recall_n": f"{found}/{total}", "mrr": sum(rr) / len(rr), "decoys_per_query": decoys / nq})
    return rows


def _frac(s: str) -> tuple[int, int]:
    a, b = s.split("/")
    return int(a), int(b)


def agent_run(investigator, retriever: Retriever, ledger, bench: dict, mode: str, date_filter: bool = True,
              on_progress=None) -> dict:
    r = copy.copy(retriever)
    r.mode = mode
    by_month = defaultdict(list)
    for c in bench["cases"]:
        by_month[c["month"]].append(c)
    keys = ["doc_hit_n", "cause_found_n", "citation_n", "false_attribution_n", "groundedness_n"]
    agg = {k: [0, 0] for k in keys}
    calls, per_case, trap_cited, claimed, done = [], {}, 0, 0, 0
    for month, cases in sorted(by_month.items()):
        lines = [dict(c["line"]) for c in cases]
        findings = run_investigation(lines, month, month_label(month), investigator, r, ledger, date_filter)
        gold = {f"{c['line']['department']}|{c['line']['account_code']}":
                {"expected": c["expected"], "gold_docs": c["gold_docs"], "keywords": c["keywords"]} for c in cases}
        s = score(findings, gold)
        for k in keys:
            a, b = _frac(s[k])
            agg[k][0] += a
            agg[k][1] += b
        calls += [len(f.trace) for f in findings.values()]
        for c in cases:
            key = f"{c['line']['department']}|{c['line']['account_code']}"
            f = findings[key]
            per_case[c["id"]] = s["per_line"].get(key, "")
            if f.status in ("supported", "unverified") and f.evidence:
                claimed += 1
                trap_cited += any(x in _traps(c) for x in f.sources)
            done += 1
            if on_progress:
                on_progress(done, len(bench["cases"]))
    out = {k: f"{a}/{b}" for k, (a, b) in agg.items()}
    out.update(avg_tool_calls=sum(calls) / max(len(calls), 1), trap_cited_n=f"{trap_cited}/{claimed}", per_case=per_case,
               config=f"{investigator.name}, {mode} search, date filter {'on' if date_filter else 'off'}")
    return out


def subset(bench: dict, n: int) -> dict:
    """n cases spread evenly over the list, so the mix of explained and no-evidence cases is kept."""
    cases = bench["cases"]
    if n >= len(cases):
        return bench
    idx = sorted({round(i * len(cases) / n) for i in range(n)})
    return dict(bench, cases=[cases[i] for i in idx])


def summary(include_claude: bool = False, limit: int | None = None, on_progress=None) -> dict:
    bench = load_bench()
    retriever, ledger = make_retriever(), load_ledger(BENCH / "transactions.csv")
    agents = []
    for mode in ("bm25", "hybrid"):
        agents.append(agent_run(KeywordBaseline(), retriever, ledger, bench, mode))
    if include_claude:
        sub = subset(bench, limit) if limit else bench
        inv = ClaudeInvestigator()
        r = agent_run(inv, retriever, ledger, sub, "hybrid", on_progress=on_progress)
        if limit and len(sub["cases"]) < len(bench["cases"]):
            r["config"] += f" ({len(sub['cases'])} of {len(bench['cases'])} cases)"
        agents.append(r)
    n = len(bench["cases"])
    return {"cases": n, "explained": sum(c["expected"] == "supported" for c in bench["cases"]),
            "no_evidence": sum(c["expected"] == "no_evidence" for c in bench["cases"]),
            "documents": len(retriever.docs), "embedder": retriever.embedder_name,
            "retrieval": retrieval_table(retriever, bench),
            "retrieval_no_date_filter": retrieval_table(retriever, bench, date_filter=False),
            "agents": agents}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", "--llm", dest="claude", action="store_true",
                    help="also run the LLM agent (Claude or Gemini, whichever key is set), hybrid search only")
    ap.add_argument("--limit", type=int, default=None, help="run the LLM agent on only N cases, spread evenly")
    a = ap.parse_args(argv)
    from .llm import provider
    if a.claude and not provider():
        raise SystemExit("--llm needs ANTHROPIC_API_KEY or GEMINI_API_KEY set.")
    s = summary(a.claude, a.limit, on_progress=lambda d, t: print(f"  agent: {d}/{t} cases", end="\r", flush=True) if a.claude else None)
    print(f"{s['cases']} cases ({s['explained']} explained, {s['no_evidence']} with no real explanation), "
          f"{s['documents']} documents, embedder: {s['embedder']}\n")
    for title, rows in (("date filter ON", s["retrieval"]), ("date filter OFF", s["retrieval_no_date_filter"])):
        print(f"Retrieval, top-4, {title}\n| Search | Queries | Recall@4 | MRR | Decoys / query |\n|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['mode']} | {r['queries']} ({r['n_queries']}) | {r['recall_n']} ({r['recall']:.0%}) | {r['mrr']:.2f} | {r['decoys_per_query']:.2f} |")
        print()
    print("Agent, end to end\n| Setup | Right doc | Cause stated | Citation precision | False attribution | Grounded | Trap cited | Tool calls |\n|---|---|---|---|---|---|---|---|")
    for g in s["agents"]:
        print(f"| {g['config']} | {g['doc_hit_n']} | {g['cause_found_n']} | {g['citation_n']} | {g['false_attribution_n']} | "
              f"{g['groundedness_n']} | {g['trap_cited_n']} | {g['avg_tool_calls']:.1f} |")


if __name__ == "__main__":
    main()
