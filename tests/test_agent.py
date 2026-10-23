"""Tests for retrieval, tools, the agent loop, grounding and the eval harness.
No API key needed: the model is replaced by a scripted fake."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gve import build_facts, compute_variances, draft_commentary, load_coa, load_gl, prepare_for_analysis, validate
from gve.agent import ClaudeInvestigator, Finding, KeywordBaseline, ground, run_investigation
from gve.commentary import inject_errors
from gve.evals import material_lines, retrieval_scores, score
from gve.retrieval import Retriever, window_for_month
from gve.tools import AgentTools, load_ledger
from gve.variance import summarize

DATA = ROOT / "data"
MONTH = "2026-09"


@pytest.fixture(scope="module")
def env():
    import json
    return dict(retriever=Retriever(DATA / "docs"), ledger=load_ledger(DATA / "transactions.csv"),
                gold=json.loads((DATA / "corpus_key.json").read_text()),
                lines={f"{l['department']}|{l['account_code']}": l for l in material_lines(MONTH)})


def tools_for(env, date_filter=True):
    return AgentTools(env["retriever"], env["ledger"], MONTH, date_filter)


# ---------------------------------------------------------------- data
def test_transactions_reconcile_to_gl():
    gl = load_gl(DATA / "sample_gl.csv")
    coa = load_coa(DATA / "chart_of_accounts.csv")
    clean = prepare_for_analysis(gl, coa)
    t = load_ledger(DATA / "transactions.csv").groupby(["month", "department", "account_code"]).amount.sum().round(2)
    g = clean.groupby(["month", "department", "account_code"]).actual.sum().round(2)
    assert (t - g).abs().max() < 0.011


# ---------------------------------------------------------------- retrieval
def test_search_finds_the_audit_invoice(env):
    ids = [h["doc_id"] for h in env["retriever"].search("audit fee invoice", k=3)]
    assert "D12" in ids


def test_date_filter_removes_last_years_invoice(env):
    lo, hi = window_for_month(MONTH)
    with_filter = {h["doc_id"] for h in env["retriever"].search("audit fee invoice", k=6, date_from=lo, date_to=hi)}
    without = {h["doc_id"] for h in env["retriever"].search("audit fee invoice", k=6)}
    assert "D14" in without and "D14" not in with_filter


def test_window_covers_pre_and_post_month_evidence():
    lo, hi = window_for_month(MONTH)
    assert lo <= "2026-07-14" and "2026-10-02" <= hi  # engagement letter and AWS statement


# ---------------------------------------------------------------- tools
def test_get_transactions_total_and_logging(env):
    t = tools_for(env)
    res = t.get_transactions("G&A", "6800", MONTH)
    assert res["count"] == 3 and abs(res["total"] - 49542.72) < 0.01
    assert res["transactions"][0]["reference"] == "WC-4471"
    assert res["transactions"][0]["txn_id"] in t.seen_txns


def test_bad_tool_arguments_do_not_crash(env):
    t = tools_for(env)
    assert "error" in t.dispatch("get_transactions", {"department": "Nope", "account_code": "1", "month": MONTH})
    assert "Unknown tool" in t.dispatch("delete_everything", {})
    assert "error" in t.dispatch("search_documents", {"query": "x", "top_k": "many"})


# ---------------------------------------------------------------- grounding
def retrieved_audit_tools(env):
    t = tools_for(env)
    t.dispatch("search_documents", {"query": "audit fee invoice"})
    t.dispatch("get_transactions", {"department": "G&A", "account_code": "6800", "month": MONTH})
    return t


def finding(evidence, text="The annual audit fee of $34.5K was billed in full in September."):
    return Finding("G&A|6800", "test", "supported", text, "high", evidence)


def test_grounded_finding_passes(env):
    t = retrieved_audit_tools(env)
    f = ground(finding([{"source_id": "D12", "quote": "Amount due: $34,542.72. Billed in full on completion of fieldwork."}]),
               t, env["lines"]["G&A|6800"])
    assert f.status == "supported" and f.problems == []


def test_invented_quote_is_downgraded(env):
    t = retrieved_audit_tools(env)
    f = ground(finding([{"source_id": "D12", "quote": "The audit was a one-time special review."}]), t, env["lines"]["G&A|6800"])
    assert f.status == "unverified" and any("word for word" in p for p in f.problems)


def test_citing_a_document_never_retrieved_is_downgraded(env):
    t = retrieved_audit_tools(env)
    f = ground(finding([{"source_id": "D99", "quote": "anything"}]), t, env["lines"]["G&A|6800"])
    assert f.status == "unverified" and any("never retrieved" in p for p in f.problems)


def test_invented_figure_is_downgraded(env):
    t = retrieved_audit_tools(env)
    f = ground(finding([{"source_id": "D12", "quote": "Billed in full on completion of fieldwork."}],
                       "The audit fee of $52K was billed in September."), t, env["lines"]["G&A|6800"])
    assert f.status == "unverified" and any("$52,000" in p for p in f.problems)


def test_transaction_citation_needs_no_quote(env):
    t = retrieved_audit_tools(env)
    tid = next(k for k, v in t.seen_txns.items() if v["reference"] == "WC-4471")
    f = ground(finding([{"source_id": tid}]), t, env["lines"]["G&A|6800"])
    assert f.status == "supported"


def test_supported_without_evidence_is_downgraded(env):
    f = ground(finding([]), retrieved_audit_tools(env), env["lines"]["G&A|6800"])
    assert f.status == "unverified"


# ---------------------------------------------------------------- agent loop (fake model)
def tu(i, name, **inp):
    return SimpleNamespace(type="tool_use", id=i, name=name, input=inp)


class FakeLLM:
    def __init__(self, script):
        self.script, self.seen_tool_results = list(script), []

    def create(self, system, messages, tools):
        last = messages[-1]["content"]
        if isinstance(last, list):
            self.seen_tool_results.append(last)
        return SimpleNamespace(content=self.script.pop(0))


def run_fake(env, script, key="G&A|6800", max_steps=8):
    llm = FakeLLM(script)
    out = run_investigation([env["lines"][key]], MONTH, "September 2026", ClaudeInvestigator(llm, max_steps),
                            env["retriever"], env["ledger"])
    return out[key], llm


GOOD_FINDING = dict(status="supported", confidence="high",
                    explanation="The annual audit fee of $34.5K was billed in full in September.",
                    evidence=[{"source_id": "D12", "quote": "Annual financial statement audit, fiscal year 2026: fieldwork and report."},
                              {"source_id": "D13", "quote": "invoiced in a single amount on completion of fieldwork, expected in September 2026"}])


def test_agent_loop_end_to_end(env):
    f, llm = run_fake(env, [
        [tu("a", "get_transactions", department="G&A", account_code="6800", month=MONTH)],
        [tu("b", "search_documents", query="audit fee invoice")],
        [tu("c", "record_finding", **GOOD_FINDING)]])
    assert f.status == "supported" and f.problems == [] and f.steps == 3
    assert [t["tool"] for t in f.trace] == ["get_transactions", "search_documents"]
    assert f.sources == ["D12", "D13"]
    assert "WC-4471" in llm.seen_tool_results[0][0]["content"]  # real tool output went back to the model


def test_agent_that_hallucinates_a_quote_is_caught(env):
    bad = dict(GOOD_FINDING, evidence=[{"source_id": "D12", "quote": "Audit fees were prepaid last quarter."}])
    f, _ = run_fake(env, [[tu("b", "search_documents", query="audit fee invoice")], [tu("c", "record_finding", **bad)]])
    assert f.status == "unverified"


def test_agent_citing_a_document_it_never_searched_for_is_caught(env):
    f, _ = run_fake(env, [[tu("c", "record_finding", **GOOD_FINDING)]])  # no retrieval at all
    assert f.status == "unverified" and len(f.problems) == 2


def test_agent_nudged_when_it_forgets_to_record(env):
    f, llm = run_fake(env, [
        [SimpleNamespace(type="text", text="I think it was the audit.")],
        [tu("b", "search_documents", query="audit fee invoice")],
        [tu("c", "record_finding", **GOOD_FINDING)]])
    assert f.status == "supported"


def test_agent_that_never_concludes_is_unverified(env):
    f, _ = run_fake(env, [[SimpleNamespace(type="text", text="hmm")]] * 3, max_steps=3)
    assert f.status == "unverified" and "step limit" in f.problems[0]


def test_agent_can_say_no_evidence(env):
    f, _ = run_fake(env, [
        [tu("a", "search_documents", query="engineering salaries")],
        [tu("c", "record_finding", status="no_evidence", confidence="medium",
            explanation="Nothing in the records explains this increase.", evidence=[])]], key="Engineering|6000")
    assert f.status == "no_evidence" and f.problems == []


# ---------------------------------------------------------------- baseline and eval
def test_baseline_findings_are_always_grounded(env):
    out = run_investigation(list(env["lines"].values()), MONTH, "September 2026", KeywordBaseline(),
                            env["retriever"], env["ledger"])
    assert all(f.status != "unverified" for f in out.values())


def test_baseline_does_not_blame_the_unexplained_line(env):
    out = run_investigation(list(env["lines"].values()), MONTH, "September 2026", KeywordBaseline(),
                            env["retriever"], env["ledger"])
    assert out["Engineering|6000"].status == "no_evidence"


def test_score_counts_false_attribution_and_misses():
    gold = {"A|1": {"expected": "supported", "gold_docs": ["D01"]},
            "B|2": {"expected": "supported", "gold_docs": ["D02"]},
            "C|3": {"expected": "no_evidence", "gold_docs": []}}
    mk = lambda k, status, ev: Finding(k, "t", status, "x", "low", [{"source_id": e} for e in ev])
    s = score({"A|1": mk("A|1", "supported", ["D01", "D09"]), "B|2": mk("B|2", "unverified", ["D02"]),
               "C|3": mk("C|3", "supported", ["D16"])}, gold)
    # precision counts only supported findings on lines that have a real explanation: D01 right, D09 wrong
    assert s["cause_found_n"] == "1/2" and s["citation_n"] == "1/2"
    assert s["false_attribution_n"] == "1/1" and s["groundedness_n"] == "2/3"


def test_date_filter_reduces_stale_retrieval(env):
    pick = lambda rows: [r for r in rows if r["config"] == "bm25, key query"][0]
    on = pick(retrieval_scores(env["retriever"], env["gold"], MONTH))
    off = pick(retrieval_scores(env["retriever"], env["gold"], MONTH, date_filter=False))
    assert on["stale"] == 0 and off["stale"] >= 1


def test_hybrid_beats_keyword_on_paraphrased_queries(env):
    rows = {r["config"]: r["recall_pct"] for r in retrieval_scores(env["retriever"], env["gold"], MONTH)}
    assert rows["hybrid, paraphrased"] > rows["bm25, paraphrased"]
    assert rows["hybrid, key query"] >= rows["bm25, key query"]


# ---------------------------------------------------------------- integration with commentary
def make_facts(suggestion=True, note=""):
    gl, coa = load_gl(DATA / "sample_gl.csv"), load_coa(DATA / "chart_of_accounts.csv")
    clean = prepare_for_analysis(gl, coa)
    var = compute_variances(clean, MONTH)
    sug = {("G&A", "6800"): {"text": "The annual audit fee of $34.5K was billed in full in September.",
                             "sources": ["D12", "D13"]}} if suggestion else {}
    notes = {("G&A", "6800"): note} if note else {}
    return build_facts(var, summarize(clean, MONTH), notes, sug)


def test_verified_suggestion_flows_into_a_valid_draft():
    facts = make_facts()
    text = draft_commentary(facts, "September 2026", mode="offline")
    assert "Documents suggest: The annual audit fee" in text and "[D12, D13]" in text
    assert validate(text, facts) == []


def test_analyst_note_overrides_suggestion():
    facts = make_facts(note="Audit billed early at partner's request")
    line = next(l for l in facts["lines"] if l["account_code"] == "6800")
    assert line["suggestion"] == "" and line["sources"] == []
    assert "Per analyst: Audit billed early" in draft_commentary(facts, "September 2026", mode="offline")


def test_citing_an_unverified_source_is_flagged():
    facts = make_facts()
    text = draft_commentary(facts, "September 2026", mode="offline").replace("[D12, D13]", "[D12, D14]")
    f = validate(text, facts)
    assert any(x.kind == "cause" and "D14" in x.detail for x in f)


def test_doc_ids_on_a_line_without_a_suggestion_are_flagged():
    facts = make_facts(suggestion=False)
    text = draft_commentary(facts, "September 2026", mode="offline")
    text = text.replace("**G&A / Professional Fees:**", "**G&A / Professional Fees:** [D12]", 1)
    assert any(x.kind == "cause" for x in validate(text, facts))


def test_simulated_mistakes_still_caught_with_suggestions_present():
    facts = make_facts()
    text = draft_commentary(facts, "September 2026", mode="offline")
    bad, applied = inject_errors(text, facts)
    kinds = {f.kind for f in validate(bad, facts)}
    assert len(applied) == 3 and {"number", "direction", "cause"} <= kinds


def test_score_requires_the_cause_not_just_the_right_document():
    gold = {"A|1": {"expected": "supported", "gold_docs": ["D01"], "keywords": ["audit"]}}
    header_only = Finding("A|1", "t", "supported", "A related document says: From: Dana, VP", "low",
                          [{"source_id": "D01", "quote": "From: Dana, VP"}])
    states_cause = Finding("A|1", "t", "supported", "The annual audit was billed in September.", "low",
                           [{"source_id": "D01", "quote": "x"}])
    assert score({"A|1": header_only}, gold)["doc_hit_n"] == "1/1"
    assert score({"A|1": header_only}, gold)["cause_found_n"] == "0/1"
    assert score({"A|1": states_cause}, gold)["cause_found_n"] == "1/1"
