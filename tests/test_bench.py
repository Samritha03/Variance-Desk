import pandas as pd
import pytest

from gve import bench

B = bench.load_bench()


@pytest.fixture(scope="module")
def retriever():
    return bench.make_retriever()


def test_shape():
    cs = B["cases"]
    assert len(cs) == 36
    assert sum(c["expected"] == "supported" for c in cs) == 28
    assert sum(c["expected"] == "no_evidence" for c in cs) == 8
    assert len({(c["month"], c["line"]["department"], c["line"]["account_code"]) for c in cs}) == 36


def test_ledger_reconciles_and_cases_are_material():
    t = pd.read_csv(bench.BENCH / "transactions.csv", dtype={"account_code": str})
    for c in B["cases"]:
        l = c["line"]
        got = t[(t.month == c["month"]) & (t.department == l["department"]) & (t.account_code == l["account_code"])].amount.sum()
        assert abs(got - l["actual"]) < 0.02
        assert abs(l["variance"]) >= 5000 and abs(l["variance_pct"]) >= 5


def test_gold_docs_exist_and_are_in_window(retriever):
    from gve.retrieval import window_for_month
    for c in B["cases"]:
        lo, hi = window_for_month(c["month"])
        for d in c["gold_docs"]:
            assert lo <= retriever.docs[d]["date"] <= hi, (c["id"], d)
        for d in c["stale_docs"]:
            assert not (lo <= retriever.docs[d]["date"] <= hi), (c["id"], d)


def test_hybrid_beats_keyword_on_paraphrases_and_filter_helps(retriever):
    on = {(r["mode"], r["queries"]): r for r in bench.retrieval_table(retriever, B)}
    off = {(r["mode"], r["queries"]): r for r in bench.retrieval_table(retriever, B, date_filter=False)}
    assert on[("hybrid", "paraphrased")]["recall"] > on[("bm25", "paraphrased")]["recall"]
    assert on[("hybrid", "paraphrased")]["recall"] > off[("hybrid", "paraphrased")]["recall"]


def test_baseline_never_invents_a_cause_for_no_evidence_lines(retriever):
    from gve.agent import KeywordBaseline
    from gve.tools import load_ledger
    r = bench.agent_run(KeywordBaseline(), retriever, load_ledger(bench.BENCH / "transactions.csv"), B, "hybrid")
    assert r["false_attribution_n"].endswith("/8")
    assert r["groundedness_n"].split("/")[0] == r["groundedness_n"].split("/")[1]
