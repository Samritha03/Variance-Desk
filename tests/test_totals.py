"""The headline numbers, recomputed from the raw ledger with plain pandas, independent of the app's code path."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gve.api import DATA, app

client = TestClient(app)
gl = pd.read_csv(DATA / "sample_gl.csv", dtype={"account_code": str})
coa = pd.read_csv(DATA / "chart_of_accounts.csv", dtype={"account_code": str})


def raw_totals(month):
    # the app drops exact duplicate rows and accounts missing from the chart (both reported on Data checks)
    d = gl[gl.month == month].drop_duplicates()
    d = d.merge(coa[["account_code", "account_type"]], on="account_code", how="inner")
    g = d.groupby("account_type")[["budget", "actual"]].sum()
    return g.loc["revenue"], g.loc["expense"]


@pytest.mark.parametrize("month", sorted(gl.month.unique()))
def test_totals_match_an_independent_recompute(month):
    t = client.post("/api/variances", json={"dataset": "sample", "month": month}).json()["totals"]
    rev, exp = raw_totals(month)
    assert t["revenue"]["actual"] == pytest.approx(rev.actual, abs=0.01) and t["revenue"]["budget"] == pytest.approx(rev.budget, abs=0.01)
    assert t["expenses"]["actual"] == pytest.approx(exp.actual, abs=0.01) and t["expenses"]["budget"] == pytest.approx(exp.budget, abs=0.01)
    assert t["operating_income"]["actual"] == pytest.approx(rev.actual - exp.actual, abs=0.01)
    assert t["operating_income"]["budget"] == pytest.approx(rev.budget - exp.budget, abs=0.01)


def test_favourable_direction_rules():
    rows = client.post("/api/variances", json={"dataset": "sample", "month": "2026-09", "min_dollars": 0, "min_pct": 0}).json()["all_rows"]
    for r in rows:
        if r["variance"] == 0:
            continue
        over = r["variance"] > 0
        assert r["favorable"] == (over if r["account_type"] == "revenue" else not over)
        assert r["variance"] == pytest.approx(r["actual"] - r["budget"], abs=0.01)
