"""Variance math. All numbers the commentary is allowed to use come from here,
never from the language model."""

from __future__ import annotations

import math

import pandas as pd


def fmt_money(x: float) -> str:
    """$47.7K / $1.25M / $850 style, sign-free (direction is stated in words)."""
    a = abs(x)
    if a >= 1_000_000:
        return f"${a / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"${a / 1_000:.1f}K"
    return f"${a:,.0f}"


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{abs(x):.1f}%"


def prepare_for_analysis(df: pd.DataFrame, coa: pd.DataFrame) -> pd.DataFrame:
    """Apply the safe, mechanical fixes the preflight reported:
    drop exact duplicates, drop unreadable rows, exclude unmapped accounts."""
    key = ["month", "department", "account_code", "budget", "actual"]
    out = df.dropna(subset=["month", "budget", "actual"]).drop_duplicates(subset=key)
    out = out[out["account_code"].isin(set(coa["account_code"]))]
    types = coa.set_index("account_code")["account_type"]
    out = out.assign(account_type=out["account_code"].map(types))
    return out.reset_index(drop=True)


def compute_variances(df: pd.DataFrame, month: str, min_dollars: float = 10_000,
                      min_pct: float = 5.0) -> pd.DataFrame:
    """Variance by department and account for one month.

    favorable: revenue above budget, or expense below budget.
    material:  |variance| >= min_dollars AND |variance %| >= min_pct.
    """
    m = df[df["month"] == month]
    g = (m.groupby(["department", "account_code", "account_name", "account_type"], as_index=False)
          [["budget", "actual"]].sum())
    g["variance"] = g["actual"] - g["budget"]
    g["variance_pct"] = g.apply(
        lambda r: (r["variance"] / abs(r["budget"]) * 100) if r["budget"] else float("nan"), axis=1)
    g["direction"] = g["variance"].apply(lambda v: "over" if v > 0 else ("under" if v < 0 else "on"))
    g["favorable"] = g.apply(
        lambda r: (r["variance"] >= 0) if r["account_type"] == "revenue" else (r["variance"] <= 0),
        axis=1)
    g["material"] = (g["variance"].abs() >= min_dollars) & (
        g["variance_pct"].abs().fillna(float("inf")) >= min_pct)
    g = g.sort_values("variance", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return g


def summarize(df: pd.DataFrame, month: str) -> dict:
    """Top-line totals for the month: revenue, expenses, operating income."""
    m = df[df["month"] == month]
    rev = m[m["account_type"] == "revenue"][["budget", "actual"]].sum()
    exp = m[m["account_type"] == "expense"][["budget", "actual"]].sum()
    oi_b, oi_a = rev["budget"] - exp["budget"], rev["actual"] - exp["actual"]
    return {
        "revenue": {"budget": rev["budget"], "actual": rev["actual"],
                    "variance": rev["actual"] - rev["budget"]},
        "expenses": {"budget": exp["budget"], "actual": exp["actual"],
                     "variance": exp["actual"] - exp["budget"]},
        "operating_income": {"budget": oi_b, "actual": oi_a, "variance": oi_a - oi_b},
    }
