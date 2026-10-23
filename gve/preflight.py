"""Pre-flight data checks.

Run before any analysis or AI step. The idea: if the data is wrong, the
commentary will be confidently wrong too, so surface problems first.

Each check returns Issue objects. Severity:
  "blocker"  - affects the period being analyzed; numbers for it can't be trusted as-is
  "warning"  - real problem, but outside the period or limited in impact
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import pandas as pd


@dataclass
class Issue:
    check: str
    severity: str  # "blocker" | "warning"
    message: str
    month: str | None = None
    department: str | None = None
    account_code: str | None = None
    rows: list = field(default_factory=list)  # row indexes in the input

    def to_dict(self):
        return asdict(self)


def check_invalid_values(df: pd.DataFrame) -> list[Issue]:
    issues = []
    bad = df[df["month"].isna() | df["budget"].isna() | df["actual"].isna()]
    for idx, r in bad.iterrows():
        what = [c for c in ("month", "budget", "actual") if pd.isna(r[c])]
        issues.append(Issue("invalid_value", "blocker",
                            f"Row {idx}: unreadable {', '.join(what)}",
                            r["month"] if pd.notna(r["month"]) else None,
                            r["department"], r["account_code"], [int(idx)]))
    return issues


def check_unmapped_accounts(df: pd.DataFrame, coa: pd.DataFrame) -> list[Issue]:
    known = set(coa["account_code"])
    issues = []
    unmapped = df[~df["account_code"].isin(known)]
    for (code, name), grp in unmapped.groupby(["account_code", "account_name"]):
        total = grp["actual"].sum()
        issues.append(Issue(
            "unmapped_account", "blocker",
            f"Account {code} '{name}' is not in the chart of accounts "
            f"({len(grp)} row(s), ${total:,.0f} actual). Excluded from variance analysis.",
            ", ".join(sorted(grp["month"].dropna().unique())),
            ", ".join(sorted(grp["department"].unique())), code,
            [int(i) for i in grp.index]))
    return issues


def check_duplicates(df: pd.DataFrame) -> list[Issue]:
    key = ["month", "department", "account_code", "budget", "actual"]
    dups = df[df.duplicated(subset=key, keep=False)]
    issues = []
    for k, grp in dups.groupby(key[:3]):
        issues.append(Issue(
            "duplicate_row", "warning",
            f"{len(grp)} identical rows for {k[1]} / {grp['account_name'].iloc[0]} in {k[0]} "
            f"(likely double import). Duplicates dropped for analysis.",
            k[0], k[1], k[2], [int(i) for i in grp.index]))
    return issues


def check_missing_periods(df: pd.DataFrame) -> list[Issue]:
    months = sorted(df["month"].dropna().unique())
    issues = []
    for dept, grp in df.groupby("department"):
        have = set(grp["month"].dropna())
        for m in months:
            if m not in have:
                issues.append(Issue(
                    "missing_period", "warning",
                    f"{dept} has no rows at all for {m}. YTD and trend figures for {dept} "
                    f"will be understated.", m, dept))
    return issues


def check_signs(df: pd.DataFrame, coa: pd.DataFrame) -> list[Issue]:
    types = dict(zip(coa["account_code"], coa["account_type"]))
    issues = []
    for idx, r in df.iterrows():
        if r["account_code"] not in types or pd.isna(r["actual"]):
            continue
        if r["actual"] < 0 or (pd.notna(r["budget"]) and r["budget"] < 0):
            kind = types[r["account_code"]]
            issues.append(Issue(
                "sign_reversal", "warning",
                f"{r['department']} / {r['account_name']} ({kind}) in {r['month']} has a "
                f"negative value (actual -${abs(r['actual']):,.0f}). Could be a reversal, credit, "
                f"or a sign error in the export.",
                r["month"], r["department"], r["account_code"], [int(idx)]))
    return issues


def check_spikes(df: pd.DataFrame, ratio: float = 3.0, min_dollars: float = 10_000,
                 skip_month: str | None = None) -> list[Issue]:
    """Flag a prior month whose actual is far above both that line's typical month
    and its own budget (a planned one-off has a budget; a miscoding usually doesn't).

    Uses the median of the *other* months as the baseline so the spike itself
    doesn't inflate it. The month being analyzed is skipped: large moves there are
    what the variance step is for.
    """
    issues = []
    clean = df.dropna(subset=["actual", "month"])
    for (dept, code), grp in clean.groupby(["department", "account_code"]):
        if len(grp) < 4:
            continue
        for idx, r in grp.iterrows():
            if skip_month and r["month"] == skip_month:
                continue
            others = grp.drop(index=idx)["actual"].abs()
            base = others.median()
            if base <= 0:
                continue
            over_budget = pd.isna(r["budget"]) or abs(r["actual"]) > ratio * abs(r["budget"])
            if (abs(r["actual"]) > ratio * base and abs(r["actual"]) - base > min_dollars
                    and over_budget):
                issues.append(Issue(
                    "spike", "warning",
                    f"{dept} / {r['account_name']} in {r['month']} is ${abs(r['actual']):,.0f} "
                    f"vs a typical ${base:,.0f} ({abs(r['actual']) / base:.1f}x). "
                    f"Worth confirming it isn't a miscoding.",
                    r["month"], dept, code, [int(idx)]))
    return issues


def run_preflight(df: pd.DataFrame, coa: pd.DataFrame, focus_month: str | None = None) -> list[Issue]:
    """Run every check. Issues that touch the focus month are escalated to blockers
    for that month only when they directly distort its numbers."""
    issues = (check_invalid_values(df) + check_unmapped_accounts(df, coa) +
              check_duplicates(df) + check_missing_periods(df) +
              check_signs(df, coa) + check_spikes(df, skip_month=focus_month))
    if focus_month:
        for i in issues:
            touches = i.month is not None and focus_month in str(i.month)
            if i.check == "unmapped_account":
                i.severity = "blocker" if touches else "warning"
            elif touches and i.check in ("duplicate_row", "sign_reversal", "spike", "missing_period"):
                i.severity = "blocker"
    order = {"blocker": 0, "warning": 1}
    return sorted(issues, key=lambda i: (order[i.severity], i.check, str(i.month)))
