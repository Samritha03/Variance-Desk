"""Load and normalize a budget-vs-actual extract and a chart of accounts."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["month", "department", "account_code", "account_name", "budget", "actual"]
COA_COLUMNS = ["account_code", "account_name", "account_type"]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_gl(source) -> pd.DataFrame:
    """Read a CSV (path or file-like) and normalize types.

    Raises ValueError with a readable message if required columns are missing.
    Non-numeric budget/actual values become NaN so the preflight can flag them
    instead of the app crashing.
    """
    df = _clean_columns(pd.read_csv(source, dtype={"account_code": str}))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. "
                         f"Expected: {', '.join(REQUIRED_COLUMNS)}")
    df["account_code"] = df["account_code"].astype(str).str.strip()
    df["department"] = df["department"].astype(str).str.strip()
    df["account_name"] = df["account_name"].astype(str).str.strip()
    df["month"] = pd.to_datetime(df["month"].astype(str), errors="coerce").dt.strftime("%Y-%m")
    for col in ("budget", "actual"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    return df


def load_coa(source) -> pd.DataFrame:
    df = _clean_columns(pd.read_csv(source, dtype={"account_code": str}))
    missing = [c for c in COA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Chart of accounts missing column(s): {', '.join(missing)}")
    df["account_code"] = df["account_code"].astype(str).str.strip()
    df["account_type"] = df["account_type"].astype(str).str.strip().str.lower()
    bad = set(df["account_type"]) - {"revenue", "expense"}
    if bad:
        raise ValueError(f"account_type must be 'revenue' or 'expense'; found {sorted(bad)}")
    return df
