"""Generate synthetic budget-vs-actual data for a fictional mid-size company.

Everything here is made up. The point is to have data we control, so we know
exactly which variances and data problems are "real" and can check whether the
app catches them.

Outputs (written next to this script):
  chart_of_accounts.csv  - account master: code, name, type, category
  sample_gl.csv          - monthly budget vs actual by department and account
  answer_key.json        - the planted variances and data issues

Run:  python data/generate_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).parent
COMPANY = "Harborline Goods Co. (fictional)"
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]  # Jan-Sep 2026
FOCUS_MONTH = "2026-09"

CHART_OF_ACCOUNTS = [
    # code, name, type, category
    ("4000", "Product Revenue", "revenue", "Revenue"),
    ("4100", "Services Revenue", "revenue", "Revenue"),
    ("5000", "Cost of Goods Sold", "expense", "COGS"),
    ("6000", "Salaries & Wages", "expense", "People"),
    ("6100", "Digital Advertising", "expense", "Marketing"),
    ("6200", "Events & Trade Shows", "expense", "Marketing"),
    ("6300", "Freight & Shipping", "expense", "Operations"),
    ("6400", "Facilities & Rent", "expense", "Facilities"),
    ("6500", "Software & Cloud", "expense", "Technology"),
    ("6600", "Travel & Entertainment", "expense", "T&E"),
    ("6650", "Contractors & Agencies", "expense", "Outside Services"),
    ("6800", "Professional Fees", "expense", "Outside Services"),
    ("6900", "Insurance", "expense", "G&A"),
]
COA_NAME = {c: n for c, n, _, _ in CHART_OF_ACCOUNTS}

# Monthly budget base by (department, account code)
BUDGET_BASE = {
    ("Sales", "4000"): 850_000,
    ("Sales", "4100"): 120_000,
    ("Sales", "6000"): 140_000,
    ("Sales", "6600"): 18_000,
    ("Marketing", "6000"): 60_000,
    ("Marketing", "6100"): 45_000,
    ("Marketing", "6200"): 20_000,
    ("Marketing", "6650"): 15_000,
    ("Engineering", "6000"): 210_000,
    ("Engineering", "6500"): 38_000,
    ("Engineering", "6650"): 25_000,
    ("Operations", "5000"): 380_000,
    ("Operations", "6000"): 90_000,
    ("Operations", "6300"): 42_000,
    ("Operations", "6400"): 30_000,
    ("G&A", "6000"): 80_000,
    ("G&A", "6400"): 12_000,
    ("G&A", "6800"): 15_000,
    ("G&A", "6900"): 9_000,
}

# Planted variances in the focus month: (dept, code) -> actual multiplier on budget
PLANTED_VARIANCES = {
    ("Marketing", "6200"): (3.40, "Fall trade show (booth, travel, freight) landed in September"),
    ("Sales", "4000"): (0.93, "Two enterprise orders slipped from September into Q4"),
    ("Engineering", "6500"): (1.22, "Cloud usage spike from a new data pipeline"),
    ("Operations", "6300"): (1.30, "Carrier fuel surcharge plus expedited shipments"),
    ("G&A", "6800"): (3.30, "Annual audit fees billed in one month"),
    ("Operations", "5000"): (0.95, "Lower volume following the revenue shortfall"),
}


def build():
    rng = np.random.default_rng(SEED)
    rows = []
    for i, month in enumerate(MONTHS):
        season = 1 + 0.04 * np.sin(i / 2)  # small seasonality on revenue/COGS
        for (dept, code), base in BUDGET_BASE.items():
            budget = base * (season if code in ("4000", "4100", "5000", "6300") else 1.0)
            budget = round(budget, -2)
            actual = budget * (1 + rng.normal(0, 0.025))
            if month == FOCUS_MONTH and (dept, code) in PLANTED_VARIANCES:
                actual = budget * PLANTED_VARIANCES[(dept, code)][0] * (1 + rng.normal(0, 0.005))
            rows.append({
                "month": month,
                "department": dept,
                "account_code": code,
                "account_name": COA_NAME[code],
                "budget": round(budget, 2),
                "actual": round(actual, 2),
            })
    df = pd.DataFrame(rows)

    issues = []

    # Issue 1: Engineering has no rows for July (missing month)
    df = df[~((df.department == "Engineering") & (df.month == "2026-07"))]
    issues.append({"type": "missing_period", "department": "Engineering", "month": "2026-07",
                   "detail": "All Engineering rows for July removed"})

    # Issue 2: an account code that isn't in the chart of accounts
    df = pd.concat([df, pd.DataFrame([{
        "month": FOCUS_MONTH, "department": "G&A", "account_code": "6999",
        "account_name": "Misc - Uncategorized", "budget": 0.0, "actual": 23_750.00,
    }])], ignore_index=True)
    issues.append({"type": "unmapped_account", "department": "G&A", "month": FOCUS_MONTH,
                   "account_code": "6999", "detail": "Account 6999 not in chart of accounts"})

    # Issue 3: exact duplicate row (double-posted import)
    dup = df[(df.department == "Marketing") & (df.account_code == "6100") & (df.month == "2026-08")]
    df = pd.concat([df, dup], ignore_index=True)
    issues.append({"type": "duplicate_row", "department": "Marketing", "month": "2026-08",
                   "account_code": "6100", "detail": "Digital Advertising August row duplicated"})

    # Issue 4: sign flip on revenue (negative actual)
    m = (df.department == "Sales") & (df.account_code == "4100") & (df.month == "2026-05")
    df.loc[m, "actual"] = -df.loc[m, "actual"]
    issues.append({"type": "sign_reversal", "department": "Sales", "month": "2026-05",
                   "account_code": "4100", "detail": "Services Revenue May actual posted negative"})

    # Issue 5: one-off spike (likely miscoding) in a prior month
    m = (df.department == "Operations") & (df.account_code == "6400") & (df.month == "2026-03")
    df.loc[m, "actual"] = df.loc[m, "actual"] * 5
    issues.append({"type": "spike", "department": "Operations", "month": "2026-03",
                   "account_code": "6400", "detail": "Facilities March actual ~5x normal"})

    df = df.sort_values(["month", "department", "account_code"]).reset_index(drop=True)

    coa = pd.DataFrame(CHART_OF_ACCOUNTS,
                       columns=["account_code", "account_name", "account_type", "category"])
    answer_key = {
        "company": COMPANY,
        "focus_month": FOCUS_MONTH,
        "planted_variances": [
            {"department": d, "account_code": c, "account_name": COA_NAME[c],
             "multiplier": mult, "true_cause": cause}
            for (d, c), (mult, cause) in PLANTED_VARIANCES.items()
        ],
        "planted_data_issues": issues,
    }
    return df, coa, answer_key


def main():
    df, coa, key = build()
    df.to_csv(OUT / "sample_gl.csv", index=False)
    coa.to_csv(OUT / "chart_of_accounts.csv", index=False)
    (OUT / "answer_key.json").write_text(json.dumps(key, indent=2))
    print(f"Wrote {len(df)} GL rows, {len(coa)} accounts, "
          f"{len(key['planted_variances'])} planted variances, "
          f"{len(key['planted_data_issues'])} planted data issues.")


if __name__ == "__main__":
    main()
