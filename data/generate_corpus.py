"""Generate the document corpus and transaction ledger the investigating agent works with.

Everything here is fictional. Two things are produced, both built to agree with
sample_gl.csv:

  transactions.csv   transaction-level ledger. For every (month, department, account)
                     the transactions add up exactly to the GL actual.
  docs/*.txt         invoices, emails and notes. Some explain the planted variances.
                     Others are distractors: wrong year, proposals that were never
                     approved, a hire that starts next month, or plain noise.
  corpus_key.json    gold answers for the eval harness: which documents truly explain
                     each line, and which line has no explanation at all.

Run after generate_data.py:   python data/generate_corpus.py
"""

import json
import random
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FOCUS = "2026-09"
RNG = random.Random(7)

GL = pd.read_csv(HERE / "sample_gl.csv", dtype={"account_code": str})
COA = pd.read_csv(HERE / "chart_of_accounts.csv", dtype={"account_code": str})
NAMES = dict(zip(COA.account_code, COA.account_name))

# Same mechanical clean-up the app applies: drop exact duplicates and unmapped accounts.
GL = GL.drop_duplicates()
GL = GL[GL.account_code.isin(set(COA.account_code))].reset_index(drop=True)

VENDORS = {
    "4000": ["Northgate Outfitters", "Bayline Retail Group", "Summit Trail Co.", "Oakridge Stores", "Larkspur Home"],
    "4100": ["Harbor Care Plans", "Install & Setup Services"],
    "5000": ["Delta Textile Mills", "Pacific Fabrication", "Eastgate Components"],
    "6000": ["Payroll run (first half)", "Payroll run (second half)"],
    "6100": ["Meta Ads", "Google Ads", "Programmatic Partners"],
    "6200": ["Event sponsorships (recurring)", "Local pop-up events"],
    "6300": ["Northline Freight", "Coastal Parcel"],
    "6400": ["Rivergate Properties"],
    "6500": ["Atlassian", "Workspace subscriptions", "Amazon Web Services"],
    "6600": ["Corporate cards", "Skyline Travel"],
    "6650": ["Brightpath Agency", "Contract developers"],
    "6800": ["Legal retainer", "Accounting support"],
    "6900": ["Insurance premium"],
}

# Named transactions inside the planted September variances.
# (vendor, description, share of the variance, reference, doc id, posting day)
PLANT = {
    ("Marketing", "6200"): [
        ("Expo Partners LLC", "Booth 418 and display package, NorthAmerica Retail Expo", 0.66, "EP-2209", "D02", 3),
        ("Skyline Travel", "Team travel, NorthAmerica Retail Expo (6 travelers)", 0.21, "SK-77412", "D03", 26),
        ("Meridian Logistics", "Booth crates, round-trip freight, Retail Expo", 0.13, "ML-30518", "D04", 20),
    ],
    ("Engineering", "6500"): [
        ("Amazon Web Services", "Compute, ingestion pipeline v2 workloads", 0.88, "AWS-SEP-2026", "D09", 30),
        ("Amazon Web Services", "Storage and data transfer increase", 0.12, "AWS-SEP-2026-B", "D09", 30),
    ],
    ("Operations", "6300"): [
        ("Northline Freight", "Fuel surcharge (14%, effective Sept 1)", 0.52, "NF-SC-0926", "D10", 30),
        ("Coastal Air Cargo", "Expedited air freight, 11 rush orders", 0.48, "CA-55120", "D11", 19),
    ],
    ("G&A", "6800"): [
        ("Whitfield & Crane LLP", "Annual financial statement audit, FY2026 fieldwork and report", 1.0, "WC-4471", "D12", 24),
    ],
}


def split(total_cents: int, n: int) -> list[int]:
    """Split an integer number of cents into n positive-ish parts that sum exactly."""
    if n == 1:
        return [total_cents]
    w = [RNG.uniform(0.6, 1.4) for _ in range(n)]
    parts = [int(round(total_cents * x / sum(w))) for x in w[:-1]]
    parts.append(total_cents - sum(parts))
    return parts


def cents(x: float) -> int:
    return int(round(x * 100))


def build_transactions() -> pd.DataFrame:
    rows, n = [], 0

    def add(month, dept, code, vendor, desc, amount_c, ref, day):
        nonlocal n
        n += 1
        rows.append({
            "txn_id": f"T{n:05d}", "date": f"{month}-{day:02d}", "month": month, "department": dept,
            "account_code": code, "account_name": NAMES[code], "counterparty": vendor,
            "description": desc, "amount": round(amount_c / 100, 2), "reference": ref,
        })

    for r in GL.itertuples():
        actual_c, budget_c = cents(r.actual), cents(r.budget)
        key = (r.department, r.account_code)
        if r.month == FOCUS and key in PLANT:
            var_c = actual_c - budget_c
            named, used = PLANT[key], 0
            for i, (vendor, desc, share, ref, _doc, day) in enumerate(named):
                amt = var_c - used if i == len(named) - 1 else int(round(var_c * share))
                used += amt
                add(r.month, r.department, r.account_code, vendor, desc, amt, ref, day)
            base_total = budget_c  # the regular spend the budget already expected
        else:
            base_total = actual_c
        vendors = VENDORS[r.account_code]
        k = min(len(vendors), RNG.choice([2, 3]))
        pick = RNG.sample(vendors, k)
        sign = -1 if base_total < 0 else 1
        for v, c in zip(pick, split(abs(base_total), k)):
            desc = "Monthly invoices" if r.account_code in ("4000", "4100") else "Regular monthly charges"
            add(r.month, r.department, r.account_code, v, desc, sign * c, f"R-{n + 1:05d}", RNG.randint(2, 27))
    return pd.DataFrame(rows)


def build_docs(txn: pd.DataFrame) -> dict[str, dict]:
    def amt(dept, code, i):
        t = txn[(txn.month == FOCUS) & (txn.department == dept) & (txn.account_code == code)
                & (txn.reference == PLANT[(dept, code)][i][3])]
        return f"${t.amount.iloc[0]:,.2f}"

    mk_total = sum(txn[(txn.month == FOCUS) & (txn.department == "Marketing") & (txn.account_code == "6200")
                       & txn.reference.isin([p[3] for p in PLANT[("Marketing", "6200")]])].amount)
    mk_round = f"${round(mk_total, -2):,.0f}"

    D = {}

    def doc(i, date, kind, title, body):
        D[f"D{i:02d}"] = {"date": date, "type": kind, "title": title, "body": body.strip()}

    doc(1, "2026-08-12", "email", "Approval to attend NorthAmerica Retail Expo", f"""
From: Marcus Hale, VP Marketing
To: Elena Ortiz, CFO
Subject: Approval to attend NorthAmerica Retail Expo, Sept 22-24

Elena, confirming the team will exhibit at the NorthAmerica Retail Expo, September 22-24 in Columbus. The show was originally planned for October, but the organizers moved the dates up, so the booth, travel and freight all land in September instead of Q4. Expected total is about {mk_round}. This was not in the September budget. I am not asking for extra funds, because we will trim the Q4 events line by the same amount.

Marcus""")
    doc(2, "2026-09-03", "invoice", "Expo Partners LLC invoice EP-2209", f"""
Expo Partners LLC
Invoice EP-2209, dated September 3, 2026
Bill to: Harborline Goods Co.
Event: NorthAmerica Retail Expo, September 22-24
Booth 418 (20x20 island) and display package: {amt("Marketing", "6200", 0)}
Terms: net 15. Total due: {amt("Marketing", "6200", 0)}""")
    doc(3, "2026-09-26", "expense report", "Expense report SK-77412, Retail Expo travel", f"""
Expense report summary SK-77412, Skyline Travel.
Six travelers attended the NorthAmerica Retail Expo, September 21-25: airfare, hotel and ground transport. Total {amt("Marketing", "6200", 1)}. Approved by M. Hale.""")
    doc(4, "2026-09-20", "freight bill", "Meridian Logistics bill ML-30518", f"""
Meridian Logistics, bill of lading ML-30518.
Three booth crates, round-trip freight to Columbus for the NorthAmerica Retail Expo. Total freight charges {amt("Marketing", "6200", 2)}.""")
    doc(5, "2026-09-10", "email", "Proposal: December holiday pop-up (not approved)", """
From: Marcus Hale, VP Marketing
To: Marketing team
Subject: Proposal: December holiday pop-up

Draft for a holiday pop-up shop in December, estimated at $18,000. This is only a proposal. It has not been approved or booked. Hold until Q4 planning.""")
    doc(6, "2026-09-29", "email", "September close: two enterprise orders slip to October", """
From: Jordan Reyes, Sales Operations
To: Finance
Subject: September close, two enterprise orders slip to October

Two large enterprise orders will not ship in September. Cascade Outfitters (order value $38,200) is waiting on their purchase order approval, and Pinecrest Stores ($24,900) asked to delay delivery while they finish a warehouse remodel. Both are now expected to ship in October, on October 6 and October 13. Together that is about $63,100 of September product revenue moving into Q4. Smaller orders from other accounts offset a little of it. No orders were cancelled.""")
    doc(7, "2026-09-15", "note", "Weekly pipeline review", """
Weekly pipeline review, September 15.
September product revenue is trending below plan. Cascade Outfitters and Pinecrest Stores are both flagged at risk of slipping past month-end. All other committed orders are on track.""")
    doc(8, "2026-09-08", "email", "Ingestion pipeline v2 is live", """
From: Priyanka Nair, Engineering Lead
To: Engineering, Finance
Subject: Ingestion pipeline v2 is live

Pipeline v2 went into production on September 8 and runs continuously on AWS compute. Expect cloud spend to run about $8,000 to $9,000 per month higher until we right-size the instances in Q4. This was not in the FY2026 budget.""")
    doc(9, "2026-10-02", "invoice", "AWS September 2026 usage summary", f"""
Amazon Web Services, September 2026 usage summary AWS-SEP-2026
Compute (ingestion pipeline v2 workloads): {amt("Engineering", "6500", 0)}
Storage and data transfer increase: {amt("Engineering", "6500", 1)}
Charges accrued at September 30.""")
    doc(10, "2026-08-28", "notice", "Northline Freight rate notice", """
Northline Freight, rate notice.
Effective September 1, 2026 the fuel surcharge increases from 8% to 14% on all ground shipments. This applies to every account.""")
    doc(11, "2026-09-17", "email", "Rush shipments for back-to-school restock", f"""
From: Tom Alvarez, Operations Manager
To: Finance
Subject: Rush shipments for back-to-school restock

We expedited 11 orders by air freight this month to meet retailer delivery windows. Coastal Air Cargo billed {amt("Operations", "6300", 1)}. Normally we ship ground.""")
    doc(12, "2026-09-24", "invoice", "Whitfield & Crane LLP invoice WC-4471", f"""
Whitfield & Crane LLP, invoice WC-4471, September 24, 2026
Annual financial statement audit, fiscal year 2026: fieldwork and report.
Amount due: {amt("G&A", "6800", 0)}. Billed in full on completion of fieldwork.""")
    doc(13, "2026-07-14", "contract", "Whitfield & Crane engagement letter", """
Engagement letter, Whitfield & Crane LLP.
The fee for the FY2026 audit will be invoiced in a single amount on completion of fieldwork, expected in September 2026.""")
    doc(14, "2025-09-22", "invoice", "Whitfield & Crane LLP invoice WC-3902 (last year)", """
Whitfield & Crane LLP, invoice WC-3902, September 22, 2025
Annual financial statement audit, fiscal year 2025. Amount due: $29,800.""")
    doc(15, "2025-12-10", "memo", "Merit cycle memo", """
HR memo. Annual merit increases take effect January 1, 2026 for all departments.""")
    doc(16, "2026-09-18", "email", "Offer accepted: Senior ML Engineer", """
From: Recruiting
To: Engineering leadership
Subject: Offer accepted, Senior ML Engineer

The Senior ML Engineer role in Engineering has been filled. Base salary $185,000. Start date: October 6.""")
    doc(17, "2026-09-05", "email", "Second floor kitchen remodel", """
Facilities notice. The second floor kitchen will be closed for remodeling from September 8 to September 19. Please use the first floor kitchen.""")
    doc(18, "2026-09-12", "email", "Security awareness training", """
IT reminder. Annual security awareness training must be completed by September 30.""")
    doc(19, "2026-09-25", "agenda", "Board meeting agenda", """
Board meeting agenda, September 25.
1. Q3 close and budget review. 2. Product roadmap. 3. Hiring plan. 4. Variance discussion deferred to the October meeting.""")
    doc(20, "2026-09-30", "checklist", "September close checklist", """
September close checklist. Accruals, bank reconciliations, intercompany matching, payroll tie-out, review of open purchase orders.""")
    return D


GOLD = {
    "Marketing|6200": {"expected": "supported", "gold_docs": ["D01", "D02", "D03", "D04"],
                       "keywords": ["trade show", "expo"], "query": "trade show booth expo"},
    "Sales|4000": {"expected": "supported", "gold_docs": ["D06", "D07"], "keywords": ["slip", "october"], "query": "orders slip October"},
    "Engineering|6500": {"expected": "supported", "gold_docs": ["D08", "D09"], "keywords": ["pipeline"], "query": "pipeline cloud AWS"},
    "Operations|6300": {"expected": "supported", "gold_docs": ["D10", "D11"], "keywords": ["fuel", "expedit"], "query": "fuel surcharge expedited freight"},
    "G&A|6800": {"expected": "supported", "gold_docs": ["D12", "D13"], "keywords": ["audit"], "query": "audit fee invoice"},
    # Real variance with no explanation anywhere in the files. A good agent says so.
    "Engineering|6000": {"expected": "no_evidence", "gold_docs": [], "keywords": [],
                         "traps": ["D16", "D15"]},
}


def main():
    txn = build_transactions()
    chk = (txn.groupby(["month", "department", "account_code"]).amount.sum().round(2)
           .reset_index().merge(GL, on=["month", "department", "account_code"]))
    assert ((chk.amount - chk.actual).abs() < 0.011).all(), "transactions must reconcile to the GL"
    txn.to_csv(HERE / "transactions.csv", index=False)

    docs_dir = HERE / "docs"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir()
    docs = build_docs(txn)
    for did, d in docs.items():
        (docs_dir / f"{did}.txt").write_text(
            f"id: {did}\ndate: {d['date']}\ntype: {d['type']}\ntitle: {d['title']}\n---\n{d['body']}\n")
    (HERE / "corpus_key.json").write_text(json.dumps({"focus_month": FOCUS, "lines": GOLD,
                                                       "out_of_period_docs": ["D14", "D15"]}, indent=2))
    print(f"Wrote {len(txn)} transactions (reconciled to GL), {len(docs)} documents, {len(GOLD)} gold lines.")


if __name__ == "__main__":
    main()
