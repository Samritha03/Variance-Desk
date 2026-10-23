"""Generate the evaluation benchmark: 36 independent variance cases, each with its own documents and ledger.

Everything is fictional. The demo company (data/) has 6 variance lines in one month, which is a
walkthrough, not a test. This benchmark spreads 36 cases across 18 months so that the pooled document
set contains realistic neighbours: other months' invoices, other departments' emails.

Per explained case (28):   2 true documents (a decision/email and an invoice or statement)
                           3 decoys: last year's version, an unapproved proposal, a quote-only estimate
Per no-evidence case (8):  a line that moved with no real reason on file, plus 2 traps
                           (a forward-looking note, last year's memo)

Outputs (in this folder):  docs/*.txt  transactions.csv  bench_key.json
Run:  python data/bench/generate_bench.py
"""

import json
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gve.variance import fmt_money, fmt_pct  # noqa: E402

HERE = Path(__file__).parent
RNG = random.Random(11)
MONTHS = [f"2025-{m:02d}" for m in range(2, 13)] + [f"2026-{m:02d}" for m in range(1, 8)]   # 18 months
PEOPLE = ["Marcus Hale", "Priya Nair", "Dana Whitfield", "Tom Reyes", "Elena Ortiz", "Jonas Berg", "Aisha Karim", "Leo Marchetti"]
MN = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# cause: dept, code, name, type, direction, named_txn, variants[(event, vendor, detail, keywords)], templates, queries
CAUSES = [
    dict(dept="Marketing", code="6200", name="Events & Trade Shows", type="expense", dir="over", named=True,
         variants=[("Retail Expo", "Expo Partners LLC", "booth and display package", ["expo", "booth"]),
                   ("Home and Garden Summit", "Summit Events Inc", "sponsorship and exhibitor fees", ["summit", "sponsor"])],
         email=("Approval to attend the {event}", "{p1} to {p2}: approved. We will exhibit at the {event} this month. It was not in the plan. {vendor} will bill the {detail}, and the team travels the week before."),
         doc2=("invoice", "{vendor} invoice {ref}", "{vendor}, invoice {ref}. {event}: {detail}. Amount due: {amt}."),
         q=["{event} booth exhibit", "conference costs and attendee travel", "who approved the trade show", "exhibitor fees and display"]),
    dict(dept="Engineering", code="6500", name="Software & Cloud", type="expense", dir="over", named=True,
         variants=[("data warehouse migration", "Amazon Web Services", "dual-running the old and new warehouses", ["migration", "warehouse"]),
                   ("forecasting model training", "Google Cloud", "GPU hours for model training", ["training", "gpu"])],
         email=("{event} is under way", "{p1}: the {event} started this month. Until cutover we are {detail}, so cloud usage runs well above plan. Expect it to normalise after cutover."),
         doc2=("invoice", "{vendor} usage summary {ref}", "{vendor} usage summary {ref}. Increase driven by {detail}. Charge: {amt}."),
         q=["{event} cloud usage", "hosting bill increase", "infrastructure spend jumped", "compute charges above plan"]),
    dict(dept="Operations", code="6300", name="Freight & Shipping", type="expense", dir="over", named=True,
         variants=[("fuel surcharge", "Northline Freight", "a 14% fuel surcharge effective the first of the month", ["fuel", "surcharge"]),
                   ("port congestion rerouting", "Coastal Parcel", "rerouting around the congested port at premium rates", ["port", "reroute"])],
         email=("Carrier notice: {event}", "{p1}: {vendor} notified us of {detail}. Finance should expect higher freight until it ends."),
         doc2=("invoice", "{vendor} invoice {ref}", "{vendor}, invoice {ref}. Includes {detail}. Total: {amt}."),
         q=["{event} freight", "shipping costs higher than planned", "carrier extra charges", "transport bill increase"]),
    dict(dept="G&A", code="6800", name="Professional Fees", type="expense", dir="over", named=True,
         variants=[("annual audit", "Whitfield & Crane LLP", "fieldwork and report for the annual audit", ["audit", "fieldwork"]),
                   ("litigation settlement counsel", "Marlow Hart LLP", "outside counsel for the vendor dispute settlement", ["settlement", "counsel"])],
         email=("{event}: billing", "{p1} to {p2}: {vendor} finished {detail}. They bill the full engagement at once rather than monthly."),
         doc2=("invoice", "{vendor} invoice {ref}", "{vendor}, invoice {ref}. Services: {detail}. Amount due: {amt}."),
         q=["{event} fee", "external accountant charges", "legal and professional services billing", "outside counsel invoice"]),
    dict(dept="Sales", code="4000", name="Product Revenue", type="revenue", dir="under", named=False,
         variants=[("two enterprise orders", "Northgate Outfitters", "pushed their delivery acceptance to next month", ["slip", "push"]),
                   ("holiday reorder", "Bayline Retail Group", "delayed the reorder until their warehouse reopens", ["delay", "reorder"])],
         email=("Month-end: {event} move out", "{p1} to {p2}: {vendor} {detail}, so the {event} will land after month-end. The revenue shortfall is timing, not lost business."),
         doc2=("note", "Pipeline review", "Pipeline review. {vendor}: {event} slipped, customer {detail}. Expected next month."),
         q=["{event} slip next month", "revenue shortfall deals delayed", "customers postponed signing", "why did bookings miss plan"]),
    dict(dept="Sales", code="4000", name="Product Revenue", type="revenue", dir="over", named=False,
         variants=[("seasonal promotion", "Oakridge Stores", "ran the promotion two weeks earlier than planned", ["promotion", "earlier"]),
                   ("new distributor launch order", "Larkspur Home", "placed the opening stocking order", ["distributor", "stocking"])],
         email=("{event} results", "{p1}: {vendor} {detail}, which pulled sales forward into this month. Treat part of the beat as timing."),
         doc2=("note", "Sales ops note", "Sales ops note. {vendor} {detail} ({event}). Orders booked this month."),
         q=["{event} sales beat", "revenue above plan", "pulled forward orders", "stocking order"]),
    dict(dept="Marketing", code="6100", name="Digital Advertising", type="expense", dir="over", named=True,
         variants=[("product launch campaign", "Meta Ads", "an extra launch burst across paid social", ["launch", "campaign"]),
                   ("holiday pre-season push", "Programmatic Partners", "front-loaded holiday display buying", ["holiday", "pre-season"])],
         email=("Approved: {event}", "{p1} to {p2}: approved extra spend for the {event}: {detail}. It runs for one month."),
         doc2=("invoice", "{vendor} statement {ref}", "{vendor}, statement {ref}. {event}: {detail}. Total charges: {amt}."),
         q=["{event} ad spend", "advertising budget overrun", "paid social costs", "media buying increase"]),
    dict(dept="G&A", code="6400", name="Facilities & Rent", type="expense", dir="over", named=True,
         variants=[("lease reconciliation", "Rivergate Properties", "a catch-up charge for last year's operating expenses", ["lease", "reconciliation"]),
                   ("office expansion", "Harbor Commercial", "a deposit for the added floor", ["expansion", "deposit"])],
         email=("Landlord notice: {event}", "{p1}: {vendor} sent {detail}. One-time item, not a change to monthly rent."),
         doc2=("invoice", "{vendor} invoice {ref}", "{vendor}, invoice {ref}. {event}: {detail}. Amount due: {amt}."),
         q=["{event} charge", "rent higher than budget", "landlord catch-up bill", "property cost one-time"]),
    dict(dept="Engineering", code="6650", name="Contractors & Agencies", type="expense", dir="over", named=True,
         variants=[("security remediation", "Brightpath Agency", "contract engineers to fix the penetration test findings", ["security", "remediation"]),
                   ("mobile app rebuild", "Contract developers", "a short-term team to finish the app rebuild", ["mobile", "rebuild"])],
         email=("{event}: contractors engaged", "{p1} to {p2}: we engaged {vendor} for {detail}. Four weeks, fixed scope."),
         doc2=("invoice", "{vendor} invoice {ref}", "{vendor}, invoice {ref}. {event}: {detail}. Amount due: {amt}."),
         q=["{event} contractors", "outside engineers cost", "agency fees above plan", "temporary developer spend"]),
    dict(dept="Operations", code="5000", name="Cost of Goods Sold", type="expense", dir="under", named=True,
         variants=[("supplier volume rebate", "Delta Textile Mills", "a year-end volume rebate credited against purchases", ["rebate", "credit"]),
                   ("renegotiated fabric pricing", "Pacific Fabrication", "new lower unit prices effective this month", ["pricing", "renegotiated"])],
         email=("{event} confirmed", "{p1}: {vendor} confirmed {detail}. Cost of goods will run below plan until budgets are reset."),
         doc2=("statement", "{vendor} credit memo {ref}", "{vendor}, credit memo {ref}. {event}: {detail}. Credit: {amt}."),
         q=["{event} cost savings", "materials cost below plan", "supplier credit", "cheaper purchasing"]),
    dict(dept="Sales", code="6600", name="Travel & Entertainment", type="expense", dir="over", named=True,
         variants=[("annual sales kickoff", "Skyline Travel", "flights and hotel for the whole sales team", ["kickoff", "flights"]),
                   ("customer summit travel", "Corporate cards", "executive travel to the customer summit", ["summit", "executive"])],
         email=("{event} logistics", "{p1} to {p2}: the {event} happened this month: {detail}. Budgeted in Q1 originally, moved up."),
         doc2=("expense report", "Expense report {ref}", "Expense report {ref}. {event}: {detail}. Total: {amt}."),
         q=["{event} cost", "team offsite expenses", "sales travel spend", "hotel and airfare overrun"]),
    dict(dept="G&A", code="6900", name="Insurance", type="expense", dir="over", named=True,
         variants=[("policy renewal", "Atlas Mutual", "a renewal premium increase after two claims last year", ["renewal", "premium"]),
                   ("cyber coverage upgrade", "Beacon Underwriters", "higher limits added mid-term", ["cyber", "coverage"])],
         email=("{event}", "{p1}: {vendor} bound the {event} with {detail}. Premium is billed in one instalment."),
         doc2=("invoice", "{vendor} premium notice {ref}", "{vendor}, premium notice {ref}. {event}: {detail}. Amount due: {amt}."),
         q=["{event} premium", "insurance cost increase", "coverage change bill", "annual policy payment"]),
    dict(dept="Sales", code="4100", name="Services Revenue", type="revenue", dir="under", named=False,
         variants=[("installation backlog", "Install & Setup Services", "two technicians left and jobs queued", ["backlog", "technician"]),
                   ("support plan renewals", "Harbor Care Plans", "renewals slipped while pricing was reviewed", ["renewal", "pricing"])],
         email=("{event} update", "{p1} to {p2}: the {event}: {detail}. Services revenue is behind plan until the queue clears."),
         doc2=("note", "Services ops note", "Services ops note. {event}: {detail}. Work moves to next month."),
         q=["{event} services revenue", "installation work delayed", "customer renewals pending", "services income below plan"]),
    dict(dept="Engineering", code="6000", name="Salaries & Wages", type="expense", dir="over", named=True,
         variants=[("retention bonus payout", "Payroll bonus run", "a one-time retention bonus paid to the platform team", ["retention", "bonus"]),
                   ("early start of two senior hires", "Payroll run (supplemental)", "two senior engineers who started a month early", ["senior", "started"])],
         email=("{event}", "{p1} to {p2}: approved the {event}: {detail}. Not recurring."),
         doc2=("payroll", "Payroll register {ref}", "Payroll register {ref}. {event}: {detail}. Amount: {amt}."),
         q=["{event} payroll", "one-time compensation", "bonus costs", "headcount cost above plan"]),
]

NO_EVIDENCE = [  # (dept, code, name, type)
    ("Marketing", "6000", "Salaries & Wages", "expense"), ("Sales", "6000", "Salaries & Wages", "expense"),
    ("Operations", "6000", "Salaries & Wages", "expense"), ("G&A", "6000", "Salaries & Wages", "expense"),
    ("Engineering", "6000", "Salaries & Wages", "expense"), ("Operations", "6400", "Facilities & Rent", "expense"),
    ("Marketing", "6650", "Contractors & Agencies", "expense"), ("Operations", "6300", "Freight & Shipping", "expense"),
]


def d(month: str, day: int) -> date:
    y, m = map(int, month.split("-"))
    return date(y, m, day)


def shift_year(dt: date, years: int) -> date:
    return dt.replace(year=dt.year + years)


def money(x: float) -> str:
    return f"${abs(x):,.2f}"


def main():
    docs, txns, cases = {}, [], []
    used = set()
    n_doc, n_txn = 100, 10000

    def add_doc(dt: date, kind: str, title: str, body: str) -> str:
        nonlocal n_doc
        n_doc += 1
        i = f"D{n_doc}"
        docs[i] = {"date": dt.isoformat(), "type": kind, "title": title, "body": body.strip()}
        return i

    def add_txn(month, dept, code, name, vendor, desc, amount, ref, day):
        nonlocal n_txn
        n_txn += 1
        txns.append({"txn_id": f"T{n_txn}", "date": f"{month}-{day:02d}", "month": month, "department": dept,
                     "account_code": code, "account_name": name, "counterparty": vendor, "description": desc,
                     "amount": round(amount, 2), "reference": ref})

    def make_line(dept, code, name, typ, budget, actual):
        var = actual - budget
        pct = var / abs(budget) * 100
        fav = var >= 0 if typ == "revenue" else var <= 0
        return {"id": "F1", "department": dept, "account_code": code, "account_name": name, "account_type": typ,
                "budget": budget, "actual": actual, "variance": var, "variance_pct": pct,
                "direction": "over" if var > 0 else "under", "favorable": fav,
                "budget_str": fmt_money(budget), "actual_str": fmt_money(actual), "variance_str": fmt_money(var),
                "variance_pct_str": fmt_pct(pct), "note": "", "suggestion": "", "sources": []}

    def base_rows(month, dept, code, name, total, typ):
        vendors = [f"{name.split()[0]} Vendor A", f"{name.split()[0]} Vendor B", f"{name.split()[0]} Vendor C"]
        parts = [RNG.uniform(0.7, 1.3) for _ in range(2)]
        a = round(total * parts[0] / sum(parts), 2)
        for v, amt in zip(vendors[:2], (a, round(total - a, 2))):
            add_txn(month, dept, code, name, v, "Regular monthly charges", amt, f"R-{RNG.randint(10000, 99999)}", RNG.randint(2, 27))

    case_no = 0
    plan = []
    for k, c in enumerate(CAUSES):
        for v in range(2):
            plan.append((c, c["variants"][v], (k + 14 * v) % len(MONTHS)))
    for i, (dept, code, name, typ) in enumerate(NO_EVIDENCE):
        plan.append((None, (dept, code, name, typ), (2 + 2 * i) % len(MONTHS)))

    for c, var, mi in plan:
        month = MONTHS[mi]
        key = (c["dept"] if c else var[0], c["code"] if c else var[1])
        while (month, *key) in used:
            mi = (mi + 1) % len(MONTHS)
            month = MONTHS[mi]
        used.add((month, *key))
        case_no += 1
        cid = f"C{case_no:02d}"
        p1, p2 = RNG.sample(PEOPLE, 2)
        month_name = f"{MN[int(month[5:])]} {month[:4]}"
        if c:
            dept, code, name, typ = c["dept"], c["code"], c["name"], c["type"]
            event, vendor, detail, kws = var
            budget = float(RNG.choice([18000, 24000, 30000, 42000, 55000, 80000, 120000]))
            pct = RNG.uniform(0.28, 0.9)
            delta = round(budget * pct, -2) * (1 if c["dir"] == "over" else -1)
            if c["type"] == "revenue":
                budget *= 6
                delta = round(budget * RNG.uniform(0.07, 0.12), -2) * (1 if c["dir"] == "over" else -1)
            actual = round(budget + delta, 2)
            ref = f"{vendor.split()[0][:3].upper()}-{RNG.randint(1000, 9999)}"
            line = make_line(dept, code, name, typ, budget, actual)
            if c["named"]:
                add_txn(month, dept, code, name, vendor, f"{event.capitalize()}: {detail}", delta, ref, 24)
                base_rows(month, dept, code, name, budget, typ)
            else:
                base_rows(month, dept, code, name, actual, typ)
            amt = money(delta)
            fill = dict(event=event, vendor=vendor, detail=detail, ref=ref, amt=amt, p1=p1, p2=p2, month=month_name)
            sub, body = c["email"]
            g1 = add_doc(d(month, RNG.randint(3, 12)) - timedelta(days=RNG.choice([0, 0, 25])), "email", sub.format(**fill),
                         f"From: {p1}\nTo: {p2}\n\n" + body.format(**fill))
            kind2, t2, b2 = c["doc2"]
            g2 = add_doc(d(month, RNG.randint(15, 27)), kind2, t2.format(**fill), b2.format(**fill))
            stale = add_doc(shift_year(d(month, 20), -1), kind2, t2.format(**fill) + " (last year)",
                            b2.format(**{**fill, "amt": money(abs(delta) * RNG.uniform(0.55, 0.8)), "ref": ref[:4] + "0000"}))
            prop = add_doc(d(month, 5) - timedelta(days=RNG.choice([20, 40])), "email", f"Proposal: expanded {event} (not approved)",
                           f"From: {RNG.choice(PEOPLE)}\nTo: {p2}\n\nProposal to add a second phase to the {event} with {vendor}. This was reviewed and not approved, so no spend is planned. Revisit next year.")
            quote = add_doc(d(month, 28) + timedelta(days=RNG.choice([8, 18])), "email", f"Estimate request: {vendor}",
                            f"From: {RNG.choice(PEOPLE)}\nTo: {p2}\n\n{vendor} sent a price estimate for follow-on work related to the {event}. Quote only. Nothing has been ordered and nothing was charged this month.")
            qs = [c["q"][0].format(**fill)] + c["q"][1:]
            cases.append({"id": cid, "month": month, "line": line, "expected": "supported", "gold_docs": [g1, g2],
                          "keywords": kws, "queries": qs, "decoy_docs": [prop, quote], "stale_docs": [stale]})
        else:
            dept, code, name, typ = var
            budget = float(RNG.choice([60000, 90000, 140000, 210000]))
            actual = round(budget * RNG.uniform(1.06, 1.11), 2)
            line = make_line(dept, code, name, typ, budget, actual)
            base_rows(month, dept, code, name, actual, typ)
            who = RNG.choice(PEOPLE)
            t1 = add_doc(d(month, 20) + timedelta(days=RNG.choice([9, 18])), "email", f"{dept}: new role opening next month",
                         f"From: HR\nTo: {who}\n\nOffer accepted for an open {dept} role. Start date is the first of next month. No cost in the current month.")
            t2 = add_doc(shift_year(d(month, 10), -1), "memo", f"{dept} planning memo (last year)",
                         "Planning memo from last year. Annual increases take effect January 1. No other changes were approved.")
            cases.append({"id": cid, "month": month, "line": line, "expected": "no_evidence", "gold_docs": [],
                          "keywords": [], "queries": [f"{dept} {name.lower()} increase", f"why is {dept.lower()} {name.lower()} over budget",
                                                      f"{name.lower()} higher than plan", f"{dept.lower()} extra cost this month"],
                          "decoy_docs": [t1], "stale_docs": [t2]})

    out = HERE / "docs"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for i, dd in docs.items():
        (out / f"{i}.txt").write_text(f"id: {i}\ndate: {dd['date']}\ntype: {dd['type']}\ntitle: {dd['title']}\n---\n{dd['body']}\n")
    df = pd.DataFrame(txns)
    df.to_csv(HERE / "transactions.csv", index=False)

    # reconcile: transactions must sum to each case's actual
    for cs in cases:
        l = cs["line"]
        t = df[(df.month == cs["month"]) & (df.department == l["department"]) & (df.account_code == l["account_code"])]
        assert abs(t.amount.sum() - l["actual"]) < 0.02, (cs["id"], t.amount.sum(), l["actual"])
        assert abs(l["variance"]) >= 5000 and abs(l["variance_pct"]) >= 5, cs["id"]
    json.dump({"cases": cases}, open(HERE / "bench_key.json", "w"), indent=1)
    print(f"{len(cases)} cases ({sum(c['expected'] == 'supported' for c in cases)} explained, "
          f"{sum(c['expected'] == 'no_evidence' for c in cases)} no-evidence), {len(docs)} documents, {len(df)} transactions")


if __name__ == "__main__":
    main()
