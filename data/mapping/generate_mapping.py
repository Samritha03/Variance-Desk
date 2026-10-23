"""Synthetic charts of accounts for the mapping demo. Fictional entities; the answer key is written by hand.

Standard chart: the target structure. Three entities, each with its own numbering and naming, including the
traps a real mapping project has: same code, different meaning (Bayside 5000 is hosting, not COGS), names in
another language (Kestrel), vague names (Misc Expense), and accounts with no home in the standard chart.
gold == "NONE" means a person has to decide; the right behaviour is to send it to review, not to guess.
"""
import csv
import json
from pathlib import Path

OUT = Path(__file__).parent

STANDARD = [
    ("4000", "Product Revenue", "revenue", "Sales of physical goods to customers"),
    ("4100", "Services Revenue", "revenue", "Installation, implementation, support, logistics and other services billed to customers"),
    ("4200", "Subscription Revenue", "revenue", "Recurring software subscription and licence fees"),
    ("4900", "Other Income", "revenue", "Interest income, gains and other non-operating income"),
    ("5000", "Cost of Goods Sold", "expense", "Direct cost of what is sold: inventory purchases, materials, carriers and fuel for delivered services"),
    ("5100", "Inbound Freight", "expense", "Freight paid to bring inventory in from suppliers"),
    ("5200", "Hosting Costs", "expense", "Cloud hosting that serves customers' production usage"),
    ("6000", "Salaries & Wages", "expense", "Base pay for employees"),
    ("6010", "Payroll Taxes & Benefits", "expense", "Employer payroll taxes, health insurance, pensions, social contributions"),
    ("6020", "Bonuses & Commissions", "expense", "Variable pay: sales commissions, bonuses"),
    ("6030", "Recruiting", "expense", "Recruiter and staffing agency placement fees, job advertising"),
    ("6100", "Digital Advertising", "expense", "Paid online advertising and customer acquisition"),
    ("6200", "Events & Trade Shows", "expense", "Trade shows, conferences, exhibits and field events"),
    ("6300", "Freight & Shipping", "expense", "Outbound delivery and shipping to customers"),
    ("6400", "Facilities & Rent", "expense", "Office, store and warehouse rent and lease costs"),
    ("6410", "Utilities", "expense", "Electricity, gas, water, telecom"),
    ("6500", "Software & Cloud", "expense", "Internal software subscriptions, IT licences and cloud tools used to run the business"),
    ("6600", "Travel & Entertainment", "expense", "Travel, meals and entertainment"),
    ("6650", "Contractors & Agencies", "expense", "External contractors and agencies for project work"),
    ("6800", "Professional Fees", "expense", "Legal, audit, accounting, tax and consulting fees"),
    ("6900", "Insurance", "expense", "Business, liability and cyber insurance premiums"),
    ("6950", "Bank & Card Fees", "expense", "Bank charges and payment processing fees"),
    ("7000", "Depreciation & Amortization", "expense", "Depreciation of fixed assets and amortization"),
    ("7100", "Bad Debt Expense", "expense", "Uncollectible receivables written off"),
    ("7200", "Interest Expense", "expense", "Interest paid on borrowings"),
    ("7300", "Income Taxes", "expense", "Income tax expense"),
    ("7400", "Repairs & Maintenance", "expense", "Repairs and upkeep of equipment and premises"),
    ("7500", "Office Supplies", "expense", "Stationery and consumables"),
]

# (entity, code, name, type, description, gold, tag)
R, E = "revenue", "expense"
ACCOUNTS = [
    ("Northwind Retail", "40100", "Merchandise Sales", R, "", "4000", "synonym"),
    ("Northwind Retail", "40500", "Customer Returns & Allowances", R, "", "NONE", "no home"),
    ("Northwind Retail", "41000", "Installation Services Income", R, "", "4100", "synonym"),
    ("Northwind Retail", "49000", "Interest & Misc Income", R, "", "4900", "synonym"),
    ("Northwind Retail", "50100", "Merchandise Purchases", E, "", "5000", "synonym"),
    ("Northwind Retail", "50300", "Freight-In", E, "", "5100", "synonym"),
    ("Northwind Retail", "60100", "Store Payroll", E, "", "6000", "synonym"),
    ("Northwind Retail", "60150", "Mgmt Salaries", E, "", "6000", "abbreviation"),
    ("Northwind Retail", "60200", "Employer Taxes & Benefits", E, "", "6010", "synonym"),
    ("Northwind Retail", "60250", "Seasonal Hiring Agency Fees", E, "Fees paid to staffing agency for holiday hires", "6030", "ambiguous"),
    ("Northwind Retail", "61000", "Google & Meta Ads", E, "", "6100", "synonym"),
    ("Northwind Retail", "61500", "Trade Show Exhibits", E, "", "6200", "synonym"),
    ("Northwind Retail", "62000", "Delivery Expense", E, "", "6300", "synonym"),
    ("Northwind Retail", "63000", "Store Rent", E, "", "6400", "synonym"),
    ("Northwind Retail", "63100", "Electric & Gas", E, "", "6410", "synonym"),
    ("Northwind Retail", "64000", "POS & IT Subscriptions", E, "", "6500", "synonym"),
    ("Northwind Retail", "65000", "Travel & Meals", E, "", "6600", "synonym"),
    ("Northwind Retail", "66000", "Legal & Audit", E, "", "6800", "synonym"),
    ("Northwind Retail", "67000", "Business Insurance", E, "", "6900", "exact"),
    ("Northwind Retail", "68000", "Merchant Processing Fees", E, "", "6950", "synonym"),
    ("Northwind Retail", "69000", "Depr Expense", E, "", "7000", "abbreviation"),
    ("Northwind Retail", "69500", "Shrink & Inventory Adjustments", E, "Inventory losses and count adjustments", "5000", "ambiguous"),

    ("Kestrel Logistics GmbH", "800000", "Transportumsatz / Freight Revenue", R, "", "4100", "foreign"),
    ("Kestrel Logistics GmbH", "800500", "Warehousing Fees Income", R, "", "4100", "synonym"),
    ("Kestrel Logistics GmbH", "890000", "Sonstige Erträge", R, "", "4900", "foreign"),
    ("Kestrel Logistics GmbH", "500000", "Subcontracted Carriers", E, "", "5000", "ambiguous"),
    ("Kestrel Logistics GmbH", "500500", "Fuel", E, "", "5000", "ambiguous"),
    ("Kestrel Logistics GmbH", "600000", "Löhne und Gehälter", E, "", "6000", "foreign"),
    ("Kestrel Logistics GmbH", "600500", "Sozialabgaben", E, "", "6010", "foreign"),
    ("Kestrel Logistics GmbH", "601000", "Provisionen", E, "", "6020", "foreign"),
    ("Kestrel Logistics GmbH", "610000", "Werbung / Advertising", E, "", "6100", "foreign"),
    ("Kestrel Logistics GmbH", "615000", "Messekosten", E, "", "6200", "foreign"),
    ("Kestrel Logistics GmbH", "620000", "Miete Lager", E, "", "6400", "foreign"),
    ("Kestrel Logistics GmbH", "621000", "Strom & Wasser", E, "", "6410", "foreign"),
    ("Kestrel Logistics GmbH", "630000", "IT Lizenzen", E, "", "6500", "foreign"),
    ("Kestrel Logistics GmbH", "640000", "Reisekosten", E, "", "6600", "foreign"),
    ("Kestrel Logistics GmbH", "650000", "Rechtsberatung & Steuerberatung", E, "", "6800", "foreign"),
    ("Kestrel Logistics GmbH", "660000", "Versicherungen", E, "", "6900", "foreign"),
    ("Kestrel Logistics GmbH", "670000", "Bankgebühren", E, "", "6950", "foreign"),
    ("Kestrel Logistics GmbH", "680000", "Abschreibungen", E, "", "7000", "foreign"),
    ("Kestrel Logistics GmbH", "690000", "Zinsaufwand", E, "", "7200", "foreign"),
    ("Kestrel Logistics GmbH", "699000", "Intercompany Management Fee - Parent", E, "", "NONE", "no home"),
    ("Kestrel Logistics GmbH", "699500", "Suspense / Clearing", E, "", "NONE", "no home"),

    ("Bayside Software", "4000", "Subscriptions - ARR", R, "", "4200", "code clash"),
    ("Bayside Software", "4010", "Implementation & Onboarding", R, "", "4100", "synonym"),
    ("Bayside Software", "4050", "Interest Income", R, "", "4900", "synonym"),
    ("Bayside Software", "5000", "Hosting - AWS", E, "Production hosting for customer workloads", "5200", "code clash"),
    ("Bayside Software", "5050", "Customer Support Staff", E, "", "6000", "ambiguous"),
    ("Bayside Software", "6000", "Engineering Salaries", E, "", "6000", "synonym"),
    ("Bayside Software", "6010", "Benefits", E, "", "6010", "synonym"),
    ("Bayside Software", "6020", "Sales Commissions", E, "", "6020", "synonym"),
    ("Bayside Software", "6030", "Recruiting Fees", E, "", "6030", "exact"),
    ("Bayside Software", "6100", "Paid Acquisition", E, "", "6100", "synonym"),
    ("Bayside Software", "6110", "Conferences & Field Events", E, "", "6200", "synonym"),
    ("Bayside Software", "6200", "Office Lease", E, "", "6400", "synonym"),
    ("Bayside Software", "6300", "SaaS Tools", E, "Internal tools: Slack, Notion, Jira", "6500", "synonym"),
    ("Bayside Software", "6310", "Dev Contractors", E, "", "6650", "synonym"),
    ("Bayside Software", "6400", "T&E", E, "", "6600", "abbreviation"),
    ("Bayside Software", "6500", "Legal", E, "", "6800", "code clash"),
    ("Bayside Software", "6510", "Accounting & Audit", E, "", "6800", "synonym"),
    ("Bayside Software", "6600", "D&O and Cyber Insurance", E, "", "6900", "synonym"),
    ("Bayside Software", "6700", "Stripe Fees", E, "", "6950", "synonym"),
    ("Bayside Software", "6800", "Misc Expense", E, "", "NONE", "vague"),
    ("Bayside Software", "6900", "Stock-Based Comp", E, "", "NONE", "no home"),
]


def main():
    with open(OUT / "standard_coa.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["code", "name", "type", "description"]); w.writerows(STANDARD)
    with open(OUT / "entity_accounts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["entity", "code", "name", "type", "description"])
        w.writerows(a[:5] for a in ACCOUNTS)
    gold = {f"{a[0]}|{a[1]}": {"target": a[5], "tag": a[6]} for a in ACCOUNTS}
    codes = {s[0] for s in STANDARD} | {"NONE"}
    assert all(g["target"] in codes for g in gold.values())
    types = {s[0]: s[2] for s in STANDARD}
    assert all(types[a[5]] == a[3] for a in ACCOUNTS if a[5] != "NONE"), "gold target must match account type"
    (OUT / "mapping_key.json").write_text(json.dumps(gold, indent=1))
    print(len(STANDARD), "standard,", len(ACCOUNTS), "entity accounts,", sum(g["target"] == "NONE" for g in gold.values()), "with no home")


if __name__ == "__main__":
    main()
