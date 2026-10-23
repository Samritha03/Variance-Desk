"""The two tools the agent can call, plus a record of everything it has seen.

The model never touches the files directly. Code runs the search and the ledger
query, and logs every document chunk and transaction that came back. That log is
what the grounding check later uses: the agent may only cite what it retrieved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .retrieval import Retriever, window_for_month

TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": ("Search the company's invoices, emails, contracts and notes. Returns the best matching "
                        "passages with document id, date and type. Results are limited to documents dated near "
                        "the month being analysed. Use short, specific queries (vendor names, events, amounts)."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "top_k": {"type": "integer", "description": "How many passages to return (1-6, default 4)"}},
            "required": ["query"]},
    },
    {
        "name": "get_transactions",
        "description": ("List ledger transactions for one department and account in one month, largest first, "
                        "with the total. Use it to see which vendors or invoices make up a variance."),
        "input_schema": {"type": "object", "properties": {
            "department": {"type": "string"},
            "account_code": {"type": "string"},
            "month": {"type": "string", "description": "YYYY-MM"},
            "limit": {"type": "integer", "description": "Max rows (default 8)"}},
            "required": ["department", "account_code", "month"]},
    },
    {
        "name": "record_finding",
        "description": ("Submit your conclusion for this line. Call exactly once, when done. Cite only documents "
                        "or transactions you retrieved. Each document citation needs a quote copied word for word "
                        "from the retrieved text. If nothing you found explains the variance, use status "
                        "no_evidence."),
        "input_schema": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["supported", "no_evidence"]},
            "explanation": {"type": "string", "description": "One or two plain sentences on why the line moved"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "evidence": {"type": "array", "items": {"type": "object", "properties": {
                "source_id": {"type": "string", "description": "Document id like D02 or transaction id like T00306"},
                "quote": {"type": "string", "description": "Exact words from the document (not needed for transactions)"}},
                "required": ["source_id"]}}},
            "required": ["status", "explanation", "confidence", "evidence"]},
    },
]


def load_ledger(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"account_code": str})


class AgentTools:
    def __init__(self, retriever: Retriever, ledger: pd.DataFrame, month: str, date_filter: bool = True):
        self.retriever, self.ledger, self.month = retriever, ledger, month
        self.window = window_for_month(month) if date_filter else (None, None)
        self.seen_docs: dict[str, list[str]] = {}   # doc id -> chunk texts returned
        self.seen_txns: dict[str, dict] = {}        # txn id -> row
        self.trace: list[dict] = []

    # -- tools --------------------------------------------------------------
    def search_documents(self, query: str, top_k: int = 4) -> dict:
        top_k = max(1, min(int(top_k or 4), 6))
        hits = self.retriever.search(query, k=top_k, date_from=self.window[0], date_to=self.window[1])
        for h in hits:
            self.seen_docs.setdefault(h["doc_id"], []).append(h["text"])
        return {"results": [{k: h[k] for k in ("doc_id", "date", "type", "title", "text")} for h in hits],
                "note": "No matching documents." if not hits else ""}

    def get_transactions(self, department: str, account_code: str, month: str, limit: int = 8) -> dict:
        t = self.ledger
        t = t[(t.department == department) & (t.account_code == str(account_code)) & (t.month == month)]
        if t.empty:
            return {"error": f"No transactions for {department} / {account_code} in {month}."}
        t = t.reindex(t.amount.abs().sort_values(ascending=False).index)
        rows = t.head(max(1, min(int(limit or 8), 20)))
        out = []
        for r in rows.itertuples():
            row = {"txn_id": r.txn_id, "date": r.date, "counterparty": r.counterparty,
                   "description": r.description, "amount": r.amount, "reference": r.reference}
            self.seen_txns[r.txn_id] = row
            out.append(row)
        return {"count": int(len(t)), "total": round(float(t.amount.sum()), 2), "transactions": out}

    # -- dispatch -----------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "search_documents":
                res = self.search_documents(**{k: args[k] for k in ("query", "top_k") if k in args})
                summary = f"{len(res['results'])} passage(s): " + ", ".join(r["doc_id"] for r in res["results"])
            elif name == "get_transactions":
                res = self.get_transactions(**{k: args[k] for k in ("department", "account_code", "month", "limit") if k in args})
                summary = res.get("error") or f"{res['count']} transaction(s), total {res['total']:,.2f}"
            else:
                res, summary = {"error": f"Unknown tool {name}"}, "unknown tool"
        except Exception as e:  # a bad argument from the model must not crash the run
            res, summary = {"error": f"{type(e).__name__}: {e}"}, "error"
        self.trace.append({"tool": name, "args": args, "summary": summary})
        return json.dumps(res)
