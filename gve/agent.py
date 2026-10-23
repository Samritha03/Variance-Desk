"""The investigating agent.

For one material variance, the agent searches company documents and ledger
transactions to find out why the line moved, then records a finding with
citations. Two investigators share one interface:

  ClaudeInvestigator   tool-use loop: the model decides which queries to run,
                       code executes them, the model submits a finding.
  KeywordBaseline      no model. One search, best matching sentence. Exists so the
                       eval has something to beat and so the demo runs without a key.

Neither one is trusted. Every finding goes through ground(), which checks that:
  - each cited source was actually retrieved in this run,
  - each document quote appears word for word in the text that came back,
  - each dollar figure and percentage in the explanation traces to a cited quote,
    a cited transaction, or the computed variance.
A finding that fails any check is downgraded to "unverified" and is never used
as a suggested cause.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

from .retrieval import tokenize
from .tools import TOOL_SCHEMAS, AgentTools

DEFAULT_MODEL = os.environ.get("GVE_AGENT_MODEL", os.environ.get("GVE_MODEL", "claude-sonnet-5-5"))
MAX_STEPS = 8

SYSTEM = """You are a research assistant for a finance analyst. Your job is to find out, from the company's own records, why one budget line moved in one month.

How to work:
1. Use get_transactions to see what made up the line.
2. Use search_documents for the vendors, events or reasons you see. If the first query finds nothing useful, try another.
3. When you are done, call record_finding once.

Rules:
- Cite only documents and transactions that your tools returned. Never cite anything from memory.
- For each document you cite, quote the exact words from the retrieved text. Do not paraphrase inside a quote.
- A document counts only if it explains this month's variance. Ignore documents about other years, proposals that were not approved, events that have not happened yet, and anything that does not account for this amount.
- Do not state any figure that is not in the retrieved text or in the numbers you were given.
- If no document explains the variance, record status no_evidence. That is the right answer when it is true. Do not guess.
- Keep the explanation to one or two plain sentences."""


@dataclass
class Finding:
    line_key: str
    agent: str
    status: str                      # supported | no_evidence | unverified
    explanation: str
    confidence: str
    evidence: list[dict]
    problems: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    steps: int = 0

    @property
    def sources(self) -> list[str]:
        seen = []
        for e in self.evidence:
            s = (e.get("source_id") or "").strip()
            if s and s not in seen:
                seen.append(s)
        return seen


# ------------------------------------------------------------------ grounding
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?([KkMm])?")
PCT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?%")


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def money_values(text: str) -> list[float]:
    out = []
    for num, suf in MONEY.findall(text or ""):
        v = float(num.replace(",", ""))
        out.append(v * {"k": 1e3, "m": 1e6}.get(suf.lower(), 1))
    return out


def pct_values(text: str) -> list[float]:
    return [float(p.replace(",", "")) for p in PCT.findall(text or "")]


def _close(v: float, allowed, floor: float) -> bool:
    return any(abs(v - a) <= max(floor, 0.006 * abs(a)) for a in allowed)


def ground(f: Finding, tools: AgentTools, line: dict) -> Finding:
    """Verify a finding against what the tools actually returned."""
    probs: list[str] = []
    if f.status == "supported" and not f.evidence:
        probs.append("Marked as supported but cites no evidence.")

    allowed_money = {abs(line["budget"]), abs(line["actual"]), abs(line["variance"])}
    pct = line.get("variance_pct")
    allowed_pct = set() if pct is None or (isinstance(pct, float) and math.isnan(pct)) else {abs(pct)}
    txn_amounts, quote_amounts = [], []

    for ev in f.evidence:
        sid = (ev.get("source_id") or "").strip()
        quote = (ev.get("quote") or "").strip()
        if sid in tools.seen_docs:
            if not quote:
                probs.append(f"{sid}: no quote given.")
            elif not any(_norm(quote) in _norm(c) for c in tools.seen_docs[sid]):
                probs.append(f"{sid}: the quote is not word for word in the text that was retrieved.")
            else:
                vals = money_values(quote)
                allowed_money |= set(vals)
                quote_amounts += vals
                allowed_pct |= set(pct_values(quote))
        elif sid in tools.seen_txns:
            a = abs(float(tools.seen_txns[sid]["amount"]))
            txn_amounts.append(a)
            allowed_money.add(a)
        else:
            probs.append(f"{sid or '(blank)'}: cited, but never retrieved in this run.")
    if len(txn_amounts) > 1:
        allowed_money.add(sum(txn_amounts))
    if len(set(quote_amounts)) > 1:
        allowed_money.add(sum(set(quote_amounts)))

    for v in money_values(f.explanation):
        if not _close(v, allowed_money, 1.0):
            probs.append(f"${v:,.0f} in the explanation does not appear in the cited evidence or the computed figures.")
    for p in pct_values(f.explanation):
        if not _close(p, allowed_pct, 0.06):
            probs.append(f"{p:g}% in the explanation does not appear in the cited evidence or the computed figures.")

    f.problems = probs
    if probs and f.status == "supported":
        f.status = "unverified"
    f.trace = list(tools.trace)
    return f


# ------------------------------------------------------------------ investigators
def _prompt_for(line: dict, month_label: str) -> str:
    return (f"Investigate why {line['department']} / {line['account_name']} (account {line['account_code']}) was "
            f"{line['variance_str']} ({line['variance_pct_str']}) {line['direction']} budget in {month_label}. "
            f"Budget {line['budget_str']}, actual {line['actual_str']}. "
            f"Month for ledger queries: {line['month']}.")


class AnthropicLLM:
    """Thin wrapper so the loop can be tested with a fake client."""

    def __init__(self, model: str = DEFAULT_MODEL):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def create(self, system, messages, tools):
        return self.client.messages.create(model=self.model, max_tokens=1500, system=system,
                                           messages=messages, tools=tools)


class ClaudeInvestigator:
    """The tool-use loop. Runs on Claude by default, or on Gemini when only a Gemini key is set (see llm.py)."""

    def __init__(self, llm=None, max_steps: int = MAX_STEPS):
        self.llm, self.max_steps = llm, max_steps

    @property
    def name(self) -> str:
        from .llm import agent_name
        return agent_name(self.llm)

    def investigate(self, line: dict, month_label: str, tools: AgentTools) -> Finding:
        from .llm import make_llm
        llm = self.llm or make_llm()
        key = f"{line['department']}|{line['account_code']}"
        messages = [{"role": "user", "content": _prompt_for(line, month_label)}]
        final, steps = None, 0
        for steps in range(1, self.max_steps + 1):
            resp = llm.create(SYSTEM, messages, TOOL_SCHEMAS)
            calls = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            messages.append({"role": "assistant", "content": resp.content})
            if not calls:
                messages.append({"role": "user", "content": "Call record_finding now to submit your conclusion."})
                continue
            results = []
            for c in calls:
                if c.name == "record_finding":
                    final = dict(c.input)
                    results.append({"type": "tool_result", "tool_use_id": c.id, "content": "Recorded."})
                else:
                    results.append({"type": "tool_result", "tool_use_id": c.id,
                                    "content": tools.dispatch(c.name, dict(c.input))})
            messages.append({"role": "user", "content": results})
            if final is not None:
                break
        if final is None:
            f = Finding(key, self.name, "unverified", "The agent did not submit a conclusion.", "low", [],
                        ["No finding was recorded within the step limit."], list(tools.trace), steps)
            return f
        status = final.get("status") if final.get("status") in ("supported", "no_evidence") else "unverified"
        f = Finding(key, self.name, status, str(final.get("explanation", "")).strip(),
                    str(final.get("confidence", "low")), list(final.get("evidence") or []), steps=steps)
        return ground(f, tools, line)


class KeywordBaseline:
    """One search, best sentence. A floor to measure the agent against."""

    name = "keyword-baseline"

    def investigate(self, line: dict, month_label: str, tools: AgentTools) -> Finding:
        key = f"{line['department']}|{line['account_code']}"
        q = f"{line['department']} {line['account_name']}"
        qt = set(tokenize(q))
        res = json.loads(tools.dispatch("search_documents", {"query": q, "top_k": 3}))["results"]
        for r in res:
            overlap = qt & set(tokenize(f"{r['title']} {r['text']}"))
            if len(overlap) < 2:
                continue
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", r["text"]) if len(s.strip()) > 25]
            if not sents:
                continue
            best = max(sents, key=lambda s: len(qt & set(tokenize(s))))
            f = Finding(key, self.name, "supported", f"A related document says: {best}", "low",
                        [{"source_id": r["doc_id"], "quote": best}], steps=1)
            return ground(f, tools, line)
        f = Finding(key, self.name, "no_evidence", "No document matched this line.", "low", [], steps=1)
        return ground(f, tools, line)


def run_investigation(lines: list[dict], month: str, month_label: str, investigator, retriever, ledger,
                      date_filter: bool = True, on_progress=None) -> dict[str, Finding]:
    """One fresh tool session per line, so a line can only cite what it retrieved itself."""
    out = {}
    for i, line in enumerate(lines):
        tools = AgentTools(retriever, ledger, month, date_filter)
        line = dict(line, month=month)
        f = investigator.investigate(line, month_label, tools)
        out[f.line_key] = f
        if on_progress:
            on_progress(i + 1, len(lines), f)
    return out
