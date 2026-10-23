"""Draft variance commentary.

The model only writes words. Every number it may use is pre-computed and
pre-formatted in a facts table, and the validator checks the draft against
that same table afterwards.

Two drafters:
  - "claude":  Anthropic Messages API (needs ANTHROPIC_API_KEY)
  - "offline": deterministic template, so the demo works with no key
"""

from __future__ import annotations

import os
import re

import pandas as pd

from .variance import fmt_money, fmt_pct

DEFAULT_MODEL = os.environ.get("GVE_MODEL", "claude-sonnet-5-5")
NO_DRIVER = "Driver not yet confirmed; needs analyst input."


def build_facts(variances: pd.DataFrame, summary: dict, notes: dict | None = None,
                suggestions: dict | None = None) -> dict:
    """Turn computed variances into the only facts the drafter may cite.

    notes:       {(department, account_code): "analyst note"} - human-supplied context.
    suggestions: {(department, account_code): {"text": ..., "sources": [...]}} - drivers the
                 investigating agent found AND that passed the grounding check AND that a
                 person chose to accept. An analyst note always wins over a suggestion.
    """
    notes = notes or {}
    suggestions = suggestions or {}
    lines = []
    for i, r in variances[variances["material"]].iterrows():
        note = (notes.get((r["department"], r["account_code"])) or "").strip()
        sug = suggestions.get((r["department"], r["account_code"])) or {}
        lines.append({
            "id": f"F{len(lines) + 1}",
            "department": r["department"],
            "account_code": r["account_code"],
            "account_name": r["account_name"],
            "account_type": r["account_type"],
            "budget": float(r["budget"]), "actual": float(r["actual"]),
            "variance": float(r["variance"]), "variance_pct": float(r["variance_pct"]),
            "direction": r["direction"],
            "favorable": bool(r["favorable"]),
            "budget_str": fmt_money(r["budget"]),
            "actual_str": fmt_money(r["actual"]),
            "variance_str": fmt_money(r["variance"]),
            "variance_pct_str": fmt_pct(r["variance_pct"]),
            "note": note,
            "suggestion": "" if note else (sug.get("text") or "").strip(),
            "sources": [] if note else list(sug.get("sources") or []),
        })
    totals = {}
    for k, v in summary.items():
        pct = v["variance"] / abs(v["budget"]) * 100 if v["budget"] else float("nan")
        totals[k] = {**{f: float(v[f]) for f in ("budget", "actual", "variance")},
                     "variance_pct": pct,
                     "budget_str": fmt_money(v["budget"]), "actual_str": fmt_money(v["actual"]),
                     "variance_str": fmt_money(v["variance"]), "variance_pct_str": fmt_pct(pct),
                     "direction": "over" if v["variance"] > 0 else "under"}
    return {"lines": lines, "totals": totals}


def _facts_table(facts: dict) -> str:
    out = ["TOTALS"]
    for k, t in facts["totals"].items():
        out.append(f"- {k.replace('_', ' ')}: budget {t['budget_str']}, actual {t['actual_str']}, "
                   f"{t['variance_str']} ({t['variance_pct_str']}) {t['direction']} budget")
    out.append("\nMATERIAL VARIANCES")
    for f in facts["lines"]:
        fav = "favorable" if f["favorable"] else "unfavorable"
        out.append(f"- [{f['id']}] {f['department']} / {f['account_name']} ({f['account_type']}): "
                   f"budget {f['budget_str']}, actual {f['actual_str']}, "
                   f"{f['variance_str']} ({f['variance_pct_str']}) {f['direction']} budget, {fav}. "
                   f"Analyst note: {f['note'] or 'NONE'}"
                   + (f" | Verified suggestion from documents: {f['suggestion']} "
                      f"(sources: {', '.join(f['sources'])})" if f["suggestion"] else ""))
    return "\n".join(out)


SYSTEM_PROMPT = f"""You write monthly budget-vs-actual commentary for a CFO.

Hard rules:
1. Use ONLY the numbers in the facts table, written exactly as they appear there (e.g. "$47.7K", "238.4%"). Never calculate, round differently, add up, or introduce any other number.
2. Describe direction with the words given: "over budget" / "under budget", and "favorable" / "unfavorable". Do not contradict the facts table.
3. Only state a cause if the line has an analyst note or a verified suggestion from documents.
   - Analyst note: stay within what the note says.
   - Verified suggestion: start the cause with "Documents suggest" and end the bullet with its sources in square brackets, e.g. [D02, D03]. Cite only the sources listed on that line.
   - Neither: end that bullet with exactly: "{NO_DRIVER}"
4. Do not mention account codes or fact IDs (F1, F2...).

Format (markdown, nothing else):
- One short overview paragraph using the TOTALS lines.
- Then one bullet per material variance, in the order given, each starting with "**<Department> / <Account name>:**"."""


def _draft_claude(facts: dict, month_label: str, company: str, model: str) -> str:
    """The AI writer. Runs on Claude, or on Gemini when only a Gemini key is set (see llm.py)."""
    from .llm import make_llm, write_text
    llm = make_llm()
    if hasattr(llm, "model") and getattr(llm, "label", "") != "Gemini":
        llm.model = model
    return write_text(SYSTEM_PROMPT, f"Company: {company}\nPeriod: {month_label}\n\n{_facts_table(facts)}", llm)


def _draft_offline(facts: dict, month_label: str, company: str) -> str:
    t = facts["totals"]
    rev, exp, oi = t["revenue"], t["expenses"], t["operating_income"]
    parts = [
        f"In {month_label}, revenue came in at {rev['actual_str']}, {rev['variance_str']} "
        f"({rev['variance_pct_str']}) {rev['direction']} budget. Expenses were {exp['actual_str']}, "
        f"{exp['variance_str']} ({exp['variance_pct_str']}) {exp['direction']} budget, leaving "
        f"operating income {oi['variance_str']} ({oi['variance_pct_str']}) {oi['direction']} budget "
        f"at {'a loss of ' if oi['actual'] < 0 else ''}{oi['actual_str']}.",
        "",
    ]
    for f in facts["lines"]:
        fav = "favorable" if f["favorable"] else "unfavorable"
        if f["note"]:
            driver = f"Per analyst: {f['note']}"
        elif f["suggestion"]:
            driver = f"Documents suggest: {f['suggestion'].rstrip('.')}. [{', '.join(f['sources'])}]"
        else:
            driver = NO_DRIVER
        parts.append(
            f"- **{f['department']} / {f['account_name']}:** {f['actual_str']} against a "
            f"{f['budget_str']} budget, {f['variance_str']} ({f['variance_pct_str']}) "
            f"{f['direction']} budget ({fav}). {driver}")
    return "\n".join(parts)


def draft_commentary(facts: dict, month_label: str, company: str = "the company",
                     mode: str = "offline", model: str = DEFAULT_MODEL) -> str:
    if not facts["lines"]:
        return f"No material variances in {month_label} at the current thresholds."
    if mode == "claude":
        return _draft_claude(facts, month_label, company, model)
    return _draft_offline(facts, month_label, company)


def inject_errors(text: str, facts: dict) -> tuple[str, list[str]]:
    """Plant the three classic LLM mistakes so the validator can be demoed:
    a wrong number, a flipped direction, and an invented cause."""
    applied = []
    lines = facts["lines"]
    if lines:
        f = lines[0]
        wrong = fmt_money(f["variance"] * 1.11)
        if f["variance_str"] in text and wrong != f["variance_str"]:
            text = text.replace(f["variance_str"], wrong, 1)
            applied.append(f"Changed {f['account_name']} variance {f['variance_str']} -> {wrong}")
    if len(lines) > 1:
        f = lines[1]
        flip = {"over": "under", "under": "over"}.get(f["direction"])
        pat = re.compile(rf"(\*\*{re.escape(f['department'])} / {re.escape(f['account_name'])}:\*\*[^\n]*?)\b{f['direction']} budget")
        if flip and pat.search(text):
            text = pat.sub(rf"\1{flip} budget", text, count=1)
            applied.append(f"Flipped {f['account_name']} direction to '{flip} budget'")
    no_note = [f for f in lines if not f["note"] and not f.get("suggestion")]
    if no_note:
        f = no_note[-1]
        pat = re.compile(rf"(\*\*{re.escape(f['department'])} / {re.escape(f['account_name'])}:\*\*[^\n]*?){re.escape(NO_DRIVER)}")
        if pat.search(text):
            text = pat.sub(r"\1This was driven by a one-time vendor price increase.", text, count=1)
            applied.append(f"Invented a cause for {f['account_name']}")
    return text, applied
