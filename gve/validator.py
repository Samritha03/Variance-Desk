"""Check a drafted commentary against the computed facts.

Checks, in order of importance:
  number      every $ and % figure must be one of the pre-computed strings for
              that bullet (or the totals, for the overview paragraph)
  direction   "over budget" / "under budget" must match the computed direction
  cause       a bullet may only explain *why* if the analyst supplied a note;
              otherwise it must carry the "driver not confirmed" line
  coverage    every material variance gets a bullet

Nothing here uses an LLM. It is string matching against numbers the code computed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .commentary import NO_DRIVER

MONEY_RE = re.compile(r"\$\d[\d,]*(?:\.\d+)?[KM]?")
PCT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%")
BULLET_RE = re.compile(r"^\s*[-*]\s+\*\*(.+?)\s*/\s*(.+?):\*\*(.*)$")
ID_RE = re.compile(r"\b[DT]\d{2,5}\b")
CAUSE_WORDS = re.compile(
    r"\b(driven by|due to|because|caused by|resulted from|reflects|owing to|as a result of|attributable to)\b",
    re.IGNORECASE)


@dataclass
class Finding:
    kind: str      # number | direction | cause | coverage
    severity: str  # "error"
    where: str     # which bullet / paragraph
    detail: str
    snippet: str = ""

    def to_dict(self):
        return asdict(self)


def _tokens(text: str) -> list[str]:
    # strip a trailing sentence period so "$47.7K." matches "$47.7K"
    return MONEY_RE.findall(text) + PCT_RE.findall(text)


def _line_allowed(f: dict) -> set[str]:
    base = {f["budget_str"], f["actual_str"], f["variance_str"], f["variance_pct_str"]}
    # figures inside a verified, grounded suggestion are allowed to be repeated
    return base | set(_tokens(f.get("suggestion") or ""))


def _totals_allowed(facts: dict) -> set[str]:
    out = set()
    for t in facts["totals"].values():
        out |= {t["budget_str"], t["actual_str"], t["variance_str"], t["variance_pct_str"]}
    return out


def validate(text: str, facts: dict) -> list[Finding]:
    findings: list[Finding] = []
    by_key = {(f["department"].lower(), f["account_name"].lower()): f for f in facts["lines"]}
    seen = set()
    overview_lines = []

    for raw in text.splitlines():
        m = BULLET_RE.match(raw)
        if not m:
            if raw.strip():
                overview_lines.append(raw)
            continue
        dept, acct, body = m.group(1).strip(), m.group(2).strip(), m.group(3)
        where = f"{dept} / {acct}"
        f = by_key.get((dept.lower(), acct.lower()))
        if f is None:
            findings.append(Finding("coverage", "error", where,
                                    "Bullet refers to a line that isn't a material variance in the data.",
                                    raw.strip()))
            continue
        seen.add(f["id"])

        # numbers
        allowed = _line_allowed(f)
        for tok in _tokens(body):
            if tok not in allowed:
                findings.append(Finding(
                    "number", "error", where,
                    f"{tok} does not match any computed figure for this line "
                    f"(expected one of: {', '.join(sorted(allowed))}).", tok))

        # direction
        low = body.lower()
        says_over, says_under = "over budget" in low, "under budget" in low
        if f["direction"] == "over" and says_under and not says_over:
            findings.append(Finding("direction", "error", where,
                                    "Says under budget, but actual is over budget.", "under budget"))
        elif f["direction"] == "under" and says_over and not says_under:
            findings.append(Finding("direction", "error", where,
                                    "Says over budget, but actual is under budget.", "over budget"))
        for word, ok in (("unfavorable", not f["favorable"]), ("favorable", f["favorable"])):
            # "unfavorable" contains "favorable": test the longer word first
            if re.search(rf"(?<!un){word}\b", low) and not ok:
                findings.append(Finding(
                    "direction", "error", where,
                    f"Calls the variance {word}, but it is "
                    f"{'favorable' if f['favorable'] else 'unfavorable'}.", word))
                break

        # cause
        cited = set(ID_RE.findall(body))
        allowed_ids = set(f.get("sources") or [])
        if cited - allowed_ids:
            findings.append(Finding(
                "cause", "error", where,
                f"Cites {', '.join(sorted(cited - allowed_ids))}, which is not a verified source for this line.",
                ", ".join(sorted(cited - allowed_ids))))
        if not f["note"] and not f.get("suggestion"):
            has_placeholder = NO_DRIVER.lower() in low
            if CAUSE_WORDS.search(body):
                findings.append(Finding(
                    "cause", "error", where,
                    "States a cause, but there is no analyst note or verified document for this line.",
                    body.strip()[-120:]))
            elif not has_placeholder:
                findings.append(Finding(
                    "cause", "error", where,
                    "No note or verified document for this line, so it must say the driver is not yet confirmed.",
                    body.strip()[-120:]))

    # overview paragraph numbers
    allowed_tot = _totals_allowed(facts)
    for line in overview_lines:
        for tok in _tokens(line):
            if tok not in allowed_tot:
                findings.append(Finding(
                    "number", "error", "Overview",
                    f"{tok} does not match any computed total.", tok))

    # coverage
    for f in facts["lines"]:
        if f["id"] not in seen:
            findings.append(Finding(
                "coverage", "error", f"{f['department']} / {f['account_name']}",
                "Material variance is missing from the commentary."))
    return findings


def annotate(text: str, findings: list[Finding]) -> str:
    """Return markdown with offending tokens wrapped in red-highlighted HTML-safe marks."""
    out = text
    for fd in findings:
        if fd.kind in ("number", "direction") and fd.snippet:
            out = out.replace(fd.snippet, f":red[**{fd.snippet}**]", 1)
    return out
