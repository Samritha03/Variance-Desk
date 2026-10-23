"""Turn an uploaded file into a document the retriever can index.

A document needs an id, a date, a type, a title and text. The date is the one that matters: the search
filters by it, so a document with no date would be silently skipped. We read it from the file when we can
(front matter, an email "Date:" header, or the first date in the text) and say where it came from, and
refuse the file if there is none rather than guess.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime

MAX_BYTES = 2_000_000
MAX_CHARS = 200_000
EXTS = (".txt", ".md", ".eml", ".pdf")
MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
LONG = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
                  r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b", re.I)
LONG2 = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|"
                   r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+(20\d{2})\b", re.I)
SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b")
DOC_TYPES = ("email", "invoice", "contract", "memo", "note", "report", "other")


def _mon(name: str) -> int | None:
    n = name.lower().rstrip(".")
    for full, i in MONTHS.items():
        if full.startswith(n[:3]) and (len(n) <= len(full)):
            return i
    return None


def _valid(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str) -> list[str]:
    """Every date in the text, in order of appearance, as ISO strings."""
    hits: list[tuple[int, str]] = []
    for mo in ISO.finditer(text):
        v = _valid(*map(int, mo.groups()))
        if v: hits.append((mo.start(), v))
    for mo in LONG.finditer(text):
        m = _mon(mo.group(1))
        v = m and _valid(int(mo.group(3)), m, int(mo.group(2)))
        if v: hits.append((mo.start(), v))
    for mo in LONG2.finditer(text):
        m = _mon(mo.group(2))
        v = m and _valid(int(mo.group(3)), m, int(mo.group(1)))
        if v: hits.append((mo.start(), v))
    for mo in SLASH.finditer(text):   # US order, month first
        v = _valid(int(mo.group(3)), int(mo.group(1)), int(mo.group(2)))
        if v: hits.append((mo.start(), v))
    return [v for _, v in sorted(hits)]


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if not name.endswith(EXTS):
        raise ValueError(f"{filename}: only {', '.join(EXTS)} files are supported.")
    if len(data) > MAX_BYTES:
        raise ValueError(f"{filename}: larger than {MAX_BYTES // 1_000_000} MB.")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        try:
            pages = [(p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages]
        except Exception as e:
            raise ValueError(f"{filename}: could not read this PDF ({type(e).__name__}).")
        text = "\n\n".join(pages).strip()
        if len(text) < 20:
            raise ValueError(f"{filename}: no text found in this PDF. It may be a scan; this app does not do OCR.")
    else:
        text = data.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"{filename}: the file is empty.")
    return text[:MAX_CHARS]


def _guess_type(filename: str, text: str) -> str:
    low = text[:1500].lower()
    if filename.lower().endswith(".eml") or ("from:" in low and ("subject:" in low or "to:" in low)):
        return "email"
    if "invoice" in low[:300]:
        return "invoice"
    if "memo" in low[:200]:
        return "memo"
    if "agreement" in low[:300] or "contract" in low[:300]:
        return "contract"
    return "note"


def _guess_title(filename: str, text: str) -> str:
    m = re.search(r"^subject:\s*(.+)$", text, re.I | re.M)
    if m:
        return m.group(1).strip()[:120]
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and not re.match(r"^(from|to|date|cc):", line, re.I):
            return line[:120]
    return re.sub(r"\.\w+$", "", filename)


def parse_document(filename: str, data: bytes, *, doc_date: str | None = None, doc_type: str | None = None,
                   title: str | None = None) -> dict:
    """Returns {"date", "type", "title", "text", "filename", "date_source"}; raises ValueError with a plain message."""
    text = extract_text(filename, data)
    meta: dict[str, str] = {}
    if text.startswith("id:") or "\n---\n" in text[:400]:           # same header format as the sample documents
        head, sep, body = text.partition("\n---\n")
        if sep:
            meta = dict(l.split(": ", 1) for l in head.strip().splitlines() if ": " in l)
            if "date" in meta or "title" in meta:
                text = body.strip()
            else:
                meta = {}
    src = None
    d = (doc_date or "").strip()
    if d:
        try:
            d = datetime.strptime(d, "%Y-%m-%d").date().isoformat()
        except ValueError:
            raise ValueError(f"{filename}: date must look like 2026-09-14.")
        src = "entered"
    elif meta.get("date") and find_dates(meta["date"]):
        d, src = find_dates(meta["date"])[0], "file header"
    else:
        hdr = re.search(r"^date:\s*(.+)$", text, re.I | re.M)
        found = find_dates(hdr.group(1)) if hdr else []
        if found:
            d, src = found[0], "Date: line"
        else:
            found = find_dates(text)
            if found:
                d, src = found[0], "first date in the text"
    if not d:
        raise ValueError(f"{filename}: no date found. Enter the document's date; the search filters by date, "
                         "so an undated document would never be found.")
    t = (doc_type or meta.get("type") or "").strip().lower() or _guess_type(filename, text)
    return {"date": d, "type": t[:20], "title": (title or meta.get("title") or _guess_title(filename, text)).strip()[:160],
            "text": text, "filename": filename, "date_source": src}
