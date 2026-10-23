"""HTTP API for the Grounded Variance Explainer.

A thin layer over the engine in this package. No business logic lives here: it parses requests,
calls load/preflight/variance/agent/commentary/validator, and returns JSON.

Run:  uvicorn gve.api:app --reload --port 8000
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import ClaudeInvestigator, KeywordBaseline
from . import mapping as mp
from .tools import AgentTools
from .app_helpers import month_label
from .commentary import build_facts, draft_commentary, inject_errors
from .ingest import parse_document
from .evals import DATA, PARAPHRASES, format_table, retrieval_scores, run_config
from .loader import load_coa, load_gl
from .preflight import run_preflight
from .retrieval import Retriever
from .tools import load_ledger
from .validator import annotate, validate
from .variance import compute_variances, prepare_for_analysis, summarize

ROOT = Path(__file__).resolve().parents[1]
WEB_DIST = ROOT / "web" / "dist"
SAMPLE_COMPANY = "Harborline Goods Co."
SAMPLE_ID = "sample"

app = FastAPI(title="Grounded Variance Explainer", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("GVE_CORS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ shared state
_uploads: dict[str, dict] = {}   # id -> {"gl": df, "coa": df, "name": str}


@lru_cache(maxsize=1)
def base_retriever() -> Retriever:
    r = Retriever(DATA / "docs", mode="hybrid")
    r._dense()  # fit the embedder once, at startup, so request-scoped copies share the index
    return r


# Documents a visitor uploads live in memory, keyed by a random id the browser keeps, so one visitor's
# files are never searched for another. Lost on restart; capped so a public demo cannot fill memory.
MAX_SESSIONS, MAX_USER_DOCS = 20, 50
_sessions: dict[str, dict] = {}   # sid -> {"docs": {id: doc}, "ret": Retriever | None, "n": int}


def _session(sid: str) -> dict:
    s = _sessions.pop(sid, None) or {"docs": {}, "ret": None, "n": 0}
    _sessions[sid] = s                      # most recently used goes last
    while len(_sessions) > MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)))
    return s


def retriever(sid: str = "") -> Retriever:
    """The sample documents, plus whatever this visitor has uploaded."""
    s = _sessions.get(sid)
    if not s or not s["docs"]:
        return base_retriever()
    if s["ret"] is None:
        s["ret"] = base_retriever().with_docs(list(s["docs"].values()))
        s["ret"]._dense()
    return s["ret"]


def SID(x_gve_session: str = Header("", alias="X-GVE-Session")) -> str:
    return x_gve_session[:64]


@lru_cache(maxsize=1)
def ledger() -> pd.DataFrame:
    return load_ledger(DATA / "transactions.csv")


@lru_cache(maxsize=1)
def sample_frames():
    return load_gl(DATA / "sample_gl.csv"), load_coa(DATA / "chart_of_accounts.csv")


def frames(dataset: str):
    if dataset == SAMPLE_ID:
        return sample_frames()
    d = _uploads.get(dataset)
    if not d:
        raise HTTPException(404, "Unknown dataset. Upload files again or use the sample data.")
    return d["gl"], d["coa"]


def clean(o):
    """Make pandas/numpy output JSON-safe (NaN -> None)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "item"):
        return clean(o.item())
    return o


def analysis(dataset: str, month: str, min_dollars: float, min_pct: float, notes=None, suggestions=None):
    gl, coa = frames(dataset)
    if month not in set(gl["month"].dropna().unique()):
        raise HTTPException(422, f"No data for month {month}.")
    c = prepare_for_analysis(gl, coa)
    var = compute_variances(c, month, min_dollars, min_pct)
    summ = summarize(c, month)
    return var, summ, build_facts(var, summ, notes or {}, suggestions or {})


def claude_available() -> bool:
    """True when any model key is set (Claude or Gemini). The name is kept for the API's sake."""
    from .llm import provider
    return provider() is not None


# ------------------------------------------------------------------ request models
class Params(BaseModel):
    dataset: str = SAMPLE_ID
    month: str
    min_dollars: float = Field(5000, ge=0)
    min_pct: float = Field(5.0, ge=0)


class Note(BaseModel):
    department: str
    account_code: str
    text: str = ""


class Suggestion(BaseModel):
    department: str
    account_code: str
    text: str
    sources: list[str] = []


class InvestigateReq(Params):
    agent: Literal["baseline", "claude"] = "baseline"
    retrieval: Literal["bm25", "dense", "hybrid"] = "hybrid"
    date_filter: bool = True


class CommentaryReq(Params):
    notes: list[Note] = []
    suggestions: list[Suggestion] = []
    mode: Literal["offline", "claude"] = "offline"
    plant_errors: bool = False


class ValidateReq(CommentaryReq):
    text: str


def _maps(req):
    notes = {(n.department, n.account_code): n.text for n in req.notes if n.text.strip()}
    sugg = {(s.department, s.account_code): {"text": s.text, "sources": s.sources} for s in req.suggestions}
    return notes, sugg


# ------------------------------------------------------------------ endpoints
@app.get("/api/meta")
def meta(sid: str = Depends(SID)):
    r = retriever(sid)
    gl, _ = sample_frames()
    return {
        "company": SAMPLE_COMPANY,
        "claude_available": claude_available(), "anthropic_available": bool(os.environ.get("ANTHROPIC_API_KEY")), "llm_label": __import__("gve.llm", fromlist=["x"]).provider_label(),
        "embedder": r.embedder_name,
        "documents": len(r.docs), "chunks": len(r.chunks), "uploaded_documents": len(_sessions.get(sid, {}).get("docs", {})),
        "ledger_rows": int(len(ledger())),
        "sample_months": sorted(gl["month"].dropna().unique().tolist()),
        "default_model": (os.environ.get("GVE_GEMINI_MODEL") or "gemini-3.8-flash") if __import__("gve.llm", fromlist=["x"]).provider() == "gemini" else (os.environ.get("GVE_AGENT_MODEL") or os.environ.get("GVE_MODEL") or "claude-sonnet-5-5"),
    }


@app.post("/api/datasets")
async def upload(gl: UploadFile = File(...), coa: UploadFile = File(...)):
    try:
        gl_df = load_gl(io.BytesIO(await gl.read()))
        coa_df = load_coa(io.BytesIO(await coa.read()))
    except ValueError as e:
        raise HTTPException(422, str(e))
    did = uuid.uuid4().hex[:10]
    _uploads[did] = {"gl": gl_df, "coa": coa_df, "name": gl.filename}
    if len(_uploads) > 20:
        _uploads.pop(next(iter(_uploads)))
    return {"dataset": did, "months": sorted(gl_df["month"].dropna().unique().tolist()),
            "investigation_supported": False}


@app.get("/api/datasets/{dataset}/months")
def months(dataset: str):
    gl, _ = frames(dataset)
    return {"months": sorted(gl["month"].dropna().unique().tolist())}


@app.post("/api/preflight")
def preflight(p: Params):
    gl, coa = frames(p.dataset)
    issues = run_preflight(gl, coa, focus_month=p.month)
    return {"issues": [i.to_dict() for i in issues],
            "blockers": sum(i.severity == "blocker" for i in issues),
            "warnings": sum(i.severity == "warning" for i in issues),
            "rows": int(len(gl))}


@app.post("/api/variances")
def variances(p: Params):
    var, summ, facts = analysis(p.dataset, p.month, p.min_dollars, p.min_pct)
    rows = var.to_dict("records")
    return clean({"month": p.month, "label": month_label(p.month), "totals": facts["totals"],
                  "lines": facts["lines"], "all_rows": rows,
                  "investigation_supported": p.dataset == SAMPLE_ID})


@app.post("/api/investigate")
def investigate(req: InvestigateReq, sid: str = Depends(SID)):
    """Streams newline-delimited JSON: one {"type":"progress"} per line, then {"type":"done"}."""
    if req.dataset != SAMPLE_ID:
        raise HTTPException(422, "Investigation needs documents and a transaction ledger; only the sample company has them.")
    if req.agent == "claude" and not claude_available():
        raise HTTPException(422, "Set ANTHROPIC_API_KEY or GEMINI_API_KEY on the server to use the AI agent.")
    _, _, facts = analysis(req.dataset, req.month, req.min_dollars, req.min_pct)
    lines = facts["lines"]
    r = copy.copy(retriever(sid))
    r.mode = req.retrieval

    def events():
        yield json.dumps({"type": "start", "total": len(lines), "agent": req.agent, "retrieval": req.retrieval}) + "\n"
        inv = ClaudeInvestigator() if req.agent == "claude" else KeywordBaseline()
        label = month_label(req.month)
        for i, line in enumerate(lines, 1):
            tools = AgentTools(r, ledger(), req.month, req.date_filter)   # fresh session per line
            try:
                f = inv.investigate(dict(line, month=req.month), label, tools)
            except Exception as e:  # bad key, network, API error
                yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}) + "\n"
                return
            yield json.dumps(clean({"type": "progress", "done": i, "total": len(lines),
                                    "finding": finding_json(f)})) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


def finding_json(f) -> dict:
    return {"line_key": f.line_key, "agent": f.agent, "status": f.status, "explanation": f.explanation,
            "confidence": f.confidence, "evidence": f.evidence, "problems": f.problems,
            "trace": f.trace, "steps": f.steps, "sources": f.sources}


def _doc_row(d: dict, source: str) -> dict:
    return {"id": d["id"], "date": d["date"], "type": d["type"], "title": d["title"], "chars": len(d["text"]),
            "source": source, **({"filename": d.get("filename"), "date_source": d.get("date_source")} if source == "uploaded" else {})}


@app.get("/api/documents")
def list_documents(sid: str = Depends(SID)):
    mine = _sessions.get(sid, {}).get("docs", {})
    rows = [_doc_row(d, "sample") for d in base_retriever().docs.values()] + [_doc_row(d, "uploaded") for d in mine.values()]
    return {"documents": sorted(rows, key=lambda r: (r["date"], r["id"]), reverse=True), "uploaded": len(mine),
            "max_uploaded": MAX_USER_DOCS}


@app.post("/api/documents")
async def add_document(file: UploadFile = File(...), date: str = Form(""), type: str = Form(""), title: str = Form(""),
                       sid: str = Depends(SID)):
    if not sid:
        raise HTTPException(422, "Missing session header.")
    s = _session(sid)
    if len(s["docs"]) >= MAX_USER_DOCS:
        raise HTTPException(422, f"Limit of {MAX_USER_DOCS} uploaded documents reached. Remove some first.")
    try:
        d = parse_document(file.filename or "upload.txt", await file.read(), doc_date=date, doc_type=type, title=title)
    except ValueError as e:
        raise HTTPException(422, str(e))
    s["n"] += 1
    d["id"] = f"D9{s['n']:03d}"          # D-prefix so the verifier's citation check applies to uploads too
    s["docs"][d["id"]] = d
    s["ret"] = None                       # re-index on next use
    retriever(sid)                        # build now, so the first search after an upload is fast
    return _doc_row(d, "uploaded")


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str, sid: str = Depends(SID)):
    s = _sessions.get(sid)
    if not s or doc_id not in s["docs"]:
        raise HTTPException(404, "Only documents you uploaded can be removed.")
    del s["docs"][doc_id]
    s["ret"] = None
    return {"removed": doc_id}


@app.get("/api/documents/{doc_id}")
def document(doc_id: str, sid: str = Depends(SID)):
    d = retriever(sid).docs.get(doc_id)
    if not d:
        raise HTTPException(404, "No such document.")
    return {k: v for k, v in d.items() if k in ("id", "date", "type", "title", "text")}


class SearchReq(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    month: str | None = None
    date_filter: bool = True
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(4, ge=1, le=10)


@app.post("/api/search")
def search(req: SearchReq, sid: str = Depends(SID)):
    """The search the agent uses, on its own, so you can see what it would find."""
    from .retrieval import window_for_month
    lo, hi = window_for_month(req.month) if (req.date_filter and req.month) else (None, None)
    r = retriever(sid)
    hits = r.search(req.query, k=req.k, date_from=lo, date_to=hi, mode=req.mode)
    for h in hits:
        h["uploaded"] = h["doc_id"].startswith("D9")
        h["text"] = h["text"][:400]
    return {"hits": hits, "window": [lo, hi]}


@app.post("/api/commentary")
def commentary(req: CommentaryReq):
    notes, sugg = _maps(req)
    _, _, facts = analysis(req.dataset, req.month, req.min_dollars, req.min_pct, notes, sugg)
    try:
        text = draft_commentary(facts, month_label(req.month),
                                SAMPLE_COMPANY if req.dataset == SAMPLE_ID else "the company", req.mode)
    except Exception as e:
        raise HTTPException(502, f"Drafting failed ({type(e).__name__}): {e}")
    applied: list[str] = []
    if req.plant_errors:
        text, applied = inject_errors(text, facts)
    return _validated(text, facts, applied)


@app.post("/api/validate")
def validate_text(req: ValidateReq):
    notes, sugg = _maps(req)
    _, _, facts = analysis(req.dataset, req.month, req.min_dollars, req.min_pct, notes, sugg)
    return _validated(req.text, facts, [])


def _validated(text: str, facts: dict, applied: list[str]) -> dict:
    found = validate(text, facts)
    return {"text": text, "applied": applied, "findings": [f.to_dict() for f in found],
            "annotated": annotate(text, found), "ok": not found,
            "facts_checked": len(facts["lines"])}


@lru_cache(maxsize=1)
def _eval_payload() -> dict:
    gold = json.loads((DATA / "corpus_key.json").read_text())
    r, led = base_retriever(), ledger()
    month = "2026-09"
    res = []
    for mode in ("bm25", "hybrid"):
        rm = copy.copy(r); rm.mode = mode
        for df in (True, False):
            res.append(run_config(f"baseline, {mode} search, date filter {'on' if df else 'off'}", KeywordBaseline(), df,
                                  month, rm, led, gold))
    configs = [{k: v for k, v in c.items() if k != "findings"} for c in res]
    ret = []
    ret = retrieval_scores(r, gold, month)
    return {"agent_configs": clean(configs), "retrieval": clean(ret),
            "embedder": r.embedder_name, "paraphrase_queries": sum(len(v) for v in PARAPHRASES.values()),
            "claude_measured": False,
            "table": format_table(res)}


@lru_cache(maxsize=1)
def _bench_payload() -> dict:
    from .bench import summary
    return clean(summary(include_claude=False))


@app.get("/api/bench")
def bench_results():
    return _bench_payload()


@app.get("/api/evals")
def evals():
    return _eval_payload()


# ------------------------------------------------------------------ chart-of-accounts mapping
class MapRule(BaseModel):
    source_code: str
    target_code: str


class MapReq(BaseModel):
    entity: str
    agent: Literal["baseline", "claude"] = "baseline"
    rules: list[MapRule] = Field(default_factory=list, max_length=500)


@app.get("/api/mapping/data")
def mapping_data():
    return {"standard": mp.load_standard(), "entities": mp.load_entities()}


@app.post("/api/mapping/run")
def mapping_run(req: MapReq):
    ents = mp.load_entities()
    if req.entity not in ents:
        raise HTTPException(404, "Unknown entity.")
    if req.agent == "claude" and not claude_available():
        raise HTTPException(422, "Set ANTHROPIC_API_KEY or GEMINI_API_KEY on the server to use the AI mapper.")
    std = mp.load_standard()
    mapper = mp.ClaudeMapper(standard=std) if req.agent == "claude" else _sim_mapper()
    try:
        rows = mapper.map_entity(req.entity, ents[req.entity])
    except Exception as e:
        raise HTTPException(502, f"The mapper failed ({type(e).__name__}): {e}")
    rows = mp.apply_rules(rows, [dict(r.model_dump(), entity=req.entity) for r in req.rules], std)
    return {"agent": mapper.name, "rows": rows}


@lru_cache(maxsize=1)
def _sim_mapper():
    return mp.SimilarityMapper(mp.load_standard())


def _measured() -> dict:
    p = DATA / "measured.json"
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=1)
def _mapping_eval():
    s = mp.summary(include_claude=False)
    s["results"] += [dict(r, measured=True) for r in _measured().get("mapping", [])]
    return s


@app.get("/api/mapping/eval")
def mapping_eval():
    return _mapping_eval()


@app.get("/api/health")
def health():
    return {"ok": True}


# ------------------------------------------------------------------ built frontend (single-host deploys)
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = WEB_DIST / path
        return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")
