"""The Gemini adapter, tested against a scripted HTTP server (no key, no network)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gve import llm as L
from gve.agent import ClaudeInvestigator
from gve.evals import material_lines
from gve.mapping import ClaudeMapper, load_standard
from gve.retrieval import Retriever
from gve.tools import TOOL_SCHEMAS, AgentTools, load_ledger

DATA = ROOT / "data"


def server(*replies):
    """A fake Gemini endpoint that returns the given (status, json) in order and records requests."""
    seen, queue = [], list(replies)

    def handler(req: httpx.Request):
        seen.append(json.loads(req.content)); assert req.headers["x-goog-api-key"] == "k"
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler), seen


def call(name, args, sig=None):
    p = {"functionCall": {"name": name, "args": args}}
    if sig: p["thoughtSignature"] = sig
    return (200, {"candidates": [{"content": {"parts": [p]}}]})


def gem(transport, **kw):
    return L.GeminiLLM(model="m", key="k", transport=transport, min_interval=0, sleep=lambda s: None, **kw)


def test_schema_and_tool_config_translation():
    req = L.to_gemini("sys", [{"role": "user", "content": "hi"}], TOOL_SCHEMAS)
    decl = {d["name"]: d for d in req["tools"][0]["function_declarations"]}
    p = decl["record_finding"]["parameters"]
    assert p["type"] == "OBJECT" and p["properties"]["status"]["enum"] == ["supported", "no_evidence"]
    assert p["properties"]["evidence"]["items"]["type"] == "OBJECT" and "tool_config" not in req
    one = L.to_gemini("s", [{"role": "user", "content": "x"}], TOOL_SCHEMAS[:1])
    assert one["tool_config"]["function_calling_config"]["mode"] == "ANY"


def test_tool_results_are_matched_to_calls_and_signatures_round_trip():
    t, seen = server(call("search_documents", {"query": "q"}, sig="SIG1"), call("record_finding", {"status": "no_evidence", "explanation": "x", "confidence": "low", "evidence": []}))
    g = gem(t)
    msgs = [{"role": "user", "content": "go"}]
    r1 = g.create("s", msgs, TOOL_SCHEMAS)
    assert r1.content[0].type == "tool_use" and r1.content[0].raw["thoughtSignature"] == "SIG1"
    msgs += [{"role": "assistant", "content": r1.content},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": r1.content[0].id, "content": json.dumps({"results": []})}]}]
    g.create("s", msgs, TOOL_SCHEMAS)
    sent = seen[1]["contents"]
    assert sent[1]["role"] == "model" and sent[1]["parts"][0]["thoughtSignature"] == "SIG1"
    fr = sent[2]["parts"][0]["functionResponse"]
    assert fr["name"] == "search_documents" and fr["response"] == {"results": []}


def test_retries_rate_limits_then_gives_a_plain_error():
    t, seen = server((429, {"error": {"message": "slow down"}}), call("record_finding", {"status": "no_evidence"}))
    assert gem(t).create("s", [{"role": "user", "content": "x"}], TOOL_SCHEMAS).content[0].name == "record_finding" and len(seen) == 2
    t, _ = server((400, {"error": {"message": "API key not valid"}}))
    with pytest.raises(RuntimeError, match="Gemini API error 400: API key not valid.*GVE_GEMINI_MODEL"):
        gem(t).create("s", [{"role": "user", "content": "x"}], TOOL_SCHEMAS)


def test_blocked_prompt_does_not_crash():
    t, _ = server((200, {"promptFeedback": {"blockReason": "SAFETY"}}))
    out = gem(t).create("s", [{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    assert out.content[0].type == "text" and "SAFETY" in out.content[0].text


def test_full_investigation_loop_runs_on_gemini_and_is_grounded():
    ret, led = Retriever(DATA / "docs"), load_ledger(DATA / "transactions.csv")
    line = dict({f"{l['department']}|{l['account_code']}": l for l in material_lines("2026-09")}["G&A|6800"], month="2026-09")
    probe = AgentTools(ret, led, "2026-09").search_documents("audit fee invoice")["results"]
    quote = next(r for r in probe if r["doc_id"] == "D12")["text"].split(".")[0].strip()
    t, _ = server(call("search_documents", {"query": "audit fee invoice"}),
                  call("record_finding", {"status": "supported", "explanation": "Audit invoice.", "confidence": "high",
                                          "evidence": [{"source_id": "D12", "quote": quote}]}))
    inv = ClaudeInvestigator(gem(t))
    f = inv.investigate(line, "September 2026", AgentTools(ret, led, "2026-09"))
    assert inv.name == "gemini-agent" and f.status == "supported" and f.sources == ["D12"] and f.problems == []


def test_mapper_runs_on_gemini_and_names_itself():
    t, _ = server(call("propose_mappings", {"mappings": [{"source_code": "40100", "target_code": "4000", "confidence": "high", "reason": "sales"}]}))
    m = ClaudeMapper(gem(t), load_standard())
    rows = m.map_entity("E", [{"code": "40100", "name": "Merchandise Sales", "type": "revenue", "description": ""}])
    assert m.name == "gemini-mapper" and rows[0]["status"] == "auto"


def test_provider_choice(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GVE_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    assert L.provider() is None
    monkeypatch.setenv("GEMINI_API_KEY", "g"); assert L.provider() == "gemini" and L.provider_label() == "Gemini"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a"); assert L.provider() == "anthropic"
    monkeypatch.setenv("GVE_PROVIDER", "gemini"); assert L.provider() == "gemini"


def test_writer_runs_on_gemini_and_sends_no_tools():
    tr, seen = server((200, {"candidates": [{"content": {"parts": [{"text": "Revenue was $1."}]}, "finishReason": "STOP"}]}))
    text = L.write_text("sys", "facts", gem(tr))
    assert text == "Revenue was $1."
    assert "tools" not in seen[0] and "tool_config" not in seen[0]


def test_writer_raises_on_empty_gemini_answer():
    tr, _ = server((200, {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}))
    with pytest.raises(RuntimeError):
        L.write_text("sys", "facts", gem(tr))
