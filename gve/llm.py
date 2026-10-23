"""Which language model runs the agent, the follow-up questions and the mapper.

Everything else in the project talks to one tiny interface, llm.create(system, messages, tools), using
Anthropic-style messages and tool definitions. AnthropicLLM (in agent.py) is that interface for Claude.
GeminiLLM below is the same interface for Google's Gemini API, so the agent can be run and measured with a
free Gemini key. Results are then about Gemini, and are labelled that way.

Pick with GVE_PROVIDER=anthropic|gemini, or just set one key:
  ANTHROPIC_API_KEY                    -> Claude
  GEMINI_API_KEY (or GOOGLE_API_KEY)   -> Gemini   (model: GVE_GEMINI_MODEL, default gemini-2.5-flash)
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def _gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def provider() -> str | None:
    p = (os.environ.get("GVE_PROVIDER") or "").lower()
    if p in ("anthropic", "claude"):
        return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else None
    if p == "gemini":
        return "gemini" if _gemini_key() else None
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "gemini" if _gemini_key() else None


def provider_label() -> str:
    return {"anthropic": "Claude", "gemini": "Gemini"}.get(provider() or "", "Claude")


def make_llm():
    p = provider()
    if p == "gemini":
        return GeminiLLM()
    from .agent import AnthropicLLM
    return AnthropicLLM()


def agent_name(llm=None) -> str:
    label = getattr(llm, "label", None) or provider_label()
    return f"{label.lower()}-agent"


# ------------------------------------------------------------------ Gemini
def _schema(s):
    """Anthropic/JSON-schema tool parameters -> the subset Gemini accepts."""
    if isinstance(s, list):
        return [_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k in ("type",) and isinstance(v, str):
            out[k] = v.upper()
        elif k == "properties":
            out[k] = {n: _schema(p) for n, p in v.items()}
        elif k in ("items", "enum", "required", "description", "format"):
            out[k] = _schema(v) if k == "items" else v
    return out


def to_gemini(system: str, messages: list[dict], tools: list[dict]) -> dict:
    names: dict[str, str] = {}           # tool_use id -> tool name, so results can be matched back
    contents = []
    for m in messages:
        c = m["content"]
        if m["role"] == "assistant":
            parts = []
            for b in c:
                t = getattr(b, "type", "")
                if t == "text":
                    if getattr(b, "text", ""):
                        parts.append({"text": b.text})
                elif t == "tool_use":
                    names[b.id] = b.name
                    parts.append(getattr(b, "raw", None) or {"functionCall": {"name": b.name, "args": dict(b.input)}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif isinstance(c, str):
            contents.append({"role": "user", "parts": [{"text": c}]})
        else:
            parts = []
            for r in c:
                try:
                    body = json.loads(r["content"])
                except Exception:
                    body = r["content"]
                parts.append({"functionResponse": {"name": names.get(r["tool_use_id"], "tool"),
                                                   "response": body if isinstance(body, dict) else {"result": body}}})
            contents.append({"role": "user", "parts": parts})
    req = {"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
           "generationConfig": {"temperature": 0.2, "maxOutputTokens": 16384}}
    if tools:
        req["tools"] = [{"function_declarations": [{"name": t["name"], "description": t["description"],
                                                    "parameters": _schema(t["input_schema"])} for t in tools]}]
    if len(tools) == 1:                  # one tool offered: make the model call it
        req["tool_config"] = {"function_calling_config": {"mode": "ANY", "allowed_function_names": [tools[0]["name"]]}}
    return req


def from_gemini(data: dict, start: int = 0) -> SimpleNamespace:
    cands = data.get("candidates") or []
    blocks = []
    if not cands:
        why = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"[no answer: {why}]")])
    parts = (cands[0].get("content") or {}).get("parts") or []
    for i, p in enumerate(parts):
        if "functionCall" in p:
            fc = p["functionCall"]
            blocks.append(SimpleNamespace(type="tool_use", id=f"call_{start + i}", name=fc["name"], input=fc.get("args") or {}, raw=p))
        elif p.get("text") and not p.get("thought"):
            blocks.append(SimpleNamespace(type="text", text=p["text"]))
    if not any(b.type == "tool_use" for b in blocks):          # say why, so an empty answer can be diagnosed
        blocks.append(SimpleNamespace(type="text", text=f"[no tool call; finishReason={cands[0].get('finishReason', '?')}]"))
    return SimpleNamespace(content=blocks)


class GeminiLLM:
    label = "Gemini"

    def __init__(self, model: str | None = None, key: str | None = None, transport=None, min_interval: float | None = None,
                 sleep=time.sleep):
        self.model = model or os.environ.get("GVE_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self._key = key or _gemini_key()
        self._transport, self._sleep, self._n = transport, sleep, 0
        self.min_interval = float(os.environ.get("GVE_LLM_MIN_INTERVAL", "4")) if min_interval is None else min_interval
        self._last = 0.0

    def create(self, system, messages, tools):
        import httpx
        body = to_gemini(system, messages, tools)
        with httpx.Client(timeout=120, transport=self._transport) as c:
            for attempt in range(6):
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    self._sleep(wait)
                self._last = time.monotonic()
                r = c.post(GEMINI_URL.format(model=self.model), json=body, headers={"x-goog-api-key": self._key})
                if r.status_code in (429, 500, 503) and attempt < 5:
                    self._sleep(min(float(r.headers.get("retry-after", 0) or 0) or 5 * 2 ** attempt, 60))
                    continue
                if r.status_code >= 400:
                    try:
                        msg = r.json()["error"]["message"]
                    except Exception:
                        msg = r.text[:200]
                    hint = " Check the key, and the model name in GVE_GEMINI_MODEL." if r.status_code in (400, 401, 403, 404) else ""
                    raise RuntimeError(f"Gemini API error {r.status_code}: {msg[:300]}.{hint}")
                out = from_gemini(r.json(), self._n)
                self._n += len(out.content)
                return out
        raise RuntimeError("Gemini API kept returning errors (rate limit?). Try again in a minute.")


def write_text(system: str, user: str, llm=None) -> str:
    """One plain-text completion (no tools) on whichever provider is configured. Used by the commentary writer."""
    llm = llm or make_llm()
    if isinstance(llm, GeminiLLM):
        resp = llm.create(system, [{"role": "user", "content": user}], [])
        texts = [b.text for b in resp.content if getattr(b, "type", "") == "text" and not b.text.startswith("[no tool call;")]
        out = "".join(texts).strip()
        if not out or out.startswith("[no answer"):
            raise RuntimeError(out or "Gemini returned an empty answer.")
        return out
    msg = llm.client.messages.create(model=llm.model, max_tokens=1200, temperature=0.2, system=system,
                                     messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def main():
    """python -m gve.llm : one tiny call to check the key and model before a long run."""
    p = provider()
    if not p:
        raise SystemExit("No key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY (or GVE_PROVIDER to choose).")
    ping = {"name": "ping", "description": "Reply by calling this with ok=true.",
            "input_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}
    llm = make_llm()
    resp = llm.create("You are a connectivity check.", [{"role": "user", "content": "Call ping."}], [ping])
    called = any(getattr(b, "type", "") == "tool_use" and b.name == "ping" for b in resp.content)
    print(f"{provider_label()} ({getattr(llm, 'model', '?')}): reached the API, tool call {'worked' if called else 'did NOT happen (the agent needs tool calling)'}.")


if __name__ == "__main__":
    main()
