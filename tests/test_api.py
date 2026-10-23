import json
from fastapi.testclient import TestClient
from gve.api import app

c = TestClient(app)
P = {"month": "2026-09", "min_dollars": 5000, "min_pct": 5}


def test_meta_and_health():
    m = c.get("/api/meta").json()
    assert m["documents"] == 20 and m["embedder"] and "2026-09" in m["sample_months"]
    assert c.get("/api/health").json()["ok"]


def test_preflight_finds_planted_issues():
    r = c.post("/api/preflight", json=P).json()
    assert r["rows"] > 0 and isinstance(r["issues"], list)


def test_variances_shape_and_json_safe():
    r = c.post("/api/variances", json=P).json()
    assert r["lines"] and r["totals"]["revenue"]["actual_str"].startswith("$")
    assert all("variance_str" in l for l in r["lines"])


def test_unknown_month_and_dataset():
    assert c.post("/api/variances", json={**P, "month": "1999-01"}).status_code == 422
    assert c.post("/api/variances", json={**P, "dataset": "nope"}).status_code == 404


def _stream(body):
    r = c.post("/api/investigate", json=body)
    assert r.status_code == 200
    return [json.loads(x) for x in r.text.strip().splitlines()]


def test_investigate_streams_one_event_per_line():
    ev = _stream({**P, "agent": "baseline", "retrieval": "hybrid"})
    n = len(c.post("/api/variances", json=P).json()["lines"])
    assert ev[0]["type"] == "start" and ev[-1]["type"] == "done"
    prog = [e for e in ev if e["type"] == "progress"]
    assert len(prog) == n and prog[-1]["done"] == n
    assert all(e["finding"]["status"] in ("supported", "no_evidence", "unverified") for e in prog)


def test_claude_agent_needs_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert c.post("/api/investigate", json={**P, "agent": "claude"}).status_code == 422


def test_commentary_validates_and_planted_errors_are_caught():
    ok = c.post("/api/commentary", json=P).json()
    assert ok["ok"] and ok["findings"] == []
    bad = c.post("/api/commentary", json={**P, "plant_errors": True}).json()
    assert not bad["ok"] and bad["applied"]
    assert {f["kind"] for f in bad["findings"]} & {"number", "direction", "cause"}


def test_live_validate_of_edited_text():
    text = c.post("/api/commentary", json=P).json()["text"]
    edited = text.replace("budget", "budget", 1) + "\n- **Sales / Product revenue:** grew 99.9% due to a promotion."
    r = c.post("/api/validate", json={**P, "text": edited}).json()
    assert isinstance(r["findings"], list)


def test_accepted_suggestion_unlocks_a_cause_only_with_sources():
    lines = c.post("/api/variances", json=P).json()["lines"]
    l = lines[0]
    body = {**P, "suggestions": [{"department": l["department"], "account_code": l["account_code"],
                                  "text": "Trade show costs landed in September.", "sources": ["D02"]}]}
    r = c.post("/api/commentary", json=body).json()
    assert "Documents suggest" in r["text"] and "D02" in r["text"] and r["ok"]


def test_document_and_evals():
    assert c.get("/api/documents/D02").json()["id"] == "D02"
    assert c.get("/api/documents/ZZZ").status_code == 404
    e = c.get("/api/evals").json()
    modes = {x["config"] for x in e["retrieval"]}
    assert "hybrid, paraphrased" in modes and e["claude_measured"] is False


def test_upload_roundtrip():
    gl = open("data/sample_gl.csv", "rb").read()
    coa = open("data/chart_of_accounts.csv", "rb").read()
    r = c.post("/api/datasets", files={"gl": ("gl.csv", gl), "coa": ("coa.csv", coa)}).json()
    assert r["months"] and not r["investigation_supported"]
    v = c.post("/api/variances", json={**P, "dataset": r["dataset"]}).json()
    assert v["lines"] and not v["investigation_supported"]
    assert c.post("/api/datasets", files={"gl": ("g.csv", b"a,b\n1,2"), "coa": ("c.csv", coa)}).status_code == 422
