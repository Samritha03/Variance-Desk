import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gve.api import app
from gve.mapping import (ClaudeMapper, SimilarityMapper, apply_rules, evaluate, load_entities, load_key, load_standard, run_all)

STD = load_standard()
KEY = load_key()


def fake(mappings):
    class L:
        def create(self, system, messages, tools):
            assert tools[0]["name"] == "propose_mappings" and "Do NOT assume the source code" in system
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="1", name="propose_mappings", input={"mappings": mappings})])
    return ClaudeMapper(L(), STD)


ACCTS = [{"code": "40100", "name": "Merchandise Sales", "type": "revenue", "description": ""},
         {"code": "699500", "name": "Suspense / Clearing", "type": "expense", "description": ""},
         {"code": "1", "name": "Mystery", "type": "expense", "description": ""},
         {"code": "2", "name": "Wrong kind", "type": "expense", "description": ""},
         {"code": "3", "name": "Skipped", "type": "expense", "description": ""},
         {"code": "4", "name": "Unsure", "type": "expense", "description": ""}]


def test_claude_output_is_validated_not_trusted():
    m = fake([
        {"source_code": "40100", "target_code": "4000", "confidence": "high", "reason": "sales of goods"},
        {"source_code": "699500", "target_code": "NONE", "confidence": "high", "reason": "clearing"},
        {"source_code": "1", "target_code": "9999", "confidence": "high", "reason": "made up"},
        {"source_code": "2", "target_code": "4000", "confidence": "high", "reason": "type clash"},
        {"source_code": "4", "target_code": "6800", "confidence": "medium", "reason": "maybe"},
    ])
    r = {x["source_code"]: x for x in m.map_entity("E", ACCTS)}
    assert (r["40100"]["status"], r["40100"]["target_code"]) == ("auto", "4000")
    assert r["699500"]["status"] == "review" and r["699500"]["target_code"] is None      # NONE is never auto-applied
    assert r["1"]["status"] == "review" and "not in the standard chart" in r["1"]["reason"]
    assert r["2"]["status"] == "review" and "revenue account" in r["2"]["reason"]
    assert r["3"]["status"] == "review" and "no answer" in r["3"]["reason"]
    assert r["4"]["status"] == "review" and r["4"]["target_code"] == "6800"              # medium confidence goes to a person


def test_rules_win_and_are_validated():
    rows = fake([{"source_code": "40100", "target_code": "4000", "confidence": "high", "reason": "x"}]).map_entity("E", ACCTS[:1])
    out = apply_rules(rows, [{"entity": "E", "source_code": "40100", "target_code": "4200"}], STD)
    assert out[0]["target_code"] == "4200" and out[0]["status"] == "rule"
    out = apply_rules(rows, [{"entity": "E", "source_code": "40100", "target_code": "ZZZ"}], STD)
    assert out[0]["target_code"] == "4000"                                               # an invalid rule is ignored


def test_code_clash_is_not_matched_by_code():
    rows = {(r["entity"], r["source_code"]): r for r in run_all(SimilarityMapper(STD))}
    hosting = rows[("Bayside Software", "5000")]       # same code as standard COGS, but it is hosting
    assert hosting["target_code"] != "5000" or hosting["status"] == "review"


def test_baseline_never_applies_a_wrong_mapping_automatically_on_this_set():
    s = evaluate(run_all(SimilarityMapper(STD)), KEY)
    assert s["n"] == len(KEY) and s["wrong_auto"] == 0 and s["wrong"] > 0 and s["wrong_caught"] == s["wrong"]


def test_eval_counts_a_wrong_auto_mapping():
    rows = run_all(SimilarityMapper(STD))
    bad = [dict(r, target_code="7300", status="auto") if r["source_code"] == "40100" else r for r in rows]
    assert evaluate(bad, KEY)["wrong_auto"] >= 1


def test_key_is_consistent():
    codes = {s["code"] for s in STD} | {"NONE"}
    ents = load_entities()
    assert sum(len(v) for v in ents.values()) == len(KEY) and all(g["target"] in codes for g in KEY.values())


def test_endpoints():
    c = TestClient(app)
    d = c.get("/api/mapping/data").json()
    assert set(d["entities"]) == set(load_entities()) and len(d["standard"]) == len(STD)
    r = c.post("/api/mapping/run", json={"entity": "Northwind Retail"}).json()
    assert len(r["rows"]) == len(d["entities"]["Northwind Retail"]) and r["agent"] == "similarity-baseline"
    one = r["rows"][0]
    r2 = c.post("/api/mapping/run", json={"entity": "Northwind Retail", "rules": [{"source_code": one["source_code"], "target_code": "7300"}]}).json()
    assert r2["rows"][0]["status"] == "rule" and r2["rows"][0]["target_code"] == "7300"
    assert c.post("/api/mapping/run", json={"entity": "Nope"}).status_code == 404
    assert c.post("/api/mapping/run", json={"entity": "Northwind Retail", "agent": "claude"}).status_code in (200, 422)
    assert c.get("/api/mapping/eval").json()["results"][0]["mapper"] == "similarity-baseline"


def test_empty_replies_are_retried_and_then_explained():
    calls = []

    class L:
        def create(self, system, messages, tools):
            calls.append(1)
            if len(calls) == 1:
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="[no tool call; finishReason=MAX_TOKENS]")])
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="1", name="propose_mappings", input={"mappings": [
                {"source_code": "40100", "target_code": "4000", "confidence": "high", "reason": "sales"}]})])
    r = ClaudeMapper(L(), STD).map_entity("E", ACCTS[:1])
    assert len(calls) == 2 and r[0]["status"] == "auto"
    calls.clear()

    class Dead:
        def create(self, system, messages, tools):
            calls.append(1)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="[no tool call; finishReason=MAX_TOKENS]")])
    r = ClaudeMapper(Dead(), STD).map_entity("E", ACCTS[:1])
    assert len(calls) == 3 and "MAX_TOKENS" in r[0]["reason"] and r[0]["status"] == "review"


def test_big_entities_are_sent_in_batches():
    seen = []

    class L:
        def create(self, system, messages, tools):
            seen.append(len(messages[0]["content"].split("one entry each:\n")[1].splitlines()))
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="1", name="propose_mappings", input={"mappings": []})])
    big = [{"code": str(i), "name": f"A{i}", "type": "expense", "description": ""} for i in range(25)]
    m = ClaudeMapper(L(), STD); m.TRIES = 1
    m.map_entity("E", big)
    assert seen == [12, 12, 1]
