"""Tests: the engine must catch every planted data issue and every injected AI error."""
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gve import (build_facts, compute_variances, draft_commentary, load_coa, load_gl,
                 prepare_for_analysis, run_preflight, validate)
from gve.commentary import NO_DRIVER, inject_errors
from gve.variance import summarize

FOCUS = "2026-09"


@pytest.fixture(scope="module")
def world():
    gl = load_gl(ROOT / "data/sample_gl.csv")
    coa = load_coa(ROOT / "data/chart_of_accounts.csv")
    key = json.loads((ROOT / "data/answer_key.json").read_text())
    clean = prepare_for_analysis(gl, coa)
    var = compute_variances(clean, FOCUS)
    notes = {("Marketing", "6200"): "Fall trade show landed in September"}
    facts = build_facts(var, summarize(clean, FOCUS), notes)
    return dict(gl=gl, coa=coa, key=key, clean=clean, var=var, facts=facts)


def test_every_planted_data_issue_is_caught(world):
    issues = run_preflight(world["gl"], world["coa"], FOCUS)
    found = {i.check for i in issues}
    planted = {x["type"] for x in world["key"]["planted_data_issues"]}
    assert planted <= found, f"missed: {planted - found}"


def test_unmapped_account_in_focus_month_is_blocker(world):
    issues = run_preflight(world["gl"], world["coa"], FOCUS)
    assert any(i.check == "unmapped_account" and i.severity == "blocker" for i in issues)


def test_unmapped_and_duplicates_excluded_from_analysis(world):
    clean = world["clean"]
    assert "6999" not in set(clean.account_code)
    mk = clean[(clean.department == "Marketing") & (clean.account_code == "6100") & (clean.month == "2026-08")]
    assert len(mk) == 1


def test_planted_variances_are_material_and_right_direction(world):
    v = world["var"].set_index(["department", "account_code"])
    for p in world["key"]["planted_variances"]:
        row = v.loc[(p["department"], p["account_code"])]
        assert (row.variance > 0) == (p["multiplier"] > 1)
    # the big ones must be flagged material
    for k in [("Marketing", "6200"), ("G&A", "6800"), ("Sales", "4000")]:
        assert v.loc[k].material


def test_favorability_follows_account_type(world):
    v = world["var"].set_index(["department", "account_code"])
    assert not v.loc[("Sales", "4000")].favorable      # revenue under budget
    assert v.loc[("Operations", "5000")].favorable     # expense under budget


def test_clean_offline_draft_passes_validation(world):
    text = draft_commentary(world["facts"], "September 2026", mode="offline")
    assert validate(text, world["facts"]) == []


def test_offline_draft_never_invents_cause(world):
    text = draft_commentary(world["facts"], "September 2026", mode="offline")
    assert text.count(NO_DRIVER) == sum(1 for f in world["facts"]["lines"] if not f["note"])


def test_validator_catches_each_injected_error(world):
    text = draft_commentary(world["facts"], "September 2026", mode="offline")
    bad, applied = inject_errors(text, world["facts"])
    assert len(applied) == 3
    kinds = {f.kind for f in validate(bad, world["facts"])}
    assert {"number", "direction", "cause"} <= kinds


def test_validator_catches_missing_bullet(world):
    text = draft_commentary(world["facts"], "September 2026", mode="offline")
    trimmed = "\n".join(l for l in text.splitlines() if "G&A / Professional Fees" not in l)
    assert any(f.kind == "coverage" for f in validate(trimmed, world["facts"]))


def test_validator_catches_wrong_overview_number(world):
    text = draft_commentary(world["facts"], "September 2026", mode="offline")
    first = text.splitlines()[0]
    tampered = text.replace(first, first.replace("$881.0K", "$900.0K"))
    assert any(f.where == "Overview" for f in validate(tampered, world["facts"]))


def test_validator_catches_wrong_favorability(world):
    text = draft_commentary(world["facts"], "September 2026", mode="offline")
    tampered = text.replace("(unfavorable)", "(favorable)", 1)
    assert any(f.kind == "direction" for f in validate(tampered, world["facts"]))


def test_claude_path_sends_only_computed_facts(world):
    """No API key needed: mock the client and check what would be sent."""
    fake = mock.MagicMock()
    fake.messages.create.return_value.content = [mock.MagicMock(type="text", text="ok")]
    with mock.patch("anthropic.Anthropic", return_value=fake):
        out = draft_commentary(world["facts"], "September 2026", "Acme", mode="claude")
    assert out == "ok"
    kw = fake.messages.create.call_args.kwargs
    prompt = kw["messages"][0]["content"]
    assert "$47.7K" in prompt and "Analyst note: Fall trade show" in prompt
    assert "Analyst note: NONE" in prompt
    assert "ONLY the numbers" in kw["system"]
    assert "6999" not in prompt  # excluded account never reaches the model


def test_loader_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("month,dept,actual\n2026-01,A,1\n")
    with pytest.raises(ValueError, match="Missing required"):
        load_gl(p)
