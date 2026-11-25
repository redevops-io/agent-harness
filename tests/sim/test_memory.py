# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Self-Improvement Memory record: what it must never lose or let anyone rewrite."""
import json

import pytest

from agent_harness.sim import MemoryCondition, Provenance, SelfImprovementMemory, SIMError

PROV = Provenance(domain="coding", benchmark="bench-a", policy="model-x@1", optimizer="opt@1", evaluator="tests@1", run="src")


def _mem():
    m = SelfImprovementMemory(run="src")
    p = m.open_problem("premature completion", 1)
    h = m.open_attempt(p.id, round=1, sample=0, what="add a completion gate", files=["systemprompt.md"], rationale="it stops early",
                       predicted=["t1", "t2"], provenance=PROV, candidate_harness="h1")
    m.close_attempt(h, accepted=True, passed_tasks=["t1", "t3"], failed_tasks=["t2", "t4"], previously_passed=["t3", "t4"])
    return m, p


def test_fixed_and_missed_come_from_the_prediction_on_record_and_a_regression_is_recorded():
    m, p = _mem()
    a = p.attempts[0]
    assert a.predicted == ("t1", "t2") and a.fixed == ("t1",) and a.missed == ("t2",) and a.regressed == ("t4",)
    assert (a.passed, a.total, a.accepted, a.evaluated) == (2, 4, True, True)


def test_an_outcome_cannot_be_written_twice_or_without_a_prediction_on_record():
    m, p = _mem()
    h = m.open_attempt(p.id, round=2, sample=0, what="tighten the gate", files=["systemprompt.md"], rationale="r", predicted=["t2"], provenance=PROV)
    before = json.dumps(m.to_json(), sort_keys=True)
    assert '"evaluated": false' in before and '"predicted": ["t2"]' in before          # the prediction exists before any outcome
    m.close_attempt(h, accepted=False, passed_tasks=[], failed_tasks=["t2"])
    with pytest.raises(SIMError, match="not open"):
        m.close_attempt(h, accepted=True, passed_tasks=["t2"], failed_tasks=[])           # no second, better outcome
    other = SelfImprovementMemory(run="x")
    with pytest.raises(SIMError):
        other.close_attempt(h, accepted=True, passed_tasks=[], failed_tasks=[])
    with pytest.raises(SIMError, match="both passed and failed"):
        m.close_attempt(m.open_attempt(p.id, round=3, sample=0, what="w", files=[], rationale="", predicted=[], provenance=PROV),
                        accepted=False, passed_tasks=["t1"], failed_tasks=["t1"])


def test_there_is_no_way_to_delete_history_and_a_merge_keeps_every_attempt_with_an_alias():
    m, p = _mem()
    assert not [n for n in dir(m) if n.startswith(("delete", "remove", "drop", "clear", "forget"))]
    q = m.open_problem("fragile tool-call parsing", 2)
    m.close_attempt(m.open_attempt(q.id, round=2, sample=1, what="repair the parser", files=["skills/parse/SKILL.md"], rationale="r", predicted=["t4"], provenance=PROV),
                    accepted=False, passed_tasks=[], failed_tasks=["t4"])
    merged = m.merge(q.id, p.id)
    assert [a.what for a in merged.attempts] == ["add a completion gate", "repair the parser"]
    assert m.aliases == {q.id: p.id} and m.problem(q.id) is merged and len(m.problems) == 1


def test_identical_root_cause_text_is_one_entry_and_a_different_text_is_a_new_one():
    m, p = _mem()
    assert m.open_problem("  premature   completion ", 5) is p and p.opened_round == 1
    assert m.open_problem("premature completion on long tasks", 5) is not p
    doc = m.to_json()
    doc["problems"].append({"id": "P9", "name": "premature completion", "opened_round": 0, "attempts": []})
    doc["next_id"] = 10
    loaded = SelfImprovementMemory.from_json(doc)
    assert loaded.merge_identical() == 1 and loaded.aliases == {"P9": "P1"} and loaded.problem("P9").opened_round == 0


def test_the_three_memory_conditions_are_views_of_one_complete_record():
    m, _ = _mem()
    assert m.render(MemoryCondition.NONE) == ""
    only = json.loads(m.render(MemoryCondition.PROBLEM_ONLY))
    assert only == {"problems": [{"id": "P1", "name": "premature completion"}]}          # no edit, outcome or provenance
    full = json.loads(m.render(MemoryCondition.FULL))
    assert full["problems"][0]["attempts"][0]["rationale"] == "it stops early" and full == m.to_json()


def test_a_saved_record_round_trips_and_carrying_it_into_a_target_run_keeps_ids_and_names_the_source(tmp_path):
    m, p = _mem()
    m.save(tmp_path / "sim.json")
    again = SelfImprovementMemory.load(tmp_path / "sim.json")
    assert again.to_json() == m.to_json()
    target = m.carry_into("tgt", domain="coding")
    assert target.carried_from == {"arm": "src", "domain": "coding", "into": "tgt"} and target.problem("P1").attempts == p.attempts
    target.open_problem("a target-only problem", 1)
    assert len(m.problems) == 1                                                            # the source record is not touched
    open_one = SelfImprovementMemory(run="s")
    open_one.open_attempt(open_one.open_problem("x", 1).id, round=1, sample=0, what="w", files=[], rationale="", predicted=[], provenance=PROV)
    with pytest.raises(SIMError, match="still open"):
        open_one.carry_into("t")
