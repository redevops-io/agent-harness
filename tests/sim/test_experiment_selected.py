# SPDX-License-Identifier: AGPL-3.0-or-later
"""The bookkeeping of the complete-versus-selected experiment: what the proposer read is checked against the record, and spending is counted."""
import json

from agent_harness.sim.experiment_selected import MEASURED, Meter, MeteredClient, history_report


def _attempt(n: int, **over):
    a = {"round": n, "sample": 0, "what": f"edit {n}", "rationale": f"because {n}", "files": ["systemprompt.md"], "predicted": ["t1"], "provenance": {"run": "r"},
         "accepted": n % 2 == 0, "backfilled": False, "fixed": ["t1"], "missed": [], "regressed": [], "passed": 3, "total": 5, "candidate_harness": f"h{n}", "evaluated": True}
    a.update(over)
    return a


RECORD = {"problems": [{"id": "P1", "attempts": [_attempt(1), _attempt(2)]}, {"id": "P2", "attempts": [_attempt(3)]}]}


def _text(record):
    return json.dumps(record)


def test_the_complete_record_preserves_everything():
    r = history_report(RECORD, _text(RECORD))
    assert r["attempts"] == r["attempts_shown"] == r["attempts_with_identical_measured_fields"] == 3
    assert r["attempt_descriptions_verbatim"] == r["rationales_verbatim"] == 3 and r["problems_not_shown"] == []


def test_compaction_that_shortens_prose_keeps_the_measured_fields_and_says_what_it_shortened():
    view = json.loads(_text(RECORD))
    for p in view["problems"]:
        for a in p["attempts"]:
            a["what"] = a["what"][:3]
            a.pop("rationale")
    r = history_report(RECORD, _text(view))
    assert r["attempts_with_identical_measured_fields"] == 3
    assert r["attempt_descriptions_verbatim"] == 0 and r["rationales_verbatim"] == 0 and r["rationales_in_record"] == 3


def test_a_changed_measurement_a_dropped_attempt_and_a_missing_problem_are_each_counted():
    view = json.loads(_text(RECORD))
    view["problems"][0]["attempts"][0]["missed"] = ["t9"]          # a measurement changed
    view["problems"][0]["attempts"].pop(1)                        # an attempt dropped
    view["problems"].pop(1)                                       # a problem gone
    r = history_report(RECORD, _text(view))
    assert r["attempts_shown"] == 1 and r["attempts_with_identical_measured_fields"] == 0 and r["problems_not_shown"] == ["P2"]


def test_text_that_is_not_the_record_preserves_nothing():
    r = history_report(RECORD, "not json")
    assert r["attempts_shown"] == 0 and r["problems_not_shown"] == ["P1", "P2"]


def test_the_meter_counts_what_the_server_reports_and_what_it_does_not():
    class Inner:
        def complete(self, messages, tools=None):
            return {"message": {"content": "x"}, "usage": {"prompt_tokens": 7, "completion_tokens": 3}} if messages else {"message": {"content": ""}}

    m = Meter()
    c = MeteredClient(Inner(), m)
    c.complete([{"role": "user", "content": "a"}])
    c.complete([])
    t = m.total()
    assert (t["calls"], t["prompt_tokens"], t["completion_tokens"], t["calls_without_usage"]) == (2, 7, 3, 1)


def test_the_measured_fields_are_the_ones_a_view_may_never_change():
    assert {"predicted", "fixed", "missed", "regressed", "passed", "total", "accepted"} <= set(MEASURED)


def test_a_carried_attempt_and_a_target_attempt_with_the_same_round_are_not_confused():
    a = _attempt(1, provenance={"run": "source"}, candidate_harness="hS")
    b = _attempt(1, provenance={"run": "target"}, candidate_harness="hT", fixed=["t7"])
    rec = {"problems": [{"id": "P1", "attempts": [a, b]}]}
    r = history_report(rec, json.dumps(rec))
    assert r["attempts"] == r["attempts_shown"] == r["attempts_with_identical_measured_fields"] == 2
