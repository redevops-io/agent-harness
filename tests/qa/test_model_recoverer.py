# SPDX-License-Identifier: AGPL-3.0-or-later
"""The model recoverer on its own (no runner). The strict gate on a reply, and every hard bound proven at its
boundary: calls, tokens, candidates, wall time, response size, the kill switch (before the call and after the response),
unknown usage, and the opt-in default. Fake client only."""
import json

import pytest

from agent_harness.qa import recovery_model as RM, records as R
from agent_harness.qa.errors import RecoveryPayloadRejected, SpecInvalid
from agent_harness.qa.ports import RecoveryContext
from agent_harness.qa.recovery import KillSwitch

from ._fx_runner import FakeClock
from ._fx_model import (AdapterSnapshots, FakeModelClient, Reply, SAVE, SAVE_BY_ID, StaticSnapshots, array, good, opt_in)
from .test_recovery import observation, step

NODES = StaticSnapshots(RM.SnapshotNode(role="button", name="Save", test_id="save"),
                        RM.SnapshotNode(role="button", name="Delete", test_id="delete"))


def make(*replies, limits=None, kill=None, snapshots=None, ledger=None):
    clock = FakeClock()
    client = FakeModelClient(clock, *replies)
    rec = RM.ModelCandidateRecoverer(client=client, snapshots=snapshots if snapshots is not None else NODES, clock=clock,
                                     limits=limits if limits is not None else opt_in(), kill_switch=kill, ledger=ledger)
    return rec, client, clock


def ctx(step_id="s-x", **kw):
    st = step(step_id=step_id, **kw)
    return RecoveryContext(step=st, observation=observation(st.step_id))


def propose(rec, **kw):
    return rec.propose(ctx(**kw))


def keys(cands):
    return [(c.role, c.name, c.label, c.test_id) for c in cands]


# =====================================================================================================================
# a good reply
# =====================================================================================================================

def test_a_wellformed_reply_becomes_strict_model_candidates_in_a_fixed_order():
    rec, client, _ = make(good({"role": "button", "name": "Save", "test_id": "save"}, {"name": "Save", "role": "button"}))
    got = propose(rec)
    assert client.calls == 1 and rec.reason is None and not rec.halted
    assert all(type(c) is R.LocatorCandidate and c.strategy is R.LocatorStrategy.MODEL_CANDIDATE for c in got)
    assert keys(got) == [("button", "Save", None, None), ("button", "Save", None, "save")]


def test_the_order_repetition_and_case_of_a_reply_change_nothing():
    a = {"role": "button", "name": "Save"}
    b = {"role": "button", "name": "Save", "test_id": "save"}
    c = {"label": "Save form"}
    base = keys(propose(make(good(a, b, c))[0]))
    for text in (array(c, b, a), array(b, b, a, c), array(c, a, b)):
        assert keys(propose(make(Reply(text=text))[0])) == base
    same_but_for_case = propose(make(good({"role": "BUTTON", "name": " save "}, {"role": "button", "name": "Save"}))[0])
    assert len(same_but_for_case) == 1                                      # equal after normalisation: one proposal


def test_null_values_are_the_same_as_absent_keys():
    got = propose(make(good({"role": "button", "name": "Save", "label": None, "test_id": None}))[0])
    assert keys(got) == [("button", "Save", None, None)]


# =====================================================================================================================
# the strict gate: anything that is not a locator is rejected (an integrity event, as in deterministic recovery)
# =====================================================================================================================

REJECTED_ITEMS = [
    {"role": "button", "name": "Save", "expected": "Saved"},                 # an expected value
    {"role": "button", "name": "Save", "url": "https://evil.example/"},      # a URL field
    {"role": "button", "name": "Save", "action": "click"},                   # an action
    {"role": "button", "name": "Save", "assertion": {"op": "EQUALS"}},       # an assertion
    {"role": "button", "name": "Save", "predicate": {"op": "EQUALS", "expected": "Deleted"}},
    {"role": "button", "name": "Save", "script": "document.cookie"},
    {"role": "button", "name": "Save", "confidence": 0.99},                  # advisory fields are not accepted either
    {"role": "button", "name": "Save", "justification": "it is the right one"},
    {"role": "button", "name": "Save", "reason": "free text"},
    {"role": "button", "name": "Save", "rank": 1},
    {"role": "button", "name": "Save", "strategy": "SEMANTIC_ALTERNATE"},   # the model cannot pick its own strategy
    {"role": "button", "name": "Save", "strategy": "MODEL_CANDIDATE"},
    {"role": "button", "name": "Save", "extras": {}},
    {"role": "button", "name": "Save", "schema_id": "runtime-contracts/qa/locator-candidate"},
    {"role": "button", "name": "Save", "Role": "x"},
]


@pytest.mark.parametrize("item", REJECTED_ITEMS, ids=lambda i: sorted(set(i) - {"role", "name"})[0])
def test_any_field_other_than_role_name_label_and_test_id_is_rejected_by_the_strict_gate(item):
    rec, client, _ = make(good(SAVE, item))
    with pytest.raises(RecoveryPayloadRejected):
        propose(rec)
    assert client.calls == 1


BAD_VALUES = [
    {"role": "button", "name": "https://evil.example/pay"},
    {"role": "button", "name": "Save", "test_id": "javascript:alert(1)"},
    {"role": "button", "name": "Save", "label": "data:text/html;base64,AAAA"},
    {"role": "button", "name": "<script>alert(1)</script>"},
    {"role": "button", "name": "Save`rm -rf /`"},
    {"role": "button", "name": "${jndi:ldap://x}"},
    {"role": "button", "name": "x" * 81},
    {"role": "button", "name": ""},
    {"role": "button", "name": "Sa\x00ve"},
    {"role": "button", "name": "Sa‮ve"},
    {"role": "button", "name": "Sa\nve"},
    {"role": "button", "name": 7},
    {"role": "button", "name": True},
    {"role": "button", "name": ["Save"]},
    {"role": "button", "name": {"text": "Save"}},
    {"role": "button", "name": 1.5},
]


@pytest.mark.parametrize("item", BAD_VALUES, ids=lambda i: repr(i["name"])[:30])
def test_a_value_that_is_free_text_a_url_markup_script_or_not_a_short_plain_string_is_rejected(item):
    with pytest.raises(RecoveryPayloadRejected):
        propose(make(good(item))[0])


def test_the_rejection_message_never_echoes_what_the_model_said():
    secret = "SECRET-MODEL-TEXT-7731"
    with pytest.raises(RecoveryPayloadRejected) as caught:
        propose(make(good({"role": "button", "name": "Save", secret: "x"}))[0])
    assert secret not in str(caught.value) and secret not in repr(caught.value)


def test_the_gate_checks_every_element_before_anything_is_used():
    rec, client, _ = make(good(SAVE, SAVE_BY_ID, {"role": "button", "name": "Save", "expected": "Saved"}))
    with pytest.raises(RecoveryPayloadRejected):
        propose(rec)


@pytest.mark.parametrize("text, code", [
    ("not json", RM.R_MODEL_MALFORMED),
    ("", RM.R_MODEL_MALFORMED),
    ("   ", RM.R_MODEL_MALFORMED),
    ("```json\n[{\"role\":\"button\",\"name\":\"Save\"}]\n```", RM.R_MODEL_MALFORMED),
    ("Sure! [{\"role\":\"button\",\"name\":\"Save\"}]", RM.R_MODEL_MALFORMED),
    ("{\"role\":\"button\",\"name\":\"Save\"}", RM.R_MODEL_MALFORMED),            # an object, not a list
    ("\"Save\"", RM.R_MODEL_MALFORMED),
    ("null", RM.R_MODEL_MALFORMED),
    ("[\"Save\"]", RM.R_MODEL_MALFORMED),                                        # a list of non-objects
    ("[[\"Save\"]]", RM.R_MODEL_MALFORMED),
    ("[{\"role\":\"button\",\"role\":\"link\",\"name\":\"Save\"}]", RM.R_MODEL_MALFORMED),     # duplicate key
    ("[{\"role\":\"button\",\"name\":NaN}]", RM.R_MODEL_MALFORMED),
    ("[{\"role\":\"button\",\"name\":\"Save\"},]", RM.R_MODEL_MALFORMED),
    ("[" * 1500 + "]" * 1500, RM.R_MODEL_MALFORMED),                              # nesting
    ("[]", RM.R_MODEL_NO_CANDIDATES),
])
def test_output_that_is_not_a_list_of_objects_is_discarded_with_a_reason_and_cost_is_still_recorded(text, code):
    rec, client, _ = make(Reply(text=text))
    assert propose(rec) == () and rec.reason == code and not rec.halted
    assert client.calls == 1 and rec.ledger.calls == 1 and rec.ledger.tokens == 42


def test_a_reply_that_is_not_text_is_discarded():
    rec, client, _ = make(Reply(text=b"[]", usage=(1, 1)))
    assert propose(rec) == () and rec.reason == RM.R_MODEL_MALFORMED
    rec, client, _ = make(Reply(wrong_type=True))
    assert propose(rec) == () and rec.reason == RM.R_MODEL_MALFORMED


# =====================================================================================================================
# bounds, each at its boundary
# =====================================================================================================================

def test_the_default_limits_make_no_call_at_all():
    rec, client, _ = make(good(SAVE), limits=RM.ModelRecoveryLimits())
    assert propose(rec) == () and rec.reason == RM.R_MODEL_DISABLED and client.calls == 0
    assert RM.ModelRecoveryLimits().enabled is False and opt_in().enabled is True


@pytest.mark.parametrize("field", ["max_calls_per_step", "max_calls_per_run", "max_tokens_per_step", "max_tokens_per_run",
                                   "max_candidates", "max_candidates_per_step", "max_candidates_per_run",
                                   "max_output_tokens", "max_wall_ms"])
def test_a_zero_anywhere_keeps_the_model_off(field):
    rec, client, _ = make(good(SAVE), limits=opt_in(**{field: 0}))
    assert propose(rec) == () and rec.reason == RM.R_MODEL_DISABLED and client.calls == 0


def test_limits_reject_negative_non_integer_and_oversized_values():
    for bad in (-1, 1.0, "3", True, None):
        with pytest.raises(SpecInvalid):
            RM.ModelRecoveryLimits(max_calls_per_step=bad)
    with pytest.raises(SpecInvalid):
        opt_in(max_candidates=17)
    assert opt_in(max_candidates=16).max_candidates == 16


def test_calls_per_step_boundary():
    rec, client, _ = make(good(SAVE), good(SAVE), limits=opt_in(max_calls_per_step=1, max_calls_per_run=5))
    assert propose(rec, step_id="s-a") and client.calls == 1
    assert propose(rec, step_id="s-a") == () and rec.reason == RM.R_MODEL_CALL_CAP and client.calls == 1
    assert propose(rec, step_id="s-b") and client.calls == 2               # another step has its own allowance


def test_calls_per_run_boundary():
    rec, client, _ = make(good(SAVE), good(SAVE), good(SAVE), limits=opt_in(max_calls_per_step=2, max_calls_per_run=2))
    assert propose(rec, step_id="s-a") and propose(rec, step_id="s-b")
    assert propose(rec, step_id="s-c") == () and rec.reason == RM.R_MODEL_CALL_CAP and client.calls == 2
    assert propose(rec, step_id="s-a") == () and client.calls == 2


def test_the_call_is_charged_before_the_client_runs_even_if_it_raises():
    rec, client, _ = make(Reply(raises=RuntimeError("boom")), good(SAVE), limits=opt_in(max_calls_per_step=1))
    assert propose(rec) == () and rec.reason == RM.R_MODEL_CLIENT_ERROR and client.calls == 1
    assert propose(rec) == () and rec.reason == RM.R_MODEL_CALL_CAP and client.calls == 1       # no free retry


def test_tokens_per_step_boundary_exact_fit_then_nothing_left():
    lim = opt_in(max_tokens_per_step=100, max_calls_per_step=5, max_calls_per_run=5)
    rec, client, _ = make(good(SAVE, usage=(30, 12)), good(SAVE, usage=(40, 18)), good(SAVE), limits=lim)
    assert propose(rec) and rec.ledger.tokens == 42
    assert propose(rec) and rec.ledger.tokens == 100                        # 58 left, 58 used: exactly the cap
    assert propose(rec) == () and rec.reason == RM.R_MODEL_TOKEN_CAP and client.calls == 2


def test_one_token_over_the_remaining_budget_is_refused_and_the_whole_remainder_is_charged():
    lim = opt_in(max_tokens_per_step=100, max_calls_per_step=5, max_calls_per_run=5)
    rec, client, _ = make(good(SAVE, usage=(30, 12)), good(SAVE, usage=(40, 19)), limits=lim)
    assert propose(rec)
    assert propose(rec) == () and rec.reason == RM.R_MODEL_OVER_BUDGET
    assert rec.ledger.tokens == 100                                         # 58 left; 59 reported; all 58 charged
    kind, event = rec.events[-1]
    assert kind == "ModelResponse" and event["outcome"] == RM.R_MODEL_OVER_BUDGET
    assert event["usage"] == {"input_tokens": 40, "output_tokens": 19}


def test_tokens_per_run_boundary():
    lim = opt_in(max_tokens_per_step=1000, max_tokens_per_run=100, max_calls_per_step=5, max_calls_per_run=5)
    rec, client, _ = make(good(SAVE, usage=(50, 50)), good(SAVE), limits=lim)
    assert propose(rec, step_id="s-a") and rec.ledger.tokens == 100
    assert propose(rec, step_id="s-b") == () and rec.reason == RM.R_MODEL_TOKEN_CAP and client.calls == 1


def test_the_output_limit_given_to_the_client_is_the_smaller_of_the_cap_and_what_is_left():
    lim = opt_in(max_output_tokens=256, max_tokens_per_step=300, max_calls_per_step=5, max_calls_per_run=5)
    rec, client, _ = make(good(SAVE, usage=(100, 100)), good(SAVE, usage=(10, 10)), limits=lim)
    propose(rec)
    assert client.requests[0].max_output_tokens == 256 and client.requests[0].timeout_ms == lim.max_wall_ms
    propose(rec)
    assert client.requests[1].max_output_tokens == 100                      # 300 - 200 left


def test_a_reply_that_exceeds_the_calls_own_output_limit_is_over_budget():
    lim = opt_in(max_output_tokens=20)
    rec, client, _ = make(good(SAVE, usage=(5, 21)), limits=lim)
    assert propose(rec) == () and rec.reason == RM.R_MODEL_OVER_BUDGET
    rec, client, _ = make(good(SAVE, usage=(5, 20)), limits=lim)
    assert propose(rec)


@pytest.mark.parametrize("reply", [
    Reply(text=array(SAVE), usage=None),
    Reply(text=array(SAVE), usage_object=RM.ModelUsage(input_tokens=-1, output_tokens=5)),
    Reply(text=array(SAVE), usage_object=RM.ModelUsage(input_tokens=5, output_tokens=-1)),
    Reply(text=array(SAVE), usage_object=RM.ModelUsage(input_tokens=5.0, output_tokens=5)),
    Reply(text=array(SAVE), usage_object=RM.ModelUsage(input_tokens=True, output_tokens=5)),
    Reply(text=array(SAVE), usage_object=RM.ModelUsage(input_tokens="5", output_tokens=5)),
    Reply(text=array(SAVE), usage_object={"input_tokens": 1, "output_tokens": 1}),
    Reply(text=array(SAVE), usage_object=(1, 1)),
], ids=["none", "negative-in", "negative-out", "float", "bool", "str", "dict", "tuple"])
def test_unknown_or_unusable_token_usage_is_refused_and_the_whole_remainder_is_charged(reply):
    lim = opt_in(max_tokens_per_step=500, max_tokens_per_run=900)
    rec, client, _ = make(reply, limits=lim)
    assert propose(rec) == () and rec.reason == RM.R_MODEL_USAGE_UNKNOWN
    assert rec.ledger.tokens == 500 and rec.ledger.calls == 1                # nothing more may be spent for this step
    assert propose(rec) == () and rec.reason in (RM.R_MODEL_CALL_CAP, RM.R_MODEL_TOKEN_CAP) and client.calls == 1


def test_a_client_that_raises_costs_the_whole_remainder():
    rec, client, _ = make(Reply(raises=ConnectionError("down")), limits=opt_in(max_tokens_per_step=500))
    assert propose(rec) == () and rec.reason == RM.R_MODEL_CLIENT_ERROR and rec.ledger.tokens == 500


def test_candidates_per_response_boundary_an_over_long_reply_is_discarded_whole_never_truncated():
    lim = opt_in(max_candidates=2)
    two = good({"role": "button", "name": "A"}, {"role": "button", "name": "B"})
    three = good({"role": "button", "name": "A"}, {"role": "button", "name": "B"}, {"role": "button", "name": "C"})
    assert len(propose(make(two, limits=lim)[0])) == 2
    rec, _, _ = make(three, limits=lim)
    assert propose(rec) == () and rec.reason == RM.R_MODEL_TOO_MANY


def test_candidates_per_step_and_per_run_boundaries():
    lim = opt_in(max_candidates=3, max_candidates_per_step=4, max_candidates_per_run=5, max_calls_per_step=5,
                 max_calls_per_run=5)
    three = good({"name": "A"}, {"name": "B"}, {"name": "C"})
    two = good({"name": "D"}, {"name": "E"})
    rec, client, _ = make(three, two, two, three, limits=lim)
    assert len(propose(rec, step_id="s-a")) == 3
    assert propose(rec, step_id="s-a") == () and rec.reason == RM.R_MODEL_TOO_MANY       # 2 > 1 left for the step
    assert len(propose(rec, step_id="s-b")) == 2 and rec.ledger.candidates == 5          # run cap reached
    assert propose(rec, step_id="s-c") == () and rec.reason == RM.R_MODEL_CANDIDATE_CAP


def test_the_response_size_boundary():
    body = array(SAVE)
    exact = opt_in(max_response_bytes=len(body.encode()))
    assert propose(make(Reply(text=body), limits=exact)[0])
    rec, _, _ = make(Reply(text=body), limits=opt_in(max_response_bytes=len(body.encode()) - 1))
    assert propose(rec) == () and rec.reason == RM.R_MODEL_TOO_LARGE
    padded = Reply(text=body + " " * 10_000)
    rec, _, _ = make(padded)
    assert propose(rec) == () and rec.reason == RM.R_MODEL_TOO_LARGE


def test_the_response_size_is_counted_in_bytes_not_characters():
    text = array({"role": "button", "name": "é" * 40})
    assert len(text) < len(text.encode())
    rec, _, _ = make(Reply(text=text), limits=opt_in(max_response_bytes=len(text)))
    assert propose(rec) == () and rec.reason == RM.R_MODEL_TOO_LARGE


def test_the_wall_time_boundary_on_the_injected_clock():
    lim = opt_in(max_wall_ms=1000)
    rec, client, _ = make(good(SAVE, latency_ms=999), limits=lim)
    assert propose(rec) and not rec.halted
    rec, client, _ = make(good(SAVE, latency_ms=1000), limits=lim)
    assert propose(rec) == () and rec.reason == RM.R_MODEL_WALL_TIME and rec.halted
    assert rec.ledger.calls == 1 and rec.ledger.tokens == 42                # the cost of a slow call is still recorded


def test_the_kill_switch_is_checked_before_the_call():
    kill = KillSwitch()
    kill.engage("operator")
    rec, client, _ = make(good(SAVE), kill=kill)
    assert propose(rec) == () and rec.reason == "KILL_SWITCH" and rec.halted
    assert client.calls == 0 and rec.ledger.calls == 0 and rec.ledger.tokens == 0


def test_the_kill_switch_is_checked_after_the_response_and_nothing_the_model_said_is_used():
    kill = KillSwitch()
    rec, client, _ = make(good(SAVE, kill=kill), kill=kill)
    assert propose(rec) == () and rec.reason == "KILL_SWITCH" and rec.halted
    assert client.calls == 1 and rec.ledger.calls == 1 and rec.ledger.tokens == 42     # the call happened and is accounted
    assert propose(rec) == () and client.calls == 1                         # and once engaged it stays engaged


def test_a_kill_switch_engaged_mid_flight_beats_a_hostile_reply():
    kill = KillSwitch()
    hostile = good(SAVE, {"role": "button", "name": "Save", "expected": "x"}, kill=kill)
    rec, client, _ = make(hostile, kill=kill)
    assert propose(rec) == () and rec.halted                                # halted before the reply is even parsed


def test_no_snapshot_means_no_call():
    rec, client, _ = make(good(SAVE), snapshots=StaticSnapshots())
    assert propose(rec) == () and rec.reason == RM.R_MODEL_NO_SNAPSHOT and client.calls == 0 and rec.ledger.calls == 0

    class Raising:
        def nodes(self, observation):
            raise RuntimeError("blob store down")
    rec, client, _ = make(good(SAVE), snapshots=Raising())
    assert propose(rec) == () and rec.reason == RM.R_MODEL_NO_SNAPSHOT and client.calls == 0


def test_a_step_without_an_authored_role_and_name_gets_no_call():
    rec, client, _ = make(good(SAVE))
    assert propose(rec, role=None) == () and rec.reason == RM.R_MODEL_NO_ANCHOR and client.calls == 0


def test_the_recoverer_needs_a_client_a_snapshot_source_a_clock_and_real_limits():
    clock = FakeClock()
    with pytest.raises(SpecInvalid):
        RM.ModelCandidateRecoverer(client=object(), snapshots=NODES, clock=clock)
    with pytest.raises(SpecInvalid):
        RM.ModelCandidateRecoverer(client=FakeModelClient(clock), snapshots=object(), clock=clock)
    with pytest.raises(SpecInvalid):
        RM.ModelCandidateRecoverer(client=FakeModelClient(clock), snapshots=NODES, clock=object())
    with pytest.raises(SpecInvalid):
        RM.ModelCandidateRecoverer(client=FakeModelClient(clock), snapshots=NODES, clock=clock, limits=object())
    with pytest.raises(SpecInvalid):
        RM.ModelCandidateRecoverer(client=FakeModelClient(clock), snapshots=NODES, clock=clock, kill_switch=object())


def test_the_recoverer_satisfies_the_recoverer_port():
    from agent_harness.qa.ports import Recoverer
    assert isinstance(make()[0], Recoverer) and isinstance(FakeModelClient(FakeClock()), RM.CandidateModelClient)
