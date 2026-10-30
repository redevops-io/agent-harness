# SPDX-License-Identifier: AGPL-3.0-or-later
"""H2: idempotency keys and the pure retry decision, checked over the whole input space against an independent oracle
and against the safety invariants (never replay without a guard)."""
import itertools

import pytest

from agent_harness.qa import canon, idempotency as I, records as R

SE, ID, RS, PC = R.SideEffect, R.Idempotency, R.RetrySafety, R.PostconditionResult
RETRY_SAME = (R.PermittedRecovery.RETRY_SAME,)

READ_ONLY = {SE.NONE, SE.READS_EXTERNAL}
INTERNAL = {SE.WRITES_INTERNAL}
RISKY = {SE.WRITES_EXTERNAL, SE.NOTIFIES_HUMAN, SE.SPENDS_MONEY}      # send, payment, approval, deletion, submit


def inputs(**kw):
    base = dict(side_effect=SE.WRITES_INTERNAL, idempotency=ID.AT_MOST_ONCE, retry_safety=RS.SAFE_WITH_IDEMPOTENCY_KEY,
                permitted_recovery=RETRY_SAME, may_have_executed=True, has_idempotency_key=True,
                postcondition=PC.NOT_SATISFIED, error_retryable=True, recovery_enabled=True, retries_left=1)
    base.update(kw)
    return I.RetryInputs(**base)


def oracle(i):
    """Independent statement of the rules, written as one boolean expression per class."""
    if not i.recovery_enabled or i.retries_left <= 0 or RETRY_SAME[0] not in i.permitted_recovery:
        return False
    if not i.error_retryable or i.postcondition is PC.SATISFIED:
        return False
    if not i.may_have_executed:
        return True
    if i.retry_safety is RS.UNSAFE:
        return False
    keyed = (i.retry_safety is RS.SAFE_WITH_IDEMPOTENCY_KEY and i.idempotency in (ID.IDEMPOTENT, ID.AT_MOST_ONCE)
             and i.has_idempotency_key)
    checked = i.postcondition is PC.NOT_SATISFIED
    if i.side_effect in READ_ONLY:
        return i.idempotency is ID.IDEMPOTENT or checked
    if i.side_effect in INTERNAL:
        return keyed or checked
    return keyed and checked


GRID = list(itertools.product(list(SE), list(ID), list(RS), list(PC), [False, True], [False, True], [False, True],
                              [False, True], [(), RETRY_SAME], [0, 1]))


def test_the_grid_is_the_whole_input_space():
    assert len(GRID) == 6 * 3 * 3 * 3 * 2 * 2 * 2 * 2 * 2 * 2


def test_decide_retry_matches_the_oracle_on_every_point():
    for (se, idem, rs, pc, executed, key, retryable, enabled, permitted, left) in GRID:
        i = inputs(side_effect=se, idempotency=idem, retry_safety=rs, postcondition=pc, may_have_executed=executed,
                   has_idempotency_key=key, error_retryable=retryable, recovery_enabled=enabled,
                   permitted_recovery=permitted, retries_left=left)
        got = I.decide_retry(i)
        assert got.allow is oracle(i), (i, got)
        assert got == I.decide_retry(i)                           # pure


def test_safety_invariants_over_the_whole_space():
    for (se, idem, rs, pc, executed, key, retryable, enabled, permitted, left) in GRID:
        i = inputs(side_effect=se, idempotency=idem, retry_safety=rs, postcondition=pc, may_have_executed=executed,
                   has_idempotency_key=key, error_retryable=retryable, recovery_enabled=enabled,
                   permitted_recovery=permitted, retries_left=left)
        d = I.decide_retry(i)
        if not d.allow:
            continue
        assert i.recovery_enabled and i.retries_left > 0 and RETRY_SAME[0] in i.permitted_recovery
        assert i.error_retryable and i.postcondition is not PC.SATISFIED          # never replay an applied effect
        if i.may_have_executed:
            assert i.retry_safety is not RS.UNSAFE
            assert i.idempotency is ID.IDEMPOTENT or i.postcondition is PC.NOT_SATISFIED \
                or (i.idempotency is ID.AT_MOST_ONCE and i.has_idempotency_key)
            assert i.postcondition is not PC.NOT_EVALUATED or i.idempotency in (ID.IDEMPOTENT, ID.AT_MOST_ONCE)
            if i.side_effect in RISKY:                          # send/payment/approval/deletion/submit: both guards
                assert i.has_idempotency_key and i.postcondition is PC.NOT_SATISFIED
                assert i.retry_safety is RS.SAFE_WITH_IDEMPOTENCY_KEY and i.idempotency in (ID.IDEMPOTENT, ID.AT_MOST_ONCE)
            if i.idempotency is ID.UNKNOWN:                     # UNKNOWN is never a guard on its own
                assert i.postcondition is PC.NOT_SATISFIED


@pytest.mark.parametrize("effect", sorted(RISKY, key=str))
def test_risky_classes_are_never_blindly_replayed(effect):
    base = dict(side_effect=effect)
    assert I.decide_retry(inputs(**base)).allow                                   # both guards present
    assert not I.decide_retry(inputs(**base, has_idempotency_key=False)).allow
    assert not I.decide_retry(inputs(**base, postcondition=PC.NOT_EVALUATED)).allow
    assert not I.decide_retry(inputs(**base, retry_safety=RS.SAFE)).allow          # no key promise
    assert not I.decide_retry(inputs(**base, retry_safety=RS.UNSAFE)).allow
    assert not I.decide_retry(inputs(**base, idempotency=ID.UNKNOWN)).allow
    assert not I.decide_retry(inputs(**base, postcondition=PC.SATISFIED)).allow


def test_unknown_idempotency_never_counts_as_a_guard():
    for se in SE:
        d = I.decide_retry(inputs(side_effect=se, idempotency=ID.UNKNOWN, postcondition=PC.NOT_EVALUATED,
                                  retry_safety=RS.SAFE_WITH_IDEMPOTENCY_KEY))
        assert d.allow is False and d.reason == I.DENY_NO_GUARD


def test_an_applied_effect_is_never_replayed_and_is_reported_as_applied():
    for se in SE:
        d = I.decide_retry(inputs(side_effect=se, postcondition=PC.SATISFIED))
        assert d.allow is False and d.already_applied and d.reason == I.DENY_ALREADY_APPLIED


def test_an_attempt_that_never_reached_act_is_not_a_replay():
    d = I.decide_retry(inputs(side_effect=SE.SPENDS_MONEY, may_have_executed=False, has_idempotency_key=False,
                              postcondition=PC.NOT_EVALUATED, idempotency=ID.UNKNOWN, retry_safety=RS.UNSAFE))
    assert d.allow and d.reason == I.ALLOW_NOT_EXECUTED
    # but the budget and permission gates still apply
    assert not I.decide_retry(inputs(may_have_executed=False, retries_left=0)).allow
    assert not I.decide_retry(inputs(may_have_executed=False, permitted_recovery=())).allow
    assert not I.decide_retry(inputs(may_have_executed=False, recovery_enabled=False)).allow


def test_the_first_failing_gate_gives_the_reason():
    assert I.decide_retry(inputs(recovery_enabled=False, retries_left=0)).reason == I.DENY_RECOVERY_DISABLED
    assert I.decide_retry(inputs(retries_left=0, permitted_recovery=())).reason == I.DENY_BUDGET_EXHAUSTED
    assert I.decide_retry(inputs(permitted_recovery=(), error_retryable=False)).reason == I.DENY_NOT_PERMITTED
    assert I.decide_retry(inputs(error_retryable=False)).reason == I.DENY_NOT_RETRYABLE
    assert I.decide_retry(inputs(retry_safety=RS.UNSAFE)).reason == I.DENY_RETRY_UNSAFE
    assert I.decide_retry(inputs(postcondition=PC.NOT_EVALUATED, retry_safety=RS.SAFE,
                                 idempotency=ID.UNKNOWN)).reason == I.DENY_NO_GUARD


def test_reasons_for_the_allowed_paths():
    assert I.decide_retry(inputs(side_effect=SE.NONE, idempotency=ID.IDEMPOTENT, retry_safety=RS.SAFE,
                                 postcondition=PC.NOT_EVALUATED)).reason == I.ALLOW_IDEMPOTENT_DECLARED
    assert I.decide_retry(inputs(side_effect=SE.WRITES_INTERNAL, postcondition=PC.NOT_EVALUATED)).reason \
        == I.ALLOW_IDEMPOTENCY_KEY
    assert I.decide_retry(inputs(side_effect=SE.WRITES_INTERNAL, retry_safety=RS.SAFE, idempotency=ID.UNKNOWN,
                                 postcondition=PC.NOT_SATISFIED)).reason == I.ALLOW_POSTCONDITION_NOT_APPLIED
    assert I.decide_retry(inputs(side_effect=SE.NOTIFIES_HUMAN)).reason == I.ALLOW_KEY_AND_POSTCONDITION


@pytest.mark.parametrize("bad", [None, "retry", {"side_effect": "NONE"}, 3])
def test_something_that_is_not_retry_inputs_is_a_denial(bad):
    got = I.decide_retry(bad)
    assert got.allow is False and got.reason == I.DENY_INVALID_INPUT


@pytest.mark.parametrize("override", [
    dict(side_effect="NONE"), dict(idempotency="IDEMPOTENT"), dict(retry_safety="SAFE"), dict(postcondition="SATISFIED"),
    dict(permitted_recovery=["RETRY_SAME"]), dict(permitted_recovery=("RETRY_SAME",)), dict(may_have_executed=1),
    dict(has_idempotency_key=None), dict(error_retryable="yes"), dict(recovery_enabled=1), dict(retries_left=True),
    dict(retries_left=1.0),
])
def test_a_malformed_field_is_a_denial(override):
    got = I.decide_retry(inputs(**override))
    assert got.allow is False and got.reason == I.DENY_INVALID_INPUT


# --- keys ---------------------------------------------------------------------------------------------------------

def test_the_key_is_stable_for_a_step_and_distinct_across_steps_and_runs():
    k = I.idempotency_key("run-1", "s-save")
    assert k == I.idempotency_key("run-1", "s-save")                 # same for every attempt of the step
    assert k != I.idempotency_key("run-1", "s-nav") and k != I.idempotency_key("run-2", "s-save")
    assert I.idempotency_key("a", "bc") != I.idempotency_key("ab", "c")
    assert R.ActionRequest.__dataclass_fields__                       # the key must be usable as ActionRequest.key
    assert len(k) <= 128 and k.startswith("idem-")
    R.ActionRequest(action=R.Action(kind=R.ActionKind.CLICK), idempotency_key=k, policy_decision_ref="rcv1:" + "0" * 64,
                    step_id="s-save")


def test_the_key_is_a_digest_of_run_and_step_only():
    assert I.idempotency_key("run-1", "s-save") == "idem-" + canon.digest_hex({"run_id": "run-1", "step_id": "s-save"})[:40]
