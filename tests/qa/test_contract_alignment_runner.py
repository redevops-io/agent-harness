# SPDX-License-Identifier: AGPL-3.0-or-later
"""The runner and the recovery engine against the contracts' reading of a recovery record, and a timeout that is
recoverable.

``FaultRecovery.reason_code`` says why a recovery as a whole did not recover (closed vocabulary); the per-candidate
``candidates_rejected_reasons`` only says why each candidate that was considered was rejected. A timeout is a
recoverable fault, and recovery stays fail-closed: it needs enablement, a permitted mode, the caps and the idempotency
guard, a recoverable timeout never produces a pass, and nothing is repeated when its side effect may have been applied.
"""
import pytest

from agent_harness.qa import errors, recovery as RV, recovery_model as RM, records as R
from agent_harness.qa.errors import ActionTimeout, SessionLost
from agent_harness.qa.idempotency import (DENY_ALREADY_APPLIED, DENY_BUDGET_EXHAUSTED, DENY_INVALID_INPUT, DENY_NO_GUARD,
                                          DENY_NOT_PERMITTED, DENY_NOT_RETRYABLE, DENY_RECOVERY_DISABLED,
                                          DENY_RETRY_UNSAFE, DENY_UNKNOWN_SIDE_EFFECT_CLASS)
from agent_harness.qa.runstate import RunState

from agent_harness.qa.hooks import NotEvaluatedPostcondition

from ._fx_recovery import (BUDGET, classification, clicked, duplicate_save, lose_session_on_save, make_recovery_env,
                           move_save, recover_spec, run, stale_fixture)
from ._fx_runner import Fault, default_policy, make_config, make_env, make_spec, nav_step
from .test_recovery import Fixed, cand, observation, step
from .test_recovery_runner import DenyFrom, StubEngine, decision, only, selection
from .test_runner import Hook, clicks, retry_spec, save_step, semantic, timeout_on

RC, RR, RS, SUP, O = R.RecoveryReason, R.RecoveryResult, R.RecoveryStrategy, R.SuppressedReason, R.Outcome


# =====================================================================================================================
# the mapping of the engine's causes onto the closed vocabulary
# =====================================================================================================================

def test_every_cause_maps_to_a_member_of_the_closed_vocabulary():
    assert set(RV.REASON_CODE_OF.values()) <= set(R.RecoveryReason)
    for member in R.RecoveryReason:                                              # every member is the image of a cause
        if member is not RC.POSTCONDITION_NOT_VERIFIED:
            assert member in RV.REASON_CODE_OF.values(), member
    assert RV.reason_code_for(None) is None
    assert RV.reason_code_for("NOT_A_CAUSE_THIS_VERSION_KNOWS") is RC.NOT_PERMITTED       # the conservative member
    assert RV.reason_code_for("SESSION_LOST") is RC.POSTCONDITION_NOT_VERIFIED            # a typed adapter error code
    assert RV.reason_code_for("REPLAY_" + DENY_RETRY_UNSAFE) is RC.RETRY_UNSAFE


def test_the_selector_refusals_the_guard_denials_and_the_model_refusals_are_all_mapped():
    selector = {RV.R_NO_UI_TARGET: RC.NO_UI_TARGET, RV.R_NO_ANCHOR: RC.NO_ROLE_NAME_ANCHOR,
                RV.R_RECOVERER_ERROR: RC.RECOVERER_ERROR, RV.R_CANDIDATE_CAP: RC.CANDIDATE_CAP_EXCEEDED,
                RV.R_ELAPSED_CAP: RC.ELAPSED_CAP, RV.R_KILL_SWITCH: RC.KILL_SWITCH, RV.R_NO_CANDIDATE: RC.NO_CANDIDATE,
                RV.R_AMBIGUOUS: RC.AMBIGUOUS_CANDIDATES}
    guard = {DENY_INVALID_INPUT: RC.INVALID_INPUT, DENY_RECOVERY_DISABLED: RC.RECOVERY_DISABLED,
             DENY_BUDGET_EXHAUSTED: RC.BUDGET_EXHAUSTED, DENY_NOT_PERMITTED: RC.NOT_PERMITTED,
             DENY_NOT_RETRYABLE: RC.NOT_RETRYABLE, DENY_ALREADY_APPLIED: RC.ALREADY_APPLIED,
             DENY_RETRY_UNSAFE: RC.RETRY_UNSAFE, DENY_NO_GUARD: RC.NO_IDEMPOTENCY_OR_POSTCONDITION_GUARD,
             DENY_UNKNOWN_SIDE_EFFECT_CLASS: RC.UNKNOWN_SIDE_EFFECT_CLASS}
    for cause, member in {**selector, **guard}.items():
        assert RV.reason_code_for(cause) is member, cause
    model_constants = {v for k, v in vars(RM).items() if k.startswith("R_MODEL_")}
    assert model_constants <= set(RV.REASON_CODE_OF), model_constants - set(RV.REASON_CODE_OF)
    # a candidate-level rejection is not a cause: it is never mapped, it stays one token per candidate
    for token in (RV.R_NOT_FOUND, RV.R_NOT_UNIQUE, RV.R_ROLE_MISMATCH, RV.R_NAME_MISMATCH, RV.R_NOT_ACTIONABLE):
        assert token not in RV.REASON_CODE_OF


def test_a_proposal_that_is_refused_before_it_is_located_still_counts_as_considered():
    proposals = Fixed(cand(strategy=R.LocatorStrategy.MODEL_CANDIDATE, role="button", name="Save"),
                      cand())                                                  # a wrong strategy, then an empty one
    selector = RV.LocatorSelector(proposals, RV.RecoveryLimits())
    got = selector.select(step(), observation(), lambda d: R.LocateResult(candidates=()), lambda: None)
    assert got.rejected == (RV.R_STRATEGY, RV.R_EMPTY) and got.considered == 2
    assert len(got.rejected) <= got.considered


# =====================================================================================================================
# the runner records why a recovery did not recover
# =====================================================================================================================

def timeout_after_applied_write(adapter):
    adapter.faults.update(act=[None, Fault(ActionTimeout("late"), apply=True)])


def timeout_clean(adapter):
    adapter.faults.update(act=[None, Fault(ActionTimeout("late"))])


@pytest.mark.parametrize("make, expected, result", [
    (lambda: make_recovery_env(recover_spec(save=("RETRY_SAME",), keyed=False), adapter_setup=timeout_after_applied_write),
     RC.ALREADY_APPLIED, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(save=("RETRY_SAME",), keyed=False), adapter_setup=timeout_after_applied_write,
                               postconditions=NotEvaluatedPostcondition()), RC.RETRY_UNSAFE, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(save=("CHECKPOINT_RESUME",), keyed=False), adapter_setup=lose_session_on_save),
     RC.RETRY_UNSAFE, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(), adapter_setup=move_save, config_kw={"recovery_enabled": False}),
     RC.RECOVERY_DISABLED, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(save=()), adapter_setup=move_save), RC.NOT_PERMITTED, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(save=("RESET_FIXTURE",), fixtures=[]), adapter_setup=stale_fixture),
     RC.NOT_PERMITTED, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(budget={**BUDGET, "max_recovery_attempts": 0}), adapter_setup=move_save),
     RC.BUDGET_EXHAUSTED, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(), adapter_setup=move_save, engine=StubEngine(decision=decision(RS.MODEL_CANDIDATE))),
     RC.NOT_PERMITTED, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(), adapter_setup=move_save, engine=StubEngine(plan_raises=True)),
     RC.NOT_PERMITTED, RR.NOT_ATTEMPTED),
    (lambda: make_recovery_env(recover_spec(), adapter_setup=move_save,
                               engine=StubEngine(decision=decision(RS.SEMANTIC_LOCATOR),
                                                 selection=selection("btn-delete", name="Delete"))),
     RC.NO_CANDIDATE, RR.NOT_RECOVERED),
], ids=["already-applied", "unsafe-write", "session-on-unsafe-write", "recovery-disabled", "mode-not-permitted",
        "no-known-fixture", "attempt-budget", "model-not-usable", "engine-raises", "runner-recheck-failed"])
def test_the_reason_a_recovery_did_not_recover_is_in_reason_code(make, expected, result):
    env = make()
    r = run(env)
    rec = only(r).recovery
    assert rec.result is result and rec.reason_code is expected
    assert r.outcome in (O.INCONCLUSIVE, O.ENVIRONMENT_FAIL)
    assert rec.candidates_rejected_reasons == () or rec.candidates_considered >= len(rec.candidates_rejected_reasons)


def test_the_kill_switch_is_recorded_as_the_suppressed_reason_and_the_reason_code():
    engine = RV.DeterministicRecovery()
    engine.kill_switch.engage("operator stop")
    r = run(make_recovery_env(recover_spec(), adapter_setup=move_save, engine=engine))
    rec = only(r).recovery
    assert rec.suppressed_reason is SUP.KILL_SWITCH and rec.reason_code is RC.KILL_SWITCH
    assert rec.attempted is False and rec.result is RR.NOT_ATTEMPTED


def test_a_recovery_that_recovered_carries_no_reason_code_and_hashes_as_it_always_did():
    r = run(make_recovery_env(recover_spec(), adapter_setup=move_save))
    rec = only(r).recovery
    assert rec.result is RR.RECOVERED and rec.reason_code is None and "reason_code" not in rec.canonical_form()


def test_a_policy_denied_recovery_is_not_permitted():
    spec = recover_spec()
    env = make_recovery_env(spec, adapter_setup=move_save, policy=DenyFrom(default_policy(spec), spec, "s-save", 2))
    r = run(env)
    rec = only(r).recovery
    assert rec.suppressed_reason is SUP.POLICY_DENIED and rec.reason_code is RC.NOT_PERMITTED
    assert rec.result is RR.NOT_ATTEMPTED and clicked(env, "btn-save") == []


def test_a_candidate_that_was_considered_is_rejected_per_candidate_and_the_round_has_its_own_reason():
    r = run(make_recovery_env(recover_spec(), adapter_setup=duplicate_save))
    rec = only(r).recovery
    assert rec.reason_code is RC.AMBIGUOUS_CANDIDATES
    assert rec.candidates_considered >= 1 and rec.candidates_rejected_reasons
    assert len(rec.candidates_rejected_reasons) <= rec.candidates_considered
    assert set(rec.candidates_rejected_reasons) <= {RV.R_AMBIGUOUS_RESULT, RV.R_NOT_FOUND, RV.R_NOT_UNIQUE}


# =====================================================================================================================
# A5: a timeout is recoverable, and the runner stays fail-closed
# =====================================================================================================================

def test_a_timeout_is_both_retryable_and_recoverable():
    assert errors.ActionTimeout.retryable is True and errors.ActionTimeout.recoverable is True
    assert errors.ActionTimeout.category is errors.ErrorCategory.ACTION
    assert errors.ActionTimeout.class_hint is errors.ClassHint.BENIGN_DRIFT_OR_ENVIRONMENT


def hooked_spec():
    return make_spec([nav_step(), save_step(permitted=["SEMANTIC_LOCATOR"])], allowed_side_effects=["WRITES_INTERNAL"])


def test_a_timeout_whose_action_may_have_been_applied_is_never_re_targeted_or_repeated():
    hook = Hook(semantic())
    env = make_env(hooked_spec(), adapter_setup=timeout_on(1, apply=True), recovery=hook,
                   config=make_config(recovery_enabled=True))
    r = env.run()
    assert hook.requests == []                                              # no re-targeting after a possible effect
    assert len(clicks(env)) == 1 and env.adapter.values == {"save-status": "Saved"}      # applied exactly once
    assert r.outcome is O.INCONCLUSIVE and RunState.RECOVERABLE not in r.states


def test_a_timeout_that_did_not_reach_the_action_is_recoverable_and_still_never_passes():
    hook = Hook(semantic())
    env = make_env(hooked_spec(), adapter_setup=lambda a: a.faults.update(locate=[None, Fault(ActionTimeout("late"))]),
                   recovery=hook, config=make_config(recovery_enabled=True))
    r = env.run()
    assert len(hook.requests) == 1 and hook.requests[0].error_code == "ACTION_TIMEOUT"       # recoverable is honoured
    assert r.outcome is O.INCONCLUSIVE and r.exit_code != 0                               # whatever the hook found
    assert classification(r)[0] == "INCONCLUSIVE" and r.classification.halt_batch is False


def test_a_timeout_is_not_recovered_unless_recovery_is_enabled():
    hook = Hook(semantic())
    off = make_env(hooked_spec(), adapter_setup=lambda a: a.faults.update(locate=[None, Fault(ActionTimeout("late"))]),
                   recovery=hook)
    r = off.run()
    assert hook.requests == [] and r.outcome is O.INCONCLUSIVE and RunState.RECOVERABLE not in r.states


def test_a_timeout_is_not_recovered_by_a_mode_the_step_does_not_permit():
    hook = Hook(semantic())
    spec = make_spec([nav_step(), save_step()], allowed_side_effects=["WRITES_INTERNAL"])
    env = make_env(spec, adapter_setup=lambda a: a.faults.update(locate=[None, Fault(ActionTimeout("late"))]), recovery=hook,
                   config=make_config(recovery_enabled=True))
    r = env.run()
    assert hook.requests == [] and r.outcome is O.INCONCLUSIVE


def test_a_timeout_on_an_unsafe_write_is_not_retried_with_recovery_enabled():
    spec = retry_spec(save=save_step(permitted=["RETRY_SAME"], retry_safety="UNSAFE"))
    env = make_env(spec, adapter_setup=timeout_on(1, apply=True), config=make_config(recovery_enabled=True))
    r = env.run()
    assert r.outcome is O.INCONCLUSIVE and len(clicks(env)) == 1


def test_the_engine_still_retries_a_timeout_only_under_the_idempotency_guard():
    # a keyed, idempotent write whose effect is observed absent: the guard allows one replay and the step is judged
    # by its own postcondition; an unkeyed one is never replayed (covered above and in the recovery runner tests)
    env = make_recovery_env(recover_spec(save=("RETRY_SAME",), keyed=True), adapter_setup=timeout_clean)
    r = run(env)
    ev = only(r)
    assert ev.recovery.strategy is RS.RETRY_SAME
    assert ev.recovery.result in (RR.RECOVERED, RR.NOT_RECOVERED)
    if ev.recovery.result is RR.RECOVERED:
        assert ev.recovery.reason_code is None and r.outcome is O.PASS
    assert len(clicked(env, "btn-save")) <= 2
