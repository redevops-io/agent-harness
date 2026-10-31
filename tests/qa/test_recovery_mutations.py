# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mutation tests. Each recovery safety mechanism is removed from a copy of its source (an exact-text replacement that
must match once) and a probe that holds on the real code must fail on the mutant: remove the uniqueness check, the
role+name rule, the detection-first ordering, a cap, the idempotency check, the step-postcondition judgement ... and
the suite goes red. The probes are small scenarios, so they check the behaviour the mechanism exists for."""
import importlib

import pytest

from agent_harness.qa import records as R
from agent_harness.qa.errors import ActionTimeout, RecoveryPayloadRejected

from ._fx_verify import Mutants
from ._fx_runner import AttestAll, Fault, default_policy, operations
from ._fx_recovery import (acts, after_form, clicked, compose, delay_save, duplicate_save, lose_session_on_save, make_recovery_env,
                  move_save, recover_spec, rename_label, reuse_test_id_on_delete, stale_fixture, wrong_total)
from .test_recovery import Fixed, World, cand, inputs, observation, prior, step, env_inputs
from .test_recovery_runner import DenyFrom, StubEngine, decision, only, selection

RR, RS, SUP = R.RecoveryResult, R.RecoveryStrategy, R.SuppressedReason
PC = R.PostconditionResult


@pytest.fixture
def mutants():
    m = Mutants()
    yield m
    m.close()


def holds(probe, module):
    """True if the probe holds; an exception from a mutant counts as 'does not hold'."""
    try:
        return bool(probe(module))
    except Exception:
        return False


def real(filename):
    return importlib.import_module("agent_harness.qa." + filename[:-3])


def M(filename, replacements, probe, why):
    return filename, replacements, probe, why


# =====================================================================================================================
# probes on recovery.py (the module under test is passed in)
# =====================================================================================================================

def select(mod, world, st=None, recoverer=None, limits=None, within=lambda: True, kill=False):
    switch = mod.KillSwitch()
    if kill:
        switch.engage("operator")
    engine = mod.DeterministicRecovery(recoverer=recoverer, limits=limits, kill_switch=switch)
    st = st or step()
    return engine.select_locator(st, observation(st.step_id), world.locate, within)


def plan(mod, i, limits=None, kill=False):
    return mod.plan_recovery(i, limits or mod.RecoveryLimits(), kill_switch=kill)


def probe_positive_control_recovers(mod):
    got = select(mod, World(save={"test_id": "x"}))
    return got.target is not None and got.target.candidate_ref == "save"


def probe_multi_control_results_are_ambiguous(mod):
    world = World(a={"unique_flag": True}, b={"name": "Delete", "unique_flag": True})
    got = select(mod, world, recoverer=Fixed(cand(role="button")))
    return got.target is None and probe_positive_control_recovers(mod)


def probe_non_unique_control_is_refused(mod):
    got = select(mod, World(save={"unique_flag": False}))
    return got.target is None and probe_positive_control_recovers(mod)


def probe_two_matching_controls_are_never_resolved(mod):
    world = World(a={"test_id": "t1"}, b={"test_id": "t2"})
    got = select(mod, world, recoverer=Fixed(cand(test_id="t1"), cand(test_id="t2")))
    return got.target is None and got.refusal == "AMBIGUOUS_CANDIDATES" and probe_positive_control_recovers(mod)


def probe_a_wrong_role_is_refused(mod):
    world = World(link={"role": "link", "name": "Save"})
    got = select(mod, world, recoverer=Fixed(cand(name="Save")))
    return got.target is None and probe_positive_control_recovers(mod)


def probe_a_wrong_name_is_refused(mod):
    world = World(delete={"name": "Delete", "test_id": "save"})
    got = select(mod, world, step(test_id="save"), recoverer=Fixed(cand(test_id="save")))
    return got.target is None and probe_positive_control_recovers(mod)


def probe_labels_are_normalised(mod):
    return select(mod, World(x={"name": "  SAVE\n"})).target is not None and \
        select(mod, World(x={"name": "Save changes"})).target is None


def probe_candidate_cap_on_proposals(mod):
    many = Fixed(*[cand(test_id=f"t{i}") for i in range(4)])
    got = select(mod, World(save={}), recoverer=many, limits=mod.RecoveryLimits(max_candidates=3))
    return got.target is None and got.aborted and probe_positive_control_recovers(mod)


def probe_candidate_cap_on_controls(mod):
    got = select(mod, World(**{f"c{i}": {} for i in range(5)}), limits=mod.RecoveryLimits(max_candidates=4))
    return got.target is None and got.aborted


def probe_elapsed_cap_stops_selection(mod):
    world = World(save={})
    return select(mod, world, within=lambda: False).aborted and world.calls == []


def probe_kill_switch_stops_selection(mod):
    return select(mod, World(save={}), kill=True).aborted


def probe_model_strategy_is_not_used(mod):
    got = select(mod, World(save={}), recoverer=Fixed(cand(strategy=R.LocatorStrategy.MODEL_CANDIDATE, role="button",
                                                          name="Save")))
    return got.target is None


def probe_recoverer_output_passes_the_strict_gate(mod):
    class Hostile:
        def propose(self, ctx):
            return [{"strategy": "SEMANTIC_ALTERNATE", "role": "button", "name": "Save", "expected": "Saved"}]
    try:
        select(mod, World(save={}), recoverer=Hostile())
    except RecoveryPayloadRejected:
        return True
    return False


def probe_transition_not_observed_is_not_drift(mod):
    got = mod.detect_fault(ActionTimeout("x", signal="TRANSITION_NOT_OBSERVED"), elapsed_ms=0)
    return got.claimed_class is R.ClaimedClass.PRODUCT_DEFECT_SUSPECTED


def probe_plan_kill_switch(mod):
    return not plan(mod, inputs(), kill=True).attempt and plan(mod, inputs()).attempt


def probe_plan_requires_permission(mod):
    return not plan(mod, inputs(st=step(permitted=[]))).attempt and plan(mod, inputs()).attempt


def probe_plan_step_attempt_cap(mod):
    return not plan(mod, inputs(step_attempts_used=2)).attempt and plan(mod, inputs(step_attempts_used=1)).attempt


def probe_plan_elapsed_cap(mod):
    return not plan(mod, inputs(step_recovery_ms=10_000)).attempt and plan(mod, inputs(step_recovery_ms=1)).attempt


def probe_plan_idempotency_guard(mod):
    unsafe = step(permitted=["RETRY_SAME"], side_effect="WRITES_INTERNAL", idempotency="AT_MOST_ONCE",
                  retry_safety="UNSAFE")
    keyed = step(permitted=["RETRY_SAME"], side_effect="WRITES_INTERNAL", idempotency="AT_MOST_ONCE",
                 retry_safety="SAFE_WITH_IDEMPOTENCY_KEY")
    bad = plan(mod, inputs("ACTION_TIMEOUT", st=unsafe, error_retryable=True, acted=True, postcondition=PC.NOT_SATISFIED))
    good = plan(mod, inputs("ACTION_TIMEOUT", st=keyed, error_retryable=True, acted=True, postcondition=PC.NOT_SATISFIED))
    return (not bad.attempt) and bad.suppressed is SUP.NON_IDEMPOTENT and good.attempt


def probe_plan_replays_only_guarded_steps(mod):
    unsafe = prior("WRITES_INTERNAL", "AT_MOST_ONCE", "UNSAFE")
    return (not plan(mod, env_inputs(priors=[prior(), unsafe])).attempt) and plan(mod, env_inputs(priors=[prior()])).attempt


def probe_plan_replay_cap(mod):
    priors = [prior(step_id=f"s-{i}") for i in range(3)]
    return (not plan(mod, env_inputs(priors=priors), mod.RecoveryLimits(max_replay_steps=2)).attempt
            and plan(mod, env_inputs(priors=priors), mod.RecoveryLimits(max_replay_steps=3)).attempt)


def probe_plan_reset_needs_a_fixture(mod):
    base = env_inputs("ENVIRONMENT_UNHEALTHY", "RESET_FIXTURE")
    no_fixture = type(base)(**{**base.__dict__, "has_fixture": False})
    return plan(mod, base).attempt and not plan(mod, no_fixture).attempt


def probe_plan_already_applied(mod):
    st = step(permitted=["RETRY_SAME"], side_effect="WRITES_INTERNAL", idempotency="AT_MOST_ONCE",
              retry_safety="SAFE_WITH_IDEMPOTENCY_KEY")
    got = plan(mod, inputs("ACTION_TIMEOUT", st=st, error_retryable=True, acted=True, postcondition=PC.SATISFIED))
    return got.already_applied and not got.attempt


def probe_plan_policy_denial_is_not_recovered(mod):
    every = step(permitted=[p.value for p in R.PermittedRecovery])
    got = plan(mod, inputs("ACTION_BLOCKED_BY_POLICY", st=every))
    return not got.attempt and got.suppressed is SUP.POLICY_DENIED


# =====================================================================================================================
# probes on runner.py
# =====================================================================================================================

def renv(mod, spec, setup, **kw):
    return make_recovery_env(spec, adapter_setup=setup, mod=mod, **kw)


def probe_detection_is_sealed_before_any_recovery_action(mod):
    env = renv(mod, recover_spec(), move_save)
    env.run()
    log = env.log
    det = log.index(("evidence", "FaultDetection"))
    return log[det + 1] == ("adapter", "observe") and det < log.index(("evidence", "FaultRecovery"))


def probe_detection_is_sealed_first_for_environment_modes(mod):
    for spec, setup, first in ((recover_spec(save=("CHECKPOINT_RESUME",)), lose_session_on_save, "close_session"),
                               (recover_spec(save=("RESET_FIXTURE",)), stale_fixture, "reset")):
        env = renv(mod, spec, setup)
        env.run()
        det = env.log.index(("evidence", "FaultDetection"))
        if env.log[det + 1] != ("adapter", first):
            return False
    return True


def probe_result_comes_from_the_step_postcondition(mod):
    from agent_harness.qa.hooks import NotEvaluatedPostcondition
    env = renv(mod, recover_spec(), move_save, postconditions=NotEvaluatedPostcondition())
    r = env.run()
    return only(r).recovery.result is RR.NOT_RECOVERED and not only(r).recovery.postcondition_verified


def probe_a_cancelled_recovery_is_still_sealed(mod):
    from agent_harness.qa.hooks import CancelToken
    cancel = CancelToken()

    def setup(adapter):
        move_save(adapter)
        original = adapter.locate

        def locate(s, target, *, mode):
            if mode is R.LocateMode.SEMANTIC:
                cancel.cancel("operator")
            return original(s, target, mode=mode)
        adapter.locate = locate
    env = renv(mod, recover_spec(), setup, cancel=cancel)
    r = env.run()
    return len(r.manifest.fault_events) == 1 and only(r).recovery.result is RR.ABORTED


def probe_runner_rechecks_the_permitted_mode(mod):
    stub = StubEngine(decision=decision(RS.RESET_FIXTURE))
    env = renv(mod, recover_spec(), move_save, engine=stub)
    r = env.run()
    return only(r).recovery.suppressed_reason is SUP.NOT_PERMITTED and not [c for c in env.adapter.calls if c[0] == "reset"]


def probe_runner_rechecks_the_kill_switch_and_the_config(mod):
    stub = StubEngine(decision=decision(RS.SEMANTIC_LOCATOR), selection=selection("btn-save"))
    stub.kill_switch.engage("operator")
    env = renv(mod, recover_spec(), move_save, engine=stub)
    r = env.run()
    off = StubEngine(decision=decision(RS.SEMANTIC_LOCATOR), selection=selection("btn-save"))
    env2 = renv(mod, recover_spec(), move_save, engine=off, config_kw={"recovery_enabled": False})
    r2 = env2.run()
    return (only(r).recovery.suppressed_reason is SUP.KILL_SWITCH and only(r2).recovery.suppressed_reason is SUP.KILL_SWITCH
            and clicked(env, "btn-save") == [] and clicked(env2, "btn-save") == [])


def probe_runner_rechecks_the_selected_role_and_name(mod):
    stub = StubEngine(decision=decision(RS.SEMANTIC_LOCATOR), selection=selection("btn-delete", name="Delete"))
    env = renv(mod, recover_spec(), move_save, engine=stub)
    env.run()
    return env.adapter.deleted is False and clicked(env, "btn-delete") == []


def probe_runner_charges_the_recovery_budget(mod):
    stub = StubEngine(decision=decision(RS.SEMANTIC_LOCATOR), selection=selection("btn-save"))
    spec = recover_spec(budget={"deadline_ms": 120000, "max_model_calls": 0, "max_recovery_attempts": 0,
                                "max_retries": 1, "max_tokens": 0})
    env = renv(mod, spec, move_save, engine=stub)
    r = env.run()
    return only(r).recovery.suppressed_reason is SUP.BUDGET_EXHAUSTED and clicked(env, "btn-save") == []


def probe_runner_charges_the_retry_budget(mod):
    stub = StubEngine(decision=decision(RS.RETRY_SAME))
    spec = recover_spec(save=("RETRY_SAME",), budget={"deadline_ms": 120000, "max_model_calls": 0,
                                                       "max_recovery_attempts": 2, "max_retries": 0, "max_tokens": 0})
    env = renv(mod, spec, delay_save, engine=stub)
    r = env.run()
    return only(r).recovery.suppressed_reason is SUP.BUDGET_EXHAUSTED


def probe_runner_checks_the_replayed_state_against_the_checkpoint(mod):
    def drifting(adapter):
        lose_session_on_save(adapter)
        original = adapter.open_session

        def reopen(profile):
            handle = original(profile)
            if adapter.session_count > 1:
                adapter.dom_meta = {"title": "Different page"}
            return handle
        adapter.open_session = reopen
    env = renv(mod, recover_spec(save=("CHECKPOINT_RESUME",)), drifting)
    r = env.run()
    return only(r).recovery.result is RR.NOT_RECOVERED and len(clicked(env, "btn-save")) == 1


def probe_runner_requires_a_confirmed_reset(mod):
    def refuses(adapter):
        stale_fixture(adapter)
        adapter.reset_ok = False
    env = renv(mod, recover_spec(save=("RESET_FIXTURE",)), refuses)
    r = env.run()
    rec = only(r).recovery
    return rec.result is RR.NOT_RECOVERED and rec.reason_code is R.RecoveryReason.POSTCONDITION_NOT_VERIFIED


def probe_runner_requires_a_verified_fixture_after_reset(mod):
    class Stale(AttestAll):
        calls = 0

        def fixtures_match(self, spec):
            Stale.calls += 1
            return Stale.calls == 1
    env = renv(mod, recover_spec(save=("RESET_FIXTURE",)), stale_fixture, checks=Stale())
    r = env.run()
    rec = only(r).recovery
    return rec.result is RR.NOT_RECOVERED and rec.reason_code is R.RecoveryReason.POSTCONDITION_NOT_VERIFIED


def probe_runner_policy_decides_every_replayed_step(mod):
    from agent_harness.qa.policy import make_decision

    class DenyReplay:
        def __init__(self, inner, spec):
            self.inner, self.app = inner, spec.target.app_id

        def decide(self, run_id, step, attempt):
            if step.step_id == "s-nav" and attempt >= 2:
                return make_decision(run_id=run_id, step=step, attempt=attempt, allow=False, rule_id="no-replay",
                                     reason="x", principal="runner:qa", app=self.app, provider="test")
            return self.inner.decide(run_id, step, attempt)
    spec = recover_spec(save=("CHECKPOINT_RESUME",))
    env = renv(mod, spec, lose_session_on_save, policy=DenyReplay(default_policy(spec), spec))
    env.run()
    return len(acts(env, "navigate")) == 1


def probe_runner_records_a_policy_denied_recovery(mod):
    spec = recover_spec()
    env = renv(mod, spec, move_save, policy=DenyFrom(default_policy(spec), spec, "s-save", 2))
    r = env.run()
    return only(r).recovery.suppressed_reason is SUP.POLICY_DENIED and clicked(env, "btn-save") == []


def probe_runner_does_not_call_a_failed_to_close_session_clean(mod):
    env = renv(mod, recover_spec(save=("CHECKPOINT_RESUME",)),
               compose(lose_session_on_save, lambda a: a.faults.update(close_session=[Fault(RuntimeError("dead"))])))
    r = env.run()
    return r.classification.reason == "CLEANUP_UNVERIFIED"


def probe_a_failure_on_a_recovered_locator_is_not_reproduced(mod):
    env = renv(mod, recover_spec(), compose(move_save, wrong_total))
    r = env.run()
    return r.classification.reason == "FAILURE_NOT_REPRODUCED"


def probe_a_recovered_locator_never_passes(mod):
    env = renv(mod, recover_spec(), move_save)
    r = env.run()
    return r.outcome is R.Outcome.INCONCLUSIVE and r.exit_code != 0


def probe_a_plain_retry_met_twice_is_one_episode(mod):
    spec = recover_spec(save=("RETRY_SAME",), budget={"deadline_ms": 120000, "max_model_calls": 0,
                                                       "max_recovery_attempts": 2, "max_retries": 3, "max_tokens": 0})
    env = renv(mod, spec, lambda a: a.faults.update(act=[None, Fault(ActionTimeout("late")), Fault(ActionTimeout("late"))]))
    r = env.run()
    return len(r.manifest.fault_events) == 1 and only(r).recovery.attempts == 2


# =====================================================================================================================
# the table
# =====================================================================================================================

MUTATIONS = {
    # --- recovery.py: uniqueness ----------------------------------------------------------------------------------
    "recovery: a result with several controls is not ambiguous":
        M("recovery.py", [("            if len(found.candidates) > 1:                   # several controls: never the first match",
                           "            if False:                   # several controls: never the first match")],
          probe_multi_control_results_are_ambiguous, "several controls in one result"),
    "recovery: the adapter's uniqueness flag is ignored":
        M("recovery.py", [("            if not cand.unique:", "            if False:")], probe_non_unique_control_is_refused,
          "uniqueness flag"),
    "recovery: two matching controls are not ambiguous":
        M("recovery.py", [("        if len(matching_refs) != 1:", "        if len(matching_refs) < 1:")],
          probe_two_matching_controls_are_never_resolved, "exactly one control"),
    # --- recovery.py: the role+name rule --------------------------------------------------------------------------
    "recovery: the role rule is removed":
        M("recovery.py", [("    if not labels_match(authored.role, role):\n        return R_ROLE_MISMATCH",
                           "    if False:\n        return R_ROLE_MISMATCH")],
          probe_a_wrong_role_is_refused, "role must match"),
    "recovery: the name rule is removed":
        M("recovery.py", [("    if not labels_match(authored.name, name):\n        return R_NAME_NEAR_MISS",
                           "    if False:\n        return R_NAME_NEAR_MISS")],
          probe_a_wrong_name_is_refused, "name must match"),
    "recovery: labels are compared without NFC, case folding or whitespace collapsing":
        M("recovery.py", [("    return \" \".join(unicodedata.normalize(\"NFC\", text).casefold().split())",
                           "    return text")],
          probe_labels_are_normalised, "normalisation"),
    "recovery: a model-candidate proposal is accepted by the deterministic engine":
        M("recovery.py", [("            if proposal.strategy is not LocatorStrategy.SEMANTIC_ALTERNATE:", "            if False:")],
          probe_model_strategy_is_not_used, "no model strategy"),
    "recovery: the recoverer's output skips the strict gate":
        M("recovery.py", [("            proposals = propose_safely(self._recoverer, RecoveryContext(step=step, observation=observation))",
                           "            proposals = tuple(self._recoverer.propose(RecoveryContext(step=step, observation=observation)))")],
          probe_recoverer_output_passes_the_strict_gate, "single strict gate"),
    # --- recovery.py: caps ----------------------------------------------------------------------------------------
    "recovery: the candidate cap on proposals is removed":
        M("recovery.py", [("        if len(proposals) > self._limits.max_candidates:", "        if False:")],
          probe_candidate_cap_on_proposals, "candidate cap"),
    "recovery: the candidate cap on located controls is removed":
        M("recovery.py", [("            if len(found.candidates) > self._limits.max_candidates:", "            if False:")],
          probe_candidate_cap_on_controls, "candidate cap"),
    "recovery: the elapsed cap is not checked during a selection":
        M("recovery.py", [("            if stop is not None:", "            if False:")],
          probe_elapsed_cap_stops_selection, "time cap"),
    "recovery: the kill switch is not checked during a selection":
        M("recovery.py", [("            if self._kill.engaged:\n                return R_KILL_SWITCH", "            if False:\n                return R_KILL_SWITCH")],
          probe_kill_switch_stops_selection, "kill switch"),
    "recovery: the per-step attempt cap is removed":
        M("recovery.py", [("    if i.step_attempts_used >= limits.max_attempts_per_step:", "    if False:")],
          probe_plan_step_attempt_cap, "attempt cap"),
    "recovery: the per-step time cap is removed":
        M("recovery.py", [("    if i.step_recovery_ms >= limits.max_elapsed_ms:", "    if False:")],
          probe_plan_elapsed_cap, "time cap"),
    "recovery: the replay cap is removed":
        M("recovery.py", [("        if len(i.prior_steps) > limits.max_replay_steps:", "        if False:")],
          probe_plan_replay_cap, "replay cap"),
    "recovery: the kill switch and the enable flag are ignored":
        M("recovery.py", [("    if kill_switch or not i.recovery_enabled:", "    if False:")],
          probe_plan_kill_switch, "kill switch"),
    # --- recovery.py: permission, idempotency ---------------------------------------------------------------------
    "recovery: a mode the step does not permit may run":
        M("recovery.py", [("    permitted = [s for s in ladder if PERMIT_OF[s] in i.step.permitted_recovery]",
                           "    permitted = list(ladder)")],
          probe_plan_requires_permission, "permitted recovery modes"),
    "recovery: the idempotency check is removed":
        M("recovery.py", [("    return decide_retry(inputs)", "    return RetryDecision(True, \"MUTANT\")")],
          probe_plan_idempotency_guard, "idempotency"),
    "recovery: an earlier step is replayed without the idempotency check":
        M("recovery.py", [("            if not replay.allow:", "            if False:")],
          probe_plan_replays_only_guarded_steps, "replay guard"),
    "recovery: an applied effect is not recognised":
        M("recovery.py", [("    if guard.reason == DENY_ALREADY_APPLIED:", "    if False:")],
          probe_plan_already_applied, "no replay of an applied effect"),
    "recovery: a fixture reset without a known fixture":
        M("recovery.py", [("        if strategy is RecoveryStrategy.RESET_FIXTURE and not i.has_fixture:", "        if False:")],
          probe_plan_reset_needs_a_fixture, "known fixture"),
    "recovery: a policy denial is retried":
        M("recovery.py", [("    if i.error_code == \"ACTION_BLOCKED_BY_POLICY\":", "    if False:")],
          probe_plan_policy_denial_is_not_recovered, "policy denial"),
    "recovery: a missed transition is claimed as drift":
        M("recovery.py", [("    if error.code == \"ACTION_TIMEOUT\" and error.details.get(\"signal\") == \"TRANSITION_NOT_OBSERVED\":",
                           "    if False:")],
          probe_transition_not_observed_is_not_drift, "detection claim"),
    # --- runner.py ------------------------------------------------------------------------------------------------
    "runner: the detection is sealed after the recovery instead of before":
        M("runner.py", [("        self._emit(c, \"FaultDetection\", detection)\n        c.pending = pending",
                         "        c.pending = pending"),
                        ("        self._emit(c, \"FaultRecovery\", recovery)",
                         "        self._emit(c, \"FaultDetection\", p.detection)\n        self._emit(c, \"FaultRecovery\", recovery)")],
          probe_detection_is_sealed_before_any_recovery_action, "detection first"),
    "runner: the detection is sealed late for environment modes":
        M("runner.py", [("        self._emit(c, \"FaultDetection\", detection)\n        c.pending = pending",
                         "        c.pending = pending"),
                        ("        self._emit(c, \"FaultRecovery\", recovery)",
                         "        self._emit(c, \"FaultDetection\", p.detection)\n        self._emit(c, \"FaultRecovery\", recovery)")],
          probe_detection_is_sealed_first_for_environment_modes, "detection first"),
    "runner: the recovery result is inferred from the absence of an error":
        M("runner.py", [("        verified = made.error is None and made.postcondition.result is PostconditionResult.SATISFIED",
                         "        verified = made.error is None")],
          probe_result_comes_from_the_step_postcondition, "step postcondition"),
    "runner: a recovery cut short by an exception is not sealed":
        M("runner.py", [("                self._abandon_pending(c)\n", "                pass\n")],
          probe_a_cancelled_recovery_is_still_sealed, "every detection gets its recovery record"),
    "runner: the engine's plan is not re-checked against the step's permitted modes":
        M("runner.py", [("        return self._enabled() and PermittedRecovery(strategy.value) in step.permitted_recovery",
                         "        return self._enabled()")],
          probe_runner_rechecks_the_permitted_mode, "runner re-check"),
    "runner: the engine's plan is not re-checked against the kill switch and the configuration":
        M("runner.py", [("        return bool(self._cfg.recovery_enabled and not self._engine.kill_switch.engaged)", "        return True")],
          probe_runner_rechecks_the_kill_switch_and_the_config, "runner re-check"),
    "runner: a selected target is not re-checked against the authored role and name":
        M("runner.py", [("                  and _same_label(authored.role, cand.role) and _same_label(authored.name, cand.name))",
                         "                  )")],
          probe_runner_rechecks_the_selected_role_and_name, "runner re-check"),
    "runner: recovery attempts are not charged":
        M("runner.py", [("            else:\n                c.budgets.charge_recovery_attempt()\n        except BudgetExceeded:",
                         "            else:\n                pass\n        except BudgetExceeded:")],
          probe_runner_charges_the_recovery_budget, "spec budget"),
    "runner: retries are not charged":
        M("runner.py", [("            if strategy is RecoveryStrategy.RETRY_SAME:\n                c.budgets.charge_retry()\n            else:",
                         "            if strategy is RecoveryStrategy.RETRY_SAME:\n                pass\n            else:")],
          probe_runner_charges_the_retry_budget, "spec budget"),
    "runner: the replayed state is not compared with the checkpoint":
        M("runner.py", [("        if shown is None or state_hash(shown) != last.state_hash:", "        if shown is None:")],
          probe_runner_checks_the_replayed_state_against_the_checkpoint, "verified checkpoint"),
    "runner: a reset the adapter does not confirm is accepted":
        M("runner.py", [("            if type(done) is not ResetResult or not done.ok or done.scope is not ResetScope.FIXTURE:", "            if False:")],
          probe_runner_requires_a_confirmed_reset, "known fixture"),
    "runner: the fixture is not re-verified after a reset":
        M("runner.py", [("            if self._attest(self._checks.fixtures_match, c.spec) is not True:", "            if False:")],
          probe_runner_requires_a_verified_fixture_after_reset, "known fixture"),
    "runner: a replayed step is not policy-decided":
        M("runner.py", [("        if not is_allowed(decision):\n            self._receipt(c, prior, attempt, ref,", "        if False:\n            self._receipt(c, prior, attempt, ref,")],
          probe_runner_policy_decides_every_replayed_step, "policy for replays"),
    "runner: a policy-denied recovery is not recorded as such":
        M("runner.py", [("                self._seal_pending(c, attempted=False, result=RecoveryResult.NOT_ATTEMPTED, verified=False,\n                                   suppressed=SuppressedReason.POLICY_DENIED, cause=\"POLICY_DENIED\")",
                         "                pass")],
          probe_runner_records_a_policy_denied_recovery, "policy-denied recovery"),
    "runner: a lost session that cannot be closed counts as clean":
        M("runner.py", [("                c.stale_cleanup = Signal.FAILED        # the lost session could not be verifiably closed",
                         "                pass")],
          probe_runner_does_not_call_a_failed_to_close_session_clean, "cleanup of the lost session"),
    "runner: a failure on a recovered locator counts as reproduced":
        M("runner.py", [("            return Signal.UNKNOWN\n        return Signal.OK", "            return Signal.OK\n        return Signal.OK")],
          probe_a_failure_on_a_recovered_locator_is_not_reproduced, "reproduction"),
    "runner: a run with a recovered locator may pass":
        M("runner.py", [("        if c.locator_recovery_used and got.outcome in (Outcome.PASS, Outcome.PRODUCT_FAIL):", "        if False:")],
          probe_a_recovered_locator_never_passes, "locator recovery never certifies"),
    "runner: every failed retry opens a new event":
        M("runner.py", [("        if made.error is not None and p.strategy is RecoveryStrategy.RETRY_SAME and self._same_fault(p, made.error):",
                         "        if False:")],
          probe_a_plain_retry_met_twice_is_one_episode, "attempts counted in one record"),
}


def test_every_mutation_anchor_is_unique_and_every_probe_holds_on_the_real_code():
    for name, (filename, replacements, probe, why) in MUTATIONS.items():
        assert probe(real(filename)) is True, name


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_removing_the_mechanism_makes_the_probe_fail(name, mutants):
    filename, replacements, probe, why = MUTATIONS[name]
    mutant = mutants.load(filename, replacements)
    assert not holds(probe, mutant), f"mutation survived: {name} ({why})"


def test_the_mutation_table_covers_the_listed_recovery_mechanisms():
    touched = {filename for filename, *_ in MUTATIONS.values()}
    assert touched == {"recovery.py", "runner.py"}
    wanted = ("several controls", "uniqueness flag", "exactly one control", "role must match", "name must match",
              "detection first", "candidate cap", "attempt cap", "time cap", "idempotency", "replay guard",
              "step postcondition", "kill switch", "permitted recovery modes", "single strict gate",
              "known fixture", "verified checkpoint", "policy for replays", "locator recovery never certifies")
    assert all(any(w in why for _, _, _, why in MUTATIONS.values()) for w in wanted)
    assert len(MUTATIONS) >= 40
