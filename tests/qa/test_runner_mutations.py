# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mutation tests. Each safety mechanism of the single-run layer is removed from a copy of its source (an exact-text
replacement that must match once) and a probe that holds on the real code must fail on the mutant: remove the
deny-by-default, a deadline check, the idempotency guard, the transition check, the chain verification, ... and the
suite goes red. The probes are small end-to-end scenarios against the fake adapter, so they check the behaviour the
mechanism exists for, not the text of the code."""
import importlib
import json
import random

import pytest

from agent_harness.qa import records as R
from agent_harness.qa.classify import Classification
from agent_harness.qa.errors import ActionTimeout
from agent_harness.qa.limits import LimitCaps
from agent_harness.qa.policy import DefaultActionPolicy, make_decision
from agent_harness.qa.run_errors import ReceiptChainBroken
from agent_harness.qa.verifier import HaltLatch

from ._fx_verify import Mutants, REF_DOCS
from ._fx_runner import (APP, FakeClock, Fault, UrlEndsPostcondition, default_policy, make_env, make_spec, nav_step, operations,
                  save_step)
from .test_idempotency import GRID, inputs as retry_inputs, oracle as retry_oracle
from .test_receipts import event_tampers, receipt, tamper_cases
from .test_runner import (BUDGET, EditingStore, Fixed, Hook, Raising, Slow, clicks, recover_spec, renamed_save,
                          retry_spec, save_timeout, semantic, timed)
from .test_runstate import ALL_PAIRS, EXPECTED
from .test_policy import ALL_KINDS


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


# =====================================================================================================================
# probes: policy
# =====================================================================================================================

def spec_with_effect(effect, listed=True):
    save = save_step(side_effect=effect, idempotency="AT_MOST_ONCE")
    return make_spec([nav_step(), save], allowed_side_effects=[effect] if listed else [])


def probe_policy_default_denies_all_kinds(P):
    spec = make_spec()
    bare = P.DefaultActionPolicy(spec=spec, allowed_origins=[APP])        # origins allowed, no action kind listed
    return all(not bare.evaluate(s).allow for s in spec.steps)


def probe_policy_side_effect_must_be_listed(P):
    spec = make_spec(allowed_side_effects=[])
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    return not pol.evaluate(spec.steps[1]).allow and pol.evaluate(spec.steps[0]).allow


def probe_policy_origin_must_be_allowed(P):
    spec = make_spec()
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=["https://other.example.test"])
    return all(not pol.evaluate(s).allow for s in spec.steps)


def probe_policy_external_classes_need_a_local_sink(P):
    for effect in ("WRITES_EXTERNAL", "NOTIFIES_HUMAN", "SPENDS_MONEY"):
        spec = spec_with_effect(effect)
        pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
        if pol.evaluate(spec.steps[1]).allow:
            return False
        sink = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP],
                                     local_sink_origins=[APP])
        if not sink.evaluate(spec.steps[1]).allow:
            return False
    return True


def probe_policy_unknown_side_effect_class_is_denied(P):
    spec = make_spec()
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    saved = P.SIDE_EFFECT_RULES
    P.SIDE_EFFECT_RULES = {k: v for k, v in saved.items() if k is not R.SideEffect.WRITES_INTERNAL}
    try:
        return not pol.evaluate(spec.steps[1]).allow
    finally:
        P.SIDE_EFFECT_RULES = saved


def probe_policy_unknown_idempotency_is_denied(P):
    spec = make_spec([nav_step(), save_step(idempotency="UNKNOWN")])
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    return not pol.evaluate(spec.steps[1]).allow


def probe_policy_errors_deny(P):
    spec = make_spec()
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])

    def boom(step):
        raise RuntimeError("x")
    pol.evaluate = boom
    return pol.decide("run-1", spec.steps[1], 1).decision is R.PolicyDecisionKind.DENY


def probe_policy_only_the_reviewed_step_is_authorised(P):
    spec = make_spec()
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    changed = R.Step.from_json({**spec.steps[1].to_json(), "intent": "something else"})
    return not pol.evaluate(changed).allow


def probe_policy_require_approval_is_not_an_allow(P):
    spec = make_spec()
    base = make_decision(run_id="run-1", step=spec.steps[1], attempt=1, allow=True, rule_id="r", reason="r",
                         principal="p", app="a", provider="x")
    pending = R.PolicyDecisionRecord.from_json({**base.to_json(), "decision": "REQUIRE_APPROVAL"})
    return P.is_allowed(base) and not P.is_allowed(pending)


def probe_policy_decision_must_bind_run_and_step(P):
    spec = make_spec()
    good = make_decision(run_id="run-1", step=spec.steps[1], attempt=1, allow=True, rule_id="r", reason="r",
                         principal="p", app="a", provider="x")
    return (P.binding_problem(good, "run-1", spec.steps[1]) is None
            and P.binding_problem(good, "run-2", spec.steps[1]) is not None
            and P.binding_problem(good, "run-1", spec.steps[0]) is not None)


# =====================================================================================================================
# probes: retry decision, state machine, limits, receipts
# =====================================================================================================================

def probe_retry_matches_the_oracle(I):
    for point in GRID:
        (se, idem, rs, pc, executed, key, retryable, enabled, permitted, left) = point
        args = dict(side_effect=se, idempotency=idem, retry_safety=rs, postcondition=pc, may_have_executed=executed,
                    has_idempotency_key=key, error_retryable=retryable, recovery_enabled=enabled,
                    permitted_recovery=permitted, retries_left=left)
        if I.decide_retry(I.RetryInputs(**args)).allow is not retry_oracle(retry_inputs(**args)):
            return False
    return True


def probe_machine_table(M):
    for src, dst in ALL_PAIRS:
        m = M.RunStateMachine()
        m._state = M.RunState(src.value)
        try:
            m.to(M.RunState(dst.value))
            moved = True
        except Exception:
            moved = False
        if moved is not ((src, dst) in EXPECTED):
            return False
    return True


def probe_run_deadline(L):
    clock = FakeClock()
    d = L.Deadlines(clock, L.effective_limits(R.Budget(deadline_ms=100)))
    clock.advance(100)
    try:
        d.check_run()
    except Exception:
        return True
    return False


def probe_step_deadline(L):
    clock = FakeClock()
    timer = L.Deadlines(clock, L.effective_limits(R.Budget(deadline_ms=1000), L.LimitCaps(step_deadline_ms=10))).begin_step()
    clock.advance(10)
    try:
        timer.check()
    except Exception:
        return True
    return False


def probe_observation_deadline(L):
    clock = FakeClock()
    d = L.Deadlines(clock, L.effective_limits(R.Budget(deadline_ms=1000)))
    clock.advance(5)
    try:
        d.check_observation(0, 5)
    except Exception:
        return True
    return False


def probe_budget_never_exceeds(L):
    b = L.Budgets(L.effective_limits(R.Budget(deadline_ms=10, max_retries=1)))
    b.charge_retry()
    try:
        b.charge_retry()
    except Exception:
        return b.retries == 1
    return False


def probe_caps_only_lower(L):
    spec = R.Budget(deadline_ms=100, max_retries=1)
    return L.effective_limits(spec, L.LimitCaps(max_retries=50, deadline_ms=9999)).max_retries == 1


def probe_chain_detects_tampering(RC):
    for _, receipts, links in tamper_cases():
        try:
            RC.verify_chain(receipts, links)
            return False
        except ReceiptChainBroken:
            continue
    return True


def probe_ledger_attempts_are_consecutive(RC):
    led = RC.ReceiptLedger("run-1")
    step = make_spec().steps[1]
    try:
        led.append(receipt(step, attempt=3))
    except ReceiptChainBroken:
        return True
    return False


def probe_ledger_rejects_another_run(RC):
    led = RC.ReceiptLedger("run-1")
    try:
        led.append(receipt(run_id="run-2"))
    except ReceiptChainBroken:
        return True
    return False


def probe_manifest_binding(RC):
    r = make_env().run()
    altered = [r.receipts[0], R.ActionReceipt.from_json({**r.receipts[1].to_json(), "result": "ACTION_TIMEOUT"})]
    try:
        RC.verify_manifest_receipts(r.manifest, altered)
    except ReceiptChainBroken:
        return True
    return False


def probe_events_detect_tampering(RC):
    for _, events in event_tampers():
        try:
            RC.verify_events(events)
            return False
        except ReceiptChainBroken:
            continue
    return True


def probe_state_hash_covers_dom_meta(RC):
    from .test_receipts import obs
    return RC.state_hash(obs()) != RC.state_hash(obs(dom_meta={"title": "T", "nodes": 4}))


def probe_unknown_state_is_per_attempt(RC):
    return RC.unknown_state_hash("run-1", "s1", 1) != RC.unknown_state_hash("run-1", "s1", 2)


# =====================================================================================================================
# probes: the runner (each takes the mutated or real runner module)
# =====================================================================================================================

def run_env(RN, spec=None, **kw):
    env = make_env(spec, mod=RN, **kw)
    return env, env.run()


def probe_runner_never_acts_without_an_allow(RN):
    spec = make_spec()
    env, r = run_env(RN, spec, policy=DefaultActionPolicy(spec=spec))
    return operations(env.adapter, "act") == [] and r.outcome is R.Outcome.INCONCLUSIVE


def probe_runner_checks_the_decision_binding(RN):
    forged = lambda run_id, step, attempt: make_decision(run_id="another-run", step=step, attempt=attempt, allow=True,
                                                         rule_id="x", reason="x", principal="p", app="a", provider="evil")
    env, r = run_env(RN, policy=Fixed(forged))
    return operations(env.adapter, "act") == []


def probe_runner_denies_when_the_policy_raises(RN):
    env, r = run_env(RN, policy=Raising())
    return operations(env.adapter, "act") == [] and r.outcome is R.Outcome.INCONCLUSIVE


def probe_runner_denies_without_a_policy(RN):
    env, r = run_env(RN, policy=object())
    env2 = make_env(mod=RN)
    env2.runner._policy = None
    r2 = env2.run()
    return (operations(env.adapter, "act") == [] and operations(env2.adapter, "act") == []
            and r.outcome is R.Outcome.INCONCLUSIVE and r2.policy_decisions[0].rule_id == "policy-missing")


def probe_runner_denies_when_the_policy_is_too_slow(RN):
    spec = make_spec()
    env = make_env(spec, mod=RN, config_kw={"policy_timeout_ms": 50})
    env.runner._policy = Slow({"clock": env.clock}, default_policy(spec), 50)
    r = env.run()
    return operations(env.adapter, "act") == [] and r.outcome is R.Outcome.INCONCLUSIVE


def probe_runner_enforces_the_run_deadline(RN):
    rng = random.Random(77)
    for _ in range(25):
        deadline = rng.randint(1, 300)
        lat = {op: rng.randint(0, 60) for op in ("open_session", "observe", "locate", "act", "assert_")}
        env, r = run_env(RN, timed(deadline_ms=deadline), adapter_setup=lambda a, lat=lat: a.latency.update(lat))
        if any(started >= deadline for op, started in env.adapter.times
               if op in ("open_session", "locate", "act", "assert_")):
            return False
    env, r = run_env(RN, timed(deadline_ms=100), adapter_setup=lambda a: a.latency.update(act=200))
    return r.outcome is R.Outcome.ENVIRONMENT_FAIL


def probe_runner_enforces_the_step_deadline(RN):
    env = make_env(mod=RN, config_kw={"caps": LimitCaps(step_deadline_ms=50)},
                   adapter_setup=lambda a: a.latency.update(observe=20, locate=20, act=20))
    return env.run().outcome is R.Outcome.INCONCLUSIVE


def probe_runner_enforces_the_observation_deadline(RN):
    env, r = run_env(RN, adapter_setup=lambda a: a.latency.update(observe=5000))
    return operations(env.adapter, "act") == [] and r.outcome is R.Outcome.INCONCLUSIVE


def probe_runner_never_replays_an_unsafe_write(RN):
    save = save_step(permitted=["RETRY_SAME"], retry_safety="UNSAFE")
    env, r = run_env(RN, retry_spec(save=save), adapter_setup=save_timeout(apply=True),
                     config_kw={"recovery_enabled": True})
    return len(clicks(env)) == 1


def probe_runner_does_not_replay_an_applied_effect(RN):
    save = save_step(permitted=["RETRY_SAME"], retry_safety="SAFE_WITH_IDEMPOTENCY_KEY")
    env, r = run_env(RN, retry_spec(save=save), adapter_setup=save_timeout(apply=True),
                     config_kw={"recovery_enabled": True}, postconditions=UrlEndsPostcondition())
    return r.outcome is R.Outcome.PASS and len(clicks(env)) == 1


def probe_runner_never_takes_the_first_of_two_matches(RN):
    def twins(a):
        base = a.on_navigate

        def nav(adapter, url):
            base(adapter, url)
            adapter.elements["btn-save-2"] = dict(adapter.elements["btn-save"])
        a.on_navigate = nav
    env, r = run_env(RN, adapter_setup=twins)
    return all(c[1][2] != "btn-save" for c in operations(env.adapter, "act"))


def probe_runner_checks_the_located_name(RN):
    def lying(a):
        real = a.locate

        def locate(s, target, *, mode):
            res = real(s, target, mode=mode)
            if target.name == "Save":
                return R.LocateResult(candidates=(R.LocateCandidate(candidate_ref="btn-delete", enabled=True, unique=True,
                                                                    visible=True, name="Delete", role="button"),))
            return res
        a.locate = locate
    env, r = run_env(RN, adapter_setup=lying)
    return env.adapter.deleted is False


def probe_runner_checks_a_recovered_control_is_the_authored_one(RN):
    hook = Hook(semantic("btn-delete"))
    env, r = run_env(RN, recover_spec(), recovery=hook, config_kw={"recovery_enabled": True},
                     adapter_setup=lambda a: (renamed_save(a), a.semantic.update({"Save": "btn-delete"})))
    return env.adapter.deleted is False


def probe_runner_loads_the_assertions_again_at_verify(RN):
    docs = json.loads(json.dumps(REF_DOCS))
    docs[0]["predicate"]["expected"] = "Anything"
    env, r = run_env(RN, store=EditingStore(REF_DOCS, docs))
    return operations(env.adapter, "assert_") == [] and r.classification.reason == "ASSERTION_INTEGRITY"


def probe_runner_never_passes_a_locator_recovery(RN):
    env, r = run_env(RN, recover_spec(), adapter_setup=renamed_save, recovery=Hook(semantic()),
                     config_kw={"recovery_enabled": True})
    return r.outcome is not R.Outcome.PASS and r.outcome is not R.Outcome.PRODUCT_FAIL


def probe_runner_appends_run_started_first(RN):
    env, r = run_env(RN)
    first = [x for x in env.log if x in (("evidence", "RunStarted"), ("adapter", "open_session"))]
    return first == [("evidence", "RunStarted"), ("adapter", "open_session")]


def probe_runner_closes_the_session_on_abort(RN):
    env, r = run_env(RN, adapter_setup=lambda a: a.faults.update({"act": [Fault(ActionTimeout("x"))]}))
    return len(operations(env.adapter, "close_session")) == 1


def probe_runner_honours_the_batch_latch(RN):
    halt = HaltLatch()
    halt.trip(ReceiptChainBroken("earlier"))
    env, r = run_env(RN, halt=halt)
    if env.adapter.calls != [] or not r.halt_batch:
        return False
    # and a latch tripped while the run is in progress (by another run of the batch) stops the next step
    halt2 = HaltLatch()
    env2 = make_env(mod=RN, halt=halt2)
    base = env2.adapter.act

    def act(s, a, t):
        halt2.trip(ReceiptChainBroken("tripped meanwhile"))
        return base(s, a, t)
    env2.adapter.act = act
    env2.run()
    return clicks(env2) == []


def probe_runner_honours_cancellation(RN):
    from agent_harness.qa.hooks import CancelToken
    cancel = CancelToken()
    cancel.cancel("stop")
    env, r = run_env(RN, cancel=cancel)
    return env.adapter.calls == [] and r.cancelled


def probe_runner_refuses_unredacted_observations(RN):
    env, r = run_env(RN, adapter_setup=lambda a: setattr(a, "unredacted", True))
    return operations(env.adapter, "act") == [] and r.outcome is R.Outcome.INCONCLUSIVE


def probe_runner_refuses_a_classification_the_table_refuses(RN):
    spec = make_spec()
    env = make_env(spec, mod=RN, policy=DefaultActionPolicy(spec=spec))
    forged = Classification(outcome=R.Outcome.PASS, reason=None, verdict=R.Verdict.PASS, halt_batch=False,
                            review_required=False, rule=6)
    original = RN.classify
    RN.classify = lambda facts: forged
    try:
        r = env.run()
    finally:
        RN.classify = original
    return r.outcome is R.Outcome.INCONCLUSIVE and r.classification.reason == "ASSERTION_INTEGRITY"


def probe_runner_charges_model_calls(RN):
    hook = Hook(semantic(strategy=R.RecoveryStrategy.MODEL_CANDIDATE, model_calls=1, tokens=10))
    env, r = run_env(RN, make_spec([nav_step(), save_step(permitted=["MODEL_CANDIDATE"])]), adapter_setup=renamed_save,
                     recovery=hook, config_kw={"recovery_enabled": True, "model_recovery_enabled": True})
    return all(c[1][3] != "MODEL_CANDIDATE" for c in operations(env.adapter, "act")) and r.manifest.counters.model_calls == 0


def probe_runner_gates_model_recovery(RN):
    spec = make_spec([nav_step(), save_step(permitted=["MODEL_CANDIDATE"])],
                     budget={**BUDGET, "max_model_calls": 1, "max_tokens": 100})
    hook = Hook(semantic(strategy=R.RecoveryStrategy.MODEL_CANDIDATE, model_calls=1, tokens=10))
    env, r = run_env(RN, spec, adapter_setup=renamed_save, recovery=hook, config_kw={"recovery_enabled": True})
    return hook.requests == []


# =====================================================================================================================
# the table of mutations
# =====================================================================================================================

def M(filename, replacements, probe, *, why):
    return (filename, replacements, probe, why)


MUTATIONS = {
    # --- policy.py: remove the deny-by-default ------------------------------------------------------------------
    "policy: an empty action-kind allow-list allows everything":
        M("policy.py", [("        self._kinds = frozenset(kinds)\n", "        self._kinds = frozenset(kinds) or frozenset(ActionKind)\n")],
          probe_policy_default_denies_all_kinds, why="default deny"),
    "policy: the action-kind rule is dropped":
        M("policy.py", [("        if kind not in self._kinds:", "        if False:")],
          probe_policy_default_denies_all_kinds, why="action kinds"),
    "policy: unlisted side effects are allowed":
        M("policy.py", [("        if need != NEED_NOTHING and effect not in self._allowed_effects:", "        if False:")],
          probe_policy_side_effect_must_be_listed, why="side-effect allowances come from the spec"),
    "policy: any origin is allowed":
        M("policy.py", [("        if origin not in self._origins:", "        if False:")],
          probe_policy_origin_must_be_allowed, why="origins"),
    "policy: external classes no longer need a local sink":
        M("policy.py", [("        if need == NEED_LISTED_AND_LOCAL_SINK and origin not in self._sinks:", "        if False:")],
          probe_policy_external_classes_need_a_local_sink, why="local sink"),
    "policy: an unknown side-effect class is not denied":
        M("policy.py", [("        if need is None:", "        if False:")],
          probe_policy_unknown_side_effect_class_is_denied, why="unknown class"),
    "policy: unknown idempotency is not denied":
        M("policy.py", [("        if not isinstance(step.idempotency, Idempotency) or step.idempotency is Idempotency.UNKNOWN:", "        if False:")],
          probe_policy_unknown_idempotency_is_denied, why="unknown idempotency"),
    "policy: an internal error is an allow":
        M("policy.py", [("            verdict = _deny(\"policy-internal-error\", \"the policy failed while evaluating the action\")",
                         "            verdict = Verdict(True, \"x\", \"x\")")],
          probe_policy_errors_deny, why="exceptions deny"),
    "policy: any step with a known id is authorised":
        M("policy.py", [("        if known is None or known[0] != step:", "        if known is None:")],
          probe_policy_only_the_reviewed_step_is_authorised, why="step binding"),
    "policy: REQUIRE_APPROVAL counts as an allow":
        M("policy.py", [("record.decision is PolicyDecisionKind.ALLOW", "record.decision is not PolicyDecisionKind.DENY")],
          probe_policy_require_approval_is_not_an_allow, why="only allow authorises"),
    "policy: a decision for another run or step is accepted":
        M("policy.py", [("    if record.run_id != run_id or record.step_id != step.step_id:", "    if False:")],
          probe_policy_decision_must_bind_run_and_step, why="decision binding"),
    # --- idempotency.py: remove the idempotency guard -------------------------------------------------------------
    "idempotency: an unsafe step may be retried":
        M("idempotency.py", [("    if i.retry_safety is RetrySafety.UNSAFE:", "    if False:")],
          probe_retry_matches_the_oracle, why="UNSAFE is never retried"),
    "idempotency: an applied effect may be replayed":
        M("idempotency.py", [("    if i.postcondition is PostconditionResult.SATISFIED:", "    if False:")],
          probe_retry_matches_the_oracle, why="no replay of an applied effect"),
    "idempotency: the budget gate is dropped":
        M("idempotency.py", [("    if i.retries_left <= 0:", "    if False:")],
          probe_retry_matches_the_oracle, why="bounded retries"),
    "idempotency: risky classes need only a key or a check":
        M("idempotency.py", [("        if keyed and not_applied:", "        if keyed or not_applied:")],
          probe_retry_matches_the_oracle, why="send/payment need both guards"),
    "idempotency: anything counts as idempotent":
        M("idempotency.py", [("        if i.idempotency is Idempotency.IDEMPOTENT:\n            return RetryDecision(True, ALLOW_IDEMPOTENT_DECLARED)",
                              "        if True:\n            return RetryDecision(True, ALLOW_IDEMPOTENT_DECLARED)")],
          probe_retry_matches_the_oracle, why="UNKNOWN is not a guard"),
    "idempotency: every attempt counts as never executed":
        M("idempotency.py", [("    if not i.may_have_executed:\n        return RetryDecision(True, ALLOW_NOT_EXECUTED)",
                              "    if True:\n        return RetryDecision(True, ALLOW_NOT_EXECUTED)")],
          probe_retry_matches_the_oracle, why="blind replay"),
    "idempotency: a key is assumed to be present":
        M("idempotency.py", [("             and i.has_idempotency_key)", "             and True)")],
          probe_retry_matches_the_oracle, why="the key must have been sent"),
    # --- runstate.py ----------------------------------------------------------------------------------------------
    "runstate: illegal transitions are allowed":
        M("runstate.py", [("        if not is_allowed(self._state, target):", "        if False:")],
          probe_machine_table, why="transition check"),
    "runstate: VERIFY may go back to ACT":
        M("runstate.py", [("    S.VERIFY: frozenset({S.PASS, S.PRODUCT_FAIL}) | _ABORTS,", "    S.VERIFY: frozenset({S.PASS, S.PRODUCT_FAIL, S.ACT}) | _ABORTS,")],
          probe_machine_table, why="table"),
    "runstate: PASS is not terminal":
        M("runstate.py", [("    S.PASS: frozenset(),", "    S.PASS: frozenset({S.ACT}),")],
          probe_machine_table, why="terminal states"),
    "runstate: ACT may jump to PASS":
        M("runstate.py", [("    S.ACT: frozenset({S.OBSERVE, S.VERIFY, S.RECOVERABLE}) | _ABORTS,", "    S.ACT: frozenset({S.OBSERVE, S.VERIFY, S.RECOVERABLE, S.PASS}) | _ABORTS,")],
          probe_machine_table, why="PASS only from VERIFY"),
    # --- limits.py ------------------------------------------------------------------------------------------------
    "limits: the run deadline check is removed":
        M("limits.py", [("        if elapsed >= self._limits.deadline_ms:", "        if False:")], probe_run_deadline, why="deadline"),
    "limits: the step deadline check is removed":
        M("limits.py", [("        if elapsed >= self._limit:", "        if False:")], probe_step_deadline, why="deadline"),
    "limits: the observation deadline check is removed":
        M("limits.py", [("        if elapsed >= deadline_ms:", "        if False:")], probe_observation_deadline, why="deadline"),
    "limits: a budget can be exceeded":
        M("limits.py", [("        if used + amount > limit:", "        if False:")], probe_budget_never_exceeds, why="budgets"),
    "limits: a cap can raise a limit":
        M("limits.py", [("    return spec_value if cap is None else min(spec_value, cap)", "    return spec_value if cap is None else cap")],
          probe_caps_only_lower, why="caps only lower"),
    # --- receipts.py ----------------------------------------------------------------------------------------------
    "receipts: the chain is not verified":
        M("receipts.py", [("        if link.index != index or link.prev != prev or link.receipt_hash != rhash or link.link_hash != expected:", "        if False:")],
          probe_chain_detects_tampering, why="receipt chain"),
    "receipts: attempts may skip":
        M("receipts.py", [("        if receipt.attempt != self._last_attempt.get(receipt.step_id, 0) + 1:", "        if False:")],
          probe_ledger_attempts_are_consecutive, why="append-only attempts"),
    "receipts: a receipt of another run is accepted":
        M("receipts.py", [("        if receipt.run_id != self._run_id:", "        if False:")],
          probe_ledger_rejects_another_run, why="one run"),
    "receipts: the manifest is not checked against the receipts":
        M("receipts.py", [("        if recorded.get(step_id) != tuple(refs):", "        if False:")],
          probe_manifest_binding, why="receipts hashed into the manifest"),
    "receipts: the event chain is not verified":
        M("receipts.py", [("            if body[\"seq\"] != seq or body[\"prev\"] != prev or event[\"event_hash\"] != canon.canon_hash(body):", "            if False:")],
          probe_events_detect_tampering, why="event chain"),
    "receipts: the state hash ignores DOM metadata":
        M("receipts.py", [("             \"dom_meta\": form.get(\"dom_meta\"), ", "             ")],
          probe_state_hash_covers_dom_meta, why="state hashes"),
    "receipts: an unobserved state is the same for every attempt":
        M("receipts.py", [("    return canon.to_contract_ref(canon.canon_hash({\"attempt\": attempt, \"form\": STATE_FORM,", "    return canon.to_contract_ref(canon.canon_hash({\"attempt\": 0, \"form\": STATE_FORM,")],
          probe_unknown_state_is_per_attempt, why="unknown state"),
    # --- runner.py ------------------------------------------------------------------------------------------------
    "runner: the adapter is called without an allow":
        M("runner.py", [("            if not is_allowed(decision):", "            if False:")],
          probe_runner_never_acts_without_an_allow, why="policy gate"),
    "runner: a decision for another run is accepted":
        M("runner.py", [("                elif binding_problem(record, c.run_id, step) is not None:", "                elif False:")],
          probe_runner_checks_the_decision_binding, why="decision binding"),
    "runner: a policy exception is not a denial":
        M("runner.py", [("            except Exception:\n                rule, why = \"policy-error\", \"the policy raised while deciding\"",
                        "            except ZeroDivisionError:\n                rule, why = \"policy-error\", \"the policy raised while deciding\"")],
          probe_runner_denies_when_the_policy_raises, why="exceptions deny"),
    "runner: a policy timeout is not a denial":
        M("runner.py", [("                if self._clock.monotonic_ms() - started >= self._cfg.policy_timeout_ms:", "                if False:")],
          probe_runner_denies_when_the_policy_is_too_slow, why="timeouts deny"),
    "runner: a missing policy authorises the action":
        M("runner.py", [("        if not callable(getattr(self._policy, \"decide\", None)):\n            rule, why = \"policy-missing\", \"there is no policy: nothing is authorised\"\n        else:",
                         "        if False:\n            pass\n        else:")],
          probe_runner_denies_without_a_policy, why="a missing policy is a denial"),
    "runner: the run deadline is not checked at boundaries":
        M("runner.py", [("            raise RunCancelled(\"the run was cancelled\", reason=self._cancel.reason)\n        c.deadlines.check_run()",
                         "            raise RunCancelled(\"the run was cancelled\", reason=self._cancel.reason)")],
          probe_runner_enforces_the_run_deadline, why="run deadline"),
    "runner: the run deadline is not checked inside a step":
        M("runner.py", [("        c.deadlines.check_run()\n        timer.check()", "        timer.check()")],
          probe_runner_enforces_the_run_deadline, why="run deadline"),
    "runner: the step deadline is not checked":
        M("runner.py", [("        c.deadlines.check_run()\n        timer.check()", "        c.deadlines.check_run()")],
          probe_runner_enforces_the_step_deadline, why="step deadline"),
    "runner: the observation deadline is not checked":
        M("runner.py", [("        obs = self._call(self._qa.observe, c.session, ObserveRequest(phase=phase, step_id=step.step_id))\n        c.deadlines.check_observation(started, step.observation_deadline_ms)",
                         "        obs = self._call(self._qa.observe, c.session, ObserveRequest(phase=phase, step_id=step.step_id))")],
          probe_runner_enforces_the_observation_deadline, why="observation deadline"),
    "runner: every failed step is replayed":
        M("runner.py", [("            if decision.allow:\n                c.machine.to(RunState.RECOVERABLE)", "            if True:\n                c.machine.to(RunState.RECOVERABLE)")],
          probe_runner_never_replays_an_unsafe_write, why="idempotency guard"),
    "runner: an executed action is treated as not executed":
        M("runner.py", [("                acted = True\n                outcome = self._call(self._qa.act", "                acted = False\n                outcome = self._call(self._qa.act")],
          probe_runner_never_replays_an_unsafe_write, why="idempotency guard"),
    "runner: an applied effect is not recognised":
        M("runner.py", [("            if decision.already_applied:\n                return \"applied\"", "            if False:\n                return \"applied\"")],
          probe_runner_does_not_replay_an_applied_effect, why="no replay of an applied effect"),
    "runner: the first of several matches is taken":
        M("runner.py", [("        if len(matching) > 1 or not matching[0].unique:", "        if False:")],
          probe_runner_never_takes_the_first_of_two_matches, why="uniqueness"),
    "runner: the located control's name is not checked":
        M("runner.py", [("                    and (descriptor.name is None or cand.name == descriptor.name)]", "                    and True]")],
          probe_runner_checks_the_located_name, why="never click Delete"),
    "runner: a recovered control may differ from the authored one":
        M("runner.py", [("        if not _same_label(authored.role, cand.role) or not _same_label(authored.name, cand.name):", "        if False:")],
          probe_runner_checks_a_recovered_control_is_the_authored_one, why="never click Delete"),
    "runner: the assertions are not loaded again at verification":
        M("runner.py", [("        again = load_frozen_assertions(self._store, c.pin, lineage=self._lineage, programs=self._programs,\n                                       halt=self._halt)",
                         "        again = c.frozen")],
          probe_runner_loads_the_assertions_again_at_verify, why="verifier loads by hash"),
    "runner: a locator recovery may pass":
        M("runner.py", [("        if c.locator_recovery_used and got.outcome in (Outcome.PASS, Outcome.PRODUCT_FAIL):", "        if False:")],
          probe_runner_never_passes_a_locator_recovery, why="unsealed recovery"),
    "runner: the session is opened before RunStarted":
        M("runner.py", [("        self._emit(c, \"RunStarted\", {", "        self._call(self._qa.open_session, SessionProfile(egress_profile=spec.egress_profile, redaction_policy_id=self._cfg.redaction_policy_id))\n        self._emit(c, \"RunStarted\", {")],
          probe_runner_appends_run_started_first, why="RunStarted first"),
    "runner: cleanup is skipped":
        M("runner.py", [("            cleanup_signal = self._cleanup(c)", "            cleanup_signal = Signal.OK")],
          probe_runner_closes_the_session_on_abort, why="cleanup"),
    "runner: the batch latch is ignored":
        M("runner.py", [("        self._halt.check()\n        if self._cancel is not None", "        if self._cancel is not None")],
          probe_runner_honours_the_batch_latch, why="halt path"),
    "runner: cancellation is ignored":
        M("runner.py", [("        if self._cancel is not None and self._cancel.cancelled:", "        if False:")],
          probe_runner_honours_cancellation, why="cancel path"),
    "runner: unredacted observations are accepted":
        M("runner.py", [("        if not obs.redaction_applied or obs.redaction_policy_id != self._cfg.redaction_policy_id:", "        if False:")],
          probe_runner_refuses_unredacted_observations, why="redaction"),
    "runner: an unreachable classification is not refused":
        M("runner.py", [("        if not is_allowed_transition(c.machine.state, OUTCOME_STATE[classification.outcome]):", "        if False:")],
          probe_runner_refuses_a_classification_the_table_refuses, why="the machine follows the table"),
    "runner: model calls are not charged":
        M("runner.py", [("            c.budgets.charge_model(calls, response.tokens)     # zero unless the reviewed spec allows model calls", "            pass")],
          probe_runner_charges_model_calls, why="model budget"),
    "runner: model recovery is not gated":
        M("runner.py", [("        if RecoveryStrategy.MODEL_CANDIDATE in permitted and not self._cfg.model_recovery_enabled:", "        if False:")],
          probe_runner_gates_model_recovery, why="model recovery off by default"),
}


def real(filename):
    return importlib.import_module("agent_harness.qa." + filename[:-3])


def test_every_mutation_anchor_is_unique_and_every_probe_holds_on_the_real_code():
    for name, (filename, replacements, probe, why) in MUTATIONS.items():
        assert probe(real(filename)) is True, name


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_removing_the_mechanism_makes_the_probe_fail(name, mutants):
    filename, replacements, probe, why = MUTATIONS[name]
    mutant = mutants.load(filename, replacements)
    assert not holds(probe, mutant), f"mutation survived: {name} ({why})"


def test_the_mutation_table_covers_every_runner_module_that_enforces_something():
    touched = {filename for filename, *_ in MUTATIONS.values()}
    assert touched == {"policy.py", "idempotency.py", "runstate.py", "limits.py", "receipts.py", "runner.py"}
    assert len(MUTATIONS) >= 45
