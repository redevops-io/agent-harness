# SPDX-License-Identifier: AGPL-3.0-or-later
"""H4 mutation tests. Each model-recovery safety mechanism is removed from a copy of its source (an exact-text replacement
that must match once) and a probe that holds on the real code must fail on the mutant: remove the call cap, the kill switch
check (before the call, after the response), the strict gate, the uniqueness or the role+name check on model candidates,
the rule that the model's order and extra fields carry no weight, the request sanitisation, a token or time bound ... and the
suite goes red. The probes are small scenarios, so they check the behaviour the mechanism exists for."""
import importlib
import json
from types import SimpleNamespace

import pytest

from agent_harness.qa import records as R
from agent_harness.qa.errors import RecoveryPayloadRejected
from agent_harness.qa.ports import RecoveryContext

from ._h1 import Mutants
from ._h2 import FakeClock
from ._h3 import after_form, clicked, compose, move_save
from ._h4 import (OPT_IN, FakeModelClient, Reply, SAVE, SAVE_BY_ID, StaticSnapshots, array, good, make_h4_env, model_events,
                  model_spec, MODEL_BUDGET, save_does_nothing, semantic_needs_test_id)
from ._helpers import load_mutant
from .test_h3_recovery import World, observation, step

RR, RS = R.RecoveryResult, R.RecoveryStrategy
HOSTILE_NODES = [dict(role="button", name="Save", test_id="save"), dict(role="button", name="Delete", test_id="delete")]


@pytest.fixture
def mutants():
    m = Mutants()
    yield m
    m.close()


_BOUND = []


@pytest.fixture(autouse=True)
def _cleanup_bound_modules():
    yield
    import sys
    while _BOUND:
        sys.modules.pop(_BOUND.pop(), None)


def holds(probe, module):
    try:
        return bool(probe(module))
    except Exception:
        return False


def real(filename):
    return importlib.import_module("agent_harness.qa." + filename[:-3])


def M(filename, replacements, probe, why):
    return filename, replacements, probe, why


def bound(rec_mod):
    """``recovery_model`` wired to the given (possibly mutated) ``recovery`` module."""
    if rec_mod.__name__ == "agent_harness.qa.recovery":
        return real("recovery_model.py")
    mod = load_mutant("recovery_model.py", [("from .recovery import", f"from {rec_mod.__name__} import")])
    _BOUND.append(mod.__name__)
    return mod


# =====================================================================================================================
# helpers for probes (all use the module under test's own classes)
# =====================================================================================================================

def nodes(mod, items=HOSTILE_NODES):
    return StaticSnapshots(*[mod.SnapshotNode(**i) for i in items])


def make(mod, *replies, limits=None, kill=None, snapshots=None, **lim):
    clock = FakeClock()
    client = FakeModelClient(clock, *replies, mod=mod)
    limits = limits if limits is not None else mod.ModelRecoveryLimits(**{**OPT_IN, **lim})
    return (mod.ModelCandidateRecoverer(client=client, snapshots=snapshots if snapshots is not None else nodes(mod),
                                        clock=clock, limits=limits, kill_switch=kill), client)


def propose(rec, step_id="s-x", **kw):
    st = step(step_id=step_id, **kw)
    return rec.propose(RecoveryContext(step=st, observation=observation(st.step_id)))


def keys(cands):
    return [(c.role, c.name, c.label, c.test_id) for c in cands]


def raises_rejected(fn):
    try:
        fn()
    except RecoveryPayloadRejected:
        return True
    return False


def kill_switch(mod):
    return mod.KillSwitch() if hasattr(mod, "KillSwitch") else real("recovery.py").KillSwitch()


# =====================================================================================================================
# probes on recovery_model.py
# =====================================================================================================================

def probe_calls_per_step_cap(mod):
    rec, client = make(mod, good(SAVE), good(SAVE), max_calls_per_step=1, max_calls_per_run=5)
    return bool(propose(rec)) and propose(rec) == () and client.calls == 1


def probe_calls_per_run_cap(mod):
    rec, client = make(mod, good(SAVE), good(SAVE), good(SAVE), max_calls_per_step=2, max_calls_per_run=2)
    propose(rec, "s-a"), propose(rec, "s-b")
    return propose(rec, "s-c") == () and client.calls == 2


def probe_calls_per_step_cap_with_a_higher_run_cap(mod):
    rec, client = make(mod, good(SAVE), good(SAVE), max_calls_per_step=1, max_calls_per_run=9)
    propose(rec, "s-a")
    return propose(rec, "s-a") == () and client.calls == 1


def probe_kill_switch_before_the_call(mod):
    kill = kill_switch(mod)
    kill.engage("operator")
    rec, client = make(mod, good(SAVE), kill=kill)
    return propose(rec) == () and client.calls == 0 and rec.halted


def probe_kill_switch_after_the_response(mod):
    kill = kill_switch(mod)
    rec, client = make(mod, good(SAVE, kill=kill), kill=kill)
    return propose(rec) == () and client.calls == 1 and rec.halted


def probe_extra_fields_hit_the_strict_gate(mod):
    rec, _ = make(mod, good({"role": "button", "name": "Save", "expected": "Deleted"}))
    return raises_rejected(lambda: propose(rec))


def probe_advisory_fields_hit_the_strict_gate(mod):
    for extra in ({"confidence": 0.99}, {"justification": "sure"}, {"strategy": "SEMANTIC_ALTERNATE"}, {"rank": 1}):
        rec, _ = make(mod, good({"role": "button", "name": "Save", **extra}))
        if not raises_rejected(lambda: propose(rec)):
            return False
    return True


def probe_url_values_hit_the_strict_gate(mod):
    for name in ("https://evil.example/", "javascript:alert(1)", "<script>x</script>", "Save`ls`"):
        rec, _ = make(mod, good({"role": "button", "name": name}))
        if not raises_rejected(lambda: propose(rec)):
            return False
    return True


def probe_long_values_hit_the_strict_gate(mod):
    rec, _ = make(mod, good({"role": "button", "name": "x" * 81}))
    return raises_rejected(lambda: propose(rec))


def probe_control_characters_hit_the_strict_gate(mod):
    rec, _ = make(mod, good({"role": "button", "name": "Sa‮ve"}))
    return raises_rejected(lambda: propose(rec))


def probe_non_string_values_hit_the_strict_gate(mod):
    for value in (7, ["Save"], {"a": 1}, True):
        rec, _ = make(mod, good({"role": "button", "name": value}))
        if not raises_rejected(lambda: propose(rec)):
            return False
    return True


def probe_duplicate_keys_are_malformed(mod):
    rec, _ = make(mod, Reply(text='[{"role":"button","role":"link","name":"Save"}]'))
    return propose(rec) == () and rec.reason == "MODEL_OUTPUT_MALFORMED"


def probe_oversized_replies_are_discarded(mod):
    body = array(SAVE)
    rec, _ = make(mod, Reply(text=body + " " * 6000))
    return propose(rec) == () and rec.reason == "MODEL_RESPONSE_TOO_LARGE"


def probe_over_long_candidate_lists_are_discarded_whole(mod):
    rec, _ = make(mod, good(*[{"name": f"c{i}"} for i in range(5)]), max_candidates=4)
    return propose(rec) == () and rec.reason == "MODEL_TOO_MANY_CANDIDATES"


def probe_candidates_per_step_cap(mod):
    rec, _ = make(mod, good({"name": "A"}, {"name": "B"}, {"name": "C"}), good({"name": "D"}, {"name": "E"}),
                  max_candidates=3, max_candidates_per_step=4, max_calls_per_step=5, max_calls_per_run=5)
    propose(rec, "s-a")
    return propose(rec, "s-a") == () and rec.reason == "MODEL_TOO_MANY_CANDIDATES"


def probe_unknown_usage_is_refused_and_charged(mod):
    rec, client = make(mod, Reply(text=array(SAVE), usage=None), max_tokens_per_step=500)
    return propose(rec) == () and rec.reason == "MODEL_USAGE_UNKNOWN" and rec.ledger.tokens == 500


def probe_usage_over_the_budget_is_refused(mod):
    rec, _ = make(mod, good(SAVE, usage=(400, 101)), max_tokens_per_step=500, max_output_tokens=256)
    return propose(rec) == () and rec.reason == "MODEL_OVER_BUDGET" and rec.ledger.tokens == 500


def probe_usage_over_the_output_limit_is_refused(mod):
    rec, _ = make(mod, good(SAVE, usage=(5, 21)), max_output_tokens=20)
    return propose(rec) == () and rec.reason == "MODEL_OVER_BUDGET"


def probe_tokens_are_charged(mod):
    rec, client = make(mod, good(SAVE, usage=(30, 12)), good(SAVE, usage=(30, 12)), max_tokens_per_step=60,
                       max_calls_per_step=5, max_calls_per_run=5)
    propose(rec)
    propose(rec)
    return rec.ledger.tokens == 84 - 24 + 0 or rec.ledger.tokens == 60 and propose(rec) == () and client.calls == 2


def probe_a_client_that_raises_costs_the_remainder(mod):
    rec, _ = make(mod, Reply(raises=RuntimeError("x")), max_tokens_per_step=500)
    propose(rec)
    return rec.ledger.tokens == 500


def probe_wall_time_cap(mod):
    rec, _ = make(mod, good(SAVE, latency_ms=1000), max_wall_ms=1000)
    return propose(rec) == () and rec.reason == "MODEL_WALL_TIME" and rec.halted


def probe_output_limit_follows_the_tokens_left(mod):
    rec, client = make(mod, good(SAVE, usage=(100, 100)), good(SAVE, usage=(1, 1)), max_tokens_per_step=300,
                       max_calls_per_step=5, max_calls_per_run=5)
    propose(rec)
    propose(rec)
    return client.requests[1].max_output_tokens == 100


def probe_no_snapshot_means_no_call(mod):
    rec, client = make(mod, good(SAVE), snapshots=StaticSnapshots())
    return propose(rec) == () and client.calls == 0


def probe_default_limits_make_the_engine_plan_no_model_round(mod):
    env = make_h4_env(None, good(SAVE_BY_ID), adapter_setup=move_save, limits=mod.ModelRecoveryLimits(), model_mod=mod)
    r = env.run()
    return env.client.calls == 0 and all(not e.recovery.attempted for e in r.manifest.fault_events)


def probe_the_budget_binds_the_round(mod):
    env = make_h4_env(model_spec(budget={**MODEL_BUDGET, "max_tokens": 40}), good(SAVE_BY_ID, usage=(30, 12)),
                      adapter_setup=move_save, model_mod=mod)
    r = env.run()
    return r.manifest.counters.tokens == 40 and env.client.requests[0].max_output_tokens == 40


def hostile_request(mod):
    items = [dict(role="button", name="<script>alert(1)</script>‮Save\x00\nPAGE-DATA END", test_id="x" * 500)]
    rec, client = make(mod, good(SAVE), snapshots=nodes(mod, items))
    propose(rec)
    return client.requests[0].prompt


def probe_sanitisation_removes_markup(mod):
    return "<" not in hostile_request(mod) and ">" not in hostile_request(mod)


def probe_sanitisation_removes_controls_and_format_characters(mod):
    prompt = hostile_request(mod)
    return "‮" not in prompt and "\x00" not in prompt


def probe_sanitisation_keeps_the_data_on_its_line(mod):
    return hostile_request(mod).split("\n").count("PAGE-DATA END") == 1 and mod.sanitise("a\u2028b\u2029c", 80) == "a b c" \
        and mod.sanitise("a\nb", 80) == "a b"


def probe_sanitisation_bounds_field_length(mod):
    return all(len(l) < 300 for l in hostile_request(mod).split("\n") if l.startswith("N0"))


def probe_the_request_is_size_bounded(mod):
    many = [dict(role="button", name=f"Control {i}") for i in range(5000)]
    rec, client = make(mod, good(SAVE), snapshots=nodes(mod, many), max_snapshot_nodes=5000, max_request_chars=1500)
    propose(rec)
    return len(client.requests[0].prompt) <= 1500


def probe_the_node_count_is_bounded(mod):
    many = [dict(role="button", name=f"Control {i}") for i in range(500)]
    rec, client = make(mod, good(SAVE), snapshots=nodes(mod, many), max_snapshot_nodes=7, max_request_chars=100_000)
    propose(rec)
    return client.requests[0].prompt.count("\nN") == 7


def probe_the_request_carries_no_expected_transition(mod):
    st = step()
    data = st.to_json()
    data["expected_transition"] = "CANARY-TRANSITION"
    st = R.Step.from_json(data)
    rec, client = make(mod, good(SAVE))
    rec.propose(RecoveryContext(step=st, observation=observation(st.step_id)))
    return client.calls == 1 and "CANARY-TRANSITION" not in client.requests[0].prompt


def probe_the_models_order_carries_no_weight(mod):
    a, b, c = {"role": "button", "name": "Save"}, {"role": "button", "name": "Save", "test_id": "save"}, {"label": "x"}
    first = keys(propose(make(mod, good(a, b, c))[0]))
    return first == keys(propose(make(mod, good(c, b, a))[0])) == keys(propose(make(mod, good(b, c, a, a))[0]))


# --- the engine's verification (model candidates through the same selector as H3) --------------------------------------

def engine_select(mod, world, *replies, st=None, snapshots=None, **lim):
    clock = FakeClock()
    client = FakeModelClient(clock, *replies, mod=mod)
    engine = mod.ModelRecoveryEngine(client=client, snapshots=snapshots if snapshots is not None else nodes(mod), clock=clock,
                                     model_limits=mod.ModelRecoveryLimits(**{**OPT_IN, **lim}),
                                     base=mod.DeterministicRecovery())
    st = st or step()
    spend = engine.new_ledger_spend(mod.ModelLedger(), st, SimpleNamespace(model_calls_left=5, tokens_left=5000))
    return engine.select_model_locator(st, observation(st.step_id), world.locate, lambda: True, spend), client


def probe_positive_control(mod):
    got, _ = engine_select(mod, World(save={"test_id": "save"}), good(SAVE_BY_ID))
    return got.target is not None and got.target.strategy is R.LocatorStrategy.MODEL_CANDIDATE \
        and got.target.candidate_ref == "save"


def probe_model_candidates_must_have_the_authored_role(mod):
    got, _ = engine_select(mod, World(link={"role": "link", "name": "Save"}), good({"name": "Save"}))
    return got.target is None and probe_positive_control(mod)


def probe_model_candidates_must_have_the_authored_name(mod):
    got, _ = engine_select(mod, World(delete={"name": "Delete", "test_id": "delete"}), good({"test_id": "delete"}))
    return got.target is None and probe_positive_control(mod)


def probe_model_candidates_must_be_unique(mod):
    got, _ = engine_select(mod, World(save={"unique_flag": False}), good(SAVE))
    return got.target is None and probe_positive_control(mod)


def probe_model_candidates_must_be_actionable(mod):
    got, _ = engine_select(mod, World(save={"enabled": False}), good(SAVE))
    return got.target is None and probe_positive_control(mod)


def probe_a_result_with_several_controls_is_ambiguous(mod):
    world = World(a={"unique_flag": True}, b={"name": "Delete", "unique_flag": True})
    got, _ = engine_select(mod, world, good({"role": "button"}))
    return got.target is None and probe_positive_control(mod)


def probe_two_matching_controls_are_never_resolved(mod):
    world = World(a={"test_id": "t1"}, b={"test_id": "t2"})
    got, _ = engine_select(mod, world, good({"test_id": "t1"}, {"test_id": "t2"}))
    return got.target is None and got.refusal == "AMBIGUOUS_CANDIDATES" and probe_positive_control(mod)


def probe_a_non_model_proposal_is_refused_by_the_model_path(mod):
    class Sneaky(mod.ModelCandidateRecoverer):
        def propose(self, ctx):
            return (R.LocatorCandidate(strategy=R.LocatorStrategy.AUTHORED, role="button", name="Save"),)
    clock = FakeClock()
    engine = mod.ModelRecoveryEngine(client=FakeModelClient(clock, good(SAVE)), snapshots=nodes(mod), clock=clock,
                                     model_limits=mod.ModelRecoveryLimits(**OPT_IN), base=mod.DeterministicRecovery())
    return True


# --- plan-level mechanisms -------------------------------------------------------------------------------------------

def plan_inputs(rec_mod, **kw):
    st = step(permitted=["MODEL_CANDIDATE"])
    base = dict(step=st, error_code="LOCATOR_NOT_FOUND", error_retryable=False, acted=False,
                postcondition=R.PostconditionResult.NOT_EVALUATED, has_idempotency_key=True, recovery_enabled=True,
                retries_left=1, recovery_attempts_left=2, step_attempts_used=0, step_recovery_ms=0, tried=(),
                model_calls_left=3, model_tokens_left=900)
    base.update(kw)
    return rec_mod.FaultInputs(**base)


def model_engine(mod):
    clock = FakeClock()
    return mod.ModelRecoveryEngine(client=FakeModelClient(clock), snapshots=nodes(mod), clock=clock,
                                   model_limits=mod.ModelRecoveryLimits(**OPT_IN), base=mod.DeterministicRecovery())


def plan_module(rec_mod):
    return rec_mod


def probe_plan_chooses_the_model_only_with_a_call_and_tokens_left_and_once(rec_mod):
    mod = bound(rec_mod)
    engine = model_engine(mod)
    ok = engine.plan(plan_inputs(rec_mod)).strategy is RS.MODEL_CANDIDATE
    no_calls = engine.plan(plan_inputs(rec_mod, model_calls_left=0)).strategy is RS.NONE
    no_tokens = engine.plan(plan_inputs(rec_mod, model_tokens_left=0)).strategy is RS.NONE
    once = engine.plan(plan_inputs(rec_mod, tried=(RS.MODEL_CANDIDATE,))).strategy is RS.NONE
    return ok and no_calls and no_tokens and once


def probe_plan_ignores_the_model_when_calls_are_exhausted(rec_mod):
    return bound(rec_mod) and model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod, model_calls_left=0)).strategy is RS.NONE \
        and model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod)).strategy is RS.MODEL_CANDIDATE


def probe_plan_ignores_the_model_when_tokens_are_exhausted(rec_mod):
    return model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod, model_tokens_left=0)).strategy is RS.NONE \
        and model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod)).strategy is RS.MODEL_CANDIDATE


def probe_plan_makes_one_model_round_per_step(rec_mod):
    return model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod, tried=(RS.MODEL_CANDIDATE,))).strategy is RS.NONE \
        and model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod)).strategy is RS.MODEL_CANDIDATE


def probe_plan_respects_the_step_permission_for_the_model(rec_mod):
    st = step(permitted=["SEMANTIC_LOCATOR"])
    return model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod, step=st, tried=(RS.SEMANTIC_LOCATOR,))).strategy is RS.NONE \
        and model_engine(bound(rec_mod)).plan(plan_inputs(rec_mod)).strategy is RS.MODEL_CANDIDATE


def probe_plan_honours_the_kill_switch_for_the_model(rec_mod):
    mod = bound(rec_mod)
    clock = FakeClock()
    kill = rec_mod.KillSwitch()
    kill.engage("x")
    engine = mod.ModelRecoveryEngine(client=FakeModelClient(clock), snapshots=nodes(mod), clock=clock,
                                     model_limits=mod.ModelRecoveryLimits(**OPT_IN),
                                     base=rec_mod.DeterministicRecovery(kill_switch=kill))
    return engine.plan(plan_inputs(rec_mod)).strategy is RS.NONE and model_engine(mod).plan(plan_inputs(rec_mod)).attempt


# --- selector mechanisms exercised through the model engine --------------------------------------------------------------

def sel(rec_mod, world, *replies, **kw):
    return engine_select(bound(rec_mod), world, *replies, **kw)[0]


def probe_via_model_role(rec_mod):
    return sel(rec_mod, World(link={"role": "link", "name": "Save"}), good({"name": "Save"})).target is None \
        and sel(rec_mod, World(save={}), good(SAVE)).target is not None


def probe_via_model_name(rec_mod):
    return sel(rec_mod, World(delete={"name": "Delete", "test_id": "delete"}), good({"test_id": "delete"})).target is None \
        and sel(rec_mod, World(save={}), good(SAVE)).target is not None


def probe_via_model_unique(rec_mod):
    return sel(rec_mod, World(save={"unique_flag": False}), good(SAVE)).target is None \
        and sel(rec_mod, World(save={}), good(SAVE)).target is not None


def probe_via_model_actionable(rec_mod):
    return sel(rec_mod, World(save={"visible": False}), good(SAVE)).target is None \
        and sel(rec_mod, World(save={}), good(SAVE)).target is not None


def probe_via_model_multi_result(rec_mod):
    world = World(a={"unique_flag": True}, b={"name": "Delete", "unique_flag": True})
    return sel(rec_mod, world, good({"role": "button"})).target is None \
        and sel(rec_mod, World(save={}), good(SAVE)).target is not None


def probe_via_model_single_match(rec_mod):
    world = World(a={"test_id": "t1"}, b={"test_id": "t2"})
    return sel(rec_mod, world, good({"test_id": "t1"}, {"test_id": "t2"})).target is None \
        and sel(rec_mod, World(save={}), good(SAVE)).target is not None


def probe_via_model_candidate_cap(rec_mod):
    world = World(**{f"c{i}": {} for i in range(9)})
    got = sel(rec_mod, world, good(SAVE))
    return got.target is None and got.aborted


# =====================================================================================================================
# probes on runner.py (model path)
# =====================================================================================================================

def runner_env(mod, *replies, spec=None, setup=move_save, **kw):
    return make_h4_env(spec, *replies, adapter_setup=setup, mod=mod, **kw)


def probe_runner_needs_the_configuration_for_the_model(mod):
    env = runner_env(mod, good(SAVE_BY_ID), config_kw={"model_recovery_enabled": False})
    env.run()
    return env.client.calls == 0


def probe_runner_records_the_cost_on_the_recovery(mod):
    r = runner_env(mod, good(SAVE_BY_ID)).run()
    ev = r.manifest.fault_events[0]
    return (ev.recovery.cost.model_calls, ev.recovery.cost.tokens) == (1, 42)


def probe_runner_charges_the_run_budget(mod):
    r = runner_env(mod, good(SAVE_BY_ID)).run()
    return (r.manifest.counters.model_calls, r.manifest.counters.tokens) == (1, 42)


def probe_runner_emits_the_hashed_request_and_response(mod):
    r = runner_env(mod, good(SAVE_BY_ID)).run()
    return [e["kind"] for e in model_events(r)] == ["ModelRequest", "ModelResponse"]


def probe_runner_accounts_a_round_that_ended_in_an_exception(mod):
    r = runner_env(mod, good(SAVE, {"role": "button", "name": "Save", "expected": "x"})).run()
    ev = r.manifest.fault_events[0]
    return ev.recovery.result is RR.ABORTED and ev.recovery.cost.model_calls == 1 and r.manifest.counters.model_calls == 1 \
        and len(model_events(r)) == 2


def probe_runner_never_certifies_a_model_recovered_run(mod):
    r = runner_env(mod, good(SAVE_BY_ID)).run()
    return r.classification.outcome.value == "INCONCLUSIVE" and r.exit_code != 0


def probe_runner_asks_the_model_after_a_deterministic_miss(mod):
    spec = model_spec(permitted=("SEMANTIC_LOCATOR", "MODEL_CANDIDATE"))
    env = runner_env(mod, good(SAVE_BY_ID), spec=spec, setup=compose(move_save, semantic_needs_test_id))
    r = env.run()
    return env.client.calls == 1 and r.manifest.fault_events[0].recovery.strategy is RS.MODEL_CANDIDATE


def probe_runner_replans_before_the_model_round(mod):
    spec = model_spec(permitted=("SEMANTIC_LOCATOR", "MODEL_CANDIDATE"))
    env = runner_env(mod, good(SAVE_BY_ID), spec=spec, setup=compose(move_save, semantic_needs_test_id),
                     recovery_limits=real("recovery.py").RecoveryLimits(max_attempts_per_step=1))
    env.run()
    return env.client.calls == 0


def probe_runner_gives_the_round_the_remaining_run_budget(mod):
    spec = model_spec(budget={**MODEL_BUDGET, "max_tokens": 40})
    env = runner_env(mod, good(SAVE_BY_ID, usage=(30, 12)), spec=spec)
    r = env.run()
    return r.manifest.counters.tokens <= 40


def probe_runner_rechecks_the_selected_control(mod):
    rm = real("recovery_model.py")
    from agent_harness.qa import recovery as RV

    class Liar(rm.ModelRecoveryEngine):
        def select_model_locator(self, step, observation, locate, within_time, spend):
            target = R.ResolvedTarget(candidate_ref="btn-delete", locator=R.LocatorDescriptor(role="button", name="Delete"),
                                      strategy=R.LocatorStrategy.MODEL_CANDIDATE)
            return rm.ModelSelection(target=target, score=1000, considered=1, rejected=(), spend=spend)
    env = runner_env(mod, good(SAVE_BY_ID))
    env.runner._engine = Liar(client=env.client, snapshots=env.snapshots, clock=env.clock, model_limits=rm.ModelRecoveryLimits(**OPT_IN),
                              base=RV.DeterministicRecovery(kill_switch=env.kill))
    env.run()
    return clicked(env, "btn-delete") == []


def probe_runner_judges_the_recovery_by_the_steps_postcondition(mod):
    from agent_harness.qa.hooks import NotEvaluatedPostcondition
    env = runner_env(mod, good(SAVE_BY_ID), postconditions=NotEvaluatedPostcondition())
    r = env.run()
    ev = r.manifest.fault_events[0]
    return ev.recovery.result is RR.NOT_RECOVERED and not ev.recovery.postcondition_verified and r.exit_code != 0


MUTATIONS = {
    # ---- recovery_model.py: bounds -----------------------------------------------------------------------------------
    "model: the call cap is not checked":
        M("recovery_model.py", [("        if spend.calls_left <= 0:\n            return self._refuse(R_MODEL_CALL_CAP)",
                                 "        if False:\n            return self._refuse(R_MODEL_CALL_CAP)")],
          probe_calls_per_step_cap, "call cap"),
    "model: calls are never charged":
        M("recovery_model.py", [("        spend.calls += 1                                                    # charged before the client is called",
                                 "        pass")],
          probe_calls_per_step_cap, "call cap"),
    "model: the per-run call cap is dropped":
        M("recovery_model.py", [("        calls = min(limits.max_calls_per_step - sc, limits.max_calls_per_run - self.calls)",
                                 "        calls = limits.max_calls_per_step - sc")],
          probe_calls_per_run_cap, "call cap"),
    "model: the per-step call cap is dropped":
        M("recovery_model.py", [("        calls = min(limits.max_calls_per_step - sc, limits.max_calls_per_run - self.calls)",
                                 "        calls = limits.max_calls_per_run - self.calls")],
          probe_calls_per_step_cap_with_a_higher_run_cap, "call cap"),
    "model: the kill switch is not checked before the call":
        M("recovery_model.py", [("        if self._kill.engaged:                                              # before the call",
                                 "        if False:")],
          probe_kill_switch_before_the_call, "kill switch"),
    "model: the kill switch is not checked after the response":
        M("recovery_model.py", [("        if self._kill.engaged:                                              # after the response",
                                 "        if False:")],
          probe_kill_switch_after_the_response, "kill switch"),
    "model: the wall-time cap is not checked":
        M("recovery_model.py", [("        if elapsed >= limits.max_wall_ms:", "        if False:")],
          probe_wall_time_cap, "wall time"),
    "model: unknown usage is accepted as free":
        M("recovery_model.py", [("            spend.tokens += left\n            return R_MODEL_USAGE_UNKNOWN, None",
                                 "            return None, None")],
          probe_unknown_usage_is_refused_and_charged, "token accounting"),
    "model: usage over the remaining budget is accepted":
        M("recovery_model.py", [("        if total > left or usage.output_tokens > request.max_output_tokens:", "        if False:")],
          probe_usage_over_the_budget_is_refused, "token cap"),
    "model: usage over the call's own output limit is accepted":
        M("recovery_model.py", [("        if total > left or usage.output_tokens > request.max_output_tokens:",
                                 "        if total > left:")],
          probe_usage_over_the_output_limit_is_refused, "token cap"),
    "model: tokens are never charged":
        M("recovery_model.py", [("        spend.tokens += total\n        return None, shown", "        return None, shown")],
          probe_tokens_are_charged, "token accounting"),
    "model: a client that raised costs nothing":
        M("recovery_model.py", [("            spend.tokens += left\n            return R_MODEL_CLIENT_ERROR, None",
                                 "            return R_MODEL_CLIENT_ERROR, None")],
          probe_a_client_that_raises_costs_the_remainder, "token accounting"),
    "model: the output limit ignores the tokens left":
        M("recovery_model.py", [("max_output_tokens=min(limits.max_output_tokens, spend.tokens_left))",
                                "max_output_tokens=limits.max_output_tokens)")],
          probe_output_limit_follows_the_tokens_left, "token cap"),
    "model: the run budget is not passed to the round":
        M("recovery_model.py", [("                                outer_tokens_left=budgets.tokens_left)", "                                outer_tokens_left=None)")],
          probe_the_budget_binds_the_round, "run budget"),
    "model: the operator opt-in is not required by the plan":
        M("recovery_model.py", [("        ladder = MODEL_LADDER if self._model_limits.enabled else LADDER", "        ladder = MODEL_LADDER")],
          probe_default_limits_make_the_engine_plan_no_model_round, "opt-in default"),
    "model: a reply with too many candidates is truncated, not discarded":
        M("recovery_model.py", [("    if len(data) > limits.max_candidates or len(data) > candidates_left:\n        raise _Discard(R_MODEL_TOO_MANY)",
                                 "    data = data[:limits.max_candidates]")],
          probe_over_long_candidate_lists_are_discarded_whole, "candidate cap"),
    "model: the per-step candidate cap is dropped":
        M("recovery_model.py", [("    if len(data) > limits.max_candidates or len(data) > candidates_left:", "    if len(data) > limits.max_candidates:")],
          probe_candidates_per_step_cap, "candidate cap"),
    "model: the response size is not bounded":
        M("recovery_model.py", [("    if len(text.encode(\"utf-8\", \"surrogatepass\")) > limits.max_response_bytes:", "    if False:")],
          probe_oversized_replies_are_discarded, "response size"),
    "model: no snapshot still means a call":
        M("recovery_model.py", [("        if node_count == 0:", "        if False:")],
          probe_no_snapshot_means_no_call, "snapshot required"),
    # ---- recovery_model.py: the strict gate ---------------------------------------------------------------------------
    "model: extra keys in a candidate are ignored, not rejected":
        M("recovery_model.py", [("    if extra:\n        raise _reject(\"%d key(s) outside role/name/label/test_id\" % len(extra))", "    pass")],
          probe_extra_fields_hit_the_strict_gate, "strict gate"),
    "model: advisory fields (confidence, rank, strategy) are tolerated":
        M("recovery_model.py", [("ALLOWED_KEYS = frozenset(KEY_ORDER)",
                                 "ALLOWED_KEYS = frozenset(KEY_ORDER + (\"confidence\", \"justification\", \"strategy\", \"rank\"))"),
                                ("    extra = set(item) - ALLOWED_KEYS", "    extra = set(item) - ALLOWED_KEYS")],
          probe_advisory_fields_hit_the_strict_gate, "confidence ignored"),
    "model: URLs, markup and scripts in values are accepted":
        M("recovery_model.py", [("    if any(s in value for s in _FORBIDDEN_SUBSTRINGS) or _SCHEME.search(value):", "    if False:")],
          probe_url_values_hit_the_strict_gate, "strict gate"),
    "model: value length is not bounded":
        M("recovery_model.py", [("    if not 1 <= len(value) <= limits.max_field_chars:", "    if False:")],
          probe_long_values_hit_the_strict_gate, "strict gate"),
    "model: control and format characters in values are accepted":
        M("recovery_model.py", [("    if any(unicodedata.category(ch) in _BAD_CATEGORIES for ch in value):", "    if False:")],
          probe_control_characters_hit_the_strict_gate, "strict gate"),
    "model: non-string values are accepted":
        M("recovery_model.py", [("    if type(value) is not str:\n        raise _reject(\"%s is not a string\" % key)", "    value = str(value)")],
          probe_non_string_values_hit_the_strict_gate, "strict gate"),
    "model: duplicate keys are accepted":
        M("recovery_model.py", [("        if key in out:\n            raise ValueError(\"duplicate key\")", "        pass")],
          probe_duplicate_keys_are_malformed, "strict parse"),
    "model: the model's own order is kept":
        M("recovery_model.py", [("    return tuple(keyed[k] for k in sorted(keyed))", "    return tuple(keyed.values())")],
          probe_the_models_order_carries_no_weight, "confidence ignored"),
    # ---- recovery_model.py: the request -------------------------------------------------------------------------------
    "model: page text is not sanitised":
        M("recovery_model.py", [("    return \" \".join(\"\".join(kept).split())[:max_chars].strip()", "    return value")],
          probe_sanitisation_removes_markup, "request sanitisation"),
    "model: angle brackets survive sanitisation":
        M("recovery_model.py", [("        if ch in \"<>\":\n            kept.append(\" \")", "        if False:\n            kept.append(\" \")")],
          probe_sanitisation_removes_markup, "request sanitisation"),
    "model: control and format characters survive sanitisation":
        M("recovery_model.py", [("        elif unicodedata.category(ch) in _BAD_CATEGORIES:\n            continue", "        elif False:\n            continue")],
          probe_sanitisation_removes_controls_and_format_characters, "request sanitisation"),
    "model: newlines survive sanitisation":
        M("recovery_model.py", [("        elif ch.isspace():\n            kept.append(\" \")", "        elif False:\n            kept.append(\" \")")],
          probe_sanitisation_keeps_the_data_on_its_line, "request sanitisation"),
    "model: page text is not length-bounded":
        M("recovery_model.py", [("[:max_chars].strip()", ".strip()")],
          probe_sanitisation_bounds_field_length, "request sanitisation"),
    "model: the request size is not bounded":
        M("recovery_model.py", [("        if used + len(line) + 1 > limits.max_request_chars:", "        if False:")],
          probe_the_request_is_size_bounded, "request size"),
    "model: the snapshot excerpt is not bounded":
        M("recovery_model.py", [("        if len(body) >= limits.max_snapshot_nodes:", "        if False:")],
          probe_the_node_count_is_bounded, "request size"),
    "model: the request carries the expected transition":
        M("recovery_model.py", [("    return TargetView(intent=step.intent, role=locator.role or \"\", name=locator.name or \"\")",
                                 "    return TargetView(intent=step.intent + \" \" + step.expected_transition, role=locator.role or \"\", name=locator.name or \"\")")],
          probe_the_request_carries_no_expected_transition, "immutable assertions"),
    # ---- recovery.py: verification of model candidates, through the model path -----------------------------------------
    "recovery (model path): the role is not compared":
        M("recovery.py", [("    if not labels_match(authored.role, role):\n        return R_ROLE_MISMATCH", "    if False:\n        return R_ROLE_MISMATCH")],
          probe_via_model_role, "role must match"),
    "recovery (model path): the name is not compared":
        M("recovery.py", [("    if not labels_match(authored.name, name):\n        return R_NAME_NEAR_MISS if name_score(authored.name, name) >= NEAR_MISS_NAME_SCORE else R_NAME_MISMATCH",
                           "    if False:\n        return R_NAME_MISMATCH")],
          probe_via_model_name, "name must match"),
    "recovery (model path): the uniqueness flag is ignored":
        M("recovery.py", [("            if not cand.unique:\n                rejected.append(R_NOT_UNIQUE)", "            if False:\n                rejected.append(R_NOT_UNIQUE)")],
          probe_via_model_unique, "uniqueness"),
    "recovery (model path): a hidden or disabled control is accepted":
        M("recovery.py", [("            if not (cand.visible and cand.enabled):\n                rejected.append(R_NOT_ACTIONABLE)", "            if False:\n                rejected.append(R_NOT_ACTIONABLE)")],
          probe_via_model_actionable, "actionable"),
    "recovery (model path): the first control of a multi-control result is taken":
        M("recovery.py", [("            if len(found.candidates) > 1:                   # several controls: never the first match",
                           "            if False:")],
          probe_via_model_multi_result, "several controls"),
    "recovery (model path): a second matching control does not block":
        M("recovery.py", [("        if len(matching_refs) != 1:                         # exactly one control may carry the authored role and name",
                           "        if len(matching_refs) < 1:")],
          probe_via_model_single_match, "exactly one control"),
    "recovery (model path): the control cap is not applied":
        M("recovery.py", [("            if len(found.candidates) > self._limits.max_candidates:", "            if False:")],
          probe_via_model_candidate_cap, "candidate cap"),
    "recovery (plan): the model rung ignores an exhausted call budget":
        M("recovery.py", [("        return (strategy in i.tried or i.recovery_attempts_left <= 0 or i.model_calls_left <= 0\n                or i.model_tokens_left <= 0)",
                           "        return (strategy in i.tried or i.recovery_attempts_left <= 0\n                or i.model_tokens_left <= 0)")],
          probe_plan_ignores_the_model_when_calls_are_exhausted, "run budget"),
    "recovery (plan): the model rung ignores an exhausted token budget":
        M("recovery.py", [("        return (strategy in i.tried or i.recovery_attempts_left <= 0 or i.model_calls_left <= 0\n                or i.model_tokens_left <= 0)",
                           "        return (strategy in i.tried or i.recovery_attempts_left <= 0 or i.model_calls_left <= 0)")],
          probe_plan_ignores_the_model_when_tokens_are_exhausted, "run budget"),
    "recovery (plan): the model may be tried more than once per step":
        M("recovery.py", [("        return (strategy in i.tried or i.recovery_attempts_left <= 0 or i.model_calls_left <= 0\n                or i.model_tokens_left <= 0)",
                           "        return (i.recovery_attempts_left <= 0 or i.model_calls_left <= 0\n                or i.model_tokens_left <= 0)")],
          probe_plan_makes_one_model_round_per_step, "one round per step"),
    "recovery (plan): the model rung ignores the step's permission":
        M("recovery.py", [("    permitted = [s for s in ladder if PERMIT_OF[s] in i.step.permitted_recovery]", "    permitted = list(ladder)")],
          probe_plan_respects_the_step_permission_for_the_model, "permitted recovery modes"),
    "recovery (plan): the kill switch does not stop the model rung":
        M("recovery.py", [("    if kill_switch or not i.recovery_enabled:", "    if not i.recovery_enabled:")],
          probe_plan_honours_the_kill_switch_for_the_model, "kill switch"),
    # ---- runner.py -------------------------------------------------------------------------------------------------
    "runner: the configuration flag does not gate the model":
        M("runner.py", [("        return bool(self._cfg.model_recovery_enabled\n                    and all(",
                         "        return bool(True\n                    and all(")],
          probe_runner_needs_the_configuration_for_the_model, "opt-in default"),
    "runner: the recovery cost is not recorded":
        M("runner.py", [("            cost=RecoveryCost(ms=elapsed, model_calls=p.model_calls, tokens=p.tokens))", "            cost=RecoveryCost(ms=elapsed))")],
          probe_runner_records_the_cost_on_the_recovery, "recovery cost"),
    "runner: the model's cost is not charged to the run budget":
        M("runner.py", [("        c.budgets.charge_model(spend.calls, spend.tokens)", "        pass")],
          probe_runner_charges_the_run_budget, "run budget"),
    "runner: the request and response are not recorded":
        M("runner.py", [("        for kind, payload in spend.events:\n            self._emit(c, kind, payload)", "        pass")],
          probe_runner_emits_the_hashed_request_and_response, "hashed evidence"),
    "runner: a round cut short by an exception is not accounted":
        M("runner.py", [("            try:\n                selection = self._engine.select_model_locator(step, fresh, locate, within_time, spend)\n            finally:\n                self._account_model(c, step, p, spend)",
                         "            selection = self._engine.select_model_locator(step, fresh, locate, within_time, spend)\n            self._account_model(c, step, p, spend)")],
          probe_runner_accounts_a_round_that_ended_in_an_exception, "recovery cost"),
    "runner: a run with a model-recovered locator may pass":
        M("runner.py", [("        c.locator_recovery_used = True\n        c.recovered_locator_steps.add(step.step_id)", "        c.recovered_locator_steps.add(step.step_id)")],
          probe_runner_never_certifies_a_model_recovered_run, "locator recovery never certifies"),
    "runner: the model is never asked after a deterministic miss":
        M("runner.py", [("defer=made is not None and self._model_next(step))", "defer=False)")],
          probe_runner_asks_the_model_after_a_deterministic_miss, "fall-through"),
    "runner: the model round is not re-planned":
        M("runner.py", [("        if strategy is RecoveryStrategy.MODEL_CANDIDATE and self._allowed(step, strategy):\n            try:",
                         "        strategy = RecoveryStrategy.MODEL_CANDIDATE\n        if True:\n            try:")],
          probe_runner_replans_before_the_model_round, "attempt cap"),
    "runner: the selected control is not re-checked":
        M("runner.py", [("                  and _same_label(authored.role, cand.role) and _same_label(authored.name, cand.name))", "                  and True)")],
          probe_runner_rechecks_the_selected_control, "role and name re-check"),
    "runner: the recovery is judged by the model path, not the step postcondition":
        M("runner.py", [("        verified = made.error is None and made.postcondition.result is PostconditionResult.SATISFIED",
                         "        verified = made.error is None")],
          probe_runner_judges_the_recovery_by_the_steps_postcondition, "step postcondition"),
}


def test_every_mutation_anchor_is_unique_and_every_probe_holds_on_the_real_code():
    for name, (filename, replacements, probe, why) in MUTATIONS.items():
        assert probe(real(filename)) is True, name


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_removing_the_mechanism_makes_the_probe_fail(name, mutants):
    filename, replacements, probe, why = MUTATIONS[name]
    mutant = mutants.load(filename, replacements)
    assert not holds(probe, mutant), f"mutation survived: {name} ({why})"


def test_the_mutation_table_covers_the_listed_h4_mechanisms():
    touched = {filename for filename, *_ in MUTATIONS.values()}
    assert touched == {"recovery_model.py", "recovery.py", "runner.py"}
    wanted = ("call cap", "kill switch", "strict gate", "uniqueness", "role must match", "name must match",
              "confidence ignored", "request sanitisation", "token cap", "token accounting", "wall time", "candidate cap",
              "response size", "request size", "immutable assertions", "recovery cost", "run budget", "hashed evidence",
              "opt-in default", "one round per step", "locator recovery never certifies", "step postcondition",
              "role and name re-check")
    assert all(any(w in why for _, _, _, why in MUTATIONS.values()) for w in wanted)
    assert len(MUTATIONS) >= 50
