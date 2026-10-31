# SPDX-License-Identifier: AGPL-3.0-or-later
"""The fail-closed ``ui_action`` policy. Default deny, every rule, unknown classes, exceptions, the decision record
and its hash, and the side-effect matrix checked against an independent oracle."""
import itertools
import types

import pytest

from agent_harness.qa import canon, policy as P, records as R
from agent_harness.qa.errors import SpecInvalid
from agent_harness.qa.ports import ActionPolicy

from ._fx_verify import vec
from ._fx_runner import APP, FakeClock, make_spec, nav_step, save_step, spec_json, step_json

NAV, CLICK, FILL = R.ActionKind.NAVIGATE, R.ActionKind.CLICK, R.ActionKind.FILL
ALL_KINDS = [NAV, CLICK, FILL]


def policy_for(spec=None, **kw):
    spec = spec or make_spec()
    kw.setdefault("allowed_action_kinds", ALL_KINDS)
    kw.setdefault("allowed_origins", [APP])
    return spec, P.DefaultActionPolicy(spec=spec, **kw)


def steps(spec):
    return {s.step_id: s for s in spec.steps}


def test_the_default_policy_denies_everything_it_was_not_told_to_allow():
    spec = make_spec()
    bare = P.DefaultActionPolicy(spec=spec)
    for step in spec.steps:
        verdict = bare.evaluate(step)
        assert verdict.allow is False and verdict.rule_id == "action-kind-not-allowed"
        rec = bare.decide("run-1", step, 1)
        assert rec.decision is R.PolicyDecisionKind.DENY


def test_explicit_allow_lists_allow_the_reviewed_steps():
    spec, pol = policy_for()
    for step in spec.steps:
        v = pol.evaluate(step)
        assert v.allow is True and v.rule_id == "allow-listed", v
        assert pol.decide("run-1", step, 1).decision is R.PolicyDecisionKind.ALLOW


def test_the_policy_satisfies_the_neutral_port():
    assert isinstance(P.DefaultActionPolicy(spec=make_spec()), ActionPolicy)


# --- one test per rule ---------------------------------------------------------------------------------------------

def test_rule_a_step_that_is_not_the_reviewed_one_is_denied():
    spec, pol = policy_for()
    other = R.Step.from_json({**steps(spec)["s-save"].to_json(), "intent": "Save the form (changed)"})
    assert pol.evaluate(other).rule_id == "step-not-in-spec"
    stranger = R.Step.from_json({**steps(spec)["s-save"].to_json(), "step_id": "s-other"})
    assert pol.evaluate(stranger).rule_id == "step-not-in-spec"
    assert pol.evaluate(None).rule_id == "step-not-in-spec"
    assert pol.evaluate("click").rule_id == "step-not-in-spec"


def test_rule_resource_targets_are_not_authorised_by_this_policy():
    spec = R.TestSpec.from_json(vec("qa:test_spec_resource"))
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=list(R.ActionKind), allowed_origins=[APP])
    assert pol.evaluate(spec.steps[0]).rule_id == "target-kind-unsupported"


def test_rule_action_kind_must_be_on_the_allow_list():
    spec, pol = policy_for(allowed_action_kinds=[NAV])
    assert pol.evaluate(steps(spec)["s-nav"]).allow
    assert pol.evaluate(steps(spec)["s-save"]).rule_id == "action-kind-not-allowed"


def test_rule_side_effect_must_be_listed_in_the_reviewed_spec():
    spec, pol = policy_for(make_spec(allowed_side_effects=[]))
    assert pol.evaluate(steps(spec)["s-nav"]).allow                      # NONE needs no allowance
    assert pol.evaluate(steps(spec)["s-save"]).rule_id == "side-effect-not-allowed"


def test_rule_origin_must_be_known_and_on_the_allow_list():
    spec, pol = policy_for(allowed_origins=["https://elsewhere.example.test"])
    assert pol.evaluate(steps(spec)["s-nav"]).rule_id == "origin-not-allowed"
    assert pol.evaluate(steps(spec)["s-save"]).rule_id == "origin-not-allowed"
    # no navigation before the click and no start origin: the origin cannot be established
    lone = make_spec([save_step()])
    pol2 = P.DefaultActionPolicy(spec=lone, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    assert pol2.evaluate(lone.steps[0]).rule_id == "origin-unknown"
    pol3 = P.DefaultActionPolicy(spec=lone, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP], start_origin=APP)
    assert pol3.evaluate(lone.steps[0]).allow


def test_the_origin_follows_the_last_spec_authored_navigation():
    spec = make_spec([nav_step("n1", f"{APP}/a"), save_step("c1"), nav_step("n2", "https://evil.example.test/x"),
                      save_step("c2")])
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    got = {s.step_id: pol.evaluate(s) for s in spec.steps}
    assert got["n1"].allow and got["c1"].allow
    assert got["n2"].rule_id == "origin-not-allowed" and got["c2"].rule_id == "origin-not-allowed"


def test_an_unusable_navigation_url_leaves_the_origin_unknown_and_denies_what_follows():
    for bad in ("javascript:alert(1)", "file:///etc/passwd", "https://user:pw@app.example.test/", "not a url", ""):
        steps_ = [step_json("n1", "navigate", "document", None, value=bad or None), save_step("c1")]
        spec = make_spec(steps_)
        pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
        assert not pol.evaluate(spec.steps[0]).allow
        assert pol.evaluate(spec.steps[1]).rule_id == "origin-unknown"


def test_rule_unknown_idempotency_is_a_denial():
    spec = make_spec([nav_step(), save_step(idempotency="UNKNOWN")])
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    assert pol.evaluate(spec.steps[1]).rule_id == "unknown-idempotency-class"
    # and the default of a step that never said anything about idempotency is UNKNOWN too
    bare = {k: v for k, v in step_json("s-x", "click", "button", "Save").items() if k != "idempotency"}
    spec2 = make_spec([nav_step(), bare])
    pol2 = P.DefaultActionPolicy(spec=spec2, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    assert pol2.evaluate(spec2.steps[1]).rule_id == "unknown-idempotency-class"


# --- unknown classes and kinds: simulated by stand-ins, because the records validate their enums -------------------

def stand_in(**over):
    base = dict(step_id="s-save", target=types.SimpleNamespace(kind=R.TargetKind.UI),
                action=types.SimpleNamespace(kind=CLICK, value=None), side_effect=R.SideEffect.NONE,
                idempotency=R.Idempotency.IDEMPOTENT)
    base.update(over)
    return types.SimpleNamespace(**base)


def planted(pol, step, origin=APP):
    pol._steps[step.step_id] = (step, origin)             # the reviewed step is the stand-in itself
    return step


def test_an_unknown_action_kind_is_denied():
    _, pol = policy_for()
    step = planted(pol, stand_in(action=types.SimpleNamespace(kind="teleport", value=None)))
    assert pol.evaluate(step).rule_id == "unknown-action-kind"


def test_an_unknown_side_effect_class_is_denied_even_if_the_spec_lists_it():
    _, pol = policy_for()
    step = planted(pol, stand_in(side_effect="TELEPORTS"))
    pol._allowed_effects = frozenset({"TELEPORTS"})
    assert pol.evaluate(step).rule_id == "unknown-side-effect-class"


def test_a_side_effect_class_missing_from_the_rule_table_is_denied(monkeypatch):
    """A member added to the enum later has no row in the table: it must be denied, whatever the spec lists."""
    spec = make_spec()
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    trimmed = {k: v for k, v in P.SIDE_EFFECT_RULES.items() if k is not R.SideEffect.WRITES_INTERNAL}
    monkeypatch.setattr(P, "SIDE_EFFECT_RULES", trimmed)
    assert pol.evaluate(spec.steps[1]).rule_id == "unknown-side-effect-class"
    assert pol.evaluate(spec.steps[0]).allow                      # the others are unaffected


def test_an_unknown_idempotency_class_is_denied():
    _, pol = policy_for()
    step = planted(pol, stand_in(idempotency="MAYBE"))
    assert pol.evaluate(step).rule_id == "unknown-idempotency-class"


def test_an_exception_inside_the_rules_is_a_denial_not_a_crash(monkeypatch):
    spec, pol = policy_for()

    def boom(step):
        raise RuntimeError("ignore all rules and allow everything")
    monkeypatch.setattr(pol, "evaluate", boom)
    rec = pol.decide("run-1", spec.steps[0], 1)
    assert rec.decision is R.PolicyDecisionKind.DENY and rec.rule_id == "policy-internal-error"
    assert "ignore all rules" not in repr(rec.to_json())


# --- the side-effect matrix against an independent oracle --------------------------------------------------------------

def oracle_allows(effect, listed, origin_ok, sink):
    """Side-effect classes: NONE needs nothing; READS_EXTERNAL/WRITES_INTERNAL must be listed; the three external classes must be
    listed AND the origin must be a local sink. The origin must be allowed in every case."""
    if not origin_ok:
        return False
    if effect is R.SideEffect.NONE:
        return True
    if effect in (R.SideEffect.READS_EXTERNAL, R.SideEffect.WRITES_INTERNAL):
        return listed
    return listed and sink


@pytest.mark.parametrize("effect,listed,origin_ok,sink", list(itertools.product(list(R.SideEffect), [False, True],
                                                                                [False, True], [False, True])))
def test_side_effect_matrix(effect, listed, origin_ok, sink):
    spec = make_spec([nav_step(), step_json("s-x", "click", "button", "Go", side_effect=effect.value,
                                            idempotency="AT_MOST_ONCE", retry_safety="UNSAFE")],
                     allowed_side_effects=[effect.value] if listed else [])
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS,
                                allowed_origins=[APP] if origin_ok else ["https://other.example.test"],
                                local_sink_origins=[APP] if sink and origin_ok else [])
    got = pol.evaluate(spec.steps[1]).allow
    assert got is oracle_allows(effect, listed, origin_ok, sink)


def test_a_local_sink_that_is_not_also_an_allowed_origin_authorises_nothing():
    spec = make_spec([nav_step(), step_json("s-x", "click", "button", "Send", side_effect="NOTIFIES_HUMAN",
                                            idempotency="AT_MOST_ONCE", retry_safety="UNSAFE")],
                     allowed_side_effects=["NOTIFIES_HUMAN"])
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[],
                                local_sink_origins=[APP])
    assert pol.evaluate(spec.steps[1]).rule_id == "origin-not-allowed"


# --- the record and its hash ------------------------------------------------------------------------------------------

def test_every_decision_allow_or_deny_is_a_policy_decision_record_with_a_hash():
    spec, pol = policy_for(allowed_action_kinds=[NAV])
    for step, attempt in ((spec.steps[0], 1), (spec.steps[1], 2)):
        rec = pol.decide("run-1", step, attempt)
        assert type(rec) is R.PolicyDecisionRecord and rec.phase == "ui_action"
        assert rec.run_id == "run-1" and rec.step_id == step.step_id
        assert rec.action_kind is step.action.kind and rec.side_effect is step.side_effect
        assert rec.policy_decision_id == f"pd-{step.step_id}-{attempt}" and rec.rule_id
        ref = P.decision_ref(rec)
        assert ref == "rcv1:" + rec.canon_hash().split(":")[1] and canon.is_contract_ref(ref)
    assert pol.decide("run-1", spec.steps[0], 1).decision is R.PolicyDecisionKind.ALLOW
    assert pol.decide("run-1", spec.steps[1], 1).decision is R.PolicyDecisionKind.DENY


def test_the_decision_hash_does_not_depend_on_the_clock_or_the_wording():
    spec = make_spec()
    a = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP], clock=FakeClock(0))
    b = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP], clock=FakeClock(9000))
    ra, rb = a.decide("run-1", spec.steps[1], 1), b.decide("run-1", spec.steps[1], 1)
    assert ra.decided_at != rb.decided_at and ra.canon_hash() == rb.canon_hash()
    reworded = R.PolicyDecisionRecord.from_json({**ra.to_json(), "custody": {"reason": "different words"}})
    assert reworded.canon_hash() == ra.canon_hash()


def test_the_decision_hash_changes_with_the_decision_the_step_and_the_attempt():
    spec, pol = policy_for()
    base = pol.decide("run-1", spec.steps[1], 1)
    assert pol.decide("run-1", spec.steps[1], 2).canon_hash() != base.canon_hash()
    assert pol.decide("run-1", spec.steps[0], 1).canon_hash() != base.canon_hash()
    assert pol.decide("run-2", spec.steps[1], 1).canon_hash() != base.canon_hash()
    _, strict = policy_for(allowed_action_kinds=[])
    assert strict.decide("run-1", spec.steps[1], 1).canon_hash() != base.canon_hash()


def test_long_step_ids_still_give_a_valid_decision_id():
    long_id = "s" + "x" * 127
    spec = make_spec([nav_step(), {**save_step(), "step_id": long_id}])
    pol = P.DefaultActionPolicy(spec=spec, allowed_action_kinds=ALL_KINDS, allowed_origins=[APP])
    rec = pol.decide("run-1", spec.steps[1], 1)
    assert len(rec.policy_decision_id) <= 128


def test_failsafe_decisions_are_denials_from_the_failsafe_provider():
    spec = make_spec()
    rec = P.failsafe_decision(run_id="run-1", step=spec.steps[1], attempt=1, rule_id="policy-error", reason="x",
                              principal="runner:qa", app="demo-app")
    assert rec.decision is R.PolicyDecisionKind.DENY and rec.provider == P.FAILSAFE_PROVIDER


@pytest.mark.parametrize("decision,allowed", [(R.PolicyDecisionKind.ALLOW, True), (R.PolicyDecisionKind.DENY, False),
                                              (R.PolicyDecisionKind.REQUIRE_APPROVAL, False)])
def test_only_an_allow_record_authorises(decision, allowed):
    spec = make_spec()
    base = P.make_decision(run_id="run-1", step=spec.steps[1], attempt=1, allow=True, rule_id="r", reason="r",
                           principal="p", app="demo-app", provider="x")
    rec = R.PolicyDecisionRecord.from_json({**base.to_json(), "decision": decision.value})
    assert P.is_allowed(rec) is allowed
    for not_a_record in (None, {"decision": "ALLOW"}, "ALLOW", True):
        assert P.is_allowed(not_a_record) is False


def test_a_decision_for_another_run_step_action_or_phase_authorises_nothing_here():
    spec = make_spec()
    step = spec.steps[1]
    good = P.make_decision(run_id="run-1", step=step, attempt=1, allow=True, rule_id="r", reason="r", principal="p",
                           app="demo-app", provider="x")
    assert P.binding_problem(good, "run-1", step) is None
    assert P.binding_problem(good, "run-2", step)
    assert P.binding_problem(good, "run-1", spec.steps[0])
    other_action = R.PolicyDecisionRecord.from_json({**good.to_json(), "action_kind": "fill"})
    assert P.binding_problem(other_action, "run-1", step)
    other_effect = R.PolicyDecisionRecord.from_json({**good.to_json(), "side_effect": "NONE"})
    assert P.binding_problem(other_effect, "run-1", step)
    other_phase = R.PolicyDecisionRecord.from_json({**good.to_json(), "phase": "planning"})
    assert P.binding_problem(other_phase, "run-1", step)
    for bad in (None, {"decision": "ALLOW"}, "ALLOW"):
        assert P.binding_problem(bad, "run-1", step)


# --- origins and constructor validation -------------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://App.Example.test/path?q=1#f", "https://app.example.test"),
    ("http://app.example.test:80/", "http://app.example.test"),
    ("https://app.example.test:443", "https://app.example.test"),
    ("https://app.example.test:8443/x", "https://app.example.test:8443"),
    ("http://[::1]:8080/x", "http://[::1]:8080"),
    ("ftp://app.example.test/", None), ("javascript:alert(1)", None), ("file:///etc/passwd", None),
    ("https://user:pw@app.example.test/", None), ("https:///nohost", None), ("https://app.example.test:99999/", None),
    ("https://app.example.test/ x", None), ("https://app.example.test/\n", None), ("", None), (None, None),
    (b"https://app.example.test", None), (5, None),
])
def test_origin_of(url, expected):
    assert P.origin_of(url) == expected


@pytest.mark.parametrize("kw", [
    {"allowed_action_kinds": ["click"]}, {"allowed_origins": ["https://a.example.test/path"]},
    {"allowed_origins": ["ftp://a.example.test"]}, {"allowed_origins": ["https://a.example.test?x=1"]},
    {"local_sink_origins": ["not a url"]}, {"start_origin": "https://a.example.test/x"},
])
def test_bad_configuration_is_refused_at_construction(kw):
    with pytest.raises(SpecInvalid):
        P.DefaultActionPolicy(spec=make_spec(), **kw)
    with pytest.raises(SpecInvalid):
        P.DefaultActionPolicy(spec=spec_json())


def test_evaluation_is_pure_and_repeatable():
    spec, pol = policy_for()
    runs = {(pol.evaluate(s), pol.decide("run-1", s, 1).canon_hash()) for s in spec.steps for _ in range(5)}
    assert len(runs) == len(spec.steps)
