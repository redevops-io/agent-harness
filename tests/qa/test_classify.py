# SPDX-License-Identifier: AGPL-3.0-or-later
"""The classifier: precedence of outcomes (exhaustive over the signal lattice), coverage invariants, verdict mapping."""
import copy
import dataclasses
import itertools

import pytest

from agent_harness.qa import classify as C, records as R, verifier as V
from agent_harness.qa.classify import Signal, Signals
from agent_harness.qa.errors import QAError, SpecInvalid
from agent_harness.qa.records import Outcome, Verdict

from ._fx_verify import (MANIFEST, Mutants, facts, m_empty, m_fail, m_fail_and_indeterminate, m_indeterminate, m_no_fault,
                  m_nondeterministic, m_skipped, m_with_fault, manifest, manifest_json, ref_specs, vec)

INC, PF, TD, EF = Outcome.INCONCLUSIVE, Outcome.PRODUCT_FAIL, Outcome.TEST_DEFECT, Outcome.ENVIRONMENT_FAIL
INTEGRITY = "ASSERTION_INTEGRITY"


def cl(**kw):
    return C.classify(facts(**kw))


# --- manifests for the scenarios (in _fx_verify.py, shared with the determinism test) -----------------------------------

def reason_of(c):
    return c.outcome, c.reason


# --- the control and rule 6 --------------------------------------------------------------------------------------

def test_a_fully_attested_all_pass_run_is_a_pass():
    c = cl()
    assert (c.outcome, c.reason, c.verdict, c.rule) == (Outcome.PASS, None, Verdict.PASS, 6)
    assert not c.halt_batch and not c.review_required
    assert c.pair == (Outcome.PASS, None)


@pytest.mark.parametrize("name,reason", [
    ("preconditions", "PRECONDITIONS_UNVERIFIED"), ("fixture_hash", "FIXTURE_UNVERIFIED"),
    ("environment_before", "ENVIRONMENT_UNVERIFIED"), ("environment_after", "ENVIRONMENT_UNVERIFIED"),
    ("recovery_checks", "RECOVERY_UNVERIFIED"), ("cleanup", "CLEANUP_UNVERIFIED"), ("policy", "POLICY_UNVERIFIED"),
])
@pytest.mark.parametrize("state", [Signal.UNKNOWN, Signal.FAILED])
def test_a_pass_needs_every_attestation_to_be_ok(name, reason, state):
    c = cl(**{name: state})
    if name == "fixture_hash" and state is Signal.FAILED:
        assert reason_of(c) == (TD, "FIXTURE_MISMATCH")                  # a fixture mismatch is a test defect (rule 3)
    elif name.startswith("environment") and state is Signal.FAILED:
        assert reason_of(c) == (EF, "ENVIRONMENT_UNHEALTHY")             # a failed health probe is rule 2
    else:
        assert reason_of(c) == (INC, reason) and c.review_required and not c.halt_batch


def test_the_recovery_attestation_is_only_needed_when_a_recovery_was_used():
    assert reason_of(C.classify(facts(m=m_no_fault(), recovery_checks=Signal.UNKNOWN))) == (Outcome.PASS, None)
    assert reason_of(C.classify(facts(recovery_checks=Signal.UNKNOWN))) == (INC, "RECOVERY_UNVERIFIED")


def test_a_run_with_no_assertions_is_never_a_pass():
    assert reason_of(C.classify(facts(m=m_empty(), assertions=()))) == (INC, "NO_ASSERTIONS")


def test_every_attestation_defaults_to_unknown_so_nothing_passes_by_omission():
    f = C.RunFacts(manifest=manifest(), assertions=ref_specs())
    assert reason_of(C.classify(f)) == (INC, "PRECONDITIONS_UNVERIFIED")
    assert all(getattr(f, n) is Signal.UNKNOWN for n in (
        "environment_before", "environment_after", "preconditions", "fixture_hash", "cleanup", "policy",
        "recovery_checks", "failure_reproduced", "spec_validity", "baseline_conformance", "rerun_consistency"))


# --- rule 1: integrity violation, halt ---------------------------------------------------------------------------

def broken_accounting(**changes):
    j = manifest_json(outcome="INCONCLUSIVE", outcome_reason="ASSERTION_INTEGRITY")
    j.update(changes)
    return R.RunManifest.from_json(j)


def swapped_executor_result():
    j = manifest_json()
    j["assertion_results"][0]["executor"] = "SCRIPT"
    return R.RunManifest.from_json(j)


def other_expected_result():
    j = manifest_json()
    j["assertion_results"][0]["expected_hash"] = j["assertion_results"][1]["expected_hash"]
    return R.RunManifest.from_json(j)


def other_set_result():
    j = manifest_json()
    j["assertion_results"][0]["assertion_set_hash"] = "rcv1:" + "1" * 64
    return R.RunManifest.from_json(j)


def other_check_id():
    j = manifest_json()
    j["assertion_results"][0]["check"]["check_id"] = "a-url"
    return R.RunManifest.from_json(j)


EDITED_SPEC = tuple(R.AssertionSpec.from_json(d) for d in vec("qa-assertion-set:expected_changed")["assertions"])
INTEGRITY_CASES = {
    "declared_not_accounted": lambda: facts(m=broken_accounting(assertions_executed=["a-saved"],
                                                                assertion_results=MANIFEST["assertion_results"][:1])),
    "executed_and_skipped": lambda: facts(m=broken_accounting(
        assertions_skipped=[{"assertion_id": "a-url", "reason_code": "NOT_REACHED"}])),
    "executed_not_declared": lambda: facts(m=broken_accounting(assertions_declared=["a-saved"])),
    "manifest_declares_other_ids": lambda: facts(m=broken_accounting(assertions_declared=["a-saved", "a-url", "a-x"])),
    "manifest_pins_other_set": lambda: facts(assertions=EDITED_SPEC),
    "manifest_pin_altered": lambda: facts(m=manifest(assertion_set_hash="rcv1:" + "1" * 64)),
    "spec_has_extra_assertion": lambda: facts(assertions=ref_specs() + (R.AssertionSpec.from_json(
        {**vec("qa-assertion-set:reference")["assertions"][0], "assertion_id": "a-extra"}),)),
    "result_executor_swapped": lambda: facts(m=swapped_executor_result()),
    "result_expected_hash_differs": lambda: facts(m=other_expected_result()),
    "result_names_other_set": lambda: facts(m=other_set_result()),
    "result_check_for_other_assertion": lambda: facts(m=other_check_id()),
    "claimed_pass_holds_failed_assertion": lambda: facts(m=manifest(verdicts={"a-saved": "FAIL"}, outcome="PASS")),
    "claimed_pass_holds_indeterminate": lambda: facts(m=manifest(verdicts={"a-url": "INDETERMINATE"}, outcome="PASS")),
    "manifest_records_integrity": lambda: facts(m=manifest(outcome="INCONCLUSIVE", outcome_reason="ASSERTION_INTEGRITY")),
    "halted_integrity_skip": lambda: facts(m=m_skipped("HALTED_INTEGRITY")),
    "refusal_hash_mismatch": lambda: facts(error_codes=("ASSERTION_HASH_MISMATCH",)),
    "refusal_expected_mismatch": lambda: facts(error_codes=("EXPECTED_MISMATCH",)),
    "refusal_executor_mismatch": lambda: facts(error_codes=("EXECUTOR_MISMATCH",)),
    "refusal_recovery_payload": lambda: facts(error_codes=("RECOVERY_PAYLOAD_REJECTED",)),
    "refusal_coverage": lambda: facts(error_codes=("COVERAGE_VIOLATION",)),
    "refusal_batch_halted": lambda: facts(error_codes=("BATCH_HALTED",)),
    "refusal_stored_hash": lambda: facts(error_codes=("HASH_MISMATCH",)),
    "refusal_fault_binding": lambda: facts(error_codes=("FAULT_BINDING_MISMATCH",)),
    "unknown_error_code": lambda: facts(error_codes=("SOMETHING_NEW",)),
}


@pytest.mark.parametrize("name", sorted(INTEGRITY_CASES))
def test_an_integrity_violation_is_inconclusive_assertion_integrity_and_halts_the_batch(name):
    c = C.classify(INTEGRITY_CASES[name]())
    assert (c.outcome, c.reason, c.rule) == (INC, INTEGRITY, 1)
    assert c.halt_batch and c.review_required and c.verdict is Verdict.INDETERMINATE
    assert c.details, "the findings are reported"


def test_integrity_outranks_every_other_condition():
    f = facts(m=swapped_executor_result(), environment_after=Signal.FAILED, fixture_hash=Signal.FAILED,
              spec_validity=Signal.FAILED, error_codes=("BROWSER_CRASH", "SPEC_INVALID", "REDACTION_FAILED"))
    assert reason_of(C.classify(f)) == (INC, INTEGRITY)


def test_an_integrity_result_reason_code_is_an_integrity_violation():
    j = manifest_json(verdicts={"a-saved": "INDETERMINATE"}, outcome="INCONCLUSIVE")
    j["assertion_results"][0]["reason_code"] = "ASSERTION_HASH_MISMATCH"
    assert C.classify(facts(m=R.RunManifest.from_json(j))).halt_batch


# --- rule 2: environment ----------------------------------------------------------------------------------------

ENV_CASES = {
    "health_before": (dict(environment_before=Signal.FAILED), "ENVIRONMENT_UNHEALTHY"),
    "health_after": (dict(environment_after=Signal.FAILED), "ENVIRONMENT_UNHEALTHY"),
    "browser_crash": (dict(error_codes=("BROWSER_CRASH",)), "BROWSER_CRASH"),
    "adapter_crash": (dict(error_codes=("ADAPTER_CRASH",)), "ADAPTER_CRASH"),
    "reset_failed": (dict(error_codes=("RESET_FAILED",)), "RESET_FAILED"),
    "resource_limit": (dict(error_codes=("RESOURCE_LIMIT_EXCEEDED",)), "RESOURCE_LIMIT_EXCEEDED"),
    "navigation_denied": (dict(error_codes=("NAVIGATION_DENIED",)), "NAVIGATION_DENIED"),
    "session_lost": (dict(error_codes=("SESSION_LOST",)), "SESSION_LOST"),
    "environment_unhealthy": (dict(error_codes=("ENVIRONMENT_UNHEALTHY",)), "ENVIRONMENT_UNHEALTHY"),
    "skipped_for_environment": (dict(m=m_skipped("ENVIRONMENT")), "ENVIRONMENT"),
    "unrecovered_env_fault_event": (dict(m=m_with_fault("ENV_UNHEALTHY", "NOT_RECOVERED", "NONE", "INCONCLUSIVE")),
                                    "ENV_UNHEALTHY"),
    "network_blocked_fault_event": (dict(m=m_with_fault("NETWORK_BLOCKED", "NOT_ATTEMPTED", "NONE", "INCONCLUSIVE")),
                                    "NETWORK_BLOCKED"),
}


@pytest.mark.parametrize("name", sorted(ENV_CASES))
def test_an_unhealthy_environment_is_environment_fail(name):
    kw, reason = ENV_CASES[name]
    c = cl(**kw)
    assert (c.outcome, c.reason, c.rule, c.verdict) == (EF, reason, 2, Verdict.INDETERMINATE)
    assert not c.halt_batch and not c.review_required


def test_an_adapter_error_recorded_on_a_result_is_read_as_environment():
    j = manifest_json(verdicts={"a-saved": "INDETERMINATE"}, outcome="INCONCLUSIVE")
    j["assertion_results"][0]["reason_code"] = "BROWSER_CRASH"
    assert reason_of(C.classify(facts(m=R.RunManifest.from_json(j)))) == (EF, "BROWSER_CRASH")


def test_environment_outranks_a_real_failure_and_every_lower_rule():
    f = facts(m=m_fail(), environment_after=Signal.FAILED, fixture_hash=Signal.FAILED, spec_validity=Signal.FAILED)
    assert reason_of(C.classify(f)) == (EF, "ENVIRONMENT_UNHEALTHY")


# --- rule 3: test defect ----------------------------------------------------------------------------------------

DEFECT_CASES = {
    "spec_invalid": (dict(spec_validity=Signal.FAILED), "SPEC_INVALID"),
    "baseline": (dict(baseline_conformance=Signal.FAILED), "BASELINE_NONCONFORMANT"),
    "fixture_mismatch": (dict(fixture_hash=Signal.FAILED), "FIXTURE_MISMATCH"),
    "rerun_inconsistent": (dict(rerun_consistency=Signal.FAILED), "RERUN_INCONSISTENT"),
    "code_spec_invalid": (dict(error_codes=("SPEC_INVALID",)), "SPEC_INVALID"),
    "code_check_program_ref": (dict(error_codes=("CHECK_PROGRAM_REF_REQUIRED",)), "CHECK_PROGRAM_REF_REQUIRED"),
    "code_unknown_enum": (dict(error_codes=("UNKNOWN_ENUM_VALUE",)), "UNKNOWN_ENUM_VALUE"),
    "code_unsupported_operation": (dict(error_codes=("UNSUPPORTED_OPERATION",)), "UNSUPPORTED_OPERATION"),
    "code_schema_version": (dict(error_codes=("SCHEMA_VERSION_UNSUPPORTED",)), "SCHEMA_VERSION_UNSUPPORTED"),
}


@pytest.mark.parametrize("name", sorted(DEFECT_CASES))
def test_a_spec_baseline_or_fixture_defect_is_a_test_defect(name):
    kw, reason = DEFECT_CASES[name]
    c = cl(**kw)
    assert (c.outcome, c.reason, c.rule, c.verdict) == (TD, reason, 3, Verdict.INDETERMINATE)
    assert not c.halt_batch and not c.review_required


def test_a_test_defect_outranks_a_real_failure():
    assert reason_of(C.classify(facts(m=m_fail(), baseline_conformance=Signal.FAILED))) == (TD, "BASELINE_NONCONFORMANT")


# --- rule 4: product fail ----------------------------------------------------------------------------------------

def test_a_deterministic_fail_with_everything_verified_is_a_product_fail():
    c = C.classify(facts(m=m_fail()))
    assert (c.outcome, c.reason, c.rule, c.verdict) == (PF, "ASSERTION_FAILED", 4, Verdict.FAIL)
    assert not c.halt_batch and not c.review_required


def test_a_real_defect_outranks_an_unrelated_inconclusive_assertion_and_a_skipped_one():
    assert C.classify(facts(m=m_fail_and_indeterminate())).outcome is PF
    j = manifest_json(verdicts={"a-saved": "FAIL"}, outcome="PRODUCT_FAIL")
    j["assertions_executed"], j["assertions_skipped"] = ["a-saved"], [{"assertion_id": "a-url", "reason_code": "NOT_REACHED"}]
    j["assertion_results"] = j["assertion_results"][:1]
    assert C.classify(facts(m=R.RunManifest.from_json(j))).outcome is PF


@pytest.mark.parametrize("name", ["preconditions", "environment_before", "environment_after"])
def test_a_failure_whose_preconditions_or_environment_are_not_verified_is_not_a_product_fail(name):
    c = C.classify(facts(m=m_fail(), **{name: Signal.UNKNOWN}))
    assert reason_of(c) == (INC, "PRODUCT_FAIL_UNVERIFIED") and c.review_required


def test_a_failure_with_a_fixture_mismatch_is_a_test_defect_and_with_unverified_fixtures_inconclusive():
    assert C.classify(facts(m=m_fail(), fixture_hash=Signal.FAILED)).outcome is TD
    assert reason_of(C.classify(facts(m=m_fail(), fixture_hash=Signal.UNKNOWN))) == (INC, "PRODUCT_FAIL_UNVERIFIED")


def test_a_failure_seen_through_a_recovered_locator_must_reproduce_through_the_authored_one():
    unproven = C.classify(facts(m=m_fail(), failure_reproduced=Signal.UNKNOWN))        # the frozen manifest used SEMANTIC_LOCATOR
    assert reason_of(unproven) == (INC, "FAILURE_NOT_REPRODUCED")
    not_repro = C.classify(facts(m=m_fail(), failure_reproduced=Signal.FAILED))
    assert reason_of(not_repro) == (INC, "FAILURE_NOT_REPRODUCED")
    assert C.classify(facts(m=m_fail(), failure_reproduced=Signal.OK)).outcome is PF
    j = m_fail().to_json()
    j["fault_events"] = []                                                              # no recovered locator: nothing to reproduce
    assert C.classify(facts(m=R.RunManifest.from_json(j), failure_reproduced=Signal.UNKNOWN)).outcome is PF


def test_an_unrecovered_fault_or_a_redaction_failure_blocks_a_product_fail():
    m = m_fail()
    assert reason_of(C.classify(facts(m=m, error_codes=("ACTION_TIMEOUT",)))) == (INC, "UNRECOVERED_FAULT")
    assert reason_of(C.classify(facts(m=m, error_codes=("REDACTION_FAILED",)))) == (INC, "REDACTION_FAILED")


def test_a_non_deterministic_fail_is_not_a_product_fail():
    j = m_fail().to_json()
    j["assertion_results"][0]["check"]["determinism"] = "NON_DETERMINISTIC"
    assert reason_of(C.classify(facts(m=R.RunManifest.from_json(j)))) == (INC, "NON_DETERMINISTIC_EVIDENCE")


# --- rule 5: inconclusive ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("m,reason", [
    (m_indeterminate("SUBJECT_UNRESOLVED"), "ASSERTION_INDETERMINATE"),
    (m_indeterminate("SUBJECT_AMBIGUOUS"), "ASSERTION_INDETERMINATE"),
    (m_skipped("NOT_REACHED"), "ASSERTIONS_SKIPPED"), (m_skipped("PRECONDITION_UNMET"), "ASSERTIONS_SKIPPED"),
    (m_skipped("POLICY_DENIED"), "ASSERTIONS_SKIPPED"), (m_skipped("BUDGET_EXHAUSTED"), "ASSERTIONS_SKIPPED"),
    (m_skipped("SUBJECT_UNRESOLVED"), "ASSERTIONS_SKIPPED"),
    (m_nondeterministic(), "NON_DETERMINISTIC_EVIDENCE"),
    (m_with_fault("LOCATOR_NOT_FOUND", "NOT_RECOVERED", "SEMANTIC_LOCATOR", "INCONCLUSIVE"), "UNRECOVERED_FAULT"),
    (m_with_fault("TRANSITION_NOT_OBSERVED", "NOT_ATTEMPTED", "NONE", "INCONCLUSIVE"), "UNRECOVERED_FAULT"),
    (m_with_fault("POLICY_DENIED", "ABORTED", "NONE", "INCONCLUSIVE"), "UNRECOVERED_FAULT"),
])
def test_omissions_indeterminate_and_non_deterministic_evidence_are_inconclusive_with_a_review_item(m, reason):
    c = C.classify(facts(m=m))
    assert (c.outcome, c.reason, c.rule, c.verdict) == (INC, reason, 5, Verdict.INDETERMINATE)
    assert c.review_required and not c.halt_batch
    item = C.review_item_for("run-1", c)
    assert item.run_id == "run-1" and item.reason == reason


def test_a_detected_assertion_failure_event_is_not_an_infrastructure_fault():
    c = C.classify(facts(m=m_with_fault("ASSERTION_FAILED", "NOT_ATTEMPTED", "NONE", "PASS")))
    assert c.outcome is Outcome.PASS
    assert C.classify(facts(m=m_with_fault("NONE", "NOT_ATTEMPTED", "NONE", "PASS"))).outcome is Outcome.PASS


@pytest.mark.parametrize("code", ["LOCATOR_NOT_FOUND", "LOCATOR_AMBIGUOUS", "TARGET_NOT_ACTIONABLE", "ACTION_TIMEOUT",
                                  "UNEXPECTED_DIALOG", "ACTION_BLOCKED_BY_POLICY"])
def test_unrecovered_non_environment_adapter_errors_are_inconclusive(code):
    assert reason_of(cl(error_codes=(code,))) == (INC, "UNRECOVERED_FAULT")


def test_a_redaction_failure_is_inconclusive_with_its_own_reason_and_does_not_halt():
    c = cl(error_codes=("REDACTION_FAILED",))
    assert (c.outcome, c.reason, c.halt_batch) == (INC, "REDACTION_FAILED", False)


def test_an_inconclusive_run_always_has_a_review_item_and_other_outcomes_never_do():
    for f in (facts(), facts(preconditions=Signal.UNKNOWN), facts(m=m_fail()), facts(environment_after=Signal.FAILED),
              facts(spec_validity=Signal.FAILED), INTEGRITY_CASES["unknown_error_code"]()):
        c = C.classify(f)
        item = C.review_item_for("run-9", c)
        assert (item is not None) == (c.outcome is INC) == c.review_required
        if item:
            assert item.review_item_id == "review-run-9" and item.reason == c.reason
    long_id = "r" * 128
    assert len(C.review_item_for(long_id, C.classify(facts(preconditions=Signal.UNKNOWN))).review_item_id) <= 128


# --- coverage invariants ---------------------------------------------------------------------------------

@pytest.mark.parametrize("declared,executed,skipped,expect", [
    (["a", "b"], ["a"], [], (INC, INTEGRITY)),                          # declared, not accounted
    (["a", "b"], ["a", "b"], ["b"], (INC, INTEGRITY)),                  # executed and skipped overlap
    (["a"], ["a", "z"], [], (INC, INTEGRITY)),                          # executed, never declared
    (["a", "b"], ["a"], ["b"], None),                                   # control
    (["a", "b"], ["a", "b"], [], None),
    ([], [], [], None),
    (["a", "a"], ["a"], [], (INC, INTEGRITY)),                          # duplicates are accounting errors
    (["a", "b"], ["a", "a"], ["b"], (INC, INTEGRITY)),
    (["a", "b"], ["a"], ["b", "b"], (INC, INTEGRITY)),
    (["a"], ["a"], ["z"], (INC, INTEGRITY)),
])
def test_classify_coverage(declared, executed, skipped, expect):
    assert C.classify_coverage(declared, executed, skipped) == expect


def test_classify_coverage_accepts_skipped_records_and_checks_declared_against_the_spec():
    skipped = [R.SkippedAssertion(assertion_id="b", reason_code="NOT_REACHED")]
    assert C.classify_coverage(["a", "b"], ["a"], skipped) is None
    assert C.classify_coverage(["a", "b"], ["a"], [{"assertion_id": "b", "reason_code": "NOT_REACHED"}]) is None
    assert C.classify_coverage(["a", "b"], ["a"], skipped, spec_ids=["a", "b"]) is None
    assert C.classify_coverage(["a", "b"], ["a"], skipped, spec_ids=["a", "b", "c"]) == (INC, INTEGRITY)
    assert C.classify_coverage(["a", "b"], ["a"], skipped, spec_ids=["a"]) == (INC, INTEGRITY)


def test_a_pass_requires_no_skipped_assertion_in_the_classifier_and_in_the_record():
    j = manifest_json()
    j["assertions_executed"], j["assertions_skipped"] = ["a-saved"], [{"assertion_id": "a-url", "reason_code": "NOT_REACHED"}]
    j["assertion_results"] = j["assertion_results"][:1]
    with pytest.raises(QAError) as err:
        R.RunManifest.from_json(j)                                      # the record itself refuses a PASS with a skip
    assert err.value.code == "COVERAGE_VIOLATION"
    j["outcome"] = "INCONCLUSIVE"
    assert C.classify(facts(m=R.RunManifest.from_json(j))).outcome is INC


# --- verdict, exit codes -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("outcome,verdict", [("PASS", "PASS"), ("PRODUCT_FAIL", "FAIL"), ("TEST_DEFECT", "INDETERMINATE"),
                                             ("ENVIRONMENT_FAIL", "INDETERMINATE"), ("INCONCLUSIVE", "INDETERMINATE"),
                                             ("GREAT", "INDETERMINATE"), ("", "INDETERMINATE"), (None, "INDETERMINATE"),
                                             ("pass", "INDETERMINATE"), (7, "INDETERMINATE")])
def test_outcome_to_verdict(outcome, verdict):
    assert C.outcome_to_verdict(outcome) == verdict


def test_every_outcome_maps_to_a_verdict_and_only_pass_is_a_pass():
    assert [o for o in Outcome if C.outcome_to_verdict(o) is Verdict.PASS] == [Outcome.PASS]
    assert [o for o in Outcome if C.outcome_to_verdict(o) is Verdict.FAIL] == [Outcome.PRODUCT_FAIL]


def test_default_exit_codes_fail_closed():
    assert C.exit_code(Outcome.PASS) == 0
    assert all(C.exit_code(o) != 0 for o in Outcome if o is not Outcome.PASS)
    assert C.exit_code("GREAT") != 0 and C.exit_code(None) != 0
    assert len({C.exit_code(o) for o in Outcome}) == len(Outcome)


def test_exit_codes_are_configurable_but_a_non_pass_never_exits_zero():
    assert C.exit_code(Outcome.INCONCLUSIVE, {"INCONCLUSIVE": 9}) == 9
    assert C.exit_code(Outcome.PASS, {"TEST_DEFECT": 42}) == 0
    for bad in ({"INCONCLUSIVE": 0}, {"TEST_DEFECT": 0}, {"ENVIRONMENT_FAIL": 0}, {"PRODUCT_FAIL": 0}, {"PASS": 1},
                {"BOGUS": 3}, {"INCONCLUSIVE": 256}, {"INCONCLUSIVE": -1}, {"INCONCLUSIVE": True}, {"INCONCLUSIVE": "1"}):
        with pytest.raises(SpecInvalid):
            C.exit_code(Outcome.PASS, bad)


# --- determinism, idempotence ------------------------------------------------------------------------------------

def test_classification_is_idempotent_and_independent_of_set_ordering():
    f = facts(m=m_fail())
    a, b = C.classify(f), C.classify(f)
    assert a == b and a.to_json() == b.to_json()
    j = m_fail().to_json()
    j["assertions_declared"] = list(reversed(j["assertions_declared"]))
    j["assertion_results"] = list(reversed(j["assertion_results"]))
    j["assertions_executed"] = list(reversed(j["assertions_executed"]))
    reordered = facts(m=R.RunManifest.from_json(j), assertions=tuple(reversed(ref_specs())))
    assert C.classify(reordered) == a


def test_facts_validate_their_inputs():
    with pytest.raises(SpecInvalid):
        C.RunFacts(manifest=MANIFEST, assertions=ref_specs())
    with pytest.raises(SpecInvalid):
        C.RunFacts(manifest=manifest(), assertions=("a",))
    with pytest.raises(SpecInvalid):
        C.RunFacts(manifest=manifest(), assertions=ref_specs() + ref_specs())
    with pytest.raises(SpecInvalid):
        C.RunFacts(manifest=manifest(), assertions=ref_specs(), cleanup="MAYBE")
    with pytest.raises(SpecInvalid):
        C.RunFacts(manifest=manifest(), assertions=ref_specs(), error_codes=(1,))
    assert C.RunFacts(manifest=manifest(), assertions=ref_specs(), cleanup="OK").cleanup is Signal.OK
    with pytest.raises(SpecInvalid):
        C.classify(manifest())


# --- the signal lattice, exhaustively ------------------------------------------------------------------------------

BOOL_FLAGS = ("integrity", "deterministic_fail", "indeterminate", "skipped", "nondeterministic", "fault",
              "redaction_failed", "no_assertions", "environment_verified", "preconditions_verified", "fixture_verified",
              "reproduction_ok", "recovery_ok", "cleanup_ok", "policy_ok")
BAD = ("deterministic_fail", "indeterminate", "skipped", "nondeterministic", "fault", "redaction_failed", "no_assertions")
VERIFIED = ("environment_verified", "preconditions_verified", "fixture_verified", "recovery_ok", "cleanup_ok", "policy_ok")
ENV_REASON, DEFECT_REASON = "ENVIRONMENT_UNHEALTHY", "SPEC_INVALID"


def all_signals():
    for bits in itertools.product((False, True), repeat=len(BOOL_FLAGS)):
        for env in (None, ENV_REASON):
            for defect in (None, DEFECT_REASON):
                yield Signals(**dict(zip(BOOL_FLAGS, bits)), environment_reason=env, test_defect_reason=defect)


def oracle(s):
    """An independent statement of the precedence (ordered rows), written without looking at decide()'s code shape."""
    if s.integrity:
        return INC, INTEGRITY
    if s.environment_reason:
        return EF, s.environment_reason
    if s.test_defect_reason:
        return TD, s.test_defect_reason
    product_fail_requirements = (s.preconditions_verified and s.fixture_verified and s.environment_verified
                                 and s.reproduction_ok and not s.fault and not s.redaction_failed)
    if s.deterministic_fail and product_fail_requirements:
        return PF, "ASSERTION_FAILED"
    ordered = [
        (s.redaction_failed, "REDACTION_FAILED"),
        (s.fault, "UNRECOVERED_FAULT"),
        (s.deterministic_fail and not s.reproduction_ok, "FAILURE_NOT_REPRODUCED"),
        (s.deterministic_fail, "PRODUCT_FAIL_UNVERIFIED"),
        (s.skipped, "ASSERTIONS_SKIPPED"),
        (s.indeterminate, "ASSERTION_INDETERMINATE"),
        (s.nondeterministic, "NON_DETERMINISTIC_EVIDENCE"),
        (s.no_assertions, "NO_ASSERTIONS"),
        (not s.preconditions_verified, "PRECONDITIONS_UNVERIFIED"),
        (not s.fixture_verified, "FIXTURE_UNVERIFIED"),
        (not s.environment_verified, "ENVIRONMENT_UNVERIFIED"),
        (not s.recovery_ok, "RECOVERY_UNVERIFIED"),
        (not s.cleanup_ok, "CLEANUP_UNVERIFIED"),
        (not s.policy_ok, "POLICY_UNVERIFIED"),
    ]
    for hit, reason in ordered:
        if hit:
            return INC, reason
    return Outcome.PASS, None


def lattice_failures(decide):
    """Run every point of the lattice through ``decide`` and check it against the oracle and the independent laws."""
    bad = []
    for s in all_signals():
        c = decide(s)
        if (c.outcome, c.reason) != oracle(s):
            bad.append(("oracle", s, c.pair))
            continue
        passes = c.outcome is Outcome.PASS
        clean = (not s.integrity and s.environment_reason is None and s.test_defect_reason is None
                 and not any(getattr(s, f) for f in BAD) and all(getattr(s, f) for f in VERIFIED))
        if passes != clean:                                                        # PASS iff nothing blocks it
            bad.append(("pass-iff-clean", s, c.pair))
        if c.halt_batch != (c.reason == INTEGRITY and c.outcome is INC) or c.halt_batch != s.integrity:
            bad.append(("halt", s, c.pair))
        if c.review_required != (c.outcome is INC):
            bad.append(("review", s, c.pair))
        if c.verdict is not C.outcome_to_verdict(c.outcome):
            bad.append(("verdict", s, c.pair))
        if s.integrity and c.rule != 1 or (not s.integrity and s.environment_reason and c.rule != 2):
            bad.append(("rule", s, c.pair))
        if len(bad) > 5:
            break
    return bad


def test_the_precedence_is_exhaustive_over_the_whole_signal_lattice():
    n = sum(1 for _ in all_signals())
    assert n == 2 ** (len(BOOL_FLAGS) + 2) == 131072
    assert lattice_failures(C.decide) == []


def test_pass_is_monotone_and_product_fail_is_robust_to_unrelated_evidence():
    clean = Signals(environment_verified=True, preconditions_verified=True, fixture_verified=True, reproduction_ok=True,
                    recovery_ok=True, cleanup_ok=True, policy_ok=True)
    assert C.decide(clean).outcome is Outcome.PASS
    for flag in BAD:
        assert C.decide(dataclasses.replace(clean, **{flag: True})).outcome is not Outcome.PASS
    for flag in VERIFIED:
        assert C.decide(dataclasses.replace(clean, **{flag: False})).outcome is not Outcome.PASS
    assert C.decide(dataclasses.replace(clean, integrity=True)).outcome is INC
    failing = dataclasses.replace(clean, deterministic_fail=True)
    assert C.decide(failing).outcome is PF
    for flag in ("indeterminate", "skipped", "nondeterministic", "no_assertions"):
        assert C.decide(dataclasses.replace(failing, **{flag: True})).outcome is PF         # a real defect outranks them
    for flag in ("fault", "redaction_failed"):
        assert C.decide(dataclasses.replace(failing, **{flag: True})).outcome is INC


def test_decide_is_a_function_of_its_signals_only():
    s = Signals(deterministic_fail=True)
    assert C.decide(s) == C.decide(dataclasses.replace(s))
    assert [p for p in C.decide.__code__.co_varnames[:C.decide.__code__.co_argcount]] == ["s", "details"]


# --- mutation tests: each precedence rule and the halt, removed, must be caught --------------------------------------

@pytest.fixture
def mutants():
    m = Mutants()
    yield m
    m.close()


def _lattice_ok(mod):
    return lattice_failures(mod.decide) == []


MUTATIONS = {
    "integrity_rule_removed": [("    if s.integrity:                                                         # (1)\n        return _result(Outcome.INCONCLUSIVE, ASSERTION_INTEGRITY, 1, details)\n", "")],
    "environment_rule_removed": [("    if s.environment_reason is not None:                                    # (2)\n        return _result(Outcome.ENVIRONMENT_FAIL, s.environment_reason, 2, details)\n", "")],
    "test_defect_rule_removed": [("    if s.test_defect_reason is not None:                                    # (3)\n        return _result(Outcome.TEST_DEFECT, s.test_defect_reason, 3, details)\n", "")],
    "defect_before_environment": [("    if s.environment_reason is not None:                                    # (2)", "    if False:                                                               # (2)"),
                                  ("    if s.test_defect_reason is not None:                                    # (3)", "    if s.test_defect_reason is not None:                                    # (3)\n        return _result(Outcome.TEST_DEFECT, s.test_defect_reason, 3, details)\n    if s.environment_reason is not None:")],
    "product_fail_ignores_preconditions": [("(s.deterministic_fail and s.preconditions_verified and s.fixture_verified", "(s.deterministic_fail and s.fixture_verified")],
    "product_fail_ignores_reproduction": [("            and s.reproduction_ok and not s.fault and not s.redaction_failed):          # (4)", "            and not s.fault and not s.redaction_failed):          # (4)")],
    "skipped_does_not_block_pass": [("    elif s.skipped:\n        reason = \"ASSERTIONS_SKIPPED\"\n", "")],
    "indeterminate_does_not_block_pass": [("    elif s.indeterminate:\n        reason = \"ASSERTION_INDETERMINATE\"\n", "")],
    "empty_run_passes": [("    elif s.no_assertions:\n        reason = \"NO_ASSERTIONS\"\n", "")],
    "cleanup_not_required": [("    elif not s.cleanup_ok:\n        reason = \"CLEANUP_UNVERIFIED\"\n", "")],
    "halt_flag_removed": [("halt_batch=(outcome is Outcome.INCONCLUSIVE and reason == ASSERTION_INTEGRITY),", "halt_batch=False,")],
    "no_review_for_inconclusive": [("review_required=(outcome is Outcome.INCONCLUSIVE), rule=rule", "review_required=False, rule=rule")],
    "fail_reads_as_pass": [("    Outcome.PRODUCT_FAIL: Verdict.FAIL,", "    Outcome.PRODUCT_FAIL: Verdict.PASS,")],
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutating_the_classifier_is_caught_by_the_lattice_suite(name, mutants):
    assert _lattice_ok(C)                                              # control: the real classifier is clean
    mod = mutants.load("classify.py", MUTATIONS[name])
    assert not _lattice_ok(mod), f"mutation {name} survived"


def test_mutating_the_coverage_invariant_is_caught(mutants):
    def probe(mod):
        return (mod.classify_coverage(["a", "b"], ["a"], []) == (INC, INTEGRITY)
                and mod.classify_coverage(["a", "b"], ["a", "b"], ["b"]) == (INC, INTEGRITY)
                and mod.classify_coverage(["a"], ["a", "z"], []) == (INC, INTEGRITY)
                and mod.classify_coverage(["a", "a"], ["a"], []) == (INC, INTEGRITY)
                and mod.classify_coverage(["a", "b"], ["a", "a"], ["b"]) == (INC, INTEGRITY)
                and mod.classify_coverage(["a", "b"], ["a"], ["b", "b"]) == (INC, INTEGRITY)
                and mod.classify_coverage(["a", "b"], ["a"], ["b"]) is None)

    assert probe(C)
    for old, new in (("or not set(e).isdisjoint(s) ", "or False "), ("or (set(e) | set(s)) != set(d))", "or False)"),
                     ("len(set(d)) != len(d) or len(set(e)) != len(e) or len(set(s)) != len(s)\n           ", "False\n           ")):
        assert not probe(mutants.load("classify.py", [(old, new)])), old


def test_mutating_the_integrity_finders_is_caught(mutants):
    cases = {n: INTEGRITY_CASES[n] for n in ("result_executor_swapped", "result_expected_hash_differs",
                                             "manifest_pin_altered", "claimed_pass_holds_failed_assertion",
                                             "executed_not_declared", "unknown_error_code", "declared_not_accounted")}

    def probe(mod, name):
        f = cases[name]()
        return mod.classify(mod.RunFacts(**{k: getattr(f, k) for k in ("manifest", "assertions", "error_codes")})).halt_batch

    for name in cases:
        assert probe(C, name)
    muts = {
        "result_executor_swapped": ("check_result_matches_assertion(r, a, assertion_set_hash=pin)\n        except QAError as exc:\n            found.append(exc.code)",
                                    "check_result_matches_assertion(r, a, assertion_set_hash=pin)\n        except QAError as exc:\n            pass"),
        "manifest_pin_altered": ("    if not canon.same_digest(m.assertion_set_hash, pin):\n        found.append(\"ASSERTION_SET_HASH_MISMATCH\")\n", ""),
        "claimed_pass_holds_failed_assertion": ("        found.append(\"PASS_CONTRADICTED_BY_OWN_RECORD\")", "        pass"),
        "unknown_error_code": ("if code in INTEGRITY_CODES or code not in ERRORS_BY_CODE:", "if code in INTEGRITY_CODES:"),
    }
    for name, (old, new) in muts.items():
        mod = mutants.load("classify.py", [(old, new)])
        assert not probe(mod, name), f"mutation of the {name} finder survived"
