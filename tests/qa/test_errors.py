# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed errors: stable code strings that match the golden vocabulary."""
import pytest

from agent_harness.qa import errors, records as R

from ._helpers import load_inventory

INV = load_inventory()

#: refusal codes of the refusal vocabulary, as read from the golden refusal data (qa.json "refusals"[*].expect.code)
FROZEN_REFUSAL_CODES = {
    "ASSERTION_HASH_MISMATCH", "HASH_MISMATCH", "EXPECTED_MISMATCH", "EXECUTOR_MISMATCH", "RECOVERY_PAYLOAD_REJECTED",
    "COVERAGE_VIOLATION", "UNSUPPORTED_COVERAGE_CONTRACT", "UNKNOWN_FIELD", "UNKNOWN_ENUM_VALUE",
    "CHECK_PROGRAM_REF_REQUIRED", "SPEC_INVALID", "SCHEMA_VERSION_UNSUPPORTED", "FAULT_BINDING_MISMATCH",
    "ASSERTION_PROGRAM_MISMATCH", "ASSERTION_PROGRAM_UNREGISTERED", "ASSERTION_PIN_NOT_FOUND",
    "ASSERTION_PIN_MALFORMED", "BATCH_HALTED",
}
#: refusals whose golden expectation is run outcome INCONCLUSIVE with reason ASSERTION_INTEGRITY
INTEGRITY_CODES = {"ASSERTION_HASH_MISMATCH", "EXPECTED_MISMATCH", "EXECUTOR_MISMATCH", "RECOVERY_PAYLOAD_REJECTED",
                   "COVERAGE_VIOLATION", "ASSERTION_PROGRAM_MISMATCH", "ASSERTION_PROGRAM_UNREGISTERED",
                   "ASSERTION_PIN_NOT_FOUND", "ASSERTION_PIN_MALFORMED", "BATCH_HALTED"}


def test_adapter_error_codes_are_exactly_the_non_ok_result_codes():
    assert set(errors.ADAPTER_ERRORS) == set(INV["enums"]["ResultCode"]) - {"OK"}
    assert len(errors.ADAPTER_ERRORS) == 15
    assert {m.value for m in R.ResultCode} == set(INV["enums"]["ResultCode"])


def test_every_frozen_refusal_code_has_an_error_class_with_that_code():
    for code in FROZEN_REFUSAL_CODES:
        assert errors.error_for_code(code).code == code
    neutral_only = set(errors.ERRORS_BY_CODE) - set(errors.ADAPTER_ERRORS) - FROZEN_REFUSAL_CODES
    assert neutral_only == {"CANON_REFUSED"}                       # the one code the golden data does not name


def test_integrity_errors_carry_the_frozen_run_outcome():
    for code, cls in errors.ERRORS_BY_CODE.items():
        if code in INTEGRITY_CODES:
            assert (cls.run_outcome, cls.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY"), code
        else:
            assert (cls.run_outcome, cls.run_reason) == (None, None), code


def test_codes_are_unique_stable_upper_case_strings():
    codes = [cls.code for cls in errors.ERRORS_BY_CODE.values()]
    assert len(codes) == len(set(codes)) == len(errors.ERRORS_BY_CODE)
    assert all(c == c.upper() and c.replace("_", "").isalpha() for c in codes)
    for code, cls in errors.ERRORS_BY_CODE.items():
        assert cls.code == code and issubclass(cls, errors.QAError)
    with pytest.raises(KeyError):
        errors.error_for_code("NOT_A_CODE")


def test_adapter_error_taxonomy_follows_adr_section_8():
    assert all(issubclass(c, errors.QAAdapterError) for c in errors.ADAPTER_ERRORS.values())
    cats = {m.value for m in errors.ErrorCategory}
    hints = {m.value for m in errors.ClassHint}
    assert cats == set(INV["enums"]["ErrorCategory"]) and hints == set(INV["enums"]["ClassHint"])
    for cls in errors.ADAPTER_ERRORS.values():
        assert cls.category in errors.ErrorCategory and cls.class_hint in errors.ClassHint
        assert isinstance(cls.retryable, bool) and isinstance(cls.recoverable, bool)
    E = errors
    assert E.LocatorNotFound.recoverable and E.LocatorNotFound.class_hint is E.ClassHint.BENIGN_DRIFT_CANDIDATE
    assert E.LocatorAmbiguous.class_hint is E.ClassHint.INCONCLUSIVE
    assert E.ActionBlockedByPolicy.category is E.ErrorCategory.POLICY and not E.ActionBlockedByPolicy.recoverable
    assert E.NavigationDenied.class_hint is E.ClassHint.ENVIRONMENT_OR_POLICY and not E.NavigationDenied.recoverable
    assert E.RedactionFailed.category is E.ErrorCategory.INTEGRITY and not E.RedactionFailed.recoverable
    assert E.ResetFailed.class_hint is E.ClassHint.ENVIRONMENT and not E.ResetFailed.recoverable
    assert E.UnsupportedOperation.class_hint is E.ClassHint.TEST_DEFECT
    assert E.ActionTimeout.retryable and E.SessionLost.recoverable and E.BrowserCrash.recoverable


def test_errors_render_with_their_code_and_keep_details():
    err = errors.SpecInvalid("a.b: bad", path="a.b")
    assert str(err) == "SPEC_INVALID: a.b: bad" and err.code == "SPEC_INVALID" and err.details == {"path": "a.b"}
    assert str(errors.ResetFailed()) == "RESET_FAILED"
    with pytest.raises(errors.QAError):
        raise errors.LocatorNotFound("no match")
