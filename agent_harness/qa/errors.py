# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed errors with stable ``code`` strings for the neutral QA module.

Two families, both subclasses of :class:`QAError`:

* integrity / validation errors raised while reading, hashing or checking records
  (``SPEC_INVALID``, ``ASSERTION_HASH_MISMATCH``, ``RECOVERY_PAYLOAD_REJECTED`` ...). Those whose frozen contract
  says the run ends ``INCONCLUSIVE(ASSERTION_INTEGRITY)`` carry ``run_outcome`` and ``run_reason``;
* adapter errors (:class:`QAAdapterError`), one per non-``OK`` ``ResultCode`` of qa-adapter/1, with
  ``category``, ``retryable``, ``recoverable`` and ``class_hint`` as in ADR section 8.

A bridge maps ``code`` to the contracts error class; the code string is the only thing that crosses the
boundary. Standard library only.
"""
from __future__ import annotations

import enum
from typing import Dict, Optional, Type

OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
REASON_ASSERTION_INTEGRITY = "ASSERTION_INTEGRITY"


class ErrorCategory(enum.StrEnum):
    LOCATOR = "LOCATOR"
    ACTION = "ACTION"
    SESSION = "SESSION"
    NETWORK = "NETWORK"
    POLICY = "POLICY"
    ENVIRONMENT = "ENVIRONMENT"
    INTEGRITY = "INTEGRITY"
    SPEC = "SPEC"


class ClassHint(enum.StrEnum):
    BENIGN_DRIFT_CANDIDATE = "BENIGN_DRIFT_CANDIDATE"
    BENIGN_DRIFT_OR_ENVIRONMENT = "BENIGN_DRIFT_OR_ENVIRONMENT"
    ENVIRONMENT_OR_BENIGN = "ENVIRONMENT_OR_BENIGN"
    ENVIRONMENT_OR_POLICY = "ENVIRONMENT_OR_POLICY"
    ENVIRONMENT = "ENVIRONMENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    POLICY = "POLICY"
    INTEGRITY = "INTEGRITY"
    TEST_DEFECT = "TEST_DEFECT"


class QAError(Exception):
    """Base class. ``code`` is a stable upper-case string; never parse ``str(error)``."""

    code: str = "QA_ERROR"
    #: set when the frozen contract says this refusal ends the run INCONCLUSIVE(ASSERTION_INTEGRITY)
    run_outcome: Optional[str] = None
    run_reason: Optional[str] = None

    def __init__(self, message: str = "", **details: object) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = dict(details)

    def __str__(self) -> str:  # "<CODE>: <message>"
        return f"{self.code}: {self.message}" if self.message != self.code else self.code


class _Integrity(QAError):
    run_outcome = OUTCOME_INCONCLUSIVE
    run_reason = REASON_ASSERTION_INTEGRITY


# --- validation and integrity errors (codes from the frozen refusal vocabulary, plus CANON_REFUSED) -------------


class CanonicalizationRefused(QAError):
    """canon-v1 refuses the value (a float, a non-string key, a duplicate key after NFC, an unsupported type).

    ``CANON_REFUSED`` is a neutral-only code: the frozen vectors say a value is refused, not under what code.
    """

    code = "CANON_REFUSED"


class SpecInvalid(QAError):
    code = "SPEC_INVALID"


class SchemaVersionUnsupported(QAError):
    code = "SCHEMA_VERSION_UNSUPPORTED"


class UnknownField(QAError):
    code = "UNKNOWN_FIELD"


class UnknownEnumValue(QAError):
    code = "UNKNOWN_ENUM_VALUE"


class CheckProgramRefRequired(QAError):
    code = "CHECK_PROGRAM_REF_REQUIRED"


class HashMismatch(QAError):
    code = "HASH_MISMATCH"


class AssertionHashMismatch(_Integrity):
    code = "ASSERTION_HASH_MISMATCH"


class ExpectedMismatch(_Integrity):
    code = "EXPECTED_MISMATCH"


class ExecutorMismatch(_Integrity):
    code = "EXECUTOR_MISMATCH"


class RecoveryPayloadRejected(_Integrity):
    code = "RECOVERY_PAYLOAD_REJECTED"


class CoverageViolation(_Integrity):
    code = "COVERAGE_VIOLATION"


class FaultBindingMismatch(QAError):
    code = "FAULT_BINDING_MISMATCH"


class UnsupportedCoverageContract(QAError):
    code = "UNSUPPORTED_COVERAGE_CONTRACT"


# --- adapter errors (ADR section 8; one per non-OK ResultCode) -------------------------------------------------


class QAAdapterError(QAError):
    """Adapter-side typed error. ``category``/``retryable``/``recoverable``/``class_hint`` are class attributes."""

    category: ErrorCategory = ErrorCategory.ENVIRONMENT
    retryable: bool = False
    recoverable: bool = False
    class_hint: ClassHint = ClassHint.ENVIRONMENT


class LocatorNotFound(QAAdapterError):
    code = "LOCATOR_NOT_FOUND"
    category = ErrorCategory.LOCATOR
    recoverable = True          # semantic locator, model candidate
    class_hint = ClassHint.BENIGN_DRIFT_CANDIDATE


class LocatorAmbiguous(QAAdapterError):
    code = "LOCATOR_AMBIGUOUS"
    category = ErrorCategory.LOCATOR
    recoverable = True          # semantic locator with a uniqueness check; never auto-picks the first match
    class_hint = ClassHint.INCONCLUSIVE


class TargetNotActionable(QAAdapterError):
    code = "TARGET_NOT_ACTIONABLE"
    category = ErrorCategory.ACTION
    retryable = True            # wait, re-observe; bounded by the deadline
    recoverable = True
    class_hint = ClassHint.BENIGN_DRIFT_CANDIDATE


class ActionTimeout(QAAdapterError):
    code = "ACTION_TIMEOUT"
    category = ErrorCategory.ACTION
    retryable = True            # only if the step is idempotent
    class_hint = ClassHint.BENIGN_DRIFT_OR_ENVIRONMENT


class UnexpectedDialog(QAAdapterError):
    code = "UNEXPECTED_DIALOG"
    category = ErrorCategory.ACTION
    recoverable = True          # dismiss only if the spec permits
    class_hint = ClassHint.BENIGN_DRIFT_CANDIDATE


class SessionLost(QAAdapterError):
    code = "SESSION_LOST"
    category = ErrorCategory.SESSION
    recoverable = True          # re-open the session and resume from a checkpoint
    class_hint = ClassHint.ENVIRONMENT_OR_BENIGN


class NavigationDenied(QAAdapterError):
    code = "NAVIGATION_DENIED"
    category = ErrorCategory.NETWORK
    class_hint = ClassHint.ENVIRONMENT_OR_POLICY


class ActionBlockedByPolicy(QAAdapterError):
    code = "ACTION_BLOCKED_BY_POLICY"
    category = ErrorCategory.POLICY
    class_hint = ClassHint.POLICY


class EnvironmentUnhealthy(QAAdapterError):
    code = "ENVIRONMENT_UNHEALTHY"
    category = ErrorCategory.ENVIRONMENT
    recoverable = True          # reset
    class_hint = ClassHint.ENVIRONMENT


class AdapterCrash(QAAdapterError):
    code = "ADAPTER_CRASH"
    category = ErrorCategory.ENVIRONMENT
    recoverable = True          # restart the session
    class_hint = ClassHint.ENVIRONMENT


class BrowserCrash(QAAdapterError):
    code = "BROWSER_CRASH"
    category = ErrorCategory.ENVIRONMENT
    recoverable = True
    class_hint = ClassHint.ENVIRONMENT


class ResetFailed(QAAdapterError):
    code = "RESET_FAILED"
    category = ErrorCategory.ENVIRONMENT
    class_hint = ClassHint.ENVIRONMENT


class ResourceLimitExceeded(QAAdapterError):
    code = "RESOURCE_LIMIT_EXCEEDED"
    category = ErrorCategory.ENVIRONMENT
    class_hint = ClassHint.ENVIRONMENT


class RedactionFailed(QAAdapterError):
    code = "REDACTION_FAILED"
    category = ErrorCategory.INTEGRITY
    class_hint = ClassHint.INTEGRITY


class UnsupportedOperation(QAAdapterError):
    code = "UNSUPPORTED_OPERATION"
    category = ErrorCategory.SPEC
    class_hint = ClassHint.TEST_DEFECT


ADAPTER_ERRORS: Dict[str, Type[QAAdapterError]] = {
    cls.code: cls
    for cls in (
        LocatorNotFound, LocatorAmbiguous, TargetNotActionable, ActionTimeout, UnexpectedDialog, SessionLost,
        NavigationDenied, ActionBlockedByPolicy, EnvironmentUnhealthy, AdapterCrash, BrowserCrash, ResetFailed,
        ResourceLimitExceeded, RedactionFailed, UnsupportedOperation,
    )
}

ERRORS_BY_CODE: Dict[str, Type[QAError]] = {
    **{
        cls.code: cls
        for cls in (
            CanonicalizationRefused, SpecInvalid, SchemaVersionUnsupported, UnknownField, UnknownEnumValue,
            CheckProgramRefRequired, HashMismatch, AssertionHashMismatch, ExpectedMismatch, ExecutorMismatch,
            RecoveryPayloadRejected, CoverageViolation, FaultBindingMismatch, UnsupportedCoverageContract,
        )
    },
    **ADAPTER_ERRORS,
}


def error_for_code(code: str) -> Type[QAError]:
    """The error class for a stable code string; unknown codes raise ``KeyError`` (never guessed)."""
    return ERRORS_BY_CODE[code]
