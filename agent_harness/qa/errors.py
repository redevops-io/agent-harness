# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed errors with stable ``code`` strings for the neutral QA module.

Two families, both subclasses of :class:`QAError`:

* integrity / validation errors raised while reading, hashing or checking records
  (``SPEC_INVALID``, ``ASSERTION_HASH_MISMATCH``, ``RECOVERY_PAYLOAD_REJECTED`` ...). Those whose contract
  says the run ends ``INCONCLUSIVE(ASSERTION_INTEGRITY)`` carry ``run_outcome`` and ``run_reason``; the integrity
  refusal vocabulary is :data:`INTEGRITY_REFUSAL_CODES` and every code in it halts the batch;
* adapter errors (:class:`QAAdapterError`), one per non-``OK`` ``ResultCode`` of qa-adapter/1, with
  ``category``, ``retryable``, ``recoverable`` and ``class_hint`` of the adapter port.

A bridge maps ``code`` to the contracts error class; the code string is the only thing that crosses the
boundary. Standard library only.
"""
from __future__ import annotations

import enum
from typing import Dict, Optional, Type

OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
REASON_ASSERTION_INTEGRITY = "ASSERTION_INTEGRITY"

#: The integrity refusal vocabulary: every refusal that ends a run ``INCONCLUSIVE(ASSERTION_INTEGRITY)`` and halts the
#: batch. ``ASSERTION_INTEGRITY`` is the outcome reason itself. The last five codes tell apart refusals that would
#: otherwise share ``ASSERTION_HASH_MISMATCH`` (which keeps its meaning: a content hash recomputed from what was handed
#: over differs from the hash it was pinned or recorded under). A consumer that does not know a code treats it the same
#: way (see ``classify.classify_refusal``).
INTEGRITY_REFUSAL_CODES = (
    "ASSERTION_INTEGRITY", "ASSERTION_HASH_MISMATCH", "EXECUTOR_MISMATCH", "EXPECTED_MISMATCH",
    "RECOVERY_PAYLOAD_REJECTED", "COVERAGE_VIOLATION", "ASSERTION_PROGRAM_MISMATCH", "ASSERTION_PROGRAM_UNREGISTERED",
    "ASSERTION_PIN_NOT_FOUND", "ASSERTION_PIN_MALFORMED", "BATCH_HALTED",
)

#: Codes the contracts side raises for structural problems; the neutral module folds all of them into ``SPEC_INVALID``
#: (it never raises them) but knows them as codes that are not integrity refusals.
STRUCTURAL_CODES = ("DUPLICATE_ENTRY", "FLOAT_REFUSED", "INVALID_FIELD", "MISSING_FIELD", "PLAN_INVALID",
                    "RECORD_INCONSISTENT", "SCHEMA_MISMATCH")


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
    #: set when the contract says this refusal ends the run INCONCLUSIVE(ASSERTION_INTEGRITY)
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


# --- validation and integrity errors (codes from the refusal vocabulary, plus CANON_REFUSED) -------------


class CanonicalizationRefused(QAError):
    """canon-v1 refuses the value (a float, a non-string key, a duplicate key after NFC, an unsupported type).

    ``CANON_REFUSED`` is a neutral-only code: the golden vectors say a value is refused, not under what code.
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


class AssertionPinMalformed(_Integrity):
    """The pinned ``assertion_set_hash`` is not ``rcv1:`` followed by 64 lowercase hex digits (checked first)."""

    code = "ASSERTION_PIN_MALFORMED"


class AssertionPinNotFound(_Integrity):
    """A well-formed pin names an assertion set the spec store does not hold."""

    code = "ASSERTION_PIN_NOT_FOUND"


class AssertionProgramUnregistered(_Integrity):
    """The check program an assertion names has no registry entry."""

    code = "ASSERTION_PROGRAM_UNREGISTERED"


class AssertionProgramMismatch(_Integrity):
    """The registry holds the check program under a different hash than the assertion froze."""

    code = "ASSERTION_PROGRAM_MISMATCH"


class BatchHalted(_Integrity):
    """A call was refused because an earlier integrity refusal halted the batch. The refusal that halted the batch
    keeps its own code; this is what every later call gets."""

    code = "BATCH_HALTED"


class RecoveryPayloadRejected(_Integrity):
    code = "RECOVERY_PAYLOAD_REJECTED"


class CoverageViolation(_Integrity):
    code = "COVERAGE_VIOLATION"


class FaultBindingMismatch(QAError):
    code = "FAULT_BINDING_MISMATCH"


class UnsupportedCoverageContract(QAError):
    code = "UNSUPPORTED_COVERAGE_CONTRACT"


# --- adapter errors (one per non-OK ResultCode) -------------------------------------------------


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
    recoverable = True          # recovery still needs enablement, a permitted mode, caps and the idempotency guard
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
            AssertionPinMalformed, AssertionPinNotFound, AssertionProgramUnregistered, AssertionProgramMismatch,
            BatchHalted, RecoveryPayloadRejected, CoverageViolation, FaultBindingMismatch, UnsupportedCoverageContract,
        )
    },
    **ADAPTER_ERRORS,
}


def error_for_code(code: str) -> Type[QAError]:
    """The error class for a stable code string; unknown codes raise ``KeyError`` (never guessed)."""
    return ERRORS_BY_CODE[code]
