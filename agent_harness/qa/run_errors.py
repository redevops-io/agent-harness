# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed errors of the single-run execution layer.

Same conventions as ``errors.py``: a stable upper-case ``code`` string, never parsed from the message. These are
neutral-only codes (the qa/v1 refusal vocabulary names causes, not the runner's own control flow), so they are
deliberately not registered in ``errors.ERRORS_BY_CODE``; like ``verifier.BatchHalted`` they never reach the classifier
as a code. The runner maps each condition onto a code the classifier knows (see ``runner.py``), except
``RECEIPT_CHAIN_BROKEN``, which is an evidence-integrity violation and is handed to the classifier as it is: an
unknown code is an integrity violation there (``INCONCLUSIVE(ASSERTION_INTEGRITY)`` and the batch halts).

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from .errors import OUTCOME_INCONCLUSIVE, QAError, REASON_ASSERTION_INTEGRITY


class RunError(QAError):
    """Base class of the runner's own errors."""

    code = "RUN_ERROR"


class IllegalTransition(RunError):
    """A run-state transition that the allowed-transition table does not contain."""

    code = "ILLEGAL_TRANSITION"


class DeadlineExceeded(RunError):
    """A run, step or observation deadline is spent. ``scope`` is ``RUN``, ``STEP`` or ``OBSERVATION``."""

    code = "DEADLINE_EXCEEDED"

    def __init__(self, message: str = "", *, scope: str = "RUN", **details: object) -> None:
        super().__init__(message, scope=scope, **details)
        self.scope = scope


class BudgetExceeded(RunError):
    """A bounded counter (retries, recovery attempts, model calls, tokens) would pass its limit. ``kind`` names it."""

    code = "BUDGET_EXCEEDED"

    def __init__(self, message: str = "", *, kind: str = "", **details: object) -> None:
        super().__init__(message, kind=kind, **details)
        self.kind = kind


class RunCancelled(RunError):
    """The operator cancelled the run (a cancel token). Not an integrity event and not a batch halt."""

    code = "RUN_CANCELLED"


class EvidenceUnavailable(RunError):
    """The evidence stream refused an append. Nothing may be certified that was not sealed."""

    code = "EVIDENCE_UNAVAILABLE"


class ReceiptChainBroken(RunError):
    """The append-only receipt chain does not verify (a receipt was altered, removed, reordered or misfiled)."""

    code = "RECEIPT_CHAIN_BROKEN"
    run_outcome = OUTCOME_INCONCLUSIVE
    run_reason = REASON_ASSERTION_INTEGRITY


RUN_ERRORS = {cls.code: cls for cls in (RunError, IllegalTransition, DeadlineExceeded, BudgetExceeded, RunCancelled,
                                        EvidenceUnavailable, ReceiptChainBroken)}

__all__ = ["RunError", "IllegalTransition", "DeadlineExceeded", "BudgetExceeded", "RunCancelled",
           "EvidenceUnavailable", "ReceiptChainBroken", "RUN_ERRORS"]
