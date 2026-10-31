# SPDX-License-Identifier: AGPL-3.0-or-later
"""Idempotency keys and the retry decision.

:func:`decide_retry` is a pure function. It answers one question: may this step's action be performed again after a
failed attempt? It never retries blindly:

* nothing is retried unless recovery is enabled, retries are left, the spec permits ``RETRY_SAME`` for the step and
  the error is retryable;
* if the declared postcondition is already observed, the action took effect: **do not replay** (``ALREADY_APPLIED``);
* an attempt that never reached the adapter's ``act`` (a locate failure) is not a replay and needs no guard;
* ``retry_safety = UNSAFE`` (the default) is never retried;
* otherwise a guard is required, by side-effect class:

  ================================================  =====================================================
  ``NONE``, ``READS_EXTERNAL``                      declared ``IDEMPOTENT``, or an observed postcondition
                                                    check showing the effect is **not** applied
  ``WRITES_INTERNAL``                               an idempotency key honoured by the target
                                                    (``SAFE_WITH_IDEMPOTENCY_KEY`` and ``IDEMPOTENT`` or
                                                    ``AT_MOST_ONCE``), or the postcondition check
  ``WRITES_EXTERNAL``, ``NOTIFIES_HUMAN``,          **both**: the idempotency key **and** a postcondition
  ``SPENDS_MONEY`` (send, payment, approval,        check showing the effect is not applied
  deletion, submit)
  ================================================  =====================================================

  ``Idempotency.UNKNOWN`` never counts as a guard, and a postcondition that was not evaluated is not a check.

The key is stable for a step across its attempts: the target can deduplicate a replay. Standard library and
neutral-module imports only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from . import canon
from .records import Idempotency, PermittedRecovery, PostconditionResult, RetrySafety, SideEffect

#: decision reasons (stable strings)
ALLOW_NOT_EXECUTED = "NOT_EXECUTED"
ALLOW_IDEMPOTENT_DECLARED = "IDEMPOTENT_DECLARED"
ALLOW_IDEMPOTENCY_KEY = "IDEMPOTENCY_KEY"
ALLOW_POSTCONDITION_NOT_APPLIED = "POSTCONDITION_NOT_APPLIED"
ALLOW_KEY_AND_POSTCONDITION = "IDEMPOTENCY_KEY_AND_POSTCONDITION"
DENY_INVALID_INPUT = "INVALID_INPUT"
DENY_RECOVERY_DISABLED = "RECOVERY_DISABLED"
DENY_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
DENY_NOT_PERMITTED = "NOT_PERMITTED"
DENY_NOT_RETRYABLE = "NOT_RETRYABLE"
DENY_ALREADY_APPLIED = "ALREADY_APPLIED"
DENY_RETRY_UNSAFE = "RETRY_UNSAFE"
DENY_UNKNOWN_SIDE_EFFECT_CLASS = "UNKNOWN_SIDE_EFFECT_CLASS"
DENY_NO_GUARD = "NO_IDEMPOTENCY_OR_POSTCONDITION_GUARD"

_READ_ONLY = frozenset({SideEffect.NONE, SideEffect.READS_EXTERNAL})
_INTERNAL_WRITE = frozenset({SideEffect.WRITES_INTERNAL})
_EXTERNAL_OR_COSTLY = frozenset({SideEffect.WRITES_EXTERNAL, SideEffect.NOTIFIES_HUMAN, SideEffect.SPENDS_MONEY})
_KEYED = frozenset({Idempotency.IDEMPOTENT, Idempotency.AT_MOST_ONCE})


def idempotency_key(run_id: str, step_id: str) -> str:
    """The idempotency key of a step: a digest of ``(run_id, step_id)``. The same for every attempt of the step (so a
    replay is recognisable to the target) and different for every step and run. Matches the ``Id`` pattern."""
    return "idem-" + canon.digest_hex({"run_id": run_id, "step_id": step_id})[:40]


@dataclass(frozen=True, kw_only=True)
class RetryInputs:
    side_effect: SideEffect
    idempotency: Idempotency
    retry_safety: RetrySafety
    permitted_recovery: Tuple[PermittedRecovery, ...]
    may_have_executed: bool          # the adapter's act() was invoked for the failed attempt
    has_idempotency_key: bool        # the action request carried the step's idempotency key
    postcondition: PostconditionResult   # observed after the failed attempt
    error_retryable: bool
    recovery_enabled: bool
    retries_left: int


@dataclass(frozen=True)
class RetryDecision:
    allow: bool
    reason: str

    @property
    def already_applied(self) -> bool:
        return self.reason == DENY_ALREADY_APPLIED


def _deny(reason: str) -> RetryDecision:
    return RetryDecision(False, reason)


def _valid(i: Any) -> bool:
    return (type(i) is RetryInputs and isinstance(i.side_effect, SideEffect) and isinstance(i.idempotency, Idempotency)
            and isinstance(i.retry_safety, RetrySafety) and isinstance(i.postcondition, PostconditionResult)
            and isinstance(i.permitted_recovery, tuple)
            and all(isinstance(p, PermittedRecovery) for p in i.permitted_recovery)
            and type(i.may_have_executed) is bool and type(i.has_idempotency_key) is bool
            and type(i.error_retryable) is bool and type(i.recovery_enabled) is bool
            and type(i.retries_left) is int)


def decide_retry(i: RetryInputs) -> RetryDecision:
    """May the action be performed again? Pure: the same inputs always give the same decision."""
    if not _valid(i):
        return _deny(DENY_INVALID_INPUT)
    if not i.recovery_enabled:
        return _deny(DENY_RECOVERY_DISABLED)
    if i.retries_left <= 0:
        return _deny(DENY_BUDGET_EXHAUSTED)
    if PermittedRecovery.RETRY_SAME not in i.permitted_recovery:
        return _deny(DENY_NOT_PERMITTED)
    if not i.error_retryable:
        return _deny(DENY_NOT_RETRYABLE)
    if i.postcondition is PostconditionResult.SATISFIED:
        return _deny(DENY_ALREADY_APPLIED)                   # the effect is there: never replay it
    if not i.may_have_executed:
        return RetryDecision(True, ALLOW_NOT_EXECUTED)       # act() was never reached: nothing to replay
    if i.retry_safety is RetrySafety.UNSAFE:
        return _deny(DENY_RETRY_UNSAFE)
    not_applied = i.postcondition is PostconditionResult.NOT_SATISFIED
    keyed = (i.retry_safety is RetrySafety.SAFE_WITH_IDEMPOTENCY_KEY and i.idempotency in _KEYED
             and i.has_idempotency_key)
    if i.side_effect in _READ_ONLY:
        if i.idempotency is Idempotency.IDEMPOTENT:
            return RetryDecision(True, ALLOW_IDEMPOTENT_DECLARED)
        if not_applied:
            return RetryDecision(True, ALLOW_POSTCONDITION_NOT_APPLIED)
        return _deny(DENY_NO_GUARD)
    if i.side_effect in _INTERNAL_WRITE:
        if keyed:
            return RetryDecision(True, ALLOW_IDEMPOTENCY_KEY)
        if not_applied:
            return RetryDecision(True, ALLOW_POSTCONDITION_NOT_APPLIED)
        return _deny(DENY_NO_GUARD)
    if i.side_effect in _EXTERNAL_OR_COSTLY:
        if keyed and not_applied:
            return RetryDecision(True, ALLOW_KEY_AND_POSTCONDITION)
        return _deny(DENY_NO_GUARD)
    return _deny(DENY_UNKNOWN_SIDE_EFFECT_CLASS)             # a class this version has no rule for


__all__ = ["idempotency_key", "RetryInputs", "RetryDecision", "decide_retry"]
