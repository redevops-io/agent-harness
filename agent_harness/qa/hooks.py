# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small ports the single-run layer (plan H2) is injected with, kept apart from ``runner.py`` so that a recovery
implementation (plan H3) can depend on them without depending on the verifier or the classifier.

* :class:`EvidenceStream`: the append-only event stream (ADR section 6 rule 2: ``RunStarted`` first). The default is
  memory-only; a filesystem or content-addressed sink is plan H5 and lives behind the same one method.
* :class:`PostconditionChecker`: evaluates a step's declared ``expected_transition`` deterministically. The default
  evaluates nothing (``NOT_EVALUATED``), which blocks every retry of a side-effecting step (see ``idempotency.py``).
* :class:`PrepareChecks`: environment health, spec preconditions and fixture hashes. ``None`` means "not verified", and
  an unverified attestation can never produce ``PASS`` (the classifier treats it as ``UNKNOWN``).
* :class:`RecoveryHook`: where plan H3 plugs the recovery *step* in. Disabled by default. A request carries the frozen
  step and the current observation and nothing else (no assertion, no expected value, no verdict); a response is a
  strategy, a resolved target and a rationale, and the runner re-checks all of it.
* :class:`CancelToken`: operator cancellation. Not the batch-halt latch (that is ``verifier.HaltLatch``).

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

from .errors import SpecInvalid
from .records import (Observation, Postcondition, PostconditionResult, RecoveryStrategy, ResolvedTarget, Step,
                      TestSpec)


@runtime_checkable
class EvidenceStream(Protocol):
    """Append-only. ``append`` must either store the event or raise; it returns nothing and removes nothing."""

    def append(self, event: Mapping[str, Any]) -> None: ...


class MemoryEvidenceStream:
    """In-memory evidence stream. No filesystem, no network. Events are kept as given (plain JSON dictionaries)."""

    def __init__(self) -> None:
        self._events: List[Mapping[str, Any]] = []

    def append(self, event: Mapping[str, Any]) -> None:
        self._events.append(event)

    @property
    def events(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self._events)


@runtime_checkable
class PostconditionChecker(Protocol):
    """Decides whether the step's declared transition was observed. Must be deterministic and must not call a model."""

    def evaluate(self, step: Step, before: Observation, after: Observation) -> Postcondition: ...


class NotEvaluatedPostcondition:
    """Default checker: the declared transition is recorded and not evaluated."""

    def evaluate(self, step: Step, before: Observation, after: Observation) -> Postcondition:
        return Postcondition(declared=step.expected_transition, result=PostconditionResult.NOT_EVALUATED)


@runtime_checkable
class PrepareChecks(Protocol):
    """Attestations the runner cannot make itself. Return ``True`` (verified), ``False`` (failed) or ``None``
    (not verified)."""

    def environment_healthy(self, phase: str) -> Optional[bool]: ...      # phase is "before" or "after"

    def preconditions_met(self, spec: TestSpec) -> Optional[bool]: ...

    def fixtures_match(self, spec: TestSpec) -> Optional[bool]: ...


class UnverifiedChecks:
    """Default: nothing is attested. A spec that declares preconditions or fixtures cannot reach ``PASS`` with it."""

    def environment_healthy(self, phase: str) -> Optional[bool]:
        return None

    def preconditions_met(self, spec: TestSpec) -> Optional[bool]:
        return None

    def fixtures_match(self, spec: TestSpec) -> Optional[bool]:
        return None


@dataclass(frozen=True, kw_only=True)
class RecoveryRequest:
    """What the recovery hook may see. There is no assertion field and none may be added (ADR section 6 rule 4)."""

    run_id: str
    step: Step
    observation: Observation
    error_code: str
    attempt: int                      # the attempt number the recovery would become
    recovery_attempts_left: int
    model_calls_left: int
    tokens_left: int


@dataclass(frozen=True, kw_only=True)
class RecoveryResponse:
    """A proposal: act on ``target`` with ``strategy``. ``model_calls``/``tokens`` are what producing it cost."""

    strategy: RecoveryStrategy
    target: Optional[ResolvedTarget] = None
    rationale: str = ""
    model_calls: int = 0
    tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, RecoveryStrategy):
            raise SpecInvalid("RecoveryResponse.strategy must be a RecoveryStrategy")
        if self.target is not None and type(self.target) is not ResolvedTarget:
            raise SpecInvalid("RecoveryResponse.target must be a ResolvedTarget")
        if type(self.rationale) is not str:
            raise SpecInvalid("RecoveryResponse.rationale must be a string")
        for name in ("model_calls", "tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SpecInvalid(f"RecoveryResponse.{name} must be a non-negative integer")


@runtime_checkable
class RecoveryHook(Protocol):
    """Plan H3 implements this. Return ``None`` to decline (the run then ends ``INCONCLUSIVE``)."""

    def recover(self, request: RecoveryRequest) -> Optional[RecoveryResponse]: ...


class CancelToken:
    """Operator cancellation: once cancelled it stays cancelled."""

    __slots__ = ("_reason",)

    def __init__(self) -> None:
        self._reason: Optional[str] = None

    def cancel(self, reason: str = "cancelled") -> None:
        if self._reason is None:
            self._reason = str(reason)[:256] or "cancelled"

    @property
    def cancelled(self) -> bool:
        return self._reason is not None

    @property
    def reason(self) -> Optional[str]:
        return self._reason


__all__ = ["EvidenceStream", "MemoryEvidenceStream", "PostconditionChecker", "NotEvaluatedPostcondition",
           "PrepareChecks", "UnverifiedChecks", "RecoveryRequest", "RecoveryResponse", "RecoveryHook", "CancelToken"]
