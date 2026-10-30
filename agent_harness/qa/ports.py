# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neutral structural interfaces (``typing.Protocol``) for the QA module.

An adapter written elsewhere satisfies these structurally; neither side imports the other. The method and parameter
names mirror the frozen qa/v1 protocols (``QAAdapter`` qa-adapter/1, ``CheckAdapter`` qa-check-adapter/1,
``BlobStore``, ``Redactor``), so a conformance check can compare names. Rules that every implementation must keep
(ADR section 8):

* ``assert_`` / ``run_check`` take a subject only and return an :class:`ObservedValue`: no ``expected`` in, no verdict out;
* ``act`` takes a :class:`ResolvedTarget` from ``locate`` and an :class:`ActionRequest` that carries a policy decision
  reference, and raises ``ActionBlockedByPolicy`` without one;
* ``locate`` returns every candidate; the runner, not the adapter, decides which is acceptable;
* ``capture`` returns redacted artifacts only; a failed redaction raises ``RedactionFailed``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .records import (ActionOutcome, ActionRequest, AssertionSubject, CaptureKind, CheckProfile, CheckSubject,
                      CleanupReport, EvidenceRef, LocateMode, LocateResult, LocatorCandidate, LocatorDescriptor,
                      Observation, ObservedValue, ObserveRequest, PolicyDecisionRecord, ResetResult, ResetScope,
                      ResolvedResource, ResolvedTarget, ResourceRef, SessionHandle, SessionProfile, Step,
                      WorkspaceHandle)


@runtime_checkable
class NeutralQAAdapter(Protocol):
    """Browser-side adapter (``api_version = "qa-adapter/1"``)."""

    adapter_id: str
    adapter_version: str
    api_version: str

    def open_session(self, profile: SessionProfile) -> SessionHandle: ...

    def observe(self, s: SessionHandle, req: ObserveRequest) -> Observation: ...

    def locate(self, s: SessionHandle, target: LocatorDescriptor, *, mode: LocateMode) -> LocateResult: ...

    def act(self, s: SessionHandle, action: ActionRequest, target: ResolvedTarget) -> ActionOutcome: ...

    def assert_(self, s: SessionHandle, subject: AssertionSubject) -> ObservedValue: ...

    def capture(self, s: SessionHandle, kinds: Sequence[CaptureKind]) -> Sequence[EvidenceRef]: ...

    def reset(self, s: SessionHandle, scope: ResetScope) -> ResetResult: ...

    def close_session(self, s: SessionHandle) -> CleanupReport: ...


@runtime_checkable
class NeutralCheckAdapter(Protocol):
    """Resource-side adapter for Track I checks (``api_version = "qa-check-adapter/1"``)."""

    adapter_id: str
    adapter_version: str
    api_version: str

    def open_workspace(self, profile: CheckProfile) -> WorkspaceHandle: ...

    def resolve(self, w: WorkspaceHandle, ref: ResourceRef) -> ResolvedResource: ...

    def run_check(self, w: WorkspaceHandle, subject: CheckSubject) -> ObservedValue: ...

    def capture(self, w: WorkspaceHandle, kinds: Sequence[CaptureKind]) -> Sequence[EvidenceRef]: ...

    def reset(self, w: WorkspaceHandle, scope: ResetScope) -> ResetResult: ...

    def close_workspace(self, w: WorkspaceHandle) -> CleanupReport: ...


@runtime_checkable
class Redactor(Protocol):
    """Redacts bytes before they are stored. ``policy_id`` is the digest of the redaction policy in force."""

    policy_id: str

    def redact(self, data: bytes, media_type: str) -> bytes: ...


@runtime_checkable
class BlobSink(Protocol):
    """Content-addressed blob store. Digests are the full ``sha256:<64 hex>`` form; shorter forms are never valid."""

    def put(self, data: bytes, media_type: str) -> str: ...

    def get(self, digest: str) -> bytes: ...

    def exists(self, digest: str) -> bool: ...

    def size(self, digest: str) -> int: ...


@dataclass(frozen=True, kw_only=True)
class RecoveryContext:
    """What a recoverer may see: the frozen step and the current observation. There is no assertion field, and none may
    be added: recovery changes how a step finds its target, never what is expected (ADR section 6 rule 4)."""

    step: Step
    observation: Observation


@runtime_checkable
class Recoverer(Protocol):
    """Proposes locator candidates for a step. Output is parsed into the strict ``LocatorCandidate`` or discarded."""

    def propose(self, ctx: RecoveryContext) -> Sequence[LocatorCandidate]: ...


@runtime_checkable
class ActionPolicy(Protocol):
    """Decides one action. Fail closed: no matching rule, an unknown side effect or action, an error or a timeout is
    a ``DENY``. Every decision, allow or deny, is returned as a record whose hash goes into the action receipt."""

    def decide(self, run_id: str, step: Step, attempt: int) -> PolicyDecisionRecord: ...


@runtime_checkable
class Clock(Protocol):
    """Time source: ISO 8601 UTC strings for custody fields, monotonic milliseconds for deadlines."""

    def now_utc(self) -> str: ...

    def monotonic_ms(self) -> int: ...


PORTS = (NeutralQAAdapter, NeutralCheckAdapter, Redactor, BlobSink, Recoverer, ActionPolicy, Clock)
