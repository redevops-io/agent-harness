# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redact before storing, bound what is stored, and record what was done.

Two classes, one rule: **nothing reaches storage or the evidence stream unredacted, and a failure stores nothing.**

* :class:`ArtifactVault` wraps a ``BlobSink`` and a ``Redactor``. It *is* a ``BlobSink`` (``put``, ``get``, ``exists``,
  ``size``), so a component handed the vault in place of a raw sink cannot store raw bytes: ``put`` redacts first. A
  redaction failure of any kind (typed, or an unexpected exception from a custom redactor, or output that is not stable
  under a second pass) drops the artifact, records a failure marker (kind and reason code, never the content) and raises
  ``RedactionFailed``. Size bounds apply to the raw input, the redacted output and the run total. ``manifest()`` gives
  the hashed :class:`~agent_harness.qa.blobs.ArtifactManifest` with the retention data.
* :class:`EvidenceGuard` is what the runner applies to every evidence event before it is hashed into the chain: the
  payload is redacted (key names decide first; strings are redacted as text; a record payload is rebuilt from its
  redacted form so its content hash stays true), reference fields are checked for full-width digests, and the size of
  the event, of all events and their count are bounded. It keeps the redaction tally and the list of events that were
  changed, which the runner writes into the evidence stream as the run's ``RedactionManifest`` event.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from . import canon
from .blobs import (ArtifactEntry, ArtifactManifest, BlobRejected, EvidenceBounds, RetentionPolicy, digest_of_bytes,
                    evidence_ref, evidence_refs_in, require_blob_ref)
from .errors import RedactionFailed, ResourceLimitExceeded, SpecInvalid
from .ports import BlobSink, Redactor
from .records import EvidenceRef
from .redaction import PatternRedactor, RedactionManifest, RedactionTally, normalise_key

FAILURE_FORM = "qa-evidence-failure/1"
RUN_REDACTION_FORM = "qa-redaction-run/1"
#: record fields that are structure, not secrets: their names contain a secret-looking word (the reference to a secret
#: held elsewhere, never its value)
STRUCTURAL_KEYS: FrozenSet[str] = frozenset({"secretref"})
#: closing events: they are not counted against the run's event-size bound (a run that hit the bound must still be able
#: to say so), only against the small reserve
SEAL_KINDS: FrozenSet[str] = frozenset({"RunSealed", "RedactionManifest", "EgressSummary", "EvidenceFailure",
                                        "ReviewItem"})
_KIND = re.compile(r"[A-Za-z][A-Za-z0-9]{0,39}")
_REASON = re.compile(r"[A-Z][A-Z0-9_]{0,39}")
_BOUND = re.compile(r"[a-z][a-z_]{0,19}")


@dataclass(frozen=True)
class ArtifactFailure:
    """A stored-nothing marker: the artifact kind, the failure code and a reason from a fixed vocabulary. No content."""

    kind: str
    code: str
    reason: str
    input_bytes: int

    def to_json(self) -> Dict[str, Any]:
        return {"code": self.code, "input_bytes": self.input_bytes, "kind": self.kind, "reason": self.reason}


@dataclass(frozen=True)
class StoredArtifact:
    entry: ArtifactEntry
    ref: EvidenceRef
    redaction: RedactionManifest


def _reason_of(exc: BaseException) -> str:
    reason = getattr(exc, "details", {}).get("reason") if hasattr(exc, "details") else None
    return reason if type(reason) is str and _REASON.fullmatch(reason) else "UNSPECIFIED"


class ArtifactVault:
    """A ``BlobSink`` that redacts before it stores. ``sink`` should be fresh for the run (the bounds are per run)."""

    def __init__(self, redactor: Redactor, sink: BlobSink, *, bounds: Optional[EvidenceBounds] = None,
                 retention: Optional[RetentionPolicy] = None) -> None:
        for what, obj, methods in (("redactor", redactor, ("redact",)), ("sink", sink, ("put", "get", "exists", "size"))):
            if obj is None or not all(callable(getattr(obj, m, None)) for m in methods):
                raise SpecInvalid(f"ArtifactVault: {what} must provide {', '.join(methods)}")
        if not isinstance(getattr(redactor, "policy_id", None), str):
            raise SpecInvalid("ArtifactVault: the redactor must have a policy_id")
        self._redactor = redactor
        self._sink = sink
        self._bounds = bounds if bounds is not None else EvidenceBounds()
        self._retention = retention if retention is not None else RetentionPolicy()
        self._entries: List[ArtifactEntry] = []
        self._failures: List[ArtifactFailure] = []
        self._total = 0

    @property
    def policy_id(self) -> str:
        return self._redactor.policy_id

    @property
    def failures(self) -> Tuple[ArtifactFailure, ...]:
        return tuple(self._failures)

    def _drop(self, kind: str, size: int, exc: BaseException, code: str) -> None:
        self._failures.append(ArtifactFailure(kind=kind, code=code, reason=_reason_of(exc), input_bytes=size))

    def _redacted(self, data: bytes, media_type: str) -> Tuple[bytes, Optional[RedactionManifest]]:
        manifest: Optional[RedactionManifest] = None
        try:
            if callable(getattr(self._redactor, "redact_with_manifest", None)):
                out, manifest = self._redactor.redact_with_manifest(data, media_type)
            else:
                out = self._redactor.redact(data, media_type)
            if type(out) is not bytes:
                raise RedactionFailed("the redactor did not return bytes", reason="BAD_OUTPUT")
            if self._redactor.redact(out, media_type) != out:           # a second pass must change nothing
                raise RedactionFailed("redacting the redacted output changed it", reason="UNSTABLE")
        except RedactionFailed:
            raise
        except Exception as exc:                                         # fail closed: nothing raw is stored
            raise RedactionFailed("the redactor failed (%s)" % type(exc).__name__, reason="INTERNAL") from exc
        return out, manifest

    def store(self, kind: str, data: bytes, media_type: str, *, step_id: Optional[str] = None) -> StoredArtifact:
        """Redact, bound, store, record. Raises ``RedactionFailed``, ``ResourceLimitExceeded`` or ``BlobRejected`` and
        stores nothing when any step fails."""
        size = len(data) if type(data) in (bytes, bytearray) else 0
        try:
            if type(data) not in (bytes, bytearray):
                raise BlobRejected("an artifact is bytes")
            limit = self._bounds.artifact_limit(media_type) if type(media_type) is str else self._bounds.max_artifact_bytes
            if size > limit:
                raise ResourceLimitExceeded("the artifact is over its size bound", bound="artifact", limit=limit)
            out, manifest = self._redacted(bytes(data), media_type)
            if len(out) > limit:
                raise ResourceLimitExceeded("the redacted artifact is over its size bound", bound="artifact", limit=limit)
            if len(self._entries) >= self._bounds.max_artifacts:
                raise ResourceLimitExceeded("too many artifacts in this run", bound="count",
                                            limit=self._bounds.max_artifacts)
            if self._total + len(out) > self._bounds.max_run_bytes:
                raise ResourceLimitExceeded("the run's artifacts are over the run size bound", bound="run",
                                            limit=self._bounds.max_run_bytes)
            digest = self._sink.put(out, media_type)
            if digest != digest_of_bytes(out):
                raise BlobRejected("the sink returned a digest that is not the content address of the bytes")
            if manifest is None:
                manifest = RedactionManifest(rule_set_hash=self.policy_id, counts=(), input_bytes=size,
                                             output_bytes=len(out), media_type=media_type)
            entry = ArtifactEntry(kind=kind, digest=digest, size=len(out), media_type=media_type,
                                  redaction_manifest=manifest.manifest_hash, step_id=step_id)
        except RedactionFailed as exc:
            self._drop(kind if type(kind) is str else "unknown", size, exc, "REDACTION_FAILED")
            raise
        except (ResourceLimitExceeded, BlobRejected) as exc:
            self._drop(kind if type(kind) is str else "unknown", size, exc, exc.code)
            raise
        self._entries.append(entry)
        self._total += len(out)
        return StoredArtifact(entry=entry, ref=evidence_ref(digest, media_type), redaction=manifest)

    # -- the BlobSink port ---------------------------------------------------------------------------------
    def put(self, data: bytes, media_type: str) -> str:
        return self.store("blob", data, media_type).entry.digest

    def get(self, digest: str) -> bytes:
        return self._sink.get(digest)

    def exists(self, digest: str) -> bool:
        return self._sink.exists(digest)

    def size(self, digest: str) -> int:
        return self._sink.size(digest)

    def manifest(self, outcome: Optional[str] = None) -> ArtifactManifest:
        """The hashed manifest of what was stored, with the retention class and days for ``outcome``."""
        return ArtifactManifest(entries=tuple(self._entries), retention_class=self._retention.retention_class(outcome),
                                retention_days=self._retention.days_for(outcome), redaction_policy_id=self.policy_id)


def _strip_record_hashes(value: Any) -> Any:
    """Drop the ``content_hash`` that ``to_json(with_hash=True)`` adds to each record (a dictionary with a
    ``schema_id``), so the record can be rebuilt from its redacted form and re-hashed. The ``content_hash`` field of an
    ``EvidenceRef`` is data and stays."""
    if isinstance(value, Mapping):
        is_record = "schema_id" in value
        return {k: _strip_record_hashes(v) for k, v in value.items() if not (is_record and k == "content_hash")}
    if isinstance(value, list):
        return [_strip_record_hashes(v) for v in value]
    return value


class EvidenceGuard:
    """Applied by the runner to every evidence event. One guard per run (it counts)."""

    def __init__(self, redactor: PatternRedactor, bounds: Optional[EvidenceBounds] = None, *,
                 exempt_keys: FrozenSet[str] = STRUCTURAL_KEYS) -> None:
        if not callable(getattr(redactor, "redact_structure", None)) or not isinstance(getattr(redactor, "policy_id", None), str):
            raise SpecInvalid("EvidenceGuard: the redactor must provide redact_structure and a policy_id")
        self._redactor = redactor
        self._bounds = bounds if bounds is not None else EvidenceBounds()
        self._exempt = frozenset(normalise_key(k) for k in exempt_keys)
        self._tally = RedactionTally(redactor.policy_id)
        self._redacted: List[Tuple[int, str]] = []
        self._events = 0
        self._bytes = 0
        self._seal_bytes = 0

    @property
    def policy_id(self) -> str:
        return self._redactor.policy_id

    @property
    def redacted_events(self) -> Tuple[Tuple[int, str], ...]:
        """``(sequence number, kind)`` of every event whose payload was changed by redaction."""
        return tuple(self._redacted)

    def apply(self, kind: str, record: Any, payload: Dict[str, Any], payload_hash: str, *,
              seq: int) -> Tuple[Dict[str, Any], str]:
        """The payload and hash to put in the chain for an event: redacted, bounded, with references checked. Raises
        ``RedactionFailed``, ``ResourceLimitExceeded`` or ``BlobRejected``; the event is then not recorded."""
        seal = kind in SEAL_KINDS
        if not seal and self._events >= self._bounds.max_events:
            raise ResourceLimitExceeded("too many evidence events in this run", bound="events",
                                        limit=self._bounds.max_events)
        new_payload, counts = self._redactor.redact_structure(payload, exempt_keys=self._exempt)
        self._tally.add(counts)
        if counts:
            if hasattr(record, "canon_hash") and hasattr(type(record), "from_json"):
                try:
                    rebuilt = type(record).from_json(_strip_record_hashes(new_payload))
                    new_payload, payload_hash = rebuilt.to_json(with_hash=True), rebuilt.canon_hash()
                except Exception as exc:                 # the redacted form is not a valid record: fail closed
                    raise RedactionFailed("the redacted record is no longer a valid record", reason="RECORD_INVALID") from exc
            else:
                payload_hash = canon.canon_hash(new_payload)
            self._redacted.append((seq, kind))
        if kind == "Observation":
            for _, value in evidence_refs_in(new_payload):
                require_blob_ref(value)                  # a short or foreign hash in an evidence reference is refused
        size = len(canon.canonical_bytes(new_payload))
        if not seal and size > self._bounds.max_event_bytes:
            raise ResourceLimitExceeded("an evidence event is over its size bound", bound="event",
                                        limit=self._bounds.max_event_bytes)
        if seal:
            if self._seal_bytes + size > self._bounds.seal_reserve_bytes:
                raise ResourceLimitExceeded("the closing events are over their reserve", bound="seal",
                                            limit=self._bounds.seal_reserve_bytes)
            self._seal_bytes += size
        else:
            if self._bytes + size > self._bounds.max_events_bytes:
                raise ResourceLimitExceeded("the run's evidence events are over the run bound", bound="events_bytes",
                                            limit=self._bounds.max_events_bytes)
            self._bytes += size
            self._events += 1
        return new_payload, payload_hash

    def failure_payload(self, kind: str, exc: BaseException) -> Dict[str, Any]:
        """The ``EvidenceFailure`` payload for an event that could not be recorded: the kind, the code and a reason from
        a fixed vocabulary. No content of the dropped event."""
        out = {"code": str(getattr(exc, "code", "UNKNOWN")), "form": FAILURE_FORM,
               "kind": kind if type(kind) is str and _KIND.fullmatch(kind) else "Unknown", "reason": _reason_of(exc)}
        bound = getattr(exc, "details", {}).get("bound") if hasattr(exc, "details") else None
        if type(bound) is str and _BOUND.fullmatch(bound):
            out["bound"] = bound                         # which size bound: "event", "events", "events_bytes", "seal"
        return out

    def manifest_payload(self) -> Dict[str, Any]:
        """The run's ``RedactionManifest`` event payload: counts per rule, the rule-set hash, which events were
        changed, and the sizes. Counts and names only; nothing that was removed."""
        return {"bounds": self._bounds.to_json(), "event_bytes": self._bytes, "events": self._events,
                "form": RUN_REDACTION_FORM, "policy_id": self.policy_id,
                "redacted_events": [{"kind": k, "seq": s} for s, k in self._redacted], "tally": self._tally.to_json()}


__all__ = ["FAILURE_FORM", "RUN_REDACTION_FORM", "STRUCTURAL_KEYS", "SEAL_KINDS", "ArtifactFailure", "StoredArtifact",
           "ArtifactVault", "EvidenceGuard"]
