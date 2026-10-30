# SPDX-License-Identifier: AGPL-3.0-or-later
"""Content-addressed blobs, full-width digests, size bounds and retention data (plan H5; ADR section 5 ``EvidenceRef``,
Addendum A.2.4 ``BlobSink``).

* **Full-width digests only.** A blob is addressed by ``sha256:<64 lowercase hex>``; a shorter form, an uppercase form,
  another prefix or a different algorithm is never accepted (:func:`require_full_digest`), and the 16-hex form of the
  harness ``TraceRecorder`` is never used for QA evidence. :func:`find_truncated_digests` lists the places in a JSON
  value that look like a cut-off digest.
* :class:`MemoryBlobSink` implements the neutral ``BlobSink`` port in memory: content-addressed (the digest is computed
  here from the bytes, never taken from the caller), size-bounded per artifact, per media type and per run, and
  verifying on read (a stored blob that no longer hashes to its address raises). The SQLite or filesystem store is
  ``redevops-qa`` Q2; it implements the same port.
* :class:`EvidenceBounds` and :class:`RetentionPolicy` are **policy data**. The bounds are enforced by the sink and the
  vault in this package; retention (14 days for a passing run, 60 days for a failing or review run, configurable) is
  recorded in the artifact manifest and **not enforced here**: deletion is Q2.
* :class:`ArtifactManifest` hashes the artifacts of a run (kind, digest, size, media type, redaction manifest hash) so a
  later check (:func:`verify_artifact_manifest`) can prove the blobs are still the ones that were recorded.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import canon
from .errors import (OUTCOME_INCONCLUSIVE, REASON_ASSERTION_INTEGRITY, QAError, ResourceLimitExceeded, SpecInvalid)
from .records import EvidenceRef

DIGEST_PREFIX = "sha256:"
MANIFEST_FORM = "qa-artifact-manifest/1"
MAX_MEDIA_TYPE_CHARS = 255

_FULL_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_FULL_HASH = re.compile(r"(?:sha256|rcv1|canon-v1):[0-9a-f]{64}")
_TRUNCATED = re.compile(r"(?:(?:sha256|rcv1|canon-v1):[0-9a-f]{1,63}(?![0-9a-f])|(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z]))")
_MEDIA_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*(?:\s*;.*)?")
_KIND = re.compile(r"[a-z][a-z0-9_]{0,39}")
_REF_KEYS = ("a11y_snapshot_ref", "console_excerpt_ref", "network_excerpt_ref", "screenshot_ref")


class BlobRejected(QAError):
    """A reference or blob that does not meet the evidence rules (a short or malformed digest, a bad media type).
    An integrity event, like a broken receipt chain: the run ends ``INCONCLUSIVE(ASSERTION_INTEGRITY)``."""

    code = "BLOB_REJECTED"
    run_outcome = OUTCOME_INCONCLUSIVE
    run_reason = REASON_ASSERTION_INTEGRITY


class BlobNotFound(QAError):
    code = "BLOB_NOT_FOUND"


class BlobCorrupt(QAError):
    """A stored blob no longer hashes to its address."""

    code = "BLOB_CORRUPT"
    run_outcome = OUTCOME_INCONCLUSIVE
    run_reason = REASON_ASSERTION_INTEGRITY


# ---------------------------------------------------------------------------------------------------------------
# digests and references
# ---------------------------------------------------------------------------------------------------------------

def is_full_digest(text: Any) -> bool:
    """True only for ``sha256:`` followed by exactly 64 lowercase hex digits."""
    return type(text) is str and _FULL_DIGEST.fullmatch(text) is not None


def is_full_hash(text: Any) -> bool:
    """True for a full-width ``sha256:``, ``rcv1:`` or ``canon-v1:`` hash (64 lowercase hex digits)."""
    return type(text) is str and _FULL_HASH.fullmatch(text) is not None


def require_full_digest(text: Any) -> str:
    if not is_full_digest(text):
        raise BlobRejected("an evidence digest must be sha256: followed by 64 lowercase hex digits")
    return text


def digest_of_bytes(data: bytes) -> str:
    """``sha256:<64 hex>`` of the bytes."""
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


def evidence_ref(digest: str, media_type: Optional[str] = None, *, source: Optional[str] = None,
                 modality: Optional[str] = None) -> EvidenceRef:
    """The ``EvidenceRef`` for a stored blob: ``ref`` and ``content_hash`` are both the full digest."""
    require_full_digest(digest)
    return EvidenceRef(ref=digest, content_hash=digest, media_type=media_type, modality=modality, source=source)


def require_blob_ref(ref: Any) -> str:
    """The digest of an ``EvidenceRef`` (or its JSON form) that points at a blob; refuses a short or foreign hash."""
    if isinstance(ref, EvidenceRef):
        content, name = ref.content_hash, ref.ref
    elif isinstance(ref, Mapping):
        content, name = ref.get("content_hash"), ref.get("ref")
    else:
        raise BlobRejected("not an evidence reference")
    require_full_digest(content)
    if name != content:
        raise BlobRejected("the reference does not name its own content hash")
    return content


def find_truncated_digests(value: Any, path: str = "$") -> Tuple[str, ...]:
    """Paths in a JSON-shaped value whose string holds a cut-off digest: a ``sha256:``, ``rcv1:`` or ``canon-v1:``
    prefix with fewer than 64 hex digits, or a bare 16-hex token (the ``TraceRecorder`` form). Sorted, deterministic."""
    found: List[str] = []

    def walk(v: Any, p: str) -> None:
        if isinstance(v, str):
            if _TRUNCATED.search(v):
                found.append(p)
        elif isinstance(v, Mapping):
            for key in sorted(v, key=str):
                if isinstance(key, str) and _TRUNCATED.search(key):
                    found.append(f"{p}.<key>")
                walk(v[key], f"{p}.{key}")
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                walk(item, f"{p}[{i}]")

    walk(value, path)
    return tuple(sorted(set(found)))


def evidence_refs_in(payload: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """The reference fields of an ``Observation`` payload that are present: ``(field name, value)``."""
    return [(k, payload[k]) for k in _REF_KEYS if payload.get(k) is not None]


# ---------------------------------------------------------------------------------------------------------------
# policy data: bounds and retention
# ---------------------------------------------------------------------------------------------------------------

def _positive(name: str, value: Any, high: int = 1 << 40) -> int:
    if type(value) is not int or not 1 <= value <= high:
        raise SpecInvalid(f"{name}: expected an integer between 1 and {high}")
    return value


@dataclass(frozen=True, kw_only=True)
class EvidenceBounds:
    """Size bounds, as policy data. The numbers are the plan's proposals (50 MB per run, 5 MB per artifact, 25 MB for a
    trace), not measurements. ``per_media_type`` raises or lowers the artifact bound for one bare media type."""

    max_artifact_bytes: int = 5 * 1024 * 1024
    max_run_bytes: int = 50 * 1024 * 1024
    max_artifacts: int = 256
    per_media_type: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    max_event_bytes: int = 64 * 1024            # one evidence event (canonical JSON)
    max_events: int = 20_000
    max_events_bytes: int = 16 * 1024 * 1024    # all evidence events of one run
    seal_reserve_bytes: int = 256 * 1024        # room kept for the closing events after the run bound is hit

    def __post_init__(self) -> None:
        for name in ("max_artifact_bytes", "max_run_bytes", "max_artifacts", "max_event_bytes", "max_events",
                     "max_events_bytes", "seal_reserve_bytes"):
            _positive(f"EvidenceBounds.{name}", getattr(self, name))
        if not isinstance(self.per_media_type, Mapping):
            raise SpecInvalid("EvidenceBounds.per_media_type: expected a mapping")
        table = {}
        for media, limit in self.per_media_type.items():
            if type(media) is not str or "/" not in media or ";" in media:
                raise SpecInvalid("EvidenceBounds.per_media_type: keys are bare media types")
            table[media.strip().lower()] = _positive("EvidenceBounds.per_media_type", limit)
        object.__setattr__(self, "per_media_type", MappingProxyType(dict(sorted(table.items()))))

    def artifact_limit(self, media_type: str) -> int:
        base = media_type.split(";", 1)[0].strip().lower()
        return self.per_media_type.get(base, self.max_artifact_bytes)

    def to_json(self) -> Dict[str, Any]:
        return {"max_artifact_bytes": self.max_artifact_bytes, "max_artifacts": self.max_artifacts,
                "max_event_bytes": self.max_event_bytes, "max_events": self.max_events,
                "max_events_bytes": self.max_events_bytes, "max_run_bytes": self.max_run_bytes,
                "per_media_type": dict(self.per_media_type), "seal_reserve_bytes": self.seal_reserve_bytes}


RETENTION_PASSING = "PASSING"
RETENTION_FAILING = "FAILING"


@dataclass(frozen=True, kw_only=True)
class RetentionPolicy:
    """How long a run's artifacts are meant to be kept: policy data recorded in the manifest. **Not enforced here**
    (deletion is the evidence store, Q2). A run that passed keeps ``passing_days``; every other outcome (product
    failure, test defect, environment failure, inconclusive, or an outcome not given) keeps ``failing_days``: the longer
    period is the safe default for evidence of something that was not a clean pass."""

    passing_days: int = 14
    failing_days: int = 60

    def __post_init__(self) -> None:
        _positive("RetentionPolicy.passing_days", self.passing_days, 3650)
        _positive("RetentionPolicy.failing_days", self.failing_days, 3650)

    def retention_class(self, outcome: Optional[str]) -> str:
        return RETENTION_PASSING if outcome == "PASS" else RETENTION_FAILING

    def days_for(self, outcome: Optional[str]) -> int:
        return self.passing_days if self.retention_class(outcome) == RETENTION_PASSING else self.failing_days

    def to_json(self) -> Dict[str, int]:
        return {"failing_days": self.failing_days, "passing_days": self.passing_days}


# ---------------------------------------------------------------------------------------------------------------
# the in-memory content-addressed store
# ---------------------------------------------------------------------------------------------------------------

def _check_media_type(media_type: Any) -> str:
    if type(media_type) is not str or not media_type or len(media_type) > MAX_MEDIA_TYPE_CHARS \
            or _MEDIA_TYPE.fullmatch(media_type) is None:
        raise BlobRejected("a blob needs a valid media type")
    return media_type


class MemoryBlobSink:
    """Memory-only ``BlobSink``. The digest is computed from the bytes; ``get`` re-hashes what it returns."""

    def __init__(self, bounds: Optional[EvidenceBounds] = None) -> None:
        self._bounds = bounds if bounds is not None else EvidenceBounds()
        if type(self._bounds) is not EvidenceBounds:
            raise SpecInvalid("MemoryBlobSink: bounds must be EvidenceBounds")
        self._blobs: Dict[str, bytes] = {}
        self._media: Dict[str, str] = {}
        self._total = 0

    @property
    def bounds(self) -> EvidenceBounds:
        return self._bounds

    def put(self, data: bytes, media_type: str) -> str:
        if type(data) not in (bytes, bytearray):
            raise BlobRejected("a blob is bytes")
        media_type = _check_media_type(media_type)
        data = bytes(data)
        limit = self._bounds.artifact_limit(media_type)
        if len(data) > limit:
            raise ResourceLimitExceeded("the artifact is over its size bound", bound="artifact", limit=limit)
        digest = digest_of_bytes(data)
        if digest in self._blobs:
            return digest                                    # content-addressed: one copy, counted once
        if len(self._blobs) >= self._bounds.max_artifacts:
            raise ResourceLimitExceeded("too many artifacts in this run", bound="count",
                                        limit=self._bounds.max_artifacts)
        if self._total + len(data) > self._bounds.max_run_bytes:
            raise ResourceLimitExceeded("the run's artifacts are over the run size bound", bound="run",
                                        limit=self._bounds.max_run_bytes)
        self._blobs[digest] = data
        self._media[digest] = media_type
        self._total += len(data)
        return digest

    def get(self, digest: str) -> bytes:
        require_full_digest(digest)
        try:
            data = self._blobs[digest]
        except KeyError:
            raise BlobNotFound("no such blob") from None
        if digest_of_bytes(data) != digest:
            raise BlobCorrupt("a stored blob does not hash to its address")
        return data

    def exists(self, digest: str) -> bool:
        require_full_digest(digest)
        return digest in self._blobs

    def size(self, digest: str) -> int:
        require_full_digest(digest)
        try:
            return len(self._blobs[digest])
        except KeyError:
            raise BlobNotFound("no such blob") from None

    def media_type(self, digest: str) -> str:
        require_full_digest(digest)
        try:
            return self._media[digest]
        except KeyError:
            raise BlobNotFound("no such blob") from None

    @property
    def total_bytes(self) -> int:
        return self._total

    def __len__(self) -> int:
        return len(self._blobs)

    def digests(self) -> Tuple[str, ...]:
        return tuple(sorted(self._blobs))


# ---------------------------------------------------------------------------------------------------------------
# the artifact manifest
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactEntry:
    kind: str
    digest: str
    size: int
    media_type: str
    redaction_manifest: Optional[str] = None      # ``canon-v1:`` hash of the redaction manifest, if one was made
    step_id: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.kind) is not str or _KIND.fullmatch(self.kind) is None:
            raise BlobRejected("an artifact kind is a short lower-case identifier")
        require_full_digest(self.digest)
        if type(self.size) is not int or self.size < 0:
            raise BlobRejected("an artifact size is a non-negative integer")
        _check_media_type(self.media_type)
        if self.redaction_manifest is not None and not is_full_hash(self.redaction_manifest):
            raise BlobRejected("the redaction manifest hash must be full width")

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"digest": self.digest, "kind": self.kind, "media_type": self.media_type,
                               "size": self.size}
        if self.redaction_manifest is not None:
            out["redaction_manifest"] = self.redaction_manifest
        if self.step_id is not None:
            out["step_id"] = self.step_id
        return out


@dataclass(frozen=True)
class ArtifactManifest:
    """The artifacts of one run, in the order they were stored, with the retention data that applies to them."""

    entries: Tuple[ArtifactEntry, ...]
    retention_class: str = RETENTION_FAILING
    retention_days: int = 60
    redaction_policy_id: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"entries": [e.to_json() for e in self.entries], "form": MANIFEST_FORM,
                               "retention_class": self.retention_class, "retention_days": self.retention_days,
                               "total_bytes": sum(e.size for e in self.entries)}
        if self.redaction_policy_id is not None:
            out["redaction_policy_id"] = self.redaction_policy_id
        return out

    @property
    def manifest_hash(self) -> str:
        """``canon-v1:<64 hex>`` of the whole manifest."""
        return canon.canon_hash(self.to_json())


def verify_artifact_manifest(manifest: Any, blobs: Any) -> Tuple[Tuple[str, int], ...]:
    """Check an artifact manifest (an :class:`ArtifactManifest` or its JSON form) against the blob store. Returns the
    findings as ``(code, entry index)``: ``MANIFEST_MALFORMED``, ``BAD_DIGEST``, ``MISSING_BLOB``, ``SIZE_MISMATCH``,
    ``CONTENT_MISMATCH`` (the bytes no longer hash to the recorded digest). An empty tuple means every artifact is
    still exactly what was recorded."""
    doc = manifest.to_json() if isinstance(manifest, ArtifactManifest) else manifest
    if not isinstance(doc, Mapping) or doc.get("form") != MANIFEST_FORM or not isinstance(doc.get("entries"), list):
        return (("MANIFEST_MALFORMED", -1),)
    findings: List[Tuple[str, int]] = []
    for i, entry in enumerate(doc["entries"]):
        if not isinstance(entry, Mapping) or not is_full_digest(entry.get("digest")):
            findings.append(("BAD_DIGEST", i))
            continue
        digest = entry["digest"]
        try:
            if not blobs.exists(digest):
                findings.append(("MISSING_BLOB", i))
                continue
            if blobs.size(digest) != entry.get("size"):
                findings.append(("SIZE_MISMATCH", i))
                continue
            data = blobs.get(digest)
        except BlobCorrupt:
            findings.append(("CONTENT_MISMATCH", i))
            continue
        except QAError:
            findings.append(("MISSING_BLOB", i))
            continue
        if digest_of_bytes(data) != digest:
            findings.append(("CONTENT_MISMATCH", i))
    return tuple(findings)


__all__ = ["DIGEST_PREFIX", "MANIFEST_FORM", "BlobRejected", "BlobNotFound", "BlobCorrupt", "is_full_digest",
           "is_full_hash", "require_full_digest", "digest_of_bytes", "evidence_ref", "require_blob_ref",
           "find_truncated_digests", "evidence_refs_in", "EvidenceBounds", "RetentionPolicy", "RETENTION_PASSING",
           "RETENTION_FAILING", "MemoryBlobSink", "ArtifactEntry", "ArtifactManifest", "verify_artifact_manifest"]
