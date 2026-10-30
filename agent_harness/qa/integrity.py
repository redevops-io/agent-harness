# SPDX-License-Identifier: AGPL-3.0-or-later
"""One check over the whole evidence of a run (plan H5): the receipt chain, the evidence event chain, the way they refer
to each other, and the artifact hashes.

:func:`verify_run_evidence` never raises for tampering: it returns an :class:`EvidenceReport` whose ``findings`` are
stable codes with the place they were found, so an integrator can show every problem at once;
:func:`require_valid` turns a report with findings into a ``ReceiptChainBroken`` (an integrity event). It reuses the
verification already in ``receipts.py`` (``verify_chain``, ``verify_events``, ``verify_manifest_receipts``) and adds the
cross-checks no single list can make:

* the ``ActionReceipt`` events in the stream are, in order, the receipts of the ledger (same count, same identity hash),
  except for an event the run's ``RedactionManifest`` lists as changed by redaction (its stored form is the redacted one);
* the closing ``RunSealed`` event names the ledger head and the manifest hash;
* every evidence reference in an ``Observation`` carries a full ``sha256:`` digest and, when a blob store is given, the
  blob exists and still hashes to it; the artifact manifest still matches the store;
* nothing in the stream looks like a truncated digest.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Set, Tuple

from .blobs import (BlobCorrupt, evidence_refs_in, find_truncated_digests, is_full_digest, verify_artifact_manifest)
from .errors import QAError
from .receipts import ChainLink, verify_chain, verify_events, verify_manifest_receipts
from .records import ActionReceipt, RunManifest
from .run_errors import ReceiptChainBroken


@dataclass(frozen=True)
class Finding:
    code: str
    where: str = ""

    def __str__(self) -> str:
        return f"{self.code}@{self.where}" if self.where else self.code


@dataclass(frozen=True)
class EvidenceReport:
    findings: Tuple[Finding, ...]
    event_head: Optional[str] = None
    receipt_head: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(f.code for f in self.findings)


def _redacted_seqs(events: Sequence[Mapping[str, Any]]) -> Set[int]:
    out: Set[int] = set()
    for event in events:
        if isinstance(event, Mapping) and event.get("kind") == "RedactionManifest":
            payload = event.get("payload")
            items = payload.get("redacted_events") if isinstance(payload, Mapping) else None
            for item in items if isinstance(items, list) else ():
                if isinstance(item, Mapping) and type(item.get("seq")) is int:
                    out.add(item["seq"])
    return out


def verify_run_evidence(*, events: Sequence[Mapping[str, Any]], receipts: Optional[Sequence[ActionReceipt]] = None,
                        links: Optional[Sequence[ChainLink]] = None, manifest: Optional[RunManifest] = None,
                        artifacts: Optional[Any] = None, blobs: Optional[Any] = None,
                        require_sealed: bool = True) -> EvidenceReport:
    """Verify everything that was given. Parts left out (``None``) are not checked, never assumed good."""
    findings: List[Finding] = []
    event_head: Optional[str] = None
    receipt_head: Optional[str] = None
    if not isinstance(events, (list, tuple)) or not all(isinstance(e, Mapping) for e in events):
        return EvidenceReport(findings=(Finding("EVENT_CHAIN", "malformed"),))

    # 1. the event chain: sequence, prev links, event hashes and payload identity
    try:
        event_head = verify_events(list(events))
    except ReceiptChainBroken as exc:
        findings.append(Finding("EVENT_CHAIN", str(exc.details.get("seq", ""))))
    except Exception:
        findings.append(Finding("EVENT_CHAIN", "malformed"))
    chain_ok = not findings

    # 2. the receipt chain (and the manifest's references to the receipts)
    if receipts is not None and links is not None:
        try:
            receipt_head = verify_chain(list(receipts), list(links))
        except ReceiptChainBroken as exc:
            findings.append(Finding("RECEIPT_CHAIN", str(exc.details.get("index", ""))))
    elif receipts is not None or links is not None:
        findings.append(Finding("RECEIPT_CHAIN", "incomplete"))
    if manifest is not None and receipts is not None:
        try:
            verify_manifest_receipts(manifest, receipts)
        except ReceiptChainBroken:
            findings.append(Finding("MANIFEST_RECEIPTS"))

    # 3. the stream refers to the ledger
    if chain_ok and receipts is not None:
        redacted = _redacted_seqs(events)
        stream = [(seq, e) for seq, e in enumerate(events) if e.get("kind") == "ActionReceipt"]
        if len(stream) != len(receipts):
            findings.append(Finding("RECEIPT_EVENT_COUNT", f"{len(stream)}!={len(receipts)}"))
        else:
            for (seq, event), receipt in zip(stream, receipts):
                try:
                    same = ActionReceipt.from_json(event["payload"]).canon_hash() == receipt.canon_hash()
                except Exception:
                    same = False
                if not same and seq not in redacted:
                    findings.append(Finding("RECEIPT_EVENT_MISMATCH", str(seq)))

    # 4. the seal
    sealed = [(seq, e) for seq, e in enumerate(events) if e.get("kind") == "RunSealed"]
    if not sealed:
        if require_sealed:
            findings.append(Finding("NOT_SEALED"))
    else:
        seq, event = sealed[-1]
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        if seq != len(events) - 1:
            findings.append(Finding("EVENTS_AFTER_SEAL", str(seq)))
        if links is not None and payload.get("receipt_head") != (links[-1].link_hash if links else None):
            findings.append(Finding("SEAL_RECEIPT_HEAD"))
        if manifest is not None and payload.get("manifest_hash") != manifest.canon_hash():
            findings.append(Finding("SEAL_MANIFEST_HASH"))

    # 5. evidence references and artifacts
    for seq, event in enumerate(events):
        payload = event.get("payload") if isinstance(event, Mapping) else None
        if isinstance(event, Mapping) and event.get("kind") == "Observation" and isinstance(payload, Mapping):
            for name, ref in evidence_refs_in(payload):
                digest = ref.get("content_hash") if isinstance(ref, Mapping) else None
                if not is_full_digest(digest) or ref.get("ref") != digest:
                    findings.append(Finding("BAD_EVIDENCE_REF", f"{seq}.{name}"))
                elif blobs is not None:
                    try:
                        if not blobs.exists(digest):
                            findings.append(Finding("MISSING_BLOB", f"{seq}.{name}"))
                        else:
                            blobs.get(digest)
                    except BlobCorrupt:
                        findings.append(Finding("BLOB_CONTENT", f"{seq}.{name}"))
                    except QAError:
                        findings.append(Finding("MISSING_BLOB", f"{seq}.{name}"))
    if artifacts is not None:
        if blobs is None:
            findings.append(Finding("ARTIFACTS_UNCHECKED", "no blob store"))
        else:
            for code, index in verify_artifact_manifest(artifacts, blobs):
                findings.append(Finding("ARTIFACT_" + code, str(index)))
    for where in find_truncated_digests([e.get("payload") if isinstance(e, Mapping) else None for e in events]):
        findings.append(Finding("TRUNCATED_DIGEST", where))
    return EvidenceReport(findings=tuple(findings), event_head=event_head, receipt_head=receipt_head)


def require_valid(report: EvidenceReport) -> EvidenceReport:
    """The report itself when it has no findings; otherwise ``ReceiptChainBroken`` (an integrity event) naming them."""
    if not report.ok:
        raise ReceiptChainBroken("the run's evidence does not verify: " + ", ".join(str(f) for f in report.findings[:8]),
                                 findings=[str(f) for f in report.findings])
    return report


__all__ = ["Finding", "EvidenceReport", "verify_run_evidence", "require_valid"]
