# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hash-chain verification over a run's whole evidence (receipt chain, event chain, the way they refer to each other,
artifact hashes) with tamper tests. Tampering is a finding, never an exception; a clean run has none."""
import copy
import dataclasses
import json

import pytest

from agent_harness.qa import blobs as B, canon, integrity as I, receipts as RC, records as R
from agent_harness.qa.run_errors import ReceiptChainBroken
from agent_harness.qa.vault import ArtifactVault, _strip_record_hashes

from . import _fx_evidence as H


@pytest.fixture(scope="module")
def run_result():
    env = H.evidence_env(adapter_setup=lambda a: setattr(a, "dom_meta", {"title": f"x {H.CANARY_A}", "nodes": 1}))
    r = env.run()
    assert r.outcome is R.Outcome.PASS and r.sealed
    return r


def verify(r, **kw):
    kw.setdefault("events", r.events)
    kw.setdefault("receipts", r.receipts)
    kw.setdefault("links", r.receipt_links)
    kw.setdefault("manifest", r.manifest)
    return I.verify_run_evidence(**kw)


def rechain(events, mutate):
    """Rebuild a correctly chained event list after ``mutate(index, kind, payload)`` edited (or dropped, or replaced)
    events: the tamper then lives in the content, not in a broken link."""
    chain = RC.EventChain()
    out = []
    for i, event in enumerate(events):
        got = mutate(i, event["kind"], copy.deepcopy(event["payload"]))
        if got is None:
            continue
        items = got if isinstance(got, list) else [got]
        for kind, payload in items:
            cls = R.RECORDS_BY_SCHEMA_ID.get(payload.get("schema_id")) if isinstance(payload, dict) else None
            if cls is not None:
                record = cls.from_json(_strip_record_hashes(payload))      # a consistent record: its hash follows its content
                payload, h = record.to_json(with_hash=True), record.canon_hash()
            else:
                h = canon.canon_hash(payload)
            out.append(chain.link(kind, payload, h))
    return out


def keep(i, kind, payload):
    return kind, payload


def codes(report):
    return sorted(set(report.codes))


def test_a_clean_guarded_run_verifies_end_to_end(run_result):
    report = verify(run_result)
    assert report.ok and report.findings == () and report.event_head == run_result.events[-1]["event_hash"]
    assert report.receipt_head == run_result.receipt_links[-1].link_hash
    assert I.require_valid(report) is report
    assert rechain(run_result.events, keep) == list(run_result.events)           # the helper reproduces the chain exactly


def test_parts_that_are_not_given_are_not_checked_and_never_assumed_good(run_result):
    assert I.verify_run_evidence(events=run_result.events).ok
    assert I.verify_run_evidence(events=run_result.events, manifest=run_result.manifest).ok


# --- the event chain ------------------------------------------------------------------------------------------------

def edited(events, index, **changes):
    out = [dict(e) for e in events]
    out[index] = {**out[index], **changes}
    return out


def test_every_edit_to_the_event_stream_breaks_the_chain(run_result):
    events = list(run_result.events)
    obs = next(i for i, e in enumerate(events) if e["kind"] == "Observation")
    mid = len(events) // 2
    cases = {
        "payload field changed": edited(events, obs, payload={**events[obs]["payload"], "url": "https://evil.test/"}),
        "payload hash changed": edited(events, obs, payload_hash="canon-v1:" + "0" * 64),
        "event hash changed": edited(events, mid, event_hash="canon-v1:" + "1" * 64),
        "kind changed": edited(events, mid, kind="Tampered"),
        "prev changed": edited(events, mid, prev="canon-v1:" + "2" * 64),
        "seq changed": edited(events, mid, seq=99),
        "event dropped": events[:mid] + events[mid + 1:],
        "event duplicated": events[:mid] + [events[mid]] + events[mid:],
        "two events swapped": events[:mid] + [events[mid + 1], events[mid]] + events[mid + 2:],
        "first event dropped": events[1:],
        "an event from another run spliced in": events[:mid] + list(H.evidence_env().run().events[2:3]) + events[mid:],
    }
    for name, tampered in cases.items():
        report = verify(run_result, events=tampered)
        assert "EVENT_CHAIN" in report.codes and not report.ok, name
        with pytest.raises(ReceiptChainBroken):
            I.require_valid(report)


def test_malformed_events_are_findings_not_crashes(run_result):
    for junk in ([None], [{}], [{"kind": "RunStarted"}], ["text"], [1, 2, 3]):
        report = I.verify_run_evidence(events=junk)
        assert "EVENT_CHAIN" in report.codes


def test_an_unsealed_run_and_events_after_the_seal_are_findings(run_result):
    events = list(run_result.events)
    unsealed = verify(run_result, events=events[:-1])
    assert "NOT_SEALED" in unsealed.codes and not unsealed.ok
    assert "NOT_SEALED" not in verify(run_result, events=events[:-1], require_sealed=False).codes
    extra = rechain(events, lambda i, k, p: [(k, p)] + ([("Late", {"a": 1})] if k == "RunSealed" else []))
    assert "EVENTS_AFTER_SEAL" in verify(run_result, events=extra).codes


def test_the_seal_must_name_the_ledger_head_and_the_manifest(run_result):
    def head(i, kind, payload):
        if kind == "RunSealed":
            payload["receipt_head"] = "canon-v1:" + "3" * 64
        return kind, payload

    def manifest_hash(i, kind, payload):
        if kind == "RunSealed":
            payload["manifest_hash"] = "canon-v1:" + "4" * 64
        return kind, payload
    assert codes(verify(run_result, events=rechain(run_result.events, head))) == ["SEAL_RECEIPT_HEAD"]
    assert codes(verify(run_result, events=rechain(run_result.events, manifest_hash))) == ["SEAL_MANIFEST_HASH"]
    other = dataclasses.replace(run_result.manifest, outcome=R.Outcome.INCONCLUSIVE, outcome_reason="X")
    assert "SEAL_MANIFEST_HASH" in verify(run_result, manifest=other).codes


# --- the receipt chain ---------------------------------------------------------------------------------------------

def test_every_edit_to_the_receipts_or_the_links_is_found(run_result):
    receipts, links = list(run_result.receipts), list(run_result.receipt_links)
    altered = dataclasses.replace(receipts[1], result=R.ResultCode.ACTION_TIMEOUT)
    cases = {
        "receipt altered": ([receipts[0], altered], links),
        "receipt dropped": (receipts[:1], links),
        "receipts reordered": (receipts[::-1], links),
        "link altered": (receipts, [links[0], dataclasses.replace(links[1], link_hash="canon-v1:" + "5" * 64)]),
        "link dropped": (receipts, links[:1]),
        "prev link altered": (receipts, [links[0], dataclasses.replace(links[1], prev="canon-v1:" + "6" * 64)]),
    }
    for name, (rs, ls) in cases.items():
        report = verify(run_result, receipts=rs, links=ls)
        assert "RECEIPT_CHAIN" in report.codes and not report.ok, name
    assert "RECEIPT_CHAIN" in verify(run_result, receipts=receipts, links=None).codes
    assert "RECEIPT_CHAIN" in verify(run_result, receipts=None, links=links).codes


def test_the_stream_and_the_ledger_must_agree_receipt_for_receipt(run_result):
    def tweak(i, kind, payload):
        if kind == "ActionReceipt" and payload["step_id"] == "s-save":
            payload["result"] = "ACTION_TIMEOUT"
        return kind, payload
    report = verify(run_result, events=rechain(run_result.events, tweak))
    assert codes(report) == ["RECEIPT_EVENT_MISMATCH"]
    dropped = rechain(run_result.events, lambda i, k, p: None if k == "ActionReceipt" and p["step_id"] == "s-save" else (k, p))
    assert "RECEIPT_EVENT_COUNT" in verify(run_result, events=dropped).codes
    manifest_missing = dataclasses.replace(run_result.manifest, step_results=run_result.manifest.step_results[:1])
    assert "MANIFEST_RECEIPTS" in verify(run_result, manifest=manifest_missing).codes


def test_an_event_changed_by_redaction_is_allowed_to_differ_from_the_ledger_only_when_the_manifest_says_so(run_result):
    save = next(i for i, e in enumerate(run_result.events) if e["kind"] == "ActionReceipt" and e["payload"]["step_id"] == "s-save")

    # the stored form of a redacted receipt differs from the ledger's in its identity
    def identity(i, kind, payload):
        if i == save:
            payload["recovery_rationale_marker"] = "x"
        return kind, payload
    unlisted = rechain(run_result.events, identity)
    assert codes(verify(run_result, events=unlisted)) == ["RECEIPT_EVENT_MISMATCH"]

    def listed(i, kind, payload):
        kind, payload = identity(i, kind, payload)
        if kind == "RedactionManifest":
            payload["redacted_events"] = payload["redacted_events"] + [{"kind": "ActionReceipt", "seq": save}]
        return kind, payload
    assert verify(run_result, events=rechain(run_result.events, listed)).ok


# --- evidence references, artifacts, digests ------------------------------------------------------------------------

def with_ref(ref):
    def mutate(i, kind, payload):
        if kind == "Observation" and payload["observation_id"].startswith("obs-s-save-after"):
            payload["screenshot_ref"] = ref
        return kind, payload
    return mutate


def test_evidence_references_in_observations_must_be_full_width_blob_digests(run_result):
    short = "sha256:" + "ab" * 8
    for bad in ({"ref": short, "content_hash": short}, {"ref": "shot-1"}, {"ref": "x", "content_hash": "rcv1:" + "0" * 64}):
        report = verify(run_result, events=rechain(run_result.events, with_ref(bad)))
        assert "BAD_EVIDENCE_REF" in report.codes, bad
    assert "TRUNCATED_DIGEST" in verify(run_result, events=rechain(run_result.events, with_ref({"ref": short, "content_hash": short}))).codes


def test_referenced_blobs_must_exist_and_still_hash_to_their_digest(run_result):
    sink = B.MemoryBlobSink()
    digest = sink.put(b"png bytes", "text/plain")
    ref = B.evidence_ref(digest, "text/plain").to_json()
    events = rechain(run_result.events, with_ref(ref))
    assert verify(run_result, events=events).ok                               # no blob store given: the reference is only checked for form
    assert verify(run_result, events=events, blobs=sink).ok
    assert "MISSING_BLOB" in verify(run_result, events=events, blobs=B.MemoryBlobSink()).codes
    sink._blobs[digest] = b"swapped bytes"
    assert "BLOB_CONTENT" in verify(run_result, events=events, blobs=sink).codes


def test_the_artifact_manifest_is_checked_against_the_store(run_result):
    vault = ArtifactVault(H.make_redactor(), B.MemoryBlobSink())
    vault.store("network", json.dumps({"a": 1}).encode(), "application/json")
    vault.store("console", b"log line", "text/plain")
    manifest = vault.manifest("PASS")
    assert verify(run_result, artifacts=manifest, blobs=vault).ok
    assert "ARTIFACTS_UNCHECKED" in verify(run_result, artifacts=manifest).codes
    digest = manifest.entries[0].digest
    vault._sink._blobs[digest] = b'{"a":2}'
    assert any(c.startswith("ARTIFACT_") for c in verify(run_result, artifacts=manifest, blobs=vault).codes)
    doc = manifest.to_json()
    doc["entries"][1]["size"] += 3
    assert "ARTIFACT_SIZE_MISMATCH" in verify(run_result, artifacts=doc, blobs=vault).codes


def test_a_truncated_digest_anywhere_in_the_stream_is_a_finding(run_result):
    for form in ("sha256:" + "ab" * 8, "rcv1:" + "cd" * 20, "ab" * 8):
        def mutate(i, kind, payload):
            if kind == "RunStarted":
                payload["note"] = f"ref {form}"
            return kind, payload
        report = verify(run_result, events=rechain(run_result.events, mutate))
        assert "TRUNCATED_DIGEST" in report.codes, form
    assert verify(run_result).ok                                              # no digest in a real run is cut short


def test_findings_name_where_they_were_found(run_result):
    events = list(run_result.events)
    report = verify(run_result, events=events[:3] + events[4:])
    finding = next(f for f in report.findings if f.code == "EVENT_CHAIN")
    assert finding.where == "3" and str(finding) == "EVENT_CHAIN@3"
    report = verify(run_result, receipts=run_result.receipts[::-1], links=run_result.receipt_links)
    assert next(f for f in report.findings if f.code == "RECEIPT_CHAIN").where == "0"
    with pytest.raises(ReceiptChainBroken) as err:
        I.require_valid(report)
    assert "RECEIPT_CHAIN@0" in str(err.value) and err.value.run_reason == "ASSERTION_INTEGRITY"
