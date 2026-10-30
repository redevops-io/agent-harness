# SPDX-License-Identifier: AGPL-3.0-or-later
"""H5: content-addressed blobs, full-width digests only, size bounds, retention data and the artifact manifest."""
import hashlib
import json

import pytest

from agent_harness.qa import blobs as B, records as R
from agent_harness.qa.errors import ResourceLimitExceeded
from agent_harness.qa.ports import BlobSink

HEX = hashlib.sha256(b"hello").hexdigest()
FULL = "sha256:" + HEX


# --- digests -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [HEX, "sha256:" + HEX[:16], "sha256:" + HEX[:63], "sha256:" + HEX + "0", "sha256:" + HEX.upper(),
                                 "sha512:" + HEX, "rcv1:" + HEX, "canon-v1:" + HEX, "SHA256:" + HEX, "sha256 " + HEX,
                                 "sha256:" + HEX[:-1] + "g", "sha256:", "", None, 5, b"sha256:" + HEX.encode(), "sha256:" + HEX + "\n",
                                 " sha256:" + HEX])
def test_only_a_full_lowercase_sha256_digest_is_a_valid_blob_digest(bad):
    assert not B.is_full_digest(bad)
    with pytest.raises(B.BlobRejected):
        B.require_full_digest(bad)
    sink = B.MemoryBlobSink()
    for call in (sink.get, sink.exists, sink.size):
        with pytest.raises(B.BlobRejected):
            call(bad)


def test_a_full_digest_is_valid_and_computed_from_the_bytes():
    assert B.is_full_digest(FULL) and B.require_full_digest(FULL) == FULL
    assert B.digest_of_bytes(b"hello") == FULL
    assert B.is_full_hash("rcv1:" + HEX) and B.is_full_hash("canon-v1:" + HEX) and B.is_full_hash(FULL)
    assert not B.is_full_hash("rcv1:" + HEX[:16]) and not B.is_full_hash("md5:" + HEX)


def test_the_sixteen_hex_trace_recorder_form_is_never_accepted():
    sixteen = HEX[:16]
    for form in (sixteen, "sha256:" + sixteen, "blobs/" + sixteen + ".txt"):
        assert not B.is_full_digest(form)
    with pytest.raises(B.BlobRejected):
        B.evidence_ref("sha256:" + sixteen)


def test_evidence_refs_carry_the_full_digest_as_reference_and_content_hash():
    ref = B.evidence_ref(FULL, "image/png", source="adapter", modality="screenshot")
    assert isinstance(ref, R.EvidenceRef) and ref.ref == FULL and ref.content_hash == FULL
    assert B.require_blob_ref(ref) == FULL and B.require_blob_ref(ref.to_json()) == FULL
    with pytest.raises(B.BlobRejected):
        B.require_blob_ref(R.EvidenceRef(ref=FULL, content_hash="sha256:" + HEX[:16]))
    with pytest.raises(B.BlobRejected):
        B.require_blob_ref(R.EvidenceRef(ref="sha256:" + hashlib.sha256(b"x").hexdigest(), content_hash=FULL))   # names other content
    with pytest.raises(B.BlobRejected):
        B.require_blob_ref(R.EvidenceRef(ref="blob-1"))
    with pytest.raises(B.BlobRejected):
        B.require_blob_ref("sha256:" + HEX)


def test_truncated_digest_forms_are_found_in_json_values():
    short = HEX[:16]
    value = {"ok": FULL, "also": "rcv1:" + HEX, "list": ["canon-v1:" + HEX, f"id@rcv1:{HEX}"],
             "short_prefixed": "sha256:" + short, "short_rcv": f"x rcv1:{HEX[:40]} y", "bare16": short,
             "nested": {"deep": [{"p": "blobs/" + short + ".txt"}]}, "key" + short: 1, "trace": HEX[:32], "uuid": "123e4567-e89b-12d3-a456-426614174000"}
    found = B.find_truncated_digests(value)
    assert found == ("$.bare16", "$.nested.deep[0].p", "$.short_prefixed", "$.short_rcv")
    assert not {"$.ok", "$.also", "$.list[0]", "$.list[1]", "$.trace", "$.uuid"} & set(found)
    assert B.find_truncated_digests({"a": [FULL, "rcv1:" + HEX]}) == ()
    assert B.find_truncated_digests("plain") == ()


# --- the sink -----------------------------------------------------------------------------------------------------

def test_the_sink_is_content_addressed_and_satisfies_the_port():
    sink = B.MemoryBlobSink()
    assert isinstance(sink, BlobSink)
    digest = sink.put(b"hello", "text/plain")
    assert digest == FULL and sink.exists(digest) and sink.size(digest) == 5 and sink.get(digest) == b"hello"
    assert sink.media_type(digest) == "text/plain"
    assert sink.put(b"hello", "text/plain") == digest and len(sink) == 1 and sink.total_bytes == 5      # one copy, counted once
    assert sink.put(bytearray(b"hello"), "text/plain") == digest
    other = sink.put(b"", "application/json")
    assert other == "sha256:" + hashlib.sha256(b"").hexdigest() and sink.get(other) == b"" and sink.digests() == tuple(sorted([digest, other]))
    assert not sink.exists("sha256:" + "0" * 64)
    for call in (sink.get, sink.size, sink.media_type):
        with pytest.raises(B.BlobNotFound):
            call("sha256:" + "0" * 64)


def test_the_sink_computes_the_digest_itself_and_checks_what_it_returns():
    sink = B.MemoryBlobSink()
    digest = sink.put(b"payload", "text/plain")
    sink._blobs[digest] = b"tampered"
    with pytest.raises(B.BlobCorrupt):
        sink.get(digest)
    assert B.BlobCorrupt.run_reason == "ASSERTION_INTEGRITY"


@pytest.mark.parametrize("data,media", [("text", "text/plain"), (None, "text/plain"), (b"x", ""), (b"x", None), (b"x", "plain"),
                                        (b"x", "text/"), (b"x", "/plain"), (b"x", "text/plain\n"), (b"x", "t" * 300 + "/x"), (b"x", 7)])
def test_the_sink_refuses_malformed_input(data, media):
    with pytest.raises(B.BlobRejected):
        B.MemoryBlobSink().put(data, media)


def test_the_artifact_bound_applies_per_artifact_and_per_media_type():
    bounds = B.EvidenceBounds(max_artifact_bytes=100, per_media_type={"application/zip": 500, "image/png": 10})
    sink = B.MemoryBlobSink(bounds)
    sink.put(b"a" * 100, "text/plain")
    with pytest.raises(ResourceLimitExceeded) as err:
        sink.put(b"b" * 101, "text/plain")
    assert err.value.details["bound"] == "artifact"
    sink.put(b"z" * 500, "application/zip")
    with pytest.raises(ResourceLimitExceeded):
        sink.put(b"z" * 501, "application/zip")
    with pytest.raises(ResourceLimitExceeded):
        sink.put(b"p" * 11, "image/png")
    sink.put(b"p" * 10, "IMAGE/PNG; x=y")                                  # matched by base type, case-insensitively


def test_the_run_bound_and_the_count_bound_stop_further_storage_and_store_nothing():
    sink = B.MemoryBlobSink(B.EvidenceBounds(max_artifact_bytes=100, max_run_bytes=150, max_artifacts=3))
    sink.put(b"a" * 100, "text/plain")
    with pytest.raises(ResourceLimitExceeded) as err:
        sink.put(b"b" * 51, "text/plain")
    assert err.value.details["bound"] == "run" and len(sink) == 1 and sink.total_bytes == 100
    sink.put(b"c" * 50, "text/plain")
    assert sink.total_bytes == 150
    count = B.MemoryBlobSink(B.EvidenceBounds(max_artifacts=2))
    count.put(b"1", "text/plain")
    count.put(b"2", "text/plain")
    with pytest.raises(ResourceLimitExceeded) as err:
        count.put(b"3", "text/plain")
    assert err.value.details["bound"] == "count" and len(count) == 2
    assert count.put(b"1", "text/plain")                                    # a duplicate is not a new artifact


def test_default_bounds_are_the_plans_proposals_and_bad_bounds_are_refused():
    b = B.EvidenceBounds()
    assert (b.max_artifact_bytes, b.max_run_bytes) == (5 * 1024 * 1024, 50 * 1024 * 1024)
    for kw in ({"max_artifact_bytes": 0}, {"max_run_bytes": -1}, {"max_artifacts": 1.5}, {"per_media_type": {"png": 5}},
               {"per_media_type": {"image/png": 0}}, {"per_media_type": []}, {"max_events": True} if False else {"max_events": 0}):
        with pytest.raises(Exception):
            B.EvidenceBounds(**kw)
    with pytest.raises(Exception):
        B.MemoryBlobSink("bounds")


# --- retention (data only) ---------------------------------------------------------------------------------------

def test_retention_is_fourteen_days_for_a_pass_and_sixty_for_everything_else_by_default():
    p = B.RetentionPolicy()
    assert (p.passing_days, p.failing_days) == (14, 60)
    assert (p.retention_class("PASS"), p.days_for("PASS")) == ("PASSING", 14)
    for outcome in ("PRODUCT_FAIL", "TEST_DEFECT", "ENVIRONMENT_FAIL", "INCONCLUSIVE", "NEW_FUTURE_OUTCOME", None, "pass", ""):
        assert (p.retention_class(outcome), p.days_for(outcome)) == ("FAILING", 60), outcome


def test_retention_is_configurable_and_validated_and_not_enforced_here():
    p = B.RetentionPolicy(passing_days=7, failing_days=30)
    assert p.days_for("PASS") == 7 and p.days_for("PRODUCT_FAIL") == 30 and p.to_json() == {"failing_days": 30, "passing_days": 7}
    for kw in ({"passing_days": 0}, {"failing_days": -5}, {"passing_days": 10 ** 6}, {"passing_days": 1.5}, {"failing_days": True}):
        with pytest.raises(Exception):
            B.RetentionPolicy(**kw)
    sink = B.MemoryBlobSink()
    assert not [m for m in dir(sink) if m.startswith(("delete", "expire", "evict", "purge", "remove"))]      # deletion is Q2


# --- the artifact manifest ---------------------------------------------------------------------------------------

def make_manifest(sink, items=((b"one", "text/plain"), (b'{"a":1}', "application/json"))):
    entries = []
    for i, (data, media) in enumerate(items):
        d = sink.put(data, media)
        entries.append(B.ArtifactEntry(kind="note", digest=d, size=len(data), media_type=media, step_id=f"s{i}"))
    return B.ArtifactManifest(entries=tuple(entries), retention_class="PASSING", retention_days=14, redaction_policy_id="rcv1:" + "0" * 64)


def test_the_manifest_hashes_the_artifacts_and_the_retention_data():
    sink = B.MemoryBlobSink()
    m = make_manifest(sink)
    doc = m.to_json()
    assert doc["form"] == B.MANIFEST_FORM and doc["total_bytes"] == 10 and doc["retention_days"] == 14
    assert m.manifest_hash.startswith("canon-v1:") and len(m.manifest_hash) == 73
    assert make_manifest(B.MemoryBlobSink()).manifest_hash == m.manifest_hash
    other = make_manifest(B.MemoryBlobSink(), ((b"one", "text/plain"), (b"changed", "application/json")))
    assert other.manifest_hash != m.manifest_hash
    reordered = B.ArtifactManifest(entries=tuple(reversed(m.entries)))
    assert reordered.manifest_hash != m.manifest_hash


def test_a_manifest_entry_refuses_a_short_digest_a_bad_kind_and_a_short_redaction_hash():
    with pytest.raises(B.BlobRejected):
        B.ArtifactEntry(kind="note", digest="sha256:" + HEX[:16], size=1, media_type="text/plain")
    with pytest.raises(B.BlobRejected):
        B.ArtifactEntry(kind="Bad Kind", digest=FULL, size=1, media_type="text/plain")
    with pytest.raises(B.BlobRejected):
        B.ArtifactEntry(kind="note", digest=FULL, size=-1, media_type="text/plain")
    with pytest.raises(B.BlobRejected):
        B.ArtifactEntry(kind="note", digest=FULL, size=1, media_type="nope")
    with pytest.raises(B.BlobRejected):
        B.ArtifactEntry(kind="note", digest=FULL, size=1, media_type="text/plain", redaction_manifest="canon-v1:" + HEX[:16])


def test_a_verified_manifest_has_no_findings_and_every_tamper_is_a_finding():
    sink = B.MemoryBlobSink()
    m = make_manifest(sink)
    assert B.verify_artifact_manifest(m, sink) == () and B.verify_artifact_manifest(m.to_json(), sink) == ()
    d0 = m.entries[0].digest
    # content changed under the same address
    sink._blobs[d0] = b"altered"
    assert B.verify_artifact_manifest(m, sink) == (("SIZE_MISMATCH", 0),)
    sink._blobs[d0] = b"ONE"                                                     # same size, different bytes
    assert B.verify_artifact_manifest(m, sink) == (("CONTENT_MISMATCH", 0),)
    sink._blobs[d0] = b"one"
    assert B.verify_artifact_manifest(m, sink) == ()
    # blob removed
    del sink._blobs[d0]
    assert B.verify_artifact_manifest(m, sink) == (("MISSING_BLOB", 0),)
    # entry edited in the manifest document
    sink2 = B.MemoryBlobSink()
    m2 = make_manifest(sink2)
    doc = json.loads(json.dumps(m2.to_json()))
    doc["entries"][1]["size"] += 1
    assert B.verify_artifact_manifest(doc, sink2) == (("SIZE_MISMATCH", 1),)
    doc = json.loads(json.dumps(m2.to_json()))
    doc["entries"][0]["digest"] = "sha256:" + HEX[:16]
    assert B.verify_artifact_manifest(doc, sink2) == (("BAD_DIGEST", 0),)
    doc = json.loads(json.dumps(m2.to_json()))
    doc["entries"][0]["digest"] = "sha256:" + hashlib.sha256(b"never stored").hexdigest()
    assert B.verify_artifact_manifest(doc, sink2) == (("MISSING_BLOB", 0),)
    for junk in (None, {}, {"form": "other"}, {"form": B.MANIFEST_FORM}, {"form": B.MANIFEST_FORM, "entries": "x"}, []):
        assert B.verify_artifact_manifest(junk, sink2) == (("MANIFEST_MALFORMED", -1),)
    doc = json.loads(json.dumps(m2.to_json()))
    doc["entries"].append("not an entry")
    assert B.verify_artifact_manifest(doc, sink2) == (("BAD_DIGEST", 2),)
