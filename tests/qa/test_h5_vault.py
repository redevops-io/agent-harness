# SPDX-License-Identifier: AGPL-3.0-or-later
"""H5: redact-before-store. The vault is a BlobSink that cannot store raw bytes; a redaction failure drops the artifact and
leaves a marker without content; bounds apply to raw input, redacted output and the run; the evidence guard redacts, bounds
and checks every event."""
import json

import pytest

from agent_harness.qa import blobs as B, redaction as RD, records as R
from agent_harness.qa.errors import RedactionFailed, ResourceLimitExceeded
from agent_harness.qa.ports import BlobSink
from agent_harness.qa.vault import ArtifactVault, EvidenceGuard, SEAL_KINDS, STRUCTURAL_KEYS

from . import _h5 as H


def make_vault(**kw):
    sink = kw.pop("sink", None)
    if sink is None:
        sink = B.MemoryBlobSink(kw.get("bounds"))
    return ArtifactVault(kw.pop("redactor", None) or H.make_redactor(), sink, **kw), sink


def stored_text(sink):
    return b"".join(sink.get(d) for d in sink.digests()).decode("utf-8", "replace")


# --- the vault ------------------------------------------------------------------------------------------------------

def test_the_vault_stores_only_redacted_bytes_and_returns_a_full_width_reference():
    vault, sink = make_vault()
    body = json.dumps({"headers": {"Authorization": f"Bearer {H.JWT}"}, "console": [f"log {H.CANARY_A}"], "url": H.BASIC_URL})
    art = vault.store("network", body.encode(), "application/json", step_id="s-1")
    stored = stored_text(sink)
    assert not H.no_secret_survives(stored) and H.CANARY_A not in stored and "alice" not in stored
    assert art.entry.digest == B.digest_of_bytes(sink.get(art.entry.digest)) and art.ref.content_hash == art.entry.digest
    assert B.is_full_digest(art.entry.digest) and art.entry.size == len(sink.get(art.entry.digest))
    assert art.redaction.total >= 3 and art.entry.redaction_manifest == art.redaction.manifest_hash
    manifest_text = json.dumps(vault.manifest("PASS").to_json())
    assert not H.no_secret_survives(manifest_text) and "alice" not in manifest_text


def test_the_vault_is_a_blob_sink_whose_put_redacts_first():
    vault, sink = make_vault()
    assert isinstance(vault, BlobSink)
    digest = vault.put(f"token {H.GITHUB_TOKEN}".encode(), "text/plain")
    assert vault.exists(digest) and vault.size(digest) == len(vault.get(digest))
    assert H.GITHUB_TOKEN.encode() not in vault.get(digest) and digest == B.digest_of_bytes(vault.get(digest))
    assert H.GITHUB_TOKEN.encode() not in b"".join(sink.get(d) for d in sink.digests())
    with pytest.raises(RedactionFailed):
        vault.put(b"\x89PNG raw screenshot bytes", "image/png")
    assert len(sink) == 1


def test_a_redaction_failure_stores_nothing_and_leaves_a_marker_with_no_content():
    vault, sink = make_vault()
    cases = [(b"\xff\xfe broken " + H.AWS_KEY_ID.encode(), "text/plain", "NOT_UTF8"),
             (H.AWS_KEY_ID.encode(), "image/png", "UNSUPPORTED_MEDIA_TYPE"),
             (b"x" * (RD.DEFAULT_MAX_INPUT_BYTES + 1), "text/plain", None)]
    for data, media, reason in cases[:2]:
        with pytest.raises(RedactionFailed):
            vault.store("capture", data, media)
    assert len(sink) == 0 and sink.total_bytes == 0 and vault.manifest().entries == ()
    assert [f.reason for f in vault.failures] == ["NOT_UTF8", "UNSUPPORTED_MEDIA_TYPE"]
    assert all(f.code == "REDACTION_FAILED" and f.kind == "capture" for f in vault.failures)
    text = json.dumps([f.to_json() for f in vault.failures])
    assert H.AWS_KEY_ID not in text


def test_a_redactor_that_raises_or_returns_junk_is_a_failure_not_a_raw_store():
    class Raising:
        policy_id = "rcv1:" + "1" * 64

        def redact(self, data, media_type):
            raise ValueError("secret " + H.AWS_KEY_ID)

    class Junk:
        policy_id = "rcv1:" + "2" * 64

        def redact(self, data, media_type):
            return "not bytes"

    class Passthrough:                    # the dangerous one: returns its input
        policy_id = "rcv1:" + "3" * 64
        calls = 0

        def redact(self, data, media_type):
            Passthrough.calls += 1
            return data if Passthrough.calls % 2 else data + b"!"       # and is not stable under a second pass

    for redactor, reason in ((Raising(), "INTERNAL"), (Junk(), "BAD_OUTPUT"), (Passthrough(), "UNSTABLE")):
        vault, sink = make_vault(redactor=redactor)
        with pytest.raises(RedactionFailed) as err:
            vault.store("note", b"raw secret bytes", "text/plain")
        assert err.value.details["reason"] == reason and H.AWS_KEY_ID not in str(err.value)
        assert len(sink) == 0 and vault.failures[0].reason == reason


def test_a_stable_custom_redactor_is_accepted_and_gets_an_opaque_manifest():
    class Upper:
        policy_id = "rcv1:" + "4" * 64

        def redact(self, data, media_type):
            return b"[X]"

    vault, sink = make_vault(redactor=Upper())
    art = vault.store("note", b"anything", "text/plain")
    assert sink.get(art.entry.digest) == b"[X]" and art.redaction.counts == ()


def test_the_artifact_bound_applies_to_the_raw_input_and_to_the_redacted_output():
    vault, sink = make_vault(bounds=B.EvidenceBounds(max_artifact_bytes=100))
    with pytest.raises(ResourceLimitExceeded):
        vault.store("note", b"z" * 101, "text/plain")
    assert len(sink) == 0 and vault.failures[0].code == "RESOURCE_LIMIT_EXCEEDED"
    # the redacted form can be larger than the input (a short secret becomes a longer token)
    short = "pwd=" + "k" * 4
    inflating = (short + " ") * 6
    big = ArtifactVault(H.make_redactor(), B.MemoryBlobSink(), bounds=B.EvidenceBounds(max_artifact_bytes=len(inflating)))
    with pytest.raises(ResourceLimitExceeded):
        big.store("note", inflating.encode(), "text/plain")


def test_the_run_bound_and_the_artifact_count_bound_apply_and_store_nothing_more():
    vault, sink = make_vault(bounds=B.EvidenceBounds(max_artifact_bytes=100, max_run_bytes=120, max_artifacts=2))
    vault.store("a", b"y" * 60, "text/plain")
    with pytest.raises(ResourceLimitExceeded) as err:
        vault.store("b", b"x" * 80, "text/plain")
    assert err.value.details["bound"] == "run" and len(sink) == 1
    vault.store("c", b"w" * 50, "text/plain")
    with pytest.raises(ResourceLimitExceeded) as err:
        vault.store("d", b"v" * 1, "text/plain")
    assert err.value.details["bound"] == "count" and len(sink) == 2 and len(vault.failures) == 2


def test_a_sink_that_lies_about_the_digest_is_refused():
    class Lying(B.MemoryBlobSink):
        def put(self, data, media_type):
            super().put(data, media_type)
            return "sha256:" + "0" * 64

    vault, _ = make_vault(sink=Lying())
    with pytest.raises(B.BlobRejected):
        vault.store("note", b"fine", "text/plain")


def test_the_manifest_carries_the_retention_class_and_days_for_the_outcome_and_the_policy_id():
    vault, _ = make_vault(retention=B.RetentionPolicy(passing_days=3, failing_days=9))
    vault.store("note", b"fine", "text/plain")
    for outcome, cls, days in (("PASS", "PASSING", 3), ("PRODUCT_FAIL", "FAILING", 9), (None, "FAILING", 9)):
        m = vault.manifest(outcome)
        assert (m.retention_class, m.retention_days) == (cls, days) and m.redaction_policy_id == vault.policy_id
    assert B.verify_artifact_manifest(vault.manifest("PASS"), vault) == ()


def test_the_vault_validates_what_it_is_given():
    with pytest.raises(Exception):
        ArtifactVault(object(), B.MemoryBlobSink())
    with pytest.raises(Exception):
        ArtifactVault(H.make_redactor(), object())
    class NoPolicy:
        def redact(self, d, m):
            return d
    with pytest.raises(Exception):
        ArtifactVault(NoPolicy(), B.MemoryBlobSink())
    vault, _ = make_vault()
    for bad in ("text", None, 5):
        with pytest.raises(B.BlobRejected):
            vault.store("note", bad, "text/plain")


# --- the evidence guard --------------------------------------------------------------------------------------------

def make_guard(**kw):
    return EvidenceGuard(H.make_redactor(), **kw)


def apply(guard, kind, payload, seq=0, record=None):
    return guard.apply(kind, record, payload, "canon-v1:" + "0" * 64, seq=seq)


def test_the_guard_redacts_a_dictionary_payload_and_rehashes_it():
    guard = make_guard()
    payload = {"note": f"seen {H.AWS_KEY_ID}", "n": 1, "password": "x", "list": [H.CANARY_A]}
    out, h = apply(guard, "Custom", payload)
    assert not H.no_secret_survives(json.dumps(out)) and out["n"] == 1 and out["password"] == "[REDACTED:key_name]"
    from agent_harness.qa import canon
    assert h == canon.canon_hash(out)
    assert guard.redacted_events == ((0, "Custom"),)
    clean, h2 = apply(guard, "Custom", {"a": 1}, seq=1)
    assert clean == {"a": 1} and h2 == "canon-v1:" + "0" * 64          # an unchanged payload keeps its hash


def test_the_guard_rebuilds_a_record_from_its_redacted_form_so_its_hash_stays_true():
    guard = make_guard()
    obs = R.Observation(observation_id="o-1", step_id="s-1", phase=R.Phase.BEFORE,
                        source=R.ObservationSource(adapter="a", adapter_version="1", browser="chromium", browser_version="1"),
                        redaction_applied=True, redaction_policy_id=guard.policy_id, url=f"https://x.test/p?token={H.TOK_C}&n={H.EMAIL}",
                        dom_meta={"title": f"Hello {H.CANARY_A}", "password": "p4ss", "nodes": 3})
    out, h = guard.apply("Observation", obs, obs.to_json(with_hash=True), obs.canon_hash(), seq=4)
    rebuilt = R.Observation.from_json(out)                              # parses, and its stored content_hash is checked
    assert rebuilt.canon_hash() == h != obs.canon_hash()
    assert H.TOK_C not in rebuilt.url and H.EMAIL not in rebuilt.url and rebuilt.dom_meta["password"] == "[REDACTED:key_name]"
    assert rebuilt.dom_meta["nodes"] == 3 and H.CANARY_A not in json.dumps(out)
    assert guard.redacted_events == ((4, "Observation"),)


def test_a_record_that_no_longer_parses_after_redaction_fails_closed():
    guard = make_guard()
    step = R.Postcondition(declared="x", result=R.PostconditionResult.NOT_EVALUATED)

    class Fake:
        def canon_hash(self):
            return "canon-v1:" + "0" * 64

        @classmethod
        def from_json(cls, data):
            raise ValueError("invalid")
    with pytest.raises(RedactionFailed) as err:
        guard.apply("Observation", Fake(), {"x": f"secret {H.AWS_KEY_ID}"}, "canon-v1:" + "0" * 64, seq=0)
    assert err.value.details["reason"] == "RECORD_INVALID"


def test_structural_keys_are_not_redacted_but_the_rest_of_the_payload_is():
    guard = make_guard()
    out, _ = apply(guard, "X", {"action": {"secret_ref": {"provider": "vault", "key": "kv/x"}}, "note": H.AWS_KEY_ID})
    assert out["action"]["secret_ref"] == {"provider": "vault", "key": "kv/x"} and H.AWS_KEY_ID not in json.dumps(out)
    assert "secretref" in STRUCTURAL_KEYS


def test_references_in_an_observation_must_be_full_width_blob_digests():
    guard = make_guard()
    good = B.evidence_ref(B.digest_of_bytes(b"x"), "image/png").to_json()
    apply(guard, "Observation", {"screenshot_ref": good})
    for bad in ({"ref": "sha256:" + "ab" * 8, "content_hash": "sha256:" + "ab" * 8}, {"ref": "x", "content_hash": "rcv1:" + "0" * 64},
                {"ref": "blob-1"}, {"ref": good["ref"], "content_hash": good["content_hash"].upper()}, "text"):
        with pytest.raises(B.BlobRejected):
            apply(guard, "Observation", {"a11y_snapshot_ref": bad})
    apply(guard, "Other", {"screenshot_ref": {"ref": "x"}})              # only an Observation's references are checked


def test_event_size_event_count_and_total_bounds():
    guard = make_guard(bounds=B.EvidenceBounds(max_event_bytes=200, max_events=3, max_events_bytes=300, seal_reserve_bytes=400))
    with pytest.raises(ResourceLimitExceeded) as err:
        apply(guard, "Big", {"a": "x" * 300})
    assert err.value.details["bound"] == "event"
    apply(guard, "A", {"a": "x" * 100})
    apply(guard, "A", {"a": "x" * 100})
    with pytest.raises(ResourceLimitExceeded) as err:
        apply(guard, "A", {"a": "x" * 120})                              # 300 total
    assert err.value.details["bound"] == "events_bytes"
    apply(guard, "A", {"a": "y"})
    with pytest.raises(ResourceLimitExceeded) as err:
        apply(guard, "A", {"a": "z"})                                    # a fourth event
    assert err.value.details["bound"] == "events"


def test_the_closing_events_still_fit_after_the_run_bound_was_hit():
    guard = make_guard(bounds=B.EvidenceBounds(max_event_bytes=1000, max_events=1, max_events_bytes=100, seal_reserve_bytes=400))
    apply(guard, "A", {"a": "x" * 50})
    with pytest.raises(ResourceLimitExceeded):
        apply(guard, "A", {"a": "x"})
    for kind in sorted(SEAL_KINDS):
        apply(guard, kind, {"a": "x" * 10})
    assert {"RunSealed", "RedactionManifest", "EvidenceFailure"} <= SEAL_KINDS
    with pytest.raises(ResourceLimitExceeded) as err:
        apply(guard, "RunSealed", {"a": "x" * 380})                      # the reserve is bounded too
    assert err.value.details["bound"] == "seal"
    roomy = make_guard(bounds=B.EvidenceBounds(max_event_bytes=50, seal_reserve_bytes=400))
    apply(roomy, "RunSealed", {"a": "x" * 200})                          # a closing event is not held to the per-event bound


def test_a_redaction_failure_inside_the_guard_surfaces_as_redaction_failed():
    guard = make_guard()
    with pytest.raises(RedactionFailed):
        apply(guard, "X", {"a": object()})
    with pytest.raises(RedactionFailed):
        apply(guard, "X", {1: "non-string key"})


def test_the_failure_and_manifest_payloads_hold_no_content():
    guard = make_guard()
    apply(guard, "X", {"a": H.AWS_KEY_ID, "b": H.EMAIL}, seq=0)
    apply(guard, "Y", {"c": "clean"}, seq=1)
    err = RedactionFailed("msg " + H.AWS_KEY_ID, reason="NOT_UTF8")
    failure = guard.failure_payload("Observation", err)
    assert failure == {"code": "REDACTION_FAILED", "form": "qa-evidence-failure/1", "kind": "Observation", "reason": "NOT_UTF8"}
    assert guard.failure_payload("bad kind!\n" + H.AWS_KEY_ID, RedactionFailed("x", reason="lower case"))["kind"] == "Unknown"
    assert guard.failure_payload("K", RedactionFailed("x", reason="lower case"))["reason"] == "UNSPECIFIED"
    doc = guard.manifest_payload()
    assert doc["redacted_events"] == [{"kind": "X", "seq": 0}] and doc["events"] == 2 and doc["policy_id"] == guard.policy_id
    assert doc["tally"]["counts"] == {"aws_key_id": 1, "pii_email": 1} and doc["tally"]["items_total"] == 2
    assert not H.no_secret_survives(json.dumps([failure, doc])) and H.EMAIL not in json.dumps(doc)


def test_the_guard_validates_its_redactor():
    with pytest.raises(Exception):
        EvidenceGuard(object())
    class OnlyRedact:
        policy_id = "rcv1:" + "0" * 64
        def redact(self, d, m):
            return d
    with pytest.raises(Exception):
        EvidenceGuard(OnlyRedact())
