# SPDX-License-Identifier: AGPL-3.0-or-later
"""The runner with the evidence guard and the egress guard wired in. A run's evidence stream holds only redacted
material; every egress decision, allow or deny, is a hashed event that names the receipt it belongs to; a blocked
navigation never reaches the adapter; the size bounds end a run without losing the closing events; a run without the
guards behaves exactly as before."""
import json

import pytest

from agent_harness.qa import blobs as B, canon, egress as E, receipts as RC, records as R, redaction as RD
from agent_harness.qa.errors import RedactionFailed, SpecInvalid
from agent_harness.qa.policy import DefaultActionPolicy
from agent_harness.qa.vault import SEAL_KINDS, EvidenceGuard

from . import _fx_evidence as H
from ._fx_runner import APP, UrlEndsPostcondition, default_policy, make_config, make_env, make_spec, nav_step, operations, save_step
from .test_runner import check_invariants


def run(env):
    return check_invariants(env, env.run())


def cls(r):
    return r.classification.outcome.value, r.classification.reason


EVIDENCE_KINDS = {"EgressDecision", "EgressSummary", "RedactionManifest", "EvidenceFailure"}


def poisoned_dom(adapter):
    adapter.dom_meta = {"title": f"Order {H.CANARY_A} for {H.EMAIL}",
                        "headers": {"Authorization": f"Bearer {H.JWT}", "Cookie": f"sid={H.SESS_A}"},
                        "console": [f"error: {H.CANARY_B} at {H.BASIC_URL}", f"key {H.AWS_KEY_ID}"],
                        "password": H.PW_A, "nodes": 4}


# --- a clean run is unchanged -------------------------------------------------------------------------------------

def test_without_the_guards_nothing_changes_and_no_evidence_event_appears():
    r = run(make_env())
    assert not EVIDENCE_KINDS & {e["kind"] for e in r.events}


def test_a_clean_run_with_both_guards_passes_and_its_evidence_is_the_same_evidence_plus_the_evidence_events():
    env = H.evidence_env()
    plain = run(make_env(config=make_config(egress_policy_id=env.egress.policy_id, redaction_policy_id=env.guard.policy_id)))
    guarded = run(env)
    assert cls(guarded) == cls(plain) == ("PASS", None)
    assert env.guard.redacted_events == ()                                    # nothing to redact: the guard changed nothing
    def core(r):
        return [(e["kind"], e["payload_hash"], e["payload"]) for e in r.events if e["kind"] not in EVIDENCE_KINDS]
    assert core(guarded) == core(plain)
    assert guarded.manifest.egress_policy_id == env.egress.policy_id and guarded.manifest.redaction_policy_id == env.guard.policy_id
    kinds = H.kinds(guarded)
    assert kinds[-3:] == ["EgressSummary", "RedactionManifest", "RunSealed"] and kinds.count("EgressDecision") >= 3
    assert guarded.sealed and RC.verify_events(list(guarded.events)) is not None


def test_a_guarded_run_is_reproducible():
    a, b = run(H.evidence_env()), run(H.evidence_env())
    assert [e["event_hash"] for e in a.events] == [e["event_hash"] for e in b.events]
    assert a.manifest.canon_hash() == b.manifest.canon_hash()


def test_the_runner_refuses_guards_whose_policy_ids_are_not_the_configured_ones():
    guard = EvidenceGuard(H.make_redactor())
    egress, _ = H.make_guard(H.app_policy(), H.APP_TABLE)
    with pytest.raises(SpecInvalid):
        make_env(evidence_guard=guard)                                        # the default config names other policy ids
    with pytest.raises(SpecInvalid):
        make_env(egress=egress)
    good = make_config(egress_policy_id=egress.policy_id, redaction_policy_id=guard.policy_id)
    make_env(config=good, evidence_guard=guard, egress=egress)
    for bad in ("guard", object(), H.make_redactor()):
        with pytest.raises(SpecInvalid):
            make_env(config=good, evidence_guard=bad, egress=egress)
        with pytest.raises(SpecInvalid):
            make_env(config=good, evidence_guard=guard, egress=bad)


# --- redaction before storage ------------------------------------------------------------------------------------

def test_a_canary_in_a_page_a_header_a_url_and_a_console_message_is_in_no_stored_event():
    spec = make_spec([nav_step(url=f"{APP}/form?token={H.TOK_B}&email={H.EMAIL}&code={H.CODE_A}"), save_step()])
    env = H.evidence_env(spec=spec, adapter_setup=poisoned_dom)
    r = run(env)
    assert r.sealed
    for text in (H.stream_text(r), json.dumps(list(env.stream.events), sort_keys=True, default=str)):
        assert not H.no_secret_survives(text), H.no_secret_survives(text)
        for leaked in (H.EMAIL, H.TOK_B, H.PW_A, H.SESS_A, "alice", H.CODE_A):
            assert leaked not in text, leaked
    assert "[REDACTED:" in H.stream_text(r)
    # what the adapter saw is not what was stored: the redaction happened in the runner, not in the page
    assert H.CANARY_A in json.dumps(env.adapter.dom_meta)
    assert RC.verify_events(list(r.events)) is not None and list(env.stream.events) == list(r.events)


def test_the_redaction_manifest_event_counts_what_was_removed_and_names_the_changed_events():
    env = H.evidence_env(adapter_setup=poisoned_dom)
    r = run(env)
    (manifest,) = H.events_of(r, "RedactionManifest")
    doc = manifest["payload"]
    assert doc["policy_id"] == env.guard.policy_id == r.manifest.redaction_policy_id
    counts = doc["tally"]["counts"]
    assert counts["denylist"] >= 2 and counts["pii_email"] >= 1 and counts["key_name"] >= 2 and counts["aws_key_id"] >= 1
    changed = {(x["seq"], x["kind"]) for x in doc["redacted_events"]}
    assert changed and all(r.events[seq]["kind"] == kind == "Observation" for seq, kind in changed)
    assert doc["events"] == len([e for e in r.events if e["kind"] not in SEAL_KINDS])
    assert not H.no_secret_survives(json.dumps(doc)) and H.EMAIL not in json.dumps(doc)
    assert r.events.index(manifest) == len(r.events) - 2                      # just before RunSealed


def test_redacted_observations_are_still_valid_records_and_verify():
    env = H.evidence_env(adapter_setup=poisoned_dom)
    r = run(env)
    for event in H.events_of(r, "Observation"):
        obs = R.Observation.from_json(event["payload"])
        assert obs.canon_hash() == event["payload_hash"] and obs.redaction_applied
        assert obs.dom_meta["password"] == "[REDACTED:key_name]" and type(obs.dom_meta["nodes"]) is int


def test_a_redaction_failure_drops_the_event_ends_the_run_inconclusive_and_records_a_marker_without_content():
    class Poisoned(RD.PatternRedactor):
        def redact_structure(self, value, *, exempt_keys=()):
            if "POISON" in json.dumps(value, default=str):
                raise RedactionFailed("cannot redact " + "POISON", reason="NOT_UTF8")
            return super().redact_structure(value, exempt_keys=exempt_keys)

    def setup(adapter):
        adapter.dom_meta = {"title": "POISON"}
    env = H.evidence_env(redactor=Poisoned(RD.RedactionConfig(denylist=H.CANARIES)), adapter_setup=setup)
    r = env.run()
    assert cls(r) == ("INCONCLUSIVE", "REDACTION_FAILED") and r.exit_code != 0 and r.review_item is not None and r.sealed
    (failure,) = H.events_of(r, "EvidenceFailure")
    assert failure["payload"] == {"code": "REDACTION_FAILED", "form": "qa-evidence-failure/1", "kind": "Observation",
                                  "reason": "NOT_UTF8"}
    assert "POISON" not in H.stream_text(r) and "POISON" not in json.dumps(list(env.stream.events), default=str)
    assert H.kinds(r).count("Observation") == 0 or all("POISON" not in json.dumps(e["payload"]) for e in H.events_of(r, "Observation"))
    assert RC.verify_events(list(r.events)) is not None
    assert not r.manifest.assertions_executed                                  # nothing was certified on unrecorded evidence


def test_a_run_never_passes_when_the_evidence_it_rests_on_could_not_be_stored_redacted():
    class Failing(RD.PatternRedactor):
        def redact_structure(self, value, *, exempt_keys=()):
            raise RedactionFailed("always", reason="INTERNAL")
    env = H.evidence_env(redactor=Failing(RD.RedactionConfig()))
    r = env.run()
    assert r.outcome is not R.Outcome.PASS and r.exit_code != 0 and not r.sealed
    assert operations(env.adapter, "open_session") == [] and operations(env.adapter, "act") == []     # RunStarted could not be recorded
    assert H.kinds(r) == ["EvidenceFailure", "EvidenceFailure"] and "RunStarted" not in H.kinds(r)


# --- size bounds -------------------------------------------------------------------------------------------------

def test_the_event_count_bound_ends_the_run_as_an_environment_failure_and_the_run_is_still_sealed():
    env = H.evidence_env(bounds=B.EvidenceBounds(max_events=6))
    r = env.run()
    assert cls(r) == ("ENVIRONMENT_FAIL", "RESOURCE_LIMIT_EXCEEDED") and r.exit_code != 0 and r.sealed
    kinds = H.kinds(r)
    assert kinds[-1] == "RunSealed" and "EvidenceFailure" in kinds
    failure = H.events_of(r, "EvidenceFailure")[0]["payload"]
    assert failure["code"] == "RESOURCE_LIMIT_EXCEEDED" and failure["bound"] == "events"
    assert RC.verify_events(list(r.events)) is not None


def test_the_event_size_and_total_size_bounds_end_the_run_too():
    for bounds, bound in ((B.EvidenceBounds(max_event_bytes=350), "event"), (B.EvidenceBounds(max_events_bytes=900), "events_bytes")):
        env = H.evidence_env(bounds=bounds)
        r = env.run()
        assert cls(r)[0] == "ENVIRONMENT_FAIL" and r.sealed and H.kinds(r)[-1] == "RunSealed", (bound, cls(r))
        assert H.events_of(r, "EvidenceFailure")[0]["payload"]["bound"] == bound


def test_a_run_inside_the_bounds_is_unaffected():
    env = H.evidence_env(bounds=B.EvidenceBounds(max_event_bytes=64 * 1024, max_events=1000, max_events_bytes=10 * 1024 * 1024))
    assert cls(run(env)) == ("PASS", None)


# --- egress -----------------------------------------------------------------------------------------------------

def egress_events(r):
    return [(e["payload"]["decision"], e["payload"]) for e in H.events_of(r, "EgressDecision")]


def test_every_egress_decision_is_a_hashed_event_with_the_receipt_it_belongs_to():
    r = run(H.evidence_env())
    decisions = egress_events(r)
    assert decisions and all(d["verdict"] == "ALLOW" for d, _ in decisions)
    receipts_by_ref = {RC.receipt_ref(x): x for x in r.receipts}
    for decision, payload in decisions:
        assert canon.canon_hash(decision) == payload["decision_hash"]
        assert payload["receipt_ref"] in receipts_by_ref and receipts_by_ref[payload["receipt_ref"]].step_id == payload["step_id"]
    purposes = [d["purpose"] for d, _ in decisions]
    assert "navigation" in purposes and "observed-url" in purposes
    assert H.events_of(r, "EgressSummary")[0]["payload"]["counts"] == {"allowed": len(decisions), "denied": 0, "dropped": 0}


@pytest.mark.parametrize("url,reason,origin", [
    ("http://169.254.169.254/latest/meta-data/", "DENY_METADATA_ADDRESS", "http://169.254.169.254"),
    ("http://0xa9fea9fe/latest/", "DENY_METADATA_ADDRESS", "http://0xa9fea9fe"),
    ("http://[::ffff:a9fe:a9fe]/", "DENY_METADATA_ADDRESS", "http://[::ffff:a9fe:a9fe]"),
    ("http://127.0.0.1:8080/admin", "DENY_NOT_LISTED", "http://127.0.0.1:8080"),
    ("http://metadata.google.internal/computeMetadata/v1/", "DENY_METADATA_NAME", "http://metadata.google.internal"),
    ("https://unlisted.example.test/", "DENY_NOT_LISTED", "https://unlisted.example.test"),
])
def test_a_blocked_navigation_never_reaches_the_adapter_and_is_a_receipt_linked_event(url, reason, origin):
    spec = make_spec([nav_step(url=url), save_step()])
    policy = default_policy(spec, allowed_origins=[APP, origin])               # the action policy says yes; egress says no
    env = H.evidence_env(spec=spec, policy=policy, egress=None)
    r = env.run()
    assert cls(r) == ("ENVIRONMENT_FAIL", "NAVIGATION_DENIED") and r.exit_code != 0
    assert operations(env.adapter, "act") == []                                # the adapter was never asked to navigate
    (receipt,) = r.receipts
    assert receipt.step_id == "s-nav" and receipt.result is R.ResultCode.NAVIGATION_DENIED
    blocked = [(d, p) for d, p in egress_events(r) if d["verdict"] == "DENY"]
    assert len(blocked) == 1
    decision, payload = blocked[0]
    assert reason in decision["reasons"] and decision["purpose"] == "navigation"
    assert payload["receipt_ref"] == RC.receipt_ref(receipt) and payload["step_id"] == "s-nav"
    assert canon.canon_hash(decision) == payload["decision_hash"]
    kinds = H.kinds(r)
    assert kinds.index("ActionReceipt") < kinds.index("EgressDecision") < kinds.index("EgressSummary") < kinds.index("RunSealed")
    summary = H.events_of(r, "EgressSummary")[0]["payload"]
    assert summary["counts"]["denied"] == 1 and len(summary["denied_hosts"]) == 1
    assert "latest" not in json.dumps(decision) and "admin" not in json.dumps(decision)       # host only


def test_the_operators_listing_of_a_metadata_address_does_not_let_a_navigation_through():
    spec = make_spec([nav_step(url="http://169.254.169.254:8082/x"), save_step()])
    entry = E.AllowEntry(schemes=("http",), ports=(8082,), host="169.254.169.254", networks=("169.254.169.254",),
                         private_ok=True, label="oops")
    egress, _ = H.make_guard(E.EgressPolicy(entries=(entry,)), {})
    policy = default_policy(spec, allowed_origins=[APP, "http://169.254.169.254:8082"])
    env = H.evidence_env(spec=spec, policy=policy, egress=egress)
    r = env.run()
    assert cls(r) == ("ENVIRONMENT_FAIL", "NAVIGATION_DENIED") and operations(env.adapter, "act") == []
    assert "DENY_METADATA_ADDRESS" in egress_events(r)[0][0]["reasons"]


def test_an_operator_listed_private_target_is_allowed_and_recorded_as_such():
    spec = make_spec([nav_step(url="http://compose-app:8080/form"), save_step()])
    entry = E.AllowEntry(schemes=("http",), ports=(8080,), host="compose-app", networks=("10.20.0.5",), private_ok=True,
                         label="compose")
    egress, _ = H.make_guard(E.EgressPolicy(entries=(entry,)), {"compose-app": ["10.20.0.5"]})
    policy = default_policy(spec, allowed_origins=[APP, "http://compose-app:8080"])
    env = H.evidence_env(spec=spec, policy=policy, egress=egress)
    r = run(env)
    decisions = [d for d, _ in egress_events(r)]
    assert decisions[0]["mode"] == "PRIVATE_OPERATOR" and decisions[0]["entry"] == "compose" and decisions[0]["pinned_address"] == "10.20.0.5"
    assert operations(env.adapter, "act")


def test_a_page_that_ends_up_off_the_list_ends_the_run_and_the_decision_is_recorded():
    def setup(adapter):
        def go(adapter, url):
            adapter.url = "https://evil.example.test/landing?token=" + H.LEAK
        adapter.on_navigate = go
    env = H.evidence_env(adapter_setup=setup)
    r = env.run()
    assert cls(r) == ("ENVIRONMENT_FAIL", "NAVIGATION_DENIED")
    blocked = [d for d, _ in egress_events(r) if d["verdict"] == "DENY"]
    assert blocked and blocked[0]["purpose"] == "observed-url" and blocked[0]["host"] == "evil.example.test"
    assert dict(env.egress.denied_hosts()) == {"evil.example.test": 1}
    assert H.LEAK not in H.stream_text(r)


def test_decisions_made_by_the_enforcement_point_during_a_step_are_linked_to_that_steps_receipt():
    egress, _ = H.make_guard(H.app_policy(), H.APP_TABLE)

    def setup(adapter):
        original = adapter.effects[("btn-save", "click")]

        def save_and_beacon(adapter, action):
            egress.check("https://beacon.exfil.test/?d=" + H.LEAK, purpose="subresource")
            egress.check("http://169.254.169.254/latest/", purpose="subresource")
            original(adapter, action)
        adapter.effects[("btn-save", "click")] = save_and_beacon
    env = H.evidence_env(egress=egress, adapter_setup=setup)
    r = run(env)
    assert cls(r) == ("PASS", None)                                           # a blocked beacon does not fail the test
    blocked = [(d, p) for d, p in egress_events(r) if d["verdict"] == "DENY"]
    assert [d["purpose"] for d, _ in blocked] == ["subresource", "subresource"]
    assert {"DENY_NOT_LISTED", "DENY_METADATA_ADDRESS"} == {x for d, _ in blocked for x in d["reasons"]}
    save_receipt = [x for x in r.receipts if x.step_id == "s-save"][0]
    assert all(p["receipt_ref"] == RC.receipt_ref(save_receipt) and p["step_id"] == "s-save" for _, p in blocked)
    assert H.LEAK not in H.stream_text(r)
    assert dict(egress.denied_hosts()) == {"169.254.169.254": 1, "beacon.exfil.test": 1}


def test_decisions_made_after_the_last_receipt_are_flushed_at_the_seal_and_none_is_lost():
    egress, _ = H.make_guard(H.app_policy(), H.APP_TABLE)

    def setup(adapter):
        original = adapter.close_session

        def close_and_beacon(s):
            egress.check("https://late.exfil.test/")
            return original(s)
        adapter.close_session = close_and_beacon
    env = H.evidence_env(egress=egress, adapter_setup=setup)
    r = run(env)
    late = [(d, p) for d, p in egress_events(r) if d["host"] == "late.exfil.test"]
    assert len(late) == 1 and late[0][1]["receipt_ref"] == RC.receipt_ref(r.receipts[-1])
    counts = H.events_of(r, "EgressSummary")[0]["payload"]["counts"]
    assert counts["denied"] == 1 and counts["allowed"] + counts["denied"] == len(egress_events(r)) and counts["dropped"] == 0


def test_the_same_navigation_without_the_egress_guard_reaches_the_adapter_so_the_guard_is_what_stops_it():
    spec = make_spec([nav_step(url="http://169.254.169.254/x"), save_step()])
    policy = default_policy(spec, allowed_origins=[APP, "http://169.254.169.254"])
    env = make_env(spec, policy=policy)
    env.run()
    assert operations(env.adapter, "act")


def test_decisions_made_during_a_retried_step_are_linked_to_the_attempt_they_happened_in():
    from .test_runner import nav_timeout, retry_spec
    egress, _ = H.make_guard(H.app_policy(), H.APP_TABLE)

    def setup(adapter):
        nav_timeout()(adapter)                                              # the first navigation attempt times out
        original = adapter.act
        seen = []

        def act(s, action, target):
            if not seen:
                seen.append(1)
                egress.check("https://beacon.exfil.test/")                  # a blocked request during attempt 1
            return original(s, action, target)
        adapter.act = act
    guard = EvidenceGuard(H.make_redactor())
    config = make_config(egress_policy_id=egress.policy_id, redaction_policy_id=guard.policy_id, recovery_enabled=True)
    env = make_env(retry_spec(), config=config, adapter_setup=setup, evidence_guard=guard, egress=egress)
    r = env.run()
    attempts = [x for x in r.receipts if x.step_id == "s-nav"]
    assert [x.attempt for x in attempts] == [1, 2] and attempts[0].result is R.ResultCode.ACTION_TIMEOUT
    blocked = [e["payload"] for e in H.events_of(r, "EgressDecision") if e["payload"]["decision"]["verdict"] == "DENY"]
    assert len(blocked) == 1 and blocked[0]["receipt_ref"] == RC.receipt_ref(attempts[0]) and blocked[0]["step_id"] == "s-nav"
