# SPDX-License-Identifier: AGPL-3.0-or-later
"""Action receipts, state hashes, the append-only receipt chain, manifest binding and the evidence event chain.
Tamper tests: every kind of alteration of a sealed run is detected."""
import copy

import pytest

from agent_harness.qa import canon, receipts as RC, records as R
from agent_harness.qa.errors import SpecInvalid
from agent_harness.qa.run_errors import ReceiptChainBroken

from ._fx_runner import make_env, make_spec

DECISION = "rcv1:" + "ab" * 32
POST = R.Postcondition(declared="url ends /saved", result=R.PostconditionResult.NOT_EVALUATED)
TOOL = R.ToolRef(name="fake-qa", version="0.1.0")


def obs(step_id="s1", phase="BEFORE", oid="obs-1", **kw):
    data = {"observation_id": oid, "step_id": step_id, "phase": phase,
            "source": {"adapter": "fake-qa", "adapter_version": "0.1.0", "browser": "chromium", "browser_version": "130"},
            "redaction_applied": True, "redaction_policy_id": "rcv1:" + "1f" * 32, "url": "https://app.example.test/a",
            "frame": "main", "dom_meta": {"title": "T", "nodes": 3}}
    data.update(kw)
    return R.Observation.from_json(data)


def receipt(step=None, attempt=1, run_id="run-1", **kw):
    step = step or make_spec().steps[1]
    args = dict(run_id=run_id, step=step, attempt=attempt, tool=TOOL, policy_decision_ref=DECISION,
                idempotency_key="idem-1", result=R.ResultCode.OK, postcondition=POST,
                state_before_hash="rcv1:" + "01" * 32, state_after_hash="rcv1:" + "02" * 32)
    args.update(kw)
    return RC.build_receipt(**args)


# --- state hashes -------------------------------------------------------------------------------------------------

def test_state_hash_is_independent_of_the_observation_identity():
    a, b = obs(oid="obs-1", phase="BEFORE"), obs(oid="obs-9", phase="AFTER", step_id="s2")
    assert a.canon_hash() != b.canon_hash()
    assert RC.state_hash(a) == RC.state_hash(b)
    assert canon.is_contract_ref(RC.state_hash(a))


@pytest.mark.parametrize("change", [{"url": "https://app.example.test/b"}, {"dom_meta": {"title": "T", "nodes": 4}},
                                    {"frame": "iframe-1"}])
def test_state_hash_changes_when_the_shown_state_changes(change):
    assert RC.state_hash(obs()) != RC.state_hash(obs(**change))


def test_state_hash_covers_snapshot_and_screenshot_digests():
    ref = lambda n: {"ref": "r", "content_hash": "sha256:" + n * 64}
    base = RC.state_hash(obs())
    assert RC.state_hash(obs(a11y_snapshot_ref=ref("a"))) != base
    assert RC.state_hash(obs(screenshot_ref=ref("b"))) != base
    assert RC.state_hash(obs(screenshot_ref=ref("b"))) != RC.state_hash(obs(screenshot_ref=ref("c")))


def test_state_hash_refuses_anything_but_an_observation():
    for bad in (None, {"url": "x"}, "obs"):
        with pytest.raises(SpecInvalid):
            RC.state_hash(bad)


def test_an_unobserved_state_never_equals_a_real_state_or_another_attempt():
    u1, u2 = RC.unknown_state_hash("run-1", "s1", 1), RC.unknown_state_hash("run-1", "s1", 2)
    assert u1 != u2 != RC.state_hash(obs()) and u1 != RC.state_hash(obs())
    assert u1 == RC.unknown_state_hash("run-1", "s1", 1)


# --- the receipt --------------------------------------------------------------------------------------------------

def test_a_receipt_carries_what_the_adr_lists():
    step = make_spec().steps[1]
    r = receipt(step, attempt=2, locator_used=R.LocatorUsed(descriptor=step.target.locator,
                                                            strategy=R.LocatorStrategy.AUTHORED),
                recovery_rationale="retry after timeout", duration_ms=17)
    j = r.to_json()
    assert j["schema_id"] == "runtime-contracts/qa/action-receipt"
    assert (j["run_id"], j["step_id"], j["attempt"]) == ("run-1", "s-save", 2)
    assert j["tool"] == {"name": "fake-qa", "version": "0.1.0"}
    assert j["action"]["kind"] == "click" and j["policy_decision_ref"] == DECISION
    assert j["state_before_hash"].startswith("rcv1:") and j["state_after_hash"].startswith("rcv1:")
    assert j["idempotency_key"] == "idem-1" and j["postcondition"]["declared"] == "url ends /saved"
    assert j["locator_used"]["descriptor"]["role"] == "button" and j["locator_used"]["strategy"] == "AUTHORED"
    assert j["custody"] == {"duration_ms": 17, "recovery_rationale": "retry after timeout"}
    assert R.ActionReceipt.from_json(r.to_json(with_hash=True)) == r


def test_duration_and_rationale_are_custody_and_do_not_change_the_hash():
    a = receipt(duration_ms=1, recovery_rationale="one")
    b = receipt(duration_ms=999, recovery_rationale="two")
    c = receipt()
    assert a.canon_hash() == b.canon_hash() == c.canon_hash()
    assert "recovery_rationale" not in a.canonical_form() and "duration_ms" not in a.canonical_form()


def test_the_value_is_never_stored_only_a_digest():
    step = make_spec([{**make_spec().to_json()["steps"][1], "action": {"kind": "fill", "value": "hunter2-secret"}}]).steps[0]
    r = receipt(step)
    assert "hunter2-secret" not in repr(r.to_json())
    assert r.action.value_digest == RC.value_digest(step.action) and canon.is_contract_ref(r.action.value_digest)
    other = R.Action(kind=R.ActionKind.FILL, value="different")
    assert RC.value_digest(other) != r.action.value_digest
    assert RC.value_digest(R.Action(kind=R.ActionKind.CLICK)) is None
    secret = R.Action(kind=R.ActionKind.FILL, secret_ref=R.SecretRef(provider="vault", key="k"))
    assert canon.is_contract_ref(RC.value_digest(secret))


def test_receipt_ids_stay_within_the_id_pattern_for_long_step_ids():
    long_step = R.Step.from_json({**make_spec().steps[1].to_json(), "step_id": "s" + "y" * 127})
    r = receipt(long_step)
    assert len(r.receipt_id) <= 128
    assert receipt(long_step, attempt=2).receipt_id != r.receipt_id


def test_a_long_rationale_is_clipped_not_refused():
    assert len(receipt(recovery_rationale="x" * 9000).recovery_rationale) == 4096


# --- the ledger and the chain ---------------------------------------------------------------------------------------

def ledger_of(*receipts):
    led = RC.ReceiptLedger("run-1")
    for r in receipts:
        led.append(r)
    return led


def three():
    spec = make_spec()
    return [receipt(spec.steps[0]), receipt(spec.steps[1], 1), receipt(spec.steps[1], 2)]


def test_the_ledger_is_append_only_and_chained():
    led = RC.ReceiptLedger("run-1")
    assert led.head is None and len(led) == 0 and led.verify() is None
    links = [led.append(r) for r in three()]
    assert [l.index for l in links] == [0, 1, 2] and links[0].prev is None and links[1].prev == links[0].link_hash
    assert led.head == links[-1].link_hash and led.verify() == led.head and len(led) == 3
    public = {n for n in dir(led) if not n.startswith("_")}
    assert public == {"run_id", "receipts", "links", "head", "append", "verify", "refs_for_step", "last_for_step"}
    for mutator in ("remove", "pop", "clear", "insert", "extend", "replace", "__setitem__", "__delitem__"):
        assert not hasattr(led, mutator)


def test_the_snapshot_properties_are_copies():
    led = ledger_of(*three())
    snapshot = list(led.receipts)
    assert isinstance(led.receipts, tuple) and isinstance(led.links, tuple)
    snapshot.pop()
    assert len(led.receipts) == 3 and led.verify()


def test_append_rules():
    spec = make_spec()
    led = RC.ReceiptLedger("run-1")
    with pytest.raises(ReceiptChainBroken):
        led.append(receipt(run_id="run-2"))                                     # another run
    with pytest.raises(ReceiptChainBroken):
        led.append(receipt(spec.steps[1], attempt=2))                           # attempts start at 1
    led.append(receipt(spec.steps[1], 1))
    with pytest.raises(ReceiptChainBroken):
        led.append(receipt(spec.steps[1], 1))                                   # no repeats
    with pytest.raises(ReceiptChainBroken):
        led.append(receipt(spec.steps[1], 3))                                   # no gaps
    for not_a_receipt in (None, {"attempt": 2}, "rcpt"):
        with pytest.raises(ReceiptChainBroken):
            led.append(not_a_receipt)
    led.append(receipt(spec.steps[0], 1))                                       # another step has its own count
    assert len(led) == 2 and led.verify()


def test_refs_and_last_receipt_per_step():
    spec = make_spec()
    rs = three()
    led = ledger_of(*rs)
    assert led.refs_for_step("s-save") == (RC.receipt_ref(rs[1]), RC.receipt_ref(rs[2]))
    assert led.refs_for_step("s-nav") == (RC.receipt_ref(rs[0]),) and led.refs_for_step("nope") == ()
    assert led.last_for_step("s-save") is rs[2] and led.last_for_step("nope") is None
    assert all(canon.is_contract_ref(x) for x in led.refs_for_step("s-save"))


def tamper_cases():
    rs = three()
    led = ledger_of(*rs)
    links = list(led.links)
    other = receipt(make_spec().steps[1], 1, state_after_hash="rcv1:" + "ff" * 32)
    yield "altered receipt", [rs[0], other, rs[2]], links
    yield "dropped receipt", [rs[0], rs[2]], links
    yield "dropped receipt and link", [rs[0], rs[2]], [links[0], links[2]]
    yield "reordered", [rs[1], rs[0], rs[2]], links
    yield "inserted receipt", [rs[0], rs[1], rs[1], rs[2]], links
    yield "extra receipt", rs + [receipt(make_spec().steps[0], 2)], links
    yield "truncated links", rs, links[:2]
    yield "link altered", rs, [links[0], RC.ChainLink(1, links[1].receipt_hash, links[1].prev, canon.canon_hash("x")), links[2]]
    yield "link prev altered", rs, [links[0], RC.ChainLink(1, links[1].receipt_hash, "canon-v1:" + "0" * 64,
                                                           links[1].link_hash), links[2]]
    yield "link index altered", rs, [links[0], RC.ChainLink(5, links[1].receipt_hash, links[1].prev, links[1].link_hash),
                                     links[2]]
    yield "link receipt hash altered", rs, [links[0], RC.ChainLink(1, canon.canon_hash("y"), links[1].prev,
                                                                   links[1].link_hash), links[2]]
    yield "not a receipt", [rs[0], {"x": 1}, rs[2]], links


@pytest.mark.parametrize("name,receipts,links", list(tamper_cases()), ids=[c[0] for c in tamper_cases()])
def test_every_alteration_of_the_chain_is_detected(name, receipts, links):
    with pytest.raises(ReceiptChainBroken) as err:
        RC.verify_chain(receipts, links)
    assert err.value.code == "RECEIPT_CHAIN_BROKEN" and err.value.run_reason == "ASSERTION_INTEGRITY"


def test_tampering_with_the_ledger_internals_is_caught_by_verify():
    led = ledger_of(*three())
    led._receipts[1] = receipt(make_spec().steps[1], 1, result=R.ResultCode.ACTION_TIMEOUT)
    with pytest.raises(ReceiptChainBroken):
        led.verify()


def test_the_chain_commits_to_order_not_just_content():
    spec = make_spec()
    a, b = receipt(spec.steps[0]), receipt(spec.steps[1])
    assert ledger_of(a, b).head != ledger_of(b, a).head


def test_chain_heads_are_deterministic():
    assert ledger_of(*three()).head == ledger_of(*three()).head


# --- the manifest binding (end to end) --------------------------------------------------------------------------------

def test_receipt_hashes_are_written_into_the_manifest_and_checked_against_it():
    r = make_env().run()
    RC.verify_manifest_receipts(r.manifest, r.receipts)
    by_step = {sr.step_id: sr.receipts for sr in r.manifest.step_results}
    assert by_step["s-nav"] == (RC.receipt_ref(r.receipts[0]),) and by_step["s-save"] == (RC.receipt_ref(r.receipts[1]),)
    assert r.manifest.canon_hash() != R.RunManifest.from_json(
        {**r.manifest.to_json(), "step_results": [{**sr.to_json(), "receipts": []} for sr in r.manifest.step_results]}
    ).canon_hash()                                               # the receipt hashes are identity fields of the manifest


def test_a_changed_removed_or_added_receipt_no_longer_matches_the_manifest():
    r = make_env().run()
    altered = [r.receipts[0], R.ActionReceipt.from_json({**r.receipts[1].to_json(), "result": "ACTION_TIMEOUT"})]
    for bad in (altered, r.receipts[:1], list(r.receipts) + [receipt(make_spec().steps[1], 2, run_id=r.run_id)],
                list(reversed(r.receipts))[:1]):
        with pytest.raises(ReceiptChainBroken):
            RC.verify_manifest_receipts(r.manifest, bad)
    j = r.manifest.to_json()
    j["step_results"][0]["receipts"] = ["rcv1:" + "9" * 64]
    with pytest.raises(ReceiptChainBroken):
        RC.verify_manifest_receipts(R.RunManifest.from_json(j), r.receipts)


# --- the event chain --------------------------------------------------------------------------------------------------

def events_of_a_run():
    return [copy.deepcopy(dict(e)) for e in make_env().run().events]


def test_the_event_chain_verifies_and_commits_to_every_payload():
    events = events_of_a_run()
    assert events[0]["kind"] == "RunStarted" and events[-1]["kind"] == "RunSealed"
    assert RC.verify_events(events) == events[-1]["event_hash"]
    assert [e["seq"] for e in events] == list(range(len(events))) and events[0]["prev"] is None
    assert RC.verify_events([]) is None


def event_tampers():
    events = events_of_a_run()

    def mutate(fn):
        copy_ = copy.deepcopy(events)
        fn(copy_)
        return copy_
    flip = lambda e: e[2]["payload"].__setitem__("decision", "DENY" if e[2]["payload"]["decision"] == "ALLOW" else "ALLOW")
    yield "payload field", mutate(flip)
    yield "run started payload", mutate(lambda e: e[0]["payload"].__setitem__("assertion_set_hash", "rcv1:" + "0" * 64))
    yield "payload hash", mutate(lambda e: e[1].__setitem__("payload_hash", canon.canon_hash("z")))
    yield "event hash", mutate(lambda e: e[1].__setitem__("event_hash", canon.canon_hash("z")))
    yield "dropped event", mutate(lambda e: e.pop(2))
    yield "reordered events", mutate(lambda e: e.insert(1, e.pop(2)))
    yield "duplicated event", mutate(lambda e: e.insert(2, e[2]))
    yield "prev link", mutate(lambda e: e[3].__setitem__("prev", canon.canon_hash("p")))
    yield "seq", mutate(lambda e: e[1].__setitem__("seq", 7))
    yield "missing key", mutate(lambda e: e[1].pop("payload"))
    yield "kind", mutate(lambda e: e[1].__setitem__("kind", "Other"))
    yield "sealed payload", mutate(lambda e: e[-1]["payload"]["classification"].__setitem__("outcome", "FAIL"))


@pytest.mark.parametrize("name,events", list(event_tampers()), ids=[n for n, _ in event_tampers()])
def test_every_alteration_of_the_event_chain_is_detected(name, events):
    with pytest.raises(ReceiptChainBroken):
        RC.verify_events(events)
