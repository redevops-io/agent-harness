# SPDX-License-Identifier: AGPL-3.0-or-later
"""Agreement with the contracts' added vectors and fixtures, and the behaviours they pin.

``tests/qa/golden/qa_canon_vectors_addendum.json`` extends the canonical vectors (predicate ``expected_hash``, value
digests, receipts, recovery records and the order of every set-valued field) and ``qa_clarifications.json`` holds the
refusal and classification fixtures for the integrity codes, the idempotency key, ``reason_code`` and the identity
boundary. Both are data copied from the contracts side; every entry is reproduced here by the neutral module alone.
"""
import copy
import hashlib
import json
import re
import unicodedata

import pytest

from agent_harness.qa import canon, classify as C, errors, receipts as RC, records as R, verifier as V
from agent_harness.qa.errors import CanonicalizationRefused, QAError
from agent_harness.qa.idempotency import idempotency_key
from agent_harness.qa.specstore import InMemoryProgramRegistry, InMemorySpecStore

from ._fx_verify import PROGRAM_KEY, REF_DOCS, REF_PIN, SCRIPT_DOCS, SCRIPT_PIN, facts, vec
from ._helpers import (ADDENDUM_VECTOR_COUNT, CANON_VECTORS_SHA256, load_addendum, load_clarifications, load_inventory,
                       load_vectors)
from .test_verifier import TAMPERS, tampered

ADD = load_addendum()
CLR = load_clarifications()
VECTORS = {v["id"]: v for v in ADD["vectors"]}
CASES = {c["id"]: c for c in ADD["set_ordering"]["cases"]}


def digest(rcv1):
    return rcv1.removeprefix("rcv1:")


def neutral_digest(obj_or_form):
    form = obj_or_form.canonical_form() if hasattr(obj_or_form, "canonical_form") else obj_or_form
    return canon.digest_hex(form)


def record(data):
    return R.RECORDS_BY_SCHEMA_ID[data["schema_id"]].from_json(copy.deepcopy(data))


def refused(code, fn, *a, **k):
    with pytest.raises(QAError) as err:
        fn(*a, **k)
    assert err.value.code == code, (err.value.code, err.value.message)
    return err.value


# =====================================================================================================================
# the file itself
# =====================================================================================================================

def test_the_addendum_extends_the_base_vectors_and_adds_55_new_ones():
    assert ADD["extends"]["sha256"] == CANON_VECTORS_SHA256
    assert len(ADD["vectors"]) == len(VECTORS) == ADDENDUM_VECTOR_COUNT == 55
    assert not set(VECTORS) & {v["id"] for v in load_vectors()["vectors"]}      # an entry never restates a base one
    assert ADD["contract"] == CLR["contract"] == "qa/v1" and ADD["canonicalization"] == "rcv1"


@pytest.mark.parametrize("vid", sorted(VECTORS))
def test_every_addendum_vector_reproduces_byte_for_byte(vid):
    v = VECTORS[vid]
    text = canon.canonical_text(v["input"])
    assert text == v["canonical_json"]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == v["sha256_hex"] == digest(v["rcv1"])
    assert canon.canon_hash(v["input"]) == "canon-v1:" + v["sha256_hex"]
    assert canon.to_contract_ref(canon.canon_hash(v["input"])) == v["rcv1"]


def test_a_float_has_no_value_digest():
    (entry,) = ADD["refusals"]
    assert entry["expect"] == "refused"
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text(entry["input"])
    with pytest.raises(CanonicalizationRefused):
        RC.json_value_digest(1.5)


# =====================================================================================================================
# K1: the value digest is rcv1 of the canonical JSON text of the value alone
# =====================================================================================================================

@pytest.mark.parametrize("entry", ADD["value_digests"], ids=[e["id"] for e in ADD["value_digests"]])
def test_the_value_digest_of_every_entry(entry):
    assert entry["digest"] == VECTORS[entry["hashed_form_vector"]]["rcv1"]
    if entry["kind"] == "string":
        got = RC.value_digest(entry["value"])
        assert got == RC.json_value_digest(entry["value"])
        assert got == RC.action_value_digest(R.Action(kind=R.ActionKind.FILL, value=entry["value"]))
    elif entry["kind"] == "secret_ref":
        got = RC.action_value_digest(R.Action(kind=R.ActionKind.FILL, secret_ref=R.SecretRef(**entry["value"])))
        assert got == RC.value_digest(R.Action(kind=R.ActionKind.FILL, secret_ref=R.SecretRef(**entry["value"])))
    else:
        got = RC.json_value_digest(entry["value"])
    assert got == entry["digest"]


def test_a_string_is_hashed_after_nfc_and_with_its_quotes_and_nothing_else():
    assert RC.value_digest("hunter2") == "rcv1:" + hashlib.sha256(b'"hunter2"').hexdigest()
    nfd, nfc = unicodedata.normalize("NFD", "café"), unicodedata.normalize("NFC", "café")
    assert nfd != nfc and RC.value_digest(nfd) == RC.value_digest(nfc)
    assert RC.value_digest("") == "rcv1:" + hashlib.sha256(b'""').hexdigest()
    plain = RC.value_digest("hunter2")
    assert plain != "rcv1:" + canon.digest_hex({"action_value": {"value": "hunter2"}, "kind": "fill"})   # no wrapper


def test_the_action_kind_does_not_enter_the_digest_of_its_value():
    digests = {RC.action_value_digest(R.Action(kind=k, value="same")) for k in (R.ActionKind.FILL, R.ActionKind.PRESS,
                                                                             R.ActionKind.SELECT, R.ActionKind.NAVIGATE)}
    assert digests == {RC.value_digest("same")}
    assert RC.action_value_digest(R.Action(kind=R.ActionKind.CLICK)) is None


def test_a_secret_reference_digests_the_reference_and_never_the_secret():
    ref = R.SecretRef(provider="vault", key="token", namespace="tenant-a", path="kv/app", version="3")
    got = RC.action_value_digest(R.Action(kind=R.ActionKind.FILL, secret_ref=ref))
    assert got == canon.to_contract_ref(canon.canon_hash(ref.canonical_form()))
    other = R.SecretRef(provider="vault", key="token", namespace="tenant-a", path="kv/app", version="4")
    assert RC.action_value_digest(R.Action(kind=R.ActionKind.FILL, secret_ref=other)) != got


def test_a_receipt_records_the_value_digest_of_the_contracts_text():
    step = R.Step(step_id="s1", intent="type a name", target=R.StepTarget(kind=R.TargetKind.UI,
                  locator=R.LocatorDescriptor(role="textbox", name="Name")),
                  action=R.Action(kind=R.ActionKind.FILL, value="hunter2"), expected_transition="field holds the name",
                  observation_deadline_ms=1000, side_effect=R.SideEffect.NONE)
    receipt = RC.build_receipt(run_id="r", step=step, attempt=1, tool=R.ToolRef(name="t", version="1"),
                               policy_decision_ref="rcv1:" + "1" * 64, idempotency_key="idem-x",
                               result=R.ResultCode.OK, postcondition=R.Postcondition(
                                   declared="d", result=R.PostconditionResult.SATISFIED),
                               state_before_hash="rcv1:" + "2" * 64, state_after_hash="rcv1:" + "3" * 64)
    assert receipt.action.value_digest == "rcv1:" + hashlib.sha256(b'"hunter2"').hexdigest()
    assert "hunter2" not in json.dumps(receipt.to_json())


def test_the_value_digest_takes_a_string_or_an_action_and_nothing_else():
    for bad in (7, None, {"value": "x"}, ["x"], b"x"):
        with pytest.raises(QAError) as err:
            RC.value_digest(bad)
        assert err.value.code == "SPEC_INVALID"


# =====================================================================================================================
# the expected_hash of a predicate
# =====================================================================================================================

@pytest.mark.parametrize("entry", ADD["predicates"], ids=[e["id"] for e in ADD["predicates"]])
def test_the_predicate_expected_hash(entry):
    op = R.PredicateOp(entry["op"])
    pred = R.Predicate(op=op, **({"expected": entry["expected"]} if "expected" in entry else {}))
    assert "rcv1:" + neutral_digest(pred) == entry["expected_hash"] == VECTORS[entry["hashed_form_vector"]]["rcv1"]


@pytest.mark.parametrize("rel", ADD["predicate_relations"], ids=range(len(ADD["predicate_relations"])))
def test_the_predicate_relations(rel):
    specs = [R.AssertionSpec.from_json(copy.deepcopy(a)) for a in rel["assertions"]]
    hashes = [V.expected_hash(a) for a in specs]
    assert (len(set(hashes)) == 1) is rel["expected_hash_equal"]
    if rel["expected_hash_equal"]:
        assert hashes[0] == rel["expected_hash"]
    assert [canon.to_contract_ref(a.canon_hash()) for a in specs] == rel["assertion_hashes"]
    assert (len(set(rel["assertion_hashes"])) == len(specs)) is rel["assertion_hashes_differ"]
    assert not set(hashes) & set(rel["assertion_hashes"])                 # expected_hash never equals assertion_hash


# =====================================================================================================================
# K2: the order of a set is per field
# =====================================================================================================================

def navigate(obj, path):
    for part in path.split("."):
        obj = obj[part]
    return obj


@pytest.mark.parametrize("cid", sorted(c for c, v in CASES.items() if v["record"] != "QAPlan" and v["record"] != "PlanGate"))
def test_every_set_ordering_case_reproduces_its_hash(cid):
    case = CASES[cid]
    obj = record(case["record_listed"])
    assert "rcv1:" + neutral_digest(obj) == case["rcv1"] == VECTORS["set-order:" + cid]["rcv1"]
    field = case["field"] if case["record"] != "AssertionResult" else case["field"]
    listed_hash = case.get("rcv1_if_ordered_the_other_way")
    if listed_hash:
        assert "rcv1:" + neutral_digest(obj) != listed_hash                 # the other order is the wrong one
    canonical = navigate(obj.canonical_form(), field)
    assert canonical == case["canonical"]                                   # the canonical form lists it in rule order


@pytest.mark.parametrize("cid", sorted(c for c, v in CASES.items() if v["record"] in ("QAPlan", "PlanGate")))
def test_the_plan_cases_are_plain_data_the_canonicalization_hashes_as_given(cid):
    # the harness receives no plan record: only the canonical bytes of the ordered data are checked
    case = CASES[cid]
    assert "rcv1:" + canon.digest_hex(case["canonical"]) != case.get("rcv1_if_ordered_the_other_way", "")
    assert VECTORS["set-order:" + cid]["rcv1"] == case["rcv1"]
    assert canon.canonical_text(VECTORS["set-order:" + cid]["input"]) == VECTORS["set-order:" + cid]["canonical_json"]


def test_the_rule_of_each_listed_field_is_the_one_the_contracts_name():
    by_rule = {(r["record"], r["field"]): r for r in ADD["set_ordering"]["rules"]}
    mine = {(cls.__name__, i.name): i for cls in R.RECORD_TYPES + R.VALUE_TYPES
            for i in R.field_infos(cls) if i.spec.kind == "tuple" and i.spec.is_set}
    for (record_name, field), rule in by_rule.items():
        if record_name in ("QAPlan", "PlanGate"):
            continue
        cls_name = "PolicyDecisionRecord" if record_name == "QAPolicyDecision" else record_name
        info = mine[(cls_name, field)]
        if rule["order"] == "element_key":
            assert info.spec.set_key == rule["key"], (record_name, field)
        elif record_name == "Check":
            assert info.spec.keep_duplicates and info.spec.set_key is None
        else:
            assert info.spec.set_key is None and not info.spec.keep_duplicates, (record_name, field)
    ordered = {(r["record"], r["field"]) for r in ADD["set_ordering"]["ordered_not_sets"]}
    for record_name, field in ordered:
        for cls in R.RECORD_TYPES + R.VALUE_TYPES:
            if cls.__name__ == record_name:
                assert not [i for i in R.field_infos(cls) if i.name == field and i.spec.is_set], (record_name, field)


def test_code_point_order_is_not_numeric_and_not_case_folded():
    ids = ["a-9", "a-10", "a-1", "B-1", "a-2"]
    docs = [{**copy.deepcopy(REF_DOCS[0]), "assertion_id": i} for i in ids]
    spec = ts_with(assertions=docs)
    assert [a.assertion_id for a in spec.assertions] == ["B-1", "a-1", "a-10", "a-2", "a-9"]


def ts_with(**over):
    """The golden UI test spec with some fields replaced (its stored content hash is dropped: it no longer applies)."""
    base = vec("qa:test_spec_ui")
    base.pop("content_hash", None)
    base.update(over)
    return R.TestSpec.from_json(base)


def test_an_object_set_is_not_ordered_by_its_canonical_text():
    # FixtureRef: the canonical text starts with content_hash, the contracts' key is id
    fixtures = [{"id": "seed-b", "content_hash": "sha256:" + "0" * 63 + "3"},
                {"id": "seed-a", "content_hash": "sha256:" + "0" * 63 + "9"}]
    spec = ts_with(fixtures=fixtures)
    assert [f.id for f in spec.fixtures] == ["seed-a", "seed-b"]
    assert [f.id for f in ts_with(fixtures=list(reversed(fixtures))).fixtures] == ["seed-a", "seed-b"]
    by_text = sorted(fixtures, key=lambda f: canon.canonical_text(f))
    assert [f["id"] for f in by_text] == ["seed-b", "seed-a"]                # the rule this module no longer applies


def test_the_evidence_refs_of_a_detection_are_ordered_by_canonical_text_and_a_pin_of_a_check_by_value():
    case = CASES["evidence_refs_by_canonical_text"]
    obj = record(case["record_listed"])
    assert [canon.canonical_text(e.canonical_form()) for e in obj.evidence_refs] == sorted(
        canon.canonical_text(e.canonical_form()) for e in obj.evidence_refs)
    pins = CASES["check_evidence_pins_natural"]
    result = record(pins["record_listed"])
    assert result.check.evidence_refs == tuple(navigate(pins["record_listed"], "check.evidence_refs"))   # order kept
    assert navigate(result.canonical_form(), "check.evidence_refs") == sorted(result.check.evidence_refs)


def test_a_duplicate_set_element_is_refused_and_an_element_key_is_unique():
    (case,) = ADD["set_ordering"]["refusals"]
    assert case["expect"] == {"code": "DUPLICATE_ENTRY", "refused": True}
    err = refused("SPEC_INVALID", R.TestSpec.from_json, copy.deepcopy(case["input"]))
    assert "duplicate" in err.message.lower() or "unique" in err.message.lower()
    docs = [REF_DOCS[0], {**REF_DOCS[0], "predicate": {"op": "EQUALS", "expected": "Other"}}]
    refused("SPEC_INVALID", lambda: ts_with(assertions=docs))               # one key, two different elements


def test_a_pin_of_a_check_that_is_listed_twice_is_kept():
    pin = "zz-ref@1#sha256:" + "0" * 63 + "7"
    check = R.Check(check_id="c", verdict=R.Verdict.PASS, determinism=R.Determinism.DETERMINISTIC,
                    evidence_refs=(pin, pin))
    assert check.evidence_refs == (pin, pin) and check.canonical_form()["evidence_refs"] == [pin, pin]
    once = R.Check(check_id="c", verdict=R.Verdict.PASS, determinism=R.Determinism.DETERMINISTIC, evidence_refs=(pin,))
    assert check.canon_hash() != once.canon_hash()


# =====================================================================================================================
# A1 - A4: how the reused Check, EvidenceRef and SecretRef hash
# =====================================================================================================================

def test_the_detail_of_a_check_is_wording_and_never_enters_the_hash():
    base = R.Check(check_id="c", verdict=R.Verdict.PASS, determinism=R.Determinism.DETERMINISTIC)
    worded = R.Check(check_id="c", verdict=R.Verdict.PASS, determinism=R.Determinism.DETERMINISTIC, detail="free wording")
    assert worded.canon_hash() == base.canon_hash() and "detail" not in worded.canonical_form()
    assert worded.to_json()["detail"] == "free wording"                      # kept, and written back
    assert R.Check.from_json(worded.to_json()) == worded
    data = vec("qa:assertion_result_pass")
    plain = R.AssertionResult.from_json(copy.deepcopy(data))
    data["check"]["detail"] = "some other words"
    again = R.AssertionResult.from_json(data)                                # the stored content_hash still verifies
    assert again.canon_hash() == plain.canon_hash() and again.check.detail == "some other words"
    with pytest.raises(QAError):
        R.Check(check_id="c", verdict=R.Verdict.PASS, detail="")             # an empty detail is still refused


def test_an_evidence_ref_that_omits_modality_and_ref_type_hashes_with_the_defaults():
    bare = R.EvidenceRef(ref="obs-1", content_hash="sha256:" + "0" * 64)
    explicit = R.EvidenceRef(ref="obs-1", content_hash="sha256:" + "0" * 64, modality="text", ref_type="evidence")
    assert bare.canonical_form()["modality"] == "text" and bare.canonical_form()["ref_type"] == "evidence"
    assert bare.canon_hash() == explicit.canon_hash()
    assert "modality" not in bare.to_json() and "ref_type" not in bare.to_json()      # absent stays absent on output
    other = R.EvidenceRef(ref="obs-1", content_hash="sha256:" + "0" * 64, modality="image", ref_type="artifact")
    assert other.canon_hash() != bare.canon_hash()
    assert R.EVIDENCE_REF_DEFAULT_MODALITY == "text" and R.EVIDENCE_REF_DEFAULT_REF_TYPE == "evidence"
    data = vec("qa:observation")
    spelled, omitted = copy.deepcopy(data), copy.deepcopy(data)
    for key in ("screenshot_ref", "a11y_snapshot_ref"):
        spelled[key].update(modality="text", ref_type="evidence")
        for field in ("modality", "ref_type"):
            omitted[key].pop(field)
    for d in (spelled, omitted):
        d.pop("content_hash", None)
    assert R.Observation.from_json(omitted).canon_hash() == R.Observation.from_json(spelled).canon_hash()
    assert R.Observation.from_json(omitted).canon_hash() != R.Observation.from_json(data).canon_hash()


def test_a_secret_reference_without_a_path_hashes_the_path_as_the_empty_string():
    bare = R.SecretRef(provider="vault", key="k")
    empty = R.SecretRef(provider="vault", key="k", path="")
    assert bare.canonical_form() == {"key": "k", "path": "", "provider": "vault"}
    assert empty.canonical_form() == bare.canonical_form() and empty.canon_hash() == bare.canon_hash()
    assert "path" not in bare.to_json() and R.SecretRef.from_json(bare.to_json()) == bare
    assert R.SecretRef(provider="vault", key="k", path="ci/app").canon_hash() != bare.canon_hash()
    assert "namespace" not in bare.canonical_form()                            # only the path has a default
    with pytest.raises(QAError):
        R.SecretRef(provider="vault", namespace="")                            # the other optional strings stay non-empty
    # the fingerprint of an action that carries it
    assert RC.action_value_digest(R.Action(kind=R.ActionKind.FILL, secret_ref=bare)) == \
        canon.to_contract_ref(canon.canon_hash(bare.canonical_form()))


# =====================================================================================================================
# receipts, idempotency and recovery records
# =====================================================================================================================

def test_receipt_hashes_and_their_identity_relations():
    hashes = {}
    for entry in ADD["idempotency_receipts"]:
        receipt = record(VECTORS["qa-clarification:" + entry["id"]]["input"])
        hashes[entry["id"]] = "rcv1:" + neutral_digest(receipt)
        assert hashes[entry["id"]] == entry["hash"] == CLR["idempotency"]["receipt_hashes"][entry["id"]]
    for rel in CLR["state_and_value_digests"]["identity_relations"]:
        if "equal" in rel:
            assert len({hashes[n] for n in rel["equal"]}) == 1, rel["why"]
        else:
            assert len({hashes[n] for n in rel["differ"]}) == len(rel["differ"]), rel["why"]


def violations(receipts):
    """The documented rule, written against the text: one key per step across its attempts, a different key per step."""
    found, per_step, holders = [], {}, {}
    for r in receipts:
        per_step.setdefault((r["run_id"], r["step_id"]), set()).add(r["idempotency_key"])
        holders.setdefault((r["run_id"], r["idempotency_key"]), set()).add(r["step_id"])
    if any(len(keys) > 1 for keys in per_step.values()):
        found.append("IDEMPOTENCY_KEY_CHANGED_ACROSS_ATTEMPTS")
    if any(len(steps) > 1 for steps in holders.values()):
        found.append("IDEMPOTENCY_KEY_SHARED_ACROSS_STEPS")
    return found


@pytest.mark.parametrize("case", CLR["idempotency"]["cases"], ids=[c["id"] for c in CLR["idempotency"]["cases"]])
def test_the_idempotency_rule_holds_for_the_keys_this_module_produces(case):
    assert violations(case["receipts"]) == case["violations"]               # the fixture agrees with the rule as written
    produced = [dict(r, idempotency_key=idempotency_key(r["run_id"], r["step_id"])) for r in case["receipts"]]
    assert violations(produced) == []                                        # per step, never per attempt, never shared
    assert all(re.fullmatch(r"idem-[0-9a-f]{40}", r["idempotency_key"]) for r in produced)


def test_a_recovery_record_hashes_as_the_contracts_say_and_reason_code_is_identity():
    hashes = {}
    for entry in ADD["fault_recoveries"]:
        rec = record(VECTORS["qa-clarification:fault_recovery_" + entry["id"]]["input"])
        hashes[entry["id"]] = "rcv1:" + neutral_digest(rec)
        assert hashes[entry["id"]] == entry["hash"] == CLR["fault_recovery"]["hashes"][entry["id"]]
    for rel in CLR["fault_recovery"]["relations"]:
        assert hashes[rel["differ"][0]] != hashes[rel["differ"][1]], rel["why"]
    # a recovery that omits reason_code hashes as it did before the field existed
    assert hashes["recovered_without_reason"] != hashes["not_recovered_without_reason"]
    assert "reason_code" not in record(VECTORS["qa-clarification:fault_recovery_not_recovered_without_reason"]["input"]
                                       ).canonical_form()


def test_the_recovery_vocabulary_is_the_closed_eighteen():
    assert [m.value for m in R.RecoveryReason] == CLR["fault_recovery"]["reason_code_vocabulary"]
    assert len(R.RecoveryReason) == 18 and R.ENUMS["RecoveryReason"] is R.RecoveryReason
    assert [m.value for m in R.RecoveryReason] == load_inventory()["enums"]["RecoveryReason"]


@pytest.mark.parametrize("fx", CLR["fault_recovery"]["refusals"], ids=[f["id"] for f in CLR["fault_recovery"]["refusals"]])
def test_the_fault_recovery_fixtures(fx):
    assert fx["operation"] == "parse_fault_recovery"
    if fx["expect"]["refused"]:
        want = {"RECORD_INCONSISTENT": "SPEC_INVALID"}.get(fx["expect"]["code"], fx["expect"]["code"])   # folded code
        refused(want, R.FaultRecovery.from_json, copy.deepcopy(fx["input"]))
    else:
        got = R.FaultRecovery.from_json(copy.deepcopy(fx["input"]))
        assert got.reason_code is R.RecoveryReason.NO_CANDIDATE and got.result is R.RecoveryResult.NOT_RECOVERED


def test_every_member_of_the_vocabulary_survives_a_round_trip_on_a_not_recovered_record():
    base = VECTORS["qa-clarification:fault_recovery_not_recovered_no_candidate"]["input"]
    for member in R.RecoveryReason:
        data = dict(copy.deepcopy(base), reason_code=member.value)
        rec = R.FaultRecovery.from_json(data)
        assert rec.reason_code is member and R.FaultRecovery.from_json(rec.to_json(with_hash=True)) == rec
    for result in ("NOT_ATTEMPTED", "ABORTED"):
        assert R.FaultRecovery.from_json(dict(copy.deepcopy(base), result=result)).reason_code is not None


# =====================================================================================================================
# K3: the integrity refusal codes
# =====================================================================================================================

PIN = "rcv1:94c9e0bd652146534232836c1de1cea4d1aab066939f8e0755f8b930a2898858"


def run_refusal(fx):
    """The neutral module's answer to one refusal fixture, as ``{refused, code, run_outcome, run_reason}``."""
    data, op = copy.deepcopy(fx["input"]), fx["operation"]

    def loader(pin, stored, programs=None):
        sets = {pin: stored} if stored is not None and isinstance(pin, str) and canon.is_contract_ref(pin) else {}
        approved = [pin] if isinstance(pin, str) and canon.is_contract_ref(pin) else []
        return V.load_frozen_assertions(InMemorySpecStore(sets), pin, lineage=V.ApprovedLineage(approved),
                                        programs=InMemoryProgramRegistry(programs or {}))
    try:
        if op == "verify_pin_well_formed":
            loader(data["pin"], copy.deepcopy(REF_DOCS))
        elif op == "load_pinned_assertion_set":
            loader(data["pin"], data["stored"])
        elif op == "verify_check_program":
            ref = data["check_program_ref"]
            docs = copy.deepcopy(SCRIPT_DOCS)
            docs[0]["check_program_ref"] = ref
            pin = "rcv1:" + R.assertion_set_hash([R.AssertionSpec.from_json(copy.deepcopy(d)) for d in docs]).removeprefix("canon-v1:")
            programs = {} if data["registered_hash"] is None else {(ref["id"], ref["version"]): data["registered_hash"]}
            loader(pin, docs, programs)
        elif op == "check_batch_not_halted":
            latch = V.HaltLatch()
            if data["halted_by"] is not None:
                latch.trip(errors.ERRORS_BY_CODE[data["halted_by"]]("halting refusal"))
            latch.check()
        else:
            raise AssertionError(f"unknown operation {op}")
    except QAError as exc:
        out = {"refused": True, "code": exc.code}
        if exc.run_outcome:
            out.update(run_outcome=exc.run_outcome, run_reason=exc.run_reason)
        return out
    return {"refused": False}


@pytest.mark.parametrize("fx", CLR["refusals"], ids=[f["id"] for f in CLR["refusals"]])
def test_every_integrity_refusal_fixture_gets_its_own_code(fx):
    assert run_refusal(fx) == fx["expect"]


def test_the_five_new_codes_are_typed_errors_with_stable_codes_that_end_the_run_inconclusive():
    expected = {"ASSERTION_PROGRAM_MISMATCH": errors.AssertionProgramMismatch,
                "ASSERTION_PROGRAM_UNREGISTERED": errors.AssertionProgramUnregistered,
                "ASSERTION_PIN_NOT_FOUND": errors.AssertionPinNotFound,
                "ASSERTION_PIN_MALFORMED": errors.AssertionPinMalformed, "BATCH_HALTED": errors.BatchHalted}
    for code, cls in expected.items():
        assert cls.code == code and errors.ERRORS_BY_CODE[code] is cls and errors.error_for_code(code) is cls
        assert issubclass(cls, QAError) and not issubclass(cls, errors.AssertionHashMismatch)
        assert (cls.run_outcome, cls.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY")
        found = C.classify(facts(error_codes=(code,)))
        assert found.pair == ("INCONCLUSIVE", "ASSERTION_INTEGRITY") and found.halt_batch is True
    assert V.BatchHalted is errors.BatchHalted                                 # one class, importable from both modules


def test_the_integrity_vocabulary_is_the_contracts_eleven_in_order():
    assert errors.INTEGRITY_REFUSAL_CODES == tuple(CLR["classification"]["cases"][i]["code"] for i in range(11))
    assert [m.value for m in R.IntegrityRefusalCode] == list(errors.INTEGRITY_REFUSAL_CODES)
    assert [m.value for m in R.IntegrityRefusalCode] == load_inventory()["enums"]["IntegrityRefusalCode"]
    assert R.ENUMS["IntegrityRefusalCode"] is R.IntegrityRefusalCode
    # every code of the vocabulary except the outcome reason itself has an error class; each halts the batch
    for code in errors.INTEGRITY_REFUSAL_CODES:
        if code != "ASSERTION_INTEGRITY":
            assert errors.ERRORS_BY_CODE[code].run_reason == "ASSERTION_INTEGRITY"
            assert C.refusal_halts_batch(code) is True


@pytest.mark.parametrize("case", CLR["classification"]["cases"], ids=[c["code"] or "EMPTY" for c in CLR["classification"]["cases"]])
def test_classify_refusal_follows_the_fixture_and_fails_closed(case):
    got = C.classify_refusal(case["code"])
    assert (None if got is None else [got[0].value, got[1]]) == case["classification"]
    assert C.refusal_halts_batch(case["code"]) is (case["classification"] is not None)


@pytest.mark.parametrize("code", [None, 7, b"BATCH_HALTED", ["BATCH_HALTED"], "batch_halted", "BATCH_HALTED ",
                                  "ASSERTION_PIN_FROM_THE_FUTURE", ""])
def test_a_refusal_code_that_is_not_known_exactly_is_an_integrity_refusal(code):
    assert C.classify_refusal(code) == (R.Outcome.INCONCLUSIVE, "ASSERTION_INTEGRITY")
    assert C.refusal_halts_batch(code) is True


def test_an_older_consumer_reaches_the_outcome_a_current_one_reaches():
    older = CLR["classification"]["older_consumer"]
    for case in older["cases"]:
        assert C.classify_refusal(case["code"]) == (R.Outcome.INCONCLUSIVE, "ASSERTION_INTEGRITY")
    # every code the older consumer knew and is not an integrity refusal classifies as none, here too
    for code in older["known_codes"]:
        known_integrity = code in errors.INTEGRITY_REFUSAL_CODES
        assert (C.classify_refusal(code) is not None) is known_integrity, code


def test_hash_mismatch_keeps_its_meaning_for_a_recomputed_content_hash():
    err = refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions,
                  InMemorySpecStore({REF_PIN: tampered(TAMPERS["expected_value"])}), REF_PIN,
                  lineage=V.ApprovedLineage([REF_PIN]), programs=InMemoryProgramRegistry({}))
    assert err.run_reason == "ASSERTION_INTEGRITY"
    refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions, InMemorySpecStore({REF_PIN: REF_DOCS}), REF_PIN,
            lineage=V.ApprovedLineage([]), programs=InMemoryProgramRegistry({}))      # outside the approved lineage
    refused("ASSERTION_HASH_MISMATCH", V.FrozenAssertionSet, assertion_set_hash="canon-v1:" + "0" * 64,
            assertions=tuple(R.AssertionSpec.from_json(d) for d in REF_DOCS))         # verify_assertion_set: any difference


def test_the_checks_run_in_the_documented_order():
    # the shape of the pin comes before the store, whatever the store holds
    refused("ASSERTION_PIN_MALFORMED", V.load_frozen_assertions, InMemorySpecStore({REF_PIN: REF_DOCS}), "rcv1:xyz",
            lineage=V.ApprovedLineage([REF_PIN]), programs=InMemoryProgramRegistry({}))
    # the store comes before the recomputed hash
    refused("ASSERTION_PIN_NOT_FOUND", V.load_frozen_assertions, InMemorySpecStore({}), REF_PIN,
            lineage=V.ApprovedLineage([REF_PIN]), programs=InMemoryProgramRegistry({}))
    # the registry entry comes before the program hash
    refused("ASSERTION_PROGRAM_UNREGISTERED", V.load_frozen_assertions, InMemorySpecStore({SCRIPT_PIN: SCRIPT_DOCS}),
            SCRIPT_PIN, lineage=V.ApprovedLineage([SCRIPT_PIN]), programs=InMemoryProgramRegistry({}))
    refused("ASSERTION_PROGRAM_MISMATCH", V.load_frozen_assertions, InMemorySpecStore({SCRIPT_PIN: SCRIPT_DOCS}),
            SCRIPT_PIN, lineage=V.ApprovedLineage([SCRIPT_PIN]),
            programs=InMemoryProgramRegistry({PROGRAM_KEY: "sha256:" + "ab" * 32}))


def test_the_refusal_that_halts_the_batch_keeps_its_code_and_every_later_call_gets_batch_halted():
    latch = V.HaltLatch()
    refused("ASSERTION_PIN_NOT_FOUND", V.load_frozen_assertions, InMemorySpecStore({}), REF_PIN,
            lineage=V.ApprovedLineage([REF_PIN]), programs=InMemoryProgramRegistry({}), halt=latch)
    assert latch.tripped and latch.cause.code == "ASSERTION_PIN_NOT_FOUND"
    err = refused("BATCH_HALTED", V.load_frozen_assertions, InMemorySpecStore({REF_PIN: REF_DOCS}), REF_PIN,
                  lineage=V.ApprovedLineage([REF_PIN]), programs=InMemoryProgramRegistry({}), halt=latch)
    assert (err.run_outcome, err.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY")
    assert latch.cause.code == "ASSERTION_PIN_NOT_FOUND"                    # the first refusal is not overwritten


# =====================================================================================================================
# the identity boundary
# =====================================================================================================================

@pytest.mark.parametrize("fx", CLR["identity_boundary"]["refusals"], ids=[f["id"] for f in CLR["identity_boundary"]["refusals"]])
def test_engine_limits_and_kill_switch_state_are_not_manifest_fields(fx):
    assert fx["operation"] == "parse_run_manifest" and fx["expect"] == {"code": "UNKNOWN_FIELD", "refused": True}
    refused("UNKNOWN_FIELD", R.RunManifest.from_json, copy.deepcopy(fx["input"]))
    clean = copy.deepcopy(fx["input"])
    for key in ("kill_switch", "engine_limits", "operator_caps"):
        clean.pop(key, None)
    assert R.RunManifest.from_json(clean).outcome is R.Outcome.PASS         # without the key it is a valid manifest


def test_the_effect_of_the_kill_switch_is_recorded_where_it_happened():
    where = CLR["identity_boundary"]["where_the_effect_is_recorded"]
    assert where == {"field": "suppressed_reason", "member": "KILL_SWITCH", "record": "FaultRecovery"}
    assert R.SuppressedReason(where["member"]) is R.SuppressedReason.KILL_SWITCH
    assert R.RecoveryReason.KILL_SWITCH.value == "KILL_SWITCH"
    assert set(CLR["identity_boundary"]["custody_counters"]) == {f.name for f in R.field_infos(R.Counters)}
    assert {f.name: f.custody for f in R.field_infos(R.RunManifest)}["counters"] is True
    assert {f.name: f.custody for f in R.field_infos(R.FaultRecovery)}["cost"] is True
    assert {f.name: f.custody for f in R.field_infos(R.FaultRecovery)}["reason_code"] is False


# =====================================================================================================================
# determinism: the added vectors reproduce in any process, whatever the hash seed, locale or time zone
# =====================================================================================================================

ALIGNMENT_SCRIPT = r"""
import json, sys
from agent_harness.qa import canon, classify as C, receipts as RC, records as R
add = json.load(open(sys.argv[1], encoding="utf-8"))
out = {}
for v in add["vectors"]:
    out[v["id"]] = canon.canon_hash(v["input"])
    sid = v["input"].get("schema_id") if isinstance(v["input"], dict) else None
    cls = R.RECORDS_BY_SCHEMA_ID.get(sid)
    if cls is not None:
        out["record:" + v["id"]] = cls.from_json(v["input"]).canon_hash()
for case in add["set_ordering"]["cases"]:
    cls = R.RECORDS_BY_SCHEMA_ID.get(case["record_listed"].get("schema_id"))
    if cls is not None:
        out["order:" + case["id"]] = cls.from_json(case["record_listed"]).canon_hash()
for e in add["value_digests"]:
    if e["kind"] == "string":
        out["digest:" + e["id"]] = RC.value_digest(e["value"])
out["classify"] = [repr(C.classify_refusal(c)) for c in ("BATCH_HALTED", "", "SPEC_INVALID", "A_FUTURE_CODE")]
print(json.dumps(out, sort_keys=True))
"""


def _run_alignment_child(**env_overrides):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(env_overrides)
    done = subprocess.run([sys.executable, "-c", ALIGNMENT_SCRIPT, str(Path(__file__).with_name("golden") /
                                                                  "qa_canon_vectors_addendum.json")],
                          env=env, capture_output=True, text=True, timeout=120, cwd=str(root))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_added_vectors_are_identical_across_hash_seeds_locales_and_time_zones():
    runs = [_run_alignment_child(PYTHONHASHSEED=seed) for seed in ("0", "1", "2", "12345", "random")]
    runs.append(_run_alignment_child(PYTHONHASHSEED="7", LC_ALL="C", LANG="C", TZ="UTC"))
    runs.append(_run_alignment_child(PYTHONHASHSEED="8", LC_ALL="C.UTF-8", LANG="C.UTF-8", TZ="Pacific/Kiritimati"))
    first = runs[0]
    assert all(other == first for other in runs[1:])
    for vid, v in VECTORS.items():
        assert first[vid] == "canon-v1:" + v["sha256_hex"]
    for cid, case in CASES.items():
        if "order:" + cid in first:
            assert first["order:" + cid] == "canon-v1:" + digest(case["rcv1"])
    for entry in ADD["value_digests"]:
        if entry["kind"] == "string":
            assert first["digest:" + entry["id"]] == entry["digest"]
    integrity = "(<Outcome.INCONCLUSIVE: 'INCONCLUSIVE'>, 'ASSERTION_INTEGRITY')"
    assert first["classify"] == [integrity, integrity, "None", integrity]      # BATCH_HALTED, "", SPEC_INVALID, unknown
