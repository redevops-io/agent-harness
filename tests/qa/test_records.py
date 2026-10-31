# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neutral records: golden round trips, parity with the golden field inventory, strictness, custody, extras."""
import copy
import dataclasses

import pytest

from agent_harness.qa import canon, records as R
from agent_harness.qa.errors import (CanonicalizationRefused, CheckProgramRefRequired, CoverageViolation,
                                     FaultBindingMismatch, HashMismatch, QAError, RecoveryPayloadRejected,
                                     SchemaVersionUnsupported, SpecInvalid, UnknownEnumValue, UnknownField)
from agent_harness.qa.schemas import SCHEMA_VERSION, SUPPORTED_SCHEMAS

from ._helpers import load_inventory, load_vectors

VECTORS = {v["id"]: v for v in load_vectors()["vectors"]}
INV = load_inventory()


def vec(vid):
    return copy.deepcopy(VECTORS[vid]["input"])


def digest_of(vid):
    return "canon-v1:" + VECTORS[vid]["sha256_hex"]


#: QAPlan and its value types are not part of the harness
PLAN_ONLY = {"qa:qa_plan", "qa-value:plan_item", "qa-value:plan_gate", "qa-value:resource_claim"}

VALUE_CLASS = {
    "component_ref": R.ComponentRef, "locator_descriptor": R.LocatorDescriptor,
    "resource_ref_manifest_path": R.ResourceRef, "resource_ref_image_digest": R.ResourceRef,
    "resource_ref_namespace_object": R.ResourceRef, "check_program_ref": R.CheckProgramRef,
    "step_target_ui": R.StepTarget, "step_target_resource": R.StepTarget, "action_click": R.Action,
    "step_ui": R.Step, "step_resource": R.Step, "predicate_equals": R.Predicate,
    "assertion_subject_ui": R.AssertionSubject, "test_target": R.TestTarget, "fixture_ref": R.FixtureRef,
    "budget": R.Budget, "observation_source": R.ObservationSource, "tool_ref": R.ToolRef,
    "locator_used": R.LocatorUsed, "action_record": R.ActionRecord, "postcondition": R.Postcondition,
    "recovery_cost": R.RecoveryCost, "skipped_assertion": R.SkippedAssertion, "version_pin": R.VersionPin,
    "model_ref": R.ModelRef, "step_result": R.StepResult, "counters": R.Counters,
    "session_profile": R.SessionProfile, "session_handle": R.SessionHandle, "observe_request": R.ObserveRequest,
    "locate_candidate": R.LocateCandidate, "locate_result": R.LocateResult, "resolved_target": R.ResolvedTarget,
    "action_request": R.ActionRequest, "action_outcome": R.ActionOutcome, "reset_result": R.ResetResult,
    "check_profile_offline": R.CheckProfile, "check_profile_k": R.CheckProfile,
    "workspace_handle": R.WorkspaceHandle, "resolved_resource": R.ResolvedResource,
    "check_subject": R.CheckSubject,
}

RECORD_VECTORS = sorted(v for v in VECTORS if v.startswith("qa:") and v not in PLAN_ONLY)
VALUE_VECTORS = sorted(v for v in VECTORS if v.startswith("qa-value:") and v not in PLAN_ONLY)
SET_VECTORS = sorted(v for v in VECTORS if v.startswith("qa-assertion-set:"))


def refused_with(code, fn, *args, **kwargs):
    with pytest.raises(QAError) as err:
        fn(*args, **kwargs)
    assert err.value.code == code, (err.value.code, err.value.message)
    return err.value


# --- the harness has no QAPlan --------------------------------------------------------------------------------

def test_the_harness_receives_no_qaplan_record():
    assert not hasattr(R, "QAPlan") and not hasattr(R, "PlanItem")
    assert "runtime-contracts/qa/qa-plan" not in SUPPORTED_SCHEMAS
    plan = vec("qa:qa_plan")
    assert plan["schema_id"] not in R.RECORDS_BY_SCHEMA_ID
    with pytest.raises(SchemaVersionUnsupported):
        from agent_harness.qa.schemas import require_supported
        require_supported(plan["schema_id"], plan["schema_version"])


def test_every_non_plan_vector_has_a_record_or_value_class():
    assert set(VALUE_VECTORS) == {"qa-value:" + n for n in VALUE_CLASS}
    assert len(RECORD_VECTORS) == 24 and len(SET_VECTORS) == 12


# --- golden: parse, canonical form, hash ------------------------------------------------------------------------

@pytest.mark.parametrize("vid", RECORD_VECTORS)
def test_record_vector_roundtrips_and_hashes(vid):
    data = vec(vid)
    cls = R.RECORDS_BY_SCHEMA_ID[data["schema_id"]]
    obj = cls.from_json(data)
    assert obj.canonical_form() == data
    assert canon.canonical_text(obj.canonical_form()) == VECTORS[vid]["canonical_json"]
    assert obj.canon_hash() == digest_of(vid)
    assert canon.to_contract_ref(obj.canon_hash()) == VECTORS[vid]["rcv1"]
    again = cls.from_json(obj.to_json(with_hash=True))          # full JSON form incl. content_hash
    assert again == obj and again.canon_hash() == obj.canon_hash()
    assert hash(again) == hash(obj)


@pytest.mark.parametrize("vid", VALUE_VECTORS)
def test_value_vector_roundtrips_and_hashes(vid):
    cls = VALUE_CLASS[vid.split(":", 1)[1]]
    data = vec(vid)
    obj = cls.from_json(data)
    assert obj.canonical_form() == data
    assert obj.canon_hash() == digest_of(vid)
    assert cls.from_json(obj.to_json()) == obj


@pytest.mark.parametrize("vid", SET_VECTORS)
def test_assertion_set_vector(vid):
    data = vec(vid)
    assert data["schema_id"] == "runtime-contracts/qa/assertion-set" and data["schema_version"] == SCHEMA_VERSION
    specs = [R.AssertionSpec.from_json(a) for a in data["assertions"]]
    assert R.assertion_set_document(specs) == data
    assert R.assertion_set_hash(specs) == digest_of(vid)
    assert R.assertion_set_hash(list(reversed(specs))) == digest_of(vid)          # order-independent


def test_assertion_set_relations():
    h = {vid: VECTORS[vid]["sha256_hex"] for vid in SET_VECTORS}
    assert h["qa-assertion-set:reference"] == h["qa-assertion-set:order_variation"]
    assert h["qa-assertion-set:unicode_nfc"] == h["qa-assertion-set:unicode_nfd"]
    ref = h["qa-assertion-set:reference"]
    for name in ("expected_changed", "locator_descriptor_changed", "requirement_ref_changed",
                 "requirement_ref_removed", "operator_changed"):
        assert h["qa-assertion-set:" + name] != ref
    assert h["qa-assertion-set:script_assertion"] != h["qa-assertion-set:script_program_changed"]
    assert h["qa-assertion-set:script_replaces_playwright"] != ref


def test_assertion_set_hash_is_a_testspec_derived_value_and_pinned_in_the_manifest():
    spec = R.TestSpec.from_json(vec("qa:test_spec_ui"))
    manifest = R.RunManifest.from_json(vec("qa:run_manifest_pass"))
    assert canon.to_contract_ref(spec.assertion_set_hash()) == manifest.assertion_set_hash
    assert spec.spec_ref() == manifest.spec_ref
    with pytest.raises(SpecInvalid):
        R.assertion_set_hash([spec.assertions[0], spec.assertions[0]])      # duplicate ids
    with pytest.raises(SpecInvalid):
        R.assertion_set_hash([{"assertion_id": "x"}])                       # only AssertionSpec objects


def test_assertion_without_requirement_ref_hashes_as_before_the_field_existed():
    a = vec("qa:assertion_spec_without_requirement_ref")
    assert "requirement_ref" not in a
    obj = R.AssertionSpec.from_json(a)
    assert obj.requirement_ref is None and obj.canon_hash() == digest_of("qa:assertion_spec_without_requirement_ref")
    explicit = dict(a, requirement_ref=None)
    assert R.AssertionSpec.from_json(explicit).canon_hash() == obj.canon_hash()      # absent equals null


# --- parity with the golden field inventory (qa.inventory.json is used as data) ---------------------------------

def _split(s, sep):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def parse_inventory_type(s):
    """'Optional[Tuple[str[max_length=512]]]' -> normalized dict."""
    i = s.find("[")
    if i < 0:
        return {"base": s, "optional": False, "constraints": {}}
    name, inner = s[:i], s[i + 1:-1]
    assert s.endswith("]"), s
    if name == "Optional":
        out = parse_inventory_type(inner)
        out["optional"] = True
        return out
    if name == "Tuple":
        return {"base": "tuple", "optional": False, "elem": parse_inventory_type(inner), "constraints": {}}
    cons = {}
    for part in _split(inner, ";"):
        k, v = part.split("=", 1)
        cons[k] = v if k == "pattern" else int(v)
    return {"base": name, "optional": False, "constraints": cons}


def normalize_spec(spec):
    base = {"str": "str", "int": "int", "bool": "bool", "json": "Json", "tuple": "tuple"}.get(spec.kind)
    if spec.kind in ("enum", "record"):
        base = spec.cls.__name__
        base = {"PolicyDecisionRecord": "QAPolicyDecision"}.get(base, base)
    out = {"base": base, "optional": spec.optional}
    if spec.kind == "tuple":
        out["elem"] = normalize_spec(spec.elem)
        out["constraints"] = {}
        return out
    cons = {}
    for key, value in (("pattern", spec.pattern), ("max_length", spec.max_length), ("min_length", spec.min_length),
                       ("minimum", spec.minimum), ("maximum", spec.maximum), ("max_json_bytes", spec.max_json_bytes)):
        if value is not None:
            cons[key] = value
    out["constraints"] = cons
    return out


def _strip(t):
    t = dict(t)
    if "elem" in t:
        t["elem"] = _strip(t["elem"])
    return t


CLASS_BY_INVENTORY_NAME = {c.__name__: c for c in R.RECORD_TYPES + R.VALUE_TYPES if c is not R.PolicyDecisionRecord}
CLASS_BY_INVENTORY_NAME["QAPolicyDecision"] = R.PolicyDecisionRecord           # contracts name; same schema id
INVENTORY_PLAN_FAMILY = {"QAPlan", "PlanGate", "PlanItem", "ResourceClaim"}
INVENTORY_CLASSES = {}
for _section in ("records", "values", "reused"):
    for _name, _body in INV[_section].items():
        if _name not in INVENTORY_PLAN_FAMILY:
            INVENTORY_CLASSES[_name] = (_section, _body)


def test_class_set_equals_the_inventory_minus_the_plan():
    assert set(INVENTORY_CLASSES) == set(CLASS_BY_INVENTORY_NAME)
    assert {n for n, (s, _) in INVENTORY_CLASSES.items() if s == "records"} == {
        c.__name__ if c is not R.PolicyDecisionRecord else "QAPolicyDecision" for c in R.RECORD_TYPES}
    assert INV["contract"] == "qa/v1" and INV["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("name", sorted(INVENTORY_CLASSES))
def test_fields_required_custody_and_types_match_the_inventory(name):
    section, body = INVENTORY_CLASSES[name]
    cls = CLASS_BY_INVENTORY_NAME[name]
    mine = {i.name: i for i in R.field_infos(cls)}
    assert set(mine) == set(body["fields"])
    for fname, theirs in body["fields"].items():
        info = mine[fname]
        assert info.required == theirs["required"], (name, fname, "required")
        assert info.custody == theirs["custody"], (name, fname, "custody")
        expected = parse_inventory_type(theirs["type"])
        got = normalize_spec(info.spec)
        if section == "reused" and expected["base"] == "str" and not expected["optional"] and not theirs["required"]:
            # documented reading for Check/EvidenceRef/SecretRef: an unset optional string is None, "" is refused
            assert info.spec.optional and info.spec.min_length == 1 and not info.required
            got = dict(got, optional=False, constraints={})
        # the inventory writes sets as Tuple[...] too; whether a tuple is a set is checked below
        assert got == expected, (name, fname, got, expected)
    if section == "records":
        assert cls.IS_RECORD and cls.SCHEMA_ID == body["schema_id"]
        assert SCHEMA_VERSION == body["schema_version"]
        assert SUPPORTED_SCHEMAS[cls.SCHEMA_ID] == (body["schema_version"],)
    else:
        assert not cls.IS_RECORD


def test_set_fields_are_exactly_the_unique_item_fields_of_the_frozen_json_schema():
    # qa.schema.json is not vendored; the set fields it marks uniqueItems are listed here as read from the golden data
    expected = {
        "CleanupReport": {"leaked", "released"}, "FaultDetection": {"evidence_refs"}, "ObserveRequest": {"capture"},
        "PolicyDecisionRecord": {"scope"}, "ReviewItem": {"evidence_refs"},
        "RunManifest": {"assertion_results", "assertions_declared", "assertions_executed", "assertions_skipped",
                        "step_results", "versions"},
        "Step": {"permitted_recovery"},
        "TestSpec": {"allowed_side_effects", "assertions", "depends_on", "fixtures", "sensitivity_labels"},
    }
    got = {}
    for cls in R.RECORD_TYPES + R.VALUE_TYPES:
        names = {i.name for i in R.field_infos(cls) if i.spec.kind == "tuple" and i.spec.is_set}
        if names:
            got[cls.__name__] = names
    assert got == expected


def test_enums_match_the_inventory():
    from agent_harness.qa import errors
    mine = dict(R.ENUMS, ErrorCategory=errors.ErrorCategory, ClassHint=errors.ClassHint)
    not_in_harness = {"CoverageContract", "EvidenceLevel"}          # QAPlan-only vocabularies
    assert set(INV["enums"]) - not_in_harness == set(mine)
    for name, members in INV["enums"].items():
        if name in not_in_harness:
            continue
        assert [m.value for m in mine[name]] == members, name
        assert [m.name for m in mine[name]] == [m.upper() for m in members], name


# --- defaults are materialized into the canonical form ----------------------------------------------------------

def test_defaults_are_conservative_and_materialized():
    s = R.Step.from_json(vec("qa-value:step_ui"))
    assert R.Step(step_id="s", intent="i", target=s.target, action=s.action, expected_transition="t",
                  observation_deadline_ms=10, side_effect=R.SideEffect.NONE).retry_safety is R.RetrySafety.UNSAFE
    bare = R.Budget(deadline_ms=1000)
    assert (bare.max_retries, bare.max_recovery_attempts, bare.max_model_calls, bare.max_tokens) == (0, 0, 0, 0)
    assert R.Check(check_id="c", verdict=R.Verdict.PASS).determinism is R.Determinism.NON_DETERMINISTIC
    assert R.Step(step_id="s", intent="i", target=s.target, action=s.action, expected_transition="t",
                  observation_deadline_ms=10, side_effect=R.SideEffect.NONE).idempotency is R.Idempotency.UNKNOWN
    sp = R.SessionProfile(egress_profile="p", redaction_policy_id="sha256:" + "0" * 64)
    assert (sp.allow_downloads, sp.allow_popups, sp.allow_file_choosers) == (False, False, False)
    assert sp.canonical_form()["allow_popups"] is False                 # materialized, as in the golden vector


# --- custody ---------------------------------------------------------------------------------------------------

def test_custody_is_excluded_from_the_hash_and_kept_under_custody():
    base = vec("qa:test_spec_ui")
    a = R.TestSpec.from_json(dict(base, custody={"author": "author-1"}))
    b = R.TestSpec.from_json(dict(base, custody={"author": "someone-else", "reviewer": "r", "approved_at": "t"}))
    assert a.canon_hash() == b.canon_hash() == digest_of("qa:test_spec_ui") == R.TestSpec.from_json(base).canon_hash()
    assert a.to_json()["custody"] == {"author": "author-1"} and "author" not in a.canonical_form()
    assert "custody" not in R.TestSpec.from_json(base).to_json()
    # a stored hash computed without custody still verifies
    stored = canon.to_contract_ref(a.canon_hash())
    assert R.TestSpec.from_json(dict(base, custody={"author": "x"}, content_hash=stored)).canon_hash() == a.canon_hash()


def test_custody_fields_must_sit_under_custody_and_identity_fields_must_not():
    base = vec("qa:test_spec_ui")
    refused_with("SPEC_INVALID", R.TestSpec.from_json, dict(base, author="a"))
    refused_with("UNKNOWN_FIELD", R.TestSpec.from_json, dict(base, custody={"nope": 1}))
    refused_with("SPEC_INVALID", R.TestSpec.from_json, dict(base, custody={"id": "x"}))
    refused_with("SPEC_INVALID", R.TestSpec.from_json, dict(base, custody="author"))
    m = vec("qa:run_manifest_pass")
    obj = R.RunManifest.from_json(dict(m, custody={"counters": {"infrastructure_ms": 5, "cost": "0.5"},
                                                   "started_at": "t0"}))
    assert obj.counters.cost == "0.5" and obj.canon_hash() == digest_of("qa:run_manifest_pass")


def test_custody_fields_never_appear_in_any_canonical_form():
    for cls in R.RECORD_TYPES:
        custody = {i.name for i in R.field_infos(cls) if i.custody}
        vid = next((v for v in RECORD_VECTORS if VECTORS[v]["input"]["schema_id"] == cls.SCHEMA_ID), None)
        assert vid is not None, cls
        form = cls.from_json(vec(vid)).canonical_form()
        assert custody.isdisjoint(form), (cls.__name__, custody & set(form))


# --- extras passthrough ----------------------------------------------------------------------------------------

def test_unknown_additive_fields_are_kept_and_hashed_at_every_level():
    base = vec("qa:test_spec_ui")
    plain = R.TestSpec.from_json(base)
    newer = copy.deepcopy(base)
    newer["x_new_top"] = {"k": [1, "two", None]}
    newer["budget"]["x_new_nested_value"] = "v"
    newer["assertions"][0]["x_new_nested_record"] = 7
    obj = R.TestSpec.from_json(newer)
    assert obj.extras["x_new_top"]["k"] == (1, "two", None)
    assert obj.budget.extras["x_new_nested_value"] == "v"
    assert obj.assertions[0].extras["x_new_nested_record"] == 7
    assert obj.canonical_form()["x_new_top"] == {"k": [1, "two", None]}
    assert obj.canon_hash() != plain.canon_hash()
    assert obj.canon_hash() == canon.canon_hash(newer)                      # recomputes exactly what the writer hashed
    assert R.TestSpec.from_json(obj.to_json()) == obj                           # survives a round trip
    assert obj.to_json()["x_new_top"] == {"k": [1, "two", None]}
    assert obj.assertion_set_hash() != plain.assertion_set_hash()               # a nested extra changes the set hash


def test_a_newer_writers_stored_hash_verifies_and_a_tampered_one_does_not():
    newer = vec("qa:checkpoint")
    newer["x_added_later"] = "v"
    newer["content_hash"] = "rcv1:" + canon.digest_hex({k: v for k, v in newer.items()})
    assert R.Checkpoint.from_json(newer).extras["x_added_later"] == "v"
    tampered = dict(newer, step_index=3)
    refused_with("HASH_MISMATCH", R.Checkpoint.from_json, tampered)
    dropped = {k: v for k, v in newer.items() if k != "x_added_later"}
    refused_with("HASH_MISMATCH", R.Checkpoint.from_json, dropped)             # dropping an unknown field is detected
    for bad in ("nonsense", "rcv1:abc", 7):
        refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(vec("qa:checkpoint"), content_hash=bad))


def test_extras_are_frozen_json_and_cannot_shadow_known_fields():
    cp = R.Checkpoint.from_json(dict(vec("qa:checkpoint"), x=[{"a": 1}]))
    with pytest.raises(TypeError):
        cp.extras["y"] = 1
    with pytest.raises(TypeError):
        cp.extras["x"][0]["a"] = 2
    with pytest.raises(AttributeError):
        cp.extras["x"][0].update({})
    refused_with("SPEC_INVALID", R.Checkpoint, checkpoint_id="c", run_id="r", step_index=0,
                 state_hash="rcv1:" + "0" * 64, extras={"step_index": 1})
    refused_with("SPEC_INVALID", R.Checkpoint, checkpoint_id="c", run_id="r", step_index=0,
                 state_hash="rcv1:" + "0" * 64, extras={"schema_id": "x"})
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(vec("qa:checkpoint"), x=1.5))
    refused_with("SPEC_INVALID", R.Checkpoint, checkpoint_id="c", run_id="r", step_index=0,
                 state_hash="rcv1:" + "0" * 64, extras={1: "x"})


def test_locator_candidate_is_strict_and_has_no_extras():
    ok = R.LocatorCandidate.from_json({"name": "Save", "role": "button", "strategy": "SEMANTIC_ALTERNATE"})
    assert ok.extras == {} and ok.canonical_form() == vec("qa:locator_candidate")
    assert R.LocatorCandidate.from_json(vec("qa:locator_candidate")).canon_hash() == digest_of("qa:locator_candidate")
    assert R.LocatorCandidate.ALLOW_EXTRAS is False
    for payload in (
        {"expected": "Anything", "role": "button", "strategy": "SEMANTIC_ALTERNATE"},
        {"selector": "#save", "strategy": "MODEL_CANDIDATE", "test_id": "save"},
        {"assertion": vec("qa:assertion_spec_playwright"), "role": "button", "strategy": "MODEL_CANDIDATE"},
        {"role": "button", "strategy": "NOT_A_STRATEGY"},
        {"role": 5, "strategy": "MODEL_CANDIDATE"},
        {"role": "button"},
        "role=button",
        None,
    ):
        err = refused_with("RECOVERY_PAYLOAD_REJECTED", R.LocatorCandidate.from_json, payload)
        assert (err.run_outcome, err.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY")
    refused_with("RECOVERY_PAYLOAD_REJECTED", R.LocatorCandidate, strategy=R.LocatorStrategy.AUTHORED, extras={"a": 1})


# --- frozen, immutable ------------------------------------------------------------------------------------------

def test_records_are_frozen_hashable_and_deeply_immutable():
    spec = R.TestSpec.from_json(vec("qa:test_spec_ui"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.assertions[0].predicate.op = R.PredicateOp.NOT_EQUALS
    with pytest.raises(AttributeError):
        spec.assertions.append(spec.assertions[0])
    assert isinstance(spec.steps, tuple) and isinstance(spec.assertions, tuple)
    assert hash(spec) == hash(R.TestSpec.from_json(vec("qa:test_spec_ui")))
    src = {"nested": [1, {"k": "v"}]}
    pred = R.Predicate(op=R.PredicateOp.EQUALS, expected=src)
    src["nested"].append("tampered")
    src["nested"][1]["k"] = "tampered"
    assert pred.canonical_form() == {"expected": {"nested": [1, {"k": "v"}]}, "op": "EQUALS"}
    with pytest.raises(TypeError):
        pred.expected["new"] = 1
    assert isinstance(pred.expected, R.FrozenMap) and pred.expected == {"nested": (1, {"k": "v"})}
    with pytest.raises(AttributeError):
        pred.expected._d = {}


def test_frozen_map_basics():
    m = R.FrozenMap({"a": 1})
    assert m == {"a": 1} and m["a"] == 1 and len(m) == 1 and list(m) == ["a"] and hash(m) == hash(R.FrozenMap({"a": 1}))
    assert m != {"a": 2} and m != 5
    assert "FrozenMap" in repr(m)


# --- sets ------------------------------------------------------------------------------------------------------

def test_sets_are_sorted_at_construction_and_duplicates_are_refused():
    data = vec("qa:run_manifest_pass")
    shuffled = copy.deepcopy(data)
    for key in ("assertion_results", "assertions_declared", "assertions_executed", "versions"):
        shuffled[key] = list(reversed(shuffled[key]))
    obj = R.RunManifest.from_json(shuffled)
    assert obj.canonical_form() == data and obj.canon_hash() == digest_of("qa:run_manifest_pass")
    refused_with("SPEC_INVALID", R.RunManifest.from_json, dict(data, assertions_declared=["a-saved", "a-saved", "a-url"]))
    refused_with("SPEC_INVALID", R.Step.from_json, dict(vec("qa-value:step_ui"), permitted_recovery=["RETRY_SAME", "RETRY_SAME"]))
    # NFC and NFD spellings are the same element
    refused_with("SPEC_INVALID", R.CleanupReport.from_json,
                 {"handle_id": "h", "verified": True, "released": ["café", "café"]})
    # the order of a sequence that is not a set is preserved
    st = R.TestSpec.from_json(vec("qa:test_spec_ui"))
    two = copy.deepcopy(vec("qa:test_spec_ui"))
    two["cleanup"] = ["b", "a"]
    assert R.TestSpec.from_json(two).cleanup == ("b", "a") and R.TestSpec.from_json(two).canon_hash() != st.canon_hash()


def test_object_sets_order_independent_hash_for_every_set_field():
    data = vec("qa:run_manifest_inconclusive_skipped")
    a = R.RunManifest.from_json(data)
    spec = vec("qa:test_spec_ui")
    spec["fixtures"] = [{"content_hash": "sha256:" + "0" * 63 + "9", "id": "z-fixture"},
                        {"content_hash": "sha256:" + "0" * 63 + "1", "id": "a-fixture"}]
    spec["sensitivity_labels"] = ["zeta", "alpha"]
    spec["depends_on"] = ["b@rcv1:" + "1" * 64, "a@rcv1:" + "2" * 64]
    spec["allowed_side_effects"] = ["WRITES_INTERNAL", "READS_EXTERNAL"]
    flipped = copy.deepcopy(spec)
    for key in ("fixtures", "sensitivity_labels", "depends_on", "allowed_side_effects", "assertions"):
        flipped[key] = list(reversed(flipped[key]))
    assert R.TestSpec.from_json(spec).canon_hash() == R.TestSpec.from_json(flipped).canon_hash()
    assert a.canon_hash() == digest_of("qa:run_manifest_inconclusive_skipped")


# --- schema identity and versions -----------------------------------------------------------------------------

def test_supported_schemas_table():
    assert all(v == (SCHEMA_VERSION,) for v in SUPPORTED_SCHEMAS.values())
    assert set(SUPPORTED_SCHEMAS) == {c.SCHEMA_ID for c in R.RECORD_TYPES} | {"runtime-contracts/qa/assertion-set"}
    with pytest.raises(TypeError):
        SUPPORTED_SCHEMAS["x"] = ("0.1",)


@pytest.mark.parametrize("version", ["1.0", "0.2", "0", "2", "", "0.1.0", None, 0.1, 1])
def test_unknown_schema_versions_are_refused(version):
    refused_with("SCHEMA_VERSION_UNSUPPORTED", R.Checkpoint.from_json, dict(vec("qa:checkpoint"), schema_version=version))


def test_schema_id_must_match_the_record():
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(vec("qa:checkpoint"), schema_id="runtime-contracts/qa/review-item"))
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(vec("qa:checkpoint"), schema_id=None))
    # absent ids are accepted and filled in: a recoverer's candidate carries none
    data = vec("qa:checkpoint")
    del data["schema_id"], data["schema_version"]
    assert R.Checkpoint.from_json(data).canon_hash() == digest_of("qa:checkpoint")


# --- strictness ------------------------------------------------------------------------------------------------

def test_wrong_types_missing_fields_and_floats_are_refused():
    cp = vec("qa:checkpoint")
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(cp, step_index=1.0))
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(cp, step_index=True))
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(cp, step_index="2"))
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(cp, step_index=-1))
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(cp, run_id=7))
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(cp, run_id=None))
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, [cp])
    refused_with("SPEC_INVALID", R.Observation.from_json, dict(vec("qa:observation"), redaction_applied="yes"))
    refused_with("SPEC_INVALID", R.Predicate.from_json, {"op": "EQUALS", "expected": 1.5})
    refused_with("SPEC_INVALID", R.Predicate.from_json, {"op": "EQUALS", "expected": {"a": [1, 2.5]}})
    refused_with("SPEC_INVALID", R.TestSpec.from_json, dict(vec("qa:test_spec_ui"), steps="s1"))
    refused_with("SPEC_INVALID", R.Postcondition.from_json, {"declared": "", "result": "SATISFIED"})
    for missing in ("run_id", "state_hash", "checkpoint_id", "step_index"):
        refused_with("SPEC_INVALID", R.Checkpoint.from_json, {k: v for k, v in cp.items() if k != missing})
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, {1: "x"})
    refused_with("UNKNOWN_ENUM_VALUE", R.Observation.from_json, dict(vec("qa:observation"), phase="DURING"))
    refused_with("UNKNOWN_ENUM_VALUE", R.Observation.from_json, dict(vec("qa:observation"), phase="after"))


@pytest.mark.parametrize("field,bad", [
    ("run_id", "has space"), ("run_id", "-leading"), ("run_id", "x" * 129), ("state_hash", "rcv1:" + "0" * 16),
    ("state_hash", "rcv1:" + "A" * 64), ("state_hash", "sha256:" + "0" * 64), ("state_hash", "canon-v1:" + "0" * 64),
    ("fixture_snapshot_id", ""),
])
def test_frozen_patterns_are_enforced(field, bad):
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(vec("qa:checkpoint"), **{field: bad}))


def test_embedded_hash_references_use_the_contracts_spelling():
    # neutral hash values are canon-v1:, but the golden patterns and vectors write embedded references as rcv1:
    h = R.Checkpoint.from_json(vec("qa:checkpoint")).canon_hash()
    assert h.startswith("canon-v1:")
    refused_with("SPEC_INVALID", R.Checkpoint.from_json, dict(vec("qa:checkpoint"), state_hash=h))
    ok = R.Checkpoint.from_json(dict(vec("qa:checkpoint"), state_hash=canon.to_contract_ref(h)))
    assert ok.state_hash == "rcv1:" + h[len("canon-v1:"):]


def test_json_size_limits_use_the_canonical_bytes():
    R.Predicate(op=R.PredicateOp.EQUALS, expected="x" * (4096 - 2))
    refused_with("SPEC_INVALID", R.Predicate, op=R.PredicateOp.EQUALS, expected="x" * 4096)
    refused_with("SPEC_INVALID", R.Predicate, op=R.PredicateOp.EQUALS, expected={"k": "x" * 4096})
    refused_with("SPEC_INVALID", R.Predicate, op=R.PredicateOp.EQUALS, expected={1: "x"})
    refused_with("SPEC_INVALID", R.Predicate, op=R.PredicateOp.EQUALS, expected=object())
    refused_with("SPEC_INVALID", R.Predicate, op=R.PredicateOp.EQUALS, expected=[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]])


# --- cross-field rules of the semantics --------------------------------------------------------------------

SCRIPT = vec("qa:assertion_spec_script")
PLAYWRIGHT = vec("qa:assertion_spec_playwright")


def test_script_assertions_carry_a_check_program_ref_and_playwright_ones_do_not():
    assert R.AssertionSpec.from_json(SCRIPT).check_program_ref.program_hash.startswith("sha256:")
    refused_with("CHECK_PROGRAM_REF_REQUIRED", R.AssertionSpec.from_json, {k: v for k, v in SCRIPT.items() if k != "check_program_ref"})
    refused_with("CHECK_PROGRAM_REF_REQUIRED", R.AssertionSpec.from_json, dict(SCRIPT, check_program_ref=None))
    refused_with("SPEC_INVALID", R.AssertionSpec.from_json, dict(PLAYWRIGHT, check_program_ref=SCRIPT["check_program_ref"]))
    refused_with("UNKNOWN_ENUM_VALUE", R.AssertionSpec.from_json, dict(PLAYWRIGHT, executor="TELEPATHY"))
    refused_with("UNKNOWN_ENUM_VALUE", R.AssertionSpec.from_json, dict(PLAYWRIGHT, executor="playwright"))
    refused_with("SPEC_INVALID", R.AssertionSpec.from_json, dict(PLAYWRIGHT, determinism="NON_DETERMINISTIC"))
    refused_with("SPEC_INVALID", R.AssertionSpec.from_json, dict(PLAYWRIGHT, on_unresolvable="FAIL"))
    refused_with("SPEC_INVALID", R.AssertionSpec.from_json, {k: v for k, v in PLAYWRIGHT.items() if k != "executor"})
    refused_with("SPEC_INVALID", R.AssertionSpec.from_json, dict(PLAYWRIGHT, requirement_ref="has space"))
    refused_with("SPEC_INVALID", R.AssertionSpec.from_json, dict(PLAYWRIGHT, requirement_ref="x" * 129))
    refused_with("SPEC_INVALID", R.AssertionSpec.from_json, dict(PLAYWRIGHT, check_program_ref={"id": "a", "version": "1", "program_hash": "sha256:" + "0" * 16}))
    changed = R.AssertionSpec.from_json(dict(SCRIPT, check_program_ref=dict(SCRIPT["check_program_ref"], version="2.0.0")))
    assert changed.canon_hash() != R.AssertionSpec.from_json(SCRIPT).canon_hash()      # the program identity is hashed


def test_step_target_is_a_tagged_union():
    ui, res = vec("qa-value:step_target_ui"), vec("qa-value:step_target_resource")
    refused_with("SPEC_INVALID", R.StepTarget.from_json, {"kind": "ui"})
    refused_with("SPEC_INVALID", R.StepTarget.from_json, dict(ui, resource=res["resource"]))
    refused_with("SPEC_INVALID", R.StepTarget.from_json, {"kind": "resource"})
    refused_with("SPEC_INVALID", R.StepTarget.from_json, dict(res, locator=ui["locator"]))
    refused_with("UNKNOWN_ENUM_VALUE", R.StepTarget.from_json, {"kind": "UI", "locator": ui["locator"]})


def test_resource_refs_carry_exactly_their_typed_fields():
    path, digest, ns = (vec("qa-value:resource_ref_" + n) for n in ("manifest_path", "image_digest", "namespace_object"))
    refused_with("SPEC_INVALID", R.ResourceRef.from_json, {"kind": "MANIFEST_PATH"})
    refused_with("SPEC_INVALID", R.ResourceRef.from_json, dict(path, digest=digest["digest"]))
    refused_with("SPEC_INVALID", R.ResourceRef.from_json, {"kind": "IMAGE_DIGEST"})
    refused_with("SPEC_INVALID", R.ResourceRef.from_json, dict(digest, digest="sha256:" + "0" * 16))
    refused_with("SPEC_INVALID", R.ResourceRef.from_json, {k: v for k, v in ns.items() if k != "name"})
    refused_with("SPEC_INVALID", R.ResourceRef.from_json, dict(ns, path="x"))
    R.ResourceRef.from_json({k: v for k, v in ns.items() if k != "namespace"})            # cluster-scoped object


def test_tier_k_needs_a_disposable_proof_and_actions_do_not_carry_both_value_and_secret():
    refused_with("SPEC_INVALID", R.CheckProfile.from_json, {"case_root_digest": "sha256:" + "0" * 64, "tier": "K"})
    R.CheckProfile.from_json(vec("qa-value:check_profile_offline"))
    refused_with("SPEC_INVALID", R.Action.from_json, {"kind": "fill", "value": "x", "secret_ref": {"provider": "vault"}})
    R.Action.from_json({"kind": "fill", "secret_ref": {"provider": "vault", "path": "p"}})


def test_unique_ids_inside_a_spec():
    spec = vec("qa:test_spec_ui")
    refused_with("SPEC_INVALID", R.TestSpec.from_json, dict(spec, steps=spec["steps"] * 2))
    a = copy.deepcopy(spec["assertions"][0])
    a["predicate"] = {"expected": "Other", "op": "EQUALS"}
    refused_with("SPEC_INVALID", R.TestSpec.from_json, dict(spec, assertions=spec["assertions"] + [a]))


def test_stored_content_hash_of_a_nested_record_is_verified():
    spec = vec("qa:test_spec_ui")
    spec["assertions"][0]["content_hash"] = "rcv1:" + "0" * 64
    refused_with("HASH_MISMATCH", R.TestSpec.from_json, spec)
    spec["assertions"][0]["content_hash"] = canon.to_contract_ref(R.AssertionSpec.from_json(vec("qa:test_spec_ui")["assertions"][0]).canon_hash())
    R.TestSpec.from_json(spec)


MANIFEST = vec("qa:run_manifest_pass")


def _manifest(**changes):
    return dict(copy.deepcopy(MANIFEST), **changes)


def test_run_manifest_accounting():
    R.RunManifest.from_json(MANIFEST)
    skipped_a_url = _manifest(assertions_executed=["a-saved"], assertions_skipped=[{"assertion_id": "a-url", "reason_code": "NOT_REACHED"}],
                              assertion_results=MANIFEST["assertion_results"][:1])
    # a PASS may not carry a skipped assertion
    refused_with("COVERAGE_VIOLATION", R.RunManifest.from_json, skipped_a_url)
    # the same accounting is a legitimate INCONCLUSIVE
    ok = R.RunManifest.from_json(dict(skipped_a_url, outcome="INCONCLUSIVE"))
    assert ok.assertions_skipped[0].reason_code is R.SkipReason.NOT_REACHED
    broken = {
        "executed not declared": _manifest(assertions_declared=["a-saved"]),
        "declared not accounted": _manifest(assertions_executed=["a-saved"], assertion_results=MANIFEST["assertion_results"][:1]),
        "overlap": _manifest(assertions_skipped=[{"assertion_id": "a-url", "reason_code": "NOT_REACHED"}]),
        "executed without a result": _manifest(assertions_executed=["a-saved", "a-url", "a-extra"], assertions_declared=["a-saved", "a-url", "a-extra"]),
        "result without executed id": _manifest(assertions_executed=["a-saved"], assertions_declared=["a-saved"]),
    }
    for name, data in broken.items():
        for outcome in ("PASS", "PRODUCT_FAIL", "TEST_DEFECT", "ENVIRONMENT_FAIL", "INCONCLUSIVE"):
            err = refused_with("COVERAGE_VIOLATION", R.RunManifest.from_json, dict(data, outcome=outcome))
            assert (err.run_outcome, err.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY"), name
        # only the frozen integrity outcome may record it
        R.RunManifest.from_json(dict(data, outcome="INCONCLUSIVE", outcome_reason="ASSERTION_INTEGRITY"))
    refused_with("SPEC_INVALID", R.RunManifest.from_json, _manifest(
        outcome="INCONCLUSIVE", outcome_reason="ASSERTION_INTEGRITY", assertions_executed=[], assertion_results=[],
        assertions_declared=["a-saved"], assertions_skipped=[{"assertion_id": "a-saved", "reason_code": "NOT_REACHED"}] * 2))


def test_fault_event_binds_recovery_to_the_detection_hash():
    ev = vec("qa:fault_event")
    obj = R.FaultEvent.from_json(ev)
    assert obj.recovery.detection_ref == canon.to_contract_ref(obj.detection.canon_hash())
    rewritten = copy.deepcopy(ev)
    rewritten["detection"]["signal"] = "ASSERTION_FAILED"           # recovery rewrites what was detected
    refused_with("FAULT_BINDING_MISMATCH", R.FaultEvent.from_json, rewritten)
    refused_with("FAULT_BINDING_MISMATCH", R.FaultEvent.from_json, dict(ev, recovery=dict(ev["recovery"], detection_ref="rcv1:" + "0" * 64)))
    # detection custody (timing) is not part of the binding
    timed = dict(ev, detection=dict(ev["detection"], custody={"detected_at_ms_into_step": 12}))
    assert R.FaultEvent.from_json(timed).canon_hash() == obj.canon_hash()
    refused_with("SPEC_INVALID", R.FaultRecovery.from_json, dict(ev["recovery"], postcondition_verified=False))


def test_every_error_the_records_raise_is_a_typed_qa_error():
    for exc in (SpecInvalid, UnknownField, UnknownEnumValue, HashMismatch, CoverageViolation, FaultBindingMismatch,
                CheckProgramRefRequired, RecoveryPayloadRejected, SchemaVersionUnsupported, CanonicalizationRefused):
        assert issubclass(exc, QAError) and isinstance(exc.code, str) and exc.code.isupper()


# --- mutation tests: removing a guard must be noticed ------------------------------------------------------------

import sys   # noqa: E402

from ._helpers import load_mutant   # noqa: E402


@pytest.fixture
def mutate():
    made = []

    def make(replacements):
        mod = load_mutant("records.py", replacements)
        made.append(mod.__name__)
        return mod

    yield make
    for name in made:
        sys.modules.pop(name, None)


def _raises(fn, code):
    try:
        fn()
    except QAError as exc:
        return exc.code == code
    except Exception:
        return False
    return False


_PASS_WITH_SKIP = _manifest(assertions_executed=["a-saved"], assertions_skipped=[{"assertion_id": "a-url", "reason_code": "NOT_REACHED"}],
                            assertion_results=MANIFEST["assertion_results"][:1])
_BROKEN_PASS = _manifest(assertions_declared=["a-saved"])
_NO_RESULT = _manifest(assertion_results=MANIFEST["assertion_results"][:1], outcome="PRODUCT_FAIL")
_FAULT_EV = vec("qa:fault_event")
_REWRITTEN = copy.deepcopy(_FAULT_EV)
_REWRITTEN["detection"]["signal"] = "ASSERTION_FAILED"
_TS = vec("qa:test_spec_ui")


def _set_order_probe(m):
    flipped = copy.deepcopy(_TS)
    flipped["assertions"] = list(reversed(flipped["assertions"]))
    return m.TestSpec.from_json(flipped).canon_hash() == m.TestSpec.from_json(_TS).canon_hash()


def _custody_probe(m):
    return (m.TestSpec.from_json(dict(_TS, custody={"author": "a"})).canon_hash()
            == m.TestSpec.from_json(_TS).canon_hash())


def _extras_probe(m):
    return m.TestSpec.from_json(dict(_TS, x_new=1)).canon_hash() != m.TestSpec.from_json(_TS).canon_hash()


GUARDS = {
    "script_requires_check_program_ref": (
        [('raise CheckProgramRefRequired("AssertionSpec: executor SCRIPT requires check_program_ref")', "pass")],
        lambda m: _raises(lambda: m.AssertionSpec.from_json({k: v for k, v in SCRIPT.items() if k != "check_program_ref"}), "CHECK_PROGRAM_REF_REQUIRED")),
    "playwright_forbids_check_program_ref": (
        [('raise SpecInvalid("AssertionSpec: check_program_ref is only for executor SCRIPT")', "pass")],
        lambda m: _raises(lambda: m.AssertionSpec.from_json(dict(PLAYWRIGHT, check_program_ref=SCRIPT["check_program_ref"])), "SPEC_INVALID")),
    "determinism_is_fixed": (
        [('raise SpecInvalid("AssertionSpec: a gating assertion must be DETERMINISTIC")', "pass")],
        lambda m: _raises(lambda: m.AssertionSpec.from_json(dict(PLAYWRIGHT, determinism="NON_DETERMINISTIC")), "SPEC_INVALID")),
    "on_unresolvable_is_fixed": (
        [('raise SpecInvalid("AssertionSpec: on_unresolvable is fixed to INDETERMINATE")', "pass")],
        lambda m: _raises(lambda: m.AssertionSpec.from_json(dict(PLAYWRIGHT, on_unresolvable="FAIL")), "SPEC_INVALID")),
    "unknown_enum_values_fail_closed": (
        [("return spec.cls(value)", "return value")],
        lambda m: _raises(lambda: m.AssertionSpec.from_json(dict(PLAYWRIGHT, executor="TELEPATHY")), "UNKNOWN_ENUM_VALUE")),
    "unknown_schema_version_refused": (
        [("require_supported(cls.SCHEMA_ID, raw)", "pass")],
        lambda m: _raises(lambda: m.Checkpoint.from_json(dict(vec("qa:checkpoint"), schema_version="1.0")), "SCHEMA_VERSION_UNSUPPORTED")),
    "stored_hash_is_verified": (
        [('raise HashMismatch(f"{path}: stored content_hash does not match the recomputed hash")', "pass")],
        lambda m: _raises(lambda: m.Checkpoint.from_json(dict(vec("qa:checkpoint"), content_hash="rcv1:" + "0" * 64)), "HASH_MISMATCH")),
    "fault_recovery_is_bound_to_the_detection": (
        [('raise FaultBindingMismatch("FaultEvent: recovery.detection_ref is not the hash of the detection it responds to")', "pass")],
        lambda m: _raises(lambda: m.FaultEvent.from_json(_REWRITTEN), "FAULT_BINDING_MISMATCH")),
    "recovered_needs_a_verified_postcondition": (
        [('raise SpecInvalid("FaultRecovery: RECOVERED requires postcondition_verified")', "pass")],
        lambda m: _raises(lambda: m.FaultRecovery.from_json(dict(_FAULT_EV["recovery"], postcondition_verified=False)), "SPEC_INVALID")),
    "broken_accounting_cannot_claim_pass": (
        [('if not (self.outcome is Outcome.INCONCLUSIVE and self.outcome_reason == "ASSERTION_INTEGRITY"):', "if False:")],
        lambda m: _raises(lambda: m.RunManifest.from_json(_BROKEN_PASS), "COVERAGE_VIOLATION")),
    "pass_requires_no_skipped_assertion": (
        [("if self.outcome is Outcome.PASS and self.assertions_skipped:", "if False:")],
        lambda m: _raises(lambda: m.RunManifest.from_json(_PASS_WITH_SKIP), "COVERAGE_VIOLATION")),
    "executed_ids_are_the_ids_with_a_result": (
        [("                and len(set(results)) == len(results) and set(results) == executed)", "                )")],
        lambda m: _raises(lambda: m.RunManifest.from_json(_NO_RESULT), "COVERAGE_VIOLATION")),
    "locator_candidate_is_strict": (
        [("ALLOW_EXTRAS: ClassVar[bool] = False", "ALLOW_EXTRAS: ClassVar[bool] = True")],
        lambda m: _raises(lambda: m.LocatorCandidate.from_json({"selector": "#x", "strategy": "MODEL_CANDIDATE"}), "RECOVERY_PAYLOAD_REJECTED")),
    "floats_are_refused_in_json": (
        [('raise SpecInvalid(f"{path}: floats are refused; carry decimals as strings")', "return value")],
        lambda m: _raises(lambda: m.Checkpoint.from_json(dict(vec("qa:checkpoint"), x_new=1.5)), "SPEC_INVALID")),
    "integers_are_exact": (
        [("if type(value) is not int:", "if False:")],
        lambda m: _raises(lambda: m.Budget(deadline_ms=1.5), "SPEC_INVALID")),
    "patterns_are_enforced": (
        [("if spec.compiled is not None and spec.compiled.fullmatch(value) is None:", "if False:")],
        lambda m: _raises(lambda: m.ComponentRef(id="has space", version="1"), "SPEC_INVALID")),
    "required_fields_are_required": (
        [("if info.required and name not in kwargs:", "if False:")],
        lambda m: _raises(lambda: m.Checkpoint.from_json({k: v for k, v in vec("qa:checkpoint").items() if k != "run_id"}), "SPEC_INVALID")),
    "step_target_is_a_tagged_union": (
        [('raise SpecInvalid("StepTarget(ui) needs a locator and no resource")', "pass")],
        lambda m: _raises(lambda: m.StepTarget.from_json({"kind": "ui"}), "SPEC_INVALID")),
    "resource_refs_carry_their_typed_fields": (
        [('raise SpecInvalid(f"ResourceRef({self.kind}): missing {sorted(need - present)}")', "pass")],
        lambda m: _raises(lambda: m.ResourceRef.from_json({"kind": "MANIFEST_PATH"}), "SPEC_INVALID")),
    "tier_k_needs_a_disposable_proof": (
        [('raise SpecInvalid("CheckProfile: disposable_proof_ref is required when tier is K")', "pass")],
        lambda m: _raises(lambda: m.CheckProfile.from_json({"case_root_digest": "sha256:" + "0" * 64, "tier": "K"}), "SPEC_INVALID")),
    "assertion_ids_are_unique": (
        [('raise SpecInvalid("TestSpec: assertion ids must be unique")', "pass")],
        lambda m: _raises(lambda: m.TestSpec.from_json(dict(_TS, assertions=_TS["assertions"] + [dict(_TS["assertions"][0], predicate={"expected": "X", "op": "EQUALS"})])), "SPEC_INVALID")),
    "sets_are_sorted_so_order_does_not_matter": (
        [("keyed = sorted(((_set_key(e), e) for e in items), key=lambda p: p[0])", "keyed = [(_set_key(e), e) for e in items]")],
        _set_order_probe),
    "set_duplicates_are_refused": (
        [('raise SpecInvalid(f"{path}: duplicate elements in a set")', "pass")],
        lambda m: _raises(lambda: m.CleanupReport(handle_id="h", verified=True, released=("a", "a")), "SPEC_INVALID")),
    "custody_is_not_hashed": (
        [("                if not identity:\n                    custody[info.name] = _plain(value, identity, with_hash)\n                continue\n",
          "                out[info.name] = _plain(value, identity, with_hash)\n                continue\n")],
        _custody_probe),
    "extras_are_hashed": (
        [("        for k, v in self.extras.items():\n            out[k] = _plain(v, identity, with_hash)\n", "")],
        _extras_probe),
    "custody_belongs_under_custody": (
        [('raise SpecInvalid(f"{path}.{key}: custody fields belong under \'custody\', not at the top level")', "pass")],
        lambda m: _raises(lambda: m.TestSpec.from_json(dict(_TS, author="a")), "SPEC_INVALID")),
}


@pytest.mark.parametrize("name", sorted(GUARDS))
def test_removing_a_guard_is_noticed(mutate, name):
    replacements, holds = GUARDS[name]
    assert holds(R), f"the real records must satisfy the guard probe {name}"
    assert not holds(mutate(replacements)), f"mutation {name} survived: removing the guard changes nothing"


def test_the_unmutated_copy_still_holds_every_guard(mutate):
    clone = mutate([])
    for name, (_, holds) in GUARDS.items():
        assert holds(clone), name
