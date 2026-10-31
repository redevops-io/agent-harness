# SPDX-License-Identifier: AGPL-3.0-or-later
"""The deterministic verifier: loading by hash, tamper refusals, executor and program checks, predicate semantics."""
import copy
import inspect
import os

import pytest

from agent_harness.qa import canon, records as R, verifier as V
from agent_harness.qa.errors import BrowserCrash, QAError
from agent_harness.qa.specstore import DirectorySpecStore, InMemoryProgramRegistry

from ._fx_verify import (EQ_SAVED_HASH, PROGRAM_HASH, PROGRAM_KEY, REF_DOCS, REF_PIN, SCRIPT_DOCS, SCRIPT_PIN, VECTORS,
                  ReadOnlyBrowserAdapter, ScriptedCheckAdapter, lineage, manifest, observed, pin_of, programs,
                  ref_answers, ref_specs, store, vec)


def refused(code, fn, *a, **k):
    with pytest.raises(QAError) as err:
        fn(*a, **k)
    assert err.value.code == code, (err.value.code, err.value.message)
    return err.value


def integrity(err):
    assert (err.run_outcome, err.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY")


def load(docs=REF_DOCS, pin=REF_PIN, **kw):
    kw.setdefault("lineage", lineage(pin))
    kw.setdefault("programs", programs())
    return V.load_frozen_assertions(store((pin, docs)), pin, **kw)


def tampered(path_edit):
    docs = copy.deepcopy(REF_DOCS)
    path_edit(docs)
    return docs


# --- loading by hash ---------------------------------------------------------------------------------------------

def test_the_reference_set_loads_by_its_pinned_hash_and_recomputes_to_the_same_digest():
    frozen = load()
    assert frozen.assertion_set_hash == REF_PIN
    assert frozen.assertion_ids == ("a-saved", "a-url")
    assert canon.same_digest(R.assertion_set_hash(frozen.assertions), REF_PIN)      # canon-v1 digest == rcv1 digest
    assert V.assertion_set_pin(frozen.assertions) == REF_PIN


def test_every_frozen_assertion_set_vector_loads_under_its_own_pin():
    sets = [v for v in VECTORS if v.startswith("qa-assertion-set:")]
    assert len(sets) == 12
    for vid in sets:
        docs = vec(vid)["assertions"]
        registry = {(d["check_program_ref"]["id"], d["check_program_ref"]["version"]): d["check_program_ref"]["program_hash"]
                    for d in docs if "check_program_ref" in d}
        frozen = V.load_frozen_assertions(store((pin_of(vid), docs)), pin_of(vid), lineage=lineage(pin_of(vid)),
                                          programs=InMemoryProgramRegistry(registry))
        assert frozen.assertion_set_hash == pin_of(vid)


def test_the_order_of_the_stored_documents_does_not_matter():
    assert load(list(reversed(REF_DOCS))).assertion_ids == ("a-saved", "a-url")


def test_loading_is_idempotent_and_deterministic():
    a, b = load(), load()
    assert a == b and hash(a) == hash(b)
    assert [x.canon_hash() for x in a.assertions] == [x.canon_hash() for x in b.assertions]


@pytest.mark.parametrize("pin", [None, "", "canon-v1:" + "0" * 64, "rcv1:" + "0" * 63, "rcv1:" + "G" * 64, 7])
def test_a_missing_or_malformed_pin_is_refused_before_anything_is_read(pin):
    err = refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions, store((REF_PIN, REF_DOCS)), pin,
                  lineage=lineage(REF_PIN), programs=programs())
    integrity(err)


def test_a_pin_outside_the_approved_lineage_is_refused():
    err = refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions, store((REF_PIN, REF_DOCS)), REF_PIN,
                  lineage=lineage(SCRIPT_PIN), programs=programs())
    integrity(err)
    refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions, store((REF_PIN, REF_DOCS)), REF_PIN,
            lineage=None, programs=programs())


def test_a_pin_the_store_does_not_hold_is_refused():
    err = refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions, store(), REF_PIN, lineage=lineage(REF_PIN),
                  programs=programs())
    integrity(err)


# --- tamper tests: altered assertion, altered expected value, swapped executor, wrong program hash ------------------

def _set(docs, i, path, value):
    node = docs[i]
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value


TAMPERS = {
    "expected_value": lambda d: _set(d, 0, ["predicate", "expected"], "Hacked"),
    "operator": lambda d: _set(d, 0, ["predicate", "op"], "CONTAINS"),
    "locator": lambda d: _set(d, 0, ["subject", "locator", "test_id"], "other"),
    "requirement_ref": lambda d: _set(d, 0, ["requirement_ref"], "REQ-OTHER"),
    "requirement_ref_removed": lambda d: d[0].pop("requirement_ref"),
    "pattern": lambda d: _set(d, 1, ["subject", "pattern"], "/other$"),
    "verifier_id": lambda d: _set(d, 0, ["verifier_id"], "other-verifier"),
    "assertion_removed": lambda d: d.pop(),
    "assertion_added": lambda d: d.append({**copy.deepcopy(d[0]), "assertion_id": "a-extra"}),
    "alternate_added": lambda d: _set(d, 0, ["subject", "subject_alternates"], [{"frame_path": [], "test_id": "x"}]),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_an_edited_assertion_under_the_pinned_hash_is_refused_and_nothing_runs(name):
    docs = tampered(TAMPERS[name])
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    err = refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions, store((REF_PIN, docs)), REF_PIN,
                  lineage=lineage(REF_PIN), programs=programs())
    integrity(err)
    assert adapter.assert_calls == []


def test_the_control_with_no_edit_loads():
    assert load(tampered(lambda d: None)).assertion_set_hash == REF_PIN


def test_a_swapped_executor_set_does_not_hash_to_the_playwright_pin():
    swapped = vec("qa-assertion-set:script_replaces_playwright")["assertions"]
    refs = {(d["check_program_ref"]["id"], d["check_program_ref"]["version"]): d["check_program_ref"]["program_hash"]
            for d in swapped if "check_program_ref" in d}
    err = refused("ASSERTION_HASH_MISMATCH", V.load_frozen_assertions, store((REF_PIN, swapped)), REF_PIN,
                  lineage=lineage(REF_PIN), programs=InMemoryProgramRegistry(refs))
    integrity(err)


def test_a_result_with_a_swapped_executor_is_refused():
    frozen = load()
    res = V.DeterministicVerifier(programs=programs(), qa_adapter=ReadOnlyBrowserAdapter(ref_answers())) \
        .verify_assertion(frozen, "a-saved")
    bad = res.to_json()
    bad["executor"] = "SCRIPT"
    err = refused("EXECUTOR_MISMATCH", V.check_result_matches_assertion, bad, frozen.get("a-saved"))
    integrity(err)
    err = refused("EXECUTOR_MISMATCH", V.check_result_matches_assertion, R.AssertionResult.from_json(bad),
                  frozen.get("a-saved"))
    integrity(err)


@pytest.mark.parametrize("value", ["WASM", "playwright", "", None, 3, ["PLAYWRIGHT"]])
def test_a_result_with_an_unknown_executor_is_non_certifying_and_refused(value):
    frozen = load()
    res = V.DeterministicVerifier(programs=programs(), qa_adapter=ReadOnlyBrowserAdapter(ref_answers())) \
        .verify_assertion(frozen, "a-saved")
    bad = res.to_json()
    bad["executor"] = value
    refused("EXECUTOR_MISMATCH", V.check_result_matches_assertion, bad, frozen.get("a-saved"))


def test_a_result_computed_against_another_expected_value_is_refused():
    frozen = load()
    a = frozen.get("a-saved")
    res = V.DeterministicVerifier(programs=programs(), qa_adapter=ReadOnlyBrowserAdapter(ref_answers())) \
        .verify_assertion(frozen, "a-saved")
    assert res.expected_hash == EQ_SAVED_HASH == V.expected_hash(a)
    bad = res.to_json()
    bad["expected_hash"] = V.expected_hash(frozen.get("a-url"))
    integrity(refused("EXPECTED_MISMATCH", V.check_result_matches_assertion, bad, a))
    V.check_result_matches_assertion(res.to_json(), a, assertion_set_hash=REF_PIN)          # control
    refused("COVERAGE_VIOLATION", V.check_result_matches_assertion, res, frozen.get("a-url"))
    other = pin_of("qa-assertion-set:expected_changed")
    refused("ASSERTION_HASH_MISMATCH", V.check_result_matches_assertion, res, a, assertion_set_hash=other)


def test_the_expected_hash_is_the_canon_hash_of_the_frozen_predicate():
    for a in ref_specs():
        assert V.expected_hash(a) == canon.to_contract_ref(canon.canon_hash(
            {"op": a.predicate.op.value, "expected": a.predicate.expected}))


# --- check_program_ref (O7) -------------------------------------------------------------------------------------

def load_script(registry=None, docs=SCRIPT_DOCS, **kw):
    return V.load_frozen_assertions(store((SCRIPT_PIN, docs)), SCRIPT_PIN, lineage=lineage(SCRIPT_PIN),
                                    programs=registry if registry is not None else programs(), **kw)


def test_a_script_assertion_with_the_registered_program_hash_loads():
    assert load_script().get("a-image-pinned").check_program_ref.program_hash == PROGRAM_HASH


def test_a_script_assertion_without_check_program_ref_fails_spec_validation_at_load():
    docs = copy.deepcopy(SCRIPT_DOCS)
    docs[0].pop("check_program_ref")
    err = refused("CHECK_PROGRAM_REF_REQUIRED", load_script, docs=docs)
    assert err.run_outcome is None                      # a spec validation failure (TEST_DEFECT), not an integrity halt


def test_a_playwright_assertion_that_carries_a_check_program_ref_is_rejected():
    docs = copy.deepcopy(REF_DOCS)
    docs[0]["check_program_ref"] = copy.deepcopy(SCRIPT_DOCS[0]["check_program_ref"])
    refused("SPEC_INVALID", V.load_frozen_assertions, store((REF_PIN, docs)), REF_PIN, lineage=lineage(REF_PIN),
            programs=programs())


def test_a_program_hash_that_differs_from_the_registered_program_ends_integrity_before_any_execution():
    check = ScriptedCheckAdapter(observed("pinned"))
    wrong = InMemoryProgramRegistry({PROGRAM_KEY: "sha256:" + "ab" * 32})
    err = refused("ASSERTION_HASH_MISMATCH", load_script, wrong)
    integrity(err)
    assert check.calls == []


def test_an_unregistered_program_ends_integrity_before_any_execution():
    integrity(refused("ASSERTION_HASH_MISMATCH", load_script, InMemoryProgramRegistry({})))


def test_a_program_hash_edited_in_the_stored_assertion_does_not_hash_to_the_pin():
    docs = copy.deepcopy(SCRIPT_DOCS)
    docs[0]["check_program_ref"]["program_hash"] = "sha256:" + "ab" * 32
    registry = InMemoryProgramRegistry({PROGRAM_KEY: "sha256:" + "ab" * 32})           # registry agrees with the edit
    integrity(refused("ASSERTION_HASH_MISMATCH", load_script, registry, docs))          # but the pin does not


class FlippableRegistry:
    def __init__(self, h):
        self.h = h

    def registered_program_hash(self, program_id, version):
        return self.h


def test_the_program_hash_is_checked_again_immediately_before_each_execution():
    registry = FlippableRegistry(PROGRAM_HASH)
    frozen = load_script(registry)
    check = ScriptedCheckAdapter(observed("pinned"))
    verifier = V.DeterministicVerifier(programs=registry, check_adapter=check)
    assert verifier.verify_assertion(frozen, "a-image-pinned").check.verdict is R.Verdict.PASS
    registry.h = "sha256:" + "cd" * 32                  # the registry changes after load
    n = len(check.calls)
    integrity(refused("ASSERTION_HASH_MISMATCH", verifier.verify_assertion, frozen, "a-image-pinned"))
    assert len(check.calls) == n and verifier.halt.tripped


def test_a_script_assertion_runs_through_the_check_adapter_with_the_program_ref_and_inputs_only():
    frozen = load_script()
    check = ScriptedCheckAdapter(observed("pinned", obs="obs-c"))
    verifier = V.DeterministicVerifier(programs=programs(), check_adapter=check)
    res = verifier.verify_assertion(frozen, "a-image-pinned", workspace="W", check_tier=R.CheckTier.K)
    a = frozen.get("a-image-pinned")
    (w, subject), = check.calls
    assert w == "W" and isinstance(subject, R.CheckSubject)
    assert subject.check_program_ref == a.check_program_ref and subject.inputs == a.subject.inputs
    assert subject.tier is R.CheckTier.K
    assert res.check.verdict is R.Verdict.PASS and res.executor is R.Executor.SCRIPT
    assert res.expected_hash == V.expected_hash(a)


# --- invalid specs are test defects, not evaluated ---------------------------------------------------------------

def _doc(**changes):
    d = copy.deepcopy(REF_DOCS[0])
    d.update(changes)
    return d


@pytest.mark.parametrize("changes", [
    {"predicate": {"op": "EQUALS"}},                                           # needs an expected value
    {"predicate": {"op": "EXISTS", "expected": "x"}},                          # takes none
    {"predicate": {"op": "MATCHES", "expected": "("}},                         # does not compile
    {"predicate": {"op": "MATCHES", "expected": 5}},
    {"predicate": {"op": "LESS_THAN", "expected": "abc"}},
    {"subject": {"kind": "RESOURCE_INPUTS", "inputs": [{"kind": "IMAGE_DIGEST", "digest": "sha256:" + "0" * 64}]}},
    {"subject": {"kind": "UI_ELEMENT"}},
    {"subject": {"kind": "URL"}},
])
def test_an_assertion_the_verifier_cannot_evaluate_is_a_spec_validation_failure(changes):
    doc = _doc(**changes)
    specs = (R.AssertionSpec.from_json(doc),)
    pin = V.assertion_set_pin(specs)
    err = refused("SPEC_INVALID", V.load_frozen_assertions, store((pin, [doc])), pin, lineage=lineage(pin),
                  programs=programs())
    assert err.run_outcome is None


def test_an_unknown_executor_in_a_stored_assertion_is_refused_at_load():
    refused("UNKNOWN_ENUM_VALUE", V.load_frozen_assertions, store((REF_PIN, [_doc(executor="WASM")])), REF_PIN,
            lineage=lineage(REF_PIN), programs=programs())


# --- the read-only stores ----------------------------------------------------------------------------------------

def test_the_in_memory_store_returns_copies_and_cannot_be_written():
    s = store((REF_PIN, REF_DOCS))
    got = s.get_assertion_set(REF_PIN)
    got[0]["predicate"]["expected"] = "Hacked"
    assert s.get_assertion_set(REF_PIN)[0]["predicate"]["expected"] == "Saved"
    with pytest.raises(AttributeError):
        s.anything = 1
    assert not [n for n in dir(s) if n.lstrip("_") in ("put", "write", "set", "delete", "add", "update")]
    with pytest.raises(KeyError):
        s.get_assertion_set("not-a-hash")


def _write(dirpath, pin, text):
    path = dirpath / (canon.digest_of(pin) + ".json")
    path.write_text(text, encoding="utf-8")
    return path


def test_the_directory_store_loads_arrays_and_assertion_set_documents(tmp_path):
    import json
    _write(tmp_path, REF_PIN, json.dumps(REF_DOCS))
    store_dir = DirectorySpecStore(tmp_path)
    frozen = V.load_frozen_assertions(store_dir, REF_PIN, lineage=lineage(REF_PIN), programs=programs())
    assert frozen.assertion_set_hash == REF_PIN
    doc = vec("qa-assertion-set:reference")
    _write(tmp_path, REF_PIN, json.dumps(doc))
    assert V.load_frozen_assertions(store_dir, REF_PIN, lineage=lineage(REF_PIN), programs=programs()) == frozen


def test_a_spec_file_edited_between_load_and_verify_is_refused_with_nothing_evaluated(tmp_path):
    import json
    path = _write(tmp_path, REF_PIN, json.dumps(REF_DOCS))
    store_dir = DirectorySpecStore(tmp_path)
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    verifier = V.DeterministicVerifier(programs=programs(), qa_adapter=adapter)
    frozen = V.load_frozen_assertions(store_dir, REF_PIN, lineage=lineage(REF_PIN), programs=programs(),
                                      halt=verifier.halt)
    path.write_text(json.dumps(tampered(TAMPERS["expected_value"])), encoding="utf-8")        # the edit
    integrity(refused("ASSERTION_HASH_MISMATCH", verifier.verify_run, store_dir, REF_PIN, lineage=lineage(REF_PIN)))
    assert adapter.assert_calls == []
    assert frozen.get("a-saved").predicate.expected == "Saved"                               # the loaded set is unchanged


def test_the_directory_store_refuses_duplicate_keys_oversize_missing_and_traversal(tmp_path):
    s = DirectorySpecStore(tmp_path)
    _write(tmp_path, REF_PIN, '[{"assertion_id": "a", "assertion_id": "b"}]')
    refused("SPEC_INVALID", s.get_assertion_set, REF_PIN)
    _write(tmp_path, REF_PIN, "x" * (4 * 1024 * 1024 + 2))
    refused("SPEC_INVALID", s.get_assertion_set, REF_PIN)
    _write(tmp_path, REF_PIN, '{"schema_id": "runtime-contracts/qa/test-spec", "schema_version": "0.1", "assertions": []}')
    refused("SPEC_INVALID", s.get_assertion_set, REF_PIN)
    _write(tmp_path, REF_PIN, '"just a string"')
    refused("SPEC_INVALID", s.get_assertion_set, REF_PIN)
    with pytest.raises(KeyError):
        s.get_assertion_set("rcv1:" + "9" * 64)
    with pytest.raises(KeyError):
        s.get_assertion_set("../../etc/passwd")


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs a non-root user to prove read-only")
def test_the_directory_store_works_on_a_read_only_directory(tmp_path):
    import json
    path = _write(tmp_path, REF_PIN, json.dumps(REF_DOCS))
    path.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        assert len(DirectorySpecStore(tmp_path).get_assertion_set(REF_PIN)) == 2
    finally:
        tmp_path.chmod(0o755)


# --- predicate semantics (pure) ----------------------------------------------------------------------------------

def P(op, expected=None):
    return R.Predicate.from_json({"op": op, **({"expected": expected} if expected is not None else {})})


def OV(value=None, **kw):
    return observed(value, **kw)


PASS, FAIL, IND = R.Verdict.PASS, R.Verdict.FAIL, R.Verdict.INDETERMINATE


@pytest.mark.parametrize("op,expected,value,verdict", [
    ("EQUALS", "Saved", "Saved", PASS), ("EQUALS", "Saved", "saved", FAIL), ("EQUALS", 1, 1, PASS),
    ("EQUALS", 1, True, FAIL), ("EQUALS", True, 1, FAIL), ("EQUALS", "1", 1, FAIL), ("EQUALS", 1, "1", FAIL),
    ("EQUALS", "é", "é", PASS), ("EQUALS", {"a": [1, 2]}, {"a": [1, 2]}, PASS),
    ("EQUALS", {"a": [1, 2]}, {"a": [2, 1]}, FAIL), ("EQUALS", [1], [1, 2], FAIL), ("EQUALS", False, False, PASS),
    ("NOT_EQUALS", "x", "y", PASS), ("NOT_EQUALS", "x", "x", FAIL), ("NOT_EQUALS", 1, True, PASS),
    ("CONTAINS", "av", "Saved", PASS), ("CONTAINS", "zz", "Saved", FAIL), ("CONTAINS", 2, [1, 2, 3], PASS),
    ("CONTAINS", "2", [1, 2, 3], FAIL), ("CONTAINS", "k", {"k": 1}, PASS), ("CONTAINS", "k", {"j": 1}, FAIL),
    ("CONTAINS", 1, "abc", IND), ("CONTAINS", "a", 5, IND),
    ("MATCHES", "/saved$", "https://x/saved", PASS), ("MATCHES", "/saved$", "https://x/saved/", FAIL),
    ("MATCHES", "^a", "ba", FAIL), ("MATCHES", "a", 5, IND),
    ("LESS_THAN", 10, 9, PASS), ("LESS_THAN", 10, 10, FAIL), ("LESS_THAN", "10", "9.5", PASS),
    ("GREATER_THAN", "1.50", 2, PASS), ("GREATER_THAN", "1.50", 1, FAIL), ("GREATER_THAN", 5, 6, PASS), ("GREATER_THAN", 5, "4", FAIL),
    ("GREATER_THAN", 5, "abc", IND), ("LESS_THAN", 5, True, IND), ("LESS_THAN", 5, [1], IND),
])
def test_predicate_semantics(op, expected, value, verdict):
    assert V.evaluate_predicate(P(op, expected), OV(value))[0] is verdict


def test_exists_and_absent_work_on_a_uniquely_resolved_subject_only():
    assert V.evaluate_predicate(P("EXISTS"), OV("x")) == (PASS, None)
    assert V.evaluate_predicate(P("EXISTS"), OV(None)) == (PASS, None)
    assert V.evaluate_predicate(P("ABSENT"), OV(None)) == (PASS, None)
    assert V.evaluate_predicate(P("ABSENT"), OV("x")) == (FAIL, None)
    assert V.evaluate_predicate(P("ABSENT"), OV(None, digest="rcv1:" + "1" * 64)) == (FAIL, None)
    assert V.evaluate_predicate(P("ABSENT"), OV(None, resolution="UNRESOLVED", count=0))[0] is IND
    assert V.evaluate_predicate(P("EXISTS"), OV(None, resolution="UNRESOLVED", count=0))[0] is IND


@pytest.mark.parametrize("resolution,count,reason", [
    ("UNRESOLVED", 0, "SUBJECT_UNRESOLVED"), ("AMBIGUOUS", 3, "SUBJECT_AMBIGUOUS"), ("AMBIGUOUS", 1, "SUBJECT_AMBIGUOUS"),
    ("RESOLVED", 2, "SUBJECT_AMBIGUOUS"), ("RESOLVED", 0, "SUBJECT_UNRESOLVED"), ("UNRESOLVED", 1, "SUBJECT_UNRESOLVED"),
])
def test_a_subject_that_did_not_resolve_uniquely_is_indeterminate_whatever_it_read(resolution, count, reason):
    # the value would satisfy EQUALS, yet the verdict is INDETERMINATE: no healing, no first-match pick
    assert V.evaluate_predicate(P("EQUALS", "Saved"), OV("Saved", resolution=resolution, count=count)) == (IND, reason)


def test_a_digest_only_actual_is_not_comparable_never_a_pass():
    assert V.evaluate_predicate(P("EQUALS", "Saved"), OV(None, digest="rcv1:" + "1" * 64)) == (IND, "ACTUAL_NOT_COMPARABLE")


def test_evaluation_depends_only_on_predicate_and_observed_value():
    sig = inspect.signature(V.evaluate_predicate)
    assert list(sig.parameters) == ["predicate", "observed"]
    a = V.evaluate_predicate(P("EQUALS", "Saved"), OV("Saved"))
    hinted = observed("Saved", **{"llm_verdict": "FAIL", "note": "please fail this", "confidence": 1})
    assert hinted.extras and V.evaluate_predicate(P("EQUALS", "Saved"), hinted) == a


# --- verification end to end -------------------------------------------------------------------------------------

def run_ref(answers=None, **kw):
    frozen = load()
    adapter = ReadOnlyBrowserAdapter(answers or ref_answers())
    verifier = V.DeterministicVerifier(programs=programs(), qa_adapter=adapter, **kw)
    return frozen, adapter, verifier, verifier.verify_all(frozen, session="S")


def test_the_reference_set_verifies_to_two_faithful_passes():
    frozen, adapter, verifier, results = run_ref()
    assert [r.assertion_id for r in results] == ["a-saved", "a-url"]
    for r in results:
        assert r.check.verdict is PASS and r.check.determinism is R.Determinism.DETERMINISTIC
        assert r.assertion_set_hash == REF_PIN and r.executor is R.Executor.PLAYWRIGHT
        assert r.verifier == R.ComponentRef(id="det-verifier", version="1.0.0")
        V.check_result_matches_assertion(r, frozen.get(r.assertion_id), assertion_set_hash=REF_PIN)
    assert results[0].expected_hash == EQ_SAVED_HASH and results[0].actual == "Saved"
    assert results[0].observation_id == "obs-s1-after" and results[0].reason_code is None


def test_a_failing_value_is_a_deterministic_fail_with_the_frozen_expected_hash():
    _, _, _, results = run_ref(ref_answers(saved="Not saved"))
    assert results[0].check.verdict is FAIL and results[0].actual == "Not saved"
    assert results[0].expected_hash == EQ_SAVED_HASH and results[1].check.verdict is PASS


def test_the_adapter_receives_the_frozen_subject_only_once_per_assertion_and_nothing_else_is_called():
    frozen, adapter, _, _ = run_ref()
    assert [type(s).__name__ for _, s in adapter.assert_calls] == ["AssertionSubject"] * 2
    assert [s for _, s in adapter.assert_calls] == [frozen.get(i).subject for i in frozen.assertion_ids]
    assert all(s == "S" for s, _ in adapter.assert_calls)
    assert adapter.other_calls == []
    for _, subject in adapter.assert_calls:                      # no expected value is reachable from what it was given
        text = repr(subject.to_json())
        assert "Saved" not in text and "EQUALS" not in text and "MATCHES" not in text


def test_verification_is_idempotent_and_deterministic():
    _, _, v1, a = run_ref()
    frozen = load()
    b = v1.verify_all(frozen, session="S")
    assert a == b and [r.canon_hash() for r in a] == [r.canon_hash() for r in b]


def test_evidence_refs_and_observed_extras_cannot_change_a_verdict():
    plain = run_ref()[3]
    frozen = load()
    adapter = ReadOnlyBrowserAdapter({"save-status": observed("Saved", model_says="PASS"),
                                      "/saved$": observed("https://app.example/saved", verdict="PASS")})
    v = V.DeterministicVerifier(programs=programs(), qa_adapter=adapter)
    with_refs = [v.verify_assertion(frozen, i, session="S", evidence_refs=("shot-1@1#sha256:" + "6" * 64,))
                 for i in frozen.assertion_ids]
    assert [r.check.verdict for r in with_refs] == [r.check.verdict for r in plain]
    assert [r.reason_code for r in with_refs] == [r.reason_code for r in plain]


def test_no_verify_entry_point_takes_an_expected_value_a_verdict_or_free_text():
    banned = {"expected", "verdict", "hint", "prompt", "judgement", "judgment", "text", "explanation", "model", "llm"}
    for fn in (V.DeterministicVerifier.verify_assertion, V.DeterministicVerifier.verify_all,
               V.DeterministicVerifier.verify_run, V.evaluate_predicate, V.load_frozen_assertions):
        assert not (set(inspect.signature(fn).parameters) & banned), fn


def test_adapter_errors_become_indeterminate_results_with_the_error_code():
    frozen = load()
    adapter = ReadOnlyBrowserAdapter({"save-status": BrowserCrash("boom"), "/saved$": observed("https://x/saved")})
    r = V.DeterministicVerifier(programs=programs(), qa_adapter=adapter).verify_assertion(frozen, "a-saved", session="S")
    assert (r.check.verdict, r.reason_code, r.actual) == (IND, "BROWSER_CRASH", None)


def test_a_malformed_adapter_answer_is_indeterminate_never_trusted():
    frozen = load()
    for bad in ({"verdict": "PASS"}, "PASS", None, 7):
        adapter = ReadOnlyBrowserAdapter({"save-status": bad})
        r = V.DeterministicVerifier(programs=programs(), qa_adapter=adapter).verify_assertion(frozen, "a-saved")
        assert (r.check.verdict, r.reason_code) == (IND, "OBSERVED_VALUE_INVALID")


def test_a_missing_adapter_for_the_executor_is_indeterminate_and_calls_nothing():
    frozen = load()
    r = V.DeterministicVerifier(programs=programs()).verify_assertion(frozen, "a-saved")
    assert (r.check.verdict, r.reason_code) == (IND, "EXECUTOR_UNAVAILABLE")
    sframe = load_script()
    r = V.DeterministicVerifier(programs=programs(), qa_adapter=ReadOnlyBrowserAdapter()).verify_assertion(sframe, "a-image-pinned")
    assert (r.check.verdict, r.reason_code) == (IND, "EXECUTOR_UNAVAILABLE")


def test_an_assertion_for_another_verifier_or_version_is_not_evaluated():
    frozen = load()
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    for kw in ({"verifier_id": "other"}, {"verifier_version": "2.0.0"}, {"verifier_version": "0.9"},
               {"verifier_version": "x"}):
        r = V.DeterministicVerifier(programs=programs(), qa_adapter=adapter, **kw).verify_assertion(frozen, "a-saved")
        assert (r.check.verdict, r.reason_code) == (IND, "VERIFIER_MISMATCH")
    assert adapter.assert_calls == []


@pytest.mark.parametrize("version,spec,ok", [
    ("1.0.0", ">=1.0,<2", True), ("1.9.9", ">=1.0,<2", True), ("2.0.0", ">=1.0,<2", False), ("0.9", ">=1.0", False),
    ("1.0", "==1.0.0", True), ("1.0.1", "!=1.0.1", False), ("1", ">0", True), ("1.0", "~=1.0", False),
    ("1.0", "*", False), ("1.0", "", False), ("1.0", ">=1.0,", False), ("1.0", ">= 1.0", True), ("abc", ">=1.0", False),
    ("1.0", None, False),
])
def test_version_ranges_fail_closed(version, spec, ok):
    assert V.version_in_range(version, spec) is ok


def test_verifying_an_assertion_that_is_not_in_the_set_is_a_coverage_violation():
    frozen = load()
    err = refused("COVERAGE_VIOLATION", V.DeterministicVerifier(programs=programs()).verify_assertion, frozen, "a-nope")
    integrity(err)


# --- manifest against spec ---------------------------------------------------------------------------------------

def test_a_manifest_of_this_spec_verifies_and_a_manifest_of_another_assertion_set_does_not():
    m = manifest()
    V.verify_manifest_against_spec(m, ref_specs())
    other = tuple(R.AssertionSpec.from_json(d) for d in vec("qa-assertion-set:expected_changed")["assertions"])
    integrity(refused("ASSERTION_HASH_MISMATCH", V.verify_manifest_against_spec, m, other))


def test_a_manifest_whose_results_do_not_match_the_frozen_assertions_is_refused():
    j = manifest().to_json()
    j["assertion_results"][0]["executor"] = "SCRIPT"
    m = R.RunManifest.from_json(j)
    integrity(refused("EXECUTOR_MISMATCH", V.verify_manifest_against_spec, m, ref_specs()))


# --- mutation tests for the verifier's own checks ----------------------------------------------------------------

from ._fx_verify import Mutants   # noqa: E402


@pytest.fixture
def mutants():
    m = Mutants()
    yield m
    m.close()


def _swapped_executor_refused(mod):
    frozen = load()
    res = V.DeterministicVerifier(programs=programs(), qa_adapter=ReadOnlyBrowserAdapter(ref_answers())) \
        .verify_assertion(frozen, "a-saved").to_json()
    res["executor"] = "SCRIPT"
    try:
        mod.check_result_matches_assertion(res, frozen.get("a-saved"))
    except QAError as exc:
        return exc.code == "EXECUTOR_MISMATCH"
    return False


def _unknown_executor_refused(mod):
    frozen = load()
    res = V.DeterministicVerifier(programs=programs(), qa_adapter=ReadOnlyBrowserAdapter(ref_answers())) \
        .verify_assertion(frozen, "a-saved").to_json()
    res["executor"] = "WASM"
    try:
        mod.check_result_matches_assertion(res, frozen.get("a-saved"))
    except QAError as exc:
        return exc.code == "EXECUTOR_MISMATCH"
    return False


def _expected_refused(mod):
    frozen = load()
    res = V.DeterministicVerifier(programs=programs(), qa_adapter=ReadOnlyBrowserAdapter(ref_answers())) \
        .verify_assertion(frozen, "a-saved").to_json()
    res["expected_hash"] = V.expected_hash(frozen.get("a-url"))
    try:
        mod.check_result_matches_assertion(res, frozen.get("a-saved"))
    except QAError as exc:
        return exc.code == "EXPECTED_MISMATCH"
    return False


def _program_hash_refused(mod):
    wrong = InMemoryProgramRegistry({PROGRAM_KEY: "sha256:" + "ab" * 32})
    try:
        mod.load_frozen_assertions(store((SCRIPT_PIN, SCRIPT_DOCS)), SCRIPT_PIN, lineage=mod.ApprovedLineage([SCRIPT_PIN]),
                                   programs=wrong)
    except QAError as exc:
        return exc.code == "ASSERTION_HASH_MISMATCH"
    return False


def _program_rechecked_before_execution(mod):
    registry = FlippableRegistry(PROGRAM_HASH)
    frozen = mod.load_frozen_assertions(store((SCRIPT_PIN, SCRIPT_DOCS)), SCRIPT_PIN,
                                        lineage=mod.ApprovedLineage([SCRIPT_PIN]), programs=registry)
    check = ScriptedCheckAdapter(observed("pinned"))
    verifier = mod.DeterministicVerifier(programs=registry, check_adapter=check)
    registry.h = "sha256:" + "cd" * 32
    try:
        verifier.verify_assertion(frozen, "a-image-pinned")
    except QAError:
        return check.calls == []
    return False


def _declared_ids_checked(mod):
    """The spec has three assertions, the manifest pins that set but declares only two: only this check notices."""
    extra = {**copy.deepcopy(REF_DOCS[0]), "assertion_id": "a-extra"}
    spec3 = tuple(R.AssertionSpec.from_json(d) for d in copy.deepcopy(REF_DOCS) + [extra])
    pin3 = V.assertion_set_pin(spec3)
    j = manifest_json_with_pin(pin3)
    try:
        mod.verify_manifest_against_spec(R.RunManifest.from_json(j), spec3)
    except QAError:
        return True
    return False


def manifest_json_with_pin(pin):
    from ._fx_verify import manifest_json
    j = manifest_json()
    j["assertion_set_hash"] = pin
    for r in j["assertion_results"]:
        r["assertion_set_hash"] = pin
    return j


VERIFIER_MUTATIONS = {
    "executor_check": (_swapped_executor_refused, "    if result.executor is not assertion.executor:", "    if False:"),
    "unknown_executor_check": (_unknown_executor_refused, "    if isinstance(result, Mapping):\n        raw_executor = result.get(\"executor\")\n        if raw_executor not in tuple(e.value for e in Executor):",
                               "    if isinstance(result, Mapping):\n        raw_executor = result.get(\"executor\")\n        if False:"),
    "expected_check": (_expected_refused, "    if not _same(result.expected_hash, expected_hash(assertion)):", "    if False:"),
    "program_hash_check": (_program_hash_refused, "    if registered != ref.program_hash:", "    if False:"),
    "program_recheck": (_program_rechecked_before_execution, "        _check_program(assertion, self._programs)          # again, immediately before execution (registry may change)", "        pass"),
    "declared_equals_spec": (_declared_ids_checked, "    if set(manifest.assertions_declared) != {a.assertion_id for a in items}:", "    if False:"),
}


@pytest.mark.parametrize("name", sorted(VERIFIER_MUTATIONS))
def test_removing_a_verifier_check_is_caught(name, mutants):
    probe, old, new = VERIFIER_MUTATIONS[name]
    assert probe(V)
    mutant = mutants.load("verifier.py", [(old, new)])
    assert not probe(mutant), f"mutation {name} survived"


def test_mutating_predicate_evaluation_is_caught(mutants):
    cases = [("EQUALS", "Saved", "Saved", PASS), ("EQUALS", "Saved", "Other", FAIL), ("NOT_EQUALS", "x", "y", PASS),
             ("CONTAINS", "av", "Saved", PASS), ("MATCHES", "/saved$", "x/saved", PASS), ("MATCHES", "/saved$", "x/saved/", FAIL),
             ("LESS_THAN", 10, 9, PASS), ("LESS_THAN", 10, 10, FAIL), ("GREATER_THAN", 10, 9, FAIL),
             ("GREATER_THAN", 10, 10, FAIL), ("EQUALS", 1, True, FAIL)]

    def probe(mod):
        return all(mod.evaluate_predicate(P(op, e), OV(v))[0] is verdict for op, e, v, verdict in cases)

    assert probe(V)
    for old, new in (
        ("    if op is PredicateOp.NOT_EQUALS:\n        return (Verdict.FAIL if json_equal(expected, actual) else Verdict.PASS), None",
         "    if op is PredicateOp.NOT_EQUALS:\n        return (Verdict.PASS if json_equal(expected, actual) else Verdict.FAIL), None"),
        ("        holds = left < right if op is PredicateOp.LESS_THAN else left > right", "        holds = left <= right if op is PredicateOp.LESS_THAN else left >= right"),
        ("    if type(a) is bool or type(b) is bool:\n        return type(a) is type(b) and a == b", "    if type(a) is bool or type(b) is bool:\n        return a == b"),
        ("            hit = re.search(expected, _nfc(actual)) is not None", "            hit = re.fullmatch(expected, _nfc(actual)) is not None"),
        ("        return (Verdict.PASS if json_equal(expected, actual) else Verdict.FAIL), None\n    if op is PredicateOp.NOT_EQUALS",
         "        return Verdict.PASS, None\n    if op is PredicateOp.NOT_EQUALS"),
    ):
        assert not probe(mutants.load("verifier.py", [(old, new)])), old
