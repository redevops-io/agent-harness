# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mutation tests for the contract alignment: a copy of a module with one mechanism removed must fail the probe that
the real module passes (the same method as ``test_recovery_mutations.py``).

Each entry names the module, the exact text to replace (it must occur once), and a probe over the module under test.
The probes only look at behaviour the contracts' vectors and fixtures pin."""
import copy
import importlib

import pytest

from agent_harness.qa import canon, records as R
from agent_harness.qa.errors import ActionTimeout

from ._fx_recovery import make_recovery_env, move_save, recover_spec
from ._fx_runner import Fault, make_config, make_env, make_spec, nav_step
from ._fx_verify import (PROGRAM_KEY, REF_DOCS, REF_PIN, SCRIPT_DOCS, SCRIPT_PIN, Mutants, vec)
from ._helpers import load_addendum, load_clarifications
from .test_runner import Hook, clicks, save_step, semantic, timeout_on

ADD = load_addendum()
CLR = load_clarifications()
CASES = {c["id"]: c for c in ADD["set_ordering"]["cases"]}
VALUE_DIGESTS = {e["id"]: e for e in ADD["value_digests"]}


@pytest.fixture
def mutants():
    m = Mutants()
    yield m
    m.close()


def real(filename):
    return importlib.import_module("agent_harness.qa." + filename[:-3])


def holds(probe, module):
    """True if the probe holds; an exception from a mutant counts as 'does not hold'."""
    try:
        return bool(probe(module))
    except Exception:
        return False


def digest_of(mod, obj):
    return "rcv1:" + canon.digest_hex(obj.canonical_form())


# =====================================================================================================================
# probes
# =====================================================================================================================

def _set_case(case_id):
    def probe(mod):
        case = CASES[case_id]
        cls = {"TestSpec": "TestSpec", "RunManifest": "RunManifest", "AssertionResult": "AssertionResult"}[case["record"]]
        obj = getattr(mod, cls).from_json(copy.deepcopy(case["record_listed"]))
        return digest_of(mod, obj) == case["rcv1"]
    return probe


def p_string_sets_sort_after_nfc(mod):
    import unicodedata
    rec = mod.PolicyDecisionRecord(policy_decision_id="d", run_id="r", step_id="s", principal="p", app="a",
                                   decision=mod.PolicyDecisionKind.ALLOW, provider="x", action_kind=mod.ActionKind.CLICK,
                                   side_effect=mod.SideEffect.NONE, scope=("e\u0301", "f"))
    keys = [unicodedata.normalize("NFC", x) for x in rec.scope]
    return keys == sorted(keys) == ["f", "\u00e9"]


def p_enum_sets_sort_by_value(mod):
    step = mod.Step.from_json({**vec("qa-value:step_ui"), "permitted_recovery": ["SEMANTIC_LOCATOR", "RETRY_SAME"]})
    return step.permitted_recovery == (mod.PermittedRecovery.RETRY_SAME, mod.PermittedRecovery.SEMANTIC_LOCATOR)


def p_check_detail_is_wording(mod):
    base = mod.Check(check_id="c", verdict=mod.Verdict.PASS, determinism=mod.Determinism.DETERMINISTIC)
    worded = mod.Check(check_id="c", verdict=mod.Verdict.PASS, determinism=mod.Determinism.DETERMINISTIC, detail="words")
    return worded.canon_hash() == base.canon_hash()


def p_evidence_ref_defaults(mod):
    bare = mod.EvidenceRef(ref="obs-1")
    explicit = mod.EvidenceRef(ref="obs-1", modality="text", ref_type="evidence")
    return bare.canon_hash() == explicit.canon_hash() and bare.canonical_form()["modality"] == "text"


def p_secret_ref_path_default(mod):
    return mod.SecretRef(provider="vault", key="k").canonical_form()["path"] == ""


def p_reason_code_is_identity(mod):
    base = copy.deepcopy(CLR["fault_recovery"]["refusals"][-1]["input"])
    plain = {k: v for k, v in base.items() if k != "reason_code"}
    a, b = mod.FaultRecovery.from_json(plain), mod.FaultRecovery.from_json(base)
    return b.reason_code is mod.RecoveryReason.NO_CANDIDATE and a.canon_hash() != b.canon_hash()


def p_recovered_has_no_reason_code(mod):
    fx = next(f for f in CLR["fault_recovery"]["refusals"] if f["id"] == "recovered_with_reason_code")
    try:
        mod.FaultRecovery.from_json(copy.deepcopy(fx["input"]))
    except Exception as exc:
        return getattr(exc, "code", None) == "SPEC_INVALID"
    return False


def p_rejections_never_exceed_considered(mod):
    fx = next(f for f in CLR["fault_recovery"]["refusals"] if f["id"] == "more_rejections_than_candidates")
    try:
        mod.FaultRecovery.from_json(copy.deepcopy(fx["input"]))
    except Exception as exc:
        return getattr(exc, "code", None) == "SPEC_INVALID"
    return False


def p_engine_state_is_not_a_manifest_field(mod):
    fx = CLR["identity_boundary"]["refusals"][0]
    try:
        mod.RunManifest.from_json(copy.deepcopy(fx["input"]))
    except Exception as exc:
        return getattr(exc, "code", None) == "UNKNOWN_FIELD"
    return False


def p_string_digest(mod):
    entry = VALUE_DIGESTS["string_plain"]
    return (mod.value_digest(entry["value"]) == entry["digest"]
            == mod.action_value_digest(R.Action(kind=R.ActionKind.FILL, value=entry["value"])))


def p_secret_ref_fingerprint(mod):
    entry = VALUE_DIGESTS["secret_ref"]
    action = R.Action(kind=R.ActionKind.FILL, secret_ref=R.SecretRef(**entry["value"]))
    return mod.action_value_digest(action) == entry["digest"]


def p_kind_does_not_enter_the_digest(mod):
    return (mod.action_value_digest(R.Action(kind=R.ActionKind.FILL, value="x"))
            == mod.action_value_digest(R.Action(kind=R.ActionKind.PRESS, value="x")))


def _is_integrity(cls):
    return (cls.run_outcome, cls.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY")


def _class_probe(name):
    return lambda mod: _is_integrity(getattr(mod, name)) and getattr(mod, name).code in mod.INTEGRITY_REFUSAL_CODES


def p_every_code_of_the_vocabulary_has_a_class(mod):
    return all(code == "ASSERTION_INTEGRITY" or mod.ERRORS_BY_CODE[code].run_reason == "ASSERTION_INTEGRITY"
               for code in mod.INTEGRITY_REFUSAL_CODES) and len(mod.INTEGRITY_REFUSAL_CODES) == 11


def p_a_timeout_is_recoverable(mod):
    return mod.ActionTimeout.recoverable is True and mod.ActionTimeout.retryable is True


def p_unknown_codes_fail_closed(mod):
    got = [mod.classify_refusal(c) for c in ("A_CODE_FROM_A_LATER_VERSION", "", None)]
    return all(g is not None and g[1] == "ASSERTION_INTEGRITY" for g in got)


def p_integrity_codes_classify(mod):
    return all(mod.classify_refusal(c) is not None for c in ("ASSERTION_PIN_MALFORMED", "ASSERTION_PIN_NOT_FOUND",
                                                            "ASSERTION_PROGRAM_MISMATCH", "ASSERTION_PROGRAM_UNREGISTERED",
                                                            "BATCH_HALTED", "ASSERTION_HASH_MISMATCH"))


def p_known_non_integrity_codes_do_not_classify(mod):
    return mod.classify_refusal("SPEC_INVALID") is None and mod.classify_refusal("CHECK_PROGRAM_REF_REQUIRED") is None


def _load(mod, pin, docs, *, programs=None, lineage=True):
    from agent_harness.qa.specstore import InMemoryProgramRegistry, InMemorySpecStore
    sets = {pin: docs} if docs is not None and isinstance(pin, str) and canon.is_contract_ref(pin) else {}
    return mod.load_frozen_assertions(InMemorySpecStore(sets), pin,
                                      lineage=mod.ApprovedLineage([pin] if lineage and isinstance(pin, str) and
                                                                  canon.is_contract_ref(pin) else []),
                                      programs=InMemoryProgramRegistry(programs or {}))


def _refusal_code(fn):
    try:
        fn()
    except Exception as exc:
        return getattr(exc, "code", None)
    return None


def p_malformed_pin(mod):
    return _refusal_code(lambda: _load(mod, "rcv1:xyz", REF_DOCS)) == "ASSERTION_PIN_MALFORMED"


def p_pin_not_found(mod):
    return _refusal_code(lambda: _load(mod, REF_PIN, None)) == "ASSERTION_PIN_NOT_FOUND"


def p_program_unregistered(mod):
    return _refusal_code(lambda: _load(mod, SCRIPT_PIN, SCRIPT_DOCS)) == "ASSERTION_PROGRAM_UNREGISTERED"


def p_program_mismatch(mod):
    return _refusal_code(lambda: _load(mod, SCRIPT_PIN, SCRIPT_DOCS, programs={PROGRAM_KEY: "sha256:" + "ab" * 32})
                         ) == "ASSERTION_PROGRAM_MISMATCH"


def p_a_refusal_with_a_new_code_halts_the_batch(mod):
    from agent_harness.qa.specstore import InMemoryProgramRegistry, InMemorySpecStore
    latch = mod.HaltLatch()
    try:
        mod.load_frozen_assertions(InMemorySpecStore({}), REF_PIN, lineage=mod.ApprovedLineage([REF_PIN]),
                                   programs=InMemoryProgramRegistry({}), halt=latch)
    except Exception:
        pass
    return latch.tripped and latch.cause.code == "ASSERTION_PIN_NOT_FOUND"


def p_runner_sets_reason_code(mod):
    env = make_recovery_env(recover_spec(save=()), adapter_setup=move_save, mod=mod)
    r = env.run()
    return r.manifest.fault_events[0].recovery.reason_code is R.RecoveryReason.NOT_PERMITTED


def p_runner_reason_for_a_model_round_or_a_selection_refusal(mod):
    from ._fx_recovery import duplicate_save
    env = make_recovery_env(recover_spec(), adapter_setup=duplicate_save, mod=mod)
    r = env.run()
    return r.manifest.fault_events[0].recovery.reason_code is R.RecoveryReason.AMBIGUOUS_CANDIDATES


def p_runner_does_not_retarget_after_a_possible_effect(mod):
    spec = make_spec([nav_step(), save_step(permitted=["SEMANTIC_LOCATOR"])], allowed_side_effects=["WRITES_INTERNAL"])
    hook = Hook(semantic())
    env = make_env(spec, adapter_setup=timeout_on(1, apply=True), recovery=hook, mod=mod,
                   config_kw={"recovery_enabled": True})
    env.run()
    return hook.requests == [] and len(clicks(env)) == 1


def p_selector_counts_a_refused_proposal_as_considered(mod):
    from .test_recovery import Fixed, cand, observation, step
    selector = mod.LocatorSelector(Fixed(cand(strategy=R.LocatorStrategy.MODEL_CANDIDATE, role="button", name="Save")),
                                   mod.RecoveryLimits())
    got = selector.select(step(), observation(), lambda d: R.LocateResult(candidates=()), lambda: None)
    return len(got.rejected) <= got.considered


# =====================================================================================================================
# the table
# =====================================================================================================================

def M(filename, replacements, probe, why):
    return filename, replacements, probe, why


MUTATIONS = {
    # K2: the order of a set is per field
    "records: object sets are ordered by their element key":
        M("records.py", [("    if key is not None:\n        named = getattr(elem, key)",
                          "    if False:\n        named = getattr(elem, key)")], _set_case("fixtures_by_id"), "per-field ordering"),
    "records: step results are ordered by step_id":
        M("records.py", [("    if key is not None:\n        named = getattr(elem, key)",
                          "    if False:\n        named = getattr(elem, key)")], _set_case("step_results_by_step_id"), "per-field ordering"),
    "records: assertion results are ordered by assertion_id":
        M("records.py", [("    if key is not None:\n        named = getattr(elem, key)",
                          "    if False:\n        named = getattr(elem, key)")], _set_case("assertion_results_by_assertion_id"),
          "per-field ordering"),
    "records: the pins of a check are sorted when hashed":
        M("records.py", [("                value = sorted(value, key=_set_key)", "                pass")],
          _set_case("check_evidence_pins_natural"), "per-field ordering"),
    "records: string sets are ordered after NFC":
        M("records.py", [("    if isinstance(elem, str):\n        return unicodedata.normalize(\"NFC\", elem)",
                          "    if isinstance(elem, str):\n        return elem")], p_string_sets_sort_after_nfc,
          "per-field ordering"),
    "records: enum sets are ordered by member value":
        M("records.py", [("    if isinstance(elem, enum.Enum):\n        return str(elem.value)",
                          "    if isinstance(elem, enum.Enum):\n        return \"\"")],
          p_enum_sets_sort_by_value, "per-field ordering"),
    # K1: the value digest
    "receipts: the digest of a string value is the digest of its canonical text alone":
        M("receipts.py", [("        return json_value_digest(action.value)",
                           "        return canon.to_contract_ref(canon.canon_hash({\"action_value\": {\"value\": action.value}, "
                           "\"kind\": action.kind.value}))")], p_string_digest, "value digest"),
    "receipts: the digest of a secret reference is its fingerprint":
        M("receipts.py", [("        return json_value_digest(action.secret_ref.canonical_form())",
                           "        return canon.to_contract_ref(canon.canon_hash({\"action_value\": {\"secret_ref\": "
                           "action.secret_ref.canonical_form()}, \"kind\": action.kind.value}))")],
          p_secret_ref_fingerprint, "value digest"),
    "receipts: the action kind does not enter the digest":
        M("receipts.py", [("        return json_value_digest(action.value)",
                           "        return json_value_digest([action.kind.value, action.value])")],
          p_kind_does_not_enter_the_digest, "value digest"),
    # A1 - A4
    "records: the detail of a check is wording":
        M("records.py", [("            if info.wording and identity:\n                continue",
                          "            if False:\n                continue")], p_check_detail_is_wording, "A1"),
    "records: an evidence reference hashes with the contracts' defaults":
        M("records.py", [("            out.setdefault(\"modality\", EVIDENCE_REF_DEFAULT_MODALITY)", "            pass")],
          p_evidence_ref_defaults, "A4"),
    "records: a secret reference hashes an absent path as empty":
        M("records.py", [("            out.setdefault(\"path\", \"\")", "            pass")], p_secret_ref_path_default, "A3"),
    # K4
    "records: reason_code is a field of the recovery record":
        M("records.py", [("    reason_code: Optional[RecoveryReason] = None               # why the recovery as a whole did not recover\n",
                          "")], p_reason_code_is_identity, "reason_code"),
    "records: a recovered record carries no reason_code":
        M("records.py", [("raise SpecInvalid(\"FaultRecovery: a RECOVERED recovery carries no reason_code\")", "pass")],
          p_recovered_has_no_reason_code, "reason_code"),
    "records: rejection tokens never outnumber the candidates considered":
        M("records.py", [("raise SpecInvalid(\"FaultRecovery: candidates_rejected_reasons holds at most one code per candidate considered\")",
                          "pass")], p_rejections_never_exceed_considered, "reason_code"),
    "records: engine state is not a manifest field":
        M("records.py", [("            if named:\n                raise UnknownField(", "            if False:\n                raise UnknownField(")],
          p_engine_state_is_not_a_manifest_field, "identity boundary"),
    "recovery: the selector counts a refused proposal as considered":
        M("recovery.py", [("                considered += 1                              # one rejection token per candidate considered, never more\n",
                           "")], p_selector_counts_a_refused_proposal_as_considered, "reason_code"),
    "runner: the recovery record carries its reason_code":
        M("runner.py", [("            reason_code=None if result is RecoveryResult.RECOVERED else reason_code_for(cause),",
                         "            reason_code=None,")], p_runner_sets_reason_code, "reason_code"),
    "runner: a selection refusal is the reason_code, not a rejection":
        M("runner.py", [("        p.cause = selection.refusal or R_NO_CANDIDATE_CAUSE     # why the round as a whole found nothing",
                         "        p.cause = None")], p_runner_reason_for_a_model_round_or_a_selection_refusal, "reason_code"),
    # K3
    "errors: a malformed pin is an integrity refusal":
        M("errors.py", [("class AssertionPinMalformed(_Integrity):", "class AssertionPinMalformed(QAError):")],
          _class_probe("AssertionPinMalformed"), "fail-closed code"),
    "errors: a missing pin is an integrity refusal":
        M("errors.py", [("class AssertionPinNotFound(_Integrity):", "class AssertionPinNotFound(QAError):")],
          _class_probe("AssertionPinNotFound"), "fail-closed code"),
    "errors: an unregistered program is an integrity refusal":
        M("errors.py", [("class AssertionProgramUnregistered(_Integrity):", "class AssertionProgramUnregistered(QAError):")],
          _class_probe("AssertionProgramUnregistered"), "fail-closed code"),
    "errors: a program mismatch is an integrity refusal":
        M("errors.py", [("class AssertionProgramMismatch(_Integrity):", "class AssertionProgramMismatch(QAError):")],
          _class_probe("AssertionProgramMismatch"), "fail-closed code"),
    "errors: a halted batch is an integrity refusal":
        M("errors.py", [("class BatchHalted(_Integrity):", "class BatchHalted(QAError):")],
          _class_probe("BatchHalted"), "fail-closed code"),
    "errors: the vocabulary lists every code":
        M("errors.py", [("    \"ASSERTION_PIN_NOT_FOUND\", \"ASSERTION_PIN_MALFORMED\", \"BATCH_HALTED\",\n)",
                         "    \"ASSERTION_PIN_NOT_FOUND\", \"BATCH_HALTED\",\n)")], p_every_code_of_the_vocabulary_has_a_class, "vocabulary"),
    "errors: a timeout is recoverable":
        M("errors.py", [("    recoverable = True          # recovery still needs enablement, a permitted mode, caps and the idempotency guard\n",
                         "")], p_a_timeout_is_recoverable, "A5"),
    "classify: an unknown refusal code fails closed":
        M("classify.py", [("    return (Outcome.INCONCLUSIVE, ASSERTION_INTEGRITY)\n\n\ndef refusal_halts_batch",
                           "    return None\n\n\ndef refusal_halts_batch")], p_unknown_codes_fail_closed, "fail-closed unknown code"),
    "classify: every integrity code classifies":
        M("classify.py", [("(set(ERRORS_BY_CODE) | set(STRUCTURAL_CODES)) - set(INTEGRITY_REFUSAL_CODES))",
                           "(set(ERRORS_BY_CODE) | set(STRUCTURAL_CODES)))")], p_integrity_codes_classify, "fail-closed new code"),
    "classify: a code known not to be an integrity refusal does not classify":
        M("classify.py", [("    if type(code) is str and code in KNOWN_NON_INTEGRITY_CODES:\n        return None\n",
                           "")], p_known_non_integrity_codes_do_not_classify, "known non-integrity code"),
    "verifier: a malformed pin has its own code":
        M("verifier.py", [("            raise AssertionPinMalformed(", "            raise AssertionHashMismatch(")],
          p_malformed_pin, "distinct code"),
    "verifier: a pin the store does not hold has its own code":
        M("verifier.py", [("            raise AssertionPinNotFound(", "            raise AssertionHashMismatch(")],
          p_pin_not_found, "distinct code"),
    "verifier: an unregistered program has its own code":
        M("verifier.py", [("        raise AssertionProgramUnregistered(", "        raise AssertionHashMismatch(")],
          p_program_unregistered, "distinct code"),
    "verifier: a program under another hash has its own code":
        M("verifier.py", [("        raise AssertionProgramMismatch(", "        raise AssertionHashMismatch(")],
          p_program_mismatch, "distinct code"),
    "verifier: a refusal with a new code still halts the batch":
        M("verifier.py", [("        if exc is not None and isinstance(exc, QAError) and exc.run_reason == REASON_ASSERTION_INTEGRITY:",
                           "        if exc is not None and isinstance(exc, QAError) and exc.code == \"ASSERTION_HASH_MISMATCH\":")],
          p_a_refusal_with_a_new_code_halts_the_batch, "halt on every integrity code"),
    # A5
    "runner: a timeout whose action was sent is not re-targeted":
        M("runner.py", [("        if error.code == \"ACTION_TIMEOUT\" and made.acted:", "        if False:")],
          p_runner_does_not_retarget_after_a_possible_effect, "A5 fail-closed"),
}


def test_every_probe_holds_on_the_real_modules():
    for name, (filename, replacements, probe, why) in MUTATIONS.items():
        assert probe(real(filename)) is True, name


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_removing_the_mechanism_makes_the_probe_fail(name, mutants):
    filename, replacements, probe, why = MUTATIONS[name]
    mutant = mutants.load(filename, replacements)
    assert not holds(probe, mutant), f"mutation survived: {name} ({why})"


def test_the_table_covers_every_item_of_the_alignment():
    wanted = ("per-field ordering", "value digest", "fail-closed code", "fail-closed unknown code", "distinct code",
              "halt on every integrity code", "reason_code", "A1", "A3", "A4", "A5 fail-closed", "A5")
    assert all(any(w == why for _, _, _, why in MUTATIONS.values()) for w in wanted)
    assert {filename for filename, *_ in MUTATIONS.values()} >= {"records.py", "receipts.py", "errors.py", "classify.py",
                                                                  "verifier.py", "runner.py", "recovery.py"}
    assert len(MUTATIONS) >= 30


def test_the_unmutated_copies_pass_every_probe(mutants):
    for filename in {f for f, *_ in MUTATIONS.values()}:
        clone = mutants.load(filename, [])
        for name, (fname, _, probe, _) in MUTATIONS.items():
            if fname == filename:
                assert holds(probe, clone), name
