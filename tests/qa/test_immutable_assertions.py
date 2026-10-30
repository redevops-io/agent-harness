# SPDX-License-Identifier: AGPL-3.0-or-later
"""The immutable-assertion rule (ADR section 6), mechanism by mechanism, each proven by a mutation test.

Every mechanism has a *probe*: a function that takes the module(s) under test and answers "does the mechanism hold?".
The real modules must pass their probes; a copy of the module with the mechanism removed (``load_mutant``: exact-text
replacement, each anchor occurring once) must fail them. A mutant that survives means the suite would not notice the
mechanism going missing.

  1 type separation            4 capability partition     7 no assertion healing
  2 hash pinning               5 adapter never sees expected  8 changes only through review (lineage)
  3 verifier loads by hash     6 read-only spec store     9 stop rule (halt the batch)
"""
import ast
import dataclasses
import itertools
import typing

import pytest

from agent_harness.qa import classify as C, ports, records as R, recovery_types as RT, specstore as SS, verifier as V
from agent_harness.qa.errors import (AssertionHashMismatch, QAError, RecoveryPayloadRejected, SpecInvalid)

from ._h1 import (MANIFEST, REF_DOCS, REF_PIN, Mutants, ReadOnlyBrowserAdapter, facts, lineage, manifest,
                  manifest_json, mutated_source, observed, programs, ref_answers, ref_specs, store, vec)
from ._helpers import QA_SRC


@pytest.fixture
def mutants():
    m = Mutants()
    yield m
    m.close()


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception:
        return False
    return False


def holds(probe, *args, **kwargs):
    """A probe that raises counts as "the mechanism does not hold" (a mutant may break the code path it probes)."""
    try:
        return bool(probe(*args, **kwargs))
    except Exception:
        return False


def tampered_docs():
    import copy
    docs = copy.deepcopy(REF_DOCS)
    docs[0]["predicate"]["expected"] = "Hacked"
    return docs


def new_verifier(mod=V, adapter=None, **kw):
    return mod.DeterministicVerifier(programs=programs(), qa_adapter=adapter or ReadOnlyBrowserAdapter(ref_answers()), **kw)


def loader(mod, docs=REF_DOCS, pin=REF_PIN, **kw):
    kw.setdefault("lineage", mod.ApprovedLineage([pin]))
    kw.setdefault("programs", programs())
    return mod.load_frozen_assertions(store((pin, docs)), pin, **kw)


# =============================================================================================================
# 1. type separation
# =============================================================================================================

def probe_type_separation(records_mod=R, verifier_mod=V):
    spec = records_mod.AssertionSpec.from_json(REF_DOCS[0])
    frozen_fields = [(spec, "assertion_id"), (spec.predicate, "expected"), (spec.subject, "kind"), (spec.subject.locator, "test_id")]
    for obj, name in frozen_fields:
        if not raises((dataclasses.FrozenInstanceError, AttributeError), setattr, obj, name, "x"):
            return False
    frozen_set = loader(verifier_mod)
    if not raises((dataclasses.FrozenInstanceError, AttributeError), setattr, frozen_set, "assertions", ()):
        return False
    if not isinstance(frozen_set.assertions, tuple):
        return False
    # the only mutable runtime facts live on the receipt, a different type, which carries no assertion field
    receipt = {f.name for f in dataclasses.fields(records_mod.ActionReceipt)}
    assertion = {f.name for f in dataclasses.fields(records_mod.AssertionSpec)}
    return ({"locator_used", "attempt"} <= receipt and not ({"locator_used", "attempt"} & assertion)
            and not ({"predicate", "assertion_id", "expected"} & receipt))


def test_mechanism_1_type_separation(mutants):
    assert probe_type_separation()
    records_mutant = mutants.load("records.py", [       # a dataclass cannot mix frozen and non-frozen bases: change all three
        ("@dataclass(frozen=True, kw_only=True)\nclass _Value:", "@dataclass(kw_only=True)\nclass _Value:"),
        ("def _value(cls):\n    return dataclass(frozen=True, kw_only=True)(cls)", "def _value(cls):\n    return dataclass(kw_only=True)(cls)"),
        ("        return dataclass(frozen=True, kw_only=True)(cls)\n    return deco", "        return dataclass(kw_only=True)(cls)\n    return deco"),
    ])
    assert not holds(probe_type_separation, records_mod=records_mutant)
    verifier_mutant = mutants.load("verifier.py", [("@dataclass(frozen=True)\nclass FrozenAssertionSet:",
                                                    "@dataclass\nclass FrozenAssertionSet:")])
    assert not holds(probe_type_separation, verifier_mod=verifier_mutant)


# =============================================================================================================
# 2. hash pinning before execution
# =============================================================================================================

class Unpinned:
    """Looks like a frozen set (it has the assertions) but was never pinned or recomputed."""

    def __init__(self, docs):
        self.assertions = tuple(R.AssertionSpec.from_json(d) for d in docs)
        self.assertion_set_hash = REF_PIN

    def get(self, assertion_id):
        return next(a for a in self.assertions if a.assertion_id == assertion_id)


def probe_pinning(mod=V):
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    verifier = new_verifier(mod, adapter)
    unpinned = Unpinned(tampered_docs())
    refused_one = raises(AssertionHashMismatch, verifier.verify_assertion, unpinned, "a-saved", session="S")
    refused_all = raises(AssertionHashMismatch, new_verifier(mod, adapter).verify_all, unpinned, session="S")
    refused_raw = raises(AssertionHashMismatch, new_verifier(mod, adapter).verify_assertion, list(unpinned.assertions),
                         "a-saved")
    no_pin = raises(AssertionHashMismatch, mod.load_frozen_assertions, store((REF_PIN, REF_DOCS)), None,
                    lineage=mod.ApprovedLineage([REF_PIN]), programs=programs())
    return refused_one and refused_all and refused_raw and no_pin and adapter.assert_calls == []


def test_mechanism_2_hash_pinning(mutants):
    assert probe_pinning()
    mutant = mutants.load("verifier.py", [("    if type(frozen) is not FrozenAssertionSet:\n        raise AssertionHashMismatch(\"only a pinned FrozenAssertionSet can be verified\")\n    return frozen",
                                           "    return frozen")])
    assert not holds(probe_pinning, mutant)


def test_mechanism_2_a_frozen_set_cannot_be_built_unpinned_or_from_other_content():
    for pin in (None, "", "canon-v1:" + "0" * 64, "rcv1:" + "0" * 64):
        assert raises(AssertionHashMismatch, V.FrozenAssertionSet, assertion_set_hash=pin, assertions=ref_specs())
    assert raises(AssertionHashMismatch, V.FrozenAssertionSet, assertion_set_hash=REF_PIN,
                  assertions=tuple(R.AssertionSpec.from_json(d) for d in tampered_docs()))
    assert raises(SpecInvalid, V.FrozenAssertionSet, assertion_set_hash=REF_PIN, assertions=())
    assert raises(SpecInvalid, V.FrozenAssertionSet, assertion_set_hash=REF_PIN, assertions=("a-saved",))
    assert V.FrozenAssertionSet(assertion_set_hash=REF_PIN, assertions=ref_specs()).assertion_ids == ("a-saved", "a-url")


# =============================================================================================================
# 3. the verifier loads by hash and recomputes with canon-v1
# =============================================================================================================

def probe_recompute(mod=V):
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    refuses_load = raises(AssertionHashMismatch, loader, mod, tampered_docs())
    verifier = new_verifier(mod, adapter)
    edited = store((REF_PIN, tampered_docs()))
    refuses_verify = raises(AssertionHashMismatch, verifier.verify_run, edited, REF_PIN, lineage=mod.ApprovedLineage([REF_PIN]))
    specs = tuple(R.AssertionSpec.from_json(d) for d in tampered_docs())
    refuses_direct = raises(AssertionHashMismatch, mod.FrozenAssertionSet, assertion_set_hash=REF_PIN, assertions=specs)
    other = tuple(R.AssertionSpec.from_json(d) for d in vec("qa-assertion-set:expected_changed")["assertions"])
    refuses_manifest = raises(AssertionHashMismatch, mod.verify_manifest_against_spec, manifest(), other)
    control_loads = loader(mod).assertion_set_hash == REF_PIN
    return refuses_load and refuses_verify and refuses_direct and refuses_manifest and control_loads and adapter.assert_calls == []


def test_mechanism_3_verifier_loads_by_hash_and_recomputes(mutants):
    assert probe_recompute()
    mutant = mutants.load("verifier.py", [("    if not _same(recomputed, pin):\n        raise AssertionHashMismatch(\"the assertion set does not hash",
                                           "    if False:\n        raise AssertionHashMismatch(\"the assertion set does not hash")])
    assert not holds(probe_recompute, mutant)


def test_mechanism_3_the_verifier_fetches_only_by_the_pinned_hash():
    asked = []

    class Spy:
        def get_assertion_set(self, h):
            asked.append(h)
            return REF_DOCS

    V.load_frozen_assertions(Spy(), REF_PIN, lineage=lineage(REF_PIN), programs=programs())
    assert asked == [REF_PIN]
    assert list(__import__("inspect").signature(V.load_frozen_assertions).parameters)[:2] == ["store", "pinned_assertion_set_hash"]


# =============================================================================================================
# 4. capability partition: the recovery interface cannot carry an assertion
# =============================================================================================================

ASSERTION_TYPES = {"AssertionSpec", "AssertionResult", "Predicate", "ObservedValue", "Check", "FrozenAssertionSet",
                   "AssertionSubject", "TestSpec", "RunManifest", "CheckSubject"}
ASSERTION_FIELD_NAMES = {"expected", "expected_hash", "predicate", "assertion", "assertions", "assertion_id", "verdict",
                         "check", "verifier", "requirement_ref", "assertion_set_hash"}


def reachable(cls, seen=None):
    """All dataclass types reachable from ``cls`` through field annotations."""
    seen = seen if seen is not None else {}
    if cls in seen or not dataclasses.is_dataclass(cls):
        return seen
    seen[cls] = typing.get_type_hints(cls, include_extras=True)
    for hint in list(seen[cls].values()):
        stack = [hint]
        while stack:
            h = stack.pop()
            stack.extend(typing.get_args(h))
            if isinstance(h, type) and dataclasses.is_dataclass(h):
                reachable(h, seen)
    return seen


@dataclasses.dataclass(frozen=True, kw_only=True)
class FatCandidate(R.LocatorCandidate):
    expected: str = "Hacked"


def recovery_surface(ports_mod=ports, records_mod=R):
    return [ports_mod.RecoveryContext, records_mod.LocatorCandidate]


def probe_partition(ports_mod=ports, records_mod=R, recovery_mod=RT):
    for root in recovery_surface(ports_mod, records_mod):
        graph = reachable(root)
        names = {c.__name__ for c in graph}
        if names & ASSERTION_TYPES:
            return False
        for cls in graph:
            if {f.name for f in dataclasses.fields(cls)} & ASSERTION_FIELD_NAMES:
                return False
    ctx = ports_mod.RecoveryContext
    if {f.name for f in dataclasses.fields(ctx)} != {"step", "observation"}:
        return False
    if not raises(RecoveryPayloadRejected, records_mod.LocatorCandidate.from_json,
                  {"strategy": "SEMANTIC_ALTERNATE", "role": "button", "expected": "x"}):
        return False
    # the single gate: hostile output of every kind is rejected, the strict candidate is not
    spec = R.AssertionSpec.from_json(REF_DOCS[0])
    hostile = [
        [{"strategy": "SEMANTIC_ALTERNATE", "role": "button", "expected": "Hacked"}],
        [{"strategy": "SEMANTIC_ALTERNATE", "role": "button", "extra": 1}],
        [spec], [spec.to_json()], [{"strategy": "MODEL_CANDIDATE", "assertion": spec.to_json()}],
        {"strategy": "SEMANTIC_ALTERNATE", "role": "button"}, "role=button", None,
        [{"strategy": "SEMANTIC_ALTERNATE", "role": "x", "predicate": {"op": "EQUALS", "expected": "x"}}],
        [FatCandidate(strategy="SEMANTIC_ALTERNATE")],
    ]
    for payload in hostile:
        if not raises(RecoveryPayloadRejected, recovery_mod.parse_recovery_output, payload):
            return False
    good = recovery_mod.parse_recovery_output([{"strategy": "SEMANTIC_ALTERNATE", "role": "button", "name": "Save"}])
    return len(good) == 1 and good[0].role == "button"


def test_mechanism_4_capability_partition(mutants):
    assert probe_partition()
    # (a) a recovery context that grows an assertion-ish field or a field of an assertion type
    m1 = mutants.load("ports.py", [("    observation: Observation\n", "    observation: Observation\n    expected: str = \"\"\n")])
    assert not holds(probe_partition, ports_mod=m1)
    m2 = mutants.load("ports.py", [("from .records import (ActionOutcome,", "from .records import (AssertionSpec, ActionOutcome,"),
                                   ("    observation: Observation\n", "    observation: Observation\n    carried: AssertionSpec = None\n")])
    assert not holds(probe_partition, ports_mod=m2)
    m3 = mutants.load("records.py", [("    ALLOW_EXTRAS: ClassVar[bool] = False\n\n    @classmethod\n    def from_json(cls, data: Any, *, _path: Optional[str] = None):\n        try:",
                                      "    @classmethod\n    def from_json(cls, data: Any, *, _path: Optional[str] = None):\n        try:")])
    assert not holds(probe_partition, records_mod=m3)
    m4 = mutants.load("recovery_types.py", [("        elif isinstance(item, Mapping):\n            out.append(LocatorCandidate.from_json(item))                # strict: unknown keys are rejected",
                                             "        elif isinstance(item, Mapping):\n            out.append(item)")])
    assert not holds(probe_partition, recovery_mod=m4)
    m5 = mutants.load("recovery_types.py", [("    if not isinstance(raw, (list, tuple)):\n        raise RecoveryPayloadRejected(f\"recovery output must be a list of locator candidates, got {type(raw).__name__}\")\n",
                                             "")])
    assert not holds(probe_partition, recovery_mod=m5)
    m6 = mutants.load("recovery_types.py", [("        if type(item) is LocatorCandidate:\n            out.append(LocatorCandidate.from_json(item.to_json()))      # re-validated through the strict path",
                                             "        if isinstance(item, LocatorCandidate):\n            out.append(item)")])
    assert not holds(probe_partition, recovery_mod=m6)


def test_mechanism_4_candidate_schema_is_strict_in_the_records():
    assert R.LocatorCandidate.ALLOW_EXTRAS is False
    for extra in ({"expected": "x"}, {"assertion": {}}, {"predicate": {}}, {"selector": "css=x"}, {"verdict": "PASS"}):
        assert raises(RecoveryPayloadRejected, R.LocatorCandidate.from_json,
                      {"strategy": "SEMANTIC_ALTERNATE", "role": "button", **extra})
    assert raises(RecoveryPayloadRejected, R.LocatorCandidate, strategy="SEMANTIC_ALTERNATE", extras={"expected": "x"})


class HostileRecoverer:
    def __init__(self, out):
        self.out = out
        self.seen = None

    def propose(self, ctx):
        self.seen = ctx
        return self.out


def _context():
    s = R.TestSpec.from_json(vec("qa:test_spec_ui"))
    obs = R.Observation.from_json(vec("qa:observation"))
    return ports.RecoveryContext(step=s.steps[0], observation=obs)


@pytest.mark.parametrize("name,out", [
    ("changed_expected", [{"strategy": "SEMANTIC_ALTERNATE", "role": "button", "expected": "Hacked"}]),
    ("extra_field", [{"strategy": "SEMANTIC_ALTERNATE", "role": "button", "css": "#x"}]),
    ("assertion_object", [R.AssertionSpec.from_json(REF_DOCS[0])]),
    ("assertion_json", [REF_DOCS[0]]),
    ("single_object_not_a_list", {"strategy": "SEMANTIC_ALTERNATE", "role": "button"}),
    ("too_many", [{"strategy": "SEMANTIC_ALTERNATE", "role": "button"}] * (RT.MAX_CANDIDATES_PER_CALL + 1)),
    ("a_string", "click the other button"),
])
def test_a_hostile_recoverer_is_rejected_and_the_run_is_inconclusive_assertion_integrity(name, out):
    rec = HostileRecoverer(out)
    with pytest.raises(RecoveryPayloadRejected) as err:
        RT.propose_safely(rec, _context())
    assert (err.value.run_outcome, err.value.run_reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY")
    assert err.value.code == "RECOVERY_PAYLOAD_REJECTED"
    # and the classifier turns that refusal into the halt
    assert C.classify(facts(error_codes=(err.value.code,))).halt_batch


def test_a_strict_candidate_list_passes_and_the_recoverer_saw_only_the_step_and_the_observation():
    rec = HostileRecoverer([{"strategy": "SEMANTIC_ALTERNATE", "role": "button", "name": "Save"},
                            R.LocatorCandidate(strategy="MODEL_CANDIDATE", label="Save")])
    ctx = _context()
    out = RT.propose_safely(rec, ctx)
    assert [c.strategy.value for c in out] == ["SEMANTIC_ALTERNATE", "MODEL_CANDIDATE"] and rec.seen is ctx
    with pytest.raises(RecoveryPayloadRejected):
        RT.propose_safely(rec, object())
    assert raises((dataclasses.FrozenInstanceError, AttributeError), setattr, ctx, "step", None)


def test_a_subclassed_candidate_cannot_smuggle_a_field():
    assert raises(RecoveryPayloadRejected, RT.parse_recovery_output, [FatCandidate(strategy="SEMANTIC_ALTERNATE")])


# =============================================================================================================
# 5. the adapter never sees expected values
# =============================================================================================================

def probe_no_expected_to_adapter(mod=V):
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    frozen = loader(mod)
    new_verifier(mod, adapter).verify_all(frozen, session="S")
    if len(adapter.assert_calls) != 2:
        return False
    for _, subject in adapter.assert_calls:
        if type(subject) is not R.AssertionSubject:
            return False
        text = repr(subject.to_json()) + repr(subject)
        if any(word in text for word in ("Saved", "EQUALS", "MATCHES", "predicate", "expected")):
            return False
    return adapter.other_calls == []


def test_mechanism_5_adapter_never_sees_expected(mutants):
    assert probe_no_expected_to_adapter()
    mutant = mutants.load("verifier.py", [("observed = self._qa.assert_(session, assertion.subject)       # the subject only: no expected value",
                                           "observed = self._qa.assert_(session, assertion)")])
    assert not holds(probe_no_expected_to_adapter, mutant)


def test_mechanism_5_the_ports_have_no_expected_in_and_no_verdict_out():
    for proto, name in ((ports.NeutralQAAdapter, "assert_"), (ports.NeutralCheckAdapter, "run_check"),
                        (V.SubjectReader, "assert_"), (V.CheckRunner, "run_check")):
        sig = __import__("inspect").signature(getattr(proto, name))
        params = [p for p in sig.parameters if p != "self"]
        assert len(params) == 2 and not any("expect" in p or "verdict" in p or "predicate" in p for p in params)
        assert sig.return_annotation in ("ObservedValue", R.ObservedValue)
    assert not {"expected", "predicate", "verdict"} & {f.name for f in dataclasses.fields(R.ObservedValue)}
    assert not {"expected", "predicate", "verdict"} & {f.name for f in dataclasses.fields(R.CheckSubject)}
    assert not {"expected", "predicate", "verdict"} & {f.name for f in dataclasses.fields(R.AssertionSubject)}


def test_mechanism_5_the_verifier_can_only_read_through_assert_and_run_check():
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    new_verifier(adapter=adapter).verify_all(loader(V), session="S")
    assert adapter.other_calls == []
    src = (QA_SRC / "verifier.py").read_text(encoding="utf-8")
    called = {n.func.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Attribute) and n.func.value.attr in ("_qa", "_check")}
    assert called == {"assert_", "run_check"}


# =============================================================================================================
# 6. read-only spec store
# =============================================================================================================

WRITE_CALLS = {"put", "write", "write_text", "write_bytes", "unlink", "rename", "truncate", "chmod", "mkdir", "rmdir",
               "touch", "symlink_to", "hardlink_to"}


def open_modes(source):
    """The mode of every ``open()`` call in ``source``, plus a marker for every call of a file-writing method."""
    modes = []
    for n in ast.walk(ast.parse(source)):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Name) and n.func.id == "open":
            mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
            modes.append(mode.value if isinstance(mode, ast.Constant) else None)
        elif isinstance(n.func, ast.Attribute) and n.func.attr in WRITE_CALLS:
            modes.append("WRITE-CALL:" + n.func.attr)
    return modes


def probe_read_only(ss=SS, specstore_source=None):
    src = specstore_source if specstore_source is not None else (QA_SRC / "specstore.py").read_text(encoding="utf-8")
    if set(open_modes(src)) != {"rb"}:
        return False
    ops = [n for n, v in vars(ss.SpecStore).items() if callable(v) and not n.startswith("_")]
    if ops != ["get_assertion_set"]:
        return False
    s = ss.InMemorySpecStore({REF_PIN: REF_DOCS})
    if not raises(AttributeError, setattr, s, "_docs", {}) or not raises(AttributeError, setattr, s, "put", 1):
        return False
    got = s.get_assertion_set(REF_PIN)
    got[0]["assertion_id"] = "mutated"
    if s.get_assertion_set(REF_PIN)[0]["assertion_id"] != "a-saved":
        return False
    reg = ss.InMemoryProgramRegistry({("a", "1"): "sha256:" + "0" * 64})
    return raises(AttributeError, setattr, reg, "_hashes", {})


def test_mechanism_6_read_only_spec_store(mutants):
    assert probe_read_only()
    src = (QA_SRC / "specstore.py").read_text(encoding="utf-8")
    assert not holds(probe_read_only, specstore_source=mutated_source("specstore.py", [('open(path, "rb")', 'open(path, "r+b")')]))
    assert not holds(probe_read_only, specstore_source=mutated_source("specstore.py", [('open(path, "rb")', 'open(path, "wb")')]))
    m1 = mutants.load("specstore.py", [("    def get_assertion_set(self, assertion_set_hash: str) -> Sequence[Mapping[str, Any]]: ...\n",
                                        "    def get_assertion_set(self, assertion_set_hash: str) -> Sequence[Mapping[str, Any]]: ...\n\n    def put(self, assertion_set_hash: str, docs: Sequence[Mapping[str, Any]]) -> None: ...\n")])
    assert not holds(probe_read_only, m1, src)
    m2 = mutants.load("specstore.py", [('        raise AttributeError("InMemorySpecStore is read-only")', "        object.__setattr__(self, name, value)")])
    assert not holds(probe_read_only, m2, src)
    m3 = mutants.load("specstore.py", [('        raise AttributeError("InMemoryProgramRegistry is read-only")', "        object.__setattr__(self, name, value)")])
    assert not holds(probe_read_only, m3, src)


@pytest.mark.parametrize("name", ["verifier.py", "classify.py", "specstore.py", "recovery_types.py"])
def test_mechanism_6_no_h1_module_writes_files_or_mutates_a_store(name):
    src = (QA_SRC / name).read_text(encoding="utf-8")
    tree = ast.parse(src)
    attr_calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not attr_calls & {"put", "write", "write_text", "write_bytes", "unlink", "rename", "truncate", "chmod",
                             "mkdir", "rmdir", "touch", "symlink_to", "hardlink_to"}
    assert set(open_modes(src)) <= {"rb"}


# =============================================================================================================
# 7. no assertion healing
# =============================================================================================================

def probe_no_healing(mod=V):
    P = R.Predicate.from_json({"op": "EQUALS", "expected": "Saved"})
    for resolution, count in (("AMBIGUOUS", 3), ("UNRESOLVED", 0), ("RESOLVED", 2), ("RESOLVED", 0), ("AMBIGUOUS", 1)):
        verdict, reason = mod.evaluate_predicate(P, observed("Saved", resolution=resolution, count=count))
        if verdict is not R.Verdict.INDETERMINATE or reason not in ("SUBJECT_UNRESOLVED", "SUBJECT_AMBIGUOUS"):
            return False
    # the verifier asks once per assertion, with the frozen subject itself, and never retries with another
    adapter = ReadOnlyBrowserAdapter({"save-status": observed("Saved", resolution="AMBIGUOUS", count=2),
                                      "/saved$": observed(None, resolution="UNRESOLVED", count=0)})
    frozen = loader(mod)
    results = new_verifier(mod, adapter).verify_all(frozen, session="S")
    if [r.check.verdict for r in results] != [R.Verdict.INDETERMINATE] * 2:
        return False
    return (len(adapter.assert_calls) == 2
            and [s for _, s in adapter.assert_calls] == [frozen.get("a-saved").subject, frozen.get("a-url").subject])


def test_mechanism_7_no_assertion_healing(mutants):
    assert probe_no_healing()
    anchor = "    if observed.resolution is not Resolution.RESOLVED or observed.match_count != 1:"
    for new in ("    if False:", "    if observed.resolution is not Resolution.RESOLVED:", "    if observed.match_count != 1:"):
        assert not holds(probe_no_healing, mutants.load("verifier.py", [(anchor, new)])), new


# =============================================================================================================
# 8. changes only through review: a pin outside the approved lineage is refused
# =============================================================================================================

def probe_lineage(mod=V):
    other = vec("qa-assertion-set:expected_changed")["assertions"]
    from agent_harness.qa import verifier as real
    h = real.assertion_set_pin(tuple(R.AssertionSpec.from_json(d) for d in other))
    approved = mod.ApprovedLineage([REF_PIN])
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    refused_unapproved = raises(AssertionHashMismatch, mod.load_frozen_assertions, store((h, other)), h,
                                lineage=approved, programs=programs())
    refused_no_lineage = raises(AssertionHashMismatch, mod.load_frozen_assertions, store((REF_PIN, REF_DOCS)), REF_PIN,
                                lineage=None, programs=programs())
    approved_ok = mod.load_frozen_assertions(store((REF_PIN, REF_DOCS)), REF_PIN, lineage=approved,
                                            programs=programs()).assertion_set_hash == REF_PIN
    immutable = raises((dataclasses.FrozenInstanceError, AttributeError), setattr, approved,
                       "approved_assertion_set_hashes", frozenset({h}))
    return refused_unapproved and refused_no_lineage and approved_ok and immutable and adapter.assert_calls == []


def test_mechanism_8_changes_only_through_review(mutants):
    assert probe_lineage()
    mutant = mutants.load("verifier.py", [("        if not isinstance(lineage, ApprovedLineage) or not lineage.contains(pin):",
                                           "        if False:")])
    assert not holds(probe_lineage, mutant)


def test_mechanism_8_a_changed_assertion_is_a_new_set_with_a_new_hash_that_must_be_approved_on_its_own():
    changed = vec("qa-assertion-set:expected_changed")["assertions"]
    new_pin = V.assertion_set_pin(tuple(R.AssertionSpec.from_json(d) for d in changed))
    assert new_pin != REF_PIN
    unapproved = V.ApprovedLineage([REF_PIN])
    assert raises(AssertionHashMismatch, V.load_frozen_assertions, store((new_pin, changed)), new_pin, lineage=unapproved,
                  programs=programs())
    approved = V.ApprovedLineage([REF_PIN, new_pin])
    assert V.load_frozen_assertions(store((new_pin, changed)), new_pin, lineage=approved, programs=programs()).get(
        "a-saved").predicate.expected == "Different"


# =============================================================================================================
# 9. stop rule: an integrity refusal halts the batch
# =============================================================================================================

def probe_stop_rule(mod=V):
    latch = mod.HaltLatch()
    adapter = ReadOnlyBrowserAdapter(ref_answers())
    verifier = new_verifier(mod, adapter, halt=latch)
    first = raises(AssertionHashMismatch, mod.load_frozen_assertions, store((REF_PIN, tampered_docs())), REF_PIN,
                   lineage=mod.ApprovedLineage([REF_PIN]), programs=programs(), halt=latch)
    if not first or not latch.tripped:
        return False
    # everything after it in the batch is refused, even a perfectly good load and verification
    good_load = raises(mod.BatchHalted, mod.load_frozen_assertions, store((REF_PIN, REF_DOCS)), REF_PIN,
                       lineage=mod.ApprovedLineage([REF_PIN]), programs=programs(), halt=latch)
    good_frozen = V.load_frozen_assertions(store((REF_PIN, REF_DOCS)), REF_PIN, lineage=V.ApprovedLineage([REF_PIN]),
                                           programs=programs())
    good_verify = raises(mod.BatchHalted, verifier.verify_assertion, good_frozen, "a-saved", session="S")
    return good_load and good_verify and adapter.assert_calls == [] and isinstance(latch.cause, AssertionHashMismatch)


def test_mechanism_9_stop_rule(mutants):
    assert probe_stop_rule()
    m1 = mutants.load("verifier.py", [("        if exc is not None and isinstance(exc, QAError) and exc.run_reason == REASON_ASSERTION_INTEGRITY:",
                                       "        if False:")])
    assert not holds(probe_stop_rule, m1)
    m2 = mutants.load("verifier.py", [("        if self._cause is None:\n            object.__setattr__(self, \"_cause\", cause)",
                                       "        pass")])
    assert not holds(probe_stop_rule, m2)
    m3 = mutants.load("verifier.py", [("        if self._cause is not None:\n            raise BatchHalted(", "        if False:\n            raise BatchHalted(")])
    assert not holds(probe_stop_rule, m3)


def test_mechanism_9_only_integrity_refusals_halt_and_the_latch_cannot_be_reset():
    latch = V.HaltLatch()
    bad = {**REF_DOCS[0], "executor": "SCRIPT"}
    assert raises(QAError, V.load_frozen_assertions, store((REF_PIN, [bad])), REF_PIN, lineage=lineage(REF_PIN),
                  programs=programs(), halt=latch)                        # a spec validation failure (TEST_DEFECT)
    assert not latch.tripped
    assert raises(AssertionHashMismatch, V.load_frozen_assertions, store(), REF_PIN, lineage=lineage(REF_PIN),
                  programs=programs(), halt=latch)
    assert latch.tripped
    assert raises(AttributeError, setattr, latch, "_cause", None)
    assert not [n for n in dir(latch) if n.lstrip("_") in ("reset", "clear", "release", "untrip")]
    err = pytest.raises(V.BatchHalted, latch.check).value
    assert (err.run_outcome, err.run_reason, err.code) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY", "BATCH_HALTED")


def test_mechanism_9_the_classifier_halts_on_every_integrity_refusal_code():
    from agent_harness.qa.errors import ERRORS_BY_CODE
    codes = [c for c, k in ERRORS_BY_CODE.items() if k.run_reason == "ASSERTION_INTEGRITY"]
    assert set(codes) == {"ASSERTION_HASH_MISMATCH", "EXPECTED_MISMATCH", "EXECUTOR_MISMATCH",
                          "RECOVERY_PAYLOAD_REJECTED", "COVERAGE_VIOLATION"}
    for code in codes:
        c = C.classify(facts(error_codes=(code,)))
        assert (c.outcome.value, c.reason, c.halt_batch) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY", True)


# =============================================================================================================
# seeded true regression: whatever the attestations, the verdict is never PASS
# =============================================================================================================

def test_a_seeded_true_regression_is_never_a_pass_whatever_the_runner_attests_or_recovers():
    states = (C.Signal.OK, C.Signal.FAILED, C.Signal.UNKNOWN)
    names = ("environment_before", "environment_after", "preconditions", "fixture_hash", "cleanup", "policy",
             "recovery_checks", "failure_reproduced")
    for strategy in ("MODEL_CANDIDATE", "SEMANTIC_LOCATOR", "RETRY_SAME"):
        j = manifest_json(verdicts={"a-saved": "FAIL"}, outcome="PRODUCT_FAIL")
        j["fault_events"][0]["recovery"]["strategy"] = strategy
        m = R.RunManifest.from_json(j)
        for combo in itertools.product(states, repeat=len(names)):
            c = C.classify(facts(m=m, **dict(zip(names, combo))))
            assert c.outcome is not R.Outcome.PASS, (strategy, combo)
            assert c.verdict is not R.Verdict.PASS
    # and a run that claims PASS while holding that failure is an integrity error, not a pass
    claimed = C.classify(facts(m=manifest(verdicts={"a-saved": "FAIL"}, outcome="PASS")))
    assert (claimed.outcome.value, claimed.reason) == ("INCONCLUSIVE", "ASSERTION_INTEGRITY")


def test_no_model_input_path_reaches_the_verifier_or_the_classifier():
    banned_roots = {"llm", "openai", "anthropic", "agent", "loop", "tools", "guardrails", "spend", "bench"}
    for name in ("verifier.py", "classify.py", "specstore.py", "recovery_types.py"):
        tree = ast.parse((QA_SRC / name).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom):
                mods = [("." * n.level) + (n.module or "")]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            for m in mods:
                top = m.lstrip(".").split(".")[0]
                assert top not in banned_roots, (name, m)
                assert not m.startswith("agent_harness.") or m.startswith("agent_harness.qa"), (name, m)


# =============================================================================================================
# control: every probe holds on an unmodified copy of the module it probes (so a mutant fails for the mutation)
# =============================================================================================================

def test_every_probe_holds_on_an_unmodified_copy_of_its_module(mutants):
    v, r = mutants.load("verifier.py", []), mutants.load("records.py", [])
    p, rt, ss = mutants.load("ports.py", []), mutants.load("recovery_types.py", []), mutants.load("specstore.py", [])
    assert holds(probe_type_separation, records_mod=r) and holds(probe_type_separation, verifier_mod=v)
    assert holds(probe_pinning, v) and holds(probe_recompute, v) and holds(probe_no_expected_to_adapter, v)
    assert holds(probe_no_healing, v) and holds(probe_lineage, v) and holds(probe_stop_rule, v)
    assert holds(probe_partition, ports_mod=p) and holds(probe_partition, records_mod=r)
    assert holds(probe_partition, recovery_mod=rt)
    assert holds(probe_read_only, ss, (QA_SRC / "specstore.py").read_text(encoding="utf-8"))
