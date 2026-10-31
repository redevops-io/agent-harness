# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixtures for the verifier, classifier and recovery-type tests, built from the golden vectors vendored under
``golden/qa_canon_vectors.json``: the reference assertion sets, the manifest, the pinned hashes."""
import copy
import sys

from agent_harness.qa import canon, records as R, verifier as V
from agent_harness.qa.classify import RunFacts, Signal
from agent_harness.qa.specstore import InMemoryProgramRegistry, InMemorySpecStore
from agent_harness.qa.verifier import ApprovedLineage

from ._helpers import QA_SRC, load_mutant, load_vectors

VECTORS = {v["id"]: v for v in load_vectors()["vectors"]}


def vec(vid):
    return copy.deepcopy(VECTORS[vid]["input"])


def pin_of(vid):
    return "rcv1:" + VECTORS[vid]["sha256_hex"]


#: the reference assertion set: a-saved (EQUALS "Saved" on a UI element) and a-url (MATCHES "/saved$" on a URL)
REF_DOCS = vec("qa-assertion-set:reference")["assertions"]
REF_PIN = pin_of("qa-assertion-set:reference")
#: a SCRIPT assertion set: a-image-pinned (EQUALS "pinned"), check program image-pinned@1.0.0
SCRIPT_DOCS = vec("qa-assertion-set:script_assertion")["assertions"]
SCRIPT_PIN = pin_of("qa-assertion-set:script_assertion")
PROGRAM_KEY = ("image-pinned", "1.0.0")
PROGRAM_HASH = SCRIPT_DOCS[0]["check_program_ref"]["program_hash"]

EQ_SAVED_HASH = "rcv1:1b995d3503f549f89ae235791af06f1923a99bb30ec7904e2e77921cd2830c80"


def ref_specs():
    return tuple(R.AssertionSpec.from_json(d) for d in vec("qa-assertion-set:reference")["assertions"])


def store(*pairs):
    """InMemorySpecStore holding (pin, docs) pairs."""
    return InMemorySpecStore({pin: docs for pin, docs in pairs})


def lineage(*pins):
    return ApprovedLineage(pins)


def programs(**kw):
    return InMemoryProgramRegistry(kw.get("hashes", {PROGRAM_KEY: PROGRAM_HASH}))


def observed(value="Saved", *, resolution="RESOLVED", count=1, digest=None, obs="obs-s1-after", **extra):
    data = {"resolution": resolution, "match_count": count, "observation_id": obs, **extra}
    if value is not None:
        data["value"] = value
    if digest is not None:
        data["value_digest"] = digest
    return R.ObservedValue.from_json(data)


class ReadOnlyBrowserAdapter:
    """Scripted adapter for the verifier: ``assert_`` answers from a mapping keyed by locator test id / URL pattern.
    Every other operation of the neutral protocol is present and records an attempt, so tests can prove the verifier
    never uses it."""

    adapter_id = "scripted"
    adapter_version = "0"
    api_version = "qa-adapter/1"

    def __init__(self, answers=None, default=None):
        self.answers = dict(answers or {})
        self.default = default
        self.assert_calls = []
        self.other_calls = []

    def assert_(self, s, subject):
        self.assert_calls.append((s, subject))
        key = subject.locator.test_id if subject.locator is not None else subject.pattern
        value = self.answers.get(key, self.default)
        if isinstance(value, Exception):
            raise value
        return value

    def _other(name):
        def method(self, *a, **k):
            self.other_calls.append(name)
            raise AssertionError(f"the verifier must not call {name}")
        return method

    open_session = _other("open_session")
    observe = _other("observe")
    locate = _other("locate")
    act = _other("act")
    capture = _other("capture")
    reset = _other("reset")
    close_session = _other("close_session")


class ScriptedCheckAdapter:
    adapter_id = "scripted-check"
    adapter_version = "0"
    api_version = "qa-check-adapter/1"

    def __init__(self, value=None):
        self.value = value
        self.calls = []

    def run_check(self, w, subject):
        self.calls.append((w, subject))
        return self.value

    def _other(name):
        def method(self, *a, **k):
            raise AssertionError(f"the verifier must not call {name}")
        return method

    open_workspace = _other("open_workspace")
    resolve = _other("resolve")
    capture = _other("capture")
    reset = _other("reset")
    close_workspace = _other("close_workspace")


def ref_answers(saved="Saved", url="https://app.example/saved"):
    return {"save-status": observed(saved), "/saved$": observed(url)}


# --- manifests and facts ------------------------------------------------------------------------------------------

MANIFEST = vec("qa:run_manifest_pass")


def manifest_json(*, verdicts=None, outcome="PASS", **changes):
    """The golden PASS manifest with chosen changes. ``verdicts`` maps assertion id -> verdict of its result."""
    m = copy.deepcopy(MANIFEST)
    for res in m["assertion_results"]:
        if verdicts and res["assertion_id"] in verdicts:
            res["check"]["verdict"] = verdicts[res["assertion_id"]]
    m["outcome"] = outcome
    m.update(changes)
    return m


def manifest(**kw):
    return R.RunManifest.from_json(manifest_json(**kw))


def m_fail(**kw):               # a-saved deterministically FAILED, a-url PASS
    return manifest(verdicts={"a-saved": "FAIL"}, outcome="PRODUCT_FAIL", **kw)


def m_fail_and_indeterminate():
    j = manifest_json(verdicts={"a-saved": "FAIL", "a-url": "INDETERMINATE"}, outcome="PRODUCT_FAIL")
    j["assertion_results"][1]["reason_code"] = "SUBJECT_UNRESOLVED"
    return R.RunManifest.from_json(j)


def m_indeterminate(reason="SUBJECT_UNRESOLVED"):
    j = manifest_json(verdicts={"a-saved": "INDETERMINATE"}, outcome="INCONCLUSIVE")
    j["assertion_results"][0]["reason_code"] = reason
    return R.RunManifest.from_json(j)


def m_skipped(reason="NOT_REACHED"):
    j = manifest_json(outcome="INCONCLUSIVE")
    j["assertions_executed"] = ["a-saved"]
    j["assertions_skipped"] = [{"assertion_id": "a-url", "reason_code": reason}]
    j["assertion_results"] = j["assertion_results"][:1]
    return R.RunManifest.from_json(j)


def m_nondeterministic():
    j = manifest_json(outcome="INCONCLUSIVE")
    j["assertion_results"][0]["check"]["determinism"] = "NON_DETERMINISTIC"
    return R.RunManifest.from_json(j)


def m_no_fault():
    j = manifest_json()
    j["fault_events"] = []
    return R.RunManifest.from_json(j)


def m_with_fault(signal="LOCATOR_NOT_FOUND", result="NOT_RECOVERED", strategy="SEMANTIC_LOCATOR", outcome="PASS",
                 verified=False):
    j = manifest_json(outcome=outcome)
    ev = j["fault_events"][0]
    ev["detection"]["signal"] = signal
    ev["recovery"]["detection_ref"] = "rcv1:" + R.FaultDetection.from_json(ev["detection"]).canon_hash().split(":")[1]
    ev["recovery"].update(result=result, strategy=strategy, postcondition_verified=verified)
    return R.RunManifest.from_json(j)


def m_empty():
    j = manifest_json()
    j.update(assertions_declared=[], assertions_executed=[], assertions_skipped=[], assertion_results=[],
             assertion_set_hash=V.assertion_set_pin(()))
    return R.RunManifest.from_json(j)



def facts(*, m=None, assertions=None, **signals):
    """RunFacts with every attestation OK unless overridden."""
    base = dict(environment_before=Signal.OK, environment_after=Signal.OK, preconditions=Signal.OK,
                fixture_hash=Signal.OK, cleanup=Signal.OK, policy=Signal.OK, recovery_checks=Signal.OK,
                failure_reproduced=Signal.OK)
    base.update(signals)
    return RunFacts(manifest=m if m is not None else manifest(),
                    assertions=assertions if assertions is not None else ref_specs(), **base)


def mutated_source(filename, replacements):
    src = (QA_SRC / filename).read_text(encoding="utf-8")
    for old, new in replacements:
        assert src.count(old) == 1, f"mutation anchor must occur exactly once in {filename}: {old!r}"
        src = src.replace(old, new)
    return src


class Mutants:
    """Loads mutated copies of qa modules and unregisters them afterwards."""

    def __init__(self):
        self.names = []

    def load(self, filename, replacements):
        mod = load_mutant(filename, replacements)
        self.names.append(mod.__name__)
        return mod

    def close(self):
        for n in self.names:
            sys.modules.pop(n, None)
