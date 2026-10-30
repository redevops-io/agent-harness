# SPDX-License-Identifier: AGPL-3.0-or-later
"""The deterministic assertion verifier (plan H1; ADR section 6 and Addendum A.2.4 item 6, A.4.2).

What it does, in order, and what makes each step structural rather than conventional:

1. ``load_frozen_assertions`` takes the *pinned* ``assertion_set_hash`` the bridge-side loader computed (``rcv1``),
   checks that the hash is in the approved lineage, fetches the assertions from a read-only :class:`SpecStore` **by that
   hash**, parses them with the strict neutral records, **recomputes the hash with ``canon-v1``** and refuses on any
   difference, then checks every ``SCRIPT`` assertion's ``check_program_ref`` against the program registry. Nothing
   has been sent to an adapter yet.
2. The result is a :class:`FrozenAssertionSet`, a frozen object that cannot exist unless its content hashes to its
   pin. :class:`DeterministicVerifier` accepts nothing else, so there is no way to verify an unpinned assertion.
3. The verifier reads the observed value through the narrow ``assert_`` / ``run_check`` capability. The call takes the
   frozen *subject* only: no expected value goes to the adapter, and none comes back with a verdict.
4. The comparison against the frozen ``predicate`` is a pure function of ``(predicate, ObservedValue)``. There is no
   model, no clock, no randomness and no free-text input anywhere on the path to a verdict.
5. Any integrity refusal trips the batch :class:`HaltLatch`: every later call is refused (ADR section 6 rule 9).

Standard library and neutral-module imports only. It never imports recovery code.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, FrozenSet, Iterable, Optional, Protocol, Sequence, Tuple, runtime_checkable

from . import canon
from .errors import (AssertionHashMismatch, CoverageViolation, ExecutorMismatch, ExpectedMismatch, QAAdapterError,
                     QAError, REASON_ASSERTION_INTEGRITY, SpecInvalid, OUTCOME_INCONCLUSIVE)
from .records import (AssertionResult, AssertionSpec, AssertionSubject, Check, CheckSubject, CheckTier, ComponentRef,
                      Determinism, Executor, ObservedValue, Predicate, PredicateOp, Resolution, RunManifest,
                      SubjectKind, Verdict, assertion_set_hash as _records_assertion_set_hash)
from .specstore import ProgramRegistry, SpecStore

#: identity of the reference verifier; an assertion names the verifier that may evaluate it (``verifier_id`` and
#: ``verifier_version_range``), and the frozen vectors use exactly this pair
VERIFIER_ID = "det-verifier"
VERIFIER_VERSION = "1.0.0"

#: per-assertion non-certifying reason codes produced by the verifier itself (adapter errors use their own code)
R_SUBJECT_UNRESOLVED = "SUBJECT_UNRESOLVED"
R_SUBJECT_AMBIGUOUS = "SUBJECT_AMBIGUOUS"
R_TYPE_MISMATCH = "PREDICATE_TYPE_MISMATCH"
R_NOT_COMPARABLE = "ACTUAL_NOT_COMPARABLE"
R_VERIFIER_MISMATCH = "VERIFIER_MISMATCH"
R_EXECUTOR_UNAVAILABLE = "EXECUTOR_UNAVAILABLE"
R_OBSERVED_INVALID = "OBSERVED_VALUE_INVALID"


# ---------------------------------------------------------------------------------------------------------------
# hashes of frozen facts
# ---------------------------------------------------------------------------------------------------------------

def assertion_set_pin(assertions: Iterable[AssertionSpec]) -> str:
    """The ``rcv1:<hex>`` spelling of the ``canon-v1`` hash of an assertion set (what a loader would pin)."""
    return canon.to_contract_ref(_records_assertion_set_hash(assertions))


def expected_hash(assertion: AssertionSpec) -> str:
    """``rcv1:<hex>`` of the frozen predicate (``{op, expected}``): the ``AssertionResult.expected_hash`` the verifier
    must have compared against (reference vectors: the ``EQUALS "Saved"`` predicate hashes to ``1b995d35...``)."""
    return canon.to_contract_ref(canon.canon_hash(assertion.predicate.canonical_form()))


def _same(a: str, b: str) -> bool:
    try:
        return canon.same_digest(a, b)
    except QAError:
        return False


# ---------------------------------------------------------------------------------------------------------------
# the stop rule: a latch that outlives the failing call (ADR section 6 rule 9)
# ---------------------------------------------------------------------------------------------------------------

class BatchHalted(QAError):
    """Raised for any call after an ``ASSERTION_INTEGRITY`` refusal in the same batch. Neutral-only code: the frozen
    refusal vocabulary names the cause, not the halt."""

    code = "BATCH_HALTED"
    run_outcome = OUTCOME_INCONCLUSIVE
    run_reason = REASON_ASSERTION_INTEGRITY


class HaltLatch:
    """One latch per batch. Once tripped it stays tripped; there is no reset."""

    __slots__ = ("_cause",)

    def __init__(self) -> None:
        object.__setattr__(self, "_cause", None)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("HaltLatch has no public state")

    @property
    def tripped(self) -> bool:
        return self._cause is not None

    @property
    def cause(self) -> Optional[QAError]:
        return self._cause

    def trip(self, cause: QAError) -> None:
        if self._cause is None:
            object.__setattr__(self, "_cause", cause)

    def check(self) -> None:
        if self._cause is not None:
            raise BatchHalted(f"batch halted after {self._cause.code}: {self._cause.message}",
                              cause_code=self._cause.code)


class _Halting:
    """Context manager: refuse if the batch is halted; trip the latch on an ASSERTION_INTEGRITY refusal."""

    def __init__(self, latch: HaltLatch) -> None:
        self._latch = latch

    def __enter__(self) -> None:
        self._latch.check()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and isinstance(exc, QAError) and exc.run_reason == REASON_ASSERTION_INTEGRITY:
            self._latch.trip(exc)
        return False


# ---------------------------------------------------------------------------------------------------------------
# approved lineage (ADR section 6 rule 8, harness side)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ApprovedLineage:
    """The assertion-set hashes a reviewed lineage has approved (two-person approval is checked where approvals are
    recorded, outside the harness; this is plain data handed in and read-only). A pin outside it is refused."""

    approved_assertion_set_hashes: FrozenSet[str]

    def __init__(self, hashes: Iterable[str]) -> None:
        object.__setattr__(self, "approved_assertion_set_hashes",
                           frozenset(canon.digest_of(h) for h in hashes))

    def contains(self, assertion_set_hash: str) -> bool:
        try:
            return canon.digest_of(assertion_set_hash) in self.approved_assertion_set_hashes
        except QAError:
            return False


# ---------------------------------------------------------------------------------------------------------------
# assertion validation (spec validation: a failure is a TEST_DEFECT, nothing executes)
# ---------------------------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
_NEEDS_EXPECTED = frozenset({PredicateOp.EQUALS, PredicateOp.NOT_EQUALS, PredicateOp.CONTAINS, PredicateOp.MATCHES,
                             PredicateOp.LESS_THAN, PredicateOp.GREATER_THAN})
MAX_PATTERN_CHARS = 512


def _number(value: Any) -> Optional[Decimal]:
    if type(value) is int:
        return Decimal(value)
    if type(value) is str and len(value) <= 64 and _NUMBER.fullmatch(value):
        return Decimal(value)
    return None


def validate_assertion(assertion: AssertionSpec) -> None:
    """Refuse an assertion this verifier cannot evaluate faithfully (``SpecInvalid``). Pure; no I/O."""
    aid = assertion.assertion_id
    pred = assertion.predicate
    if pred.op in _NEEDS_EXPECTED:
        if pred.expected is None:
            raise SpecInvalid(f"assertion {aid}: predicate {pred.op} needs an expected value")
    elif pred.expected is not None:
        raise SpecInvalid(f"assertion {aid}: predicate {pred.op} takes no expected value")
    if pred.op is PredicateOp.MATCHES:
        if type(pred.expected) is not str or len(pred.expected) > MAX_PATTERN_CHARS:
            raise SpecInvalid(f"assertion {aid}: MATCHES needs a pattern string of at most {MAX_PATTERN_CHARS} characters")
        try:
            re.compile(pred.expected)
        except re.error as exc:
            raise SpecInvalid(f"assertion {aid}: MATCHES pattern does not compile: {exc}") from None
    if pred.op in (PredicateOp.LESS_THAN, PredicateOp.GREATER_THAN) and _number(pred.expected) is None:
        raise SpecInvalid(f"assertion {aid}: {pred.op} needs an integer or a decimal string")
    subject = assertion.subject
    if assertion.executor is Executor.SCRIPT:
        if subject.kind is not SubjectKind.RESOURCE_INPUTS or not subject.inputs:
            raise SpecInvalid(f"assertion {aid}: a SCRIPT assertion needs RESOURCE_INPUTS with at least one input")
    else:
        if subject.kind is SubjectKind.RESOURCE_INPUTS:
            raise SpecInvalid(f"assertion {aid}: a PLAYWRIGHT assertion cannot have RESOURCE_INPUTS")
        if subject.kind is SubjectKind.UI_ELEMENT and subject.locator is None:
            raise SpecInvalid(f"assertion {aid}: a UI_ELEMENT subject needs a locator")
        if subject.kind in (SubjectKind.URL, SubjectKind.RESPONSE) and subject.pattern is None:
            raise SpecInvalid(f"assertion {aid}: a {subject.kind} subject needs a pattern")


def _version_tuple(text: str) -> Optional[Tuple[int, ...]]:
    if type(text) is str and re.fullmatch(r"[0-9]+(\.[0-9]+)*", text):
        return tuple(int(p) for p in text.split("."))
    return None


def _cmp(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a > b) - (a < b)


_CLAUSE = re.compile(r"(>=|<=|==|!=|>|<)\s*([0-9]+(?:\.[0-9]+)*)")


def version_in_range(version: str, spec: str) -> bool:
    """True iff ``version`` satisfies every comma-separated clause of ``spec`` (``>= <= == != > <`` on dotted
    integers). Anything else in the range, or a non-numeric version, is *not* satisfied (fail closed)."""
    v = _version_tuple(version)
    if v is None or type(spec) is not str:
        return False
    clauses = [c.strip() for c in spec.split(",")]
    if not clauses:
        return False
    for clause in clauses:
        m = _CLAUSE.fullmatch(clause)
        if m is None:
            return False
        c = _cmp(v, _version_tuple(m.group(2)))
        ok = {">=": c >= 0, "<=": c <= 0, "==": c == 0, "!=": c != 0, ">": c > 0, "<": c < 0}[m.group(1)]
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# the frozen set, and the one place its hash is recomputed
# ---------------------------------------------------------------------------------------------------------------

def _require_pinned_digest(assertions: Tuple[AssertionSpec, ...], pin: str) -> None:
    """A.2.4 item 6: recompute the assertion-set hash with ``canon-v1`` and refuse on any difference from the pin."""
    recomputed = _records_assertion_set_hash(assertions)
    if not _same(recomputed, pin):
        raise AssertionHashMismatch("the assertion set does not hash to its pinned assertion_set_hash "
                                    "(canon-v1 recomputation differs from the loader's hash)",
                                    pinned=pin, recomputed=canon.to_contract_ref(recomputed))


@dataclass(frozen=True)
class FrozenAssertionSet:
    """Assertions together with the pin they hash to. It cannot be built from content that does not hash to the pin,
    and nothing in it can be changed afterwards: it is the only thing :class:`DeterministicVerifier` will verify."""

    assertion_set_hash: str                       # the pin, rcv1:<hex> spelling
    assertions: Tuple[AssertionSpec, ...]

    def __post_init__(self) -> None:
        if not canon.is_contract_ref(self.assertion_set_hash):
            raise AssertionHashMismatch("a frozen assertion set needs a pinned rcv1:<64 hex> assertion_set_hash")
        items = tuple(self.assertions)
        if not items or not all(type(a) is AssertionSpec for a in items):
            raise SpecInvalid("a frozen assertion set holds one or more AssertionSpec records and nothing else")
        items = tuple(sorted(items, key=lambda a: a.assertion_id))
        if len({a.assertion_id for a in items}) != len(items):
            raise SpecInvalid("assertion ids must be unique")
        object.__setattr__(self, "assertions", items)
        _require_pinned_digest(items, self.assertion_set_hash)

    def get(self, assertion_id: str) -> AssertionSpec:
        for a in self.assertions:
            if a.assertion_id == assertion_id:
                return a
        raise CoverageViolation(f"assertion {assertion_id!r} is not in the frozen assertion set")

    @property
    def assertion_ids(self) -> Tuple[str, ...]:
        return tuple(a.assertion_id for a in self.assertions)


def _require_frozen(frozen: Any) -> FrozenAssertionSet:
    """Only a pinned, self-verifying :class:`FrozenAssertionSet` may be verified (ADR section 6 rule 2)."""
    if type(frozen) is not FrozenAssertionSet:
        raise AssertionHashMismatch("only a pinned FrozenAssertionSet can be verified")
    return frozen


def _check_program(assertion: AssertionSpec, programs: ProgramRegistry) -> None:
    if assertion.executor is not Executor.SCRIPT:
        return
    ref = assertion.check_program_ref
    if ref is None:       # unreachable for a validated record; kept so the rule is stated where it is enforced
        raise SpecInvalid(f"assertion {assertion.assertion_id}: SCRIPT without check_program_ref")
    registered = programs.registered_program_hash(ref.id, ref.version)
    if registered != ref.program_hash:
        raise AssertionHashMismatch(
            f"assertion {assertion.assertion_id}: check program {ref.id}@{ref.version} is "
            + ("not registered" if registered is None else "registered under a different program_hash"),
            assertion_id=assertion.assertion_id, program=f"{ref.id}@{ref.version}")


def load_frozen_assertions(store: SpecStore, pinned_assertion_set_hash: str, *, lineage: ApprovedLineage,
                           programs: ProgramRegistry, halt: Optional[HaltLatch] = None) -> FrozenAssertionSet:
    """Load the assertions pinned by ``pinned_assertion_set_hash`` and prove they are the pinned ones.

    Raises, before any adapter exists in the picture: ``AssertionHashMismatch`` (pin malformed, not in the approved
    lineage, not in the store, recomputed hash differs, program hash differs or program unregistered; run
    ``INCONCLUSIVE(ASSERTION_INTEGRITY)`` and the batch halts), or the record validation errors for an invalid
    assertion (``SpecInvalid``, ``CheckProgramRefRequired``, ``UnknownEnumValue``, ...; run ``TEST_DEFECT``).
    """
    latch = halt if halt is not None else HaltLatch()
    with _Halting(latch):
        pin = pinned_assertion_set_hash
        if not canon.is_contract_ref(pin):
            raise AssertionHashMismatch("a run needs a pinned rcv1:<64 hex> assertion_set_hash before verification")
        if not isinstance(lineage, ApprovedLineage) or not lineage.contains(pin):
            raise AssertionHashMismatch("the pinned assertion set is not in the approved lineage", pinned=pin)
        try:
            docs = store.get_assertion_set(pin)
        except KeyError:
            raise AssertionHashMismatch("no assertion set is stored under the pinned hash", pinned=pin) from None
        if isinstance(docs, (str, bytes)) or not isinstance(docs, (list, tuple)):
            raise SpecInvalid("a stored assertion set is a list of assertion documents")
        assertions = tuple(AssertionSpec.from_json(d) for d in docs)
        frozen = FrozenAssertionSet(assertion_set_hash=pin, assertions=assertions)
        for a in frozen.assertions:
            validate_assertion(a)
            _check_program(a, programs)
        return frozen


# ---------------------------------------------------------------------------------------------------------------
# result consistency (A.2.4 item 6, A.4.2)
# ---------------------------------------------------------------------------------------------------------------

def check_result_matches_assertion(result: Any, assertion: AssertionSpec, *,
                                   assertion_set_hash: Optional[str] = None) -> AssertionResult:
    """Refuse a result that is not about this frozen assertion: a different executor (an unknown executor value is
    non-certifying and refused too), a different expected value (``expected_hash``), another assertion, or another
    assertion set. Returns the parsed :class:`AssertionResult` when it is faithful."""
    if isinstance(result, Mapping):
        raw_executor = result.get("executor")
        if raw_executor not in tuple(e.value for e in Executor):
            raise ExecutorMismatch("the result names an executor this version does not define; non-certifying",
                                   executor=str(raw_executor))
        result = AssertionResult.from_json(result)
    if type(result) is not AssertionResult:
        raise SpecInvalid("expected an AssertionResult")
    if result.assertion_id != assertion.assertion_id or result.check.check_id != assertion.assertion_id:
        raise CoverageViolation("the result is about a different assertion", assertion_id=assertion.assertion_id)
    if result.executor is not assertion.executor:
        raise ExecutorMismatch(f"result executor {result.executor} differs from the frozen assertion's "
                               f"{assertion.executor}", assertion_id=assertion.assertion_id)
    if not _same(result.expected_hash, expected_hash(assertion)):
        raise ExpectedMismatch("the result was computed against an expected value other than the frozen one",
                               assertion_id=assertion.assertion_id)
    if assertion_set_hash is not None and not _same(result.assertion_set_hash, assertion_set_hash):
        raise AssertionHashMismatch("the result names a different assertion set than the one pinned",
                                    assertion_id=assertion.assertion_id)
    return result


def verify_manifest_against_spec(manifest: RunManifest, assertions: Sequence[AssertionSpec]) -> None:
    """The run pinned the spec's assertion set, declared exactly the spec's assertions, and every result is faithful
    to its frozen assertion. Raises an integrity error otherwise."""
    items = tuple(assertions)
    pin = assertion_set_pin(items)
    if not _same(manifest.assertion_set_hash, pin):
        raise AssertionHashMismatch("the manifest pins an assertion set the spec does not hold",
                                    pinned=manifest.assertion_set_hash, recomputed=pin)
    if set(manifest.assertions_declared) != {a.assertion_id for a in items}:
        raise CoverageViolation("assertions_declared is not the spec's assertion ids")
    by_id = {a.assertion_id: a for a in items}
    for res in manifest.assertion_results:
        if res.assertion_id not in by_id:
            raise CoverageViolation(f"a result exists for {res.assertion_id!r}, which the spec never declared")
        check_result_matches_assertion(res, by_id[res.assertion_id], assertion_set_hash=pin)


# ---------------------------------------------------------------------------------------------------------------
# predicate evaluation: a pure function of (frozen predicate, observed value)
# ---------------------------------------------------------------------------------------------------------------

def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def json_equal(a: Any, b: Any) -> bool:
    """Type-strict equality of JSON values: ``true`` is not ``1``, a string is not a number, strings compare NFC."""
    if a is None or b is None:
        return a is b
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    if type(a) is int or type(b) is int:
        return type(a) is type(b) and a == b
    if type(a) is str or type(b) is str:
        return type(a) is type(b) and _nfc(a) == _nfc(b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        ka = {_nfc(k): v for k, v in a.items()}
        kb = {_nfc(k): v for k, v in b.items()}
        return ka.keys() == kb.keys() and all(json_equal(ka[k], kb[k]) for k in ka)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    return False


Evaluation = Tuple[Verdict, Optional[str]]


def evaluate_predicate(predicate: Predicate, observed: ObservedValue) -> Evaluation:
    """Deterministically compare ``observed`` with the frozen ``predicate``: ``(verdict, reason_code)``.

    * The subject must have resolved to exactly one match; anything else is ``INDETERMINATE`` (``SUBJECT_UNRESOLVED``
      / ``SUBJECT_AMBIGUOUS``), never healed and never evaluated (ADR section 6 rule 7).
    * A value that is not available (digest only) or of a type the operator cannot compare is ``INDETERMINATE``.
    * ``PASS`` or ``FAIL`` only when the comparison itself is decided.
    """
    if observed.resolution is not Resolution.RESOLVED or observed.match_count != 1:
        if observed.resolution is Resolution.AMBIGUOUS or observed.match_count > 1:
            return Verdict.INDETERMINATE, R_SUBJECT_AMBIGUOUS
        return Verdict.INDETERMINATE, R_SUBJECT_UNRESOLVED
    op, expected, actual = predicate.op, predicate.expected, observed.value
    if op is PredicateOp.EXISTS:
        return Verdict.PASS, None
    if op is PredicateOp.ABSENT:
        absent = actual is None and observed.value_digest is None
        return (Verdict.PASS if absent else Verdict.FAIL), None
    if actual is None:                                  # resolved, but only a digest (or nothing) was read
        return Verdict.INDETERMINATE, R_NOT_COMPARABLE
    if op is PredicateOp.EQUALS:
        return (Verdict.PASS if json_equal(expected, actual) else Verdict.FAIL), None
    if op is PredicateOp.NOT_EQUALS:
        return (Verdict.FAIL if json_equal(expected, actual) else Verdict.PASS), None
    if op is PredicateOp.CONTAINS:
        if type(actual) is str and type(expected) is str:
            return (Verdict.PASS if _nfc(expected) in _nfc(actual) else Verdict.FAIL), None
        if isinstance(actual, (tuple, list)):
            return (Verdict.PASS if any(json_equal(expected, e) for e in actual) else Verdict.FAIL), None
        if isinstance(actual, Mapping) and type(expected) is str:
            return (Verdict.PASS if _nfc(expected) in {_nfc(k) for k in actual} else Verdict.FAIL), None
        return Verdict.INDETERMINATE, R_TYPE_MISMATCH
    if op is PredicateOp.MATCHES:
        if type(actual) is not str or type(expected) is not str:
            return Verdict.INDETERMINATE, R_TYPE_MISMATCH
        try:
            hit = re.search(expected, _nfc(actual)) is not None
        except re.error:
            return Verdict.INDETERMINATE, R_TYPE_MISMATCH
        return (Verdict.PASS if hit else Verdict.FAIL), None
    if op in (PredicateOp.LESS_THAN, PredicateOp.GREATER_THAN):
        left, right = _number(actual), _number(expected)
        if left is None or right is None:
            return Verdict.INDETERMINATE, R_TYPE_MISMATCH
        holds = left < right if op is PredicateOp.LESS_THAN else left > right
        return (Verdict.PASS if holds else Verdict.FAIL), None
    return Verdict.INDETERMINATE, R_TYPE_MISMATCH       # an operator this version does not know: never a pass


# ---------------------------------------------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------------------------------------------

@runtime_checkable
class SubjectReader(Protocol):
    """The only thing the verifier may ask of a browser adapter: read the actual value of a subject. A
    :class:`NeutralQAAdapter` satisfies it; the verifier cannot call ``act``, ``locate`` or anything else."""

    def assert_(self, s: Any, subject: AssertionSubject) -> ObservedValue: ...


@runtime_checkable
class CheckRunner(Protocol):
    """The only thing the verifier may ask of a check adapter: run one registered, hashed check program."""

    def run_check(self, w: Any, subject: CheckSubject) -> ObservedValue: ...


class DeterministicVerifier:
    """Verifies frozen assertions against observed state. Holds no model, no clock and no mutable assertion state."""

    def __init__(self, *, programs: ProgramRegistry, qa_adapter: Optional[SubjectReader] = None,
                 check_adapter: Optional[CheckRunner] = None, halt: Optional[HaltLatch] = None,
                 verifier_id: str = VERIFIER_ID, verifier_version: str = VERIFIER_VERSION) -> None:
        self._programs = programs
        self._qa = qa_adapter
        self._check = check_adapter
        self._halt = halt if halt is not None else HaltLatch()
        self._id = verifier_id
        self._version = verifier_version

    @property
    def halt(self) -> HaltLatch:
        return self._halt

    def component(self) -> ComponentRef:
        return ComponentRef(id=self._id, version=self._version)

    # -- one assertion ---------------------------------------------------------------------------------------
    def verify_assertion(self, frozen: FrozenAssertionSet, assertion_id: str, *, session: Any = None,
                         workspace: Any = None, check_tier: CheckTier = CheckTier.OFFLINE,
                         evidence_refs: Sequence[str] = ()) -> AssertionResult:
        with _Halting(self._halt):
            assertion = _require_frozen(frozen).get(assertion_id)
            observed, verdict, reason = self._observe_and_decide(assertion, session, workspace, check_tier)
            result = AssertionResult(
                assertion_id=assertion.assertion_id,
                assertion_set_hash=frozen.assertion_set_hash,
                expected_hash=expected_hash(assertion),
                executor=assertion.executor,
                check=Check(check_id=assertion.assertion_id, verdict=verdict, determinism=Determinism.DETERMINISTIC,
                            evidence_refs=tuple(evidence_refs)),
                verifier=self.component(),
                actual=observed.value if observed is not None else None,
                actual_digest=observed.value_digest if observed is not None else None,
                observation_id=observed.observation_id if observed is not None else None,
                reason_code=reason,
            )
            check_result_matches_assertion(result, assertion, assertion_set_hash=frozen.assertion_set_hash)
            return result

    def verify_all(self, frozen: FrozenAssertionSet, *, session: Any = None, workspace: Any = None,
                   check_tier: CheckTier = CheckTier.OFFLINE) -> Tuple[AssertionResult, ...]:
        """Every assertion of the frozen set, in ``assertion_id`` order. An integrity refusal stops the loop (the
        exception propagates and the batch latch is tripped)."""
        with _Halting(self._halt):
            _require_frozen(frozen)
        return tuple(self.verify_assertion(frozen, a.assertion_id, session=session, workspace=workspace,
                                           check_tier=check_tier) for a in frozen.assertions)

    def verify_run(self, store: SpecStore, pinned_assertion_set_hash: str, *, lineage: ApprovedLineage,
                   session: Any = None, workspace: Any = None,
                   check_tier: CheckTier = CheckTier.OFFLINE) -> Tuple[AssertionResult, ...]:
        """ADR section 6 rule 3: fetch the assertions from the store **by hash at verification time**, recompute,
        refuse on mismatch, then verify every assertion. A spec edited between the run's start and now is refused
        here, with nothing evaluated."""
        frozen = load_frozen_assertions(store, pinned_assertion_set_hash, lineage=lineage, programs=self._programs,
                                        halt=self._halt)
        return self.verify_all(frozen, session=session, workspace=workspace, check_tier=check_tier)

    # -- internals -------------------------------------------------------------------------------------------
    def _observe_and_decide(self, assertion: AssertionSpec, session: Any, workspace: Any,
                            check_tier: CheckTier) -> Tuple[Optional[ObservedValue], Verdict, Optional[str]]:
        if assertion.verifier_id != self._id or not version_in_range(self._version, assertion.verifier_version_range):
            return None, Verdict.INDETERMINATE, R_VERIFIER_MISMATCH
        _check_program(assertion, self._programs)          # again, immediately before execution (registry may change)
        try:
            if assertion.executor is Executor.SCRIPT:
                if self._check is None:
                    return None, Verdict.INDETERMINATE, R_EXECUTOR_UNAVAILABLE
                ref = assertion.check_program_ref
                subject = CheckSubject(check_program_ref=ref, inputs=assertion.subject.inputs, tier=check_tier)
                observed = self._check.run_check(workspace, subject)
            elif assertion.executor is Executor.PLAYWRIGHT:
                if self._qa is None:
                    return None, Verdict.INDETERMINATE, R_EXECUTOR_UNAVAILABLE
                observed = self._qa.assert_(session, assertion.subject)       # the subject only: no expected value
            else:                                           # an executor this version does not know: non-certifying
                return None, Verdict.INDETERMINATE, R_EXECUTOR_UNAVAILABLE
        except QAAdapterError as exc:
            return None, Verdict.INDETERMINATE, exc.code
        if type(observed) is not ObservedValue:
            return None, Verdict.INDETERMINATE, R_OBSERVED_INVALID
        verdict, reason = evaluate_predicate(assertion.predicate, observed)
        return observed, verdict, reason


__all__ = [
    "VERIFIER_ID", "VERIFIER_VERSION", "BatchHalted", "HaltLatch", "ApprovedLineage", "FrozenAssertionSet",
    "DeterministicVerifier", "SubjectReader", "CheckRunner", "assertion_set_pin", "expected_hash",
    "validate_assertion", "version_in_range", "load_frozen_assertions", "check_result_matches_assertion",
    "verify_manifest_against_spec", "evaluate_predicate", "json_equal",
]
