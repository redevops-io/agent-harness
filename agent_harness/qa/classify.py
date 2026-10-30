# SPDX-License-Identifier: AGPL-3.0-or-later
"""The run classifier (plan H1; ADR section 7 and Addendum A.4.4, A.5).

``classify(facts)`` is a pure function: the same facts always give the same :class:`Classification`, nothing is read
from outside the argument, and no model takes part. It works in two steps so that the precedence can be tested
exhaustively: :func:`derive_signals` reduces the facts to a small :class:`Signals` record, and :func:`decide` maps the
signals to an outcome with an explicit first-match-wins precedence (ADR section 7):

    1. integrity violation                       -> INCONCLUSIVE(ASSERTION_INTEGRITY), halt the batch
    2. environment health failed                 -> ENVIRONMENT_FAIL
    3. spec invalid / baseline / fixture defect  -> TEST_DEFECT
    4. deterministic FAIL meeting the PRODUCT_FAIL requirements -> PRODUCT_FAIL
    5. any other non-PASS evidence, omission, or anything unverified -> INCONCLUSIVE(reason)
    6. otherwise                                 -> PASS

Fail closed: every attestation the runner has not made is ``UNKNOWN`` and blocks ``PASS`` and ``PRODUCT_FAIL``; an
error code the classifier does not know is an integrity violation; zero assertions is never a pass.

Standard library and neutral-module imports only. The module never imports recovery code.
"""
from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from . import canon
from .errors import (ERRORS_BY_CODE, REASON_ASSERTION_INTEGRITY, QAError, SpecInvalid)
from .records import (AssertionSpec, Determinism, FaultEvent, Outcome, RecoveryResult,
                      RecoveryStrategy, ReviewItem, RunManifest, SkipReason, SkippedAssertion, Verdict,
                      FaultSignal)
from .verifier import assertion_set_pin, check_result_matches_assertion

ASSERTION_INTEGRITY = REASON_ASSERTION_INTEGRITY


class Signal(enum.StrEnum):
    """What the runner attests about one fact. Absent attestation is ``UNKNOWN`` and never counts as ``OK``."""

    OK = "OK"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------------------------------------------
# error-code families (stable codes from errors.py; adapter families follow ADR section 8)
# ---------------------------------------------------------------------------------------------------------------

#: refusals that mean the assertion rule was broken (frozen: they carry run_reason ASSERTION_INTEGRITY), plus the
#: ones that mean the evidence itself cannot be trusted
INTEGRITY_CODES = frozenset({c for c, k in ERRORS_BY_CODE.items() if k.run_reason == ASSERTION_INTEGRITY}
                            | {"BATCH_HALTED", "HASH_MISMATCH", "FAULT_BINDING_MISMATCH"})
#: the spec or its inputs are wrong: nothing could be evaluated fairly (ADR section 7, TEST_DEFECT)
SPEC_DEFECT_CODES = frozenset({"SPEC_INVALID", "CHECK_PROGRAM_REF_REQUIRED", "UNKNOWN_ENUM_VALUE", "UNKNOWN_FIELD",
                               "SCHEMA_VERSION_UNSUPPORTED", "CANON_REFUSED", "UNSUPPORTED_COVERAGE_CONTRACT",
                               "UNSUPPORTED_OPERATION"})
#: infrastructure prevented a fair test (ADR section 7, ENVIRONMENT_FAIL)
ENVIRONMENT_CODES = frozenset({"BROWSER_CRASH", "ADAPTER_CRASH", "ENVIRONMENT_UNHEALTHY", "RESET_FAILED",
                               "RESOURCE_LIMIT_EXCEEDED", "NAVIGATION_DENIED", "SESSION_LOST"})
REDACTION_CODES = frozenset({"REDACTION_FAILED"})
_ENVIRONMENT_SIGNALS = frozenset({FaultSignal.ENV_UNHEALTHY, FaultSignal.SESSION_LOST, FaultSignal.NETWORK_BLOCKED})
_ASSERTION_SIGNALS = frozenset({FaultSignal.ASSERTION_FAILED, FaultSignal.ASSERTION_UNRESOLVABLE})
_LOCATOR_STRATEGIES = frozenset({RecoveryStrategy.SEMANTIC_LOCATOR, RecoveryStrategy.MODEL_CANDIDATE})


# ---------------------------------------------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class RunFacts:
    """Everything the classifier looks at. ``manifest`` and ``assertions`` (the frozen set the verifier loaded by
    hash) are records; the rest are the runner's attestations, every one ``UNKNOWN`` unless the runner says otherwise.

    ``error_codes`` are the stable codes of the typed errors the run ended with and did not recover from.
    """

    manifest: RunManifest
    assertions: Tuple[AssertionSpec, ...]
    environment_before: Signal = Signal.UNKNOWN       # health probe before the first action
    environment_after: Signal = Signal.UNKNOWN        # health probe after the last action
    preconditions: Signal = Signal.UNKNOWN            # spec preconditions verified
    fixture_hash: Signal = Signal.UNKNOWN             # fixture content hashes matched; FAILED is a fixture mismatch
    cleanup: Signal = Signal.UNKNOWN                  # verified resource cleanup
    policy: Signal = Signal.UNKNOWN                   # OK means no policy violation
    recovery_checks: Signal = Signal.UNKNOWN          # every recovery used passed unique-target and postcondition checks
    failure_reproduced: Signal = Signal.UNKNOWN       # a failure seen through a recovered locator reproduced via the authored one
    spec_validity: Signal = Signal.UNKNOWN            # only FAILED matters
    baseline_conformance: Signal = Signal.UNKNOWN     # only FAILED matters: step target / subject missing in the baseline build
    rerun_consistency: Signal = Signal.UNKNOWN        # only FAILED matters: mixed outcomes over N reruns of an identical build
    error_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.manifest) is not RunManifest:
            raise SpecInvalid("RunFacts.manifest must be a RunManifest")
        items = tuple(self.assertions)
        if not all(type(a) is AssertionSpec for a in items):
            raise SpecInvalid("RunFacts.assertions must hold AssertionSpec records")
        if len({a.assertion_id for a in items}) != len(items):
            raise SpecInvalid("RunFacts.assertions: assertion ids must be unique")
        object.__setattr__(self, "assertions", items)
        for name in ("environment_before", "environment_after", "preconditions", "fixture_hash", "cleanup", "policy",
                     "recovery_checks", "failure_reproduced", "spec_validity", "baseline_conformance",
                     "rerun_consistency"):
            value = getattr(self, name)
            if not isinstance(value, Signal):
                try:
                    object.__setattr__(self, name, Signal(value))
                except ValueError:
                    raise SpecInvalid(f"RunFacts.{name}: {value!r} is not a Signal") from None
        codes = tuple(self.error_codes)
        if not all(type(c) is str for c in codes):
            raise SpecInvalid("RunFacts.error_codes must be strings")
        object.__setattr__(self, "error_codes", codes)


@dataclass(frozen=True, kw_only=True)
class Signals:
    """The lattice the precedence runs over. A ``*_reason`` that is not ``None`` means that condition holds."""

    integrity: bool = False
    environment_reason: Optional[str] = None
    test_defect_reason: Optional[str] = None
    deterministic_fail: bool = False
    indeterminate: bool = False
    skipped: bool = False
    nondeterministic: bool = False
    fault: bool = False
    redaction_failed: bool = False
    no_assertions: bool = False
    environment_verified: bool = False
    preconditions_verified: bool = False
    fixture_verified: bool = False
    reproduction_ok: bool = False
    recovery_ok: bool = False
    cleanup_ok: bool = False
    policy_ok: bool = False


@dataclass(frozen=True)
class Classification:
    """``classify`` result. ``pair`` is the ADR's ``(outcome, reason)``."""

    outcome: Outcome
    reason: Optional[str]
    verdict: Verdict
    halt_batch: bool
    review_required: bool
    rule: int
    details: Tuple[str, ...] = ()

    @property
    def pair(self) -> Tuple[Outcome, Optional[str]]:
        return self.outcome, self.reason

    def to_json(self) -> Dict[str, Any]:
        return {"outcome": self.outcome.value, "reason": self.reason, "verdict": self.verdict.value,
                "halt_batch": self.halt_batch, "review_required": self.review_required, "rule": self.rule,
                "details": list(self.details)}


# ---------------------------------------------------------------------------------------------------------------
# outcome -> contract Verdict, exit codes, review items
# ---------------------------------------------------------------------------------------------------------------

_VERDICT_OF = MappingProxyType({
    Outcome.PASS: Verdict.PASS,
    Outcome.PRODUCT_FAIL: Verdict.FAIL,
    Outcome.TEST_DEFECT: Verdict.INDETERMINATE,
    Outcome.ENVIRONMENT_FAIL: Verdict.INDETERMINATE,
    Outcome.INCONCLUSIVE: Verdict.INDETERMINATE,
})


def outcome_to_verdict(outcome: Any) -> Verdict:
    """ADR section 7: ``PASS`` -> ``PASS``, ``PRODUCT_FAIL`` -> ``FAIL``, every other outcome -> ``INDETERMINATE``.
    An outcome this version does not define can never be read as a pass: ``INDETERMINATE``."""
    try:
        return _VERDICT_OF[Outcome(outcome)]
    except (ValueError, TypeError, KeyError):
        return Verdict.INDETERMINATE


#: default CI exit codes: 0 only for PASS; every other outcome is non-zero (fail closed)
DEFAULT_EXIT_CODES = MappingProxyType({
    Outcome.PASS: 0, Outcome.PRODUCT_FAIL: 1, Outcome.TEST_DEFECT: 2, Outcome.ENVIRONMENT_FAIL: 3,
    Outcome.INCONCLUSIVE: 4,
})


def exit_code(outcome: Any, overrides: Optional[Mapping] = None) -> int:
    """The CI exit code for an outcome. ``overrides`` may change the non-zero numbers but never map a non-PASS
    outcome to zero, change PASS, or name an unknown outcome (``SpecInvalid``). An unknown outcome exits non-zero."""
    table = dict(DEFAULT_EXIT_CODES)
    for key, value in (overrides or {}).items():
        try:
            member = Outcome(key)
        except ValueError:
            raise SpecInvalid(f"exit code override for an unknown outcome {key!r}") from None
        if type(value) is not int or not 0 <= value <= 255:
            raise SpecInvalid(f"exit code for {member} must be an integer from 0 to 255")
        if (member is Outcome.PASS) != (value == 0):
            raise SpecInvalid("only PASS may exit 0, and PASS must exit 0 (INCONCLUSIVE and the rest never count as a pass)")
        table[member] = value
    try:
        return table[Outcome(outcome)]
    except (ValueError, TypeError, KeyError):
        return table[Outcome.INCONCLUSIVE]


def review_item_for(run_id: str, classification: Classification,
                    evidence_refs: Sequence = ()) -> Optional[ReviewItem]:
    """The ``ReviewItem`` an ``INCONCLUSIVE`` run always creates (ADR section 7); ``None`` for other outcomes."""
    if not classification.review_required:
        return None
    item_id = "review-" + run_id
    if len(item_id) > 128:
        item_id = "review-" + canon.digest_hex(run_id)
    return ReviewItem(review_item_id=item_id, run_id=run_id, reason=classification.reason or "INCONCLUSIVE",
                      evidence_refs=tuple(evidence_refs))


# ---------------------------------------------------------------------------------------------------------------
# coverage invariants (ADR A.4.4)
# ---------------------------------------------------------------------------------------------------------------

def _skipped_id(item: Any) -> Any:
    if isinstance(item, SkippedAssertion):
        return item.assertion_id
    if isinstance(item, Mapping):
        return item.get("assertion_id")
    return item


def classify_coverage(declared: Iterable[Any], executed: Iterable[Any], skipped: Iterable[Any], *,
                      spec_ids: Optional[Iterable[Any]] = None) -> Optional[Tuple[Outcome, str]]:
    """The coverage-list invariants: ``declared`` equals the spec's assertion ids (when ``spec_ids`` is given),
    ``executed`` and ``skipped`` are disjoint, there are no duplicates, and ``executed`` union ``skipped`` equals
    ``declared``. Returns ``(INCONCLUSIVE, ASSERTION_INTEGRITY)`` on any violation, ``None`` when consistent."""
    d, e = list(declared), list(executed)
    s = [_skipped_id(x) for x in skipped]
    bad = (len(set(d)) != len(d) or len(set(e)) != len(e) or len(set(s)) != len(s)
           or not set(e).isdisjoint(s) or (set(e) | set(s)) != set(d))
    if spec_ids is not None and set(spec_ids) != set(d):
        bad = True
    return (Outcome.INCONCLUSIVE, ASSERTION_INTEGRITY) if bad else None


def _integrity_findings(facts: RunFacts) -> Tuple[str, ...]:
    m = facts.manifest
    spec = {a.assertion_id: a for a in facts.assertions}
    found = []
    if m.outcome is Outcome.INCONCLUSIVE and m.outcome_reason == ASSERTION_INTEGRITY:
        found.append("MANIFEST_RECORDS_ASSERTION_INTEGRITY")
    if classify_coverage(m.assertions_declared, m.assertions_executed, m.assertions_skipped,
                         spec_ids=list(spec)) is not None:
        found.append("COVERAGE_INVARIANT")
    if sorted(r.assertion_id for r in m.assertion_results) != sorted(m.assertions_executed):
        found.append("RESULTS_DO_NOT_MATCH_EXECUTED")
    pin = assertion_set_pin(facts.assertions)
    if not canon.same_digest(m.assertion_set_hash, pin):
        found.append("ASSERTION_SET_HASH_MISMATCH")
    for r in m.assertion_results:
        a = spec.get(r.assertion_id)
        if a is None:
            found.append("RESULT_FOR_UNDECLARED_ASSERTION")
            continue
        try:
            check_result_matches_assertion(r, a, assertion_set_hash=pin)
        except QAError as exc:
            found.append(exc.code)
    if m.outcome is Outcome.PASS and (m.assertions_skipped
                                      or any(r.check.verdict is not Verdict.PASS for r in m.assertion_results)):
        found.append("PASS_CONTRADICTED_BY_OWN_RECORD")        # ADR A.5: an integrity error, reported as such
    if any(s.reason_code is SkipReason.HALTED_INTEGRITY for s in m.assertions_skipped):
        found.append("HALTED_INTEGRITY")
    for code in facts.error_codes:
        if code in INTEGRITY_CODES or code not in ERRORS_BY_CODE:
            found.append(code)                                  # an unknown code is an integrity violation
    found.extend(r.reason_code for r in m.assertion_results if r.reason_code in INTEGRITY_CODES)
    return tuple(found)


# ---------------------------------------------------------------------------------------------------------------
# facts -> signals
# ---------------------------------------------------------------------------------------------------------------

def _fault_flags(events: Iterable[FaultEvent]) -> Tuple[bool, Optional[str], bool]:
    """(unrecovered fault, environment reason, recovered-locator-used) from the sealed fault events."""
    fault, env, locator_recovery = False, None, False
    for ev in events:
        rec = ev.recovery
        if rec.result is RecoveryResult.RECOVERED:
            if rec.strategy in _LOCATOR_STRATEGIES:
                locator_recovery = True
            continue
        if not ev.detection.detected or ev.detection.signal in _ASSERTION_SIGNALS or ev.detection.signal is FaultSignal.NONE:
            continue
        if ev.detection.signal in _ENVIRONMENT_SIGNALS:
            env = env or ev.detection.signal.value
        else:
            fault = True
    return fault, env, locator_recovery


def derive_signals(facts: RunFacts) -> Signals:
    return _signals(facts, _integrity_findings(facts))


def _signals(facts: RunFacts, findings: Tuple[str, ...]) -> Signals:
    m = facts.manifest
    integrity = bool(findings)

    env_reason: Optional[str] = None
    defect_reason: Optional[str] = None
    fault, redaction = False, False
    codes = list(facts.error_codes)
    for r in m.assertion_results:
        if r.reason_code is not None:
            codes.append(r.reason_code)
    for code in codes:
        if code in ENVIRONMENT_CODES:
            env_reason = env_reason or code
        elif code in SPEC_DEFECT_CODES:
            defect_reason = defect_reason or code
        elif code in REDACTION_CODES:
            redaction = True
        elif code in ERRORS_BY_CODE and code not in INTEGRITY_CODES:
            fault = True
    if facts.environment_before is Signal.FAILED or facts.environment_after is Signal.FAILED:
        env_reason = env_reason or "ENVIRONMENT_UNHEALTHY"
    if any(s.reason_code is SkipReason.ENVIRONMENT for s in m.assertions_skipped):
        env_reason = env_reason or "ENVIRONMENT"
    ev_fault, ev_env, locator_recovery = _fault_flags(m.fault_events)
    fault = fault or ev_fault
    env_reason = env_reason or ev_env

    if facts.spec_validity is Signal.FAILED:
        defect_reason = defect_reason or "SPEC_INVALID"
    if facts.baseline_conformance is Signal.FAILED:
        defect_reason = defect_reason or "BASELINE_NONCONFORMANT"
    if facts.fixture_hash is Signal.FAILED:
        defect_reason = defect_reason or "FIXTURE_MISMATCH"
    if facts.rerun_consistency is Signal.FAILED:
        defect_reason = defect_reason or "RERUN_INCONSISTENT"

    deterministic_fail = indeterminate = nondeterministic = False
    for r in m.assertion_results:
        verdict, det = r.check.verdict, r.check.determinism
        if det is not Determinism.DETERMINISTIC:
            nondeterministic = True
        elif verdict is Verdict.FAIL:
            deterministic_fail = True
        elif verdict is not Verdict.PASS:
            indeterminate = True                     # INDETERMINATE, NOT_APPLICABLE or anything unknown

    skipped = bool(m.assertions_skipped)
    return Signals(
        integrity=integrity,
        environment_reason=env_reason,
        test_defect_reason=defect_reason,
        deterministic_fail=deterministic_fail,
        indeterminate=indeterminate,
        skipped=skipped,
        nondeterministic=nondeterministic,
        fault=fault,
        redaction_failed=redaction,
        no_assertions=not facts.assertions or not m.assertions_declared,
        environment_verified=(facts.environment_before is Signal.OK and facts.environment_after is Signal.OK),
        preconditions_verified=facts.preconditions is Signal.OK,
        fixture_verified=facts.fixture_hash is Signal.OK,
        reproduction_ok=(not locator_recovery) or facts.failure_reproduced is Signal.OK,
        recovery_ok=(not any(ev.recovery.result is RecoveryResult.RECOVERED for ev in m.fault_events))
        or facts.recovery_checks is Signal.OK,
        cleanup_ok=facts.cleanup is Signal.OK,
        policy_ok=facts.policy is Signal.OK,
    )


# ---------------------------------------------------------------------------------------------------------------
# the precedence
# ---------------------------------------------------------------------------------------------------------------

def _result(outcome: Outcome, reason: Optional[str], rule: int, details: Tuple[str, ...] = ()) -> Classification:
    return Classification(
        outcome=outcome, reason=reason, verdict=outcome_to_verdict(outcome),
        halt_batch=(outcome is Outcome.INCONCLUSIVE and reason == ASSERTION_INTEGRITY),
        review_required=(outcome is Outcome.INCONCLUSIVE), rule=rule, details=details)


def decide(s: Signals, details: Tuple[str, ...] = ()) -> Classification:
    """ADR section 7 precedence, first match wins. Pure function of the signals."""
    if s.integrity:                                                         # (1)
        return _result(Outcome.INCONCLUSIVE, ASSERTION_INTEGRITY, 1, details)
    if s.environment_reason is not None:                                    # (2)
        return _result(Outcome.ENVIRONMENT_FAIL, s.environment_reason, 2, details)
    if s.test_defect_reason is not None:                                    # (3)
        return _result(Outcome.TEST_DEFECT, s.test_defect_reason, 3, details)
    if (s.deterministic_fail and s.preconditions_verified and s.fixture_verified and s.environment_verified
            and s.reproduction_ok and not s.fault and not s.redaction_failed):          # (4)
        return _result(Outcome.PRODUCT_FAIL, "ASSERTION_FAILED", 4, details)
    # (5) anything that is not provably a pass or a product failure
    reason: Optional[str] = None
    if s.redaction_failed:
        reason = "REDACTION_FAILED"
    elif s.fault:
        reason = "UNRECOVERED_FAULT"
    elif s.deterministic_fail and not s.reproduction_ok:
        reason = "FAILURE_NOT_REPRODUCED"
    elif s.deterministic_fail:
        reason = "PRODUCT_FAIL_UNVERIFIED"
    elif s.skipped:
        reason = "ASSERTIONS_SKIPPED"
    elif s.indeterminate:
        reason = "ASSERTION_INDETERMINATE"
    elif s.nondeterministic:
        reason = "NON_DETERMINISTIC_EVIDENCE"
    elif s.no_assertions:
        reason = "NO_ASSERTIONS"
    elif not s.preconditions_verified:
        reason = "PRECONDITIONS_UNVERIFIED"
    elif not s.fixture_verified:
        reason = "FIXTURE_UNVERIFIED"
    elif not s.environment_verified:
        reason = "ENVIRONMENT_UNVERIFIED"
    elif not s.recovery_ok:
        reason = "RECOVERY_UNVERIFIED"
    elif not s.cleanup_ok:
        reason = "CLEANUP_UNVERIFIED"
    elif not s.policy_ok:
        reason = "POLICY_UNVERIFIED"
    if reason is not None:
        return _result(Outcome.INCONCLUSIVE, reason, 5, details)
    return _result(Outcome.PASS, None, 6, details)                          # (6)


def classify(facts: RunFacts) -> Classification:
    """Classify a finished run: ``classify(facts).pair`` is the ADR's ``(outcome, reason)``."""
    if type(facts) is not RunFacts:
        raise SpecInvalid("classify takes a RunFacts")
    findings = _integrity_findings(facts)
    return decide(_signals(facts, findings), findings)


__all__ = [
    "ASSERTION_INTEGRITY", "Signal", "RunFacts", "Signals", "Classification", "INTEGRITY_CODES", "SPEC_DEFECT_CODES",
    "ENVIRONMENT_CODES", "REDACTION_CODES", "DEFAULT_EXIT_CODES", "outcome_to_verdict", "exit_code", "review_item_for",
    "classify_coverage", "derive_signals", "decide", "classify",
]
