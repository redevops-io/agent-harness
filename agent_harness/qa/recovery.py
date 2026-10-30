# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic recovery controls (plan H3; ADR section 6 rule 4 and rule 7, section 8, section 9).

Recovery may change *how a step finds its target or which execution path it takes*; it can never change *what is
expected*. Everything in this module is a pure function or a small stateless object: no model, no network, no clock,
no randomness, no filesystem. The runner (``runner.py``) owns the orchestration (state machine, evidence, policy,
receipts); this module owns the judgements, so each one can be tested and mutated on its own.

What lives here:

* :func:`detect_fault` turns a failed attempt into a ``FaultDetection`` (what was seen, what class it is claimed to be).
  The runner seals it **before** any recovery decision or action.
* :func:`plan_recovery` decides, from the step's declared fields and the remaining caps, which recovery mode (if any)
  may run, and why not when none may. Recovery is off unless the run enables it, the kill switch is not engaged, the
  step's ``permitted_recovery`` lists the mode, the attempt/time caps and the spec budget allow it, and the idempotency
  guard (``idempotency.decide_retry``, the H2 rule) says the work may be done again. ``UNKNOWN`` idempotency is no
  guard. Environment modes (``CHECKPOINT_RESUME``, ``RESET_FIXTURE``) additionally require every earlier step they would
  have to replay to pass the same guard.
* :class:`LocatorSelector` is the semantic locator recovery. A ``Recoverer`` proposes strict ``LocatorCandidate``
  descriptors (through ``recovery_types.propose_safely``, the single gate); each proposal is located against the *fresh*
  observation's page; a control is accepted only if it is the **single** control, across every proposal, whose role and
  accessible name equal the frozen authored role and name (NFC, case folding, whitespace collapsing), and it is unique,
  visible and enabled. Several matching controls, none, a different role or a different name are refusals; the first
  match is never taken.
* :func:`similarity` is a *deterministic, integer* score (0 to 1000) of a found control against the authored role and
  name: role weight 400 (equal or not), name weight 600 (Dice coefficient on character bigrams of the normalised name,
  integer arithmetic). It orders candidates deterministically and names near misses in the rejection reasons. It never
  loosens acceptance: the equality gate (:func:`gate_reason`) is the only way in, so ``Delete`` can never recover
  ``Save`` however the score is read.

The recovery output cannot express an assertion, an expected value or a verdict: its only input type is
``RecoveryContext`` (the frozen step and the current observation) and its only output type is the strict
``LocatorCandidate`` (``recovery_types``). Observed content (names, roles, URLs) is compared and recorded; it is never
an input to the policy, the spec or the assertion set.

Standard library and neutral-module imports only. This module imports neither the verifier, the classifier, the spec
store, the policy nor the runner.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .errors import QAAdapterError, RecoveryPayloadRejected, SpecInvalid
from .idempotency import (DENY_ALREADY_APPLIED, DENY_BUDGET_EXHAUSTED, DENY_INVALID_INPUT, DENY_NOT_PERMITTED,
                          DENY_NOT_RETRYABLE, DENY_RECOVERY_DISABLED, RetryDecision, RetryInputs, decide_retry)
from .ports import Recoverer, RecoveryContext
from .recovery_types import propose_safely
from .records import (ClaimedClass, ComponentRef, FaultDetection, FaultSignal, LocateResult, LocatorCandidate,
                      LocatorDescriptor, LocatorStrategy, Observation, PermittedRecovery, PostconditionResult,
                      RecoveryStrategy, ResolvedTarget, Step, SuppressedReason)

DETECTOR = ComponentRef(id="agent-harness-qa-fault-detector", version="0.1.0")

#: similarity scale and weights (integers: no floating point anywhere in a recovery decision)
SCORE_MAX = 1000
ROLE_WEIGHT = 400
NAME_WEIGHT = 600
#: a name at or above this Dice score that still fails the equality gate is reported as a near miss
NEAR_MISS_NAME_SCORE = 700
#: bound on the rejection reasons one recovery record carries
MAX_REASONS = 32

#: refusal / rejection reason codes (stable ``Reason`` strings)
R_NO_UI_TARGET = "NO_UI_TARGET"
R_NO_ANCHOR = "NO_ROLE_NAME_ANCHOR"
R_RECOVERER_ERROR = "RECOVERER_ERROR"
R_CANDIDATE_CAP = "CANDIDATE_CAP_EXCEEDED"
R_ELAPSED_CAP = "ELAPSED_CAP"
R_KILL_SWITCH = "KILL_SWITCH"
R_NO_CANDIDATE = "NO_CANDIDATE"
R_AMBIGUOUS = "AMBIGUOUS_CANDIDATES"
R_NOT_FOUND = "NOT_FOUND"
R_AMBIGUOUS_RESULT = "AMBIGUOUS_RESULT"
R_NOT_UNIQUE = "NOT_UNIQUE"
R_NOT_ACTIONABLE = "NOT_ACTIONABLE"
R_ROLE_MISMATCH = "ROLE_MISMATCH"
R_NAME_MISMATCH = "NAME_MISMATCH"
R_NAME_NEAR_MISS = "NAME_NEAR_MISS"
R_STRATEGY = "STRATEGY_NOT_PERMITTED"
R_EMPTY = "EMPTY_DESCRIPTOR"
R_MALFORMED = "LOCATE_MALFORMED"

PERMIT_OF: Mapping[RecoveryStrategy, PermittedRecovery] = MappingProxyType({
    RecoveryStrategy.RETRY_SAME: PermittedRecovery.RETRY_SAME,
    RecoveryStrategy.SEMANTIC_LOCATOR: PermittedRecovery.SEMANTIC_LOCATOR,
    RecoveryStrategy.RESET_FIXTURE: PermittedRecovery.RESET_FIXTURE,
    RecoveryStrategy.CHECKPOINT_RESUME: PermittedRecovery.CHECKPOINT_RESUME,
    RecoveryStrategy.MODEL_CANDIDATE: PermittedRecovery.MODEL_CANDIDATE,    # plan H4: only a ladder given by recovery_model.py names it
})

#: the recovery modes this module can run (``MODEL_CANDIDATE`` is plan H4 and is never chosen here)
ENVIRONMENT_STRATEGIES = frozenset({RecoveryStrategy.CHECKPOINT_RESUME, RecoveryStrategy.RESET_FIXTURE})

#: failure code -> the recovery modes to try, in order (a rung is skipped when the step does not permit it)
LADDER: Mapping[str, Tuple[RecoveryStrategy, ...]] = MappingProxyType({
    "LOCATOR_NOT_FOUND": (RecoveryStrategy.RETRY_SAME, RecoveryStrategy.SEMANTIC_LOCATOR),
    "LOCATOR_AMBIGUOUS": (RecoveryStrategy.SEMANTIC_LOCATOR,),
    "TARGET_NOT_ACTIONABLE": (RecoveryStrategy.RETRY_SAME,),
    "ACTION_TIMEOUT": (RecoveryStrategy.RETRY_SAME,),
    "SESSION_LOST": (RecoveryStrategy.CHECKPOINT_RESUME,),
    "BROWSER_CRASH": (RecoveryStrategy.CHECKPOINT_RESUME,),
    "ADAPTER_CRASH": (RecoveryStrategy.CHECKPOINT_RESUME,),
    "ENVIRONMENT_UNHEALTHY": (RecoveryStrategy.RESET_FIXTURE,),
})


# ---------------------------------------------------------------------------------------------------------------
# labels, similarity and the role+name gate
# ---------------------------------------------------------------------------------------------------------------

def normalise_label(text: Optional[str]) -> Optional[str]:
    """NFC, case folding, whitespace collapsing. ``None`` stays ``None``."""
    if text is None:
        return None
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _bigrams(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for i in range(len(text) - 1):
        gram = text[i:i + 2]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def dice_permille(a: str, b: str) -> int:
    """Dice coefficient of the two strings' character-bigram multisets, as an integer 0..1000. Equal strings score
    1000; a string shorter than two characters scores 0 unless equal. Integer arithmetic only."""
    if a == b:
        return SCORE_MAX
    if len(a) < 2 or len(b) < 2:
        return 0
    ga, gb = _bigrams(a), _bigrams(b)
    common = sum(min(n, gb.get(g, 0)) for g, n in ga.items())
    return (2 * common * SCORE_MAX) // ((len(a) - 1) + (len(b) - 1))


def name_score(authored: Optional[str], found: Optional[str]) -> int:
    na, nf = normalise_label(authored), normalise_label(found)
    if na is None or nf is None:
        return 0
    return dice_permille(na, nf)


def similarity(authored: LocatorDescriptor, role: Optional[str], name: Optional[str]) -> int:
    """Role+name similarity of a found control to the authored descriptor, 0..1000, deterministic. Role counts 400 when
    equal (after normalisation) and 0 otherwise; name counts 600 times its Dice score. 1000 means role and name are
    equal after normalisation."""
    r = ROLE_WEIGHT if normalise_label(authored.role) == normalise_label(role) and role is not None else 0
    n = (NAME_WEIGHT * name_score(authored.name, name)) // SCORE_MAX
    return r + n


def labels_match(authored: Optional[str], found: Optional[str]) -> bool:
    """Equal after NFC, case folding and whitespace collapsing. An absent authored value is *not* a match here: the
    recovery gate needs both anchors (see :func:`has_anchor`)."""
    if authored is None or found is None:
        return False
    return normalise_label(authored) == normalise_label(found)


def has_anchor(authored: LocatorDescriptor) -> bool:
    """Recovery needs an authored role *and* accessible name to compare against; without them identity cannot be
    verified and nothing is recovered (stricter than the H2 hook path, which lets an absent value through)."""
    return bool(normalise_label(authored.role)) and bool(normalise_label(authored.name))


def gate_reason(authored: LocatorDescriptor, role: Optional[str], name: Optional[str]) -> Optional[str]:
    """``None`` when the found control has the authored role and name; otherwise the reason it does not."""
    if not labels_match(authored.role, role):
        return R_ROLE_MISMATCH
    if not labels_match(authored.name, name):
        return R_NAME_NEAR_MISS if name_score(authored.name, name) >= NEAR_MISS_NAME_SCORE else R_NAME_MISMATCH
    return None


# ---------------------------------------------------------------------------------------------------------------
# limits and the kill switch
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class RecoveryLimits:
    """Operator caps on recovery itself (the spec budget's ``max_retries`` / ``max_recovery_attempts`` apply as well)."""

    max_attempts_per_step: int = 2      # recovery attempts for one step, all modes together
    max_candidates: int = 8             # locator proposals and located controls considered per recovery
    max_elapsed_ms: int = 10_000        # time spent recovering one step (measured on the injected clock)
    max_replay_steps: int = 8           # earlier steps an environment recovery may replay

    def __post_init__(self) -> None:
        for name in ("max_attempts_per_step", "max_candidates", "max_elapsed_ms", "max_replay_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SpecInvalid(f"RecoveryLimits.{name}: expected a non-negative integer")


class KillSwitch:
    """Operator kill switch: once engaged it stays engaged, and no recovery starts or continues."""

    __slots__ = ("_reason",)

    def __init__(self) -> None:
        self._reason: Optional[str] = None

    def engage(self, reason: str = "kill switch") -> None:
        if self._reason is None:
            self._reason = str(reason)[:256] or "kill switch"

    @property
    def engaged(self) -> bool:
        return self._reason is not None

    @property
    def reason(self) -> Optional[str]:
        return self._reason


# ---------------------------------------------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------------------------------------------

_DETECTION: Dict[str, Tuple[FaultSignal, ClaimedClass]] = {
    "LOCATOR_NOT_FOUND": (FaultSignal.LOCATOR_NOT_FOUND, ClaimedClass.BENIGN_DRIFT),
    "LOCATOR_AMBIGUOUS": (FaultSignal.LOCATOR_AMBIGUOUS, ClaimedClass.UNKNOWN),
    "TARGET_NOT_ACTIONABLE": (FaultSignal.UNEXPECTED_OVERLAY, ClaimedClass.BENIGN_DRIFT),
    "UNEXPECTED_DIALOG": (FaultSignal.UNEXPECTED_OVERLAY, ClaimedClass.BENIGN_DRIFT),
    "ACTION_TIMEOUT": (FaultSignal.ACTION_TIMEOUT, ClaimedClass.UNKNOWN),
    "SESSION_LOST": (FaultSignal.SESSION_LOST, ClaimedClass.ENVIRONMENT),
    "BROWSER_CRASH": (FaultSignal.ENV_UNHEALTHY, ClaimedClass.ENVIRONMENT),
    "ADAPTER_CRASH": (FaultSignal.ENV_UNHEALTHY, ClaimedClass.ENVIRONMENT),
    "ENVIRONMENT_UNHEALTHY": (FaultSignal.ENV_UNHEALTHY, ClaimedClass.ENVIRONMENT),
    "RESET_FAILED": (FaultSignal.ENV_UNHEALTHY, ClaimedClass.ENVIRONMENT),
    "RESOURCE_LIMIT_EXCEEDED": (FaultSignal.ENV_UNHEALTHY, ClaimedClass.ENVIRONMENT),
    "NAVIGATION_DENIED": (FaultSignal.NETWORK_BLOCKED, ClaimedClass.ENVIRONMENT),
    "ACTION_BLOCKED_BY_POLICY": (FaultSignal.POLICY_DENIED, ClaimedClass.UNKNOWN),
}


def detect_fault(error: Any, *, elapsed_ms: int) -> Optional[FaultDetection]:
    """The ``FaultDetection`` for a failed attempt, or ``None`` for an error that is not a recoverable-class fault
    here (``REDACTION_FAILED``, ``UNSUPPORTED_OPERATION``, anything that is not a typed adapter error): those end the
    run under their own code and are not detection events. A declared transition that was not observed after an
    action that reported success is claimed as a *suspected product defect*, never as drift."""
    if not isinstance(error, QAAdapterError):
        return None
    mapped = _DETECTION.get(error.code)
    if mapped is None:
        return None
    signal, claimed = mapped
    if error.code == "ACTION_TIMEOUT" and error.details.get("signal") == "TRANSITION_NOT_OBSERVED":
        signal, claimed = FaultSignal.TRANSITION_NOT_OBSERVED, ClaimedClass.PRODUCT_DEFECT_SUSPECTED
    return FaultDetection(detected=True, signal=signal, claimed_class=claimed, detector=DETECTOR,
                          detected_at_ms_into_step=max(0, int(elapsed_ms)))


# ---------------------------------------------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class FaultInputs:
    """Everything :func:`plan_recovery` may look at: declared step fields, counters and flags. No observed content, no
    assertion, no expected value."""

    step: Step
    error_code: str
    error_retryable: bool
    acted: bool                              # the adapter's act() was invoked for the failed attempt
    postcondition: PostconditionResult       # the step's own check after the failed attempt
    has_idempotency_key: bool
    recovery_enabled: bool
    retries_left: int
    recovery_attempts_left: int
    step_attempts_used: int                  # recovery attempts already made for this step
    step_recovery_ms: int                    # milliseconds already spent recovering this step
    tried: Tuple[RecoveryStrategy, ...]      # modes already attempted for this step
    prior_steps: Tuple[Step, ...] = ()       # earlier steps an environment recovery would replay
    has_fixture: bool = False                # the spec names a known fixture to reset to
    prior_used_recovered_locator: bool = False
    model_calls_left: int = 0                # plan H4: model calls the run budget still allows (0 = none, fail closed)
    model_tokens_left: int = 0               # plan H4: tokens the run budget still allows (0 = none, fail closed)


@dataclass(frozen=True)
class RecoveryDecision:
    strategy: RecoveryStrategy               # ``NONE`` when nothing will be attempted
    suppressed: SuppressedReason
    reason: str
    already_applied: bool = False            # the step's own postcondition shows the effect is there: do not redo it

    @property
    def attempt(self) -> bool:
        return self.strategy is not RecoveryStrategy.NONE


def _no(suppressed: SuppressedReason, reason: str) -> RecoveryDecision:
    return RecoveryDecision(RecoveryStrategy.NONE, suppressed, reason)


def _suppression(reason: str) -> SuppressedReason:
    if reason == DENY_RECOVERY_DISABLED:
        return SuppressedReason.KILL_SWITCH
    if reason == DENY_BUDGET_EXHAUSTED:
        return SuppressedReason.BUDGET_EXHAUSTED
    if reason in (DENY_NOT_PERMITTED, DENY_NOT_RETRYABLE, DENY_INVALID_INPUT):
        return SuppressedReason.NOT_PERMITTED
    return SuppressedReason.NON_IDEMPOTENT      # retry unsafe, no guard, unknown side-effect class: no blind replay


def _guard(inputs: RetryInputs) -> RetryDecision:
    """The idempotency judgement: the H2 ``decide_retry`` rule, unchanged."""
    return decide_retry(inputs)


def _retry_inputs(step: Step, *, acted: bool, postcondition: PostconditionResult, has_key: bool, retryable: bool,
                  permitted: Tuple[PermittedRecovery, ...], retries_left: int) -> RetryInputs:
    return RetryInputs(side_effect=step.side_effect, idempotency=step.idempotency, retry_safety=step.retry_safety,
                       permitted_recovery=permitted, may_have_executed=acted, has_idempotency_key=has_key,
                       postcondition=postcondition, error_retryable=retryable, recovery_enabled=True,
                       retries_left=retries_left)


def _exhausted(strategy: RecoveryStrategy, i: FaultInputs) -> bool:
    if strategy is RecoveryStrategy.RETRY_SAME:
        return i.retries_left <= 0
    if strategy is RecoveryStrategy.SEMANTIC_LOCATOR:
        return strategy in i.tried or i.recovery_attempts_left <= 0     # one deterministic proposal round per step
    if strategy is RecoveryStrategy.MODEL_CANDIDATE:                    # plan H4: one model round per step, and a call to make
        return (strategy in i.tried or i.recovery_attempts_left <= 0 or i.model_calls_left <= 0
                or i.model_tokens_left <= 0)
    return i.recovery_attempts_left <= 0


def plan_recovery(i: FaultInputs, limits: RecoveryLimits, *, kill_switch: bool,
                  ladder_by_fault: Mapping[str, Tuple[RecoveryStrategy, ...]] = LADDER) -> RecoveryDecision:
    """Which recovery (if any) may run for this fault. Pure and deterministic. Order of refusals: kill switch or
    recovery disabled, policy denial, no mode for this fault, mode not permitted by the step, caps and budgets,
    idempotency."""
    if kill_switch or not i.recovery_enabled:
        return _no(SuppressedReason.KILL_SWITCH, "RECOVERY_DISABLED")
    if i.error_code == "ACTION_BLOCKED_BY_POLICY":
        return _no(SuppressedReason.POLICY_DENIED, "POLICY_DENIED")
    ladder = ladder_by_fault.get(i.error_code, ())
    if not ladder:
        return _no(SuppressedReason.NOT_PERMITTED, "NO_RECOVERY_FOR_FAULT")
    permitted = [s for s in ladder if PERMIT_OF[s] in i.step.permitted_recovery]
    if not permitted:
        return _no(SuppressedReason.NOT_PERMITTED, "MODE_NOT_PERMITTED")
    if i.step_attempts_used >= limits.max_attempts_per_step:
        return _no(SuppressedReason.BUDGET_EXHAUSTED, "STEP_ATTEMPT_CAP")
    if i.step_recovery_ms >= limits.max_elapsed_ms:
        return _no(SuppressedReason.BUDGET_EXHAUSTED, "ELAPSED_CAP")
    live = [s for s in permitted if not _exhausted(s, i)]
    if not live:
        return _no(SuppressedReason.BUDGET_EXHAUSTED, "BUDGET_EXHAUSTED")
    strategy = live[0]
    # a miss at locate time executed nothing, so retrying it is a wait-and-look-again (a delayed render), not a replay;
    # for every other mode the ladder above already decided that the fault is one it applies to
    retryable = True
    if strategy is RecoveryStrategy.RETRY_SAME:
        retryable = i.error_retryable or (not i.acted and i.error_code == "LOCATOR_NOT_FOUND")
    if strategy in ENVIRONMENT_STRATEGIES:
        if strategy is RecoveryStrategy.RESET_FIXTURE and not i.has_fixture:
            return _no(SuppressedReason.NOT_PERMITTED, "NO_KNOWN_FIXTURE")
        if i.prior_used_recovered_locator:
            return _no(SuppressedReason.NOT_PERMITTED, "REPLAY_NEEDS_RECOVERED_LOCATOR")
        if len(i.prior_steps) > limits.max_replay_steps:
            return _no(SuppressedReason.BUDGET_EXHAUSTED, "REPLAY_CAP")
    # idempotency (decide_retry is the only judge). The mode itself was checked above, so the guard is asked with
    # RETRY_SAME standing in for "the spec allows this work to be done again" except for RETRY_SAME itself.
    permit = i.step.permitted_recovery if strategy is RecoveryStrategy.RETRY_SAME else (PermittedRecovery.RETRY_SAME,)
    left = i.retries_left if strategy is RecoveryStrategy.RETRY_SAME else 1
    guard = _guard(_retry_inputs(i.step, acted=i.acted, postcondition=i.postcondition, has_key=i.has_idempotency_key,
                                 retryable=retryable, permitted=permit, retries_left=left))
    if guard.reason == DENY_ALREADY_APPLIED:
        return RecoveryDecision(RecoveryStrategy.NONE, SuppressedReason.NONE, DENY_ALREADY_APPLIED,
                                already_applied=True)
    if not guard.allow:
        return _no(_suppression(guard.reason), guard.reason)
    if strategy in ENVIRONMENT_STRATEGIES:
        for prior in i.prior_steps:
            # an earlier step is replayed in a fresh state: it ran, and nothing observed shows its effect is absent
            replay = _guard(_retry_inputs(prior, acted=True, postcondition=PostconditionResult.NOT_EVALUATED,
                                          has_key=True, retryable=True, permitted=(PermittedRecovery.RETRY_SAME,),
                                          retries_left=1))
            if not replay.allow:
                return _no(_suppression(replay.reason), "REPLAY_" + replay.reason)
    return RecoveryDecision(strategy, SuppressedReason.NONE, guard.reason)


# ---------------------------------------------------------------------------------------------------------------
# semantic locator selection
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LocatorSelection:
    """The outcome of one semantic recovery round. ``target`` is set only when exactly one control passed every check;
    ``aborted`` is set when a cap or the kill switch cut the round short."""

    target: Optional[ResolvedTarget]
    score: int
    considered: int
    rejected: Tuple[str, ...]
    refusal: Optional[str] = None
    aborted: bool = False


class AuthoredAlternatesRecoverer:
    """The default deterministic ``Recoverer``: alternate semantic descriptors derived from the *authored* locator only
    (role+name, role+label, role+name+label, test id, label, name), in a fixed order. It ignores the observation's
    content, so a page cannot steer which alternates are tried. It returns locator candidates and nothing else."""

    def propose(self, ctx: RecoveryContext) -> Tuple[LocatorCandidate, ...]:
        d = ctx.step.target.locator
        if d is None:
            return ()
        out: List[LocatorCandidate] = []

        def add(**kw: Optional[str]) -> None:
            cand = LocatorCandidate(strategy=LocatorStrategy.SEMANTIC_ALTERNATE, **kw)
            if cand not in out:
                out.append(cand)
        if d.role and d.name:
            add(role=d.role, name=d.name)
        if d.role and d.label:
            add(role=d.role, label=d.label)
        if d.role and d.name and d.label:
            add(role=d.role, name=d.name, label=d.label)
        if d.test_id:
            add(test_id=d.test_id)
        if d.label:
            add(label=d.label)
        if d.name:
            add(name=d.name)
        return tuple(out)


class LocatorSelector:
    """Semantic locator recovery over a ``Recoverer``'s proposals. See the module docstring for the rules."""

    def __init__(self, recoverer: Recoverer, limits: RecoveryLimits) -> None:
        self._recoverer = recoverer
        self._limits = limits

    def select(self, step: Step, observation: Observation, locate: Callable[[LocatorDescriptor], Any],
               halted: Callable[[], Optional[str]]) -> LocatorSelection:
        """``locate`` performs one semantic locate and returns its ``LocateResult`` (it may raise a typed adapter error,
        which propagates). ``halted`` returns a reason code when a cap or the kill switch says to stop, else ``None``.
        Raises ``RecoveryPayloadRejected`` when the recoverer's output fails the strict gate (an integrity event)."""
        authored = step.target.locator
        if authored is None:
            return LocatorSelection(None, 0, 0, (), R_NO_UI_TARGET)
        if not has_anchor(authored):
            return LocatorSelection(None, 0, 0, (), R_NO_ANCHOR)
        try:
            proposals = propose_safely(self._recoverer, RecoveryContext(step=step, observation=observation))
        except RecoveryPayloadRejected:
            raise
        except Exception:                                   # a recoverer that fails recovers nothing
            return LocatorSelection(None, 0, 0, (), R_RECOVERER_ERROR)
        if len(proposals) > self._limits.max_candidates:
            return LocatorSelection(None, 0, 0, (), R_CANDIDATE_CAP, aborted=True)
        rejected: List[str] = []
        accepted: Dict[str, Tuple[Any, LocatorDescriptor]] = {}
        matching_refs: List[str] = []                       # every control seen, in any result, with the authored role+name
        seen = set()
        considered = 0
        for proposal in proposals:
            stop = halted()
            if stop is not None:
                return LocatorSelection(None, 0, considered, tuple(rejected[:MAX_REASONS]), stop, aborted=True)
            if proposal.strategy is not LocatorStrategy.SEMANTIC_ALTERNATE:
                rejected.append(R_STRATEGY)
                continue
            if proposal.role is None and proposal.name is None and proposal.label is None and proposal.test_id is None:
                rejected.append(R_EMPTY)
                continue
            key = (proposal.role, proposal.name, proposal.label, proposal.test_id)
            if key in seen:
                continue
            seen.add(key)
            considered += 1
            descriptor = LocatorDescriptor(frame_path=authored.frame_path, role=proposal.role, name=proposal.name,
                                           label=proposal.label, test_id=proposal.test_id)
            found = locate(descriptor)
            if type(found) is not LocateResult:
                rejected.append(R_MALFORMED)
                continue
            if len(found.candidates) > self._limits.max_candidates:
                return LocatorSelection(None, 0, considered, tuple(rejected[:MAX_REASONS]), R_CANDIDATE_CAP,
                                        aborted=True)
            ordered = sorted(found.candidates,
                             key=lambda c: (-similarity(authored, c.role, c.name), c.candidate_ref))
            for cand in ordered:
                if gate_reason(authored, cand.role, cand.name) is None and cand.candidate_ref not in matching_refs:
                    matching_refs.append(cand.candidate_ref)
            if not found.candidates:
                rejected.append(R_NOT_FOUND)
                continue
            if len(found.candidates) > 1:                   # several controls: never the first match
                rejected.append(R_AMBIGUOUS_RESULT)
                continue
            cand = found.candidates[0]
            reason = gate_reason(authored, cand.role, cand.name)
            if reason is not None:
                rejected.append(reason)
                continue
            if not cand.unique:
                rejected.append(R_NOT_UNIQUE)
                continue
            if not (cand.visible and cand.enabled):
                rejected.append(R_NOT_ACTIONABLE)
                continue
            accepted.setdefault(cand.candidate_ref, (cand, descriptor))
        reasons = tuple(rejected[:MAX_REASONS])
        if len(matching_refs) != 1:                         # exactly one control may carry the authored role and name
            return LocatorSelection(None, 0, considered, reasons, R_AMBIGUOUS if matching_refs else R_NO_CANDIDATE)
        ref = matching_refs[0]
        if ref not in accepted:                             # the one control failed uniqueness or actionability
            return LocatorSelection(None, 0, considered, reasons, R_NO_CANDIDATE)
        cand, descriptor = accepted[ref]
        target = ResolvedTarget(candidate_ref=ref, locator=descriptor, strategy=LocatorStrategy.SEMANTIC_ALTERNATE)
        return LocatorSelection(target, similarity(authored, cand.role, cand.name), considered, reasons)


# ---------------------------------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------------------------------

class DeterministicRecovery:
    """What a runner is given to enable deterministic recovery. Holds the recoverer, the limits and the kill switch;
    it holds no run state."""

    def __init__(self, *, recoverer: Optional[Recoverer] = None, limits: Optional[RecoveryLimits] = None,
                 kill_switch: Optional[KillSwitch] = None) -> None:
        recoverer = recoverer if recoverer is not None else AuthoredAlternatesRecoverer()
        if not callable(getattr(recoverer, "propose", None)):
            raise SpecInvalid("DeterministicRecovery: the recoverer must provide propose(ctx)")
        if limits is not None and type(limits) is not RecoveryLimits:
            raise SpecInvalid("DeterministicRecovery: limits must be RecoveryLimits")
        if kill_switch is not None and type(kill_switch) is not KillSwitch:
            raise SpecInvalid("DeterministicRecovery: kill_switch must be a KillSwitch")
        self._limits = limits if limits is not None else RecoveryLimits()
        self._kill = kill_switch if kill_switch is not None else KillSwitch()
        self._selector = LocatorSelector(recoverer, self._limits)

    @property
    def limits(self) -> RecoveryLimits:
        return self._limits

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill

    def detect(self, error: Any, *, elapsed_ms: int) -> Optional[FaultDetection]:
        return detect_fault(error, elapsed_ms=elapsed_ms)

    def plan(self, inputs: FaultInputs) -> RecoveryDecision:
        return plan_recovery(inputs, self._limits, kill_switch=self._kill.engaged)

    def select_locator(self, step: Step, observation: Observation, locate: Callable[[LocatorDescriptor], Any],
                       within_time: Callable[[], bool]) -> LocatorSelection:
        def halted() -> Optional[str]:
            if self._kill.engaged:
                return R_KILL_SWITCH
            return None if within_time() else R_ELAPSED_CAP
        return self._selector.select(step, observation, locate, halted)


__all__ = [
    "DETECTOR", "SCORE_MAX", "LADDER", "PERMIT_OF", "ENVIRONMENT_STRATEGIES", "normalise_label", "dice_permille",
    "name_score", "similarity", "labels_match", "has_anchor", "gate_reason", "RecoveryLimits", "KillSwitch",
    "detect_fault", "FaultInputs", "RecoveryDecision", "plan_recovery", "LocatorSelection",
    "AuthoredAlternatesRecoverer", "LocatorSelector", "DeterministicRecovery",
]
