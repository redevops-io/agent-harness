# SPDX-License-Identifier: AGPL-3.0-or-later
"""The bounded single-run executor (plan H2; ADR section 3, section 7, Addendum A.2.4, A.4.4, A.7).

``QARunner.run`` executes **one** :class:`TestSpec` in **one** isolated session and returns one sealed record. It is a
primitive, not a workflow engine: no queue, no scheduling, no dependency resolution, no state that outlives the call
except what it appends to the evidence stream it was given (memory-only by default). Retries are in-run attempts bounded
by the spec budget.

States (``runstate.py``): ``PREPARE -> OBSERVE -> ACT -> VERIFY -> PASS``, with ``RECOVERABLE -> REOBSERVE -> ACT`` for a
recoverable step failure (recovery is **disabled by default**; the recovery *step* is plan H3, this module provides the
hook and the transitions), and ``PRODUCT_FAIL``, ``TEST_DEFECT``, ``ENVIRONMENT_FAIL`` and ``INCONCLUSIVE -> REVIEW``.

What it guarantees, and where:

* **Pinned before anything runs.** The frozen assertions are loaded by their pinned hash and recomputed (``verifier``),
  checked against the spec, and ``RunStarted`` is appended to the evidence stream **before** ``open_session``. The
  assertions are loaded again from the store at verification time, so a spec edited in between is refused.
* **Every action is decided.** For every attempt the policy is asked first; the decision record (allow or deny, from the
  policy, or built here when the policy raised, timed out or answered with something unusable) is hashed into the
  receipt's ``policy_decision_ref``. Only an ``ALLOW`` for exactly this run, step and action lets the adapter be called.
  ``REQUIRE_APPROVAL`` is not an allow: there is no approver in this layer.
* **Targets are verified, not trusted.** The runner picks the target from ``locate``'s candidates: one candidate whose
  role and name equal the authored descriptor, unique, visible and enabled. Zero is ``LocatorNotFound``, several is
  ``LocatorAmbiguous``; it never takes the first match.
* **Receipts are append-only** and chained; their hashes are written into the manifest's ``step_results``.
* **Bounded.** Run, step and observation deadlines and the retry, recovery, model-call and token budgets come from the
  spec budget (operator caps can only lower them). Model calls default to zero.
* **The outcome is the classifier's.** The runner collects facts (coverage, results, attestations, error codes) and
  asks ``classify``; the state machine then follows the answer. The only runner-side override makes the result *more*
  conservative (see ``_conservative``).
* **Untrusted content.** Observed content (URLs, DOM text, titles, adapter detail strings) is stored as evidence and
  hashed into state hashes; it is never an input to the policy, to the spec or to the assertion set.

Deadlines are cooperative: an adapter call cannot be interrupted through the neutral protocol, so the clock is checked
before an operation and after it returns.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

from . import canon
from .classify import (ENVIRONMENT_CODES, Classification, RunFacts, Signal, classify, exit_code, outcome_to_verdict,
                       review_item_for)
from .cleanup import close_and_report
from .checkpoint import CheckpointLog
from .errors import (ADAPTER_ERRORS, REASON_ASSERTION_INTEGRITY, ActionTimeout, AdapterCrash, AssertionHashMismatch,
                     LocatorAmbiguous, LocatorNotFound, QAAdapterError, QAError, RedactionFailed, SpecInvalid,
                     TargetNotActionable, UnsupportedOperation)
from .hooks import (CancelToken, EvidenceStream, NotEvaluatedPostcondition, PostconditionChecker, PrepareChecks,
                    RecoveryHook, RecoveryRequest, RecoveryResponse, UnverifiedChecks)
from .idempotency import RetryInputs, decide_retry, idempotency_key
from .limits import (SCOPE_RUN, Budgets, Deadlines, EffectiveLimits, LimitCaps, StepTimer, effective_limits)
from .policy import (binding_problem, decision_ref, failsafe_decision, is_allowed)
from .ports import ActionPolicy, Clock, NeutralCheckAdapter, NeutralQAAdapter
from .receipts import ChainLink, EventChain, ReceiptLedger, build_receipt, state_hash, unknown_state_hash
from .records import (ActionOutcome, ActionReceipt, ActionRequest, AssertionResult, CheckProfile,
                      ComponentRef, Counters, Observation, ObserveRequest, Outcome, Phase, PermittedRecovery,
                      PolicyDecisionRecord, Postcondition, PostconditionResult, LocateMode,
                      LocateResult, LocatorStrategy, LocatorUsed, RecoveryStrategy, ResolvedTarget, ResultCode,
                      ReviewItem, RunManifest, SessionHandle, SessionProfile, SkipReason, SkippedAssertion, Step,
                      StepResult, TargetKind, TestSpec, ToolRef, VersionPin, WorkspaceHandle, Executor)
from .records import Checkpoint
from .run_errors import (BudgetExceeded, DeadlineExceeded, EvidenceUnavailable, IllegalTransition, ReceiptChainBroken,
                         RunCancelled)
from .runstate import OUTCOME_STATE, RunState, RunStateMachine
from .runstate import is_allowed as is_allowed_transition
from .schemas import ADAPTER_API_VERSION, CHECK_ADAPTER_API_VERSION
from .specstore import ProgramRegistry, SpecStore
from .verifier import (ApprovedLineage, DeterministicVerifier, FrozenAssertionSet, HaltLatch,
                       assertion_set_pin, load_frozen_assertions)

RUNNER_VERSION = "0.1.0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_RCV_OR_SHA = re.compile(r"(rcv1|sha256):[0-9a-f]{64}")
_TRACE = re.compile(r"[0-9a-f]{32}")
_CODE_BY_RESULT = {r.value: r for r in ResultCode}


# ---------------------------------------------------------------------------------------------------------------
# configuration and result
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """Operator configuration. Everything defaults to the most restrictive setting."""

    egress_policy_id: str
    redaction_policy_id: str
    recovery_enabled: bool = False          # retries and recovery hook; off by default
    model_recovery_enabled: bool = False    # plan H4; off by default
    caps: LimitCaps = LimitCaps()           # can only lower the spec budget
    policy_timeout_ms: int = 5000           # a policy answer that took this long or longer is a denial
    principal: str = "runner:qa"
    trace_id: Optional[str] = None
    check_profile: Optional[CheckProfile] = None     # needed to run SCRIPT assertions through a check adapter

    def __post_init__(self) -> None:
        for name in ("egress_policy_id", "redaction_policy_id"):
            value = getattr(self, name)
            if type(value) is not str or _RCV_OR_SHA.fullmatch(value) is None:
                raise SpecInvalid(f"RunConfig.{name}: expected rcv1:<64 hex> or sha256:<64 hex>")
        for name in ("recovery_enabled", "model_recovery_enabled"):
            if type(getattr(self, name)) is not bool:
                raise SpecInvalid(f"RunConfig.{name}: expected a boolean")
        if type(self.caps) is not LimitCaps:
            raise SpecInvalid("RunConfig.caps: expected LimitCaps")
        if type(self.policy_timeout_ms) is not int or self.policy_timeout_ms < 1:
            raise SpecInvalid("RunConfig.policy_timeout_ms: expected a positive integer")
        if type(self.principal) is not str or not 1 <= len(self.principal) <= 512:
            raise SpecInvalid("RunConfig.principal: expected a non-empty string")
        if self.trace_id is not None and (type(self.trace_id) is not str or _TRACE.fullmatch(self.trace_id) is None):
            raise SpecInvalid("RunConfig.trace_id: expected 32 lowercase hex digits")
        if self.check_profile is not None and type(self.check_profile) is not CheckProfile:
            raise SpecInvalid("RunConfig.check_profile: expected CheckProfile")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    classification: Classification
    manifest: RunManifest
    exit_code: int
    review_item: Optional[ReviewItem]
    states: Tuple[RunState, ...]
    receipts: Tuple[ActionReceipt, ...]
    receipt_links: Tuple[ChainLink, ...]
    policy_decisions: Tuple[PolicyDecisionRecord, ...]
    checkpoints: Tuple[Checkpoint, ...]
    events: Tuple[Mapping[str, Any], ...]
    cancelled: bool
    sealed: bool                 # the final RunSealed event was appended to the stream

    @property
    def outcome(self) -> Outcome:
        return self.classification.outcome

    @property
    def halt_batch(self) -> bool:
        return self.classification.halt_batch


# ---------------------------------------------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------------------------------------------

class _Abort(Exception):
    """Ends the run early. ``code`` is the stable error code the classifier will see (or ``None``); ``skip`` is the
    reason the unexecuted assertions are recorded under."""

    def __init__(self, *, code: Optional[str], skip: SkipReason,
                 cancelled: bool = False, note: str = "") -> None:
        super().__init__(code or "ABORT")
        self.code = code
        self.skip = skip
        self.cancelled = cancelled
        self.note = note


@dataclass
class _Attempt:
    error: Optional[QAError]
    acted: bool
    postcondition: Postcondition
    after: Optional[Observation]
    receipt: ActionReceipt


@dataclass
class _Ctx:
    spec: TestSpec
    run_id: str
    pin: str
    limits: EffectiveLimits
    deadlines: Deadlines
    budgets: Budgets
    machine: RunStateMachine = field(default_factory=RunStateMachine)
    ledger: Optional[ReceiptLedger] = None
    checkpoints: Optional[CheckpointLog] = None
    chain: EventChain = field(default_factory=EventChain)
    events: List[Mapping[str, Any]] = field(default_factory=list)
    decisions: List[PolicyDecisionRecord] = field(default_factory=list)
    results: List[AssertionResult] = field(default_factory=list)
    step_results: List[StepResult] = field(default_factory=list)
    frozen: Optional[FrozenAssertionSet] = None
    session: Optional[SessionHandle] = None
    workspace: Optional[WorkspaceHandle] = None
    abort: Optional[_Abort] = None
    started_at: Optional[str] = None
    browser: Optional[str] = None
    env_before: Signal = Signal.UNKNOWN
    preconditions: Signal = Signal.UNKNOWN
    fixtures: Signal = Signal.UNKNOWN
    locator_recovery_used: bool = False


def _same_label(authored: Optional[str], found: Optional[str]) -> bool:
    """Role or accessible name of a recovered candidate versus the authored one: equal after NFC, case folding and
    whitespace collapsing. An authored value of ``None`` places no constraint. (Looser, scored similarity is plan H3.)"""
    if authored is None:
        return True
    if found is None:
        return False
    def norm(text: str) -> str:
        return " ".join(unicodedata.normalize("NFC", text).casefold().split())
    return norm(authored) == norm(found)


def _signal(value: Optional[bool]) -> Signal:
    return Signal.UNKNOWN if value is None else (Signal.OK if value else Signal.FAILED)


class _Shield:
    """Wraps an adapter so that an unexpected exception reaches the verifier as a typed ``AdapterCrash``."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def assert_(self, s: Any, subject: Any) -> Any:
        return QARunner._call(self._adapter.assert_, s, subject)

    def run_check(self, w: Any, subject: Any) -> Any:
        return QARunner._call(self._adapter.run_check, w, subject)


class QARunner:
    """Executes one spec against one adapter. Reusable and stateless between calls to :meth:`run`."""

    def __init__(self, *, qa_adapter: NeutralQAAdapter, policy: Optional[ActionPolicy], store: SpecStore,
                 lineage: ApprovedLineage, programs: ProgramRegistry, clock: Clock, config: RunConfig,
                 check_adapter: Optional[NeutralCheckAdapter] = None, evidence: Optional[EvidenceStream] = None,
                 checks: Optional[PrepareChecks] = None, postconditions: Optional[PostconditionChecker] = None,
                 recovery: Optional[RecoveryHook] = None, halt: Optional[HaltLatch] = None,
                 cancel: Optional[CancelToken] = None) -> None:
        if not isinstance(config, RunConfig):
            raise SpecInvalid("QARunner: config must be a RunConfig")
        if getattr(qa_adapter, "api_version", None) != ADAPTER_API_VERSION:
            raise SpecInvalid(f"QARunner: the adapter must speak {ADAPTER_API_VERSION}")
        if check_adapter is not None and getattr(check_adapter, "api_version", None) != CHECK_ADAPTER_API_VERSION:
            raise SpecInvalid(f"QARunner: the check adapter must speak {CHECK_ADAPTER_API_VERSION}")
        for what, obj, methods in (("clock", clock, ("now_utc", "monotonic_ms")),
                                   ("store", store, ("get_assertion_set",)),
                                   ("programs", programs, ("registered_program_hash",))):
            if obj is None or not all(callable(getattr(obj, m, None)) for m in methods):
                raise SpecInvalid(f"QARunner: {what} must provide {', '.join(methods)}")
        self._adapter_ref = ComponentRef(id=str(getattr(qa_adapter, "adapter_id", "")),
                                         version=str(getattr(qa_adapter, "adapter_version", "")))
        self._tool = ToolRef(name=self._adapter_ref.id, version=self._adapter_ref.version)
        self._qa = qa_adapter
        self._check_adapter = check_adapter
        self._policy = policy
        self._store = store
        self._lineage = lineage
        self._programs = programs
        self._clock = clock
        self._cfg = config
        self._evidence = evidence
        self._checks = checks if checks is not None else UnverifiedChecks()
        self._post = postconditions if postconditions is not None else NotEvaluatedPostcondition()
        self._recovery = recovery
        self._halt = halt if halt is not None else HaltLatch()
        self._cancel = cancel

    @property
    def halt(self) -> HaltLatch:
        return self._halt

    # -- public ----------------------------------------------------------------------------------------------
    def run(self, spec: TestSpec, *, run_id: str, pinned_assertion_set_hash: str) -> RunResult:
        if type(spec) is not TestSpec:
            raise SpecInvalid("run takes a TestSpec")
        if type(run_id) is not str or _ID.fullmatch(run_id) is None:
            raise SpecInvalid("run_id must match the Id pattern")
        limits = effective_limits(spec.budget, self._cfg.caps)
        c = _Ctx(spec=spec, run_id=run_id, pin=pinned_assertion_set_hash, limits=limits,
                 deadlines=Deadlines(self._clock, limits), budgets=Budgets(limits))
        c.ledger = ReceiptLedger(run_id)
        c.checkpoints = CheckpointLog(run_id, self._clock)
        c.started_at = self._clock.now_utc()
        cleanup_signal = Signal.UNKNOWN
        env_after = Signal.UNKNOWN
        try:
            try:
                self._prepare(c)
                self._steps(c)
                self._verify(c)
            except (_Abort, QAError) as exc:
                c.abort = self._abort_from(c, exc)
            if c.abort is None and c.session is not None:
                env_after = _signal(self._attest(self._checks.environment_healthy, "after"))
        finally:
            cleanup_signal = self._cleanup(c)
        return self._seal(c, cleanup_signal, env_after)

    # -- helpers ---------------------------------------------------------------------------------------------
    @staticmethod
    def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Call an adapter operation. Typed QA errors pass through; anything else is an ``AdapterCrash`` (only the
        exception's class name is kept: its message is untrusted)."""
        try:
            return fn(*args, **kwargs)
        except QAError:
            raise
        except Exception as exc:
            raise AdapterCrash(f"the adapter raised {type(exc).__name__}") from exc

    @staticmethod
    def _attest(fn: Any, *args: Any) -> Optional[bool]:
        try:
            value = fn(*args)
        except Exception:
            return None
        return value if type(value) is bool else None

    def _boundary(self, c: _Ctx) -> None:
        """Checked before every phase and operation: batch halt, operator cancel, run deadline."""
        self._halt.check()
        if self._cancel is not None and self._cancel.cancelled:
            raise RunCancelled("the run was cancelled", reason=self._cancel.reason)
        c.deadlines.check_run()

    @staticmethod
    def _step_check(c: _Ctx, timer: StepTimer) -> None:
        """Run deadline first (an infrastructure deadline outranks a step budget), then the step deadline."""
        c.deadlines.check_run()
        timer.check()

    def _emit(self, c: _Ctx, kind: str, record: Any) -> None:
        if isinstance(record, dict):
            payload, phash = record, canon.canon_hash(record)
        else:
            payload, phash = record.to_json(with_hash=True), record.canon_hash()
        event = c.chain.link(kind, payload, phash)
        c.events.append(event)
        if self._evidence is not None:
            try:
                self._evidence.append(event)
            except Exception as exc:
                raise EvidenceUnavailable(f"the evidence stream refused an append ({type(exc).__name__})") from exc

    def _abort_from(self, c: _Ctx, exc: BaseException) -> _Abort:
        if isinstance(exc, _Abort):
            return exc
        if isinstance(exc, RunCancelled):
            return _Abort(code=None, skip=SkipReason.NOT_REACHED, cancelled=True)
        if isinstance(exc, DeadlineExceeded):
            if exc.scope == SCOPE_RUN:        # an infrastructure deadline (ADR section 7: ENVIRONMENT_FAIL)
                return _Abort(code="RESOURCE_LIMIT_EXCEEDED", skip=SkipReason.ENVIRONMENT, note="run deadline")
            return _Abort(code="ACTION_TIMEOUT", skip=SkipReason.BUDGET_EXHAUSTED, note=f"{exc.scope} deadline")
        if isinstance(exc, BudgetExceeded):
            return _Abort(code="ACTION_TIMEOUT", skip=SkipReason.BUDGET_EXHAUSTED, note=f"budget {exc.kind}")
        if isinstance(exc, EvidenceUnavailable):
            return _Abort(code="ENVIRONMENT_UNHEALTHY", skip=SkipReason.ENVIRONMENT, note="evidence unavailable")
        if isinstance(exc, (ReceiptChainBroken, IllegalTransition)):
            self._halt.trip(exc)
            return _Abort(code=exc.code, skip=SkipReason.HALTED_INTEGRITY)
        if isinstance(exc, QAError) and exc.run_reason == REASON_ASSERTION_INTEGRITY:
            self._halt.trip(exc)           # idempotent: the verifier's own latch use already tripped it
            return _Abort(code=exc.code, skip=SkipReason.HALTED_INTEGRITY)
        if isinstance(exc, QAAdapterError):
            if exc.code == "ACTION_BLOCKED_BY_POLICY":
                return _Abort(code=exc.code, skip=SkipReason.POLICY_DENIED)
            if exc.code in ENVIRONMENT_CODES:
                return _Abort(code=exc.code, skip=SkipReason.ENVIRONMENT)
            return _Abort(code=exc.code, skip=SkipReason.NOT_REACHED)
        return _Abort(code=getattr(exc, "code", None), skip=SkipReason.NOT_REACHED)

    # -- PREPARE ---------------------------------------------------------------------------------------------
    def _prepare(self, c: _Ctx) -> None:
        spec = c.spec
        self._boundary(c)
        for step in spec.steps:
            if step.target.kind is not TargetKind.UI:
                raise UnsupportedOperation(f"step {step.step_id}: only UI targets run in this layer")
        # 1. the frozen assertions, by their pinned hash, recomputed and checked against the spec: nothing has
        #    reached an adapter yet
        frozen = load_frozen_assertions(self._store, c.pin, lineage=self._lineage, programs=self._programs,
                                        halt=self._halt)
        spec_pin = assertion_set_pin(spec.assertions)
        if not canon.same_digest(spec_pin, frozen.assertion_set_hash) or \
                {a.assertion_id for a in spec.assertions} != set(frozen.assertion_ids):
            error = AssertionHashMismatch("the spec's assertions are not the pinned assertion set")
            self._halt.trip(error)
            raise error
        c.frozen = frozen
        # 2. RunStarted, appended before the first open_session (ADR section 6 rule 2)
        self._emit(c, "RunStarted", {
            "adapter": {"id": self._adapter_ref.id, "version": self._adapter_ref.version},
            "assertion_set_hash": c.pin, "limits": c.limits.to_json(),
            "model_recovery_enabled": self._cfg.model_recovery_enabled, "recovery_enabled": self._cfg.recovery_enabled,
            "run_id": c.run_id, "runner_version": RUNNER_VERSION, "spec_ref": spec.spec_ref()})
        # 3. the isolated session
        self._boundary(c)
        handle = self._call(self._qa.open_session, SessionProfile(egress_profile=spec.egress_profile,
                                                                  redaction_policy_id=self._cfg.redaction_policy_id))
        if type(handle) is not SessionHandle:
            raise AdapterCrash("open_session did not return a SessionHandle")
        c.session = handle
        # 4. attestations the runner cannot make itself
        c.env_before = _signal(self._attest(self._checks.environment_healthy, "before"))
        c.preconditions = (Signal.OK if not spec.preconditions
                           else _signal(self._attest(self._checks.preconditions_met, spec)))
        c.fixtures = Signal.OK if not spec.fixtures else _signal(self._attest(self._checks.fixtures_match, spec))
        if c.env_before is Signal.FAILED:
            raise _Abort(code="ENVIRONMENT_UNHEALTHY", skip=SkipReason.ENVIRONMENT, note="unhealthy before the run")
        if c.fixtures is Signal.FAILED:
            raise _Abort(code=None, skip=SkipReason.PRECONDITION_UNMET, note="fixture hash mismatch")
        if c.preconditions is Signal.FAILED:
            raise _Abort(code=None, skip=SkipReason.PRECONDITION_UNMET, note="precondition unmet")

    # -- OBSERVE / ACT ---------------------------------------------------------------------------------------
    def _steps(self, c: _Ctx) -> None:
        for index, step in enumerate(c.spec.steps):
            try:
                self._run_step(c, index, step)
            finally:
                if c.ledger.last_for_step(step.step_id) is not None:
                    last = c.ledger.last_for_step(step.step_id)
                    c.step_results.append(StepResult(step_id=step.step_id, result=last.result,
                                                     postcondition=last.postcondition.result,
                                                     receipts=c.ledger.refs_for_step(step.step_id)))

    def _observe(self, c: _Ctx, step: Step, phase: Phase) -> Observation:
        started = self._clock.monotonic_ms()
        obs = self._call(self._qa.observe, c.session, ObserveRequest(phase=phase, step_id=step.step_id))
        c.deadlines.check_observation(started, step.observation_deadline_ms)
        if type(obs) is not Observation or obs.step_id != step.step_id or obs.phase != phase:
            raise AdapterCrash("observe returned an observation for another step or phase")
        if not obs.redaction_applied or obs.redaction_policy_id != self._cfg.redaction_policy_id:
            raise RedactionFailed("the observation was not redacted under the run's redaction policy")
        if c.browser is None and obs.source.browser:
            c.browser = f"{obs.source.browser} {obs.source.browser_version or ''}".strip()
        self._emit(c, "Observation", obs)
        return obs

    def _pick(self, descriptor: Any, result: Any, strategy: LocatorStrategy) -> ResolvedTarget:
        if type(result) is not LocateResult:
            raise AdapterCrash("locate did not return a LocateResult")
        matching = [cand for cand in result.candidates
                    if (descriptor.role is None or cand.role == descriptor.role)
                    and (descriptor.name is None or cand.name == descriptor.name)]
        if not matching:
            raise LocatorNotFound("no candidate matches the authored role and name")
        if len(matching) > 1 or not matching[0].unique:
            raise LocatorAmbiguous("the locator does not resolve to exactly one candidate")     # never the first match
        cand = matching[0]
        if not (cand.visible and cand.enabled):
            raise TargetNotActionable("the matching candidate is hidden or disabled")
        return ResolvedTarget(candidate_ref=cand.candidate_ref, locator=descriptor, strategy=strategy)

    def _locate_authored(self, c: _Ctx, step: Step) -> ResolvedTarget:
        started = self._clock.monotonic_ms()
        result = self._call(self._qa.locate, c.session, step.target.locator, mode=LocateMode.EXACT)
        c.deadlines.check_observation(started, step.observation_deadline_ms)
        return self._pick(step.target.locator, result, LocatorStrategy.AUTHORED)

    def _decide(self, c: _Ctx, step: Step, attempt: int) -> PolicyDecisionRecord:
        started = self._clock.monotonic_ms()
        record: Any = None
        rule = None
        if not callable(getattr(self._policy, "decide", None)):
            rule, why = "policy-missing", "there is no policy: nothing is authorised"
        else:
            try:
                record = self._policy.decide(c.run_id, step, attempt)
            except Exception:
                rule, why = "policy-error", "the policy raised while deciding"
            else:
                if self._clock.monotonic_ms() - started >= self._cfg.policy_timeout_ms:
                    rule, why = "policy-timeout", "the policy did not answer in time"
                elif binding_problem(record, c.run_id, step) is not None:
                    rule, why = "policy-decision-invalid", "the policy's answer does not decide this action"
        if rule is not None:
            record = failsafe_decision(run_id=c.run_id, step=step, attempt=attempt, rule_id=rule, reason=why,
                                       principal=self._cfg.principal, app=c.spec.target.app_id, clock=self._clock)
        c.decisions.append(record)
        self._emit(c, "PolicyDecision", record)
        return record

    def _receipt(self, c: _Ctx, step: Step, attempt: int, ref: str, *, result: ResultCode, post: Postcondition,
                 before: str, after: str, used: Optional[ResolvedTarget], started: int,
                 rationale: Optional[str]) -> ActionReceipt:
        locator = used.locator if used is not None else step.target.locator
        strategy = used.strategy if used is not None else LocatorStrategy.AUTHORED
        receipt = build_receipt(
            run_id=c.run_id, step=step, attempt=attempt, tool=self._tool, policy_decision_ref=ref,
            idempotency_key=idempotency_key(c.run_id, step.step_id), result=result, postcondition=post,
            state_before_hash=before, state_after_hash=after,
            locator_used=LocatorUsed(descriptor=locator, strategy=strategy), recovery_rationale=rationale,
            duration_ms=max(0, self._clock.monotonic_ms() - started))
        c.ledger.append(receipt)
        self._emit(c, "ActionReceipt", receipt)
        return receipt

    def _attempt(self, c: _Ctx, step: Step, attempt: int, before: Observation, ref: str,
                 override: Optional[ResolvedTarget], rationale: Optional[str]) -> _Attempt:
        started = self._clock.monotonic_ms()
        state_before = state_hash(before)
        error: Optional[QAError] = None
        acted = False
        used = override
        result = ResultCode.OK

        def failed(exc: QAError) -> ResultCode:
            return _CODE_BY_RESULT.get(exc.code, ResultCode.ACTION_TIMEOUT if isinstance(exc, DeadlineExceeded)
                                       else ResultCode.ADAPTER_CRASH)
        if used is None:
            try:
                used = self._locate_authored(c, step)
            except QAError as exc:
                error, result = exc, failed(exc)
        if error is None:
            self._boundary(c)       # a halt, a cancel or the run deadline stops the run here: nothing was acted on
            try:
                request = ActionRequest(action=step.action, idempotency_key=idempotency_key(c.run_id, step.step_id),
                                        policy_decision_ref=ref, step_id=step.step_id)
                acted = True
                outcome = self._call(self._qa.act, c.session, request, used)
                if type(outcome) is not ActionOutcome:
                    raise AdapterCrash("act did not return an ActionOutcome")
                if outcome.result is not ResultCode.OK:
                    raise ADAPTER_ERRORS[outcome.result.value](f"the adapter reported {outcome.result.value}")
            except QAError as exc:
                error, result = exc, failed(exc)
        after: Optional[Observation] = None
        if acted:
            try:
                after = self._observe(c, step, Phase.AFTER)
            except QAError as exc:
                error = error or exc
        state_after = state_hash(after) if after is not None else (
            unknown_state_hash(c.run_id, step.step_id, attempt) if acted else state_before)
        post = Postcondition(declared=step.expected_transition, result=PostconditionResult.NOT_EVALUATED)
        if after is not None:
            try:
                got = self._post.evaluate(step, before, after)
                if type(got) is Postcondition and got.declared == step.expected_transition:
                    post = got
            except Exception:
                pass                                          # an unusable checker is "not evaluated", never "satisfied"
        if error is None and post.result is PostconditionResult.NOT_SATISFIED:
            error = ActionTimeout("the declared transition was not observed", signal="TRANSITION_NOT_OBSERVED")
        receipt = self._receipt(c, step, attempt, ref, result=result, post=post, before=state_before,
                                after=state_after, used=used, started=started, rationale=rationale)
        return _Attempt(error=error, acted=acted, postcondition=post, after=after, receipt=receipt)

    def _run_step(self, c: _Ctx, index: int, step: Step) -> None:
        c.machine.to(RunState.OBSERVE)
        timer = c.deadlines.begin_step()
        self._boundary(c)
        before = self._observe(c, step, Phase.BEFORE)
        self._step_check(c, timer)
        c.machine.to(RunState.ACT)
        attempt = 1
        override: Optional[ResolvedTarget] = None
        rationale: Optional[str] = None
        while True:
            self._boundary(c)
            self._step_check(c, timer)
            decision = self._decide(c, step, attempt)
            ref = decision_ref(decision)
            if not is_allowed(decision):                 # DENY and REQUIRE_APPROVAL: the adapter is never called
                self._receipt(c, step, attempt, ref, result=ResultCode.ACTION_BLOCKED_BY_POLICY,
                              post=Postcondition(declared=step.expected_transition,
                                                 result=PostconditionResult.NOT_EVALUATED),
                              before=state_hash(before), after=state_hash(before), used=override,
                              started=self._clock.monotonic_ms(), rationale=None)
                raise _Abort(code="ACTION_BLOCKED_BY_POLICY", skip=SkipReason.POLICY_DENIED,
                             note=str(decision.rule_id))
            made = self._attempt(c, step, attempt, before, ref, override, rationale)
            self._step_check(c, timer)                   # an attempt that overran its deadline ends the run
            if made.error is None:
                break
            route = self._after_failure(c, step, attempt, made, before)
            if route is None:                            # not recoverable here: the run ends
                raise made.error
            if route == "applied":                       # an observed postcondition shows the effect is there
                break
            c.machine.to(RunState.REOBSERVE)
            before = self._observe(c, step, Phase.BEFORE)
            self._step_check(c, timer)
            c.machine.to(RunState.ACT)
            override, rationale = route
            attempt += 1
        c.checkpoints.record(index, state_hash(made.after) if made.after is not None else state_hash(before))

    # -- RECOVERABLE -----------------------------------------------------------------------------------------
    def _after_failure(self, c: _Ctx, step: Step, attempt: int, made: _Attempt, before: Observation) -> Any:
        """Decide what follows a failed attempt: ``None`` (the run ends with the error), ``"applied"`` or a
        ``(target override, rationale)`` pair after moving through ``RECOVERABLE``."""
        error = made.error
        if not isinstance(error, QAAdapterError) or not self._cfg.recovery_enabled or error.code in ENVIRONMENT_CODES:
            return None
        if not (error.retryable or error.recoverable):
            return None
        if error.retryable:
            decision = decide_retry(RetryInputs(
                side_effect=step.side_effect, idempotency=step.idempotency, retry_safety=step.retry_safety,
                permitted_recovery=tuple(step.permitted_recovery), may_have_executed=made.acted,
                has_idempotency_key=bool(made.receipt.idempotency_key),
                postcondition=made.postcondition.result, error_retryable=True,
                recovery_enabled=self._cfg.recovery_enabled, retries_left=c.budgets.retries_left))
            if decision.already_applied:
                return "applied"                       # the step is done: the machine never leaves ACT
            if decision.allow:
                c.machine.to(RunState.RECOVERABLE)
                try:
                    c.budgets.charge_retry()
                except BudgetExceeded as exc:
                    raise _Abort(code=error.code, skip=SkipReason.BUDGET_EXHAUSTED, note=exc.kind) from exc
                return (None, None)
            if decision.reason == "BUDGET_EXHAUSTED":
                raise _Abort(code=error.code, skip=SkipReason.BUDGET_EXHAUSTED, note="RETRIES")
        if self._recovery is not None and error.recoverable:
            c.machine.to(RunState.RECOVERABLE)
            return self._hook(c, step, attempt, made, before, error)
        return None

    def _hook(self, c: _Ctx, step: Step, attempt: int, made: _Attempt, before: Observation,
              error: QAAdapterError) -> Any:
        allowed = {PermittedRecovery.SEMANTIC_LOCATOR: RecoveryStrategy.SEMANTIC_LOCATOR,
                   PermittedRecovery.MODEL_CANDIDATE: RecoveryStrategy.MODEL_CANDIDATE}
        permitted = {strategy for kind, strategy in allowed.items() if kind in step.permitted_recovery}
        if RecoveryStrategy.MODEL_CANDIDATE in permitted and not self._cfg.model_recovery_enabled:
            permitted.discard(RecoveryStrategy.MODEL_CANDIDATE)
        if not permitted:
            return None
        try:
            c.budgets.charge_recovery_attempt()
        except BudgetExceeded as exc:
            raise _Abort(code=error.code, skip=SkipReason.BUDGET_EXHAUSTED, note=exc.kind) from exc
        try:
            response = self._recovery.recover(RecoveryRequest(
                run_id=c.run_id, step=step, observation=made.after or before, error_code=error.code, attempt=attempt + 1,
                recovery_attempts_left=c.budgets.recovery_attempts_left, model_calls_left=c.budgets.model_calls_left,
                tokens_left=c.budgets.tokens_left))
        except Exception:
            return None                                     # a failing hook recovers nothing
        if type(response) is not RecoveryResponse or response.strategy not in permitted or response.target is None:
            return None
        wanted = (LocatorStrategy.SEMANTIC_ALTERNATE if response.strategy is RecoveryStrategy.SEMANTIC_LOCATOR
                  else LocatorStrategy.MODEL_CANDIDATE)
        if response.target.strategy is not wanted:
            return None
        calls = max(1, response.model_calls) if response.strategy is RecoveryStrategy.MODEL_CANDIDATE \
            else response.model_calls
        try:
            c.budgets.charge_model(calls, response.tokens)     # zero unless the reviewed spec allows model calls
        except BudgetExceeded as exc:
            raise _Abort(code=error.code, skip=SkipReason.BUDGET_EXHAUSTED, note=exc.kind) from exc
        # the proposed target must be the only candidate a semantic locate finds, and actionable
        started = self._clock.monotonic_ms()
        found = self._call(self._qa.locate, c.session, response.target.locator, mode=LocateMode.SEMANTIC)
        c.deadlines.check_observation(started, step.observation_deadline_ms)
        if type(found) is not LocateResult or len(found.candidates) != 1:
            return None
        cand = found.candidates[0]
        if cand.candidate_ref != response.target.candidate_ref or not (cand.unique and cand.visible and cand.enabled):
            return None
        authored = step.target.locator              # the frozen descriptor, never the proposal's
        if not _same_label(authored.role, cand.role) or not _same_label(authored.name, cand.name):
            return None                             # a different control: "Delete" is never a recovery of "Save"
        c.locator_recovery_used = True
        return (response.target, response.rationale or None)

    # -- VERIFY ----------------------------------------------------------------------------------------------
    def _verify(self, c: _Ctx) -> None:
        c.machine.to(RunState.VERIFY)
        self._boundary(c)
        # load again, by hash, at verification time: a spec edited since PREPARE is refused with nothing evaluated
        again = load_frozen_assertions(self._store, c.pin, lineage=self._lineage, programs=self._programs,
                                       halt=self._halt)
        if again != c.frozen:
            error = AssertionHashMismatch("the assertion set changed between PREPARE and VERIFY")
            self._halt.trip(error)
            raise error
        verifier = DeterministicVerifier(
            programs=self._programs, halt=self._halt,
            qa_adapter=_Shield(self._qa),
            check_adapter=_Shield(self._check_adapter) if self._check_adapter is not None else None)
        needs_workspace = any(a.executor is Executor.SCRIPT for a in again.assertions)
        if needs_workspace and self._check_adapter is not None and self._cfg.check_profile is not None:
            ws = self._call(self._check_adapter.open_workspace, self._cfg.check_profile)
            if type(ws) is WorkspaceHandle:
                c.workspace = ws
        for assertion in again.assertions:
            self._boundary(c)
            result = verifier.verify_assertion(again, assertion.assertion_id, session=c.session,
                                               workspace=c.workspace)
            c.results.append(result)
            self._emit(c, "AssertionResult", result)

    # -- cleanup and seal ------------------------------------------------------------------------------------
    def _cleanup(self, c: _Ctx) -> Signal:
        signal = Signal.OK
        if c.session is not None:
            signal = close_and_report(self._qa.close_session, c.session, c.session.session_id).signal
        if c.workspace is not None:
            ws = close_and_report(self._check_adapter.close_workspace, c.workspace, c.workspace.workspace_id).signal
            signal = signal if ws is Signal.OK else ws
        if c.session is None and c.workspace is None:
            return Signal.OK                       # nothing was opened, so nothing can leak
        return signal

    def _manifest(self, c: _Ctx, outcome: Outcome, reason: Optional[str], elapsed_ms: int) -> RunManifest:
        declared = tuple(a.assertion_id for a in c.spec.assertions)
        executed = tuple(r.assertion_id for r in c.results)
        skip = c.abort.skip if c.abort is not None else SkipReason.NOT_REACHED
        skipped = tuple(SkippedAssertion(assertion_id=i, reason_code=skip) for i in declared if i not in executed)
        pin = c.pin if isinstance(c.pin, str) and canon.is_contract_ref(c.pin) else assertion_set_pin(c.spec.assertions)
        trace = self._cfg.trace_id or canon.digest_hex({"run_id": c.run_id, "spec_ref": c.spec.spec_ref()})[:32]
        return RunManifest(
            run_id=c.run_id, spec_ref=c.spec.spec_ref(), adapter=self._adapter_ref, assertion_set_hash=pin,
            assertions_declared=declared, assertions_executed=executed, assertions_skipped=skipped,
            egress_policy_id=self._cfg.egress_policy_id, redaction_policy_id=self._cfg.redaction_policy_id,
            model_recovery_enabled=self._cfg.model_recovery_enabled, recovery_enabled=self._cfg.recovery_enabled,
            outcome=outcome, trace_id=trace, assertion_results=tuple(c.results), browser=c.browser,
            outcome_reason=reason, resource_limits=c.limits.to_json(), step_results=tuple(c.step_results),
            versions=(VersionPin(component="agent-harness-qa-runner", version=RUNNER_VERSION),),
            counters=Counters(infrastructure_ms=elapsed_ms, model_calls=c.budgets.model_calls,
                              tokens=c.budgets.tokens),
            started_at=c.started_at, finished_at=self._clock.now_utc())

    def _conservative(self, c: _Ctx, got: Classification) -> Classification:
        """The one runner-side override, and it only makes the result more conservative. A locator recovery is not
        sealed as a ``FaultEvent`` until plan H3, so the classifier cannot see it; until then a run that used one never
        ends ``PASS`` or ``PRODUCT_FAIL``."""
        if c.locator_recovery_used and got.outcome in (Outcome.PASS, Outcome.PRODUCT_FAIL):
            return Classification(outcome=Outcome.INCONCLUSIVE, reason="RECOVERY_UNVERIFIED",
                                  verdict=outcome_to_verdict(Outcome.INCONCLUSIVE), halt_batch=False,
                                  review_required=True, rule=5, details=("LOCATOR_RECOVERY_NOT_SEALED",))
        return got

    def _seal(self, c: _Ctx, cleanup: Signal, env_after: Signal) -> RunResult:
        elapsed = c.deadlines.elapsed_ms()
        try:
            provisional = self._manifest(c, Outcome.INCONCLUSIVE, "PENDING_CLASSIFICATION", elapsed)
        except QAError:      # the accounting cannot be recorded as anything but an integrity failure
            provisional = self._manifest(c, Outcome.INCONCLUSIVE, REASON_ASSERTION_INTEGRITY, elapsed)
        codes = (c.abort.code,) if c.abort is not None and c.abort.code else ()
        facts = RunFacts(
            manifest=provisional, assertions=c.spec.assertions, environment_before=c.env_before,
            environment_after=env_after, preconditions=c.preconditions, fixture_hash=c.fixtures, cleanup=cleanup,
            policy=(Signal.FAILED if any(not is_allowed(d) for d in c.decisions) else Signal.OK),
            recovery_checks=Signal.OK, failure_reproduced=Signal.OK, error_codes=codes)
        classification = self._conservative(c, classify(facts))
        if not is_allowed_transition(c.machine.state, OUTCOME_STATE[classification.outcome]):
            # the table refuses this move: a defect in this runner, never a reason to certify anything
            self._halt.trip(IllegalTransition("the classified outcome is not reachable from the current state"))
            classification = Classification(
                outcome=Outcome.INCONCLUSIVE, reason=REASON_ASSERTION_INTEGRITY,
                verdict=outcome_to_verdict(Outcome.INCONCLUSIVE), halt_batch=True, review_required=True, rule=1,
                details=("ILLEGAL_TRANSITION",))
        sealed = True
        review: Optional[ReviewItem] = review_item_for(c.run_id, classification)
        manifest = self._manifest(c, classification.outcome, classification.reason, elapsed)
        try:
            if review is not None:
                self._emit(c, "ReviewItem", review)
            self._emit(c, "RunSealed", {
                "abort": ({"cancelled": c.abort.cancelled, "code": c.abort.code, "note": c.abort.note,
                           "skip": c.abort.skip.value} if c.abort is not None else None),
                "classification": classification.to_json(), "manifest_hash": manifest.canon_hash(),
                "receipt_head": c.ledger.head, "run_id": c.run_id})
        except EvidenceUnavailable:                 # an outcome that could not be sealed is not certified
            sealed = False
            classification = Classification(
                outcome=Outcome.ENVIRONMENT_FAIL, reason="EVIDENCE_UNAVAILABLE",
                verdict=outcome_to_verdict(Outcome.ENVIRONMENT_FAIL), halt_batch=False, review_required=False, rule=2,
                details=("RUN_NOT_SEALED",))
            review = None
            manifest = self._manifest(c, classification.outcome, classification.reason, elapsed)
        self._settle(c, classification)
        return RunResult(
            run_id=c.run_id, classification=classification, manifest=manifest,
            exit_code=exit_code(classification.outcome), review_item=review, states=c.machine.history,
            receipts=c.ledger.receipts, receipt_links=c.ledger.links, policy_decisions=tuple(c.decisions),
            checkpoints=c.checkpoints.items, events=tuple(c.events),
            cancelled=bool(c.abort is not None and c.abort.cancelled), sealed=sealed)

    def _settle(self, c: _Ctx, classification: Classification) -> None:
        """Move the machine to the state that records the outcome (``INCONCLUSIVE`` continues to ``REVIEW``)."""
        target = OUTCOME_STATE[classification.outcome]
        c.machine.to(target)
        if target is RunState.INCONCLUSIVE:
            c.machine.to(RunState.REVIEW)


__all__ = ["RUNNER_VERSION", "RunConfig", "RunResult", "QARunner"]
