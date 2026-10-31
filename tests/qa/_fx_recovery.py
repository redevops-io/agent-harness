# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test infrastructure for deterministic recovery: a fake adapter with benign-drift knobs (renamed attribute, moved
element, delayed render, modal overlay, lost session, stale fixture) and with real regressions, builders for specs and
runners that take a recovery engine, a postcondition checker that evaluates the Save and navigation steps, and the
invariants every recovery run must satisfy.

Test code only: nothing here ships in ``agent_harness/``. The fake is deterministic: time moves only through the fake
clock (per-operation latency) and scheduled page changes fire when the clock passes their time.
"""
import copy

from agent_harness.qa import canon, receipts as RC, records as R
from agent_harness.qa.errors import ActionTimeout, EnvironmentUnhealthy, SessionLost
from agent_harness.qa.recovery import DeterministicRecovery
from agent_harness.qa.runner import QARunner, RunConfig
from agent_harness.qa.runstate import TERMINAL_STATES, RunState
from agent_harness.qa.specstore import InMemoryProgramRegistry, InMemorySpecStore
from agent_harness.qa.verifier import ApprovedLineage

from ._fx_verify import PROGRAM_HASH, PROGRAM_KEY, REF_DOCS, REF_PIN, SCRIPT_DOCS, SCRIPT_PIN
from ._fx_runner import (APP, AttestAll, FakeClock, FakeQAAdapter, Fault, RecordingStream, REDACTION_ID, EGRESS_ID, Env,
                  default_policy, make_config, make_spec, nav_step, operations, save_step, step_json)

S = RunState
BUDGET = {"deadline_ms": 120000, "max_model_calls": 0, "max_recovery_attempts": 2, "max_retries": 1, "max_tokens": 0}


def norm(text):
    return " ".join((text or "").casefold().split())


class FakeH3Adapter(FakeQAAdapter):
    """The runner fake plus: semantic locating (normalised role/name, label, test id), moved elements that only a semantic
    locate finds, scheduled page changes, a fresh page for every new session, a stale-fixture fault and a real
    ``reset``."""

    def __init__(self, clock=None, log=None):
        self.timed = []                  # (fake-clock ms, callable(adapter)) fired once, in order
        self.stale = False
        self.resets = []
        self.reset_ok = True
        self.session_count = 0
        super().__init__(clock, log)
        self.semantic = {}               # unused here: the semantic match below is by attributes

    # -- page ------------------------------------------------------------------------------------------------
    def blank(self):
        self.url = "about:blank"
        self.elements = {"doc": {"role": "document", "name": "app", "test_id": None, "visible": True, "enabled": True}}
        self.values = {}

    def schedule(self, at_ms, fn):
        self.timed.append((at_ms, fn))
        self.timed.sort(key=lambda item: item[0])

    def _enter(self, op, detail=None):
        super()._enter(op, detail)
        if self.clock is not None:
            while self.timed and self.timed[0][0] <= self.clock.ms:
                self.timed.pop(0)[1](self)

    # -- protocol overrides ----------------------------------------------------------------------------------
    def open_session(self, profile):
        super().open_session(profile)
        self.session_count += 1
        if self.session_count > 1:
            self.blank()                 # a new browser context starts with no page state
        return R.SessionHandle(session_id=f"sess-{self.session_count}")

    def _match(self, d, mode):
        out = []
        for ref, e in self.elements.items():
            exact = mode is R.LocateMode.EXACT
            if exact and e.get("moved"):
                continue                 # the authored structural locator no longer reaches it
            same = (lambda a, b: a == b) if exact else (lambda a, b: norm(a) == norm(b))
            if d.role is not None and not same(d.role, e["role"]):
                continue
            if d.name is not None and not same(d.name, e["name"]):
                continue
            if d.test_id is not None and d.test_id != e["test_id"]:
                continue
            if d.label is not None and (exact or norm(d.label) != norm(e.get("label"))):
                continue
            out.append(ref)
        return out

    def act(self, s, action, target):
        if self.stale and action.action.kind is R.ActionKind.CLICK:
            self._enter("act", (action.step_id, action.action.kind.value, target.candidate_ref, target.strategy.value,
                                action.policy_decision_ref, action.idempotency_key))
            raise EnvironmentUnhealthy("the fixture is stale")
        return super().act(s, action, target)

    def reset(self, s, scope):
        self._enter("reset", scope.value)
        fault = self._fault("reset")
        if fault is not None:
            raise fault.error
        self.resets.append(scope)
        if scope is R.ResetScope.FIXTURE and self.reset_ok:
            self.stale = False
            self.blank()
        return R.ResetResult(ok=self.reset_ok, scope=scope)


# ---------------------------------------------------------------------------------------------------------------
# steps and specs
# ---------------------------------------------------------------------------------------------------------------

def ui_step(step_id, kind, role, name, *, test_id=None, label=None, **kw):
    step = step_json(step_id, kind, role, name, **kw)
    if test_id is not None:
        step["target"]["locator"]["test_id"] = test_id
    if label is not None:
        step["target"]["locator"]["label"] = label
    return step


def keyed_save(**kw):
    """A Save step that may be replayed once its effect is known to be absent: keyed, at most once."""
    base = dict(side_effect="WRITES_INTERNAL", idempotency="AT_MOST_ONCE", retry_safety="SAFE_WITH_IDEMPOTENCY_KEY",
                transition="url ends /saved")
    base.update(kw)
    return ui_step("s-save", "click", "button", "Save", **base)


def recover_spec(*, nav=(), save=("SEMANTIC_LOCATOR",), keyed=True, test_id=None, steps=None, **changes):
    nav_s = nav_step(permitted=list(nav))
    if keyed:
        save_s = keyed_save(permitted=list(save), test_id=test_id)
    else:
        save_s = ui_step("s-save", "click", "button", "Save", side_effect="WRITES_INTERNAL", idempotency="AT_MOST_ONCE",
                         retry_safety="UNSAFE", transition="url ends /saved", permitted=list(save), test_id=test_id)
    return make_spec(steps if steps is not None else [nav_s, save_s], **changes)


class H3Postconditions:
    """Deterministic checker: the navigation step is done when the URL ends /form, the Save step when it ends /saved."""

    WANT = {"s-nav": "/form", "s-save": "/saved"}

    def evaluate(self, step, before, after):
        want = self.WANT.get(step.step_id)
        if want is None:
            return R.Postcondition(declared=step.expected_transition, result=R.PostconditionResult.NOT_EVALUATED)
        ok = after.url is not None and after.url.endswith(want)
        return R.Postcondition(declared=step.expected_transition, observed="url ends " + want if ok else "url differs",
                               result=R.PostconditionResult.SATISFIED if ok else R.PostconditionResult.NOT_SATISFIED)


# ---------------------------------------------------------------------------------------------------------------
# page drift and faults (adapter_setup callables)
# ---------------------------------------------------------------------------------------------------------------

def after_form(adapter, change):
    """Run ``change(adapter)`` every time the form page is installed (after the standard Save and Delete buttons)."""
    base = adapter.on_navigate

    def nav(a, url):
        base(a, url)
        if url.endswith("/form"):
            change(a)
    adapter.on_navigate = nav


def rename_test_id(adapter):
    after_form(adapter, lambda a: a.elements["btn-save"].__setitem__("test_id", "save-primary"))


def move_save(adapter):
    after_form(adapter, lambda a: a.elements["btn-save"].__setitem__("moved", True))


def rename_label(adapter):
    after_form(adapter, lambda a: a.elements["btn-save"].__setitem__("name", "SAVE  "))


def delay_save(adapter, at_ms=350, latency=100):
    adapter.latency["observe"] = latency

    def hide(a):
        element = a.elements.pop("btn-save")
        a.schedule(at_ms, lambda b: b.elements.__setitem__("btn-save", element))
    after_form(adapter, hide)


def overlay_on_save(adapter, until_ms=350, latency=100):
    adapter.latency["observe"] = latency

    def cover(a):
        a.elements["btn-save"]["enabled"] = False
        a.schedule(until_ms, lambda b: b.elements["btn-save"].__setitem__("enabled", True))
    after_form(adapter, cover)


def lose_session_on_save(adapter, times=1):
    adapter.faults["act"] = [None] + [Fault(SessionLost("the browser context was lost"))] * times


def stale_fixture(adapter):
    adapter.stale = True


def remove_save(adapter):
    after_form(adapter, lambda a: a.elements.pop("btn-save"))


def duplicate_save(adapter):
    def dup(a):
        a.elements["btn-save-2"] = dict(a.elements["btn-save"])
    after_form(adapter, dup)


def reuse_test_id_on_delete(adapter, test_id="save"):
    """The authored test id now sits on the Delete button and Save is gone from the semantic tree."""
    def swap(a):
        a.elements["btn-delete"]["test_id"] = test_id
        a.elements.pop("btn-save")
    after_form(adapter, swap)


def wrong_total(adapter):
    """A real regression: Save works, but the page then shows the wrong result."""
    def save(a, action):
        a.url = f"{APP}/saved"
        a.values["save-status"] = "Saved: total 99 (expected 100)"
    adapter.effects[("btn-save", "click")] = save


def compose(*fns):
    def setup(adapter):
        for fn in fns:
            fn(adapter)
    return setup


# ---------------------------------------------------------------------------------------------------------------
# runner assembly
# ---------------------------------------------------------------------------------------------------------------

def make_recovery_env(spec=None, *, adapter_setup=None, engine=None, config=None, config_kw=None, policy=None,
                postconditions=None, checks=None, mod=None, no_engine=False, **runner_kw):
    """``mod`` swaps in a mutated copy of ``runner.py``; the engine may come from a mutated ``recovery.py``."""
    spec = spec if spec is not None else recover_spec()
    clock = FakeClock()
    log = []
    adapter = FakeH3Adapter(clock, log)
    if adapter_setup is not None:
        adapter_setup(adapter)
    stream = RecordingStream(log)
    kw = dict(config_kw or {})
    kw.setdefault("recovery_enabled", True)
    config = config if config is not None else make_config(mod, **kw)
    policy = policy if policy is not None else default_policy(spec)
    store = InMemorySpecStore({REF_PIN: REF_DOCS, SCRIPT_PIN: SCRIPT_DOCS})
    lineage = ApprovedLineage([REF_PIN, SCRIPT_PIN])
    programs = InMemoryProgramRegistry({PROGRAM_KEY: PROGRAM_HASH})
    if engine is None and not no_engine:
        engine = DeterministicRecovery()
    runner = (mod.QARunner if mod is not None else QARunner)(
        qa_adapter=adapter, policy=policy, store=store, lineage=lineage, programs=programs, clock=clock,
        config=config, evidence=stream, checks=checks if checks is not None else AttestAll(),
        postconditions=postconditions if postconditions is not None else H3Postconditions(),
        recovery_engine=None if no_engine else engine, **runner_kw)
    return Env(spec=spec, clock=clock, adapter=adapter, policy=policy, stream=stream, runner=runner, config=config,
               log=log)


def check_recovery_invariants(env, result):
    """What must hold for every run, with or without recovery."""
    m = result.manifest
    assert result.states[0] is S.PREPARE and result.states[-1] in TERMINAL_STATES
    assert (S.REVIEW in result.states) is (result.outcome is R.Outcome.INCONCLUSIVE)
    assert m.outcome is result.outcome and m.accounting_ok()
    assert (result.exit_code == 0) is (result.outcome is R.Outcome.PASS)
    if result.outcome is R.Outcome.PASS:
        assert not m.assertions_skipped and all(r.check.verdict is R.Verdict.PASS for r in m.assertion_results)
        assert all(e.recovery.result is R.RecoveryResult.RECOVERED and e.recovery.postcondition_verified
                   and e.recovery.strategy not in (R.RecoveryStrategy.SEMANTIC_LOCATOR,
                                                   R.RecoveryStrategy.MODEL_CANDIDATE) for e in m.fault_events)
    assert RC.verify_events(list(result.events)) is not None
    RC.verify_chain(result.receipts, result.receipt_links)
    RC.verify_manifest_receipts(m, result.receipts)
    # every act followed an ALLOW decision whose hash is in the receipt
    refs = {canon.to_contract_ref(d.canon_hash()): d for d in result.policy_decisions}
    allowed = {ref for ref, d in refs.items() if d.decision is R.PolicyDecisionKind.ALLOW}
    assert {r.policy_decision_ref for r in result.receipts} <= set(refs)
    assert all(a[1][4] in allowed for a in operations(env.adapter, "act"))
    # every session opened was closed, and the last call is a close
    opens, closes = operations(env.adapter, "open_session"), operations(env.adapter, "close_session")
    assert len(closes) == env.adapter.opened <= len(opens)
    if closes and len(opens) == env.adapter.opened:
        assert env.adapter.calls[-1][0] == "close_session"
    # fault events: each pair is bound by hash; detection precedes recovery in the evidence stream
    kinds = [e["kind"] for e in result.events]
    assert kinds.count("FaultDetection") == kinds.count("FaultRecovery") == len(m.fault_events)
    detections = [e for e in result.events if e["kind"] == "FaultDetection"]
    recoveries = [e for e in result.events if e["kind"] == "FaultRecovery"]
    for ev, det, rec in zip(m.fault_events, detections, recoveries):
        assert det["seq"] < rec["seq"]
        assert ev.recovery.detection_ref == canon.to_contract_ref(ev.detection.canon_hash())
        assert canon.to_contract_ref(det["payload_hash"]) == ev.recovery.detection_ref
        assert canon.to_contract_ref(rec["payload_hash"]) == canon.to_contract_ref(ev.recovery.canon_hash())
        if ev.recovery.result is R.RecoveryResult.RECOVERED:
            assert ev.recovery.attempted and ev.recovery.postcondition_verified
        if not ev.recovery.attempted:
            assert ev.recovery.result is R.RecoveryResult.NOT_ATTEMPTED
        # why the recovery as a whole did not recover: absent on a recovery, present when nothing was recovered
        # (a recovery cut short by a cancel, a deadline or an integrity halt may have none), and one rejection token
        # per candidate considered at most
        if ev.recovery.result is R.RecoveryResult.RECOVERED:
            assert ev.recovery.reason_code is None
        elif ev.recovery.result in (R.RecoveryResult.NOT_RECOVERED, R.RecoveryResult.NOT_ATTEMPTED):
            assert isinstance(ev.recovery.reason_code, R.RecoveryReason)
        assert len(ev.recovery.candidates_rejected_reasons) <= ev.recovery.candidates_considered
    return result


def run(env, **kw):
    return check_recovery_invariants(env, env.run(**kw))


def classification(r):
    return r.classification.outcome.value, r.classification.reason


def events_of(r, kind):
    return [e for e in r.events if e["kind"] == kind]


def acts(env, kind="click"):
    return [c for c in operations(env.adapter, "act") if c[1][1] == kind]


def clicked(env, ref):
    return [c for c in acts(env) if c[1][2] == ref]
