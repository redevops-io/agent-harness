# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``ui_action`` action policy: fail-closed, auditable, and blind to what the page says.

:class:`DefaultActionPolicy` implements the neutral ``ActionPolicy`` port. It denies everything that is not explicitly
allowed:

* **action kinds** must be in the operator's ``allowed_action_kinds`` (default: none, so everything is denied);
* **side-effect classes** are matched against the reviewed ``TestSpec.allowed_side_effects``. ``NONE`` needs no
  allowance (there is nothing to authorize); ``READS_EXTERNAL`` and ``WRITES_INTERNAL`` must be listed; ``WRITES_EXTERNAL``,
  ``NOTIFIES_HUMAN`` and ``SPENDS_MONEY`` must be listed **and** the target origin must be one of the operator's
  ``local_sink_origins``. A class with no row in :data:`SIDE_EFFECT_RULES` (a member added later) is denied;
* **target origins** must be in the operator's ``allowed_origins``. The origin comes from the reviewed spec only: a
  ``NAVIGATE`` step's own URL, or else the origin of the last spec-authored navigation before the step, or else the
  operator's ``start_origin``. If none exists the origin is unknown and the action is denied;
* a step whose ``idempotency`` is ``UNKNOWN`` is denied (an unknown class is a denial: the literal reading);
* a step that is not exactly the reviewed spec's step of that id, or whose target is not a UI target, is denied.

**The policy never sees the page.** ``decide`` takes ``(run_id, step, attempt)`` and this module does not import
``Observation`` or read any observed value: DOM text, titles, URLs and model or agent output cannot add a rule, widen
a list or change a decision (the untrusted-content rule). Allow-lists come from the frozen spec and operator
configuration, both fixed before the first adapter call.

Every decision, allow or deny, is returned as a :class:`PolicyDecisionRecord`; its ``canon-v1`` hash goes into
``ActionReceipt.policy_decision_ref`` (as ``rcv1:`` plus the same digits). The runner (not the policy) turns an
exception, a timeout, a malformed or mismatched record into a ``DENY`` record built by :func:`failsafe_decision`.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple
from urllib.parse import urlsplit

from . import canon
from .errors import SpecInvalid
from .ports import Clock
from .records import (ActionKind, Idempotency, PolicyDecisionKind, PolicyDecisionRecord, SideEffect, Step, TargetKind,
                      TestSpec)

PHASE = "ui_action"
POLICY_PROVIDER = "qa-default-policy"
FAILSAFE_PROVIDER = "qa-runner-failsafe"

NEED_NOTHING = "NOTHING"
NEED_LISTED = "LISTED"
NEED_LISTED_AND_LOCAL_SINK = "LISTED_AND_LOCAL_SINK"

#: how each side-effect class is handled. A class missing from this table is denied (unknown class, fail closed).
SIDE_EFFECT_RULES: Dict[SideEffect, str] = MappingProxyType({
    SideEffect.NONE: NEED_NOTHING,
    SideEffect.READS_EXTERNAL: NEED_LISTED,
    SideEffect.WRITES_INTERNAL: NEED_LISTED,
    SideEffect.WRITES_EXTERNAL: NEED_LISTED_AND_LOCAL_SINK,
    SideEffect.NOTIFIES_HUMAN: NEED_LISTED_AND_LOCAL_SINK,
    SideEffect.SPENDS_MONEY: NEED_LISTED_AND_LOCAL_SINK,
})

_DEFAULT_PORTS = {"http": 80, "https": 443}


def origin_of(url: Any) -> Optional[str]:
    """Normalized ``scheme://host[:port]`` of an ``http``/``https`` URL, or ``None`` for anything else.

    Rejects: non-strings, other schemes, user information, an empty host, a bad port, whitespace or control characters.
    The default port is dropped, the scheme and host are lowercased, so equal origins compare equal."""
    if type(url) is not str or not url or any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in url):
        return None
    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if scheme not in _DEFAULT_PORTS or not host or "@" in parts.netloc:
        return None
    host = host.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _origins(name: str, values: Iterable[Any]) -> FrozenSet[str]:
    out = set()
    for value in values:
        origin = origin_of(value)
        if origin is None:
            raise SpecInvalid(f"{name}: {value!r} is not a bare http(s) origin")
        parts = urlsplit(value)
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise SpecInvalid(f"{name}: {value!r} is not a bare http(s) origin")
        out.add(origin)
    return frozenset(out)


@dataclass(frozen=True)
class Verdict:
    """The outcome of the pure rule evaluation: ``allow`` or ``deny`` with a stable ``rule_id`` and a fixed reason."""

    allow: bool
    rule_id: str
    reason: str


def _deny(rule_id: str, reason: str) -> Verdict:
    return Verdict(False, rule_id, reason)


class DefaultActionPolicy:
    """Default-deny ``ui_action`` policy bound to one reviewed :class:`TestSpec`."""

    def __init__(self, *, spec: TestSpec, allowed_action_kinds: Iterable[ActionKind] = (),
                 allowed_origins: Iterable[str] = (), local_sink_origins: Iterable[str] = (),
                 start_origin: Optional[str] = None, principal: str = "runner:qa",
                 clock: Optional[Clock] = None, provider: str = POLICY_PROVIDER) -> None:
        if type(spec) is not TestSpec:
            raise SpecInvalid("DefaultActionPolicy: spec must be a TestSpec")
        kinds = tuple(allowed_action_kinds)
        if not all(isinstance(k, ActionKind) for k in kinds):
            raise SpecInvalid("DefaultActionPolicy: allowed_action_kinds must be ActionKind members")
        self._spec_id = spec.id
        self._app = spec.target.app_id
        self._kinds = frozenset(kinds)
        self._allowed_effects = frozenset(spec.allowed_side_effects)
        self._origins = _origins("allowed_origins", allowed_origins)
        self._sinks = _origins("local_sink_origins", local_sink_origins)
        start = None
        if start_origin is not None:
            (start,) = _origins("start_origin", [start_origin])
        self._steps: Dict[str, Tuple[Step, Optional[str]]] = {}
        current = start
        for step in spec.steps:
            if step.target.kind is TargetKind.UI and step.action.kind is ActionKind.NAVIGATE:
                current = origin_of(step.action.value)       # an unusable URL leaves the origin unknown from here on
            self._steps[step.step_id] = (step, current)
        self._principal = principal
        self._clock = clock
        self._provider = provider

    # -- the pure rules --------------------------------------------------------------------------------------
    def evaluate(self, step: Step) -> Verdict:
        """First failing rule wins. Pure: depends on the spec, the operator configuration and ``step`` only."""
        known = self._steps.get(getattr(step, "step_id", None))
        if known is None or known[0] != step:
            return _deny("step-not-in-spec", "the action is not a step of the reviewed spec")
        origin = known[1]
        if step.target.kind is not TargetKind.UI:
            return _deny("target-kind-unsupported", "only UI targets are authorised by this policy")
        kind = step.action.kind
        if not isinstance(kind, ActionKind):
            return _deny("unknown-action-kind", "the action kind is not known to this policy")
        if kind not in self._kinds:
            return _deny("action-kind-not-allowed", "the action kind is not on the allow-list")
        effect = step.side_effect
        need = SIDE_EFFECT_RULES.get(effect) if isinstance(effect, SideEffect) else None
        if need is None:
            return _deny("unknown-side-effect-class", "the side-effect class is not known to this policy")
        if not isinstance(step.idempotency, Idempotency) or step.idempotency is Idempotency.UNKNOWN:
            return _deny("unknown-idempotency-class", "an action with unknown idempotency is not authorised")
        if need != NEED_NOTHING and effect not in self._allowed_effects:
            return _deny("side-effect-not-allowed", "the side-effect class is not listed in allowed_side_effects")
        if origin is None:
            return _deny("origin-unknown", "the target origin cannot be established from the reviewed spec")
        if origin not in self._origins:
            return _deny("origin-not-allowed", "the target origin is not on the allow-list")
        if need == NEED_LISTED_AND_LOCAL_SINK and origin not in self._sinks:
            return _deny("not-a-local-sink", "this side-effect class is only authorised against a local sink")
        return Verdict(True, "allow-listed", "action kind, side-effect class and origin are explicitly allowed")

    # -- the port --------------------------------------------------------------------------------------------
    def decide(self, run_id: str, step: Step, attempt: int) -> PolicyDecisionRecord:
        try:
            verdict = self.evaluate(step)
        except Exception:                                    # fail closed: an internal error is a denial
            verdict = _deny("policy-internal-error", "the policy failed while evaluating the action")
        return make_decision(run_id=run_id, step=step, attempt=attempt, allow=verdict.allow, rule_id=verdict.rule_id,
                             reason=verdict.reason, principal=self._principal, app=self._app, provider=self._provider,
                             clock=self._clock)


# ---------------------------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------------------------

def _bounded_id(prefix: str, *parts: Any) -> str:
    text = "-".join([prefix, *[str(p) for p in parts]])
    if len(text) <= 128:
        return text
    return prefix + "-" + canon.digest_hex({"parts": [str(p) for p in parts]})[:40]


def make_decision(*, run_id: str, step: Step, attempt: int, allow: bool, rule_id: str, reason: str, principal: str,
                  app: str, provider: str, clock: Optional[Clock] = None) -> PolicyDecisionRecord:
    """One ``PolicyDecisionRecord``. ``decided_at`` and ``reason`` are custody fields: they are not hashed, so the
    decision hash does not depend on the clock or on wording."""
    return PolicyDecisionRecord(
        policy_decision_id=_bounded_id("pd", step.step_id, attempt),
        run_id=run_id, step_id=step.step_id, principal=principal, app=app,
        decision=PolicyDecisionKind.ALLOW if allow else PolicyDecisionKind.DENY,
        provider=provider, action_kind=step.action.kind, side_effect=step.side_effect,
        phase=PHASE, rule_id=rule_id, scope=(f"ui:{step.action.kind.value}",),
        decided_at=clock.now_utc() if clock is not None else None, reason=reason[:4096])


def failsafe_decision(*, run_id: str, step: Step, attempt: int, rule_id: str, reason: str, principal: str, app: str,
                      clock: Optional[Clock] = None) -> PolicyDecisionRecord:
    """The ``DENY`` the runner records when the policy raised, timed out, or returned something unusable."""
    return make_decision(run_id=run_id, step=step, attempt=attempt, allow=False, rule_id=rule_id, reason=reason,
                         principal=principal, app=app, provider=FAILSAFE_PROVIDER, clock=clock)


def decision_ref(record: PolicyDecisionRecord) -> str:
    """The ``rcv1:<hex>`` spelling of the record's ``canon-v1`` hash: the value of ``policy_decision_ref``."""
    return canon.to_contract_ref(record.canon_hash())


def binding_problem(record: Any, run_id: str, step: Step) -> Optional[str]:
    """Why ``record`` is not a usable decision for this action, or ``None``. A record that decides some other run,
    step, action kind or side-effect class, or another phase, authorises nothing here."""
    if type(record) is not PolicyDecisionRecord:
        return "not a PolicyDecisionRecord"
    if record.phase != PHASE:
        return "wrong phase"
    if record.run_id != run_id or record.step_id != step.step_id:
        return "decides another run or step"
    if record.action_kind != step.action.kind or record.side_effect != step.side_effect:
        return "decides another action kind or side-effect class"
    return None


def is_allowed(record: Any) -> bool:
    """Only an ``ALLOW`` record authorises an action. ``DENY`` and ``REQUIRE_APPROVAL`` do not."""
    return type(record) is PolicyDecisionRecord and record.decision is PolicyDecisionKind.ALLOW


__all__ = ["PHASE", "POLICY_PROVIDER", "FAILSAFE_PROVIDER", "SIDE_EFFECT_RULES", "origin_of", "Verdict",
           "DefaultActionPolicy", "make_decision", "failsafe_decision", "decision_ref", "binding_problem",
           "is_allowed"]
