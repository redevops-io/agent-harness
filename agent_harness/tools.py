"""Tool contracts, registry, and the approval gate.

A ``ToolPlugin`` exposes an OpenAI-style spec and a ``run(args) -> ToolResult``. The ``ToolRegistry`` is what
an agent (or a tenant) calls; it enforces that side-effecting tools pass an ``ApprovalPolicy`` before they
execute. The default policy DENIES side-effects unless explicitly allowed; there is no open grant by default.

This is the contract context-runtime's ``context_runtime.tools.base`` already uses, adopted here as the
single definition. ``ToolResult.hits`` is typed ``list[Any]`` because retrieval hit types belong to the
retrieval layer, not to this package. Shell/edit command vetting is a separate, lower layer:
``agent_harness.command_policy``.
"""
from __future__ import annotations

import contextvars as _contextvars
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable
from enum import Enum
import warnings


class ToolError(RuntimeError):
    pass


class ToolClassificationError(ValueError):
    """A tool's declared risk class contradicts what the tool is, or is not a risk class at all.

    Raised by ``ToolRegistry.register`` — that is, at import/registration time, when a stack can still fail to
    start — and never during a tool call. A registry that accepts a self-contradictory classification has
    volunteered to deny a capability it will later ask for, so the mistake is not one the call path can fix.
    """


class RegistryConfigurationError(ToolError):
    """``ToolRegistry.validate(strict=True)`` found a problem a process should not start with — no authorizer
    installed, or a tool with no risk class where classification is required. Raised at boot, deliberately:
    the alternative is discovering it as a denial in front of a user, where it looks like the policy plane
    misbehaving rather than the service being built wrong."""


#: The grant actions a tool may require: the platform's action vocabulary (``agentic_os.permissions.ACTIONS``),
#: repeated here for the same reason :class:`PolicyReason` is — this package declares no dependency, and the
#: spelling is the contract. There is deliberately no weaker grade than ``read`` and no ``execute`` action:
#: a tool that runs a process or directs one is classified by what that can accomplish (``write``), not by
#: how innocent the command looks. A tool nobody has classified has an empty action and is not permitted.
ACTIONS = ("read", "write", "delete")

#: Strength order, used only to tell a re-registration that raises a risk class from one that lowers it.
#: ``delete`` sits above ``write`` because that is the difference between changing what exists and removing
#: the thing a grant was written about.
_ACTION_RANK = {action: rank for rank, action in enumerate(ACTIONS)}

#: The ``authz/v1`` resource kind a tool is authorized against. The decision envelope asks for a resource and
#: an action; a tool call is exactly that, and naming the kind here means every consumer in the stack asks for
#: the same capability instead of each inventing a resource string at its own call site.
TOOL_RESOURCE_KIND = "tool"


@dataclass(frozen=True)
class ValidationReport:
    """What ``ToolRegistry.validate()`` found. ``problems`` are why a service must not start; ``warnings`` are
    things a reviewer should see but which do not stop the process."""

    problems: tuple = ()
    warnings: tuple = ()
    authorizer: str = "none"            # "registry", "process_default", "opt_out" or "none"
    classified: tuple = ()              # (name, action) for every tool that carries a risk class

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {"ok": self.ok, "authorizer": self.authorizer, "problems": list(self.problems),
                "warnings": list(self.warnings), "classified": dict(self.classified)}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    side_effecting: bool = False      # mutates the outside world (block_ip, file_ticket, write_file)
    approval_required: bool = False   # gate even read tools if they touch sensitive data
    action: str = ""                  # required grant action, one of ACTIONS; "" = unclassified (not permitted)

    def openai(self) -> dict:
        """OpenAI-style function spec for tool-calling models."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class ToolResult:
    ok: bool = True
    data: Any = None                              # raw structured result
    hits: list = field(default_factory=list)      # if the tool is a retrieval source (type owned by the caller)
    text: str = ""                                # human/agent-readable summary
    error: Optional[str] = None
    attachments: list = field(default_factory=list)   # model inputs the tool produced (images); the loop delivers them in a user message after the tool results


@runtime_checkable
class ToolPlugin(Protocol):
    def spec(self) -> ToolSpec: ...
    def run(self, args: dict) -> ToolResult: ...


class ToolHandle:
    """What :meth:`ToolRegistry.get` returns: a read-only view of a registered tool, offering only
    :meth:`spec`. It has no ``run`` method at all — on purpose.

    ``ToolRegistry.get(name).run(args)`` used to hand back the raw :class:`ToolPlugin`, which a caller
    could then call directly, skipping :meth:`ToolRegistry.authorize` and the approval gate entirely. That
    is an authorization bypass, not a shortcut: the one path that is actually gated is
    ``ToolRegistry.run(name, args, principal=...)``. ``get()`` exists for introspection (reading a tool's
    spec to build a capability request, describe it to a model, and so on); it has never needed to expose
    execution, so this removes a capability nothing legitimate used rather than restricting one that did.

    The underlying plugin is kept on a single-underscore attribute, the same "not public API" convention
    this module already uses for its own internal methods — reaching for ``handle._tool.run(args)`` is a
    deliberate act of digging past that convention, not an accident a caller falls into by calling the
    method a ``ToolHandle`` actually advertises, which is the shape the real bypass took.
    """

    __slots__ = ("_tool",)

    def __init__(self, tool: ToolPlugin) -> None:
        self._tool = tool

    def spec(self) -> ToolSpec:
        return self._tool.spec()

    def __repr__(self) -> str:
        return f"ToolHandle({self._tool.spec().name!r})"


# An approver: given {"tool": name, "args": args}, return True to allow a side-effecting/gated tool.
Approver = Callable[[dict], bool]


class ApprovalPolicy:
    """Decide whether a side-effecting/gated tool call may run.

    Modes:
      * ``deny_side_effects`` (default): read tools allowed; side-effects need an explicit approver
        returning True.
      * ``allowlist``: only tools whose name is in ``allow`` may side-effect.
      * ``bypass``: everything allowed (use only in trusted automation).
    Any other mode value denies every gated call.
    """

    def __init__(self, mode: str = "deny_side_effects", allow: Optional[list] = None,
                 approver: Optional[Approver] = None):
        self.mode = mode
        self.allow = set(allow or [])
        self.approver = approver

    def decide(self, spec: ToolSpec, args: dict) -> tuple:
        gated = spec.side_effecting or spec.approval_required
        if not gated:
            return True, "read-only"
        if self.mode == "bypass":
            return True, "bypass"
        if self.mode == "allowlist":
            return (spec.name in self.allow, "allowlisted" if spec.name in self.allow else "not in allowlist")
        if self.mode == "deny_side_effects":
            if self.approver is not None and self.approver({"tool": spec.name, "args": args}):
                return True, "approved"
            return False, "side-effecting tool requires approval"
        return False, f"unknown approval mode {self.mode!r}"


# -- authorization: the decision, the vocabulary, the seam ---------------------------------------------------
#
# An authorizer runs on every tool call BEFORE the tool executes and answers one question: may this principal
# run this tool with these args? The answer has to be a *decision*, never the absence of one. The seam that
# this replaces returned ``Optional[str]`` where ``None`` meant "allow", so an authorizer that returned early,
# missed a branch, or was wired to return ``None`` on every path let the tool run: the deny-by-default
# registry was fail-open for every tool, read-only tools included. ``Authorization`` makes the two cases
# different types, and a registry with no authorizer installed is a denial, not a pass-through.


class Decision(str, Enum):
    """The only two answers an authorizer may give. There is no third value meaning "no opinion"."""

    ALLOW = "allow"
    DENY = "deny"


class PolicyReason:
    """The reasons this registry may attach to a decision.

    These are the members of the closed `authz/v1` vocabulary (`runtime_contracts.authz.decision.PolicyReason`)
    that a tool call can honestly produce; the values are spelled out here, not imported, because
    ``agent_harness`` declares no dependency and must stay importable without ``runtime_contracts`` (0.3.14,
    the version installed in the stack today, ships no ``authz`` subpackage at all). The subset and its exact
    spellings are pinned against the real contract by ``tests/test_authz_contract_align.py``, which fails —
    rather than skipping — whenever that module is importable. Inventing a reason string here is how a closed
    vocabulary becomes an open one, so a new member needs a contract change first.
    """

    EXPLICIT_GRANT = "explicit_grant"                       # a grant/decision covers this call
    WILDCARD_GRANT = "wildcard_grant"                       # a catch-all granted it (an opt-out, say so)
    MISSING_EXPLICIT_GRANT = "missing_explicit_grant"       # nothing granted it (deny-by-default)
    NO_RESOURCE_PLANE = "no_resource_plane"                 # the authority could not be consulted
    MALFORMED_REQUEST = "malformed_request"                 # the answer we got is not a decision
    UNCLASSIFIED = "unclassified"                           # the emitter did not classify its own decision


@dataclass(frozen=True)
class Authorization:
    """One authorization decision about one tool call.

    ``reason`` is a :class:`PolicyReason` value and ``detail`` is the human-readable why; together they are
    what the audit trail carries. ``grant_ids`` is the attribution: an allow with no grant ids and no
    ``break_glass`` is an unattributable allow, which is the thing this type exists to make visible.
    ``plane`` names who decided, ``legacy`` marks a decision coerced from the pre-``authz/v1`` return shapes
    (bool/str) so a reviewer can grep for call sites still on the old contract.
    """

    decision: Decision
    reason: str = PolicyReason.MISSING_EXPLICIT_GRANT
    detail: str = ""
    grant_ids: tuple = ()
    plane: str = "tool_registry"
    break_glass: bool = False
    legacy: bool = False

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def message(self) -> str:
        """The single string a denial is reported and audited under: the code, then the detail."""
        return f"{self.reason}: {self.detail}" if self.detail else self.reason

    @classmethod
    def allow(cls, *, reason: str = PolicyReason.EXPLICIT_GRANT, detail: str = "",
              grant_ids: tuple = (), plane: str = "tool_registry", break_glass: bool = False,
              legacy: bool = False) -> "Authorization":
        return cls(Decision.ALLOW, reason=reason, detail=detail, grant_ids=tuple(grant_ids),
                   plane=plane, break_glass=break_glass, legacy=legacy)

    @classmethod
    def deny(cls, detail: str = "", *, reason: str = PolicyReason.MISSING_EXPLICIT_GRANT,
             grant_ids: tuple = (), plane: str = "tool_registry", legacy: bool = False) -> "Authorization":
        return cls(Decision.DENY, reason=reason, detail=detail, grant_ids=tuple(grant_ids),
                   plane=plane, legacy=legacy)

    def as_dict(self) -> dict:
        """The audit-line shape for this decision. Only the authorization layer writes keys beyond the four
        the approval gate has always written (``tool``/``args``/``allowed``/``reason``) — the approval gate's
        own records keep their historical shape, so a consumer that reads them sees what it has always seen."""
        return {"allowed": self.allowed, "policy_reason": self.reason, "detail": self.detail,
                "grant_ids": sorted(set(self.grant_ids)), "plane": self.plane,
                "break_glass": self.break_glass, "legacy": self.legacy}


def as_authorization(raw: Any) -> Authorization:
    """Read whatever an authorizer returned as a decision. Strict, and every ambiguous shape is a denial.

    The old contract was ``None`` = allow, non-empty ``str`` = deny. Both meanings that were load-bearing
    keep working (a ``str`` still denies, ``True`` still means the authorizer said yes) and are marked
    ``legacy=True`` so they are visible in the audit trail; what no longer works is the silence —
    ``None``, an empty string, and any unrecognisable object all deny.
    """
    if isinstance(raw, Authorization):
        return raw
    if raw is None:                                    # the case that made the old seam fail open
        return Authorization.deny("the authorizer returned no decision", legacy=True)
    if raw is True:
        return Authorization.allow(reason=PolicyReason.EXPLICIT_GRANT, detail="authorizer returned True",
                                    legacy=True)
    if raw is False:
        return Authorization.deny("denied by authorizer", legacy=True)
    if isinstance(raw, str) and raw:
        return Authorization.deny(raw, legacy=True)
    return Authorization(Decision.DENY, reason=PolicyReason.MALFORMED_REQUEST,
                         detail=f"authorizer returned {type(raw).__name__}, not a decision", legacy=True)


# What an authorizer may return: a typed ``Authorization``, or (deprecated, coerced and marked) the old
# ``Optional[str]`` / bool shapes. See :func:`as_authorization`.
AuthorizerReturn = Union[Authorization, Optional[str], bool]
Authorizer = Callable[[object, ToolSpec, dict], AuthorizerReturn]


def engine_authorizer_of(fn: Optional[Authorizer]) -> Optional[Authorizer]:
    """The callable a registry should actually invoke for an authorizer that came from somewhere else.

    A policy plane outside this repository answers in its own contract, where a bare ``None`` means *allowed* — the
    exact shape this repository reads as "nobody decided" and denies (see :func:`as_authorization`). A plane that
    hands its callable over for use as a process default or a registry's authorizer therefore attaches
    ``engine_authorizer``: the same decisions expressed in this repository's contract. When that attribute is
    present it is the only callable whose "yes" a registry can act on, so it is the one that gets called; an
    authorizer that speaks this contract already carries nothing and is returned untouched.
    """
    twin = getattr(fn, "engine_authorizer", None)
    return twin if callable(twin) else fn


_default_authorizer: Optional[Authorizer] = None


def set_default_authorizer(fn: Optional[Authorizer]) -> None:
    """Install (or clear, with None) a process-wide authorizer used by every ToolRegistry without its own.

    Clearing it with ``None`` is not a return to "everything allowed": a registry that has no authorizer and
    has not opted out with ``authorizer_required=False`` denies every tool call.
    """
    global _default_authorizer
    _default_authorizer = fn


def default_authorizer_installed() -> bool:
    """Whether a process-wide authorizer is installed. Lets a service assert its policy is wired at startup."""
    return _default_authorizer is not None


_local_trust_warned = False


def local_trusted_authorizer(*, plane: str = "local_trusted") -> Authorizer:
    """An authorizer that allows every call, for trusted local/direct use: the built-in ``Agent``, the bench
    harness, examples and tests. It is not a hole — the decision is attributed (``plane="local_trusted"``,
    a wildcard reason) so it is greppable in the audit trail, and the first use warns once per process.

    Reach for this when there is no policy plane to consult. It is the explicit form of what the old default
    did silently; a registry that wants the old silence has to ask for it with ``authorizer_required=False``.
    """
    global _local_trust_warned
    if not _local_trust_warned:      # once per process: the fact is process-wide, the noise is not useful
        _local_trust_warned = True
        warnings.warn("agent_harness: this process installs a local_trusted authorizer, which allows every "
                      "tool call without asking a policy plane. Correct for local/direct execution; wrong "
                      "anywhere a tenant, a principal or a budget is in scope.", stacklevel=2)

    def _authorizer(principal, spec, args) -> Authorization:
        return Authorization.allow(reason=PolicyReason.WILDCARD_GRANT,
                                   detail="local trusted execution (no policy plane consulted)", plane=plane)
    return _authorizer

_principal_ctx = _contextvars.ContextVar("agent_harness_current_principal", default=None)


def current_principal():
    """The principal for the in-flight tool call (set by ``ToolRegistry.run``), or None. Lets a tool scope its
    behaviour to the caller without changing the ToolPlugin interface."""
    return _principal_ctx.get()


class ToolRegistry:
    """Register tools; run them through the authorization seam and the approval gate; describe them to a model.

    Two gates run in order, and both have to be satisfied: the ``authorizer`` answers "may this principal run
    this tool" (data/tenant/plan authorization), and ``policy`` answers "is this kind of call approved here"
    (the side-effect gate). Neither substitutes for the other, so a registry that allows everything at one
    layer still enforces the other.

    ``authorizer_required`` (default ``True``) is what makes the absence of a policy plane a denial instead
    of a pass-through. Setting it to ``False`` restores the old behaviour and is a deliberate, loud opt-out:
    it warns at construction and every decision it produces is audited as a wildcard/legacy grant.

    ``classify_required`` (default ``False``) is read by :meth:`validate`: when true, a tool that arrives
    without a risk class is a startup failure rather than a runtime surprise. Classification is *reported* by
    the registry and *enforced* by the plane — the registry states which capability a call needs, the plane
    decides whether the principal holds it. Two places making that decision is how a stack gets a second,
    quieter authorization layer, so the registry never denies on classification grounds itself.
    """

    def __init__(self, policy: Optional[ApprovalPolicy] = None, authorizer: Optional[Authorizer] = None,
                 *, authorizer_required: bool = True, classify_required: bool = False):
        self._tools: dict = {}
        self.policy = policy or ApprovalPolicy()
        self.authorizer = authorizer      # authorization gate; falls back to the process default
        self.authorizer_required = authorizer_required
        self.classify_required = classify_required
        self.audit: list = []             # append-only record of every decision
        if not authorizer_required:
            warnings.warn("agent_harness: ToolRegistry(authorizer_required=False) runs tools with no "
                          "authorization decision. Every call is audited as a wildcard grant; use "
                          "local_trusted_authorizer() instead when local/direct execution is the intent.",
                          stacklevel=2)

    # -- what this registry requires -----------------------------------------------------------------------

    def classification(self) -> dict:
        """``{tool_name: action}`` for every registered tool, ``""`` where it carries no risk class. The point
        of reading it from the registry rather than keeping a separate table next to it is that there is then
        one list of tools and one classification of them; a dict maintained beside the registry drifts from the
        registry the first time a tool is added to one and not the other."""
        return {name: tool.spec().action for name, tool in sorted(self._tools.items())}

    def unclassified(self) -> list:
        return [name for name, action in self.classification().items() if not action]

    def required_capability(self, name: str) -> dict:
        """The ``authz/v1`` capability triple a call to ``name`` needs, in the envelope's own field names.

        This is the seam the plane reads: instead of a consumer assembling
        ``{"resource_kind": "tool", "resource_name": ..., "requested_action": ...}`` at its own call site —
        where it can name any resource and pick any action — it asks the registry that owns the tool. An
        unclassified tool yields an empty action, which no grant can match: the unknown stays unknown instead
        of becoming the developer's choice of default.
        """
        action = self.get(name).spec().action
        return {"resource_kind": TOOL_RESOURCE_KIND, "resource_name": name, "requested_action": action}

    def validate(self, *, require_classification: Optional[bool] = None,
                 strict: bool = False) -> ValidationReport:
        """Everything a boot-time check needs before this registry serves traffic.

        Called by ``Agent.__init__`` and meant to be called by every service that builds its own registry.
        Fatal problems are the ones where the registry cannot do its job — no authorizer installed, or a tool
        with no risk class when classification is required. Warnings are reachability facts a reviewer needs
        to see (a gated tool the approval policy can never approve) that do not justify killing a process.

        ``strict`` raises :class:`RegistryConfigurationError` on any problem instead of returning the report —
        what a service constructor wants, because the report only helps whoever reads it.
        """
        require_cls = self.classify_required if require_classification is None else require_classification
        problems: list = []
        warnings_: list = []
        if self.authorizer is not None:
            who = "registry"
        elif _default_authorizer is not None:
            who = "process_default"
        elif not self.authorizer_required:
            who = "opt_out"
            warnings_.append("authorizer_required=False: every call is authorized by a wildcard grant, and "
                             "the audit will say so")
        else:
            who = "none"
            problems.append("no authorizer is installed and authorizer_required is true: every tool call will "
                            f"be denied with missing_explicit_grant. Pass authorizer=..., call "
                            f"set_default_authorizer(...), or use local_trusted_authorizer() for local runs.")
        if who == "opt_out" and not self._tools:
            warnings_.append("registry is empty and opted out of authorization")
        unclassified = self.unclassified()
        if unclassified and require_cls:
            problems.append(f"{len(unclassified)} tool(s) have no risk class: {unclassified}. Pass action=... "
                            f"at registration; an unclassified tool cannot be the basis of a capability.")
        elif unclassified:
            warnings_.append(f"{len(unclassified)} tool(s) have no risk class: {unclassified}")
        for name, tool in sorted(self._tools.items()):
            spec = tool.spec()
            if not (spec.side_effecting or spec.approval_required):
                continue
            if self.policy.mode == "bypass":
                continue
            if self.policy.mode == "allowlist":
                if name not in self.policy.allow:
                    warnings_.append(f"{name}: gated (side_effecting={spec.side_effecting} or "
                                     f"approval_required={spec.approval_required}) and not in the approval "
                                     f"allowlist, so it can never run here")
            elif self.policy.mode == "deny_side_effects":
                if self.policy.approver is None:
                    warnings_.append(f"{name}: gated and the approval policy has no approver, so it can "
                                     f"never run here")
            else:
                warnings_.append(f"{name}: gated and approval mode {self.policy.mode!r} is not a mode this "
                                 f"policy knows, so every gated call is denied")
        report = ValidationReport(problems=tuple(problems), warnings=tuple(warnings_), authorizer=who,
                                  classified=tuple((n, a) for n, a in self.classification().items() if a))
        if strict and problems:
            raise RegistryConfigurationError("tool registry is not ready to serve: " + "; ".join(problems))
        return report

    # -- the authorization gate ----------------------------------------------------------------------------

    def authorize(self, spec: ToolSpec, args: dict, principal=None) -> Authorization:
        """Ask the installed authorizer about one call. Never raises; every failure mode is a denial."""
        authorizer = engine_authorizer_of(self.authorizer or _default_authorizer)
        if authorizer is None:
            if self.authorizer_required:
                return Authorization.deny(
                    "no authorizer is installed on this registry. Pass ToolRegistry(authorizer=...), install "
                    "one process-wide with set_default_authorizer(...), or — when there is genuinely no "
                    "policy plane to consult, e.g. local/direct execution — pass "
                    "authorizer=local_trusted_authorizer().",
                    reason=PolicyReason.MISSING_EXPLICIT_GRANT)
            return Authorization.allow(reason=PolicyReason.WILDCARD_GRANT,
                                       detail="authorizer_required=False (no authorization consulted)",
                                       plane="tool_registry", legacy=True)
        try:
            raw = authorizer(principal, spec, args)
        except Exception as e:  # an unreachable authority is a denial, never a pass-through
            return Authorization.deny(f"{type(e).__name__}: {e}", reason=PolicyReason.NO_RESOURCE_PLANE)
        return as_authorization(raw)

    def _record_authorization(self, name: str, args: dict, decision: Authorization) -> None:
        """Audit one authorization outcome. Denials are always recorded; allows are recorded when the call
        is gated or when the decision is not an ordinary named grant, so a wildcard/legacy allow — the two
        shapes that hide a missing policy — is never invisible."""
        if not decision.allowed:
            self.audit.append({"tool": name, "args": args, "allowed": False, "reason": decision.message,
                               **decision.as_dict()})
            return
        if decision.legacy or decision.reason == PolicyReason.WILDCARD_GRANT:
            self.audit.append({"tool": name, "args": args, "allowed": True, "reason": decision.message,
                               **decision.as_dict()})

    def _record_approval(self, name: str, args: dict, ok: bool, reason: str) -> None:
        """The approval gate's record — the four keys it has always written, unchanged in shape."""
        self.audit.append({"tool": name, "args": args, "allowed": ok, "reason": reason})

    def unattributed_allows(self) -> list:
        """Every allow in the audit trail with no grant behind it — the shape an auditor asks about first."""
        return [a for a in self.audit if a.get("allowed")
                and "policy_reason" in a and not a.get("grant_ids") and not a.get("break_glass")]

    def register(self, tool: ToolPlugin) -> ToolPlugin:
        spec = tool.spec()
        self._check_classifiable(spec)
        previous = self._tools.get(spec.name)
        if previous is not None:
            self._check_no_downgrade(previous.spec(), spec)
        self._tools[spec.name] = tool
        return tool

    def _check_classifiable(self, spec: ToolSpec) -> None:
        """Reject, at registration, a risk class that cannot be honored. Two checks, both about a
        contradiction the call path could never resolve:

        * the action is not one of :data:`ACTIONS` (a typo is not a risk class);
        * ``action="read"`` on a tool the spec itself calls side-effecting or approval-gated. A ``read`` grant
          is the weakest thing the plane issues, and a tool that mutates or launches work cannot be authorized
          with it — registering it that way guarantees a denial at the least opportune moment, which is how a
          classification table and the tools it describes drift apart unnoticed. The other direction is legal:
          a ``write`` classification on a read-only tool only ever asks for more than it needs.
        """
        if not spec.action:
            return                                    # unclassified is a startup finding, not a registration error
        if spec.action not in ACTIONS:
            raise ToolClassificationError(
                f"{spec.name}: action {spec.action!r} is not one of {ACTIONS}")
        if spec.action == "read" and (spec.side_effecting or spec.approval_required):
            raise ToolClassificationError(
                f"{spec.name}: classified 'read' but the spec says side_effecting={spec.side_effecting} / "
                f"approval_required={spec.approval_required}. Classify it 'write', or drop the flag if the "
                f"tool really does nothing but return existing state.")

    def _check_no_downgrade(self, old: ToolSpec, new: ToolSpec) -> None:
        """Re-registering a name — a hot reload, a plugin overwrite, a second builder handed the same registry —
        may keep or raise its risk class, never lower it. A class travels with the code that implements the
        tool, so a path that lets the newer registration win silently turns "this tool can delete rows" into
        "this tool reads" at the moment someone swaps the implementation. Raising is the only answer that
        leaves the process in a state worth debugging.
        """
        if not old.action:
            return                       # adding a class to a previously unclassified tool
        if not new.action:
            raise ToolClassificationError(
                f"{new.name}: re-registered without a risk class; it was {old.action!r}. Classifications are "
                f"not removed by re-registration — pass action={old.action!r} or a stronger class.")
        if _ACTION_RANK[new.action] < _ACTION_RANK[old.action]:
            raise ToolClassificationError(
                f"{new.name}: re-registered as {new.action!r}, downgrading the previous classification "
                f"{old.action!r}. A re-registration may raise a tool's risk class, never lower it.")

    def get(self, name: str) -> ToolHandle:
        """A read-only view of a registered tool: :meth:`ToolHandle.spec` only, deliberately no way to
        execute it. To run a tool, call :meth:`run` — the gated path. See :class:`ToolHandle`."""
        return ToolHandle(self._lookup(name))

    def _lookup(self, name: str) -> ToolPlugin:
        """The raw plugin, for this registry's own internal use only. The only caller that may do anything
        with what this returns besides read its ``spec()`` is :meth:`run`, strictly after
        :meth:`authorize` and ``self.policy.decide`` have both already passed — never expose this return
        value to outside code, or the authorization/approval gate it just satisfied means nothing."""
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list:
        return list(self._tools)

    def specs(self) -> list:
        """OpenAI-style tool specs for a tool-calling model."""
        return [t.spec().openai() for t in self._tools.values()]

    def run(self, name: str, args: Optional[dict] = None, principal=None) -> ToolResult:
        try:
            tool = self._lookup(name)
        except ToolError as e:
            return ToolResult(ok=False, error=str(e), text=f"[error] {e}")
        spec = tool.spec()
        args = args or {}
        token = _principal_ctx.set(principal)
        try:
            decision = self.authorize(spec, args, principal)
            self._record_authorization(name, args, decision)
            if not decision.allowed:
                return ToolResult(ok=False, error=f"denied: {decision.message}",
                                  text=f"[blocked] {name}: {decision.message}")
            ok, reason = self.policy.decide(spec, args)
            if spec.side_effecting or spec.approval_required:
                self._record_approval(name, args, ok, reason)
            if not ok:
                return ToolResult(ok=False, error=f"denied: {reason}", text=f"[blocked] {name}: {reason}")
            return tool.run(args)
        except Exception as e:  # a tool crash is a failed result, not a runtime crash
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}", text=f"[error] {name}: {e}")
        finally:
            _principal_ctx.reset(token)


def function_tool(name: str, fn: Callable[[dict], ToolResult], description: str = "",
                  parameters: Optional[dict] = None, side_effecting: bool = False,
                  approval_required: bool = False, action: str = "") -> ToolPlugin:
    """Adapt a plain ``fn(args) -> ToolResult`` into a ToolPlugin.

    ``action`` is the risk class: which capability this call will be checked for. It is set here, by the
    code that registers the tool, and never read from the function's name, its docstring, or a remote
    server's ``readOnlyHint``: the thing asking to be run is not allowed to grade its own risk. Omit it
    only while a service is mid-migration; ``ToolRegistry.validate()`` reports every tool left
    unclassified.
    """
    _spec = ToolSpec(name=name, description=description,
                     parameters=parameters or {"type": "object", "properties": {}},
                     side_effecting=side_effecting, approval_required=approval_required, action=action)

    class _FnTool:
        def spec(self) -> ToolSpec:
            return _spec

        def run(self, args: dict) -> ToolResult:
            return fn(args)

    return _FnTool()

