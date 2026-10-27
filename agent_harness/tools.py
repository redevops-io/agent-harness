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
from typing import Any, Callable, Optional, Protocol, runtime_checkable


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    side_effecting: bool = False      # mutates the outside world (block_ip, file_ticket, write_file)
    approval_required: bool = False   # gate even read tools if they touch sensitive data

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


# -- data-access authorization seam ------------------------------------------------------------------------
# An optional authorizer runs on every tool call BEFORE the tool executes: given the caller's principal
# (opaque to this package), the ToolSpec and the args, it returns None to allow or a short deny reason.
# ``set_default_authorizer`` installs one process-wide for registries that were not given their own.
Authorizer = Callable[[object, ToolSpec, dict], Optional[str]]

_default_authorizer: Optional[Authorizer] = None


def set_default_authorizer(fn: Optional[Authorizer]) -> None:
    """Install (or clear, with None) a process-wide authorizer used by every ToolRegistry without its own."""
    global _default_authorizer
    _default_authorizer = fn


_principal_ctx = _contextvars.ContextVar("agent_harness_current_principal", default=None)


def current_principal():
    """The principal for the in-flight tool call (set by ``ToolRegistry.run``), or None. Lets a tool scope its
    behaviour to the caller without changing the ToolPlugin interface."""
    return _principal_ctx.get()


class ToolRegistry:
    """Register tools; run them through the approval gate; describe them to a model."""

    def __init__(self, policy: Optional[ApprovalPolicy] = None, authorizer: Optional[Authorizer] = None):
        self._tools: dict = {}
        self.policy = policy or ApprovalPolicy()
        self.authorizer = authorizer      # data-access gate; falls back to the process default
        self.audit: list = []             # append-only record of every gated decision

    def register(self, tool: ToolPlugin) -> ToolPlugin:
        self._tools[tool.spec().name] = tool
        return tool

    def get(self, name: str) -> ToolPlugin:
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
            tool = self.get(name)
        except ToolError as e:
            return ToolResult(ok=False, error=str(e), text=f"[error] {e}")
        spec = tool.spec()
        args = args or {}
        token = _principal_ctx.set(principal)
        try:
            authorizer = self.authorizer or _default_authorizer
            if authorizer is not None:
                deny = authorizer(principal, spec, args)
                if deny:
                    self.audit.append({"tool": name, "args": args, "allowed": False, "reason": deny})
                    return ToolResult(ok=False, error=f"denied: {deny}", text=f"[blocked] {name}: {deny}")
            ok, reason = self.policy.decide(spec, args)
            if spec.side_effecting or spec.approval_required:
                self.audit.append({"tool": name, "args": args, "allowed": ok, "reason": reason})
            if not ok:
                return ToolResult(ok=False, error=f"denied: {reason}", text=f"[blocked] {name}: {reason}")
            return tool.run(args)
        except Exception as e:  # a tool crash is a failed result, not a runtime crash
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}", text=f"[error] {name}: {e}")
        finally:
            _principal_ctx.reset(token)


def function_tool(name: str, fn: Callable[[dict], ToolResult], description: str = "",
                  parameters: Optional[dict] = None, side_effecting: bool = False,
                  approval_required: bool = False) -> ToolPlugin:
    """Adapt a plain ``fn(args) -> ToolResult`` into a ToolPlugin."""
    _spec = ToolSpec(name=name, description=description,
                     parameters=parameters or {"type": "object", "properties": {}},
                     side_effecting=side_effecting, approval_required=approval_required)

    class _FnTool:
        def spec(self) -> ToolSpec:
            return _spec

        def run(self, args: dict) -> ToolResult:
            return fn(args)

    return _FnTool()
