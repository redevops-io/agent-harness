"""Minimal tool-calling agent loop."""
import os
from typing import Any, Callable, Dict, List, Optional
import agent_harness.command_policy as command_policy
import agent_harness.guardrails as guardrails
import agent_harness.sandbox as sandbox
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool

# Built-in tools
TOOLS = {}

def tool(name: str):
    def deco(fn):
        TOOLS[name] = fn
        return fn
    return deco

def _sandbox_root() -> str:
    """Directory that file tools are confined to (default: cwd)."""
    return os.path.realpath(os.environ.get("AGENT_HARNESS_ROOT", os.getcwd()))

def _resolve_in_sandbox(path: str, root: Optional[str] = None) -> str:
    """Resolve ``path`` and ensure it stays inside the sandbox root.

    Uses realpath so symlinks and ``..`` segments cannot escape. Raises
    ValueError on any attempt to read/write outside the root.
    """
    root = os.path.realpath(root) if root else _sandbox_root()
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"path escapes sandbox root: {path}")
    return target

# Policy used when the module-level tools are called outside an Agent.
_DEFAULT_POLICY = command_policy.CommandPolicy()

def _read_file(path: str, root: Optional[str] = None) -> str:
    resolved = _resolve_in_sandbox(path, root)
    with open(resolved, "r") as f:
        return f.read()

def _write_file(path: str, content: str, root: Optional[str] = None,
                policy: Optional[command_policy.CommandPolicy] = None) -> str:
    resolved = _resolve_in_sandbox(path, root)
    with open(resolved, "w") as f:
        f.write(content)
    (policy or _DEFAULT_POLICY).note_write(resolved)
    return "ok"

def _run_bash(cmd: str, root: Optional[str] = None,
              policy: Optional[command_policy.CommandPolicy] = None) -> Dict[str, Any]:
    """Run ``cmd`` under ``policy`` with cwd set to the workspace root.

    Not full isolation: see the sandbox module docstring.
    """
    pol = policy or _DEFAULT_POLICY
    root = pol._root() if pol.root else (os.path.realpath(root) if root else pol._root())
    if pol.decide({"type": "shell", "command": cmd}) != "allow":
        return {"error": "denied"}
    argv = pol.resolve_argv(cmd)
    if argv is None:
        return {"error": "denied"}
    return sandbox.run({"type": "shell", "command": cmd, "argv": argv}, cwd=root)

# Model-facing tools expose only their own parameters; root and policy are bound
# by the Agent (or default to the environment) and cannot be supplied by callers.
@tool("read_file")
def read_file(path: str) -> str:
    return _read_file(path)

@tool("write_file")
def write_file(path: str, content: str) -> str:
    return _write_file(path, content)

@tool("run_bash")
def run_bash(cmd: str) -> Dict[str, Any]:
    return _run_bash(cmd)

@tool("finish")
def finish(result: str = "done") -> str:
    return result

_BUILTIN_SPECS = (
    ("read_file", "Read a text file inside the workspace root.",
     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, False),
    ("write_file", "Write a text file inside the workspace root.",
     {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
      "required": ["path", "content"]}, True),
    ("run_bash", "Run one allowlisted command in the workspace root. Processes the command starts are stopped when it returns.",
     {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}, True),
    ("finish", "Finish the task with a result.",
     {"type": "object", "properties": {"result": {"type": "string"}}}, False),
)


def _adapt(fn: Callable) -> Callable[[Dict[str, Any]], ToolResult]:
    """Wrap a keyword-argument tool function as ``fn(args) -> ToolResult``."""
    def run(args: Dict[str, Any]) -> ToolResult:
        try:
            out = fn(**args)
        except TypeError as e:
            return ToolResult(ok=False, error=f"invalid arguments: {e}", data={"error": f"invalid arguments: {e}"})
        if isinstance(out, dict) and "error" in out:
            return ToolResult(ok=False, error=str(out["error"]), data=out)
        return ToolResult(ok=True, data=out, text=str(out))
    return run


class Agent:
    def __init__(self, llm: Optional[Callable] = None, policy: Optional[command_policy.CommandPolicy] = None, callbacks: Optional[Dict[str, Callable]] = None, root: Optional[str] = None,
                 approval: Optional[ApprovalPolicy] = None):
        # ``policy`` vets the content of shell commands and edits; ``approval`` is the tool-level gate that
        # decides whether a side-effecting tool may run at all. By default the Agent's own side-effecting
        # built-ins are allowed at the tool level and left to ``policy``; pass ``approval`` to require more.
        self.llm = llm
        self.policy = policy or command_policy.CommandPolicy()
        self.root = os.path.realpath(root) if root else _sandbox_root()
        if self.policy.root is None:
            self.policy.root = self.root
        self.callbacks = callbacks or {}
        self.tools = dict(TOOLS)
        root, policy = self.root, self.policy

        def read_file(path: str) -> str:
            return _read_file(path, root)

        def write_file(path: str, content: str) -> str:
            return _write_file(path, content, root, policy)

        def run_bash(cmd: str) -> Dict[str, Any]:
            return _run_bash(cmd, root, policy)

        self.tools["read_file"] = read_file
        self.tools["write_file"] = write_file
        self.tools["run_bash"] = run_bash
        self.loop_guard = guardrails.LoopGuard()

        self.registry = ToolRegistry(approval or ApprovalPolicy("allowlist", allow=["write_file", "run_bash"]))
        for name, desc, params, effect in _BUILTIN_SPECS:
            self.registry.register(function_tool(
                name, _adapt(self.tools[name]), description=desc, parameters=params, side_effecting=effect))

    def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """Dispatch a model-supplied tool call; unknown tools or arguments are rejected."""
        if name in self.registry.list():
            res = self.registry.run(name, args or {})
            return res.data if res.ok else (res.data if isinstance(res.data, dict) else {"error": res.error})
        # Tools added through the module-level ``@tool`` decorator are not described to the registry and keep
        # the legacy direct dispatch (no tool-level gate).
        fn = self.tools.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"invalid arguments: {e}"}

    def run(self, task: str) -> str:
        self.loop_guard.reset()
        if not guardrails.validate_input(task):
            return guardrails.refuse_unsafe("input")
        if not self.loop_guard.check():
            return "loop limit"
        # minimal loop: just finish
        msg = {"role": "user", "content": task}
        if "on_message" in self.callbacks:
            self.callbacks["on_message"](msg)
        res = self.tools["finish"](task)
        if "on_tool_call" in self.callbacks:
            self.callbacks["on_tool_call"]("finish", {"result": res})
        return res

def create_agent(**kwargs):
    return Agent(**kwargs)