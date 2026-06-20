"""Minimal tool-calling agent loop."""
import os
from typing import Any, Callable, Dict, List, Optional
import agent_harness.approval as approval
import agent_harness.guardrails as guardrails
import agent_harness.sandbox as sandbox

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

def _resolve_in_sandbox(path: str) -> str:
    """Resolve ``path`` and ensure it stays inside the sandbox root.

    Uses realpath so symlinks and ``..`` segments cannot escape. Raises
    ValueError on any attempt to read/write outside the root.
    """
    root = _sandbox_root()
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"path escapes sandbox root: {path}")
    return target

@tool("read_file")
def read_file(path: str) -> str:
    resolved = _resolve_in_sandbox(path)
    with open(resolved, "r") as f:
        return f.read()

@tool("write_file")
def write_file(path: str, content: str) -> str:
    resolved = _resolve_in_sandbox(path)
    with open(resolved, "w") as f:
        f.write(content)
    return "ok"

@tool("run_bash")
def run_bash(cmd: str) -> Dict[str, Any]:
    pol = approval.ApprovalPolicy()
    dec = pol.decide({"type": "shell", "command": cmd})
    if dec != "allow":
        return {"error": "denied"}
    return sandbox.run({"type": "shell", "command": cmd})

@tool("finish")
def finish(result: str = "done") -> str:
    return result

class Agent:
    def __init__(self, llm: Optional[Callable] = None, policy: Optional[approval.ApprovalPolicy] = None, callbacks: Optional[Dict[str, Callable]] = None):
        self.llm = llm
        self.policy = policy or approval.ApprovalPolicy()
        self.callbacks = callbacks or {}
        self.tools = TOOLS
        self.loop_guard = guardrails.LoopGuard()

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