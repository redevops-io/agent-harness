"""Minimal tool-calling agent loop."""
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

@tool("read_file")
def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

@tool("write_file")
def write_file(path: str, content: str) -> str:
    with open(path, "w") as f:
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

    def run(self, task: str) -> str:
        if not guardrails.validate_input(task):
            return guardrails.refuse_unsafe("input")
        if not guardrails.check_loop():
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