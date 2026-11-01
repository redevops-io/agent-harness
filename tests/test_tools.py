from tests._native import windows_todo
import pytest
from agent_harness.sandbox import run

@windows_todo("bare command names resolve only against the POSIX trusted directories (/usr/local/bin, /usr/bin, /bin); a Windows resolver for allowlisted executables is not implemented yet")
def test_register_and_dispatch():
    # Simulate tool via @tool-like dispatch using sandbox run
    action = {"type": "shell", "command": "echo hello"}
    result = run(action)
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]

def test_export_openai_spec():
    # Minimal OpenAI-style spec
    spec = {
        "name": "echo",
        "description": "Echo input",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}
    }
    assert spec["name"] == "echo"


# --- tool contract (same cases as context-runtime's tests/test_tools.py and test_tool_authorizer.py) ---------
from agent_harness.tools import (ApprovalPolicy, ToolPlugin, ToolRegistry, ToolResult, ToolSpec, current_principal,
                                 function_tool, set_default_authorizer)


def _read_tool():
    return function_tool("echo_docs", lambda a: ToolResult(ok=True, hits=[{"text": f"doc for {a.get('query')}"}]),
                         description="returns docs")


def _side_effect_tool():
    return function_tool("nuke", lambda a: ToolResult(ok=True, text="boom"), side_effecting=True, approval_required=True)


def test_read_tool_runs_and_satisfies_protocol():
    reg = ToolRegistry()
    assert isinstance(reg.register(_read_tool()), ToolPlugin)
    res = reg.run("echo_docs", {"query": "k8s"})
    assert res.ok and "k8s" in res.hits[0]["text"]


def test_side_effecting_tool_denied_by_default_and_audited():
    reg = ToolRegistry()
    reg.register(_side_effect_tool())
    res = reg.run("nuke", {})
    assert not res.ok and "denied" in res.error
    assert reg.audit[-1]["allowed"] is False


def test_approver_allows():
    reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=lambda a: a["tool"] == "nuke"))
    reg.register(_side_effect_tool())
    assert reg.run("nuke", {}).text == "boom" and reg.audit[-1]["allowed"] is True


def test_allowlist_and_bypass_and_unknown_mode():
    t = _side_effect_tool()
    for policy, expect in ((ApprovalPolicy("allowlist", allow=["nuke"]), True),
                           (ApprovalPolicy("allowlist", allow=["other"]), False),
                           (ApprovalPolicy("bypass"), True),
                           (ApprovalPolicy("typo"), False)):
        reg = ToolRegistry(policy)
        reg.register(t)
        assert reg.run("nuke", {}).ok is expect


def test_openai_specs():
    reg = ToolRegistry()
    reg.register(_read_tool())
    s = reg.specs()[0]
    assert s["type"] == "function" and s["function"]["name"] == "echo_docs"
    assert ToolSpec("x").openai()["function"]["parameters"] == {"type": "object", "properties": {}}


def test_unknown_tool_and_crashing_tool_are_failed_results():
    reg = ToolRegistry()
    assert "unknown tool" in reg.run("missing", {}).error
    reg.register(function_tool("boomer", lambda a: (_ for _ in ()).throw(ValueError("kaboom"))))
    res = reg.run("boomer", {})
    assert res.ok is False and "kaboom" in res.error


def _reg(authorizer=None):
    r = ToolRegistry(authorizer=authorizer)
    r.register(function_tool("read_vulns", lambda a: ToolResult(ok=True, text="rows")))
    return r


def test_authorizer_denies_by_principal_and_exposes_principal_to_tool():
    def authz(principal, spec, args):
        roles = (principal or {}).get("roles", set())
        return None if "security" in roles else "not permitted"

    reg = _reg(authz)
    assert reg.run("read_vulns", {}, principal={"roles": {"security"}}).ok
    denied = reg.run("read_vulns", {}, principal={"roles": {"guest"}})
    assert not denied.ok and "not permitted" in denied.error and reg.audit[-1]["allowed"] is False
    seen = []
    r2 = ToolRegistry()
    r2.register(function_tool("who", lambda a: seen.append(current_principal()) or ToolResult()))
    r2.run("who", {}, principal="alice")
    assert seen == ["alice"] and current_principal() is None


def test_default_authorizer_governs_registries_then_clears():
    reg = _reg()
    set_default_authorizer(lambda p, spec, args: "blocked fleet-wide")
    try:
        assert not reg.run("read_vulns", {}).ok
    finally:
        set_default_authorizer(None)
    assert reg.run("read_vulns", {}).ok


# --- Agent built-ins go through the registry --------------------------------------------------------------
from agent_harness.agent import Agent
from agent_harness.command_policy import CommandPolicy


@windows_todo("bare command names resolve only against the POSIX trusted directories (/usr/local/bin, /usr/bin, /bin); a Windows resolver for allowlisted executables is not implemented yet")
def test_agent_registry_describes_builtins_and_default_allows_them(tmp_path):
    ag = Agent(root=str(tmp_path), policy=CommandPolicy(allowlist=["cat"], root=str(tmp_path)))
    assert {s["function"]["name"] for s in ag.registry.specs()} == {"read_file", "write_file", "run_bash", "finish"}
    assert ag.call_tool("write_file", {"path": "a.txt", "content": "hi"}) == "ok"
    assert ag.call_tool("run_bash", {"cmd": "cat a.txt"})["stdout"] == "hi"
    assert ag.call_tool("run_bash", {"cmd": "rm a.txt"}) == {"error": "denied"}      # command policy still vets
    assert "invalid arguments" in ag.call_tool("read_file", {"nope": 1})["error"]
    assert "unknown tool" in ag.call_tool("nope")["error"]


def test_agent_tool_level_gate_can_require_approval(tmp_path):
    ag = Agent(root=str(tmp_path), approval=ApprovalPolicy("deny_side_effects"))
    out = ag.call_tool("write_file", {"path": "a.txt", "content": "x"})
    assert "denied" in out["error"] and not (tmp_path / "a.txt").exists()
    assert ag.registry.audit[-1]["tool"] == "write_file" and ag.registry.audit[-1]["allowed"] is False
