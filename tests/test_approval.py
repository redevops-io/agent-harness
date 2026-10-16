import pytest
from agent_harness.approval import ApprovalPolicy

def test_allowlist_allows_prefix():
    policy = ApprovalPolicy(mode="allowlist")
    assert policy.decide({"type": "shell", "command": "echo hi"}) == "allow"

def test_allowlist_denies_other():
    policy = ApprovalPolicy(mode="allowlist")
    assert policy.decide({"type": "shell", "command": "rm -rf /"}) == "deny"

def test_bypass_allows():
    policy = ApprovalPolicy(mode="bypass")
    assert policy.decide({"type": "shell", "command": "rm -rf /"}) == "allow"

def test_edits_only():
    policy = ApprovalPolicy(mode="edits-only")
    assert policy.decide({"type": "edit", "path": "x"}) == "allow"
    assert policy.decide({"type": "shell", "command": "echo hi"}) == "deny"


# --- resolved allowlist, workspace containment, written-file tracking ---------
import os
from agent_harness.agent import Agent, _run_bash as run_bash, _write_file as write_file
from agent_harness.approval import resolve_command


def _d(policy, cmd):
    return policy.decide({"type": "shell", "command": cmd})


def test_bare_command_resolves_to_trusted_absolute_path():
    p = resolve_command("echo")
    assert p and os.path.isabs(p)


def test_rejects_path_containing_command_names(tmp_path):
    p = ApprovalPolicy(root=str(tmp_path))
    assert _d(p, "./echo hi") == "deny"
    assert _d(p, "sub/cat x") == "deny"
    assert _d(p, "/tmp/echo hi") == "deny"


def test_literal_path_entry_must_match_exactly(tmp_path):
    exe = resolve_command("echo")
    p = ApprovalPolicy(allowlist=[exe], root=str(tmp_path))
    assert _d(p, f"{exe} hi") == "allow"
    assert _d(p, "echo hi") == "deny"


def test_local_file_named_like_allowlisted_command_is_not_used(tmp_path, monkeypatch):
    fake = tmp_path / "ls"
    fake.write_text("#!/bin/sh\necho local\n")
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    out = run_bash("ls", root=str(tmp_path), policy=ApprovalPolicy(root=str(tmp_path)))
    assert "local" not in out.get("stdout", "")


def test_file_arguments_stay_inside_root(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("a")
    outside = tmp_path / "secret.txt"
    outside.write_text("s")
    (root / "link").symlink_to(outside)
    p = ApprovalPolicy(root=str(root))
    assert _d(p, "cat a.txt") == "allow"
    assert _d(p, "ls") == "allow"
    assert _d(p, "cat ../secret.txt") == "deny"
    assert _d(p, f"cat {outside}") == "deny"
    assert _d(p, "cat link") == "deny"
    assert _d(p, "ls /") == "deny"
    assert _d(p, "cat --file=/etc/hostname") == "deny"


def test_written_files_cannot_be_executed_or_run_by_interpreters(tmp_path):
    root = str(tmp_path)
    p = ApprovalPolicy(allowlist=["echo", "cat", "sh", "python3", "./run.sh"], root=root)
    assert write_file("run.sh", "echo hi\n", root=root, policy=p) == "ok"
    assert _d(p, "./run.sh") == "deny"
    assert _d(p, "sh run.sh") == "deny"
    assert _d(p, "python3 ./run.sh") == "deny"
    assert _d(p, "cat run.sh") == "allow"
    p.clear_written(os.path.join(root, "run.sh"))
    assert _d(p, "sh run.sh") == "allow"


def test_run_bash_uses_agent_policy_and_root(tmp_path):
    ag = Agent(policy=ApprovalPolicy(allowlist=["pwd", "cat"]), root=str(tmp_path))
    out = ag.tools["run_bash"]("pwd")
    assert out["stdout"].strip() == os.path.realpath(tmp_path)
    assert ag.tools["run_bash"]("echo hi") == {"error": "denied"}
    ag.tools["write_file"]("x.txt", "data")
    assert ag.tools["run_bash"]("cat x.txt")["stdout"] == "data"


def test_tool_dispatch_rejects_root_and_policy_arguments(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "ok.txt").write_text("fine")
    ag = Agent(root=str(root))
    assert ag.call_tool("read_file", {"path": "ok.txt"}) == "fine"
    assert "error" in ag.call_tool("read_file", {"path": "hostname", "root": "/etc"})
    assert "error" in ag.call_tool("write_file", {"path": "a", "content": "x", "root": "/tmp"})
    assert "error" in ag.call_tool("run_bash", {"cmd": "ls", "policy": None})
    assert "error" in ag.call_tool("nope", {})


def _interp_policy(root):
    return ApprovalPolicy(allowlist=["python3", "sh", "cat", "ls"], root=str(root))


def test_interpreter_denied_for_written_packages_and_siblings(tmp_path):
    root = str(tmp_path)
    p = _interp_policy(root)
    (tmp_path / "main.py").write_text("pass\n")
    write_file("helper.py", "x = 1\n", root=root, policy=p)
    os.mkdir(tmp_path / "pkg")
    write_file("pkg/__init__.py", "", root=root, policy=p)
    assert _d(p, "python3 pkg") == "deny"
    assert _d(p, "python3 main.py") == "deny"  # sits beside a written module
    assert _d(p, "cat helper.py") == "allow"


def test_interpreter_inline_code_denied_unless_allowlisted_exactly(tmp_path):
    p = _interp_policy(tmp_path)
    assert _d(p, "python3 -c pass") == "deny"
    assert _d(p, "python3 -m pkg") == "deny"
    assert _d(p, "python3 -cpass") == "deny"
    assert _d(p, "sh -c true") == "deny"
    q = ApprovalPolicy(allowlist=["python3", "python3 -c pass"], root=str(tmp_path))
    assert _d(q, "python3 -c pass") == "allow"
    assert _d(q, "python3 -c other") == "deny"


def test_short_option_attached_values_are_checked(tmp_path):
    p = ApprovalPolicy(root=str(tmp_path))
    assert _d(p, "cat -f/etc/hostname") == "deny"
    assert _d(p, "ls -la") == "allow"
