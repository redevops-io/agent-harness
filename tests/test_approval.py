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
from agent_harness.agent import Agent, run_bash, write_file
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
