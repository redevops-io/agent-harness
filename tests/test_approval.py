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
