"""Approval/permission policy module.

Modes: allowlist | edits-only | bypass
Decide(action) returns allow/deny. Never open grant by default.
Safe shell-command prefixes in allowlist.
"""
import os
import shlex
from typing import Any, Dict, List

# Shell metacharacters that enable chaining/redirection/substitution. A command
# containing any of these is rejected outright — the allowlist only vets a
# single executable, so these would let a caller smuggle in extra commands.
_SHELL_METACHARS = (";", "|", "&", "$", "`", ">", "<", "\n", "\r", "(", ")", "{", "}")

class ApprovalPolicy:
    def __init__(self, mode: str = "allowlist", allowlist: List[str] = None):
        self.mode = mode
        self.allowlist = allowlist or [
            "echo", "ls", "cat", "pwd", "true", "false"
        ]

    def decide(self, action: Dict[str, Any]) -> str:
        if self.mode == "bypass":
            return "allow"
        action_type = action.get("type", "")
        if action_type == "shell":
            if self.mode == "edits-only":
                return "deny"
            cmd = action.get("command", "")
            if not cmd or any(ch in cmd for ch in _SHELL_METACHARS):
                return "deny"
            try:
                argv = shlex.split(cmd)
            except ValueError:
                # Unbalanced quotes etc. — reject rather than guess.
                return "deny"
            if not argv:
                return "deny"
            # Validate the real executable (basename) against the allowlist, not
            # just the raw first token, so paths like /bin/echo are normalized.
            if os.path.basename(argv[0]) in self.allowlist:
                return "allow"
            return "deny"
        if action_type == "edit":
            return "allow" if self.mode in ("allowlist", "edits-only") else "deny"
        if self.mode == "edits-only":
            return "deny"
        # default deny, never open grant
        return "deny"


class Approver(ApprovalPolicy):
    """Backwards-compatible alias for ApprovalPolicy (older docs/examples used
    this name). Also accepts ``mode="never"`` as an explicit deny-everything
    policy.
    """

    def decide(self, action: Dict[str, Any]) -> str:
        if self.mode == "never":
            return "deny"
        return super().decide(action)
