"""Approval/permission policy module.

Modes: allowlist | edits-only | bypass
Decide(action) returns allow/deny. Never open grant by default.
Safe shell-command prefixes in allowlist.
"""
import os
from typing import Any, Dict, List

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
            prefix = cmd.split()[0] if cmd else ""
            if prefix in self.allowlist:
                return "allow"
            return "deny"
        if action_type == "edit":
            return "allow" if self.mode in ("allowlist", "edits-only") else "deny"
        if self.mode == "edits-only":
            return "deny"
        # default deny, never open grant
        return "deny"
