"""Approval/permission policy module.

Modes: allowlist | edits-only | bypass
Decide(action) returns allow/deny. Never open grant by default.

In ``allowlist`` mode a shell command is allowed only when:

* its executable resolves, through a fixed trusted search path (not the
  caller's ``PATH`` or working directory), to the same absolute path as an
  allowlist entry;
* the command name contains no ``/`` unless that exact literal is allowlisted;
* file-like arguments stay inside the workspace root (after ``realpath``, so
  symlinks and ``..`` are accounted for);
* it does not execute, or hand to an interpreter, a file written during the
  session (see :meth:`ApprovalPolicy.note_write`).

This is a policy check, not isolation: an allowed command still runs as the
host user. Use an OS-level sandbox or container for untrusted workloads.
"""
import os
import shlex
from typing import Any, Dict, List, Optional, Set

# Shell metacharacters that enable chaining/redirection/substitution. A command
# containing any of these is rejected outright — the allowlist only vets a
# single executable, so these would let a caller smuggle in extra commands.
_SHELL_METACHARS = (";", "|", "&", "$", "`", ">", "<", "\n", "\r", "(", ")", "{", "}")

# Fixed search path used to resolve bare command names.
TRUSTED_PATH = ("/usr/local/bin", "/usr/bin", "/bin")

# Commands whose arguments are never treated as file names.
_NO_FILE_ARGS = frozenset({"echo", "pwd", "true", "false"})

# Commands that load and run code from a file argument.
_INTERPRETERS = frozenset({
    "sh", "bash", "dash", "zsh", "ksh", "csh", "fish", "env", "xargs", "find",
    "python", "python3", "perl", "ruby", "node", "php", "lua", "awk", "gawk",
    "make", "source", "exec", "busybox", "nice", "timeout", "nohup",
})


def _is_interpreter(path: str) -> bool:
    name = os.path.basename(path)
    return name in _INTERPRETERS or name.startswith(("python", "perl", "ruby"))


def resolve_command(name: str, search_path=TRUSTED_PATH) -> Optional[str]:
    """Resolve a bare command name to an absolute path using ``search_path``.

    Returns None for names containing a path separator or when not found.
    """
    if not name or "/" in name:
        return None
    for d in search_path:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _contained(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


class ApprovalPolicy:
    def __init__(self, mode: str = "allowlist", allowlist: List[str] = None,
                 root: Optional[str] = None, trusted_path=TRUSTED_PATH):
        self.mode = mode
        self.allowlist = allowlist or [
            "echo", "ls", "cat", "pwd", "true", "false"
        ]
        self.root = root
        self.trusted_path = tuple(trusted_path)
        self.written: Set[str] = set()

    # -- helpers -----------------------------------------------------------
    def _root(self) -> str:
        return os.path.realpath(
            self.root or os.environ.get("AGENT_HARNESS_ROOT") or os.getcwd())

    def note_write(self, path: str) -> None:
        """Record a file written during the session; it needs re-approval to run."""
        self.written.add(os.path.realpath(path))

    def clear_written(self, path: Optional[str] = None) -> None:
        """Re-approve a written file (or all of them when ``path`` is None)."""
        if path is None:
            self.written.clear()
        else:
            self.written.discard(os.path.realpath(path))

    def resolve(self, name: str) -> Optional[str]:
        """Absolute path an allowlisted command name would run as, or None."""
        if "/" in name:
            if name not in self.allowlist:
                return None
            return name if os.path.isabs(name) else os.path.join(self._root(), name)
        if name not in self.allowlist:
            return None
        return resolve_command(name, self.trusted_path)

    def _args_ok(self, exe: str, args: List[str], root: str) -> bool:
        check_files = os.path.basename(exe) not in _NO_FILE_ARGS
        interp = _is_interpreter(exe)
        for arg in args:
            cands = [arg]
            if arg.startswith("-") and "=" in arg:
                cands.append(arg.split("=", 1)[1])
            for c in cands:
                if not c:
                    continue
                is_opt = c.startswith("-") and c is arg
                if is_opt and not interp:
                    continue
                real = os.path.realpath(os.path.join(root, os.path.expanduser(c)))
                if interp and real in self.written:
                    return False
                if check_files and not is_opt and not _contained(real, root):
                    return False
        return True

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
            exe = self.resolve(argv[0])
            if exe is None:
                return "deny"
            root = self._root()
            if os.path.realpath(exe) in self.written:
                return "deny"
            if not self._args_ok(exe, argv[1:], root):
                return "deny"
            return "allow"
        if action_type == "edit":
            return "allow" if self.mode in ("allowlist", "edits-only") else "deny"
        if self.mode == "edits-only":
            return "deny"
        # default deny, never open grant
        return "deny"

    def resolve_argv(self, cmd: str) -> Optional[List[str]]:
        """argv for ``cmd`` with argv[0] replaced by its resolved absolute path."""
        try:
            argv = shlex.split(cmd)
        except ValueError:
            return None
        if not argv:
            return None
        exe = self.resolve(argv[0]) if self.mode != "bypass" else (
            argv[0] if "/" in argv[0] else resolve_command(argv[0], self.trusted_path))
        if exe is None:
            return None
        return [exe] + argv[1:]


class Approver(ApprovalPolicy):
    """Backwards-compatible alias for ApprovalPolicy (older docs/examples used
    this name). Also accepts ``mode="never"`` as an explicit deny-everything
    policy.
    """

    def decide(self, action: Dict[str, Any]) -> str:
        if self.mode == "never":
            return "deny"
        return super().decide(action)
