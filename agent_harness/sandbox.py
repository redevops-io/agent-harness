"""Sandbox runner for shell/tool actions.

Runs a command as a non-shell argv (``shell=False``), confined to a working
directory with a timeout, and captures stdout/stderr/returncode.

NOTE: this is *not* a real isolation boundary. The command still runs as the
host user with full filesystem/network access — ``cwd`` and ``timeout`` only
constrain where it starts and how long it runs. Pair this with the command policy
allowlist (and, for real isolation, an OS-level sandbox/container) before
running untrusted commands.
"""
import os
import shlex
from typing import Any, Callable, Dict, List, Optional

from agent_harness.command_policy import TRUSTED_PATH, resolve_command
from agent_harness.process_runner import ProcessRunner, current_control

# Per stream; the process runner counts what it drops. Tool output is clipped much lower by the loop.
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
TIMEOUT_RC = 124      # the timeout(1) convention, kept for callers that test for it
CANCELLED_RC = 130

def _is_path(name: str) -> bool:
    return "/" in name or os.sep in name or (os.altsep is not None and os.altsep in name)


def run_shell(cmd: str, cwd: str = None, timeout: int = 30,
              argv: Optional[List[str]] = None,
              cancel: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Run ``cmd`` (or a pre-resolved ``argv``) without a shell.

    Bare command names are resolved against a fixed trusted search path, never
    the caller's ``PATH`` or the working directory.

    The child runs under :mod:`agent_harness.process_runner`: it and everything it starts are stopped together on a
    timeout or a cancel (``cancel``, or the one set by an enclosing ``run_control``), and the effective timeout is
    never longer than the time left in an enclosing ``run_control``. The result also carries ``tree_gone``.
    """
    if cwd is None:
        cwd = os.getcwd()
    try:
        if argv is None:
            argv = shlex.split(cmd)
        if not argv:
            return {"stdout": "", "stderr": "empty command", "returncode": 1}
        if not _is_path(argv[0]):
            exe = resolve_command(argv[0])
            if exe is None:
                return {"stdout": "", "stderr": "command not found", "returncode": 127}
            argv = [exe] + list(argv[1:])
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(TRUSTED_PATH)
        ctl = current_control()
        if ctl is not None:
            remaining = ctl.remaining()
            if remaining is not None:
                timeout = min(timeout, remaining)
            if ctl.cancel is not None:
                cancel = ctl.cancel if cancel is None else (lambda a=cancel, b=ctl.cancel: bool(a() or b()))
        res = ProcessRunner(max_output=MAX_OUTPUT_BYTES).run(argv, cwd=cwd, env=env, timeout=timeout, cancel=cancel)
        if res.cancelled:
            return {"stdout": "", "stderr": "cancelled", "returncode": CANCELLED_RC, "tree_gone": res.tree_gone}
        if res.timed_out:
            return {"stdout": "", "stderr": "timeout", "returncode": TIMEOUT_RC, "tree_gone": res.tree_gone}
        if res.returncode is None:
            return {"stdout": "", "stderr": res.error, "returncode": 1}
        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode,
            "tree_gone": res.tree_gone,
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": 1}

def run(action: Dict[str, Any], cwd: str = None, timeout: int = 30,
        cancel: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    if action.get("type") == "shell":
        return run_shell(action.get("command", ""), cwd, timeout,
                         argv=action.get("argv"), cancel=cancel)
    return {"stdout": "", "stderr": "unknown action", "returncode": 1}


class Sandbox:
    """Context-manager wrapper around :func:`run`/:func:`run_shell`.

    Binds a working directory and timeout for the duration of the ``with`` block.
    See the module docstring: this is a soft confinement, not OS-level isolation.
    """

    def __init__(self, cwd: str = None, timeout: int = 30):
        self.cwd = cwd if cwd is not None else os.getcwd()
        self.timeout = timeout

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def run(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return run(action, cwd=self.cwd, timeout=self.timeout)
