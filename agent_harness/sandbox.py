"""Sandbox runner for shell/tool actions.

Runs a command as a non-shell argv (``shell=False``), confined to a working
directory with a timeout, and captures stdout/stderr/returncode.

NOTE: this is *not* a real isolation boundary. The command still runs as the
host user with full filesystem/network access — ``cwd`` and ``timeout`` only
constrain where it starts and how long it runs. Pair this with the approval
allowlist (and, for real isolation, an OS-level sandbox/container) before
running untrusted commands.
"""
import os
import shlex
import subprocess
from typing import Any, Dict

def run_shell(cmd: str, cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
    if cwd is None:
        cwd = os.getcwd()
    try:
        argv = shlex.split(cmd)
        if not argv:
            return {"stdout": "", "stderr": "empty command", "returncode": 1}
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": 124}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": 1}

def run(action: Dict[str, Any], cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
    if action.get("type") == "shell":
        return run_shell(action.get("command", ""), cwd, timeout)
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
