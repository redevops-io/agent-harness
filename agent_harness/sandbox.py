"""Sandbox runner for shell/tool actions.

Confines to working directory, timeout, captures stdout/stderr/returncode.
"""
import os
import subprocess
import tempfile
from typing import Any, Dict

def run_shell(cmd: str, cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
    if cwd is None:
        cwd = os.getcwd()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
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
