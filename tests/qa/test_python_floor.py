# SPDX-License-Identifier: AGPL-3.0-or-later
"""Python floors (D9): core stays 3.10-compatible and never imports qa; qa needs 3.11+ and says so clearly.

This file must stay importable on Python 3.10: standard library only, no 3.11-only syntax."""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "agent_harness"
CORE_FILES = sorted(p for p in PKG.rglob("*.py") if PKG / "qa" not in p.parents)


def child(code, python=None, extra_env=None):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(extra_env or {})
    return subprocess.run([python or sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=120, cwd=str(ROOT))


def test_core_sources_parse_with_python_3_10_grammar():
    assert CORE_FILES, "no core files found"
    for path in CORE_FILES + [PKG / "qa" / "__init__.py"]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


def test_the_grammar_check_can_fail():
    # sanity: a 3.11-only construct is rejected under feature_version=(3, 10), so the test above has teeth
    with pytest.raises(SyntaxError):
        ast.parse("try:\n    pass\nexcept* ValueError:\n    pass\n", feature_version=(3, 10))


def test_importing_agent_harness_does_not_import_qa():
    done = child("import sys, agent_harness; print(any(m == 'agent_harness.qa' or m.startswith('agent_harness.qa.') for m in sys.modules))")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


GUARD_CODE = """
import sys
import agent_harness                       # core imports first, on the real version
sys.version_info = (3, 10, 12, 'final', 0)  # then pretend to be 3.10
try:
    import agent_harness.qa
except ImportError as exc:
    print('ImportError:', exc)
    print('submodules loaded:', sorted(m for m in sys.modules if m.startswith('agent_harness.qa.')))
else:
    print('NO ERROR')
"""


def test_the_guard_fails_clearly_below_3_11_and_loads_nothing_else():
    done = child(GUARD_CODE)
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "ImportError:" in out and "NO ERROR" not in out
    assert "requires Python 3.11 or newer" in out and "3.10.12" in out and "core still supports" in out
    assert "submodules loaded: []" in out                # the guard ran before any submodule was imported


def test_the_guard_passes_on_3_11_and_newer():
    if sys.version_info < (3, 11):
        done = child("import agent_harness.qa")
        assert done.returncode != 0 and "requires Python 3.11 or newer" in done.stderr
        return
    done = child("import agent_harness.qa as q; print(q.NEUTRAL_API, q.MINIMUM_PYTHON)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "qa-neutral/1 (3, 11)"


def test_the_minimum_is_3_11_and_the_message_names_both_versions():
    text = (PKG / "qa" / "__init__.py").read_text(encoding="utf-8")
    assert "MINIMUM_PYTHON = (3, 11)" in text
    assert "sys.version_info < MINIMUM_PYTHON" in text
    # the guard sits before the first import of a submodule
    assert text.index("raise ImportError") < text.index("from . import")


def test_real_older_interpreters_on_this_machine_if_any():
    """Run the real guard on any python3.10 found on PATH (none is required; CI runs the 3.10 job)."""
    exercised = []
    for name in ("python3.10",):
        exe = shutil.which(name)
        if not exe:
            continue
        core = child("import agent_harness", python=exe)
        assert core.returncode == 0, core.stderr
        qa = child("import agent_harness.qa", python=exe)
        assert qa.returncode != 0 and "requires Python 3.11 or newer" in qa.stderr
        exercised.append(name)
    assert isinstance(exercised, list)
