# SPDX-License-Identifier: AGPL-3.0-or-later
"""CI gate for the QA suite: run ``tests/qa`` and fail unless every test ran.

Used by ``.github/workflows/qa.yml`` (Python 3.11 and 3.12). It runs pytest on the QA tests with a JUnit report, then
fails when

* pytest itself fails (a failing test, a collection error, or no test collected: pytest exit code 5);
* zero tests were collected;
* any test was skipped (or xfailed) and is not on the allow-list.

The allow-list is ``tests/qa/ci_skip_allowlist.txt``: one test id per line (``<file>::<name>``, shell-style wildcards
allowed), each entry preceded by a comment line that says why that skip is acceptable. It is empty: the QA suite has
no permitted skip. Standard library only; this file is not collected by pytest (its name does not start with ``test_``).

    python tests/qa/ci_gate.py                       # run tests/qa
    python tests/qa/ci_gate.py --tests DIR --allowlist FILE
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_ALLOWLIST = HERE / "ci_skip_allowlist.txt"


def read_allowlist(text: str) -> List[str]:
    """Entries of an allow-list file: non-blank lines that do not start with ``#``."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def unexplained_entries(text: str) -> List[str]:
    """Entries that are not directly preceded by a comment line (an allowed skip must say why)."""
    bad, previous_comment = [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            previous_comment = False
        elif line.startswith("#"):
            previous_comment = True
        else:
            if not previous_comment:
                bad.append(line)
            previous_comment = False
    return bad


def test_id(case: ET.Element) -> str:
    """``<file>::<name>`` from a JUnit ``testcase`` element (classname ``tests.qa.test_x`` or ``tests.qa.test_x.Cls``)."""
    parts = [p for p in (case.get("classname") or "").split(".") if p]
    module = next((p for p in parts if p.startswith("test_")), parts[-1] if parts else "")
    return f"{module}.py::{case.get('name') or ''}"


def evaluate(junit_xml: str, allowlist: Sequence[str]) -> Tuple[int, List[str], List[str]]:
    """Return ``(collected, skipped_ids, problems)`` for a JUnit report."""
    root = ET.fromstring(junit_xml)
    cases = list(root.iter("testcase"))
    skipped, problems = [], []
    for case in cases:
        ident = test_id(case)
        if case.find("failure") is not None or case.find("error") is not None:
            problems.append(f"FAILED: {ident}")
        if case.find("skipped") is not None:
            skipped.append(ident)
            if not any(fnmatch.fnmatchcase(ident, pattern) for pattern in allowlist):
                problems.append(f"SKIPPED and not allow-listed: {ident}")
    if not cases:
        problems.append("no QA tests were collected")
    return len(cases), skipped, problems


def run(tests: Path, allowlist_path: Optional[Path], extra: Sequence[str] = ()) -> int:
    allow_text = Path(allowlist_path).read_text(encoding="utf-8") if allowlist_path and Path(allowlist_path).exists() else ""
    for entry in unexplained_entries(allow_text):
        print(f"ci_gate: allow-list entry without a preceding comment: {entry}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", str(tests), "-q", "-rs", "-p", "no:cacheprovider",
               f"--junitxml={report}", *extra]
        done = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ))
        problems: List[str] = []
        collected, skipped = 0, []
        if report.exists():
            collected, skipped, problems = evaluate(report.read_text(encoding="utf-8"), read_allowlist(allow_text))
        else:
            problems.append("pytest wrote no report")
        if done.returncode != 0:
            problems.append(f"pytest exited with status {done.returncode}")
    print(f"ci_gate: collected={collected} skipped={len(skipped)} allow-listed-patterns={len(read_allowlist(allow_text))}")
    for line in problems:
        print(f"ci_gate: {line}", file=sys.stderr)
    return 1 if problems else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tests", type=Path, default=HERE)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    args, extra = parser.parse_known_args(argv)
    return run(args.tests, args.allowlist, extra)


if __name__ == "__main__":
    raise SystemExit(main())
