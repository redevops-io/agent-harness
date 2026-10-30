# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import boundary (G4) and licence boundary: the QA module imports the standard library and harness core only."""
import ast
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "agent_harness"
QA = PKG / "qa"

#: top-level packages the QA module must never import (plain AGPL must take no Commons-Clause dependency in code)
FORBIDDEN = {"runtime_contracts", "context_runtime", "redevops_qa", "playwright"}
#: harness modules that do not exist at the 6b8ba3a pin; QA code must not reach for them
ABSENT_AT_PIN = {"netpolicy", "supervisor"}
STDLIB = set(sys.stdlib_module_names)
CORE_MODULES = {p.stem if p.suffix == ".py" else p.name for p in PKG.iterdir()
                if (p.suffix == ".py" and p.name != "__init__.py") or (p.is_dir() and (p / "__init__.py").exists())} - {"qa"}


def imports_of(source: str, package: str = "agent_harness.qa"):
    """All imports in ``source`` as (kind, target) with relative imports resolved against ``package``.

    Also reports literal ``importlib.import_module("x")`` and ``__import__("x")`` calls."""
    out = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(("import", alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                prefix = ".".join(base)
                mod = prefix + ("." + node.module if node.module else "")
                if node.module:
                    out.append(("from", mod))
                else:
                    out.extend(("from", f"{mod}.{a.name}") for a in node.names)
            else:
                out.append(("from", node.module))
                if node.module == "agent_harness":
                    out.extend(("from", f"agent_harness.{a.name}") for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                out.append(("dynamic", node.args[0].value))
    return out


def violations(source: str, package: str = "agent_harness.qa"):
    bad = []
    for kind, target in imports_of(source, package):
        top = target.split(".")[0]
        parts = target.split(".")
        if top in FORBIDDEN:
            bad.append((target, "forbidden package"))
        elif top == "agent_harness":
            if len(parts) > 1 and parts[1] in ABSENT_AT_PIN:
                bad.append((target, "module absent at the pin"))
            elif len(parts) > 1 and parts[1] != "qa" and parts[1] not in CORE_MODULES:
                bad.append((target, "not a harness core module"))
        elif top not in STDLIB:
            bad.append((target, "not standard library"))
    return bad


QA_FILES = sorted(QA.rglob("*.py"))
CORE_FILES = sorted(p for p in PKG.rglob("*.py") if QA not in p.parents)


def test_the_expected_files_exist():
    assert {p.name for p in QA_FILES} == {"__init__.py", "canon.py", "errors.py", "ports.py", "records.py", "schemas.py"}


@pytest.mark.parametrize("path", QA_FILES, ids=lambda p: p.name)
def test_qa_modules_import_only_the_standard_library_and_harness_core(path):
    assert violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", QA_FILES, ids=lambda p: p.name)
def test_qa_modules_carry_the_agpl_spdx_header(path):
    assert path.read_text(encoding="utf-8").splitlines()[0] == "# SPDX-License-Identifier: AGPL-3.0-or-later"


#: Imports of a contracts-side package that already exist in harness core at the 6b8ba3a pin: two lazy, optional
#: ``context_runtime`` imports in bench/. They are not QA code, are left byte-identical, and are listed here so a third
#: one cannot appear unnoticed. (The ADR says the harness never imports Context Runtime; at the pin it does, lazily.)
PREEXISTING_CORE_IMPORTS = {
    ("agent_harness/bench/selfcheck.py", "context_runtime"),
    ("agent_harness/bench/jobbench.py", "context_runtime.integrations.agent_harness"),
}


def test_no_file_under_agent_harness_imports_a_contracts_side_package():
    found = set()
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for _, target in imports_of(path.read_text(encoding="utf-8"), _package_of(path)):
            if target.split(".")[0] in FORBIDDEN:
                found.add((rel, target))
    assert found == PREEXISTING_CORE_IMPORTS, found ^ PREEXISTING_CORE_IMPORTS
    assert not [f for f in found if f[0].startswith("agent_harness/qa/")]
    assert {t.split(".")[0] for _, t in found} == {"context_runtime"}          # never runtime_contracts/redevops_qa/playwright


def _package_of(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        return ".".join(parts)
    return ".".join(parts[:-1])


def test_core_never_imports_the_qa_module():
    for path in CORE_FILES:
        found = [t for _, t in imports_of(path.read_text(encoding="utf-8"), _package_of(path))
                 if t == "agent_harness.qa" or t.startswith("agent_harness.qa.")]
        assert found == [], (path, found)


def test_core_init_names_no_qa():
    text = (PKG / "__init__.py").read_text(encoding="utf-8")
    assert "qa" not in [n for _, t in imports_of(text, "agent_harness") for n in t.split(".")[1:]]


def test_pyproject_declares_an_empty_qa_extra_and_no_contracts_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["dependencies"] == []
    assert project["optional-dependencies"]["qa"] == []
    assert project["optional-dependencies"]["dev"] == ["pytest>=8.0"]
    assert set(project["optional-dependencies"]) == {"dev", "qa"}
    assert project["requires-python"] == ">=3.10"
    assert project["version"] == "0.1.0"
    assert project["license"] == "AGPL-3.0-or-later"
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for name in ("runtime-contracts", "runtime_contracts", "context-runtime", "context_runtime", "redevops-qa",
                 "redevops_qa", "playwright", "pyyaml"):
        assert name not in text, name
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["agent_harness"]


# --- the scanner must be able to fail (mutation tests for the boundary tests themselves) ---------------------------

@pytest.mark.parametrize("source,why", [
    ("import runtime_contracts", "forbidden package"),
    ("import runtime_contracts.qa.records as r", "forbidden package"),
    ("from runtime_contracts.qa import TestSpec", "forbidden package"),
    ("from context_runtime.tools import base", "forbidden package"),
    ("import redevops_qa", "forbidden package"),
    ("from redevops_qa.bridge import x", "forbidden package"),
    ("import playwright.sync_api", "forbidden package"),
    ("def f():\n    from playwright.sync_api import sync_playwright", "forbidden package"),
    ("try:\n    import runtime_contracts\nexcept ImportError:\n    pass", "forbidden package"),
    ("import importlib\nimportlib.import_module('runtime_contracts')", "forbidden package"),
    ("__import__('playwright')", "forbidden package"),
    ("import yaml", "not standard library"),
    ("import requests", "not standard library"),
    ("from agent_harness.netpolicy import ip_blocked", "module absent at the pin"),
    ("from agent_harness import supervisor", "module absent at the pin"),
    ("from .. import netpolicy", "module absent at the pin"),
    ("from agent_harness.not_a_module import x", "not a harness core module"),
])
def test_the_boundary_scanner_flags_violations(source, why):
    bad = violations(source)
    assert bad and bad[0][1] == why, bad


@pytest.mark.parametrize("source", [
    "import json\nimport hashlib\nfrom dataclasses import dataclass",
    "from . import canon\nfrom .records import Step",
    "from agent_harness import llm",
    "from agent_harness.tools import ToolRegistry",
    "import agent_harness.qa.canon",
    "from __future__ import annotations",
])
def test_the_boundary_scanner_accepts_the_allowed_imports(source):
    assert violations(source) == []
