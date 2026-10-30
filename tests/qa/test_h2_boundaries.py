# SPDX-License-Identifier: AGPL-3.0-or-later
"""H2: who may import whom inside the QA module, and what the single-run modules may not touch (clock, randomness,
filesystem, network, processes). The H0/H1 boundary tests already scan every file under ``agent_harness/qa`` for
forbidden packages; these add the H2 layering."""
import ast

import pytest

from .test_boundaries import QA, imports_of, qa_internal_imports

H2_MODULES = ["checkpoint", "cleanup", "clock", "hooks", "idempotency", "limits", "policy", "receipts", "run_errors",
              "runner", "runstate"]
H0_H1_MODULES = ["canon", "classify", "errors", "ports", "records", "recovery_types", "schemas", "specstore", "verifier"]


def top(names):
    return {n.split(".")[0] for n in names}


def closure(start):
    seen, todo = set(), [start]
    while todo:
        for t in qa_internal_imports(todo.pop() + ".py"):
            mod = t.split(".")[0]
            if mod not in seen:
                seen.add(mod)
                todo.append(mod)
    return seen


@pytest.mark.parametrize("name", H2_MODULES)
def test_every_h2_module_exists_and_imports_only_qa_modules_inside_the_package(name):
    assert (QA / f"{name}.py").is_file()
    assert top(qa_internal_imports(f"{name}.py")) <= set(H2_MODULES + H0_H1_MODULES)


def test_h0_and_h1_modules_do_not_import_any_h2_module():
    for name in H0_H1_MODULES:
        assert not top(qa_internal_imports(f"{name}.py")) & set(H2_MODULES), name


def test_recovery_types_cannot_reach_the_runner_the_policy_or_any_h2_module():
    assert not closure("recovery_types") & set(H2_MODULES)
    assert not closure("recovery_types") & {"verifier", "classify", "specstore"}


def test_the_hooks_module_is_light_enough_for_a_recovery_implementation_to_depend_on():
    got = top(qa_internal_imports("hooks.py"))
    assert got <= {"errors", "records"}, got
    assert not closure("hooks") & {"verifier", "classify", "specstore", "runner", "policy", "receipts"}


def test_the_policy_module_depends_on_records_and_ports_only():
    assert top(qa_internal_imports("policy.py")) <= {"canon", "errors", "ports", "records"}
    assert not closure("policy") & {"verifier", "classify", "specstore", "runner", "receipts", "hooks"}


def test_receipts_idempotency_limits_and_runstate_do_not_depend_on_the_runner_or_the_verifier():
    for name in ("receipts", "idempotency", "limits", "runstate", "checkpoint", "run_errors"):
        assert not closure(name) & {"runner", "verifier", "classify", "specstore", "policy"}, name


def test_only_the_runner_ties_the_h2_pieces_to_the_verifier_and_the_classifier():
    users = {n for n in H2_MODULES if closure(n) & {"verifier", "classify"}}
    assert users == {"runner", "cleanup"}                         # cleanup reads the classifier's Signal type


FORBIDDEN_FOR_DETERMINISM = {"random", "time", "datetime", "uuid", "secrets", "os", "locale", "socket", "subprocess",
                             "threading", "pathlib", "shutil", "tempfile", "glob", "io", "multiprocessing", "asyncio",
                             "urllib.request", "http", "ssl"}


def imported_roots(source):
    tree = ast.parse(source)
    roots = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and not n.level}
    return roots


@pytest.mark.parametrize("name", [m for m in H2_MODULES if m != "clock"])
def test_h2_modules_read_no_clock_randomness_environment_files_or_network(name):
    source = (QA / f"{name}.py").read_text(encoding="utf-8")
    roots = imported_roots(source)
    assert not {r for r in roots if r in FORBIDDEN_FOR_DETERMINISM or r.split(".")[0] in FORBIDDEN_FOR_DETERMINISM}, roots
    tree = ast.parse(source)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"open", "exec", "eval", "compile", "__import__", "input", "print"}, called


def test_the_clock_module_is_the_only_place_that_reads_real_time():
    roots = imported_roots((QA / "clock.py").read_text(encoding="utf-8"))
    assert roots == {"__future__", "datetime", "time"}
    for name in H0_H1_MODULES + [m for m in H2_MODULES if m != "clock"]:
        assert not imported_roots((QA / f"{name}.py").read_text(encoding="utf-8")) & {"time", "datetime"}, name


def test_the_package_init_does_not_import_any_h2_module():
    text = (QA / "__init__.py").read_text(encoding="utf-8")
    names = {t.split(".")[2] for _, t in imports_of(text) if t.startswith("agent_harness.qa.") and t.count(".") >= 2}
    assert names and not names & set(H2_MODULES), names
