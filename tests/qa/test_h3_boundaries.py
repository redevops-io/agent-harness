# SPDX-License-Identifier: AGPL-3.0-or-later
"""H3: layering and licence boundaries of the recovery module. The H0/H1 boundary tests already scan every file under
``agent_harness/qa`` for forbidden packages and the SPDX header; these add what is specific to recovery: what it may
import, who may import it, that it can never reach the verifier or an adapter, and that its scoring is integer-only."""
import ast

import pytest

from .test_boundaries import QA, imports_of, qa_internal_imports
from .test_h2_boundaries import FORBIDDEN_FOR_DETERMINISM, closure, imported_roots, top

RECOVERY = (QA / "recovery.py")
ALL_QA = sorted(p.stem for p in QA.glob("*.py") if p.name != "__init__.py")


def test_recovery_imports_only_the_neutral_modules_it_needs():
    assert top(qa_internal_imports("recovery.py")) <= {"errors", "idempotency", "ports", "recovery_types", "records"}


def test_recovery_cannot_reach_the_verifier_the_classifier_the_spec_store_the_policy_or_the_runner():
    reach = closure("recovery")
    assert not reach & {"verifier", "classify", "specstore", "policy", "runner", "hooks", "receipts", "cleanup",
                        "checkpoint", "limits", "runstate", "run_errors", "clock"}, reach
    assert reach <= {"errors", "idempotency", "ports", "recovery_types", "records", "canon", "schemas"}, reach


def test_only_the_runner_imports_the_recovery_module():
    users = {name for name in ALL_QA if name != "recovery" and "recovery" in top(qa_internal_imports(f"{name}.py"))}
    assert users == {"runner", "recovery_model"}          # plan H4 builds on it; nothing else may


def test_nothing_the_verdict_depends_on_imports_recovery():
    for name in ("verifier", "classify", "specstore", "records", "canon", "errors", "ports", "recovery_types",
                 "idempotency", "policy", "receipts", "limits", "runstate"):
        assert "recovery" not in closure(name), name


def test_recovery_reads_no_clock_randomness_environment_files_or_network():
    source = RECOVERY.read_text(encoding="utf-8")
    roots = imported_roots(source)
    assert not {r for r in roots if r in FORBIDDEN_FOR_DETERMINISM or r.split(".")[0] in FORBIDDEN_FOR_DETERMINISM}, roots
    tree = ast.parse(source)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"open", "exec", "eval", "compile", "__import__", "input", "print", "float", "hash", "id"}, called


def test_recovery_scoring_is_integer_only():
    tree = ast.parse(RECOVERY.read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in {"random", "time", "uuid"}]


def test_recovery_never_calls_an_adapter_operation_itself():
    """It is handed a ``locate`` callable by the runner; it holds no adapter, session or clock."""
    tree = ast.parse(RECOVERY.read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"act", "observe", "assert_", "reset", "open_session", "close_session", "capture", "run_check",
                        "open_workspace", "monotonic_ms", "now_utc", "get_assertion_set", "verify_assertion"}, attrs


def test_the_recovery_types_that_cross_the_runner_boundary_hold_no_assertion_expected_value_or_verdict():
    from dataclasses import fields
    from agent_harness.qa import recovery as RV
    banned = {"expected", "assertion", "assertions", "predicate", "verdict", "actual", "outcome", "check"}
    for cls in (RV.FaultInputs, RV.RecoveryDecision, RV.LocatorSelection, RV.RecoveryLimits):
        assert not {f.name for f in fields(cls)} & banned, cls
    assert not [m for m in dir(RV.DeterministicRecovery) if any(b in m for b in ("assert", "verdict", "expected"))]


def test_a_scanner_for_the_recovery_boundary_can_fail():
    bad = "from .verifier import load_frozen_assertions\nimport random\n"
    targets = {t.split(".")[-1] for _, t in imports_of(bad)}
    assert {"verifier", "random"} <= targets
    assert imported_roots(bad) & FORBIDDEN_FOR_DETERMINISM
