# SPDX-License-Identifier: AGPL-3.0-or-later
"""H4: layering and licence boundaries of model-candidate recovery. The H0 to H3 boundary tests already scan every file under
``agent_harness/qa`` for forbidden packages and the SPDX header; these add what is specific to H4: the module imports the
standard library and a handful of neutral modules only, no model or HTTP client (not even the harness core ``llm`` module,
which speaks HTTP), nothing that can reach the verifier, the classifier, the spec store or an adapter, and only the runner
imports it."""
import ast

import pytest

from .test_boundaries import PKG, QA, STDLIB, imports_of, qa_internal_imports, violations
from .test_h2_boundaries import FORBIDDEN_FOR_DETERMINISM, closure, imported_roots, top

MODEL = QA / "recovery_model.py"
ALL_QA = sorted(p.stem for p in QA.glob("*.py") if p.name != "__init__.py")

#: model and HTTP SDKs and clients, by import root
SDKS = {"anthropic", "openai", "httpx", "requests", "urllib3", "http", "aiohttp", "socket", "ssl", "boto3",
        "botocore", "google", "vertexai", "litellm", "langchain", "langchain_core", "cohere", "mistralai", "ollama",
        "transformers", "torch", "tiktoken", "websocket", "websockets", "grpc", "playwright", "selenium"}
#: stdlib network entry points ("urllib.parse" is string handling and stays allowed)
NETWORK_TARGETS = ("urllib.request", "urllib.error", "urllib.response", "urllib.robotparser")
#: harness core modules that talk to a model, a network or a process
CORE_NETWORK = {"llm", "nim", "egress_proxy", "spend_proxy", "openshell", "sandbox", "loop", "agent", "netpolicy"}


def test_recovery_model_imports_only_the_standard_library_and_a_few_neutral_modules():
    assert top(qa_internal_imports("recovery_model.py")) <= {"canon", "errors", "ports", "recovery", "recovery_types", "records"}
    roots = {r.split(".")[0] for r in imported_roots(MODEL.read_text(encoding="utf-8"))}
    assert roots <= STDLIB | {"agent_harness"}, roots
    assert roots - {"agent_harness"} <= {"__future__", "dataclasses", "hashlib", "json", "re", "typing", "unicodedata"}, roots
    assert violations(MODEL.read_text(encoding="utf-8")) == []


def test_no_qa_module_imports_a_model_or_http_sdk_or_the_core_llm_module():
    for name in ALL_QA:
        source = (QA / f"{name}.py").read_text(encoding="utf-8")
        targets = {t for _, t in imports_of(source)}
        roots = {t.split(".")[0] for t in targets}
        assert not roots & SDKS, (name, roots & SDKS)
        assert not [t for t in targets if t.startswith(NETWORK_TARGETS)], (name, targets)
        core = {t.split(".")[1] for t in targets if t.startswith("agent_harness.") and t.count(".") >= 1}
        assert not core & CORE_NETWORK, (name, core & CORE_NETWORK)


def test_recovery_model_cannot_reach_the_verifier_the_classifier_the_spec_store_the_policy_the_runner_or_an_adapter():
    reach = closure("recovery_model")
    assert not reach & {"verifier", "classify", "specstore", "policy", "runner", "hooks", "receipts", "cleanup", "checkpoint",
                        "limits", "runstate", "run_errors", "clock"}, reach
    assert reach <= {"canon", "errors", "idempotency", "ports", "recovery", "recovery_types", "records", "schemas"}, reach


def test_only_the_runner_imports_recovery_model():
    users = {name for name in ALL_QA if name != "recovery_model" and "recovery_model" in top(qa_internal_imports(f"{name}.py"))}
    assert users == {"runner"}


def test_nothing_the_verdict_depends_on_imports_recovery_model():
    for name in ("verifier", "classify", "specstore", "records", "canon", "errors", "ports", "recovery_types", "idempotency",
                 "policy", "receipts", "limits", "runstate", "recovery"):
        assert "recovery_model" not in closure(name), name


def test_recovery_model_reads_no_clock_randomness_environment_files_or_network():
    source = MODEL.read_text(encoding="utf-8")
    roots = imported_roots(source)
    assert not {r for r in roots if r in FORBIDDEN_FOR_DETERMINISM or r.split(".")[0] in FORBIDDEN_FOR_DETERMINISM}, roots
    tree = ast.parse(source)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"open", "exec", "eval", "compile", "__import__", "input", "print", "hash", "id"}, called
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"now", "urlopen", "getenv", "environ", "random", "uuid4", "act", "observe", "assert_", "reset",
                        "open_session", "close_session", "capture", "run_check", "get_assertion_set", "verify_assertion",
                        "now_utc"}, attrs
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]


def test_the_only_clock_use_is_the_injected_clocks_monotonic_reading():
    tree = ast.parse(MODEL.read_text(encoding="utf-8"))
    clock_calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "_clock"}
    assert clock_calls == {"monotonic_ms"}


def test_the_model_client_port_is_defined_here_and_carries_no_sdk_types():
    from agent_harness.qa import recovery_model as RM
    from dataclasses import fields
    assert RM.CandidateModelClient.__module__ == "agent_harness.qa.recovery_model"
    for cls in (RM.CandidateModelRequest, RM.CandidateModelResponse, RM.ModelUsage, RM.SnapshotNode):
        assert all(f.type in ("str", "int", "Optional[str]", "Optional[ModelUsage]") for f in fields(cls)), cls


def test_the_test_infrastructure_for_h4_does_no_io_either():
    from pathlib import Path
    tests = Path(__file__).parent
    for name in ("_h4.py", "test_h4_recoverer.py", "test_h4_runner.py", "test_h4_injection.py", "test_h4_request.py"):
        roots = {r.split(".")[0] for r in imported_roots((tests / name).read_text(encoding="utf-8"))}
        assert not roots & (SDKS | {"subprocess", "socket", "os"}), (name, roots)


def test_a_scanner_for_the_model_boundary_can_fail():
    bad = "import anthropic\nfrom agent_harness.llm import chat\nimport urllib.request\n"
    targets = {t for _, t in imports_of(bad)}
    assert {t.split(".")[0] for t in targets} & SDKS == {"anthropic"} and [t for t in targets if t.startswith(NETWORK_TARGETS)]
    assert {t.split(".")[1] for t in targets if t.startswith("agent_harness.")} & CORE_NETWORK == {"llm"}
