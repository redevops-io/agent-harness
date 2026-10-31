# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layering and licence boundaries of the egress, redaction, blob, vault and integrity modules. The earlier boundary
tests already scan every file under ``agent_harness/qa`` for forbidden packages (contracts, Context Runtime, the product package,
Playwright), non-standard-library imports, the SPDX header and any network SDK; these add what is specific to the egress and evidence modules: the
policy logic opens no socket and resolves no name, reads no clock or randomness, reaches nothing that decides a verdict,
only the runner imports it, and no digest is cut short anywhere in the module."""
import ast
import re

import pytest

from .test_boundaries import PKG, QA, ROOT, STDLIB, imports_of, qa_internal_imports, violations
from .test_runner_boundaries import FORBIDDEN_FOR_DETERMINISM, closure, imported_roots, top

EVIDENCE_MODULES = ["egress", "redaction", "blobs", "vault", "integrity"]
#: the test and fixture files of the egress, redaction and evidence modules (scanned for secret-shaped strings below)
EVIDENCE_TESTS = ("_fx_evidence", "test_blobs", "test_evidence_boundaries", "test_evidence_determinism", "test_egress_guard",
                  "test_egress_vectors", "test_evidence_integrity", "test_evidence_mutations", "test_redaction",
                  "test_evidence_runner", "test_vault")
ALL_QA = sorted(p.stem for p in QA.glob("*.py") if p.name != "__init__.py")

#: stdlib modules each egress/evidence module may import (anything else fails the test)
ALLOWED_STDLIB = {
    "egress": {"__future__", "enum", "ipaddress", "re", "unicodedata", "dataclasses", "typing", "urllib.parse"},
    "redaction": {"__future__", "base64", "hashlib", "html", "json", "math", "re", "unicodedata", "dataclasses", "typing",
                  "urllib.parse"},
    "blobs": {"__future__", "hashlib", "re", "dataclasses", "types", "typing"},
    "vault": {"__future__", "re", "dataclasses", "typing"},
    "integrity": {"__future__", "dataclasses", "typing"},
}
ALLOWED_INTERNAL = {
    "egress": {"canon", "errors"}, "redaction": {"canon", "errors"}, "blobs": {"canon", "errors", "records"},
    "vault": {"blobs", "canon", "errors", "ports", "records", "redaction"},
    "integrity": {"blobs", "errors", "receipts", "records", "run_errors"},
}
NETWORK = {"socket", "ssl", "http", "urllib.request", "urllib.error", "asyncio", "select", "selectors", "ftplib", "smtplib",
           "telnetlib", "xmlrpc", "socketserver", "email"}


@pytest.mark.parametrize("name", EVIDENCE_MODULES)
def test_each_evidence_module_imports_only_the_listed_standard_library_and_neutral_modules(name):
    source = (QA / f"{name}.py").read_text(encoding="utf-8")
    roots = imported_roots(source)
    assert roots <= ALLOWED_STDLIB[name], roots - ALLOWED_STDLIB[name]
    assert top(qa_internal_imports(f"{name}.py")) <= ALLOWED_INTERNAL[name], top(qa_internal_imports(f"{name}.py"))
    assert violations(source) == []
    core = [t for _, t in imports_of(source) if t.startswith("agent_harness.") and not t.startswith("agent_harness.qa")]
    assert core == []                      # no harness core module at all: not netpolicy, egress_proxy, guardrails or trace


@pytest.mark.parametrize("name", EVIDENCE_MODULES)
def test_evidence_modules_open_no_socket_resolve_no_name_and_read_no_clock_randomness_environment_or_files(name):
    source = (QA / f"{name}.py").read_text(encoding="utf-8")
    roots = imported_roots(source)
    assert not {r for r in roots if r in FORBIDDEN_FOR_DETERMINISM or r.split(".")[0] in FORBIDDEN_FOR_DETERMINISM}, roots
    assert not {r for r in roots if r in NETWORK or r.split(".")[0] in NETWORK}, roots
    tree = ast.parse(source)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"open", "exec", "eval", "compile", "__import__", "input", "print", "hash", "id"}, called
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"now", "urlopen", "getenv", "environ", "random", "uuid4", "getaddrinfo", "gethostbyname", "connect",
                        "sendall", "recv", "bind", "listen", "accept", "create_connection", "system", "popen", "time",
                        "monotonic", "perf_counter", "sleep", "now_utc", "monotonic_ms"}, attrs


def test_the_egress_module_resolves_names_only_through_an_injected_resolver():
    source = (QA / "egress.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    resolver_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "resolve"
                      and isinstance(n.value, ast.Attribute) and n.value.attr == "_resolver"]
    assert len(resolver_calls) == 1                                  # one resolution site, behind the injected object
    assert "NoResolver" in source and "raise LookupError" in source  # the default refuses; there is no default that looks anything up


def test_no_evidence_module_can_reach_anything_that_decides_a_verdict_or_runs_an_adapter():
    for name in EVIDENCE_MODULES:
        reach = closure(name)
        assert not reach & {"verifier", "classify", "specstore", "policy", "runner", "hooks", "clock", "limits", "runstate",
                            "checkpoint", "cleanup", "recovery", "recovery_model", "idempotency"}, (name, reach)
    assert closure("egress") <= {"canon", "errors"}
    assert closure("redaction") <= {"canon", "errors"}
    assert closure("blobs") <= {"canon", "errors", "records", "schemas"}
    assert closure("vault") <= {"blobs", "canon", "errors", "ports", "records", "redaction", "schemas"}
    assert closure("integrity") <= {"blobs", "canon", "errors", "receipts", "records", "run_errors", "schemas"}


def test_only_the_runner_imports_the_evidence_modules_that_it_wires_and_nothing_a_verdict_depends_on_does():
    for name in ("egress", "vault"):
        users = {n for n in ALL_QA if n != name and name in top(qa_internal_imports(f"{n}.py"))}
        assert users == {"runner"}, (name, users)
    assert {n for n in ALL_QA if "integrity" in top(qa_internal_imports(f"{n}.py"))} == set()
    assert {n for n in ALL_QA if "redaction" in top(qa_internal_imports(f"{n}.py"))} == {"vault"}
    assert {n for n in ALL_QA if "blobs" in top(qa_internal_imports(f"{n}.py"))} == {"integrity", "runner", "vault"}
    for name in ("verifier", "classify", "specstore", "records", "canon", "errors", "ports", "recovery_types", "idempotency",
                 "policy", "receipts", "limits", "runstate", "recovery", "recovery_model", "hooks", "checkpoint", "cleanup"):
        assert not closure(name) & set(EVIDENCE_MODULES), name


def test_the_package_init_and_the_core_do_not_import_the_evidence_modules():
    text = (QA / "__init__.py").read_text(encoding="utf-8")
    names = {t.split(".")[2] for _, t in imports_of(text) if t.startswith("agent_harness.qa.") and t.count(".") >= 2}
    assert not names & set(EVIDENCE_MODULES), names
    for path in PKG.rglob("*.py"):
        if QA in path.parents:
            continue
        assert "agent_harness.qa" not in path.read_text(encoding="utf-8"), path


def test_the_runner_names_the_evidence_modules_only_through_its_listed_imports():
    assert {"blobs", "egress", "vault"} <= top(qa_internal_imports("runner.py"))
    assert not {"redaction", "integrity"} & top(qa_internal_imports("runner.py"))


# --- no truncated digest in the module ------------------------------------------------------------------------------

HASH_CALLS = {"hexdigest", "digest_hex", "canon_hash", "digest", "sha256", "digest_of_bytes", "to_contract_ref", "record_hash"}
#: the four places that derive a bounded *identifier* (not an evidence reference or a content hash) from a digest, all from
#: before the egress and evidence modules: an idempotency key, two over-long id fallbacks, and the 128-bit W3C trace id. Nothing new may be added.
KNOWN_SHORT_IDS = {("idempotency.py", "idempotency_key"), ("policy.py", "_bounded_id"), ("receipts.py", "bounded_id"),
                   ("runner.py", "_manifest")}


def truncations(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()

    class V(ast.NodeVisitor):
        def __init__(self):
            self.scope = ["<module>"]

        def visit_FunctionDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_Subscript(self, node):
            sl = node.slice
            if isinstance(sl, ast.Slice) and sl.upper is not None and (sl.lower is None or isinstance(sl.lower, ast.Constant)):
                v = node.value
                name = None
                if isinstance(v, ast.Call):
                    f = v.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                elif isinstance(v, ast.Name):
                    name = v.id
                elif isinstance(v, ast.Attribute):
                    name = v.attr
                if name and (name in HASH_CALLS or re.search(r"digest|hash|sha", name, re.I)):
                    found.add((path.name, self.scope[-1]))
            self.generic_visit(node)
    V().visit(tree)
    return found


def test_no_qa_module_cuts_a_digest_short_except_the_four_known_derived_identifiers():
    found = set()
    for path in sorted(QA.glob("*.py")):
        found |= truncations(path)
    assert found == KNOWN_SHORT_IDS, found ^ KNOWN_SHORT_IDS
    assert not [f for f in found if f[0].removesuffix(".py") in EVIDENCE_MODULES]


def test_the_scanner_would_find_a_truncation():
    sample = QA / "__scratch__"
    tree = ast.parse("import hashlib\ndef f(d):\n    return hashlib.sha256(d).hexdigest()[:16]\n")
    assert any(isinstance(n, ast.Subscript) for n in ast.walk(tree))
    src = "import hashlib\ndef f(d):\n    return hashlib.sha256(d).hexdigest()[:16]\ndef g(digest):\n    return digest[:8]\n"
    path = ROOT / "tests" / "qa" / "__scan_sample__.py"
    try:
        path.write_text(src, encoding="utf-8")
        assert truncations(path) == {("__scan_sample__.py", "f"), ("__scan_sample__.py", "g")}
    finally:
        path.unlink(missing_ok=True)


# --- repository hygiene ---------------------------------------------------------------------------

SECRET_SHAPES = [re.compile(p) for p in (
    r"AKIA[0-9A-Z]{16}", r"ghp_[A-Za-z0-9]{36}", r"xox[abprs]-[0-9]{6,}", r"sk_live_[A-Za-z0-9]{16,}",
    r"AIza[0-9A-Za-z_-]{35}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.",
    r"npm_[A-Za-z0-9]{36}", r"sk-ant-[A-Za-z0-9_-]{20,}")]


def test_no_evidence_file_contains_a_secret_shaped_string():
    files = [QA / f"{n}.py" for n in EVIDENCE_MODULES] + [ROOT / 'tests' / 'qa' / f'{n}.py' for n in EVIDENCE_TESTS]
    assert len(files) >= 10
    for path in files:
        text = path.read_text(encoding="utf-8")
        for shape in SECRET_SHAPES:
            assert not shape.search(text), (path.name, shape.pattern)
