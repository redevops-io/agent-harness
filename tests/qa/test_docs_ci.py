# SPDX-License-Identifier: AGPL-3.0-or-later
"""Documentation, compatibility statement, CI workflow and release-preparation checks.

These tests keep the documents tied to the files they describe (a module map that lists every QA module, recorded
hashes that match the golden data, loop statements that match ``loop.py``, line numbers that match the source) and
check the properties the QA workflow and the release checklist rely on. They read files only; nothing here changes
behaviour."""
import hashlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent_harness.qa import schemas

from . import ci_gate
from ._helpers import CANON_VECTOR_COUNT, CONTRACTS_QA_V1_COMMIT, GOLDEN, GOLDEN_SHA256

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
WORKFLOWS = ROOT / ".github" / "workflows"
QA_PKG = ROOT / "agent_harness" / "qa"



def read(path):
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------
# docs/qa.md
# ---------------------------------------------------------------------------------------------------------------

def test_the_qa_doc_exists_and_names_the_python_floor_and_the_empty_extra():
    text = read(DOCS / "qa.md")
    assert "Python 3.11 or newer" in text
    assert "The `qa` extra is empty" in text
    assert "requires Python 3.11 or newer" in text            # the guard message the doc quotes


@pytest.mark.parametrize("path", sorted(p for p in QA_PKG.glob("*.py") if p.name != "__init__.py"),
                         ids=lambda p: p.name)
def test_the_module_map_lists_every_qa_module(path):
    rows = re.findall(r"^\| [A-Z][A-Za-z ]+ \| `([a-z_]+\.py)` \|", read(DOCS / "qa.md"), flags=re.M)
    assert path.name in rows, f"the docs/qa.md module map does not list {path.name}"


def test_the_module_map_lists_only_modules_that_exist():
    text = read(DOCS / "qa.md")
    mapped = set(re.findall(r"^\| [A-Z][A-Za-z ]+ \| `([a-z_]+\.py)` \|", text, flags=re.M))
    assert mapped, "no module map rows found"
    existing = {p.name for p in QA_PKG.glob("*.py")}
    assert mapped <= existing, sorted(mapped - existing)


def test_the_documented_ports_exist_in_the_code():
    from agent_harness.qa import hooks, ports, specstore
    from agent_harness.qa import egress, recovery_model
    text = read(DOCS / "qa.md")
    for module, names in ((ports, ("NeutralQAAdapter", "NeutralCheckAdapter", "Redactor", "BlobSink", "Recoverer",
                                   "ActionPolicy", "Clock")),
                          (hooks, ("EvidenceStream", "PostconditionChecker", "PrepareChecks", "RecoveryHook")),
                          (specstore, ("SpecStore", "ProgramRegistry")),
                          (egress, ("Resolver", "NoResolver")),
                          (recovery_model, ("CandidateModelClient",))):
        for name in names:
            assert hasattr(module, name), (module.__name__, name)
            assert name in text, f"docs/qa.md does not mention {name}"


def test_the_documented_recovery_modes_are_the_real_enum():
    from agent_harness.qa.records import RecoveryStrategy
    text = read(DOCS / "qa.md")
    for member in RecoveryStrategy:
        if member is RecoveryStrategy.NONE:          # "no recovery", not a mode
            continue
        assert f"`{member.value}`" in text, member.value


def test_the_documented_identifiers_match_the_code():
    text = read(DOCS / "qa.md")
    assert f"`{schemas.NEUTRAL_API}`" in text
    assert f"`{schemas.CONTRACT_ID}`" in text
    assert f"`{schemas.HASH_ALGORITHM}`" in text and f"`{schemas.CONTRACT_HASH_ALGORITHM}`" in text
    assert f"schema version `{schemas.SCHEMA_VERSION}`" in text


def test_the_compatibility_statement_quotes_the_golden_hashes_and_they_match_the_files():
    text = read(DOCS / "qa.md")
    assert CONTRACTS_QA_V1_COMMIT in text and "Contracts commit mirrored" in text
    assert f"{CANON_VECTOR_COUNT} vectors" in text
    for name, digest in GOLDEN_SHA256.items():
        assert digest in text, f"docs/qa.md does not quote the sha256 of {name}"
        assert hashlib.sha256((GOLDEN / name).read_bytes()).hexdigest() == digest, f"{name} no longer matches the golden vectors"


def test_the_compatibility_statement_names_the_release_critical_gate():
    text = read(DOCS / "qa.md")
    for needle in ("compat-receipt/1", "release-critical", "never skips", "zero skipped", "stale",
                   "independent mirror", "canon-v1:<64 hex>", "rcv1:<64 hex>"):
        assert needle in text, needle
    assert "No receipt is produced by this repository" in text


def test_the_licence_statement_is_present_and_the_licence_files_say_agpl():
    text = read(DOCS / "qa.md")
    assert "remains plain AGPL-3.0-or-later" in text
    assert "imports no\nCommons-Clause-licensed code" in text or "imports no Commons-Clause-licensed code" in text
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in read(ROOT / "LICENSE")
    assert 'license = "AGPL-3.0-or-later"' in read(ROOT / "pyproject.toml")


def test_the_doc_says_what_the_module_does_not_do():
    text = read(DOCS / "qa.md")
    for needle in ("No durable scheduling", "No model calls of its own", "No network", "No browser"):
        assert needle in text, needle


# ---------------------------------------------------------------------------------------------------------------
# README, STATUS, architecture: corrections checked against the files at this tree
# ---------------------------------------------------------------------------------------------------------------

LOOP = read(ROOT / "agent_harness" / "loop.py")
AGENT = read(ROOT / "agent_harness" / "agent.py")
STATEMENT_DOCS = ("README.md", "STATUS.md", "docs/architecture.md", "docs/qa.md")


def test_the_loop_statements_match_the_code():
    assert "def run_loop(" in LOOP and "max_turns" in LOOP and "while True:" in LOOP      # multi-turn
    assert "# minimal loop: just finish" in AGENT                                           # Agent.run is the stub
    for rel in ("README.md", "STATUS.md", "docs/architecture.md"):
        text = read(ROOT / rel)
        assert "run_loop" in text, rel
        assert not re.search(r"(shipped|agent) loop is a minimal single-step stub", text), rel
        assert "single-step agent loop" not in text, rel


def test_the_docs_do_not_describe_files_that_are_absent_at_this_tree():
    assert not (ROOT / "agent_harness" / "netpolicy.py").exists()
    assert not (ROOT / "agent_harness" / "supervisor.py").exists()
    assert not (DOCS / "execution-supervision.md").exists()
    for rel in STATEMENT_DOCS:
        text = read(ROOT / rel).lower()
        for absent in ("netpolicy", "supervisor", "execution-supervision"):
            assert absent not in text, f"{rel} mentions {absent}, which does not exist at this tree"


def test_the_readme_status_and_architecture_point_at_the_qa_doc():
    for rel in ("README.md", "STATUS.md", "docs/architecture.md"):
        assert "qa.md" in read(ROOT / rel), rel
    assert "QA_RELEASE_CHECKLIST.md" in read(ROOT / "README.md")


def test_relative_markdown_links_resolve():
    """The same check as .github/workflows/docs-links.yml, over the files this PR touches."""
    bad = []
    for md in [ROOT / "README.md", ROOT / "STATUS.md", *sorted(DOCS.glob("*.md"))]:
        text = re.sub(r"```.*?```", "", read(md), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            path = target.split("#")[0]
            if path and not (md.parent / path).exists():
                bad.append(f"{md.relative_to(ROOT)}: {target}")
    assert bad == []


def test_in_page_anchors_into_the_qa_doc_exist():
    def slug(heading):
        return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).strip().replace(" ", "-")
    anchors = {slug(h) for h in re.findall(r"^#+ (.+)$", read(DOCS / "qa.md"), flags=re.M)}
    for md in [ROOT / "README.md", ROOT / "STATUS.md", *sorted(DOCS.glob("*.md"))]:
        for target in re.findall(r"\]\(qa\.md#([^)\s]+)\)|\]\(docs/qa\.md#([^)\s]+)\)", read(md)):
            anchor = target[0] or target[1]
            assert anchor in anchors, (md.name, anchor)


# ---------------------------------------------------------------------------------------------------------------
# CHANGELOG and the release checklist: preparation only
# ---------------------------------------------------------------------------------------------------------------

def test_the_changelog_has_the_dated_release_entry():
    text = read(DOCS / "CHANGELOG.md")
    assert re.search(r"^## \[0\.2\.0\] - 2026-10-31$", text, flags=re.M)
    assert "(tag `v0.2.0`; the previous tag is `v0.1.0`)" in text
    assert not re.search(r"^## \[Unreleased\]$", text, flags=re.M)


def test_the_checklist_records_the_release_and_that_the_tag_was_authorised():
    checklist = read(DOCS / "QA_RELEASE_CHECKLIST.md")
    assert "released as `v0.2.0` on 2026-10-31" in checklist
    assert "The tag `v0.2.0` was authorised by the maintainer on 2026-10-31" in checklist
    assert "preparation only" not in checklist


def test_the_checklist_carries_every_required_precondition():
    text = read(DOCS / "QA_RELEASE_CHECKLIST.md")
    for needle in ("Maintainer approval", "compat-receipt/1", "exact pins", "3.10, 3.11 and 3.12", "Boundary tests green",
                   "No-secrets scan", "6b8ba3a", "Unrelated commits on\n`main` are not included", "netpolicy.py", "bench/",
                   "stale", "`result = PASS`", "skipped_tests = 0", "Version bump at release time only"):
        assert needle in text, needle


PUBLIC_FILES = [ROOT / "README.md", ROOT / "STATUS.md", DOCS / "architecture.md", DOCS / "qa.md",
                DOCS / "QA_RELEASE_CHECKLIST.md", DOCS / "CHANGELOG.md", WORKFLOWS / "qa.yml",
                Path(ci_gate.__file__), ci_gate.DEFAULT_ALLOWLIST]

# ---------------------------------------------------------------------------------------------------------------
# Public wording. The published code, tests, documents and workflow describe what the module does and why; they
# carry no project-management vocabulary (stage or plan labels, references to design documents, programme names).
# The words below are matched by patterns, and the self-check samples are written in pieces, so that this file does
# not itself contain what it forbids.
# ---------------------------------------------------------------------------------------------------------------

#: the name of the product package that embeds this module, written in pieces
_PKG = "re" "devops-qa"

#: (label, pattern). A label in SHARED_DOC_SKIPS is not applied to README, STATUS and architecture, which describe the
#: whole harness (its evaluation runner and its approval log are unrelated to this module).
FORBIDDEN = [
    ("stage label", re.compile(r"(?<![A-Za-z0-9])[Hh][0-6](?![A-Za-z0-9])")),
    ("plan label", re.compile(r"\bplans? [Hh]\d|\bplan gates?\b|\bPR [A-Z]\d\b|\bthe pl" r"an(?:'s)? "
                              r"(?:proposals?|says|requires|item)|\bper the pl" r"an\b", re.IGNORECASE)),
    ("design-document reference", re.compile(r"\bADR\b|\bAdden" r"dum\b|\bA\.\d+(?:\.\d+)?\b|\bsection \d|\u00a7|"
                                             r"\b(?:is|of|store,) Q\d\b|\bgates? G\d\b|\(G\d\)|\bthreat T\d\b")),
    ("review reference", re.compile(r"\baudit\b(?![ -](?:log|trail))", re.IGNORECASE)),
    ("frozen-input record", re.compile(r"\bIF-?1\b|FRE" r"EZE_|(?i:\bfre" r"eze record|\bthe fre" r"eze\b|\bfre" r"eze data)")),
    ("programme name", re.compile(r"\bTrack [A-Z]\b|\bSAB\b|\bArm [A-D]\b|\bbench" r"mark(?!ing)|\bheld-?out\b|"
                             r"\bground[ -]truth\b|\bmile" r"stone", re.IGNORECASE)),
    ("role label", re.compile(r"\bintegra" r"tors?\b|\bowners?\b|\bowner-|\bapproved" r" by\b|\bhand" r"off\b",
                                       re.IGNORECASE)),
    ("decision label", re.compile(r"\bD\d{1,2}\b")),
    ("product package", re.compile(re.escape(_PKG) + r"|RE" r"DEVOPS|RD" r"QA")),
]
SHARED_DOC_SKIPS = {"programme name", "review reference"}
SHARED_DOCS = ("README.md", "STATUS.md", "docs/architecture.md")
#: technical uses that stay: the core module `agent_harness.h2`, the import name of the product package, the product
#: package named as such (and as a quoted dependency name), and private keys and address ranges.
ALLOWED = [
    re.compile(r"agent_harness\.h" r"2\b"),
    re.compile(re.escape(_PKG.replace("-", "_"))),
    re.compile("`" + re.escape(_PKG) + "` product package"),
    re.compile('"' + re.escape(_PKG) + '"'),
]
#: words that say how visible a repository or package is; they never appear in a public document
PUBLIC_DOC_ONLY = re.compile(r"\bprivate\b(?! (?:keys?|address(?:es)?|ranges?)\b)|\bsealed bench" r"mark|"
                             r"source-available|\bvisibility\b", re.IGNORECASE)


def scan_text(text, *, shared=False, public_doc=False):
    """-> sorted ``(label, matched text)`` for every forbidden word in ``text``."""
    for allowed in ALLOWED:
        text = allowed.sub("", text)
    found = set()
    for label, pattern in FORBIDDEN:
        if shared and label in SHARED_DOC_SKIPS:
            continue
        found.update((label, m.group(0)) for m in pattern.finditer(text))
    if public_doc:
        found.update(("visibility", m.group(0)) for m in PUBLIC_DOC_ONLY.finditer(text))
    return sorted(found)


def published_files():
    """Every file the wording rules cover: the QA module, its tests (not the golden data), the QA documents, the
    pointers in README, STATUS and architecture, the workflow and pyproject.toml."""
    tests = ROOT / "tests" / "qa"
    files = sorted(QA_PKG.rglob("*.py"))
    files += sorted(p for p in tests.rglob("*")
                    if p.is_file() and "golden" not in p.relative_to(tests).parts and "__pycache__" not in p.parts
                    and p.suffix in (".py", ".txt"))
    files += [DOCS / "qa.md", DOCS / "QA_RELEASE_CHECKLIST.md", DOCS / "CHANGELOG.md", ROOT / "README.md",
              ROOT / "STATUS.md", DOCS / "architecture.md", WORKFLOWS / "qa.yml", ROOT / "pyproject.toml"]
    return files


def test_the_published_files_carry_no_internal_vocabulary():
    problems = []
    for path in published_files():
        rel = path.relative_to(ROOT).as_posix()
        for label, word in scan_text(read(path), shared=rel in SHARED_DOCS, public_doc=path in PUBLIC_FILES):
            problems.append(f"{rel}: {label}: {word!r}")
    assert problems == []


def test_the_wording_scan_covers_the_published_files_and_not_the_golden_data():
    files = {p.relative_to(ROOT).as_posix() for p in published_files()}
    for expected in ("agent_harness/qa/runner.py", "agent_harness/qa/egress.py", "tests/qa/_helpers.py",
                     "tests/qa/conftest.py", "tests/qa/ci_gate.py", "tests/qa/test_docs_ci.py", "docs/qa.md",
                     "docs/QA_RELEASE_CHECKLIST.md", "README.md", ".github/workflows/qa.yml", "pyproject.toml"):
        assert expected in files, expected
    assert not any("golden" in f for f in files)
    assert len(files) > 80
    assert {p.name for p in (ROOT / "tests" / "qa" / "golden").iterdir()} == {
        "qa.inventory.json", "qa.schema.json", "qa_canon_vectors.json", "qa_canon_vectors_addendum.json",
        "qa_clarifications.json"}


def test_the_wording_scan_can_fail():
    samples = {
        "stage label": "added in H" "5",
        "plan label": "the pl" "an's proposals",
        "design-document reference": "see the A" "DR, section" " 7",
        "review reference": "the aud" "it's gate",
        "frozen-input record": "IF" "-1 fre" "eze record",
        "programme name": "the bench" "mark programme",
        "role label": "the integ" "rator publishes",
        "decision label": "decision D" "10",
        "product package": "the " + _PKG + " repository",
    }
    assert {label for label, _ in FORBIDDEN} == set(samples)
    for label, sample in samples.items():
        assert any(found == label for found, _ in scan_text(sample)), (label, sample)
    assert scan_text("Own" "er approval, held" "-out set") != []
    assert scan_text("the private repo" "sitory", public_doc=True) != []
    # a label skipped for the shared documents is still applied everywhere else
    assert scan_text("the bench" "mark programme", shared=True) == []
    assert scan_text("the bench" "mark programme") != []


def test_legitimate_technical_terms_remain_allowed():
    for text in ("`RunSealed` event and a sealed receipt chain, evidence sealed at the end of the run",
                 "PEM private key blocks and private address ranges (RFC 1918), loopback and link-local targets",
                 "frozen assertions, a frozen dataclass and `_freeze_json`",
                 "RFC 2544 benchmarking range 198.18.0.0/15, an audit log and an audit trail",
                 "`agent_harness.h2` (the file-tree spec) and `import redevops_qa` as a forbidden import",
                 "the `" + _PKG + "` product package, and the quoted dependency name \"" + _PKG + "\"",
                 "C0 controls, Q1 and Q2 test variables, 3 refusals, version 0.1"):
        assert scan_text(text, public_doc=True) == [], text


def test_the_documents_do_not_describe_the_visibility_of_the_product_package():
    for path in PUBLIC_FILES:
        text = read(path)
        for line in text.splitlines():
            if _PKG in line:
                assert not re.search(r"private|public|visib|source-available|closed", line, re.IGNORECASE), line
    assert "`" + _PKG + "` product package" in read(DOCS / "qa.md")


@pytest.mark.parametrize("path", PUBLIC_FILES, ids=lambda p: p.name)
def test_the_public_qa_documents_do_not_reveal_internal_details(path):
    rel = path.relative_to(ROOT).as_posix()
    assert scan_text(read(path), shared=rel in SHARED_DOCS, public_doc=True) == [], path.name


def test_in_file_anchors_in_the_qa_doc_exist():
    def slug(heading):
        return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).strip().replace(" ", "-")
    text = read(DOCS / "qa.md")
    anchors = {slug(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.M)}
    linked = re.findall(r"\]\(#([^)\s]+)\)", text)
    assert linked
    for anchor in linked:
        assert anchor in anchors, anchor


def _line_of(path, needle):
    for number, line in enumerate(read(path).splitlines(), 1):
        if needle in line:
            return number
    raise AssertionError((path, needle))


def test_the_checklist_lists_the_lazy_context_runtime_imports_with_the_right_line_numbers():
    text = read(DOCS / "QA_RELEASE_CHECKLIST.md")
    selfcheck = ROOT / "agent_harness" / "bench" / "selfcheck.py"
    jobbench = ROOT / "agent_harness" / "bench" / "jobbench.py"
    assert (f"| `agent_harness/bench/selfcheck.py` | {_line_of(selfcheck, 'import context_runtime  # noqa')} |"
            in text)
    assert (f"| `agent_harness/bench/jobbench.py` | "
            f"{_line_of(jobbench, 'from context_runtime.integrations.agent_harness import ContextRuntimeManager')} |"
            in text)


def test_the_documented_lazy_imports_are_all_the_context_runtime_imports_outside_qa():
    found = []
    for path in sorted((ROOT / "agent_harness").rglob("*.py")):
        if QA_PKG in path.parents:
            continue
        for number, line in enumerate(read(path).splitlines(), 1):
            if re.match(r"\s*(import|from)\s+context_runtime\b", line):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{number}")
    assert found == ["agent_harness/bench/jobbench.py:80", "agent_harness/bench/selfcheck.py:121"], found


# ---------------------------------------------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------------------------------------------

QA_YML = read(WORKFLOWS / "qa.yml")
WRITE_OR_PUBLISH = re.compile(
    r"(contents|packages|id-token|deployments|actions|pull-requests|issues):\s*write|^\s*tags:|"
    r"pypi|twine|gh release|action-gh-release|git tag|git push|npm publish|docker push|\bpublish\b",
    re.IGNORECASE | re.MULTILINE)


def test_the_qa_workflow_runs_the_qa_suite_on_3_11_and_3_12_in_a_strict_environment():
    assert re.search(r'python-version:\s*\["3\.11",\s*"3\.12"\]', QA_YML)
    assert re.search(r'env -i HOME="\$\(mktemp -d\)"[^\n]* QA_STRICT_ENV=1', QA_YML)
    assert "tests/qa/ci_gate.py" in QA_YML
    assert 'pip install -e ".[dev,qa]"' in QA_YML
    assert "contents: read" in QA_YML
    assert "fail-fast: false" in QA_YML


def test_the_qa_workflow_keeps_a_python_3_10_job_for_the_core_import_and_the_guard():
    job = QA_YML.split("\n  python-3-10:\n")[1]
    assert 'python-version: "3.10"' in job
    assert "import agent_harness" in job and "import agent_harness.qa" in job
    assert "requires Python 3.11 or newer" in job
    assert "tests/qa/test_python_floor.py" in job


def test_the_existing_tests_workflow_is_unchanged():
    text = read(WORKFLOWS / "tests.yml")
    assert 'python-version: ["3.10", "3.12"]' in text
    assert 'python -m pip install -e ".[dev]"' in text
    assert text.rstrip().endswith("- run: python -m pytest -q")
    assert "QA_STRICT_ENV" not in text


def without_comments(text):
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_no_workflow_has_a_tag_trigger_a_publish_step_or_a_write_permission():
    paths = sorted(WORKFLOWS.glob("*.yml"))
    assert {p.name for p in paths} >= {"tests.yml", "qa.yml", "docs-links.yml", "secret-scan.yml"}
    for path in paths:
        hits = WRITE_OR_PUBLISH.findall(without_comments(read(path)))
        assert not hits, (path.name, hits)


def test_the_workflow_scan_can_fail():
    assert WRITE_OR_PUBLISH.search("permissions:\n  contents: write\n")
    assert WRITE_OR_PUBLISH.search("on:\n  push:\n    tags: ['v*']\n")
    assert WRITE_OR_PUBLISH.search("      - run: twine upload dist/*\n")


# ---------------------------------------------------------------------------------------------------------------
# The CI gate: it must fail on zero collected tests and on any skip outside the allow-list
# ---------------------------------------------------------------------------------------------------------------

def junit(*cases):
    body = "".join(cases)
    return f'<?xml version="1.0"?><testsuites><testsuite name="pytest">{body}</testsuite></testsuites>'


def case(name, outcome=None, classname="tests.qa.test_x"):
    inner = {"skip": "<skipped/>", "fail": "<failure/>", "error": "<error/>", None: ""}[outcome]
    return f'<testcase classname="{classname}" name="{name}">{inner}</testcase>'


def test_the_gate_accepts_a_clean_run():
    collected, skipped, problems = ci_gate.evaluate(junit(case("a"), case("b")), [])
    assert (collected, skipped, problems) == (2, [], [])


def test_the_gate_fails_when_nothing_was_collected():
    collected, _, problems = ci_gate.evaluate(junit(), [])
    assert collected == 0 and problems == ["no QA tests were collected"]


def test_the_gate_fails_on_a_skip_that_is_not_allow_listed():
    _, skipped, problems = ci_gate.evaluate(junit(case("a"), case("b", "skip")), [])
    assert skipped == ["test_x.py::b"]
    assert problems == ["SKIPPED and not allow-listed: test_x.py::b"]


def test_the_gate_accepts_an_allow_listed_skip_and_only_that_one():
    xml = junit(case("b", "skip"), case("c", "skip", classname="tests.qa.test_y"), case("a"))
    _, _, problems = ci_gate.evaluate(xml, ["test_x.py::b"])
    assert problems == ["SKIPPED and not allow-listed: test_y.py::c"]
    _, _, problems = ci_gate.evaluate(xml, ["test_x.py::*", "test_y.py::c"])
    assert problems == []


def test_the_gate_fails_on_failures_and_errors():
    for outcome in ("fail", "error"):
        _, _, problems = ci_gate.evaluate(junit(case("a", outcome)), [])
        assert problems == ["FAILED: test_x.py::a"]


def test_the_gate_reads_a_parametrized_test_in_a_class():
    xml = junit(case("test_it[a-1]", "skip", classname="tests.qa.test_x.TestK"))
    _, skipped, _ = ci_gate.evaluate(xml, [])
    assert skipped == ["test_x.py::test_it[a-1]"]


def test_allow_list_parsing_ignores_comments_and_requires_a_reason():
    text = "# why\ntest_x.py::a\n\n# also why\n  test_y.py::*  \n"
    assert ci_gate.read_allowlist(text) == ["test_x.py::a", "test_y.py::*"]
    assert ci_gate.unexplained_entries(text) == []
    assert ci_gate.unexplained_entries("test_x.py::a\n") == ["test_x.py::a"]
    assert ci_gate.unexplained_entries("# why\n\ntest_x.py::a\n") == ["test_x.py::a"]


def test_the_shipped_allow_list_is_parseable_documented_and_has_no_unexplained_entry():
    text = read(ci_gate.DEFAULT_ALLOWLIST)
    assert text.lstrip().startswith("#")
    assert "no permitted skip" in text
    assert ci_gate.unexplained_entries(text) == []


def run_gate(tmp_path, source, allowlist=""):
    tests = tmp_path / "suite"
    tests.mkdir()
    if source is not None:
        (tests / "test_sample.py").write_text(textwrap.dedent(source), encoding="utf-8")
    allow = tmp_path / "allow.txt"
    allow.write_text(allowlist, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "QA_STRICT_ENV"}
    return subprocess.run([sys.executable, str(Path(ci_gate.__file__)), "--tests", str(tests), "--allowlist", str(allow),
                           "--rootdir", str(tests), "-c", os.devnull],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)


def test_the_gate_end_to_end_passes_a_clean_suite(tmp_path):
    done = run_gate(tmp_path, "def test_ok():\n    assert True\n")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "collected=1 skipped=0" in done.stdout


def test_the_gate_end_to_end_fails_on_an_empty_suite(tmp_path):
    done = run_gate(tmp_path, None)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "no QA tests were collected" in done.stderr


def test_the_gate_end_to_end_fails_on_a_skip_and_accepts_it_when_allow_listed(tmp_path):
    source = """
        import pytest
        def test_ok():
            pass
        @pytest.mark.skip(reason="x")
        def test_skipped():
            pass
    """
    done = run_gate(tmp_path, source)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "SKIPPED and not allow-listed" in done.stderr
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    done = run_gate(allowed, source, "# the one skip this suite is allowed\ntest_sample.py::test_skipped\n")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "skipped=1" in done.stdout


def test_the_gate_end_to_end_fails_on_a_failing_test(tmp_path):
    done = run_gate(tmp_path, "def test_bad():\n    assert False\n")
    assert done.returncode == 1
    assert "FAILED" in done.stderr
