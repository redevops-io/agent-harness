import json

import pytest

from agent_harness.h2 import (ArtifactError, HarnessSpec, lineage, load_artifact, render_codex_agents_md,
                              save_artifact, stock_spec, tree_hash, validate)


def rules(spec):
    return {i.rule for i in validate(spec)}


def good():
    s = stock_spec()
    s.set("systemprompt.md", "Inventory inputs, then verify outputs.\n")
    s.set("skills/reconcile/SKILL.md", "---\nname: reconcile\ndescription: Use when numbers must match\n---\nRun scripts/check.py\n")
    s.set("scripts/check.py", "print('ok')\n")
    return s


def test_stock_and_good_trees_are_valid():
    assert validate(stock_spec()) == []
    assert validate(good()) == []


def test_skill_without_trigger_rejected():
    s = good()
    s.set("skills/reconcile/SKILL.md", "Just some notes.\n")
    assert "skill-trigger" in rules(s)


def test_skill_directory_without_skill_file():
    s = good()
    s.set("skills/orphan/helper.md", "x")
    assert "skill-missing-file" in rules(s)


@pytest.mark.parametrize("text", ["hello {{name}}", "{% if x %}", "a }} b"])
def test_template_braces_in_always_on_rejected(text):
    s = good()
    s.set("guardrails.md", text)
    assert "template-braces" in rules(s)


def test_template_braces_allowed_outside_always_on():
    s = good()
    s.set("skills/reconcile/notes.md", "{{ fine here }}")
    assert "template-braces" not in rules(s)


def test_missing_script_reference_rejected():
    s = good()
    s.set("LongTermMEMORY.md", "Use scripts/nope.py before delivering.")
    assert "script-missing" in rules(s)


def test_script_syntax_error_rejected():
    s = good()
    s.set("scripts/check.py", "def broken(:\n")
    assert "script-compile" in rules(s)


def test_plugin_unbalanced_rejected():
    s = good()
    s.set("plugin/guard.ts", "export default () => { return (1;\n")
    assert "plugin-structure" in rules(s)


@pytest.mark.parametrize("name", ["scripts/RUBRICS.json", "scripts/score.json", "skills/x/verdict.md"])
def test_leakage_names_rejected(name):
    s = good()
    s.set(name, "{}")
    assert "leakage" in rules(s)


def test_unknown_path_rejected():
    s = good()
    s.set("notes/random.md", "x")
    assert "unknown-path" in rules(s)


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.md", "a/../../x"])
def test_unsafe_paths_cannot_be_set(bad):
    with pytest.raises(ValueError):
        HarnessSpec().set(bad, "x")


def test_codex_agents_md_contains_only_always_on_text():
    s = good()
    out = render_codex_agents_md(s)
    assert "Inventory inputs" in out and "reconcile" not in out


def test_artifact_roundtrip_dedup_and_lineage(tmp_path):
    a = save_artifact(tmp_path, stock_spec(), meta={"note": "base"})
    assert a == tree_hash(stock_spec())
    assert save_artifact(tmp_path, stock_spec()) == a  # identical tree -> same artifact
    b = save_artifact(tmp_path, good(), parent=a)
    assert lineage(tmp_path, b) == [b, a]
    assert load_artifact(tmp_path, b).files == good().files
    assert json.loads((tmp_path / b / "manifest.json").read_text())["parent"] == a


def test_artifact_rejects_invalid_tree_and_unknown_parent(tmp_path):
    bad = good()
    bad.set("skills/reconcile/SKILL.md", "no trigger")
    with pytest.raises(ArtifactError):
        save_artifact(tmp_path, bad)
    with pytest.raises(ArtifactError):
        save_artifact(tmp_path, good(), parent="deadbeef")
    assert list(tmp_path.iterdir()) == []  # nothing half-written


def test_artifact_tamper_detected(tmp_path):
    a = save_artifact(tmp_path, good())
    (tmp_path / a / "tree" / "systemprompt.md").write_text("tampered")
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path, a)
