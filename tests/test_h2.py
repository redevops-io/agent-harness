import json

import pytest

from agent_harness.h2 import (CODEX, OPENCODE, ArtifactError, HarnessSpec, lineage, load_artifact,
                              save_artifact, stock_spec, tree_hash, validate)


def rules(spec):
    return {i.rule for i in validate(spec)}


def oc_good():
    s = stock_spec(OPENCODE)
    s.set("guardrails.md", "- When numbers must match, load the `reconcile` skill.\nRun python3 scripts/check.py\n")
    s.set("skills/reconcile/SKILL.md", "---\nname: reconcile\ndescription: Match numbers\n---\nbody\n")
    s.set("scripts/check.py", "print('ok')\n")
    return s


def cx_good():
    s = HarnessSpec(CODEX)
    s.set("AGENTS.md", "Inventory inputs. When numbers must match, read .agents/skills/reconcile/SKILL.md.\n")
    s.set(".agents/skills/reconcile/SKILL.md", "---\ndescription: Match numbers\n---\nbody\n")
    return s


def test_stock_opencode_valid_and_stock_codex_needs_content():
    assert validate(stock_spec(OPENCODE)) == []
    assert "always-on-empty" in rules(stock_spec(CODEX))  # Codex rejects an empty AGENTS.md


def test_good_trees_valid():
    assert validate(oc_good()) == []
    assert validate(cx_good()) == []


def test_unnamed_skill_is_untriggered():
    s = oc_good()
    s.set("guardrails.md", "nothing here")
    assert "skill-trigger" in rules(s)


def test_skill_requires_description():
    s = oc_good()
    s.set("skills/reconcile/SKILL.md", "---\nname: reconcile\n---\n")
    assert "skill-description" in rules(s)


def test_opencode_skill_name_must_match_codex_name_may_be_absent():
    s = oc_good()
    s.set("skills/reconcile/SKILL.md", "---\ndescription: x\n---\n")
    assert "skill-name" in rules(s)
    c = cx_good()  # no name: line is fine on Codex
    assert "skill-name" not in rules(c)
    c.set(".agents/skills/reconcile/SKILL.md", "---\nname: other\ndescription: x\n---\n")
    assert "skill-name" in rules(c)


def test_subagent_mode_required_only_on_opencode():
    for surf, path, ok in ((OPENCODE, "sub_agents/helper/AGENT.md", "---\nmode: subagent\ndescription: d\n---\n"),
                           (CODEX, "sub_agents/helper/AGENT.md", "---\ndescription: d\n---\n")):
        s = oc_good() if surf is OPENCODE else cx_good()
        s.set(path, ok)
        assert validate(s) == []
    s = oc_good()
    s.set("sub_agents/helper/AGENT.md", "---\ndescription: d\n---\n")
    assert "subagent-mode" in rules(s)


def test_always_on_cap_per_surface():
    s = oc_good()
    s.set("systemprompt.md", "x" * 24_001)
    assert "always-on-cap" in rules(s)
    c = cx_good()
    c.set("AGENTS.md", c.get("AGENTS.md") + "x" * 8_000)
    assert "always-on-cap" in rules(c)


@pytest.mark.parametrize("text", ["{{name}}", "{% if x %}", "{# c #}"])
def test_template_braces_rejected_in_always_on_only(text):
    s = oc_good()
    s.set("systemprompt.md", text)
    assert "template-braces" in rules(s)
    s = oc_good()
    s.set("skills/reconcile/notes.md", text)  # off-surface file, flagged for a different reason
    assert "template-braces" not in rules(s)


def test_missing_and_reserved_script_refs():
    s = oc_good()
    s.set("systemprompt.md", "run python3 scripts/nope.py")
    assert "script-missing" in rules(s)
    s.set("systemprompt.md", "run python3 scripts/read_doc.py")  # reserved name, provided by the runner
    assert "script-missing" not in rules(s)


def test_script_syntax_error():
    s = oc_good()
    s.set("scripts/check.py", "def broken(:\n")
    assert "script-compile" in rules(s)


def test_script_names_must_be_lowercase_snake():
    s = oc_good()
    s.set("scripts/Bad-Name.py", "x=1")
    assert "unknown-path" in rules(s)


def test_codex_hooks_schema():
    c = cx_good()
    c.set(".codex/hooks.json", json.dumps({"hooks": {"PostToolUse": [{"matcher": "Bash",
          "hooks": [{"type": "command", "command": "echo ok"}]}]}}))
    assert validate(c) == []
    for bad in ("not json", json.dumps({"hooks": {"Nope": []}}),
                json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "prompt"}]}]}})):
        c.set(".codex/hooks.json", bad)
        assert "hooks" in rules(c)


def test_codex_has_no_plugin_dir_and_opencode_plugin_balance():
    c = cx_good()
    c.set("plugin/x.ts", "export default {}")
    assert "unknown-path" in rules(c)
    s = oc_good()
    s.set("plugin/guard.ts", "export default () => { return (1;\n")
    assert "plugin-structure" in rules(s)


@pytest.mark.parametrize("name", ["scripts/rubric.py", "scripts/score.py", "sub_agents/x/verdict.md"])
def test_leakage_names(name):
    s = oc_good()
    s.set(name, "x=1")
    assert "leakage" in rules(s)


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.md", "a/../../x", ""])
def test_unsafe_paths_cannot_be_set(bad):
    with pytest.raises(ValueError):
        HarnessSpec(OPENCODE).set(bad, "x")


def test_artifact_roundtrip_dedup_lineage_and_surface(tmp_path):
    a = save_artifact(tmp_path, stock_spec(OPENCODE), meta={"note": "base"})
    assert save_artifact(tmp_path, stock_spec(OPENCODE)) == a
    b = save_artifact(tmp_path, oc_good(), parent=a)
    assert lineage(tmp_path, b) == [b, a]
    loaded = load_artifact(tmp_path, b)
    assert loaded.files == oc_good().files and loaded.surface is OPENCODE
    assert json.loads((tmp_path / b / "manifest.json").read_text())["surface"] == "opencode"
    # same bytes on a different surface are a different artifact
    assert tree_hash(HarnessSpec(CODEX, {"AGENTS.md": "x"})) != tree_hash(HarnessSpec(OPENCODE, {"AGENTS.md": "x"}))


def test_artifact_rejects_invalid_tree_and_unknown_parent(tmp_path):
    bad = oc_good()
    bad.set("guardrails.md", "")
    with pytest.raises(ArtifactError):
        save_artifact(tmp_path, bad)
    with pytest.raises(ArtifactError):
        save_artifact(tmp_path, oc_good(), parent="deadbeef")
    assert list(tmp_path.iterdir()) == []


def test_artifact_tamper_detected(tmp_path):
    a = save_artifact(tmp_path, oc_good())
    (tmp_path / a / "tree" / "systemprompt.md").write_text("tampered")
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path, a)
