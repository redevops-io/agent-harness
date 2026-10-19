"""Compare agent_harness.h2 verdicts with upstream harness2's own validator, when upstream is installed.

Skipped unless ``harness2`` (google-research/harness2, Python >= 3.12) is importable. Our extra checks
(leakage names, path safety, plugin balance, template braces) are excluded from the comparison; the
shared rules must agree on accept/reject for every corpus case.
"""
import pytest

from agent_harness.h2 import CODEX, OPENCODE, HarnessSpec, validate

up_tree = pytest.importorskip("harness2.core.tree")
up_oc = pytest.importorskip("harness2.substrate.opencode")
up_cx = pytest.importorskip("harness2.substrate.codex")

OURS_ONLY = {"leakage", "path", "plugin-structure", "template-braces"}
UP = {"opencode": up_oc.OPENCODE_SURFACE, "codex": up_cx.CODEX_SURFACE}

SKILL_OK = "---\nname: s1\ndescription: d\n---\nbody\n"
CORPUS = {
    "opencode": [
        {"guardrails.md": "load the `s1` skill", "skills/s1/SKILL.md": SKILL_OK},
        {"guardrails.md": "no mention", "skills/s1/SKILL.md": SKILL_OK},
        {"skills/s1/SKILL.md": "---\ndescription: d\n---\n", "guardrails.md": "s1"},
        {"skills/s1/SKILL.md": "---\nname: s1\n---\n", "guardrails.md": "s1"},
        {"scripts/a.py": "x=1", "systemprompt.md": "python3 scripts/a.py"},
        {"systemprompt.md": "python3 scripts/missing.py"},
        {"scripts/a.py": "def (:"},
        {"sub_agents/h/AGENT.md": "---\ndescription: d\n---\n"},
        {"sub_agents/h/AGENT.md": "---\nmode: subagent\ndescription: d\n---\n"},
        {"systemprompt.md": "x" * 24_001},
        {},
    ],
    "codex": [
        {"AGENTS.md": "read .agents/skills/s1/SKILL.md", ".agents/skills/s1/SKILL.md": "---\ndescription: d\n---\n"},
        {"AGENTS.md": "nothing", ".agents/skills/s1/SKILL.md": "---\ndescription: d\n---\n"},
        {"AGENTS.md": "s1", ".agents/skills/s1/SKILL.md": "---\nname: z\ndescription: d\n---\n"},
        {"AGENTS.md": ""},
        {"AGENTS.md": "x" * 8_001},
        {"AGENTS.md": "ok", ".codex/hooks.json": "not json"},
        {"AGENTS.md": "ok", ".codex/hooks.json": '{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}}'},
        {"AGENTS.md": "ok", ".agents/skills/s1/scripts/t.py": "def (:"},
        {"AGENTS.md": "ok", "sub_agents/h/AGENT.md": "---\ndescription: d\n---\n"},
    ],
}


@pytest.mark.parametrize("surface,files", [(n, f) for n, fs in CORPUS.items() for f in fs])
def test_same_accept_reject_as_upstream(surface, files):
    ours_surface = {"opencode": OPENCODE, "codex": CODEX}[surface]
    ours = {i.rule for i in validate(HarnessSpec(ours_surface, dict(files)))} - OURS_ONLY
    try:
        up_tree.validate(UP[surface], dict(files))
        upstream_ok = True
    except ValueError:
        upstream_ok = False
    assert (not ours) == upstream_ok, f"ours={sorted(ours)} upstream_ok={upstream_ok}"
