"""Harness² editable surfaces: which files an agent harness may contain, per executor.

Mirrors ``SurfaceSpec`` / ``OPENCODE_SURFACE`` / ``CODEX_SURFACE`` from google-research/harness2 at commit
1cfe254e8d25b15f37048c3730134686abe60335 (Apache-2.0; see docs/harness2.md). It is data copied for
validation only; ``tests/test_h2_conformance.py`` compares verdicts with upstream when it is installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Pattern, Tuple

UPSTREAM_REPO = "https://github.com/google-research/harness2"
UPSTREAM_COMMIT = "1cfe254e8d25b15f37048c3730134686abe60335"

_NAME = r"[a-z0-9][a-z0-9_\-]*"


@dataclass(frozen=True)
class Surface:
    name: str
    always_on: Tuple[str, ...]
    always_on_max: int
    skill_dir: str
    skill_name_rule: str  # "must_match" (OpenCode) or "overrides_dir" (Codex)
    subagent_requires_mode: bool
    plugin_dir: Optional[str]
    hooks_file: Optional[str]
    hook_events: FrozenSet[str] = frozenset()
    compile_skill_scripts: bool = False
    reject_empty_always_on: bool = False


OPENCODE = Surface(
    name="opencode",
    always_on=("systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md"),
    always_on_max=24_000,
    skill_dir="skills",
    skill_name_rule="must_match",
    subagent_requires_mode=True,
    plugin_dir="plugin",
    hooks_file=None,
)

CODEX = Surface(
    name="codex",
    always_on=("AGENTS.md",),
    always_on_max=8_000,
    skill_dir=".agents/skills",
    skill_name_rule="overrides_dir",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=".codex/hooks.json",
    hook_events=frozenset({
        "PreToolUse", "PostToolUse", "PermissionRequest", "PreCompact", "PostCompact",
        "SessionStart", "SubagentStart", "SubagentStop", "UserPromptSubmit", "Stop",
    }),
    compile_skill_scripts=True,
    reject_empty_always_on=True,
)

SURFACES: Dict[str, Surface] = {s.name: s for s in (OPENCODE, CODEX)}

# Human-readable map of the paper's eight components (Table 5) to files, per surface.
COMPONENT_FILES = {
    "C1 system prompt": {"opencode": "systemprompt.md", "codex": "AGENTS.md (shared)"},
    "C2 guardrails": {"opencode": "guardrails.md", "codex": "AGENTS.md (shared)"},
    "C3 memory": {"opencode": "LongTermMEMORY.md", "codex": "AGENTS.md (shared)"},
    "C4 tool guidance": {"opencode": "tool_descriptions/tool_guidance.md", "codex": "AGENTS.md (shared)"},
    "C5 subagents": {"opencode": "sub_agents/<name>/AGENT.md", "codex": "sub_agents/<name>/AGENT.md"},
    "C6 skills": {"opencode": "skills/<name>/SKILL.md", "codex": ".agents/skills/<name>/SKILL.md"},
    "C7 scripts": {"opencode": "scripts/<name>.py", "codex": "scripts/<name>.py"},
    "C8 plugins": {"opencode": "plugin/<name>.ts", "codex": ".codex/hooks.json"},
}


@lru_cache(maxsize=None)
def patterns(surface: Surface) -> Dict[str, Pattern[str]]:
    sd = re.escape(surface.skill_dir)
    out = {
        "skill": re.compile(rf"{sd}/({_NAME})/SKILL\.md"),
        "skill_script": re.compile(rf"{sd}/{_NAME}/scripts/[A-Za-z0-9_.\-]+"),
        "subagent": re.compile(rf"sub_agents/({_NAME})/AGENT\.md"),
        "script": re.compile(r"scripts/[a-z0-9_][a-z0-9_]*\.py"),
        "script_ref": re.compile(rf"(?:python3?\s+)((?:{sd}/[\w\-]+/)?scripts/[\w.\-]+\.py)"),
    }
    if surface.plugin_dir is not None:
        out["plugin"] = re.compile(rf"{re.escape(surface.plugin_dir)}/({_NAME})\.ts")
    return out


def normalize(path: str) -> str:
    """Safe relative POSIX path; ValueError for absolute paths or traversal."""
    r = str(path).strip().replace("\\", "/")
    if r.startswith("/") or ".." in r.split("/"):
        raise ValueError(f"path escapes the harness: {path!r}")
    while r.startswith("./"):
        r = r[2:]
    r = r.strip("/")
    if not r:
        raise ValueError("empty harness path")
    return r


def is_allowed(surface: Surface, path: str) -> bool:
    """Whether ``path`` is on the editable surface. Never raises."""
    try:
        rel = normalize(path)
    except ValueError:
        return False
    if rel in surface.always_on or (surface.hooks_file is not None and rel == surface.hooks_file):
        return True
    p = patterns(surface)
    if any(p[k].fullmatch(rel) for k in ("skill", "skill_script", "subagent", "script")):
        return True
    return "plugin" in p and bool(p["plugin"].fullmatch(rel))


@dataclass
class HarnessSpec:
    """An in-memory harness tree for one surface: relative POSIX path -> text."""

    surface: Surface
    files: Dict[str, str] = field(default_factory=dict)

    def set(self, path: str, content: str) -> None:
        self.files[normalize(path)] = content

    def get(self, path: str, default: str = "") -> str:
        return self.files.get(path, default)

    def always_on_text(self) -> str:
        return "\n\n".join(self.files.get(a, "") for a in self.surface.always_on)

    def skills(self) -> List[str]:
        rx = patterns(self.surface)["skill"]
        return sorted({m.group(1) for p in self.files for m in [rx.fullmatch(p)] if m})

    def subagents(self) -> List[str]:
        rx = patterns(self.surface)["subagent"]
        return sorted({m.group(1) for p in self.files for m in [rx.fullmatch(p)] if m})


def stock_spec(surface: Surface) -> HarnessSpec:
    """The base tree: empty always-on files. (Upstream also seeds README stubs; they are not part of the tree.)"""
    s = HarnessSpec(surface)
    for a in surface.always_on:
        s.set(a, "")
    return s
