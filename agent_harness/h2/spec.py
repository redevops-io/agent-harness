"""Harness² file-tree spec: the eight editable components, mounted as files.

Layout follows Table 5 of arXiv 2610.14224v1. It has NOT yet been checked against the pinned upstream
revision of google-research/harness2 (see docs/harness2.md); treat the paths as provisional until it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Tuple

ALWAYS_ON = {
    "C1": "systemprompt.md",
    "C2": "guardrails.md",
    "C3": "LongTermMEMORY.md",
    "C4": "tool_descriptions/tool_guidance.md",
}


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    group: str
    path: str  # file path, or a directory prefix when ``multi`` is true
    activation: str
    multi: bool = False


COMPONENTS: Tuple[Component, ...] = (
    Component("C1", "System prompt", "context", "systemprompt.md", "every call"),
    Component("C2", "Guardrails", "context", "guardrails.md", "every call"),
    Component("C3", "Memory", "context", "LongTermMEMORY.md", "every call"),
    Component("C4", "Tool guidance", "tool-interface", "tool_descriptions/tool_guidance.md", "every call"),
    Component("C5", "Subagents", "orchestration", "sub_agents/", "task tool; own context", multi=True),
    Component("C6", "Skills", "context", "skills/", "skill tool, on invocation", multi=True),
    Component("C7", "Scripts", "tool-interface", "scripts/", "run via bash", multi=True),
    Component("C8", "Plugins", "middleware", "plugin/", "inside the tool loop", multi=True),
)

SKILL_FILE = "SKILL.md"
AGENT_FILE = "AGENT.md"


def component_of(path: str) -> str | None:  # noqa: UP007 (3.9 compatible via __future__)
    """Return the component id a relative path belongs to, or None if it is not part of the spec."""
    for c in COMPONENTS:
        if c.multi and path.startswith(c.path):
            return c.id
        if not c.multi and path == c.path:
            return c.id
    return None


def normalize(path: str) -> str:
    """Return a safe relative POSIX path; raise ValueError for absolute paths or traversal."""
    p = PurePosixPath(path.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"unsafe harness path: {path!r}")
    return str(p)


@dataclass
class HarnessSpec:
    """An in-memory harness tree: relative POSIX path -> text content."""

    files: Dict[str, str] = field(default_factory=dict)

    def set(self, path: str, content: str) -> None:
        self.files[normalize(path)] = content

    def get(self, path: str, default: str = "") -> str:
        return self.files.get(path, default)

    def paths(self, component_id: str) -> List[str]:
        return sorted(p for p in self.files if component_of(p) == component_id)

    def skills(self) -> List[str]:
        names = {PurePosixPath(p).parts[1] for p in self.files if p.startswith("skills/") and len(PurePosixPath(p).parts) > 2}
        return sorted(names)

    def always_on(self) -> Dict[str, str]:
        return {cid: self.files[p] for cid, p in ALWAYS_ON.items() if p in self.files}

    def unknown_paths(self) -> List[str]:
        return sorted(p for p in self.files if component_of(p) is None)


def stock_spec() -> HarnessSpec:
    """Empty base harness with placeholder files naming each lever, so an editor can discover them."""
    s = HarnessSpec()
    s.set("systemprompt.md", "")
    s.set("guardrails.md", "")
    s.set("LongTermMEMORY.md", "")
    s.set("tool_descriptions/tool_guidance.md", "")
    s.set("skills/README.md", "Skills go in skills/<name>/SKILL.md. Each needs a trigger line.\n")
    s.set("sub_agents/README.md", "Subagents go in sub_agents/<name>/AGENT.md.\n")
    s.set("scripts/README.md", "Helper scripts go here and are run via the shell.\n")
    s.set("plugin/README.md", "TypeScript middleware plugins go here.\n")
    return s


def render_codex_agents_md(spec: HarnessSpec) -> str:
    """Codex keeps always-on text in AGENTS.md; concatenate C1-C4 under headed sections."""
    parts: List[str] = []
    for cid, path in ALWAYS_ON.items():
        body = spec.get(path).strip()
        if body:
            name = next(c.name for c in COMPONENTS if c.id == cid)
            parts.append(f"# {name}\n\n{body}\n")
    return "\n".join(parts)


def iter_text(spec: HarnessSpec, paths: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for p in paths:
        yield p, spec.files.get(p, "")
