"""Static validation of a harness tree against its surface.

Re-implements the rules of upstream ``harness2.core.tree.validate`` (at the pinned commit) as a list of
issues instead of one exception, and adds three checks upstream does not make: path safety, a leakage-name
guard, and a cheap TypeScript balance check. Structure only, NOT safety: scripts, hooks and plugins are
executable code and still need runtime policy before use (docs/harness2.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List

from .spec import HarnessSpec, is_allowed, normalize, patterns

RESERVED_SCRIPTS = frozenset({"read_doc.py", "pack.py", "unpack.py"})
_BRACES = re.compile(r"\{\{|\{%|\{#")
_LEAK_NAME = re.compile(r"(rubric|verdict|grader|score|answer[_-]?key|ground[_-]?truth)", re.I)


@dataclass(frozen=True)
class Issue:
    rule: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.path}: {self.message}"


def _frontmatter(text: str) -> str:
    return text[:600]


def validate(spec: HarnessSpec) -> List[Issue]:
    s, files, issues = spec.surface, spec.files, []
    rx = patterns(s)

    for path in files:
        try:
            normalize(path)
        except ValueError as e:
            issues.append(Issue("path", path, str(e)))
            continue
        if not is_allowed(s, path):
            issues.append(Issue("unknown-path", path, f"not on the {s.name} editable surface"))
        if _LEAK_NAME.search(PurePosixPath(path).name):
            issues.append(Issue("leakage", path, "file name looks like a grader artifact"))

    body = spec.always_on_text()
    if len(body) > s.always_on_max:
        issues.append(Issue("always-on-cap", s.always_on[0], f"{len(body)} chars exceeds the {s.always_on_max} cap"))
    if s.reject_empty_always_on and not body.strip():
        issues.append(Issue("always-on-empty", s.always_on[0], "empty always-on file: the harness would not exist"))
    for a in s.always_on:
        m = _BRACES.search(files.get(a, ""))
        if m:
            issues.append(Issue("template-braces", a, f"template braces {m.group(0)!r} are not allowed in always-on text"))

    for name in spec.skills():
        p = f"{s.skill_dir}/{name}/SKILL.md"
        fm = _frontmatter(files[p])
        if not re.search(r"^description:\s*\S", fm, re.M):
            issues.append(Issue("skill-description", p, "needs a `description:` in frontmatter or it is dropped silently"))
        m = re.search(r"^name:\s*(\S+)\s*$", fm, re.M)
        if s.skill_name_rule == "must_match":
            if not (m and m.group(1) == name):
                issues.append(Issue("skill-name", p, f"needs frontmatter `name: {name}` matching its directory"))
        elif m and m.group(1) != name:
            issues.append(Issue("skill-name", p, f"`name: {m.group(1)}` overrides the directory name `{name}`"))
        if name not in body:
            issues.append(Issue("skill-trigger", p, f"skill `{name}` is never named in the always-on text"))

    for name in spec.subagents():
        p = f"sub_agents/{name}/AGENT.md"
        fm = _frontmatter(files[p])
        if s.subagent_requires_mode and not re.search(r"^mode:\s*subagent\s*$", fm, re.M):
            issues.append(Issue("subagent-mode", p, "needs frontmatter `mode: subagent`"))
        if not re.search(r"^description:\s*\S", fm, re.M):
            issues.append(Issue("subagent-description", p, "needs a `description:`"))

    for path, src in files.items():
        compiles = rx["script"].fullmatch(path) or (
            s.compile_skill_scripts and rx["skill_script"].fullmatch(path) and path.endswith(".py"))
        if compiles:
            try:
                compile(src, path, "exec")
            except (SyntaxError, ValueError) as e:
                issues.append(Issue("script-compile", path, f"does not compile: {e}"[:200]))

    have = {p for p in files if rx["script"].fullmatch(p) or rx["skill_script"].fullmatch(p)}
    for path, src in files.items():
        if not (rx["skill"].fullmatch(path) or path in s.always_on):
            continue
        for ref in rx["script_ref"].findall(src):
            if ref.rsplit("/", 1)[-1] not in RESERVED_SCRIPTS and ref not in have:
                issues.append(Issue("script-missing", path, f"invokes `{ref}` but that file is not in the harness"))

    if s.hooks_file and s.hooks_file in files:
        issues.extend(_check_hooks(spec))

    if "plugin" in rx:
        for path, src in files.items():
            if rx["plugin"].fullmatch(path) and (src.count("{") != src.count("}") or src.count("(") != src.count(")")):
                issues.append(Issue("plugin-structure", path, "unbalanced braces or parentheses (tsc not run)"))
    return issues


def _check_hooks(spec: HarnessSpec) -> List[Issue]:
    hf = spec.surface.hooks_file
    try:
        doc = json.loads(spec.files[hf])
    except json.JSONDecodeError as e:
        return [Issue("hooks", hf, f"not valid JSON: {e}")]
    hooks = doc.get("hooks") if isinstance(doc, dict) else None
    if not isinstance(hooks, dict) or not hooks:
        return [Issue("hooks", hf, 'needs a top-level {"hooks": {"<Event>": [...]}} object')]
    out: List[Issue] = []
    for event, entries in hooks.items():
        if event not in spec.surface.hook_events:
            out.append(Issue("hooks", hf, f"unknown event {event!r}"))
        elif not isinstance(entries, list):
            out.append(Issue("hooks", hf, f"{event} must map to a list of matcher entries"))
        else:
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict) or not isinstance(entry.get("hooks"), list):
                    out.append(Issue("hooks", hf, f"{event}[{i}] needs a `hooks` list"))
                    continue
                for h in entry["hooks"]:
                    if not isinstance(h, dict) or h.get("type") != "command":
                        out.append(Issue("hooks", hf, f'{event}[{i}] hook type must be "command"'))
                    elif not str(h.get("command", "")).strip():
                        out.append(Issue("hooks", hf, f"{event}[{i}] has an empty `command`"))
    return out
