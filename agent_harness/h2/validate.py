"""Static validation of a harness tree.

Implements the editor hard rules reported in the Harness² paper (skills need a trigger line, no template
braces in always-on text, every named script must exist and compile) plus a leakage guard (no grader
verdicts, rubrics or scores in the tree). These checks establish structure, NOT safety: scripts and
plugins are executable code and must still pass runtime policy before use (docs/harness2.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List

from .spec import ALWAYS_ON, HarnessSpec, SKILL_FILE, component_of, normalize

_TEMPLATE_BRACES = re.compile(r"\{\{|\}\}|\{%|%\}")
_SCRIPT_REF = re.compile(r"\bscripts/([\w\-./]+\.(?:py|sh|js|ts))")
_TRIGGER = re.compile(r"^\s*(?:trigger\s*:|use\s+when\b|when\s+to\s+use\b)", re.I | re.M)
_FRONTMATTER_DESC = re.compile(r"\A---\s*\n(?:.*\n)*?description\s*:\s*\S", re.I)
_LEAK_NAME = re.compile(r"(rubric|verdict|grader|score|answer[_-]?key|ground[_-]?truth)", re.I)


@dataclass(frozen=True)
class Issue:
    rule: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.path}: {self.message}"


def _has_trigger(text: str) -> bool:
    return bool(_FRONTMATTER_DESC.search(text) or _TRIGGER.search(text))


def validate(spec: HarnessSpec) -> List[Issue]:
    issues: List[Issue] = []

    for path in spec.files:
        try:
            normalize(path)
        except ValueError as e:
            issues.append(Issue("path", path, str(e)))
            continue
        if component_of(path) is None:
            issues.append(Issue("unknown-path", path, "not part of any of the eight components"))
        if _LEAK_NAME.search(PurePosixPath(path).name):
            issues.append(Issue("leakage", path, "file name looks like a grader artifact (rubric/verdict/score)"))

    # Skills need a trigger line in the same edit.
    for name in spec.skills():
        p = f"skills/{name}/{SKILL_FILE}"
        if p not in spec.files:
            issues.append(Issue("skill-missing-file", f"skills/{name}/", f"no {SKILL_FILE}"))
        elif not _has_trigger(spec.files[p]):
            issues.append(Issue("skill-trigger", p, "no trigger line (frontmatter description, 'Trigger:' or 'Use when')"))

    # No template braces in always-on text.
    for path in ALWAYS_ON.values():
        if path in spec.files and _TEMPLATE_BRACES.search(spec.files[path]):
            issues.append(Issue("template-braces", path, "template braces are not allowed in always-on text"))

    # Every named script must exist.
    for path, text in spec.files.items():
        if path in ALWAYS_ON.values() or path.endswith(".md"):
            for ref in sorted(set(_SCRIPT_REF.findall(text))):
                if f"scripts/{ref}" not in spec.files:
                    issues.append(Issue("script-missing", path, f"references scripts/{ref}, which does not exist"))

    # Python scripts must compile.
    for path in spec.paths("C7"):
        if path.endswith(".py"):
            try:
                compile(spec.files[path], path, "exec")
            except SyntaxError as e:
                issues.append(Issue("script-compile", path, f"syntax error: {e.msg} (line {e.lineno})"))

    # Plugins: only a cheap structural check; TypeScript is not compiled here.
    for path in spec.paths("C8"):
        if path.endswith(".ts"):
            t = spec.files[path]
            if t.count("{") != t.count("}") or t.count("(") != t.count(")"):
                issues.append(Issue("plugin-structure", path, "unbalanced braces or parentheses (tsc not run)"))

    return issues
