"""Our harness WITH the changes Google's Harness² pipeline composed: same loop, tools and context-runtime as ``ours``, plus
the edits the pipeline made to the stock OpenCode harness for this task and cell.

Only the DIFFERENCE between the stock tree (``base``) and the composed one (``agg1``) is applied: always-on text that changed
is appended to our system prompt, new or changed skills and scripts are placed in the workspace at their tree paths (``skills/..``, ``scripts/..``) and named
in that text. Unchanged stock text is left out (it is OpenCode wording written for OpenCode's tools). The composed tree is
validated with ``agent_harness.h2`` first; an invalid tree is refused.

Needs the ``harness2`` arm to have run for the cell (its run directory holds the trees), so list it BEFORE this arm. The
pipeline sees the task while it refines, as in the paper's test-time setting; that is a property of the method, not hidden
information (no rubric or answer reaches it)."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from ...h2 import OPENCODE, HarnessSpec, validate
from ..cells import arm_id, task_dir_name
from . import harness2 as h2
from . import ours
from .ours import diagnostics

NAME = "ours-h2"
ALWAYS_ON = ("systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md")
MAX_ADDENDUM_CHARS = 24000
IGNORED_RULES = ("template-braces",)


def tree_dir(exp, cell, task: str, which: str) -> Path:
    a = h2.adapt_dir(exp, cell, task)              # the pipeline's chunk directory that holds this task
    return (a if a is not None else h2.runs_dir(exp, cell) / "jb" / "adapt" / h2.NS / task_dir_name(task)) / "harnesses" / which


# upstream scaffolds an explanatory README.md into each component directory; they are not editable harness files
SCAFFOLD = re.compile(r"^(plugin|scripts|skills|sub_agents|tool_descriptions)/README\.md$")


def read_tree(d: Path) -> Dict[str, str]:
    return {str(p.relative_to(d)): p.read_text(encoding="utf-8") for p in sorted(d.rglob("*"))
            if p.is_file() and not SCAFFOLD.match(str(p.relative_to(d)))}


def _description(skill_md: str) -> str:
    m = re.search(r"^description:\s*(.+)$", skill_md, re.M)
    return m.group(1).strip().strip("\"'") if m else ""


def delta(base: Dict[str, str], final: Dict[str, str]) -> Tuple[str, Dict[str, str], List[str]]:
    """(text appended to the system prompt, files for ``<workspace>/harness/``, the paths that differ from stock)."""
    changed = sorted(p for p in final if final[p] != base.get(p))
    # refuse on problems in what the pipeline CHANGED (or whole-tree problems); stock text we do not use is not ours to police.
    # "template-braces" is skipped: our validator is stricter than upstream there, and our prompt is not a template
    issues = [i for i in validate(HarnessSpec(OPENCODE, dict(final)))
              if (not i.path or i.path in changed) and i.rule not in IGNORED_RULES]
    if issues:
        raise ValueError("composed harness is invalid: " + "; ".join(str(i) for i in issues[:3]))
    parts, files = [], {}
    for p in ALWAYS_ON:
        if p in changed:
            parts.append(f"### {p}\n{final[p].strip()}")
    skills = sorted({p.split("/")[1] for p in changed if p.startswith("skills/") and p.endswith("SKILL.md")})
    for n in skills:
        files[f"skills/{n}/SKILL.md"] = final[f"skills/{n}/SKILL.md"]
    scripts = [p for p in changed if p.startswith("scripts/")]
    for p in scripts:
        files[p] = final[p]
    if skills:
        parts.append("### Skills (a skill is a file: \"loading\" it means reading it with your file tool)\n" + "\n".join(
            f"- skills/{n}/SKILL.md: {_description(final[f'skills/{n}/SKILL.md'])}" for n in skills))
    if scripts:
        parts.append("### Scripts (run with python3 from the workspace root)\n" + "\n".join(f"- {p}" for p in scripts))
    text = ("## Additional guidance for this task (refined harness)\n\n" + "\n\n".join(parts))[:MAX_ADDENDUM_CHARS] if parts else ""
    return text, files, changed


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME,
             context: str = "context-runtime", context_options: dict = None) -> None:
    from .. import jobbench as jb
    from ...llm import ChatClient
    run_task = run_task or jb.run_task
    lim = exp.cfg.get("limits", {})
    max_out = int(exp.cfg.get("max_output_tokens", 16384))
    arm = arm_id(name, cell.name)

    def one(task: str) -> None:
        if exp.board.finished(arm, task):
            return
        base, final = tree_dir(exp, cell, task, "base"), tree_dir(exp, cell, task, "agg1")
        if not (base.is_dir() and final.is_dir()):
            print(f"{arm} {task}: no composed harness yet (run the harness2 arm for this cell first); skipped", flush=True)
            return
        out = exp.root / name / cell.name / task_dir_name(task)
        client = (client_factory or (lambda: ours.make_client(exp, cell, max_out)))()
        try:
            text, files, changed = delta(read_tree(base), read_tree(final))
            s = run_task(exp.tasks_base, task, out, client, max_turns=lim.get("max_turns", 200), max_seconds=lim.get("max_seconds", 3600),
                         max_context_tokens=cell.window, attempts=lim.get("attempts", 2), context=context,
                         extra_system=text, extra_files=files, trace=bool(exp.cfg.get("trace", False)),
                         context_options=context_options)
        except Exception as e:             # a crash is NOT finished: neither judged nor scored 0
            exp.board._append(arm, task, generated=False, error=f"{type(e).__name__}: {e}"[:300])
            return
        exp.board.mark_generated(arm, task, deliverables=s["deliverables"], stop=s["stop"], dir=str(out / "output"),
                                 archived_items=s.get("archived_items"), retrieval_calls=s.get("retrieval_calls"),
                                 seconds=s.get("seconds"), context=context, context_options=context_options or None, sampling=exp.cfg.get("sampling") or None, harness_changed_files=changed, **diagnostics(out))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, exp.tasks))
