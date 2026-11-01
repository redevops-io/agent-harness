"""Run JobBench tasks with agent_harness's own executor and score them with JobBench's own judge script.

    python -m agent_harness.bench.jobbench --tasks-base <job-bench-eval>/dataset/main --task technical_writers/task1 \\
        --label own_q27 --base-url http://127.0.0.1:8765/proxmox/own/v1 --model Qwen3.8-27B --out runs/own \\
        --judge-python <job-bench-eval>/.venv/bin/python --judge-script <job-bench-eval>/eval/judge.py \\
        --judge-model grok-4.7 --judge-base-url http://127.0.0.1:8765/xai/judge/v1

Task framing, output contract and judge are the same as the upstream OpenCode runner; only the executor differs.
The agent never sees RUBRICS.json: only ``task_folder`` is copied into its workspace.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm import ChatClient
from ..loop import LoopResult, run_loop
from .sandbox import DockerSandbox
from .tools import bench_registry

PROMPT = """=== TASK FOLDER ===
/work/task_folder

=== INSTRUCTIONS ===
1. Read the TASK_INSTRUCTIONS.txt file in the task folder above
2. Based on the Reference Files section in TASK_INSTRUCTIONS.txt, read the corresponding files from the same task folder using appropriate tools.
3. Complete the task as specified in TASK_INSTRUCTIONS.txt
4. Only save the final deliverables to the output directory specified below. Do not save any intermediate or temporary files.

=== OUTPUT DIRECTORY ===
/work/output

IMPORTANT:
- All reference files are in the task folder: /work/task_folder
- Only save the final deliverables to the output directory /work/output. Do not save any intermediate or temporary files.
- You MUST only access files within /work or search online for new reference files if you find needed. Do NOT access any files or directories in this system outside of this path.
- If you encounter ambiguous or conflicting information, analyze the conflict, explain your reasoning, and justify the approach you choose.
- If a file cannot be read directly (e.g., .xlsx, .docx, .db, .pptx), use appropriate tools, MCP servers, or code to extract and process its contents."""

# The stock system prompt of this harness: deliberately generic, no domain advice.
BASE_SYSTEM = """You are an autonomous agent that completes professional work tasks end to end using the tools provided.
Work step by step: inspect the inputs, do the work, check the result against the task statement, then save the deliverables
exactly where the task says. When the deliverables are saved, reply with a short summary and stop calling tools."""


_GRADER = re.compile(r"^(rubrics?\b|eval_result|.*_judge\.json$)", re.I)


def assert_workspace_clean(ws: Path) -> None:
    """The agent must never see grader material. Abort the run (not just the task) if any is in its workspace."""
    leaks = sorted(p.relative_to(ws).as_posix() for p in ws.rglob("*") if _GRADER.search(p.name))
    if leaks:
        raise RuntimeError(f"grader artifacts in the agent workspace, refusing to run: {leaks[:5]}")


def _count_tool_calls(trajectory: Path, names) -> int:
    try:
        return sum(1 for line in trajectory.read_text().splitlines() if json.loads(line).get("kind") == "tool"
                   and json.loads(line).get("name") in names)
    except (OSError, ValueError):
        return 0


def make_context_manager(kind: str, archive_dir: Path, *, rag: bool = False, **options):
    """``context-runtime`` (default): lossless archive + retrieval, from the context-runtime package; ``basic``:
    agent_harness's dependency-free lossy elision; ``none``: no proactive management at all (last-resort clipping only). A missing package is an error, never a silent downgrade."""
    if kind == "basic":
        return None
    if kind == "none":
        from ..context import NoCompactionManager
        return NoCompactionManager()
    if kind != "context-runtime":
        raise ValueError(f"unknown context manager {kind!r}")
    try:
        from context_runtime.integrations.agent_harness import ContextRuntimeManager
    except ImportError as e:
        raise RuntimeError("context manager 'context-runtime' needs the context-runtime package "
                           "(pip install -e <context-runtime checkout>); pass context='basic' to opt out") from e
    mgr = ContextRuntimeManager(archive_dir, use_redevops_rag=rag, **options)
    if not (hasattr(mgr, "is_placeholder") and hasattr(mgr, "visibility")):
        raise RuntimeError("the installed context-runtime predates the transcript contract (is_placeholder, visibility): the loop could no longer recognise its stubs. "
                           "Use a context-runtime that has context_runtime/compression/contract.py (branch feat/execution-contract, e6599d0 or later).")
    return mgr


def pin_task_instructions(mgr, task_folder: Path):
    """The task statement is read from a file, not pinned in the prompt, so compaction could archive it away. Pin its text with the
    manager (creating the default Basic manager when none was given). Returns the manager to use."""
    if mgr is None:
        from ..context import BasicContextManager
        mgr = BasicContextManager()
    for f in sorted(Path(task_folder).glob("TASK_INSTRUCTIONS*")):
        if hasattr(mgr, "pin_text") and f.is_file():
            mgr.pin_text(f.read_text(errors="ignore"))
    return mgr


EXTRA_DIRS = ("skills", "scripts", "sub_agents", "plugin")


def _write_extra_files(ws: Path, files: Optional[Dict[str, str]]) -> None:
    """Files the harness itself provides (skills, scripts), written into the workspace at the same relative paths they have in
    the harness tree, so the harness's own instructions ("run scripts/x.py") stay valid. Only under EXTRA_DIRS: never over
    ``task_folder/`` or ``output/``, never outside the workspace."""
    root = ws.resolve()
    for rel, text in (files or {}).items():
        dest = (root / rel).resolve()
        top = dest.relative_to(root).parts[0] if root in dest.parents else None
        if top not in EXTRA_DIRS:
            raise ValueError(f"harness file outside the allowed directories {EXTRA_DIRS}: {rel!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")


def _preserve_incomplete(ws: Path, dest: Path, max_files: int = 200, max_bytes: int = 50_000_000) -> str:
    """Copy what an unfinished run leaves outside ``output/`` (scripts, drafts, intermediate data) so the evidence survives the workspace. Never scored, never
    moved out of the workspace listing of ``outside_output``. Returns the destination name or ''."""
    n = total = 0
    for dirpath, dirs, files in os.walk(ws):
        rel_dir = Path(dirpath).relative_to(ws)
        dirs[:] = [d for d in dirs if not d.startswith(".") and not (rel_dir == Path(".") and d in ("task_folder", "output"))]
        for f in files:
            src = Path(dirpath) / f
            try:
                size = src.stat().st_size
            except OSError:
                continue
            if n >= max_files or total + size > max_bytes:
                return dest.name if n else ""
            tgt = dest / rel_dir / f
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
            n, total = n + 1, total + size
    return dest.name if n else ""


def run_task(tasks_base: Path, task_id: str, out_dir: Path, client: ChatClient, system: str = BASE_SYSTEM, *,
             max_turns: int = 200, max_seconds: float = 3600, max_context_tokens: int = 40960, web: bool = True,
             extra_system: str = "", attempts: int = 2, context: str = "context-runtime", rag: bool = False,
             index_inputs: bool = True, extra_files: Optional[Dict[str, str]] = None, trace: bool = False,
             pin_instructions: bool = True, context_options: Optional[Dict[str, Any]] = None,
             capability_notes: bool = False, web_paging: bool = False, web_downloads: bool = False,
             tools: str = "legacy", network: str = "none", supervise: bool = False, supervisor_config: Optional[Dict[str, Any]] = None,
             vision: str = "off", vision_evidence: str = "", check_integrity: bool = False, preserve_incomplete: bool = True, max_images: Optional[int] = None) -> Dict[str, Any]:
    """Roll one task. Returns a summary dict; deliverables land in ``out_dir/output``, the trajectory in ``out_dir``.

    Like the upstream OpenCode runner, a second attempt (fresh workspace) is made only when the first left no
    deliverables. ``max_seconds`` applies per attempt.

    ``supervise`` runs the loop under the execution supervisor (repeat/stall recovery, budget notices, deliverable checkpoint); the default keeps arm definitions
    that predate it as a bare loop (the old ``[repeat]`` guard no longer exists). ``vision``: ``off`` | ``on`` (verified configuration; give ``vision_evidence``) |
    ``probe`` (two known images are sent to the endpoint first); ``view_image`` is registered only when vision is supported and tools are ``parity``."""
    src = Path(tasks_base) / task_id / "task_folder"
    if not src.is_dir():
        raise FileNotFoundError(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        ws = Path(tempfile.mkdtemp(prefix="ah-jb-"))
        try:
            shutil.copytree(src, ws / "task_folder")
            (ws / "output").mkdir()
            assert_workspace_clean(ws)
            _write_extra_files(ws, extra_files)
            for p in [ws, ws / "task_folder", ws / "output", *ws.rglob("*")]:    # container user is uid 1000, not the host user
                os.chmod(p, 0o777 if p.is_dir() else 0o666)
            blocked: List[str] = []
            mgr = make_context_manager(context, out_dir / f"archive{attempt}", rag=rag, **(context_options or {}))
            if hasattr(mgr, "index_files") and index_inputs:
                mgr.index_files(ws / "task_folder")
            if pin_instructions:
                mgr = pin_task_instructions(mgr, ws / "task_folder")
            rec = None
            if trace:
                from ..trace import TraceRecorder
                rec = TraceRecorder(out_dir / f"trace{attempt}")
            parity = tools == "parity"
            if tools not in ("legacy", "parity"):
                raise ValueError(f"unknown tool set {tools!r}")
            with DockerSandbox(ws, network=network) as sbx:
                found: list = []
                if parity or capability_notes:
                    from .capabilities import probe_environment
                    found = probe_environment(sbx)
                from ..multimodal import VisionCapability, probe_vision
                cap = VisionCapability()
                if vision == "on":
                    cap = VisionCapability(True, "configuration", {"evidence": vision_evidence})
                elif vision == "probe":
                    cap = probe_vision(lambda m: (client.complete(m, None)["message"].get("content") or ""))
                elif vision != "off":
                    raise ValueError(f"unknown vision mode {vision!r}")
                budget = min(12000, int(0.2 * max_context_tokens))
                if parity:
                    from .tools_parity import parity_registry
                    reg = parity_registry(ws, sbx, web=web, network=network, found=found, blocked_log=blocked, output_budget=budget, vision=cap.supported, max_images=max_images)
                else:
                    reg = bench_registry(ws, sbx, blocked_log=blocked, web=web, web_paging=web_paging, web_downloads=web_downloads)
                notes = ""
                if capability_notes:
                    from .capabilities import environment_note
                    notes = environment_note(found, web_paging=(web_paging or parity) and web, web_downloads=(web_downloads or parity) and web,
                                             network=network, edit=parity)
                supervision = None
                if supervise:
                    from .capabilities import supervision_for
                    from ..supervisor import SupervisorConfig
                    instr = "\n".join(f.read_text(errors="ignore") for f in sorted((ws / "task_folder").glob("TASK_INSTRUCTIONS*")) if f.is_file())
                    inputs = [p.name for p in (ws / "task_folder").rglob("*") if p.is_file()]
                    supervision = supervision_for(ws, instr, inputs, SupervisorConfig(**(supervisor_config or {})))
                extra = "\n\n".join(x for x in (extra_system, notes) if x)
                schemas = [sp["function"] if isinstance(sp, dict) and "function" in sp else sp for sp in reg.specs()]    # the exact schemas the model is shown
                (out_dir / f"tool_schemas{'' if attempt == 1 else f'_attempt{attempt}'}.json").write_text(json.dumps(
                    {"tools": tools, "network": network, "sandbox_tools_found": found, "vision": {"supported": cap.supported, "source": cap.source, "evidence": cap.evidence, "max_images": max_images},
                     "output_budget": budget, "schemas": schemas}, indent=1))
                res: LoopResult = run_loop(client, reg, (system + ("\n\n" + extra if extra else "")), PROMPT, supervision=supervision, check_integrity=check_integrity, max_images=max_images,
                                           max_turns=max_turns, max_seconds=max_seconds,
                                           max_context_tokens=max_context_tokens, context_manager=mgr, trace=rec,
                                           trajectory=out_dir / f"trajectory{'' if attempt == 1 else f'_attempt{attempt}'}.jsonl")
            for sub in (".downloads", ".artifacts", ".tool_output"):          # persist the audit trail before the workspace is removed
                if (ws / sub).is_dir():
                    dst = out_dir / f"{sub.strip('.')}{attempt}"
                    shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(ws / sub, dst)
            deliver = out_dir / "output"
            shutil.rmtree(deliver, ignore_errors=True)
            shutil.copytree(ws / "output", deliver)
            from .capabilities import stray_deliverables
            from ..completion import check_deliverables
            outside = stray_deliverables(ws)
            instr_text = "\n".join(f.read_text(errors="ignore") for f in sorted((ws / "task_folder").glob("TASK_INSTRUCTIONS*")) if f.is_file())
            completion = check_deliverables(ws / "output", instr_text, [p.name for p in (ws / "task_folder").rglob("*") if p.is_file()])
            kept = ""
            if preserve_incomplete and (res.stop not in ("done", "verified_complete") or not completion.ok):
                kept = _preserve_incomplete(ws, out_dir / f"incomplete{attempt}")
            files = sorted(p.relative_to(deliver).as_posix() for p in deliver.rglob("*") if p.is_file())
            summary = {"task": task_id, "attempt": attempt, "stop": res.stop, "turns": res.turns, "seconds": res.seconds,
                       "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                       "tool_calls": res.tool_calls, "tool_errors": res.tool_errors, "deliverables": files,
                       "blocked_fetches": blocked, "outside_output": outside, "supervisor": res.supervisor, "completion": completion.to_dict(), "incomplete_preserved": kept, "model": client.model, "context_manager": context, "rag": rag,
                       "archived_items": len(mgr.archive.entries) if hasattr(mgr, "archive") else 0,
                       "retrieval_calls": _count_tool_calls(out_dir / f"trajectory{'' if attempt == 1 else f'_attempt{attempt}'}.jsonl",
                                                            ("search_context", "read_archive"))}
            from .failure_class import classify
            summary["failure"] = classify(out_dir / f"trajectory{'' if attempt == 1 else f'_attempt{attempt}'}.jsonl", summary)
            from .edit_stats import edit_stats, load as load_edits
            summary["edit_stats"] = edit_stats(load_edits(ws / ".artifacts" / "edits.jsonl"))
            summary["run_options"] = {"tools": tools, "network": network, "supervise": supervise, "supervisor_config": supervisor_config, "vision": vision, "vision_supported": cap.supported, "capability_notes": capability_notes, "check_integrity": check_integrity, "max_images": max_images}
            runs.append({k: summary[k] for k in ("attempt", "stop", "turns", "seconds", "deliverables")} | {"failure": summary["failure"]["class"]})
        finally:
            shutil.rmtree(ws, ignore_errors=True)
        if summary["deliverables"]:
            break
    summary["attempts"] = runs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def judge(task_id: str, tasks_base: Path, deliver: Path, details_file: Path, *, judge_python: str, judge_script: str,
          judge_model: str, judge_base_url: str, reasoning_effort: str = "default", timeout: int = 1800,
          finished: bool = True) -> Dict[str, Any]:
    """Score a finished rollout (see ``evalkit.judge_output``). ``graded: False`` means ungraded, never zero."""
    from .evalkit import JudgeConfig, judge_output
    r = judge_output(task_id, tasks_base, deliver, details_file,
                     JudgeConfig(judge_python, judge_script, judge_model, judge_base_url, reasoning_effort, timeout_s=timeout),
                     finished=finished)
    if r.status == "no_output":
        return {"graded": True, "score": 0.0, "no_output": True}
    if r.status == "graded":
        return {"graded": True, "score": r.score, "total": r.total, "max": r.max, "passed": r.passed, "count": r.count}
    return {"graded": False, "score": None, "reason": r.reason, "judge_failures": r.failures}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks-base", required=True)
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system-file")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=3600)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--context-manager", choices=["context-runtime", "basic", "none"], default="context-runtime")
    ap.add_argument("--rag", action="store_true", help="back retrieval with redevops-rag (hybrid vector+BM25) instead of offline BM25")
    ap.add_argument("--no-index-inputs", action="store_true")
    ap.add_argument("--context", type=int, default=40960)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--judge-python"); ap.add_argument("--judge-script")
    ap.add_argument("--judge-model"); ap.add_argument("--judge-base-url")
    a = ap.parse_args(argv)
    system = Path(a.system_file).read_text() if a.system_file else BASE_SYSTEM
    client = ChatClient(a.base_url, a.model, temperature=a.temperature, max_tokens=a.max_tokens)
    rows = []
    for t in a.task:
        out = Path(a.out) / a.label / t.replace("/", "__")
        s = run_task(Path(a.tasks_base), t, out, client, system, max_turns=a.max_turns, max_seconds=a.max_seconds,
                     max_context_tokens=a.context, web=not a.no_web, attempts=a.attempts,
                     context=a.context_manager, rag=a.rag, index_inputs=not a.no_index_inputs)
        if a.judge_python:
            s["judge"] = judge(t, Path(a.tasks_base), out / "output", out / "judge_details.json", judge_python=a.judge_python,
                               judge_script=a.judge_script, judge_model=a.judge_model, judge_base_url=a.judge_base_url)
            (out / "summary.json").write_text(json.dumps(s, indent=2))
        rows.append(s)
        print(json.dumps({k: s[k] for k in ("task", "stop", "turns", "seconds", "deliverables", "judge") if k in s}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
