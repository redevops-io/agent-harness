"""Our harness: agent-harness executor + context-runtime context management (the default manager)."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from ..cells import arm_id, task_dir_name

NAME = "ours"


def diagnostics(out_dir: Path) -> Dict[str, Any]:
    peak, compactions, turns = 0, 0, 0
    for tf in sorted(out_dir.glob("trajectory*.jsonl")):
        for line in tf.read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "assistant":
                peak = max(peak, int((e.get("usage") or {}).get("prompt_tokens") or 0))
                turns += 1
            elif e["kind"] == "compact":
                compactions += 1
    return {"peak_prompt": peak, "compactions": compactions, "model_turns": turns}


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME,
             context: str = "context-runtime") -> None:
    from .. import jobbench as jb
    from ...llm import ChatClient
    run_task = run_task or jb.run_task
    lim = exp.cfg.get("limits", {})
    max_out = int(exp.cfg.get("max_output_tokens", 16384))
    arm = arm_id(name, cell.name)

    def one(task: str) -> None:
        if exp.board.finished(arm, task):
            return
        out = exp.root / name / cell.name / task_dir_name(task)
        client = (client_factory or (lambda: ChatClient(f"{exp.proxy}/{cell.provider}/{cell.label('ours')}/v1", cell.model,
                                                        max_tokens=max_out)))()
        try:
            s = run_task(exp.tasks_base, task, out, client, max_turns=lim.get("max_turns", 200), max_seconds=lim.get("max_seconds", 3600),
                         max_context_tokens=cell.window, attempts=lim.get("attempts", 2), context=context)
        except Exception as e:             # a crash is recorded as NOT finished, so it is neither judged nor scored 0
            exp.board._append(arm, task, generated=False, error=f"{type(e).__name__}: {e}"[:300])
            return
        exp.board.mark_generated(arm, task, deliverables=s["deliverables"], stop=s["stop"], dir=str(out / "output"), context=context,
                                 archived_items=s.get("archived_items"), retrieval_calls=s.get("retrieval_calls"),
                                 seconds=s.get("seconds"), **diagnostics(out))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, exp.tasks))
