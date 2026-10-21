"""OpenCode's stock harness (no refinement), adapted for this experiment.

Runs one rollout per task through upstream Harness²'s JobBench adapter on its OpenCode substrate with an EMPTY starting
harness ("bare stock OpenCode" in upstream's words). Nothing of Harness²'s refinement is involved.

Adaptations (none of them change what OpenCode does; all are routing and bookkeeping):
* every provider is routed through the spend proxy via ``OPENCODE_CONFIG_CONTENT`` (the proxy label ``think`` switches the
  executor's reasoning on; the default is off);
* web access goes through the egress filter (benchmark hosts blocked);
* a runner that exits non-zero after exhausting its time and retries is recorded as *finished with no output* (the stock
  harness produced nothing), any other failure as a crash (not finished, never scored 0);
* each task runs in its own child process so a hung rollout cannot take the experiment down.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..cells import arm_id, task_dir_name
from ._common import opencode_diagnostics, opencode_env

NAME = "opencode"
NO_OUTPUT_MARK = "Benchmark runner exited with status"


# ---- child process (runs where harness2 is importable, with cwd = the harness2 checkout) -------------------------------
def run_one(task: str, label: str, timeout: int) -> Dict[str, Any]:
    from harness2.benchmarks.jobbench import JobBenchAdapter     # upstream package; imported lazily
    ad = JobBenchAdapter(substrate="oc")
    try:
        out = ad.execute(task, None, f"{label}/{task}/base1", timeout=timeout)
    except RuntimeError as e:
        if NO_OUTPUT_MARK in str(e):
            return {"out": None, "reason": str(e)[:200]}
        raise
    return {"out": str(out)}


def _child_main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run-one"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)
    print("RESULT " + json.dumps(run_one(a.task, a.label, a.timeout)), flush=True)
    return 0


# ---- parent ----------------------------------------------------------------------------------------------------------
def runs_dir(exp, cell) -> Path:
    return exp.root / NAME / cell.name


def child_command(task: str, label: str, timeout: int) -> List[str]:
    return [sys.executable, "-m", "agent_harness.bench.harnesses.opencode", "run-one", "--task", task, "--label", label,
            "--timeout", str(timeout)]


def child_env(exp, cell) -> Dict[str, str]:
    max_out = int(exp.cfg.get("max_output_tokens", 16384))
    env = opencode_env(exp.proxy, {cell.provider: {"model": cell.model, "context": cell.window, "output": max_out}},
                       exp.cfg.get("egress_proxy"), labels={cell.provider: cell.label("oc")})
    env.update({"HARNESS2_SOLVER_MODEL": f"{cell.provider}/{cell.model}", "HARNESS2_IMPROVER_MODEL": f"{cell.provider}/{cell.model}",
                "HARNESS2_RUNS_DIR": str(runs_dir(exp, cell))})
    return env


def parse_result(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise ValueError("child printed no RESULT line")


def generate(exp, cell, workers: int = 1, runner=subprocess.run) -> None:
    import os
    lim = exp.cfg.get("limits", {})
    cwd = Path(exp.cfg["harness2_root"]).expanduser()
    env = {**os.environ, **child_env(exp, cell)}
    arm = arm_id(NAME, cell.name)
    label = f"oc-{cell.name}"

    def one(task: str) -> None:
        if exp.board.finished(arm, task):
            return
        try:
            p = runner(child_command(task, label, int(lim.get("max_seconds", 3600))), cwd=cwd, env=env, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"child exit {p.returncode}: {(p.stderr or '')[-200:]}")
            res = parse_result(p.stdout)
        except Exception as e:                    # a crash is NOT finished: never judged, never scored 0
            exp.board._append(arm, task, generated=False, error=f"{type(e).__name__}: {e}"[:300])
            return
        out = Path(res["out"]) if res.get("out") else None
        names = sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()) if out and out.is_dir() else []
        traj = runs_dir(exp, cell) / "jb" / "dataset" / task / "model_traj" / out.name if out else None
        exp.board.mark_generated(arm, task, deliverables=names, stop="done" if names else "no_output", dir=str(out) if out else "",
                                 **(opencode_diagnostics(traj) if traj else {}))
    with ThreadPoolExecutor(workers) as ex_:
        list(ex_.map(one, exp.tasks))


if __name__ == "__main__":
    raise SystemExit(_child_main())
