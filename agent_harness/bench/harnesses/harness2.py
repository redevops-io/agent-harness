"""Google's Harness² (google-research/harness2), adapted for this experiment.

Runs upstream's pipeline unmodified: for every task, a reference rollout with the starting harness (``base1``), two probe
rollouts with an invariant and a domain-specific candidate harness (``inv1``, ``dom1``), a composer that merges them, and a
final rollout with the composed harness (``agg1``); then a reflection step that updates a per-domain library. At k=1 the final
rollout is the result (selection is skipped). Settings: ``--k 1 --mode parallel`` on the OpenCode substrate.

The arm recorded here is the FINAL (refined) rollout. With ``h2-base`` in the experiment's arm list, the reference rollout is
recorded too as a by-product: it is a second sample of the stock OpenCode harness and gives a run-to-run noise estimate.

Adaptations (routing and bookkeeping only; upstream code is not modified):
* executor and editor/composer/reflector are routed through the spend proxy (``OPENCODE_CONFIG_CONTENT``); the executor model is
  the cell's, the editor is a FIXED model identical in every cell (``improver`` in the config), so only the executor varies;
* web access through the egress filter; the proxy label ``think`` switches the executor's reasoning on;
* one runs directory per cell (library and state are never shared between cells), splits file written from the task list;
* no ground truth reaches the pipeline: scoring is done afterwards, outside, by the experiment's judge.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..cells import arm_id, task_dir_name
from ._common import opencode_diagnostics, opencode_env

NAME = "harness2"
NS = "jb-oc-agg1"                  # upstream's run namespace for substrate=oc, mode=parallel, k=1


def runs_dir(exp, cell) -> Path:
    return exp.root / NAME / cell.name


def read_cell(runs: Path, tasks: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per finished task: where the reference (base) and the final (chosen) deliverables are, or None when a rollout produced
    nothing. A task without a summary has not finished and is absent from the result."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        sj = runs / "jb" / "adapt" / NS / task_dir_name(t) / "summary.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        mo, traj = runs / "jb" / "dataset" / t / "model_output", runs / "jb" / "dataset" / t / "model_traj"

        def find(label: Optional[str]) -> Optional[Path]:
            if not label:
                return None
            d = mo / f"{NS}__{task_dir_name(t)}__{label}"
            return d if d.is_dir() and any(p.is_file() for p in d.rglob("*")) else None
        base_label = (s.get("base") or [{}])[0].get("label")
        sel = (s.get("selections") or {}).get("agg") or {}
        chosen = sel.get("chosen_label")
        if not chosen:
            finals = ((s.get("modes") or {}).get("agg") or {}).get("finals") or []
            chosen = finals[0] if finals else None
        out[t] = {"status": s.get("status"), "base_label": base_label, "chosen_label": chosen,
                  "base_dir": find(base_label), "chosen_dir": find(chosen),
                  "base_traj": traj / f"{NS}__{task_dir_name(t)}__{base_label}" if base_label else None,
                  "chosen_traj": traj / f"{NS}__{task_dir_name(t)}__{chosen}" if chosen else None}
    return out


def command(splits: Path, workers: int) -> List[str]:
    return ["harness2", "--bench", "jb", "--substrate", "oc", "--k", "1", "--mode", "parallel", "--split-file", str(splits),
            "--split", "test", "--workers", str(workers)]


def env_for(exp, cell) -> Dict[str, str]:
    imp = exp.cfg["improver"]
    max_out = int(exp.cfg.get("max_output_tokens", 16384))
    env = opencode_env(exp.proxy, {cell.provider: {"model": cell.model, "context": cell.window, "output": max_out},
                                   imp["provider"]: {"model": imp["model"], "context": imp["window"], "output": 32768}},
                       exp.cfg.get("egress_proxy"), labels={cell.provider: cell.label("oc")})
    env.update({"HARNESS2_SOLVER_MODEL": f"{cell.provider}/{cell.model}", "HARNESS2_IMPROVER_MODEL": f"{imp['provider']}/{imp['model']}",
                "HARNESS2_RUNS_DIR": str(runs_dir(exp, cell))})
    return env


def collect(exp, cell) -> None:
    for task, r in read_cell(runs_dir(exp, cell), exp.tasks).items():
        pairs = [(NAME, r["chosen_dir"], r["chosen_traj"], r["chosen_label"])]
        if "h2-base" in exp.arms:
            pairs.append(("h2-base", r["base_dir"], r["base_traj"], r["base_label"]))
        for arm, d, traj, label in pairs:
            a = arm_id(arm, cell.name)
            if exp.board.finished(a, task):
                continue
            names = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d else []
            exp.board.mark_generated(a, task, deliverables=names, stop=r["status"] or "unknown", dir=str(d) if d else "", label=label,
                                     **(opencode_diagnostics(traj) if traj else {}))


def generate(exp, cell, workers: int = 1, runner=subprocess.run) -> None:
    splits = exp.root / "splits.json"
    splits.write_text(json.dumps({"benchmark": "jb", "train": [], "test": exp.tasks}))
    log = exp.root / NAME / f"{cell.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        runner(command(splits, workers), cwd=Path(exp.cfg["harness2_root"]).expanduser(), env={**os.environ, **env_for(exp, cell)},
               stdout=f, stderr=subprocess.STDOUT)
    collect(exp, cell)
