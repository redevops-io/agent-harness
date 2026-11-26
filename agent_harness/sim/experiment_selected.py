# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete memory against Context Runtime's selected memory, on the pair a finished run produced.

    python -m agent_harness.sim.experiment_selected --run DIR --out DIR2 --context-runtime PATH --base-url URL --model NAME
        [--budget-tokens 3000] [--replicates 3] [--rounds 6] [--candidates 2]

``--run`` is a finished :mod:`agent_harness.sim.experiment` run: its ``source/`` supplies the evolved harness and the memory
carried into the target. Every arm starts from that harness and that memory, uses the same tasks, rounds, candidates,
feedback version and models, and sees the memory under the full condition. The only difference is what the proposer and the
selector read:

* ``complete``  the whole record (the method of arXiv:2611.18856, section 3.2);
* ``selected``  ``context_runtime.learning.sim_select.select`` under ``--budget-tokens``, per round, with a query built from the
  tasks that currently fail, the editable harness files and the evaluator's observations. The tool returns the complete
  record whenever it fits; a view otherwise. If the evidence that must be shown does not fit, it raises ``BudgetTooSmall``:
  the arm stops there, the error is recorded, and no fallback is used.

Context Runtime is imported from ``--context-runtime``; its git commit is recorded in ``results.json``. This package does not
depend on it. Arms alternate within each replicate so that drift in the model server falls on both. A held-out half is
evaluated outside every loop, at the start and at the end of each arm, together with the source task set (retention).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..h2.artifact import load_artifact
from ..llm import ChatClient
from .experiment import _ids, _optimize, seed_harness
from .llm import PolicyEvaluator, TaskResult
from .loop import LoopRefused, evaluate_heldout
from .memory import MemoryCondition, Provenance, SelfImprovementMemory
from .tasksets import TASKSETS

ARMS = ("complete", "selected")


def _import_select(path: Path):
    sys.path.insert(0, str(path))
    from context_runtime.learning import sim_select  # noqa: PLC0415  (an optional, pinned dependency of the experiment only)
    return sim_select


def _commit(path: Path) -> Dict[str, Any]:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True, check=False).stdout.strip()
    return {"path": str(path), "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


class Meter:
    """What one client spent in one arm: calls, the usage the server reports for them, and wall-clock seconds."""

    def __init__(self) -> None:
        self.calls: list = []

    def total(self) -> Dict[str, Any]:
        u = lambda k: sum(int((c.get("usage") or {}).get(k) or 0) for c in self.calls)  # noqa: E731
        return {"calls": len(self.calls), "prompt_tokens": u("prompt_tokens"), "completion_tokens": u("completion_tokens"),
                "calls_without_usage": sum(1 for c in self.calls if not c.get("usage")), "seconds": round(sum(c["seconds"] for c in self.calls), 2)}


class MeteredClient:
    """Wraps a chat client; every call is recorded, none is altered."""

    def __init__(self, inner: Any, meter: Meter) -> None:
        self._inner, self._meter = inner, meter

    def complete(self, messages, tools=None):
        t0 = time.monotonic()
        out = self._inner.complete(messages, tools) if tools is not None else self._inner.complete(messages)
        self._meter.calls.append({"usage": out.get("usage") or {}, "seconds": time.monotonic() - t0})
        return out

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


#: the measured part of an attempt: a view must never change, shorten or drop any of it
MEASURED = ("round", "sample", "files", "predicted", "provenance", "accepted", "backfilled", "fixed", "missed", "regressed", "passed", "total",
            "candidate_harness", "evaluated")


def history_report(record: Dict[str, Any], view_text: str) -> Dict[str, Any]:
    """What of the record's intervention history is in the text the proposer read: parsed back from that text, not from the tool's report."""
    try:
        shown = {p["id"]: p for p in json.loads(view_text).get("problems", [])}
    except (ValueError, AttributeError, KeyError, TypeError):
        shown = {}
    n_att = n_shown = n_measured = n_what = n_rat = 0
    problems_missing = []
    for p in record.get("problems", []):
        got = shown.get(p["id"])
        if got is None:
            problems_missing.append(p["id"])
        by_round = {(a.get("round"), a.get("sample")): a for a in (got or {}).get("attempts", [])}
        for a in p.get("attempts", []):
            n_att += 1
            b = by_round.get((a.get("round"), a.get("sample")))
            if b is None:
                continue
            n_shown += 1
            n_measured += all(a.get(k) == b.get(k) for k in MEASURED)
            n_what += a.get("what") == b.get("what")
            n_rat += bool(a.get("rationale")) and a.get("rationale") == b.get("rationale")
    return {"problems": len(record.get("problems", [])), "problems_not_shown": problems_missing, "attempts": n_att, "attempts_shown": n_shown,
            "attempts_with_identical_measured_fields": n_measured, "attempt_descriptions_verbatim": n_what, "rationales_verbatim": n_rat,
            "rationales_in_record": sum(1 for p in record.get("problems", []) for a in p.get("attempts", []) if a.get("rationale"))}


def _hook(S, budget: Optional[int], trace: list, editable: Sequence[str]):
    """``budget=None``: the complete record (what the optimizer shows by default), recorded the same way as the selected arm."""
    def memory_text(memory: SelfImprovementMemory, current) -> tuple:
        record = memory.view(MemoryCondition.FULL)
        if budget is None:
            text = S.render(record)
            trace.append({"tokens": S.estimate_tokens(text), "complete": True, "compacted": [], "omitted": [], "history": history_report(record, text)})
            return text, {"method": "complete-record"}
        failing = sorted(current.failed)
        query_text = " ".join(r.detail for r in current.results if not r.passed)[:2000]
        try:
            view, rep = S.select(record, S.Query.of(failing, editable, query_text), budget_tokens=budget)
        except S.BudgetTooSmall as exc:
            trace.append({"budget_too_small": True, "needed": exc.needed, "budget": exc.budget})
            raise LoopRefused(f"BudgetTooSmall: needs {exc.needed} tokens, budget {exc.budget}") from exc
        text = S.render(view)
        trace.append({"tokens": rep.get("tokens"), "tokens_complete": rep.get("tokens_complete"), "complete": rep.get("complete"),
                      "compacted": rep.get("compacted"), "omitted": rep.get("omitted"), "history": history_report(record, text)})
        return text, rep
    return memory_text


def run(out: Path, *, run_dir: Path, context_runtime: Path, base_url: str, model: str, budget: int, replicates: int, rounds: int, candidates: int,
        workers: int = 4) -> Dict[str, Any]:
    S = _import_select(context_runtime)
    prior = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    source, target = prior["source"], prior["target"]
    feedback, record_rejected = prior["configuration"]["feedback"], prior["configuration"]["record_rejected"]
    state = json.loads((run_dir / "source" / "state.json").read_text(encoding="utf-8"))
    evolved = load_artifact(run_dir / "source" / "harnesses", state["current"])
    src_mem = SelfImprovementMemory.load(run_dir / "source" / "sim.json")
    out.mkdir(parents=True, exist_ok=True)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    policy = ChatClient(base_url, model, temperature=0, max_tokens=400, timeout=300, retries=2, extra_body=extra)
    engineer = ChatClient(base_url, model, temperature=0.3, max_tokens=4000, timeout=600, retries=2, extra_body=extra)
    src, tgt = TASKSETS[source](), TASKSETS[target]()
    held_src, held_tgt = PolicyEvaluator(policy, src.heldout, workers=workers), PolicyEvaluator(policy, tgt.heldout, workers=workers)
    start_score = evaluate_heldout(held_tgt, evolved, _ids(tgt.heldout)).score
    results: Dict[str, Any] = {}
    for rep in range(1, replicates + 1):
        for arm in ARMS:
            name = f"{arm}-{rep}"
            mem = src_mem.carry_into(name, domain="policy-vetting")
            trace: list = []
            hook = _hook(S, budget if arm == "selected" else None, trace, ("systemprompt.md",))
            eng_meter, pol_meter = Meter(), Meter()
            prov = Provenance(domain="policy-vetting", benchmark=target, policy=model, optimizer=model, evaluator="oracle:" + target, run=name)
            t0, status, err = time.monotonic(), "completed", ""
            log: Dict[str, Any] = {}
            opt = None
            try:
                opt = _optimize(out, name, condition=MemoryCondition.FULL, memory_text=hook, client=MeteredClient(engineer, eng_meter), policy_client=MeteredClient(policy, pol_meter), ts=tgt, harness=evolved, memory=mem, rounds=rounds, candidates=candidates,
                                          prov=prov, log=log, feedback=feedback, record_rejected=record_rejected)
            except LoopRefused as exc:
                status, err = "stopped", str(exc)
            (out / f"{name}-selection-trace.json").write_text(json.dumps(trace, indent=1), encoding="utf-8")
            entry: Dict[str, Any] = {"status": status, "error": err, "selection_trace": trace, "seconds": round(time.monotonic() - t0, 1),
                                     "engineer_usage": eng_meter.total(), "policy_usage": pol_meter.total()}
            if opt is not None:
                entry.update(log.get(name, {}))
                entry["target_heldout_final"] = evaluate_heldout(held_tgt, opt.harness, _ids(tgt.heldout)).score
                entry["source_heldout_final"] = evaluate_heldout(held_src, opt.harness, _ids(src.heldout)).score
            results[name] = entry
            print(name, status, entry.get("target_heldout_final"), flush=True)
    doc = {"schema": "agent-harness.sim-selected/1", "model": model, "from_run": str(run_dir), "source": source, "target": target, "rounds": rounds,
           "candidates_per_round": candidates, "replicates": replicates, "budget_tokens": budget, "token_counter": "context_runtime estimate_tokens (4 characters per token)",
           "configuration": {"feedback": feedback, "record_rejected": record_rejected}, "context_runtime": _commit(context_runtime),
           "target_heldout_start": start_score, "arms": results,
           "cost_note": "local GPU inference: no per-token price applies; tokens (as reported by the server) and wall-clock seconds are the cost measures",
           "caveats": ["replicates alternate the two arms; sampling of the harness engineers is not seeded", "30 held-out tasks: one task is 3.3 points",
                       "the carried memory holds ONE problem, so selection can compact its prose but cannot omit or rank problems", "same model proposes, selects and acts"]}
    (out / "results.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context-runtime", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-tokens", type=int, default=3000)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=2)
    a = ap.parse_args(argv)
    r = run(a.out, run_dir=a.run, context_runtime=a.context_runtime, base_url=a.base_url, model=a.model, budget=a.budget_tokens,
            replicates=a.replicates, rounds=a.rounds, candidates=a.candidates)
    print(json.dumps({k: {x: v.get(x) for x in ("status", "target_heldout_final", "source_heldout_final")} for k, v in r["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
