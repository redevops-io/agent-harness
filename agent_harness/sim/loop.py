# SPDX-License-Identifier: AGPL-3.0-or-later
"""The proposer-evaluator-selector loop over real harness edits, with Self-Improvement Memory (arXiv:2611.18856, 3.1-3.3).

A round: the evaluator runs the current harness on the EVOLVE tasks; the proposer reads that feedback, the run's own
history and the memory, and proposes edits to harness files; the selector screens them, the survivors are evaluated, and
the selector chooses one or none. The chosen edit and what it measured go into the memory, also when it regressed or
missed its prediction. If nothing is chosen the harness and the memory stay as they were.

What is structural here rather than left to a prompt:

* **Held-out tasks never reach the loop.** The optimizer is given the evolve tasks only and refuses to start when a
  declared held-out set overlaps them; every evaluator call it makes is checked against the evolve set. Held-out scoring
  is :func:`evaluate_heldout`, a function the optimizer never calls.
* **The optimizer cannot edit what governs it.** A proposal that touches a protected path (guardrails, hooks, plugins by
  default) or a path the harness surface does not allow is refused before anything runs, and so is a tree that fails
  harness validation. Authorization, policy and the evaluator live outside the harness tree and are not reachable from
  a proposal at all: a proposal is a mapping of harness file paths to text and nothing else.
* **Predictions precede outcomes.** The round's change manifest (every proposal, the root cause it cites, the tasks it
  predicts to fix) is written to disk before the first candidate is evaluated.
* **Every promoted harness is a versioned artifact** with its parent recorded (agent_harness.h2.artifact), so a
  promotion can be rolled back to any ancestor; a rollback is itself recorded and removes nothing from the memory.
* **A candidate that scores below the current harness is not promoted** unless the run was configured to allow it; the
  selector cannot override that.

The proposer and the selector are ports: an LLM in production, a script in the tests. This module holds no model client.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..h2.artifact import ArtifactError, load_artifact, save_artifact, tree_hash
from ..h2.spec import HarnessSpec, is_allowed, normalize
from ..h2.validate import validate
from .memory import MemoryCondition, Provenance, SelfImprovementMemory, SIMError

#: harness files an optimizer may not touch, whatever the proposer says (matched as a path or a directory prefix)
DEFAULT_PROTECTED: Tuple[str, ...] = ("guardrails.md", "hooks.json", "plugin/", "plugins/")


class LoopRefused(SIMError):
    """The loop was asked to do something its guarantees forbid."""


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    score: float = 0.0
    detail: str = ""               # a short trace or error; shown to the proposer
    observed: str = ""             # what the policy answered, as extracted by the evaluator ("" when it gave no answer)
    truncated: bool = False        # the reply ended because the output budget ran out


@dataclass(frozen=True)
class Feedback:
    harness_id: str
    results: Tuple[TaskResult, ...]
    cost_usd: float = 0.0
    seconds: float = 0.0

    @property
    def passed(self) -> FrozenSet[str]:
        return frozenset(r.task_id for r in self.results if r.passed)

    @property
    def failed(self) -> FrozenSet[str]:
        return frozenset(r.task_id for r in self.results if not r.passed)

    @property
    def score(self) -> float:
        return (len(self.passed) / len(self.results)) if self.results else 0.0

    def to_json(self) -> Dict[str, Any]:
        return {"harness_id": self.harness_id, "score": round(self.score, 6), "passed": sorted(self.passed),
                "failed": sorted(self.failed), "cost_usd": self.cost_usd, "seconds": self.seconds,
                "details": {r.task_id: r.detail for r in self.results if r.detail},
                "observed": {r.task_id: r.observed for r in self.results if not r.passed},
                "truncated": sorted(r.task_id for r in self.results if r.truncated)}


@dataclass(frozen=True)
class Proposal:
    root_cause: str                              # an existing problem id (P3) or the text of a new hypothesis
    what: str
    rationale: str
    edits: Mapping[str, Optional[str]]           # harness path -> new text, or None to delete the file
    predicted: Tuple[str, ...] = ()              # evolve tasks this edit is expected to fix

    def to_json(self) -> Dict[str, Any]:
        return {"root_cause": self.root_cause, "what": self.what, "rationale": self.rationale,
                "files": sorted(self.edits), "predicted": list(self.predicted)}


class Evaluator(Protocol):
    def evaluate(self, harness: HarnessSpec, tasks: Sequence[str]) -> Feedback: ...


class Proposer(Protocol):
    def propose(self, *, harness: HarnessSpec, feedback: Feedback, history: Sequence[Mapping[str, Any]],
                memory: str, candidates: int) -> Sequence[Proposal]: ...


class Selector(Protocol):
    def screen(self, proposals: Sequence[Proposal], *, harness: HarnessSpec, memory: str) -> Sequence[bool]: ...

    def choose(self, evaluated: Sequence[Tuple[Proposal, Feedback]], *, current: Feedback,
               memory: str) -> Optional[int]: ...


class TransferCondition(str, Enum):
    """The four conditions of the paper (section 3.3, appendix C.3)."""

    COLD_START = "cold_start"                # seed harness, no memory
    HARNESS_TRANSFER = "harness_transfer"    # source-evolved harness, no memory
    PROBLEM_ONLY = "problem_only"            # source-evolved harness, root causes only
    FULL_SIM = "full_sim"                    # source-evolved harness, the complete record, updated on the target

    @property
    def memory(self) -> MemoryCondition:
        return {TransferCondition.COLD_START: MemoryCondition.NONE, TransferCondition.HARNESS_TRANSFER: MemoryCondition.NONE,
                TransferCondition.PROBLEM_ONLY: MemoryCondition.PROBLEM_ONLY, TransferCondition.FULL_SIM: MemoryCondition.FULL}[self]


def start_target_run(condition: TransferCondition, *, seed: HarnessSpec, source_harness: HarnessSpec,
                     source_memory: SelfImprovementMemory, target_run: str, domain: str = ""
                     ) -> Tuple[HarnessSpec, SelfImprovementMemory, MemoryCondition]:
    """The starting harness, the memory and the view for one condition. The three transfer conditions start from the
    same source-evolved harness. The memory object is always a full record (new target attempts are stored in every
    condition, so the runs stay comparable afterwards); the condition decides what the proposer and selector SEE."""
    condition = TransferCondition(condition)
    if condition is TransferCondition.COLD_START:
        return seed, SelfImprovementMemory(run=target_run), MemoryCondition.NONE
    if condition is TransferCondition.HARNESS_TRANSFER:
        return source_harness, SelfImprovementMemory(run=target_run), MemoryCondition.NONE
    return source_harness, source_memory.carry_into(target_run, domain=domain), condition.memory


def _protected(path: str, protected: Sequence[str]) -> bool:
    return any(path == p.rstrip("/") or path.startswith(p if p.endswith("/") else p + "/") for p in protected)


def apply_edits(harness: HarnessSpec, proposal: Proposal, *, protected: Sequence[str] = DEFAULT_PROTECTED) -> HarnessSpec:
    """The harness with a proposal's edits applied, or ``LoopRefused`` naming the first reason it may not be."""
    if not proposal.edits:
        raise LoopRefused("a proposal must edit at least one harness file")
    out = HarnessSpec(harness.surface, dict(harness.files))
    for raw, text in proposal.edits.items():
        try:
            path = normalize(raw)
        except Exception as exc:  # noqa: BLE001  (the surface's own path errors)
            raise LoopRefused(f"not a harness path: {raw!r} ({type(exc).__name__})") from None
        if _protected(path, protected):
            raise LoopRefused(f"{path} is outside the optimizer's authority (protected)")
        if not is_allowed(harness.surface, path):
            raise LoopRefused(f"{path} is not a file of the {harness.surface.name} harness surface")
        if text is None:
            out.files.pop(path, None)
        elif not isinstance(text, str):
            raise LoopRefused(f"{path}: an edit is text or a deletion")
        else:
            out.set(path, text)
    issues = validate(out)
    if issues:
        raise LoopRefused("the edited harness is not valid: " + "; ".join(str(i) for i in issues[:3]))
    if tree_hash(out) == tree_hash(harness):
        raise LoopRefused("the proposal changes nothing")
    return out


def evaluate_heldout(evaluator: Evaluator, harness: HarnessSpec, heldout_tasks: Sequence[str]) -> Feedback:
    """Held-out scoring. Deliberately a free function: nothing in :class:`HarnessOptimizer` can call it or see its result."""
    return evaluator.evaluate(harness, list(heldout_tasks))


@dataclass
class RoundRecord:
    round: int
    current: Dict[str, Any]
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen: Optional[int] = None
    promoted: Optional[str] = None
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"round": self.round, "current": self.current, "candidates": self.candidates, "chosen": self.chosen,
                "promoted": self.promoted, "note": self.note}


class HarnessOptimizer:
    def __init__(self, *, evaluator: Evaluator, proposer: Proposer, selector: Selector, workdir: "str | Path",
                 harness: HarnessSpec, memory: SelfImprovementMemory, condition: MemoryCondition,
                 evolve_tasks: Sequence[str], provenance: Provenance, heldout_tasks: Sequence[str] = (),
                 candidates_per_round: int = 2, protected: Sequence[str] = DEFAULT_PROTECTED,
                 allow_score_drop: bool = False, record_rejected: bool = False) -> None:
        self.evolve = tuple(dict.fromkeys(evolve_tasks))
        if not self.evolve:
            raise LoopRefused("there is no evolve task to optimize on")
        leak = set(self.evolve) & set(heldout_tasks)
        if leak:
            raise LoopRefused(f"evolve and held-out tasks overlap: {sorted(leak)[:5]}")
        self._evaluator, self.proposer, self.selector = evaluator, proposer, selector
        self.workdir = Path(workdir)
        self.artifacts = self.workdir / "harnesses"
        (self.workdir / "rounds").mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.memory, self.condition = memory, MemoryCondition(condition)
        self.provenance = provenance
        self.candidates_per_round = max(1, int(candidates_per_round))
        self.protected = tuple(protected)
        self.allow_score_drop = bool(allow_score_drop)
        #: False is the paper's rule (section 3.2: only the chosen candidate enters the memory). True also records every
        #: candidate that was evaluated and NOT adopted, as an attempt with accepted=False, so the memory shows what was
        #: tried and did not work. An experimental configuration; a run states which one it used.
        self.record_rejected = bool(record_rejected)
        self.history: List[Dict[str, Any]] = []
        self.round = 0
        self.harness = harness
        self.harness_id = save_artifact(self.artifacts, harness, meta={"role": "start", "run": memory.run})
        self.promotions: List[Dict[str, Any]] = [{"round": 0, "harness": self.harness_id, "kind": "start"}]
        self._current: Optional[Feedback] = None
        self._save_state()

    # ── the only way this class runs tasks ───────────────────────────────────────────────────────────────
    def _evaluate(self, harness: HarnessSpec) -> Feedback:
        fb = self._evaluator.evaluate(harness, list(self.evolve))
        seen = {r.task_id for r in fb.results}
        if not seen <= set(self.evolve):
            raise LoopRefused(f"the evaluator returned tasks outside the evolve set: {sorted(seen - set(self.evolve))[:5]}")
        return fb

    def _save_state(self) -> None:
        self.memory.save(self.workdir / "sim.json")
        (self.workdir / "state.json").write_text(json.dumps(
            {"schema": "agent-harness.sim-run/1", "run": self.memory.run, "condition": self.condition.value,
             "current": self.harness_id, "round": self.round, "promotions": self.promotions, "record_rejected": self.record_rejected,
             "provenance": self.provenance.to_json()}, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    # ── one round ────────────────────────────────────────────────────────────────────────────────────────
    def step(self) -> RoundRecord:
        self.round += 1
        if self._current is None:
            self._current = self._evaluate(self.harness)
        current = self._current
        shown = self.memory.render(self.condition)
        rec = RoundRecord(round=self.round, current=current.to_json())
        proposals = list(self.proposer.propose(harness=self.harness, feedback=current, history=tuple(self.history),
                                               memory=shown, candidates=self.candidates_per_round))[: self.candidates_per_round]
        screened = list(self.selector.screen(proposals, harness=self.harness, memory=shown)) if proposals else []
        if len(screened) != len(proposals):
            raise LoopRefused("the selector must answer once per proposal when screening")
        built: List[Tuple[int, Proposal, HarnessSpec]] = []
        for i, (p, keep) in enumerate(zip(proposals, screened)):
            entry = {**p.to_json(), "screened_in": bool(keep), "refused": ""}
            unknown = sorted(set(p.predicted) - set(self.evolve))
            if unknown:
                entry["refused"] = f"predicts tasks outside the evolve set: {unknown[:5]}"
            elif keep:
                try:
                    built.append((i, p, apply_edits(self.harness, p, protected=self.protected)))
                except LoopRefused as why:
                    entry["refused"] = str(why)
            rec.candidates.append(entry)
        # the change manifest: every proposal, its cited root cause and its prediction, on disk before any candidate runs
        manifest = self.workdir / "rounds" / f"round-{self.round:03d}.manifest.json"
        manifest.write_text(json.dumps({"round": self.round, "harness": self.harness_id, "memory_condition": self.condition.value,
                                        "candidates": rec.candidates}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        evaluated: List[Tuple[int, Proposal, HarnessSpec, Feedback]] = []
        for i, p, spec in built:
            fb = self._evaluate(spec)
            rec.candidates[i]["outcome"] = fb.to_json()
            evaluated.append((i, p, spec, fb))
        pick = self.selector.choose([(p, fb) for _, p, _, fb in evaluated], current=current, memory=shown) if evaluated else None
        if pick is not None and not (0 <= pick < len(evaluated)):
            raise LoopRefused("the selector chose a candidate that was not evaluated")
        if pick is not None:
            i, p, spec, fb = evaluated[pick]
            if fb.score < current.score and not self.allow_score_drop:
                rec.note = f"the selector's choice scored {fb.score:.3f}, below the current {current.score:.3f}: not promoted"
                rec.candidates[i]["refused"] = "scores below the current harness"
                pick = None
        if pick is not None:
            i, p, spec, fb = evaluated[pick]
            parent = self.harness_id
            try:
                new_id = save_artifact(self.artifacts, spec, parent=parent, meta={"round": self.round, "what": p.what, "run": self.memory.run})
            except ArtifactError as why:
                raise LoopRefused(f"the chosen harness could not be stored: {why}") from why
            problem = self.memory.problem(p.root_cause) if self._is_known(p.root_cause) else self.memory.open_problem(p.root_cause, self.round)
            handle = self.memory.open_attempt(problem.id, round=self.round, sample=i, what=p.what, files=sorted(p.edits),
                                              rationale=p.rationale, predicted=p.predicted, candidate_harness=new_id,
                                              provenance=Provenance(**{**self.provenance.to_json(), "harness_version": parent}))
            self.memory.close_attempt(handle, accepted=True, passed_tasks=fb.passed, failed_tasks=fb.failed, previously_passed=current.passed)
            self.harness, self.harness_id, self._current = spec, new_id, fb
            self.promotions.append({"round": self.round, "harness": new_id, "parent": parent, "kind": "promotion", "problem": problem.id})
            rec.chosen, rec.promoted = i, new_id
        elif not rec.note:
            rec.note = "no candidate was accepted: the harness" + (" is unchanged" if self.record_rejected else " and the memory are unchanged")
        if self.record_rejected:
            for i, p, spec, fb in evaluated:
                if rec.chosen == i:
                    continue
                problem = self.memory.problem(p.root_cause) if self._is_known(p.root_cause) else self.memory.open_problem(p.root_cause, self.round)
                handle = self.memory.open_attempt(problem.id, round=self.round, sample=i, what=p.what, files=sorted(p.edits), rationale=p.rationale,
                                                  predicted=p.predicted, candidate_harness=tree_hash(spec),
                                                  provenance=Provenance(**{**self.provenance.to_json(), "harness_version": rec.current.get("harness_id") or self.promotions[-1]["harness"]}))
                self.memory.close_attempt(handle, accepted=False, passed_tasks=fb.passed, failed_tasks=fb.failed, previously_passed=current.passed)
        self.history.append(rec.to_json())
        (self.workdir / "rounds" / f"round-{self.round:03d}.outcome.json").write_text(json.dumps(rec.to_json(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self._save_state()
        return rec

    def _is_known(self, ref: str) -> bool:
        try:
            self.memory.problem(ref)
            return True
        except SIMError:
            return False

    def run(self, rounds: int) -> List[RoundRecord]:
        return [self.step() for _ in range(int(rounds))]

    # ── versions ─────────────────────────────────────────────────────────────────────────────────────────
    def rollback(self, harness_id: str, *, reason: str) -> None:
        """Make an earlier promoted harness current again. The memory keeps every attempt; the rollback is recorded."""
        if harness_id not in {p["harness"] for p in self.promotions}:
            raise LoopRefused("a rollback goes to a harness this run started from or promoted")
        if not str(reason).strip():
            raise LoopRefused("a rollback needs a reason")
        self.harness = load_artifact(self.artifacts, harness_id)
        self.promotions.append({"round": self.round, "harness": harness_id, "kind": "rollback", "from": self.harness_id, "reason": str(reason)})
        self.harness_id, self._current = harness_id, None
        self._save_state()
