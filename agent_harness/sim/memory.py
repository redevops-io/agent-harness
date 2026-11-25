# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Self-Improvement Memory record (arXiv:2611.18856, section 3.2 and appendix D.3).

One JSON file per optimization run. It groups harness edits by the recurring problem they address::

    { "schema": "agent-harness.sim/1", "next_id": 10,
      "carried_from": {"arm": <source run>, "domain": ..., "into": <target run>} | null,
      "aliases": {"P6": "P5"},                       # P6 was merged into P5
      "problems": [ {"id", "name", "opened_round", "attempts": [...]} ] }

and each attempt holds ``round, sample, accepted, backfilled, what, files, rationale, predicted, fixed, missed, passed,
total, regressed`` and its provenance (``domain, benchmark, policy, harness_version, optimizer, evaluator, run``).

What the code guarantees, each with a test in tests/sim/:

* **A prediction is written before the outcome exists.** ``open_attempt`` stores the edit, its rationale and the tasks it
  is predicted to fix, and returns a pending handle; only ``close_attempt`` can add the observed outcome, and it cannot
  change what was predicted. A second close, or a close of an unknown handle, is refused.
* **History is never dropped.** There is no delete. An edit that is later revised, replaced or removed keeps its
  attempts; merging two problems keeps every attempt of both and leaves an alias for the merged id.
* **Entries with identical root-cause text are one entry** (deterministic merge, D.1); whether a new diagnosis matches
  an existing one is otherwise the proposer's decision, recorded in its proposal.
* **Outcomes are recorded as measured**: a regression or a missed prediction is stored like a success.
* **The memory condition is a view, not a different store** (section 3.3, appendix C.3): ``full`` renders everything,
  ``problem_only`` renders root causes alone, ``none`` renders nothing. The record on disk is complete either way, so
  the conditions can be compared on the same history.

Standard library only. No model, no network, no clock except the one passed in.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "agent-harness.sim/1"


class SIMError(ValueError):
    """A use of the memory that would falsify or lose history."""


class MemoryCondition(str, Enum):
    """What of the record the proposer and the selector are shown (the three memory conditions of the paper)."""

    NONE = "none"                    # cold start and harness transfer: no SIM
    PROBLEM_ONLY = "problem_only"    # root causes only: no edits, outcomes or provenance
    FULL = "full"                    # the complete record, without retrieval or truncation


@dataclass(frozen=True)
class Provenance:
    """Where an attempt was made and measured. Every field is recorded; none is verified by this class."""

    domain: str
    benchmark: str
    policy: str                      # the task-solving model (and its version) the harness was run with
    harness_version: str = ""        # id of the harness the edit was applied to (an h2 artifact id)
    optimizer: str = ""              # the proposer/selector model
    evaluator: str = ""              # what produced the scores (judge or test suite, with its version)
    run: str = ""

    def to_json(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Attempt:
    round: int
    sample: int
    what: str                        # the modification, in words
    files: Tuple[str, ...]           # harness files it touched
    rationale: str
    predicted: Tuple[str, ...]       # tasks expected to be fixed, written BEFORE evaluation
    provenance: Provenance
    accepted: bool = False           # the selector chose this edit in its round
    backfilled: bool = False         # recorded after the fact from an earlier run's history
    fixed: Tuple[str, ...] = ()      # predicted tasks that did pass afterwards
    missed: Tuple[str, ...] = ()     # predicted tasks that did not
    regressed: Tuple[str, ...] = ()  # tasks that passed before the edit and failed after it
    passed: Optional[int] = None     # evolve-set trials passed in this attempt's evaluation
    total: Optional[int] = None
    candidate_harness: str = ""      # id of the harness with the edit applied
    evaluated: bool = False

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_json()
        for k in ("files", "predicted", "fixed", "missed", "regressed"):
            d[k] = list(d[k])
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Attempt":
        return Attempt(round=int(d["round"]), sample=int(d.get("sample", 0)), what=str(d["what"]),
                       files=tuple(d.get("files", ())), rationale=str(d.get("rationale", "")),
                       predicted=tuple(d.get("predicted", ())), provenance=Provenance(**d["provenance"]),
                       accepted=bool(d.get("accepted", False)), backfilled=bool(d.get("backfilled", False)),
                       fixed=tuple(d.get("fixed", ())), missed=tuple(d.get("missed", ())),
                       regressed=tuple(d.get("regressed", ())), passed=d.get("passed"), total=d.get("total"),
                       candidate_harness=str(d.get("candidate_harness", "")), evaluated=bool(d.get("evaluated", False)))


@dataclass
class Problem:
    id: str
    name: str                        # the root-cause hypothesis
    opened_round: int
    attempts: List[Attempt] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "opened_round": self.opened_round,
                "attempts": [a.to_json() for a in self.attempts]}


@dataclass(frozen=True)
class PendingAttempt:
    """Handle of an attempt whose prediction is on record and whose outcome is not yet."""

    problem_id: str
    index: int


def _norm(name: str) -> str:
    return " ".join(str(name).split())


class SelfImprovementMemory:
    def __init__(self, *, run: str = "", carried_from: Optional[Dict[str, str]] = None) -> None:
        self.run = run
        self.carried_from: Optional[Dict[str, str]] = dict(carried_from) if carried_from else None
        self.problems: List[Problem] = []
        self.aliases: Dict[str, str] = {}
        self.next_id = 1
        self._pending: set = set()

    # ── reading ──────────────────────────────────────────────────────────────────────────────────────────
    def resolve(self, pid: str) -> str:
        """Follow aliases to the id an entry lives under today."""
        seen = set()
        while pid in self.aliases:
            if pid in seen:
                raise SIMError(f"alias cycle at {pid}")
            seen.add(pid)
            pid = self.aliases[pid]
        return pid

    def problem(self, pid: str) -> Problem:
        target = self.resolve(pid)
        for p in self.problems:
            if p.id == target:
                return p
        raise SIMError(f"no problem {pid!r} in this memory")

    def find(self, name: str) -> Optional[Problem]:
        want = _norm(name)
        return next((p for p in self.problems if _norm(p.name) == want), None)

    def attempts(self) -> List[Tuple[Problem, Attempt]]:
        return [(p, a) for p in self.problems for a in p.attempts]

    # ── writing ──────────────────────────────────────────────────────────────────────────────────────────
    def open_problem(self, name: str, round: int) -> Problem:
        """The entry for a root-cause hypothesis: the existing one when its text is identical, else a new one."""
        if not _norm(name):
            raise SIMError("a problem needs a root-cause hypothesis")
        existing = self.find(name)
        if existing is not None:
            return existing
        p = Problem(id=f"P{self.next_id}", name=_norm(name), opened_round=int(round))
        self.next_id += 1
        self.problems.append(p)
        return p

    def open_attempt(self, problem_id: str, *, round: int, sample: int, what: str, files: Sequence[str], rationale: str,
                     predicted: Sequence[str], provenance: Provenance, candidate_harness: str = "",
                     backfilled: bool = False) -> PendingAttempt:
        """Put an edit and its prediction on record BEFORE it is evaluated."""
        if not str(what).strip():
            raise SIMError("an attempt must say what was modified")
        p = self.problem(problem_id)
        p.attempts.append(Attempt(round=int(round), sample=int(sample), what=str(what), files=tuple(files),
                                  rationale=str(rationale), predicted=tuple(dict.fromkeys(predicted)),
                                  provenance=provenance, candidate_harness=candidate_harness, backfilled=backfilled))
        handle = PendingAttempt(p.id, len(p.attempts) - 1)
        self._pending.add(handle)
        return handle

    def close_attempt(self, handle: PendingAttempt, *, accepted: bool, passed_tasks: Iterable[str],
                      failed_tasks: Iterable[str], previously_passed: Iterable[str] = (),
                      passed: Optional[int] = None, total: Optional[int] = None) -> Attempt:
        """Add the measured outcome. ``fixed`` and ``missed`` are computed here from the prediction on record: the caller
        supplies what passed and what failed, not which predictions came true."""
        if handle not in self._pending:
            raise SIMError("this attempt is not open: it was already closed, or never opened in this memory")
        p = self.problem(handle.problem_id)
        a = p.attempts[handle.index]
        ok, bad = set(passed_tasks), set(failed_tasks)
        if ok & bad:
            raise SIMError(f"tasks reported both passed and failed: {sorted(ok & bad)[:5]}")
        closed = Attempt(**{**a.__dict__, "accepted": bool(accepted), "evaluated": True,
                            "fixed": tuple(t for t in a.predicted if t in ok),
                            "missed": tuple(t for t in a.predicted if t not in ok),
                            "regressed": tuple(sorted(set(previously_passed) & bad)),
                            "passed": len(ok) if passed is None else int(passed),
                            "total": len(ok) + len(bad) if total is None else int(total)})
        p.attempts[handle.index] = closed
        self._pending.discard(handle)
        return closed

    def merge(self, absorbed: str, into: str) -> Problem:
        """Merge one problem into another, keeping every attempt of both and an alias for the absorbed id."""
        a, b = self.problem(absorbed), self.problem(into)
        if a is b:
            return b
        if any(h.problem_id == a.id for h in self._pending):
            raise SIMError("close the open attempts of a problem before merging it")
        b.attempts = sorted(b.attempts + a.attempts, key=lambda x: (x.round, x.sample))
        b.opened_round = min(a.opened_round, b.opened_round)
        self.problems.remove(a)
        self.aliases[a.id] = b.id
        return b

    def merge_identical(self) -> int:
        """Deterministic merge of entries whose root-cause text is identical (appendix D.1). Returns how many merged."""
        n, seen = 0, {}
        for p in list(self.problems):
            key = _norm(p.name)
            if key in seen:
                self.merge(p.id, seen[key].id)
                n += 1
            else:
                seen[key] = p
        return n

    def carry_into(self, target_run: str, *, domain: str = "") -> "SelfImprovementMemory":
        """The memory a target run starts with: every problem and attempt of this one, ids unchanged."""
        if self._pending:
            raise SIMError("an attempt is still open: close it before carrying the memory into another run")
        out = SelfImprovementMemory.from_json(self.to_json())
        out.run = target_run
        out.carried_from = {"arm": self.run, "domain": domain, "into": target_run}
        return out

    # ── what the proposer and the selector are shown ─────────────────────────────────────────────────────
    def view(self, condition: MemoryCondition) -> Dict[str, Any]:
        """The record as shown under a memory condition. FULL is everything, with no retrieval and no truncation."""
        condition = MemoryCondition(condition)
        if condition is MemoryCondition.NONE:
            return {}
        if condition is MemoryCondition.PROBLEM_ONLY:
            return {"problems": [{"id": p.id, "name": p.name} for p in self.problems]}
        return self.to_json()

    def render(self, condition: MemoryCondition) -> str:
        """The text placed in the proposer's and the selector's prompt."""
        v = self.view(condition)
        return "" if not v else json.dumps(v, indent=1, sort_keys=True, ensure_ascii=False)

    # ── storage ──────────────────────────────────────────────────────────────────────────────────────────
    def to_json(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "run": self.run, "next_id": self.next_id, "carried_from": self.carried_from,
                "aliases": dict(self.aliases), "problems": [p.to_json() for p in self.problems]}

    @staticmethod
    def from_json(doc: Dict[str, Any]) -> "SelfImprovementMemory":
        if doc.get("schema") != SCHEMA:
            raise SIMError(f"not a {SCHEMA} record")
        m = SelfImprovementMemory(run=str(doc.get("run", "")), carried_from=doc.get("carried_from"))
        m.next_id = int(doc["next_id"])
        m.aliases = {str(k): str(v) for k, v in (doc.get("aliases") or {}).items()}
        for p in doc.get("problems", []):
            m.problems.append(Problem(id=str(p["id"]), name=str(p["name"]), opened_round=int(p["opened_round"]),
                                      attempts=[Attempt.from_json(a) for a in p.get("attempts", [])]))
        ids = [p.id for p in m.problems]
        if len(ids) != len(set(ids)) or set(ids) & set(m.aliases):
            raise SIMError("problem ids repeat, or an id is both an entry and an alias")
        return m

    def save(self, path: "str | Path") -> None:
        """Write atomically. An attempt still open is saved with its prediction and no outcome, which is the truth."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".sim-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_json(), fh, indent=1, sort_keys=True, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def load(path: "str | Path") -> "SelfImprovementMemory":
        return SelfImprovementMemory.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
