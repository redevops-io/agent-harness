# SPDX-License-Identifier: AGPL-3.0-or-later
"""The LLM proposer and selector (the "harness engineers" of arXiv:2611.18856) and a model-backed evaluator.

``LLMProposer`` and ``LLMSelector`` implement the ports of :mod:`agent_harness.sim.loop` over any chat client with a
``complete(messages) -> {"message": {"content": ...}}`` method (``agent_harness.llm.ChatClient`` in production). The
memory is placed, whole, in the system prompt of both, followed by the usage guidance of appendix D.2.

Their output is DATA to be checked: a reply that is not the requested JSON yields no proposal (or no choice), never a
guess; the loop validates every edit against the harness surface and the protected paths, and the selector cannot
promote a candidate that scored below the current harness. Neither class sees a task's expected answer: the feedback
they are given is the evolve tasks' ids, prompts, the model's replies and pass/fail.

``PolicyEvaluator`` runs a task set through a policy model with the harness tree as its instructions. The tasks, their
answers and the scoring live in this process and are not part of the harness tree, so no harness edit can change them.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..h2.spec import HarnessSpec
from .loop import Feedback, Proposal, TaskResult
from .tasksets import Task, answer_of

#: appendix D.2 of the paper, as instructions
SIM_GUIDANCE = """How to use the Self-Improvement Memory below:
1. Treat its entries as prior experience. You remain free to choose a different edit.
2. Check the domain, benchmark and model behind each entry before relying on it.
3. Start from an existing diagnosis when it explains the current failure; cite its id.
4. Reuse the rationale, and reassess the edit itself after a benchmark or model change.
5. Revise, replace or override a harness modification when new evidence warrants it.
6. Revisit a direction that was already tested only when new evidence justifies it.
7. Create a new problem entry when no existing diagnosis explains the failure.
8. Give task-specific evidence more weight in similar settings, and less when both the benchmark and the model changed."""

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _json_in(text: str) -> Any:
    """The first JSON value in a model reply (after removing reasoning tags and code fences), or None."""
    body = _THINK.sub("", text or "").strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body.strip())
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = body.find(opener), body.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(body[i:j + 1])
            except ValueError:
                continue
    return None


def _reply(client: Any, system: str, user: str) -> Tuple[str, bool]:
    """The reply text and whether it was cut off by the output budget."""
    out = client.complete([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return str((out.get("message") or {}).get("content") or ""), out.get("finish_reason") == "length"


def _text(client: Any, system: str, user: str) -> str:
    return _reply(client, system, user)[0]


#: How a failing evolve task is described to the proposer. A run records which version it used.
#:   v1  the task, and the tail of the model's reply. (First experiment: 5 of 6 source failures were replies cut off before
#:       any verdict, which v1 does not say; the proposer read the visible reasoning as the model's mistake and proposed the
#:       opposite of the rule.)
#:   v2  adds what the evaluator observed: the verdict it extracted, or that there was none, and whether the reply was cut
#:       off by the output budget. Nothing from the held-out half, and no answer key: the evaluator's observation only.
FEEDBACK_VERSIONS = ("v1", "v2")


def describe_failure(result: TaskResult, task_text: str, version: str) -> str:
    if version == "v1":                                   # exactly the wording of the first experiment
        return f"[{result.task_id}]\n{task_text}\nMODEL REPLY: {result.detail}"
    base = f"[{result.task_id}]\n{task_text}\nMODEL REPLY (last 300 characters): {result.detail}"
    if result.observed:
        seen = f"EVALUATOR: graded incorrect. The model's final verdict was {result.observed}."
    elif result.truncated:
        seen = "EVALUATOR: graded incorrect because NO VERDICT was given: the reply used its whole output budget and was cut off before an answer."
    else:
        seen = "EVALUATOR: graded incorrect because NO VERDICT was given in the reply."
    return base + "\n" + seen


def _harness_text(harness: HarnessSpec, limit: int = 12000) -> str:
    parts = [f"### {path}\n{harness.files[path]}" for path in sorted(harness.files) if harness.files[path].strip()]
    return ("\n\n".join(parts) or "(every harness file is empty)")[:limit]


def _memory_block(memory: str) -> str:
    return (SIM_GUIDANCE + "\n\nSelf-Improvement Memory (complete record):\n" + memory) if memory else "No Self-Improvement Memory is provided in this run."


class LLMProposer:
    def __init__(self, client: Any, *, editable: Sequence[str] = ("systemprompt.md", "tool_descriptions/tool_guidance.md", "LongTermMEMORY.md"),
                 task_text: Optional[Mapping[str, str]] = None, max_failures_shown: int = 12, feedback: str = "v1") -> None:
        if feedback not in FEEDBACK_VERSIONS:
            raise ValueError(f"feedback must be one of {FEEDBACK_VERSIONS}")
        self.feedback = feedback
        self.client, self.editable = client, tuple(editable)
        self.task_text = dict(task_text or {})
        self.max_failures_shown = max_failures_shown
        self.raw: List[str] = []                       # every reply, kept for the evidence

    def propose(self, *, harness: HarnessSpec, feedback: Feedback, history: Sequence[Mapping[str, Any]], memory: str,
                candidates: int) -> Sequence[Proposal]:
        failures = [r for r in feedback.results if not r.passed][: self.max_failures_shown]
        shown = "\n\n".join(describe_failure(r, self.task_text.get(r.task_id, ""), self.feedback) for r in failures) or "(no failing task)"
        past = "\n".join(f"round {h['round']}: " + "; ".join(f"{c['what']} -> {'promoted' if h.get('chosen') == i else (c.get('refused') or 'not chosen')}"
                                                                 f" (score {c.get('outcome', {}).get('score', 'n/a')})" for i, c in enumerate(h["candidates"])) for h in history[-6:]) or "(first round)"
        system = ("You are a harness engineer. You improve the instructions (the harness) given to a fixed task-solving model; you do not solve the tasks yourself.\n"
                  f"You may edit only these harness files: {', '.join(self.editable)}. Other files are outside your authority.\n" + _memory_block(memory))
        user = (f"Current harness:\n{_harness_text(harness)}\n\nEvolve-set score: {feedback.score:.3f} ({len(feedback.passed)}/{len(feedback.results)} passed).\n\n"
                f"Failing evolve tasks (the expected answers are not shown to you):\n{shown}\n\nEarlier rounds of this run:\n{past}\n\n"
                f"Propose up to {candidates} DIFFERENT harness edits. Reply with a JSON array only. Each element:\n"
                '{"root_cause": "<an existing problem id such as P2, or a short new root-cause hypothesis>", "what": "<the modification in one sentence>", '
                '"rationale": "<why it should help>", "predicted": ["<ids of failing tasks you expect this edit to fix>"], '
                '"edits": {"<harness file>": "<the COMPLETE new text of that file>"}}')
        reply = _text(self.client, system, user)
        self.raw.append(reply)
        data = _json_in(reply)
        out: List[Proposal] = []
        for item in (data if isinstance(data, list) else []):
            if not isinstance(item, dict) or not isinstance(item.get("edits"), dict) or not str(item.get("root_cause", "")).strip():
                continue
            edits = {str(k): (v if isinstance(v, str) else None) for k, v in item["edits"].items() if isinstance(v, (str, type(None)))}
            out.append(Proposal(root_cause=str(item["root_cause"]).strip(), what=str(item.get("what", "")).strip() or "unspecified edit",
                                rationale=str(item.get("rationale", "")).strip(), edits=edits,
                                predicted=tuple(str(t) for t in (item.get("predicted") or []) if isinstance(t, str))))
        return out[:candidates]


class LLMSelector:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.raw: List[str] = []

    def screen(self, proposals: Sequence[Proposal], *, harness: HarnessSpec, memory: str) -> Sequence[bool]:
        if not proposals:
            return []
        system = ("You review proposed edits to an agent harness BEFORE they are evaluated. Keep a proposal unless it is clearly not worth an evaluation: "
                  "it repeats a direction the memory shows was tested without new evidence, it is specific to single tasks rather than a class of failures, or it is needlessly complex.\n"
                  + _memory_block(memory))
        user = ("Proposals:\n" + "\n".join(f"{i}: root cause {p.root_cause!r}; {p.what}; rationale: {p.rationale}; files: {sorted(p.edits)}" for i, p in enumerate(proposals))
                + '\n\nReply with a JSON array of booleans, one per proposal, in order (true = evaluate it).')
        reply = _text(self.client, system, user)
        self.raw.append(reply)
        data = _json_in(reply)
        if isinstance(data, list) and len(data) == len(proposals) and all(isinstance(x, bool) for x in data):
            return data
        return [True] * len(proposals)                 # an unusable screening answer screens nothing out

    def choose(self, evaluated: Sequence[Tuple[Proposal, Feedback]], *, current: Feedback, memory: str) -> Optional[int]:
        if not evaluated:
            return None
        system = ("You select at most one evaluated harness edit to adopt. Weigh measured performance on the evolve set, complexity, how specific the edit is to single tasks, "
                  "and regression risk. Choose none when no candidate is a real improvement.\n" + _memory_block(memory))
        lines = []
        for i, (p, fb) in enumerate(evaluated):
            lines.append(f"{i}: {p.what} | root cause {p.root_cause!r} | score {fb.score:.3f} (current {current.score:.3f}) | newly passing: {sorted(fb.passed - current.passed)} | "
                         f"newly failing: {sorted(current.passed & fb.failed)} | predicted to fix: {list(p.predicted)} | edit size: {sum(len(v or '') for v in p.edits.values())} chars")
        reply = _text(self.client, system, "Candidates:\n" + "\n".join(lines) + '\n\nReply with JSON only: {"choice": <index or null>, "reason": "<one sentence>"}')
        self.raw.append(reply)
        data = _json_in(reply)
        pick = data.get("choice") if isinstance(data, dict) else None
        return pick if isinstance(pick, int) and not isinstance(pick, bool) and 0 <= pick < len(evaluated) else None


class PolicyEvaluator:
    """Scores a harness by running a task set through a policy model. Holds the tasks and their answers; a harness only
    ever supplies instructions."""

    def __init__(self, client: Any, tasks: Sequence[Task], *, workers: int = 4, suffix: str = "Answer with one word on the last line: ALLOW or DENY.") -> None:
        self.client, self.workers, self.suffix = client, max(1, int(workers)), suffix
        self._tasks: Dict[str, Task] = {t.task_id: t for t in tasks}
        self.calls = 0

    def evaluate(self, harness: HarnessSpec, tasks: Sequence[str]) -> Feedback:
        unknown = [t for t in tasks if t not in self._tasks]
        if unknown:
            raise KeyError(f"this evaluator does not hold {unknown[:3]}")
        system = harness.always_on_text().strip() or "You are a careful assistant."
        t0 = time.monotonic()

        def one(tid: str) -> TaskResult:
            task = self._tasks[tid]
            try:
                reply, cut = _reply(self.client, system, task.prompt + "\n\n" + self.suffix)
            except Exception as exc:  # noqa: BLE001  (a failed call is a failed task, with the reason)
                return TaskResult(tid, False, 0.0, f"(the model call failed: {type(exc).__name__})")
            got = answer_of(_THINK.sub("", reply))
            return TaskResult(tid, got == task.answer, 1.0 if got == task.answer else 0.0, (_THINK.sub("", reply).strip()[-300:] or "(empty reply)"),
                              observed=got, truncated=bool(cut))

        with ThreadPoolExecutor(self.workers) as pool:
            results = list(pool.map(one, list(tasks)))
        self.calls += len(results)
        return Feedback(harness_id="", results=tuple(results), seconds=round(time.monotonic() - t0, 2))
