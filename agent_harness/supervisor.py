"""The execution supervisor: ONE authority for repeated operations, stalled work, recovery, budget and completion in ``run_loop``.

It replaces the earlier overlapping mechanisms (the ``[repeat]`` guard, the consecutive-failure note, the progress tracker's nudges and checkpoint). The loop
asks it four things: may this call run (``before``), what should the model see of the result (``after``), may the model stop (``on_finish_attempt``), and
should the run end (``stop_reason``). Everything it decides is a function of EVIDENCE about the run, never of a bare repeat count:

* **Operations are grouped into families**: the same tool and, for shell commands, a command that is at least ``family_similarity`` alike after digits and
  whitespace are normalised; for web tools the same URL and offset; for file tools the same path and arguments. A near-identical script that only changes a crop
  box is the same family as the last one; a different script is not.
* **Progress is a change that matters.** For a family, only these reset its stall count: a change to the output directory's content, a change to a file the
  operation refers to made by ANOTHER call, an error turning into a different error or into success. Scratch files the operation writes itself, background
  filesystem noise and rewriting identical bytes are not progress (the workspace is compared by content hash, per call, so each change is attributed to the
  call that made it).
* **Recovery is a bounded state machine per family**: ``normal`` -> (one concise note) -> ``suppressed`` (calls in that family are answered without running,
  with a time-to-live after which one probe run is allowed) -> back to ``normal`` on relevant change. Materially different operations are never blocked.
  *Recovery is exhausted* only when a suppressed family keeps being requested ``exhaust_after`` times with no different operation executed in between AND
  nothing has progressed for ``exhaust_min_idle_turns``; then the run ends as ``blocked`` with the evidence preserved.
* **Refusals do not pile up in the model's history**: the first ``max_visible_notes`` suppression answers of a family are informative, every later one is a
  one-line stub, and a family that is refused ``exhaust_after`` times with nothing different tried ends the run (bounded, never an unlimited stream).
* **Refusals say what is true**: if an identical earlier result was archived or clipped, the answer is a pointer to it (asked from the context manager's
  ``visibility``), mentioning only parameters the tool actually has; if it is unavailable the call is allowed to run again.
* **Budget and completion**: remaining turns and time are tracked; notices (never claiming success) go out at fixed fractions and when the reserve for
  building and checking deliverables is reached; a model that stops with missing or invalid deliverables is turned back at most ``checkpoint_pushes`` times;
  once the deliverables pass the structural checks and nothing has changed for ``verified_idle_end`` turns the run ends as ``verified_complete`` (structurally
  valid, not reviewed for correctness).
"""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .completion import DeliverableReport
from .multimodal import image_is_visible

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
_NOTE = re.compile(r"\n?\[(?:supervisor|budget)[^\]]*\]", re.S)
_ERROR = re.compile(r"^ERROR\b|Traceback \(most recent call last\)|\[exit code [1-9]|\[timed out|^\[stderr\]|Error:|Exception:", re.M)
Snapshot = Dict[str, str]


@dataclass
class SupervisorConfig:
    family_similarity: float = 0.90
    identical_suppress_after: int = 2        # executions returning the same result before the next identical call is not run
    changing_watch_after: int = 6            # executions of one family with changing, non-error results and no relevant change before a note
    changing_suppress_after: int = 10        # ... before the family is suppressed
    suppress_ttl_turns: int = 15             # a suppressed family may run one probe execution after this many turns
    exhaust_after: int = 6                   # suppressed requests with no different operation executed in between ...
    exhaust_min_idle_turns: int = 30         # ... AND this many turns without progress (output change, relevant file change, a new kind of operation, a new error)
    end_on_exhaustion: bool = True           # False = advisory: record that recovery is exhausted but keep the run going (for controlled comparisons)
    pointer_answers: int = 2                 # identical re-requests answered with a pointer to an archived / clipped earlier result
    max_visible_notes: int = 2               # suppression answers per family that keep their full text in the history
    delivery_stall_factor: float = 0.5       # thresholds shrink by this factor once the delivery reserve is reached
    reserve_turns: int = 12
    reserve_fraction: float = 0.08
    reserve_seconds: float = 300.0
    reserve_blocks: Tuple[str, ...] = ("web_fetch", "web_download", "search_context")   # exploration the reserve for building and checking deliverables may not be spent on
    notice_fractions: Tuple[float, ...] = (0.5, 0.8)
    checkpoint_pushes: int = 2
    stall_checkpoint: bool = False             # opt-in (not used by any pinned run): after ``stall_checkpoint_ignored`` refusals of one operation in a row, while no valid deliverable exists, say what is missing and start delivery mode now
    stall_checkpoint_ignored: int = 3
    min_turns_for_push: int = 3
    min_seconds_for_push: float = 120.0
    verified_idle_notice: int = 6
    verified_idle_end: int = 12
    output_prefix: str = "output/"
    max_events: int = 400


@dataclass
class Decision:
    run: bool
    text: str = ""
    kind: str = "run"            # run | pointer | suppressed | exhausted | probe
    exhausted: bool = False


@dataclass
class FinishDecision:
    action: str                  # accept | push
    text: str = ""


@dataclass
class _Family:
    id: int
    tool: str
    norm: str
    example: str
    refs: Set[str]
    state: str = "normal"                    # normal | suppressed
    stall: int = 0
    identical: int = 0
    last_fp: Optional[str] = None
    last_error_fp: Optional[str] = None
    last_seq: int = 0
    execs: int = 0
    ops: Dict[str, Tuple[str, int, str]] = field(default_factory=dict)       # exact key -> (call_id, turn, fingerprint)
    watched: bool = False
    pointers: int = 0
    ttl_until: int = 0
    ttl: int = 0
    probing: bool = False
    suppressed_total: int = 0
    ignored: int = 0
    last_alt_seen: int = 0


def fingerprint(text: str) -> str:
    import hashlib
    return hashlib.sha1(_WS.sub(" ", _NOTE.sub("", text or "")).strip().encode("utf-8", "replace")).hexdigest()[:16]


def _canon(args: Dict[str, Any]) -> str:
    return json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def _norm_url(u: str) -> str:
    return re.sub(r"[?#].*$", "", str(u or "").strip()).rstrip("/").lower()


class Supervisor:
    def __init__(self, cfg: Optional[SupervisorConfig] = None, *, max_turns: int = 200, max_seconds: float = 3600.0, context: Any = None,
                 probe: Optional[Callable[[], Snapshot]] = None, checker: Optional[Callable[[], DeliverableReport]] = None,
                 specs: Optional[Dict[str, Dict[str, Any]]] = None, emit: Optional[Callable[..., None]] = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or SupervisorConfig()
        self.max_turns, self.max_seconds, self.cm, self.probe, self.checker = max_turns, max_seconds, context, probe, checker
        self.specs, self._emit_cb, self.clock = specs or {}, emit, clock
        self.families: List[_Family] = []
        self.turn = 0
        self.elapsed = 0.0
        self.seq = 0
        self.alt_execs = 0
        self.change_log: List[Tuple[int, int, frozenset]] = []          # (seq, family id, changed relevant paths)
        self._snap: Optional[Snapshot] = self.probe() if self.probe else None
        self.last_output_digest = self._out_digest(self._snap)
        self.last_output_change_turn = 0
        self.last_progress_turn = 0
        self.last_any_change_turn = 0
        self._last_delivery_notice = -99
        self.pending: List[str] = []
        self.exhausted = False
        self.exhausted_family: Optional[int] = None
        self.pushes = 0
        self.stall_checkpoint_turn: Optional[int] = None
        self._noticed: Set[str] = set()
        self._report_cache: Tuple[str, Optional[DeliverableReport]] = ("", None)
        self.counts: Dict[str, int] = {"suppressed": 0, "pointers": 0, "watch_notes": 0, "probes": 0, "executed": 0, "budget_notices": 0, "pushes": 0}
        self.events: List[Dict[str, Any]] = []
        self.verified_ended = False

    # ---- bookkeeping -------------------------------------------------------------------------------------------------------
    def _event(self, kind: str, **kw: Any) -> None:
        if len(self.events) < self.cfg.max_events:
            self.events.append({"turn": self.turn, "kind": kind, **kw})
        if self._emit_cb:
            self._emit_cb("supervisor", action=kind, turn=self.turn, **kw)

    def _out_digest(self, snap: Optional[Snapshot]) -> str:
        if snap is None:
            return ""
        import hashlib
        h = hashlib.sha1()
        for k in sorted(k for k in snap if k.startswith(self.cfg.output_prefix)):
            h.update(f"{k}={snap[k]};".encode())
        return h.hexdigest()[:16]

    def _zone(self, path: str) -> str:
        if path.startswith(self.cfg.output_prefix):
            return "output"
        if path.startswith("task_folder/") or any(part.startswith(".") for part in path.split("/")):
            return "ignore"
        return "scratch"

    @staticmethod
    def _refs(name: str, args: Dict[str, Any]) -> Set[str]:
        text = " ".join(str(v) for v in (args or {}).values() if isinstance(v, str))
        return {m.group(0).lower() for m in re.finditer(r"[\w][\w.\-]*\.[A-Za-z0-9]{1,6}\b", text)}

    def _assign(self, name: str, args: Dict[str, Any]) -> _Family:
        if name in ("bash", "python"):
            norm = _DIGITS.sub("N", _WS.sub(" ", str(args.get("command", ""))))[:3000]
        elif name in ("web_fetch", "web_download"):
            norm = f"{_norm_url(args.get('url'))}|{args.get('offset', 0)}|{args.get('path', '')}"
        else:
            norm = _canon(args)
        ordered = sorted((f for f in self.families if f.tool == name), key=lambda f: -f.last_seq)       # the most recently used family first: loops hit it
        for f in ordered:
            if f.norm == norm:
                return f
        if name in ("bash", "python"):
            for f in ordered:
                if abs(len(f.norm) - len(norm)) > 0.25 * max(len(norm), 1):
                    continue
                m = difflib.SequenceMatcher(None, f.norm, norm, autojunk=False)
                if m.real_quick_ratio() >= self.cfg.family_similarity and m.quick_ratio() >= self.cfg.family_similarity and m.ratio() >= self.cfg.family_similarity:
                    return f
        f = _Family(len(self.families), name, norm, _canon(args)[:300], self._refs(name, args))
        self.families.append(f)
        return f

    def _relevant_change_since(self, fam: _Family) -> bool:
        for seq, fid, paths in self.change_log:
            if seq <= fam.last_seq or fid == fam.id:
                continue
            for p in paths:
                if self._zone(p) == "output" or p.rsplit("/", 1)[-1].lower() in fam.refs:
                    return True
        return False

    def _tool_has(self, tool: str, param: str) -> bool:
        return param in ((self.specs.get(tool) or {}).get("properties") or {})

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.turn)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    @property
    def delivery_mode(self) -> bool:
        rt = max(self.cfg.reserve_turns, int(self.cfg.reserve_fraction * self.max_turns))
        return self.stall_checkpoint_turn is not None or self.remaining_turns <= rt or self.remaining_seconds <= self.cfg.reserve_seconds

    def _thresholds(self) -> Tuple[int, int]:
        f = self.cfg.delivery_stall_factor if self.delivery_mode else 1.0
        return max(2, int(self.cfg.changing_watch_after * f)), max(3, int(self.cfg.changing_suppress_after * f))

    def deliverable_report(self) -> Optional[DeliverableReport]:
        if not self.checker:
            return None
        key = self.last_output_digest
        if self.probe is None or self._report_cache[0] != key or self._report_cache[1] is None:      # without a workspace probe nothing says when the output changed
            self._report_cache = (key, self.checker())
        return self._report_cache[1]

    # ---- turn level ------------------------------------------------------------------------------------------------------------
    def begin_turn(self, turn: int, elapsed: float) -> None:
        self.turn, self.elapsed = turn, elapsed
        used = max(turn / max(1, self.max_turns), elapsed / max(1.0, self.max_seconds))
        rep = self.deliverable_report()
        ok = bool(rep and rep.ok)
        for frac in self.cfg.notice_fractions:
            key = f"frac{frac}"
            if used >= frac and key not in self._noticed and not ok:
                self._noticed.add(key)
                self.pending.append(self._budget_text(rep))
        if self.delivery_mode and not ok and turn - self._last_delivery_notice >= 5:
            self._last_delivery_notice = turn
            self.pending.append(self._budget_text(rep, urgent=True))

    def _budget_text(self, rep: Optional[DeliverableReport], urgent: bool = False) -> str:
        self.counts["budget_notices"] += 1
        status = ("the deliverable check cannot be run" if rep is None else
                  {"none": "no file is in /work/output yet", "draft_only": "only draft-named files are in /work/output",
                   "invalid": "the files in /work/output fail the structural check (" + rep.issues() + ")",
                   "partial": "the output is incomplete (" + rep.issues() + ")",
                   "structurally_valid": "the files pass the structural check"}[rep.status])
        lead = "Budget nearly used up" if urgent else "Budget check"
        tail = (" Stop exploring: build the required files in /work/output now, run them, and verify what they contain." if urgent else
                " Plan so the files exist with time left to run and verify them.")
        return f"[budget] {lead}: {self.remaining_turns} turns and {int(self.remaining_seconds // 60)} min left; {status}.{tail} (This check is structural only.)"

    # ---- call level -------------------------------------------------------------------------------------------------------------
    def before(self, call_id: str, name: str, args: Dict[str, Any], messages: List[Dict[str, Any]]) -> Decision:
        if name in self.cfg.reserve_blocks and self.delivery_mode:
            rep = self.deliverable_report()
            if rep is not None and not rep.ok:
                self.counts["reserve_refusals"] = self.counts.get("reserve_refusals", 0) + 1
                self._event("reserve_refusal", tool=name)
                return Decision(False, (f"[supervisor] Not run: the reserve for building and checking the deliverables has started ({self.remaining_turns} turns and "
                                        f"{int(self.remaining_seconds // 60)} min left) and {('no valid deliverable exists yet: ' + rep.issues()) if rep.status in ('none', 'draft_only', 'invalid') else 'the output is incomplete: ' + rep.issues()}. "
                                        f"Build and save the files from what you already have."), "reserve")
        fam = self._assign(name, args)
        key = _canon(args)
        if name == "view_image" and isinstance(args.get("path"), str) and not image_is_visible(messages, args["path"]):
            fam.identical = 0                                         # the earlier image is no longer in view: looking again is the only way to see it, not a repeat
            if fam.state == "suppressed":
                self._lift(fam, "image no longer in view")
            return Decision(True)
        if fam.state == "suppressed":
            if self._relevant_change_since(fam):
                self._lift(fam, "relevant change")
            elif self.turn >= fam.ttl_until:
                fam.probing = True
                self.counts["probes"] += 1
                self._event("probe", family=fam.id, tool=name)
                return Decision(True, kind="probe")
            else:
                return self._suppress(fam, call_id, name, key, messages)
        entry = fam.ops.get(key)
        if entry and not self._relevant_change_since(fam):
            view = self._view(messages, entry[0])
            if view and view.state == "unavailable":
                return Decision(True)                                 # the earlier result is gone and cannot be recovered: running again is the only way back
            if view and view.state in ("archived", "clipped") and getattr(view, "archive_id", None) and fam.pointers < self.cfg.pointer_answers:
                fam.pointers += 1
                fam.stall += 1
                self.counts["pointers"] += 1
                self._event("pointer", family=fam.id, archive=view.archive_id)
                return Decision(False, self._pointer_text(name, entry, view), "pointer")
            if fam.identical >= self.cfg.identical_suppress_after:
                self._to_suppressed(fam, "identical result repeated")
                return self._suppress(fam, call_id, name, key, messages)
        return Decision(True)

    def _view(self, messages: List[Dict[str, Any]], call_id: str) -> Any:
        if self.cm is None or not hasattr(self.cm, "visibility"):
            return None
        try:
            return self.cm.visibility(messages, call_id)
        except Exception:                                                # noqa: BLE001 - a context manager bug must not stop the run
            return None

    def _pointer_text(self, name: str, entry: Tuple[str, int, str], view: Any) -> str:
        how = f'read_archive("{view.archive_id}")' if self._tool_has("read_archive", "id") or "read_archive" in self.specs else f"the archived text {view.archive_id}"
        if view.state == "clipped":
            more = f" or call {name} again with a larger offset" if self._tool_has(name, "offset") else ""
            return f"[supervisor] Not run again: the identical call at turn {entry[1]} returned a result that was shortened for context; the full text is archived as {view.archive_id}: {how}{more}."
        return f"[supervisor] Not run again: the identical call at turn {entry[1]} returned a result that has since been moved out of the conversation; it is archived as {view.archive_id}: {how}."

    def _to_suppressed(self, fam: _Family, why: str) -> None:
        fam.state = "suppressed"
        fam.ttl = fam.ttl * 2 if fam.ttl else self.cfg.suppress_ttl_turns
        fam.ttl = min(fam.ttl, 60)
        fam.ttl_until = self.turn + fam.ttl
        self._event("suppress", family=fam.id, tool=fam.tool, why=why, ttl=fam.ttl, example=fam.example[:120])

    def _stall_checkpoint(self, fam: _Family) -> None:
        """One structured message when a refused operation is requested again and again and nothing deliverable exists: what stalled, what is missing, what is now unavailable, what to do. It also starts
        delivery mode (exploration tools in ``reserve_blocks`` are refused) instead of waiting for the end-of-budget reserve."""
        rep = self.deliverable_report()
        if rep is None or rep.ok:
            return
        self.stall_checkpoint_turn = self.turn
        self.counts["stall_checkpoints"] = self.counts.get("stall_checkpoints", 0) + 1
        missing = ", ".join(rep.unmatched_names or [("." + e) for e in rep.missing_exts]) or rep.issues()
        text = (f"[checkpoint] Stalled at turn {self.turn} of {self.max_turns}: `{fam.example[:100]}` has run {fam.execs} times with an unchanged result and is refused from now on. "
                f"Still missing: {missing} ({rep.issues()}). Unavailable now: {', '.join(self.cfg.reserve_blocks)} (the remaining {self.remaining_turns} turns / {int(self.remaining_seconds // 60)} min are for "
                f"building files). Next: write one script that builds every file you can from the data you already have (a partial file is better than none), run it, and check what it wrote.")
        self.pending.append(text)
        self._event("stall_checkpoint", family=fam.id, missing=missing[:200], remaining_turns=self.remaining_turns)

    def _lift(self, fam: _Family, why: str) -> None:
        fam.state, fam.stall, fam.identical, fam.watched, fam.ttl, fam.probing, fam.ignored = "normal", 0, 0, False, 0, False, 0
        self._event("lift", family=fam.id, why=why)

    def _suppress(self, fam: _Family, call_id: str, name: str, key: str, messages: List[Dict[str, Any]]) -> Decision:
        if self.alt_execs != fam.last_alt_seen:
            fam.ignored = 0
        fam.last_alt_seen = self.alt_execs
        fam.ignored += 1
        fam.suppressed_total += 1
        self.counts["suppressed"] += 1
        if self.cfg.stall_checkpoint and self.stall_checkpoint_turn is None and fam.ignored >= self.cfg.stall_checkpoint_ignored:
            self._stall_checkpoint(fam)
        idle = self.turn - self.last_progress_turn
        if self.delivery_mode:
            idle *= 2                                                      # the reserve for delivering is being used up: patience halves
        if fam.ignored >= self.cfg.exhaust_after and idle >= self.cfg.exhaust_min_idle_turns:
            self._event("exhausted", family=fam.id, tool=name, suppressed=fam.suppressed_total, idle_turns=idle, example=fam.example[:160], ends_run=self.cfg.end_on_exhaustion)
            self.counts["exhaustion_reached"] = self.counts.get("exhaustion_reached", 0) + 1
            if self.cfg.end_on_exhaustion:
                self.exhausted, self.exhausted_family = True, fam.id
                return Decision(False, f"[supervisor] Recovery exhausted: this operation was refused {fam.ignored} times in a row, nothing has progressed for {idle} turns and no different operation was tried. The run ends here; work and files are preserved.", "exhausted", True)
        first = fam.suppressed_total <= self.cfg.max_visible_notes
        if first:
            alt = ""
            entry = fam.ops.get(key)
            if entry:
                alt = f" Its result is the one from turn {entry[1]}."
            text = (f"[supervisor] Not run: this operation (same tool, same intent) has run {fam.execs} times with no change to the output or to the files it uses.{alt} "
                    f"Do something materially different, or give the final answer with what you have. It may run again in {max(0, fam.ttl_until - self.turn)} turns or once a relevant file changes.")
        else:
            text = f"[supervisor] not run (repeat #{fam.suppressed_total}; see the earlier note)"
        self._event("suppressed", family=fam.id, tool=name, n=fam.suppressed_total)
        return Decision(False, text, "suppressed")

    def after(self, call_id: str, name: str, args: Dict[str, Any], text: str, ok: bool) -> str:
        """Record an executed operation; returns the text the model should see (a note or budget notice may be appended)."""
        fam = self._assign(name, args)
        self.seq += 1
        self.counts["executed"] += 1
        fp = fingerprint(text)
        is_err = (not ok) or bool(_ERROR.search(text or ""))
        pre = self._snap
        post = self.probe() if self.probe else None
        own_paths: Set[str] = set()
        out_changed_by_own = False
        if pre is not None and post is not None:
            changed = {p for p in set(pre) | set(post) if pre.get(p) != post.get(p)}
            relevant = {p for p in changed if self._zone(p) in ("output", "scratch")}
            own_paths = relevant
            if relevant:
                self.change_log.append((self.seq, fam.id, frozenset(relevant)))
            self._snap = post
            out_now = self._out_digest(post)
            if out_now != self.last_output_digest:
                out_changed_by_own = True
                self.last_output_digest = out_now
                self.last_output_change_turn = self.turn
                self.last_progress_turn = self.turn
        foreign = self._relevant_change_since(fam)
        was_error_change = is_err and fam.last_error_fp is not None and fp != fam.last_error_fp
        transition = (fam.last_fp is not None and fam.last_error_fp is not None and not is_err)
        progressed = fam.execs == 0 or foreign or out_changed_by_own or was_error_change or transition
        if progressed:
            fam.stall = 1
            fam.watched = False
            fam.identical = 1
            if fam.state == "suppressed":
                self._lift(fam, "probe made progress" if fam.probing else "progress")
        else:
            fam.stall += 1
            fam.identical = fam.identical + 1 if fp == fam.last_fp else 1
        if fam.probing and not progressed:
            self._to_suppressed(fam, "probe made no progress")
        fam.probing = False
        fam.last_fp = fp
        fam.last_error_fp = fp if is_err else None
        fam.last_seq = self.seq
        fam.execs += 1
        fam.ops[_canon(args)] = (call_id, self.turn, fp)
        if progressed:
            self.last_progress_turn = self.turn
        if own_paths:
            self.last_any_change_turn = self.turn
        self.alt_execs += 1
        out = text
        watch, suppress = self._thresholds()
        if not progressed and not is_err and fam.identical < 2:        # changing, successful results with no relevant change: slower thresholds
            if fam.stall >= suppress and fam.state == "normal":
                self._to_suppressed(fam, "changing results, no relevant change")
            elif fam.stall >= watch and not fam.watched:
                fam.watched = True
                self.counts["watch_notes"] += 1
                out += (f"\n[supervisor] This is the {fam.stall}th run of the same kind of operation with no change to the output or to the files it uses. "
                        f"Results that look different are not progress if they do not change what you will deliver; consider a different approach.")
                self._event("watch", family=fam.id, stall=fam.stall)
        if self.pending:
            out += "\n" + "\n".join(self.pending)
            self.pending = []
        return out

    def attach_pending(self, text: str) -> str:
        """Append queued budget notices to a text that did not come from an executed operation (a suppressed call), so notices are never starved."""
        if self.pending:
            text += "\n" + "\n".join(self.pending)
            self.pending = []
        return text

    # ---- ending ---------------------------------------------------------------------------------------------------------------
    def on_finish_attempt(self) -> FinishDecision:
        """The model replied without a tool call. Accept, or send it back (bounded) when the deliverables are missing or invalid and there is budget."""
        rep = self.deliverable_report()
        if rep is None or rep.ok:
            return FinishDecision("accept")
        if self.pushes < self.cfg.checkpoint_pushes and self.remaining_turns >= self.cfg.min_turns_for_push and self.remaining_seconds >= self.cfg.min_seconds_for_push:
            self.pushes += 1
            self.counts["pushes"] += 1
            self._event("push", status=rep.status, issues=rep.issues()[:200])
            return FinishDecision("push", (f"[harness check] You ended your turn, but the deliverable check found: {rep.issues()}. "
                                           f"{self.remaining_turns} turns and {int(self.remaining_seconds // 60)} minutes remain. Create or repair the required files in /work/output and "
                                           f"verify them, or, if that is impossible, say exactly which input or capability is missing. (This check is structural; it does not judge correctness.)"))
        return FinishDecision("accept")

    def stop_reason(self) -> Optional[str]:
        if self.exhausted:
            return "blocked"
        rep = self.deliverable_report()
        if rep is not None and rep.ok and self.probe is not None:          # without workspace observation nobody can say how long nothing has changed
            idle = self.turn - max(self.last_output_change_turn, self.last_any_change_turn)
            if idle >= self.cfg.verified_idle_notice and "verified" not in self._noticed:
                self._noticed.add("verified")
                self.pending.append("[budget] The output files pass the structural check and nothing has changed for several turns. Unless you know of a specific defect to fix, reply with a short summary and stop. (This check is structural; it does not judge correctness.)")
            known = bool(rep.required_exts) and not rep.unmatched_names          # only end the run when the task named its outputs and all of them are present and valid
            if known and idle >= self.cfg.verified_idle_end:
                self.verified_ended = True
                return "verified_complete"
        return None

    def report(self) -> Dict[str, Any]:
        rep = self.deliverable_report()
        fams = sorted(self.families, key=lambda f: -(f.execs + f.suppressed_total))[:8]
        return {"exhausted": self.exhausted, "verified_complete": self.verified_ended, "counts": dict(self.counts), "last_progress_turn": self.last_progress_turn,
                "last_output_change_turn": self.last_output_change_turn, "pushes": self.pushes,
                "completion": rep.to_dict() if rep else None,
                "families": [{"id": f.id, "tool": f.tool, "state": f.state, "executions": f.execs, "suppressed": f.suppressed_total, "example": f.example[:160]} for f in fams],
                "events": self.events[-60:]}
