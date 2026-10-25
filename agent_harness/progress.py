"""Progress-aware recovery for the agent loop (opt-in; ``run_loop(..., recovery=RecoveryPolicy(...))``).

The failure this addresses: a model that stops making progress while the harness keeps executing and paying for turns. Three situations look alike
(the same call again and again) and need different handling, so the tracker classifies from evidence, not from the call text alone:

* **redundant**: the same normalized call returned the same result (fingerprint) and nothing in the workspace changed since. Re-running cannot help;
  the call is not executed again and the model is told so, plainly, with what is known.
* **repeated error**: the same normalized call failed with the same error several times and nothing changed. Same treatment; the message names the error.
* **polling**: the same call keeps returning *different* results (a status check, a growing file). Legitimate; allowed up to a separate bound.

A *capability* hit (the result text matches a pattern the harness registered, for example "no network") appends a note once, with the alternatives the harness
supplies. The policy knows nothing about OCR or the web: the harness states what its environment can and cannot do.

Recovery has its own budget, separate from the turn limit: only steps that made no progress (blocked calls and executions of an already-seen result) count, a progress
event (workspace fingerprint changed, or a new call returned new information) resets the streak, and when the streak reaches the budget the run ends as ``blocked`` with a
structured, truthful list of blockers. Nothing is ever forced: no deliverable is written or submitted on the model's behalf, and the one-time checkpoint note only
offers a choice ("finish with what you hold, or state what you cannot obtain").
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

_WS = re.compile(r"\s+")
_HARNESS_NOTE = re.compile(r"\n?\[(?:note|no progress|checkpoint|environment)[^\]]*\]", re.S)


def _norm_value(key: str, v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if key in ("url", "href") or re.match(r"https?://", s):
            u = urlsplit(s)
            if u.scheme and u.netloc:
                return urlunsplit((u.scheme.lower(), u.netloc.lower(), u.path.rstrip("/") or "/", u.query, ""))
        return _WS.sub(" ", s)
    if isinstance(v, dict):
        return {k: _norm_value(k, x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm_value(key, x) for x in v]
    return v


def normalize_call(name: str, args: Dict[str, Any]) -> str:
    """Canonical text of a tool call: sorted keys, whitespace collapsed in strings, URL fragments and trailing slashes dropped."""
    return name + ":" + json.dumps({k: _norm_value(k, v) for k, v in (args or {}).items()}, sort_keys=True, ensure_ascii=False)


def fingerprint(text: str) -> str:
    """Hash of a tool result that ignores whitespace layout and any note this module appended."""
    return hashlib.sha1(_WS.sub(" ", _HARNESS_NOTE.sub("", text or "")).strip().encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class Capability:
    """Something the environment cannot do, recognised from result text, with what the harness offers instead (the harness must state it truthfully)."""
    name: str
    patterns: Tuple[str, ...]
    note: str


@dataclass
class RecoveryPolicy:
    redundant_after: int = 2          # identical result this many times in a row -> the call is redundant
    error_after: int = 3              # identical error this many times in a row -> the call is not retried unchanged
    poll_limit: int = 30              # identical call with changing results, per call
    checkpoint_after: int = 8         # no-progress steps in a row at which the one-time checkpoint note is added
    budget: Optional[int] = 20        # no-progress steps in a row after which the run ends as 'blocked'; None disables the hard stop (guidance, classification and telemetry stay)
    capabilities: Sequence[Capability] = ()
    progress_probe: Optional[Callable[[], str]] = None   # returns a fingerprint of the workspace; a change counts as progress


@dataclass
class _CallState:
    last_fp: Optional[str] = None
    same: int = 0                     # identical results in a row
    err_fp: Optional[str] = None
    err_streak: int = 0
    changes: int = 0                  # times the result differed from the previous one (polling)
    epoch: int = 0                    # workspace epoch of the last execution
    runs: int = 0


class ProgressTracker:
    def __init__(self, policy: RecoveryPolicy):
        self.p = policy
        self.state: Dict[str, _CallState] = {}
        self.seen_fp: set = set()
        self.epoch = 0
        self._probe = policy.progress_probe() if policy.progress_probe else None
        self.streak = 0               # consecutive steps without progress
        self.max_streak = 0
        self.total_blocked = 0
        self.checkpointed = False
        self.noted: set = set()
        self.blockers: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {"redundant": 0, "repeated_error": 0, "poll_exhausted": 0, "capability": 0, "progress": 0}

    def _refresh(self) -> bool:
        """Probe the workspace; a change starts a new epoch and counts as progress."""
        if not self.p.progress_probe:
            return False
        now = self.p.progress_probe()
        if now == self._probe:
            return False
        self._probe, self.epoch, self.streak = now, self.epoch + 1, 0
        self.counts["progress"] += 1
        return True

    # ---- before executing -------------------------------------------------------------------------------------------
    def before(self, name: str, args: Dict[str, Any]) -> Optional[str]:
        """A message to return instead of running the call, or None to run it."""
        self._refresh()
        key = normalize_call(name, args)
        st = self.state.get(key)
        if not st:
            return None
        unchanged = st.epoch == self.epoch                    # nothing in the workspace moved since this call last ran
        kind = None
        if unchanged and st.err_streak >= self.p.error_after:
            kind = "repeated_error"
        elif unchanged and st.same >= self.p.redundant_after:
            kind = "redundant"
        elif st.changes >= self.p.poll_limit:
            kind = "poll_exhausted"
        if not kind:
            return None
        self.counts[kind] += 1
        self.total_blocked += 1
        self.streak += 1
        self.max_streak = max(self.max_streak, self.streak)
        self._blocker(kind, name, key, st)
        msg = {"redundant": f"[no progress] {name} was already run with these exact arguments {st.same} times and returned the same result each time; "
                            f"nothing in the workspace has changed since. It was not run again.",
               "repeated_error": f"[no progress] {name} failed {st.err_streak} times in a row with the same error and nothing in the workspace has changed since. "
                                 f"It was not run again unchanged.",
               "poll_exhausted": f"[no progress] {name} has been repeated {st.changes} times with changing results; the polling limit is {self.p.poll_limit}. "
                                 f"It was not run again."}[kind]
        notes = [c.note for c in self.p.capabilities if c.name in self.noted]
        msg += " Change the arguments, use a different tool, or work with what you already have."
        if notes:
            msg += " " + " ".join(notes)
        return msg + self._checkpoint()

    # ---- after executing --------------------------------------------------------------------------------------------
    def after(self, name: str, args: Dict[str, Any], text: str, ok: bool) -> str:
        """Record an executed call; returns the text to show the model (a capability note may be appended)."""
        key = normalize_call(name, args)
        st = self.state.setdefault(key, _CallState())
        fp = fingerprint(text)
        first = st.runs == 0
        if st.last_fp is not None and fp != st.last_fp:
            st.changes += 1
        st.same = (st.same + 1 if fp == st.last_fp else 1) if ok else 0        # errors are judged by their own streak (a transient failure may deserve a retry)
        st.last_fp = fp
        if not ok:
            st.err_streak = st.err_streak + 1 if fp == st.err_fp else 1
            st.err_fp = fp
        else:
            st.err_streak, st.err_fp = 0, None
        st.runs += 1
        progressed = self._refresh()
        if ok and first and fp not in self.seen_fp:
            progressed = True                                  # a new call that returned information not seen before
            self.counts["progress"] += 1
        # only an identical call that gave the identical answer again is a wasted step; a different call that happens to return the same
        # (for example an empty list) is exploration, not a loop
        redundant_exec = not progressed and ((ok and st.same > 1) or (not ok and st.err_streak > 1))
        self.seen_fp.add(fp)
        st.epoch = self.epoch
        if progressed:
            self.streak = 0
        elif redundant_exec:
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)
        out = text
        for c in self.p.capabilities:
            if re.search("|".join(c.patterns), text or "", re.I):
                self.counts["capability"] += 1
                if c.name not in self.noted:
                    self.noted.add(c.name)
                    self._blocker("capability", name, key, st, c.name)
                    out += f"\n[environment: {c.note}]"
        return out

    def _checkpoint(self) -> str:
        if self.checkpointed or self.streak < self.p.checkpoint_after:
            return ""
        self.checkpointed = True
        return (f"\n[checkpoint] {self.streak} recent steps made no progress. Decide now: if the files and facts you already hold are enough, "
                f"write the deliverable from them and state any limits in your final answer. If a required input cannot be obtained with the tools you "
                f"have, say exactly which one in your final answer instead of retrying.")

    def _blocker(self, kind: str, name: str, key: str, st: _CallState, detail: str = "") -> None:
        if any(b["kind"] == kind and b["call"] == key for b in self.blockers):
            return
        self.blockers.append({"kind": kind, "tool": name, "call": key[:200], "detail": detail, "same_result": st.same, "error_streak": st.err_streak})

    @property
    def exhausted(self) -> bool:
        return self.p.budget is not None and self.streak >= self.p.budget

    def report(self) -> Dict[str, Any]:
        return {"counts": dict(self.counts), "streak": self.streak, "max_streak": self.max_streak, "hard_stop": self.p.budget is not None,
                "blocked_calls": self.total_blocked, "exhausted": self.exhausted,
                "blockers": list(self.blockers)}
