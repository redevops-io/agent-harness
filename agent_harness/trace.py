"""Opt-in run trace: what the model was actually shown at every call, and what the canonical (original) messages were.

Compaction changes the model-visible projection of the conversation. To tell a context-management failure from an agent, tool
or evaluation failure afterwards, a run needs three things the plain trajectory does not keep:

* the **canonical** message as first appended (never altered by a manager), content-addressed in a blob store;
* the **projection** shown to the model at each call (per message: hash and length shown versus canonical, rewritten or not,
  archive ids referenced), plus which messages a compaction changed since the previous call;
* **full** tool results and tool-call arguments (hash, length, admitted length), and the full assistant text.

Large texts live once in ``blobs/<sha16>.txt``; ``trace.jsonl`` holds only hashes and numbers. Nothing here affects the run: the
recorder is only called when a ``TraceRecorder`` is passed to ``run_loop``, and a write failure is reported once and then ignored.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ARCHIVE_REF = re.compile(r"A\d{4}")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def message_text(m: Dict[str, Any]) -> str:
    """Everything the model sees of one message: content plus the exact tool-call arguments."""
    calls = m.get("tool_calls") or []
    if not calls:
        return m.get("content") or ""
    return json.dumps({"content": m.get("content") or "", "tool_calls": [
        {"id": c.get("id"), "name": c["function"]["name"], "arguments": c["function"].get("arguments")} for c in calls]},
        sort_keys=True)


class TraceRecorder:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        (self.dir / "blobs").mkdir(parents=True, exist_ok=True)
        self._log = open(self.dir / "trace.jsonl", "a", encoding="utf-8")
        self._t0 = time.time()
        self._canon: Dict[int, str] = {}        # message index -> sha of the canonical text
        self._canon_len: Dict[int, int] = {}
        self._prev: Dict[int, str] = {}         # message index -> sha shown at the previous call
        self._failed = False

    # -- storage ---------------------------------------------------------------------------------------------------
    def blob(self, text: str) -> str:
        h = sha(text)
        p = self.dir / "blobs" / f"{h}.txt"
        if not p.exists():
            p.write_text(text, encoding="utf-8")
        return h

    def _emit(self, kind: str, **kw: Any) -> None:
        try:
            self._log.write(json.dumps({"t": round(time.time() - self._t0, 1), "kind": kind, **kw}) + "\n")
            self._log.flush()
        except Exception as e:                    # a recorder problem must never end a long run
            if not self._failed:
                self._failed = True
                print(f"[trace] disabled after error: {e}", flush=True)

    def _safe(self, fn, *a, **kw):
        if self._failed:
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:
            self._failed = True
            print(f"[trace] disabled after error: {e}", flush=True)
            return None

    # -- events ----------------------------------------------------------------------------------------------------
    def add(self, index: int, message: Dict[str, Any]) -> None:
        """Called once, at the moment a message is appended (before any manager can rewrite it)."""
        def go():
            text = message_text(message)
            self._canon[index] = self.blob(text)
            self._canon_len[index] = len(text)
        self._safe(go)

    def projection(self, turn: int, messages: List[Dict[str, Any]], estimated_tokens: int, last_prompt: int) -> None:
        """Called right before each model call: what is about to be sent."""
        def go():
            items, changed, rewritten = [], [], 0
            for i, m in enumerate(messages):
                text = message_text(m)
                shown = sha(text)
                canon = self._canon.get(i)
                if canon is None:                 # a message the loop did not announce (should not happen); keep it visible
                    canon = self._canon[i] = self.blob(text)
                    self._canon_len[i] = len(text)
                if shown != canon:
                    rewritten += 1
                    self.blob(text)
                if self._prev.get(i) not in (None, shown):
                    changed.append(i)
                self._prev[i] = shown
                refs = sorted(set(_ARCHIVE_REF.findall(text[:400]))) if shown != canon else []
                items.append([i, m.get("role", "?")[0], shown, canon, len(text), self._canon_len.get(i, 0)] + ([refs] if refs else []))
            self._emit("projection", turn=turn, n=len(messages), est_tokens=estimated_tokens, last_prompt=last_prompt,
                       rewritten=rewritten, changed_since_prev=changed, items=items)
        self._safe(go)

    def assistant(self, turn: int, message: Dict[str, Any], usage: Dict[str, Any], finish: Optional[str]) -> None:
        def go():
            calls = [{"id": c.get("id"), "name": c["function"]["name"], "args_sha": self.blob(c["function"].get("arguments") or ""),
                      "args_len": len(c["function"].get("arguments") or "")} for c in message.get("tool_calls") or []]
            self._emit("assistant_full", turn=turn, content_sha=self.blob(message.get("content") or ""),
                       content_len=len(message.get("content") or ""), calls=calls, usage=usage, finish=finish)
        self._safe(go)

    def tool_result(self, turn: int, name: str, args: Dict[str, Any], full: str, admitted: str) -> None:
        def go():
            self._emit("tool_full", turn=turn, name=name, args_sha=self.blob(json.dumps(args, sort_keys=True, default=str)),
                       out_sha=self.blob(full), out_len=len(full), admitted_sha=sha(admitted), admitted_len=len(admitted),
                       cut=len(admitted) < len(full), refs=sorted(set(_ARCHIVE_REF.findall(admitted[-400:] + admitted[:400]))))
        self._safe(go)

    def note(self, kind: str, **kw: Any) -> None:
        self._safe(self._emit, kind, **kw)

    def close(self) -> None:
        try:
            self._log.close()
        except Exception:
            pass
