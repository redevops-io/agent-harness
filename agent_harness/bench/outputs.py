"""Recoverable tool output and a content-hash ledger of what the agent leaves in its workspace.

* ``OutputStore``: when a tool result is too long for the model's context, the **full** text is saved under ``/work/.tool_output/`` (inside the workspace, so
  bash can grep or slice it) and the model is told the path and how to page through it with ``read_file(path, offset, limit)``. Nothing is lost to truncation.
* ``FileLedger``: after each tool call, new or changed files under ``/work`` (not inputs, not hidden directories) are hashed and appended to
  ``/work/.artifacts/files.jsonl`` with tool, size, SHA-256 and time: download hashes and workspace persistence for every route a file can arrive by
  (web_download, a bash command, write_file, edit_file).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STORE_DIR = ".tool_output"
LEDGER_DIR = ".artifacts"
SKIP_TOP = ("task_folder",)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


class OutputStore:
    def __init__(self, workspace: Path):
        self.ws = Path(workspace)
        self.dir = self.ws / STORE_DIR
        self.n = 0

    def save(self, kind: str, text: str) -> str:
        """Write ``text`` and return its path as the agent sees it (``/work/.tool_output/<kind>-<n>.txt``)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o777)
        self.n += 1
        name = f"{kind}-{self.n:04d}.txt"
        p = self.dir / name
        tmp = self.dir / (name + ".tmp")
        with open(tmp, "w", encoding="utf-8", errors="replace", newline="") as f:      # no newline translation: the bytes are what the model was shown
            f.write(text)
        os.chmod(tmp, 0o666)
        os.replace(tmp, p)
        return f"/work/{STORE_DIR}/{name}"

    def clip(self, kind: str, text: str, limit: int) -> str:
        """``text`` unchanged if it fits; otherwise head and tail with a marker naming the saved full text."""
        if len(text) <= limit:
            return text
        path = self.save(kind, text)
        head, tail = limit * 2 // 3, limit // 3
        return (text[:head] + f"\n[... {len(text) - head - tail} of {len(text)} chars omitted from the middle; the full text is saved at {path}. "
                f"Read it in pages with read_file(path, offset, limit), or search it with bash grep ...]\n" + text[-tail:])


class FileLedger:
    def __init__(self, workspace: Path, max_hash_bytes: int = 500_000_000):
        self.ws = Path(workspace)
        self.max_hash_bytes = max_hash_bytes
        self._seen: Dict[str, Tuple[int, int]] = {}
        self.path = self.ws / LEDGER_DIR / "files.jsonl"
        self.records: List[dict] = []

    def _walk(self):
        for dirpath, dirs, files in os.walk(self.ws):
            top = Path(dirpath) == self.ws
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (top and d in SKIP_TOP))
            for f in sorted(files):
                yield Path(dirpath) / f

    def record(self, tool: str, only: Optional[Path] = None) -> List[dict]:
        """Hash files that are new or changed since the last call; returns the new ledger rows."""
        new = []
        for p in ([only] if only else self._walk()):
            try:
                st = p.stat()
            except OSError:
                continue
            rel = p.relative_to(self.ws).as_posix()
            key = (st.st_size, st.st_mtime_ns)
            if self._seen.get(rel) == key:
                continue
            self._seen[rel] = key
            if st.st_size > self.max_hash_bytes:
                digest = "skipped:too-large"
            else:
                try:
                    digest = sha256_file(p)
                except OSError:
                    continue
            new.append({"t": round(time.time(), 1), "tool": tool, "path": "/work/" + rel, "size": st.st_size, "sha256": digest})
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o777)
            with open(self.path, "a", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r) + "\n")
            self.records += new
        return new
