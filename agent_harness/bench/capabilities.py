"""What the benchmark environment can and cannot do, stated to the model from facts (probed in the sandbox, or true by construction).

This module is the benchmark's honest statement of its environment (what tools exist, what the network allows) and the workspace observation the supervisor
(``agent_harness.supervisor``) needs. Nothing here names a task, and no workaround is hardcoded: the model is told what exists and chooses.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence


DELIVERABLE_EXTS = (".xlsx", ".xls", ".docx", ".doc", ".pptx", ".pdf", ".csv", ".ods", ".odt", ".odp", ".md", ".txt", ".json", ".zip")
PROBES = ("tesseract", "pdftotext", "pandoc", "libreoffice", "soffice", "convert", "curl", "wget")


def probe_environment(sandbox, run_timeout: int = 20) -> List[str]:
    """Names from PROBES that exist in the sandbox (``command -v``). A failed probe means "not found", never an exception."""
    found = []
    for tool in PROBES:
        try:
            out = str(sandbox.run(f"command -v {tool} >/dev/null 2>&1 && echo yes || echo no", run_timeout))
        except Exception:                    # noqa: BLE001 - a probe must never stop a run
            continue
        if "yes" in out.split():
            found.append(tool)
    return found


def environment_note(found: Sequence[str], *, web_paging: bool, web_downloads: bool, network: str = "none", edit: bool = False) -> str:
    net = ("- bash reaches the network only through a filtering proxy (HTTP and HTTPS); blocked destinations answer 403; direct sockets do not work."
           if network == "egress" else "- bash has no network access; web content is reachable only through the web tools.")
    lines = ["Environment facts:", net,
             "- You cannot look at images yourself; image and scan content has to be processed with code or tools that exist in the sandbox.",
             "- Command-line tools found in the sandbox: " + (", ".join(found) if found else "none of tesseract, pdftotext, pandoc, libreoffice, curl, wget") + "."]
    if web_paging:
        lines.append("- web_fetch returns 12,000 characters at a time; use its offset argument to read further.")
    if web_downloads:
        lines.append("- web_download saves a URL's file into /work so bash can read it (data files up to 50 MB).")
    if edit:
        lines.append("- edit_file changes part of an existing file in place; write_file replaces a whole file.")
    lines.append("- Deliverables must be written into /work/output. Files elsewhere are not collected.")
    return "\n".join(lines)


def workspace_fingerprint(root: Path, skip: Sequence[str] = ("task_folder",)) -> str:
    """Hash of (relative path, size, mtime) of every file under ``root`` except the static inputs. A change means the workspace moved."""
    h = hashlib.sha1()
    root = Path(root)
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not (Path(dirpath) == root and d in skip) and not d.startswith("."))
        for f in sorted(files):
            try:
                st = os.stat(os.path.join(dirpath, f))
            except OSError:
                continue
            h.update(f"{os.path.relpath(os.path.join(dirpath, f), root)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def stray_deliverables(workspace: Path, output_dir: Optional[Path] = None) -> List[str]:
    """Deliverable-looking files that ended up outside ``output/`` and ``task_folder/`` (reported, never moved or copied)."""
    ws = Path(workspace)
    out = Path(output_dir) if output_dir else ws / "output"
    found = []
    for dirpath, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if Path(dirpath, d) not in (out, ws / "task_folder") and not d.startswith(".")]
        for f in files:
            if f.lower().endswith(DELIVERABLE_EXTS):
                found.append(os.path.relpath(os.path.join(dirpath, f), ws))
    return sorted(found)


def workspace_snapshot(root: Path, max_hash_bytes: int = 4_000_000) -> dict:
    """{relative path: content digest} of every file under ``root``. Files above ``max_hash_bytes`` are identified by size and mtime (a rewrite of identical
    bytes cannot be told from a change there). Digests are cached by (size, mtime) so repeated snapshots are cheap."""
    out = {}
    root = Path(root)
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            full = os.path.join(dirpath, f)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = Path(os.path.relpath(full, root)).as_posix()      # logical, OS-neutral names
            key = (full, st.st_size, st.st_mtime_ns)
            d = _DIGESTS.get(key)
            if d is None:
                if st.st_size > max_hash_bytes:
                    d = f"big:{st.st_size}:{st.st_mtime_ns}"
                else:
                    try:
                        d = hashlib.sha1(open(full, "rb").read()).hexdigest()[:16]
                    except OSError:
                        continue
                _DIGESTS[key] = d
            out[rel] = d
    return out


_DIGESTS: dict = {}


def supervision_for(ws: Path, task_text: str = "", input_names=(), cfg=None):
    """Supervision for a benchmark workspace: content-based workspace observation and, when the task text names its outputs, the deliverable check."""
    from ..completion import check_deliverables
    from ..loop import Supervision
    from ..supervisor import SupervisorConfig
    ws = Path(ws)
    return Supervision(cfg or SupervisorConfig(), probe=lambda: workspace_snapshot(ws),
                       checker=lambda: check_deliverables(ws / "output", task_text, input_names), inputs=tuple(input_names))
