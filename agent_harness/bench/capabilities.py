"""What the benchmark environment can and cannot do, stated to the model from facts (probed in the sandbox, or true by construction).

Used with the progress-aware recovery policy (``agent_harness.progress``): the policy is generic, this module is the benchmark's honest statement of its
environment. Nothing here names a task, and no workaround is hardcoded: the model is told what exists and chooses.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence

from ..progress import Capability, RecoveryPolicy

DELIVERABLE_EXTS = (".xlsx", ".xls", ".docx", ".doc", ".pptx", ".pdf", ".csv", ".ods", ".odt", ".odp", ".md", ".txt", ".json", ".zip")
PROBES = ("tesseract", "pdftotext", "pandoc", "libreoffice", "soffice", "convert")


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


def environment_note(found: Sequence[str], *, web_paging: bool, web_downloads: bool) -> str:
    lines = ["Environment facts:",
             "- bash has no network access; web content is reachable only through the web tools.",
             "- You cannot look at images yourself; image and scan content has to be processed with code or tools that exist in the sandbox.",
             "- Command-line tools found in the sandbox: " + (", ".join(found) if found else "none of tesseract, pdftotext, pandoc, libreoffice") + "."]
    if web_paging:
        lines.append("- web_fetch returns 15,000 characters at a time; use its offset argument to read further.")
    if web_downloads:
        lines.append("- web_download saves a URL's file into /work so bash can read it (data files up to 25 MB).")
    lines.append("- Deliverables must be written into /work/output. Files elsewhere are not collected.")
    return "\n".join(lines)


def capabilities(*, web_paging: bool, web_downloads: bool) -> List[Capability]:
    net = "bash has no network access. Web content is available only through web_fetch"
    if web_paging:
        net += " (read long pages with its offset argument)"
    if web_downloads:
        net += " and web_download (saves a file into /work for bash to read)"
    return [Capability("network", (r"Temporary failure in name resolution", r"Network is unreachable", r"Name or service not known", r"No address associated"), net + "."),
            Capability("image", (r"does not support image", r"cannot read images?", r"image input"),
                       "The model cannot view images. Use code or sandbox tools to process image content.")]


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


def recovery_policy(ws: Path, *, web_paging: bool = False, web_downloads: bool = False, with_capabilities: bool = True, **overrides) -> RecoveryPolicy:
    return RecoveryPolicy(capabilities=tuple(capabilities(web_paging=web_paging, web_downloads=web_downloads)) if with_capabilities else (),
                          progress_probe=lambda: workspace_fingerprint(ws), **overrides)
