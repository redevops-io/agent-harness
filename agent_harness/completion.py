"""Deliverable checks and the completion state of a run (stdlib only).

``check_deliverables`` answers one question honestly: what is in the output directory, and is each file structurally usable? It never says a task is
*correct*: the strongest status is ``structurally_valid``. Statuses (worst to best): none, draft_only, invalid, partial, structurally_valid.

* *Draft or intermediate* files are recognised by name (draft, tmp, scratch, wip, backup, ``~$`` lock files, hidden files) and never count as final output.
* *Invalid* means the file cannot be what its extension says or has no substance: an xlsx without a populated sheet, a docx without text, a csv without a
  data row, a SQLite file whose tables are all empty, a zero-byte anything.
* *Required artifacts* come from the task text (``expected_artifacts``): file names the task asks for that are not among its inputs. Names the model chose
  differently are matched by extension, so a naming difference alone is not a failure; a missing extension is.
"""
from __future__ import annotations

import json
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DRAFT_NAME = re.compile(r"(^|[\W_])(draft|tmp|temp|scratch|wip|backup|bak|untitled|test)([\W_]|$)|^~\$|^\.", re.I)
OUTPUT_EXTS = ("xlsx", "xls", "docx", "pptx", "pdf", "csv", "md", "json", "txt", "db", "sqlite", "html", "png", "jpg", "zip")
_FILENAME = re.compile(r"[\"'`]([^\"'`/\\]+?\.(?:" + "|".join(OUTPUT_EXTS) + r"))[\"'`]|(?<![\w.\-])([\w][\w.\-()&%+]*\.(?:" + "|".join(OUTPUT_EXTS) + r"))\b", re.I)
_ASKS = re.compile(r"deliver|output|save|produce|create|submit|prepare|provide|generate|write|export|workbook|report|memo", re.I)
ORDER = ["none", "draft_only", "invalid", "partial", "structurally_valid"]


@dataclass
class FileCheck:
    name: str
    status: str                      # valid | invalid | draft
    reason: str = ""
    size: int = 0


@dataclass
class DeliverableReport:
    status: str = "none"
    files: List[FileCheck] = field(default_factory=list)
    required_exts: List[str] = field(default_factory=list)
    missing_exts: List[str] = field(default_factory=list)
    expected_names: List[str] = field(default_factory=list)
    unmatched_names: List[str] = field(default_factory=list)
    digest: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "structurally_valid"

    def issues(self) -> str:
        if self.status == "none":
            return "no files are in the output directory"
        out = [f"{f.name}: {f.reason}" for f in self.files if f.status != "valid"]
        if self.missing_exts:
            out.append("no valid file of type " + ", ".join("." + e for e in self.missing_exts))
        return "; ".join(out) or "all files pass the structural checks"

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "files": [f.__dict__ for f in self.files], "required_exts": self.required_exts, "missing_exts": self.missing_exts,
                "expected_names": self.expected_names, "unmatched_names": self.unmatched_names}


def expected_artifacts(task_text: str, input_names: Sequence[str] = ()) -> List[str]:
    """File names a task text asks for as outputs: names on lines that talk about delivering/saving/producing, minus the task's own input files."""
    inputs = {n.lower() for n in input_names}
    names: List[str] = []
    for line in task_text.splitlines():
        if not _ASKS.search(line):
            continue
        for m in _FILENAME.finditer(line):
            n = (m.group(1) or m.group(2)).strip(" .,;:()")
            if n.lower() not in inputs and n not in names and not DRAFT_NAME.search(n):
                names.append(n)
    return names


def _text_of_xml(xml: str, tag: str) -> str:
    return " ".join(re.findall(rf"<{tag}[^>]*>([^<]*)</{tag}>", xml))


def check_file(path: Path) -> FileCheck:
    p = Path(path)
    name = p.name
    try:
        size = p.stat().st_size
    except OSError as e:
        return FileCheck(name, "invalid", f"unreadable ({e.__class__.__name__})")
    if DRAFT_NAME.search(name):
        return FileCheck(name, "draft", "draft/intermediate file name", size)
    if size == 0:
        return FileCheck(name, "invalid", "zero bytes", size)
    ext = p.suffix.lower().lstrip(".")
    try:
        if ext in ("xlsx", "docx", "pptx"):
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                if ext == "xlsx":
                    sheets = [n for n in names if n.startswith("xl/worksheets/sheet")]
                    shared = _text_of_xml(z.read("xl/sharedStrings.xml").decode("utf-8", "replace"), "t") if "xl/sharedStrings.xml" in names else ""
                    cells = 0
                    for sname in sheets:
                        x = z.read(sname).decode("utf-8", "replace")
                        cells += len(re.findall(r"<v>[^<]+</v>|<is>", x))
                    if not sheets or (cells == 0 and not shared.strip()):
                        return FileCheck(name, "invalid", "workbook has no populated cell", size)
                elif ext == "docx":
                    if "word/document.xml" not in names or len(_text_of_xml(z.read("word/document.xml").decode("utf-8", "replace"), "w:t").strip()) < 20:
                        return FileCheck(name, "invalid", "document has (almost) no text", size)
                else:
                    slides = [n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)]
                    if not slides:
                        return FileCheck(name, "invalid", "presentation has no slide", size)
        elif ext == "pdf":
            b = p.read_bytes()
            if not b.startswith(b"%PDF") or b"/Page" not in b:
                return FileCheck(name, "invalid", "not a readable PDF with a page", size)
        elif ext == "csv":
            lines = [l for l in p.read_text(errors="replace").splitlines() if l.strip()]
            if len(lines) < 2:
                return FileCheck(name, "invalid", "CSV needs a header and at least one data row", size)
        elif ext == "json":
            json.loads(p.read_text(errors="replace"))
        elif ext in ("md", "txt", "html"):
            if len(re.sub(r"\s+", "", p.read_text(errors="replace"))) < 20:
                return FileCheck(name, "invalid", "almost no text", size)
        elif ext in ("db", "sqlite"):
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
                rows = sum(con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables)
            finally:
                con.close()
            if not tables or rows == 0:
                return FileCheck(name, "invalid", "database has no rows in any table", size)
    except (zipfile.BadZipFile, KeyError, ValueError, sqlite3.DatabaseError, OSError) as e:
        return FileCheck(name, "invalid", f"cannot be opened as .{ext} ({e.__class__.__name__})", size)
    return FileCheck(name, "valid", "", size)


def check_deliverables(output_dir: Path, task_text: str = "", input_names: Sequence[str] = ()) -> DeliverableReport:
    d = Path(output_dir)
    files = sorted(f for f in d.rglob("*") if f.is_file()) if d.is_dir() else []
    checks = [check_file(f) for f in files]
    expected = expected_artifacts(task_text, input_names) if task_text else []
    req = sorted({n.rsplit(".", 1)[1].lower() for n in expected})
    valid_exts = {c.name.rsplit(".", 1)[-1].lower() for c in checks if c.status == "valid"}
    have = {c.name.lower() for c in checks if c.status == "valid"}
    rep = DeliverableReport(files=checks, required_exts=req, expected_names=expected,
                            missing_exts=[e for e in req if e not in valid_exts], unmatched_names=[n for n in expected if n.lower() not in have])
    if not checks:
        rep.status = "none"
    elif all(c.status == "draft" for c in checks):
        rep.status = "draft_only"
    elif not any(c.status == "valid" for c in checks):
        rep.status = "invalid"
    elif rep.missing_exts or any(c.status == "invalid" for c in checks):
        rep.status = "partial"
    else:
        rep.status = "structurally_valid"
    import hashlib
    h = hashlib.sha256()
    for f, c in zip(files, checks):
        h.update(f"{f.relative_to(d)}|{c.size}|{c.status}|".encode())
        h.update(f.read_bytes()[:65536] if c.size else b"")
    rep.digest = h.hexdigest()[:16]
    return rep
