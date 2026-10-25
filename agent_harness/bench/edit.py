"""In-place text edit with strict matching, atomic replacement and failure messages that say what to do next.

``edit_text(path, old, new, replace_all)`` replaces ``old`` with ``new`` only if ``old`` occurs exactly once (or ``replace_all``); it never guesses.
The new content is written to a temporary file in the same directory (same mode) and moved over the original with ``os.replace``, so a crash or a failed
check leaves the original untouched. Returns ``(ok, message)``.
"""
from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path
from typing import Tuple


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def edit_text(path: Path, old: str, new: str, replace_all: bool = False) -> Tuple[bool, str]:
    p = Path(path)
    if not p.is_file():
        return False, f"no such file: {p.name}. Use write_file to create a new file."
    if old == "":
        return False, "old_string is empty. To create or overwrite a whole file use write_file; edit_file replaces existing text."
    if old == new:
        return False, "old_string and new_string are identical; nothing to change."
    data = p.read_bytes()
    if b"\0" in data[:4000]:
        return False, "binary file: edit_file only edits text files."
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False, "the file is not valid UTF-8; edit_file only edits UTF-8 text."
    if "\r\n" in text and "\r\n" not in old:         # a CRLF file edited with LF text: match and write in the file's own line endings (still exact otherwise)
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    n = text.count(old)
    if n == 0:
        hint = ""
        first = old.strip().splitlines()[0] if old.strip() else ""
        if first:
            lines = text.splitlines()
            close = difflib.get_close_matches(first.strip(), [l.strip() for l in lines], n=1, cutoff=0.6)
            if close:
                at = next(i for i, l in enumerate(lines) if l.strip() == close[0]) + 1
                hint = f" The closest line is {at}: {lines[at - 1].strip()[:120]!r}. Check whitespace, indentation and line endings; read the file again before editing."
        return False, "old_string was not found in the file; nothing was changed." + hint
    if n > 1 and not replace_all:
        at = []
        i = -1
        while len(at) < 5 and (i := text.find(old, i + 1)) != -1:
            at.append(_line_of(text, i))
        return False, (f"old_string occurs {n} times (lines {', '.join(map(str, at))}{'...' if n > 5 else ''}); nothing was changed. Include more surrounding "
                       f"lines to make it unique, or pass replace_all=true to change every occurrence.")
    out = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    st = p.stat()
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(out.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, st.st_mode & 0o7777)
        os.replace(tmp, p)
    except Exception as e:                    # noqa: BLE001 - report and keep the original
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False, f"could not write the edit ({type(e).__name__}: {e}); the original file is unchanged."
    return True, f"edited {p.name}: replaced {n if replace_all else 1} occurrence{'s' if replace_all and n != 1 else ''} ({len(old)} -> {len(new)} chars)."
