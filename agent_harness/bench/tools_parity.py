"""The benchmark tool set at parity with the capabilities stock OpenCode gives its model (``tools="parity"``; the legacy set in ``tools.py`` stays available
for controlled comparisons).

Differences from the legacy set, each one a documented gap:

* **bash can reach the network, but only through the egress gateway** (``DockerSandbox(network="egress")``; the policy is enforced in a separate container, see
  ``EgressGateway``). With ``network="none"`` the tool says so.
* **Nothing is silently truncated**: long bash output, web pages and file reads are paged; the full text is saved under ``/work/.tool_output`` and read back with
  ``read_file(path, offset, limit)``.
* **``edit_file``**: exact-match replacement, unique unless ``replace_all``, atomic, failure messages that say what to do.
* **``web_download``**: streams to a file under ``/work`` through the same policy (public addresses only, redirects re-checked, size cap), hashes it and records it.
* **Tool descriptions state what is and is not possible** (network, curl/wget, OCR, images), from facts passed in by the caller (probed in the sandbox).
* Consistent semantics: every tool call returns text; a command's exit status is always in the text (``[exit code N]``); a timeout kills the command's whole process
  group and says so; file tools are confined to ``/work``; the tool-level error flag is reserved for the tool itself failing (blocked URL, bad path, failed edit).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..netpolicy import NetPolicy, safe_fetch
from ..tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool
from .edit import edit_text
from .outputs import FileLedger, OutputStore, STORE_DIR
from .tools import _confine, _virtual, html_to_text

PAGE = 12000                       # characters returned per read or fetch page (the loop admits at most 12,000 per tool result)
BASH_STDOUT, BASH_STDERR = 10000, 4000
MAX_FETCH_BYTES = 5_000_000
MAX_DOWNLOAD_BYTES = 50_000_000
MAX_TIMEOUT = 600


def describe_bash(network: str, found: Sequence[str]) -> str:
    base = "Run a shell command in a sandbox (Python 3 with pandas, openpyxl, python-docx, python-pptx, pdfplumber; pandoc; pdftotext). Working dir /work. "
    if network == "egress":
        clients = [c for c in ("curl", "wget") if c in found]
        net = ("Network: HTTP and HTTPS only, through a filtering proxy that is already configured (HTTP_PROXY/HTTPS_PROXY); direct sockets and your own DNS lookups do "
               "not work; private, local and cloud-metadata addresses and some answer-hosting sites are blocked; redirects are re-checked. "
               + (f"Command-line clients: {', '.join(clients)}. " if clients else "curl and wget are not installed; use python (urllib or requests honour the proxy) or the web_download tool. "))
    else:
        net = "Network: none. Use the web_fetch and web_download tools for web content. "
    ocr = "OCR: tesseract is installed. " if "tesseract" in found else "OCR: no OCR program is installed. "
    return (base + net + ocr + "You cannot view images yourself. "
            f"Output longer than {BASH_STDOUT} characters is shortened in the reply, but the full text is saved in /work/{STORE_DIR}/ and can be read with read_file "
            f"(path, offset, limit) or searched with grep. The reply ends with [exit code N] when the command failed, or a timed-out notice. Optional timeout in seconds "
            f"(default 180, at most {MAX_TIMEOUT}); on timeout the command and its child processes are killed.")


def parity_registry(workspace: Path, sandbox, *, policy: Optional[NetPolicy] = None, web: bool = True, bash_timeout: int = 180,
                    network: str = "egress", found: Sequence[str] = (), blocked_log: Optional[List[str]] = None,
                    fetch: Callable[..., Any] = safe_fetch, store: Optional[OutputStore] = None, ledger: Optional[FileLedger] = None) -> ToolRegistry:
    ws = Path(workspace).resolve()
    policy = policy or NetPolicy()
    store = store or OutputStore(ws)
    ledger = ledger or FileLedger(ws)
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=["bash", "write_file", "edit_file", "web_download"]))
    s = {"type": "string"}
    i = lambda d: {"type": "integer", "description": d}   # noqa: E731
    cache: Dict[str, str] = {}

    def conf(path: str) -> Path:
        return _confine(ws, _virtual(ws, str(path)))

    # ---- bash ----------------------------------------------------------------------------------------------------------
    def bash(a):
        t = max(1, min(int(a.get("timeout") or bash_timeout), MAX_TIMEOUT))
        r = sandbox.run_raw(str(a["command"]), t)
        out = store.clip("bash-stdout", r.stdout, BASH_STDOUT)
        parts = [out] if out.strip() else []
        if r.stderr.strip():
            parts.append("[stderr]\n" + store.clip("bash-stderr", r.stderr, BASH_STDERR))
        if r.timed_out:
            parts.append(f"[timed out after {t}s: the command and its child processes were killed]")
        elif r.returncode:
            parts.append(f"[exit code {r.returncode}]")
        ledger.record("bash")
        return ToolResult(text="\n".join(parts) or "[no output]")

    # ---- files -----------------------------------------------------------------------------------------------------------
    def read_file(a):
        import re
        stem = re.sub(r"\.txt$", "", str(a["path"]).replace("\\", "/").rsplit("/", 1)[-1])
        if re.fullmatch(r"A\d{3,5}", stem):
            return ToolResult(ok=False, error=(f"'{stem}' is an archive id from earlier in this conversation, not a file. Call read_archive with id='{stem}' "
                                               f"(and offset to page through it)."))
        p = conf(a["path"])
        if not p.is_file():
            return ToolResult(ok=False, error=f"no such file: {a['path']}")
        data = p.read_bytes()
        if b"\0" in data[:2000]:
            return ToolResult(ok=False, error="binary file: use bash with python, pdftotext or pandoc to extract text (images cannot be read directly)")
        text = data.decode("utf-8", errors="replace")
        off = max(0, int(a.get("offset") or 0))
        lim = max(1, min(int(a.get("limit") or PAGE), PAGE))
        if off >= len(text) and len(text):
            return ToolResult(text=f"[offset {off} is past the end: the file has {len(text)} characters]")
        part = text[off:off + lim]
        more = len(text) - off - len(part)
        return ToolResult(text=part + (f"\n[... {more} more characters; call read_file with offset={off + len(part)} for the next page]" if more > 0 else ""))

    def write_file(a):
        p = conf(a["path"])
        if p.is_dir():
            return ToolResult(ok=False, error=f"{a['path']} is a directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(a["content"]))
        os.chmod(tmp, 0o666)
        os.replace(tmp, p)
        ledger.record("write_file", only=p)
        return ToolResult(text=f"wrote {len(str(a['content']))} chars to {a['path']}")

    def edit_file(a):
        p = conf(a["path"])
        ok, msg = edit_text(p, str(a["old_string"]), str(a["new_string"]), bool(a.get("replace_all", False)))
        if ok:
            ledger.record("edit_file", only=p)
        return ToolResult(ok=ok, text=msg, error=None if ok else msg)

    def list_dir(a):
        p = conf(a.get("path", "/work"))
        if not p.is_dir():
            return ToolResult(ok=False, error=f"not a directory: {a.get('path')}")
        return ToolResult(text="\n".join(sorted(f"{c.name}{'/' if c.is_dir() else ''}" for c in p.iterdir() if not c.name.startswith("."))) or "[empty]")

    reg.register(function_tool("bash", bash, describe_bash(network, found),
                               {"type": "object", "properties": {"command": s, "timeout": i("seconds, default 180")}, "required": ["command"]}, side_effecting=True))
    reg.register(function_tool("read_file", read_file, f"Read a UTF-8 text file under /work, {PAGE} characters per call; offset and limit page through longer files.",
                               {"type": "object", "properties": {"path": s, "offset": i("first character"), "limit": i(f"characters, at most {PAGE}")}, "required": ["path"]}))
    reg.register(function_tool("write_file", write_file, "Create or overwrite a UTF-8 text file under /work (atomic). To change part of an existing file use edit_file.",
                               {"type": "object", "properties": {"path": s, "content": s}, "required": ["path", "content"]}, side_effecting=True))
    reg.register(function_tool("edit_file", edit_file,
                               "Replace text in an existing UTF-8 file under /work. old_string must match exactly (including whitespace) and occur exactly once, unless "
                               "replace_all is true. The edit is atomic; on any failure the file is unchanged and the reply says why.",
                               {"type": "object", "properties": {"path": s, "old_string": s, "new_string": s, "replace_all": {"type": "boolean"}},
                                "required": ["path", "old_string", "new_string"]}, side_effecting=True))
    reg.register(function_tool("list_dir", list_dir, "List a directory under /work.", {"type": "object", "properties": {"path": s}, "required": ["path"]}))

    # ---- web -------------------------------------------------------------------------------------------------------------
    def note_block(res):
        if res.blocked is not None and blocked_log is not None:
            from urllib.parse import urlsplit
            blocked_log.append(urlsplit(res.chain[-1]).hostname or "?")

    def web_fetch(a):
        url = str(a["url"]).strip()
        off = max(0, int(a.get("offset") or 0))
        if url not in cache:
            res = fetch(url, policy, max_bytes=MAX_FETCH_BYTES)
            if not res.ok:
                note_block(res)
                return ToolResult(ok=False, error="ERROR: " + res.error, text="ERROR: " + res.error)
            raw = res.data.decode("utf-8", errors="replace")
            text = html_to_text(raw) if "html" in res.content_type.lower() else raw
            if res.truncated:
                text += f"\n[download stopped at {MAX_FETCH_BYTES} bytes]"
            cache[url] = text
            if len(text) > PAGE:
                cache[url + "\0path"] = store.save("web", text)
        text = cache[url]
        if off >= len(text):
            return ToolResult(text=f"[offset {off} is past the end: the page has {len(text)} characters]")
        part = text[off:off + PAGE]
        more = len(text) - off - len(part)
        tail = ""
        if more > 0:
            tail = (f"\n[... {more} more characters; call web_fetch with offset={off + len(part)} for the next page. The full text is saved at "
                    f"{cache.get(url + chr(0) + 'path')}: read it with read_file or search it with grep]")
        return ToolResult(text=part + tail)

    def web_download(a):
        url = str(a["url"]).strip()
        dest = conf(a["path"])
        if dest.is_dir():
            return ToolResult(ok=False, error=f"{a['path']} is a directory")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".part")
        f = os.fdopen(fd, "wb")
        try:
            res = fetch(url, policy, max_bytes=MAX_DOWNLOAD_BYTES, keep=False, sink=f.write, total_timeout=300)
        finally:
            f.close()
        if not res.ok or res.truncated:
            os.unlink(tmp)                                   # never leave a partial file behind
            why = res.error if not res.ok else f"the file is larger than the {MAX_DOWNLOAD_BYTES} byte limit; nothing was saved"
            note_block(res)
            return ToolResult(ok=False, error="ERROR: " + why, text="ERROR: " + why)
        os.chmod(tmp, 0o666)
        os.replace(tmp, dest)
        man = ws / ".downloads" / "manifest.jsonl"
        man.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(man.parent, 0o777)
        with open(man, "a", encoding="utf-8") as m:
            m.write(json.dumps({"t": round(time.time(), 1), "url": url, "final_url": res.final_url, "redirects": len(res.chain) - 1, "chain": res.chain,
                                "path": "/work/" + str(dest.relative_to(ws)), "size": res.size, "sha256": res.sha256, "content_type": res.content_type}) + "\n")
        ledger.record("web_download", only=dest)
        return ToolResult(text=f"saved {res.size} bytes to {a['path']} (sha256 {res.sha256[:16]}...; {len(res.chain) - 1} redirect(s); final URL {res.final_url})")

    if web:
        reg.register(function_tool("web_fetch", web_fetch,
                                   f"Fetch a web page as text, {PAGE} characters per call; offset reads further into a long page, and the full text is saved under /work/{STORE_DIR}/. "
                                   "Public http(s) addresses only; some hosts are blocked.",
                                   {"type": "object", "properties": {"url": s, "offset": i("first character")}, "required": ["url"]}))
        reg.register(function_tool("web_download", web_download,
                                   f"Download a URL's file into /work (up to {MAX_DOWNLOAD_BYTES // 1_000_000} MB) so bash can read it. Records the SHA-256. Public http(s) "
                                   "addresses only; redirects are re-checked; nothing is saved if the file is too large.",
                                   {"type": "object", "properties": {"url": s, "path": s}, "required": ["url", "path"]}, side_effecting=True))
    return reg
