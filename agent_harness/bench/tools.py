"""The benchmark tool set: bash (in the no-network container), confined file tools, and a guarded web fetch."""
from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from ..egress_proxy import DEFAULT_ALLOW, DEFAULT_DENY, host_denied
from ..tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool

_SCRIPT = re.compile(r"(?is)<(script|style|noscript)\b.*?</\1>")
_TAG = re.compile(r"(?s)<[^>]+>")


def html_to_text(raw: str) -> str:
    t = html.unescape(_TAG.sub(" ", _SCRIPT.sub(" ", raw)))
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\r\f\v]+", " ", t)).strip()


def _confine(workspace: Path, path: str) -> Path:
    p = (workspace / path.lstrip("/") if not str(path).startswith(str(workspace)) else Path(path)).resolve()
    if p != workspace and workspace not in p.parents:
        raise ValueError(f"path escapes the workspace: {path}")
    return p


def _virtual(workspace: Path, path: str) -> str:
    """The agent sees /work/...; map to the host workspace."""
    return path[len("/work"):] if path == "/work" or path.startswith("/work/") else path


class NoRedirectToDenied(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if host_denied(urllib.parse.urlsplit(newurl).hostname or ""):
            raise urllib.error.URLError(f"redirect to blocked host {urllib.parse.urlsplit(newurl).hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def web_fetch(url: str, max_chars: int = 15000, timeout: int = 30,
              deny=DEFAULT_DENY, allow=DEFAULT_ALLOW, blocked_log: Optional[List[str]] = None, offset: int = 0, paged: bool = False) -> str:
    """Fetch a page as text. Legacy mode returns the first ``max_chars`` characters; ``paged=True`` honours ``offset`` and says where the next part starts."""
    text = _fetch(url, deny, allow, blocked_log, timeout, as_text=True)
    if text.startswith("ERROR:"):
        return text
    if not paged:
        return text[:max_chars] + (f"\n[... truncated, {len(text) - max_chars} more chars]" if len(text) > max_chars else "")
    offset = max(0, int(offset or 0))
    part = text[offset:offset + max_chars]
    if offset >= len(text):
        return f"[offset {offset} is past the end: the page has {len(text)} characters]"
    more = len(text) - offset - len(part)
    return part + (f"\n[... truncated, {more} more chars; call web_fetch with offset={offset + len(part)} for the next part, or web_download to save the file]" if more > 0 else "")


def web_download(url: str, dest: Path, max_bytes: int = 25_000_000, timeout: int = 60,
                 deny=DEFAULT_DENY, allow=DEFAULT_ALLOW, blocked_log: Optional[List[str]] = None) -> str:
    """Save a URL's bytes to ``dest`` (already confined by the caller). Same egress policy as web_fetch."""
    raw = _fetch(url, deny, allow, blocked_log, timeout, as_text=False, limit=max_bytes)
    if isinstance(raw, str):
        return raw
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return f"saved {len(raw)} bytes to {dest.name}" + (" (stopped at the size limit; the file is incomplete)" if len(raw) >= max_bytes else "")


def _fetch(url: str, deny, allow, blocked_log, timeout: int, *, as_text: bool, limit: int = 2_000_000):
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme in ("http", "https") and u.hostname:    # percent-encode non-ASCII (models paste URLs with an ellipsis etc.)
        url = urllib.parse.urlunsplit((u.scheme, u.netloc.encode("idna").decode("ascii"), urllib.parse.quote(u.path, safe="/%:@!$&'()*+,;=~-._"),
                                       urllib.parse.quote(u.query, safe="=&%+:/?@!$'()*,;~-._"), ""))
        u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return "ERROR: only http(s) URLs are supported"
    if host_denied(u.hostname, deny, allow):
        if blocked_log is not None:
            blocked_log.append(u.hostname)
        return f"ERROR: {u.hostname} is blocked by the benchmark egress policy"
    opener = urllib.request.build_opener(NoRedirectToDenied)
    try:
        with opener.open(urllib.request.Request(url, headers={"User-Agent": "agent-harness-bench/1"}), timeout=timeout) as r:
            data = r.read(limit)
            ctype = r.headers.get("Content-Type", "")
            charset = r.headers.get_content_charset() or "utf-8"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"ERROR: fetch failed: {e}"
    if not as_text:
        return data
    raw = data.decode(charset, errors="replace")
    return html_to_text(raw) if "html" in ctype else raw


def bench_registry(workspace: Path, sandbox, blocked_log: Optional[List[str]] = None,
                   web: bool = True, bash_timeout: int = 180, web_paging: bool = False, web_downloads: bool = False) -> ToolRegistry:
    ws = Path(workspace).resolve()
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=["bash", "write_file", "web_download"]))
    obj = lambda **props: {"type": "object", "properties": props, "required": list(props)}   # noqa: E731
    s = {"type": "string"}

    def bash(a):
        return ToolResult(text=sandbox.run(str(a["command"]), int(a.get("timeout", bash_timeout))))

    def read_file(a):
        stem = re.sub(r"\.txt$", "", str(a["path"]).replace("\\", "/").rsplit("/", 1)[-1])
        if re.fullmatch(r"A\d{3,5}", stem):
            return ToolResult(ok=False, error=(f"'{stem}' is an archive id from earlier in this conversation, not a file. Call "
                                               f"read_archive with id='{stem}' (and offset to page through it)."))
        p = _confine(ws, _virtual(ws, a["path"]))
        data = p.read_bytes()
        if b"\0" in data[:2000]:
            return ToolResult(ok=False, error="binary file: use bash with python/pdftotext/pandoc to extract text")
        return ToolResult(text=data.decode("utf-8", errors="replace"))

    def write_file(a):
        p = _confine(ws, _virtual(ws, a["path"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(a["content"]), encoding="utf-8")
        return ToolResult(text=f"wrote {len(str(a['content']))} chars to {a['path']}")

    def list_dir(a):
        p = _confine(ws, _virtual(ws, a.get("path", "/work")))
        return ToolResult(text="\n".join(sorted(f"{c.name}{'/' if c.is_dir() else ''}" for c in p.iterdir())) or "[empty]")

    reg.register(function_tool("bash", bash, "Run a shell command in a sandbox (Python 3 with pandas, openpyxl, python-docx, "
                               "python-pptx, pdfplumber; pandoc; pdftotext). No network. Working dir /work.",
                               obj(command=s), side_effecting=True))
    reg.register(function_tool("read_file", read_file, "Read a UTF-8 text file under /work.", obj(path=s)))
    reg.register(function_tool("write_file", write_file, "Write a UTF-8 text file under /work.", obj(path=s, content=s),
                               side_effecting=True))
    reg.register(function_tool("list_dir", list_dir, "List a directory under /work.", obj(path=s)))
    if web:
        def fetch(a):
            t = web_fetch(str(a["url"]), blocked_log=blocked_log, offset=a.get("offset", 0), paged=web_paging)
            return ToolResult(ok=not t.startswith("ERROR:"), text=t, error=t if t.startswith("ERROR:") else None)
        if web_paging:
            reg.register(function_tool("web_fetch", fetch, "Fetch a web page as text, 15,000 characters at a time (some hosts are blocked). "
                                       "Use offset to read further into a long page.",
                                       {"type": "object", "properties": {"url": s, "offset": {"type": "integer", "description": "first character to return"}},
                                        "required": ["url"]}))
        else:
            reg.register(function_tool("web_fetch", fetch, "Fetch a web page as text (some hosts are blocked).", obj(url=s)))
        if web_downloads:
            def download(a):
                t = web_download(str(a["url"]), _confine(ws, _virtual(ws, a["path"])), blocked_log=blocked_log)
                return ToolResult(ok=not t.startswith("ERROR:"), text=t, error=t if t.startswith("ERROR:") else None)
            reg.register(function_tool("web_download", download, "Download a URL (data file, up to 25 MB) into /work, to be read with bash. "
                                       "Same blocked hosts as web_fetch.", obj(url=s, path=s), side_effecting=True))
    return reg
