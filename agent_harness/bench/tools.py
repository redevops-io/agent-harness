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
              deny=DEFAULT_DENY, allow=DEFAULT_ALLOW, blocked_log: Optional[List[str]] = None) -> str:
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
            raw = r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace")
            ctype = r.headers.get("Content-Type", "")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"ERROR: fetch failed: {e}"
    text = html_to_text(raw) if "html" in ctype else raw
    return text[:max_chars] + (f"\n[... truncated, {len(text) - max_chars} more chars]" if len(text) > max_chars else "")


def bench_registry(workspace: Path, sandbox, blocked_log: Optional[List[str]] = None,
                   web: bool = True, bash_timeout: int = 180) -> ToolRegistry:
    ws = Path(workspace).resolve()
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=["bash", "write_file"]))
    obj = lambda **props: {"type": "object", "properties": props, "required": list(props)}   # noqa: E731
    s = {"type": "string"}

    def bash(a):
        return ToolResult(text=sandbox.run(str(a["command"]), int(a.get("timeout", bash_timeout))))

    def read_file(a):
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
        reg.register(function_tool("web_fetch", lambda a: (lambda t: ToolResult(ok=not t.startswith("ERROR:"), text=t,
                                   error=t if t.startswith("ERROR:") else None))(web_fetch(str(a["url"]), blocked_log=blocked_log)),
                                   "Fetch a web page as text (some hosts are blocked).", obj(url=s)))
    return reg
