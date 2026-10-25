"""Network egress policy and a policy-enforcing HTTP(S) fetcher, shared by the egress proxy (outside the agent) and the in-process web tools.

What it enforces, in one place, so the proxy and the tools cannot drift apart:

* **Destinations**: only public addresses. The host name is resolved *here*, every resulting address is checked (loopback, private, link-local including the
  169.254.169.254 metadata address, CGNAT, multicast, reserved, IPv4-mapped IPv6 are refused) and the connection is made to the checked address itself, never to the
  name again, so DNS rebinding between check and connect is not possible. If any address of a name is refused the whole name is refused.
* **Names**: ``localhost``, ``*.localhost``, ``*.local``, ``*.internal`` and the cloud metadata host names are refused before resolution; a deny list (benchmark answer
  hosts) applies on top, with an allow list that only overrides the deny list.
* **Ports**: 80, 443, 8080, 8443 by default.
* **Redirects**: ``safe_fetch`` follows redirects itself, re-checking scheme, host, port and addresses at every hop and bounding their number.
* **Size and time**: a byte cap and an overall deadline per fetch; the SHA-256 of what was received is returned with the result.

Resolver and connector are injectable so every rule is testable without a network.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import ssl
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Upstream harness2 EXPERIMENTS.md, "Web access": block these for the executor.
DEFAULT_DENY = ("huggingface.co", "hf.co", "*.hf.co", "github.com", "raw.githubusercontent.com")
# Gists are not benchmark repositories and tasks may cite them as references.
DEFAULT_ALLOW = ("gist.github.com", "gist.githubusercontent.com")
DEFAULT_PORTS = (80, 443, 8080, 8443)
BLOCKED_NAMES = ("localhost", "metadata.google.internal", "metadata.goog", "instance-data", "metadata")
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".lan", ".home.arpa")


def _match(host: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        return host.endswith(pattern[1:])           # *.hf.co matches cdn.hf.co, not hf.co
    return host == pattern or host.endswith("." + pattern)


def host_denied(host: str, deny: Iterable[str] = DEFAULT_DENY, allow: Iterable[str] = DEFAULT_ALLOW) -> bool:
    h = host.lower().rstrip(".")
    if any(_match(h, a) for a in allow):
        return False
    return any(_match(h, d) for d in deny)


def ip_blocked(addr: str) -> bool:
    """True for anything that is not a public unicast address (IPv4-mapped IPv6 is unwrapped first)."""
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


@dataclass
class Verdict:
    ok: bool
    reason: str = ""                  # port | name | host_denied | private_address | unresolvable | scheme
    ips: List[str] = field(default_factory=list)


@dataclass
class NetPolicy:
    deny_hosts: Sequence[str] = DEFAULT_DENY
    allow_hosts: Sequence[str] = DEFAULT_ALLOW
    ports: Sequence[int] = DEFAULT_PORTS
    resolver: Callable[..., Any] = socket.getaddrinfo

    def check(self, host: str, port: int) -> Verdict:
        h = (host or "").strip().lower().rstrip(".")
        if not h:
            return Verdict(False, "name")
        if int(port) not in tuple(self.ports):
            return Verdict(False, "port")
        if h in BLOCKED_NAMES or h.endswith(BLOCKED_SUFFIXES):
            return Verdict(False, "name")
        if host_denied(h, self.deny_hosts, self.allow_hosts):
            return Verdict(False, "host_denied")
        try:
            infos = self.resolver(h, int(port), type=socket.SOCK_STREAM)
        except (OSError, UnicodeError):
            return Verdict(False, "unresolvable")
        ips: List[str] = []
        for info in infos:
            ip = info[4][0]
            if ip_blocked(ip):
                return Verdict(False, "private_address")     # one bad record refuses the name
            if ip not in ips:
                ips.append(ip)
        return Verdict(bool(ips), "" if ips else "unresolvable", ips)


# ---- connections to a checked address ------------------------------------------------------------------------------------
class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects to ``ip`` but verifies the certificate and sends SNI for ``server_name``."""

    def __init__(self, ip: str, port: int, server_name: str, timeout: float):
        super().__init__(ip, port, timeout=timeout, context=ssl.create_default_context())
        self._server_name = server_name

    def connect(self):
        sock = socket.create_connection((self.host, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self._server_name)


def default_connect(ip: str, port: int, host: str, tls: bool, timeout: float):
    return _PinnedHTTPSConnection(ip, port, host, timeout) if tls else http.client.HTTPConnection(ip, port, timeout=timeout)


@dataclass
class FetchResult:
    ok: bool
    error: str = ""
    status: int = 0
    final_url: str = ""
    chain: List[str] = field(default_factory=list)       # every URL requested, in order
    content_type: str = ""
    size: int = 0
    sha256: str = ""
    truncated: bool = False                              # stopped at max_bytes
    data: bytes = b""
    blocked: Optional[Verdict] = None


def safe_fetch(url: str, policy: NetPolicy, *, max_bytes: int = 25_000_000, max_redirects: int = 5, timeout: float = 30,
               total_timeout: float = 120, connect: Callable[..., Any] = default_connect, keep: bool = True,
               sink: Optional[Callable[[bytes], None]] = None, user_agent: str = "agent-harness-bench/1") -> FetchResult:
    """GET ``url`` under ``policy``. ``sink`` receives the body in chunks (for large files); with ``keep`` the body is also returned in memory."""
    deadline = time.time() + total_timeout
    chain: List[str] = []
    cur = url.strip()
    for hop in range(max_redirects + 1):
        chain.append(cur)
        u = urllib.parse.urlsplit(cur)
        if u.scheme not in ("http", "https") or not u.hostname:
            return FetchResult(False, "only http(s) URLs are supported", chain=chain, blocked=Verdict(False, "scheme"))
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        v = policy.check(host, port)
        if not v.ok:
            return FetchResult(False, f"{host}:{port} is blocked by the egress policy ({v.reason})", chain=chain, blocked=v)
        path = urllib.parse.quote(u.path or "/", safe="/%:@!$&'()*+,;=~-._") + (("?" + urllib.parse.quote(u.query, safe="=&%+:/?@!$'()*,;~-._")) if u.query else "")
        conn = None
        try:
            conn = connect(v.ips[0], port, host, u.scheme == "https", min(timeout, max(1.0, deadline - time.time())))
            conn.request("GET", path, headers={"Host": u.netloc.rsplit("@", 1)[-1], "User-Agent": user_agent, "Accept-Encoding": "identity", "Accept": "*/*"})
            r = conn.getresponse()
            if r.status in (301, 302, 303, 307, 308) and r.getheader("Location"):
                cur = urllib.parse.urljoin(cur, r.getheader("Location"))
                r.read(0)
                continue
            h = hashlib.sha256()
            buf = bytearray()
            size = 0
            truncated = False
            while True:
                if time.time() > deadline:
                    return FetchResult(False, "fetch failed: total time limit exceeded", status=r.status, chain=chain)
                chunk = r.read(min(65536, max_bytes - size + 1))
                if not chunk:
                    break
                if size + len(chunk) > max_bytes:
                    chunk = chunk[: max_bytes - size]
                    truncated = True
                h.update(chunk)
                size += len(chunk)
                if keep:
                    buf += chunk
                if sink:
                    sink(chunk)
                if truncated:
                    break
            if r.status >= 400:
                return FetchResult(False, f"fetch failed: HTTP Error {r.status}: {r.reason}", status=r.status, final_url=cur, chain=chain)
            return FetchResult(True, status=r.status, final_url=cur, chain=chain, content_type=r.getheader("Content-Type", "") or "", size=size,
                               sha256=h.hexdigest(), truncated=truncated, data=bytes(buf))
        except (OSError, http.client.HTTPException, ssl.SSLError, ValueError) as e:
            return FetchResult(False, f"fetch failed: {e}", chain=chain)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
    return FetchResult(False, f"fetch failed: more than {max_redirects} redirects", chain=chain)
