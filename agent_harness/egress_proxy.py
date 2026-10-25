"""Loopback HTTP/HTTPS forward proxy with a host denylist, for benchmark runs.

Benchmark rubrics and answers are public on some hosts (Hugging Face, GitHub). Agents still need the open web
(tasks tell them to consult external references), so block only those hosts. Point an agent at it with
``HTTP_PROXY``/``HTTPS_PROXY=http://127.0.0.1:PORT`` and ``NO_PROXY`` for the model endpoints.

It is a soft control: a process that ignores proxy settings bypasses it. For a hard block run the agent in
a container with no network and mediate web access through a tool (see ``agent_harness.bench``).
Every denied request is logged by host only (no paths, no headers, no bodies), as evidence for reports.
"""
from __future__ import annotations

import argparse
import select
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter
from typing import Iterable, List, Optional

from .netpolicy import DEFAULT_ALLOW, DEFAULT_DENY, NetPolicy, host_denied   # noqa: F401  (re-exported: callers import them from here)


class EgressProxy:
    """``policy=None`` is the legacy host denylist only. With a ``NetPolicy`` the proxy also refuses non-public addresses (checked after its own DNS
    resolution, then connected by address so a name cannot change meaning between check and connect), unlisted ports, and caps bytes per tunnel."""

    def __init__(self, deny: Iterable[str] = DEFAULT_DENY, allow: Iterable[str] = DEFAULT_ALLOW, policy: Optional[NetPolicy] = None,
                 max_tunnel_bytes: int = 500_000_000):
        self.deny, self.allow = tuple(deny), tuple(allow)
        self.policy, self.max_tunnel_bytes = policy, max_tunnel_bytes
        self.reasons: Counter = Counter()
        self.denied: List[str] = []
        self.allowed_count = 0
        self._lock = threading.Lock()

    def _note(self, host: str, denied: bool, reason: str = "host_denied") -> None:
        with self._lock:
            if denied:
                self.denied.append(host)
                self.reasons[reason] += 1
            else:
                self.allowed_count += 1

    def handler(self):
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _deny(self, host):
                body = b"blocked by benchmark egress policy"
                self.send_response(403)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_CONNECT(self):
                host, _, port = self.path.rpartition(":")
                try:
                    pnum = int(port or 443)
                except ValueError:
                    return self._deny(host)
                target = host
                if proxy.policy:
                    v = proxy.policy.check(host, pnum)
                    if not v.ok:
                        proxy._note(host, True, v.reason)
                        return self._deny(host)
                    target = v.ips[0]
                elif host_denied(host, proxy.deny, proxy.allow):
                    proxy._note(host, True)
                    return self._deny(host)
                proxy._note(host, False)
                try:
                    upstream = socket.create_connection((target, pnum), timeout=30)
                except OSError:
                    self.send_error(502)
                    return
                self.send_response(200, "Connection Established")
                self.end_headers()
                socks = [self.connection, upstream]
                moved = 0
                try:
                    while True:
                        r, _, x = select.select(socks, [], socks, 120)
                        if x or not r:
                            break
                        for s in r:
                            data = s.recv(65536)
                            if not data:
                                return
                            moved += len(data)
                            if moved > proxy.max_tunnel_bytes:
                                return
                            (upstream if s is self.connection else self.connection).sendall(data)
                except OSError:
                    pass
                finally:
                    upstream.close()
                    self.close_connection = True

            def _plain(self):
                u = urllib.parse.urlsplit(self.path)
                host = u.hostname or ""
                target = host
                if proxy.policy:
                    if u.scheme != "http":
                        proxy._note(host, True, "scheme")
                        return self._deny(host)
                    v = proxy.policy.check(host, u.port or 80)
                    if not v.ok:
                        proxy._note(host, True, v.reason)
                        return self._deny(host)
                    target = v.ips[0]
                elif host_denied(host, proxy.deny, proxy.allow):
                    proxy._note(host, True)
                    return self._deny(host)
                proxy._note(host, False)
                import http.client
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                conn = http.client.HTTPConnection(target, u.port or 80, timeout=60)
                try:
                    hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("proxy-connection", "connection")}
                    conn.request(self.command, (u.path or "/") + ("?" + u.query if u.query else ""), body, hdrs)
                    r = conn.getresponse()
                    data = r.read()
                    self.send_response(r.status)
                    for k, v in r.getheaders():
                        if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    self.send_error(502)
                finally:
                    conn.close()
                    self.close_connection = True

            do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = _plain

        return H


def serve(proxy: EgressProxy, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), proxy.handler())
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--deny", action="append", default=[], help="extra host to block (repeatable)")
    ap.add_argument("--host", default="127.0.0.1", help="listen address (0.0.0.0 only inside the egress sidecar container)")
    ap.add_argument("--strict", action="store_true", help="also refuse private/metadata addresses and unlisted ports (connect by checked address)")
    a = ap.parse_args(argv)
    proxy = EgressProxy(DEFAULT_DENY + tuple(a.deny), policy=NetPolicy(DEFAULT_DENY + tuple(a.deny)) if a.strict else None)
    srv = serve(proxy, a.host, a.port)
    print(f"egress proxy on {a.host}:{srv.server_address[1]} strict={a.strict} deny={proxy.deny} allow={proxy.allow}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
