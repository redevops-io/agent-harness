"""Budget-enforcing reverse proxy for OpenAI-compatible providers (OpenAI, xAI, Kimi/Moonshot).

Point a client's base URL at ``http://127.0.0.1:PORT/<provider>/<label>/v1`` with any dummy API key. The proxy
injects the real key from its own environment (so harness subprocesses never see it), forwards the request,
meters usage from the response, appends it to a ledger and refuses new calls (HTTP 402) once a provider is
within ``headroom`` dollars of its cap. ``<label>`` tags the role (executor, editor, judge...) for reports.

Refuses (fails closed): unknown provider or label path, unpriced models, priority/fast service tiers.
A 2xx response with no usage is charged a conservative estimate and flagged ``unmetered``.
It never logs request bodies, headers or keys. Loopback only by default; not a general-purpose proxy.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .spend import Ledger, SpendError, Usage, cost_usd, find_model, load_prices, usage_from_response

_PATH = re.compile(r"^/(?P<provider>[a-z0-9]+)/(?P<label>[A-Za-z0-9_\-]{1,40})(?P<rest>/.*)$")
_METERED = ("/chat/completions", "/responses")
_HOP = {"host", "authorization", "content-length", "connection", "transfer-encoding", "accept-encoding",
        "x-api-key", "proxy-authorization"}
_EXPENSIVE_TIERS = {"priority", "fast", "ultrafast"}


class Proxy:
    def __init__(self, prices: Dict[str, Any], ledger: Ledger, keys: Dict[str, str], safety: float = 1.10,
                 upstreams: Optional[Dict[str, str]] = None):
        self.prices, self.ledger, self.keys, self.safety = prices, ledger, keys, safety
        self.upstreams = upstreams or {p: v["base_url"] for p, v in prices["providers"].items()}

    def handler(self):
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"   # close-delimited bodies: simple and fine for SSE

            def log_message(self, *a):      # never log paths/headers
                pass

            def _send_json(self, status, typ, msg):
                body = json.dumps({"error": {"message": msg, "type": typ}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def _handle(self, method):
                m = _PATH.match(self.path.split("?", 1)[0])
                if not m or m["provider"] not in proxy.upstreams or m["provider"] not in proxy.keys:
                    return self._send_json(404, "not_found", "unknown provider or malformed path")
                provider, label, rest = m["provider"], m["label"], m["rest"]
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                metered = method == "POST" and rest.endswith(_METERED)
                if method == "GET" and rest.rstrip("/") != "/v1/models":
                    return self._send_json(405, "method_not_allowed", "only GET /models and metered POSTs")
                if method == "POST" and not metered:
                    return self._send_json(405, "method_not_allowed", "endpoint not metered; refused")
                body: Dict[str, Any] = {}
                model, stream = "", False
                if metered:
                    if not proxy.ledger.allowed(provider):
                        return self._send_json(402, "budget_exceeded",
                                               f"{provider} budget exhausted (spent ${proxy.ledger.spent(provider):.2f})")
                    try:
                        body = json.loads(raw or b"{}")
                        model = str(body["model"])
                        find_model(proxy.prices, provider, model)
                    except (ValueError, KeyError, SpendError) as e:
                        return self._send_json(400, "unpriced_model", f"refused: {e}")
                    if str(body.get("service_tier", "")).lower() in _EXPENSIVE_TIERS:
                        return self._send_json(400, "unpriced_tier", "priority/fast service tiers are not priced; refused")
                    for k, v in (proxy.prices["providers"][provider].get("request_defaults") or {}).items():
                        body.setdefault(k, v)   # configured per provider; never overrides what the client set
                    ceiling = proxy.prices["providers"][provider].get("clamp_output_tokens")
                    if ceiling:
                        for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                            if isinstance(body.get(k), int) and body[k] > ceiling:
                                body[k] = ceiling
                    stream = bool(body.get("stream"))
                    if stream and rest.endswith("/chat/completions"):
                        so = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
                        body["stream_options"] = {**so, "include_usage": True}
                    raw = json.dumps(body).encode()
                up = urllib.parse.urlparse(proxy.upstreams[provider])
                path = up.path.rstrip("/") + (rest[3:] if rest.startswith("/v1/") else rest)
                if self.path.find("?") >= 0:
                    path += "?" + self.path.split("?", 1)[1]
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
                if proxy.keys[provider]:   # free (self-hosted) providers get no key
                    headers["Authorization"] = "Bearer " + proxy.keys[provider]
                if raw:
                    headers["Content-Length"] = str(len(raw))
                conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
                conn = conn_cls(up.netloc, timeout=600)
                usage: Optional[Usage] = None
                status = 502
                try:
                    conn.request(method, path, body=raw or None, headers=headers)
                    resp = conn.getresponse()
                    status = resp.status
                    self.send_response(status)
                    ctype = resp.getheader("Content-Type", "application/json")
                    self.send_header("Content-Type", ctype)
                    self.end_headers()
                    if "text/event-stream" in ctype:
                        for line in iter(resp.readline, b""):
                            self.wfile.write(line)
                            self.wfile.flush()
                            if line.startswith(b"data:") and b"[DONE]" not in line:
                                try:
                                    u = usage_from_response(json.loads(line[5:]))
                                except ValueError:
                                    u = None
                                usage = u or usage
                    else:
                        data = resp.read()
                        self.wfile.write(data)
                        if metered:
                            try:
                                obj = json.loads(data)
                                usage = usage_from_response(obj)
                                model = str(obj.get("model") or model)
                            except ValueError:
                                pass
                except (OSError, http.client.HTTPException):
                    if status == 502:
                        try:
                            self._send_json(502, "upstream_error", "upstream unreachable")
                        except OSError:
                            pass
                finally:
                    conn.close()
                    if metered and (status < 300 or usage):
                        proxy._account(provider, label, model, rest, status, usage, stream, len(raw), body)
                    elif metered:   # failed calls cost nothing but must stay visible for diagnosis
                        proxy.ledger.record(provider, label, str(body.get("model", model)), rest, status, None, 0.0, stream)

        return H

    def _account(self, provider, label, model, path, status, usage, stream, req_bytes, body):
        unmetered = usage is None
        if unmetered:   # conservative: ~3 bytes/token in, and the largest output the request allowed (default 8192)
            out = int(body.get("max_output_tokens") or body.get("max_completion_tokens") or body.get("max_tokens") or 8192)
            usage = Usage(input_tokens=max(1, req_bytes // 3), cached_tokens=0, output_tokens=out)
        try:
            cost = cost_usd(self.prices, provider, model, usage, self.safety)
        except SpendError:   # response named a model we cannot price: charge by the requested model instead
            cost = cost_usd(self.prices, provider, str(body.get("model", model)), usage, self.safety)
        self.ledger.record(provider, label, str(body.get("model", model)), path, status, usage, cost, stream, unmetered)


def serve(proxy: Proxy, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), proxy.handler())
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cap", action="append", default=[], metavar="PROVIDER=USD", help="repeatable, e.g. openai=20")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--headroom", type=float, default=1.0, help="refuse new calls this many USD below the cap")
    ap.add_argument("--report", action="store_true", help="print the ledger report and exit")
    a = ap.parse_args(argv)
    caps = {k: float(v) for k, v in (c.split("=", 1) for c in a.cap)}
    prices = load_prices()
    free = {p for p, v in prices["providers"].items() if v.get("free")}
    ledger = Ledger(a.ledger, caps, a.headroom, free)
    if a.report:
        print(json.dumps(ledger.report(), indent=2, sort_keys=True))
        return 0
    keys = {p: os.environ[prices["providers"][p]["key_env"]] for p in caps if os.environ.get(prices["providers"][p]["key_env"] or "")}
    keys.update({p: "" for p in free})
    missing = [p for p in caps if p not in keys]
    if missing:
        print("missing API key env for: " + ", ".join(missing), file=sys.stderr)
        return 2
    srv = serve(Proxy(prices, ledger, keys), "127.0.0.1", a.port)
    print(f"spend proxy on 127.0.0.1:{srv.server_address[1]} caps={caps} (ledger {a.ledger})", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
