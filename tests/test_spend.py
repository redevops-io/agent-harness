import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_harness.spend import Ledger, SpendError, Usage, cost_usd, find_model, load_prices, usage_from_response
from agent_harness.spend_proxy import Proxy, serve

PRICES = load_prices()


def test_price_table_is_complete_and_sane():
    assert PRICES["as_of"] and set(PRICES["providers"]) == {"openai", "xai", "kimi", "local", "proxmox"}
    for prov, d in PRICES["providers"].items():
        for name, p in d["models"].items():
            if d.get("free"):
                assert d["key_env"] is None and p["input"] == p["cached"] == p["output"] == 0, (prov, name)
                continue
            assert 0 < p["cached"] <= p["input"] <= p["output"] * 1.0 + 1e-9 or name == "grok-4.3", (prov, name)
            if "long" in p:
                assert p["long"]["input"] >= p["input"] and p["long"]["output"] >= p["output"]


def test_cost_examples_by_hand():
    # gpt-5.4-mini: 1M input (0.5M cached) + 0.2M output = 0.5*0.75 + 0.5*0.075 + 0.2*4.5 = 1.3125
    c = cost_usd(PRICES, "openai", "gpt-5.4-mini", Usage(1_000_000, 500_000, 200_000), safety=1.0)
    assert c == pytest.approx(1.3125)
    # xai long-context tier applies to the WHOLE request once input >= 200k
    short = cost_usd(PRICES, "xai", "grok-4.7", Usage(199_999, 0, 1000), safety=1.0)
    long_ = cost_usd(PRICES, "xai", "grok-4.7", Usage(200_000, 0, 1000), safety=1.0)
    assert short == pytest.approx(199_999 * 2 / 1e6 + 1000 * 6 / 1e6)
    assert long_ == pytest.approx(200_000 * 4 / 1e6 + 1000 * 12 / 1e6)
    assert cost_usd(PRICES, "kimi", "kimi-k2.7-code", Usage(1000, 0, 1000)) == pytest.approx(1.1 * (0.95 + 4.0) / 1000)


def test_dated_alias_and_unknown_model():
    assert find_model(PRICES, "openai", "gpt-5.4-mini-2026-03-17") is PRICES["providers"]["openai"]["models"]["gpt-5.4-mini"]
    with pytest.raises(SpendError):
        find_model(PRICES, "openai", "gpt-9")


def test_usage_parsing_variants():
    assert usage_from_response({"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                          "prompt_tokens_details": {"cached_tokens": 4}}}) == Usage(10, 4, 5)
    assert usage_from_response({"type": "response.completed", "response": {"usage": {
        "input_tokens": 7, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 2}}}}) == Usage(7, 2, 3)
    assert usage_from_response({"choices": [{"usage": {"prompt_tokens": 1, "completion_tokens": 2, "cached_tokens": 1}}]}) == Usage(1, 1, 2)
    assert usage_from_response({"id": "x"}) is None


class Fake(BaseHTTPRequestHandler):
    seen = []
    omit_usage = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body,
                          "xkey": self.headers.get("X-Api-Key")})
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 0}}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
            if body.get("stream_options", {}).get("include_usage") and not Fake.omit_usage:
                self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        out = {"model": body["model"], "choices": [{"message": {"content": "ok"}}]}
        if not Fake.omit_usage:
            out["usage"] = usage
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def rig(tmp_path):
    Fake.seen, Fake.omit_usage = [], False
    up = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=up.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    ledger = Ledger(tmp_path / "l.jsonl", {"openai": 5.0, "xai": 5.0}, headroom=0.0)
    proxy = Proxy(PRICES, ledger, {"openai": "REAL-KEY", "xai": "XAI-KEY"}, safety=1.0,
                  upstreams={"openai": f"http://127.0.0.1:{up.server_address[1]}/v1",
                             "xai": f"http://127.0.0.1:{up.server_address[1]}/v1"})
    srv = serve(proxy)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, ledger, tmp_path
    srv.shutdown()
    up.shutdown()


def post(base, path, body, key="dummy"):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key, "X-Api-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_non_stream_metered_key_injected_and_label_recorded(rig):
    base, ledger, _ = rig
    st, _ = post(base, "/openai/executor/v1/chat/completions", {"model": "gpt-5.4-mini", "messages": []})
    assert st == 200
    assert Fake.seen[0]["auth"] == "Bearer REAL-KEY" and Fake.seen[0]["xkey"] is None   # client key never forwarded
    assert Fake.seen[0]["path"] == "/v1/chat/completions"
    expect = (1000 * 0.75 + 500 * 4.5) / 1e6
    assert ledger.spent("openai") == pytest.approx(expect)
    assert ledger.report()["by_label"]["openai/executor/gpt-5.4-mini"]["calls"] == 1


def test_stream_gets_usage_injected_and_metered(rig):
    base, ledger, _ = rig
    st, body = post(base, "/xai/editor/v1/chat/completions", {"model": "grok-4.7", "stream": True, "messages": []})
    assert st == 200 and b"[DONE]" in body
    assert Fake.seen[0]["body"]["stream_options"]["include_usage"] is True
    assert ledger.spent("xai") == pytest.approx((1000 * 2 + 500 * 6) / 1e6)


def test_missing_usage_is_charged_conservatively_and_flagged(rig):
    base, ledger, _ = rig
    Fake.omit_usage = True
    post(base, "/openai/x/v1/chat/completions", {"model": "gpt-5.4-mini", "max_tokens": 1000, "messages": []})
    assert ledger.spent("openai") > (1000 * 4.5) / 1e6 * 0.99     # at least the full max_tokens at output price
    assert ledger.report()["providers"]["openai"]["unmetered_calls"] == 1


def test_cap_blocks_with_402_and_survives_restart(rig):
    base, ledger, tmp = rig
    ledger.caps["openai"] = 0.001
    st, body = post(base, "/openai/x/v1/chat/completions", {"model": "gpt-5.4-mini", "messages": []})
    assert st == 200                       # first call is allowed: nothing spent yet
    st, body = post(base, "/openai/x/v1/chat/completions", {"model": "gpt-5.4-mini", "messages": []})
    assert st == 402 and b"budget_exceeded" in body and len(Fake.seen) == 1   # second never reaches upstream
    assert not Ledger(tmp / "l.jsonl", {"openai": 0.001}, 0.0).allowed("openai")  # rebuilt from the file


@pytest.mark.parametrize("path,body,code", [
    ("/openai/x/v1/chat/completions", {"model": "gpt-unknown", "messages": []}, 400),            # unpriced model
    ("/openai/x/v1/chat/completions", {"model": "gpt-5.4-mini", "service_tier": "priority"}, 400),
    ("/nope/x/v1/chat/completions", {"model": "gpt-5.4-mini"}, 404),                              # unknown provider
    ("/openai/x/v1/embeddings", {"model": "text-embedding-3-small", "input": "x"}, 405),          # unmetered endpoint
    ("/kimi/x/v1/chat/completions", {"model": "kimi-k2.7-code"}, 404),                            # provider without key
])
def test_refusals_never_reach_upstream(rig, path, body, code):
    base, _, _ = rig
    assert post(base, path, body)[0] == code
    assert Fake.seen == []


def test_ledger_keys_on_requested_model_not_dated_alias(rig):
    base, ledger, _ = rig
    post(base, "/openai/x/v1/chat/completions", {"model": "gpt-5.4-mini", "messages": []})
    assert list(ledger.report()["by_label"]) == ["openai/x/gpt-5.4-mini"]


def test_free_provider_is_uncapped_unkeyed_and_recorded_at_zero(tmp_path):
    Fake.seen, Fake.omit_usage = [], False
    up = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=up.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    prices = load_prices()
    assert prices["providers"]["proxmox"]["free"] and prices["providers"]["proxmox"]["key_env"] is None
    ledger = Ledger(tmp_path / "l.jsonl", {}, headroom=1.0, free={"proxmox"})
    proxy = Proxy(prices, ledger, {"proxmox": ""}, safety=1.1,
                  upstreams={"proxmox": f"http://127.0.0.1:{up.server_address[1]}/v1"})
    srv = serve(proxy)
    try:
        for _ in range(3):   # no cap, so never refused
            assert post(f"http://127.0.0.1:{srv.server_address[1]}", "/proxmox/exec/v1/chat/completions",
                        {"model": "Qwen3.8-27B", "messages": []}, key="client-key")[0] == 200
        assert all(s["auth"] is None for s in Fake.seen)               # no Authorization sent upstream
        r = ledger.report()
        assert r["providers"]["proxmox"]["usd"] == 0 and r["by_label"]["proxmox/exec/Qwen3.8-27B"]["input"] == 3000
    finally:
        srv.shutdown()
        up.shutdown()


def test_request_defaults_injected_but_never_override_and_failures_are_recorded(tmp_path):
    Fake.seen, Fake.omit_usage = [], False
    up = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=up.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    prices = load_prices()
    assert prices["providers"]["proxmox"]["request_defaults"] == {"chat_template_kwargs": {"enable_thinking": False}}
    ledger = Ledger(tmp_path / "l.jsonl", {}, headroom=1.0, free={"proxmox"})
    proxy = Proxy(prices, ledger, {"proxmox": ""}, upstreams={"proxmox": f"http://127.0.0.1:{up.server_address[1]}/v1"})
    srv = serve(proxy)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        post(base, "/proxmox/x/v1/chat/completions", {"model": "Qwen3.8-27B", "messages": []})
        post(base, "/proxmox/x/v1/chat/completions", {"model": "Qwen3.8-27B", "messages": [],
                                                      "chat_template_kwargs": {"enable_thinking": True}})
        assert Fake.seen[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert Fake.seen[1]["body"]["chat_template_kwargs"] == {"enable_thinking": True}   # client choice wins
    finally:
        srv.shutdown()
        up.shutdown()
    # a refused-by-upstream call (no usage, 4xx) is recorded as a failure, not silently dropped
    class Bad(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(400); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
    bad = ThreadingHTTPServer(("127.0.0.1", 0), Bad)
    threading.Thread(target=bad.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    ledger2 = Ledger(tmp_path / "l2.jsonl", {}, free={"proxmox"})
    srv2 = serve(Proxy(prices, ledger2, {"proxmox": ""}, upstreams={"proxmox": f"http://127.0.0.1:{bad.server_address[1]}/v1"}))
    try:
        assert post(f"http://127.0.0.1:{srv2.server_address[1]}", "/proxmox/x/v1/chat/completions",
                    {"model": "Qwen3.8-27B", "messages": []})[0] == 400
        r = ledger2.report()
        assert r["providers"]["proxmox"]["failed_calls"] == 1 and r["failures"] == {"proxmox/x/Qwen3.8-27B HTTP 400": 1}
        assert r["providers"]["proxmox"]["calls"] == 0
    finally:
        srv2.shutdown()
        bad.shutdown()


def test_output_token_ceiling_clamps_only_larger_values(tmp_path):
    Fake.seen, Fake.omit_usage = [], False
    up = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=up.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    prices = load_prices()
    assert prices["providers"]["proxmox"]["clamp_output_tokens"] == 8192
    proxy = Proxy(prices, Ledger(tmp_path / "l.jsonl", {}, free={"proxmox"}), {"proxmox": ""},
                  upstreams={"proxmox": f"http://127.0.0.1:{up.server_address[1]}/v1"})
    srv = serve(proxy)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        post(base, "/proxmox/x/v1/chat/completions", {"model": "Qwen3.8-27B", "messages": [], "max_completion_tokens": 65536})
        post(base, "/proxmox/x/v1/chat/completions", {"model": "Qwen3.8-27B", "messages": [], "max_tokens": 500})
        post(base, "/proxmox/x/v1/chat/completions", {"model": "Qwen3.8-27B", "messages": []})
        b = [s["body"] for s in Fake.seen]
        assert b[0]["max_completion_tokens"] == 8192 and b[1]["max_tokens"] == 500 and "max_tokens" not in b[2]
    finally:
        srv.shutdown()
        up.shutdown()
