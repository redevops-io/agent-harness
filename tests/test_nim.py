"""Model-free, network-free tests for the NIM adapter — ``urllib`` is monkeypatched, so no GPU,
no NGC endpoint, no torch. Mirrors the redevops-rag embedder test idiom."""
import json

import pytest

from agent_harness.nim import (
    NimClient, NimError, NimResult, classify_retry, CAPABILITY_RESULT_KEYS,
)


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _openai_chat_payload(text="hi there", model="nvidia/nemotron-4-340b-instruct",
                         pt=5, ct=3, finish="stop"):
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    }


def test_chat_posts_openai_shape_and_captures_identity(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        sent["auth"] = req.headers.get("Authorization")
        sent["timeout"] = timeout
        return _Resp(_openai_chat_payload())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = NimClient(base_url="http://nim:8000", api_key="secret",
                  model="nvidia/nemotron-4-340b-instruct",
                  price_per_1k_prompt=1.0, price_per_1k_completion=2.0, clock=iter([0.0, 0.5]).__next__)
    r = c.chat("hello", temperature=0.0, timeout=30)

    assert sent["url"] == "http://nim:8000/v1/chat/completions"
    assert sent["body"]["model"] == "nvidia/nemotron-4-340b-instruct"
    assert sent["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["body"]["temperature"] == 0.0        # extra params pass through
    assert sent["auth"] == "Bearer secret"           # api key -> Authorization
    assert sent["timeout"] == 30                      # deadline propagated
    assert r.model_id == "nvidia/nemotron-4-340b-instruct"   # identity captured from server echo
    assert r.text == "hi there"
    assert r.cost_usd == pytest.approx(5 / 1000 * 1.0 + 3 / 1000 * 2.0)
    assert r.latency_ms == pytest.approx(500.0)       # injected clock: (0.5-0.0)*1000
    assert not r.offline


def test_request_serialization_is_deterministic_and_order_independent(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Resp(_openai_chat_payload()))
    c = NimClient(base_url="http://nim:8000", model="m")
    d1 = c.chat("q", temperature=0.0, top_p=1.0).request_digest
    d2 = c.chat("q", top_p=1.0, temperature=0.0).request_digest   # kwargs in different order
    assert d1 == d2 and d1.startswith("sha256:")


def test_server_echoed_model_is_authoritative(monkeypatch):
    # even if we pin "m", the served model wins for identity capture
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _Resp(_openai_chat_payload(model="nvidia/actual-served")))
    c = NimClient(base_url="http://nim:8000", model="m")
    assert c.chat("q").model_id == "nvidia/actual-served"


def test_unpinned_model_refuses_in_production():
    with pytest.raises(NimError):
        NimClient(base_url="http://nim:8000").chat("q")


def test_unpinned_model_allowed_with_loud_override(monkeypatch):
    monkeypatch.setenv("AGENT_HARNESS_ALLOW_UNPINNED_MODEL", "1")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _Resp(_openai_chat_payload(model="")))
    r = NimClient(base_url="http://nim:8000").chat("q")
    assert r.model_id in ("UNPINNED", "")  # server may echo empty; pinned falls back to UNPINNED


def test_http_error_is_classified(monkeypatch):
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(NimError) as ei:
        NimClient(base_url="http://nim:8000", model="m").chat("q")
    assert ei.value.status == 503 and ei.value.retry_class == "retryable"


def test_transport_error_is_retryable(monkeypatch):
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(NimError) as ei:
        NimClient(base_url="http://nim:8000", model="m").chat("q")
    assert ei.value.retry_class == "retryable"


def test_offline_stub_runs_without_network():
    r = NimClient(model="m").chat("echo me")   # no base_url -> offline
    assert r.offline and r.text == "echo me" and r.endpoint_identity == "offline"


def test_capability_result_envelope_matches_v1_keys():
    r = NimClient(model="m").chat("x")
    cap = r.as_capability_result(output_ref="art-1")
    assert set(cap) == set(CAPABILITY_RESULT_KEYS)
    assert cap["kind"] == "model" and cap["output_ref"] == "art-1"
    assert cap["provider"]["provider"] == "nvidia-nim"


def test_identity_does_not_call_the_model(monkeypatch):
    def fail(*a, **k):  # any network use should blow up
        raise AssertionError("identity must not touch the network")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    ident = NimClient(base_url="http://api.nvidia.com", model="m").identity()
    assert ident["provider"] == "nvidia-nim" and ident["locality"] == "hosted"


@pytest.mark.parametrize("status,transport,expected", [
    (200, False, "ok"), (429, False, "retryable"), (500, False, "retryable"),
    (503, False, "retryable"), (400, False, "terminal"), (404, False, "terminal"),
    (None, True, "retryable"),
])
def test_classify_retry(status, transport, expected):
    assert classify_retry(status, transport_error=transport) == expected
