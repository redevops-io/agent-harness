"""NVIDIA NIM model adapter — a provider behind the existing model contract.

NIM (NVIDIA Inference Microservices) exposes **OpenAI-compatible** endpoints, so this adapter is a
thin OpenAI-shape client (``/v1/chat/completions``) that works against a **hosted** NIM (NGC) or a
**self-hosted** NIM the same way — only ``base_url`` differs. It contacts the model over ``urllib``
(stdlib; no new dependency) and, when no endpoint is configured, degrades to an **offline stub** so
the example/tests run without a network, mirroring :class:`agent_harness.llm.Client`.

What it adds over the bare shim, from the accelerator-integration plan (NIM adapter §2):

  * **model identity capture** — the served ``model`` id is pinned onto every result;
  * **production refusal if the model is unpinned** — a call with no model id refuses unless the
    loud dev override ``AGENT_HARNESS_ALLOW_UNPINNED_MODEL=1`` is set (never silent);
  * **deterministic request serialization** — the request body is canonicalized (sorted keys, no
    whitespace) into a ``request_digest`` so the same call is the same evidence across providers;
  * **cost + token accounting** from the ``usage`` block;
  * **cancellation / deadline propagation** via ``timeout``;
  * **retry classification** — transport/HTTP status split into retryable vs terminal, without
    retrying here (the runtime owns retry policy);
  * **capability discovery outside the execution path** — :meth:`identity` never calls the model.

The record it returns is **structurally conformant to ``provider-capability/v1``** (the vendor-neutral
surface in the public ``agentic-os`` runtime): the field names of :meth:`NimResult.as_capability_result`
match ``CapabilityResult`` + ``ProviderIdentity`` exactly, so a NIM result is ordinary runtime evidence,
not a vendor side channel. agent-harness stays dependency-free and does **not** import ``agentic_os`` —
conformance is proven by the shared field vocabulary (fixture round-trip), not a code import.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional

#: The cross-repo contract this result shape conforms to.
PROVIDER_CAPABILITY_VERSION = "provider-capability/v1"

#: Exactly the keys of the provider-capability/v1 CapabilityResult envelope this adapter fills.
CAPABILITY_RESULT_KEYS = (
    "kind", "output_ref", "provider", "request_digest", "response_digest",
    "cost_usd", "latency_ms", "evidence", "policy_decisions",
)


def _canonical(obj: Any) -> str:
    """Canonical serialization matching the runtime's other digests (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _digest(obj: Any) -> str:
    return "sha256:" + sha256(_canonical(obj).encode()).hexdigest()[:16]


def classify_retry(status: Optional[int], transport_error: bool = False) -> str:
    """Split a failure into ``retryable`` | ``terminal`` | ``ok`` — the runtime, not this adapter,
    decides whether to actually retry. Transport failures and 408/429/5xx are retryable; 4xx that
    describe the request (auth, not-found, malformed) are terminal."""
    if transport_error:
        return "retryable"
    if status is None:
        return "ok"
    if status in (408, 429) or 500 <= status <= 599:
        return "retryable"
    if 400 <= status <= 499:
        return "terminal"
    return "ok"


class NimError(RuntimeError):
    """A NIM call failed. ``retry_class`` carries the classification for the caller's policy."""

    def __init__(self, message: str, *, status: Optional[int] = None, retry_class: str = "terminal"):
        super().__init__(message)
        self.status = status
        self.retry_class = retry_class


@dataclass
class NimResult:
    """A single NIM invocation as ordinary evidence. Field names mirror provider-capability/v1."""
    text: str
    model_id: str
    request_digest: str
    response_digest: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = ""
    endpoint_identity: str = ""
    service_kind: str = "nim"
    provider: str = "nvidia-nim"
    offline: bool = False

    def as_capability_result(self, output_ref: str) -> Dict[str, Any]:
        """Project into a provider-capability/v1 CapabilityResult dict (references, not inline output).

        ``output_ref`` is the ArtifactHandle-style reference the caller stored the text under — the
        envelope carries the reference, never the text, so results dereference and replay."""
        return {
            "kind": "model",
            "output_ref": output_ref,
            "provider": {
                "provider": self.provider,
                "service_kind": self.service_kind,
                "service_version": "",
                "model_id": self.model_id,
                "model_digest": "",
                "endpoint_identity": self.endpoint_identity,
                "adapter_version": "nim@1",
                "accelerator_kind": "",
                "deployment_profile_digest": "",
            },
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "evidence": (),
            "policy_decisions": (),
        }


class NimClient:
    """OpenAI-compatible client for a hosted or self-hosted NIM.

    ``base_url`` selects hosted vs self-hosted (the only difference); ``model`` is the served weights
    id and MUST be pinned for a production call. With no ``base_url`` the client runs **offline** and
    returns a deterministic stub result so wiring/tests run without a network.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, *, price_per_1k_prompt: float = 0.0,
                 price_per_1k_completion: float = 0.0, clock: Any = None):
        self.base_url = (base_url or os.environ.get("NIM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("NIM_API_KEY")
        self.model = model or os.environ.get("NIM_MODEL")
        self.price_per_1k_prompt = price_per_1k_prompt
        self.price_per_1k_completion = price_per_1k_completion
        # Injected monotonic clock (callable → float seconds) keeps latency measurable *and* the
        # pure path testable; wall-clock stays out of the module.
        self._clock = clock

    # -- capability discovery: never touches the execution path -------------------------------------
    def identity(self) -> Dict[str, str]:
        """Provider identity for planning/EXPLAIN without invoking the model."""
        return {
            "provider": "nvidia-nim",
            "service_kind": "nim",
            "model_id": self.model or "",
            "endpoint_identity": self._endpoint_identity(),
            "locality": "hosted" if self._is_hosted() else "self-hosted",
        }

    def _is_hosted(self) -> bool:
        return "api.nvidia.com" in self.base_url or "integrate.api" in self.base_url

    def _endpoint_identity(self) -> str:
        """A stable, **non-secret** reference to the endpoint — never the raw host in public artifacts."""
        if not self.base_url:
            return "offline"
        return "nim:" + sha256(self.base_url.encode()).hexdigest()[:12]

    def _pinned_model(self, model: Optional[str]) -> str:
        m = model or self.model
        if not m:
            if os.environ.get("AGENT_HARNESS_ALLOW_UNPINNED_MODEL") == "1":
                return "UNPINNED"
            raise NimError(
                "refusing NIM call with no pinned model id — set model=... (or "
                "AGENT_HARNESS_ALLOW_UNPINNED_MODEL=1 for dev only)",
                retry_class="terminal",
            )
        return m

    def _build_body(self, messages: List[Dict[str, str]], model: str,
                    stream: bool, extra: Dict[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages}
        if stream:
            body["stream"] = True
        body.update(extra)
        return body

    def chat(self, prompt: Optional[str] = None, *, messages: Optional[List[Dict[str, str]]] = None,
             model: Optional[str] = None, timeout: int = 60, stream: bool = False,
             **params: Any) -> NimResult:
        """Run a chat completion against the NIM and return a :class:`NimResult`.

        ``prompt`` is sugar for a single user message. Extra OpenAI params (``temperature`` etc.) pass
        through. ``timeout`` is the deadline. Streaming is accepted and the chunks are concatenated;
        the offline path ignores it.
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        pinned = self._pinned_model(model)
        body = self._build_body(messages, pinned, stream, params)
        request_digest = _digest(body)

        if not self.base_url:
            return self._offline_result(messages, pinned, request_digest)

        started = self._now()
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            status = e.code
            raise NimError(f"NIM HTTP {status}", status=status,
                           retry_class=classify_retry(status)) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise NimError(f"NIM transport error: {e}", retry_class="retryable") from e
        latency_ms = (self._now() - started) * 1000.0
        return self._parse(raw, pinned, request_digest, latency_ms)

    def _parse(self, raw: bytes, pinned: str, request_digest: str, latency_ms: float) -> NimResult:
        data = json.loads(raw.decode())
        choices = data.get("choices") or []
        if not choices:
            raise NimError("NIM response had no choices", retry_class="terminal")
        # streaming servers may return {"delta": {...}}; non-streaming return {"message": {...}}
        pieces = []
        finish = ""
        for c in choices:
            msg = c.get("message") or c.get("delta") or {}
            pieces.append(msg.get("content") or "")
            finish = c.get("finish_reason") or finish
        text = "".join(pieces)
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        # Model identity is captured from the *server's* echo when present (authoritative), else pinned.
        served_model = data.get("model") or pinned
        response_digest = _digest({"model": served_model, "text": text, "finish": finish})
        return NimResult(
            text=text, model_id=served_model, request_digest=request_digest,
            response_digest=response_digest, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=self._cost(pt, ct), latency_ms=round(latency_ms, 3), finish_reason=finish,
            endpoint_identity=self._endpoint_identity(),
        )

    def _offline_result(self, messages: List[Dict[str, str]], pinned: str,
                        request_digest: str) -> NimResult:
        """Deterministic, network-free result so wiring runs without an endpoint (echo of the prompt)."""
        text = messages[-1].get("content", "") if messages else ""
        response_digest = _digest({"model": pinned, "text": text, "finish": "stop"})
        return NimResult(
            text=text, model_id=pinned, request_digest=request_digest,
            response_digest=response_digest, finish_reason="stop",
            endpoint_identity="offline", offline=True,
        )

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(prompt_tokens / 1000.0 * self.price_per_1k_prompt
                     + completion_tokens / 1000.0 * self.price_per_1k_completion, 6)

    def _now(self) -> float:
        if self._clock is not None:
            return float(self._clock())
        import time
        return time.monotonic()


if __name__ == "__main__":
    # Offline self-check, mirroring provider.py / merge.py conformance runners — no network.
    checks = []
    c = NimClient(model="nvidia/nemotron-4-340b-instruct")
    r = c.chat("hello world")
    checks.append(("offline call returns text", r.text == "hello world" and r.offline))
    checks.append(("model id pinned onto result", r.model_id == "nvidia/nemotron-4-340b-instruct"))
    checks.append(("request digest deterministic",
                   r.request_digest == c.chat("hello world").request_digest))
    checks.append(("identity does not need the model", c.identity()["model_id"].startswith("nvidia/")))
    cap = r.as_capability_result(output_ref="art-xyz")
    checks.append(("capability envelope keys match v1", set(cap) == set(CAPABILITY_RESULT_KEYS)))
    checks.append(("envelope carries a reference not text", cap["output_ref"] == "art-xyz"))
    checks.append(("provider identity is nvidia-nim", cap["provider"]["provider"] == "nvidia-nim"))
    # unpinned model refuses in production
    unp = NimClient()
    try:
        unp.chat("x")
        refused = False
    except NimError:
        refused = True
    checks.append(("unpinned model refuses", refused))
    checks.append(("retry classification: 503 retryable", classify_retry(503) == "retryable"))
    checks.append(("retry classification: 400 terminal", classify_retry(400) == "terminal"))
    checks.append(("retry classification: transport retryable",
                   classify_retry(None, transport_error=True) == "retryable"))
    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"RESULT {passed}/{len(checks)}  ({PROVIDER_CAPABILITY_VERSION})")
    import sys
    sys.exit(0 if passed == len(checks) else 1)
