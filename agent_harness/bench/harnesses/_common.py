"""Pieces shared by the OpenCode-based adapters (OpenCode itself and Harness², which drives OpenCode)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def opencode_env(proxy_base: str, providers: Dict[str, Dict[str, Any]], egress: Optional[str] = None,
                 labels: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for OpenCode: every provider is routed through the spend proxy; web goes through the egress filter.

    ``providers``: name -> {model, context, output[, route][, temperature]}. ``route`` is the spend-proxy provider the requests go to (default: the
    name itself); it lets two OpenCode providers with different limits share one backend and model, which OpenCode otherwise cannot
    tell apart. ``labels``: name -> proxy path label (default ``oc``); the label ``think`` makes the proxy switch reasoning on for
    that provider's requests."""
    labels = labels or {}
    cfg = {"provider": {name: {"npm": "@ai-sdk/openai-compatible", "name": name,
                               "options": {"baseURL": f"{proxy_base}/{p.get('route', name)}/{labels.get(name, 'oc')}/v1", "apiKey": "dummy"},
                               "models": {p["model"]: {"name": p["model"], "tool_call": True, "reasoning": bool(labels.get(name) == "think"),
                                                       # OpenCode sends a temperature only if the model declares this capability; without it NOTHING is sent
                                                       # and the server's own default applies (found on the wire 2026-10-23)
                                                       "temperature": bool(p.get("temperature", False)),
                                                       "limit": {"context": p["context"], "output": p["output"]}}}}
                         for name, p in providers.items()}}
    env = {"OPENCODE_CONFIG_CONTENT": json.dumps(cfg), "OPENAI_API_KEY": "dummy"}
    if egress:
        env.update({k: egress for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")})
        env.update({"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
    return env


def opencode_diagnostics(traj_dir: Path) -> Dict[str, Any]:
    """Peak prompt size over an OpenCode rollout: input + cache read + cache write of the largest step."""
    peak, steps = 0, 0
    for tf in sorted(traj_dir.glob("*.jsonl")) if traj_dir.is_dir() else []:
        for line in tf.read_text(errors="replace").splitlines():
            if "{" not in line:
                continue
            try:
                e = json.loads(line[line.index("{"):])
            except ValueError:
                continue
            if e.get("type") == "step_finish":
                t = (e.get("part") or {}).get("tokens") or {}
                c = t.get("cache") or {}
                peak = max(peak, int(t.get("input") or 0) + int(c.get("read") or 0) + int(c.get("write") or 0))
                steps += 1
    return {"peak_prompt": peak, "model_turns": steps}
