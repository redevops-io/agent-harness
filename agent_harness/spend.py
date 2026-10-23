"""Spend accounting: price table, per-request cost, and an append-only ledger with hard caps.

Fails closed: a model with no price, or a request for a pricier service tier than the table models, is not
priced and therefore not allowed through ``spend_proxy``. Costs are multiplied by ``safety`` (default 1.10)
to cover effects the table does not model (provider rounding, cache-write fees).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PRICES_PATH = Path(__file__).with_name("data") / "prices.json"


class SpendError(Exception):
    pass


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0     # total prompt tokens, INCLUDING cached ones
    cached_tokens: int = 0
    output_tokens: int = 0    # includes reasoning tokens


def load_prices(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads(Path(path or PRICES_PATH).read_text(encoding="utf-8"))


def find_model(prices: Dict[str, Any], provider: str, model: str) -> Dict[str, Any]:
    models = prices["providers"][provider]["models"]
    if model in models:
        return models[model]
    # dated aliases such as gpt-5.4-mini-2026-03-17 -> gpt-5.4-mini (longest key that is a prefix)
    keys = [k for k in models if model.startswith(k + "-")]
    if not keys:
        raise SpendError(f"no price for model {model!r} on provider {provider!r}")
    return models[max(keys, key=len)]


def cost_usd(prices: Dict[str, Any], provider: str, model: str, usage: Usage, safety: float = 1.10) -> float:
    p = find_model(prices, provider, model)
    tier = p
    if "long" in p and usage.input_tokens >= p["long"]["threshold"]:
        tier = p["long"]
    cached = min(usage.cached_tokens, usage.input_tokens)
    raw = ((usage.input_tokens - cached) * tier["input"] + cached * tier["cached"]
           + usage.output_tokens * tier["output"]) / 1e6
    return raw * safety


def usage_from_response(obj: Dict[str, Any]) -> Optional[Usage]:
    """Extract usage from a chat-completions or Responses API object (or a stream's final event)."""
    if not isinstance(obj, dict):
        return None
    u = obj.get("usage")
    if not isinstance(u, dict):
        choices = obj.get("choices")  # Moonshot's streaming style: usage inside the last choice
        u = choices[0].get("usage") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if not isinstance(u, dict):
        resp = obj.get("response")
        u = resp.get("usage") if isinstance(resp, dict) else None
    if not isinstance(u, dict):
        return None
    inp = u.get("prompt_tokens", u.get("input_tokens"))
    out = u.get("completion_tokens", u.get("output_tokens"))
    if inp is None or out is None:
        return None
    details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", u.get("cached_tokens", 0)) or 0
    return Usage(int(inp), int(cached), int(out))


class Ledger:
    """Append-only JSONL of metered calls; totals are rebuilt from the file so caps survive restarts."""

    def __init__(self, path: Path, caps: Dict[str, float], headroom: float = 1.0, free: Optional[set] = None):
        self.path = Path(path)
        self.caps = dict(caps)
        self.free = set(free or ())
        self.headroom = headroom
        self._lock = threading.Lock()
        self._spent: Dict[str, float] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._spent[row["provider"]] = self._spent.get(row["provider"], 0.0) + row["cost_usd"]

    def spent(self, provider: str) -> float:
        with self._lock:
            return self._spent.get(provider, 0.0)

    def allowed(self, provider: str) -> bool:
        """True while spend is more than ``headroom`` below the cap (headroom absorbs in-flight requests)."""
        if provider in self.free:
            return True
        cap = self.caps.get(provider)
        return cap is not None and self.spent(provider) < cap - self.headroom

    def record(self, provider: str, label: str, model: str, path: str, status: int,
               usage: Optional[Usage], cost: float, stream: bool, unmetered: bool = False,
               params: Optional[Dict[str, Any]] = None) -> None:
        row = {"ts": round(time.time(), 3), "provider": provider, "label": label, "model": model, "path": path,
               "status": status, "stream": stream, "unmetered": unmetered, "cost_usd": round(cost, 6),
               "input_tokens": usage.input_tokens if usage else None,
               "cached_tokens": usage.cached_tokens if usage else None,
               "output_tokens": usage.output_tokens if usage else None}
        if params:                                  # what the request asked for (never its content): lets a run be audited afterwards
            row["params"] = params
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            self._spent[provider] = self._spent.get(provider, 0.0) + cost

    def report(self) -> Dict[str, Any]:
        """Totals per provider and per (provider, label, model)."""
        out: Dict[str, Any] = {"providers": {}, "by_label": {}}
        for line in self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []:
            r = json.loads(line)
            p = out["providers"].setdefault(r["provider"], {"calls": 0, "usd": 0.0, "unmetered_calls": 0, "failed_calls": 0})
            if r["status"] >= 400 or r["status"] == 0:
                p["failed_calls"] += 1
                by = out.setdefault("failures", {})
                key = f'{r["provider"]}/{r["label"]}/{r["model"]} HTTP {r["status"]}'
                by[key] = by.get(key, 0) + 1
                continue
            p["calls"] += 1
            p["usd"] = round(p["usd"] + r["cost_usd"], 6)
            p["unmetered_calls"] += 1 if r["unmetered"] else 0
            k = f'{r["provider"]}/{r["label"]}/{r["model"]}'
            b = out["by_label"].setdefault(k, {"calls": 0, "usd": 0.0, "input": 0, "cached": 0, "output": 0})
            b["calls"] += 1
            b["usd"] = round(b["usd"] + r["cost_usd"], 6)
            b["input"] += r["input_tokens"] or 0
            b["cached"] += r["cached_tokens"] or 0
            b["output"] += r["output_tokens"] or 0
        for prov, p in out["providers"].items():
            p["cap_usd"] = self.caps.get(prov)
        return out
