"""Run the upstream JobBench judge (``eval/judge.py``) UNCHANGED except for one request parameter: thinking is switched off.

Why it exists: on a large deliverable the judge prompt can fill nearly the whole context (61,716 of 65,536 tokens in the G1 failure), so what is left for the reply (3,820 tokens) is
spent on hidden reasoning and the reply comes back empty with ``finish_reason=length``, every time. A rubric in that state has no verdict; asking again with the same request cannot change that.
This shim is used to recover such rubrics (see ``evalkit.judge_output``), and, with ``AGENT_HARNESS_JUDGE_THINKING_OFF=0``, only to WRITE DOWN the tokens the server reports for each request
(a local model has no API bill, so otherwise what a run consumed is recorded nowhere); in that mode no request, prompt, parser or verdict differs from the upstream judge. The prompt, the rubric, the model, the parsing and the per-rubric persistence are the upstream ones; every
rubric answered this way is marked ``recovered_with: thinking_off`` in the details file, so a run reports how many verdicts came from the recovery mode.

usage: python judge_shim.py PATH/TO/judge.py [the upstream judge's own arguments]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

MARK = "thinking_off"
USAGE_ENV = "AGENT_HARNESS_JUDGE_USAGE_LOG"      # where the tokens the server reports for each judge request are appended (one JSON line each)
THINKING_ENV = "AGENT_HARNESS_JUDGE_THINKING_OFF"  # "1" (the default when run as a script) switches thinking off; "0" leaves the request exactly as the upstream judge builds it (measure only)


def _record(path: Optional[str], line: dict) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:                 # O_APPEND: one short line per request is safe from the judge's worker threads
            f.write(json.dumps({"ts": round(time.time(), 1), **line}) + "\n")
    except OSError:
        pass                                                         # measuring must never change a verdict


def run(argv: List[str], *, thinking_off: bool = True, usage_log: Optional[str] = None) -> None:
    script = Path(argv[0]).resolve()
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("judge", script)
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge"] = J
    spec.loader.exec_module(J)                                      # type: ignore[union-attr]
    original_client, original_rubric = J.get_openai_client, J.judge_rubric

    class _Client:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.chat = self
            self.completions = self

        def create(self, **kw: Any) -> Any:
            if thinking_off:
                extra = dict(kw.get("extra_body") or {})
                extra["chat_template_kwargs"] = {**(extra.get("chat_template_kwargs") or {}), "enable_thinking": False}
                kw = {**kw, "extra_body": extra}
            try:
                out = self._inner.chat.completions.create(**kw)
            except Exception as e:  # noqa: BLE001  (recorded, then raised unchanged)
                _record(usage_log, {"error": type(e).__name__, "detail": str(e)[:200]})
                raise
            u = getattr(out, "usage", None)
            if usage_log and u is not None:
                det = getattr(u, "completion_tokens_details", None)
                _record(usage_log, {"prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None),
                                    "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det is not None else None})
            return out

    def judge_rubric(*a: Any, **k: Any):
        res, dbg = original_rubric(*a, **k)
        if thinking_off and not str((res.get("result") or {}).get("overall_reasoning", "")).startswith("Failed after"):
            res["recovered_with"] = MARK
        return res, dbg

    J.get_openai_client = lambda *a, **k: _Client(original_client(*a, **k))
    J.judge_rubric = judge_rubric                                   # main() resolves the name at call time
    sys.argv = [str(script)] + argv[1:]
    J.main()


if __name__ == "__main__":
    run(sys.argv[1:], thinking_off=os.environ.get(THINKING_ENV, "1") != "0", usage_log=os.environ.get(USAGE_ENV))
