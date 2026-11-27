"""Run the upstream JobBench judge (``eval/judge.py``) UNCHANGED except for one request parameter: thinking is switched off.

Why it exists: on a large deliverable the judge prompt can fill nearly the whole context (61,716 of 65,536 tokens in the G1 failure), so what is left for the reply (3,820 tokens) is
spent on hidden reasoning and the reply comes back empty with ``finish_reason=length``, every time. A rubric in that state has no verdict; asking again with the same request cannot change that.
This shim is used ONLY to recover such rubrics (see ``evalkit.judge_output``). The prompt, the rubric, the model, the parsing and the per-rubric persistence are the upstream ones; every
rubric answered this way is marked ``recovered_with: thinking_off`` in the details file, so a run reports how many verdicts came from the recovery mode.

usage: python judge_shim.py PATH/TO/judge.py [the upstream judge's own arguments]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List

MARK = "thinking_off"


def run(argv: List[str]) -> None:
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
            extra = dict(kw.get("extra_body") or {})
            extra["chat_template_kwargs"] = {**(extra.get("chat_template_kwargs") or {}), "enable_thinking": False}
            return self._inner.chat.completions.create(**{**kw, "extra_body": extra})

    def judge_rubric(*a: Any, **k: Any):
        res, dbg = original_rubric(*a, **k)
        if not str((res.get("result") or {}).get("overall_reasoning", "")).startswith("Failed after"):
            res["recovered_with"] = MARK
        return res, dbg

    J.get_openai_client = lambda *a, **k: _Client(original_client(*a, **k))
    J.judge_rubric = judge_rubric                                   # main() resolves the name at call time
    sys.argv = [str(script)] + argv[1:]
    J.main()


if __name__ == "__main__":
    run(sys.argv[1:])
