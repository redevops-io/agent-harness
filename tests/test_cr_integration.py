"""Supervisor + loop + the REAL Context Runtime compactor: what the model can see decides what a refusal may say.

Needs the Context Runtime branch with the visibility/placeholder contract (``context_runtime.compression.contract``); skipped otherwise.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("context_runtime.compression.contract")
from context_runtime.integrations.agent_harness import ContextRuntimeManager      # noqa: E402

from agent_harness.bench.sandbox import RunResult                                  # noqa: E402
from agent_harness.bench.tools_parity import DEFAULT_BUDGET, parity_registry       # noqa: E402
from agent_harness.context import BasicContextManager                              # noqa: E402
from agent_harness.integrity import check_message_integrity                        # noqa: E402
from agent_harness.llm import ChatError                                            # noqa: E402
from agent_harness.loop import Supervision, run_loop                               # noqa: E402
from agent_harness.netpolicy import FetchResult                                    # noqa: E402
from agent_harness.supervisor import SupervisorConfig                              # noqa: E402
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool   # noqa: E402
from agent_harness.trace import TraceRecorder                                      # noqa: E402

BIG = "DATA " * 400
SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


class Script:
    def __init__(self, actions, prompt_tokens=2000):
        self.actions, self.i, self.prompt_tokens, self.seen = list(actions), 0, prompt_tokens, []

    def complete(self, messages, tools=None):
        self.seen.append(json.loads(json.dumps(messages)))
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        if callable(a):
            a = a(messages)
        if isinstance(a, Exception):
            raise a
        msg = {"role": "assistant", "content": a} if isinstance(a, str) else {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{self.i}", "type": "function", "function": {"name": a[0], "arguments": json.dumps(a[1])}}]}
        return {"message": msg, "finish_reason": "stop", "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 5}}


def reg(counter=None, schema=SCHEMA, **fns):
    r = ToolRegistry(ApprovalPolicy("allowlist", allow=list(fns)))
    for name, fn in fns.items():
        def run(a, name=name, fn=fn):
            if counter is not None:
                counter[name] = counter.get(name, 0) + 1
            return ToolResult(text=fn(a))
        r.register(function_tool(name, run, name, schema, side_effecting=True))
    return r


def archive_id_in(messages):
    for m in reversed(messages):
        if m["role"] == "tool" and "archived as A" in m["content"]:
            import re
            return re.search(r"archived as (A\d+)", m["content"]).group(1)
        if m["role"] == "tool" and m["content"].startswith("[supervisor] Not run again") :
            import re
            return re.search(r"(A\d{4})", m["content"]).group(1)
    return None


# ---- R1: the archive-blind refusal -------------------------------------------------------------------------------------------------
def test_r1_a_repeat_of_a_call_whose_result_was_archived_gets_a_pointer_to_the_archive_not_a_blanket_refusal(tmp_path):
    n = {}
    mgr = ContextRuntimeManager(tmp_path / "arch", policy="v2")
    follow = lambda msgs: ("read_archive", {"id": archive_id_in(msgs)})
    acts = [("fetch", {"q": "u"})] + [("filler", {"q": str(i)}) for i in range(8)] + [("fetch", {"q": "u"}), follow, "ok"]
    r = reg(n, fetch=lambda a: BIG, filler=lambda a: "F" * 700 + a["q"])
    c = Script(acts)
    res = run_loop(c, r, "s", "t", max_turns=40, max_context_tokens=3000, context_manager=mgr, supervision=Supervision(), check_integrity=True)
    texts = [m["content"] for m in c.seen[-1] if m["role"] == "tool"]
    pointer = next(m["content"] for req in c.seen for m in req if m["role"] == "tool" and m["content"].startswith("[supervisor] Not run again"))     # as the model first saw it
    aid = archive_id_in([{"role": "tool", "content": pointer}])
    assert aid in mgr.archive.entries and mgr.archive.get(aid) == BIG                  # the id exists and the archive holds the original
    assert "use what you already have" not in pointer and "read_archive" in pointer
    assert n["fetch"] == 1 and res.supervisor["counts"]["pointers"] == 1               # the fetch was not run again; the model was told where its data is
    assert any(t.startswith("[" + aid) and "DATA DATA" in t for t in texts)            # and read_archive then returned it
    assert res.stop == "done"


def test_a_repeat_whose_result_is_still_visible_gets_the_concise_note_not_a_pointer(tmp_path):
    n = {}
    mgr = ContextRuntimeManager(tmp_path / "arch")
    c = Script([("fetch", {"q": "u"})] * 6, prompt_tokens=100)
    res = run_loop(c, reg(n, fetch=lambda a: "small and visible"), "s", "t", max_turns=30, context_manager=mgr, supervision=Supervision())
    assert res.supervisor["counts"]["pointers"] == 0 and n["fetch"] == 2 + res.supervisor["counts"]["probes"]
    assert any(m["content"].startswith("[supervisor] Not run: this operation") for m in c.seen[-1] if m["role"] == "tool")


def test_a_result_that_is_gone_and_not_recoverable_may_be_run_again(tmp_path):
    n = {}
    mgr = BasicContextManager()                                                          # lossy elision: nothing is stored
    acts = [("fetch", {"q": "u"})] + [("filler", {"q": str(i)}) for i in range(7)] + [("fetch", {"q": "u"}), "ok"]
    r = reg(n, fetch=lambda a: BIG, filler=lambda a: "F" * 700 + a["q"])
    c = Script(acts)
    res = run_loop(c, r, "s", "t", max_turns=40, max_context_tokens=3000, context_manager=mgr, supervision=Supervision())
    assert n["fetch"] == 2 and res.supervisor["counts"]["pointers"] == 0 and res.supervisor["counts"]["suppressed"] == 0


@pytest.mark.parametrize("has_offset", [True, False])
def test_pointer_for_a_clipped_result_mentions_offset_only_if_the_tool_has_one(tmp_path, has_offset):
    schema = {"type": "object", "properties": {"q": {"type": "string"}, **({"offset": {"type": "integer"}} if has_offset else {})}}
    mgr = ContextRuntimeManager(tmp_path / "arch")
    r = reg({}, schema=schema, page=lambda a: "".join(f"line {i} " + "x" * 50 + "\n" for i in range(400)))
    c = Script([("page", {"q": "u"}), ("page", {"q": "u"}), "ok"], prompt_tokens=100)
    res = run_loop(c, r, "s", "t", max_turns=10, max_tool_chars=3000, context_manager=mgr, supervision=Supervision())
    pointer = next(m["content"] for m in c.seen[-1] if m["role"] == "tool" and m["content"].startswith("[supervisor]"))
    assert "shortened for context" in pointer and "archived as A" in pointer
    assert ("larger offset" in pointer) == has_offset                                     # no advice about a parameter that does not exist


# ---- placeholder contract --------------------------------------------------------------------------------------------------------
def test_placeholder_guard_uses_the_managers_own_detector_for_its_v2_stub(tmp_path):
    n = {}
    mgr = ContextRuntimeManager(tmp_path / "arch", policy="v2")
    stub = "<<ARCHIVED-PLACEHOLDER A0150: write_file.content, 12420 chars. This is NOT the content: never reuse this text as an argument. Original: read_archive(\"A0150\").>>"
    r = reg(n, write=lambda a: "written")
    c = Script([("write", {"q": stub}), ("write", {"q": "real content"}), "ok"], prompt_tokens=100)
    res = run_loop(c, r, "s", "t", max_turns=10, context_manager=mgr, supervision=Supervision())
    assert n["write"] == 1 and res.tool_errors == 1                                      # the placeholder was refused, the real write ran
    assert mgr.is_placeholder(stub) and not BasicContextManager().is_placeholder(stub)  # the format belongs to Context Runtime, not to the harness


# ---- recovery reads stay visible ---------------------------------------------------------------------------------------------------
def test_evidence_the_model_just_recovered_is_not_archived_again_while_it_is_working_with_it(tmp_path):
    mgr = ContextRuntimeManager(tmp_path / "arch", policy="v2")
    follow = lambda msgs: ("read_archive", {"id": archive_id_in(msgs)})
    acts = [("fetch", {"q": "u"})] + [("filler", {"q": str(i)}) for i in range(8)] + [("fetch", {"q": "u"}), follow] + [("filler", {"q": f"b{i}"}) for i in range(4)] + ["ok"]
    r = reg({}, fetch=lambda a: BIG, filler=lambda a: "F" * 700 + a["q"])
    c = Script(acts)
    res = run_loop(c, r, "s", "t", max_turns=60, max_context_tokens=3000, context_manager=mgr, supervision=Supervision(), check_integrity=True)
    last = c.seen[-1]
    recovered = [m for m in last if m["role"] == "tool" and "DATA DATA DATA" in m["content"] and not m["content"].startswith("[archived")]
    assert res.stop == "done" and recovered                                              # four tool results later, the recovered slice is still verbatim


# ---- tool output, budgets, pagination ---------------------------------------------------------------------------------------------
class FakeSbx:
    def run_raw(self, cmd, timeout=120):
        return RunResult("L" * 60000, "E" * 20000, 1)


def test_every_parity_tool_result_fits_the_admission_budget_so_admission_never_cuts_a_page(tmp_path):
    (tmp_path / "t.txt").write_text("0123456789" * 5000)
    text = "word " * 20000
    fetch = lambda url, policy, **kw: FetchResult(True, 200, content_type="text/plain", data=text.encode(), size=len(text), chain=[url])
    r = parity_registry(tmp_path, FakeSbx(), fetch=fetch, output_budget=DEFAULT_BUDGET)
    mgr = ContextRuntimeManager(tmp_path / "arch")
    results = [r.run("bash", {"command": "x"}).text, r.run("read_file", {"path": "/work/t.txt"}).text, r.run("read_file", {"path": "/work/t.txt", "offset": 40000}).text,
               r.run("web_fetch", {"url": "https://x.org/p"}).text, r.run("web_fetch", {"url": "https://x.org/p", "offset": 11200}).text]
    for t in results:
        assert len(t) <= DEFAULT_BUDGET and mgr.admit("tool", t, DEFAULT_BUDGET) == t         # unchanged by admission: continuation hints and pages are intact
    assert "offset=11200" in results[1] and "offset=11200" in results[3]
    small = parity_registry(tmp_path, FakeSbx(), fetch=fetch, output_budget=4000)
    assert len(small.run("read_file", {"path": "/work/t.txt"}).text) <= 4000 and len(small.run("bash", {"command": "x"}).text) <= 4000


# ---- model-visible integrity through compaction, context errors and pushes --------------------------------------------------------------
def test_every_request_sent_through_real_compaction_and_server_context_errors_is_structurally_sound(tmp_path):
    mgr = ContextRuntimeManager(tmp_path / "arch", policy="v2")
    big_args = {"q": "x" * 3000}
    acts = []
    for i in range(30):
        acts.append(("filler", {"q": f"{i}" + "y" * 900}))
        if i in (9, 19):
            acts.append(ChatError("HTTP 400: This model's maximum context length is 3000 tokens. However, your prompt contains at least 3100"))
    acts += ["done"]
    r = reg({}, filler=lambda a: "R" * 1500)
    c = Script(acts)
    res = run_loop(c, r, "s", "t", max_turns=100, max_context_tokens=3000, context_manager=mgr, supervision=Supervision(), check_integrity=True)
    assert res.stop == "done"
    assert all(check_message_integrity(req) == [] for req in c.seen)                      # no request ever carried a corrupt history
    assert all(len(b) >= len(a) for a, b in zip(c.seen, c.seen[1:]))                      # messages are rewritten in place, never removed or reordered
    assert len(mgr.archive.entries) > 0


def test_the_trace_keeps_the_canonical_history_while_compaction_and_supervision_act(tmp_path):
    mgr = ContextRuntimeManager(tmp_path / "arch", policy="v2")
    acts = [("fetch", {"q": "u"})] + [("filler", {"q": str(i)}) for i in range(8)] + [("fetch", {"q": "u"}), ("fetch", {"q": "u"}), "ok"]
    rec = TraceRecorder(tmp_path / "trace")
    r = reg({}, fetch=lambda a: BIG, filler=lambda a: "F" * 700 + a["q"])
    res = run_loop(Script(acts), r, "s", "t", max_turns=40, max_context_tokens=3000, context_manager=mgr, supervision=Supervision(), trace=rec)
    rows = [json.loads(l) for l in (tmp_path / "trace" / "trace.jsonl").read_text().splitlines()]
    tools = [x for x in rows if x["kind"] == "tool_full"]
    first = (tmp_path / "trace" / "blobs" / f"{tools[0]['out_sha']}.txt").read_text()
    assert first == BIG                                                                   # the full original of the archived result is in the trace
    proj = [x for x in rows if x["kind"] == "projection"]
    assert proj[-1]["rewritten"] > 0 and proj[0]["rewritten"] == 0                        # the model saw rewrites; the canonical copies are untouched
