import json

import pytest

from agent_harness.llm import ChatError
from agent_harness.loop import ELIDED, elide_old_tool_outputs, run_loop
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool


class Script:
    """A fake ChatClient that replays scripted assistant messages."""
    def __init__(self, replies, prompt_tokens=100):
        self.replies, self.i, self.prompt_tokens, self.seen = list(replies), 0, prompt_tokens, []

    def complete(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        r = self.replies[self.i]
        self.i += 1
        if isinstance(r, Exception):
            raise r
        return {"message": r, "finish_reason": "stop", "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 5}}


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def reg():
    r = ToolRegistry(ApprovalPolicy("allowlist", allow=["write"]))
    r.register(function_tool("echo", lambda a: ToolResult(text=a["x"] * 2), parameters={"type": "object", "properties": {}}))
    r.register(function_tool("write", lambda a: ToolResult(text="written"), side_effecting=True))
    r.register(function_tool("nuke", lambda a: ToolResult(text="boom"), side_effecting=True))
    return r


def test_runs_tools_then_stops_on_plain_answer(tmp_path):
    c = Script([call("echo", x="ab"), {"role": "assistant", "content": "all done"}])
    res = run_loop(c, reg(), "sys", "do it", trajectory=tmp_path / "t.jsonl")
    assert res.stop == "done" and res.final_text == "all done" and res.turns == 2 and res.tool_calls == 1
    assert c.seen[1][-1] == {"role": "tool", "tool_call_id": "cecho", "content": "abab"}
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert kinds == ["assistant", "tool", "assistant", "end"]


def test_approval_gate_applies_to_loop_tool_calls():
    c = Script([call("nuke"), {"role": "assistant", "content": "ok"}])
    res = run_loop(c, reg(), "s", "t")
    assert res.tool_errors == 1 and "denied" in c.seen[1][-1]["content"]


def test_bad_arguments_unknown_tools_and_crashes_do_not_stop_the_loop():
    bad = {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function",
           "function": {"name": "echo", "arguments": "{not json"}}]}
    c = Script([bad, call("missing"), {"role": "assistant", "content": "fine"}])
    res = run_loop(c, reg(), "s", "t")
    assert res.stop == "done" and res.tool_errors == 2


def test_limits():
    forever = [call("echo", x="a")] * 10
    assert run_loop(Script(forever), reg(), "s", "t", max_turns=3).stop == "max_turns"
    assert run_loop(Script(forever), reg(), "s", "t", max_seconds=-1).stop == "max_seconds"
    assert run_loop(Script([ChatError("down")]), reg(), "s", "t").stop == "model_error"


def test_old_tool_outputs_are_elided_when_context_is_high_and_overflow_is_detected():
    c = Script([call("echo", x="a")] * 8 + [{"role": "assistant", "content": "end"}], prompt_tokens=900)
    res = run_loop(c, reg(), "s", "t", max_context_tokens=1000, compact_at=0.7)
    assert res.stop == "done"
    last = c.seen[-1]
    tools = [m["content"] for m in last if m["role"] == "tool"]
    assert tools.count(ELIDED) == len(tools) - 4          # keeps the 4 most recent
    # nothing left to elide and still above 95% -> stop instead of sending an oversized prompt
    c2 = Script([call("echo", x="a")] * 3, prompt_tokens=990)
    assert run_loop(c2, reg(), "s", "t", max_context_tokens=1000).stop == "context_overflow"


def test_elide_helper_keeps_recent():
    msgs = [{"role": "tool", "content": str(i)} for i in range(6)]
    assert elide_old_tool_outputs(msgs, keep_last=2) == 4 and [m["content"] for m in msgs[-2:]] == ["4", "5"]


def test_context_error_from_server_triggers_compaction_and_retry():
    big = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": json.dumps({"x": "y" * 50})}}]}
    err = ChatError("HTTP 400: This model's maximum context length is 40960 tokens. However, your prompt contains at least 40961")
    c = Script([big, big, big, big, big, big, err, {"role": "assistant", "content": "ok"}])
    res = run_loop(c, reg(), "s", "t", max_context_tokens=10**9)       # proactive limit disabled: only the reactive path
    assert res.stop == "done" and res.final_text == "ok"
    last = c.seen[-1]
    assert [m["content"] for m in last if m["role"] == "tool"].count(ELIDED) == 2   # 6 tool outputs, keep 4 -> 2 elided


def test_persistent_context_error_gives_up_after_three_compactions():
    err = ChatError("HTTP 400: maximum context length is 100 tokens")
    c = Script([call("echo", x="a"), err, err, err, err])
    assert run_loop(c, reg(), "s", "t").stop == "model_error"


def test_parallel_tool_outputs_are_estimated_before_sending():
    many = {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "echo", "arguments": json.dumps({"x": "z" * 6000})}} for i in range(4)]}
    c = Script([call("echo", x="a"), many, {"role": "assistant", "content": "end"}], prompt_tokens=100)
    run_loop(c, reg(), "s", "t", max_context_tokens=10000, compact_at=0.7, max_tool_chars=6000)
    # after the 4 big outputs (~16k estimated tokens > 7k) old outputs were elided before the last call
    assert any(m.get("content") == ELIDED for m in c.seen[-1] if m["role"] == "tool")


# --- robustness fixes found on the Qwen3.8-27B runs ----------------------------------------------------------
from agent_harness.loop import missing_args, shrink_tool_call_args


def reg_req():
    r = ToolRegistry(ApprovalPolicy("allowlist", allow=["write"]))
    r.register(function_tool("write", lambda a: ToolResult(text="ok"), side_effecting=True,
                             parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                         "required": ["path", "content"]}))
    r.register(function_tool("flaky", lambda a: ToolResult(ok=False, error="ERROR: nope"), parameters={"type": "object", "properties": {}}))
    return r


def test_missing_required_arguments_get_an_actionable_message():
    r = reg_req()
    assert missing_args(r, "write", {"path": "a"}).startswith("missing required argument(s): content. write expects: path (string), content (string)")
    assert missing_args(r, "write", {"path": "a", "content": "b"}) is None and missing_args(r, "nope", {}) is None
    c = Script([call("write", path="a"), {"role": "assistant", "content": "done"}])
    res = run_loop(c, r, "s", "t")
    tool_msg = [m for m in c.seen[1] if m["role"] == "tool"][0]["content"]
    assert tool_msg.startswith("ERROR: missing required argument(s): content") and res.tool_errors == 1


def test_error_prefix_is_not_doubled_and_repeated_failures_get_a_nudge():
    c = Script([call("flaky")] * 7 + [{"role": "assistant", "content": "end"}])
    run_loop(c, reg_req(), "s", "t")
    outs = [[m["content"] for m in seen if m["role"] == "tool"] for seen in c.seen][-1]
    assert outs[0] == "ERROR: nope" and "note:" not in outs[4] and "failed 6 times in a row" in outs[5]


def test_shrink_keeps_json_valid_and_recent_calls_intact():
    big = "x" * 5000
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(4):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function", "function": {
            "name": "write", "arguments": json.dumps({"path": f"p{i}", "content": big})}}]})
    assert shrink_tool_call_args(msgs, keep_last=2) == 2
    old = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])
    assert old["path"] == "p0" and len(old["content"]) < 400 and "omitted" in old["content"]
    assert json.loads(msgs[5]["tool_calls"][0]["function"]["arguments"])["content"] == big      # recent call untouched


# --- ContextManager interface ----------------------------------------------------------------------------------
from agent_harness.context import BasicContextManager, ContextManager


class FakeManager:
    def __init__(self):
        self.prepared, self.errors = [], []

    def prepare(self, messages, *, target_tokens, estimated_tokens):
        self.prepared.append((target_tokens, estimated_tokens))
        return 0

    def on_context_error(self, messages, attempt):
        self.errors.append(attempt)
        return 1

    def admit(self, tool, text, max_chars):
        self.admitted = getattr(self, "admitted", []) + [(tool, len(text), max_chars)]
        return text

    def tools(self):
        return [function_tool("search_context", lambda a: ToolResult(text="found"), "retrieve", parameters={"type": "object", "properties": {}})]


def test_custom_manager_is_asked_to_prepare_and_its_tools_are_offered():
    assert isinstance(BasicContextManager(), ContextManager) and isinstance(FakeManager(), ContextManager)
    fm = FakeManager()
    seen_tools = []

    class C(Script):
        def complete(self, messages, tools=None):
            seen_tools.append([t["function"]["name"] for t in tools])
            return super().complete(messages, tools)

    c = C([call("echo", x="a"), call("search_context"), {"role": "assistant", "content": "ok"}], prompt_tokens=900)
    res = run_loop(c, reg(), "s", "t", max_context_tokens=1000, context_manager=fm)
    assert "search_context" in seen_tools[0] and res.stop == "done"
    assert fm.prepared and fm.prepared[0][0] == 500            # target = compact_to * window
    assert any("found" == m["content"] for m in c.seen[-1] if m["role"] == "tool")   # the manager's tool ran


def test_manager_is_consulted_before_giving_up_on_overflow_and_on_server_errors():
    fm = FakeManager()
    c = Script([call("echo", x="a"), call("echo", x="a"), {"role": "assistant", "content": "ok"}], prompt_tokens=990)
    assert run_loop(c, reg(), "s", "t", max_context_tokens=1000, context_manager=fm).stop == "done"   # on_context_error freed room
    assert fm.errors == [3, 3]
    fm2 = FakeManager()
    err = ChatError("HTTP 400: maximum context length is 100 tokens")
    c2 = Script([err, {"role": "assistant", "content": "ok"}])
    assert run_loop(c2, reg(), "s", "t", context_manager=fm2).stop == "done" and fm2.errors == [1]


def test_every_tool_result_goes_through_the_managers_admit_hook():
    fm = FakeManager()
    c = Script([call("echo", x="abc"), {"role": "assistant", "content": "ok"}])
    run_loop(c, reg(), "s", "t", context_manager=fm, max_tool_chars=77)
    assert fm.admitted == [("echo", 6, 77)]
    # the default manager clips: long results are shortened with an omission marker
    c2 = Script([call("echo", x="z" * 500), {"role": "assistant", "content": "ok"}])
    run_loop(c2, reg(), "s", "t", max_tool_chars=300)
    tool = [m for m in c2.seen[1] if m["role"] == "tool"][0]["content"]
    assert len(tool) < 1000 and "omitted" in tool
