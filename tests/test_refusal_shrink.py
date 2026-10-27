"""A refusal loop must not grow the prompt without bound, and an unrecoverable context error must be reported as one. Regression for the 11-task run: statisticians attempt 1 reached
65,537 tokens after ~130 refused call/stub pairs (each repeating a ~380-token command) and ended as model_error."""
import json

import pytest

from agent_harness.integrity import check_message_integrity
from agent_harness.llm import ChatError
from agent_harness.loop import REFUSED_ARGS_NOTE, Supervision, run_loop, shrink_refused_calls
from agent_harness.supervisor import SupervisorConfig
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool

BIG = "python3 - <<'EOF'\n" + "\n".join(f"x{i} = load('file{i}.csv')  # check the join and print the keys" for i in range(14)) + "\nEOF"


def reg():
    r = ToolRegistry(ApprovalPolicy("allowlist", allow=["bash"]))
    r.register(function_tool("bash", lambda a: ToolResult(text="True False"), "bash", {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}, side_effecting=True))
    return r


class Loopy:
    def __init__(self, n, final="done", overflow_above=None):
        self.n, self.i, self.seen, self.overflow_above, self.final = n, 0, [], overflow_above, final

    def complete(self, messages, tools=None):
        size = sum(len(json.dumps(m)) for m in messages)
        self.seen.append((size, json.loads(json.dumps(messages))))
        if self.overflow_above and size > self.overflow_above:
            raise ChatError("HTTP 400: This model's maximum context length is 65536 tokens. However, your prompt contains at least 65537 input tokens")
        self.i += 1
        if self.i > self.n:
            return {"message": {"role": "assistant", "content": self.final}, "finish_reason": "stop", "usage": {"prompt_tokens": size // 3, "completion_tokens": 5}}
        return {"message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{self.i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": BIG})}}]},
                "finish_reason": "tool_calls", "usage": {"prompt_tokens": size // 3, "completion_tokens": 5}}


def test_old_refused_calls_are_shortened_and_the_newest_stay_whole():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(5):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": BIG})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "[supervisor] not run"})
    assert shrink_refused_calls(msgs, {"c0", "c1", "c2", "c3", "c4"}, keep_full=2) == 3
    args = [m["tool_calls"][0]["function"]["arguments"] for m in msgs if m["role"] == "assistant"]
    assert [REFUSED_ARGS_NOTE in a for a in args] == [True, True, True, False, False] and len(msgs) == 12
    assert check_message_integrity(msgs) == [] and shrink_refused_calls(msgs, {"c0", "c1", "c2", "c3", "c4"}, keep_full=2) == 0      # idempotent


def test_a_call_that_ran_is_never_shortened():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "ran", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": BIG})}}]}, {"role": "tool", "tool_call_id": "ran", "content": "ok"}]
    assert shrink_refused_calls(msgs, set(), keep_full=0) == 0 and json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"])["command"] == BIG


def test_a_130_turn_refusal_loop_keeps_the_prompt_flat():
    c = Loopy(130)
    res = run_loop(c, reg(), "s", "t", max_turns=200, supervision=Supervision(SupervisorConfig(end_on_exhaustion=False)), check_integrity=True, max_context_tokens=65536)
    assert res.stop == "done" and res.supervisor["counts"]["suppressed"] > 100
    sizes = [s for s, _ in c.seen]
    pair = len(json.dumps(c.seen[-1][1][-2])) + len(json.dumps(c.seen[-1][1][-1]))     # size of one refused pair (call + stub) as it now stands
    full_pair = len(json.dumps({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": BIG})}}]})) + len(json.dumps({"role": "tool", "tool_call_id": "c1", "content": "[supervisor] not run (repeat #99; see the earlier note)"}))
    unshrunk = sizes[12] + (len(sizes) - 13) * full_pair
    assert sizes[-1] < 0.5 * unshrunk # shortened pairs cost well under half of whole ones
    last = c.seen[-1][1]
    assert sum(REFUSED_ARGS_NOTE in json.dumps(m) for m in last if m.get("role") == "assistant") > 100 and check_message_integrity(last) == []
    kept = [m for m in last if m.get("role") == "assistant" and m.get("tool_calls") and json.loads(m["tool_calls"][0]["function"]["arguments"]).get("command") == BIG]
    assert len(kept) >= 2                                              # the first executed calls and the newest refusals stay whole


def test_the_loop_trace_and_trajectory_record_the_shrinking(tmp_path):
    from agent_harness.trace import TraceRecorder
    traj = tmp_path / "t.jsonl"
    c = Loopy(25)
    run_loop(c, reg(), "s", "t", max_turns=200, supervision=Supervision(SupervisorConfig(end_on_exhaustion=False)), trace=TraceRecorder(tmp_path / "trace"), trajectory=traj)
    ev = [json.loads(l) for l in traj.read_text().splitlines()]
    assert any(e["kind"] == "refusal_shrink" for e in ev)


def test_an_unrecoverable_context_error_is_reported_as_context_overflow_not_model_error():
    class Always:
        def complete(self, messages, tools=None):
            raise ChatError("HTTP 400: This model's maximum context length is 65536 tokens. However, your prompt contains at least 65537 input tokens")
    res = run_loop(Always(), reg(), "s", "t", supervision=None)
    assert res.stop == "context_overflow"
    from agent_harness.bench.failure_class import classify
    assert classify(__import__("pathlib").Path("/nonexistent"), {"stop": res.stop, "deliverables": [], "turns": 0})["class"] == "context_overflow"


def test_a_server_context_error_is_repaired_by_shortening_old_refusals_first():
    # the server rejects any prompt above a size; only shortening the refused calls can get below it
    c = Loopy(60, overflow_above=60000)
    res = run_loop(c, reg(), "s", "t", max_turns=200, supervision=Supervision(SupervisorConfig(end_on_exhaustion=False)), check_integrity=True, max_context_tokens=10 ** 9)   # no proactive compaction
    assert res.stop == "done"
