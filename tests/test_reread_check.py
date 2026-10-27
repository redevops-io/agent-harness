"""Opt-in source re-read check (off by default). Motivation: in the 11-task licensing run the task files were read once at turn 2, were archived by compaction, and the deliverable written
at turn 111 contained principals that are in none of them. The check asks, once before the first write into the output directory and once before a finish, for a re-read of input files whose
earlier read has left the context. It never judges content."""
import json
from types import SimpleNamespace

from agent_harness.context import BasicContextManager
from agent_harness.loop import Supervision, run_loop
from agent_harness.supervisor import Supervisor, SupervisorConfig
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool

INPUTS = ["pending_applications.csv", "rules.csv", "TASK_INSTRUCTIONS.txt"]


class FakeCM:
    """Says which tool results the model can still see."""
    def __init__(self):
        self.hidden = set()

    def visibility(self, messages, call_id):
        return SimpleNamespace(state="archived" if call_id in self.hidden else "visible", archive_id="a1" if call_id in self.hidden else None)


def sup(cm=None, **cfg):
    return Supervisor(SupervisorConfig(**cfg), max_turns=200, max_seconds=3600, context=cm, inputs=INPUTS)


def read(s, cid, name, turn):
    s.begin_turn(turn, float(turn))
    assert s.before(cid, "read_file", {"path": f"/work/task_folder/{name}"}, []).run
    s.after(cid, "read_file", {"path": f"/work/task_folder/{name}"}, "contents", True)


WRITE = ("write_file", {"path": "/work/output/report.xlsx", "content": "x"})


def test_off_by_default():
    cm = FakeCM(); s = sup(cm)
    read(s, "c1", "pending_applications.csv", 1); cm.hidden.add("c1")
    s.begin_turn(50, 50.0)
    assert s.before("w", *WRITE, []).run


def test_the_first_output_write_is_held_once_when_a_source_read_has_left_the_context():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); read(s, "c2", "rules.csv", 2); cm.hidden |= {"c1", "c2"}
    s.begin_turn(50, 50.0)
    d = s.before("w1", *WRITE, [])
    assert not d.run and d.kind == "reread" and "pending_applications.csv" in d.text and "rules.csv" in d.text and "This call was not run" in d.text
    assert s.counts["reread_gates"] == 1
    assert s.before("w2", *WRITE, []).run                                 # once per run, never a loop


def test_no_hold_while_the_sources_are_still_visible_or_were_read_again():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1)
    s.begin_turn(5, 5.0); assert s.before("w", *WRITE, []).run            # visible
    s2 = sup(cm, reread_check=True)
    read(s2, "d1", "pending_applications.csv", 1); cm.hidden.add("d1")
    s2.begin_turn(40, 40.0)
    assert s2.before("d2", "bash", {"command": "cat /work/task_folder/pending_applications.csv"}, []).run
    s2.after("d2", "bash", {"command": "cat /work/task_folder/pending_applications.csv"}, "contents", True)      # read again (differently) after the first read left the context
    s2.begin_turn(50, 50.0); assert s2.before("w", *WRITE, []).run


def test_the_pointer_to_an_archived_input_and_the_read_back_count_as_a_re_read():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); cm.hidden.add("c1")
    s.begin_turn(40, 40.0)
    d = s.before("c2", "read_file", {"path": "/work/task_folder/pending_applications.csv"}, [])
    assert not d.run and d.kind == "pointer"                              # the supervisor's existing answer for an identical call whose result was archived
    s.begin_turn(41, 41.0)
    assert s.before("c3", "read_archive", {"id": "a1"}, []).run
    s.after("c3", "read_archive", {"id": "a1"}, "contents", True)
    s.begin_turn(50, 50.0)
    assert s.before("w", *WRITE, []).run and s.counts.get("reread_gates", 0) == 0


def test_writing_elsewhere_is_not_held():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); cm.hidden.add("c1")
    s.begin_turn(50, 50.0)
    assert s.before("a", "write_file", {"path": "/work/scratch.py", "content": "x"}, []).run
    assert s.before("b", "bash", {"command": "python3 analyze.py && ls /work/output"}, []).run
    assert s.before("c", "bash", {"command": "cat /work/output/a.csv | head"}, []).run


def test_shell_writes_into_the_output_directory_are_held():
    for cmd in ("df.to_excel('/work/output/a.xlsx')", "python3 -c \"wb.save('/work/output/a.xlsx')\"", "cp /work/a.csv /work/output/a.csv", "echo hi > /work/output/a.txt"):
        cm = FakeCM(); s = sup(cm, reread_check=True)
        read(s, "c1", "rules.csv", 1); cm.hidden.add("c1"); s.begin_turn(50, 50.0)
        assert not s.before("w", "bash", {"command": cmd}, []).run, cmd


def test_without_a_context_manager_that_reports_visibility_nothing_is_claimed():
    s = sup(None, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); s.begin_turn(50, 50.0)
    assert s.before("w", *WRITE, []).run


def test_finishing_after_a_write_made_without_the_sources_in_view_is_sent_back_once():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); cm.hidden.add("c1")
    s.begin_turn(50, 50.0)
    assert not s.before("w1", *WRITE, []).run                             # held once
    assert s.before("w2", *WRITE, []).run; s.after("w2", *WRITE, "wrote", True)          # the model writes anyway
    s.begin_turn(60, 60.0)
    d = s.on_finish_attempt()
    assert d.action == "push" and "pending_applications.csv" in d.text and "Before you finish" in d.text
    assert s.on_finish_attempt().action == "accept"


def test_finishing_is_not_held_when_the_sources_were_read_again_after_the_write():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); cm.hidden.add("c1"); s.begin_turn(50, 50.0)
    s.before("w1", *WRITE, []); assert s.before("w2", *WRITE, []).run; s.after("w2", *WRITE, "wrote", True)
    s.begin_turn(55, 55.0)
    cmd = {"command": "cat /work/task_folder/pending_applications.csv"}
    assert s.before("c3", "bash", cmd, []).run; s.after("c3", "bash", cmd, "contents", True); cm.hidden.add("c3")             # compared with the source after writing (and that read has scrolled away again)
    s.begin_turn(60, 60.0)
    assert s.on_finish_attempt().action == "accept"


def test_finishing_without_any_output_write_is_left_to_the_deliverable_check():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); cm.hidden.add("c1"); s.begin_turn(60, 60.0)
    assert s.on_finish_attempt().action == "accept"


def test_the_instructions_file_is_not_a_source_to_re_read():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "TASK_INSTRUCTIONS.txt", 1); cm.hidden.add("c1"); s.begin_turn(50, 50.0)
    assert s.before("w", *WRITE, []).run and s.on_finish_attempt().action == "accept"


# ---- through the real loop ----------------------------------------------------------------------------------------------------------------------
class HidingCM(BasicContextManager):
    """A context manager that hides every tool result older than ``keep`` turns, as compaction would."""
    def __init__(self, keep=2):
        super().__init__()
        self.keep = keep

    def visibility(self, messages, call_id):
        tools = [m for m in messages if m.get("role") == "tool"]
        pos = next((i for i, m in enumerate(tools) if m.get("tool_call_id") == call_id), None)
        stale = pos is not None and len(tools) - pos > self.keep
        return SimpleNamespace(state="archived" if stale else "visible", archive_id="x" if stale else None)


def test_through_the_loop_the_model_is_asked_to_re_read_then_writes():
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=["read_file", "write_file", "bash"]))
    for n in ("read_file", "write_file", "bash"):
        reg.register(function_tool(n, lambda a: ToolResult(text="ok"), n, {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "command": {"type": "string"}}}, side_effecting=True))
    script = [("read_file", {"path": "/work/task_folder/rules.csv"})] + [("bash", {"command": f"echo {i}"}) for i in range(4)] + [WRITE, ("read_file", {"path": "/work/task_folder/rules.csv"}), WRITE, "finished"]
    seen = []

    class C:
        i = 0

        def complete(self, messages, tools=None):
            seen.append(json.loads(json.dumps(messages)))
            a = script[min(self.i, len(script) - 1)]; self.i += 1
            msg = {"role": "assistant", "content": a} if isinstance(a, str) else {"role": "assistant", "content": "", "tool_calls": [{"id": f"k{self.i}", "type": "function", "function": {"name": a[0], "arguments": json.dumps(a[1])}}]}
            return {"message": msg, "finish_reason": "stop", "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
    res = run_loop(C(), reg, "s", "t", max_turns=100, context_manager=HidingCM(2), supervision=Supervision(SupervisorConfig(reread_check=True), inputs=INPUTS), check_integrity=True)
    texts = [m["content"] for m in seen[-1] if m.get("role") == "tool"]
    assert res.stop == "done" and sum("[check]" in t for t in texts) == 1 and res.supervisor["counts"]["reread_gates"] == 1


def test_no_hold_once_the_delivery_reserve_has_started():
    cm = FakeCM(); s = sup(cm, reread_check=True)
    read(s, "c1", "pending_applications.csv", 1); cm.hidden.add("c1")
    s.begin_turn(195, 100.0)                                               # 5 turns left of 200: delivering matters more than checking
    assert s.delivery_mode and s.before("w", *WRITE, []).run and s.counts.get("reread_gates", 0) == 0
