import json
from pathlib import Path

import pytest

from agent_harness.bench import jobbench as jb
from agent_harness.bench.tools import bench_registry
from agent_harness.context import (ELIDED, BasicContextManager, NoCompactionManager, PinSet, elide_old_tool_outputs, placeholder_arg,
                                   placeholder_in)
from agent_harness.loop import run_loop
from agent_harness.tools import ToolRegistry, ToolResult, function_tool
from agent_harness.trace import TraceRecorder


class Script:
    """A scripted model: each step is (content, [(tool, args), ...]) ; the last step has no calls."""
    model = "scripted"

    def __init__(self, steps, prompt_tokens=100):
        self.steps, self.i, self.pt = steps, 0, prompt_tokens

    def complete(self, messages, tools):
        content, calls = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        tc = [{"id": f"c{self.i}_{j}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for j, (n, a) in enumerate(calls)] or None
        return {"message": {"role": "assistant", "content": content, "tool_calls": tc}, "usage": {"prompt_tokens": self.pt, "completion_tokens": 1},
                "finish_reason": "stop"}


def registry(log):
    reg = ToolRegistry()

    def write(a):
        log.append(a)
        return ToolResult(text=f"wrote {len(a['content'])} chars")
    reg.register(function_tool("write_file", write, "w", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                                          "required": ["path", "content"]}, side_effecting=False))
    n = {"i": 0}

    def big(a):                                  # distinct each call, so the loop's identical-call short-circuit does not apply
        n["i"] += 1
        return ToolResult(text=("x" * 29990) + f"{n['i']:010d}")
    reg.register(function_tool("big", big, "b"))
    return reg


# -- placeholder guard -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["head...[900 chars omitted to save context]", ELIDED])
def test_placeholder_text_is_recognised(value):
    assert placeholder_in(value) and placeholder_arg({"path": "/tmp/a", "content": value}) == "content"


def test_a_managers_own_stub_format_is_recognised_through_its_detector_not_through_patterns_in_the_loop():
    stub = "<<GONE 7: that text was removed>>"                      # a format this package has never heard of
    assert not placeholder_in(stub)
    assert placeholder_arg({"content": stub}, lambda v: isinstance(v, str) and v.startswith("<<GONE")) == "content"
    assert placeholder_arg({"content": "fine"}, lambda v: isinstance(v, str) and v.startswith("<<GONE")) is None


@pytest.mark.parametrize("value", ["a normal document that mentions the [archived] section", "def f(): return 1", "", 5, None])
def test_real_content_is_not_mistaken_for_a_placeholder(value):
    assert not placeholder_in(value)


def test_an_action_carrying_a_placeholder_is_refused_not_run(tmp_path):
    log = []
    stub = "import openpyxl from openpyxl.styles import Font ...[900 chars omitted to save context]"
    client = Script([("", [("write_file", {"path": "/tmp/d1.py", "content": stub})]),
                     ("", [("write_file", {"path": "/tmp/d1.py", "content": "print('real')"})]), ("done", [])])
    traj = tmp_path / "t.jsonl"
    res = run_loop(client, registry(log), "s", "t", trajectory=traj)
    assert log == [{"path": "/tmp/d1.py", "content": "print('real')"}]                 # only the real write reached the tool
    assert res.tool_errors == 1 and res.stop == "done"
    tool_msgs = [m["content"] for m in res.messages if m["role"] == "tool"]
    assert tool_msgs[0].startswith("ERROR: argument 'content' is a context placeholder") and "Nothing was run" in tool_msgs[0]
    assert any(json.loads(l)["kind"] == "placeholder_blocked" for l in traj.read_text().splitlines())


# -- pins ------------------------------------------------------------------------------------------------------------
def conv(pinned_text):
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i, body in enumerate([pinned_text, "other result one", "other result two", "r3", "r4", "r5", "r6", "r7"]):
        m += [{"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": str(i), "content": body}]
    return m


def test_pinned_text_survives_basic_elision_but_others_do_not():
    text = "TASK INSTRUCTIONS: produce a three-sheet workbook and a PDF report. Reference files: a.csv b.csv"
    mgr = BasicContextManager()
    mgr.pin_text(text)
    msgs = conv("$ cat TASK_INSTRUCTIONS.txt\n" + text + "\n")
    mgr.prepare(msgs, target_tokens=10, estimated_tokens=10**6)
    tools = [m["content"] for m in msgs if m["role"] == "tool"]
    assert "TASK INSTRUCTIONS" in tools[0] and tools[1] == ELIDED and tools[2] == ELIDED          # pinned kept, old others elided
    assert tools[-4:] == ["r4", "r5", "r6", "r7"]                                                  # the newest four stay as before


def test_without_a_pin_the_instructions_are_elided_as_before():
    msgs = conv("TASK INSTRUCTIONS: produce a three-sheet workbook")
    BasicContextManager().prepare(msgs, target_tokens=10, estimated_tokens=10**6)
    assert [m["content"] for m in msgs if m["role"] == "tool"][0] == ELIDED


def test_pin_whitespace_differences_do_not_matter_and_tiny_pins_are_ignored():
    p = PinSet()
    p.add("   short ")
    assert not p
    p.add("Reference   Files:\n a.csv   and   b.csv  are required for this task")
    assert p.matches("$ cat x\nReference Files: a.csv and b.csv are required for this task\nmore")


def test_the_none_manager_also_honours_pins_in_its_last_resort(tmp_path):
    mgr = NoCompactionManager()
    mgr.pin_text("TASK INSTRUCTIONS: produce a three-sheet workbook and a PDF")
    msgs = conv("TASK INSTRUCTIONS: produce a three-sheet workbook and a PDF")
    assert mgr.prepare(msgs, target_tokens=1, estimated_tokens=10**6) == 0
    mgr.on_context_error(msgs, 1)
    assert "TASK INSTRUCTIONS" in [m["content"] for m in msgs if m["role"] == "tool"][0]


def test_pin_task_instructions_reads_the_file_and_creates_the_default_manager(tmp_path):
    (tmp_path / "TASK_INSTRUCTIONS.txt").write_text("Please build the quarterly reconciliation workbook with three sheets and a PDF summary.")
    mgr = jb.pin_task_instructions(None, tmp_path)
    assert isinstance(mgr, BasicContextManager) and mgr.pins
    assert mgr.pins.matches("... Please build the quarterly reconciliation workbook with three sheets and a PDF summary. ...")


# -- read_file on an archive id ---------------------------------------------------------------------------------------
class FakeSbx:
    def run(self, *a, **k):
        return ""


@pytest.mark.parametrize("path", ["A0001", "/work/A0001", "A0001.txt", "/work/output/A12345"])
def test_read_file_on_an_archive_id_points_to_read_archive(tmp_path, path):
    r = bench_registry(tmp_path, FakeSbx(), web=False).run("read_file", {"path": path})
    assert not r.ok and "read_archive" in r.error and "archive id" in r.error


def test_read_file_on_a_real_file_is_unchanged(tmp_path):
    (tmp_path / "A_notes.txt").write_text("hello")
    r = bench_registry(tmp_path, FakeSbx(), web=False).run("read_file", {"path": "/work/A_notes.txt"})
    assert r.ok and r.text == "hello"


# -- trace recorder -------------------------------------------------------------------------------------------------
def run_traced(tmp_path, with_trace):
    log = []
    steps = [("", [("big", {})])] * 6 + [("done", [])]
    rec = TraceRecorder(tmp_path / "trace") if with_trace else None
    res = run_loop(Script(steps, prompt_tokens=900), registry(log), "s", "t", max_context_tokens=1000, max_tool_chars=2000, trace=rec, supervision=None)
    return res, rec


def test_tracing_does_not_change_the_conversation(tmp_path):
    a, _ = run_traced(tmp_path / "a", True)
    b, _ = run_traced(tmp_path / "b", False)
    assert json.dumps(a.messages, sort_keys=True) == json.dumps(b.messages, sort_keys=True) and a.turns == b.turns


def test_trace_keeps_canonical_text_and_shows_what_was_rewritten(tmp_path):
    res, rec = run_traced(tmp_path, True)
    rows = [json.loads(l) for l in (tmp_path / "trace" / "trace.jsonl").read_text().splitlines()]
    proj = [r for r in rows if r["kind"] == "projection"]
    assert len(proj) == res.turns and proj[0]["rewritten"] == 0
    last = proj[-1]
    assert last["rewritten"] > 0 and last["changed_since_prev"]                        # old tool outputs were elided by the basic manager
    rewritten = [it for it in last["items"] if it[2] != it[3]]
    for it in rewritten:                                                                   # canonical original is still in the blob store
        canon = (tmp_path / "trace" / "blobs" / f"{it[3]}.txt").read_text()
        assert len(canon) == it[5] and it[4] < it[5]
    tools = [r for r in rows if r["kind"] == "tool_full"]
    assert tools and all(t["out_len"] == 30000 and t["admitted_len"] < 30000 and t["cut"] for t in tools)
    assert (tmp_path / "trace" / "blobs" / f"{tools[0]['out_sha']}.txt").read_text() == "x" * 29990 + "0000000001"   # full result kept verbatim


def test_trace_keeps_full_assistant_arguments_not_300_characters(tmp_path):
    log = []
    big = "y" * 5000
    rec = TraceRecorder(tmp_path / "trace")
    run_loop(Script([("", [("write_file", {"path": "/p", "content": big})]), ("done", [])]), registry(log), "s", "t", trace=rec)
    rows = [json.loads(l) for l in (tmp_path / "trace" / "trace.jsonl").read_text().splitlines()]
    a = [r for r in rows if r["kind"] == "assistant_full"][0]["calls"][0]
    assert json.loads((tmp_path / "trace" / "blobs" / f"{a['args_sha']}.txt").read_text())["content"] == big


def test_a_trace_write_failure_never_ends_the_run(tmp_path, capsys):
    rec = TraceRecorder(tmp_path / "trace")
    rec._log.close()                                                                        # simulate a broken disk handle
    res = run_loop(Script([("", [("big", {})]), ("done", [])]), registry([]), "s", "t", trace=rec)
    assert res.stop == "done" and "[trace] disabled" in capsys.readouterr().out
