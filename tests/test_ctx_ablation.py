from types import SimpleNamespace

from agent_harness.bench import experiment as ex, harnesses, jobbench as jb
from agent_harness.bench.cells import arm_id
from agent_harness.bench.harnesses import ours, ours_h2, ours_h2_nocr, ours_nocr
from agent_harness.context import BasicContextManager, NoCompactionManager
from agent_harness.loop import run_loop
from agent_harness.tools import ToolRegistry
from tests.test_harness_ours_h2 import put, trees


def msgs(n):
    out = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(n):
        out += [{"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": str(i), "content": "r" * 3000}]
    return out


def test_none_changes_nothing_while_the_prompt_fits_but_basic_elides():
    m = msgs(12)
    assert NoCompactionManager().prepare(m, target_tokens=10, estimated_tokens=10**6) == 0 and m == msgs(12)
    assert BasicContextManager().prepare(msgs(12), target_tokens=10, estimated_tokens=10**6) > 0       # the contrast this arm exists for


def test_none_still_has_the_last_resort_fallback_for_a_prompt_the_server_rejects():
    m = msgs(12)
    assert NoCompactionManager().on_context_error(m, 1) > 0
    assert NoCompactionManager().tools() == [] and NoCompactionManager().admit("x", "a" * 20000, 100) != "a" * 20000


def test_the_loop_with_none_never_compacts_proactively():
    class C:
        model = "m"
        n = 0

        def complete(self, messages, tools):
            self.n += 1
            call = [{"id": str(self.n), "type": "function", "function": {"name": "big", "arguments": "{}"}}] if self.n < 8 else None
            return {"message": {"role": "assistant", "content": "done", "tool_calls": call}, "usage": {"prompt_tokens": 40000, "completion_tokens": 1},
                    "finish_reason": "stop"}
    from agent_harness.tools import function_tool, ToolResult
    reg = ToolRegistry()
    reg.register(function_tool("big", lambda a: ToolResult(text="y" * 5000), "big"))
    events = []
    import tempfile, pathlib, json
    t = pathlib.Path(tempfile.mkdtemp()) / "t.jsonl"
    run_loop(C(), reg, "s", "t", max_context_tokens=50000, context_manager=NoCompactionManager(), trajectory=t)
    kinds = [json.loads(l)["kind"] for l in t.read_text().splitlines()]
    assert "compact" not in kinds                                    # 40k of 50k is past the 70% trigger, yet nothing is compacted


def test_make_context_manager_knows_none_and_the_index_and_archive_hooks_are_guarded(tmp_path):
    mgr = jb.make_context_manager("none", tmp_path)
    assert isinstance(mgr, NoCompactionManager) and not hasattr(mgr, "index_files") and not hasattr(mgr, "archive")


def test_the_nocr_arms_are_discoverable_and_pass_context_none_under_their_own_names(tmp_path):
    assert harnesses.load("ours-nocr") is ours_nocr and harnesses.load("ours-h2-nocr") is ours_h2_nocr
    (tmp_path / "tasks.txt").write_text("a/task1\n")
    e = ex.Experiment({"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"),
                       "cells": {"q8": {"provider": "px48", "model": "M", "window": 65536}}, "arms": ["ours", "ours-nocr", "ours-h2", "ours-h2-nocr"]})
    put(e, e.cells[0], "a/task1", *trees())
    seen = []

    def run_task(tasks_base, task, out, client, **kw):
        seen.append((str(out), kw["context"]))
        return {"deliverables": ["r.md"], "stop": "done", "archived_items": 0, "retrieval_calls": 0, "seconds": 1.0}
    for mod in (ours, ours_nocr, ours_h2, ours_h2_nocr):
        mod.generate(e, e.cells[0], 1, run_task=run_task, client_factory=lambda: SimpleNamespace(model="M"))
    assert [(o.split("/exp/")[1].split("/")[0], c) for o, c in seen] == [("ours", "context-runtime"), ("ours-nocr", "none"),
                                                                         ("ours-h2", "context-runtime"), ("ours-h2-nocr", "none")]
    for arm in ("ours", "ours-nocr", "ours-h2", "ours-h2-nocr"):
        assert e.board.finished(arm_id(arm, "q8"), "a/task1")
    assert e.board.info(arm_id("ours-nocr", "q8"), "a/task1")["context"] == "none"
