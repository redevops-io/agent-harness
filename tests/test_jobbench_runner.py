import json
from pathlib import Path

import pytest

import agent_harness.bench.jobbench as jb
from agent_harness.loop import LoopResult


class FakeSandbox:
    def __init__(self, ws, *a, **k): self.ws = ws
    def __enter__(self): return self
    def __exit__(self, *e): pass


def make_task(tmp_path):
    t = tmp_path / "base" / "x" / "task1" / "task_folder"
    t.mkdir(parents=True)
    (t / "TASK_INSTRUCTIONS.txt").write_text("do it")
    return tmp_path / "base"


def test_second_attempt_only_when_first_has_no_deliverables(tmp_path, monkeypatch):
    base = make_task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", FakeSandbox)
    attempts = {"n": 0}

    def fake_run_loop(client, reg, system, task, **kw):
        attempts["n"] += 1
        if attempts["n"] == 2:   # only the second attempt writes a deliverable
            reg.run("write_file", {"path": "/work/output/a.md", "content": "hi"})
        return LoopResult(stop="done", turns=1, seconds=1.0)

    monkeypatch.setattr(jb, "run_loop", fake_run_loop)
    s = jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=2, context="basic")
    assert attempts["n"] == 2 and s["deliverables"] == ["a.md"] and [a["attempt"] for a in s["attempts"]] == [1, 2]
    assert (tmp_path / "out" / "output" / "a.md").read_text() == "hi"
    # a first attempt that delivers is final: no second attempt
    attempts["n"] = 0
    monkeypatch.setattr(jb, "run_loop", lambda c, r, s_, t, **kw: (r.run("write_file", {"path": "/work/output/b.md", "content": "x"}),
                                                                   LoopResult(stop="done", turns=1))[1])
    s2 = jb.run_task(base, "x/task1", tmp_path / "out2", client=type("C", (), {"model": "m"})(), attempts=2, context="basic")
    assert len(s2["attempts"]) == 1 and s2["deliverables"] == ["b.md"]


def test_context_runtime_is_the_default_and_a_missing_package_is_an_error_not_a_silent_downgrade(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_cr(name, *a, **k):
        if name.startswith("context_runtime"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cr)
    with pytest.raises(RuntimeError, match="context-runtime"):
        jb.make_context_manager("context-runtime", tmp_path)
    assert jb.make_context_manager("basic", tmp_path) is None
    with pytest.raises(ValueError):
        jb.make_context_manager("nope", tmp_path)


def test_runner_passes_a_context_runtime_manager_into_the_loop_and_reports_archive_stats(tmp_path, monkeypatch):
    pytest.importorskip("context_runtime")
    base = make_task(tmp_path)
    (base / "x" / "task1" / "task_folder" / "notes.txt").write_text("alpha beta gamma")
    monkeypatch.setattr(jb, "DockerSandbox", FakeSandbox)
    got = {}

    def fake_run_loop(client, reg, system, task, **kw):
        got["mgr"] = kw["context_manager"]
        got["mgr"].admit("bash", "z" * 5000, 100)            # something to archive
        reg.run("write_file", {"path": "/work/output/a.md", "content": "hi"})
        return LoopResult(stop="done", turns=1, seconds=1.0)

    monkeypatch.setattr(jb, "run_loop", fake_run_loop)
    s = jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})())
    assert s["context_manager"] == "context-runtime" and s["archived_items"] == 3        # two input files + the oversized output
    assert type(got["mgr"]).__name__ == "ContextRuntimeManager"


def test_grader_artifacts_in_the_workspace_abort_the_run(tmp_path, monkeypatch):
    base = make_task(tmp_path)
    (base / "x" / "task1" / "task_folder" / "RUBRICS.json").write_text("{}")
    monkeypatch.setattr(jb, "DockerSandbox", FakeSandbox)
    monkeypatch.setattr(jb, "run_loop", lambda *a, **k: pytest.fail("the agent must never start"))
    with pytest.raises(RuntimeError, match="grader artifacts"):
        jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), context="basic")


def test_guard_flags_grader_names_but_not_ordinary_inputs(tmp_path):
    (tmp_path / "scoring_rules.xlsx").write_text("x")           # an ordinary task input that merely mentions scoring
    (tmp_path / "answer_key_template.docx").write_text("x")     # likewise: not one of JobBench's grader file names
    jb.assert_workspace_clean(tmp_path)
    for name in ("RUBRICS.json", "rubric.json", "eval_result", "gpt-5-4_judge.json"):
        (tmp_path / name).write_text("x")
        with pytest.raises(RuntimeError):
            jb.assert_workspace_clean(tmp_path)
        (tmp_path / name).unlink()
    (tmp_path / "rubric_notes.md").write_text("x")             # \b stops at "_": only the exact grader file names are flagged
    jb.assert_workspace_clean(tmp_path)
