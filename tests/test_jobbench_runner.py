import json
from pathlib import Path

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
    s = jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=2)
    assert attempts["n"] == 2 and s["deliverables"] == ["a.md"] and [a["attempt"] for a in s["attempts"]] == [1, 2]
    assert (tmp_path / "out" / "output" / "a.md").read_text() == "hi"
    # a first attempt that delivers is final: no second attempt
    attempts["n"] = 0
    monkeypatch.setattr(jb, "run_loop", lambda c, r, s_, t, **kw: (r.run("write_file", {"path": "/work/output/b.md", "content": "x"}),
                                                                   LoopResult(stop="done", turns=1))[1])
    s2 = jb.run_task(base, "x/task1", tmp_path / "out2", client=type("C", (), {"model": "m"})(), attempts=2)
    assert len(s2["attempts"]) == 1 and s2["deliverables"] == ["b.md"]
