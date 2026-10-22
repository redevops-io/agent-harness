from pathlib import Path
import json

import pytest

from agent_harness.bench import experiment as ex, harnesses
from agent_harness.bench.cells import arm_id
from agent_harness.bench.harnesses import harness2 as h2

TASKS = ["a/task1", "b/task1", "c/task1"]


def make_exp(tmp_path, arms=("harness2",), think=False):
    (tmp_path / "tasks.txt").write_text("\n".join(TASKS) + "\n")
    cfg = {"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"),
           "cells": {"q8": {"provider": "px48", "model": "M", "window": 65536, "think": think},
                     "fp4": {"provider": "px32", "model": "M", "window": 65536}},
           "improver": {"provider": "local", "model": "I", "window": 262144}, "harness2_root": str(tmp_path / "h2"),
           "arms": list(arms), "egress_proxy": "http://127.0.0.1:8766"}
    return ex.Experiment(cfg)


def write_task(runs, task, *, base_files=("b.md",), chosen_files=("c.md",), chosen="agg1", status="ok", steps=(1000, 5000)):
    t = task.replace("/", "__")
    adapt = runs / "jb" / "adapt" / h2.NS / t
    adapt.mkdir(parents=True)
    (adapt / "summary.json").write_text(json.dumps({"status": status, "base": [{"label": "base1"}], "modes": {"agg": {"finals": ["agg1"]}},
                                                    "selections": {"agg": {"chosen_label": chosen}} if chosen else {}}))
    for label, files in (("base1", base_files), (chosen or "agg1", chosen_files)):
        d = runs / "jb" / "dataset" / task / "model_output" / f"{h2.NS}__{t}__{label}"
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_text("x")
        tr = runs / "jb" / "dataset" / task / "model_traj" / f"{h2.NS}__{t}__{label}"
        tr.mkdir(parents=True, exist_ok=True)
        (tr / "a.jsonl").write_text("\n".join("n " + json.dumps({"type": "step_finish", "part": {"tokens": {"input": i, "cache": {"read": 0, "write": 0}}}}) for i in steps) + "\n")


def test_the_adapter_is_discoverable_by_the_runner():
    assert harnesses.load("harness2") is h2


def test_reader_finds_base_and_final_outputs_and_skips_unfinished_tasks(tmp_path):
    write_task(tmp_path, "a/task1")
    write_task(tmp_path, "b/task1", base_files=(), status="base-failed")
    r = h2.read_cell(tmp_path, TASKS)
    assert set(r) == {"a/task1", "b/task1"}                                       # c/task1 has no summary: unfinished, absent
    assert r["a/task1"]["base_dir"].name.endswith("__base1") and r["a/task1"]["chosen_dir"].name.endswith("__agg1")
    assert r["b/task1"]["base_dir"] is None and r["b/task1"]["chosen_dir"] is not None


def test_command_is_k1_parallel_on_the_opencode_substrate_over_the_given_tasks(tmp_path):
    cmd = h2.command(tmp_path / "s.json", 3)
    assert cmd[0].endswith("harness2") and cmd[1:3] == ["--bench", "jb"] and cmd[cmd.index("--k") + 1] == "1" and cmd[cmd.index("--mode") + 1] == "parallel"
    assert cmd[cmd.index("--substrate") + 1] == "oc" and cmd[cmd.index("--split") + 1] == "test" and cmd[cmd.index("--workers") + 1] == "3"


def test_executor_varies_by_cell_but_the_editor_and_everything_else_are_identical(tmp_path):
    e = make_exp(tmp_path)
    a, b = h2.env_for(e, e.cells[0]), h2.env_for(e, e.cells[1])
    assert a["HARNESS2_SOLVER_MODEL"] == "px48/M" and b["HARNESS2_SOLVER_MODEL"] == "px32/M"
    assert a["HARNESS2_IMPROVER_MODEL"] == b["HARNESS2_IMPROVER_MODEL"] == "local/I"
    assert a["HARNESS2_RUNS_DIR"] != b["HARNESS2_RUNS_DIR"]                       # cells never share state or libraries
    pa, pb = json.loads(a["OPENCODE_CONFIG_CONTENT"])["provider"], json.loads(b["OPENCODE_CONFIG_CONTENT"])["provider"]
    assert pa["local"] == pb["local"]                                             # the editor's configuration is the same in both cells
    assert pa["local"]["options"]["baseURL"].endswith("/local/oc/v1") and pa["local"]["models"]["I"]["reasoning"] is False


def test_a_thinking_cell_switches_only_the_executor_label(tmp_path):
    e = make_exp(tmp_path, think=True)
    pa = json.loads(h2.env_for(e, e.cells[0])["OPENCODE_CONFIG_CONTENT"])["provider"]
    assert pa["px48"]["options"]["baseURL"].endswith("/px48/think/v1") and pa["local"]["options"]["baseURL"].endswith("/local/oc/v1")


def test_collect_registers_the_final_rollout_and_optionally_the_reference(tmp_path):
    e = make_exp(tmp_path, arms=("harness2", "h2-base"))
    cell = e.cells[0]
    runs = h2.runs_dir(e, cell)
    write_task(runs, "a/task1")
    write_task(runs, "b/task1", chosen_files=())                                  # the refined rollout produced nothing
    h2.collect(e, cell)
    f, g = arm_id("harness2", "q8"), arm_id("h2-base", "q8")
    assert e.board.info(f, "a/task1")["deliverables"] == ["c.md"] and e.board.info(g, "a/task1")["deliverables"] == ["b.md"]
    assert e.board.finished(f, "b/task1") and e.board.info(f, "b/task1")["deliverables"] == []        # finished, nothing delivered: judged as no output
    assert not e.board.finished(f, "c/task1")                                     # unfinished: neither judged nor 0
    assert e.board.info(f, "a/task1")["peak_prompt"] == 5000


def test_the_reference_arm_is_only_recorded_when_asked_for(tmp_path):
    e = make_exp(tmp_path, arms=("harness2",))
    write_task(h2.runs_dir(e, e.cells[0]), "a/task1")
    h2.collect(e, e.cells[0])
    assert e.board.finished(arm_id("harness2", "q8"), "a/task1") and not e.board.finished(arm_id("h2-base", "q8"), "a/task1")


def test_generate_runs_the_pipeline_in_chunks_and_registers_each_chunk_as_it_ends(tmp_path):
    e = make_exp(tmp_path)
    calls = []

    def runner(cmd, cwd, env, stdout, stderr):
        sp = Path(cmd[cmd.index("--split-file") + 1])
        chunk = json.loads(sp.read_text())["test"]
        calls.append((chunk, str(cwd), env["HARNESS2_SOLVER_MODEL"], str(sp)))
        for t in chunk[:1]:                                  # the first task of each chunk finishes; the others produce no summary
            write_task(h2.runs_dir(e, e.cells[0]), t)
        a = arm_id("harness2", "q8")
        assert all(e.board.finished(a, t) for c, *_ in calls[:-1] for t in c[:1])          # earlier chunks were registered already
    h2.generate(e, e.cells[0], 1, runner=runner)
    assert [c[0] for c in calls] == [[t] for t in TASKS]                                       # one task per chunk at workers=1
    assert calls[0][1].endswith("/h2") and calls[0][2] == "px48/M" and "q8.splits.json" in calls[0][3]
    assert e.board.finished(arm_id("harness2", "q8"), "a/task1")
    calls.clear()
    h2.generate(e, e.cells[0], 2, runner=runner)                                               # finished tasks are not re-run
    assert all("a/task1" not in c for c, *_ in calls)


def test_executable_is_found_next_to_the_interpreter_not_via_path(tmp_path, monkeypatch):
    from agent_harness.bench.harnesses import harness2 as h
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python").write_text("")
    (fake / "harness2").write_text("")
    monkeypatch.setattr(h.sys, "executable", str(fake / "python"))
    assert h.executable() == str(fake / "harness2")
    assert h.command(tmp_path / "s.json", 1)[0] == str(fake / "harness2")
    (fake / "harness2").unlink()
    assert h.executable() == "harness2"                                    # falls back to PATH only when it is not beside us
