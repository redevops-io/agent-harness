import json
from pathlib import Path

import pytest

from agent_harness.bench import experiment as ex, harnesses
from agent_harness.bench.cells import Cell, arm_id
from agent_harness.bench.evalkit import JudgeResult
from agent_harness.bench.harnesses import _common, ours

TASKS = ["a/task1", "b/task1", "c/task1"]


def make_cfg(tmp_path, tasks=TASKS, **extra):
    (tmp_path / "tasks.txt").write_text("\n".join(tasks) + "\n")
    cfg = {"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"),
           "cells": {"q8": {"provider": "px48", "model": "M", "window": 65536},
                     "fp4": {"provider": "px32", "model": "M", "window": 65536},
                     "q8-think": {"provider": "px48", "model": "M", "window": 65536, "think": True}},
           "arms": ["ours"]}
    cfg.update(extra)
    return cfg


def test_thinking_cells_use_the_reserved_proxy_label_and_nothing_else_differs():
    plain, think = Cell("q8", "px48", "M", 65536), Cell("q8-think", "px48", "M", 65536, True)
    assert plain.label("ours") == "ours" and plain.label("oc") == "oc" and think.label("ours") == think.label("oc") == "think"
    assert (plain.provider, plain.model, plain.window) == (think.provider, think.model, think.window)


def test_unknown_and_missing_adapters_are_clear_errors():
    with pytest.raises(ValueError, match="unknown harness"):
        harnesses.load("nope")
    assert harnesses.load("ours").NAME == "ours"


def test_opencode_env_routes_everything_through_the_proxy_and_marks_thinking_cells():
    env = _common.opencode_env("http://127.0.0.1:8765", {"px48": {"model": "M", "context": 65536, "output": 16384},
                                                           "local": {"model": "I", "context": 262144, "output": 32768}},
                               "http://127.0.0.1:8766", labels={"px48": "think"})
    prov = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]
    assert prov["px48"]["options"]["baseURL"] == "http://127.0.0.1:8765/px48/think/v1" and prov["px48"]["models"]["M"]["reasoning"] is True
    assert prov["local"]["options"]["baseURL"] == "http://127.0.0.1:8765/local/oc/v1" and prov["local"]["models"]["I"]["reasoning"] is False
    assert prov["px48"]["models"]["M"]["limit"] == {"context": 65536, "output": 16384}
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:8766" and "127.0.0.1" in env["NO_PROXY"] and env["OPENAI_API_KEY"] == "dummy"


def test_opencode_peak_prompt_counts_cache_tokens(tmp_path):
    rows = [json.dumps({"type": "step_finish", "part": {"tokens": {"input": i, "cache": {"read": 10, "write": 5}}}}) for i in (1000, 5000, 2000)]
    (tmp_path / "t.jsonl").write_text("\n".join("noise " + r for r in rows) + "\nnot json\n")
    assert _common.opencode_diagnostics(tmp_path) == {"peak_prompt": 5015, "model_turns": 3}
    assert _common.opencode_diagnostics(tmp_path / "missing") == {"peak_prompt": 0, "model_turns": 0}


def test_ours_diagnostics_from_trajectory(tmp_path):
    rows = [{"kind": "assistant", "usage": {"prompt_tokens": 100}}, {"kind": "compact"}, {"kind": "tool"},
            {"kind": "assistant", "usage": {"prompt_tokens": 900}}, {"kind": "end"}]
    (tmp_path / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert ours.diagnostics(tmp_path) == {"peak_prompt": 900, "compactions": 1, "model_turns": 2}


def test_ours_generation_records_runs_a_crash_is_not_finished_and_resume_skips_finished(tmp_path):
    e = ex.Experiment(make_cfg(tmp_path))
    q8 = e.cells[0]
    calls = []

    def fake_run_task(tasks_base, task, out, client, **kw):
        calls.append(task)
        if task == "b/task1":
            raise RuntimeError("docker exploded")
        out.mkdir(parents=True, exist_ok=True)
        (out / "output").mkdir(exist_ok=True)
        (out / "output" / "r.md").write_text("x")
        (out / "trajectory.jsonl").write_text(json.dumps({"kind": "assistant", "usage": {"prompt_tokens": 321}}) + "\n")
        assert kw["context"] == "context-runtime" and kw["max_context_tokens"] == 65536
        return {"deliverables": ["r.md"], "stop": "done", "archived_items": 2, "retrieval_calls": 1, "seconds": 5}

    ours.generate(e, q8, workers=2, run_task=fake_run_task, client_factory=lambda: object())
    a = arm_id("ours", "q8")
    assert e.board.finished(a, "a/task1") and e.board.info(a, "a/task1")["peak_prompt"] == 321
    assert not e.board.finished(a, "b/task1") and "docker exploded" in e.board.info(a, "b/task1")["error"]   # crash: not finished, not 0
    calls.clear()
    ours.generate(e, q8, workers=1, run_task=fake_run_task, client_factory=lambda: object())
    assert calls == ["b/task1"]                                                                              # only the crashed task is retried


def test_ours_client_uses_the_thinking_label_and_a_common_output_ceiling(tmp_path):
    e = ex.Experiment(make_cfg(tmp_path, tasks=["a/task1"]))
    seen = {}

    def fake_run_task(tasks_base, task, out, client, **kw):
        seen[task] = (client.base_url, client.max_tokens)
        out.mkdir(parents=True, exist_ok=True)
        return {"deliverables": [], "stop": "done"}
    for cell in (e.cells[0], e.cells[2]):
        e.board = ex.Scoreboard(tmp_path / f"s-{cell.name}.jsonl")
        ours.generate(e, cell, run_task=fake_run_task)
        seen[cell.name] = seen.pop("a/task1")
    assert seen["q8"] == ("http://127.0.0.1:8765/px48/ours/v1", 16384)
    assert seen["q8-think"] == ("http://127.0.0.1:8765/px48/think/v1", 16384)


def test_judging_is_refused_without_a_matching_calibration(tmp_path):
    rep = tmp_path / "r.json"
    assert not ex.calibration_ok(rep, "J", "high")
    rep.write_text(json.dumps({"judge_model": "J", "judge_reasoning_effort": "high", "checks": [{"name": "judge calibration", "status": "FAIL"}]}))
    assert not ex.calibration_ok(rep, "J", "high")
    rep.write_text(json.dumps({"judge_model": "J", "judge_reasoning_effort": "high", "checks": [{"name": "judge calibration", "status": "PASS"}]}))
    assert ex.calibration_ok(rep, "J", "high")
    assert not ex.calibration_ok(rep, "J", "default") and not ex.calibration_ok(rep, "OTHER", "high")


def test_main_judge_step_refuses_an_uncalibrated_judge_before_judging_anything(tmp_path, capsys):
    cfg = make_cfg(tmp_path, judge={"python": "x", "script": "y", "model": "J", "base_url": "u", "reasoning_effort": "high"},
                   selfcheck_report=str(tmp_path / "missing.json"))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    assert ex.main(["judge", "--config", str(tmp_path / "c.json")]) == 2
    assert "REFUSING" in capsys.readouterr().out


def _fill(e, vals, shift_by_cell, arms, peaks=None):
    for cell, shift in shift_by_cell.items():
        for arm in arms:
            aid = arm_id(arm, cell)
            for t, v in zip(e.tasks, vals[arm]):
                e.board.mark_generated(aid, t, deliverables=["f"], stop="done", dir="d", peak_prompt=(peaks or {}).get(arm, 1000))
                e.board.record_judgement(aid, t, JudgeResult("graded", v + shift), "J")


def test_report_uses_only_tasks_graded_in_every_arm_and_flags_context_pressure(tmp_path):
    cfg = make_cfg(tmp_path, tasks=["a/task1", "b/task1"], arms=["ours", "opencode", "harness2"])
    cfg["cells"] = {k: cfg["cells"][k] for k in ("q8", "fp4")}
    e = ex.Experiment(cfg)
    _fill(e, {"ours": (0.6, 0.5), "opencode": (0.4, 0.3), "harness2": (0.5, 0.2)}, {"q8": 0.0, "fp4": -0.1},
          ("ours", "opencode", "harness2"), peaks={"ours": 62000})
    e.board.mark_generated("ours@q8", "b/task1", deliverables=["f"], stop="done", dir="d")
    e.board.record_judgement("ours@q8", "b/task1", JudgeResult("ungraded", reason="429"), "J")   # drops b/task1 for EVERY arm
    r = e.report()
    assert r["tasks_graded_in_all_arms"] == 1 and r["table"]["ours@q8"]["n_common"] == 1
    assert r["table"]["ours@q8"]["mean_on_common_tasks"] == pytest.approx(0.6) and r["table"]["opencode@fp4"]["mean_on_common_tasks"] == pytest.approx(0.3)
    assert r["paired"]["ours - opencode @ q8"]["dropped_ungraded_or_missing"] == 1
    assert "q8 - fp4" not in r["paired"] and "ours: q8 - fp4" in r["paired"]                  # precision comparison per harness
    assert r["diagnostics"]["ours@fp4"]["runs_at_90pct_of_window"] == 2 and r["diagnostics"]["opencode@fp4"]["runs_at_90pct_of_window"] == 0
    md = e.report_markdown()
    assert "| ours | 60.0 | 50.0 |" in md and "1 graded in every arm" in md


def test_passive_arms_are_never_generated_on_their_own(tmp_path, monkeypatch):
    e = ex.Experiment(make_cfg(tmp_path, arms=["ours", "h2-base"]))
    ran = []

    class Fake:
        NAME = "ours"
        @staticmethod
        def generate(exp, cell, workers):
            ran.append(cell.name)
    monkeypatch.setattr(harnesses, "load", lambda name: Fake if name == "ours" else pytest.fail(f"loaded {name}"))
    e.generate()
    assert ran == ["q8", "fp4", "q8-think"]
