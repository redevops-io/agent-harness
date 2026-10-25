import json
import time

import pytest

from agent_harness.bench import audit, experiment as ex


def cfg(tmp_path, **over):
    (tmp_path / "tasks.txt").write_text("a/task1\n")
    c = {"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"), "harness2_root": str(tmp_path / "h2"),
         "cells": {"q8": {"provider": "px48", "model": "M", "window": 65536}, "mac": {"provider": "local", "model": "L", "window": 65536}},
         "arms": ["ours-cr", "harness2"], "improver": {"provider": "local", "model": "L", "window": 262144},
         "spec": {"window": 65536, "max_output": 16384, "sampling": {"temperature": 0.55, "top_p": 1.0}, "thinking": False},
         "sampling": {"temperature": 0.55, "top_p": 1.0}}
    c.update(over)
    return c


def levels(fs):
    return [f.level for f in fs]


def test_a_matching_config_passes(tmp_path):
    fs = audit.audit_config(ex.Experiment(cfg(tmp_path)))
    assert audit.passed(fs) and "FAIL" not in levels(fs)


def test_no_declared_spec_is_a_failure(tmp_path):
    c = cfg(tmp_path)
    del c["spec"]
    assert not audit.passed(audit.audit_config(ex.Experiment(c)))


def test_our_arms_without_the_declared_sampling_fail(tmp_path):
    c = cfg(tmp_path)
    del c["sampling"]                                                 # the first run's mistake: nothing sent, server default (1.0) used
    fs = audit.audit_config(ex.Experiment(c))
    assert any(f.level == "FAIL" and "temperature" in f.message for f in fs)


def test_a_wrong_cell_window_or_output_ceiling_fails(tmp_path):
    c = cfg(tmp_path)
    c["cells"]["q8"]["window"] = 40960
    c["max_output_tokens"] = 32768
    msgs = " ".join(f.message for f in audit.audit_config(ex.Experiment(c)) if f.level == "FAIL")
    assert "window 40960" in msgs and "max_output_tokens" in msgs


def test_the_executor_limit_collision_the_first_run_had_would_now_be_caught(tmp_path, monkeypatch):
    from agent_harness.bench.harnesses import harness2 as h2
    e = ex.Experiment(cfg(tmp_path, arms=["harness2"]))
    assert audit.passed(audit.audit_config(e))                                                      # fixed adapter: editor has its own alias
    real = h2.env_for

    def collided(exp, cell, runs=None):                                                              # the old behaviour: editor overwrites executor
        env = real(exp, cell, runs)
        oc = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        prov, model = env["HARNESS2_SOLVER_MODEL"].split("/", 1)
        oc["provider"][prov]["models"][model]["limit"] = {"context": 262144, "output": 32768}
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(oc)
        return env
    monkeypatch.setattr(h2, "env_for", collided)
    assert any(f.level == "FAIL" and "262144" in f.message for f in audit.audit_config(e))


def test_generate_refuses_to_start_when_the_audit_fails_and_can_be_forced(tmp_path):
    c = cfg(tmp_path, arms=["ours-cr"])
    del c["sampling"]
    e = ex.Experiment(c)
    with pytest.raises(SystemExit, match="config audit failed"):
        e.generate()
    e.generate(skip_audit=True)                                                                       # explicit and loud; adapters then run (no tasks dir: they just skip)


def ledger(tmp_path, rows):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def row(label="ours", provider="px48", **params):
    return {"ts": time.time(), "provider": provider, "label": label, "status": 200, "params": params}


def test_wire_audit_passes_matching_requests_and_ignores_editor_and_judge(tmp_path):
    e = ex.Experiment(cfg(tmp_path))
    good = dict(temperature=0.55, top_p=1.0, max_tokens=16384, thinking=False)
    rows = [row(**good), row("oc", **good), row("editor", temperature=1.0), row("judge", temperature=1.0), row("ours", provider="elsewhere", temperature=9)]
    fs = audit.audit_wire(e, ledger(tmp_path, rows))
    assert audit.passed(fs) and fs[0].level == "OK"


def test_wire_audit_catches_a_server_default_temperature_and_an_oversized_output_request(tmp_path):
    e = ex.Experiment(cfg(tmp_path))
    rows = [row(temperature=0.55, top_p=1.0), row(top_p=1.0), row("oc", temperature=1.0, top_p=1.0, max_tokens=32768)]   # nothing sent / wrong / too large
    msg = " ".join(str(f) for f in audit.audit_wire(e, ledger(tmp_path, rows)) if f.level == "FAIL")
    assert "temperature" in msg and "output tokens" in msg
    assert not audit.passed(audit.audit_wire(e, ledger(tmp_path, rows)))


def test_wire_tolerance_allows_a_small_share_of_auxiliary_requests(tmp_path):
    e = ex.Experiment(cfg(tmp_path))
    rows = [row("oc", temperature=0.55, top_p=1.0)] * 99 + [row("oc", temperature=0.0, top_p=1.0)]
    assert not audit.passed(audit.audit_wire(e, ledger(tmp_path, rows), tolerance=0.0))
    assert audit.passed(audit.audit_wire(e, ledger(tmp_path, rows), tolerance=0.02))


def test_run_audit_fails_when_any_run_exceeded_the_window(tmp_path):
    e = ex.Experiment(cfg(tmp_path))
    e.board.mark_generated("ours-cr@q8", "a/task1", deliverables=["f"], stop="done", peak_prompt=50000)
    assert audit.passed(audit.audit_runs(e))
    e.board.mark_generated("harness2@mac", "a/task1", deliverables=["f"], stop="ok", peak_prompt=229000)         # the Mac cell's first-run bug
    fs = audit.audit_runs(e)
    assert not audit.passed(fs) and "229000" in fs[0].message


def test_opencode_arms_declare_the_temperature_capability_only_when_sampling_is_configured(tmp_path):
    from agent_harness.bench.harnesses import harness2 as h2, opencode as oc
    e = ex.Experiment(cfg(tmp_path, arms=["harness2"]))
    for env in (h2.env_for(e, e.cells[0]), oc.child_env(e, e.cells[0])):
        m = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["px48"]["models"]["M"]
        assert m["temperature"] is True                                   # without this flag OpenCode sends NO temperature
    e2 = ex.Experiment(cfg(tmp_path, arms=["harness2"], sampling=None))
    e2.cfg.pop("sampling")
    assert json.loads(h2.env_for(e2, e2.cells[0])["OPENCODE_CONFIG_CONTENT"])["provider"]["px48"]["models"]["M"]["temperature"] is False


def test_the_config_audit_fails_if_the_opencode_model_would_send_no_temperature(tmp_path, monkeypatch):
    from agent_harness.bench.harnesses import harness2 as h2
    e = ex.Experiment(cfg(tmp_path, arms=["harness2"]))
    assert audit.passed(audit.audit_config(e))
    real = h2.env_for

    def without_flag(exp, cell, runs=None):
        env = real(exp, cell, runs)
        oc = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        prov, model = env["HARNESS2_SOLVER_MODEL"].split("/", 1)
        oc["provider"][prov]["models"][model]["temperature"] = False
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(oc)
        return env
    monkeypatch.setattr(h2, "env_for", without_flag)
    assert any(f.level == "FAIL" and "temperature" in f.message for f in audit.audit_config(e))


def test_tolerated_deviations_are_reported_not_hidden(tmp_path):
    e = ex.Experiment(cfg(tmp_path))
    rows = [row("oc", temperature=0.55, top_p=1.0)] * 49 + [row("oc", top_p=1.0)]
    fs = audit.audit_wire(e, ledger(tmp_path, rows), tolerance=0.05)
    assert audit.passed(fs) and "tolerated" in fs[0].message and "'temperature': 1" in fs[0].message


def test_a_small_window_overshoot_is_a_warning_and_a_large_one_a_failure(tmp_path):
    e = ex.Experiment(cfg(tmp_path))
    e.board.mark_generated("ours-cr@q8", "a/task1", deliverables=["f"], stop="done", peak_prompt=66992)          # +2.2%: OpenCode's after-the-step check
    fs = audit.audit_runs(e)
    assert audit.passed(fs) and fs[0].level == "WARN" and "66992" in fs[0].message
    e.board.mark_generated("harness2@mac", "a/task1", deliverables=["f"], stop="ok", peak_prompt=229000)
    fs = audit.audit_runs(e)
    assert not audit.passed(fs) and any(f.level == "FAIL" and "229000" in f.message for f in fs)
    strict = ex.Experiment(cfg(tmp_path, root=str(tmp_path / "strict")))
    strict.cfg["spec"]["window_tolerance"] = 0.0
    strict.board.mark_generated("ours-cr@q8", "a/task1", deliverables=["f"], stop="done", peak_prompt=66992)
    assert not audit.passed(audit.audit_runs(strict))                                                        # with no tolerance the same run fails


def test_the_matrix_audit_catches_missing_cells_pre_repair_failures_and_mismatched_composed_harnesses(tmp_path):
    import os
    from agent_harness.bench.harnesses import harness2 as h2, ours_h2
    from tests.test_harness_ours_h2 import put, trees
    c = cfg(tmp_path, arms=["h2-base", "harness2", "ours-h2-cr"], cells={"q8": {"provider": "px48", "model": "M", "window": 65536}})
    e = ex.Experiment(c)
    cell = e.cells[0]
    msgs = lambda fs: " | ".join(str(f) for f in fs)
    assert "no finished rollout" in msgs(audit.audit_matrix(e))                                         # nothing finished yet
    base, final = trees()
    put(e, cell, "a/task1", base, final)                                                                  # a composed harness on disk
    _, _, changed = ours_h2.delta(ours_h2.read_tree(ours_h2.tree_dir(e, cell, "a/task1", "base")), ours_h2.read_tree(ours_h2.tree_dir(e, cell, "a/task1", "agg1")))
    for arm in e.arms:
        e.board.mark_generated(f"{arm}@q8", "a/task1", deliverables=["f"], stop="ok", harness_changed_files=changed if arm == "ours-h2-cr" else None)
    assert audit.passed(audit.audit_matrix(e)) and audit.audit_matrix(e)[0].level == "OK"
    e.board.mark_generated("ours-h2-cr@q8", "a/task1", deliverables=["f"], stop="done", harness_changed_files=["guardrails.md"])   # ran on a different tree
    assert "differs from the one on disk" in msgs(audit.audit_matrix(e))
    e.board.mark_generated("ours-h2-cr@q8", "a/task1", deliverables=["f"], stop="done", harness_changed_files=changed)
    summ = h2.runs_dir(e, cell) / "jb" / "adapt" / h2.NS / "a__task1" / "summary.json"                   # a partial result with an old summary
    summ.parent.mkdir(parents=True, exist_ok=True)
    summ.write_text("{}")
    os.utime(summ, (1000, 1000))
    e.board.mark_generated("harness2@q8", "a/task1", deliverables=[], stop="partial")
    assert "predates the infrastructure repair" in msgs(audit.audit_matrix(e, stable_since=2000))
    assert audit.passed(audit.audit_matrix(e, stable_since=2000, accepted=["harness2@q8|a/task1"]))        # a looked-at genuine outcome can be accepted explicitly
    os.utime(summ, (5000, 5000))
    assert audit.passed(audit.audit_matrix(e, stable_since=2000))                                          # produced after the repair: a genuine outcome


def test_the_matrix_audit_uses_a_content_hash_when_recorded_and_timestamps_otherwise(tmp_path):
    import os
    from agent_harness.bench.harnesses import ours_h2
    from tests.test_harness_ours_h2 import put, trees
    c = cfg(tmp_path, arms=["ours-h2-cr"], cells={"q8": {"provider": "px48", "model": "M", "window": 65536}})
    e = ex.Experiment(c)
    cell = e.cells[0]
    base, final = trees()
    put(e, cell, "a/task1", base, final)
    d = ours_h2.tree_dir(e, cell, "a/task1", "agg1")
    ft = ours_h2.read_tree(d)
    _, _, changed = ours_h2.delta(ours_h2.read_tree(ours_h2.tree_dir(e, cell, "a/task1", "base")), ft)
    out = tmp_path / "out"
    out.mkdir()
    # content hash recorded and equal -> fine; the same file names with different content -> caught
    e.board.mark_generated("ours-h2-cr@q8", "a/task1", deliverables=["f"], stop="done", dir=str(out), harness_changed_files=changed,
                           harness_tree_hash=ours_h2.tree_hash(ft, changed))
    assert audit.passed(audit.audit_matrix(e))
    (d / "guardrails.md").write_text(final["guardrails.md"] + "\nAn extra rule added when the pipeline was re-run.\n")
    assert "different content" in " ".join(str(f) for f in audit.audit_matrix(e))
    # no hash recorded (older runs): the run output must not be older than the tree
    e.board.mark_generated("ours-h2-cr@q8", "a/task1", deliverables=["f"], stop="done", dir=str(out), harness_changed_files=changed, harness_tree_hash=None)
    os.utime(out, (1000, 1000))
    assert "ran before the composed harness" in " ".join(str(f) for f in audit.audit_matrix(e))
    os.utime(out, (os.path.getmtime(d) + 100,) * 2)
    assert audit.passed(audit.audit_matrix(e))
