from types import SimpleNamespace

import pytest

from agent_harness.bench import experiment as ex, harnesses, jobbench as jb
from agent_harness.bench.cells import arm_id
from agent_harness.bench.harnesses import harness2 as h2, ours_h2 as oh
from agent_harness.h2 import OPENCODE, stock_spec

SKILL = "---\nname: deliverable-verify\ndescription: Check every deliverable exists and is complete before finishing.\n---\nList output/ and open each file.\n"


def trees():
    base = dict(stock_spec(OPENCODE).files)
    base["systemprompt.md"] = "STOCK-OPENCODE-WORDING use the todowrite tool.\n"
    final = dict(base)
    final["guardrails.md"] = base.get("guardrails.md", "") + "\nAlways run the deliverable-verify skill before finishing.\n"
    final["skills/deliverable-verify/SKILL.md"] = SKILL
    final["scripts/verify_outputs.py"] = "print('ok')\n"
    return base, final


def make_exp(tmp_path):
    (tmp_path / "tasks.txt").write_text("a/task1\n")
    return ex.Experiment({"root": str(tmp_path / "exp"), "tasks_file": str(tmp_path / "tasks.txt"), "tasks_base": str(tmp_path / "base"),
                          "cells": {"q8": {"provider": "px48", "model": "M", "window": 65536}}, "arms": ["ours-h2"]})


def put(exp, cell, task, base, final):
    for which, files in (("base", base), ("agg1", final)):
        for rel, text in files.items():
            p = oh.tree_dir(exp, cell, task, which) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)


def test_the_adapter_is_discoverable_under_its_hyphenated_name():
    assert harnesses.load("ours-h2") is oh


def test_only_the_difference_from_stock_is_applied():
    base, final = trees()
    text, files, changed = oh.delta(base, final)
    assert changed == ["guardrails.md", "scripts/verify_outputs.py", "skills/deliverable-verify/SKILL.md"]
    assert "Always run the deliverable-verify skill" in text and "skills/deliverable-verify/SKILL.md: Check every" in text
    assert "- scripts/verify_outputs.py" in text
    assert set(files) == {"skills/deliverable-verify/SKILL.md", "scripts/verify_outputs.py"}
    assert "STOCK-OPENCODE-WORDING" not in text                      # unchanged stock wording is not copied in


def test_an_unchanged_tree_adds_nothing_so_the_arm_equals_ours():
    base, _ = trees()
    assert oh.delta(base, dict(base)) == ("", {}, [])


def test_an_invalid_composed_tree_is_refused():
    base, final = trees()
    final["guardrails.md"] += "\nrubric: {{ answer key }}\n"
    with pytest.raises(ValueError, match="invalid"):
        oh.delta(base, final)


def test_generate_passes_the_delta_to_the_run_and_records_it(tmp_path):
    e = make_exp(tmp_path)
    put(e, e.cells[0], "a/task1", *trees())
    got = {}

    def run_task(tasks_base, task, out, client, **kw):
        got.update(kw)
        return {"deliverables": ["r.md"], "stop": "done", "archived_items": 2, "retrieval_calls": 0, "seconds": 1.0}
    oh.generate(e, e.cells[0], 1, run_task=run_task, client_factory=lambda: SimpleNamespace(model="M"))
    a = arm_id("ours-h2", "q8")
    assert "deliverable-verify" in got["extra_system"] and set(got["extra_files"]) == {"skills/deliverable-verify/SKILL.md", "scripts/verify_outputs.py"}
    assert got["context"] == "context-runtime"
    assert e.board.finished(a, "a/task1") and "guardrails.md" in e.board.info(a, "a/task1")["harness_changed_files"]


def test_without_a_composed_harness_the_task_is_skipped_not_failed(tmp_path, capsys):
    e = make_exp(tmp_path)
    oh.generate(e, e.cells[0], 1, run_task=lambda *a, **k: pytest.fail("must not run"), client_factory=lambda: None)
    assert "run the harness2 arm" in capsys.readouterr().out
    assert e.board.info(arm_id("ours-h2", "q8"), "a/task1") == {}


def test_a_crash_is_recorded_as_not_finished(tmp_path):
    e = make_exp(tmp_path)
    put(e, e.cells[0], "a/task1", *trees())

    def boom(*a, **k):
        raise RuntimeError("sandbox died")
    oh.generate(e, e.cells[0], 1, run_task=boom, client_factory=lambda: SimpleNamespace(model="M"))
    a = arm_id("ours-h2", "q8")
    assert not e.board.finished(a, "a/task1") and "sandbox died" in e.board.info(a, "a/task1")["error"]


def test_harness_files_land_at_their_tree_paths_and_cannot_go_elsewhere(tmp_path):
    jb._write_extra_files(tmp_path, {"skills/x/SKILL.md": "ok", "scripts/v.py": "print(1)"})
    assert (tmp_path / "skills/x/SKILL.md").read_text() == "ok" and (tmp_path / "scripts/v.py").is_file()
    for bad in ("../evil.py", "/etc/evil", "skills/../../evil", "output/x.md", "task_folder/a.csv", "README.md"):
        with pytest.raises(ValueError):
            jb._write_extra_files(tmp_path, {bad: "x"})


def test_upstream_scaffolding_readmes_are_not_part_of_the_tree(tmp_path):
    for rel, text in (("guardrails.md", "g"), ("skills/README.md", "scaffold"), ("scripts/README.md", "scaffold"), ("skills/a/SKILL.md", SKILL)):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    assert sorted(oh.read_tree(tmp_path)) == ["guardrails.md", "skills/a/SKILL.md"]


def test_the_trees_a_real_harness2_run_produced_load_and_diff(tmp_path):
    from pathlib import Path
    real = Path.home() / "Documents/harness2-experiment/smoke-runs/harness2/fp8-48gb/jb/adapt/jb-oc-agg1/technical_writers__task1/harnesses"
    if not (real / "agg1").is_dir():
        pytest.skip("no real Harness² run on this machine")
    text, files, changed = oh.delta(oh.read_tree(real / "base"), oh.read_tree(real / "agg1"))
    assert changed and "guardrails.md" in changed and text.startswith("## Additional guidance") and files


def test_problems_in_unchanged_stock_text_do_not_block_but_problems_in_edits_do():
    base, final = trees()
    base["systemprompt.md"] = final["systemprompt.md"] = "stock text with {{ braces }}\n"
    assert oh.delta(base, final)[2]                                              # stock braces are not our concern
    final["guardrails.md"] += "{{ edit with braces }}"
    with pytest.raises(ValueError, match="template-braces"):
        oh.delta(base, final)
