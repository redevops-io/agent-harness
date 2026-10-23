import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_harness.bench import judge_batch_script as jb
from agent_harness.bench.evalkit import JudgeConfig, judge_batch

RUBRICS = [{"rubric": "R0 text", "weight": 6, "criterion": ["c0", "c1"]}, {"rubric": "R1 text", "weight": 10, "criterion": ["d0"]}]


def fake_upstream():
    def failed(i, rubric, overall, criteria_reasoning=None):
        crit = rubric["criterion"]
        return {"index": i, "rubric": rubric["rubric"], "weight": rubric["weight"],
                "result": {"passed": False, "score": 0, "criteria_count": len(crit), "criteria_passed": 0, "overall_reasoning": overall,
                           "criteria_results": [{"index": j, "criterion": c, "passed": False, "reasoning": criteria_reasoning or overall, "evidence": ""}
                                                for j, c in enumerate(crit)]}}

    def report(name, model, results, total=None):
        return {"evaluated_model": name, "judge_model": model, "total_score": sum(r["result"]["score"] for r in results),
                "max_score": sum(r["weight"] for r in results), "passed_count": sum(1 for r in results if r["result"]["passed"]),
                "total_count": total or len(results), "rubrics": results}
    return SimpleNamespace(normalize_criteria=lambda r: list(r["criterion"]), parse_judge_json=lambda t: (json.loads(t), "direct"),
                           build_failed_rubric_result=failed, build_details_report=report,
                           extract_all_file_contents=lambda d: "".join(f"=== FILE: {p.name} ===\n{p.read_text()}\n" for p in sorted(Path(d).glob("*"))))


def cands(tmp_path, keys=("ours-cr", "ours-h2-cr", "harness2", "h2-base")):
    out = {}
    for k in keys:
        d = tmp_path / k
        d.mkdir()
        (d / "answer.md").write_text(f"answer of {k}")
        out[k] = d
    return out


def test_labels_are_a_deterministic_shuffle_independent_of_input_order():
    a = jb.assign_labels(["a", "b", "c", "d"], "seed1")
    assert sorted(a.values()) == ["A", "B", "C", "D"] and a == jb.assign_labels(["d", "c", "b", "a"], "seed1")
    assert len({tuple(sorted(jb.assign_labels(list("abcd"), f"s{i}").items())) for i in range(30)}) > 5          # different seeds really shuffle


def test_the_prompt_carries_every_candidate_and_criterion_but_no_arm_names():
    sections = [("A", "=== FILE: x ===\ncontent of A"), ("B", "=== FILE: x ===\ncontent of B")]
    p = jb.build_prompt("R text", ["c0", "c1"], sections)
    assert "content of A" in p and "content of B" in p and "Criterion 0: c0" in p and "Criterion 1: c1" in p
    assert '"candidates"' in p and "INDEPENDENTLY" in p and "exactly 2 items" in p
    assert "ours" not in p and "harness2" not in p


def reply_for(labels_to_verdicts):
    """labels_to_verdicts: letter -> list of passed booleans per criterion of the group being asked about (by criterion count)."""
    def call(prompt, timeout):
        n = prompt.count("\nCriterion ")
        cands_ = {}
        for letter in "ABCD":
            if f"CANDIDATE {letter}:" in prompt and letter in labels_to_verdicts:
                v = labels_to_verdicts[letter][:n]
                cands_[letter] = {"criteria_results": [{"index": i, "passed": p, "reasoning": f"r{letter}{i}", "evidence": "e"} for i, p in enumerate(v)],
                                  "rubric_passed": all(v), "overall_reasoning": f"o{letter}"}
        return json.dumps({"candidates": cands_})
    return call


def test_verdicts_are_mapped_back_to_the_right_candidate(tmp_path):
    c = cands(tmp_path)
    up = fake_upstream()
    labels = jb.assign_labels(list(c), "s")
    want = {"ours-cr": [True, True], "ours-h2-cr": [True, False], "harness2": [False, False], "h2-base": [True, True]}
    call = reply_for({labels[k]: v for k, v in want.items()})
    reports, got_labels = jb.run_batch(up, RUBRICS, c, "J", call, "s", max_workers=2)
    assert got_labels == labels
    for k, v in want.items():
        r0 = reports[k]["rubrics"][0]["result"]
        assert [x["passed"] for x in r0["criteria_results"]] == v and r0["passed"] == all(v) and r0["score"] == (6 if all(v) else 0)
        assert reports[k]["evaluated_model"] == k and reports[k]["protocol"]["label"] == labels[k]
    assert reports["ours-cr"]["total_score"] == 6 + 10 and reports["harness2"]["total_score"] == 0


def test_rubric_pass_is_recomputed_from_the_criteria_not_taken_from_the_model(tmp_path):
    c = cands(tmp_path, ("x", "y"))
    labels = jb.assign_labels(list(c), "s")

    def liar(prompt, timeout):
        n = prompt.count("\nCriterion ")
        return json.dumps({"candidates": {l: {"criteria_results": [{"index": i, "passed": i == 0, "reasoning": "r"} for i in range(n)],
                                              "rubric_passed": True, "overall_reasoning": "o"} for l in labels.values()}})
    reports, _ = jb.run_batch(fake_upstream(), RUBRICS, c, "J", liar, "s")
    r = reports["x"]["rubrics"][0]["result"]                  # criterion 1 failed although the model claimed the rubric passed
    assert r["passed"] is False and r["score"] == 0 and r["model_rubric_passed"] is True


def test_a_missing_candidate_section_or_a_garbage_reply_becomes_a_failure_result_the_downstream_code_recognises(tmp_path):
    c = cands(tmp_path, ("x", "y"))
    labels = jb.assign_labels(list(c), "s")
    only_x = reply_for({labels["x"]: [True, True]})
    reports, _ = jb.run_batch(fake_upstream(), RUBRICS, c, "J", only_x, "s")
    fy = reports["y"]["rubrics"][0]["result"]["criteria_results"]
    assert all(r["reasoning"].startswith(jb.FAIL_PREFIX) for r in fy) and reports["x"]["rubrics"][0]["result"]["passed"]
    reports, _ = jb.run_batch(fake_upstream(), RUBRICS, c, "J", lambda p, t: "not json at all", "s", max_retries=2)
    assert all(r["reasoning"].startswith(jb.FAIL_PREFIX) for k in reports for ru in reports[k]["rubrics"] for r in ru["result"]["criteria_results"])


def test_a_model_call_that_raises_is_retried_then_recorded_not_raised(tmp_path):
    c = cands(tmp_path, ("x", "y"))
    labels = jb.assign_labels(list(c), "s")
    n = {"calls": 0}

    def flaky(prompt, timeout):
        n["calls"] += 1
        if n["calls"] == 1:
            raise TimeoutError("slow")
        return reply_for({labels["x"]: [True, True], labels["y"]: [True, True]})(prompt, timeout)
    reports, _ = jb.run_batch(fake_upstream(), [RUBRICS[1]], c, "J", flaky, "s", max_workers=1, max_retries=2)
    assert reports["x"]["rubrics"][0]["result"]["passed"] and n["calls"] == 2


UPSTREAM = os.environ.get("JOBBENCH_JUDGE_PY")


@pytest.mark.skipif(not UPSTREAM or not Path(UPSTREAM).is_file(), reason="set JOBBENCH_JUDGE_PY to the benchmark's eval/judge.py")
def test_the_real_upstream_helpers_accept_and_produce_the_same_shapes(tmp_path):
    up = jb.load_upstream(UPSTREAM)
    c = cands(tmp_path, ("x", "y"))
    labels = jb.assign_labels(list(c), "s")
    reports, _ = jb.run_batch(up, RUBRICS, c, "J", reply_for({labels["x"]: [True, True], labels["y"]: [False, True]}), "s")
    d = reports["x"]
    assert {"evaluated_model", "judge_model", "total_score", "max_score", "pass_rate", "passed_count", "total_count", "rubrics"} <= set(d)
    assert d["total_score"] == 16 and d["max_score"] == 16 and d["rubrics"][0]["result"]["criteria_results"][0]["criterion"] == "c0"
    assert reports["y"]["total_score"] == 0 and reports["y"]["rubrics"][0]["result"]["criteria_passed"] == 1     # group score 0 (all-or-nothing) but one criterion passed


# -- evalkit.judge_batch ------------------------------------------------------------------------------------------------
FAKE_BATCH = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
def opt(name): return a[a.index(name) + 1]
cands = {s.split("=", 1)[0]: s.split("=", 1)[1] for i, s in enumerate(a) if a[i - 1] == "--candidate"}
mode = Path(opt("--details-dir"), "mode.txt").read_text().strip()
calls = Path(opt("--details-dir"), "calls.log"); calls.write_text(calls.read_text() + ",".join(sorted(cands)) + "\n" if calls.exists() else ",".join(sorted(cands)) + "\n")
if mode == "crash": sys.exit(3)
for k in cands:
    msg = None
    if mode == "quota": msg = "Failed to get judge response: Error code: 429 - insufficient balance, account is suspended"
    if mode == "y_fails_once" and k == "y" and calls.read_text().count("\n") == 1: msg = "Failed to get judge response: timeout"
    crit = [{"index": 0, "criterion": "c", "passed": msg is None, "reasoning": msg or "ok", "evidence": ""}]
    res = {"passed": msg is None, "score": 10 if msg is None else 0, "criteria_count": 1, "criteria_passed": int(msg is None), "criteria_results": crit, "overall_reasoning": ""}
    d = {"total_score": res["score"], "max_score": 10, "passed_count": int(msg is None), "total_count": 1, "rubrics": [{"index": 0, "rubric": "r", "weight": 10, "result": res}]}
    Path(opt("--details-dir"), k + ".json").write_text(json.dumps(d))
'''


@pytest.fixture
def brig(tmp_path):
    script = tmp_path / "fake_batch.py"
    script.write_text(FAKE_BATCH)
    base = tmp_path / "base" / "t" / "task1"
    base.mkdir(parents=True)
    (base / "RUBRICS.json").write_text("{}")
    cfg = JudgeConfig(sys.executable, "unused-upstream.py", "m", "http://x", attempts=3, backoff_s=0, timeout_s=20, batch_script=str(script))
    cs = cands(tmp_path / "..", ()) if False else {}
    for k in ("x", "y", "empty"):
        d = tmp_path / "out" / k
        d.mkdir(parents=True)
        if k != "empty":
            (d / "a.md").write_text("hi")
        cs[k] = d
    dd = tmp_path / "details"
    dd.mkdir()

    def run(mode):
        (dd / "mode.txt").write_text(mode)
        (dd / "calls.log").unlink(missing_ok=True)
        r = judge_batch("t/task1", tmp_path / "base", cs, dd, cfg, seed="s", sleep=lambda s: None)
        return r, (dd / "calls.log").read_text().splitlines() if (dd / "calls.log").exists() else []
    return run


def test_candidates_without_deliverables_are_no_output_and_never_sent(brig):
    r, calls = brig("ok")
    assert r["empty"].status == "no_output" and r["x"].status == "graded" and r["y"].score == 1.0
    assert calls == ["x,y"]                                                         # one shared pass, the empty one excluded


def test_only_the_incomplete_candidates_are_retried(brig):
    r, calls = brig("y_fails_once")
    assert r["x"].status == r["y"].status == "graded"
    assert calls == ["x,y", "y"]


def test_a_provider_quota_error_stops_everything_at_once(brig):
    r, calls = brig("quota")
    assert r["x"].budget_exhausted and r["y"].budget_exhausted and calls == ["x,y"]


def test_a_crashing_script_leaves_the_candidates_ungraded_after_the_attempts(brig):
    r, calls = brig("crash")
    assert r["x"].status == "ungraded" and "judge process failed" in r["x"].reason or "no readable details" in r["x"].reason
    assert len(calls) == 3
