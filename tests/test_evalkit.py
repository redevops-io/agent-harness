import json
import sys
from pathlib import Path

import pytest

from agent_harness.bench.evalkit import (JudgeConfig, NotReady, Scoreboard, JudgeResult, judge_output, spearman)

# A stand-in for JobBench's eval/judge.py. Like the real one it REUSES an existing details file instead of redoing work.
FAKE = r'''
import json, os, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
details = args["--details-file"]
state = details + ".calls"
n = int(open(state).read()) + 1 if os.path.exists(state) else 1
open(state, "w").write(str(n))
mode = os.environ["FAKE_MODE"]
if os.path.exists(details):
    sys.exit(0)                                    # real judge: "already judged", leaves the (maybe failed) file as is
if mode == "crash":
    sys.exit(3)
if mode == "slow":
    time.sleep(30)
fail = mode == "fail" or (mode.startswith("flaky") and n < int(mode[5:]))
crit = [{"reasoning": "Failed to get judge response: Error code: 429" if fail else "criterion met"}]
json.dump({"total_score": 30, "max_score": 60, "passed_count": 3, "total_count": 6,
           "rubrics": [{"result": {"criteria_results": crit}}]}, open(details, "w"))
'''


@pytest.fixture
def rig(tmp_path, monkeypatch):
    script = tmp_path / "fake_judge.py"
    script.write_text(FAKE)
    base = tmp_path / "base" / "x" / "task1"
    base.mkdir(parents=True)
    (base / "RUBRICS.json").write_text("{}")
    deliver = tmp_path / "out"
    deliver.mkdir()
    (deliver / "a.md").write_text("hello")
    cfg = JudgeConfig(sys.executable, str(script), "m", "http://x", attempts=3, backoff_s=0, timeout_s=5)
    return cfg, tmp_path / "base", deliver, tmp_path / "details.json"


def run(rig, mode, **kw):
    cfg, base, deliver, details = rig
    monkeypatch_env(mode)
    return judge_output("x/task1", base, deliver, details, cfg, sleep=lambda s: None, **kw)


def monkeypatch_env(mode):
    import os
    os.environ["FAKE_MODE"] = mode


def test_graded_score_is_total_over_max(rig):
    r = run(rig, "ok")
    assert (r.status, r.score, r.total, r.max, r.attempts) == ("graded", 0.5, 30.0, 60.0, 1)


def test_unfinished_rollout_is_refused_not_scored(rig):
    with pytest.raises(NotReady):
        run(rig, "ok", finished=False)


def test_finished_rollout_with_no_deliverables_is_no_output_zero_and_the_judge_is_not_called(rig):
    cfg, base, deliver, details = rig
    (deliver / "a.md").unlink()
    r = run(rig, "ok")
    assert (r.status, r.score) == ("no_output", 0.0) and not Path(str(details) + ".calls").exists()
    missing = judge_output("x/task1", base, deliver / "nope", details, cfg)       # a directory that does not exist
    assert missing.status == "no_output"                                         # ...but only because the caller said finished=True


def test_infrastructure_failure_is_ungraded_never_zero(rig):
    r = run(rig, "fail")
    assert r.status == "ungraded" and r.score is None and r.failures == 1 and r.attempts == 3


def test_flaky_judge_is_retried_and_a_stale_failed_details_file_is_not_reused(rig):
    cfg, base, deliver, details = rig
    details.write_text(json.dumps({"total_score": 0, "max_score": 60, "rubrics": [
        {"result": {"criteria_results": [{"reasoning": "Failed to get judge response: 429"}]}}]}))   # leftover from an earlier attempt
    r = run(rig, "flaky2")
    assert r.status == "graded" and r.attempts == 2 and r.score == 0.5


@pytest.mark.parametrize("mode", ["crash", "slow"])
def test_crashing_or_hanging_judge_is_ungraded(rig, mode):
    cfg, base, deliver, details = rig
    cfg.attempts, cfg.timeout_s = 1, 2
    assert run((cfg, base, deliver, details), mode).status == "ungraded"


def test_missing_rubrics_is_ungraded(rig):
    cfg, base, deliver, details = rig
    (base / "x" / "task1" / "RUBRICS.json").unlink()
    assert run(rig, "ok").status == "ungraded"


def test_scoreboard_ungraded_is_not_done_not_zero_and_unfinished_cannot_be_judged(tmp_path):
    sb = Scoreboard(tmp_path / "s.jsonl")
    sb.mark_generated("A", "t1", deliverables=["a"], stop="done")
    sb.mark_generated("A", "t2", deliverables=["a"], stop="done")
    sb.mark_generated("A", "t3", deliverables=[], stop="max_turns", finished=False)
    sb.record_judgement("A", "t1", JudgeResult("graded", 0.4), "j")
    sb.record_judgement("A", "t2", JudgeResult("ungraded", failures=2, reason="429"), "j")
    assert sb.score("A", "t1") == 0.4 and sb.score("A", "t2") is None and not sb.graded("A", "t2")
    assert sb.pending("A", ["t1", "t2", "t3", "t4"]) == ["t2"]                  # t3 unfinished and t4 unknown are NOT pending
    with pytest.raises(NotReady):
        sb.record_judgement("A", "t3", JudgeResult("graded", 0.9), "j")
    sb2 = Scoreboard(tmp_path / "s.jsonl")                                      # survives a restart
    assert sb2.score("A", "t1") == 0.4 and sb2.pending("A", ["t2"]) == ["t2"]


def test_paired_statistics_drop_ungraded_and_count_no_output_as_zero(tmp_path):
    sb = Scoreboard(tmp_path / "s.jsonl")
    for arm in "AB":
        for t in ("t1", "t2", "t3", "t4"):
            sb.mark_generated(arm, t, deliverables=["f"], stop="done")
    for t, (a, b) in {"t1": (0.5, 0.3), "t2": (0.2, 0.2), "t3": (0.9, 0.1)}.items():
        sb.record_judgement("A", t, JudgeResult("graded", a), "j")
        sb.record_judgement("B", t, JudgeResult("graded", b), "j")
    sb.record_judgement("A", "t4", JudgeResult("no_output", 0.0), "j")
    sb.record_judgement("B", "t4", JudgeResult("ungraded", reason="429"), "j")       # B's t4 failed to judge: dropped, not 0
    p = sb.paired("A", "B", ["t1", "t2", "t3", "t4"])
    assert p["n"] == 3 and p["dropped_ungraded_or_missing"] == 1
    assert p["mean_delta"] == pytest.approx((0.2 + 0 + 0.8) / 3) and (p["wins"], p["ties"], p["losses"]) == (2, 1, 0)


def test_sign_test_matches_a_hand_computation(tmp_path):
    sb = Scoreboard(tmp_path / "s.jsonl")
    for i in range(10):
        for arm, v in (("A", 0.5 if i < 9 else 0.1), ("B", 0.3)):
            sb.mark_generated(arm, f"t{i}", deliverables=["f"], stop="done")
            sb.record_judgement(arm, f"t{i}", JudgeResult("graded", v), "j")
    p = sb.paired("A", "B", [f"t{i}" for i in range(10)])
    assert (p["wins"], p["losses"]) == (9, 1) and p["sign_test_p"] == pytest.approx(22 / 1024)


def test_spearman_handles_ties_and_ordering():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
