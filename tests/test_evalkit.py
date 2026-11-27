import json
import sys
from pathlib import Path

import pytest

from agent_harness.bench.evalkit import (JudgeConfig, NotReady, Scoreboard, JudgeResult, judge_output, spearman)

# A stand-in for JobBench's eval/judge.py, like the real one RUBRIC BY RUBRIC: a rubric that already has a result in the details file is not asked again (so the harness decides
# which stored results to keep), a rubric without one is judged and its result is written. Rubric 0 is met; rubric 1 is a REAL verdict "not met" (30 of 60 in all).
FAKE = r'''
import json, os, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
details = args["--details-file"]
state = details + ".calls"
n = int(open(state).read()) + 1 if os.path.exists(state) else 1
open(state, "w").write(str(n))
mode = os.environ["FAKE_MODE"]
if mode == "crash":
    sys.exit(3)
if mode == "slow":
    time.sleep(30)
existing = {}
if os.path.exists(details):
    existing = {r["index"]: r for r in json.load(open(details)).get("rubrics", [])}
count = json.load(open(args["--rubrics-file"])).get("n", 2)
out = []
for i in range(count):
    if i in existing:
        out.append(existing[i])
        continue
    open(details + ".judged", "a").write(f"{n}:{i}\n")                 # which rubrics were sent to the judge on which pass
    lost = mode == "fail" or (mode.startswith("flaky") and n < int(mode[5:]) and i == 0) or (mode == "badjson" and n == 1 and i == 0)
    if lost:
        why = "Failed after 2 retries: Judge returned empty content (finish_reason=length)"
        res = {"passed": False, "score": 0, "criteria_count": 1, "criteria_passed": 0, "overall_reasoning": why,
               "criteria_results": [{"index": 0, "criterion": "c", "passed": False, "reasoning": "Failed to get judge response: " + why, "evidence": ""}]}
    else:
        met = i == 0
        res = {"passed": met, "score": 30 if met else 0, "criteria_count": 1, "criteria_passed": 1 if met else 0, "overall_reasoning": "the judge's own words",
               "criteria_results": [{"index": 0, "criterion": "c", "passed": met, "reasoning": "criterion met" if met else "criterion NOT met", "evidence": ""}]}
    out.append({"index": i, "rubric": "r", "weight": 30, "result": res})
json.dump({"total_score": sum(r["result"]["score"] for r in out), "max_score": 30 * count, "passed_count": sum(1 for r in out if r["result"]["passed"]), "total_count": count, "rubrics": out},
          open(details, "w"))
'''


@pytest.fixture
def rig(tmp_path, monkeypatch):
    script = tmp_path / "fake_judge.py"
    script.write_text(FAKE)
    base = tmp_path / "base" / "x" / "task1"
    base.mkdir(parents=True)
    (base / "RUBRICS.json").write_text(json.dumps({"n": 2}))
    deliver = tmp_path / "out"
    deliver.mkdir()
    (deliver / "a.md").write_text("hello")
    cfg = JudgeConfig(sys.executable, str(script), "m", "http://x", attempts=3, backoff_s=0, timeout_s=5, recovery_thinking_off=False)
    monkeypatch.setenv("AGENT_HARNESS_ENV_PASS", "FAKE_MODE")      # the judge child gets the allowlist plus this one named variable
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
    assert r.status == "ungraded" and r.score is None and r.failures == 2 and r.attempts == 3


def test_flaky_judge_is_retried_and_a_detail_file_of_another_output_is_not_reused(rig):
    cfg, base, deliver, details = rig
    details.write_text(json.dumps({"total_score": 60, "max_score": 60, "rubrics": [        # a leftover (no fingerprint): another output's, never reused, even though it looks perfect
        {"index": 0, "result": {"passed": True, "score": 60, "criteria_results": [{"reasoning": "criterion met"}]}},
        {"index": 1, "result": {"passed": True, "score": 0, "criteria_results": [{"reasoning": "criterion met"}]}}]}))
    r = run(rig, "flaky2")
    assert r.status == "graded" and r.attempts == 2 and r.score == 0.5


def judged(details):
    return Path(str(details) + ".judged").read_text().split()


def test_only_the_rubric_that_got_no_reply_is_asked_again_and_a_real_not_met_verdict_is_never_asked_again(rig):
    r = run(rig, "flaky2")
    assert r.status == "graded" and r.score == 0.5 and r.attempts == 2
    assert judged(rig[3]) == ["1:0", "1:1", "2:0"]       # pass 1 asked both; pass 2 asked ONLY rubric 0 (no reply). Rubric 1, a genuine "criterion NOT met", was kept as the judge decided
    d = json.loads(rig[3].read_text())
    assert [x["result"]["criteria_results"][0]["reasoning"] for x in d["rubrics"]] == ["criterion met", "criterion NOT met"]
    log = [json.loads(x) for x in Path(str(rig[3]) + ".recovery.jsonl").read_text().splitlines()]
    assert [(x["attempt"], x["rubric"]) for x in log] == [(2, 0)] and "finish_reason=length" in log[0]["why"]      # what was dropped is kept in the judge's words


def test_a_verdict_is_kept_only_for_the_same_files_rubrics_and_judge(rig):
    cfg, base, deliver, details = rig
    assert run(rig, "ok").status == "graded"
    assert judged(details) == ["1:0", "1:1"]
    assert run(rig, "ok").status == "graded" and judged(details) == ["1:0", "1:1"]          # same output again: every verdict kept, nothing sent to the judge
    (deliver / "a.md").write_text("changed")                                                  # another output: the stored verdicts say nothing about it
    Path(str(details) + ".judged").unlink()
    assert run(rig, "ok").status == "graded" and judged(details) == [f"{n}:{i}" for n in (3,) for i in (0, 1)]


def test_recovery_is_bounded_and_the_result_is_ungraded_never_a_zero(rig):
    r = run(rig, "fail")
    assert r.status == "ungraded" and r.score is None and r.attempts == 3
    assert judged(rig[3]) == ["1:0", "1:1", "2:0", "2:1", "3:0", "3:1"]       # every pass asks again for what got no reply, at most `attempts` passes in all


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


def test_rubric_timeout_is_passed_to_the_judge_script(rig, tmp_path):
    cfg, base, deliver, details = rig
    seen = tmp_path / "argv.txt"
    script = tmp_path / "echo_judge.py"
    script.write_text("import sys,json\nopen(%r,'w').write(' '.join(sys.argv))\n"
                      "a=dict(zip(sys.argv[1::2],sys.argv[2::2]))\n"
                      "json.dump({'total_score':1,'max_score':2,'rubrics':[]},open(a['--details-file'],'w'))\n" % str(seen))
    cfg2 = JudgeConfig(sys.executable, str(script), "m", "http://x", rubric_timeout_s=3600, workers=1, attempts=1)
    r = judge_output("x/task1", base, deliver, details, cfg2)
    argv = seen.read_text()
    assert r.status == "graded" and "--timeout-per-rubric 3600" in argv and "--max-workers 1" in argv


def test_a_budget_refusal_stops_judging_immediately_and_is_not_retried(rig):
    cfg, base, deliver, details = rig
    script = Path(cfg.script)
    script.write_text(r'''
import json, os, sys
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
n = a["--details-file"] + ".calls"
open(n, "a").write("x")
msg = "Failed to get judge response: Error code: 402 - {'error': {'message': 'kimi budget exhausted (spent $49.00)', 'type': 'budget_exceeded'}}"
json.dump({"total_score": 0, "max_score": 60, "rubrics": [{"result": {"criteria_results": [{"reasoning": msg}]}}]}, open(a["--details-file"], "w"))
''')
    r = judge_output("x/task1", base, deliver, details, cfg, sleep=lambda s: None)
    assert r.status == "ungraded" and r.budget_exhausted and r.attempts == 1
    assert len(Path(str(details) + ".calls").read_text()) == 1             # one call: no retries against an empty wallet


def test_a_finished_rollout_clears_the_error_of_an_earlier_crash(tmp_path):
    b = Scoreboard(tmp_path / "s.jsonl")
    b.mark_failed("a", "t", "RuntimeError: boom") if hasattr(b, "mark_failed") else b._append("a", "t", error="RuntimeError: boom")
    assert b.info("a", "t")["error"]
    b.mark_generated("a", "t", deliverables=["f"], stop="done")
    assert not b.info("a", "t")["error"] and b.finished("a", "t")
    assert not Scoreboard(tmp_path / "s.jsonl").info("a", "t")["error"]            # also after a reload


@pytest.mark.parametrize("msg", [
    "Failed to get judge response: Error code: 429 - {'error': {'message': 'Your account org-x is suspended due to insufficient balance, please recharge', 'type': 'exceeded_current_quota_error'}}",
    "Failed to get judge response: Error code: 429 - insufficient_quota",
    "Failed to get judge response: Error code: 402 - kimi budget exhausted (spent $49.00)"])
def test_provider_quota_errors_stop_judging_at_once_like_our_own_budget_refusal(rig, msg):
    cfg, base, deliver, details = rig
    Path(cfg.script).write_text(r'''
import json, sys
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
open(a["--details-file"] + ".calls", "a").write("x")
json.dump({"total_score": 0, "max_score": 60, "rubrics": [{"result": {"criteria_results": [{"reasoning": %r}]}}]}, open(a["--details-file"], "w"))
''' % msg)
    r = judge_output("x/task1", base, deliver, details, cfg, sleep=lambda s: None)
    assert r.status == "ungraded" and r.budget_exhausted and r.attempts == 1
    assert len(Path(str(details) + ".calls").read_text()) == 1                       # one call, no retries against a closed account


def test_an_ordinary_rate_limit_is_still_retried(rig):
    cfg, base, deliver, details = rig
    Path(cfg.script).write_text(r'''
import json, sys
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
open(a["--details-file"] + ".calls", "a").write("x")
json.dump({"total_score": 0, "max_score": 60, "rubrics": [{"result": {"criteria_results": [{"reasoning": "Failed to get judge response: Error code: 429 - rate limit reached for requests, retry later"}]}}]}, open(a["--details-file"], "w"))
''')
    r = judge_output("x/task1", base, deliver, details, cfg, sleep=lambda s: None)
    assert r.status == "ungraded" and not r.budget_exhausted and len(Path(str(details) + ".calls").read_text()) == cfg.attempts


# ── the recovery mode: the upstream judge unchanged except that thinking is off ──
UPSTREAM = r'''
import json, sys
from pathlib import Path

class _Completions:
    def create(self, **kw):
        extra = kw.get("extra_body") or {}
        off = (extra.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        if not off:
            raise ValueError("Judge returned empty content (finish_reason=length)")      # the reasoning used up what was left of the context
        return "verdict"

class _Client:
    def __init__(self):
        self.chat = self
        self.completions = _Completions()

def get_openai_client(base, key):
    return _Client()

def judge_rubric(idx, rubric, contents, model, base, key, timeout, retries, images):
    try:
        get_openai_client(base, key).chat.completions.create(model=model, messages=[])
        res = {"passed": True, "score": 30, "criteria_count": 1, "criteria_passed": 1, "overall_reasoning": "ok",
               "criteria_results": [{"index": 0, "criterion": "c", "passed": True, "reasoning": "criterion met", "evidence": ""}]}
    except ValueError as e:
        why = f"Failed after {retries} retries: {e}"
        res = {"passed": False, "score": 0, "criteria_count": 1, "criteria_passed": 0, "overall_reasoning": why,
               "criteria_results": [{"index": 0, "criterion": "c", "passed": False, "reasoning": "Failed to get judge response: " + why, "evidence": ""}]}
    return {"index": idx, "rubric": "r", "weight": 30, "result": res}, {}

def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    details = Path(args["--details-file"])
    existing = {r["index"]: r for r in json.loads(details.read_text()).get("rubrics", [])} if details.exists() else {}
    out = []
    for i in range(2):
        out.append(existing[i] if i in existing else judge_rubric(i, {}, "", "m", "b", "k", 1, 1, [])[0])
    details.write_text(json.dumps({"total_score": sum(r["result"]["score"] for r in out), "max_score": 60, "passed_count": 0, "total_count": 2, "rubrics": out}))
'''


@pytest.fixture
def upstream(tmp_path):
    script = tmp_path / "judge.py"
    script.write_text(UPSTREAM + "\nif __name__ == '__main__':\n    main()\n")
    base = tmp_path / "base" / "x" / "task1"
    base.mkdir(parents=True)
    (base / "RUBRICS.json").write_text("{}")
    deliver = tmp_path / "out"
    deliver.mkdir()
    (deliver / "a.md").write_text("hello")
    return script, tmp_path / "base", deliver, tmp_path / "details.json"


def test_a_cut_off_reply_is_recovered_with_thinking_off_and_the_recovered_rubrics_are_marked(upstream):
    script, base, deliver, details = upstream
    cfg = JudgeConfig(sys.executable, str(script), "m", "http://x", attempts=3, backoff_s=0, timeout_s=30, recovery_thinking_off=True)
    r = judge_output("x/task1", base, deliver, details, cfg, sleep=lambda s: None)
    assert r.status == "graded" and r.attempts == 2 and r.recovered == 2                 # pass 1 normal: both cut off; pass 2 is the recovery mode (thinking off) and answers
    d = json.loads(details.read_text())
    assert [x["recovered_with"] for x in d["rubrics"]] == ["thinking_off", "thinking_off"]


def test_with_recovery_off_a_cut_off_reply_stays_ungraded(upstream):
    script, base, deliver, details = upstream
    cfg = JudgeConfig(sys.executable, str(script), "m", "http://x", attempts=3, backoff_s=0, timeout_s=30, recovery_thinking_off=False)
    r = judge_output("x/task1", base, deliver, details, cfg, sleep=lambda s: None)
    assert r.status == "ungraded" and r.score is None and r.recovered == 0


def test_the_shim_changes_only_the_thinking_parameter(upstream):
    script, base, deliver, details = upstream
    import subprocess
    from agent_harness.bench import judge_shim
    out = subprocess.run([sys.executable, judge_shim.__file__, str(script), "--details-file", str(details), "--rubrics-file", str(base / "x" / "task1" / "RUBRICS.json"), "--output-dir", str(deliver)],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    d = json.loads(details.read_text())
    assert all(x["result"]["passed"] and x["recovered_with"] == "thinking_off" and x["result"]["criteria_results"][0]["reasoning"] == "criterion met" for x in d["rubrics"])


def test_recovery_with_thinking_off_is_off_by_default_and_in_the_g1_runner():
    import inspect
    from agent_harness.sim import experiment_jobbench
    assert JudgeConfig("p", "s", "m", "u").recovery_thinking_off is False
    assert "recovery_thinking_off=False" in inspect.getsource(experiment_jobbench.run)
