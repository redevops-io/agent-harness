"""The evaluation failures of G1 runs 1 and 2, reproduced, and the recovery shown (see docs/experiments/G1_PLAN.md, amendment 4).

Run 1 stopped because judge replies were cut off (a 61,716-token prompt in a 65,536-token context). Run 2 stopped because the judge prompt for one deliverable was larger than the judge's whole context
(98,304 tokens), a request that can never be answered. Both are reproduced below with a judge that behaves like the real one rubric by rubric: a rubric that already has a verdict is not asked again,
and every verdict is written as it arrives, so a process that stops part way leaves exactly what it finished.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from agent_harness.bench.evalkit import JudgeConfig, classify_details, context_overflow, judge_output

# The message the real server returned in run 2, verbatim (tests/data would hide it; it is the evidence).
RUN2_ERROR = ("Failed after 2 retries: Error code: 400 - {'error': {'message': \"This model's maximum context length is 98304 tokens. However, your prompt contains at least 98305 input tokens, "
              "which exceeds this limit even without any output tokens. Please reduce the length of the input prompt. (parameter=input_tokens, value=98305)\"}}")

# A judge like the upstream one: persists after every rubric (as ``append_rubric_result`` does), reuses stored results, and has a context limit and a prompt size.
JUDGE = r'''
import json, os, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
details = args["--details-file"]
n_rub = json.load(open(args["--rubrics-file"]))["n"]
ctx, prompt = int(os.environ["FAKE_CTX"]), int(os.environ["FAKE_PROMPT"])
die_after = int(os.environ.get("FAKE_DIE_AFTER", "-1"))        # stop the whole process (no clean exit) after this many rubrics were judged IN THIS CALL
cut_off = {int(x) for x in os.environ.get("FAKE_CUT_OFF", "").split(",") if x}     # rubrics whose reply needs more room than the context leaves (thinking on)
open(details + ".calls", "a").write("call\n")
have = {}
if os.path.exists(details):
    have = {r["index"]: r for r in json.load(open(details)).get("rubrics", [])}
def fail(why):
    return {"passed": False, "score": 0, "criteria_count": 2, "criteria_passed": 0, "overall_reasoning": why,
            "criteria_results": [{"index": i, "criterion": "c", "passed": False, "reasoning": "Failed to get judge response: " + why, "evidence": ""} for i in range(2)]}
def write(rs):
    json.dump({"total_score": sum(r["result"]["score"] for r in rs), "max_score": 10 * len(rs), "passed_count": sum(1 for r in rs if r["result"]["passed"]), "total_count": n_rub,
               "rubrics": sorted(rs, key=lambda r: r["index"])}, open(details, "w"))
done_here = 0
for i in range(n_rub):
    if i in have:
        continue
    open(details + ".judged", "a").write(f"{i}\n")                       # every rubric that is actually sent to the judge, across all calls
    if prompt >= ctx:
        res = fail(os.environ["FAKE_OVERFLOW_MESSAGE"].replace("N", str(prompt)))
    elif i in cut_off and prompt > ctx * 0.9:
        res = fail("Failed after 2 retries: Judge returned empty content (finish_reason=length)")
    else:
        res = {"passed": True, "score": 10, "criteria_count": 2, "criteria_passed": 2, "overall_reasoning": f"verdict for {i}",
               "criteria_results": [{"index": k, "criterion": "c", "passed": True, "reasoning": "met", "evidence": ""} for k in range(2)]}
    have[i] = {"index": i, "rubric": "r", "weight": 10, "result": res}
    write(list(have.values()))
    done_here += 1
    if die_after >= 0 and done_here >= die_after:
        os._exit(9)
'''


@pytest.fixture
def rig(tmp_path, monkeypatch):
    script = tmp_path / "judge.py"
    script.write_text(JUDGE)
    base = tmp_path / "base" / "x" / "task1"
    base.mkdir(parents=True)
    (base / "RUBRICS.json").write_text(json.dumps({"n": 8}))
    deliver = tmp_path / "out"
    deliver.mkdir()
    (deliver / "a.md").write_text("the deliverable")
    cfg = JudgeConfig(sys.executable, str(script), "m", "http://x", attempts=3, backoff_s=0, timeout_s=20, recovery_thinking_off=False)
    monkeypatch.setenv("AGENT_HARNESS_ENV_PASS", "FAKE_CTX,FAKE_PROMPT,FAKE_DIE_AFTER,FAKE_CUT_OFF,FAKE_OVERFLOW_MESSAGE")
    details = tmp_path / "details.json"

    def set_env(ctx=98304, prompt=50_000, die_after=-1, cut_off="", overflow=RUN2_ERROR.replace("98305", "N")):
        for k, v in {"FAKE_CTX": ctx, "FAKE_PROMPT": prompt, "FAKE_DIE_AFTER": die_after, "FAKE_CUT_OFF": cut_off, "FAKE_OVERFLOW_MESSAGE": overflow}.items():
            monkeypatch.setenv(k, str(v))

    def go():
        return judge_output("x/task1", tmp_path / "base", deliver, details, cfg, sleep=lambda s: None)

    def judged():
        f = Path(str(details) + ".judged")
        return [int(x) for x in f.read_text().split()] if f.exists() else []

    def calls():
        f = Path(str(details) + ".calls")
        return len(f.read_text().split()) if f.exists() else 0

    return set_env, go, judged, calls, details


# ── run 2: the prompt does not fit the judge's context ───────────────────────────────────────────────────
def test_the_real_run_2_error_is_recognised_as_a_context_overflow():
    rubric = {"index": 0, "result": {"overall_reasoning": RUN2_ERROR, "criteria_results": [{"reasoning": "Failed to get judge response: " + RUN2_ERROR}]}}
    assert context_overflow(rubric) == (98304, 98305)
    assert context_overflow({"index": 1, "result": {"overall_reasoning": "Failed after 2 retries: Judge returned empty content (finish_reason=length)",
                                                    "criteria_results": [{"reasoning": "Failed to get judge response: x"}]}}) is None      # run 1's failure is a different thing


def test_run_2_a_prompt_larger_than_the_context_is_ungraded_at_once_not_retried_and_says_why(rig):
    set_env, go, judged, calls, details = rig
    set_env(ctx=98304, prompt=144_169)
    r = go()
    assert r.status == "ungraded" and r.context_exceeded and r.prompt_limit == 98304 and r.prompt_tokens_at_least == 144_169
    assert r.attempts == 1 and calls() == 1                       # three passes would be the same request three times
    assert "not retried" in r.reason and "98304" in r.reason
    assert r.score is None                                        # never a zero


def test_run_2_graded_verdicts_are_kept_and_the_deliverable_is_graded_when_the_context_is_made_large_enough(rig):
    set_env, go, judged, calls, details = rig
    set_env(ctx=98304, prompt=144_169)
    assert go().context_exceeded
    d = json.loads(details.read_text())
    assert all("maximum context length" in r["result"]["overall_reasoning"] for r in d["rubrics"])
    set_env(ctx=163_840, prompt=144_169)                          # the judge's context changed (an owner decision about the experiment, see amendment 4)
    r = go()
    assert r.status == "graded" and r.score == 1.0 and r.count == 8
    assert sorted(judged()) == sorted(list(range(8)) * 2)         # asked once while it could not fit (all 8 failed), once now: nothing was asked a third time


def test_a_mix_of_overflow_and_other_failures_is_still_retried(rig, monkeypatch):
    set_env, go, judged, calls, details = rig
    set_env(ctx=98304, prompt=50_000)
    pre = {"total_score": 0, "max_score": 20, "passed_count": 0, "total_count": 8, "rubrics": [
        {"index": 0, "rubric": "r", "weight": 10, "result": {"passed": False, "score": 0, "criteria_count": 2, "criteria_passed": 0, "overall_reasoning": RUN2_ERROR,
                                                              "criteria_results": [{"index": 0, "criterion": "c", "passed": False, "reasoning": "Failed to get judge response: " + RUN2_ERROR}]}},
        {"index": 1, "rubric": "r", "weight": 10, "result": {"passed": False, "score": 0, "criteria_count": 2, "criteria_passed": 0, "overall_reasoning": "Failed after 2 retries: timed out",
                                                              "criteria_results": [{"index": 0, "criterion": "c", "passed": False, "reasoning": "Failed to get judge response: timed out"}]}}]}
    assert not classify_details(pre, 1).context_exceeded
    assert classify_details({**pre, "rubrics": pre["rubrics"][:1]}, 1).context_exceeded


# ── run 1: replies cut off because the prompt leaves no room ─────────────────────────────────────────────
def test_run_1_cut_off_replies_stay_ungraded_after_the_bounded_passes_and_the_answered_rubrics_are_kept(rig):
    set_env, go, judged, calls, details = rig
    set_env(ctx=65_536, prompt=61_716, cut_off="0,1,3,4")         # 94 percent of the context: the four rubrics that need room get no reply, on every pass
    r = go()
    assert r.status == "ungraded" and not r.context_exceeded and r.attempts == 3 and r.score is None
    answered = [x["index"] for x in json.loads(details.read_text())["rubrics"] if x["result"]["passed"]]
    assert sorted(answered) == [2, 5, 6, 7]
    assert judged().count(2) == 1 and judged().count(0) == 3       # the answered ones were asked once; the cut-off ones once per pass (bounded at 3)


def test_run_1_is_graded_when_the_judge_has_room_and_only_the_four_missing_rubrics_are_asked_again(rig):
    set_env, go, judged, calls, details = rig
    set_env(ctx=65_536, prompt=61_716, cut_off="0,1,3,4")
    assert go().status == "ungraded"
    set_env(ctx=98_304, prompt=61_716, cut_off="0,1,3,4")         # amendment 3: more context, nothing else changed
    before = len(judged())
    r = go()
    assert r.status == "graded" and r.count == 8
    assert sorted(judged()[before:]) == [0, 1, 3, 4]               # the four with no verdict; the four with one were not sent again


# ── interruption ─────────────────────────────────────────────────────────────────────────────────────────
def test_a_judge_that_stopped_part_way_is_not_graded_from_what_it_finished_and_resumes_where_it_stopped(rig):
    set_env, go, judged, calls, details = rig
    set_env(die_after=3)                                          # the process dies (no clean exit) after judging 3 rubrics, leaving a file whose totals cover only those
    r = go()
    assert r.status == "graded" and r.count == 8                  # later passes finish what is missing (the fake dies again after 3 each time: 3 + 3 + 2)
    assert sorted(judged()) == list(range(8))                      # every rubric was sent exactly once across the interrupted passes: nothing finished was judged again


def test_a_partial_details_file_is_never_read_as_a_grade():
    partial = {"total_score": 30, "max_score": 30, "passed_count": 3, "total_count": 8,
               "rubrics": [{"index": i, "result": {"passed": True, "score": 10, "criteria_count": 2, "criteria_passed": 2,
                                                   "criteria_results": [{"reasoning": "met"}]}} for i in range(3)]}
    r = classify_details(partial, 1)
    assert r.status == "ungraded" and "3 of 8" in r.reason and r.score is None


def test_interruption_with_no_pass_left_ends_ungraded_and_a_restart_continues(rig):
    set_env, go, judged, calls, details = rig
    set_env(die_after=2)
    r = go()                                                      # 3 passes of 2 rubrics: 6 of 8
    assert r.status == "ungraded" and "6 of 8" in r.reason
    set_env()                                                     # restart: a new call of judge_output, no state but the files
    r2 = go()
    assert r2.status == "graded" and sorted(judged()) == list(range(8))
