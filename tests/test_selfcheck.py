import json
import sys

import pytest

from agent_harness.bench import selfcheck as sc

needs_docker = pytest.mark.skipif(not sc.docker_ok(), reason="docker daemon not reachable")


@needs_docker
def test_sandbox_check_passes_on_a_correct_sandbox():
    c = sc.check_sandbox()
    assert c.status == "PASS", c.detail


@needs_docker
def test_scripted_pipeline_runs_end_to_end_with_real_sandbox_and_context_runtime():
    pytest.importorskip("context_runtime")
    c = sc.check_pipeline()
    assert c.status == "PASS", c.detail


def test_endpoint_check_reports_a_dead_endpoint_as_fail_not_crash():
    c = sc.check_endpoint("dead", "http://127.0.0.1:9/v1", "m", timeout=2)
    assert c.status == "FAIL" and "dead" in c.name


def test_judge_controls_flag_a_judge_that_returns_the_same_score_for_everything(tmp_path):
    from agent_harness.bench.evalkit import JudgeConfig
    judge = tmp_path / "j.py"
    judge.write_text('import json,sys\na=dict(zip(sys.argv[1::2],sys.argv[2::2]))\n'
                     'json.dump({"total_score":50,"max_score":100,"passed_count":1,"total_count":2,"rubrics":[]},open(a["--details-file"],"w"))\n')
    base = tmp_path / "base" / "x" / "task1"
    base.mkdir(parents=True)
    (base / "RUBRICS.json").write_text("{}")
    cal = tmp_path / "cal"
    items = []
    for i in range(6):
        d = cal / "outputs" / f"o{i}"
        d.mkdir(parents=True)
        (d / "a.md").write_text("text")
        items.append({"id": f"o{i}", "task": "x/task1", "dir": f"outputs/o{i}", "grok47": i / 10})
    (cal / "manifest.json").write_text(json.dumps({"items": items}))
    cfg = JudgeConfig(sys.executable, str(judge), "m", "http://x", attempts=1, timeout_s=10)
    res = {c.name: c for c in sc.check_judge_controls(cfg, tmp_path / "base", cal)}
    assert res["judge negative control"].status == "FAIL"       # junk scored 0.5 > 0.25: a constant judge cannot pass
    assert res["judge no-output control"].status == "PASS"
    assert res["judge calibration"].status == "FAIL"            # constant scores carry no rank information (nan/0 correlation)
