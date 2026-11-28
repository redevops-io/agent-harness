"""Repeat qualification (QUALIFICATION_CRITERIA_v3.md, sha256 c8be6756...), server context 163,840, thinking-on judge, recovery_thinking_off=False, G1 judge settings.
Nothing regenerates an agent answer. Stage 1 regression, stage 3 determinism (same output again, fresh), stage 2 agreement on three outputs the judge answered in the failed run."""
import json, sys, time
from pathlib import Path
H = Path.home()
sys.path.insert(0, str(H / "qualification/g1-amend4-wt"))
from agent_harness.bench import evalkit
JB = H / "projects/harness2-upstream/dependencies/job-bench-eval"
ARCH = H / "qualification/jobbench/g1-run1-failed-evaluation-2026-11-26/g1-run1/out/rollouts"
Q = H / "qualification/jobbench/g1-ctx160k-qualification/qual3"
base = "http://192.168.40.105:30807/v1"
cfg = evalkit.JudgeConfig(str(JB / ".venv/bin/python"), str(JB / "eval/judge.py"), "qwen3.8-27b", base, "default", workers=2, max_retries=2, attempts=3, recovery_workers=1, recovery_thinking_off=False,
                          backoff_s=30.0, rubric_timeout_s=900, timeout_s=3 * 3600)
assert cfg.recovery_thinking_off is False
TASK = "mechanical_engineering_technicians/task2"; FAIL_OUT = H / "qualification/jobbench/g1-run2-failed-evaluation-2026-11-28/g1-run2/rollouts/eval-0004/mechanical_engineering_technicians__task2__r2/output"
out = {"started": time.time(), "server_max_model_len": 163840}
def summarize(path, r, secs):
    d = json.loads(path.read_text())
    return {"status": r.status, "attempts": r.attempts, "recovered": r.recovered, "failures": r.failures, "seconds": secs,
            "rubrics": [{"index": x["index"], "recovered_with": x.get("recovered_with"), "no_reply": x["result"]["overall_reasoning"].startswith("Failed after"), "passed": x["result"]["passed"],
                         "criteria": [c["passed"] for c in x["result"]["criteria_results"]]} for x in d["rubrics"]]}
def go(name, task, deliver):
    p = Q / f"{name}.json"
    for f in Q.glob(p.name + "*"): f.unlink()
    t0 = time.time(); r = evalkit.judge_output(task, JB / "dataset/main", deliver, p, cfg)
    return summarize(p, r, round(time.time() - t0))
out["stage1"] = go("stage1", TASK, FAIL_OUT); print("stage1", json.dumps(out["stage1"]), flush=True)
json.dump(out, open(Q / "qual_result.json", "w"), indent=1)
out["stage3"] = go("stage3", TASK, FAIL_OUT); print("stage3", json.dumps(out["stage3"]), flush=True)
json.dump(out, open(Q / "qual_result.json", "w"), indent=1)
rows = []
for name in ("bookkeeping_accounting_and_auditing_clerks__task2__r1", "civil_engineers__task3__r1", "secretaries_and_administrative_assistants_except_legal_medical_and_executive__task2__r1"):
    task = name.replace("__r1", "").replace("__", "/")
    s = go("stage2_" + name[:20], task, ARCH / "eval-0001" / name / "output")
    old = {x["index"]: x for x in json.loads((ARCH / "eval-0001" / "judge" / f"{name}.json").read_text())["rubrics"]}
    for x in s["rubrics"]:
        o = old[x["index"]]
        rows.append({"task": task, "rubric": x["index"], "no_reply": x["no_reply"], "old_passed": o["result"]["passed"], "new_passed": x["passed"], "recovered": x["recovered_with"],
                     "old_criteria": [c["passed"] for c in o["result"]["criteria_results"]], "new_criteria": x["criteria"]})
    out["stage2"] = rows; json.dump(out, open(Q / "qual_result.json", "w"), indent=1); print("stage2", name[:30], flush=True)
out["ended"] = time.time(); json.dump(out, open(Q / "qual_result.json", "w"), indent=1); print("DONE", flush=True)
