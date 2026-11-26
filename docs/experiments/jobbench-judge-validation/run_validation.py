"""Judge validation (written before it ran): can the LOCAL model judge JobBench outputs the way an independent judge did?
Independent scores: Kimi k2.7-code, per-criterion verdicts stored by the pre-registered final4 experiment (rep 1). The local judge (Qwen3.8-27B FP8, vLLM)
re-scores a seeded, score-stratified sample of the same stored outputs, with the unchanged upstream judge; 4 of them are scored a second time (test-retest).
Not tuned: one pass, settings fixed here. Outputs the local judge cannot grade (for example a deliverable larger than its context) are reported as ungraded, not dropped.
usage: run_validation.py N_SAMPLE N_RETEST"""
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, "/Users/arybach/qualification/public-review/src/agent-harness")
sys.path.insert(0, "/Users/arybach/Documents/Agent_harness/harness2-experiment/final4")
import judge_final4 as J
from agent_harness.bench.evalkit import JudgeConfig, judge_output
OUT = Path("/Users/arybach/qualification/jobbench/judge-validation"); n_sample, n_retest = int(sys.argv[1]), int(sys.argv[2])
cfgj = json.loads((J.E / "rerun3b_config.json").read_text())["judge"]
local = JudgeConfig(cfgj["python"], cfgj["script"], "qwen3.8-27b", "http://192.168.40.105:30807/v1", "default", workers=2, max_retries=2, attempts=1, rubric_timeout_s=900, timeout_s=3 * 3600)
cands = []
for arm, task, d in J.outputs(1):
    kf = J.F / "judge_details" / "rep1" / arm / f"{task.replace('/', '__')}.json"
    if d is not None and kf.exists() and Path(d).exists():
        k = json.loads(kf.read_text()); cands.append({"arm": arm, "task": task, "dir": str(d), "kimi_file": str(kf), "kimi_passed": k["passed_count"], "kimi_total": k["total_count"], "kimi_share": k["passed_count"] / max(1, k["total_count"])})
cands.sort(key=lambda c: (c["kimi_share"], c["arm"], c["task"]))
rng = random.Random(20261126)
buckets = [cands[i::n_sample] for i in range(n_sample)]                      # one from each slice of the Kimi score range
sample = [rng.choice(b) for b in buckets if b][:n_sample]
plan = {"written_before_running": True, "candidates": len(cands), "sample": sample, "retest": [s["arm"] + "/" + s["task"] for s in rng.sample(sample, min(n_retest, len(sample)))], "local_judge": {"model": "qwen3.8-27b", "root": "Qwen/Qwen3.8-27B-FP8", "server": "vLLM 0.34.0", "rubric_timeout_s": 900, "workers": 2, "max_retries": 2}}
(OUT / "plan.json").write_text(json.dumps(plan, indent=1))
rows = []
for s in sample:
    for rep in (1, 2) if (s["arm"] + "/" + s["task"]) in plan["retest"] else (1,):
        det = OUT / "details" / f"{s['arm']}__{s['task'].replace('/', '__')}__r{rep}.json"
        t0 = time.time()
        r = judge_output(s["task"], J.BASE, Path(s["dir"]), det, local)
        row = {**{k: s[k] for k in ("arm", "task", "kimi_passed", "kimi_total")}, "rep": rep, "status": r.status, "reason": r.reason, "criteria_passed": r.criteria_passed, "criteria_total": r.criteria_total, "seconds": round(time.time() - t0, 1), "details": str(det)}
        rows.append(row); print(json.dumps({k: row[k] for k in ("arm", "task", "rep", "status", "criteria_passed", "criteria_total", "kimi_passed", "seconds")}), flush=True)
        (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
print("DONE")
