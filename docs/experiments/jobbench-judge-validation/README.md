# Can the local model judge JobBench outputs? Validation against an independent judge (2026-11-26)

Written before it ran (`plan.json`, `run_validation.py`: a seeded, score-stratified sample of 12 stored outputs, 4 of them scored twice). Not tuned; one pass.

**Independent scores:** Kimi k2.7-code, per-criterion verdicts stored by the pre-registered four-harness experiment (replicate 1, judge temperature 1, the unchanged upstream judge).
**Judge under test:** Qwen3.8-27B (FP8, vLLM 0.34.0, 65,536-token context), the same upstream judge script, rubric timeout 900 s, 2 workers, 2 retries per rubric, no reasoning-effort setting sent.
**Files:** `criterion_verdicts.json` (pass/fail per criterion, local and Kimi, no deliverable text), `rows.json`, `plan.json`, `run_validation.py`.

| Measure | Result |
|---|---|
| Outputs sampled / gradeable by the local judge | 12 / 11 |
| Criterion-level agreement with Kimi (345 criteria) | **94.8 %**, Cohen's kappa **0.895** |
| Disagreements | 10 criteria the local judge passes and Kimi fails; 8 the local judge fails and Kimi passes (local 187 passes, Kimi 185: no bias in the total) |
| Rubric-level (all criteria of a rubric) agreement | 99 of 103 (96.1 %) |
| Weighted score per output, local against Kimi | Pearson r **0.937** (n = 11); 8 of 11 identical, the others differ by 0.09 to 0.17 |
| Test-retest of the local judge (3 outputs scored twice, 91 criteria) | 96.3 %, 97.1 %, 96.6 % criterion agreement (passed counts moved by 1) |
| Ungradable | **1 of 12 outputs** (social_science_research_assistants/task2): "28 rubric criteria got no judge reply" after 12 s, twice. Most likely a deliverable beyond the local context; not investigated further. |
| Judge time | 4.5 to 13 minutes per output (about 12 rubrics each) |

## What this supports

On these 11 outputs the local judge reproduces an independent, larger model's verdicts closely and consistently enough to be used for the JobBench experiment, **with the following limits, which are part of the result**:

- **The sample covers low-scoring outputs only.** The stored outputs score 0 to 0.62 on Kimi's scale (24 candidates, none above 0.62). Agreement on good deliverables, where fine distinctions decide the score, is untested.
- **Eleven outputs from six tasks and four harnesses**; criteria within an output are not independent, so 345 criteria are fewer independent observations than 345. Nothing here estimates agreement on other task types.
- **The independent judge is itself a model at temperature 1** with its own error; agreement with it is not accuracy. No human-scored outputs were available.
- **About one output in twelve could not be graded locally** (probably size). An experiment that depends on such tasks must report them as ungraded, not as failures of the harness under test.
- The same model family acts and judges in the planned experiment; the test above is a judge-versus-judge comparison, not a check that the judge is blind to its own family's outputs.

The decision in `NEXT_EXPERIMENT.md` stands: local judge, agreement published beside every score.
