# Result of the judge-context qualification (criteria: QUALIFICATION_CRITERIA_v3.md, sha256 c8be6756..., fixed before the server change at 2026-11-28T17:28:31Z)
Server: qwen-reasoning, --max-model-len 98,304 -> 163,840, nothing else changed. Judge input, prompt, parser, thresholds unchanged.
1. PASS: oversize deliverable (mechanical_engineering_technicians/task2 r2, 144,169 text tokens): graded in 1 pass (1,211 s), 8/8 rubrics thinking-on verdicts, 0 recovered, 0 without reply.
2. PASS: agreement on the 3 v2 outputs (23 rubrics, 81 criteria): rubric-level 21/23 = 91.3 % (>= 85), criterion-level 74/81 = 91.4 % (>= 90), criteria passed 54 -> 55 (1.9 %, <= 10 %), 0 without reply, 0 recovered.
3. PASS: second judging of the oversize output matches on 8/8 rubrics (>= 6) and on every criterion.
4. PASS: pod Ready, 0 restarts; KV cache 185,314 tokens (>= 163,840; 1.13x); GPU 44,175/48,935 MiB used (4.7 GiB free); a 155,646-token request and a 90,951-token request completed; decode 39.9-40.0 tok/s vs baseline 40.0. (First long-prompt attempt undershot the 155,000 floor by synthetic-prompt calibration, 145,326; repeated at 155,646 and recorded in infra_after_long_retry.json; criteria unchanged.)
5. PASS: agent-harness at 1c98946d, 3,826 passed, 26 skipped; nothing in the evaluator regenerates an answer or replaces a stored verdict (stage runs used content-addressed folders and durable verdicts).
Conclusion: option A qualified; run 3 proceeds under amendment 4 (owner instruction 2026-11-28).
