# JobBench task lists (seed-42 stratified split)

`train.txt` (21) and `test_all.txt` (44) come from `scripts/split_tasks.py --bench jb --seed 42` in upstream Harness².
`seen_during_development.txt` (22) are test tasks the agent-harness executor was run on while it was being debugged; **do not use them
for claims**. `unseen.txt` (22) are the test tasks no arm had run on; `stage1.txt` (every other one) and `stage2.txt` (the rest) split
them into two evaluation stages. This split is NOT the paper's 31/34 split.
