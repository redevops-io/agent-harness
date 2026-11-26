# Run 5: complete memory against Context Runtime's selected memory

Pair and carried state: run 3's `source/` (the evolved egress-vetting harness and its one-problem memory), the command-vetting task set, 6 rounds, 2 candidates, feedback v2, rejected attempts recorded, 3 replicates per arm with the arms alternating. Both arms see the memory under the full condition and differ only in what the proposer and selector read: the complete record, or `context_runtime.learning.sim_select.select` under a budget of 3,000 tokens (`estimate_tokens`, 4 characters per token), with the query built from the failing tasks, `systemprompt.md` and the failing replies. Context Runtime commit `3b55bec98f681b1a25de09787217d715cb3256b6` (a clean checkout; recorded in `results.json`). Model: `Qwen/Qwen3.8-27B-FP8` on vLLM 0.34.0, 65,536-token context, thinking disabled; policy temperature 0, 400 output tokens; proposer and selector temperature 0.3, 4,000. The model server is local: no per-token price applies, so tokens and seconds are the cost measures.

## Why run 5 was restarted, and what of the first start remains

The first start of run 5 ran for a few minutes and was stopped by me, before any arm had finished, because I found that it recorded no token usage or per-call latency (the proposer and selector discarded the server's `usage`), which was required of this run, and the history check was not yet in the code. The code was changed only to add that measurement (a metered client around each model client, and a check of the intervention history that parses the text the proposer read); no setting of the experiment changed. **The interrupted output directory was deleted when the run was relaunched, and its log was overwritten; I cannot preserve it.** What survives from before the restart is the one-round smoke run of the metered code (`smoke/one-round-metered`) and this note. That is a loss of evidence of an interrupted run that produced no result; it should have been kept.

## Result

Target held-out (command-vetting, 30 tasks), start 0.567 for every arm (the carried harness, evaluated once):

| Arm / replicate | Status | Target held-out final | Source held-out final (retention) | Promotions | Engineer prompt tokens (calls) | Engineer seconds | Memory shown to the proposer, by round (tokens) |
|---|---|---|---|---|---|---|---|
| complete 1 | completed | 0.600 | 0.833 | 1 | 119,941 (16) | 201 | 3,665 to 5,389 |
| complete 2 | completed | 0.967 | 0.700 | 2 | 85,939 (12) | 73 | 3,665 to 4,835 |
| complete 3 | completed | 0.533 | 0.600 | 3 | 139,067 (18) | 143 | 3,665 to 5,630 |
| selected 1 | completed | 0.600 | 0.833 | 3 | 64,823 (13) | 228 | 2,059 to 2,885 |
| selected 2 | **stopped: BudgetTooSmall** (needs 3,030, budget 3,000), after round 3 | not measured | not measured | | 28,894 (6) | 48 | 2,059 to 2,762 |
| selected 3 | **stopped: BudgetTooSmall** (needs 3,035, budget 3,000), after round 3 | not measured | not measured | | 40,950 (8) | 119 | 2,059 to 2,737 |

- **Accuracy.** The complete arm's three replicates end at 0.600, 0.967 and 0.533 on the same task set from the same start: a spread of 0.43 between replicates of one arm. The one selected replicate that finished ends at 0.600, inside that spread. Two of three selected replicates could not finish at this budget. There is no comparison of accuracy to report beyond that: one completed selected run cannot be compared with three noisy complete ones.
- **Input tokens.** The proposer and selector prompts were 33 percent shorter per call with selection (mean 4,974 prompt tokens per engineer call, against 7,461 for the complete record: the memory itself 2,059 to 2,885 tokens against 3,665 to 5,630). Total engineer input depends on how many calls a run makes (6 to 18) and varies more than the per-call difference does.
- **Policy tokens and seconds** are the same kind of quantity in both arms (the policy model runs the 30 evolve tasks for every candidate); they are in `results.json` (`policy_usage`) and are not an effect of memory.
- **Latency.** Seconds per engineer call: complete 12.6, 6.1, 8.0; selected 17.5, 8.1, 14.8. Selection did not make a call faster here: most of a call's time is decoding the reply (completion tokens 2,342 to 8,086 per run), not reading 2,500 fewer prompt tokens.
- **BudgetTooSmall happened, in two of three selected replicates.** The budget was fixed before the run at 3,000, above the 2,606 minimum I measured on run 3's final memory. The memory grows during the target run as attempts are recorded, and with ONE problem the mandatory evidence is that problem's whole history, compacted: its minimum grew past 3,000 after round 3 of two runs (3,030 and 3,035). The tool did what it promises (it refused rather than drop evidence, and said how much it needed); the arm stopped there, with no fallback, as designed. The conclusion is about this budget: at 3,000 tokens selection cannot serve a one-problem memory that has more than about 8 attempts.
- **Preservation of the intervention history** (corrected check, see below; recomputed on each arm's final record, `history_recomputed_final_records.json`): for the 3 records where the budget was enough, the selected view kept all 8 of 8 attempts with their predictions, fixed, missed, regressed, accepted, passed, total, files and provenance exactly as recorded; it did not keep the wording: 0 of 8 attempt descriptions and 0 of 8 rationales verbatim (compaction shortens the description and drops the rationale). For the complete arm everything is verbatim, by construction. For the 3 final records too large for the budget (complete 1: needs 3,176; complete 3: 3,751; selected 2 and 3: 3,030 and 3,035), selection refuses.
- **No problem was ever omitted, because there was only one problem.** Every record in this run has one problem.

## What this does NOT show

**A one-problem memory cannot validate relevance ranking or selection across problems.** `sim_select` ranks problems by shared tasks, files and wording, makes some mandatory by the regression evidence, and lists the rest in an index. With one problem there is nothing to rank, nothing to omit and no mandatory-versus-optional choice: the only mechanism exercised is compaction of one problem's prose. Whether selection picks the right problems, keeps the relevant ones and serves a proposer as well as the whole record when problems compete cannot be learned from these memories, and no record was added or split to make it possible. That is the question `NEXT_EXPERIMENT.md` is designed to ask on workloads that produce several distinct failure patterns.

Also not shown: any accuracy difference between complete and selected memory (one completed selected run); variance of the selected arm; behaviour at other budgets (the budget was fixed in advance and not tuned); a model other than this Qwen; the effect of a learned or model-based selector.

## A bug in my own measurement, found and corrected

The per-round history check recorded during the run (`selection_trace` in `results.json`) keyed attempts by (round, sample), so a carried source attempt and a target attempt with the same round number collided, and one attempt per record was misattributed: it reported 5 of 6 identical for the complete arm too, where identity holds by construction. The recorded per-round `history` values are therefore not used. The check now keys on (run, round, sample, harness) with a regression test, and was recomputed on each arm's final record (the file above); that recomputation is a reconstruction after the run, with a query-free selection, labelled as such.

## Note on the runner as committed

Run 5 loaded `sim_select` by importing it as a package from the pinned checkout. The committed runner loads the same file by path instead (the repository's boundary tests forbid importing Context Runtime from this package); the module is the same file at the same commit, so the code that ran is the code that is recorded.
