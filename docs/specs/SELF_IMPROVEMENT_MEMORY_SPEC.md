# Workstream A — Self-Improvement Memory and Recursive Harness Optimization

Owner specification, received 2026-11-25. This is the authoritative implementation plan for this workstream; the text below is the owner's, reformatted as Markdown only (headings and lists), with nothing added or removed. Part of "rAIls Continual Self-Improvement and Agent-Native Application Compatibility"; the other workstream's specification is in agentic-os `docs/specs/AGENT_NATIVE_APPS_SPEC.md`.

Study the complete paper, including its implementation appendix and experimental controls:

- https://arxiv.org/abs/2611.18856
- https://arxiv.org/html/2611.18856v1
- https://lucasxinyu.github.io/sim/

## Audit

Audit the existing implementation across context-runtime, agent-harness, agentic-os, Sidekick, redevops-rag, runtime-contracts, agentic-tests, redevops-qa and relevant application repositories.

Trace the actual execution and learning paths. Do not rely on README claims, simulated reward examples or the existence of memory-related classes.

Compare the implementation against Sections 3.1–3.3 and Appendix D of the paper. Produce a matrix of implemented, partial, missing and experimentally verified capabilities, with exact source and test references.

## Implementation

1. Reuse existing execution evidence, reward contracts, learning aggregators, skill stores and optimizer infrastructure.
2. Implement persistent optimizer-level Self-Improvement Memory, separate from conversational and task-solving memory.
3. Organize improvement history by recurring root-cause hypotheses.
4. Record interventions, rationale, predicted outcomes before evaluation, observed outcomes, missed predictions, regressions, model versions, benchmark identity, harness versions and evaluation provenance.
5. Preserve historical evidence when an intervention is revised, replaced or removed.
6. Implement or complete the proposer–evaluator–selector loop for actual harness improvements, including prompts, context assembly, tool orchestration, parsing, verification, completion gates and workflow strategies.
7. Make SIM available to both proposer and selector.
8. Support continuation across compatible models, model generations, datasets, application domains and deployment environments.
9. Use Context Runtime to optimize memory selection when necessary, but compare retrieval-based memory against the paper's complete-memory approach.
10. Evaluate candidate changes in isolated environments. Keep authorization, governance, security policy and independent evaluators outside the optimizer's modification authority.
11. Support versioned promotion, rollback, regression detection and bounded experimentation.
12. Expose improvement history, evidence and measured outcomes in Projects Dashboards.

## Benchmarking

Implement five comparison arms:

- Cold start.
- Harness transfer without SIM.
- Problem-only memory.
- Full SIM.
- Existing rAIls learning implementation.
Use matched starting harnesses, target models, datasets, evaluation budgets and candidate budgets where appropriate.

Measure task success, source-performance retention, target improvement, cost, latency, optimization overhead, regressions and transfer effectiveness.

Keep held-out evaluation inaccessible to the optimizer. Record negative results and unsuccessful interventions in the experimental evidence.

Run cross-model and cross-dataset comparisons using available Claude, Gemini, Grok and local Qwen models. Include actual workspace, QA, retrieval, legal-document and cross-app tasks.

Do not claim recursive self-improvement merely because a bandit learns to select an existing strategy.

## Delivery

Treat both workstreams as implementation and release tasks, not standalone research exercises.

Audit before introducing new components. Reuse existing infrastructure. Fix verified gaps, run real acceptance tests, publish the applicable releases and provide reproducible evidence.

Proceed directly without delegation. Routine engineering, merges, releases and previously authorized deployments remain authorized. Preserve required automated checks and application-level governance. Escalate only genuine external blockers.
