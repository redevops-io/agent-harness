# Harness² integration

Status: **groundwork only.** `agent_harness.h2` models and validates the Harness² file tree and stores
immutable versioned copies. It does not run the refinement pipeline, call any model, or reproduce any
published result. No performance claim is made here.

Background: [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0) and
[arXiv 2610.14224](https://arxiv.org/abs/2610.14224). Model weights stay frozen; the files around the model
(the *harness*) are refined. "Harness" here is a different thing from the rest of this package (tool
registry, approval, sandbox, guardrails) and from `HarnessProfile` in agentic-os.

## Upstream revision

Reviewed commit: `1cfe254e8d25b15f37048c3730134686abe60335` (`UPSTREAM_COMMIT`, 3 commits on `main`, no
release tag). Layouts, validation rules and the OpenCode/Codex surfaces below were read from that commit's
`harness2/core/tree.py`, `substrate/opencode.py` and `substrate/codex.py`. Upstream requires Python
3.12-3.13 and is POSIX-only; this package does not depend on it.

## Two surfaces, not one tree

The paper's eight components (C1-C8) map to different files depending on the executor:

| Component | OpenCode | Codex |
|---|---|---|
| C1-C4 system prompt, guardrails, memory, tool guidance | `systemprompt.md`, `guardrails.md`, `LongTermMEMORY.md`, `tool_descriptions/tool_guidance.md` (cap 24,000 chars combined) | all in `AGENTS.md` (cap 8,000 chars; must not be empty) |
| C5 subagents | `sub_agents/<name>/AGENT.md` (needs `mode: subagent`, `description:`) | same path (needs `description:`) |
| C6 skills | `skills/<name>/SKILL.md` (`name:` must equal directory; `description:` required) | `.agents/skills/<name>/SKILL.md` (`description:` required; `name:` optional) |
| C7 scripts | `scripts/<name>.py` (lowercase snake case) | same |
| C8 plugins | `plugin/<name>.ts` | `.codex/hooks.json` (Claude-Code hook schema; no TypeScript plugins) |

A skill counts as triggered only if its name appears in the always-on text. Skills with no such mention
are rejected, because the agent would never invoke them.

## What this package provides

- `Surface`, `OPENCODE`, `CODEX`: the surface data above, copied from upstream.
- `HarnessSpec(surface, files)`, `stock_spec(surface)`: an in-memory tree.
- `validate(spec)`: the upstream rules as a list of issues, plus three checks upstream does not make: path
  safety, grader-looking file names (rubric, verdict, score, answer key, ground truth), and a TypeScript
  brace/parenthesis balance check. Template braces (`{{`, `{%`, `{#`) are rejected in always-on text on both
  surfaces; upstream enforces that only for its Jinja install mode, so this is deliberately stricter.
- `save_artifact` / `load_artifact` / `lineage`: SHA-256 content-addressed artifacts that record their
  surface and parent. Invalid trees are refused, identical trees collapse to one id, and a modified stored
  tree fails loading.

## Verification done (2026-10-19)

- `tests/test_h2.py`: unit tests for every rule on both surfaces.
- `tests/test_h2_conformance.py`: runs a 20-case corpus through both this validator and upstream's
  `harness2.core.tree.validate`. All 20 agree on accept/reject. The test is skipped when upstream is not
  installed (it needs Python 3.12+: `pip install` a checkout at the commit above).
- Upstream's credential-free `harness2-demo --k 2` ran on macOS with Python 3.13 (2 tasks, 0 failed). The
  five harness trees it wrote (the base and four library entries) all pass `validate()` and round-trip
  through the artifact store.

Not tested: any model-backed run, any benchmark, Codex trees from a real run, TypeScript compilation.

## What validation does not do

Passing `validate()` means a tree is well formed, **not that it is safe**. Scripts, hooks and plugins are
executable code and prompts are untrusted. A refined harness must not reach production without runtime
policy checks, regression tests and explicit promotion. Tool allowlists, secrets, egress, sandboxing,
spend limits and human approval belong to the runtime, not to editable files.

## Not done yet

- The adapter that runs upstream refinement from here (optional extra), and its run manifest.
- Review of the paper's appendices against upstream prompts; the LAB 1,251 vs 1,143 task-count question.
- Any benchmark run (WorkBuddy-Bench, JobBench, LAB). Costs are unmeasured; no spend cap is approved.
- Ownership decision for `approval.py` and migration of copies in other repos.
