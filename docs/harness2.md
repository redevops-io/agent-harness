# Harness² integration

Status: **groundwork only.** `agent_harness.h2` defines and validates the Harness² file tree and stores
immutable versioned copies. It does not run the refinement pipeline, call any model, or reproduce any
published result. No performance claim is made here.

Background: [google-research/harness2](https://github.com/google-research/harness2) (Apache-2.0) and
[arXiv 2610.14224](https://arxiv.org/abs/2610.14224). They keep model weights frozen and refine the files
around the model: a *harness* of eight components. This is a different meaning of "harness" from the rest
of this package (tool registry, approval, sandbox, guardrails) and from `HarnessProfile` in agentic-os.

## The eight components

| Id | Component | Path | Activation |
|---|---|---|---|
| C1 | System prompt | `systemprompt.md` | every call |
| C2 | Guardrails | `guardrails.md` | every call |
| C3 | Memory | `LongTermMEMORY.md` | every call |
| C4 | Tool guidance | `tool_descriptions/tool_guidance.md` | every call |
| C5 | Subagents | `sub_agents/<name>/AGENT.md` | task tool, own context |
| C6 | Skills | `skills/<name>/SKILL.md` | skill tool, on invocation |
| C7 | Scripts | `scripts/<name>.py` | run via bash |
| C8 | Plugins | `plugin/<name>.ts` | inside the tool loop |

Paths come from the paper (Table 5). **They have not yet been checked against a reviewed upstream commit.**
Candidate upstream commit seen on 2026-10-19: `1cfe254e8d25b15f37048c3730134686abe60335` (HEAD at the time;
not reviewed, not pinned).

## What this package provides

- `HarnessSpec`, `stock_spec()`: an in-memory tree and the placeholder base tree.
- `validate(spec)`: structural checks. A skill needs a trigger line; no template braces in always-on text
  (C1-C4); every `scripts/...` file named in always-on text or Markdown must exist; Python scripts must
  compile; plugin braces must balance; no file whose name looks like a grader artifact (rubric, verdict,
  score, answer key, ground truth); every path must belong to a component and stay inside the tree.
- `save_artifact` / `load_artifact` / `lineage`: content-addressed (SHA-256) artifacts with a parent link.
  Invalid trees are refused, identical trees collapse to one id, and a modified stored tree fails loading.
- `render_codex_agents_md(spec)`: Codex keeps always-on text in `AGENTS.md`.

Assumption to verify upstream: the exact "trigger line" format. The validator accepts a frontmatter
`description:`, a `Trigger:` line, or a `Use when` line.

## What validation does not do

Passing `validate()` means a tree is well formed, **not that it is safe**. Scripts (C7) and plugins (C8)
are executable code and prompts are untrusted. TypeScript is not compiled here. A refined harness must not
reach production without runtime policy checks, regression tests and explicit promotion. Tool allowlists,
secrets, egress, sandboxing, spend limits and human approval belong to the runtime, not to editable files.

## Not done yet

- Pinning and reviewing an upstream commit; the upstream adapter (`agent_harness.h2` extra).
- Verifying layouts, prompts and commands against that commit; auditing the paper's appendices.
- Any benchmark run (WorkBuddy-Bench, JobBench, LAB). Costs are unmeasured.
- Ownership decision for `approval.py` and migration of copies in other repos.
