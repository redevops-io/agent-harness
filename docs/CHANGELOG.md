# Changelog

Notable changes to `agent-harness`. The package version in `pyproject.toml` is `0.1.0` (tag `v0.1.0`); nothing below is
released. Releasing the QA module is a separate step that the maintainers approve; see [QA_RELEASE_CHECKLIST.md](QA_RELEASE_CHECKLIST.md).

## [Unreleased]

### Added

- **`agent_harness.qa`**, an optional QA module (Python 3.11 or newer; the `qa` extra is empty and the floor is enforced
  by an import guard). Standard library and harness core only; `import agent_harness` never imports it. See [qa.md](qa.md).
  - Neutral records, ports, typed errors, schema identifiers and `canon-v1` hashing, mirroring runtime-contracts `qa/v1`
    independently (`canon-v1:<hex>` equals `rcv1:<hex>`), with the 97 frozen canonical-hash vectors under `tests/qa/golden/`.
  - A deterministic assertion verifier that loads frozen assertions by pinned hash and recomputes the hash, a pure run
    classifier, and a read-only spec store.
  - A bounded single-run executor with an explicit state machine, an append-only receipt chain, a fail-closed `ui_action`
    policy, idempotency-aware retries, deadlines and budgets.
  - Deterministic recovery (semantic locator, checkpoint and fixture modes) and model-candidate recovery behind an injected
    client, off by default. A run that used a recovered locator never ends `PASS` or `PRODUCT_FAIL`.
  - An egress policy (default deny, metadata hard-deny, resolve-then-pin, per-hop redirect checks), a fail-closed redactor,
    full-width digest checks and an evidence integrity verifier.
- `tests/qa/`: the QA test suite, including mutation tests for the immutable-assertion rule, import-boundary and
  licence-boundary checks, and the Python floor tests.
- `.github/workflows/qa.yml`: runs `tests/qa` on Python 3.11 and 3.12 with `QA_STRICT_ENV=1` and fails if no QA test is
  collected or any is skipped (`tests/qa/ci_gate.py`, allow-list `tests/qa/ci_skip_allowlist.txt`, currently empty); a
  Python 3.10 job checks that the core imports and that `import agent_harness.qa` fails with the guard message. The
  existing `tests` workflow is unchanged.
- Documentation: [qa.md](qa.md) (module map, ports and what a deployment supplies, the immutable-assertion rule,
  recovery, policy, egress and redaction, evidence integrity, what the module does not do, the compatibility statement and
  the licence statement) and [QA_RELEASE_CHECKLIST.md](QA_RELEASE_CHECKLIST.md).

### Changed

- `README.md`, `STATUS.md` and `docs/architecture.md` no longer describe the agent loop as a single-step stub: that
  describes `Agent.run`; `agent_harness.loop.run_loop` is a multi-turn tool-calling loop with hard limits. They also
  describe the Context Runtime integration as it is documented in `agent_harness/context.py`.
- No behaviour of the existing modules changed, and no `LICENSE` file changed. The harness remains AGPL-3.0-or-later.
