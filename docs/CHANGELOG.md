# Changelog

Notable changes to `agent-harness`. The package version in `pyproject.toml` is `0.2.0` (tag `v0.2.0`; the previous tag is `v0.1.0`).
The 0.2.0 release was checked against [QA_RELEASE_CHECKLIST.md](QA_RELEASE_CHECKLIST.md).

## [0.2.0] - 2026-10-31

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
- `agent_harness.qa` now follows the contracts `qa/v1` text where it had derived its own definition (compatibility
  statement in [qa.md](qa.md)): the digest of an action value is `rcv1` of the value's canonical JSON text alone (a
  `SecretRef` digests as its fingerprint); each set-valued field is ordered by its own rule (element key, canonical text
  or value); the integrity refusals have distinct codes (`ASSERTION_PIN_MALFORMED`, `ASSERTION_PIN_NOT_FOUND`,
  `ASSERTION_PROGRAM_UNREGISTERED`, `ASSERTION_PROGRAM_MISMATCH`, `BATCH_HALTED`) and an unknown code classifies as an
  integrity refusal; `FaultRecovery` carries an optional `reason_code` from a closed vocabulary; `Check.detail` is not
  hashed, `Check.evidence_refs` is sorted when hashed, an `EvidenceRef` and a `SecretRef` hash with the contracts'
  defaults; and `ACTION_TIMEOUT` is recoverable (recovery stays guarded). Hashes of records that carry those values
  change accordingly. The golden data under `tests/qa/golden/` gains the added vectors, the clarification fixtures, the
  field inventory and the JSON Schema of the contracts commit named in the compatibility statement.
- No behaviour of the existing modules changed, and no `LICENSE` file changed. The harness remains AGPL-3.0-or-later.
