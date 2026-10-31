# QA module (`agent_harness.qa`)

`agent_harness.qa` is a neutral, standard-library-only module for running **one** reviewed UI test spec in **one**
isolated session and recording the result as a tamper-evident record. It provides the records, the ports (protocols), the
deterministic verifier and classifier, the run state machine, the fail-closed action policy, bounded recovery, the egress
policy, redaction and evidence integrity checks. It does not drive a browser, call a model, open a socket or store
anything durably: those parts are supplied through the ports (see [What a deployment supplies](#what-a-deployment-supplies)).

It is optional. `import agent_harness` never imports it.

## Requirements and install

* **Python 3.11 or newer.** The harness core keeps supporting Python 3.10. On an older interpreter
  `import agent_harness.qa` raises an `ImportError` whose message says `agent_harness.qa requires Python 3.11 or newer`,
  before any submodule is loaded (`agent_harness/qa/__init__.py`).
* **The `qa` extra is empty.** `pip install -e ".[qa]"` installs the same package as `pip install -e .`; the extra
  declares no dependency and the Python floor is enforced by the import guard, not by package metadata. The core keeps
  `dependencies = []`.
* **No other package is imported.** The module imports the standard library and harness core only. It does not import
  `runtime_contracts`, `context_runtime`, `redevops_qa` or Playwright (checked by `tests/qa/test_boundaries.py`).

```bash
pip install -e ".[dev,qa]"                       # Python 3.11+
python -m pytest tests/qa -q                     # needs nothing installed beyond pytest
```

## Module map

| Area | Module (`agent_harness/qa/`) | Role |
|---|---|---|
| Records and hashing | `__init__.py` | the Python 3.11 import guard; exports `NEUTRAL_API`, `canon_hash` and the submodules |
| Records and hashing | `records.py` | frozen plain-data records with strict `to_json` / `from_json`: `TestSpec` (with `Step`, `AssertionSpec`, `FixtureRef`), `Observation`, `ObservedValue`, `ActionReceipt`, `AssertionResult`, `FaultEvent`, `Checkpoint`, `ReviewItem`, `RunManifest`, `PolicyDecisionRecord`, `LocatorCandidate` and the values and enums they use |
| Records and hashing | `canon.py` | `canon-v1` canonical JSON and SHA-256; floats are refused |
| Records and hashing | `ports.py` | the structural interfaces: `NeutralQAAdapter`, `NeutralCheckAdapter`, `Redactor`, `BlobSink`, `Recoverer`, `ActionPolicy`, `Clock` |
| Records and hashing | `errors.py` | typed errors with stable `code` strings (integrity and validation errors; one adapter error per non-`OK` result code) |
| Records and hashing | `schemas.py` | `NEUTRAL_API = "qa-neutral/1"`, `CONTRACT_ID = "qa/v1"`, `SUPPORTED_SCHEMAS` (an unknown schema id or version is refused) |
| Verification | `verifier.py` | loads frozen assertions by pinned hash, recomputes the hash, evaluates observed values against the frozen predicate |
| Verification | `classify.py` | `classify(facts)`: a pure function from run facts to `(outcome, reason)` with explicit precedence |
| Verification | `specstore.py` | read-only spec store and check-program registry ports, with in-memory and directory reference implementations |
| Verification | `recovery_types.py` | the only types a recoverer can return (strict `LocatorCandidate` lists); no type can carry an expected value |
| Run execution | `runner.py` | `QARunner.run`: one spec, one session, one `RunManifest` recorded in the evidence stream |
| Run execution | `runstate.py` | the run state machine and its one explicit transition table |
| Run execution | `receipts.py` | `ActionReceipt` construction and the append-only receipt chain |
| Run execution | `policy.py` | `DefaultActionPolicy`, the fail-closed `ui_action` policy |
| Run execution | `checkpoint.py` | in-run checkpoints (step index, state hash, optional fixture snapshot id; no storage state) |
| Run execution | `cleanup.py` | verifiable `close_session` cleanup |
| Run execution | `idempotency.py` | idempotency keys and the retry decision |
| Run execution | `limits.py` | deadlines and bounded budgets measured through the injected clock |
| Run execution | `hooks.py` | small ports the runner is injected with: `EvidenceStream`, `PostconditionChecker`, `PrepareChecks`, `RecoveryHook` |
| Run execution | `run_errors.py` | typed errors of the run layer |
| Run execution | `clock.py` | the production `Clock`; the only place real time is read |
| Recovery | `recovery.py` | deterministic recovery: fault detection, recovery planning, semantic locator selection, kill switch, checkpoint and fixture modes |
| Recovery | `recovery_model.py` | model-candidate recovery behind an injected `CandidateModelClient`; off by default |
| Egress and evidence | `egress.py` | the egress policy (pure decisions, injected `Resolver`) and `EgressGuard` |
| Egress and evidence | `redaction.py` | `PatternRedactor`, a deterministic fail-closed redactor |
| Egress and evidence | `blobs.py` | content-addressed blobs, full 64-hex digests only, size bounds, retention data |
| Egress and evidence | `vault.py` | `ArtifactVault` (redact before storing) and `EvidenceGuard` (redact and bound every evidence event) |
| Egress and evidence | `integrity.py` | `verify_run_evidence`: one check over the receipt chain, the event chain, their cross-references and the artifact hashes |

Tests live in `tests/qa/`. They run on the Python 3.11 and 3.12 CI job; on 3.10 only `test_python_floor.py` is collected.

## What a deployment supplies

The module defines the ports and tests them against scripted in-memory implementations. A deployment (for example the
`redevops-qa` product package, a separate package that depends on the pinned releases of this one) supplies:

| Needed | Port or hook | Notes |
|---|---|---|
| A real browser adapter | `NeutralQAAdapter` (and `NeutralCheckAdapter` for non-UI checks) | opens a fresh session, locates, acts, observes and closes; `assert_` and `run_check` take a subject only and return an `ObservedValue`, with no expected value in and no verdict out |
| A postcondition checker | `hooks.PostconditionChecker` | evaluates a step's declared `expected_transition`. The default evaluates nothing, which blocks every retry of a side-effecting step |
| Prepare checks | `hooks.PrepareChecks` | environment health, spec preconditions, fixture hashes. Unset means "not verified", and an unverified attestation cannot produce `PASS` |
| A resolver and egress enforcement | `egress.Resolver`; the connection point | `egress.py` decides and never connects. The component that opens connections (a gateway or browser proxy) must call it and connect only to the pinned address. `NoResolver` is the default and refuses |
| An evidence store | `BlobSink`, `hooks.EvidenceStream` | the module ships `MemoryBlobSink` and `MemoryEvidenceStream`. Durable storage and retention deletion belong to the deployment |
| A model client | `recovery_model.CandidateModelClient` | needed only when model-candidate recovery is enabled. The module holds no SDK, HTTP client or key |
| A spec store | `specstore.SpecStore`, `specstore.ProgramRegistry` | read-only by construction; the writer is a different component |
| Policy data | `DefaultActionPolicy` configuration | allowed action kinds, origins and local sink origins. The default allows nothing |

## The immutable-assertion rule

A test's assertions are frozen before the run and nothing in the run can change what counts as a pass. The rule is
enforced by structure, with a mutation test for each mechanism (`tests/qa/test_immutable_assertions.py`: a copy of the
module with the mechanism removed must fail its probe):

1. **Type separation.** An `AssertionSpec` (with the expected value and predicate) is a different type from an
   `ObservedValue` (what the page showed). Recovery-side types cannot hold either.
2. **Hash pinning.** The assertion set is pinned by hash before the run.
3. **The verifier loads by hash.** `verifier.load_frozen_assertions` fetches the assertions from the spec store by the
   pinned hash, recomputes the hash with `canon-v1` and refuses on any difference. A `FrozenAssertionSet` cannot exist
   unless its content hashes to its pin, and `DeterministicVerifier` accepts nothing else. The assertions are loaded
   again at verification time, so a spec edited in between is refused.
4. **Capability partition.** A recoverer receives a `RecoveryContext` (the frozen step and the current observation; no
   assertion field) and may return only strict `LocatorCandidate` values. `recovery.py` and `recovery_types.py` import
   neither the verifier, the classifier nor the spec store.
5. **The adapter never sees the expected value.** `assert_` takes a subject only.
6. **The spec store is read-only.** It has one operation, a read.
7. **No assertion healing.** There is no code path that rewrites an assertion or a predicate.
8. **Changes only through review.** A new assertion set must be in the approved lineage.
9. **Stop rule.** Any integrity refusal halts the batch; every later call is refused.

## Run outcomes and recovery

`classify(facts)` is a pure function with a first-match-wins precedence: integrity violation
(`INCONCLUSIVE(ASSERTION_INTEGRITY)`, halt the batch), environment failure, test defect, product failure, any other
non-pass evidence (`INCONCLUSIVE`), then `PASS`. Every attestation the runner has not made is `UNKNOWN` and blocks
`PASS` and `PRODUCT_FAIL`. A `PASS` requires every declared assertion to have been executed and passed (none skipped),
the declared, executed and skipped lists to be consistent, and the checks for preconditions, fixtures, environment,
cleanup and policy to hold. A `PASS` record that holds a failed assertion is an integrity error.

**Recovery modes** are `RETRY_SAME`, `SEMANTIC_LOCATOR`, `MODEL_CANDIDATE`, `RESET_FIXTURE`, `CHECKPOINT_RESUME` and
`ESCALATE_REVIEW`. Recovery is off unless the run enables it, the kill switch is not engaged, the step's
`permitted_recovery` lists the mode, the caps and the spec budget allow it, and the idempotency guard says the work may
be done again (`UNKNOWN` idempotency is no guard; a step that cannot be shown safe is never replayed blindly).

Recovery cannot produce a `PASS`:

* It only changes how a step finds its target or which execution path it takes, never what is expected. It receives no
  assertion, and its output type cannot carry one.
* A fault is recorded in the evidence stream as a `FaultDetection` before any recovery decision. The recovery record
  references it by hash, and its result is judged by the step's own postcondition check, not inferred from the run.
* A candidate locator (from a deterministic recoverer or from a model) is accepted only if it is the single control
  whose role and accessible name equal the authored ones. The first match is never taken.
* A run that used a recovered locator (`SEMANTIC_LOCATOR` or `MODEL_CANDIDATE`) never ends `PASS` or `PRODUCT_FAIL`. It
  ends `INCONCLUSIVE` with a review item.
* `PASS` is reachable from `VERIFY` only, and only through the verifier's comparison of observed values with the frozen
  assertions. The model, when enabled, is asked for candidate locators only and never sees an assertion or a verdict.

Model-candidate recovery is off by default (`RunConfig.model_recovery_enabled = False`), is bounded by hard caps on model
calls, tokens and candidates per step and per run, and is tested with a fake client only.

## Policy: default deny

`DefaultActionPolicy` denies everything that is not explicitly allowed. The decision is made from the frozen spec and
operator configuration, fixed before the first adapter call, and never from the page.

* Action kinds must be in the operator's `allowed_action_kinds` (default: none).
* Side-effect classes must be listed in the reviewed spec. `WRITES_EXTERNAL`, `NOTIFIES_HUMAN` and `SPENDS_MONEY` also
  need a target that is one of the operator's local sink origins.
* The target origin must be in the operator's `allowed_origins` and comes from the reviewed spec.
* An unknown side-effect class, an unknown idempotency class, a missing policy, a provider error or timeout, and a step
  that is not exactly the reviewed one are all denials.
* Every decision, allow or deny, is a `PolicyDecisionRecord` whose hash goes into `ActionReceipt.policy_decision_ref`.
  An action without one is refused (`ActionBlockedByPolicy`).

## Egress and redaction

Both are opt-in in `QARunner` (an `EgressGuard` and an `EvidenceGuard`); without them the default behaviour is unchanged.

**Egress** (`egress.py`) is pure and deterministic. The allow-list is operator data (exact `scheme://host:port`
entries; a wildcard is only a subdomain wildcard of a named public domain); with no entry every request is denied.
Loopback, RFC 1918 and other non-public targets are refused unless an entry names the exact networks and ports.
Metadata and link-local addresses (`169.254.0.0/16`, `fe80::/10`, `fd00:ec2::254` and the cloud metadata names) are
refused whatever the allow-list says, including when hidden in IPv4-mapped, NAT64, 6to4 or Teredo forms. The resolver is
called once per decision; the decision is made on all returned addresses (one bad address denies the set); an allowed
decision pins the address a connector must use. Every redirect hop is decided again, and the same rules apply to
navigation, subresources and WebSockets. Numeric host spellings (decimal, octal, hex, short forms) are parsed to the
address a browser would reach. The module never opens a socket; enforcement at the connection point is the
deployment's (the `redevops-qa` product package provides it).

**Redaction** (`redaction.py`, `vault.py`) is deterministic and fails closed. `PatternRedactor` removes canary secrets
(literal and common encodings), key blocks (PEM), credential headers, URL userinfo, JWTs, provider key shapes,
secret-valued URL query parameters, secret-named assignments, bearer and basic tokens and high-entropy tokens, plus
email addresses, phone numbers and card numbers by default. Structured data is redacted by key name first. Input it
cannot handle raises `RedactionFailed`, and nothing is stored in that case. Binary media is not redacted and is not
claimed to be: it is refused unless the operator lists its media type as an acknowledged exception. `EvidenceGuard`
redacts every evidence event before it is hashed into the chain.

## Evidence integrity

* Receipts form an append-only chain; each link commits to the previous one with `canon-v1`. Altering, dropping, adding
  or reordering a receipt is detected.
* Evidence events are chained the same way; `RunStarted` is appended before `open_session`, and the closing `RunSealed` event
  names the receipt chain head and the manifest hash.
* Evidence references are full `sha256:<64 hex>` digests. A shorter or differently spelled digest (including the 16-hex
  form of `TraceRecorder`) is refused (`blobs.py`).
* `integrity.verify_run_evidence` checks the receipt chain, the event chain, how they refer to each other and the
  artifact hashes, and returns stable finding codes instead of raising, so every problem can be shown at once.
  `require_valid` turns a report with findings into an integrity error.

## What the module does not do

* **No durable scheduling.** `QARunner.run` executes one spec and returns. There is no queue, no workflow engine, no
  dependency resolution and no state that outlives the call except the evidence stream it was given.
* **No model calls of its own.** The only model use is the optional, injected `CandidateModelClient`.
* **No network.** Nothing in the module opens a socket or resolves a name. The test suite fails any non-loopback
  connection attempt.
* **No browser.** There is no Playwright or other browser dependency. A browser adapter is supplied through the ports.
* **No durable storage and no retention deletion.** Retention periods are recorded in the artifact manifest; deleting
  by them is the deployment's.
* **No plan record.** The module receives a `TestSpec`, not a plan; plan-level checks happen in the component that reads
  the plan. Inside the module the coverage lists must be consistent (declared equals executed plus
  skipped, disjoint) and a `PASS` requires no skipped assertion.

## Compatibility with runtime-contracts `qa/v1`

The contracts package (`runtime-contracts`, schema family `qa/v1`) is canonical. The records, ports and `canon-v1` in this
module are an **independent mirror** of it: nothing is imported in either direction and nothing is copied between the
repositories. On any conflict, contracts wins and this module is corrected.

| Item | Value |
|---|---|
| Contract id | `qa/v1` (`schemas.CONTRACT_ID`); schema version `0.1` |
| Neutral interface id | `qa-neutral/1` (`schemas.NEUTRAL_API`) |
| Hash spelling | the neutral `canon-v1:<64 hex>` is the contracts `rcv1:<64 hex>` with the same digits; `canon.to_contract_ref`, `canon.from_contract_ref` and `canon.same_digest` map the prefix. References embedded in records use the contracts `rcv1:` spelling |
| Contracts commit mirrored | `13e00f23da74782af091aec607c47358e9814fc6` |
| Canonical hash vectors | 97 vectors in `tests/qa/golden/qa_canon_vectors.json`, sha256 `7241c570746a63e75431591b88e42562f6604fc2a8586474a267eedb51908a13`, reproduced byte for byte by `tests/qa/test_canon.py` |
| Field inventory | `tests/qa/golden/qa.inventory.json`, sha256 `8ebd0be1488233966ba66dc2cbb7c46b42ec3973a246f405e905130a4d28ba57`, checked field by field by `tests/qa/test_records.py` |

This repository's suite checks the neutral side against those frozen inputs and needs no contracts install. It cannot
show that the two sides agree on anything the frozen inputs do not contain, and it does not try.

The check that the two sides are one contract lives in the `redevops-qa` product package (its conformance suite) and is the
**release-critical compatibility gate**. It runs against the exact pinned versions of both sides, installed from those
pins, and:

* fails when either dependency is missing or does not import (it never skips), and requires a non-zero collected-test
  count and zero skipped tests;
* checks the seven parity areas: `canon-v1` against `rcv1` for every vector in both directions, identity-field and
  custody-field sets, enums, protocol method and parameter names, refusal reason codes, and the contracts to neutral to
  contracts round trip;
* writes one machine-readable `compat-receipt/1` (`result` is `PASS` only if every check passed, the collected count is
  above zero, the skipped count is zero and both dependencies are available). The receipt is stale as soon as any pinned
  version differs.

No receipt is produced by this repository. A release needs a current receipt for the exact set of pins being released
(see [QA_RELEASE_CHECKLIST.md](QA_RELEASE_CHECKLIST.md)).

## Licence

`agent-harness` remains plain AGPL-3.0-or-later (`LICENSE`; `pyproject.toml` declares `AGPL-3.0-or-later`). Every file
in `agent_harness/qa/` carries `SPDX-License-Identifier: AGPL-3.0-or-later`. The QA module imports no
Commons-Clause-licensed code: it imports the standard library and harness core only, and `tests/qa/test_boundaries.py`
fails if a QA file imports any other package. No `LICENSE` file is changed by the QA work. Whether the module is ever
relicensed or dual licensed is a separate decision that this documentation does not make.
