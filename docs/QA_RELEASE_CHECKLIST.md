# QA module release checklist

Status: **released as `v0.2.0` on 2026-10-31**, with maintainer approval, from the QA release branch built on the base commit below. This
checklist is kept as the record of what was verified.

* **The tag `v0.2.0` was authorised by the maintainer on 2026-10-31.** Nothing in this repository or its workflows
  creates or triggers a tag.
* The existing `v0.1.0` has two duplicate GitHub releases. The `v0.2.0` tag gets exactly one.

## Release-engineering rule: the release is built from the `6b8ba3a` base

The QA module is built from the existing base commit `6b8ba3a`, with the QA work added on top. **Unrelated commits on
`main` are not included.** In particular the later `main` commit that touches `bench/` and adds `netpolicy.py`,
`supervisor.py` and `docs/execution-supervision.md` is not in this tree, and the documentation here does not describe
those files.

## Preconditions (all must hold; each is evidence, not an assertion)

1. **Maintainer approval** for the tag and the release, recorded before the tag is created.
2. **A current `compat-receipt/1` from the conformance suite of the `redevops-qa` product package for the exact pins being released**
   (contracts tag and commit, this repository's tag and commit, the product package's commit). It must show `result = PASS`,
   all seven checks passing, `collected_tests > 0`, `skipped_tests = 0`, and a verified `receipt_hash`. The receipt is
   stale as soon as any pinned version differs; a missing, stale or failing receipt blocks the release. This repository
   cannot produce it (see [qa.md](qa.md#compatibility-with-runtime-contracts-qav1)).
3. **CI green on Python 3.10, 3.11 and 3.12:**
   * `tests` workflow (core suite on 3.10 and 3.12), unchanged;
   * `qa` workflow, job `qa-suite` (`tests/qa` on 3.11 and 3.12, `QA_STRICT_ENV=1`, non-zero collection, no skips outside
     `tests/qa/ci_skip_allowlist.txt`, which is empty);
   * `qa` workflow, job `python-3-10` (`import agent_harness` works and `import agent_harness.qa` fails with the guard
     message);
   * `docs-links` and `secret-scan` workflows.
4. **Boundary tests green:** `tests/qa/test_boundaries.py` (the QA module imports the standard library and harness
   core only; across all of `agent_harness/` the only imports of `runtime_contracts`, `context_runtime`, `redevops_qa` or
   Playwright are the two pre-existing lazy `context_runtime` imports listed below, and none is in `agent_harness/qa/`;
   every QA file carries the `AGPL-3.0-or-later` SPDX header; core never imports `qa`).
5. **No-secrets scan clean:** the `secret-scan` workflow (gitleaks, redacted) passes on the full history of the release
   branch, and a scrubbed-environment run (`QA_STRICT_ENV=1`, temporary `HOME`, `APPROVAL_SECRET` unset) passes.
6. **Compatibility with the base** (run from the release branch with the base available as `6b8ba3a`):

   ```bash
   # every changed path is inside the allow-list: prints nothing
   git diff --name-only 6b8ba3a HEAD | grep -v -E '^(agent_harness/qa/|tests/qa/|docs/|STATUS\.md$|README\.md$|pyproject\.toml$|\.github/workflows/)'
   # every harness file that existed at the base is byte-identical, except the allow-listed ones: prints nothing
   git diff --name-only 6b8ba3a HEAD -- $(git ls-tree -r --name-only 6b8ba3a | grep -v -E '^(docs/|STATUS\.md$|README\.md$|pyproject\.toml$)')
   # the existing suite gives the same counts as at the base (the QA tests are in tests/qa)
   python -m pytest -q --ignore=tests/qa
   ```

   The existing suite, measured at the base before the QA work and after it, is 189 passed and 5 skipped (the skips are the
   optional `harness2` and `context_runtime` conformance tests and the Docker-dependent selfcheck tests). The only change
   to `pyproject.toml` before release is the empty `qa` extra.
7. **Documentation** reads correctly against this tree: `docs/qa.md`, the corrected loop statements in `README.md`,
   `STATUS.md` and `docs/architecture.md`, and the changelog entry. `docs-links` is green.
8. **Version bump at release time only.** On approval, and not before: set `version` in `pyproject.toml` to the approved
   target, move the `Unreleased` changelog entry under that version and its date, then tag. None of this is done in
   preparation.
9. **Downstream pins.** Context Runtime pins this repository by tag. Its pin and the archive sha256 recorded for the tag's
   commit are updated after the tag, in that repository.
10. **Licence.** Nothing here changes the licence terms. Any change to the licensing of the QA module, and any
    release, is decided separately from this checklist.

## Note for the licence review: existing lazy imports of `context_runtime`

The harness remains plain AGPL and the QA module imports no Commons-Clause-licensed code. Outside the QA module there
are two pre-existing, lazy (inside a function, guarded by `ImportError`) imports of Context Runtime. They are not part of
the QA work and were not changed:

| File | Line | Import | When it runs |
|---|---|---|---|
| `agent_harness/bench/selfcheck.py` | 121 | `import context_runtime  # noqa: F401` | `check_pipeline()`, after the Docker checks, to report whether the default context manager is installed |
| `agent_harness/bench/jobbench.py` | 80 | `from context_runtime.integrations.agent_harness import ContextRuntimeManager` | `make_context_manager("context-runtime", ...)`; a missing package raises a `RuntimeError` that names the `basic` opt-out |

Line numbers are for this tree (`6b8ba3a` plus the QA work; no QA change touches these files) and move if the tree
is rebased onto `main`. The docstring of `agent_harness/context.py` says of `ContextRuntimeManager`: "This package never
imports it; the dependency points from context-runtime to here". The two lazy imports above are the exception to that
sentence and should be read with it in the review; the docstring was not changed. Neither import is reachable from
`agent_harness.qa`.
