# SPDX-License-Identifier: AGPL-3.0-or-later
"""agent_harness.qa: neutral records, ports and ``canon-v1`` hashing for the QA module (optional ``qa`` extra).

The QA module needs Python 3.11 or newer; the harness core keeps supporting 3.10. ``agent_harness/__init__.py`` never
imports this package, so ``import agent_harness`` is unaffected. Importing ``agent_harness.qa`` on an older
interpreter fails here, with a clear message, before any submodule is loaded.

Standard library and harness core only: nothing is imported from ``runtime_contracts``, ``context_runtime``,
``redevops_qa`` or Playwright. Contracts ``qa/v1`` is canonical; this package is its non-normative mirror.
"""
from __future__ import annotations

import sys

MINIMUM_PYTHON = (3, 11)

if sys.version_info < MINIMUM_PYTHON:   # the guard: keep this syntax 3.10-compatible and run it before any import
    raise ImportError(
        "agent_harness.qa requires Python %d.%d or newer (this is Python %d.%d.%d). The harness core still supports "
        "Python 3.10; run the QA module on 3.11 or newer."
        % (MINIMUM_PYTHON + tuple(sys.version_info[:3]))
    )

from . import canon, errors, ports, records, schemas                                      # noqa: E402
from .canon import canon_hash, canonical_bytes, canonical_text                            # noqa: E402
from .schemas import CONTRACT_ID, NEUTRAL_API, SCHEMA_VERSION, SUPPORTED_SCHEMAS          # noqa: E402

__all__ = [
    "MINIMUM_PYTHON", "NEUTRAL_API", "CONTRACT_ID", "SCHEMA_VERSION", "SUPPORTED_SCHEMAS",
    "canon", "errors", "ports", "records", "schemas",
    "canon_hash", "canonical_bytes", "canonical_text",
]
