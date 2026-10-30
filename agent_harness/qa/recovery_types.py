# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recovery-side types that cannot express an assertion (ADR section 6 rule 4, the capability partition).

A recoverer is shown a :class:`~agent_harness.qa.ports.RecoveryContext` (the frozen step and the current observation;
no assertion field, and the types reachable from it hold no predicate, no expected value and no verdict) and may return
exactly one thing: a list of strict :class:`~agent_harness.qa.records.LocatorCandidate` (``strategy`` plus role, name,
test id, label; no ``extras``). :func:`parse_recovery_output` is the single gate between a recoverer's output and the
runner: a candidate is parsed into that type or the whole output is rejected with ``RECOVERY_PAYLOAD_REJECTED``
(``INCONCLUSIVE(ASSERTION_INTEGRITY)``). There is no field, no code path and no type on which an expected value could
be carried back, and this module imports neither the verifier, the classifier nor the spec store.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Tuple

from .errors import RecoveryPayloadRejected
from .ports import Recoverer, RecoveryContext
from .records import LocatorCandidate

#: more candidates than this from one call is rejected outright (a bound, not a budget: budgets are H3's)
MAX_CANDIDATES_PER_CALL = 16


def parse_recovery_output(raw: Any) -> Tuple[LocatorCandidate, ...]:
    """Parse whatever a recoverer returned into a tuple of strict ``LocatorCandidate`` or reject all of it.

    Accepts a list or tuple whose items are ``LocatorCandidate`` instances or JSON objects. Rejected, as
    ``RecoveryPayloadRejected``: anything that is not a list or tuple (a single object, a mapping, a string); an item
    that is any other record (an assertion, a spec, a result); a subclass of ``LocatorCandidate``; a candidate carrying
    unknown keys such as ``expected``; more than :data:`MAX_CANDIDATES_PER_CALL` items.
    """
    if not isinstance(raw, (list, tuple)):
        raise RecoveryPayloadRejected(f"recovery output must be a list of locator candidates, got {type(raw).__name__}")
    if len(raw) > MAX_CANDIDATES_PER_CALL:
        raise RecoveryPayloadRejected(f"recovery output has more than {MAX_CANDIDATES_PER_CALL} candidates")
    out = []
    for i, item in enumerate(raw):
        if type(item) is LocatorCandidate:
            out.append(LocatorCandidate.from_json(item.to_json()))      # re-validated through the strict path
        elif isinstance(item, Mapping):
            out.append(LocatorCandidate.from_json(item))                # strict: unknown keys are rejected
        else:
            raise RecoveryPayloadRejected(f"recovery output item {i} is a {type(item).__name__}, not a locator candidate")
    return tuple(out)


def propose_safely(recoverer: Recoverer, ctx: RecoveryContext) -> Tuple[LocatorCandidate, ...]:
    """Ask a recoverer for candidates and return only what survives :func:`parse_recovery_output`."""
    if type(ctx) is not RecoveryContext:
        raise RecoveryPayloadRejected("a recoverer is shown a RecoveryContext and nothing else")
    return parse_recovery_output(recoverer.propose(ctx))


__all__ = ["MAX_CANDIDATES_PER_CALL", "parse_recovery_output", "propose_safely"]
