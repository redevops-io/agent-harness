# SPDX-License-Identifier: AGPL-3.0-or-later
"""A MATCHES pattern that can run without bound on observed text is refused when the assertion set is loaded.

The observed text comes from the system under test and the verifier has no clock: ``(a+)+$`` on 24 ``a`` and a ``b``
already takes more than half a second and doubles with every character, and an observed value may be 16 KB."""
import copy

import pytest

from agent_harness.qa import records as R, verifier as V
from agent_harness.qa.errors import SpecInvalid

from ._fx_verify import REF_DOCS

UNBOUNDED = [r"(a+)+$", r"(.*a)*b", r"(a|b+)*c", r"(?:\w+\s?)+$", r"((ab)*)+x", r"(a*)*", r"(a+?)+?b", r"(?=(a+)+)x"]
FINE = [r"^Saved$", r"\d+ items?", r"^[A-Z]{2}-\d{4,}$", r"(ab){1,8}c+", r"^(GET|POST) /api/\w+$", r".*total: \d+\.\d{2}.*",
        r"(?:a{1,3})+b", r"colou?r", r"^\s*\S+@\S+\.\S+\s*$"]


def _matches(pattern):
    doc = copy.deepcopy(next(d for d in REF_DOCS if d["predicate"]["op"] in ("EQUALS", "CONTAINS", "MATCHES")))
    doc["predicate"] = {"op": "MATCHES", "expected": pattern}
    return R.AssertionSpec.from_json(doc)


@pytest.mark.parametrize("pattern", UNBOUNDED)
def test_a_pattern_with_nested_unbounded_repetition_is_refused_before_anything_runs(pattern):
    assert V.pattern_can_backtrack_without_bound(pattern)
    with pytest.raises(SpecInvalid, match="nests an unbounded repetition"):
        V.validate_assertion(_matches(pattern))


@pytest.mark.parametrize("pattern", FINE)
def test_ordinary_patterns_are_still_accepted(pattern):
    assert not V.pattern_can_backtrack_without_bound(pattern)
    V.validate_assertion(_matches(pattern))


def test_every_pattern_in_the_reference_vectors_is_accepted():
    import json
    from ._helpers import QA_SRC  # noqa: F401  (the vectors live next to the package)
    from . import _fx_verify as fx
    seen = [d["predicate"]["expected"] for docs in (fx.REF_DOCS, fx.SCRIPT_DOCS) for d in docs if d["predicate"]["op"] == "MATCHES"]
    assert all(not V.pattern_can_backtrack_without_bound(p) for p in seen), json.dumps(seen)
