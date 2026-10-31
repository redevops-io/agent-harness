# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared test helpers: golden data loaders and the source-mutation harness."""
import json
import sys
import types
from pathlib import Path

GOLDEN = Path(__file__).with_name("golden")
QA_SRC = Path(__file__).resolve().parents[2] / "agent_harness" / "qa"

#: the runtime-contracts commit these golden files were taken from (the ``qa/v1`` schema family)
CONTRACTS_QA_V1_COMMIT = "13e00f23da74782af091aec607c47358e9814fc6"
#: sha256 of the golden data files, as documented in docs/qa.md (compatibility statement)
CANON_VECTORS_SHA256 = "7241c570746a63e75431591b88e42562f6604fc2a8586474a267eedb51908a13"
INVENTORY_SHA256 = "8ebd0be1488233966ba66dc2cbb7c46b42ec3973a246f405e905130a4d28ba57"
GOLDEN_SHA256 = {
    "qa_canon_vectors.json": CANON_VECTORS_SHA256,
    "qa.inventory.json": INVENTORY_SHA256,
}
CANON_VECTOR_COUNT = 97


def load_vectors():
    return json.loads((GOLDEN / "qa_canon_vectors.json").read_text(encoding="utf-8"))


def load_inventory():
    return json.loads((GOLDEN / "qa.inventory.json").read_text(encoding="utf-8"))


_counter = [0]


def load_mutant(filename, replacements):
    """Compile ``agent_harness/qa/<filename>`` with exact-text replacements applied and return it as a module.

    Every anchor must occur exactly once, so a mutation test fails loudly if the code it targets changes shape.
    The caller removes ``module.__name__`` from ``sys.modules`` when done (the ``mutant`` fixture does).
    """
    path = QA_SRC / filename
    src = path.read_text(encoding="utf-8")
    for old, new in replacements:
        assert src.count(old) == 1, f"mutation anchor must occur exactly once in {filename}: {old!r}"
        src = src.replace(old, new)
    _counter[0] += 1
    name = f"agent_harness.qa._mutant_{_counter[0]}"
    mod = types.ModuleType(name)
    mod.__package__ = "agent_harness.qa"
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod
