"""Harness adapters. Each module exposes ``NAME`` and ``generate(experiment, cell, workers)``; it runs (or reads) the
rollouts of ONE harness for ONE cell and registers every finished task on ``experiment.board``.

``ours`` lives here; ``opencode`` and ``harness2`` are separate modules so each adapted third-party harness can be
developed, reviewed and merged on its own branch."""
from __future__ import annotations

import importlib
from types import ModuleType

KNOWN = ("ours", "opencode", "harness2")


def load(name: str) -> ModuleType:
    if name not in KNOWN:
        raise ValueError(f"unknown harness {name!r}; known: {', '.join(KNOWN)}")
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(f".{name}"):
            raise RuntimeError(f"the {name!r} harness adapter is not present in this checkout (it lives on its own branch)") from e
        raise
