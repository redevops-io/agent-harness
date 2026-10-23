"""Harness adapters. Each module exposes ``NAME`` and ``generate(experiment, cell, workers)``; it runs (or reads) the
rollouts of ONE harness for ONE cell and registers every finished task on ``experiment.board``.

``ours`` lives here; ``opencode`` and ``harness2`` are separate modules so each adapted third-party harness can be
developed, reviewed and merged on its own branch."""
from __future__ import annotations

import importlib
import re
from types import ModuleType, SimpleNamespace

KNOWN = ("ours", "ours-nocr", "ours-cr", "ours-cr-pin", "ours-h2-cr", "ours-h2", "ours-h2-nocr", "opencode", "harness2")
PASSIVE = ("h2-base",)      # arms recorded as a by-product of another harness's run; never generated on their own


_REPEAT = re.compile(r"^(?P<base>.+)-r(?P<k>\d+)$")


def load(name: str) -> ModuleType:
    """A repeat of an arm is the same arm under another name: ``ours-cr-r2`` runs ``ours-cr`` into its own results (so the repeats
    are separate rollouts on the scoreboard). Only arms whose ``generate`` accepts ``name=`` can be repeated."""
    m = _REPEAT.match(name)
    if name not in KNOWN and m and m.group("base") in KNOWN:
        base = load(m.group("base"))
        if "name" not in base.generate.__code__.co_varnames:
            raise ValueError(f"{m.group('base')!r} cannot be repeated (its generate() takes no name=)")
        return SimpleNamespace(NAME=name, generate=lambda exp, cell, workers=1, **kw: base.generate(exp, cell, workers, name=name, **kw),
                               diagnostics=getattr(base, "diagnostics", None))
    if name not in KNOWN:
        raise ValueError(f"unknown harness {name!r}; known: {', '.join(KNOWN)}")
    try:
        return importlib.import_module(f"{__name__}.{name.replace('-', '_')}")
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(f".{name.replace('-', '_')}"):
            raise RuntimeError(f"the {name!r} harness adapter is not present in this checkout (it lives on its own branch)") from e
        raise
