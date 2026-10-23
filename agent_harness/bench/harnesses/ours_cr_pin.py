"""Context ablation arm: ``ours-cr`` plus protection of small input-file reads (``pin_input_reads``): a short tool result that is
mostly the text of an indexed input file stays verbatim through compaction, within a quarter of the compaction target."""
from __future__ import annotations

from . import ours

NAME = "ours-cr-pin"
diagnostics = ours.diagnostics
OPTIONS = {"policy": "v2", "pin_input_reads": True}


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME) -> None:
    ours.generate(exp, cell, workers, run_task, client_factory, name=name, context="context-runtime", context_options=dict(OPTIONS))
