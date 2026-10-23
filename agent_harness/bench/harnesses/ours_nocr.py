"""Our harness WITHOUT context-runtime: identical to ``ours`` except that nothing manages the context proactively (see
``agent_harness.context.NoCompactionManager``). Paired with ``ours`` it measures what context-runtime costs or buys."""
from __future__ import annotations

from . import ours

NAME = "ours-nocr"
diagnostics = ours.diagnostics


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME) -> None:
    ours.generate(exp, cell, workers, run_task, client_factory, name=name, context="none")
