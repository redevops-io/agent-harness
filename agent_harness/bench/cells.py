"""An experimental cell: one model served one way. Cells are never mixed in a comparison."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    name: str
    provider: str            # key in the spend proxy's price table (which endpoint)
    model: str
    window: int              # context window the server was started with
    think: bool = False      # reasoning on for the executor (switched per request through the proxy label)

    def label(self, role: str) -> str:
        """Proxy path label. Thinking cells use the reserved label ``think``, which the proxy maps to enable_thinking=true."""
        return "think" if self.think else role


def arm_id(arm: str, cell: str) -> str:
    return f"{arm}@{cell}"


def task_dir_name(task: str) -> str:
    return task.replace("/", "__")
