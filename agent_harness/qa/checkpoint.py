# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-run checkpoints (plan H2; ADR section 4.1 ``Checkpoint``).

A checkpoint is the step index reached, the state hash seen there and an optional fixture snapshot id: nothing else.
It holds no cookie, no local storage, no session token and no page content ("storage state is never persisted to
evidence"), and it lives in memory for the duration of one run: this module writes nothing to disk and keeps nothing
across runs. Resuming from a checkpoint is a recovery strategy (``CHECKPOINT_RESUME``) and is plan H3.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .ports import Clock
from .receipts import bounded_id
from .records import Checkpoint


class CheckpointLog:
    """Append-only, in memory, one per run."""

    def __init__(self, run_id: str, clock: Optional[Clock] = None) -> None:
        self._run_id = run_id
        self._clock = clock
        self._items: List[Checkpoint] = []

    def record(self, step_index: int, state_hash: str, fixture_snapshot_id: Optional[str] = None) -> Checkpoint:
        """Add the checkpoint for having completed step ``step_index``. Indexes only go up."""
        if self._items and step_index <= self._items[-1].step_index:
            raise ValueError("checkpoints are recorded in increasing step order")
        item = Checkpoint(checkpoint_id=bounded_id("ckpt", self._run_id, step_index), run_id=self._run_id,
                          step_index=step_index, state_hash=state_hash, fixture_snapshot_id=fixture_snapshot_id,
                          taken_at=self._clock.now_utc() if self._clock is not None else None)
        self._items.append(item)
        return item

    @property
    def items(self) -> Tuple[Checkpoint, ...]:
        return tuple(self._items)

    @property
    def latest(self) -> Optional[Checkpoint]:
        return self._items[-1] if self._items else None


__all__ = ["CheckpointLog"]
