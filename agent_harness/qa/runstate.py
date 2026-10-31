# SPDX-License-Identifier: AGPL-3.0-or-later
"""The run state machine.

The allowed transitions are one explicit table. :class:`RunStateMachine` is the only object that moves a run between
states and it moves it only along the table: any other move raises :class:`IllegalTransition` and leaves the state
where it was. A terminal state has no successors.

::

    PREPARE ----> OBSERVE ----> ACT ----> OBSERVE (next step) ... ----> VERIFY ----> PASS
        |            |           |  \\                                      |-------> PRODUCT_FAIL
        |            |           |   +--> RECOVERABLE --> REOBSERVE --> ACT |-------> TEST_DEFECT
        |            |           |             |             |              |-------> ENVIRONMENT_FAIL
        +------------+-----------+-------------+-------------+--------------+------> INCONCLUSIVE --> REVIEW

``PASS`` and ``PRODUCT_FAIL`` are reachable from ``VERIFY`` only: they need deterministic assertion results, which
exist only after verification. Every other non-terminal state can fall into ``TEST_DEFECT``, ``ENVIRONMENT_FAIL`` or
``INCONCLUSIVE`` (an abort). ``INCONCLUSIVE`` always continues to ``REVIEW`` (a ``ReviewItem`` is always created).
The outcome itself is decided by the classifier, never here: this module only validates the move.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

import enum
from types import MappingProxyType
from typing import FrozenSet, Mapping, Tuple

from .records import Outcome
from .run_errors import IllegalTransition


class RunState(enum.StrEnum):
    PREPARE = "PREPARE"
    OBSERVE = "OBSERVE"
    ACT = "ACT"
    VERIFY = "VERIFY"
    PASS = "PASS"
    RECOVERABLE = "RECOVERABLE"
    REOBSERVE = "REOBSERVE"
    PRODUCT_FAIL = "PRODUCT_FAIL"
    TEST_DEFECT = "TEST_DEFECT"
    ENVIRONMENT_FAIL = "ENVIRONMENT_FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    REVIEW = "REVIEW"


S = RunState
_ABORTS = frozenset({S.TEST_DEFECT, S.ENVIRONMENT_FAIL, S.INCONCLUSIVE})

#: state -> the states it may move to. Terminal states map to the empty set.
TRANSITIONS: Mapping[RunState, FrozenSet[RunState]] = MappingProxyType({
    S.PREPARE: frozenset({S.OBSERVE, S.VERIFY}) | _ABORTS,              # a spec without steps goes straight to VERIFY
    S.OBSERVE: frozenset({S.ACT}) | _ABORTS,
    S.ACT: frozenset({S.OBSERVE, S.VERIFY, S.RECOVERABLE}) | _ABORTS,   # OBSERVE = next step, VERIFY = last step done
    S.RECOVERABLE: frozenset({S.REOBSERVE}) | _ABORTS,
    S.REOBSERVE: frozenset({S.ACT}) | _ABORTS,
    S.VERIFY: frozenset({S.PASS, S.PRODUCT_FAIL}) | _ABORTS,
    S.INCONCLUSIVE: frozenset({S.REVIEW}),
    S.PASS: frozenset(),
    S.PRODUCT_FAIL: frozenset(),
    S.TEST_DEFECT: frozenset(),
    S.ENVIRONMENT_FAIL: frozenset(),
    S.REVIEW: frozenset(),
})

TERMINAL_STATES: FrozenSet[RunState] = frozenset(s for s, nxt in TRANSITIONS.items() if not nxt)

#: the run state that records each run outcome (``INCONCLUSIVE`` is followed by ``REVIEW``)
OUTCOME_STATE: Mapping[Outcome, RunState] = MappingProxyType({
    Outcome.PASS: S.PASS,
    Outcome.PRODUCT_FAIL: S.PRODUCT_FAIL,
    Outcome.TEST_DEFECT: S.TEST_DEFECT,
    Outcome.ENVIRONMENT_FAIL: S.ENVIRONMENT_FAIL,
    Outcome.INCONCLUSIVE: S.INCONCLUSIVE,
})


def is_allowed(source: RunState, target: RunState) -> bool:
    """True only when the table contains the move. Anything that is not a ``RunState`` is not allowed."""
    if not isinstance(source, RunState) or not isinstance(target, RunState):
        return False
    return target in TRANSITIONS.get(source, frozenset())


class RunStateMachine:
    """Holds the current state and the ordered history of states."""

    def __init__(self) -> None:
        self._state = RunState.PREPARE
        self._history: Tuple[RunState, ...] = (RunState.PREPARE,)

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def history(self) -> Tuple[RunState, ...]:
        return self._history

    @property
    def terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    def to(self, target: RunState) -> RunState:
        """Move to ``target`` or raise :class:`IllegalTransition`; the state is unchanged after a refusal."""
        if not is_allowed(self._state, target):
            raise IllegalTransition(f"{self._state} -> {target} is not an allowed transition",
                                    source=str(self._state), target=str(target))
        self._state = target
        self._history = self._history + (target,)
        return target


__all__ = ["RunState", "TRANSITIONS", "TERMINAL_STATES", "OUTCOME_STATE", "is_allowed", "RunStateMachine"]
