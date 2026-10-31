# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deadlines and bounded budgets for one run.

Everything is measured through the injected :class:`~agent_harness.qa.ports.Clock`; nothing here reads real time.

* **Effective limits** are the spec's budget, lowered (never raised) by operator caps: a cap larger than the spec value
  is ignored, so an arm cap can only tighten what the reviewed spec allows. Model calls and tokens default to zero in
  the spec, so a run makes no model call unless a reviewed spec says it may.
* **Deadlines** are cooperative: the runner checks them before each operation and again after it returns (an adapter
  call cannot be interrupted through the neutral protocol). A deadline is spent when ``elapsed >= limit``: there is no
  time left to start anything, which is the fail-closed reading of a limit given in whole milliseconds.
* **Budgets** are counters that never pass their limit: a charge that would exceed it raises ``BudgetExceeded`` and
  changes nothing.

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .errors import SpecInvalid
from .ports import Clock
from .records import Budget
from .run_errors import BudgetExceeded, DeadlineExceeded

SCOPE_RUN = "RUN"
SCOPE_STEP = "STEP"
SCOPE_OBSERVATION = "OBSERVATION"


def _int_or_none(name: str, value: Any, minimum: int) -> Optional[int]:
    if value is None:
        return None
    if type(value) is not int:
        raise SpecInvalid(f"LimitCaps.{name}: expected an integer or None, got {type(value).__name__}")
    if value < minimum:
        raise SpecInvalid(f"LimitCaps.{name}: below the minimum {minimum}")
    return value


@dataclass(frozen=True, kw_only=True)
class LimitCaps:
    """Operator caps (for example the arm caps). ``None`` means no cap. A cap can only lower a spec value."""

    deadline_ms: Optional[int] = None
    step_deadline_ms: Optional[int] = None
    max_retries: Optional[int] = None
    max_recovery_attempts: Optional[int] = None
    max_model_calls: Optional[int] = None
    max_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        for name, minimum in (("deadline_ms", 1), ("step_deadline_ms", 1), ("max_retries", 0),
                              ("max_recovery_attempts", 0), ("max_model_calls", 0), ("max_tokens", 0)):
            _int_or_none(name, getattr(self, name), minimum)


@dataclass(frozen=True)
class EffectiveLimits:
    deadline_ms: int
    step_deadline_ms: int
    max_retries: int
    max_recovery_attempts: int
    max_model_calls: int
    max_tokens: int

    def to_json(self) -> Dict[str, int]:
        return {"deadline_ms": self.deadline_ms, "max_model_calls": self.max_model_calls,
                "max_recovery_attempts": self.max_recovery_attempts, "max_retries": self.max_retries,
                "max_tokens": self.max_tokens, "step_deadline_ms": self.step_deadline_ms}


def _low(spec_value: int, cap: Optional[int]) -> int:
    return spec_value if cap is None else min(spec_value, cap)


def effective_limits(budget: Budget, caps: Optional[LimitCaps] = None) -> EffectiveLimits:
    """The spec budget lowered by the caps. Never larger than the spec on any axis."""
    caps = caps or LimitCaps()
    if type(budget) is not Budget:
        raise SpecInvalid("effective_limits takes a Budget record")
    return EffectiveLimits(
        deadline_ms=_low(budget.deadline_ms, caps.deadline_ms),
        step_deadline_ms=_low(budget.deadline_ms, caps.step_deadline_ms),
        max_retries=_low(budget.max_retries, caps.max_retries),
        max_recovery_attempts=_low(budget.max_recovery_attempts, caps.max_recovery_attempts),
        max_model_calls=_low(budget.max_model_calls, caps.max_model_calls),
        max_tokens=_low(budget.max_tokens, caps.max_tokens),
    )


class StepTimer:
    """Elapsed time inside one step (all attempts), against the step deadline."""

    def __init__(self, clock: Clock, limit_ms: int) -> None:
        self._clock = clock
        self._limit = limit_ms
        self._start = clock.monotonic_ms()

    def elapsed_ms(self) -> int:
        return max(0, self._clock.monotonic_ms() - self._start)

    def check(self) -> None:
        elapsed = self.elapsed_ms()
        if elapsed >= self._limit:
            raise DeadlineExceeded(f"step deadline of {self._limit} ms is spent", scope=SCOPE_STEP,
                                   elapsed_ms=elapsed, limit_ms=self._limit)


class Deadlines:
    """Run, step and observation deadlines over one clock."""

    def __init__(self, clock: Clock, limits: EffectiveLimits) -> None:
        self._clock = clock
        self._limits = limits
        self._start = clock.monotonic_ms()

    def elapsed_ms(self) -> int:
        return max(0, self._clock.monotonic_ms() - self._start)

    def remaining_ms(self) -> int:
        return max(0, self._limits.deadline_ms - self.elapsed_ms())

    def check_run(self) -> None:
        elapsed = self.elapsed_ms()
        if elapsed >= self._limits.deadline_ms:
            raise DeadlineExceeded(f"run deadline of {self._limits.deadline_ms} ms is spent", scope=SCOPE_RUN,
                                   elapsed_ms=elapsed, limit_ms=self._limits.deadline_ms)

    def begin_step(self) -> StepTimer:
        return StepTimer(self._clock, self._limits.step_deadline_ms)

    def check_observation(self, started_ms: int, deadline_ms: int) -> None:
        """An observation (or locate) that started at ``started_ms`` must have finished inside ``deadline_ms``."""
        elapsed = max(0, self._clock.monotonic_ms() - started_ms)
        if elapsed >= deadline_ms:
            raise DeadlineExceeded(f"observation deadline of {deadline_ms} ms is spent", scope=SCOPE_OBSERVATION,
                                   elapsed_ms=elapsed, limit_ms=deadline_ms)


class Budgets:
    """Counters for retries, recovery attempts, model calls and tokens. A charge either fits or raises untouched."""

    def __init__(self, limits: EffectiveLimits) -> None:
        self._limits = limits
        self.retries = 0
        self.recovery_attempts = 0
        self.model_calls = 0
        self.tokens = 0

    @staticmethod
    def _amount(value: Any, what: str) -> int:
        if type(value) is not int or value < 0:
            raise SpecInvalid(f"{what}: expected a non-negative integer")
        return value

    def _take(self, kind: str, used: int, amount: int, limit: int) -> int:
        if used + amount > limit:
            raise BudgetExceeded(f"{kind} budget of {limit} would be exceeded", kind=kind, used=used, limit=limit)
        return used + amount

    def charge_retry(self) -> None:
        self.retries = self._take("RETRIES", self.retries, 1, self._limits.max_retries)

    def charge_recovery_attempt(self) -> None:
        self.recovery_attempts = self._take("RECOVERY_ATTEMPTS", self.recovery_attempts, 1,
                                            self._limits.max_recovery_attempts)

    def charge_model(self, calls: int, tokens: int = 0) -> None:
        """Charge model calls and tokens together: if either would pass its limit, neither is charged."""
        calls, tokens = self._amount(calls, "calls"), self._amount(tokens, "tokens")
        new_calls = self._take("MODEL_CALLS", self.model_calls, calls, self._limits.max_model_calls)
        new_tokens = self._take("TOKENS", self.tokens, tokens, self._limits.max_tokens)
        self.model_calls, self.tokens = new_calls, new_tokens

    @property
    def model_calls_left(self) -> int:
        return self._limits.max_model_calls - self.model_calls

    @property
    def tokens_left(self) -> int:
        return self._limits.max_tokens - self.tokens

    @property
    def recovery_attempts_left(self) -> int:
        return self._limits.max_recovery_attempts - self.recovery_attempts

    @property
    def retries_left(self) -> int:
        return self._limits.max_retries - self.retries


__all__ = ["SCOPE_RUN", "SCOPE_STEP", "SCOPE_OBSERVATION", "LimitCaps", "EffectiveLimits", "effective_limits",
           "StepTimer", "Deadlines", "Budgets"]
