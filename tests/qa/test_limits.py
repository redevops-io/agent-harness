# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deadlines and budgets. Property tests run over seeded pseudo-random sequences (no third-party library) with a
fake clock: a budget never goes past its limit, a refused charge changes nothing, a deadline is spent exactly when the
elapsed time reaches it, and operator caps can only lower a spec value."""
import random

import pytest

from agent_harness.qa import records as R
from agent_harness.qa.errors import SpecInvalid
from agent_harness.qa.limits import (Budgets, Deadlines, EffectiveLimits, LimitCaps, SCOPE_OBSERVATION, SCOPE_RUN,
                                     SCOPE_STEP, effective_limits)
from agent_harness.qa.run_errors import BudgetExceeded, DeadlineExceeded

from ._fx_runner import FakeClock


def budget(**kw):
    base = dict(deadline_ms=1000, max_model_calls=0, max_recovery_attempts=2, max_retries=1, max_tokens=0)
    base.update(kw)
    return R.Budget(**base)


def limits(**kw):
    return effective_limits(budget(**kw))


def test_a_spec_without_model_budget_allows_zero_model_calls_and_tokens():
    b = Budgets(effective_limits(R.Budget(deadline_ms=10)))
    assert b.model_calls_left == 0 and b.tokens_left == 0
    with pytest.raises(BudgetExceeded) as err:
        b.charge_model(1)
    assert err.value.kind == "MODEL_CALLS" and b.model_calls == 0
    with pytest.raises(BudgetExceeded):
        b.charge_model(0, 1)
    b.charge_model(0, 0)                      # charging nothing is always fine


def test_caps_only_lower_never_raise():
    spec = budget(deadline_ms=1000, max_retries=3, max_recovery_attempts=4, max_model_calls=5, max_tokens=600)
    low = effective_limits(spec, LimitCaps(deadline_ms=10, step_deadline_ms=20, max_retries=0, max_recovery_attempts=1,
                                           max_model_calls=2, max_tokens=7))
    assert (low.deadline_ms, low.step_deadline_ms, low.max_retries, low.max_recovery_attempts, low.max_model_calls,
            low.max_tokens) == (10, 20, 0, 1, 2, 7)
    high = effective_limits(spec, LimitCaps(deadline_ms=10**6, step_deadline_ms=10**6, max_retries=99,
                                            max_recovery_attempts=99, max_model_calls=99, max_tokens=10**9))
    assert (high.deadline_ms, high.step_deadline_ms, high.max_retries, high.max_recovery_attempts,
            high.max_model_calls, high.max_tokens) == (1000, 1000, 3, 4, 5, 600)


@pytest.mark.parametrize("name,bad", [("deadline_ms", 0), ("step_deadline_ms", 0), ("max_retries", -1),
                                      ("max_tokens", 1.5), ("max_tokens", True), ("max_model_calls", "1")])
def test_bad_caps_are_refused(name, bad):
    with pytest.raises(SpecInvalid):
        LimitCaps(**{name: bad})


def test_property_effective_limits_never_exceed_the_spec():
    rng = random.Random(20261030)
    for _ in range(300):
        spec = budget(deadline_ms=rng.randint(1, 10**5), max_retries=rng.randint(0, 50),
                      max_recovery_attempts=rng.randint(0, 50), max_model_calls=rng.randint(0, 50),
                      max_tokens=rng.randint(0, 10**6))
        caps = LimitCaps(**{k: rng.choice([None, rng.randint(0 if k.startswith("max") else 1, 10**6)])
                            for k in ("deadline_ms", "step_deadline_ms", "max_retries", "max_recovery_attempts",
                                      "max_model_calls", "max_tokens")})
        eff = effective_limits(spec, caps)
        assert eff.deadline_ms <= spec.deadline_ms and eff.step_deadline_ms <= spec.deadline_ms
        assert eff.max_retries <= spec.max_retries and eff.max_recovery_attempts <= spec.max_recovery_attempts
        assert eff.max_model_calls <= spec.max_model_calls and eff.max_tokens <= spec.max_tokens


def test_property_a_budget_never_passes_its_limit_and_a_refusal_changes_nothing():
    rng = random.Random(7)
    for _ in range(200):
        lim = EffectiveLimits(deadline_ms=1, step_deadline_ms=1, max_retries=rng.randint(0, 6),
                              max_recovery_attempts=rng.randint(0, 6), max_model_calls=rng.randint(0, 6),
                              max_tokens=rng.randint(0, 50))
        b = Budgets(lim)
        for _ in range(60):
            before = (b.retries, b.recovery_attempts, b.model_calls, b.tokens)
            op = rng.choice(["retry", "recover", "model"])
            try:
                if op == "retry":
                    b.charge_retry()
                elif op == "recover":
                    b.charge_recovery_attempt()
                else:
                    b.charge_model(rng.randint(0, 3), rng.randint(0, 20))
            except BudgetExceeded:
                assert (b.retries, b.recovery_attempts, b.model_calls, b.tokens) == before    # all-or-nothing
            assert b.retries <= lim.max_retries and b.recovery_attempts <= lim.max_recovery_attempts
            assert b.model_calls <= lim.max_model_calls and b.tokens <= lim.max_tokens
            assert b.retries_left >= 0 and b.model_calls_left >= 0 and b.tokens_left >= 0


def test_a_model_charge_that_fits_calls_but_not_tokens_charges_neither():
    b = Budgets(limits(max_model_calls=5, max_tokens=10))
    with pytest.raises(BudgetExceeded) as err:
        b.charge_model(1, 11)
    assert err.value.kind == "TOKENS" and (b.model_calls, b.tokens) == (0, 0)
    b.charge_model(2, 10)
    assert (b.model_calls, b.tokens) == (2, 10)


@pytest.mark.parametrize("bad", [-1, 1.0, "1", None, True])
def test_a_malformed_model_charge_is_refused(bad):
    with pytest.raises(SpecInvalid):
        Budgets(limits(max_model_calls=5, max_tokens=5)).charge_model(bad, 0)


def test_the_retry_budget_is_exactly_its_limit():
    b = Budgets(limits(max_retries=2))
    b.charge_retry()
    b.charge_retry()
    with pytest.raises(BudgetExceeded) as err:
        b.charge_retry()
    assert err.value.kind == "RETRIES" and b.retries == 2


# --- deadlines ----------------------------------------------------------------------------------------------------

def test_the_run_deadline_is_spent_exactly_when_elapsed_reaches_it():
    clock = FakeClock(1000)
    d = Deadlines(clock, limits(deadline_ms=100))
    d.check_run()
    clock.advance(99)
    d.check_run()
    assert d.remaining_ms() == 1 and d.elapsed_ms() == 99
    clock.advance(1)
    with pytest.raises(DeadlineExceeded) as err:
        d.check_run()
    assert err.value.scope == SCOPE_RUN == "RUN" and d.remaining_ms() == 0


def test_the_step_deadline_counts_from_the_start_of_the_step():
    clock = FakeClock()
    d = Deadlines(clock, effective_limits(budget(deadline_ms=1000), LimitCaps(step_deadline_ms=50)))
    clock.advance(400)
    timer = d.begin_step()
    clock.advance(49)
    timer.check()
    clock.advance(1)
    with pytest.raises(DeadlineExceeded) as err:
        timer.check()
    assert err.value.scope == SCOPE_STEP
    d.check_run()                                  # the run still has time: the step deadline is its own


def test_the_observation_deadline_is_measured_per_call():
    clock = FakeClock()
    d = Deadlines(clock, limits(deadline_ms=10**6))
    started = clock.monotonic_ms()
    clock.advance(4)
    d.check_observation(started, 5)
    clock.advance(1)
    with pytest.raises(DeadlineExceeded) as err:
        d.check_observation(started, 5)
    assert err.value.scope == SCOPE_OBSERVATION


def test_property_deadlines_fire_at_exactly_the_limit_for_any_step_sequence():
    rng = random.Random(99)
    for _ in range(300):
        limit = rng.randint(1, 500)
        clock = FakeClock(rng.randint(0, 10**6))
        d = Deadlines(clock, limits(deadline_ms=limit))
        spent = 0
        for _ in range(50):
            step = rng.randint(0, 60)
            clock.advance(step)
            spent += step
            if spent >= limit:
                with pytest.raises(DeadlineExceeded):
                    d.check_run()
                assert d.remaining_ms() == 0
                break
            d.check_run()
            assert d.remaining_ms() == limit - spent


def test_a_clock_that_runs_backwards_never_makes_elapsed_negative():
    clock = FakeClock(500)
    d = Deadlines(clock, limits(deadline_ms=10))
    clock.ms = 100
    assert d.elapsed_ms() == 0
    d.check_run()


def test_deadline_exceptions_carry_their_numbers_and_a_stable_code():
    clock = FakeClock()
    d = Deadlines(clock, limits(deadline_ms=10))
    clock.advance(25)
    with pytest.raises(DeadlineExceeded) as err:
        d.check_run()
    assert err.value.code == "DEADLINE_EXCEEDED"
    assert err.value.details["elapsed_ms"] == 25 and err.value.details["limit_ms"] == 10
