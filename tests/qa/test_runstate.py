# SPDX-License-Identifier: AGPL-3.0-or-later
"""H2: the run state machine. The allowed edges below are written out by hand (an oracle independent of the table in
``runstate.py``) and every one of the 12 x 12 ordered pairs is tested."""
import itertools

import pytest

from agent_harness.qa import records as R
from agent_harness.qa.run_errors import IllegalTransition
from agent_harness.qa.runstate import (OUTCOME_STATE, TERMINAL_STATES, TRANSITIONS, RunState, RunStateMachine,
                                       is_allowed)

S = RunState
ABORTS = [S.TEST_DEFECT, S.ENVIRONMENT_FAIL, S.INCONCLUSIVE]

#: the ADR section 3 / section 7 edges, by hand
EXPECTED = {
    (S.PREPARE, S.OBSERVE), (S.PREPARE, S.VERIFY),
    (S.OBSERVE, S.ACT),
    (S.ACT, S.OBSERVE), (S.ACT, S.VERIFY), (S.ACT, S.RECOVERABLE),
    (S.RECOVERABLE, S.REOBSERVE),
    (S.REOBSERVE, S.ACT),
    (S.VERIFY, S.PASS), (S.VERIFY, S.PRODUCT_FAIL),
    (S.INCONCLUSIVE, S.REVIEW),
}
for _src in (S.PREPARE, S.OBSERVE, S.ACT, S.RECOVERABLE, S.REOBSERVE, S.VERIFY):
    for _dst in ABORTS:
        EXPECTED.add((_src, _dst))

ALL_PAIRS = list(itertools.product(list(RunState), list(RunState)))


def test_there_are_twelve_states_and_the_expected_edge_count():
    assert len(RunState) == 12 and len(ALL_PAIRS) == 144
    assert len(EXPECTED) == 11 + 6 * 3


@pytest.mark.parametrize("src,dst", ALL_PAIRS, ids=lambda x: x.value)
def test_every_ordered_pair_is_allowed_exactly_when_the_adr_says_so(src, dst):
    allowed = (src, dst) in EXPECTED
    assert is_allowed(src, dst) is allowed
    assert (dst in TRANSITIONS[src]) is allowed
    m = RunStateMachine()
    m._state = src                                # place the machine on the source state
    if allowed:
        assert m.to(dst) is dst and m.state is dst
    else:
        with pytest.raises(IllegalTransition) as err:
            m.to(dst)
        assert err.value.code == "ILLEGAL_TRANSITION" and m.state is src       # unchanged after a refusal


def test_the_table_has_a_row_for_every_state_and_nothing_else():
    assert set(TRANSITIONS) == set(RunState)
    assert all(isinstance(v, frozenset) for v in TRANSITIONS.values())


def test_terminal_states_have_no_successors():
    assert TERMINAL_STATES == {S.PASS, S.PRODUCT_FAIL, S.TEST_DEFECT, S.ENVIRONMENT_FAIL, S.REVIEW}
    for t in TERMINAL_STATES:
        assert TRANSITIONS[t] == frozenset()
    assert TRANSITIONS[S.INCONCLUSIVE] == {S.REVIEW}          # INCONCLUSIVE always continues to REVIEW


def test_pass_and_product_fail_are_reachable_from_verify_only():
    for src in RunState:
        for dst in (S.PASS, S.PRODUCT_FAIL):
            assert is_allowed(src, dst) is (src is S.VERIFY)


def test_recovery_runs_through_recoverable_and_reobserve_only():
    assert [s for s in RunState if is_allowed(s, S.RECOVERABLE)] == [S.ACT]
    assert [s for s in RunState if is_allowed(s, S.REOBSERVE)] == [S.RECOVERABLE]
    assert {s for s in RunState if is_allowed(S.REOBSERVE, s)} == {S.ACT, *ABORTS}


def test_a_non_state_is_never_a_transition_target_or_source():
    m = RunStateMachine()
    assert "OBSERVE" == S.OBSERVE                # equal to a member as text, and still not a RunState
    for bogus in ("OBSERVE", None, 3):
        with pytest.raises(IllegalTransition):
            m.to(bogus)
    assert not is_allowed("PREPARE", S.OBSERVE) and not is_allowed(S.PREPARE, "OBSERVE")
    assert m.state is S.PREPARE and m.history == (S.PREPARE,)


def test_a_walk_through_the_machine_records_its_history():
    m = RunStateMachine()
    for nxt in (S.OBSERVE, S.ACT, S.RECOVERABLE, S.REOBSERVE, S.ACT, S.OBSERVE, S.ACT, S.VERIFY, S.INCONCLUSIVE,
                S.REVIEW):
        m.to(nxt)
    assert m.history[0] is S.PREPARE and m.history[-1] is S.REVIEW and m.terminal
    assert len(m.history) == 11
    with pytest.raises(IllegalTransition):
        m.to(S.PASS)


def test_each_outcome_has_a_state_and_only_inconclusive_is_not_terminal():
    assert set(OUTCOME_STATE) == set(R.Outcome)
    for outcome, state in OUTCOME_STATE.items():
        assert state.value == outcome.value
        assert (state in TERMINAL_STATES) is (outcome is not R.Outcome.INCONCLUSIVE)
