from agent_harness.bench.context_budget import exhausted, fits, input_budget, reply_room


def test_the_g1_failure_is_recognised_and_its_numbers_are_the_measured_ones():
    assert reply_room(65536, 61716) == 3820
    assert exhausted(61716, 3820, "length", "", 65536)                                    # the instrumented request: empty content, length, used all the room
    assert not exhausted(61678, 2181, "stop", '{"criteria_results": []}', 65536)          # rubric 2 fitted and answered
    assert not exhausted(61716, 3820, "length", "partial but usable text", 65536)         # content present: cut short, not exhausted-and-empty
    assert not exhausted(30000, 900, "length", "", 65536)                                 # stopped for length far below the room: the caller's own max_tokens, a different cause


def test_planning_reserves_reasoning_and_reply_instead_of_filling_the_window():
    assert input_budget(65536, reply_tokens=8000, reasoning_tokens=8000) == 65536 - 16000 - 256
    assert not fits(61716, 65536, reply_tokens=8000, reasoning_tokens=8000)               # the G1 prompt does not fit a 65,536 window with a real reserve ...
    assert fits(61716, 98304, reply_tokens=8000, reasoning_tokens=8000)                   # ... and does fit 98,304
    assert input_budget(1000, reply_tokens=5000) == 0
