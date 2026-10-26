"""Opt-in stall checkpoint (prototype, off by default): the third refusal in a row of one operation, while nothing deliverable exists, says what stalled and what is missing, and starts delivery mode."""
from agent_harness.completion import DeliverableReport
from agent_harness.supervisor import Supervisor, SupervisorConfig


def drive(cfg, report, n=4, tool="bash", args=None):
    sup = Supervisor(cfg, max_turns=200, max_seconds=3600, checker=lambda: report)
    texts = []
    for i in range(n):
        sup.begin_turn(i + 1, 10.0 * (i + 1))
        d = sup.before(f"c{i}", tool, args or {"command": "python3 check.py"}, [])
        if d.run:
            texts.append(sup.after(f"c{i}", tool, args or {"command": "python3 check.py"}, "same result", True))
        else:
            texts.append(sup.attach_pending(d.text))
    return sup, texts


NONE = DeliverableReport(status="none", expected_names=["a.db", "b.xlsx"], unmatched_names=["a.db", "b.xlsx"], required_exts=["db", "xlsx"], missing_exts=["db", "xlsx"])


def test_off_by_default_the_refusal_carries_no_checkpoint():
    sup, texts = drive(SupervisorConfig(), NONE)
    assert any("Not run" in t for t in texts) and not any("[checkpoint]" in t for t in texts) and sup.stall_checkpoint_turn is None and not sup.delivery_mode


def test_the_third_refusal_in_a_row_with_nothing_delivered_carries_one_structured_checkpoint():
    sup, texts = drive(SupervisorConfig(stall_checkpoint=True), NONE, n=8)
    cps = [t for t in texts if "[checkpoint]" in t]
    assert len(cps) == 1 and sup.counts["stall_checkpoints"] == 1
    cp = cps[0]
    assert "python3 check.py" in cp and "a.db" in cp and "b.xlsx" in cp and "turns" in cp and "web_fetch" in cp and "partial file is better than none" in cp
    assert sup.delivery_mode and sup.stall_checkpoint_turn == 5          # calls 3, 4, 5 are the first three refusals (two identical runs before them)


def test_after_the_checkpoint_exploration_tools_are_refused_while_nothing_is_valid():
    sup, _ = drive(SupervisorConfig(stall_checkpoint=True), NONE, n=5)
    sup.begin_turn(10, 100.0)
    d = sup.before("x", "web_fetch", {"url": "https://example.org/a"}, [])
    assert not d.run and "reserve" in d.text


def test_no_checkpoint_when_a_valid_deliverable_already_exists():
    ok = DeliverableReport(status="structurally_valid")
    sup, texts = drive(SupervisorConfig(stall_checkpoint=True), ok, n=6)
    assert not any("[checkpoint]" in t for t in texts) and sup.stall_checkpoint_turn is None


def test_without_a_deliverable_checker_nothing_is_claimed():
    sup = Supervisor(SupervisorConfig(stall_checkpoint=True), max_turns=200)
    for i in range(5):
        sup.begin_turn(i + 1, 1.0)
        d = sup.before(f"c{i}", "bash", {"command": "x"}, [])
        if d.run:
            sup.after(f"c{i}", "bash", {"command": "x"}, "same", True)
    assert sup.stall_checkpoint_turn is None


def test_a_single_refusal_is_not_a_stall():
    sup, texts = drive(SupervisorConfig(stall_checkpoint=True), NONE, n=4)             # calls 3 and 4 are refused: two refusals
    assert sup.stall_checkpoint_turn is None and not any("[checkpoint]" in t for t in texts)
