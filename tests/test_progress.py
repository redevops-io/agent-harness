"""Progress-aware recovery: classification, bounded budget, nothing forced, seeded defects, replay, and the opt-in tool/capability changes."""
import json
from pathlib import Path

import pytest

import agent_harness.bench.tools as bt
from agent_harness.bench.capabilities import (environment_note, probe_environment, recovery_policy, stray_deliverables, workspace_fingerprint)
from agent_harness.bench.replay import Step, load_steps, replay
from agent_harness.loop import run_loop
from agent_harness.progress import Capability, ProgressTracker, RecoveryPolicy, fingerprint, normalize_call
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool


# ---- helpers -----------------------------------------------------------------------------------------------------------
class Script:
    def __init__(self, replies):
        self.replies, self.i, self.seen = list(replies), 0, []

    def complete(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return {"message": r, "finish_reason": "stop", "usage": {"prompt_tokens": 100, "completion_tokens": 5}}


def call(name, cid="c1", **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def registry(results, runs=None, side=None):
    """Tool 'probe' returns the next scripted result each time; 'write' changes the fake workspace."""
    runs = runs if runs is not None else []
    it = iter(results)
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=["write"]))

    def probe(a):
        runs.append(dict(a))
        r = next(it, "STILL-THE-SAME")
        return ToolResult(ok=not str(r).startswith("ERROR:"), text=str(r), error=str(r) if str(r).startswith("ERROR:") else None)

    def write(a):
        if side is not None:
            side["n"] += 1
        return ToolResult(text="written")
    reg.register(function_tool("probe", probe, "probe", {"type": "object", "properties": {"q": {"type": "string"}}}))
    reg.register(function_tool("write", write, "write", {"type": "object", "properties": {}}, side_effecting=True))
    return reg


def tracker(**kw):
    state = {"w": 0}
    pol = RecoveryPolicy(progress_probe=lambda: state["w"], **kw)
    return ProgressTracker(pol), state


# ---- normalisation and fingerprints -------------------------------------------------------------------------------------
def test_normalisation_ignores_layout_not_content():
    assert normalize_call("bash", {"command": "ls   -la\n/work"}) == normalize_call("bash", {"command": "ls -la /work"})
    assert normalize_call("web_fetch", {"url": "https://Example.com/a/#frag"}) == normalize_call("web_fetch", {"url": "https://example.com/a"})
    assert normalize_call("web_fetch", {"url": "https://example.com/a?x=1"}) != normalize_call("web_fetch", {"url": "https://example.com/a?x=2"})
    assert normalize_call("bash", {"command": "ls"}) != normalize_call("read_file", {"command": "ls"})


def test_fingerprint_ignores_whitespace_and_notes_added_by_the_harness():
    assert fingerprint("a  b\n c") == fingerprint("a b c")
    assert fingerprint("result\n[environment: x]") == fingerprint("result")
    assert fingerprint("result") != fingerprint("other")


# ---- classification -----------------------------------------------------------------------------------------------------
def test_redundant_call_is_not_executed_a_third_time():
    tr, _ = tracker()
    for _ in range(2):
        assert tr.before("probe", {"q": "a"}) is None
        tr.after("probe", {"q": "a"}, "same", True)
    msg = tr.before("probe", {"q": "a"})
    assert msg and "[no progress]" in msg and "2 times" in msg and tr.counts["redundant"] == 1


def test_polling_with_changing_results_is_legitimate_until_the_poll_limit():
    tr, _ = tracker(poll_limit=5)
    for i in range(6):
        assert tr.before("probe", {"q": "status"}) is None
        tr.after("probe", {"q": "status"}, f"progress {i}%", True)
    assert tr.counts["redundant"] == 0 and tr.streak == 0
    msg = tr.before("probe", {"q": "status"})
    assert msg and "polling limit" in msg and tr.counts["poll_exhausted"] == 1


def test_repeated_identical_error_is_blocked_but_allowed_again_after_the_workspace_changes():
    tr, state = tracker(error_after=3)
    for _ in range(3):
        assert tr.before("probe", {"q": "x"}) is None
        tr.after("probe", {"q": "x"}, "ERROR: no such file", False)
    msg = tr.before("probe", {"q": "x"})
    assert msg and "failed 3 times" in msg
    state["w"] += 1                      # the model fixed something; the same call may now behave differently
    tr.after("probe", {"q": "other"}, "ok", True)       # an executed call notices the changed workspace
    assert tr.before("probe", {"q": "x"}) is None


def test_different_errors_are_not_a_loop():
    tr, _ = tracker(error_after=3)
    for i in range(5):
        assert tr.before("probe", {"q": "x"}) is None
        tr.after("probe", {"q": "x"}, f"ERROR: failure number {i}", False)


def test_progress_resets_the_streak_and_new_information_counts_as_progress():
    tr, state = tracker(budget=3)
    tr.after("probe", {"q": "a"}, "A", True)
    for _ in range(2):
        tr.after("probe", {"q": "a"}, "A", True)            # same result again: no progress
    assert tr.streak >= 1
    tr.after("probe", {"q": "brand new"}, "fresh facts", True)
    assert tr.streak == 0


def test_capability_note_is_added_once_with_the_harness_supplied_alternatives():
    cap = Capability("network", (r"name resolution",), "bash has no network; use web_fetch.")
    tr, _ = tracker(capabilities=[cap])
    first = tr.after("bash", {"command": "curl x"}, "ERROR: Temporary failure in name resolution", False)
    second = tr.after("bash", {"command": "curl y"}, "ERROR: Temporary failure in name resolution", False)
    assert "[environment: bash has no network; use web_fetch.]" in first and "[environment" not in second
    assert [b["kind"] for b in tr.blockers] == ["capability"]


def test_checkpoint_note_offers_a_choice_once_and_forces_nothing():
    tr, _ = tracker(budget=50, checkpoint_after=3, redundant_after=1)
    tr.after("probe", {"q": "a"}, "same", True)
    msgs = [tr.before("probe", {"q": "a"}) for _ in range(6)]
    with_cp = [m for m in msgs if "[checkpoint]" in m]
    assert len(with_cp) == 1 and "instead of retrying" in with_cp[0] and "write the deliverable from them" in with_cp[0]


# ---- loop integration ---------------------------------------------------------------------------------------------------
def test_without_a_policy_the_legacy_guard_still_applies_unchanged():
    runs = []
    reg = registry([], runs)
    res = run_loop(Script([call("probe", q="a")] * 6 + [{"role": "assistant", "content": "done"}]), reg, "s", "t", max_turns=10)
    assert res.recovery is None and res.stop == "done" and len(runs) == 1 + 1      # legacy blocks from the third identical result on


def test_seeded_defect_endless_identical_calls_legacy_burns_the_whole_turn_budget_policy_stops_early():
    replies = [call("probe", q="same")]
    legacy = run_loop(Script(replies), registry([]), "s", "t", max_turns=60)
    assert legacy.stop == "max_turns" and legacy.turns == 60
    pol = RecoveryPolicy(budget=5, checkpoint_after=3, progress_probe=lambda: 0)
    res = run_loop(Script(replies), registry([]), "s", "t", max_turns=60, recovery=pol)
    assert res.stop == "blocked" and res.turns <= 2 + 5 + 1
    assert res.recovery["exhausted"] and res.recovery["blockers"][0]["kind"] == "redundant"
    assert res.final_text == ""                       # nothing was invented on the model's behalf


def test_recovery_never_calls_a_tool_on_the_models_behalf():
    side = {"n": 0}
    pol = RecoveryPolicy(budget=4, progress_probe=lambda: 0)
    res = run_loop(Script([call("probe", q="same")]), registry([], side=side), "s", "t", max_turns=40, recovery=pol)
    assert res.stop == "blocked" and side["n"] == 0 and res.tool_calls == res.turns


def test_a_productive_run_with_many_distinct_calls_is_never_blocked():
    replies = [call("probe", q=str(i)) for i in range(80)] + [{"role": "assistant", "content": "finished"}]
    res = run_loop(Script(replies), registry([f"result {i}" for i in range(80)]), "s", "t", max_turns=200,
                   recovery=RecoveryPolicy(budget=5, progress_probe=lambda: 0))
    assert res.stop == "done" and res.recovery["blocked_calls"] == 0


def test_workspace_change_between_identical_calls_allows_them_to_run():
    side, state = {"n": 0}, {"w": 0}
    runs = []
    reg = registry(["R", "R", "R", "R"], runs, side)
    pol = RecoveryPolicy(budget=10, progress_probe=lambda: state["w"] + side["n"])
    replies = [call("probe", q="a"), call("probe", q="a"), call("write"), call("probe", q="a"), {"role": "assistant", "content": "ok"}]
    res = run_loop(Script(replies), reg, "s", "t", max_turns=20, recovery=pol)
    assert res.stop == "done" and len(runs) == 3          # the third identical call ran because the write changed the workspace


def test_recovery_messages_reach_the_model_and_count_as_tool_errors():
    pol = RecoveryPolicy(budget=3, progress_probe=lambda: 0)
    c = Script([call("probe", q="same")])
    res = run_loop(c, registry([]), "s", "t", max_turns=20, recovery=pol)
    tool_msgs = [m["content"] for m in c.seen[-1] if m.get("role") == "tool"]
    assert any(m.startswith("[no progress]") for m in tool_msgs) and res.tool_errors >= 1


def test_unavailable_capability_scenario_note_names_alternatives_not_a_hardcoded_fix():
    cap = Capability("image", (r"does not support image",), "The model cannot view images. Use code or sandbox tools to process image content.")
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=[]))
    reg.register(function_tool("read_image", lambda a: ToolResult(ok=False, error="ERROR: this model does not support image input"), "r",
                               {"type": "object", "properties": {}}))
    c = Script([call("read_image"), {"role": "assistant", "content": "ok"}])
    run_loop(c, reg, "s", "t", recovery=RecoveryPolicy(capabilities=[cap]))
    shown = [m["content"] for m in c.seen[1] if m.get("role") == "tool"][0]
    assert "[environment: The model cannot view images." in shown and "OCR" not in shown


# ---- replay -------------------------------------------------------------------------------------------------------------
def test_replay_finds_the_first_intervention_and_the_end_of_a_stuck_run():
    steps = [Step(1, "web_fetch", {"url": "https://x.org/a.csv"}, "header,1,2", True)] + \
            [Step(i, "web_fetch", {"url": "https://x.org/a.csv"}, "[repeat] You have made this exact web_fetch call 3 times", True) for i in range(2, 40)]
    r = replay(steps, RecoveryPolicy(budget=8))
    assert r["first_blocked_step"] == 3 and r["ended_at_step"] is not None and r["steps_saved"] > 20


def test_replay_does_not_block_a_legitimate_edit_run_cycle():
    steps = []
    for i in range(10):
        steps.append(Step(len(steps) + 1, "bash", {"command": "python build.py"}, "ERROR: KeyError 'x'", False))
        steps.append(Step(len(steps) + 1, "write_file", {"path": "/work/build.py", "content": f"v{i}"}, "wrote", True))
    r = replay(steps, RecoveryPolicy(budget=5, error_after=2))
    assert r["blocked_calls"] == 0 and r["ended_at_step"] is None


def test_load_steps_reads_a_trace_directory(tmp_path):
    from agent_harness.trace import TraceRecorder
    rec = TraceRecorder(tmp_path / "trace1")
    rec.tool_result(1, "bash", {"command": "ls"}, "a b", "a b")
    rec.close()
    steps = load_steps(tmp_path / "trace1")
    assert [(s.name, s.args, s.text, s.ok) for s in steps] == [("bash", {"command": "ls"}, "a b", True)]


# ---- capabilities, artifacts, tools -------------------------------------------------------------------------------------
def test_probe_environment_reports_only_what_the_sandbox_has():
    class Sbx:
        def run(self, cmd, timeout):
            return "yes\n" if ("tesseract" in cmd or "pandoc" in cmd) else "no\n"
    found = probe_environment(Sbx())
    assert found == ["tesseract", "pandoc"]
    note = environment_note(found, web_paging=True, web_downloads=False)
    assert "tesseract, pandoc" in note and "offset" in note and "web_download" not in note and "/work/output" in note
    assert "none of" in environment_note([], web_paging=False, web_downloads=False)


def test_probe_failures_are_not_found_not_errors():
    class Bad:
        def run(self, cmd, timeout):
            raise RuntimeError("container gone")
    assert probe_environment(Bad()) == []


def test_stray_deliverables_are_reported_not_moved(tmp_path):
    ws = tmp_path
    for d in ("task_folder", "output", "scratch"):
        (ws / d).mkdir()
    (ws / "task_folder" / "input.xlsx").write_text("in")
    (ws / "output" / "ok.xlsx").write_text("out")
    (ws / "scratch" / "draft.docx").write_text("draft")
    (ws / "results.csv").write_text("a")
    (ws / "build.py").write_text("code")
    assert stray_deliverables(ws) == ["results.csv", "scratch/draft.docx"]
    assert (ws / "scratch" / "draft.docx").exists() and not (ws / "output" / "draft.docx").exists()


def test_workspace_fingerprint_changes_on_new_files_and_ignores_the_inputs(tmp_path):
    (tmp_path / "task_folder").mkdir()
    a = workspace_fingerprint(tmp_path)
    (tmp_path / "task_folder" / "x.txt").write_text("input appears")
    assert workspace_fingerprint(tmp_path) == a
    (tmp_path / "new.py").write_text("code")
    assert workspace_fingerprint(tmp_path) != a


def test_paged_web_fetch_pages_with_offset_and_says_where_the_next_part_starts(monkeypatch):
    monkeypatch.setattr(bt, "_fetch", lambda *a, **k: "0123456789" * 3000)         # 30,000 characters
    legacy = bt.web_fetch("https://x.org/d.csv")
    assert len(legacy) < 15100 and "truncated, 15000 more chars" in legacy and "offset" not in legacy
    p1 = bt.web_fetch("https://x.org/d.csv", paged=True)
    assert "offset=15000" in p1
    p2 = bt.web_fetch("https://x.org/d.csv", paged=True, offset=15000)
    assert p2.startswith("0123456789") and "truncated" not in p2
    assert "past the end" in bt.web_fetch("https://x.org/d.csv", paged=True, offset=99999)


def test_web_download_saves_inside_the_workspace_only(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "_fetch", lambda *a, **k: b"a,b\n1,2\n")
    class Sbx:
        def run(self, c, t): return ""
    reg = bt.bench_registry(tmp_path, Sbx(), web=True, web_paging=True, web_downloads=True)
    assert {"web_fetch", "web_download"} <= set(reg.names()) if hasattr(reg, "names") else True
    r = reg.run("web_download", {"url": "https://x.org/d.csv", "path": "/work/data/d.csv"})
    assert r.ok and (tmp_path / "data" / "d.csv").read_bytes() == b"a,b\n1,2\n"
    r2 = reg.run("web_download", {"url": "https://x.org/d.csv", "path": "/work/../../etc/evil"})
    assert not r2.ok and not (tmp_path.parent / "etc").exists()


def test_new_tools_are_absent_unless_requested(tmp_path):
    class Sbx:
        def run(self, c, t): return ""
    reg = bt.bench_registry(tmp_path, Sbx(), web=True)
    specs = {s.name: s for s in reg.specs()} if hasattr(reg, "specs") and not callable(getattr(reg, "specs")) else {s["function"]["name"]: s for s in reg.specs()}
    assert "web_download" not in specs and "offset" not in json.dumps(specs["web_fetch"] if isinstance(specs["web_fetch"], dict) else specs["web_fetch"].__dict__)


def test_recovery_policy_factory_wires_the_workspace_probe(tmp_path):
    pol = recovery_policy(tmp_path, web_paging=True, web_downloads=True)
    a = pol.progress_probe()
    (tmp_path / "f.txt").write_text("x")
    assert pol.progress_probe() != a and {c.name for c in pol.capabilities} == {"network", "image"}
    assert recovery_policy(tmp_path, with_capabilities=False).capabilities == ()


# ---- run_task wiring: artifact collection is honest, options are opt-in ------------------------------------------------
def _task(tmp_path):
    t = tmp_path / "base" / "x" / "task1" / "task_folder"
    t.mkdir(parents=True)
    (t / "TASK_INSTRUCTIONS.txt").write_text("do it")
    return tmp_path / "base"


class _Sbx:
    def __init__(self, ws, *a, **k): self.ws = ws
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def run(self, cmd, timeout): return "yes\n" if "tesseract" in cmd else "no\n"


def test_run_task_reports_files_written_outside_output_and_never_collects_them(tmp_path, monkeypatch):
    import agent_harness.bench.jobbench as jb
    from agent_harness.loop import LoopResult
    base = _task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", _Sbx)
    seen = {}

    def fake(client, reg, system, task, **kw):
        seen.update(kw, system=system)
        reg.run("write_file", {"path": "/work/results.csv", "content": "a,b"})          # a deliverable-looking file in the wrong place
        return LoopResult(stop="blocked", turns=3, recovery={"exhausted": True, "blockers": [{"kind": "redundant"}]})
    monkeypatch.setattr(jb, "run_loop", fake)
    s = jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=1, context="basic",
                    recovery=True, capability_notes=True, web_paging=True, web_downloads=True)
    assert s["deliverables"] == [] and s["outside_output"] == ["results.csv"]          # reported, not silently collected
    assert s["stop"] == "blocked" and s["recovery"]["blockers"][0]["kind"] == "redundant"
    assert seen["recovery"] is not None and "tesseract" in seen["system"] and "web_download" in seen["system"]


def test_run_task_defaults_leave_behaviour_unchanged(tmp_path, monkeypatch):
    import agent_harness.bench.jobbench as jb
    from agent_harness.loop import LoopResult
    base = _task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", _Sbx)
    seen = {}
    monkeypatch.setattr(jb, "run_loop", lambda c, r, system, t, **kw: (seen.update(kw, system=system), LoopResult(stop="done"))[1])
    jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=1, context="basic")
    assert seen["recovery"] is None and "Environment facts" not in seen["system"]


# ---- hard stop disabled (arm C): guidance, classification and telemetry stay ---------------------------------------------------------
def test_without_a_hard_stop_the_run_is_never_ended_by_recovery_but_the_turn_limit_still_applies():
    pol = RecoveryPolicy(budget=None, checkpoint_after=3, progress_probe=lambda: 0)
    c = Script([call("probe", q="same")])
    res = run_loop(c, registry([]), "s", "t", max_turns=40, recovery=pol)
    assert res.stop == "max_turns" and res.turns == 40                               # only the overall limit ends it
    rep = res.recovery
    assert rep["hard_stop"] is False and rep["exhausted"] is False and rep["blocked_calls"] > 30 and rep["max_streak"] > 30
    assert rep["blockers"][0]["kind"] == "redundant"
    shown = [m["content"] for m in c.seen[-1] if m.get("role") == "tool"]
    assert any("[no progress]" in m for m in shown) and sum("[checkpoint]" in m for m in shown) == 1     # guidance continues, once-only checkpoint


def test_the_default_policy_still_has_the_hard_stop():
    assert RecoveryPolicy().budget == 20


def test_run_task_passes_no_budget_when_the_hard_stop_is_disabled(tmp_path, monkeypatch):
    import agent_harness.bench.jobbench as jb
    from agent_harness.loop import LoopResult
    base = _task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", _Sbx)
    seen = []
    monkeypatch.setattr(jb, "run_loop", lambda c, r, s, t, **kw: (seen.append(kw["recovery"]), LoopResult(stop="done"))[1])
    for stop in (True, False):
        jb.run_task(base, "x/task1", tmp_path / f"o{stop}", client=type("C", (), {"model": "m"})(), attempts=1, context="basic", recovery=True, recovery_hard_stop=stop)
    assert seen[0].budget == 20 and seen[1].budget is None
    summ = json.loads((tmp_path / "oFalse" / "summary.json").read_text())
    assert summ["run_options"]["recovery_hard_stop"] is False
