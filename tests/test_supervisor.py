"""The execution supervisor: repeated operations, stalls, recovery, budget, completion; with the real Context Runtime where context state matters.

The six deterministic reproductions from the cross-repository audit (R1-R6) are permanent tests here, together with the cases that must NOT be blocked.
Tests marked with the ``cr`` fixture need the Context Runtime branch on the path and are skipped without it.
"""
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from agent_harness.completion import check_deliverables, check_file, expected_artifacts
from agent_harness.integrity import check_message_integrity
from agent_harness.loop import Supervision, run_loop
from agent_harness.supervisor import Supervisor, SupervisorConfig
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool


# ---- helpers ------------------------------------------------------------------------------------------------------------------
class Script:
    """A fake ChatClient replaying scripted assistant actions: (name, args) for a tool call, a str for a final text, or a callable(messages) -> action."""

    def __init__(self, actions, prompt_tokens=100):
        self.actions, self.i, self.prompt_tokens, self.seen = list(actions), 0, prompt_tokens, []

    def complete(self, messages, tools=None):
        self.seen.append(json.loads(json.dumps(messages)))
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        if callable(a):
            a = a(messages)
        if isinstance(a, str):
            msg = {"role": "assistant", "content": a}
        else:
            name, args = a
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{self.i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        return {"message": msg, "finish_reason": "stop", "usage": {"prompt_tokens": self.prompt_tokens + self.i * 20, "completion_tokens": 5}}


def _as_result(v):
    return v if isinstance(v, ToolResult) else ToolResult(text=str(v))


def registry(**fns):
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=list(fns)))
    for name, fn in fns.items():
        reg.register(function_tool(name, lambda a, fn=fn: _as_result(fn(a)), name,
                                   {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}, side_effecting=True))
    return reg


class WS:
    """A fake workspace: path -> content, observed by content digest like the real probe."""

    def __init__(self):
        self.files = {}

    def snapshot(self):
        return {p: hashlib.sha1(c.encode()).hexdigest()[:12] for p, c in self.files.items()}


def sup_cfg(**kw):
    return SupervisorConfig(**kw)


def run(actions, reg, ws=None, checker=None, cfg=None, max_turns=60, **kw):
    sup = Supervision(cfg or SupervisorConfig(), probe=ws.snapshot if ws else None, checker=checker)
    c = Script(actions)
    res = run_loop(c, reg, "sys", "task", max_turns=max_turns, supervision=sup, check_integrity=True, **kw)
    return res, c


def tool_texts(c):
    return [m["content"] for m in c.seen[-1] if m["role"] == "tool"]


# ---- R3: near-identical loops are caught by family, bounded, and end accurately ------------------------------------------------
def test_r3_near_identical_loop_is_suppressed_then_recovery_is_exhausted_and_the_run_ends_blocked():
    n = {"i": 0}

    def peek(a):
        n["i"] += 1
        return f"ascii art rows {n['i']}\n" + "#" * (n["i"] % 7)                     # a different result every time, never an error

    acts = [("bash", {"q": f"python3 crop.py --box {i},{i + 10},{i},{i + 10}"}) for i in range(100)]
    res, c = run(acts, registry(bash=peek), max_turns=200)
    rep = res.supervisor
    assert res.stop == "blocked" and res.turns <= 40                                 # ten executions, suppression, 30 idle turns without progress, then the end
    assert n["i"] == 10 + rep["counts"]["probes"] and rep["counts"]["probes"] >= 1 and rep["exhausted"] and rep["counts"]["suppressed"] >= 6
    assert rep["families"][0]["state"] == "suppressed" and res.turns < 100


def test_r3_a_different_operation_between_requests_is_never_blocked_and_postpones_exhaustion():
    acts = []
    for i in range(14):
        acts.append(("bash", {"q": f"python3 crop.py --box {i}"}))
    acts += [("read", {"q": "another thing"}), ("bash", {"q": "python3 crop.py --box 99"})] * 3 + ["finished"]
    res, c = run(acts, registry(bash=lambda a: f"out {a['q']}", read=lambda a: "ok"), max_turns=100)
    assert res.stop == "done" and res.supervisor["counts"]["suppressed"] >= 3 and not res.supervisor["exhausted"]


def test_advisory_mode_records_exhaustion_without_ending_the_run():
    acts = [("peek", {"q": "x"})] * 60 + ["ok"]
    res, c = run(acts, registry(peek=lambda a: "same"), cfg=sup_cfg(end_on_exhaustion=False), max_turns=80)
    assert res.stop == "done" and not res.supervisor["exhausted"] and res.supervisor["counts"]["exhaustion_reached"] >= 1
    assert any(e["kind"] == "exhausted" and e["ends_run"] is False for e in res.supervisor["events"])


# ---- exact repeats --------------------------------------------------------------------------------------------------------------
def test_exact_repeated_call_is_not_run_a_third_time_and_the_refusal_is_short_and_bounded():
    calls = {"n": 0}

    def peek(a):
        calls["n"] += 1
        return "same"
    res, c = run([("peek", {"q": "x"})] * 40, registry(peek=peek), max_turns=60)
    outs = tool_texts(c)
    assert calls["n"] == 2 + res.supervisor["counts"]["probes"] and res.stop == "blocked"
    text_len = sum(len(o) for o in outs if o.startswith("[supervisor]"))
    assert text_len < 3000                                                           # recovery messages do not accumulate (R6): two informative notes, the rest one-line stubs
    assert sum(1 for o in outs if o.startswith("[supervisor] Not run")) == 2 and sum(1 for o in outs if o.startswith("[supervisor] not run")) >= 3


def test_r6_150_repeat_requests_never_put_150_refusals_into_the_history():
    res, c = run([("peek", {"q": "x"})] * 150, registry(peek=lambda a: "same"), max_turns=300)
    msgs = [m for m in c.seen[-1] if m["role"] == "tool"]
    assert res.turns <= 40 and len(msgs) <= 40                                       # the run ends instead of looping 150 times
    stub_chars = sum(len(m["content"]) for m in msgs if m["content"].startswith("[supervisor] not run"))
    assert stub_chars < 2500


# ---- legitimate repetition is allowed --------------------------------------------------------------------------------------------
def test_polling_with_changing_results_is_allowed_until_the_changing_threshold():
    state = {"n": 0}

    def poll(a):
        state["n"] += 1
        return f"status {state['n']}%"
    res, c = run([("poll", {"q": "job"})] * 8 + ["done"], registry(poll=poll))
    assert res.stop == "done" and state["n"] == 8 and res.supervisor["counts"]["suppressed"] == 0
    assert res.supervisor["counts"]["watch_notes"] == 1                              # one concise note at the sixth run, nothing refused


def test_editing_a_file_the_command_uses_makes_the_same_command_runnable_again():
    ws = WS()
    ws.files["build.py"] = "v1"
    out = {"v": "ERROR: boom"}

    def bash(a):
        return out["v"]

    def edit(a):
        ws.files["build.py"] = a["q"]
        out["v"] = "ERROR: boom"                                                    # still failing: the SAME error text
        return "edited"
    acts = [("bash", {"q": "python3 build.py"}), ("bash", {"q": "python3 build.py"}), ("edit", {"q": "v2"}), ("bash", {"q": "python3 build.py"}), "ok"]
    res, c = run(acts, registry(bash=bash, edit=edit), ws=ws)
    assert res.stop == "done" and res.supervisor["counts"]["suppressed"] == 0 and res.supervisor["counts"]["executed"] == 4


def test_debugging_a_script_through_changing_errors_is_progress_not_a_loop():
    errs = iter([f"Traceback (most recent call last): NameError: name 'v{i}' is not defined" for i in range(12)])
    acts = [("bash", {"q": "python3 job.py"})] * 12 + ["ok"]
    res, c = run(acts, registry(bash=lambda a: next(errs)))
    assert res.stop == "done" and res.supervisor["counts"]["suppressed"] == 0


# ---- R4: what counts as progress ------------------------------------------------------------------------------------------------
def test_r4_rewriting_identical_scratch_bytes_each_run_is_not_progress():
    ws = WS()

    def build(a):
        ws.files["scratch/out.bin"] = "same bytes"                                  # the command "writes" every time, identically
        return "built"
    res, c = run([("bash", {"q": "python3 build.py"})] * 80, registry(bash=build), ws=ws, max_turns=100)
    assert res.stop == "blocked" and res.supervisor["counts"]["executed"] == 2 + res.supervisor["counts"]["probes"]


def test_r4_a_command_that_changes_the_deliverable_each_run_is_progress():
    ws = WS()
    k = {"i": 0}

    def build(a):
        k["i"] += 1
        ws.files["output/report.txt"] = f"revision {k['i']}"
        return "built"
    res, c = run([("bash", {"q": "python3 build.py"})] * 6 + ["done"], registry(bash=build), ws=ws)
    assert res.stop == "done" and res.supervisor["counts"]["suppressed"] == 0


def test_background_filesystem_changes_do_not_count_as_progress():
    ws = WS()
    k = {"i": 0}

    def tick(a):
        k["i"] += 1
        ws.files[f".cache/tmp{k['i']}"] = "noise"                                 # hidden directory: ignored zone
        ws.files["task_folder/input.csv"] = f"x{k['i']}"                            # inputs: ignored zone
        return "same"
    res, c = run([("bash", {"q": "ls"})] * 80, registry(bash=tick), ws=ws, max_turns=100)
    assert res.stop == "blocked" and res.supervisor["counts"]["executed"] == 2 + res.supervisor["counts"]["probes"]


# ---- R5: tool output text is never reinterpreted -------------------------------------------------------------------------------
def test_r5_output_text_that_mentions_a_network_error_is_passed_through_unchanged():
    msg = "Notes: the log said 'Temporary failure in name resolution' last week."
    res, c = run([("read", {"q": "notes"}), "ok"], registry(read=lambda a: msg))
    assert tool_texts(c) == [msg]


# ---- completion and budget ---------------------------------------------------------------------------------------------------------
def make_xlsx(path, cells=("hello", "world")):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("xl/workbook.xml", "<workbook><sheets><sheet name='S1'/></sheets></workbook>")
        z.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData>" + "".join(f"<row><c t='inlineStr'><is><t>{c}</t></is></c></row>" for c in cells) + "</sheetData></worksheet>")


def test_empty_or_invalid_deliverables_are_never_structurally_valid(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    assert check_deliverables(out).status == "none"
    (out / "draft_notes.md").write_text("x" * 100)
    assert check_deliverables(out).status == "draft_only"
    (out / "empty.csv").write_text("")
    (out / "bad.xlsx").write_bytes(b"not a zip")
    (out / "hollow.xlsx").write_bytes(b"")
    con = sqlite3.connect(out / "county.db")
    con.execute("create table t (a int)")
    con.commit()
    con.close()
    rep = check_deliverables(out)
    assert rep.status == "invalid" and {f.name for f in rep.files if f.status == "invalid"} == {"empty.csv", "bad.xlsx", "hollow.xlsx", "county.db"}
    assert not rep.ok and "database has no rows" in rep.issues()


def test_a_populated_workbook_and_a_matching_task_text_give_structurally_valid(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    make_xlsx(out / "Report_Q3.xlsx")
    rep = check_deliverables(out, "Save the final workbook as Report_Q3.xlsx in the output folder.", ["input.csv"])
    assert rep.status == "structurally_valid" and rep.required_exts == ["xlsx"] and rep.unmatched_names == []
    make_xlsx(out / "Hollow.xlsx", cells=())
    assert check_file(out / "Hollow.xlsx").status == "invalid"
    only_wrong_type = check_deliverables(out, "Deliver Summary.docx and Report_Q3.xlsx", [])
    assert only_wrong_type.status == "partial" and only_wrong_type.missing_exts == ["docx"]


def test_expected_artifacts_come_from_delivery_lines_and_exclude_inputs():
    text = "Inputs: Source_Orders.xlsx and notes.txt.\nDeliver a reconciled workbook called Reconciled_Master.xlsx and a memo \"Board Memo.docx\".\nThe source file Source_Orders.xlsx must not be edited."
    assert expected_artifacts(text, ["Source_Orders.xlsx", "notes.txt"]) == ["Reconciled_Master.xlsx", "Board Memo.docx"]


def test_model_that_stops_with_no_valid_output_is_turned_back_at_most_twice_and_the_status_is_truthful(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    checker = lambda: check_deliverables(out, "Deliver Report.xlsx", [])
    res, c = run(["I am done.", "Really done.", "Finished.", "Final."], registry(), checker=checker)
    users = [m["content"] for m in c.seen[-1] if m["role"] == "user"]
    assert res.stop == "done" and res.supervisor["pushes"] == 2 and sum(u.startswith("[harness check]") for u in users) == 2
    assert res.supervisor["completion"]["status"] == "none"                          # accepted, but never reported as complete
    assert "does not judge correctness" in users[-1]


def test_a_model_that_delivers_a_valid_file_can_stop_immediately(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    ws_checker = lambda: check_deliverables(out, "Deliver Report.xlsx", [])
    res, c = run([("save", {"q": "go"}), "All done."], registry(save=lambda a: (make_xlsx(out / "Report.xlsx"), "saved")[1]), checker=ws_checker)
    assert res.stop == "done" and res.supervisor["pushes"] == 0 and res.supervisor["completion"]["status"] == "structurally_valid"


def test_valid_output_before_exhaustion_is_preserved_and_the_verified_run_ends_early_without_claiming_correctness(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    ws = WS()

    def save(a):
        make_xlsx(out / "Report.xlsx")
        ws.files["output/Report.xlsx"] = "v1"
        return "saved"
    checker = lambda: check_deliverables(out, "Deliver Report.xlsx", [])
    acts = [("save", {"q": "go"})] + [("explore", {"q": f"look {i}"}) for i in range(60)]
    res, c = run(acts, registry(save=save, explore=lambda a: f"seen {a['q']}"), ws=ws, checker=checker, cfg=sup_cfg(verified_idle_notice=3, verified_idle_end=5), max_turns=100)
    assert res.stop == "verified_complete" and res.turns < 12 and (out / "Report.xlsx").exists()
    assert any("does not judge correctness" in t for t in tool_texts(c))
    assert not any("success" in t.lower() or "complete" in t.lower() and "verified" not in t.lower() for t in tool_texts(c) if t.startswith("[budget]"))


def test_without_named_outputs_the_run_is_never_ended_by_the_supervisor_even_if_a_file_is_valid(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    make_xlsx(out / "early_result.xlsx")
    checker = lambda: check_deliverables(out, "Analyse the data and report.", [])
    res, c = run([("explore", {"q": f"x{i}"}) for i in range(30)] + ["ok"], registry(explore=lambda a: f"r {a['q']}"), checker=checker, cfg=sup_cfg(verified_idle_end=3), max_turns=80)
    assert res.stop == "done"                                                        # the model decides when to stop


def test_budget_notices_reserve_time_for_delivery_and_never_claim_success(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    checker = lambda: check_deliverables(out, "Deliver Report.xlsx", [])
    res, c = run([("explore", {"q": f"x{i}"}) for i in range(30)] + ["ok"], registry(explore=lambda a: f"r {a['q']}"), checker=checker, max_turns=30,
                 cfg=sup_cfg(reserve_turns=6, checkpoint_pushes=0))
    notices = [t for t in tool_texts(c) if "[budget]" in t]
    assert any("no file is in /work/output yet" in t for t in notices)
    assert any("Budget nearly used up" in t and "Stop exploring" in t for t in notices)
    assert all("structural" in t for t in notices) and not any("passes" in t and "no file" in t for t in notices)


def test_the_reserve_for_delivery_is_not_spent_on_new_exploration_but_building_is_allowed(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    checker = lambda: check_deliverables(out, "Deliver Report.xlsx", [])
    n, ws = {}, WS()

    def save(a):
        make_xlsx(out / "Report.xlsx")
        ws.files["output/Report.xlsx"] = "v1"
        return "saved"
    acts = [("web_fetch", {"url": f"https://x.org/{i}", "q": f"u{i}"}) for i in range(12)] + [("web_fetch", {"url": "https://x.org/late", "q": "late"}), ("bash", {"q": "build"}),
                                                                                              ("web_fetch", {"url": "https://x.org/after", "q": "after valid"}), "ok"]
    r = registry(web_fetch=lambda a: n.setdefault("f", []).append(a["q"]) or f"page {a['q']}", bash=lambda a: save(a))
    res, c = run(acts, r, ws=ws, checker=checker, cfg=sup_cfg(reserve_turns=4), max_turns=16)
    assert res.supervisor["counts"]["reserve_refusals"] >= 1 and "late" not in n["f"]              # refused inside the reserve while nothing valid exists
    assert any("reserve for building and checking" in t for t in tool_texts(c))
    assert "after valid" in n["f"]                                                                     # once the output is valid, exploration is allowed again


def test_incomplete_work_is_preserved_when_the_run_ends_blocked(tmp_path):
    ws = WS()
    out = tmp_path / "output"
    out.mkdir()
    (out / "draft_partial.csv").write_text("a,b\n1,2\n")
    checker = lambda: check_deliverables(out, "Deliver Final.csv", [])
    res, c = run([("peek", {"q": "x"})] * 30, registry(peek=lambda a: "same"), ws=ws, checker=checker, max_turns=40)
    assert res.stop == "blocked" and (out / "draft_partial.csv").exists()
    assert res.supervisor["completion"]["status"] == "draft_only"                    # reported honestly: a draft is not a deliverable


# ---- message integrity -------------------------------------------------------------------------------------------------------------
def test_integrity_checker_accepts_sound_histories_and_names_each_defect():
    ok = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"},
          {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
          {"role": "tool", "tool_call_id": "a", "content": "r"}, {"role": "assistant", "content": "done"}]
    assert check_message_integrity(ok) == []
    assert check_message_integrity(ok[:3])                                             # call without a result
    assert check_message_integrity(ok[:2] + [ok[3]])                                   # result without a call
    dup = ok[:4] + [ok[2], ok[3]]
    assert any("duplicate" in p for p in check_message_integrity(dup))
    assert check_message_integrity([ok[1], ok[0]])                                     # wrong start
    img_in_tool = ok[:3] + [{"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "x"}]}]
    assert any("tool content must be text" in p or "content parts" in p for p in check_message_integrity(img_in_tool))


def test_a_corrupt_history_is_never_sent_to_the_model():
    class Bad(Script):
        pass
    reg = registry(peek=lambda a: "x")
    c = Script([("peek", {"q": "a"}), "ok"])
    mgr_msgs = {}

    class Breaker:
        def prepare(self, messages, *, target_tokens, estimated_tokens):
            messages.pop(2)                                                           # corrupt the history: drop an assistant call whose result remains
            return 1
        def on_context_error(self, messages, attempt): return 0
        def admit(self, tool, text, max_chars): return text
        def tools(self): return []
    res = run_loop(c, reg, "s", "t", context_manager=Breaker(), max_context_tokens=100, supervision=None, check_integrity=True)
    assert res.stop == "integrity_error" and len(c.seen) == 1                         # stopped before the second request
