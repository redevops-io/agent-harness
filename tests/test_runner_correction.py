"""run_task end of the correction: honest completion status, preserved incomplete work, supervision wiring, and the opt-in live vision check."""
import base64
import json
import os
from pathlib import Path

import pytest

import agent_harness.bench.jobbench as jb
from agent_harness.loop import LoopResult
from agent_harness.multimodal import PROBE_DIR


class Sbx:
    def __init__(self, ws, *a, **k): self.ws = ws
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def run(self, c, t=120): return ""
    def run_raw(self, c, t=120):
        from agent_harness.bench.sandbox import RunResult
        return RunResult("", "", 0)


def task(tmp_path, text="Deliver Final_Report.xlsx summarising the data."):
    t = tmp_path / "base" / "x" / "task1" / "task_folder"
    t.mkdir(parents=True)
    (t / "TASK_INSTRUCTIONS.txt").write_text(text)
    return tmp_path / "base"


def fake_loop(write, result):
    def run(client, reg, system, prompt, **kw):
        write(reg)
        return result
    return run


def test_a_blocked_run_with_only_a_draft_is_reported_as_blocked_not_delivered_and_its_work_is_preserved(tmp_path, monkeypatch):
    base = task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", Sbx)

    def write(reg):
        reg.run("write_file", {"path": "/work/output/draft_partial.csv", "content": "a,b\n1,2\n"})
        reg.run("write_file", {"path": "/work/scratch/analysis.py", "content": "print('half done')"})
    monkeypatch.setattr(jb, "run_loop", fake_loop(write, LoopResult(stop="blocked", turns=60, supervisor={"exhausted": True, "events": [{"kind": "exhausted"}]})))
    s = jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=1, context="basic", supervise=True)
    assert s["stop"] == "blocked" and s["completion"]["status"] == "draft_only"
    assert s["failure"]["class"] == "invalid_deliverable"                                # a file exists, but not one that counts
    assert s["deliverables"] == ["draft_partial.csv"] and (tmp_path / "out" / "output" / "draft_partial.csv").exists()     # nothing was deleted
    kept = tmp_path / "out" / s["incomplete_preserved"]
    assert (kept / "scratch" / "analysis.py").read_text() == "print('half done')"        # the unfinished script survives the workspace
    assert s["run_options"]["supervise"] is True and s["supervisor"]["exhausted"] is True


def test_a_run_that_ended_normally_with_the_wrong_kind_of_file_is_reported_as_partial_not_delivered(tmp_path, monkeypatch):
    base = task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", Sbx)

    def run(client, reg, system, prompt, **kw):
        reg.run("write_file", {"path": "/work/output/Final_Report.csv", "content": "x,y\n1,2\n"})
        return LoopResult(stop="done", turns=5)
    monkeypatch.setattr(jb, "run_loop", run)
    s = jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=1, context="basic")
    assert s["completion"]["status"] == "partial" and s["completion"]["missing_exts"] == ["xlsx"]   # the task asked for an .xlsx and only a .csv exists
    assert s["failure"]["class"] == "invalid_deliverable"


def test_supervision_is_wired_with_the_task_text_and_content_based_probe(tmp_path, monkeypatch):
    base = task(tmp_path, "Create Memo.docx and Data.xlsx. Inputs: input.csv")
    (base / "x" / "task1" / "task_folder" / "input.csv").write_text("a\n1\n")
    monkeypatch.setattr(jb, "DockerSandbox", Sbx)
    seen = {}

    def loop(c, r, s, t, **kw):
        sv = kw["supervision"]
        if sv is not None:                                                               # observe while the workspace still exists
            seen.update(cfg=sv.cfg, integrity=kw["check_integrity"], rep=sv.checker(), snap=sv.probe())
        seen["supervision"] = sv
        return LoopResult(stop="done")
    monkeypatch.setattr(jb, "run_loop", loop)
    jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=1, context="basic", supervise=True, supervisor_config={"exhaust_after": 9}, check_integrity=True)
    assert seen["cfg"].exhaust_after == 9 and seen["integrity"] is True
    rep, snap = seen["rep"], seen["snap"]
    assert rep.required_exts == ["docx", "xlsx"] and rep.status == "none"                # outputs named in the task text; inputs excluded
    assert "task_folder/input.csv" in snap and all(len(v) >= 8 for v in snap.values())   # digests, not mtimes
    jb.run_task(base, "x/task1", tmp_path / "out2", client=type("C", (), {"model": "m"})(), attempts=1, context="basic")
    assert seen["supervision"] is None                                                   # arm definitions that predate the supervisor stay a bare loop


def test_parity_tools_receive_the_loops_admission_budget(tmp_path, monkeypatch):
    base = task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", Sbx)
    seen = {}
    monkeypatch.setattr(jb, "run_loop", lambda c, r, s, t, **kw: (seen.update(read=r.run("read_file", {"path": "/work/task_folder/TASK_INSTRUCTIONS.txt"})), LoopResult(stop="done"))[1])
    import agent_harness.bench.tools_parity as tp
    captured = {}
    real = tp.parity_registry
    monkeypatch.setattr(tp, "parity_registry", lambda *a, **k: (captured.update(k), real(*a, **k))[1])
    jb.run_task(base, "x/task1", tmp_path / "out", client=type("C", (), {"model": "m"})(), attempts=1, context="basic", tools="parity", network="egress", max_context_tokens=40960)
    assert captured["output_budget"] == 8192 and captured["vision"] is False             # min(12000, 20% of the window), same formula as run_loop


# ---- the known order form, a blank image and a control image against a live endpoint (opt-in; never run in CI or without approval) -------------------------------------
ORDER_FORM = Path(os.environ.get("JOBBENCH_TASKS", "")) / "data_entry_keyers/task1/task_folder/Order_Form_ORD_202507_0001.png"
live = pytest.mark.skipif(os.environ.get("AGENT_HARNESS_VISION_LIVE_URL") is None, reason="set AGENT_HARNESS_VISION_LIVE_URL (and JOBBENCH_TASKS) to run against a model")


@live
@pytest.mark.skipif(not ORDER_FORM.exists(), reason="the benchmark dataset is not available")
def test_live_order_form_control_and_blank_through_the_real_endpoint():
    from agent_harness.llm import ChatClient
    from agent_harness.multimodal import CONTROL_EXPECTED, image_message, load_image
    client = ChatClient(os.environ["AGENT_HARNESS_VISION_LIVE_URL"], os.environ.get("AGENT_HARNESS_VISION_LIVE_MODEL", "Qwen3.8-27B"), temperature=0.55, max_tokens=700)

    def read(path, shown):
        att, why = load_image(Path(path), shown)
        if att is None:
            return why
        msg = image_message([att])
        msg["content"].append({"type": "text", "text": "Transcribe every field as 'label: value'. Do not guess."})
        return client.complete([msg], None)["message"]["content"]
    form = read(ORDER_FORM, "form.png")
    for v in ("ORD-202507-0001", "Acme Corporation", "250", "12.50", "3125.00"):
        assert v in form                                                                  # the known form, field by field
    control = read(PROBE_DIR / "control.png", "control.png")
    assert all(v.lower() in control.lower() for v in CONTROL_EXPECTED.values())
    blank = read(PROBE_DIR / "blank.png", "blank.png")                                    # never sent: reported as blank by the harness
    assert "nothing to read" in blank
