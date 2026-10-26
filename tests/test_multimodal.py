"""Image input: from a task file to the model's request, with blank/unreadable handling, history pruning, capability probing and request format."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import agent_harness.multimodal as mm
from agent_harness.bench.sandbox import RunResult
from agent_harness.bench.tools_parity import describe_bash, parity_registry
from agent_harness.integrity import check_message_integrity
from agent_harness.llm import ChatClient
from agent_harness.loop import Supervision, run_loop
from agent_harness.multimodal import (CONTROL_EXPECTED, PROBE_DIR, blankness, content_chars, estimate_image_tokens, image_message, is_image_message, load_image, probe_vision,
                                      prune_images, sniff_image)
from agent_harness.trace import TraceRecorder, message_text

PIL = pytest.importorskip("PIL.Image")
CONTROL, BLANK = (PROBE_DIR / "control.png").read_bytes(), (PROBE_DIR / "blank.png").read_bytes()


class FakeSbx:
    def run_raw(self, cmd, timeout=120):
        return RunResult("", "", 0)

    def run(self, cmd, timeout=120):
        return ""


def workspace(tmp_path):
    (tmp_path / "task_folder").mkdir()
    (tmp_path / "task_folder" / "form.png").write_bytes(CONTROL)
    (tmp_path / "task_folder" / "blank.png").write_bytes(BLANK)
    return tmp_path


class Script:
    def __init__(self, actions):
        self.actions, self.i, self.seen = list(actions), 0, []

    def complete(self, messages, tools=None):
        self.seen.append(json.loads(json.dumps(messages)))
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        msg = {"role": "assistant", "content": a} if isinstance(a, str) else {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{self.i}", "type": "function", "function": {"name": a[0], "arguments": json.dumps(a[1])}}]}
        return {"message": msg, "finish_reason": "stop", "usage": {"prompt_tokens": 500, "completion_tokens": 5}}


# ---- file facts -----------------------------------------------------------------------------------------------------------------------
def test_headers_give_type_and_size_for_each_supported_format(tmp_path):
    import io
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("GIF", "image/gif"), ("WEBP", "image/webp")):
        b = io.BytesIO()
        PIL.new("RGB", (37, 21), "red").save(b, fmt)
        info = sniff_image(b.getvalue())
        assert info and (info.mime, info.width, info.height) == (mime, 37, 21), fmt
    assert sniff_image(b"not an image at all") is None and sniff_image(b"") is None
    assert sniff_image(CONTROL).width == 600 and sniff_image(CONTROL).height == 400


def test_image_token_estimate_matches_the_measured_cost_of_the_known_image():
    assert abs(estimate_image_tokens(600, 400) - 230) <= 6                         # measured on the pinned server: 273 vs 43 prompt tokens
    assert content_chars([{"type": "text", "text": "abc"}, mm.ImageAttachment("x", CONTROL, sniff_image(CONTROL)).part()]) >= 3 * 200


def test_blank_and_faint_images_are_detected_and_a_real_one_is_not():
    assert blankness(BLANK) == "blank" and blankness(CONTROL) == "ok"
    import io
    img = PIL.new("L", (200, 200), 128)
    img.putpixel((5, 5), 131)
    b = io.BytesIO()
    img.save(b, "PNG")
    assert blankness(b.getvalue()) == "faint"


# ---- the view_image tool ------------------------------------------------------------------------------------------------------------------
def test_a_real_image_is_attached_and_delivered_in_a_user_message_after_the_tool_result(tmp_path):
    ws = workspace(tmp_path)
    reg = parity_registry(ws, FakeSbx(), vision=True)
    c = Script([("view_image", {"path": "/work/task_folder/form.png"}), "I read it."])
    res = run_loop(c, reg, "s", "t", supervision=None, check_integrity=True)
    req = c.seen[1]
    tool_i = next(i for i, m in enumerate(req) if m["role"] == "tool")
    assert "Image attached in the next message" in req[tool_i]["content"] and "600x400" in req[tool_i]["content"]
    img = req[tool_i + 1]
    assert img["role"] == "user" and isinstance(img["content"], list) and img["content"][0]["type"] == "text"
    part = img["content"][1]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")      # the exact format the server accepted in X0
    assert base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) == CONTROL                          # byte for byte the task file
    assert res.stop == "done" and check_message_integrity(req) == []


def test_blank_images_are_reported_not_attached_and_the_model_is_told_not_to_guess(tmp_path):
    ws = workspace(tmp_path)
    reg = parity_registry(ws, FakeSbx(), vision=True)
    r = reg.run("view_image", {"path": "/work/task_folder/blank.png"})
    assert r.ok and not r.attachments and "nothing to read" in r.text and "do not infer or guess" in r.text


def test_bad_inputs_fail_clearly(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    (ws / "note.txt").write_text("hello")
    reg = parity_registry(ws, FakeSbx(), vision=True)
    assert not reg.run("view_image", {"path": "/work/note.txt"}).ok and "not a PNG" in reg.run("view_image", {"path": "/work/note.txt"}).text
    assert not reg.run("view_image", {"path": "/work/missing.png"}).ok
    assert not reg.run("view_image", {"path": "/work/../../etc/hosts"}).ok                # confined to the workspace
    monkeypatch.setattr(mm, "MAX_IMAGE_BYTES", 100)
    big = reg.run("view_image", {"path": "/work/task_folder/form.png"})
    assert not big.attachments and "not attached" in big.text


def test_view_image_exists_only_when_the_model_is_known_to_see_and_the_descriptions_say_so(tmp_path):
    ws = workspace(tmp_path)
    off = parity_registry(ws, FakeSbx(), vision=False)
    on = parity_registry(ws, FakeSbx(), vision=True)
    names = lambda r: {sp["function"]["name"] for sp in r.specs()}
    assert "view_image" not in names(off) and "view_image" in names(on)
    assert "You cannot view images yourself" in describe_bash("egress", [], vision=False)
    assert "view_image" in describe_bash("egress", [], vision=True) and "alternative" in describe_bash("egress", ["tesseract"], vision=True)


# ---- history ----------------------------------------------------------------------------------------------------------------------------------
def test_old_images_are_pruned_in_place_and_the_newest_stay(tmp_path):
    info = sniff_image(CONTROL)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(4):
        msgs.append(image_message([mm.ImageAttachment(f"/work/{i}.png", CONTROL, info)]))
    assert prune_images(msgs, keep=2) == 2
    assert [is_image_message(m) for m in msgs[2:]] == [False, False, True, True]
    assert len(msgs) == 6 and "call view_image on the file again" in msgs[2]["content"][-1]["text"] and msgs[2]["role"] == "user"


def test_the_loop_prunes_images_when_the_context_is_over_the_compaction_threshold(tmp_path):
    ws = workspace(tmp_path)
    reg = parity_registry(ws, FakeSbx(), vision=True)
    acts = [("view_image", {"path": "/work/task_folder/form.png"}), ("view_image", {"path": "/work/task_folder/form.png"}), ("view_image", {"path": "/work/task_folder/form.png"}), "ok"]
    c = Script(acts)
    res = run_loop(c, reg, "s", "t", max_turns=10, max_context_tokens=1000, supervision=None, keep_images=1)           # 500-token prompts trigger compaction at 70% of 1000
    last = c.seen[-1]
    assert sum(is_image_message(m) for m in last) <= 2 and any("removed from the history" in json.dumps(m) for m in last)
    assert check_message_integrity(last) == []


def test_the_trace_records_images_by_hash_not_as_base64(tmp_path):
    ws = workspace(tmp_path)
    msg = image_message([mm.ImageAttachment("/work/form.png", CONTROL, sniff_image(CONTROL))])
    text = message_text(msg)
    assert "base64" not in text and "sha256" in text and len(text) < 600
    rec = TraceRecorder(tmp_path / "trace")
    reg = parity_registry(ws, FakeSbx(), vision=True)
    run_loop(Script([("view_image", {"path": "/work/task_folder/form.png"}), "ok"]), reg, "s", "t", supervision=None, trace=rec)
    blobs = "".join(p.read_text() for p in (tmp_path / "trace" / "blobs").glob("*.txt"))
    assert "base64," not in blobs


# ---- capability: measured, never assumed ---------------------------------------------------------------------------------------------------------
def _completer(control_reply, blank_reply):
    def complete(messages):
        url = messages[0]["content"][0]["image_url"]["url"]
        return control_reply if base64.b64decode(url.split(",", 1)[1]) == CONTROL else blank_reply
    return complete


def test_probe_accepts_a_model_that_reads_the_control_and_says_nothing_for_the_blank():
    cap = probe_vision(_completer("Reference: ZK-6305\nColour: TEAL\nQuantity: 61", "The image is blank."))
    assert cap.supported and cap.source == "probe" and cap.evidence["control_read_exactly"]


@pytest.mark.parametrize("control,blank,why", [("Reference: ZK-6305", "x", "control_read_exactly"),                           # reads only part of it
                                              ("Reference: ZK-6305\nColour: TEAL\nQuantity: 61", "Reference: ZK-6305 Colour: TEAL", "blank_leaked_control_values"),
                                              ("I cannot see an image.", "I cannot see an image.", "control_read_exactly")])
def test_probe_rejects_partial_reads_and_models_that_invent_values_for_a_blank_image(control, blank, why):
    cap = probe_vision(_completer(control, blank))
    assert not cap.supported
    assert cap.evidence["control_read_exactly"] is False if why == "control_read_exactly" else cap.evidence["blank_leaked_control_values"] is True


def test_probe_records_a_rejected_image_as_no_vision_with_the_reason():
    def reject(messages):
        raise RuntimeError("HTTP 400: image input is not supported by this model")
    cap = probe_vision(reject)
    assert not cap.supported and "not supported" in cap.evidence["error"]


def test_probe_fixtures_are_the_known_images():
    assert CONTROL_EXPECTED == {"reference": "ZK-6305", "colour": "TEAL", "quantity": "61"} and sniff_image(CONTROL) and blankness(BLANK) == "blank"


# ---- request format on the wire ------------------------------------------------------------------------------------------------------------------
def test_the_chat_client_posts_image_parts_in_the_openai_format():
    got = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_POST(self):
            got["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}, image_message([mm.ImageAttachment("/work/f.png", CONTROL, sniff_image(CONTROL))])]
        ChatClient(f"http://127.0.0.1:{srv.server_address[1]}", "m").complete(msgs, None)
    finally:
        srv.shutdown()
    part = got["body"]["messages"][2]["content"][1]
    assert part["type"] == "image_url" and list(part["image_url"]) == ["url"] and part["image_url"]["url"].startswith("data:image/png;base64,")


# ---- run_task wiring: configuration or probe, recorded ------------------------------------------------------------------------------------------------
class RunSbx(FakeSbx):
    def __init__(self, ws, *a, **k): self.ws = ws
    def __enter__(self): return self
    def __exit__(self, *e): pass


def _task(tmp_path):
    t = tmp_path / "base" / "x" / "task1" / "task_folder"
    t.mkdir(parents=True)
    (t / "TASK_INSTRUCTIONS.txt").write_text("Read form.png and save the result.")
    (t / "form.png").write_bytes(CONTROL)
    return tmp_path / "base"


@pytest.mark.parametrize("mode,expect", [("off", False), ("on", True), ("probe", True)])
def test_run_task_registers_view_image_from_configuration_or_a_probe_and_records_how_it_knew(tmp_path, monkeypatch, mode, expect):
    import agent_harness.bench.jobbench as jb
    from agent_harness.loop import LoopResult
    base = _task(tmp_path)
    monkeypatch.setattr(jb, "DockerSandbox", RunSbx)
    seen = {}
    monkeypatch.setattr(jb, "run_loop", lambda c, r, s, t, **kw: (seen.update(names=[sp["function"]["name"] for sp in r.specs()]), LoopResult(stop="done"))[1])
    probes = []

    class Client:
        model = "m"

        def complete(self, messages, tools=None):
            probes.append(messages)
            url = messages[0]["content"][0]["image_url"]["url"]
            reply = "Reference: ZK-6305\nColour: TEAL\nQuantity: 61" if base64.b64decode(url.split(",", 1)[1]) == CONTROL else "blank"
            return {"message": {"role": "assistant", "content": reply}, "finish_reason": "stop", "usage": {}}
    jb.run_task(base, "x/task1", tmp_path / "out", Client(), attempts=1, context="basic", tools="parity", network="egress", vision=mode, vision_evidence="X0 2026-10-25")
    assert ("view_image" in seen["names"]) == expect
    cfg = json.loads((tmp_path / "out" / "tool_schemas.json").read_text())["vision"]
    assert cfg["supported"] == expect and cfg["source"] == {"off": "assumed-none", "on": "configuration", "probe": "probe"}[mode]
    assert len(probes) == (2 if mode == "probe" else 0)
    if mode == "on":
        assert cfg["evidence"] == {"evidence": "X0 2026-10-25"}
    with pytest.raises(ValueError):
        jb.run_task(base, "x/task1", tmp_path / "out2", Client(), attempts=1, context="basic", vision="maybe")
