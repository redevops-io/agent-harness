"""The per-prompt image limit is an invariant of every model request: counted in image parts, enforced regardless of token pressure, learned from the endpoint if
unknown, and never silently dropping the images of the current turn. The first v2 data-entry rerun died on 'At most 5 image(s) may be provided in one prompt'."""
import io
import json

import pytest

import agent_harness.multimodal as mm
from agent_harness.bench.failure_class import classify
from agent_harness.bench.sandbox import RunResult
from agent_harness.bench.tools_parity import parity_registry
from agent_harness.llm import ChatError
from agent_harness.loop import Supervision, run_loop
from agent_harness.multimodal import ImageCapError, count_images, enforce_image_cap, image_message, is_image_message, parse_image_limit, sniff_image

PIL = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def png(i):
    im = PIL.new("RGB", (120, 80), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([5 + i * 7, 10, 40 + i * 7, 60], fill="black")
    d.text((10, 65), f"form {i}", fill="black")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def att(i):
    data = png(i)
    return mm.ImageAttachment(f"/work/form{i}.png", data, sniff_image(data))


class FakeSbx:
    def run_raw(self, cmd, timeout=120):
        return RunResult("", "", 0)

    def run(self, cmd, timeout=120):
        return ""


def workspace(tmp_path, n):
    (tmp_path / "task_folder").mkdir()
    for i in range(n):
        (tmp_path / "task_folder" / f"form{i}.png").write_bytes(png(i))
    return tmp_path


class StrictEndpoint:
    """Behaves like the pinned vLLM deployment: HTTP 400 when a prompt holds more than ``limit`` images."""
    def __init__(self, actions, limit=5):
        self.actions, self.limit, self.i, self.seen, self.rejected = list(actions), limit, 0, [], 0

    def complete(self, messages, tools=None):
        n = count_images(messages)
        self.seen.append((n, json.loads(json.dumps(messages))))
        if n > self.limit:
            self.rejected += 1
            raise ChatError(f"HTTP 400: At most {self.limit} image(s) may be provided in one prompt. (parameter=image)")
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        if isinstance(a, str):
            msg = {"role": "assistant", "content": a}
        else:
            calls = a if isinstance(a, list) else [a]
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{self.i}_{k}", "type": "function", "function": {"name": c[0], "arguments": json.dumps(c[1])}}
                                                                    for k, c in enumerate(calls)]}
        return {"message": msg, "finish_reason": "stop", "usage": {"prompt_tokens": 500, "completion_tokens": 5}}


def view(i):
    return ("view_image", {"path": f"/work/task_folder/form{i}.png"})


def base():
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]


# ---- the enforcement function ---------------------------------------------------------------------------------------------------------------
def test_six_images_across_three_messages_keep_the_newest_five_and_label_the_dropped_one():
    msgs = base() + [image_message([att(0), att(1)]), image_message([att(2), att(3)]), image_message([att(4), att(5)])]
    n_msgs, roles = len(msgs), [m["role"] for m in msgs]
    dropped = enforce_image_cap(msgs, 5)
    assert count_images(msgs) == 5 and len(dropped) == 1 and len(msgs) == n_msgs and [m["role"] for m in msgs] == roles
    text = json.dumps(msgs[2])
    assert "form0.png" in text and "sha256:" in text and "removed from the history" in text and "at most 5 images per prompt" in text
    assert is_image_message(msgs[2]) and not any("data:image" in json.dumps(msgs[2]["content"][1])  for _ in [0])          # image 0 is a note now; image 1 still attached
    assert [d["label"].split()[1] for d in dropped] == ["/work/form0.png"]


def test_under_the_cap_nothing_changes():
    msgs = base() + [image_message([att(0), att(1)])]
    before = json.dumps(msgs)
    assert enforce_image_cap(msgs, 5) == [] and json.dumps(msgs) == before


def test_six_images_in_the_newest_message_are_refused_not_trimmed():
    msgs = base() + [image_message([att(i) for i in range(6)])]
    before = json.dumps(msgs)
    with pytest.raises(ImageCapError):
        enforce_image_cap(msgs, 5)
    assert json.dumps(msgs) == before                       # nothing was dropped on the way to the refusal


def test_images_of_the_newest_message_are_never_dropped_to_make_room():
    msgs = base() + [image_message([att(0)]), image_message([att(1), att(2), att(3)])]
    enforce_image_cap(msgs, 3)
    assert count_images(msgs) == 3 and not is_image_message(msgs[2]) and is_image_message(msgs[3]) and count_images(msgs[3:]) == 3


def test_the_endpoint_limit_is_parsed_from_its_error_text():
    assert parse_image_limit("HTTP 400: At most 5 image(s) may be provided in one prompt.") == 5
    assert parse_image_limit("at most 1 image") == 1 and parse_image_limit("context length exceeded") is None and parse_image_limit("") is None


def test_pruning_for_context_counts_images_not_messages():
    msgs = base() + [image_message([att(0), att(1), att(2)]), image_message([att(3)])]
    assert mm.prune_images(msgs, keep=2) == 2 and count_images(msgs) == 2


# ---- the loop -------------------------------------------------------------------------------------------------------------------------------
def test_ten_forms_viewed_two_at_a_time_never_exceed_the_limit_even_without_token_pressure(tmp_path):
    ws = workspace(tmp_path, 10)
    acts = [[view(2 * k), view(2 * k + 1)] for k in range(5)] + ["done"]
    ep = StrictEndpoint(acts, limit=5)
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True, max_images=5), "s", "t", max_turns=20, max_context_tokens=65536, supervision=None, max_images=5)
    assert res.stop == "done" and ep.rejected == 0 and max(n for n, _ in ep.seen) <= 5 and ep.seen[-1][0] == 5
    final = json.dumps(ep.seen[-1][1])
    assert "form0.png" in final and "removed from the history" in final             # the oldest forms are labelled notes, still in the transcript


def test_without_the_invariant_the_same_run_reproduces_the_v2_crash(tmp_path):
    ws = workspace(tmp_path, 10)
    acts = [[view(2 * k), view(2 * k + 1)] for k in range(5)] + ["done"]
    ep = StrictEndpoint(acts, limit=5)
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True), "s", "t", max_turns=20, max_context_tokens=65536, supervision=None, max_images=None, keep_images=2)
    assert res.stop == "done" and ep.rejected == 1          # the loop now learns the limit from the first 400 instead of ending the run as model_error


def test_seven_views_in_one_turn_attach_five_and_say_so_for_the_rest(tmp_path):
    ws = workspace(tmp_path, 7)
    ep = StrictEndpoint([[view(i) for i in range(7)], "done"], limit=5)
    traj = tmp_path / "t.jsonl"
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True, max_images=5), "s", "t", max_turns=10, supervision=None, max_images=5, trajectory=traj)
    assert res.stop == "done" and ep.rejected == 0 and ep.seen[-1][0] == 5
    tool_texts = [m["content"] for m in ep.seen[-1][1] if m.get("role") == "tool"]
    assert sum("NOT attached" in t for t in tool_texts) == 2 and all("form5.png" in t or "form6.png" in t for t in tool_texts if "NOT attached" in t)
    events = [json.loads(l) for l in traj.read_text().splitlines()]
    assert any(e["kind"] == "image_cap" and e.get("refused_attachments") for e in events)


def test_a_limit_nobody_told_the_harness_is_learned_from_the_first_rejection(tmp_path):
    ws = workspace(tmp_path, 4)
    ep = StrictEndpoint([[view(0), view(1), view(2)], [view(3)], "done"], limit=3)
    traj = tmp_path / "t.jsonl"
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True), "s", "t", max_turns=10, supervision=None, max_context_tokens=65536, trajectory=traj)
    assert res.stop == "done" and ep.rejected == 1                         # exactly one 400 teaches the limit; every later request already complies
    k = next(i for i, (n, _) in enumerate(ep.seen) if n > 3)          # the one rejected request
    assert all(n <= 3 for n, _ in ep.seen[k + 1:]) and ep.seen[-1][0] == 3
    events = [json.loads(l) for l in traj.read_text().splitlines()]
    assert [e["cap"] for e in events if e["kind"] == "image_cap" and e.get("learned_from")] == [3]


def test_a_rejection_that_cannot_be_repaired_is_refused_locally_with_a_clear_stop(tmp_path):
    ws = workspace(tmp_path, 2)
    ep = StrictEndpoint([[view(0), view(1)], "never reached"], limit=1)
    traj = tmp_path / "t.jsonl"
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True), "s", "t", max_turns=10, supervision=None, trajectory=traj)
    assert res.stop == "request_invalid" and ep.rejected == 1 and ep.i == 1        # one 400 taught the limit; the second request was never sent
    events = [json.loads(l) for l in traj.read_text().splitlines()]
    assert any(e["kind"] == "model_error" and "refused locally" in e["error"] for e in events) and any(e["kind"] == "image_cap" and e.get("learned_from") for e in events)
    cls = classify(traj, {"stop": res.stop, "deliverables": [], "turns": res.turns})
    assert cls["class"] == "request_invalid"


def test_the_same_form_viewed_again_after_compaction_is_present_in_the_newest_message(tmp_path):
    ws = workspace(tmp_path, 3)
    acts = [view(0), view(1), view(2), view(0), view(1), view(0), "done"]
    ep = StrictEndpoint(acts, limit=5)
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True, max_images=5), "s", "t", max_turns=20, max_context_tokens=1000, supervision=None, keep_images=1, max_images=5)
    assert res.stop == "done" and ep.rejected == 0
    n, last = ep.seen[-1]
    newest = [m for m in last if is_image_message(m)][-1]
    assert "form0.png" in json.dumps(newest) and n <= 5


# ---- what the model is told -------------------------------------------------------------------------------------------------------------------
def test_the_view_image_description_states_the_limit_only_when_there_is_one(tmp_path):
    ws = workspace(tmp_path, 1)
    desc = lambda cap: next(s["function"]["description"] for s in parity_registry(ws, FakeSbx(), vision=True, max_images=cap).specs() if s["function"]["name"] == "view_image")
    assert "At most 5 images can be shown at once" in desc(5) and "write down what you read" in desc(5)
    assert "at once" not in desc(None)


# ---- recording and re-viewing -----------------------------------------------------------------------------------------------------------------
def test_viewing_a_dropped_image_again_is_not_a_repeat(tmp_path):
    ws = workspace(tmp_path, 4)
    acts = [view(0), view(1), view(2), view(0), view(1), view(2), view(0), view(1), view(2), view(0), "done"]
    ep = StrictEndpoint(acts, limit=2)
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True, max_images=2), "s", "t", max_turns=200, max_context_tokens=65536, supervision=Supervision(), max_images=2)
    assert res.stop == "done" and ep.rejected == 0
    tool_texts = [m["content"] for _, msgs in ep.seen[-1:] for m in msgs if m.get("role") == "tool"]
    assert not any("[supervisor]" in t for t in tool_texts) and res.supervisor["counts"]["suppressed"] == 0


def test_viewing_a_still_visible_image_again_and_again_is_still_a_repeat(tmp_path):
    ws = workspace(tmp_path, 1)
    ep = StrictEndpoint([view(0), view(0), view(0), view(0), "done"], limit=5)
    res = run_loop(ep, parity_registry(ws, FakeSbx(), vision=True, max_images=5), "s", "t", max_turns=20, supervision=Supervision(), max_images=5)
    assert res.supervisor["counts"]["suppressed"] >= 1


def test_the_image_message_asks_for_a_written_record_only_when_images_can_drop_out():
    text = lambda cap: image_message([att(0)], cap)["content"][0]["text"]
    assert "Only the newest 5 images stay visible" in text(5) and "write down every value you read" in text(5)
    assert "stay visible" not in text(None)


def test_image_is_visible_follows_the_attached_state():
    msgs = base() + [image_message([att(0)]), image_message([att(1)])]
    assert mm.image_is_visible(msgs, "/work/form0.png") and not mm.image_is_visible(msgs, "/work/form9.png")
    enforce_image_cap(msgs, 1)
    assert not mm.image_is_visible(msgs, "/work/form0.png") and mm.image_is_visible(msgs, "/work/form1.png")
