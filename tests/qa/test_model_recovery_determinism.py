# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-assisted runs are reproducible. The same scenario and the same scripted replies give the same request hash,
response hash, evidence events, fault events and classification in every process, whatever the hash seed, locale or time
zone; the order of the model's candidates changes nothing."""
import json
import os
import subprocess
import sys
from pathlib import Path

from . import _fx_recovery as FR, _fx_model as H

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json
from tests.qa import _fx_recovery as FR, _fx_model as H
cases = {
    "good": (None, [H.good(H.SAVE_BY_ID)], FR.move_save),
    "reordered": (None, [H.good({"label": "x"}, H.SAVE, H.SAVE_BY_ID)], FR.move_save),
    "reordered-again": (None, [H.good(H.SAVE_BY_ID, {"label": "x"}, H.SAVE)], FR.move_save),
    "malformed": (None, [H.Reply(text="no")], FR.move_save),
    "hostile": (None, [H.good(H.SAVE, {"role": "button", "name": "Save", "expected": "x"})], FR.move_save),
    "unicode": (None, [H.good({"role": "button", "name": "Savе"})], FR.move_save),
    "fall-through": (H.model_spec(permitted=("SEMANTIC_LOCATOR", "MODEL_CANDIDATE")), [H.good(H.SAVE_BY_ID)],
                     FR.compose(FR.move_save, H.semantic_needs_test_id)),
}
out = {}
for name, (spec, replies, setup) in cases.items():
    env = H.make_model_env(spec, *replies, adapter_setup=setup)
    r = env.run()
    out[name] = {"events": [e["event_hash"] for e in r.events], "manifest": r.manifest.canon_hash(),
                 "faults": [e.canon_hash() for e in r.manifest.fault_events],
                 "model": [e["payload"] for e in H.model_events(r)],
                 "requests": [q.prompt for q in env.client.requests], "classification": r.classification.to_json()}
print(json.dumps(out, sort_keys=True))
"""


def child(**env_overrides):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(env_overrides)
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_model_assisted_runs_are_identical_across_hash_seeds_locales_and_time_zones():
    runs = [child(PYTHONHASHSEED=seed) for seed in ("0", "1", "2", "12345", "random")]
    runs.append(child(PYTHONHASHSEED="7", LC_ALL="C", LANG="C", TZ="UTC"))
    runs.append(child(PYTHONHASHSEED="8", LC_ALL="C.UTF-8", LANG="C.UTF-8", TZ="Pacific/Kiritimati"))
    for other in runs[1:]:
        assert other == runs[0]
    first = runs[0]
    assert first["good"]["classification"]["reason"] == "RECOVERY_UNVERIFIED"
    assert first["malformed"]["classification"]["reason"] == "UNRECOVERED_FAULT"
    assert first["hostile"]["classification"]["reason"] == "ASSERTION_INTEGRITY"
    assert first["unicode"]["classification"]["reason"] == "UNRECOVERED_FAULT"
    assert first["fall-through"]["classification"]["reason"] == "RECOVERY_UNVERIFIED"
    assert len(first["good"]["model"]) == 2 and len(first["good"]["requests"]) == 1
    # the order the model lists its candidates in changes nothing recorded about the run, except the raw reply's own hash
    a, b = first["reordered"], first["reordered-again"]
    assert a["faults"] == b["faults"] and a["requests"] == b["requests"]
    assert a["model"][0] == b["model"][0] and a["model"][1]["response_hash"] != b["model"][1]["response_hash"]
    assert {k: v for k, v in a["model"][1].items() if k != "response_hash"} == \
        {k: v for k, v in b["model"][1].items() if k != "response_hash"}


def test_repeating_a_run_in_one_process_gives_identical_records():
    def once():
        env = H.make_model_env(None, H.good(H.SAVE_BY_ID), adapter_setup=FR.move_save)
        r = H.run(env)
        return [e["event_hash"] for e in r.events], r.manifest.canon_hash(), env.client.requests[0]
    assert once() == once() == once()


def test_the_order_the_adapter_lists_controls_changes_the_request_only_by_the_snapshot_order_and_not_the_recovery():
    def reordered(adapter):
        FR.move_save(adapter)
        FR.after_form(adapter, lambda a: setattr(a, "elements", dict(reversed(list(a.elements.items())))))
    base = H.run(H.make_model_env(None, H.good(H.SAVE_BY_ID), adapter_setup=FR.move_save))
    flipped = H.run(H.make_model_env(None, H.good(H.SAVE_BY_ID), adapter_setup=reordered))
    assert [e.recovery.canon_hash() for e in base.manifest.fault_events] == [e.recovery.canon_hash() for e in flipped.manifest.fault_events]
    assert base.classification == flipped.classification
