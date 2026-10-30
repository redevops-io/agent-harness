# SPDX-License-Identifier: AGPL-3.0-or-later
"""H3: recovery is deterministic. The same scenario gives the same evidence hashes, fault events, receipts and
classification in every process, whatever the hash seed, locale or time zone; and the order in which an adapter lists
its controls changes nothing."""
import json
import os
import subprocess
import sys
from pathlib import Path

from agent_harness.qa import recovery as RV

from . import _h3 as H

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json
from tests.qa import _h3 as H
cases = {
    "renamed": (H.recover_spec(test_id="save"), H.rename_test_id),
    "moved": (H.recover_spec(), H.move_save),
    "delayed": (H.recover_spec(save=("RETRY_SAME",)), H.delay_save),
    "overlay": (H.recover_spec(save=("RETRY_SAME",)), H.overlay_on_save),
    "lost-session": (H.recover_spec(save=("CHECKPOINT_RESUME",)), H.lose_session_on_save),
    "stale-fixture": (H.recover_spec(save=("RESET_FIXTURE",)), H.stale_fixture),
    "ambiguous": (H.recover_spec(), H.duplicate_save),
    "removed": (H.recover_spec(save=("RETRY_SAME", "SEMANTIC_LOCATOR")), H.remove_save),
    "regression": (H.recover_spec(save=()), H.wrong_total),
}
out = {}
for name, (spec, setup) in cases.items():
    env = H.make_h3_env(spec, adapter_setup=setup)
    r = H.run(env)
    out[name] = {"events": [e["event_hash"] for e in r.events], "manifest": r.manifest.canon_hash(),
                 "faults": [e.canon_hash() for e in r.manifest.fault_events],
                 "receipts": [x.canon_hash() for x in r.receipts], "classification": r.classification.to_json()}
print(json.dumps(out, sort_keys=True))
"""


def child(**env_overrides):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(env_overrides)
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=300,
                          cwd=str(ROOT))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_recovery_runs_are_identical_across_hash_seeds_locales_and_time_zones():
    runs = [child(PYTHONHASHSEED=seed) for seed in ("0", "1", "2", "12345", "random")]
    runs.append(child(PYTHONHASHSEED="7", LC_ALL="C", LANG="C", TZ="UTC"))
    runs.append(child(PYTHONHASHSEED="8", LC_ALL="C.UTF-8", LANG="C.UTF-8", TZ="Pacific/Kiritimati"))
    for other in runs[1:]:
        assert other == runs[0]
    first = runs[0]
    assert first["moved"]["classification"]["reason"] == "RECOVERY_UNVERIFIED"
    assert first["lost-session"]["classification"]["outcome"] == "PASS"
    assert first["ambiguous"]["classification"]["reason"] == "UNRECOVERED_FAULT"
    assert first["regression"]["classification"]["outcome"] == "PRODUCT_FAIL"
    assert len(first["moved"]["faults"]) == 1 and first["regression"]["faults"] == []


def test_repeating_a_run_in_one_process_gives_identical_records():
    def once():
        r = H.run(H.make_h3_env(H.recover_spec(), adapter_setup=H.move_save))
        return [e["event_hash"] for e in r.events], r.manifest.canon_hash()
    assert once() == once() == once()


def test_the_order_the_adapter_lists_controls_does_not_change_the_recovery():
    def reordered(adapter):
        H.move_save(adapter)

        def reverse(a):
            a.elements = dict(reversed(list(a.elements.items())))
        H.after_form(adapter, reverse)
    base = H.run(H.make_h3_env(H.recover_spec(), adapter_setup=H.move_save))
    flipped = H.run(H.make_h3_env(H.recover_spec(), adapter_setup=reordered))
    assert [e.recovery.canon_hash() for e in base.manifest.fault_events] == \
        [e.recovery.canon_hash() for e in flipped.manifest.fault_events]
    assert base.manifest.canon_hash() == flipped.manifest.canon_hash()


def test_the_recovery_decision_and_selection_are_the_same_value_every_time():
    from .test_h3_recovery import World, inputs, plan, select
    a = select(World(save={}, delete={"name": "Delete"}))
    assert all(select(World(save={}, delete={"name": "Delete"})) == a for _ in range(3))
    assert len({plan(inputs()) for _ in range(5)}) == 1 and isinstance(a, RV.LocatorSelection)
