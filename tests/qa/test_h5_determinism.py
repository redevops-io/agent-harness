# SPDX-License-Identifier: AGPL-3.0-or-later
"""H5: the egress decisions, the redaction output and manifests, the policy ids, the blob digests and a guarded run's
evidence are identical in every process, whatever the hash seed, locale or time zone."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json
from tests.qa import _h5 as H, test_h5_egress_vectors as V, test_h5_redaction as T
from agent_harness.qa import blobs as B, egress as E, redaction as RD
from agent_harness.qa.vault import ArtifactVault

out = {}
guard, _ = H.make_guard()
out["egress"] = [[v[0], guard.check(v[1]).record_hash] for v in V.VECTORS]
out["egress_policy"] = guard.policy_id
chain = E.decide_chain(H.make_policy(), [("https://app.example.test/", [H.PUBLIC_V4]), ("https://api.example.test/", [H.PUBLIC_V4_B, H.PUBLIC_V6]),
                                         ("http://169.254.169.254/", None)])
out["chain"] = chain.chain_hash
r = H.make_redactor()
out["redaction_policy"] = r.policy_id
out["redaction"] = [[r.redact_with_manifest(c[1].encode(), "text/plain")[1].manifest_hash,
                     r.redact_with_manifest(c[1].encode(), "text/plain")[0].decode()] for c in T.RULE_CASES]
doc = {"k": [c[1] for c in T.RULE_CASES], "password": "x", "tokens": 3, "z": {"b": 1, "a": 2}}
out["structure"] = json.dumps(r.redact_structure(doc)[0], sort_keys=True)
vault = ArtifactVault(r, B.MemoryBlobSink())
for i, c in enumerate(T.RULE_CASES[:12]):
    vault.store("note", c[1].encode(), "text/plain", step_id="s%d" % i)
vault.store("network", json.dumps(doc).encode(), "application/json")
out["artifacts"] = vault.manifest("PASS").manifest_hash
env = H.h5_env(adapter_setup=lambda a: setattr(a, "dom_meta", {"title": "x " + H.CANARY_A + " " + H.EMAIL, "password": "p", "nodes": 2}))
res = env.run()
out["run"] = {"events": [e["event_hash"] for e in res.events], "manifest": res.manifest.canon_hash(),
              "classification": res.classification.to_json(), "redacted": [list(x) for x in env.guard.redacted_events]}
print(json.dumps(out, sort_keys=True))
"""


def child(**overrides):
    paths = [str(ROOT)] + [p for p in sys.path if p and p != str(ROOT)]          # the child needs the same packages (pytest)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": os.pathsep.join(paths), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(overrides)
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_egress_redaction_blobs_and_a_guarded_run_are_identical_across_hash_seeds_locales_and_time_zones():
    runs = [child(PYTHONHASHSEED=seed) for seed in ("0", "1", "2", "12345", "random")]
    runs.append(child(PYTHONHASHSEED="7", LC_ALL="C", LANG="C", TZ="UTC"))
    runs.append(child(PYTHONHASHSEED="8", LC_ALL="C.UTF-8", LANG="C.UTF-8", TZ="Pacific/Kiritimati"))
    runs.append(child(PYTHONHASHSEED="9", LC_ALL="tr_TR.UTF-8", LANG="tr_TR.UTF-8", TZ="America/St_Johns"))     # the Turkish-i locale
    for other in runs[1:]:
        assert other == runs[0]
    first = runs[0]
    assert len(first["egress"]) >= 150 and first["run"]["classification"]["outcome"] == "PASS" and first["run"]["redacted"]
    assert first["redaction_policy"].startswith("rcv1:") and first["egress_policy"].startswith("rcv1:")


def test_repeating_the_work_in_one_process_gives_identical_records():
    from . import _h5 as H, test_h5_egress_vectors as V
    results = []
    for _ in range(2):
        guard, _ = H.make_guard()
        results.append(([guard.check(v[1]).record_hash for v in V.VECTORS], H.make_redactor().policy_id))
    assert results[0] == results[1]
