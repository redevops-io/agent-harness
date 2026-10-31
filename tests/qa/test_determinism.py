# SPDX-License-Identifier: AGPL-3.0-or-later
"""Determinism: hashes do not depend on dict order, hash seed, locale, timezone or the process."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_harness.qa import canon, records as R

from ._helpers import QA_SRC, load_vectors

ROOT = Path(__file__).resolve().parents[2]
VECTORS = load_vectors()["vectors"]

SCRIPT = r"""
import json, sys
from agent_harness.qa import canon, records as R
data = json.load(open(sys.argv[1], encoding="utf-8"))
out = {}
for v in data["vectors"]:
    out[v["id"]] = canon.canon_hash(v["input"])
    sid = v["input"].get("schema_id") if isinstance(v["input"], dict) else None
    cls = R.RECORDS_BY_SCHEMA_ID.get(sid)
    if cls is not None:
        out["record:" + v["id"]] = cls.from_json(v["input"]).canon_hash()
specs = [R.AssertionSpec.from_json(a) for a in next(v for v in data["vectors"] if v["id"] == "qa-assertion-set:reference")["input"]["assertions"]]
out["set"] = R.assertion_set_hash(specs[::-1])
print(json.dumps(out, sort_keys=True))
"""


def run_child(**env_overrides):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(env_overrides)
    done = subprocess.run([sys.executable, "-c", SCRIPT, str(Path(__file__).with_name("golden") / "qa_canon_vectors.json")],
                          env=env, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_hashes_are_identical_across_hash_seeds_locales_and_timezones():
    expected = {v["id"]: "canon-v1:" + v["sha256_hex"] for v in VECTORS}
    runs = [run_child(PYTHONHASHSEED=seed) for seed in ("0", "1", "2", "12345", "random")]
    runs.append(run_child(PYTHONHASHSEED="7", LC_ALL="C", LANG="C", TZ="UTC"))
    runs.append(run_child(PYTHONHASHSEED="8", LC_ALL="C.UTF-8", LANG="C.UTF-8", TZ="Pacific/Kiritimati"))
    first = runs[0]
    for other in runs[1:]:
        assert other == first
    for vid, h in expected.items():
        assert first[vid] == h
        if "record:" + vid in first:
            assert first["record:" + vid] == h
    assert first["set"] == expected["qa-assertion-set:reference"]


def test_repeated_hashing_in_one_process_is_stable():
    for v in VECTORS:
        hashes = {canon.canon_hash(v["input"]) for _ in range(3)}
        assert hashes == {"canon-v1:" + v["sha256_hex"]}


def test_records_hash_the_same_whatever_the_key_order_of_the_input():
    spec = next(v["input"] for v in VECTORS if v["id"] == "qa:test_spec_ui")

    def rev(o):
        if isinstance(o, dict):
            return {k: rev(o[k]) for k in reversed(list(o))}
        return [rev(x) for x in o] if isinstance(o, list) else o

    assert R.TestSpec.from_json(rev(spec)).canon_hash() == R.TestSpec.from_json(spec).canon_hash()
    assert R.TestSpec.from_json(rev(spec)).to_json() == R.TestSpec.from_json(spec).to_json()


def test_nfd_input_hashes_as_nfc_through_the_records():
    nfc = next(v["input"] for v in VECTORS if v["id"] == "qa-assertion-set:unicode_nfc")
    nfd = next(v["input"] for v in VECTORS if v["id"] == "qa-assertion-set:unicode_nfd")
    assert nfc != nfd
    a = [R.AssertionSpec.from_json(x) for x in nfc["assertions"]]
    b = [R.AssertionSpec.from_json(x) for x in nfd["assertions"]]
    assert R.assertion_set_hash(a) == R.assertion_set_hash(b)


@pytest.mark.parametrize("name", ["canon.py", "records.py", "schemas.py", "errors.py"])
def test_hashing_and_record_code_uses_no_clock_randomness_or_environment(name):
    tree = ast.parse((QA_SRC / name).read_text(encoding="utf-8"))
    roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and not n.level}
    assert roots.isdisjoint({"random", "time", "datetime", "uuid", "secrets", "os", "locale", "socket", "subprocess", "threading"})
