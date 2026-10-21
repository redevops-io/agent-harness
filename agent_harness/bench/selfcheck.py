"""Preflight checks for a benchmark setup. Nothing here calls a paid API; run it before any benchmark.

    python -m agent_harness.bench.selfcheck --judge-python ... --judge-script ... --judge-model M --judge-base-url U \\
        --tasks-base <job-bench-eval>/dataset/main --calibration ~/Documents/harness2-calibration \\
        --endpoint name=http://127.0.0.1:8765/proxmox/chk/v1:Qwen3.8-27B

Each check returns PASS / FAIL / SKIP with a reason. The exit code is non-zero if any check FAILs, so a run script can
refuse to start. Checks (in order of cost): sandbox isolation, scripted end-to-end pipeline (real Docker sandbox, real
context manager, scripted model that forces compaction and retrieval), endpoint behaviour, judge controls, judge
calibration against labelled outputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .evalkit import JudgeConfig, judge_output, spearman


@dataclass
class Check:
    name: str
    status: str            # PASS | FAIL | SKIP
    detail: str = ""


def docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def check_sandbox() -> Check:
    if not docker_ok():
        return Check("sandbox isolation", "SKIP", "docker daemon not reachable")
    from .sandbox import DockerSandbox
    ws = Path(tempfile.mkdtemp(prefix="ah-selfcheck-"))
    try:
        ws.chmod(0o777)
        with DockerSandbox(ws) as sbx:
            net = sbx.run("python3 -c \"import urllib.request;urllib.request.urlopen('http://example.com',timeout=5);print('REACHED')\" 2>&1 | tail -1", 30)
            user = sbx.run("id -un", 15).strip()
            libs = sbx.run("python3 -c 'import docx, openpyxl, pdfplumber, pandas; print(\"libs-ok\")'", 60).strip()
            writable = sbx.run("touch /work/_probe && echo ok", 15).strip()
        problems = []
        if "REACHED" in net:
            problems.append("sandbox has network access")
        if user == "root":
            problems.append("sandbox runs as root")
        if "libs-ok" not in libs:
            problems.append("office libraries missing")
        if writable != "ok":
            problems.append("workspace not writable")
        return Check("sandbox isolation", "FAIL" if problems else "PASS", "; ".join(problems) or "no network, non-root, libraries present, workspace writable")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


class _Scripted:
    """A deterministic stand-in model: dumps big outputs, then must recover one through search_context, then delivers."""
    model = "scripted"

    def __init__(self):
        self.step = 0

    def complete(self, messages, tools=None):
        s, self.step = self.step, self.step + 1
        chars = sum(len(m.get("content") or "") for m in messages)
        usage = {"prompt_tokens": chars // 3, "completion_tokens": 20}

        def call(name, **a):
            return {"role": "assistant", "content": "", "tool_calls": [{"id": f"s{s}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(a)}}]}
        if s < 6:
            return {"message": call("bash", command=f"python3 -c \"print('LINE '*900 + ' MARKER{s}')\""), "finish_reason": "tool_calls", "usage": usage}
        if s == 6:
            return {"message": call("search_context", query="MARKER1"), "finish_reason": "tool_calls", "usage": usage}
        if s == 7:        # decide from the REAL search result the loop handed back, not from what the script expects
            last_tool = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
            verdict = "found MARKER1" if "MARKER1" in last_tool else "NOT FOUND"
            return {"message": call("write_file", path="/work/output/result.md", content=verdict), "finish_reason": "tool_calls", "usage": usage}
        return {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop", "usage": usage}


def check_pipeline() -> Check:
    if not docker_ok():
        return Check("scripted pipeline", "SKIP", "docker daemon not reachable")
    try:
        import context_runtime  # noqa: F401
    except ImportError:
        return Check("scripted pipeline", "FAIL", "context-runtime is not installed (it is the default context manager)")
    from . import jobbench as jb
    root = Path(tempfile.mkdtemp(prefix="ah-selfcheck-"))
    try:
        tf = root / "base" / "t" / "task1" / "task_folder"
        tf.mkdir(parents=True)
        (tf / "TASK_INSTRUCTIONS.txt").write_text("Write result.md.")
        s = jb.run_task(root / "base", "t/task1", root / "out", _Scripted(), max_turns=30, max_seconds=300,
                        max_context_tokens=3000, web=False, attempts=1, context="context-runtime")
        traj = [json.loads(line) for line in (root / "out" / "trajectory.jsonl").read_text().splitlines()]
        compacted = any(e["kind"] == "compact" for e in traj)
        problems = []
        if s["stop"] != "done":
            problems.append(f"stopped with {s['stop']}")
        if s["deliverables"] != ["result.md"]:
            problems.append(f"deliverables {s['deliverables']}")
        if not compacted:
            problems.append("no compaction happened (window too large for the scripted dump?)")
        if s["archived_items"] < 2:
            problems.append(f"only {s['archived_items']} archived items")
        if s["retrieval_calls"] < 1:
            problems.append("the model never called a retrieval tool")
        written = (root / "out" / "output" / "result.md").read_text() if (root / "out" / "output" / "result.md").exists() else ""
        if written != "found MARKER1":
            problems.append(f"archived content was not recoverable through search_context (deliverable says {written!r})")
        return Check("scripted pipeline", "FAIL" if problems else "PASS",
                     "; ".join(problems) or f"{s['turns']} turns, {s['archived_items']} archived items, {s['retrieval_calls']} retrieval call, deliverable written")
    except Exception as e:                                        # noqa: BLE001 - a crash is a failed check, with its message
        return Check("scripted pipeline", "FAIL", f"{type(e).__name__}: {e}"[:300])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_endpoint(name: str, base_url: str, model: str, want_ctx_at_least: int = 0, timeout: int = 180) -> Check:
    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(base_url.rstrip("/") + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer dummy"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    try:
        t0 = time.time()
        d = post("/chat/completions", {"model": model, "temperature": 0, "max_tokens": 200,
                                       "messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}]})
        dt = time.time() - t0
        text = (d["choices"][0]["message"].get("content") or "").strip()
        reasoning = (d["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        problems = []
        if "391" not in text:
            problems.append(f"wrong or empty answer {text[:40]!r} (reasoning tokens {reasoning})")
        with urllib.request.urlopen(urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer dummy"}), timeout=30) as r:
            models = json.load(r)["data"]
        ctx = next((m.get("max_model_len") or m.get("context_length") for m in models if m["id"] == model), None)
        if want_ctx_at_least and (ctx or 0) < want_ctx_at_least:
            problems.append(f"served context {ctx} < required {want_ctx_at_least}")
        return Check(f"endpoint {name}", "FAIL" if problems else "PASS",
                     "; ".join(problems) or f"answered in {dt:.1f}s, context {ctx}, reasoning tokens {reasoning}")
    except Exception as e:                                        # noqa: BLE001
        return Check(f"endpoint {name}", "FAIL", f"{type(e).__name__}: {str(e)[:200]}")


def check_judge_controls(cfg: JudgeConfig, tasks_base: Path, calibration: Path, *, min_spearman: float = 0.7,
                         max_items: Optional[int] = None, log: Callable[[str], None] = lambda s: None) -> List[Check]:
    """Negative control (garbage must score low), no-output control, then calibration against labelled outputs."""
    out: List[Check] = []
    man = json.loads((calibration / "manifest.json").read_text())
    items = man["items"][:max_items] if max_items else man["items"]
    tmp = Path(tempfile.mkdtemp(prefix="ah-judge-"))
    try:
        t = items[0]["task"]
        junk = tmp / "junk"
        junk.mkdir()
        (junk / "report.md").write_text("Lorem ipsum dolor sit amet. " * 200)
        r = judge_output(t, tasks_base, junk, tmp / "junk.json", cfg)
        ok = r.status == "graded" and r.score is not None and r.score <= 0.25
        out.append(Check("judge negative control", "PASS" if ok else "FAIL",
                         f"irrelevant text scored {r.score if r.score is None else round(r.score, 3)} ({r.status}); must be graded and <= 0.25"))
        empty = tmp / "empty"
        empty.mkdir()
        r0 = judge_output(t, tasks_base, empty, tmp / "empty.json", cfg)
        out.append(Check("judge no-output control", "PASS" if (r0.status, r0.score) == ("no_output", 0.0) else "FAIL", f"{r0.status} {r0.score}"))
        ref, cand, ungraded = [], [], 0
        for i, it in enumerate(items):
            res = judge_output(it["task"], tasks_base, calibration / it["dir"], tmp / f"cal{i}.json", cfg)
            log(f"  calibration {i + 1}/{len(items)} {it['id']}: ref {it['grok47']:.3f} candidate {res.score}")
            if res.status == "graded":
                ref.append(it["grok47"])
                cand.append(res.score)
            else:
                ungraded += 1
        if len(ref) < 5:
            out.append(Check("judge calibration", "FAIL", f"only {len(ref)} of {len(items)} outputs could be graded ({ungraded} ungraded)"))
        else:
            rho = spearman(ref, cand)
            gap = sum(cand) / len(cand) - sum(ref) / len(ref)
            out.append(Check("judge calibration", "PASS" if rho >= min_spearman and ungraded <= 1 else "FAIL",
                             f"Spearman {rho:.2f} vs the reference judge on {len(ref)} outputs (need >= {min_spearman}); "
                             f"mean level {sum(cand)/len(cand):.2f} vs {sum(ref)/len(ref):.2f} (gap {gap:+.2f}); {ungraded} ungraded"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judge-python"); ap.add_argument("--judge-script"); ap.add_argument("--judge-model"); ap.add_argument("--judge-base-url")
    ap.add_argument("--judge-workers", type=int, default=4)
    ap.add_argument("--tasks-base"); ap.add_argument("--calibration")
    ap.add_argument("--endpoint", action="append", default=[], help="name=base_url:model[:min_context]")
    ap.add_argument("--max-calibration-items", type=int)
    ap.add_argument("--min-spearman", type=float, default=0.7)
    ap.add_argument("--skip", action="append", default=[], choices=["sandbox", "pipeline", "endpoints", "judge"])
    a = ap.parse_args(argv)
    checks: List[Check] = []

    def show(c: Check) -> None:
        checks.append(c)
        print(f"[{c.status}] {c.name}: {c.detail}", flush=True)
    if "sandbox" not in a.skip:
        show(check_sandbox())
    if "pipeline" not in a.skip:
        show(check_pipeline())
    if "endpoints" not in a.skip:
        for spec in a.endpoint:
            name, rest = spec.split("=", 1)
            parts = rest.rsplit(":", 2)
            if parts[-1].isdigit() and len(parts) == 3:
                url_model, ctx = parts[0] + ":" + parts[1], int(parts[2])
            else:
                url_model, ctx = rest, 0
            url, model = url_model.rsplit(":", 1) if "/v1:" in url_model else (url_model, "")
            show(check_endpoint(name, url, model, ctx))
    if "judge" not in a.skip:
        if not (a.judge_python and a.judge_script and a.judge_model and a.judge_base_url and a.tasks_base and a.calibration):
            show(Check("judge controls", "SKIP", "judge options not given"))
        else:
            cfg = JudgeConfig(a.judge_python, a.judge_script, a.judge_model, a.judge_base_url, workers=a.judge_workers)
            for c in check_judge_controls(cfg, Path(a.tasks_base), Path(a.calibration).expanduser(), min_spearman=a.min_spearman,
                                          max_items=a.max_calibration_items, log=lambda s: print(s, flush=True)):
                show(c)
    failed = [c for c in checks if c.status == "FAIL"]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed" + (f"; FAILED: {', '.join(c.name for c in failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
