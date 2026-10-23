"""Audit an experiment against ONE declared specification, from three independent angles.

Two mistakes in the first runs were invisible in the configuration and obvious only on the wire or in the numbers: the arms were
sampled differently (OpenCode sends temperature 0.55 for Qwen models, our client sent nothing so each server used its own default of
1.0), and on one cell the editor's 262,144-token window silently replaced the executor's 65,536. This module makes such mistakes fail
loudly before and during a run:

* ``audit_config``: what each adapter WILL use (window, output ceiling, sampling), against the declared ``spec``;
* ``audit_wire``: what the spend proxy RECORDED being sent (temperature, top_p, max_tokens, thinking), against the same spec;
* ``audit_runs``: no finished run may have a prompt larger than the declared window.

The spec lives in the experiment config::

    "spec": {"window": 65536, "max_output": 16384, "sampling": {"temperature": 0.55, "top_p": 1.0}, "thinking": false}
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

EXECUTOR_LABELS = ("ours", "oc", "think")        # proxy labels used for the model under test (not the editor, not the judge)


@dataclass
class Finding:
    level: str            # FAIL | WARN | OK
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


def spec_of(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    s = cfg.get("spec")
    return s if isinstance(s, dict) and {"window", "max_output"} <= set(s) else None


def audit_config(exp) -> List[Finding]:
    from .harnesses import harness2 as h2, ours
    out: List[Finding] = []
    spec = spec_of(exp.cfg)
    if not spec:
        return [Finding("FAIL", "config", "no declared spec (window, max_output, sampling): refusing to run an unspecified experiment")]
    samp = spec.get("sampling") or {}
    for c in exp.cells:
        if c.window != spec["window"]:
            out.append(Finding("FAIL", f"cell {c.name}", f"window {c.window} != spec {spec['window']}"))
    if int(exp.cfg.get("max_output_tokens", 16384)) != spec["max_output"]:
        out.append(Finding("FAIL", "config", f"max_output_tokens {exp.cfg.get('max_output_tokens', 16384)} != spec {spec['max_output']}"))
    for arm in exp.arms:
        for c in exp.cells:
            where = f"{arm}@{c.name}"
            if arm == "harness2":
                env = h2.env_for(exp, c)
                oc = json.loads(env["OPENCODE_CONFIG_CONTENT"])
                prov, model = env["HARNESS2_SOLVER_MODEL"].split("/", 1)
                lim = oc["provider"][prov]["models"][model]["limit"]
                if lim != {"context": spec["window"], "output": spec["max_output"]}:
                    out.append(Finding("FAIL", where, f"OpenCode executor limit {lim} != spec window/output"))
                ed = env["HARNESS2_IMPROVER_MODEL"].split("/", 1)[0]
                if ed == prov:
                    out.append(Finding("FAIL", where, "the editor shares the executor's OpenCode provider entry (limits would collide)"))
                if samp.get("temperature") is not None and not oc["provider"][prov]["models"][model].get("temperature"):
                    out.append(Finding("FAIL", where, "spec asks for a temperature but the OpenCode model entry does not declare the temperature "
                                                      "capability, so OpenCode would send none (the server default would apply)"))
                out.append(Finding("WARN", where, "the temperature OpenCode sends is its own per-model policy (0.55 for Qwen); verified on the wire (audit_wire)"))
            elif arm.split("-r")[0].startswith("ours"):
                cl = ours.make_client(exp, c, int(exp.cfg.get("max_output_tokens", 16384)))
                if samp.get("temperature") is not None and cl.temperature != samp["temperature"]:
                    out.append(Finding("FAIL", where, f"client temperature {cl.temperature} != spec {samp['temperature']}"))
                if samp.get("top_p") is not None and cl.extra_body.get("top_p") != samp["top_p"]:
                    out.append(Finding("FAIL", where, f"client top_p {cl.extra_body.get('top_p')} != spec {samp['top_p']}"))
                if cl.max_tokens != spec["max_output"]:
                    out.append(Finding("FAIL", where, f"client max_tokens {cl.max_tokens} != spec {spec['max_output']}"))
    return out or [Finding("OK", "config", "all adapters match the declared spec")]


def audit_wire(exp, ledger: Path, since: float = 0.0, tolerance: float = 0.0) -> List[Finding]:
    """Check recorded requests for the model under test (labels ours/oc/think on the cells' providers) against the spec."""
    spec = spec_of(exp.cfg)
    if not spec:
        return [Finding("FAIL", "wire", "no declared spec")]
    samp = spec.get("sampling") or {}
    providers = {c.provider for c in exp.cells}
    rows = []
    for line in Path(ledger).read_text().splitlines():
        r = json.loads(line)
        if r["ts"] >= since and r["provider"] in providers and r["label"] in EXECUTOR_LABELS and r.get("status") == 200 and r.get("params"):
            rows.append(r)
    if not rows:
        return [Finding("WARN", "wire", "no recorded requests with parameters in this window; nothing to verify")]
    out: List[Finding] = []
    tolerated: Dict[str, int] = {}
    for key, want in (("temperature", samp.get("temperature")), ("top_p", samp.get("top_p"))):
        if want is None:
            continue
        bad = [r for r in rows if r["params"].get(key) != want]
        share = len(bad) / len(rows)
        if bad and share <= tolerance:
            tolerated[key] = len(bad)
        if share > tolerance:
            seen: Dict[str, int] = {}
            for r in bad:
                k = f"{r['provider']}/{r['label']}={r['params'].get(key)}"
                seen[k] = seen.get(k, 0) + 1
            out.append(Finding("FAIL", "wire", f"{len(bad)} of {len(rows)} requests have {key} != {want}: {dict(sorted(seen.items(), key=lambda x: -x[1])[:5])}"))
    over = [r for r in rows if (r["params"].get("max_tokens") or 0) > spec["max_output"]]
    if over:
        out.append(Finding("FAIL", "wire", f"{len(over)} requests asked for more than {spec['max_output']} output tokens"))
    if "thinking" in spec:
        bad = [r for r in rows if r["params"].get("thinking") is not None and bool(r["params"]["thinking"]) != bool(spec["thinking"])]
        if bad:
            out.append(Finding("FAIL", "wire", f"{len(bad)} requests have thinking != {spec['thinking']}"))
    note = f"; {tolerated} deviating request(s) tolerated as auxiliary" if tolerated else ""
    return out or [Finding("OK", "wire", f"{len(rows)} requests match the declared sampling, output ceiling and thinking mode{note}")]


def audit_runs(exp) -> List[Finding]:
    spec = spec_of(exp.cfg)
    if not spec:
        return [Finding("FAIL", "runs", "no declared spec")]
    bad = []
    for (arm, task), d in exp.board._state.items():
        if d.get("finished") and (d.get("peak_prompt") or 0) > spec["window"]:
            bad.append((arm, task.split("/")[0][:24], d["peak_prompt"]))
    if bad:
        worst = sorted(bad, key=lambda x: -x[2])[:5]
        return [Finding("FAIL", "runs", f"{len(bad)} finished runs had a prompt above the {spec['window']}-token window; worst: {worst}")]
    return [Finding("OK", "runs", f"no finished run exceeded the {spec['window']}-token window")]


def passed(findings: Sequence[Finding]) -> bool:
    return not any(f.level == "FAIL" for f in findings)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .experiment import Experiment
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ledger", help="spend proxy ledger to audit on the wire")
    ap.add_argument("--since", type=float, default=0.0, help="unix time; only requests after it are checked")
    ap.add_argument("--wire-tolerance", type=float, default=0.0, help="share of requests allowed to deviate (OpenCode auxiliary calls)")
    a = ap.parse_args(argv)
    ex = Experiment(json.loads(Path(a.config).expanduser().read_text()))
    fs = audit_config(ex) + audit_runs(ex)
    if a.ledger:
        fs += audit_wire(ex, Path(a.ledger).expanduser(), a.since, a.wire_tolerance)
    for f in fs:
        print(f)
    return 0 if passed(fs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
