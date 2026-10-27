"""Four-answers-per-call JobBench judge (run with the benchmark's own Python environment).

For ONE task and ONE model it judges the answers of several harnesses together: per rubric group a single request carries the group's
criteria and every candidate's output files, labelled A, B, C, ... in a seeded shuffle (the mapping is written next to the results),
and returns one verdict per criterion per candidate. Candidates are judged independently ("judge each only on its own files"); the point
of the shared call is one consistent standard across the harnesses, not a ranking.

Reuses the upstream judge (``--judge-script``) for file conversion, rubric handling, JSON parsing and the details-file format, so every
candidate gets a details file in exactly the upstream format and downstream code does not change. Rubric pass/fail is recomputed
from the criteria (the rule the prompt states: all criteria must pass) instead of trusting the model's own ``rubric_passed``.

This module's core (``run_batch`` and friends) takes the upstream module and a model-call function as arguments so it can be tested
without a network or the benchmark's dependencies.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

FAIL_PREFIX = "Failed to get judge response"          # the upstream judge's marker for "no usable reply"; downstream code counts these


def assign_labels(keys: Sequence[str], seed: str) -> Dict[str, str]:
    """Candidate key -> letter, a deterministic shuffle of the sorted keys (so the order shown does not follow arm order)."""
    order = sorted(keys)
    random.Random(seed).shuffle(order)
    return {k: string.ascii_uppercase[i] for i, k in enumerate(order)}


def build_prompt(rubric_text: str, criteria: Sequence[str], sections: Sequence[Tuple[str, str]], vision_labels: Sequence[str] = ()) -> str:
    crit = "\n".join(f"Criterion {i}: {c}" for i, c in enumerate(criteria))
    letters = ", ".join(l for l, _ in sections)
    files = "\n\n".join(f"### CANDIDATE {l}: output files\n\n{text}" for l, text in sections)
    shape = ",\n    ".join(
        f'"{l}": {{"criteria_results": [{{"index": 0, "passed": true/false, "reasoning": "...", "evidence": "..."}}, ...], '
        f'"rubric_passed": true/false, "overall_reasoning": "..."}}' for l, _ in sections)
    imgs = f"\n- Attached images are labelled with the candidate they belong to ({', '.join(vision_labels)})" if vision_labels else ""
    return f"""You are an evaluation judge. Evaluate ALL criteria of a single rubric for EACH of {len(sections)} candidate outputs ({letters}).

## Rubric Description
{rubric_text}

## Criteria to Evaluate (judge ALL of them, for EVERY candidate)
{crit}

## Candidate Outputs
Each candidate below is the complete set of output files produced independently for the same task.

{files}

## Evaluation Rules
- Judge each candidate INDEPENDENTLY, using only that candidate's own files. Do not compare candidates and do not let one candidate's content influence another's verdict.
- For each candidate and each criterion decide PASS or FAIL only; semantic matching is acceptable (exact wording is not required).
- A candidate's rubric passes ONLY if ALL criteria pass for that candidate.
- The candidate labels carry no meaning (their order is random).

## Output Format
Return a JSON object with EXACTLY this structure (no markdown, no extra text):
{{
  "candidates": {{
    {shape}
  }}
}}

IMPORTANT:
- every candidate must have a criteria_results array with exactly {len(criteria)} items, one per criterion, using "index" 0..{len(criteria) - 1}
- include specific evidence from that candidate's output files{imgs}
"""


def candidate_result(upstream, rubric_index: int, rubric: dict, section: Optional[dict], error: str = "") -> dict:
    """Turn one candidate's part of the reply into an upstream-format rubric result; a missing or unusable part is a failure result."""
    criteria = upstream.normalize_criteria(rubric)
    if not isinstance(section, dict) or not isinstance(section.get("criteria_results"), list):
        why = error or "reply had no usable section for this candidate"
        r = upstream.build_failed_rubric_result(rubric_index, rubric, f"Failed after retries: {why}", f"{FAIL_PREFIX}: {why}")
        return r
    got = {int(c.get("index", -1)): c for c in section["criteria_results"] if isinstance(c, dict) and str(c.get("index", "")).lstrip("-").isdigit()}
    enriched = []
    for i, criterion in enumerate(criteria):
        c = got.get(i, {})
        enriched.append({"index": i, "criterion": criterion, "passed": bool(c.get("passed", False)),
                         "reasoning": c.get("reasoning", "") if c else f"{FAIL_PREFIX}: criterion {i} missing from the reply",
                         "evidence": c.get("evidence", "")})
    passed = bool(enriched) and all(c["passed"] for c in enriched)
    weight = rubric.get("weight", 0)
    res = {"passed": passed, "score": weight if passed else 0, "criteria_count": len(criteria),
           "criteria_passed": sum(1 for c in enriched if c["passed"]), "criteria_results": enriched,
           "overall_reasoning": section.get("overall_reasoning", "")}
    if "rubric_passed" in section and bool(section["rubric_passed"]) != passed:
        res["model_rubric_passed"] = bool(section["rubric_passed"])        # kept for transparency; the criteria decide
    return {"index": rubric_index, "rubric": rubric.get("rubric", ""), "weight": weight, "result": res}


def judge_group(upstream, rubric_index: int, rubric: dict, contents: Dict[str, str], labels: Dict[str, str],
                call_model: Callable[[str, int], str], max_retries: int, timeout: int) -> Dict[str, dict]:
    """One request for one rubric group; returns key -> upstream-format result for every candidate."""
    by_letter = {l: k for k, l in labels.items()}
    sections = [(l, contents[by_letter[l]]) for l in sorted(by_letter)]
    prompt = build_prompt(rubric.get("rubric", ""), upstream.normalize_criteria(rubric), sections)
    last = ""
    for _ in range(max(1, max_retries)):
        try:
            raw = call_model(prompt, timeout)
            parsed, _status = upstream.parse_judge_json(raw)
            cands = parsed.get("candidates") if isinstance(parsed, dict) else None
            if not isinstance(cands, dict):
                raise ValueError("reply has no 'candidates' object")
            return {k: candidate_result(upstream, rubric_index, rubric, cands.get(l)) for k, l in labels.items()}
        except Exception as e:            # noqa: BLE001 - any failure of one call is a recorded failure, never a crash
            last = f"{type(e).__name__}: {e}"
    return {k: candidate_result(upstream, rubric_index, rubric, None, last) for k in labels}


def run_batch(upstream, rubrics: List[dict], candidates: Dict[str, Path], judge_model: str, call_model: Callable[[str, int], str],
              seed: str, max_workers: int = 4, max_retries: int = 2, timeout: int = 900) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Judge every rubric group over all candidates. Returns (key -> upstream-format details report, key -> letter)."""
    labels = assign_labels(list(candidates), seed)
    contents = {k: upstream.extract_all_file_contents(Path(d)) for k, d in candidates.items()}
    results: Dict[str, List[Optional[dict]]] = {k: [None] * len(rubrics) for k in candidates}

    def one(i: int) -> None:
        out = judge_group(upstream, i, rubrics[i], contents, labels, call_model, max_retries, timeout)
        for k, r in out.items():
            results[k][i] = r
    with ThreadPoolExecutor(max(1, max_workers)) as pool:
        list(pool.map(one, range(len(rubrics))))
    reports = {k: upstream.build_details_report(k, judge_model, [r for r in rs if r is not None], len(rubrics)) for k, rs in results.items()}
    for k in reports:
        reports[k]["protocol"] = {"name": "batched", "candidates": len(candidates), "label": labels[k], "seed": seed}
    return reports, labels


def load_upstream(path: str):
    spec = importlib.util.spec_from_file_location("upstream_judge", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["upstream_judge"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_call_model(upstream, judge_model: str, api_base: str, api_key: str) -> Callable[[str, int], str]:
    extra: Dict[str, Any] = {}
    effort = os.environ.get("JUDGE_REASONING_EFFORT", "low").strip()
    if effort and effort.lower() != "default":
        extra["extra_body"] = {"reasoning_effort": effort}

    def call(prompt: str, timeout: int) -> str:
        client = upstream.get_openai_client(api_base, api_key)
        r = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "system", "content": "You are an evaluation judge. You must return valid JSON only, with no markdown formatting or extra text."},
                      {"role": "user", "content": prompt}],
            max_completion_tokens=int(os.environ.get("JUDGE_MAX_COMPLETION_TOKENS", "65536")), temperature=0.0, timeout=timeout, **extra)
        msg = r.choices[0].message if r.choices else None
        content = getattr(msg, "content", None) if msg else None
        if not content:
            raise ValueError(f"Judge returned empty content (finish_reason={r.choices[0].finish_reason if r.choices else 'unknown'})")
        return content.strip()
    return call


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judge-script", required=True, help="path of the upstream eval/judge.py")
    ap.add_argument("--rubrics-file", required=True)
    ap.add_argument("--candidate", action="append", required=True, help="KEY=DIR (repeatable)")
    ap.add_argument("--details-dir", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    ap.add_argument("--seed", required=True)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--timeout-per-rubric", type=int, default=900)
    a = ap.parse_args(argv)
    up = load_upstream(a.judge_script)
    data = json.loads(Path(a.rubrics_file).read_text(encoding="utf-8"))
    rubrics = data.get("rubrics") or data.get("evaluation_rubrics") or []
    cands = {}
    for spec in a.candidate:
        k, d = spec.split("=", 1)
        cands[k] = Path(d)
    reports, labels = run_batch(up, rubrics, cands, a.judge_model, make_call_model(up, a.judge_model, a.api_base, a.api_key), a.seed,
                                a.max_workers, a.max_retries, a.timeout_per_rubric)
    out = Path(a.details_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, rep in reports.items():
        up.write_json(out / f"{k}.json", rep)
    (out / "blind_map.json").write_text(json.dumps({"seed": a.seed, "labels": labels}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
