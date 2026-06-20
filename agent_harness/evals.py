"""Eval suite runner for agent tasks."""
import json
from typing import Any, Callable, Dict, List
import agent_harness.agent as agent_mod

def run_suite(tasks: List[Dict[str, Any]], agent_factory: Callable = None) -> Dict[str, Any]:
    if agent_factory is None:
        agent_factory = agent_mod.create_agent
    results = []
    for t in tasks:
        agent = agent_factory()
        out = agent.run(t.get("input", ""))
        passed = t.get("expected", "") in str(out)
        results.append({"task": t.get("name", ""), "passed": passed, "output": out})
    report = {"results": results, "score": sum(r["passed"] for r in results) / max(1, len(results))}
    print(json.dumps(report, indent=2))
    return report