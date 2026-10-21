"""Run the same tool-gate cases through agent_harness.tools and context_runtime.tools.base, when the latter is
installed (Python >= 3.12). The contract was adopted from context-runtime; this guards against drift."""
import pytest

cr = pytest.importorskip("context_runtime.tools.base")
from agent_harness import tools as ah  # noqa: E402

MODES = [("deny_side_effects", []), ("allowlist", ["t"]), ("allowlist", ["other"]), ("bypass", [])]
FLAGS = [(False, False), (True, False), (False, True), (True, True)]


@pytest.mark.parametrize("mode,allow", MODES)
@pytest.mark.parametrize("effect,gated", FLAGS)
@pytest.mark.parametrize("approve", [None, True, False])
def test_policy_decisions_match(mode, allow, effect, gated, approve):
    approver = None if approve is None else (lambda a, v=approve: v)
    a = ah.ApprovalPolicy(mode, allow, approver).decide(ah.ToolSpec("t", side_effecting=effect, approval_required=gated), {})
    b = cr.ApprovalPolicy(mode, allow, approver).decide(cr.ToolSpec("t", side_effecting=effect, approval_required=gated), {})
    assert a == b


def test_registry_results_and_audit_match():
    def build(m):
        reg = m.ToolRegistry(m.ApprovalPolicy("deny_side_effects"))
        reg.register(m.function_tool("w", lambda a: m.ToolResult(text="x"), side_effecting=True))
        reg.register(m.function_tool("r", lambda a: m.ToolResult(text="y")))
        return reg
    ra, rb = build(ah), build(cr)
    for name in ("w", "r", "missing"):
        x, y = ra.run(name, {"k": 1}), rb.run(name, {"k": 1})
        assert (x.ok, x.error, x.text) == (y.ok, y.error, y.text)
    assert ra.audit == rb.audit
    assert ra.specs() == rb.specs()
