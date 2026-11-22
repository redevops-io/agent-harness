"""The tool contract has exactly one implementation.

``context_runtime.tools.base`` re-exports ``agent_harness.tools`` — it used to be a copy, and the copy had
nothing tying it to the original. So "conformance" here is not two implementations agreeing on a spec; it is
the absence of a fork. That distinction matters for the authorization seam: a fix landed here is the fix for
every consumer that imports through the re-export, and a fork reintroduced anywhere would silently un-fix it.
"""
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
    """Same gate, same records, through both import paths — the whole audit list compared, not a projection.

    This is the assertion that made the copy safe to delete. It is kept because it is the one that would
    notice a fork coming back: an installed ``context_runtime`` that still carries its own registry answers
    with the old ``None``-means-allow contract, and would fail here rather than pass by coincidence.
    """
    if cr.ToolRegistry is not ah.ToolRegistry:
        pytest.fail("context_runtime.tools.base no longer re-exports agent_harness.tools — a second copy of "
                    "the contract is installed. The authorization fix lives in one place; a fork means the "
                    "consumers behind that import path are still fail-open.")

    def build(m, authorizer):
        reg = m.ToolRegistry(m.ApprovalPolicy("deny_side_effects"), authorizer=authorizer)
        reg.register(m.function_tool("w", lambda a: m.ToolResult(text="x"), side_effecting=True))
        reg.register(m.function_tool("r", lambda a: m.ToolResult(text="y")))
        return reg

    allow = lambda p, spec, args: ah.Authorization.allow(grant_ids=("conformance",))
    ra, rb = build(ah, allow), build(cr, allow)
    for name in ("w", "r", "missing"):
        x, y = ra.run(name, {"k": 1}), rb.run(name, {"k": 1})
        assert (x.ok, x.error, x.text) == (y.ok, y.error, y.text)
    assert ra.audit == rb.audit
    assert ra.specs() == rb.specs()


def test_reexport_shares_every_public_symbol():
    """Identity, not structural equality — a symbol re-implemented on the far side is a fork with one file."""
    for name in ("ApprovalPolicy", "ToolRegistry", "ToolResult", "ToolSpec", "ToolPlugin", "ToolError",
                 "function_tool", "current_principal", "set_default_authorizer", "Approver", "Authorizer"):
        assert getattr(cr, name) is getattr(ah, name), f"context_runtime.tools.base.{name} is a separate object"


@pytest.mark.xfail(strict=True, reason="PR-C12: the re-export does not yet carry authz/v1, so a consumer "
                                       "importing through context_runtime.tools.base cannot express a typed "
                                       "decision and must import agent_harness.tools to do so. When the "
                                       "re-export gains these names this strict xfail turns into a failure — "
                                       "remove the marker then.")
def test_reexport_exposes_the_authorization_contract():
    for name in ("Authorization", "Decision", "PolicyReason", "as_authorization", "local_trusted_authorizer",
                 "default_authorizer_installed"):
        assert hasattr(cr, name), name
