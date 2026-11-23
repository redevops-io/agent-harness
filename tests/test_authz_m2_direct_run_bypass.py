"""M2 (independent authorization-stack review, 2026-11-22): ``ToolRegistry.get(name).run(args)`` used to
return the raw :class:`agent_harness.tools.ToolPlugin` and execute it with no authorization or approval
check at all, bypassing both gates ``ToolRegistry.run()`` enforces. The fix is :class:`ToolHandle`: ``get()``
now returns a read-only view exposing only ``spec()``, with no ``run`` method to call at all, so a caller
cannot reach for the obvious attribute and get an unguarded execution path by accident. These cases are
written so a reintroduction of the old ``get()`` behaviour (returning the raw plugin) fails loudly: each one
asserts the tool body never ran, not just that the registry *looks* fine.
"""
import pytest

from agent_harness.tools import (ApprovalPolicy, Authorization, ToolError, ToolHandle, ToolRegistry,
                                 ToolResult, function_tool, set_default_authorizer)


class Recorder:
    """A tool body that records that it ran, so "denied"/"blocked" can be proved, not just asserted."""

    def __init__(self):
        self.calls = []

    def consequential(self, name="delete_all", action="delete"):
        def body(args):
            self.calls.append((name, args))
            return ToolResult(ok=True, text=f"{name} executed")

        return function_tool(name, body, side_effecting=True, action=action)

    @property
    def ran(self) -> bool:
        return bool(self.calls)


@pytest.fixture
def body():
    return Recorder()


@pytest.fixture(autouse=True)
def _no_process_default():
    set_default_authorizer(None)
    yield
    set_default_authorizer(None)


def deny(principal, spec, args):
    return Authorization.deny("no grant")


def allow(principal, spec, args):
    return Authorization.allow(grant_ids=("test-grant",))


# -- M2-01: get() returns a handle with no run() at all, under any authorizer state --------------------

def test_get_returns_a_tool_handle_not_the_raw_plugin(body):
    reg = ToolRegistry(authorizer=allow, policy=ApprovalPolicy("deny_side_effects", approver=lambda c: True))
    reg.register(body.consequential())
    handle = reg.get("delete_all")
    assert isinstance(handle, ToolHandle)
    assert not hasattr(handle, "run"), "ToolHandle must not expose run() under any circumstances"


# -- M2-02: the exact bypass this review demonstrated must now fail structurally, not just return a denial --

def test_get_then_run_can_no_longer_execute_an_unauthorized_tool(body):
    reg = ToolRegistry(authorizer=deny, policy=ApprovalPolicy("deny_side_effects"))
    reg.register(body.consequential())
    with pytest.raises(AttributeError):
        reg.get("delete_all").run({})
    assert not body.ran, "the tool body must never have executed"


# -- M2-03: the gated path still denies correctly (no regression in the authorization check itself) -------

def test_registry_run_still_denies_without_a_grant(body):
    reg = ToolRegistry(authorizer=deny, policy=ApprovalPolicy("deny_side_effects"))
    reg.register(body.consequential())
    res = reg.run("delete_all", {})
    assert not res.ok
    assert not body.ran


# -- M2-04: the gated path still allows correctly when a real grant is present -----------------------------

def test_registry_run_still_allows_with_a_real_grant(body):
    reg = ToolRegistry(authorizer=allow, policy=ApprovalPolicy("deny_side_effects", approver=lambda c: True))
    reg.register(body.consequential())
    res = reg.run("delete_all", {})
    assert res.ok
    assert body.ran


# -- M2-05: a tool authorized once is denied after the grant is revoked (via a stateful authorizer) --------

def test_revoked_grant_is_denied_on_the_next_call_not_just_the_first(body):
    state = {"granted": True}

    def stateful(principal, spec, args):
        return Authorization.allow(grant_ids=("g1",)) if state["granted"] else Authorization.deny("revoked")

    reg = ToolRegistry(authorizer=stateful, policy=ApprovalPolicy("deny_side_effects", approver=lambda c: True))
    reg.register(body.consequential())

    first = reg.run("delete_all", {})
    assert first.ok and body.ran

    state["granted"] = False
    body.calls.clear()
    second = reg.run("delete_all", {})
    assert not second.ok, "a revoked grant must deny the very next call"
    assert not body.ran

    # and get().run() still cannot be used to work around the revocation either
    with pytest.raises(AttributeError):
        reg.get("delete_all").run({})
    assert not body.ran


# -- M2-06: ToolHandle's one legitimate use -- spec() introspection -- is unaffected ------------------------

def test_spec_introspection_through_get_still_works(body):
    reg = ToolRegistry(authorizer=allow)
    reg.register(body.consequential(action="delete"))
    assert reg.get("delete_all").spec().action == "delete"
    assert reg.required_capability("delete_all") == {
        "resource_kind": "tool", "resource_name": "delete_all", "requested_action": "delete"}


# -- M2-07: an unknown tool still raises ToolError from get(), not a different error shape -----------------

def test_get_unknown_tool_still_raises_tool_error(body):
    reg = ToolRegistry(authorizer=allow)
    with pytest.raises(ToolError):
        reg.get("does_not_exist")


# -- M2-08: reaching past the documented-private attribute is a deliberate act, not the bypass this
#    review tested -- and the review's own live reproduction was against the advertised run() method, not
#    against digging into ToolHandle's internals. Recorded here so the distinction is explicit and tested,
#    not just asserted in prose.

def test_reaching_into_the_private_attribute_is_not_the_tested_bypass_shape(body):
    reg = ToolRegistry(authorizer=deny, policy=ApprovalPolicy("deny_side_effects"))
    reg.register(body.consequential())
    handle = reg.get("delete_all")
    # This still works -- it is a documented, deliberate act of reaching past a single-underscore
    # "not public API" convention, the same convention this module already relies on for its own internal
    # methods (_check_classifiable, _check_no_downgrade, _lookup). It is not the bypass M2 was about, which
    # was calling the one method ToolHandle actually advertises.
    handle._tool.run({})
    assert body.ran, "documented: this is a deliberate violation of the privacy convention, not a fix target"
