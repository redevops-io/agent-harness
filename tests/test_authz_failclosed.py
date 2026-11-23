"""The fail-closed contract of the tool registry (audit KERNEL F-01/F-02/F-03, cases K-01…K-08).

Before this contract, an authorizer that returned ``None`` — by design, by accident, or because it was
never installed at all — meant *allow*, so every tool ran unchecked at every construction site that did not
remember to wire a policy. The cases below are the ones that would have caught it, and they are written so
that a reintroduction of the old default fails loudly: each one asserts the tool body never ran, not just
that a result came back unhappy.
"""
import threading

import pytest

from tests._authz import reset_local_trust_warning

from agent_harness.tools import (ApprovalPolicy, Authorization, Decision, PolicyReason, ToolRegistry, ToolResult,
                                 ToolSpec, as_authorization, current_principal, default_authorizer_installed,
                                 engine_authorizer_of, local_trusted_authorizer, set_default_authorizer)


# -- fixtures ---------------------------------------------------------------------------------------------

class Recorder:
    """A tool body that records that it ran, so "denied" can be proved instead of trusted."""

    def __init__(self):
        self.calls = []

    def tool(self, name="probe", *, side_effecting=False, approval_required=False):
        from agent_harness.tools import function_tool

        def body(args):
            self.calls.append((name, args, current_principal()))
            return ToolResult(ok=True, text=f"{name} ran")

        return function_tool(name, body, side_effecting=side_effecting, approval_required=approval_required)

    @property
    def ran(self) -> bool:
        return bool(self.calls)


@pytest.fixture
def body():
    return Recorder()


@pytest.fixture(autouse=True)
def _no_process_default():
    """A test that installs a process-wide authorizer must clean up after itself — a leaked default would
    silently authorize every later test in the session."""
    set_default_authorizer(None)
    yield
    set_default_authorizer(None)


def allow(principal, spec, args):
    return Authorization.allow(grant_ids=("test-grant",))


# -- K-01: no policy plane is not permission ---------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [{}, {"policy": ApprovalPolicy("bypass")},
                                    {"policy": ApprovalPolicy("allowlist", allow=["probe"])}])
@pytest.mark.parametrize("flags", [{}, {"side_effecting": True}, {"approval_required": True},
                                   {"side_effecting": True, "approval_required": True}])
def test_registry_without_authorizer_denies_every_tool_and_never_runs_the_body(body, kwargs, flags):
    reg = ToolRegistry(**kwargs)
    reg.register(body.tool(**flags))
    res = reg.run("probe", {"x": 1})
    assert not res.ok
    assert "missing_explicit_grant" in res.error and "missing_explicit_grant" in res.text
    assert not body.ran, "a tool ran with no authorization decision"


def test_approval_bypass_does_not_substitute_for_authorization(body):
    """The old code's worst case: ``ApprovalPolicy("bypass")`` (used by the bench and by tests) let a call
    through even though nobody had decided that the caller may make it."""
    reg = ToolRegistry(ApprovalPolicy("bypass"))
    reg.register(body.tool(name="mutating", side_effecting=True))
    assert not reg.run("mutating", {}).ok and not body.ran


def test_register_list_and_specs_still_work_without_an_authorizer(body):
    """Fail-closed applies to execution. Describing tools to a model is not a tool call — a registry that
    refused to enumerate its tools would break every agent loop before it reached the gate."""
    reg = ToolRegistry()
    reg.register(body.tool())
    assert reg.list() == ["probe"] and reg.specs()[0]["function"]["name"] == "probe"
    assert not body.ran


def test_unknown_tool_still_reports_as_unknown(body):
    reg = ToolRegistry()
    res = reg.run("not_registered", {})
    assert "unknown tool" in res.error and not body.ran


# -- K-02: a decision that is not recorded did not happen ---------------------------------------------------

def test_every_denial_is_audited_whatever_denied_it(body):
    reg = ToolRegistry(authorizer=lambda p, spec, args: Authorization.deny("out of scope"))
    reg.register(body.tool())
    assert not reg.run("probe", {}).ok
    row = reg.audit[-1]
    assert row["allowed"] is False and row["tool"] == "probe"
    assert row["policy_reason"] == PolicyReason.MISSING_EXPLICIT_GRANT   # nothing granted it
    assert "out of scope" in row["reason"]


def test_an_unattributed_allow_is_visible(body):
    reg = ToolRegistry(authorizer=lambda p, spec, args: True)      # legacy shape, still an allow
    reg.register(body.tool())
    assert reg.run("probe", {}).ok
    unattributed = reg.unattributed_allows()
    assert len(unattributed) == 1 and unattributed[0]["legacy"] is True


def test_approval_layer_records_keep_their_historical_shape(body):
    """The gate that predates this contract wrote four keys; consumers read them. Adding keys there would
    break the vendored copy's parity, so the extra classification lives only on authorization records."""
    reg = ToolRegistry(ApprovalPolicy("deny_side_effects"), authorizer=allow)
    reg.register(body.tool(side_effecting=True))
    assert not reg.run("probe", {}).ok
    assert set(reg.audit[-1]) == {"tool", "args", "allowed", "reason"}


# -- K-03: the principal is the one that was passed ---------------------------------------------------------

def test_decision_follows_the_principal_and_names_it_in_the_audit(body):
    seen = []

    def authz(principal, spec, args):
        seen.append(principal)
        roles = (principal or {}).get("roles", set())
        return (Authorization.allow(grant_ids=("sec-ops",)) if "security" in roles
                else Authorization.deny("role not permitted"))

    reg = ToolRegistry(authorizer=authz)
    reg.register(body.tool(name="findings.read"))
    assert reg.run("findings.read", {}, principal={"roles": {"security"}}).ok
    assert reg.audit == []                                        # a named grant is not audited as a gate
    assert reg.unattributed_allows() == []
    denied = reg.run("findings.read", {}, principal={"roles": {"guest"}})
    assert not denied.ok and "role not permitted" in denied.error
    assert reg.audit[-1]["plane"] == "tool_registry"
    assert seen == [{"roles": {"security"}}, {"roles": {"guest"}}]


def test_principal_is_visible_to_the_tool_and_does_not_leak(body):
    reg = ToolRegistry(authorizer=allow)
    reg.register(body.tool())
    reg.run("probe", {}, principal="app:svc")
    name, args, principal = body.calls[0]
    assert (name, principal) == ("probe", "app:svc") and args == {}
    assert current_principal() is None


# -- K-04: an authority that cannot be reached is a denial ----------------------------------------------------

@pytest.mark.parametrize("boom", [ConnectionError("authz store down"), KeyError("grant"), ValueError("bad epoch")])
def test_authorizer_exception_denies_and_is_classified_as_outage(body, boom):
    def authz(principal, spec, args):
        raise boom

    reg = ToolRegistry(authorizer=authz)
    reg.register(body.tool())
    res = reg.run("probe", {})
    assert not res.ok and not body.ran
    assert reg.audit[-1]["policy_reason"] == PolicyReason.NO_RESOURCE_PLANE
    assert type(boom).__name__ in reg.audit[-1]["detail"]
    # the registry is still usable, and still closed: the next call re-asks and re-denies
    assert not reg.run("probe", {}).ok and reg.audit[-1]["policy_reason"] == PolicyReason.NO_RESOURCE_PLANE


def test_default_authorizer_outage_denies_registries_that_inherit_it(body):
    def authz(principal, spec, args):
        raise RuntimeError("policy service unreachable")

    reg = ToolRegistry()
    reg.register(body.tool())
    set_default_authorizer(authz)
    assert not reg.run("probe", {}).ok and not body.ran
    assert reg.audit[-1]["policy_reason"] == PolicyReason.NO_RESOURCE_PLANE


def test_authorizer_returning_a_junk_value_denies(body):
    for junk in ({"allow": True}, ["yes"], 1, object()):
        reg = ToolRegistry(authorizer=lambda p, spec, args, j=junk: j)
        reg.register(body.tool())
        assert not reg.run("probe", {}).ok and not body.ran
        assert reg.audit[-1]["policy_reason"] == PolicyReason.MALFORMED_REQUEST


# -- K-05: a default authorizer governs every registry, and clearing it closes them ----------------------------

def test_default_authorizer_governs_and_clearing_fails_closed(body):
    reg = ToolRegistry()
    reg.register(body.tool())
    set_default_authorizer(allow)
    assert reg.run("probe", {}).ok and body.ran
    set_default_authorizer(None)
    assert not reg.run("probe", {}).ok
    assert reg.audit[-1]["policy_reason"] == PolicyReason.MISSING_EXPLICIT_GRANT


def test_registries_in_other_threads_do_not_inherit_a_cleared_default(body):
    """The default is process-wide by contract; a registry built with its own authorizer must not be moved
    by what another thread does to the default."""
    own = ToolRegistry(authorizer=allow)
    own.register(body.tool(name="own"))
    bare = ToolRegistry()
    bare.register(body.tool(name="bare"))
    results = {}

    def worker():
        set_default_authorizer(lambda p, spec, args: "denied here")
        results["bare"] = bare.run("bare", {}).ok
        results["own"] = own.run("own", {}).ok

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert results == {"bare": False, "own": True}


# -- K-06: a tool's own flags are not authorization ----------------------------------------------------------

def test_side_effect_flag_cannot_authorize_a_read_tool(body):
    """Marking every tool side-effecting is what makes the old approval gate *deny*; it is not what makes
    the authorization gate *allow*. The two layers stay independent in both directions."""
    reg = ToolRegistry(ApprovalPolicy("deny_side_effects"))
    reg.register(body.tool(name="read_only", side_effecting=True, approval_required=True))
    first = reg.run("read_only", {})
    assert not first.ok and "missing_explicit_grant" in first.error       # authorization gate
    assert not body.ran
    approved = ToolRegistry(ApprovalPolicy("deny_side_effects", approver=lambda a: True), authorizer=allow)
    approved.register(body.tool(name="read_only", side_effecting=True, approval_required=True))
    assert approved.run("read_only", {}).ok


# -- K-07: both gates speak, in order, and neither is a way around the other ------------------------------------

def test_authorizer_cannot_be_bypassed_by_the_approval_gate(body):
    denied = ToolRegistry(ApprovalPolicy("bypass"), authorizer=lambda p, spec, args: Authorization.deny("nope"))
    denied.register(body.tool())
    assert not denied.run("probe", {}).ok and not body.ran


def test_approval_gate_cannot_be_bypassed_by_the_authorizer(body):
    gated = ToolRegistry(ApprovalPolicy("deny_side_effects"), authorizer=allow)
    gated.register(body.tool(side_effecting=True))
    res = gated.run("probe", {})
    assert not res.ok and "requires approval" in res.error and not body.ran


def test_authorizer_runs_before_the_tool_and_before_the_approval_gate(body):
    order = []

    def authz(principal, spec, args):
        order.append("authorizer")
        return Authorization.allow(grant_ids=("g",))

    reg = ToolRegistry(ApprovalPolicy("deny_side_effects", approver=lambda a: order.append("approver") or True),
                       authorizer=authz)
    reg.register(body.tool(side_effecting=True))
    assert reg.run("probe", {}).ok
    assert order == ["authorizer", "approver"]


def test_a_subclass_cannot_run_a_tool_without_asking(body):
    """Independent review finding M2 (2026-11-22): this test used to assert the opposite of what it's
    called — that ``self.get(name).run(args)`` bypassing the gate was an acceptable, merely "visible in
    code" escape hatch. It was a real authorization/approval bypass, closed by making ``get()`` return a
    :class:`agent_harness.tools.ToolHandle` with no ``run`` method at all (see ``ToolHandle``'s docstring).
    A subclass "forgetting" the authorization layer by reaching for the obvious ``get(name).run(...)`` call
    now fails structurally, the same as any other caller — there is no longer a code path, forgetful or
    deliberate, that reaches execution through the registry's public surface except ``run()`` itself."""

    class Forgetful(ToolRegistry):
        def run_anyway(self, name, args=None):
            return self.get(name).run(args or {})

    reg = Forgetful()
    reg.register(body.tool())
    assert not reg.run("probe", {}).ok and not body.ran          # the normal path is closed
    with pytest.raises(AttributeError):                          # and so is the old "obvious" bypass
        reg.run_anyway("probe", {})
    assert not body.ran


# -- K-08: the mapping layer never resolves to something that was not registered ---------------------------------

def test_unregistered_name_reaches_no_plugin(body):
    reg = ToolRegistry(authorizer=allow)
    reg.register(body.tool(name="known"))
    for name in ("", "Known", "known ", "known/x", "other"):
        res = reg.run(name, {})
        assert "unknown tool" in res.error
    assert [c[0] for c in body.calls] == []


def test_authorizer_is_never_asked_about_an_unknown_tool(body):
    asked = []
    reg = ToolRegistry(authorizer=lambda p, spec, args: asked.append(spec.name) or Authorization.allow())
    assert "unknown tool" in reg.run("ghost", {}).error
    assert asked == []


# -- the legacy return shapes, now marked ---------------------------------------------------------------------

@pytest.mark.parametrize("raw,allowed,reason", [
    (None, False, PolicyReason.MISSING_EXPLICIT_GRANT),        # the shape that made the old seam fail open
    ("", False, PolicyReason.MALFORMED_REQUEST),               # an empty deny reason is not a decision
    ("denied by tenant policy", False, PolicyReason.MISSING_EXPLICIT_GRANT),
    (True, True, PolicyReason.EXPLICIT_GRANT),
    (False, False, PolicyReason.MISSING_EXPLICIT_GRANT),
    (0, False, PolicyReason.MALFORMED_REQUEST),
])
def test_as_authorization_reads_the_old_shapes(raw, allowed, reason):
    decision = as_authorization(raw)
    assert decision.allowed is allowed and decision.reason == reason
    assert decision.legacy is True


def test_a_typed_decision_passes_through_untouched():
    typed = Authorization.allow(reason=PolicyReason.WILDCARD_GRANT, grant_ids=("g-1",), plane="authz/v1")
    assert as_authorization(typed) is typed
    assert Decision.ALLOW == "allow"


# -- the explicit opt-outs --------------------------------------------------------------------------------------


# -- an authorizer installed from another repository's contract ---------------------------------------------------


def _foreign_plane():
    """A policy plane written outside this repository, handed over as the process default.

    It answers the way a platform authorizer answers — ``None`` for allowed, a string for why not — and it carries
    ``engine_authorizer``, the same decisions in this repository's contract. They cannot be the same callable: here
    a bare ``None`` is "nobody decided" and denies (K-01), so calling the plane directly would deny every call the
    plane just allowed.
    """

    def plane(principal, spec, args):
        return None if spec.name == "probe" else "the plane has no grant for that"

    def engine_shaped(principal, spec, args):
        if spec.name == "probe":
            return Authorization.allow(reason=PolicyReason.EXPLICIT_GRANT, detail="the plane granted it",
                                       plane="the foreign plane")
        return Authorization.deny("the plane has no grant for that")

    plane.engine_authorizer = engine_shaped
    return plane


def test_a_foreign_process_default_runs_the_tools_it_allows(body):
    """The install point takes one callable; the shape a registry can act on travels on it."""
    set_default_authorizer(_foreign_plane())
    reg = ToolRegistry()
    reg.register(body.tool())
    assert reg.run("probe", {}).ok and body.ran, "the plane allowed this call and the registry denied it"


def test_a_foreign_process_default_still_denies_what_the_plane_denies(body):
    """The twin is a translation, not a loosening: what the plane refused stays refused, in the plane's words."""
    set_default_authorizer(_foreign_plane())
    reg = ToolRegistry()
    reg.register(body.tool(name="other"))
    res = reg.run("other", {})
    assert not res.ok and not body.ran and "the plane has no grant for that" in res.text


def test_a_foreign_authorizer_handed_to_one_registry_is_read_the_same_way(body):
    """The convention is about the callable, not about the seat it sits in."""
    reg = ToolRegistry(authorizer=_foreign_plane())
    reg.register(body.tool())
    assert reg.run("probe", {}).ok and body.ran


def test_the_engine_shape_is_chosen_only_when_it_is_there():
    plane = _foreign_plane()
    assert engine_authorizer_of(plane) is plane.engine_authorizer
    assert engine_authorizer_of(allow) is allow            # already speaks this contract: untouched
    assert engine_authorizer_of(None) is None              # nothing installed stays nothing installed


def test_a_non_callable_engine_shape_is_not_trusted():
    """An attribute that is not a callable is a bug in whoever set it; falling back to the callable that answers
    is at least a legible denial, and never a crash inside the gate."""
    def speaks_ours(principal, spec, args):
        return Authorization.allow()

    speaks_ours.engine_authorizer = "not a callable"
    assert engine_authorizer_of(speaks_ours) is speaks_ours


def test_local_trusted_authorizer_allows_and_says_so(body):
    reg = ToolRegistry(authorizer=local_trusted_authorizer())
    reg.register(body.tool())
    assert reg.run("probe", {}).ok and body.ran
    row = reg.audit[-1]
    assert row["allowed"] is True and row["plane"] == "local_trusted"
    assert row["policy_reason"] == PolicyReason.WILDCARD_GRANT


def test_local_trust_is_not_an_approval_bypass(body):
    """Trusting the *caller* is not the same as waiving the side-effect gate — the two layers stay apart."""
    reg = ToolRegistry(ApprovalPolicy("deny_side_effects"), authorizer=local_trusted_authorizer())
    reg.register(body.tool(side_effecting=True))
    assert not reg.run("probe", {}).ok and not body.ran
    assert reg.audit[-1]["reason"] == "side-effecting tool requires approval"


def test_local_trusted_authorizer_warns_once(body, recwarn):
    reset_local_trust_warning()
    reg = ToolRegistry(authorizer=local_trusted_authorizer())
    reg.register(body.tool())
    reg.run("probe", {})
    reg.run("probe", {})
    messages = [str(w.message) for w in recwarn.list if "local_trusted authorizer" in str(w.message)]
    assert len(messages) == 1, messages
    ToolRegistry(authorizer=local_trusted_authorizer()).run("probe", {})
    assert len([w for w in recwarn.list if "local_trusted authorizer" in str(w.message)]) == 1


def test_authorizer_required_false_is_a_loud_opt_out(body):
    with pytest.warns(UserWarning, match="authorizer_required=False"):
        reg = ToolRegistry(authorizer_required=False)
    reg.register(body.tool())
    assert reg.run("probe", {}).ok and body.ran
    row = reg.audit[-1]
    assert row["policy_reason"] == PolicyReason.WILDCARD_GRANT and row["legacy"] is True


def test_the_package_exports_the_contract():
    import agent_harness.tools as tools

    for name in ("Authorization", "Decision", "PolicyReason", "as_authorization", "local_trusted_authorizer",
                 "default_authorizer_installed", "set_default_authorizer", "engine_authorizer_of", "ToolRegistry"):
        assert hasattr(tools, name), name
    assert default_authorizer_installed() is False        # nothing leaks from another test