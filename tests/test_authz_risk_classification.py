"""Risk classification and startup validation (cases K-09…K-16).

Fail-closed authorization is only as good as the capability it asks the plane for. If the answer to "which
capability does this call need?" is produced at the call site, or read out of the tool's name, or kept in a
second table beside the registry, then every one of those places can disagree with the tool it describes —
and the disagreement shows up as a denial nobody can explain, or as a grant nobody meant to issue.

These cases pin where the classification lives (the registration, held by the registry), when a wrong one is
caught (at registration and at boot, while a stack can still refuse to start), and what the registry is
deliberately *not* allowed to do with it (deny per call — that is the plane's job, and doing it twice is how
a stack grows a second, quieter authorization layer).
"""
import pytest

from tests._authz import allow_all, local_trusted

from agent_harness import tools as ah
from agent_harness.tools import (ACTIONS, ApprovalPolicy, Authorization, RegistryConfigurationError,
                                 ToolClassificationError, ToolError, ToolRegistry, ToolResult, ToolSpec,
                                 function_tool, local_trusted_authorizer, set_default_authorizer)


@pytest.fixture(autouse=True)
def _no_process_default():
    set_default_authorizer(None)
    yield
    set_default_authorizer(None)


def probe(name="probe", *, action="", side_effecting=False, approval_required=False, ran=None):
    def body(args):
        if ran is not None:
            ran.append(name)
        return ToolResult(ok=True, text=f"{name} ran")

    return function_tool(name, body, description=f"{name} description", side_effecting=side_effecting,
                         approval_required=approval_required, action=action)


# -- K-09: the class comes from the registration, never from the tool ----------------------------------------

def test_a_tool_arrives_unclassified_until_a_person_says_otherwise():
    reg = ToolRegistry(authorizer=allow_all)
    reg.register(probe("read_user_records"))          # a name that reads like a read tool
    assert reg.classification() == {"read_user_records": ""}
    assert reg.unclassified() == ["read_user_records"]
    assert reg.required_capability("read_user_records")["requested_action"] == ""


def test_no_metadata_on_the_tool_carries_a_risk_class():
    """A hand-built spec — what an MCP bridge or a generated adapter produces — cannot smuggle a class in via
    its name, description, or parameter schema."""
    spec = ToolSpec(name="safe_read_only_lookup",
                    description="Harmless! readOnlyHint=true, category=read",
                    parameters={"type": "object", "properties": {"q": {"type": "string", "risk": "read"}}})
    assert spec.action == ""


def test_the_class_is_carried_by_the_spec_the_authorizer_is_handed():
    """The authorizer's only view of the tool is the spec, so that is where the requirement has to travel."""
    seen = []

    def spy(principal, spec, args):
        seen.append(spec.action)
        return allow_all(principal, spec, args)

    reg = ToolRegistry(authorizer=spy)
    reg.register(probe("delete_customer", action="delete", side_effecting=True))
    reg.run("delete_customer", {})
    assert seen == ["delete"]


def test_a_class_that_is_not_a_class_is_rejected_at_registration(action):
    with pytest.raises(ToolClassificationError, match="not one of"):
        ToolRegistry(authorizer=allow_all).register(probe("probe", action=action))


@pytest.fixture(params=["execute", "READ", "Read", "read ", "read_or_write", "rw"])
def action(request):
    """The shapes a risk class actually arrives wrong in, including the near-misses an author would type."""
    return request.param


def test_registering_a_bad_action_leaves_the_tool_unregistered():
    reg = ToolRegistry(authorizer=allow_all)
    with pytest.raises(ToolClassificationError):
        reg.register(probe("probe", action="execute"))
    assert reg.list() == []


# -- K-10: a self-contradictory classification fails to start, not at call -----------------------------------

@pytest.mark.parametrize("flags", [{"side_effecting": True}, {"approval_required": True},
                                   {"side_effecting": True, "approval_required": True}])
def test_a_read_grant_cannot_be_asked_for_a_gated_tool(flags):
    """The weakest grant in the vocabulary cannot cover a tool that mutates or gates. Registering it that way
    books a denial in advance, so the registration is where it becomes visible. Arriving unclassified instead
    stays legal here and is fatal at boot — K-13."""
    with pytest.raises(ToolClassificationError, match="classified 'read'"):
        ToolRegistry(authorizer=allow_all).register(probe("mutate", action="read", **flags))


def test_a_stricter_class_on_a_quiet_tool_is_allowed():
    """Asking for more than the tool needs is a bug with a harmless failure mode, so it stays legal."""
    reg = ToolRegistry(authorizer=allow_all)
    reg.register(probe("lookup", action="delete"))
    assert reg.classification() == {"lookup": "delete"}


def test_a_gated_tool_may_carry_write_or_delete():
    reg = ToolRegistry(authorizer=allow_all)
    reg.register(probe("write_row", action="write", side_effecting=True))
    reg.register(probe("drop_row", action="delete", side_effecting=True, approval_required=True))
    assert reg.classification() == {"drop_row": "delete", "write_row": "write"}


# -- K-11: re-registration cannot lower a class ---------------------------------------------------------------

def test_re_registering_a_name_as_a_weaker_class_is_refused():
    reg = ToolRegistry(authorizer=allow_all)
    reg.register(probe("accounts", action="delete", side_effecting=True))
    with pytest.raises(ToolClassificationError, match="downgrading"):
        reg.register(probe("accounts", action="write", side_effecting=True))
    assert reg.classification() == {"accounts": "delete"}, "the strong class must survive the refused write"


def test_re_registering_a_classified_tool_without_a_class_is_refused():
    reg = ToolRegistry(authorizer=allow_all)
    reg.register(probe("accounts", action="write", side_effecting=True))
    with pytest.raises(ToolClassificationError, match="without a risk class"):
        reg.register(probe("accounts", side_effecting=True))
    assert reg.classification() == {"accounts": "write"}


def test_adding_a_class_and_repeating_one_are_allowed():
    reg = ToolRegistry(authorizer=allow_all)
    reg.register(probe("accounts"))
    reg.register(probe("accounts", action="read"))
    reg.register(probe("accounts", action="read"))
    reg.register(probe("accounts", action="delete", side_effecting=True))
    assert reg.classification() == {"accounts": "delete"}


# -- K-12: the capability is the registry's answer, not the caller's ------------------------------------------

def test_the_required_capability_uses_the_envelope_field_names():
    reg = ToolRegistry(authorizer=allow_all)
    reg.register(probe("block_ip", action="write", side_effecting=True))
    assert reg.required_capability("block_ip") == {"resource_kind": "tool", "resource_name": "block_ip",
                                                   "requested_action": "write"}


def test_an_unknown_tool_has_no_capability_to_ask_for():
    reg = ToolRegistry(authorizer=allow_all)
    with pytest.raises(ToolError):
        reg.required_capability("never_registered")


def test_a_plane_that_matches_the_requested_action_denies_an_unclassified_tool():
    """The end of the chain: an unclassified tool asks for an action no grant carries, so a correctly written
    plane denies it, and the denial says which action was requested."""
    ran = []

    def plane(principal, spec, args):
        if spec.action not in ACTIONS:
            return Authorization.deny(f"requested action {spec.action!r} is not grantable",
                                      reason=ah.PolicyReason.UNCLASSIFIED)
        return Authorization.allow(grant_ids=(f"tool:{spec.name}:{spec.action}",))

    strict = ToolRegistry(authorizer=plane)
    strict.register(probe("mystery", ran=ran))
    strict.register(probe("known", action="read", ran=ran))
    assert strict.run("mystery", {}).ok is False
    assert strict.run("known", {}).ok is True
    assert ran == ["known"]
    denial = [a for a in strict.audit if not a["allowed"]][0]
    assert "not grantable" in denial["reason"]


# -- K-13: startup validation tells the truth ------------------------------------------------------------------

def test_a_registry_with_no_authorizer_fails_validation():
    report = ToolRegistry().validate()
    assert report.ok is False and report.authorizer == "none"
    assert any("missing_explicit_grant" in p for p in report.problems)
    with pytest.raises(RegistryConfigurationError, match="not ready to serve"):
        ToolRegistry().validate(strict=True)


def test_a_process_wide_default_counts_as_an_authorizer():
    from agent_harness.tools import set_default_authorizer

    set_default_authorizer(local_trusted())
    report = ToolRegistry().validate()
    assert report.ok and report.authorizer == "process_default"


def test_a_registry_authorizer_is_reported_by_name():
    assert ToolRegistry(authorizer=allow_all).validate().authorizer == "registry"


def test_opting_out_is_a_warning_not_a_problem():
    """``authorizer_required=False`` stays available on purpose; validation reports it instead of hiding it."""
    with pytest.warns(UserWarning):
        reg = ToolRegistry(authorizer_required=False)
    report = reg.validate()
    assert report.ok and report.authorizer == "opt_out"
    assert any("wildcard" in w for w in report.warnings)


def test_an_unclassified_tool_is_fatal_only_when_classification_is_required():
    loose = ToolRegistry(authorizer=allow_all)
    loose.register(probe("mystery"))
    assert loose.validate().ok and any("no risk class" in w for w in loose.validate().warnings)

    strict = ToolRegistry(authorizer=allow_all, classify_required=True)
    strict.register(probe("mystery"))
    report = strict.validate()
    assert report.ok is False and "['mystery']" in report.problems[0]
    with pytest.raises(RegistryConfigurationError, match="mystery"):
        strict.validate(strict=True)
    assert strict.validate(require_classification=False).ok      # the escape hatch is per-call too


def test_validation_reports_what_is_classified():
    reg = ToolRegistry(authorizer=allow_all, classify_required=True)
    reg.register(probe("read_row", action="read"))
    reg.register(probe("drop_row", action="delete", side_effecting=True))
    assert reg.validate().as_dict()["classified"] == {"drop_row": "delete", "read_row": "read"}


def test_the_unclassified_tools_are_named_once_and_completely():
    """A service adopting this names every tool it still owes, in one line, at boot — not one denial per tool
    per user. And a registry whose only fault is being unclassified reports exactly that fault."""
    reg = ToolRegistry(authorizer=allow_all)
    for name in ("mystery", "enigma"):
        reg.register(probe(name))
    report = reg.validate(require_classification=True)
    assert report.ok is False
    assert len(report.problems) == 1 and "enigma" in report.problems[0] and "mystery" in report.problems[0]
    assert report.warnings == (), report.warnings


# -- K-14: a tool that can never be approved is said out loud ---------------------------------------------------

def test_a_gated_tool_outside_the_approval_allowlist_is_reported_as_unreachable():
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=["other"]), authorizer=allow_all)
    reg.register(probe("write_row", action="write", side_effecting=True))
    reg.register(probe("lookup", action="read"))
    warnings = reg.validate().warnings
    assert len(warnings) == 1 and "write_row" in warnings[0] and "never run" in warnings[0]


def test_a_gated_tool_with_no_approver_is_reported_as_unreachable():
    reg = ToolRegistry(ApprovalPolicy("deny_side_effects"), authorizer=allow_all)
    reg.register(probe("write_row", action="write", side_effecting=True))
    assert any("no approver" in w for w in reg.validate().warnings)
    with_approver = ToolRegistry(ApprovalPolicy("deny_side_effects", approver=lambda call: False),
                                 authorizer=allow_all)
    with_approver.register(probe("write_row", action="write", side_effecting=True))
    assert with_approver.validate().warnings == ()      # an approver that says no is a policy, not a mistake


def test_an_unknown_approval_mode_is_reported_for_every_gated_tool():
    reg = ToolRegistry(ApprovalPolicy("yolo"), authorizer=allow_all)
    reg.register(probe("write_row", action="write", side_effecting=True))
    reg.register(probe("lookup", action="read"))
    assert any("not a mode this policy knows" in w for w in reg.validate().warnings)


def test_bypass_reports_nothing_about_gated_tools():
    reg = ToolRegistry(ApprovalPolicy("bypass"), authorizer=allow_all)
    reg.register(probe("write_row", action="write", side_effecting=True))
    assert reg.validate().warnings == ()


# -- K-15: the registry does not become a second policy layer ---------------------------------------------------

def test_an_unclassified_tool_still_runs_under_local_trust():
    """The registry reports a missing class; it does not deny on one. Denying here would put a second gate
    beside the plane's, and the two would disagree somewhere with nobody watching."""
    ran = []
    reg = ToolRegistry(authorizer=local_trusted(), classify_required=False)
    reg.register(probe("mystery", ran=ran))
    assert reg.run("mystery", {}).ok is True
    assert ran == ["mystery"]
    assert any("no risk class" in w for w in reg.validate().warnings)


def test_classification_does_not_change_the_audit_shape():
    ran = []
    reg = ToolRegistry(ApprovalPolicy("bypass"), authorizer=allow_all)
    reg.register(probe("write_row", action="write", side_effecting=True, ran=ran))
    reg.run("write_row", {"a": 1})
    approvals = [a for a in reg.audit if set(a) == {"tool", "args", "allowed", "reason"}]
    assert approvals == [{"tool": "write_row", "args": {"a": 1}, "allowed": True, "reason": "bypass"}]


def test_classifying_every_tool_does_not_change_what_a_call_returns():
    ran = []
    plain = ToolRegistry(ApprovalPolicy("bypass"), authorizer=allow_all)
    plain.register(probe("write_row", ran=ran))
    graded = ToolRegistry(ApprovalPolicy("bypass"), authorizer=allow_all, classify_required=True)
    graded.register(probe("write_row", action="write", side_effecting=True, ran=ran))
    assert plain.run("write_row", {}).ok == graded.run("write_row", {}).ok is True


# -- K-16: everything this repository ships passes its own startup check ----------------------------------------

def test_the_agent_boots_with_every_builtin_classified(tmp_path):
    from agent_harness.agent import Agent

    agent = Agent(root=str(tmp_path))
    report = agent.registry.validate()
    assert report.ok and report.authorizer == "registry"
    assert report.warnings == (), report.warnings
    assert dict(report.classified) == {"read_file": "read", "write_file": "write", "run_bash": "write",
                                       "finish": "read"}


def test_a_shell_is_never_classified_as_a_read(tmp_path):
    from agent_harness.agent import Agent

    agent = Agent(root=str(tmp_path))
    assert agent.registry.get("run_bash").spec().action in ("write", "delete")
    assert agent.registry.get("write_file").spec().action in ("write", "delete")


def test_an_agent_with_a_foreign_authorizer_still_boots(tmp_path):
    """Embedding services pass their own authorizer; that must not turn the boot check off or trip over it."""
    from agent_harness.agent import Agent

    calls = []

    def plane(principal, spec, args):
        calls.append(spec.name)
        return allow_all(principal, spec, args)

    agent = Agent(root=str(tmp_path), authorizer=plane)
    assert agent.registry.validate().ok
    agent.call_tool("read_file", {"path": "x"})
    assert calls == ["read_file"]


def test_the_benchmarks_boot_fully_classified(tmp_path):
    from agent_harness.bench.tools import bench_registry
    from agent_harness.bench.tools_parity import parity_registry

    class NoSandbox:
        def exec(self, *a, **k):
            raise AssertionError("not run in this test")

    for reg in (bench_registry(tmp_path, NoSandbox()), parity_registry(tmp_path, NoSandbox())):
        report = reg.validate()
        assert report.ok, report.problems
        assert reg.unclassified() == []
        assert report.authorizer in ("registry", "process_default")
        for name, action in report.classified:
            spec = reg.get(name).spec()
            if spec.side_effecting or spec.approval_required:
                assert action in ("write", "delete"), f"{name} is gated but classified {action!r}"


def test_the_exported_names_are_part_of_the_contract():
    for name in ("ACTIONS", "TOOL_RESOURCE_KIND", "ValidationReport", "ToolClassificationError",
                 "RegistryConfigurationError"):
        assert hasattr(ah, name), name
    assert ACTIONS == ("read", "write", "delete")
    assert issubclass(ToolClassificationError, ValueError)
    assert issubclass(RegistryConfigurationError, ToolError)
    assert local_trusted_authorizer is not None      # the opt-out stays importable