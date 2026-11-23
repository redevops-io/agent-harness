"""`agent_harness.tools` must speak `authz/v1` without depending on it.

The stack ships two different `runtime-contracts` versions at once: 0.3.14 installed in the platform venv,
which has no `authz` subpackage, and the repository head, where the contract lives. So the vocabulary is
declared locally in :mod:`agent_harness.tools` and pinned here in two ways — against a snapshot carrying the
commit it was read from, and against the real module whenever one is importable. The snapshot keeps the
check meaningful on a machine with neither checkout; the live comparison stops the snapshot from rotting.
"""
import importlib.util
import pathlib
import subprocess
import sys

import pytest

from agent_harness import tools as ah

#: Snapshot of ``runtime_contracts.authz.decision.PolicyReason`` at runtime-contracts 15f02e3
#: ("authz/v1: issue exact capabilities instead of families, and call a missing checkout BLOCKED").
#: Update it only together with a contract change, and re-read the module — do not edit it to match.
CONTRACT_POLICY_REASONS = {
    "explicit_grant", "wildcard_grant", "missing_explicit_grant", "tenant_isolation", "no_resource_plane",
    "invalid_principal", "malformed_request", "sealed_store", "own_rows_requires_user",
    "principal_expired", "unclassified",
}
CONTRACT_COMMIT = "15f02e3"

#: Snapshot of the capability field names at the same commit — the three envelope fields that say *what is
#: being asked for* (``AuthorizationDecision.resource_kind`` / ``.resource_name`` / ``.requested_action``).
#: ``ToolRegistry.required_capability`` writes exactly these, because a capability named anything else is a
#: capability no plane will look up.
CAPABILITY_FIELDS = ("resource_kind", "resource_name", "requested_action")

#: Snapshot of the action vocabulary of the resource-grant plane (``agentic_os.permissions.ACTIONS``) at the
#: commit this branch was written against. Same reason as above: the registry states the requirement, the
#: plane issues the grant, and the two only meet if "write" means the same thing on both sides.
PLATFORM_ACTIONS = ("read", "write", "delete")

#: Reasons a tool call can genuinely produce. The rest belong to planes this registry does not own: a tool
#: registry has no tenant binding to break (tenant_isolation), no grant store to seal (sealed_store), no
#: principal lifetime of its own (principal_expired), and rows are not its unit (own_rows_requires_user).
LOCAL_REASONS = {
    ah.PolicyReason.EXPLICIT_GRANT, ah.PolicyReason.WILDCARD_GRANT, ah.PolicyReason.MISSING_EXPLICIT_GRANT,
    ah.PolicyReason.NO_RESOURCE_PLANE, ah.PolicyReason.MALFORMED_REQUEST, ah.PolicyReason.UNCLASSIFIED,
}


def _installed_contract():
    """The real `authz/v1` module if any is importable — from site-packages or a sibling checkout."""
    try:
        if importlib.util.find_spec("runtime_contracts.authz") is not None:
            import runtime_contracts.authz as authz

            return authz, f"installed:{getattr(sys.modules.get('runtime_contracts'), '__version__', 'unknown')}"
    except (ImportError, ValueError):
        pass                       # the parent package is absent, which is exactly the 0.3.14 case
    for candidate in (pathlib.Path(__file__).resolve().parents[2] / "runtime-contracts",):
        module = candidate / "runtime_contracts"
        if (module / "authz").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            import runtime_contracts.authz as authz

            return authz, f"checkout:{candidate}"
    return None, "absent"


def test_the_package_needs_no_runtime_contracts_to_import():
    """The dependency declaration says `dependencies = []`. The authorization seam is the place that claim
    would break first, so it is asserted rather than trusted: importing and using it must not pull in
    ``runtime_contracts`` (0.3.14, the version installed in the stack, has no `authz` subpackage at all).

    Run in a clean interpreter: this suite imports the contract elsewhere, so checking ``sys.modules`` here
    would only prove the test order, not the import graph.
    """
    probe = ("import sys, agent_harness.tools as t;"
             "r = t.ToolRegistry(authorizer=lambda p, s, a: t.Authorization.allow(grant_ids=('g',)));"
             "r.register(t.function_tool('x', lambda a: t.ToolResult(text='y')));"
             "assert r.run('x', {}).ok;"
             "assert not r.run('missing', {}).ok;"
             "assert 'runtime_contracts' not in sys.modules, sorted(m for m in sys.modules if 'runtime' in m)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=str(pathlib.Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stdout + out.stderr


def test_local_vocabulary_is_a_subset_of_the_contract():
    """Every reason this registry emits must be a member of the closed vocabulary — an invented reason is
    how a closed set becomes an open one, and this is the gate that stops it."""
    assert {r.value if hasattr(r, "value") else r for r in LOCAL_REASONS} <= CONTRACT_POLICY_REASONS


def test_every_classified_decision_carries_a_known_reason():
    """Whatever the code paths emit, they emit vocabulary — never a bare message."""
    samples = [ah.Authorization.allow(), ah.Authorization.deny("x"), ah.as_authorization(None),
               ah.as_authorization(True), ah.as_authorization(""), ah.as_authorization(object()),
               ah.ToolRegistry(authorizer=lambda p, s, a: (_ for _ in ()).throw(RuntimeError("down")))
               .authorize(ah.ToolSpec("t"), {})]
    known = {r.value if hasattr(r, "value") else r for r in LOCAL_REASONS}
    for decision in samples:
        assert decision.reason in known, decision
        assert isinstance(decision.allowed, bool)


def test_snapshot_matches_the_contract_when_it_is_available():
    """Skipped only where no `runtime-contracts` is on the machine. Where it is — including the platform
    venv, which has 0.3.14 — a stale snapshot or a reason outside the vocabulary is a hard failure."""
    contract, source = _installed_contract()
    if contract is None:
        pytest.skip(f"no runtime_contracts.authz importable ({source}); the snapshot check above still ran")
    if not hasattr(contract, "PolicyReason"):
        pytest.skip(f"runtime_contracts.authz present but has no PolicyReason ({source}) — pre-authz/v1 "
                    f"version, e.g. 0.3.14; the registry must keep working without it")
    real = {r.value for r in contract.PolicyReason}
    assert real == CONTRACT_POLICY_REASONS, (
        f"the authz/v1 vocabulary moved ({source}); re-read decision.py and update the snapshot, "
        f"contract={sorted(real)} snapshot={sorted(CONTRACT_POLICY_REASONS)}")
    assert {r.value if hasattr(r, "value") else r for r in LOCAL_REASONS} <= real
    # and the envelope really accepts what we would emit for a denied tool call
    decision = ah.Authorization.deny("no authorizer", reason=ah.PolicyReason.MISSING_EXPLICIT_GRANT)
    assert contract.PolicyReason(decision.reason) is contract.PolicyReason.MISSING_EXPLICIT_GRANT


def test_a_required_capability_is_written_in_the_envelopes_field_names():
    """The registry's answer to "what does this call need?" must be readable by a plane that has never seen
    this package: the same three field names the envelope uses, and nothing else invented along the way."""
    reg = ah.ToolRegistry(authorizer=lambda p, s, a: ah.Authorization.allow(grant_ids=("g",)))
    reg.register(ah.function_tool("block_ip", lambda a: ah.ToolResult(text="y"), action="write",
                                  side_effecting=True))
    triple = reg.required_capability("block_ip")
    assert set(triple) == set(CAPABILITY_FIELDS)
    assert triple == {"resource_kind": "tool", "resource_name": "block_ip", "requested_action": "write"}


def test_the_capability_field_names_match_the_contract_when_it_is_available():
    contract, source = _installed_contract()
    if contract is None or not hasattr(contract, "ENVELOPE_FIELDS"):
        pytest.skip(f"no authz/v1 envelope to compare against ({source})")
    assert set(CAPABILITY_FIELDS) <= set(contract.ENVELOPE_FIELDS), (
        f"the envelope no longer carries {CAPABILITY_FIELDS} as declared ({source}): "
        f"{sorted(contract.ENVELOPE_FIELDS)}")


def test_the_action_vocabulary_matches_the_grant_plane_when_it_is_available():
    """``agentic_os.permissions.ACTIONS`` is where grants are actually written. A fourth action added there —
    or one dropped — has to reach this package deliberately, with a release note, not by surprise."""
    try:
        from agentic_os import permissions as platform_permissions
    except ImportError:
        pytest.skip("agentic_os is not installed here; the snapshot above still pins the vocabulary")
    assert ah.ACTIONS == platform_permissions.ACTIONS == PLATFORM_ACTIONS
