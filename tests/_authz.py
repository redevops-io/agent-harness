"""Authorizers the test suite uses on purpose.

A registry in a test is either *about* authorization — then the test builds the authorizer it is testing —
or it is about something else (the loop, the context manager, the process runner) and needs a gate that
stays out of the way. ``allow_all`` is for the second case: an explicit, attributed allow, passed at the
construction site so the call reads as a decision.

There is deliberately no autouse fixture installing one globally. A suite that made "everything allowed"
the ambient condition could no longer see the fail-open this contract exists to prevent.
"""
from agent_harness.tools import Authorization, local_trusted_authorizer


def allow_all(principal, spec, args) -> Authorization:
    """Attribute the allow to a named test grant, so it is distinguishable from a policy plane's decision."""
    return Authorization.allow(grant_ids=("test-explicit",), plane="test")


def local_trusted():
    """The authorizer a local/direct caller installs (used where a test exercises that path itself)."""
    return local_trusted_authorizer(plane="test_local")


def reset_local_trust_warning():
    """Re-arm the once-per-process local-trust warning, for the one test that asserts on it.

    The flag is a module global on purpose — the fact it reports is process-wide — so a test that wants to
    see it has to be the first caller or reset it. Reaching into the private is deliberate and lives here so
    it reads as test plumbing rather than as API.
    """
    import agent_harness.tools as tools

    tools._local_trust_warned = False