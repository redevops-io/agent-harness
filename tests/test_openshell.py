"""SandboxCapability acceptance criteria as tests — typed denials, terminal-on-cancel, credentials
never in evidence, policy change = new identity, success requires a terminal event, backend-agnostic."""
from agent_harness.openshell import (
    LocalSandbox, SandboxPolicy, SandboxCapability, Terminal, Failure,
)


def test_denied_network_is_a_typed_failure():
    r = LocalSandbox().run({"type": "http", "requires_network": True},
                           SandboxPolicy(allow_network=False))
    assert r.terminal == Terminal.DENIED and r.failure == Failure.NETWORK_DENIED
    assert r.exit_code is None  # nothing ran


def test_denied_inference_is_typed():
    r = LocalSandbox().run({"type": "shell", "command": "echo x", "requires_inference": True},
                           SandboxPolicy(allow_process_spawn=True, allow_inference=False))
    assert r.terminal == Terminal.DENIED


def test_success_requires_a_terminal_event():
    r = LocalSandbox().run({"type": "shell", "command": "echo hi"},
                           SandboxPolicy(allow_process_spawn=True))
    assert r.terminal == Terminal.SUCCESS and r.ok and r.exit_code == 0


def test_nonzero_exit_is_failed_not_success():
    r = LocalSandbox().run({"type": "shell", "command": "false"},
                           SandboxPolicy(allow_process_spawn=True))
    assert r.terminal == Terminal.FAILED and r.failure == Failure.NONZERO_EXIT and not r.ok


def test_cancellation_terminates_with_a_terminal_event():
    r = LocalSandbox().run({"type": "shell", "command": "echo hi"},
                           SandboxPolicy(allow_process_spawn=True), cancel=lambda: True)
    assert r.terminal == Terminal.CANCELLED and r.failure == Failure.CANCELLED


def test_policy_change_creates_new_policy_identity():
    a = SandboxPolicy(allow_network=True)
    b = SandboxPolicy(allow_network=True, allowed_hosts=("api.nvidia.com",))
    assert a.identity() != b.identity()
    assert a.identity() == SandboxPolicy(allow_network=True).identity()


def test_credentials_never_enter_evidence():
    r = LocalSandbox().run(
        {"type": "shell", "command": "echo AKIAIOSFODNN7EXAMPLEKEY1234567890"},
        SandboxPolicy(allow_process_spawn=True, writable_paths=()),
        secret_reference_names=("redevops-nim-key",),
    )
    assert r.secret_reference_names == ("redevops-nim-key",)   # only the name
    assert "AKIA" not in r.stdout_ref                          # value redacted from evidence


def test_process_spawn_denied_is_typed():
    r = LocalSandbox().run({"type": "shell", "command": "echo hi"},
                           SandboxPolicy(allow_process_spawn=False))
    assert r.terminal == Terminal.DENIED and r.failure == Failure.PROCESS_DENIED


def test_backend_is_contract_agnostic():
    assert isinstance(LocalSandbox(), SandboxCapability)


def test_result_digest_is_stable():
    p = SandboxPolicy(allow_process_spawn=True)
    r1 = LocalSandbox().run({"type": "shell", "command": "echo hi"}, p)
    r2 = LocalSandbox().run({"type": "shell", "command": "echo hi"}, p)
    assert r1.digest() == r2.digest()
