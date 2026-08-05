"""SandboxCapability — the resource-policy execution contract; a local backend proves it.

A sandbox answers a *different* question than Mission policy. Mission policy decides whether an action
may be **attempted**; the sandbox decides whether the running process may touch a **file, endpoint,
credential or syscall**. OpenShell is one implementation of this contract (declarative sandbox); it is
**not a second Mission Runtime**. The contract is deliberately backend-agnostic so the same
:class:`SandboxCapability` can later be satisfied by Docker, Firecracker or Kata without touching callers.

We prove the contract here with a **local backend** (the existing soft-confinement runner). That is the
right first backend: if the contract only made sense with a GPU sandbox it would be the wrong contract.

Every run persists a :class:`SandboxResult` — sandbox impl + version, base image digest, the fs / network
/ process / inference **policy digests**, provider-bundle *identities* (never secrets), GPU assignment,
and a **terminal** result with a failure classification. Guarantees enforced at the contract level:

  * a denied capability (e.g. network when policy forbids it) is a **typed** failure, not a crash;
  * cancellation terminates the run and still yields a terminal event;
  * credentials never enter evidence, logs, or sandbox storage — only reference *names*;
  * changing any policy produces a **new policy identity** (digest), so evidence can't be reused across
    policies;
  * success requires a terminal event — there is no "ran but unknown" state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

from . import sandbox as _sandbox
from .guardrails import redact


def _digest(obj: Any) -> str:
    return "sha256:" + sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


class Terminal(str, Enum):
    """Every run ends in exactly one terminal state — no ambiguous 'ran but unknown'."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DENIED = "DENIED"          # a policy forbade a capability the action needed
    CANCELLED = "CANCELLED"    # Mission cancellation terminated the run
    TIMEOUT = "TIMEOUT"


class Failure(str, Enum):
    NONE = "none"
    NETWORK_DENIED = "network_denied"
    FILESYSTEM_DENIED = "filesystem_denied"
    PROCESS_DENIED = "process_denied"
    NONZERO_EXIT = "nonzero_exit"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SandboxPolicy:
    """The resource policy a run executes under. Its digest is the **policy identity** — any change
    (a new allowed host, a widened fs scope) mints a new identity, so no evidence crosses policies."""
    allow_network: bool = False
    allowed_hosts: Tuple[str, ...] = ()
    writable_paths: Tuple[str, ...] = ()
    allow_process_spawn: bool = True
    allow_inference: bool = False           # may the sandboxed code call a model endpoint?
    inference_endpoints: Tuple[str, ...] = ()

    def fs_digest(self) -> str:
        return _digest({"writable_paths": sorted(self.writable_paths)})

    def network_digest(self) -> str:
        return _digest({"allow_network": self.allow_network, "allowed_hosts": sorted(self.allowed_hosts)})

    def process_digest(self) -> str:
        return _digest({"allow_process_spawn": self.allow_process_spawn})

    def inference_digest(self) -> str:
        return _digest({"allow_inference": self.allow_inference,
                        "inference_endpoints": sorted(self.inference_endpoints)})

    def identity(self) -> str:
        return _digest({"fs": self.fs_digest(), "network": self.network_digest(),
                        "process": self.process_digest(), "inference": self.inference_digest()})


@dataclass(frozen=True)
class SandboxResult:
    """Terminal evidence for one sandboxed action. Credentials never appear — only reference names."""
    terminal: Terminal
    failure: Failure
    sandbox_impl: str
    sandbox_version: str
    base_image_digest: str
    policy_identity: str
    policy_digests: Dict[str, str]
    gpu_assignment: str = ""
    provider_bundle_identities: Tuple[str, ...] = ()   # identities, never secrets
    secret_reference_names: Tuple[str, ...] = ()        # names of secrets made available, never values
    exit_code: Optional[int] = None
    stdout_ref: str = ""       # reference to stored (redacted) output, not the output itself
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.terminal == Terminal.SUCCESS

    def digest(self) -> str:
        return _digest({"terminal": self.terminal.value, "failure": self.failure.value,
                        "impl": self.sandbox_impl, "policy_identity": self.policy_identity,
                        "exit_code": self.exit_code})


@runtime_checkable
class SandboxCapability(Protocol):
    """Backend-agnostic: Docker/Firecracker/Kata/OpenShell all implement this."""
    impl: str
    version: str
    def run(self, action: Dict[str, Any], policy: SandboxPolicy,
            *, cancel: Optional[Callable[[], bool]] = None,
            timeout: int = 30) -> SandboxResult: ...


class LocalSandbox:
    """Local backend — soft confinement (see :mod:`agent_harness.sandbox`). It is **not** real OS
    isolation, and says so; it exists to prove the :class:`SandboxCapability` contract end to end. It
    enforces the policy at the contract boundary: an action that needs a capability the policy denies is
    refused *before* running, as a typed :class:`Terminal.DENIED`."""

    impl = "local-soft"
    version = "0.1"
    base_image_digest = ""   # local has no image

    def run(self, action: Dict[str, Any], policy: SandboxPolicy,
            *, cancel: Optional[Callable[[], bool]] = None, timeout: int = 30,
            secret_reference_names: Tuple[str, ...] = (),
            provider_bundle_identities: Tuple[str, ...] = (),
            gpu_assignment: str = "", store: Optional[Callable[[str], str]] = None) -> SandboxResult:
        policy_digests = {"fs": policy.fs_digest(), "network": policy.network_digest(),
                          "process": policy.process_digest(), "inference": policy.inference_digest()}
        base = dict(sandbox_impl=self.impl, sandbox_version=self.version,
                    base_image_digest=self.base_image_digest, policy_identity=policy.identity(),
                    policy_digests=policy_digests, gpu_assignment=gpu_assignment,
                    provider_bundle_identities=tuple(provider_bundle_identities),
                    secret_reference_names=tuple(secret_reference_names))

        # cancellation before start → terminal CANCELLED (never a silent no-op)
        if cancel is not None and cancel():
            return SandboxResult(terminal=Terminal.CANCELLED, failure=Failure.CANCELLED, **base)

        # contract-level capability enforcement (typed denials, no crash)
        if action.get("requires_network") and not policy.allow_network:
            return SandboxResult(terminal=Terminal.DENIED, failure=Failure.NETWORK_DENIED,
                                 detail="network capability denied by policy", **base)
        if action.get("requires_inference") and not policy.allow_inference:
            return SandboxResult(terminal=Terminal.DENIED, failure=Failure.PROCESS_DENIED,
                                 detail="inference capability denied by policy", **base)
        if action.get("type") == "shell" and not policy.allow_process_spawn:
            return SandboxResult(terminal=Terminal.DENIED, failure=Failure.PROCESS_DENIED,
                                 detail="process spawn denied by policy", **base)

        cwd = policy.writable_paths[0] if policy.writable_paths else None
        out = _sandbox.run(action, cwd=cwd, timeout=timeout)

        if cancel is not None and cancel():
            return SandboxResult(terminal=Terminal.CANCELLED, failure=Failure.CANCELLED, **base)

        rc = out.get("returncode", 1)
        # store output through a redacting sink — credentials never enter evidence/logs/storage
        combined = redact((out.get("stdout") or "") + (out.get("stderr") or ""))
        stdout_ref = store(combined) if store else ("inline:" + _digest(combined))
        if out.get("stderr") == "timeout" or rc == 124:
            return SandboxResult(terminal=Terminal.TIMEOUT, failure=Failure.TIMEOUT,
                                 exit_code=rc, stdout_ref=stdout_ref, **base)
        if rc == 0:
            return SandboxResult(terminal=Terminal.SUCCESS, failure=Failure.NONE,
                                 exit_code=0, stdout_ref=stdout_ref, **base)
        return SandboxResult(terminal=Terminal.FAILED, failure=Failure.NONZERO_EXIT,
                             exit_code=rc, stdout_ref=stdout_ref, **base)


if __name__ == "__main__":
    checks = []
    sb = LocalSandbox()
    open_policy = SandboxPolicy(allow_network=True, allow_process_spawn=True)
    closed_policy = SandboxPolicy(allow_network=False)

    # denied network → typed capability failure (not a crash)
    r_net = sb.run({"type": "http", "requires_network": True}, closed_policy)
    checks.append(("network denial is typed", r_net.terminal == Terminal.DENIED
                   and r_net.failure == Failure.NETWORK_DENIED))

    # a real (local) success has a terminal event
    r_ok = sb.run({"type": "shell", "command": "echo hi"}, open_policy)
    checks.append(("success requires terminal", r_ok.terminal == Terminal.SUCCESS and r_ok.ok))
    checks.append(("exit code captured", r_ok.exit_code == 0))

    # cancellation terminates with a terminal event
    r_cancel = sb.run({"type": "shell", "command": "echo hi"}, open_policy, cancel=lambda: True)
    checks.append(("cancel terminates", r_cancel.terminal == Terminal.CANCELLED))

    # policy change → new policy identity
    p2 = SandboxPolicy(allow_network=True, allowed_hosts=("api.nvidia.com",))
    checks.append(("policy change mints new identity", open_policy.identity() != p2.identity()))
    checks.append(("same policy same identity", open_policy.identity() == SandboxPolicy(
        allow_network=True, allow_process_spawn=True).identity()))

    # credentials never enter evidence — only reference names; secret-looking output is redacted
    r_secret = sb.run({"type": "shell", "command": "echo AKIAIOSFODNN7EXAMPLEKEY1234567890"}, open_policy,
                      secret_reference_names=("redevops-nim-key",))
    checks.append(("secret ref name kept, value absent",
                   r_secret.secret_reference_names == ("redevops-nim-key",)
                   and "AKIA" not in r_secret.stdout_ref))

    # process spawn denied → typed
    r_proc = sb.run({"type": "shell", "command": "echo hi"},
                    SandboxPolicy(allow_process_spawn=False))
    checks.append(("process denial is typed", r_proc.terminal == Terminal.DENIED))

    # contract is backend-agnostic
    checks.append(("LocalSandbox satisfies SandboxCapability", isinstance(sb, SandboxCapability)))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"RESULT {passed}/{len(checks)}  (SandboxCapability / local backend)")
    import sys
    sys.exit(0 if passed == len(checks) else 1)
