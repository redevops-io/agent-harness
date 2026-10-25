"""A long-lived Docker container with NO network, the workspace mounted at /work, resource limits and a
non-root user. The agent's bash tool runs inside it, so generated code cannot reach the network (and so cannot
fetch benchmark answers) or the host filesystem beyond the mounted workspace."""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IMAGE = "agent-harness-office:1"


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    seconds: float = 0.0


class EgressGateway:
    """A filtering proxy in its own container, OUTSIDE the agent's container, on a Docker ``--internal`` network (no route to the outside).

    The sandbox container joins only that internal network, so it can reach nothing but the proxy: no direct sockets to the internet, to the host or to the
    LAN, and no DNS beyond Docker's embedded resolver (which knows only the proxy's name), so the policy cannot be bypassed by ignoring proxy variables or by
    resolving names itself. The proxy additionally joins the default bridge network for its upstream traffic and runs ``agent_harness.egress_proxy --strict``
    (public addresses only, port list, redirects re-checked by the client's next request, per-tunnel byte cap). Only that module and the policy module are
    mounted, read-only; the container has no host mounts, no secrets and no capabilities."""

    PORT = 8766

    def __init__(self, image: str = IMAGE, docker: str = "docker", extra_deny: tuple = ()):
        self.image, self.docker, self.extra_deny = image, docker, tuple(extra_deny)
        tag = uuid.uuid4().hex[:10]
        self.network, self.name = f"ah-net-{tag}", f"ah-egress-{tag}"
        self.stage: Optional[Path] = None

    @property
    def proxy_url(self) -> str:
        return f"http://egress:{self.PORT}"

    def _d(self, *args, check=True, timeout=120):
        return subprocess.run([self.docker, *args], check=check, capture_output=True, text=True, timeout=timeout)

    def start(self) -> None:
        import shutil
        import tempfile
        src = Path(__file__).resolve().parent.parent
        self.stage = Path(tempfile.mkdtemp(prefix="ah-egress-"))
        pkg = self.stage / "agent_harness"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        for f in ("egress_proxy.py", "netpolicy.py"):
            shutil.copy(src / f, pkg / f)
        for p in [self.stage, pkg, *pkg.iterdir()]:
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        self._d("network", "create", "--internal", self.network)
        deny = [x for d in self.extra_deny for x in ("--deny", d)]
        self._d("run", "-d", "--rm", "--name", self.name, "--network", self.network, "--network-alias", "egress",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp", "--memory", "256m", "--pids-limit", "128",
                "-v", f"{self.stage}:/opt/egress:ro", "-e", "PYTHONPATH=/opt/egress", self.image,
                "python3", "-m", "agent_harness.egress_proxy", "--host", "0.0.0.0", "--port", str(self.PORT), "--strict", *deny)
        self._d("network", "connect", "bridge", self.name)
        for _ in range(40):                     # wait until the proxy accepts connections
            r = self._d("exec", self.name, "python3", "-c",
                        f"import socket;socket.create_connection(('127.0.0.1',{self.PORT}),2).close()", check=False, timeout=20)
            if r.returncode == 0:
                return
            time.sleep(0.25)
        raise RuntimeError("egress gateway did not start: " + (self._d("logs", self.name, check=False).stdout or "")[-300:])

    def stop(self) -> None:
        import shutil
        subprocess.run([self.docker, "rm", "-f", self.name], capture_output=True, timeout=60)
        subprocess.run([self.docker, "network", "rm", self.network], capture_output=True, timeout=60)
        if self.stage:
            shutil.rmtree(self.stage, ignore_errors=True)


class DockerSandbox:
    """``network="none"`` (default, the legacy configuration) has no network at all. ``network="egress"`` joins the sandbox to an EgressGateway's internal network
    and exports the proxy settings to every command; nothing else is reachable."""

    def __init__(self, workspace: Path, image: str = IMAGE, memory: str = "4g", cpus: str = "2", pids: int = 512,
                 docker: str = "docker", network: str = "none", egress_deny: tuple = ()):
        if network not in ("none", "egress"):
            raise ValueError(f"network must be 'none' or 'egress', not {network!r}")
        self.workspace, self.image, self.docker = Path(workspace).resolve(), image, docker
        self.memory, self.cpus, self.pids = memory, cpus, pids
        self.network = network
        self.gateway = EgressGateway(image, docker, egress_deny) if network == "egress" else None
        self.name = "ah-sbx-" + uuid.uuid4().hex[:10]

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        net = ["--network", "none"]
        env: list = []
        if self.gateway:
            self.gateway.start()
            net = ["--network", self.gateway.network]
            url = self.gateway.proxy_url
            env = [x for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy") for x in ("-e", f"{k}={url}")]
        try:
            subprocess.run([self.docker, "run", "-d", "--rm", "--name", self.name, *net, *env,
                            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", str(self.pids),
                            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                            "-v", f"{self.workspace}:/work", "-w", "/work", self.image, "sleep", "infinity"],
                           check=True, capture_output=True, text=True, timeout=120)
        except Exception:
            if self.gateway:
                self.gateway.stop()
            raise

    def stop(self) -> None:
        subprocess.run([self.docker, "rm", "-f", self.name], capture_output=True, timeout=60)
        if self.gateway:
            self.gateway.stop()

    def run_raw(self, command: str, timeout: int = 120) -> RunResult:
        """Structured result with the real exit status. The command runs in its own process group under ``timeout``: TERM at the limit, KILL 5 s later, and the
        signal goes to the whole group, so children do not outlive the call. Exit 124 (TERM) or 137 after the limit means timed out."""
        t0 = time.time()
        try:
            p = subprocess.run([self.docker, "exec", "-w", "/work", self.name, "timeout", "--kill-after=5", "--signal=TERM", str(timeout),
                                "bash", "-lc", command], capture_output=True, text=True, timeout=timeout + 40)
        except subprocess.TimeoutExpired:
            subprocess.run([self.docker, "exec", self.name, "pkill", "-KILL", "-f", "timeout --kill-after"], capture_output=True, timeout=30)
            return RunResult("", "", 124, True, time.time() - t0)
        took = time.time() - t0
        return RunResult(p.stdout or "", p.stderr or "", p.returncode, p.returncode == 124 or (p.returncode == 137 and took >= timeout), took)

    def run(self, command: str, timeout: int = 120, max_chars: int = 20000) -> str:
        try:
            p = subprocess.run([self.docker, "exec", "-w", "/work", self.name, "timeout", "--signal=KILL", str(timeout),
                                "bash", "-lc", command], capture_output=True, text=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return f"[command timed out after {timeout}s]"
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
        if p.returncode == 137:
            out += f"\n[killed: exceeded {timeout}s]"
        elif p.returncode:
            out += f"\n[exit code {p.returncode}]"
        if len(out) > max_chars:
            out = out[: max_chars * 2 // 3] + f"\n[... {len(out) - max_chars} chars omitted ...]\n" + out[-max_chars // 3:]
        return out or "[no output]"
