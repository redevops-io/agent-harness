"""A long-lived Docker container with NO network, the workspace mounted at /work, resource limits and a
non-root user. The agent's bash tool runs inside it, so generated code cannot reach the network (and so cannot
fetch benchmark answers) or the host filesystem beyond the mounted workspace."""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Optional

IMAGE = "agent-harness-office:1"


class DockerSandbox:
    def __init__(self, workspace: Path, image: str = IMAGE, memory: str = "4g", cpus: str = "2", pids: int = 512,
                 docker: str = "docker"):
        self.workspace, self.image, self.docker = Path(workspace).resolve(), image, docker
        self.memory, self.cpus, self.pids = memory, cpus, pids
        self.name = "ah-sbx-" + uuid.uuid4().hex[:10]

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        subprocess.run([self.docker, "run", "-d", "--rm", "--name", self.name, "--network", "none",
                        "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", str(self.pids),
                        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                        "-v", f"{self.workspace}:/work", "-w", "/work", self.image, "sleep", "infinity"],
                       check=True, capture_output=True, text=True, timeout=120)

    def stop(self) -> None:
        subprocess.run([self.docker, "rm", "-f", self.name], capture_output=True, timeout=60)

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
