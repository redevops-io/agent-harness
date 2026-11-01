"""Process-tree control: children are small stdlib-only Python scripts, so the same tests run on Linux, macOS and
Windows. Each tree test starts a child that starts a grandchild which would outlive it (both keep the output pipes
open) and then checks that both processes are gone."""
import os
import subprocess
import sys
import threading
import time

import pytest

from agent_harness.process_runner import CancelToken, ProcessRunner, run_process

PY = sys.executable
IS_WINDOWS = sys.platform == "win32"

# Parent: starts a grandchild that sleeps for five minutes, prints both pids, then sleeps (or exits at once).
TREE = r"""
import os, subprocess, sys, time
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
print("GC", gc.pid, flush=True)
print("PARENT", os.getpid(), flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "exit":
    sys.exit(7)
time.sleep(300)
"""


# Same tree, but the pids go to a file (a tool's timeout result carries no output).
TREE_TO_FILE = TREE.replace('print("GC", gc.pid, flush=True)\nprint("PARENT", os.getpid(), flush=True)',
                            'open(sys.argv[1], "w").write("PARENT %d\\nGC %d\\n" % (os.getpid(), gc.pid))')


def pid_alive(pid: int) -> bool:
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False                                  # no such process (or access denied: not ours)
        try:
            code = wintypes.DWORD()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie (dead, not yet reaped by an init that does not reap) is not running
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(state) and not state.startswith("Z")


def wait_gone(pid: int, seconds: float = 10.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


def pids(stdout: str):
    found = dict(line.split() for line in stdout.splitlines() if line.startswith(("GC ", "PARENT ")))
    return int(found["PARENT"]), int(found["GC"])


@pytest.fixture
def guard():
    """Kill anything a failing test left behind, so one failure does not leak processes into the CI host."""
    leftovers = []
    yield leftovers
    for pid in leftovers:
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, 9)
        except OSError:
            pass


def test_a_normal_run_returns_output_and_the_exit_status():
    r = run_process([PY, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"], timeout=30)
    assert (r.stdout.strip(), r.stderr.strip(), r.returncode) == ("out", "err", 3)
    assert not r.timed_out and not r.cancelled and r.tree_gone and r.drain_complete and not r.tree_stopped


def test_cwd_and_env_are_what_the_caller_gave(tmp_path):
    env = {**os.environ, "RUNNER_PROBE": "yes"}
    r = run_process([PY, "-c", "import os; print(os.getcwd()); print(os.environ.get('RUNNER_PROBE'))"],
                    cwd=str(tmp_path), env=env, timeout=30)
    cwd, probe = r.stdout.split()
    assert os.path.samefile(cwd, tmp_path) and probe == "yes"


def test_child_stdin_is_closed_not_inherited():
    r = run_process([PY, "-c", "import sys; print(repr(sys.stdin.read()))"], timeout=30)
    assert r.stdout.strip() == "''"


def test_undecodable_output_is_replaced_not_raised():
    r = run_process([PY, "-c", "import sys; sys.stdout.buffer.write(b'a\\xffb')"], timeout=30)
    assert r.stdout == "a�b" and r.returncode == 0


def test_a_missing_executable_is_a_result_not_an_exception():
    r = run_process(["this-executable-does-not-exist-9f3"], timeout=5)
    assert r.returncode is None and r.error and not r.timed_out


def test_output_is_limited_and_the_excess_counted():
    code = "import sys; sys.stdout.buffer.write(b'x' * 200000); sys.stderr.buffer.write(b'y' * 5000)"
    r = run_process([PY, "-c", code], timeout=30, max_output=1000)
    assert len(r.stdout) == 1000 and r.stdout_truncated == 199000
    assert len(r.stderr) == 1000 and r.stderr_truncated == 4000
    assert r.returncode == 0 and r.drain_complete


def test_timeout_stops_the_child_and_its_grandchild(guard):
    t0 = time.monotonic()
    r = ProcessRunner(grace=1.0).run([PY, "-c", TREE], timeout=5.0)
    parent, grandchild = pids(r.stdout)
    guard += [parent, grandchild]
    assert r.timed_out and not r.cancelled and r.tree_stopped and r.tree_gone and r.drain_complete
    assert time.monotonic() - t0 < 20                  # the deadline was enforced during the run, with no hang
    assert wait_gone(parent) and wait_gone(grandchild)


def test_cancel_during_a_run_stops_the_whole_tree(guard):
    runner = ProcessRunner(grace=1.0)
    threading.Timer(4.0, runner.cancel).start()
    t0 = time.monotonic()
    r = runner.run([PY, "-c", TREE], timeout=120)
    parent, grandchild = pids(r.stdout)
    guard += [parent, grandchild]
    assert r.cancelled and not r.timed_out and r.tree_gone and r.drain_complete
    assert time.monotonic() - t0 < 30
    assert wait_gone(parent) and wait_gone(grandchild)


def test_an_external_cancel_callable_is_observed_during_a_run(guard):
    token = CancelToken()
    threading.Timer(4.0, token.cancel).start()
    r = run_process([PY, "-c", TREE], timeout=120, cancel=token)
    parent, grandchild = pids(r.stdout)
    guard += [parent, grandchild]
    assert r.cancelled and wait_gone(parent) and wait_gone(grandchild)


def test_cancel_between_runs_means_the_next_run_does_not_start(tmp_path):
    runner = ProcessRunner()
    marker = tmp_path / "started"
    code = f"open({str(marker)!r}, 'w').close()"
    assert runner.run([PY, "-c", code], timeout=30).returncode == 0 and marker.exists()
    marker.unlink()
    runner.cancel()
    r = runner.run([PY, "-c", code], timeout=30)
    assert r.cancelled and r.returncode is None and not marker.exists()
    runner.reset()
    assert runner.run([PY, "-c", code], timeout=30).returncode == 0 and marker.exists()


def test_descendants_left_after_a_normal_exit_are_stopped_and_do_not_block_the_call(guard):
    t0 = time.monotonic()
    r = ProcessRunner(grace=1.0).run([PY, "-c", TREE, "exit"], timeout=60)
    parent, grandchild = pids(r.stdout)
    guard += [parent, grandchild]
    assert r.returncode == 7 and not r.timed_out
    assert r.tree_stopped and r.tree_gone and r.drain_complete        # a leftover grandchild is reported, then gone
    assert time.monotonic() - t0 < 30                  # the grandchild holds the pipes: the call must not wait for it
    assert wait_gone(grandchild)


@pytest.mark.skipif(IS_WINDOWS, reason="Windows has no SIGTERM: the job is terminated at once, so there is no "
                                      "grace period to test (the Windows tree tests above cover the kill)")
def test_a_child_that_ignores_sigterm_is_killed_after_the_grace_period(guard):
    code = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); "
            "time.sleep(300)")
    t0 = time.monotonic()
    r = ProcessRunner(grace=0.5).run([PY, "-c", code], timeout=2.5)
    assert r.timed_out and r.tree_gone and r.returncode in (-9, -15)
    assert time.monotonic() - t0 < 15


@pytest.mark.skipif(IS_WINDOWS, reason="a Windows descendant cannot leave the job (breakaway is not enabled), so this "
                                      "POSIX-only case (a descendant that starts its own session) does not exist there")
def test_a_descendant_that_left_the_group_cannot_block_the_call(guard):
    code = (
        "import subprocess, sys\n"
        "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "print('GC', gc.pid, flush=True)\n"
    )
    t0 = time.monotonic()
    r = ProcessRunner(drain=1.0).run([PY, "-c", code], timeout=30)
    gc = int(r.stdout.split()[1])
    guard.append(gc)                                   # the runner cannot reach it; the test cleans it up
    assert time.monotonic() - t0 < 15                  # bounded drain: no wait for the 60 s sleeper
    assert r.returncode == 0 and r.drain_complete is False
    assert pid_alive(gc)                               # honest: it escaped, and the result says the drain was incomplete


def test_no_child_of_this_process_is_left_after_a_series_of_runs(guard):
    for _ in range(3):
        r = ProcessRunner(grace=0.5).run([PY, "-c", TREE], timeout=3.0)
        guard += list(pids(r.stdout))
    assert all(wait_gone(p) for p in guard)


# ---------------------------------------------------------------------------------------------------------------
# The agent loop: max_seconds and cancel apply during a tool call that runs a process
# ---------------------------------------------------------------------------------------------------------------

def _loop_registry(argv, **sandbox_kw):
    from agent_harness import sandbox
    from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool

    def run(_args):
        out = sandbox.run_shell("", argv=argv, timeout=120, **sandbox_kw)
        return ToolResult(text=f"rc={out['returncode']} {out['stderr']}\n{out['stdout']}")
    reg = ToolRegistry(ApprovalPolicy("allowlist", allow=["slow"]))
    reg.register(function_tool("slow", run, side_effecting=True))
    return reg


def _script(*replies):
    from tests.test_loop import Script
    return Script(list(replies))


def _slow_call():
    import json
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "slow", "arguments": json.dumps({})}}]}


def test_max_seconds_is_enforced_during_a_tool_call(guard, tmp_path):
    from agent_harness.loop import run_loop
    pidfile = tmp_path / "pids.txt"
    t0 = time.monotonic()
    res = run_loop(_script(_slow_call(), {"role": "assistant", "content": "unreached"}),
                   _loop_registry([PY, "-c", TREE_TO_FILE, str(pidfile)]), "sys", "task", max_seconds=5, supervision=None)
    parent, grandchild = pids(pidfile.read_text())
    guard += [parent, grandchild]
    assert wait_gone(parent) and wait_gone(grandchild)
    assert res.stop == "max_seconds"
    assert "rc=124 timeout" in res.messages[-1]["content"]
    assert time.monotonic() - t0 < 30                  # not the tool's own 120 s


def test_cancel_stops_a_running_tool_call_and_ends_the_loop(guard, tmp_path):
    from agent_harness.loop import run_loop
    pidfile = tmp_path / "pids.txt"
    flag = CancelToken()
    threading.Timer(4.0, flag.cancel).start()
    t0 = time.monotonic()
    res = run_loop(_script(_slow_call(), {"role": "assistant", "content": "unreached"}),
                   _loop_registry([PY, "-c", TREE_TO_FILE, str(pidfile)]), "sys", "task", cancel=flag, supervision=None)
    parent, grandchild = pids(pidfile.read_text())
    guard += [parent, grandchild]
    assert wait_gone(parent) and wait_gone(grandchild)
    assert res.stop == "cancelled" and "rc=130 cancelled" in res.messages[-1]["content"]
    assert time.monotonic() - t0 < 30


def test_a_loop_cancelled_before_it_starts_runs_nothing():
    from agent_harness.loop import run_loop
    flag = CancelToken()
    flag.cancel()
    res = run_loop(_script({"role": "assistant", "content": "unreached"}), _loop_registry([PY, "-c", "pass"]),
                   "sys", "task", cancel=flag, supervision=None)
    assert res.stop == "cancelled" and res.turns == 0


# ---------------------------------------------------------------------------------------------------------------
# run_subprocess: the drop-in the judge and benchmark runners use
# ---------------------------------------------------------------------------------------------------------------

def test_run_subprocess_matches_subprocess_run_for_a_normal_child():
    from agent_harness.process_runner import run_subprocess
    p = run_subprocess([PY, "-c", "import sys; print('o'); print('e', file=sys.stderr); sys.exit(2)"],
                       capture_output=True, text=True, timeout=30)
    assert (p.returncode, p.stdout.strip(), p.stderr.strip()) == (2, "o", "e")
    b = run_subprocess([PY, "-c", "print('x')"], capture_output=True, timeout=30)
    assert b.stdout.strip() == b"x"


def test_run_subprocess_raises_timeout_expired_and_leaves_no_process(guard, tmp_path):
    from agent_harness.process_runner import run_subprocess
    pidfile = tmp_path / "pids.txt"
    with pytest.raises(subprocess.TimeoutExpired):
        run_subprocess([PY, "-c", TREE_TO_FILE, str(pidfile)], capture_output=True, text=True, timeout=5)
    parent, grandchild = pids(pidfile.read_text())
    guard += [parent, grandchild]
    assert wait_gone(parent) and wait_gone(grandchild)


def test_run_subprocess_writes_to_a_file_and_can_merge_stderr(tmp_path):
    from agent_harness.process_runner import run_subprocess
    log = tmp_path / "log.bin"
    with open(log, "wb") as f:
        p = run_subprocess([PY, "-c", "import sys; print('out'); sys.stdout.flush(); print('err', file=sys.stderr)"],
                           stdout=f, stderr=subprocess.STDOUT, timeout=30)
    assert p.returncode == 0 and p.stdout is None
    assert sorted(log.read_text().split()) == ["err", "out"]


def test_run_subprocess_raises_oserror_for_a_missing_executable():
    from agent_harness.process_runner import run_subprocess
    with pytest.raises(OSError):
        run_subprocess(["this-executable-does-not-exist-9f3"], capture_output=True, timeout=5)
