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


# ---------------------------------------------------------------------------------------------------------------
# Review fixes: many open fds, parent-death signals, timeout=None under a deadline, a cancelled run_subprocess
# ---------------------------------------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def many_open_fds():
    """Hold more than 1100 descriptors open so that the next pipes get numbers above FD_SETSIZE (1024)."""
    held = []
    try:
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            want = 1500 if hard == resource.RLIM_INFINITY else min(hard, 1500)
            if soft != resource.RLIM_INFINITY and soft < want:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except ImportError:
            pass                                           # Windows: the C runtime's own limit applies
        while len(held) < 1100:
            try:
                held.append(os.open(os.devnull, os.O_RDONLY))
            except OSError:
                break
        if not IS_WINDOWS and len(held) < 1100:
            pytest.skip("cannot raise the descriptor limit above 1100 here")
        yield len(held)
    finally:
        for fd in held:
            os.close(fd)


def test_output_is_read_correctly_when_the_pipe_descriptors_are_above_1024(many_open_fds):
    r = run_process([PY, "-c", "import sys; sys.stdout.buffer.write(b'hello\\n')"], timeout=30)
    assert (r.stdout, r.returncode, r.drain_complete) == ("hello\n", 0, True)
    big = run_process([PY, "-c", "import sys; sys.stdout.buffer.write(b'x' * 200000)"], timeout=30)
    assert len(big.stdout) == 200000 and big.returncode == 0 and not big.timed_out


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX only: the reader on Windows reads through the stream, not a selector")
def test_a_reader_error_is_surfaced_not_turned_into_a_clean_end_of_file(monkeypatch):
    import selectors
    from agent_harness import process_runner as pr

    class Broken(selectors.DefaultSelector):
        def select(self, timeout=None):
            raise OSError(5, "simulated read failure")
    monkeypatch.setattr(pr.selectors, "DefaultSelector", Broken)
    r = run_process([PY, "-c", "print('hello')"], timeout=10)
    assert r.read_error and "simulated read failure" in r.read_error and r.drain_complete is False


PARENT_SCRIPT = r'''
import signal, sys
# a test runner started as a background job may have these ignored; the tests need the normal dispositions
signal.signal(signal.SIGINT, signal.default_int_handler)
for _name in ("SIGHUP", "SIGTERM"):
    if hasattr(signal, _name):
        signal.signal(getattr(signal, _name), signal.SIG_DFL)
sys.path.insert(0, sys.argv[2])
from agent_harness import sandbox
child = ("import os, signal, time\n"
         "if %r: signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
         "open(%r, 'w').write(str(os.getpid()))\n"
         "time.sleep(300)\n") % (sys.argv[3] == "ignore", sys.argv[1])
try:
    sandbox.run_shell("", argv=[sys.executable, "-c", child], timeout=300)
except KeyboardInterrupt:
    print("saw KeyboardInterrupt", flush=True)
'''


def _start_parent(tmp_path, mode="plain"):
    pidfile = tmp_path / "child.pid"
    env = {**os.environ, "PYTHONPATH": ROOT}
    parent = subprocess.Popen([PY, "-c", PARENT_SCRIPT, str(pidfile), ROOT, mode], env=env, stdout=subprocess.PIPE,
                              start_new_session=not IS_WINDOWS)
    end = time.monotonic() + 30
    while time.monotonic() < end and not (pidfile.exists() and pidfile.read_text().strip()):
        time.sleep(0.05)
    return parent, int(pidfile.read_text())


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX only: SIGHUP/SIGTERM sent to the parent's process group; the Windows "
                                      "equivalent (the parent is killed) is the next test")
@pytest.mark.parametrize("sig", ["SIGHUP", "SIGTERM"])
def test_a_group_signal_to_the_parent_does_not_leave_the_sandbox_child_running(tmp_path, guard, sig):
    import signal
    parent, child = _start_parent(tmp_path)
    guard.append(child)
    os.killpg(parent.pid, getattr(signal, sig))            # what a closing terminal or `kill -- -pgid` does
    parent.wait(timeout=30)
    assert wait_gone(child), f"{sig}: the sandbox child outlived its parent"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows only: the job object kills its processes when the parent dies")
def test_killing_the_parent_kills_the_job(tmp_path, guard):
    parent, child = _start_parent(tmp_path)
    guard.append(child)
    parent.kill()
    parent.wait(timeout=30)
    assert wait_gone(child), "the child outlived its parent (KILL_ON_JOB_CLOSE did not take effect)"


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX only: SIGINT delivered twice to the parent's process group")
def test_a_second_interrupt_during_the_grace_period_still_kills_the_child(tmp_path, guard):
    import signal
    parent, child = _start_parent(tmp_path, "ignore")       # the child ignores SIGTERM, so the grace period lasts
    guard.append(child)
    os.killpg(parent.pid, signal.SIGINT)
    time.sleep(0.5)
    os.killpg(parent.pid, signal.SIGINT)
    parent.wait(timeout=30)
    assert wait_gone(child), "a second Ctrl-C left the sandbox child running"


def test_timeout_none_under_a_run_control_deadline_is_a_deadline_not_a_crash():
    from agent_harness import sandbox
    from agent_harness.process_runner import run_control
    with run_control(max_seconds=60):
        out = sandbox.run_shell("", argv=[PY, "-c", "print('ok')"], timeout=None)
    assert out["returncode"] == 0 and out["stdout"].strip() == "ok", out
    with run_control(max_seconds=1.5):
        out = sandbox.run_shell("", argv=[PY, "-c", "import time; time.sleep(60)"], timeout=None)
    assert out["returncode"] == 124 and out["stderr"] == "timeout"


def test_a_cancelled_run_subprocess_is_not_a_normal_exit(guard, tmp_path):
    from agent_harness.process_runner import RunCancelled, run_subprocess
    pidfile = tmp_path / "pids.txt"
    token = CancelToken()
    threading.Timer(4.0, token.cancel).start()
    with pytest.raises(RunCancelled) as e:
        with __import__("agent_harness.process_runner", fromlist=["run_control"]).run_control(cancel=token):
            run_subprocess([PY, "-c", TREE_TO_FILE, str(pidfile)], capture_output=True, text=True, timeout=120)
    assert isinstance(e.value, subprocess.SubprocessError)    # the judge launchers already catch this family
    parent, grandchild = pids(pidfile.read_text())
    guard += [parent, grandchild]
    assert wait_gone(parent) and wait_gone(grandchild)


def test_sandbox_output_is_utf8_with_replacement_and_crlf_is_preserved():
    from agent_harness import sandbox
    out = sandbox.run_shell("", argv=[PY, "-c", "import sys; sys.stdout.buffer.write(b'caf\\xc3\\xa9 \\xff\\r\\nx')"], timeout=30)
    assert out["stdout"] == "café �\r\nx"


def test_the_loop_ends_as_max_seconds_when_a_tool_hit_the_deadline_even_if_the_wall_clock_lags(monkeypatch):
    """The tool's deadline is monotonic; the loop must not depend on time.time() having advanced as far (its resolution is
    coarse on Windows), or it would ask the model once more after the tool was stopped."""
    import types
    from agent_harness import loop
    from agent_harness.loop import run_loop
    monkeypatch.setattr(loop, "time", types.SimpleNamespace(time=lambda: 1000.0, monotonic=time.monotonic))
    res = run_loop(_script(_slow_call(), {"role": "assistant", "content": "unreached"}),
                   _loop_registry([PY, "-c", "import time; time.sleep(60)"]), "sys", "task", max_seconds=1.5, supervision=None)
    assert res.stop == "max_seconds" and "rc=124 timeout" in res.messages[-1]["content"]


# ---------------------------------------------------------------------------------------------------------------
# Retest fixes: a reader that fails while the child runs, fork inheritance, the documented signal behaviour
# ---------------------------------------------------------------------------------------------------------------

FLOOD = "import sys; sys.stdout.buffer.write(b'x' * 3000000); sys.stdout.flush()"


@pytest.fixture
def failing_reader(monkeypatch):
    """The output reader of the next run fails at once (what an EIO or EMFILE from the pipe would do), while the child
    is still running and is about to block on a full pipe that nobody drains."""
    from agent_harness import process_runner as pr

    def run(self):
        self.error = "OSError: [Errno 5] injected read failure"
    monkeypatch.setattr(pr._Reader, "run", run)


def _within(seconds, fn):
    """Run ``fn`` in a thread so that a hang fails the test instead of hanging the suite."""
    box = {}
    t = threading.Thread(target=lambda: box.update(result=fn()), daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"still running after {seconds} s (a hang)"
    return box["result"]


def test_a_reader_failure_while_the_child_runs_stops_the_tree_and_is_reported_first(failing_reader):
    t0 = time.monotonic()
    r = _within(40, lambda: ProcessRunner(grace=0.5).run([PY, "-c", FLOOD], timeout=20))
    assert "injected read failure" in r.read_error and not r.timed_out and r.drain_complete is False
    assert time.monotonic() - t0 < 15                      # not the 20 s timeout


def test_a_reader_failure_without_a_timeout_does_not_hang_run_subprocess(failing_reader):
    from agent_harness.process_runner import run_subprocess

    def call():
        try:
            run_subprocess([PY, "-c", FLOOD], capture_output=True, text=True)
        except OSError as e:
            return str(e)
    msg = _within(40, call)
    assert msg and "injected read failure" in msg


def test_run_shell_reports_a_reader_failure_ahead_of_a_timeout(failing_reader):
    from agent_harness import sandbox
    out = _within(40, lambda: sandbox.run_shell("", argv=[PY, "-c", FLOOD], timeout=3))
    assert out["returncode"] not in (0, 124) and "output read error" in out["stderr"] and "injected read failure" in out["stderr"]


FORK_SCRIPT = r"""
import os, signal, sys, threading, time
signal.signal(signal.SIGTERM, signal.SIG_DFL)
sys.path.insert(0, sys.argv[2])
from agent_harness import sandbox
sandbox.run_shell("", argv=[sys.executable, "-c", "pass"], timeout=30)       # first use installs the handler (main thread)
child = "import os, time; open(%r, 'w').write(str(os.getpid())); time.sleep(300)" % sys.argv[1]
t = threading.Thread(target=lambda: sandbox.run_shell("", argv=[sys.executable, "-c", child], timeout=300), daemon=True)
t.start()
while not (os.path.exists(sys.argv[1]) and open(sys.argv[1]).read().strip()):
    time.sleep(0.05)
cpid = int(open(sys.argv[1]).read())
worker = os.fork()
if worker == 0:
    time.sleep(120)
    os._exit(0)
time.sleep(0.5)
os.kill(worker, signal.SIGTERM)                        # what Process.terminate() does to a forked worker
os.waitpid(worker, 0)
time.sleep(1.0)
try:
    os.kill(cpid, 0)
    print("PARENT COMMAND ALIVE", flush=True)
except ProcessLookupError:
    print("PARENT COMMAND KILLED", flush=True)
try:
    os.kill(cpid, 9)
except ProcessLookupError:
    pass
"""


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX only: os.fork and signal handlers")
def test_a_forked_worker_does_not_kill_the_parents_running_command(tmp_path, guard):
    pidfile = tmp_path / "child.pid"
    p = subprocess.run([PY, "-c", FORK_SCRIPT, str(pidfile), ROOT], env={**os.environ, "PYTHONPATH": ROOT},
                       capture_output=True, text=True, timeout=120)
    if pidfile.exists() and pidfile.read_text().strip():
        guard.append(int(pidfile.read_text()))
    assert "PARENT COMMAND ALIVE" in p.stdout, p.stdout + p.stderr


APP_HANDLER_SCRIPT = r"""
import os, signal, sys
sys.path.insert(0, sys.argv[2])
def app_handler(signum, frame):
    print("app handler ran", flush=True)
signal.signal(signal.SIGTERM, app_handler)             # installed before the first use: the library chains to it
from agent_harness import sandbox
child = "import os, time; open(%r, 'w').write(str(os.getpid())); time.sleep(300)" % sys.argv[1]
out = sandbox.run_shell("", argv=[sys.executable, "-c", child], timeout=300)
print("rc", out["returncode"], flush=True)
"""


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX only: SIGTERM handlers")
def test_the_documented_signal_behaviour_live_groups_are_killed_first_then_the_app_handler_runs(tmp_path, guard):
    import signal
    pidfile = tmp_path / "child.pid"
    p = subprocess.Popen([PY, "-c", APP_HANDLER_SCRIPT, str(pidfile), ROOT], env={**os.environ, "PYTHONPATH": ROOT},
                         stdout=subprocess.PIPE, text=True)
    end = time.monotonic() + 30
    while time.monotonic() < end and not (pidfile.exists() and pidfile.read_text().strip()):
        time.sleep(0.05)
    guard.append(int(pidfile.read_text()))
    p.send_signal(signal.SIGTERM)
    out = p.communicate(timeout=60)[0]
    assert "app handler ran" in out and "rc -9" in out, out   # documented: SIGKILLed first, run_shell returns -9
    assert wait_gone(int(pidfile.read_text()))
