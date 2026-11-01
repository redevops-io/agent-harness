"""Run a child process and own its whole process tree (standard library only).

``subprocess.run(timeout=)`` kills only the direct child. A grandchild that outlives it keeps running, and on
Windows it can also keep the output pipes open so that the call blocks after the kill. This module runs a child
so that its descendants can be stopped together, on a timeout, on a cooperative cancel, or when the leader
exits and leaves processes behind:

* POSIX (Linux, macOS): the child starts in a new session and process group. Stopping sends ``SIGTERM`` to the
  group, waits a grace period, then sends ``SIGKILL`` to the group, then reaps the child.
* Windows: the child is created suspended in a new process group, assigned to a Job Object created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` and only then resumed, so it cannot start a grandchild that escapes the
  job. Stopping terminates the job. The job is created through :mod:`ctypes`; nothing is installed.

Output is read by reader threads with a size limit (the excess is counted, not stored) and a bounded drain after
a kill, so a descendant that still holds a pipe cannot block the caller.

This is process-tree control, not a security boundary. A job object or a process group does not restrict files,
network or system calls; a process on POSIX that starts its own session leaves the group. ``RunResult.tree_gone``
reports whether the tree was verified gone, so that a leak is visible rather than assumed away.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO, Callable, Iterator, List, Mapping, Optional, Sequence

IS_WINDOWS = sys.platform == "win32"

DEFAULT_GRACE = 2.0           # seconds between SIGTERM and SIGKILL (POSIX); Windows terminates the job at once
DEFAULT_DRAIN = 2.0           # seconds to wait for the output readers after the tree is stopped
DEFAULT_MAX_OUTPUT = 1_000_000  # bytes kept per stream; the excess is counted and dropped
_POLL = 0.02


class ProcessRunnerError(RuntimeError):
    """The child could not be started under tree control (the runner fails closed rather than run it loose)."""


class CancelToken:
    """A cooperative cancel flag. ``cancel()`` may be called from any thread, before or during a run."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    __call__ = is_cancelled


@dataclass
class RunControl:
    """Limits an enclosing caller (the agent loop) puts on every process started below it: a wall-clock deadline
    (``time.monotonic()`` value) and a cooperative cancel callable. A tool that runs a process reads these, so
    ``max_seconds`` and a cancel apply during a tool call, not only between turns."""
    deadline: Optional[float] = None
    cancel: Optional[Callable[[], bool]] = None

    def remaining(self) -> Optional[float]:
        return None if self.deadline is None else max(0.0, self.deadline - time.monotonic())


_CONTROL: "contextvars.ContextVar[Optional[RunControl]]" = contextvars.ContextVar("agent_harness_run_control",
                                                                                   default=None)


def current_control() -> Optional[RunControl]:
    return _CONTROL.get()


@contextlib.contextmanager
def run_control(*, max_seconds: Optional[float] = None, cancel: Optional[Callable[[], bool]] = None) -> Iterator[RunControl]:
    """Apply a deadline and/or cancel to every process started inside the ``with`` block (same thread or a copied
    context). Nested blocks tighten: the earlier deadline wins and either cancel stops the run."""
    outer = _CONTROL.get()
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    if outer is not None and outer.deadline is not None and (deadline is None or outer.deadline < deadline):
        deadline = outer.deadline
    outer_cancel = outer.cancel if outer is not None else None
    if outer_cancel is not None and cancel is not None:
        a, b = outer_cancel, cancel
        merged: Optional[Callable[[], bool]] = lambda: bool(a() or b())
    else:
        merged = cancel or outer_cancel
    ctl = RunControl(deadline, merged)
    token = _CONTROL.set(ctl)
    try:
        yield ctl
    finally:
        _CONTROL.reset(token)


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: Optional[int]          # None when the process could not be started
    timed_out: bool = False
    cancelled: bool = False
    stdout_truncated: int = 0          # bytes dropped past ``max_output``
    stderr_truncated: int = 0
    tree_stopped: bool = False         # the runner had to stop the tree (timeout, cancel or leftover descendants)
    tree_gone: bool = True             # the tree was verified gone at the end (False = possible leak)
    drain_complete: bool = True        # both output readers reached end of file
    elapsed: float = 0.0
    error: str = ""
    stdout_bytes: bytes = b""          # the same output before decoding
    stderr_bytes: bytes = b""


# ---------------------------------------------------------------------------------------------------------------
# Windows: Job Object through ctypes
# ---------------------------------------------------------------------------------------------------------------

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _nt = ctypes.WinDLL("ntdll", use_last_error=True)

    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001
    PROCESS_SUSPEND_RESUME = 0x0800
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    CREATE_SUSPENDED = 0x00000004
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_NO_WINDOW = 0x08000000
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectBasicAccountingInformation = 1
    JobObjectExtendedLimitInformation = 9

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _BASIC_ACCOUNTING(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                               ctypes.POINTER(wintypes.DWORD)]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
    _nt.NtResumeProcess.restype = ctypes.c_long

    class _Job:
        """A Job Object that kills every process in it when the last handle to it closes."""

        def __init__(self) -> None:
            self.handle = _k32.CreateJobObjectW(None, None)
            if not self.handle:
                raise ProcessRunnerError(f"CreateJobObjectW failed (error {ctypes.get_last_error()})")
            info = _EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _k32.SetInformationJobObject(self.handle, JobObjectExtendedLimitInformation,
                                                ctypes.byref(info), ctypes.sizeof(info)):
                err = ctypes.get_last_error()
                self.close()
                raise ProcessRunnerError(f"SetInformationJobObject failed (error {err})")

        def assign_and_resume(self, pid: int) -> None:
            """Put the (suspended) process in the job, then let it run."""
            access = PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION
            ph = _k32.OpenProcess(access, False, pid)
            if not ph:
                raise ProcessRunnerError(f"OpenProcess failed (error {ctypes.get_last_error()})")
            try:
                if not _k32.AssignProcessToJobObject(self.handle, ph):
                    raise ProcessRunnerError(f"AssignProcessToJobObject failed (error {ctypes.get_last_error()})")
                status = _nt.NtResumeProcess(ph)
                if status != 0:
                    raise ProcessRunnerError(f"NtResumeProcess failed (status {status & 0xFFFFFFFF:#x})")
            finally:
                _k32.CloseHandle(ph)

        def active_processes(self) -> int:
            info = _BASIC_ACCOUNTING()
            if not _k32.QueryInformationJobObject(self.handle, JobObjectBasicAccountingInformation,
                                                  ctypes.byref(info), ctypes.sizeof(info), None):
                return -1
            return int(info.ActiveProcesses)

        def terminate(self) -> None:
            _k32.TerminateJobObject(self.handle, 1)

        def close(self) -> None:
            if self.handle:
                _k32.CloseHandle(self.handle)
                self.handle = None


# ---------------------------------------------------------------------------------------------------------------
# Output readers
# ---------------------------------------------------------------------------------------------------------------

class _Reader(threading.Thread):
    """Reads one pipe until end of file or ``stop``; keeps at most ``limit`` bytes and counts the rest."""

    def __init__(self, stream, limit: int, sink: Optional[BinaryIO] = None) -> None:
        super().__init__(daemon=True)
        self.stream, self.limit, self.sink = stream, limit, sink
        self.chunks: List[bytes] = []
        self.kept = 0
        self.dropped = 0
        self.eof = False
        self.stop = threading.Event()

    def run(self) -> None:
        fd = self.stream.fileno() if not IS_WINDOWS else None
        try:
            while True:
                if fd is not None:
                    # POSIX: wake up regularly so a reader can be told to stop even if a descendant that left
                    # the process group still holds the write end of the pipe.
                    if self.stop.is_set():
                        return
                    ready, _, _ = select.select([fd], [], [], 0.05)
                    if not ready:
                        continue
                    data = os.read(fd, 65536)
                else:
                    data = self.stream.read(65536)
                if not data:
                    self.eof = True
                    return
                if self.sink is not None:          # tee: written as it arrives (a long run can be followed live), not kept
                    self.sink.write(data)
                    self.sink.flush()
                    continue
                room = self.limit - self.kept
                if room > 0:
                    self.chunks.append(data[:room])
                    self.kept += min(room, len(data))
                self.dropped += max(0, len(data) - max(room, 0))
        except (OSError, ValueError):
            self.eof = True

    def raw(self) -> bytes:
        return b"".join(self.chunks)

    def text(self, encoding: str) -> str:
        return self.raw().decode(encoding, errors="replace")


# ---------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------

class ProcessRunner:
    """Runs argv lists under tree control. One runner can serve many runs; its ``cancel()`` stops the run in
    progress (if any) and makes later runs return ``cancelled`` without starting, until ``reset()``."""

    def __init__(self, *, grace: float = DEFAULT_GRACE, drain: float = DEFAULT_DRAIN,
                 max_output: int = DEFAULT_MAX_OUTPUT, encoding: str = "utf-8") -> None:
        self.grace, self.drain, self.max_output, self.encoding = grace, drain, max_output, encoding
        self._token = CancelToken()

    def cancel(self) -> None:
        self._token.cancel()

    def reset(self) -> None:
        self._token.reset()

    @property
    def cancelled(self) -> bool:
        return self._token.is_cancelled()

    def run(self, argv: Sequence[str], *, cwd: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = None, cancel: Optional[Callable[[], bool]] = None,
            max_output: Optional[int] = None, stdout_to: Optional[BinaryIO] = None,
            merge_stderr: bool = False) -> RunResult:
        """Run ``argv`` (never through a shell). Never raises for a child that fails, times out or is cancelled;
        a child that cannot be started under tree control comes back as ``returncode=None`` with ``error`` set.
        ``stdout_to`` writes the child's stdout to a binary file object as it arrives instead of keeping it;
        ``merge_stderr`` sends stderr into the same stream."""
        t0 = time.monotonic()
        limit = self.max_output if max_output is None else max_output

        def want_cancel() -> bool:
            return self._token.is_cancelled() or bool(cancel is not None and cancel())

        if want_cancel():
            return RunResult("", "", None, cancelled=True, elapsed=0.0)
        try:
            return self._run(list(argv), cwd, env, timeout, want_cancel, limit, t0, stdout_to, merge_stderr)
        except (OSError, ProcessRunnerError, ValueError) as e:
            return RunResult("", "", None, error=str(e), elapsed=time.monotonic() - t0)

    # -- internals ------------------------------------------------------------------------------------------------

    def _spawn(self, argv, cwd, env, merge_stderr=False):
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE, bufsize=0,
                      cwd=cwd, env=None if env is None else dict(env), shell=False)
        if not IS_WINDOWS:
            return subprocess.Popen(argv, start_new_session=True, **kwargs), None
        job = _Job()
        try:
            proc = subprocess.Popen(argv, creationflags=CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
                                    **kwargs)
        except BaseException:
            job.close()
            raise
        try:
            job.assign_and_resume(proc.pid)
        except BaseException:
            proc.kill()
            proc.wait()
            for s in (proc.stdout, proc.stderr):
                if s is not None:
                    s.close()
            job.close()
            raise
        return proc, job

    def _run(self, argv, cwd, env, timeout, want_cancel, limit, t0, stdout_to=None, merge_stderr=False) -> RunResult:
        proc, job = self._spawn(argv, cwd, env, merge_stderr)
        out_r = _Reader(proc.stdout, limit, stdout_to)
        err_r = _Reader(proc.stderr, limit) if proc.stderr is not None else None
        readers = [r for r in (out_r, err_r) if r is not None]
        for r in readers:
            r.start()
        timed_out = cancelled = False
        tree_gone, leftovers = False, True
        deadline = None if timeout is None else t0 + timeout
        try:
            while proc.poll() is None:
                if want_cancel():
                    cancelled = True
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=_POLL if deadline is None else max(0.0, min(_POLL, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            # The leader is done or must be stopped. Either way nothing it started may outlive the call.
            tree_gone, leftovers = self._stop_tree(proc, job, graceful=timed_out or cancelled)
        except BaseException:            # e.g. Ctrl-C: the child is in its own group and would not see it, so stop it here
            self._stop_tree(proc, job, graceful=True)
            raise
        finally:
            drain_ok = self._drain(*readers)
            for r, s in ((out_r, proc.stdout), (err_r, proc.stderr)):
                if r is not None and not r.is_alive():              # never close a handle another thread is still blocked on
                    try:
                        s.close()
                    except OSError:
                        pass
            if job is not None:
                job.close()
        return RunResult(out_r.text(self.encoding), err_r.text(self.encoding) if err_r else "", proc.returncode,
                         timed_out=timed_out, cancelled=cancelled,
                         stdout_truncated=out_r.dropped, stderr_truncated=err_r.dropped if err_r else 0,
                         stdout_bytes=out_r.raw(), stderr_bytes=err_r.raw() if err_r else b"",
                         tree_stopped=timed_out or cancelled or leftovers, tree_gone=tree_gone,
                         drain_complete=drain_ok, elapsed=time.monotonic() - t0)

    def _drain(self, *readers: _Reader) -> bool:
        end = time.monotonic() + self.drain
        for r in readers:
            r.join(max(0.0, end - time.monotonic()))
        for r in readers:
            if r.is_alive():
                r.stop.set()          # POSIX readers poll this; a Windows reader ends when the job's pipes close
        for r in readers:
            r.join(0.5)
        return all(not r.is_alive() and r.eof for r in readers)

    def _stop_tree(self, proc: subprocess.Popen, job, *, graceful: bool):
        """Stop the leader's whole tree. Returns ``(tree_gone, leftovers)``: whether the tree is verified gone,
        and whether anything was still running (the leader included) when the stop began."""
        if job is not None:
            return self._stop_job(proc, job)
        return self._stop_group(proc, graceful)

    # POSIX ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _group_alive(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _signal(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _stop_group(self, proc: subprocess.Popen, graceful: bool):
        import signal
        pgid = proc.pid                      # new session: the leader's pid is the group id
        leader_running = proc.poll() is None
        if not leader_running and not self._group_alive(pgid):
            return True, False               # the leader exited and left nothing behind
        if graceful:
            self._signal(pgid, signal.SIGTERM)
            end = time.monotonic() + self.grace
            while time.monotonic() < end:
                if proc.poll() is not None and not self._group_alive(pgid):
                    break
                time.sleep(_POLL)
        self._signal(pgid, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        end = time.monotonic() + 5
        while self._group_alive(pgid) and time.monotonic() < end:
            time.sleep(_POLL)
        return (not self._group_alive(pgid)), True

    # Windows -------------------------------------------------------------------------------------------------------
    def _stop_job(self, proc: subprocess.Popen, job):
        leftovers = proc.poll() is None or job.active_processes() > 0
        if leftovers:
            job.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        end = time.monotonic() + 5
        while job.active_processes() > 0 and time.monotonic() < end:
            time.sleep(_POLL)
        return job.active_processes() == 0, leftovers


def run_process(argv: Sequence[str], **kwargs) -> RunResult:
    """One-shot convenience: ``ProcessRunner().run(argv, **kwargs)`` with the default limits."""
    return ProcessRunner().run(argv, **kwargs)


def run_subprocess(argv: Sequence[str], *, cwd=None, env=None, capture_output: bool = False, text: bool = False,
                   timeout: Optional[float] = None, stdout=None, stderr=None, max_output: int = 256 * 1024 * 1024,
                   ) -> "subprocess.CompletedProcess":
    """A drop-in for the subset of :func:`subprocess.run` the benchmark runners use, with tree control: the child and
    everything it starts are stopped on a timeout (``subprocess.TimeoutExpired`` is raised, as ``subprocess.run`` does),
    on a cancel or deadline set by an enclosing :func:`run_control`, and when the call ends. ``stdout`` may be a binary
    file object (written as the output arrives); ``stderr=subprocess.STDOUT`` merges stderr into it. A child that
    cannot be started raises ``OSError``, as ``subprocess.run`` does."""
    sink = stdout if stdout not in (None, subprocess.PIPE, subprocess.DEVNULL) else None
    ctl = current_control()
    cancel = ctl.cancel if ctl is not None else None
    if ctl is not None and ctl.remaining() is not None:
        timeout = ctl.remaining() if timeout is None else min(timeout, ctl.remaining())
    res = ProcessRunner(max_output=max_output).run(argv, cwd=None if cwd is None else str(cwd), env=env, timeout=timeout,
                                                   cancel=cancel, stdout_to=sink,
                                                   merge_stderr=stderr == subprocess.STDOUT)
    if res.returncode is None and res.error and not res.cancelled:
        raise OSError(res.error)
    if res.timed_out:
        raise subprocess.TimeoutExpired(list(argv), timeout or 0, output=res.stdout_bytes, stderr=res.stderr_bytes)
    keep = capture_output or stdout == subprocess.PIPE
    out = (res.stdout if text else res.stdout_bytes) if keep and sink is None else None
    err = (res.stderr if text else res.stderr_bytes) if (capture_output or stderr == subprocess.PIPE) else None
    return subprocess.CompletedProcess(list(argv), res.returncode if res.returncode is not None else -2, out, err)

