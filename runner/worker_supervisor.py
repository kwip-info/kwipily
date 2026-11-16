"""Internal trusted-command supervision; runtime bootstrap supplies launch authority.

No engine imports or public command-building CLI. Caller validates activity lock
identity/ownership and retains both supplied descriptors through return. Workers
must not spawn descendants. The worker arms its native lifetime guard before ML.
"""
import fcntl
import math
import json
import sys
import os
import selectors
import signal
import stat
import subprocess
import threading
import time


def supervise(command, *, parent_fd, activity_fd, grace_seconds=2.0, max_wall_seconds=None, max_resident_bytes=None):
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Worker supervision requires the main thread.')
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg or '\0' in arg for arg in command):
        raise ValueError('Expected a nonempty trusted command argument list.')
    if isinstance(grace_seconds, bool) or not isinstance(grace_seconds, (int, float)) or not math.isfinite(grace_seconds) or not 0 <= grace_seconds <= 10:
        raise ValueError('Termination grace must be finite and between zero and ten seconds.')
    if max_wall_seconds is not None and (
        isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
        or not 0 < max_wall_seconds <= 86400 or not math.isfinite(max_wall_seconds)
    ):
        raise ValueError('Worker wall-time budget must be finite, greater than zero and at most 86400 seconds.')
    if max_resident_bytes is not None and (type(max_resident_bytes) is not int or not 134217728 <= max_resident_bytes <= 1099511627776):
        raise ValueError('Resident memory budget must be a whole number of bytes from 128 MiB through 1 TiB.')
    memory_reader = None
    if max_resident_bytes is not None:
        if __package__:
            from .worker_memory import ResidentMemoryReader
        else:
            from worker_memory import ResidentMemoryReader
        memory_reader = ResidentMemoryReader()
    if type(parent_fd) is not int or parent_fd < 0 or type(activity_fd) is not int or activity_fd <= 2 or parent_fd == activity_fd:
        raise ValueError('Invalid supervision descriptors.')
    flags = fcntl.fcntl(parent_fd, fcntl.F_GETFL)
    if flags & os.O_ACCMODE != os.O_RDONLY or flags & os.O_NONBLOCK or not stat.S_ISFIFO(os.fstat(parent_fd).st_mode):
        raise ValueError('Parent lifetime descriptor must be a blocking read-only FIFO.')
    if not stat.S_ISREG(os.fstat(activity_fd).st_mode):
        raise ValueError('Activity descriptor must be a regular file.')

    cancelled = False
    child = None
    lifetime_read = lifetime_write = None
    previous = {}

    def cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    def stop_owned_child():
        if child is None or child.poll() is not None:
            return
        child.terminate()
        deadline = time.monotonic() + grace_seconds
        while child.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                child.kill()
                break
            try:
                child.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        child.wait()

    try:
        for name in (signal.SIGTERM, signal.SIGINT):
            previous[name] = signal.signal(name, cancel)
        with selectors.DefaultSelector() as selector:
            selector.register(parent_fd, selectors.EVENT_READ)
            # Do not start work when the lifetime channel is already revoked.
            if cancelled or selector.select(timeout=0):
                return 130
            lifetime_read, lifetime_write = os.pipe()
            deadline = None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
            child = subprocess.Popen(
                command + ['--kwip-lifetime-fd', str(lifetime_read), '--kwip-activity-fd', str(activity_fd)],
                stdin=subprocess.DEVNULL, close_fds=True,
                pass_fds=(lifetime_read, activity_fd))
            os.close(lifetime_read)
            lifetime_read = None
            while True:
                code = child.poll()
                if code is not None:
                    # Observed cancellation cannot become success in the exit race.
                    # A normal exit wins only while cancellation is unobserved.
                    return 130 if cancelled else code
                if cancelled:
                    stop_owned_child()
                    return 130
                if deadline is not None and time.monotonic() >= deadline:
                    # Observe parent revocation before classifying exhaustion. The
                    # owned child is always reaped before the activity lease ends.
                    if selector.select(timeout=0):
                        os.read(parent_fd, 1)
                        cancelled = True
                    stop_owned_child()
                    if selector.select(timeout=0):
                        os.read(parent_fd, 1)
                        cancelled = True
                    return 130 if cancelled else 124
                if memory_reader is not None:
                    observed = None
                    try:
                        observed = memory_reader.read(child.pid)
                        memory_error = observed > max_resident_bytes
                    except (OSError, ValueError):
                        # A child may exit between poll and libproc. Its real exit
                        # takes precedence over an unavailable observation then.
                        code = child.poll()
                        if code is not None:
                            return 130 if cancelled else code
                        memory_error = True
                    if memory_error:
                        if selector.select(timeout=0):
                            os.read(parent_fd, 1); cancelled = True
                        stop_owned_child()
                        if selector.select(timeout=0):
                            os.read(parent_fd, 1); cancelled = True
                        if cancelled:
                            return 130
                        event = {'event': 'error', 'code': 'residentMemoryLimitExceeded' if observed is not None else 'residentMemoryMonitoringFailed',
                                 'maximumResidentBytes': max_resident_bytes, 'observedResidentBytes': observed,
                                 'message': f'The worker exceeded its sampled resident memory budget ({observed} > {max_resident_bytes} bytes). Increase the memory budget or reduce the model/batch size before retrying.' if observed is not None else
                                            'Resident memory could not be observed. The worker was stopped; inspect runtime permissions before retrying.'}
                        print(event["message"], file=sys.stderr, flush=True)
                        print(json.dumps(event, allow_nan=False), flush=True)
                        return 125
                try:
                    wait_seconds = 0.05 if deadline is None else max(0, min(0.05, deadline - time.monotonic()))
                    events = selector.select(timeout=wait_seconds)
                    if events:
                        # A dedicated lifetime pipe carries no payload. EOF and
                        # unexpected bytes both revoke this launch's authority.
                        os.read(parent_fd, 1)
                        cancelled = True
                except InterruptedError:
                    continue
                except OSError:
                    cancelled = True
    finally:
        # Keep the writer alive until owned-child exit, including graceful TERM.
        # Unexpected supervisor death instead closes it in the kernel, allowing
        # the native worker guard to exit independently of Python/GIL state.
        try:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
        finally:
            for descriptor in (lifetime_read, lifetime_write):
                if descriptor is not None:
                    os.close(descriptor)
            for name, handler in previous.items():
                signal.signal(name, handler)
