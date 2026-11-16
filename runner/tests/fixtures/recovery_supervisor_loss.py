"""Test-only REC08 broker. Never signals a cached orphan PID.

The broker remains its process-group leader until self-directed group cleanup.
Worker self-stop is after inherited-authority validation and native guard arming.
"""
import argparse
import ctypes
import fcntl
import json
import os
import selectors
import signal
import subprocess
import sys
import time


def read_line(stream, timeout=10):
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise RuntimeError('Fixture handshake deadline exceeded')
        value = stream.readline()
        if not value:
            raise RuntimeError('Fixture channel closed')
        return value.decode().strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['broker', 'supervisor', 'worker'])
    p.add_argument('runner'); p.add_argument('watchdog')
    p.add_argument('group'); p.add_argument('nonce'); p.add_argument('intent'); p.add_argument('ownership')
    p.add_argument('--kwip-lifetime-fd', type=int)
    p.add_argument('--kwip-activity-fd', type=int)
    a = p.parse_args()
    sys.path.insert(0, a.runner)
    from worker_authorization import claim_launch, inherit_launch
    command = [sys.executable, '-I', os.path.realpath(__file__)]
    common = [a.runner, a.watchdog, a.group, a.nonce, a.intent, a.ownership]
    if a.mode == 'worker':
        authority = inherit_launch(a.group, a.nonce, a.intent, a.ownership, a.kwip_activity_fd)
        authority.check()
        lib = ctypes.CDLL(a.watchdog)
        arm = lib.kwip_lifetime_arm
        arm.argtypes = [ctypes.c_int]; arm.restype = ctypes.c_int
        if arm(a.kwip_lifetime_fd) != 0:
            raise RuntimeError('Native lifetime guard did not arm')
        print(json.dumps({'nonce': a.nonce, 'pid': os.getpid()}), flush=True)
        os.kill(os.getpid(), signal.SIGSTOP)
        while True:
            signal.pause()
    if a.mode == 'supervisor':
        from worker_supervisor import supervise
        with claim_launch(a.group, a.nonce, a.intent, a.ownership) as authority:
            code = supervise(command + ['worker'] + common, parent_fd=0, activity_fd=authority.activity_fd)
        raise SystemExit(code)
    # Foundation Process can already launch us as a process-group leader.
    # Otherwise create a new session/group; never adopt the caller's group.
    if os.getpgrp() != os.getpid():
        os.setsid()
    if os.getpgrp() != os.getpid():
        raise RuntimeError('Broker does not own its process group')
    broker_group = os.getpid()
    # Bounded watchdog handles test runner disappearance even if its pipe writer
    # is accidentally retained. SIGALRM executes cleanup in the broker itself.
    def signal_owned_group(sig):
        if os.getpid() != broker_group or os.getpgrp() != broker_group:
            raise RuntimeError('Broker process-group identity changed')
        os.killpg(broker_group, sig)
    def cleanup(*_):
        signal_owned_group(signal.SIGKILL)
    signal.signal(signal.SIGALRM, cleanup)
    signal.alarm(40)
    supervisor = None
    try:
        supervisor = subprocess.Popen(command + ['supervisor'] + common, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=0)
        ready = json.loads(read_line(supervisor.stdout))
        if ready['nonce'] != a.nonce:
            raise RuntimeError('Wrong launch handshake')
        # PID is used only for observation, never as a signal target.
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            state = subprocess.run(['/bin/ps', '-o', 'stat=', '-p', str(ready['pid'])],
                                   capture_output=True, text=True, timeout=2)
            if 'T' in state.stdout:
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Worker did not acknowledge its stopped barrier')
        print('WORKER_STOPPED', flush=True)
        while True:
            action = read_line(sys.stdin.buffer, 15)
            if action == 'KILL_SUPERVISOR':
                supervisor.kill()
                if supervisor.wait(timeout=5) != -signal.SIGKILL:
                    raise RuntimeError('Supervisor did not die from owned kill')
                print('SUPERVISOR_EXITED', flush=True)
            elif action == 'RESUME':
                signal_owned_group(signal.SIGCONT)
                descriptor = os.open(os.path.join(a.group, 'activity.lock'), os.O_RDWR | os.O_NOFOLLOW)
                try:
                    deadline = time.monotonic() + 8
                    while True:
                        try:
                            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise RuntimeError('Native guard did not release activity authority')
                            time.sleep(.01)
                finally:
                    os.close(descriptor)
                print('WORKER_RELEASED', flush=True)
            elif action == 'CLEANUP':
                return
            else:
                raise RuntimeError('Unknown fixture command')
    finally:
        # Self-directed group signal is safe even after supervisor reaping:
        # this live broker itself pins the group identity.
        cleanup()


if __name__ == '__main__':
    main()
