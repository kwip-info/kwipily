#!/usr/bin/env python3
"""One shared compute owner across native repository worktrees; never waits silently."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess


def lock_path(root):
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=root, text=True).strip())
    if not common.is_absolute(): common = Path(root) / common
    return common.resolve() / 'kwip-verification-compute.lock'


@contextmanager
def compute_lock(path, purpose):
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another maintained verification/build owns compute; inspect its progress before retrying.') from error
        try:
            stream.seek(0); stream.truncate()
            json.dump({'pid': os.getpid(), 'purpose': purpose}, stream)
            stream.flush(); os.fsync(stream.fileno())
            yield stream.fileno()
        finally:
            stream.seek(0); stream.truncate(); stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--purpose', required=True, choices=['full-baseline','feature','assembly'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('A command is required')
    root = Path(__file__).resolve().parents[1]
    with compute_lock(lock_path(root), args.purpose) as fd:
        return subprocess.run(command, cwd=root, pass_fds=(fd,)).returncode


if __name__ == '__main__': raise SystemExit(main())
