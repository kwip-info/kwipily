"""Bounded immutable backend-input proof; live kernel events, never a timed cache.

Darwin vnode events invalidate retained inputs, including transient writes/renames.
Other platforms and exhausted descriptor budgets retain the full verifier.
Owner locks, root namespace and the three launch controls remain the caller's job.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import sys
import threading

if __package__:
    from .worker_authorization import directory, require, signature
else:
    from worker_authorization import directory, require, signature


def binding(value):
    # Exact numeric types matter (e.g. 2.0 must not impersonate an integer size).
    try:
        raw=json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False).encode('utf-8')
    except (TypeError,ValueError) as error:
        raise ValueError('Invalid backend request representation') from error
    require(len(raw)<=256000,'Backend request exceeds its control frame')
    return raw


class BackendInputProof:
    def __init__(self, value, root, inventory):
        self.closed = False
        self.invalid = False
        self.lock = threading.RLock()
        self.descriptors = []
        self.queue = None
        self.kernel = False
        self.root = os.fspath(root)
        self.request = binding(value)
        current=directory(self.root)
        try:
            info=os.fstat(current);self._private_directory(info)
            self.root_identity=self._root_identity(info)
        finally:os.close(current)
        if sys.platform != 'darwin' or not hasattr(select, 'kqueue'):
            return
        # Leave room for two table readers (four pack readers each), SQLite,
        # launch controls and backend imports. Falling back never rejects data.
        import resource
        soft,_=resource.getrlimit(resource.RLIMIT_NOFILE)
        parents={tuple(name.split('/')[:-1]) for name in inventory}
        needed=len(inventory)+len(parents)+2
        if soft!=resource.RLIM_INFINITY and needed>max(0,soft-64):return
        try:
            self.queue = select.kqueue()
            self.root_fd = self._retain(directory(self.root))
            info = os.fstat(self.root_fd)
            self._private_directory(info)
            self.root_identity = self._root_identity(info)
            parents = {}
            events = (select.KQ_NOTE_DELETE | select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND |
                      select.KQ_NOTE_ATTRIB | select.KQ_NOTE_LINK | select.KQ_NOTE_RENAME |
                      select.KQ_NOTE_REVOKE)
            def watch(fd):
                self.queue.control([select.kevent(fd, filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR, fflags=events)], 0, 0)
            for name, entry in inventory.items():
                parts = name.split('/')
                key = tuple(parts[:-1])
                if key not in parents:
                    parent = self._retain(directory(str(Path(self.root).joinpath(*key))))
                    before = os.fstat(parent); self._private_directory(before)
                    watch(parent)
                    require(signature(os.fstat(parent)) == signature(before),
                            'Backend input directory changed during admission')
                    parents[key] = parent
                parent = parents[key]
                fd = self._retain(os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW |
                    os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent))
                before = os.fstat(fd)
                require(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
                    and before.st_nlink == 1 and before.st_mode & 0o022 == 0
                    and before.st_size == entry['bytes'], 'Unsafe backend input')
                require(signature(before) == signature(os.stat(parts[-1], dir_fd=parent,
                    follow_symlinks=False)), 'Backend input name differs from descriptor')
                watch(fd)
                sha = hashlib.sha256(); count = 0
                while True:
                    chunk = os.read(fd, min(262144, entry['bytes'] + 1 - count))
                    if not chunk: break
                    count += len(chunk); require(count <= entry['bytes'], 'Backend input grew')
                    sha.update(chunk)
                require(count == entry['bytes'] and sha.hexdigest() == entry['sha256']
                    and signature(before) == signature(os.fstat(fd))
                    == signature(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)),
                    'Backend input changed during admission')
            self.kernel = True
            self._poll()
        except OSError as error:
            self._release()
            if error.errno not in {errno.EMFILE, errno.ENFILE, errno.ENOSYS,
                                  errno.ENOTSUP, errno.EINVAL}:
                self.closed = True
                raise
            # A finite resource/unsupported-kernel fallback, never unchecked success.
            self.kernel = False
        except BaseException:
            self.closed = True
            self._release()
            raise

    @staticmethod
    def _private_directory(info):
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
                and info.st_mode & 0o022 == 0, 'Unsafe backend input directory')

    @staticmethod
    def _root_identity(info):
        return info.st_dev, info.st_ino, info.st_mode, info.st_uid

    def _retain(self, fd):
        self.descriptors.append(fd)
        return fd

    def _poll(self):
        require(not self.invalid, 'Backend input proof was invalidated')
        try:
            events = self.queue.control(None, max(1, len(self.descriptors)), 0)
        except OSError as error:
            self.invalid = True
            raise ValueError('Backend input event proof failed') from error
        if events:
            self.invalid = True
            raise ValueError('Immutable backend input changed')

    def check(self, value, root):
        with self.lock:
            require(not self.closed and not self.invalid, 'Backend input proof is unavailable')
            require(os.fspath(root) == self.root and binding(value) == self.request,
                    'Captured backend request changed')
            try:
                current = directory(self.root)
                try:
                    info = os.fstat(current); self._private_directory(info)
                    require(self._root_identity(info) == self.root_identity,
                            'Backend input root changed')
                finally:os.close(current)
                if not self.kernel:return False
                self._poll()
                return True
            except BaseException:
                self.invalid=True
                raise

    def invalidate(self):
        with self.lock:self.invalid=True

    def _release(self):
        if self.queue is not None:
            try:self.queue.close()
            except OSError:pass
            self.queue = None
        for fd in self.descriptors:
            try:os.close(fd)
            except OSError:pass
        self.descriptors.clear()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self._release()

    def __del__(self):
        self.close()
