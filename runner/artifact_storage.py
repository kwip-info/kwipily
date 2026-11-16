"""Bounded owned artifact publication using only the Python standard library."""
import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager


def _require(condition,message):
    if not condition:raise ValueError(message)


def _identity(info):return info.st_dev,info.st_ino


class ArtifactStorageBudget:
    """Conservative high-water reservation shared by an operation's output roots."""
    def __init__(self,maximum_bytes,check_cancelled=lambda:None):
        _require(type(maximum_bytes) is int and maximum_bytes>=0,'Invalid artifact storage budget')
        self.maximum_bytes=maximum_bytes;self.check_cancelled=check_cancelled
        self.used_bytes=0;self._lock=threading.Lock()

    def reserve(self,count):
        self.check_cancelled()
        with self._lock:
            _require(self.used_bytes+count<=self.maximum_bytes,'Artifact storage budget exceeded')
            self.used_bytes+=count

    def release(self,count):
        with self._lock:self.used_bytes-=count


class _BoundedSink:
    def __init__(self,stream,maximum,budget,check_cancelled):
        self._stream=stream;self._maximum=maximum;self._budget=budget
        self._check=check_cancelled;self.high_water=0

    def _extend(self,end):
        self._check()
        _require(end<=self._maximum,'Artifact storage file exceeds byte limit')
        delta=max(0,end-self.high_water)
        self._budget.reserve(delta);self.high_water+=delta

    def write(self,data):
        view=memoryview(data).cast('B')
        self._extend(self.tell()+len(view))
        count=0
        while count<len(view):
            self._check();written=self._stream.write(view[count:])
            if written is None or written<=0:raise OSError('Artifact write made no progress')
            count+=written
        return count

    def tell(self):return self._stream.tell()
    def seek(self,offset,whence=os.SEEK_SET):
        _require(type(offset) is int and whence in (os.SEEK_SET,os.SEEK_CUR,os.SEEK_END),'Invalid artifact seek')
        start=0 if whence==os.SEEK_SET else self.tell() if whence==os.SEEK_CUR else os.fstat(self._stream.fileno()).st_size
        end=start+offset;_require(end>=0,'Negative artifact seek')
        self._extend(end)
        return self._stream.seek(end)
    def flush(self):self._check();return self._stream.flush()
    def writable(self):return True
    def readable(self):return False
    def seekable(self):return True
    @property
    def closed(self):return self._stream.closed


class ArtifactStorage:
    """Exclusive atomic files under an anchored root, with pre-write accounting."""
    def __init__(self,root,maximum_bytes=None,check_cancelled=lambda:None,budget=None):
        _require((budget is None)!=(maximum_bytes is None),'Supply one artifact budget')
        _require(budget is None or isinstance(budget,ArtifactStorageBudget),'Invalid shared artifact budget')
        self.budget=budget if budget is not None else ArtifactStorageBudget(maximum_bytes,check_cancelled)
        self.check_cancelled=check_cancelled;self._root_fd=None;self._path=None
        if type(root) is int:self._root_fd=os.dup(root)
        else:
            self._path=os.fspath(root)
            self._root_fd=os.open(self._path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        info=os.fstat(self._root_fd)
        if not stat.S_ISDIR(info.st_mode):self.close();raise ValueError('Artifact root must be a directory')
        self._root_identity=_identity(info)

    def close(self):
        if self._root_fd is not None:os.close(self._root_fd);self._root_fd=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def __del__(self):
        try:self.close()
        except (AttributeError,OSError):pass

    def _parts(self,name):
        _require(isinstance(name,str) and '\0' not in name,'Invalid artifact name')
        parts=name.split('/')
        _require(all(part not in ('','.','..') and len(part.encode())<=255 for part in parts),'Invalid artifact path')
        return parts

    def _parent(self,parts,create=False):
        _require(self._root_fd is not None,'Artifact storage is closed')
        if self._path is not None:
            _require(_identity(os.stat(self._path,follow_symlinks=False))==self._root_identity,'Artifact root changed')
        current=os.dup(self._root_fd)
        try:
            for part in parts[:-1]:
                if create:
                    try:os.mkdir(part,0o700,dir_fd=current)
                    except FileExistsError:pass
                following=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=current)
                os.close(current);current=following
            return current
        except BaseException:os.close(current);raise

    @contextmanager
    def atomic_stream(self,name,maximum_bytes):
        _require(type(maximum_bytes) is int and maximum_bytes>=0,'Invalid artifact file bound')
        self.check_cancelled();parts=self._parts(name);parent=self._parent(parts,create=True)
        temporary=parts[-1]+'.tmp' if len(parts[-1].encode())<=251 else '.artifact-'+hashlib.sha256(parts[-1].encode()).hexdigest()+'.tmp'
        owned=None;sink=None;published=False;stream=None
        try:
            try:os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)
            except FileNotFoundError:pass
            else:raise FileExistsError('Artifact target already exists')
            descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=parent)
            owned=os.fstat(descriptor);stream=os.fdopen(descriptor,'wb',buffering=0)
            sink=_BoundedSink(stream,maximum_bytes,self.budget,self.check_cancelled)
            yield sink
            sink.flush();os.fsync(stream.fileno());self.check_cancelled()
            named=os.stat(temporary,dir_fd=parent,follow_symlinks=False)
            _require(_identity(named)==_identity(owned) and stat.S_ISREG(named.st_mode) and named.st_nlink==1,'Artifact temporary changed')
            current=self._parent(parts)
            try:_require(_identity(os.fstat(current))==_identity(os.fstat(parent)),'Artifact parent changed')
            finally:os.close(current)
            # link is an exclusive publication: a concurrent target is never replaced.
            os.link(temporary,parts[-1],src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
            published=True
            result=os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)
            _require(_identity(result)==_identity(owned),'Published artifact changed')
            os.unlink(temporary,dir_fd=parent)
            os.fsync(parent)
            sink.published_stat=os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)
        except BaseException:
            if owned is not None:
                for candidate in (temporary,parts[-1]) if published else (temporary,):
                    try:
                        if _identity(os.stat(candidate,dir_fd=parent,follow_symlinks=False))==_identity(owned):os.unlink(candidate,dir_fd=parent)
                    except FileNotFoundError:pass
            if sink is not None:self.budget.release(sink.high_water)
            raise
        finally:
            if stream is not None:stream.close()
            os.close(parent)

    def atomic_write(self,name,maximum_bytes,writer):
        with self.atomic_stream(name,maximum_bytes) as sink:writer(sink)
        return sink.published_stat

    def atomic_bytes(self,name,data,maximum_bytes):
        return self.atomic_write(name,maximum_bytes,lambda stream:stream.write(data))

    def atomic_json(self,name,value,maximum_bytes):
        return self.atomic_bytes(name,json.dumps(value,indent=2,allow_nan=False).encode(),maximum_bytes)
