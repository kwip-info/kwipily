"""Per-execution native-buffer residency with owned, budgeted IPC spill.

No datasets/events are logged. Arena entries retain immutable Polars clones and
weak store owners. Closing an unused store releases its buffers immediately.
The ceiling derives from captured working-data and worker-memory policies.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import threading,weakref
if __package__:
    from .workload_policy import current_policy
else:from workload_policy import current_policy

_current=ContextVar('kwip_native_frame_arena',default=None)


def current_arena():return _current.get()


class NativeFrameArena:
    def __init__(self,limit):
        self.limit=limit;self._entries=OrderedDict();self._lock=threading.RLock()
        self.used=0;self.peak=0;self.spilled=0;self.hits=0
    def _key(self,store,index):return (id(store),index)
    def retain(self,store,index,frame,content=None):
        cost=frame.estimated_size()+(len(content) if content is not None else 0)
        if cost>self.limit:return False
        with self._lock:
            key=self._key(store,index)
            if key in self._entries:raise ValueError('Native buffer custody is already assigned.')
            while self.used+cost>self.limit:
                prior,entry=next(iter(self._entries.items()));owner,number,_,encoded,bytes_=entry
                actual=owner()
                if actual is not None and not actual.closed and encoded is not None:
                    actual._spill_native_frame(number,encoded);self.spilled+=len(encoded)
                self._entries.pop(prior);self.used-=bytes_
            self._entries[key]=(weakref.ref(store),index,frame.clone(),content,cost)
            self.used+=cost;self.peak=max(self.peak,self.used)
            return True
    def read(self,store,index):
        with self._lock:
            key=self._key(store,index);entry=self._entries.get(key)
            if entry is None:return None
            if entry[0]() is not store:raise ValueError('Native buffer owner changed.')
            self._entries.move_to_end(key);self.hits+=1
            return entry[2].clone()
    def release(self,store):
        with self._lock:
            for key,entry in list(self._entries.items()):
                if entry[0]() is store:self.used-=entry[4];del self._entries[key]
    def report(self):
        with self._lock:return {'schemaVersion':1,'residentBytes':self.used,'peakResidentBytes':self.peak,'spillBytes':self.spilled,'bufferHits':self.hits,'maximumResidentBufferBytes':self.limit}


@contextmanager
def native_buffer_scope(*,maximum_bytes=None,sqlite_instructions=1000):
    if type(sqlite_instructions) is not int or sqlite_instructions not in (1000,32768):raise ValueError('Unsupported captured native VM work grain.')
    prior=current_arena()
    if prior is not None:yield prior;return
    if __package__:from .worker_resources import captured_resident_limit
    else:from worker_resources import captured_resident_limit
    # Reserve most of a captured worker ceiling for execution/native temporaries,
    # Python, models and viewers. This is a buffer cache ceiling, not an RSS claim.
    ceiling=min(current_policy().limit('scratchBytes'),captured_resident_limit() or current_policy().limit('scratchBytes'))//4
    limit=ceiling if maximum_bytes is None else maximum_bytes
    if type(limit) is not int or not 1<=limit<=ceiling:raise ValueError('Native buffer allowance exceeds captured resource policy.')
    arena=NativeFrameArena(limit);arena.sqlite_instructions=sqlite_instructions;token=_current.set(arena)
    try:yield arena
    finally:_current.reset(token)


def resident_view(function):
    """Cache virtual native batches within the same arena as physical buffers."""
    def read(store,index):
        arena=getattr(store,'_arena',None)
        if arena is None:return function(store,index)
        store.check_cancelled()
        frame=arena.read(store,index)
        if frame is not None:return frame
        frame=function(store,index);arena.retain(store,index,frame)
        return frame
    return read


def release_view(store):
    arena=getattr(store,'_arena',None)
    if arena is not None:arena.release(store)
