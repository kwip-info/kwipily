"""Owned, bounded disk storage for preparation rows and canonical streaming JSON."""
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
if __package__:from .workload_policy import current_policy
else:from workload_policy import current_policy
if __package__:from .table_storage import StoredRows,StoredIDs,SQLReadLease
else:from table_storage import StoredRows,StoredIDs,SQLReadLease

MAX_SCRATCH_BYTES = 100_000_000
MAX_ROW_BYTES = 1_048_576
MAX_SLICE_ROWS = 100_000
_scope = ContextVar('dataflow_disk_scope', default=None)
_encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)


class ScratchBudget:
    def __init__(self, limit=None):
        admitted=current_policy().limit('scratchBytes')
        limit=admitted if limit is None else limit
        if type(limit) is not int or not 1 <= limit <= admitted:
            raise ValueError('Scratch allowance exceeds the captured preparation.scratchBytes policy.')
        self.limit = limit
        self.used = 0
        self.peak = 0
        self._lock = threading.Lock()

    def resize(self, previous, current):
        if type(previous) is not int or type(current) is not int or previous<0 or current<0:
            raise ValueError('Invalid scratch allocation accounting.')
        with self._lock:
            updated=self.used-previous+current
            if updated<0:raise ValueError('Scratch allocation accounting underflow.')
            if updated > self.limit:
                raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.')
            self.used=updated
            self.peak=max(self.peak,updated)


@contextmanager
def disk_scope(directory=None, budget=None, check_cancelled=None):
    """Share one scratch ceiling across all row stores owned by an execution."""
    previous = _scope.get()
    if budget is None:
        budget = previous[1] if previous else ScratchBudget()
    elif type(budget) is int:
        budget = ScratchBudget(budget)
    if not isinstance(budget, ScratchBudget):
        raise ValueError('Invalid disk preparation scratch budget.')
    selected = directory if directory is not None else (previous[0] if previous else None)
    check = check_cancelled if check_cancelled is not None else (previous[2] if previous else lambda:None)
    token = _scope.set((selected, budget, check))
    try:
        yield budget
    finally:
        _scope.reset(token)


def current_check():
    context=_scope.get()
    return context[2] if context else lambda:None


def current_scope():
    """Owned scratch directory, shared budget and cancellation for storage adapters."""
    return _scope.get()


class DiskStore:
    def __init__(self, check_cancelled=None, directory=None, budget=None, identity_prefix=""):
        context = _scope.get()
        self.check_cancelled = current_check() if check_cancelled is None else check_cancelled
        if type(identity_prefix) is not str or '\0' in identity_prefix or len(identity_prefix.encode('utf-8')) > 1024:
            raise ValueError('Invalid disk table identity prefix.')
        self.identity_prefix = identity_prefix
        self.check_cancelled()
        self.budget = budget if budget is not None else (context[1] if context else ScratchBudget())
        if type(self.budget) is int:
            self.budget = ScratchBudget(self.budget)
        if not isinstance(self.budget, ScratchBudget):
            raise ValueError('Invalid disk preparation scratch budget.')
        parent = directory if directory is not None else (context[0] if context else None)
        self.directory = Path(tempfile.mkdtemp(prefix='.table-rows-', dir=parent))
        self.path = self.directory / 'rows.sqlite3'
        self.connection = None
        self._read_lease=None;self._cached_pairs={}
        self.count = 0
        self.reserved = 0
        self.finished = False
        self.closed = False
        try:
            self.connection = sqlite3.connect(self.path)
            self._read_lease=SQLReadLease(self.connection)
            self.connection.execute('PRAGMA cache_size=-2048')
            self.connection.execute('PRAGMA temp_store=FILE')
            self.connection.execute('PRAGMA mmap_size=0')
            self.connection.execute('PRAGMA journal_mode=OFF')
            self._allocation_limit()
            self.connection.execute('CREATE TABLE rows (position INTEGER PRIMARY KEY, identity TEXT NOT NULL UNIQUE COLLATE BINARY, payload TEXT NOT NULL)')
            self._account()
        except BaseException:
            self.close()
            raise

    def _allocation_limit(self):
        page_size = self.connection.execute('PRAGMA page_size').fetchone()[0]
        available = self.budget.limit - self.budget.used + self.reserved
        pages = available // page_size
        if pages < 3:
            raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.')
        self.connection.execute('PRAGMA max_page_count=' + str(pages))

    def _account(self):
        page_count = self.connection.execute('PRAGMA page_count').fetchone()[0]
        page_size = self.connection.execute('PRAGMA page_size').fetchone()[0]
        current = page_count * page_size
        self.budget.resize(self.reserved, current)
        self.reserved = current

    def append(self, row, rid):
        if self.closed or self.finished:
            raise ValueError('Disk row store is no longer writable.')
        try:
            if self.count % 256 == 0:
                self.check_cancelled()
            if type(rid) is not str or not rid or '\0' in rid or len(rid.encode('utf-8')) > 1024:
                raise ValueError('Disk row identity must be a bounded nonempty string.')
            if not isinstance(row, (list, tuple)):
                raise ValueError('Disk table rows must be lists or tuples.')
            # Reject an oversized record during encoding rather than after joining it.
            chunks = []
            size = 0
            for chunk in _encoder.iterencode(row):
                size += len(chunk.encode('utf-8'))
                if size > MAX_ROW_BYTES:
                    raise ValueError('A disk table row exceeds the 1 MiB record budget.')
                chunks.append(chunk)
            if not rid.startswith(self.identity_prefix):
                raise ValueError('Disk row identity differs from its declared prefix.')
            self._allocation_limit()
            self.connection.execute('INSERT INTO rows VALUES (?, ?, ?)', (self.count, rid[len(self.identity_prefix):], ''.join(chunks)))
            self.count += 1
            # Account each insertion: a single large row must not escape the ceiling.
            self._account()
        except sqlite3.OperationalError as error:
            self.close()
            if 'full' in str(error).lower():
                raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.') from error
            raise
        except sqlite3.IntegrityError as error:
            self.close()
            raise ValueError('Disk table row identities must be unique.') from error
        except BaseException:
            self.close()
            raise

    def finish(self):
        if self.closed:
            raise ValueError('Disk row store is closed.')
        try:
            self.check_cancelled()
            self.connection.commit()
            self._account()
            self.connection.execute('PRAGMA query_only=ON')
            self.finished = True
            return DiskRows(self), DiskIDs(self)
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, 'closed', True):
            return
        self.closed = True
        if self.connection is not None:
            if self._read_lease is not None:self._read_lease.close()
            else:self.connection.close()
            self.connection = None
        self._cached_pairs.clear()
        self.budget.resize(self.reserved, 0)
        self.reserved = 0
        shutil.rmtree(self.directory, ignore_errors=True)

    def __del__(self):
        self.close()

    def _cached_record(self,index):
        if self.closed or not self.finished:raise ValueError('Disk row store is closed or incomplete.')
        if index not in self._cached_pairs:
            cached={};size=0
            cursor=self._read_lease.execute('SELECT position,payload,identity FROM rows WHERE position>=? AND position<? ORDER BY position',(index,min(index+128,self.count)))
            try:
                for position,payload,identity in cursor:
                    count=len(payload)+len(identity.encode('utf-8'))
                    if cached and size+count>2_097_152:break
                    cached[position]=(payload,identity);size+=count
            finally:cursor.close()
            if index not in cached:raise ValueError('Disk observation population differs.')
            self._cached_pairs=cached
        return self._cached_pairs[index]


class _DiskSequence(Sequence):
    column = None
    def __init__(self, store):
        self.store = store

    def __len__(self):
        return self.store.count

    def _decode(self, value):
        return value

    def __eq__(self, other):
        if not isinstance(other, (tuple, list, StoredRows, StoredIDs)):
            return NotImplemented
        return len(self) == len(other) and all(left == right for left, right in zip(self, other))

    def __getitem__(self, index):
        if isinstance(index, slice):
            selected = range(*index.indices(len(self)))
            if len(selected) > MAX_SLICE_ROWS:
                raise ValueError('Disk table slices exceed the bounded materialization budget; iterate rows instead.')
            return tuple(self[i] for i in selected)
        if type(index) is not int:
            raise TypeError('Disk table indices must be integers or slices.')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('Disk table index out of range.')
        self.store.check_cancelled()
        row = self.store._read_lease.execute('SELECT ' + self.column + ' FROM rows WHERE position=?', (index,)).fetchone()
        return self._decode(row[0])

    def __iter__(self):
        yield from self._iterate('position')

    def _iterate(self, order):
        self.store.check_cancelled()
        cursor = self.store._read_lease.execute('SELECT ' + self.column + ' FROM rows ORDER BY ' + order)
        try:
            for index, value in enumerate(cursor):
                if index % 256 == 0:
                    self.store.check_cancelled()
                yield self._decode(value[0])
        finally:
            cursor.close()


class DiskRows(_DiskSequence,StoredRows):
    maximum_encoded_row_bytes=MAX_ROW_BYTES
    column = 'payload'
    def _decode(self, value):
        return tuple(json.loads(value))

    def iter_pairs(self):
        self.store.check_cancelled();seen=0;processed=0
        cursor=self.store._read_lease.execute('SELECT payload,identity FROM rows ORDER BY position')
        try:
            for payload,identity in cursor:
                size=len(payload)+len(identity.encode('utf-8'))
                if seen%256==0 or processed+size>2_097_152:self.store.check_cancelled();processed=0
                seen+=1;processed+=size
                yield tuple(json.loads(payload)),self.store.identity_prefix+identity
            if seen!=len(self):raise ValueError('Disk row population differs.')
        finally:cursor.close()

    def iter_selected_pairs(self,indices):
        self.store.check_cancelled();count=0;processed=0
        for index in indices:
            if type(index) is not int or not 0<=index<len(self):raise IndexError('Disk selection index out of range.')
            payload,identity=self.store._cached_record(index)
            size=len(payload)+6*len((self.store.identity_prefix+identity).encode('utf-8'))
            if count>=256 or processed+size>2_097_152:self.store.check_cancelled();count=0;processed=0
            count+=1;processed+=size
            yield tuple(json.loads(payload)),self.store.identity_prefix+identity


class DiskIDs(_DiskSequence,StoredIDs):
    maximum_encoded_identity_bytes=6144
    column = 'identity'
    def _decode(self, value):
        return self.store.identity_prefix + value
    def contains(self, identity):
        if type(identity) is not str or not identity.startswith(self.store.identity_prefix):
            return False
        self.store.check_cancelled()
        return self.store._read_lease.execute('SELECT 1 FROM rows WHERE identity=?',
            (identity[len(self.store.identity_prefix):],)).fetchone() is not None

    def __contains__(self, identity):
        return self.contains(identity)

    def sorted_iter(self):
        """UTF-8 byte lexical order, matching canonical string sorting."""
        return self._iterate('identity COLLATE BINARY')


def iter_json(value, buffer_bytes=65536):
    """Canonical JSON bytes without materializing disk sequences or complete output."""
    if type(buffer_bytes) is not int or buffer_bytes < 1:
        raise ValueError('JSON chunk buffer must be positive.')
    def tokens(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError('Canonical JSON objects require string keys.')
            yield '{'
            for index, key in enumerate(sorted(item)):
                if index:
                    yield ','
                yield _encoder.encode(key)
                yield ':'
                yield from tokens(item[key])
            yield '}'
        elif isinstance(item,StoredRows) and callable(getattr(item,'iter_encoded_chunks',None)):
            yield '['
            yield from item.iter_encoded_chunks(_encoder)
            yield ']'
        elif isinstance(item,StoredIDs) and getattr(item,'maximum_encoded_identity_bytes',None)==6144:
            from itertools import islice
            yield '['
            iterator=iter(item);first=True
            while True:
                batch=tuple(islice(iterator,128))
                if not batch:break
                encoded=_encoder.encode(batch)
                if len(encoded)>1_048_576:raise ValueError('Stored identity group exceeds its JSON boundary.')
                if not first:yield ','
                yield encoded[1:-1];first=False
            yield ']'
        elif isinstance(item,StoredRows) and getattr(item,'maximum_encoded_row_bytes',None)==MAX_ROW_BYTES:
            # Owned stores already admit each record within this byte boundary.
            # Encode that bounded record in C instead of recursively visiting
            # every scalar again. Population and output buffers still stream.
            yield '['
            for index,row in enumerate(item):
                encoded=_encoder.encode(row)
                if len(encoded)>MAX_ROW_BYTES:raise ValueError('Stored row exceeds its declared JSON boundary.')
                if index:yield ','
                yield encoded
            yield ']'
        elif isinstance(item, (list, tuple, StoredRows, StoredIDs)):
            # C encoding avoids a Python generator per cell while preserving the
            # identical canonical JSON. Only bounded flat arrays use this path;
            # populations and larger strings still stream through this walker.
            if isinstance(item,(list,tuple)) and len(item)<=128:
                characters=0;flat=True
                for value in item:
                    if type(value) not in (str,int,float,bool,type(None)):
                        flat=False;break
                    if type(value) is str:characters+=len(value)
                    if characters>65536:flat=False;break
                if flat:
                    yield _encoder.encode(item)
                    return
            yield '['
            for index, child in enumerate(item):
                if index:
                    yield ','
                yield from tokens(child)
            yield ']'
        else:
            yield _encoder.encode(item)
    pending = bytearray()
    for token in tokens(value):
        raw = token.encode('utf-8')
        offset = 0
        while offset < len(raw):
            count = min(buffer_bytes - len(pending), len(raw) - offset)
            pending.extend(raw[offset:offset + count])
            offset += count
            if len(pending) == buffer_bytes:
                yield bytes(pending)
                pending.clear()
    if pending:
        yield bytes(pending)
