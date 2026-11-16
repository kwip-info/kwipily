"""Deferred canonical table JSON identities for one captured execution.

Public/durable dataVersion remains the original canonical SHA256. An unretained
live preview may explicitly omit it; omission never becomes a content identity.
Only sealed owned populations qualify for deferred reads.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy
import threading

_active=ContextVar('deferred_canonical_table_identity',default=False)

def enabled():return _active.get()

@contextmanager
def deferred_identity_scope():
    token=_active.set(True)
    try:yield
    finally:_active.reset(token)

class CanonicalTableIdentity:
    def __init__(self,columns,rows,ids):
        self.columns=deepcopy(columns);self.rows=rows;self.ids=ids
        self._value=None;self._lock=threading.RLock()
    def resolve(self):
        with self._lock:
            if self._value is None:
                if __package__:from .dataflow import digest
                else:from dataflow import digest
                self._value=digest({'columns':self.columns,'rows':self.rows,'ids':self.ids})
                # The table owns its rows. Do not retain additional store custody
                # after the exact immutable digest has been computed.
                self.rows=self.ids=None
            return self._value


def version(table,columns,rows,ids):
    prior=object.__getattribute__(table,'data_version')
    if rows is table.rows and ids is table.row_ids and columns==table.columns and prior:return prior
    if __package__:from .table_storage import StoredRows,StoredIDs
    else:from table_storage import StoredRows,StoredIDs
    import sys
    if __package__:from .dataflow_disk import DiskStore,DiskRows,DiskIDs
    else:from dataflow_disk import DiskStore,DiskRows,DiskIDs
    owner=getattr(rows,'store',None)
    owned=(type(rows) is DiskRows and type(ids) is DiskIDs and type(owner) is DiskStore and owner is ids.store)
    native=sys.modules.get((__package__+'.' if __package__ else '')+'dataflow_columnar')
    shared=sys.modules.get((__package__+'.' if __package__ else '')+'dataflow_shared')
    if native is not None:
        owned|=(type(rows) is native.ColumnarRows and type(ids) is native.ColumnarIDs and owner is ids.store
            and (isinstance(owner,native.ColumnarStore) or shared is not None and isinstance(owner,shared.ColumnProjectionStore)))
    if enabled() and owned and owner.finished and not owner.closed:
        return CanonicalTableIdentity(columns,rows,ids)
    if __package__:from .dataflow import digest
    else:from dataflow import digest
    return digest({'columns':columns,'rows':rows,'ids':ids})
