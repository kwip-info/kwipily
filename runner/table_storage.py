"""Storage-independent sequence contracts for owned, bounded typed tables.

Implementations own their scratch, support bounded slices and retain stable order.
Engines must not turn these sequences into whole lists without eager admission.
"""
from collections.abc import Sequence


class StoredRows(Sequence):
    def __eq__(self,other):
        if not isinstance(other,(list,tuple,StoredRows)):return NotImplemented
        return len(self)==len(other) and all(a==b for a,b in zip(self,other))


class StoredIDs(Sequence):
    def __eq__(self,other):
        if not isinstance(other,(list,tuple,StoredIDs)):return NotImplemented
        return len(self)==len(other) and all(a==b for a,b in zip(self,other))

    def sorted_iter(self):
        raise NotImplementedError('Stored identities require bounded lexical iteration')


import weakref

class SQLReadCursor:
    def __init__(self,lease,cursor):self.lease=lease;self.cursor=cursor;self.closed=False
    def _guard(self):
        if self.closed or self.lease.closed:raise ValueError('Stored SQL cursor is closed.')
    def __iter__(self):return self
    def __next__(self):self._guard();return next(self.cursor)
    def fetchone(self):self._guard();return self.cursor.fetchone()
    def fetchall(self):self._guard();return self.cursor.fetchall()
    def close(self):
        if not self.closed:
            self.closed=True;self.cursor.close();self.lease.cursors.discard(self)

class SQLReadLease:
    def __init__(self,connection):self.connection=connection;self.closed=False;self.cursors=weakref.WeakSet()
    def execute(self,*arguments):
        if self.closed:raise ValueError('Stored SQL reader is closed.')
        cursor=SQLReadCursor(self,self.connection.execute(*arguments));self.cursors.add(cursor);return cursor
    def close(self):
        if not self.closed:
            for cursor in list(self.cursors):cursor.close()
            self.closed=True;self.connection.close()
