"""Owned bounded IPC scratch with explicit public schema and indexed identities.

IPC files contain only internal positional column names. Observation identities live
in a separate SQLite index, so no user column name is reserved by the engine.
"""
from bisect import bisect_right
from contextlib import contextmanager
from contextvars import ContextVar
import io
import json
import math
import os
import stat
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import polars as pl

if __package__:
    from .dataflow_disk import ScratchBudget, current_scope, MAX_ROW_BYTES, MAX_SLICE_ROWS
    from .table_storage import StoredRows, StoredIDs, SQLReadLease
else:
    from dataflow_disk import ScratchBudget, current_scope, MAX_ROW_BYTES, MAX_SLICE_ROWS
    from table_storage import StoredRows, StoredIDs, SQLReadLease

if __package__:from . import dataflow_temporal
else:import dataflow_temporal

MAX_BATCH_ROWS=4096
MAX_BATCH_BYTES=2_097_152
_reader_leases=ContextVar('columnar_reader_leases',default=None)




@contextmanager
def columnar_reader_scope():
    """Close a task's read-only handles before its worker thread can be reused."""
    leases=[];token=_reader_leases.set(leases)
    try:yield
    finally:
        try:
            for store,thread,connection in reversed(leases):store._drop_reader(thread,connection)
        finally:_reader_leases.reset(token)


DTYPES={'string':pl.String,'int64':pl.Int64,'float64':pl.Float64,'bool':pl.Boolean}
DTYPES['int64Sequence']=pl.List(pl.Int64)
DTYPES.update({'audio':pl.Binary,'tensor':pl.Binary})
DTYPES.update({dtype:pl.Int64 for dtype in dataflow_temporal.DTYPES})
DTYPES['image']=pl.Struct([pl.Field(name,pl.Int64 if name in ('schemaVersion','width','height','channels') else pl.String) for name in ('schemaVersion','kind','sha256','width','height','channels','dtype','layout')])


class _ChunkOverflow(ValueError):pass


class _IPCBuffer(io.BytesIO):
    overflow=False
    def write(self,value):
        if self.tell()+len(value)>MAX_BATCH_BYTES:
            self.overflow=True
            raise _ChunkOverflow('IPC batch byte budget exceeded')
        return super().write(value)


class ColumnarStore:
    def __init__(self,columns,check_cancelled=None,directory=None,budget=None,identity_prefix='',*,residency=True):
        context=current_scope()
        self.closed=True;self.finished=False;self.connection=None;self.directory=None
        self.count=0;self.reserved=0;self.file_bytes=0;self.chunks=[];self.starts=[]
        if __package__:from .native_frame_arena import current_arena
        else:from native_frame_arena import current_arena
        if type(residency) is not bool:raise ValueError('Native residency must be explicit.')
        self._arena=current_arena() if residency else None
        self.pending=[];self.pending_ids=[];self.pending_bytes=0;self.cached_chunk=None
        self.cached_identity_page=None
        self._read_lock=threading.RLock()
        self._connection_lock=threading.RLock();self._owner_thread=threading.get_ident();self._read_connections={}
        self.columns=tuple(dict(c) for c in columns)
        self._json_columns=tuple(i for i,c in enumerate(self.columns) if c['dtype'] in ('audio','tensor'))
        self.observed_nulls=[False]*len(self.columns)
        self.check_cancelled=check_cancelled if check_cancelled is not None else (context[2] if context else lambda:None)
        self.budget=budget if budget is not None else (context[1] if context else ScratchBudget())
        if type(self.budget) is int:self.budget=ScratchBudget(self.budget)
        if not isinstance(self.budget,ScratchBudget):raise ValueError('Invalid disk preparation scratch budget.')
        if type(identity_prefix) is not str or '\0' in identity_prefix or len(identity_prefix.encode('utf-8'))>1024:
            raise ValueError('Invalid columnar table identity prefix.')
        self.identity_prefix=identity_prefix
        if not 1<=len(self.columns)<=128 or any(not {'name','dtype','nullable'}<=set(c) or set(c)-{'name','dtype','nullable','displayLabel','temporal'} or type(c['name']) is not str or not c['name'].strip() or len(c['name'])>200 or c['dtype'] not in DTYPES or type(c['nullable']) is not bool or ('displayLabel' in c and (type(c['displayLabel']) is not str or len(c['displayLabel'])>400)) for c in self.columns) or len({c['name'] for c in self.columns})!=len(self.columns):
            raise ValueError('Invalid columnar table schema.')
        for column in self.columns:dataflow_temporal.validate_column(column)
        self.schema=[(f'c{i}',DTYPES[c['dtype']]) for i,c in enumerate(self.columns)]
        self.check_cancelled()
        parent=directory if directory is not None else (context[0] if context else None)
        self.closed=False
        self.directory=Path(tempfile.mkdtemp(prefix='.table-columns-',dir=parent))
        try:
            self.connection=sqlite3.connect(self.directory/'ids.sqlite3',check_same_thread=False)
            self.connection.execute('PRAGMA cache_size=-1024')
            self.connection.execute('PRAGMA temp_store=FILE')
            self.connection.execute('PRAGMA mmap_size=0')
            self.connection.execute('PRAGMA journal_mode=OFF')
            self.page_size=self.connection.execute('PRAGMA page_size').fetchone()[0]
            self._allocation_limit()
            self.connection.execute('CREATE TABLE ids (position INTEGER PRIMARY KEY, identity TEXT NOT NULL UNIQUE COLLATE BINARY)')
            self._account()
        except sqlite3.IntegrityError as error:
            self.close();raise ValueError('Columnar table row identities must be unique.') from error
        except sqlite3.OperationalError as error:
            self.close()
            if 'full' in str(error).lower():raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.') from error
            raise
        except BaseException:self.close();raise

    def _allocation_limit(self):
        available=self.budget.limit-self.budget.used+self.reserved-self.file_bytes
        pages=available//self.page_size
        if pages<3:raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.')
        self.connection.execute('PRAGMA max_page_count='+str(pages))

    @property
    def connection(self):
        if self._connection is None:return None
        thread=threading.get_ident()
        if thread==self._owner_thread:return self._connection
        if not self.finished:raise ValueError('An incomplete columnar store cannot cross thread ownership.')
        with self._connection_lock:
            if self.closed:raise ValueError('Columnar row store is closed.')
            if thread not in self._read_connections:
                path=self.directory/'ids.sqlite3';descriptor=None;connection=None
                try:
                    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
                    if self._index_fingerprint(os.fstat(descriptor))!=self._ids_fingerprint:raise ValueError('Immutable columnar identity index changed.')
                    uri=path.absolute().as_uri()+'?mode=ro&immutable=1'
                    connection=sqlite3.connect(uri,uri=True,check_same_thread=False)
                    connection.execute('PRAGMA query_only=ON');connection.execute('PRAGMA cache_size=-512');connection.execute('PRAGMA mmap_size=0')
                    if (self._index_fingerprint(path.lstat())!=self._ids_fingerprint
                        or self._index_fingerprint(os.fstat(descriptor))!=self._ids_fingerprint):raise ValueError('Immutable columnar identity index changed.')
                except BaseException as error:
                    if connection is not None:connection.close()
                    if isinstance(error,(OSError,ValueError,sqlite3.Error)):
                        raise ValueError('Immutable columnar identity index changed.') from None
                    raise
                finally:
                    if descriptor is not None:os.close(descriptor)
                connection=SQLReadLease(connection);self._read_connections[thread]=connection
                leases=_reader_leases.get()
                if leases is not None:leases.append((self,thread,connection))
            return self._read_connections[thread]

    @connection.setter
    def connection(self,value):self._connection=value;self._owner_read_lease=SQLReadLease(value) if value is not None else None

    def _reader_cursor(self,query,parameters=()):
        if self.closed:raise ValueError('Columnar row store is closed.')
        if threading.get_ident()==self._owner_thread:return self._owner_read_lease.execute(query,parameters)
        return self.connection.execute(query,parameters)

    @property
    def active_reader_count(self):
        with self._connection_lock:return len(self._read_connections)

    def _drop_reader(self,thread,connection):
        with self._connection_lock:
            if self._read_connections.get(thread) is connection:
                del self._read_connections[thread];connection.close()

    @staticmethod
    def _index_fingerprint(info):
        if not stat.S_ISREG(info.st_mode):raise ValueError('Columnar identity index must be regular.')
        return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns

    def _identity_at(self,index):
        """Private indexed read following a checked row/identity admission."""
        with self._read_lock:
            if self.closed:raise ValueError('Columnar row store is closed.')
            if type(index) is not int or not 0<=index<self.count:raise IndexError('Columnar identity index out of range.')
            start=index//128*128
            if self.cached_identity_page is None or self.cached_identity_page[0]!=start:
                cursor=self._reader_cursor('SELECT identity FROM ids WHERE position>=? AND position<? ORDER BY position',(start,min(start+128,self.count)))
                try:values=tuple(value for (value,) in cursor)
                finally:cursor.close()
                if len(values)!=min(128,self.count-start):raise ValueError('Columnar identity page population differs.')
                self.cached_identity_page=(start,values)
            return self.identity_prefix+self.cached_identity_page[1][index-start]

    def _account(self):
        size=self.file_bytes+self.connection.execute('PRAGMA page_count').fetchone()[0]*self.page_size
        self.budget.resize(self.reserved,size);self.reserved=size

    def append(self,row,rid):
        if self.closed or self.finished:raise ValueError('Columnar row store is no longer writable.')
        try:
            if self.count%256==0:self.check_cancelled()
            if type(rid) is not str or not rid or '\0' in rid or len(rid.encode('utf-8'))>1024 or not rid.startswith(self.identity_prefix):
                raise ValueError('Columnar row identity must be bounded and match its prefix.')
            if not isinstance(row,(tuple,list)) or len(row)!=len(self.columns):raise ValueError('Columnar row width differs from schema.')
            for value,column in zip(row,self.columns):
                dtype=column['dtype']
                if dtype in ('audio','tensor') and value is not None:
                    if __package__:from .media_artifacts import is_reference
                    else:from media_artifacts import is_reference
                    if not is_reference(value,dtype):raise ValueError('Invalid typed media reference.')
                if dtype=='image' and value is not None:
                    if __package__:from .image_artifacts import validate_reference
                    else:from image_artifacts import validate_reference
                    validate_reference(value)
                if dtype=='int64Sequence' and value is not None:
                    if __package__:from .named_inputs import valid_sequence
                    else:from named_inputs import valid_sequence
                    if not valid_sequence(value):raise ValueError('Invalid typed token sequence.')
                valid=(value is None and column['nullable']) or (value is not None and (
                    (dtype in ('image','audio','tensor','int64Sequence')) or
                    (dtype in dataflow_temporal.DTYPES and dataflow_temporal.valid_value(value,dtype)) or
                    (dtype=='string' and type(value) is str) or
                    (dtype=='bool' and type(value) is bool) or
                    (dtype=='int64' and type(value) is int and -(2**63)<=value<2**63) or
                    (dtype=='float64' and type(value) is float and math.isfinite(value))))
                if not valid:raise ValueError('Columnar value violates its declared dtype/nullability.')
            encoded=len(json.dumps(row,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode('utf-8'))
            if encoded>MAX_ROW_BYTES:raise ValueError('A columnar table row exceeds the 1 MiB record budget.')
            size=encoded+len(rid.encode('utf-8'))+16*len(row)
            if self.pending and (len(self.pending)>=MAX_BATCH_ROWS or self.pending_bytes+size>MAX_BATCH_BYTES):self._flush()
            self.pending.append(tuple(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii') if column['dtype'] in ('audio','tensor') and value is not None else dict(value) if column['dtype']=='image' and value is not None else value for value,column in zip(row,self.columns)));self.pending_ids.append(rid[len(self.identity_prefix):]);self.pending_bytes+=size;self.count+=1
        except sqlite3.IntegrityError as error:
            self.close();raise ValueError('Columnar table row identities must be unique.') from error
        except sqlite3.OperationalError as error:
            self.close()
            if 'full' in str(error).lower():raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.') from error
            raise
        except BaseException:self.close();raise

    def append_frame(self,frame,identities):
        """Admit a bounded native scalar batch without Python payload rows.

        False means an unqualified representation/size, with no state change;
        invalid admitted values poison the store exactly like scalar append.
        """
        if self.closed or self.finished:raise ValueError('Columnar row store is no longer writable.')
        plain={'string','int64','float64','bool'}
        if (type(frame) is not pl.DataFrame or type(identities) is not pl.Series
            or any(c['dtype'] not in plain for c in self.columns)
            or frame.schema!=dict(self.schema) or identities.dtype!=pl.String
            or frame.height>MAX_BATCH_ROWS or frame.estimated_size()+identities.estimated_size()>MAX_BATCH_BYTES):return False
        try:
            self.check_cancelled()
            if frame.height!=len(identities):raise ValueError('Columnar batch identity population differs.')
            if identities.null_count() or (len(identities) and (identities.str.len_bytes().max()>1024
                or identities.str.contains('\x00',literal=True).any() or not identities.str.starts_with(self.identity_prefix).all()
                or identities.str.len_bytes().min()==0)):
                raise ValueError('Columnar batch identities must be bounded and match their prefix.')
            bounds=[]
            for index,column in enumerate(self.columns):
                values=frame.get_column(f'c{index}')
                if not column['nullable'] and values.null_count():raise ValueError('Columnar value violates its declared dtype/nullability.')
                if column['dtype']=='float64' and not values.drop_nulls().is_finite().all():raise ValueError('Columnar value violates its declared dtype/nullability.')
                bounds.append((pl.col(f'c{index}').str.len_bytes().cast(pl.UInt64).fill_null(0)*12+2) if column['dtype']=='string' else pl.lit(32,dtype=pl.UInt64))
            if frame.height and frame.select((pl.sum_horizontal(bounds)+2+2*len(self.columns)).max()).item()>MAX_ROW_BYTES:
                return False # Keep exact scalar admission for conservative large-string bounds.
            self.check_cancelled();self._flush()
            normalized=identities.str.slice(len(self.identity_prefix)).to_list()
            self._write_frame(frame,normalized,self.count);self.count+=frame.height
            self.check_cancelled();return True
        except sqlite3.IntegrityError as error:
            self.close();raise ValueError('Columnar table row identities must be unique.') from error
        except sqlite3.OperationalError as error:
            self.close()
            if 'full' in str(error).lower():raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.') from error
            raise
        except BaseException:self.close();raise

    def _write_chunk(self,rows,ids,start):
        self.check_cancelled()
        frame=pl.DataFrame(rows,schema=self.schema,orient='row',strict=True)
        self._write_frame(frame,ids,start)

    def _write_frame(self,frame,ids,start):
        if not frame.height:return
        self.check_cancelled()
        for index,column in enumerate(frame):self.observed_nulls[index] |= column.null_count()>0
        stream=_IPCBuffer()
        try:
            frame.write_ipc(stream,compression='uncompressed')
            content=stream.getvalue()
        except Exception:
            # The Rust writer translates Python write exceptions into OSError.
            # Only our positively identified byte-ceiling refusal permits retry.
            if not stream.overflow:raise
            if frame.height==1:raise ValueError('A columnar record exceeds the 2 MiB IPC batch budget.')
            middle=frame.height//2
            del stream
            self._write_frame(frame.head(middle),ids[:middle],start)
            self._write_frame(frame.slice(middle),ids[middle:],start+middle)
            return
        if self._arena is not None and self._arena.retain(self,len(self.chunks),frame,content):
            self._allocation_limit()
            self.connection.executemany('INSERT INTO ids VALUES (?,?)',((start+i,rid) for i,rid in enumerate(ids)))
            self._account();self.chunks.append((start,frame.height,None));self.starts.append(start)
            self.check_cancelled();return
        # Reserve exact encoded bytes before touching disk. SQLite's max_page_count
        # then consumes only the remaining shared allowance, including other stores.
        new_reserved=self.reserved+len(content)
        self.budget.resize(self.reserved,new_reserved);self.reserved=new_reserved;self.file_bytes+=len(content)
        self._allocation_limit()
        path=self.directory/f'chunk-{len(self.chunks):06d}.arrow'
        with path.open('xb') as output:output.write(content)
        self.connection.executemany('INSERT INTO ids VALUES (?,?)',((start+i,rid) for i,rid in enumerate(ids)))
        self._account()
        self.chunks.append((start,frame.height,path));self.starts.append(start)
        self.check_cancelled()

    def _spill_native_frame(self,index,content):
        self.check_cancelled()
        if self.closed:raise ValueError('Native spill owner is closed.')
        start,count,path=self.chunks[index]
        if path is not None:raise ValueError('Native buffer already has spill custody.')
        if len(content)>MAX_BATCH_BYTES:raise ValueError('Native spill exceeds its IPC boundary.')
        self.budget.resize(self.reserved,self.reserved+len(content));self.reserved+=len(content);self.file_bytes+=len(content)
        if not self.finished:self._allocation_limit()
        path=self.directory/f'chunk-{index:06d}.arrow'
        with path.open('xb') as output:output.write(content)
        self.chunks[index]=(start,count,path);self._account();self.check_cancelled()

    def _flush(self):
        if not self.pending:return
        self._write_chunk(self.pending,self.pending_ids,self.count-len(self.pending))
        self.pending=[];self.pending_ids=[];self.pending_bytes=0

    def finish(self):
        if self.closed:raise ValueError('Columnar row store is closed.')
        if self.finished:return ColumnarRows(self),ColumnarIDs(self)
        try:
            self.check_cancelled();self._flush();self.connection.commit();self._account()
            self.connection.execute('PRAGMA query_only=ON')
            self._ids_fingerprint=self._index_fingerprint((self.directory/'ids.sqlite3').lstat());self.finished=True
            return ColumnarRows(self),ColumnarIDs(self)
        except sqlite3.IntegrityError as error:
            self.close();raise ValueError('Columnar table row identities must be unique.') from error
        except sqlite3.OperationalError as error:
            self.close()
            if 'full' in str(error).lower():raise ValueError('Disk preparation exceeds its aggregate scratch storage budget.') from error
            raise
        except BaseException:self.close();raise

    def _read_chunk(self,index):
        with self._read_lock:return self._read_chunk_owned(index)

    def _read_chunk_owned(self,index):
        self.check_cancelled()
        if self.closed:raise ValueError('Columnar row store is closed.')
        if self._arena is not None:
            resident=self._arena.read(self,index)
            if resident is not None:return resident
        if self.cached_chunk is not None and self.cached_chunk[0]==index:return self.cached_chunk[1]
        path=self.chunks[index][2]
        if path is None:raise ValueError('Native buffer lost its owned residency.')
        if path.stat().st_size>MAX_BATCH_BYTES:raise ValueError('Columnar IPC chunk exceeds its byte budget.')
        with path.open('rb') as source:
            content=source.read(MAX_BATCH_BYTES+1)
        if len(content)>MAX_BATCH_BYTES:raise ValueError('Columnar IPC chunk exceeds its byte budget.')
        frame=pl.read_ipc(content)
        if self._arena is not None:self._arena.retain(self,index,frame)
        else:self.cached_chunk=(index,frame)
        return frame

    def iter_pairs(self):
        """Scan each bounded row chunk and its contiguous ID range together."""
        for index,(start,count,_) in enumerate(self.chunks):
            frame=self._read_chunk(index)
            cursor=self._reader_cursor('SELECT identity FROM ids WHERE position>=? AND position<? ORDER BY position',(start,start+count))
            try:
                for position,(row,(identity,)) in enumerate(zip(frame.iter_rows(buffer_size=256),cursor)):
                    if position%256==0:self.check_cancelled()
                    yield self.logical_row(row),self.identity_prefix+identity
            finally:cursor.close()

    def logical_row(self,row):
        if not self._json_columns:return row if type(row) is tuple else tuple(row)
        result=list(row)
        for index in self._json_columns:
            if result[index] is not None:result[index]=json.loads(result[index])
        return tuple(result)

    def close(self):
        if getattr(self,'closed',True):return
        self.closed=True
        if getattr(self,'_arena',None) is not None:self._arena.release(self)
        with self._connection_lock:
            for connection in self._read_connections.values():connection.close()
            self._read_connections.clear()
            if self._owner_read_lease is not None:self._owner_read_lease.close();self._owner_read_lease=None
            self._connection=None
        self.pending=[];self.pending_ids=[];self.cached_chunk=None;self.cached_identity_page=None
        self.budget.resize(self.reserved,0);self.reserved=0
        if self.directory is not None:shutil.rmtree(self.directory,ignore_errors=True)

    def __del__(self):self.close()


class _Sequence:
    def __init__(self,store):self.store=store
    def __len__(self):return self.store.count
    def __getitem__(self,index):
        if isinstance(index,slice):
            selected=range(*index.indices(len(self)))
            if len(selected)>MAX_SLICE_ROWS:raise ValueError('Columnar slices exceed bounded materialization budget; iterate rows instead.')
            return tuple(self[i] for i in selected)
        if type(index) is not int:raise TypeError('Columnar indices must be integers or slices.')
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError('Columnar index out of range.')
        self.store.check_cancelled()
        return self._get(index)


class ColumnarRows(_Sequence,StoredRows):
    maximum_encoded_row_bytes=MAX_ROW_BYTES
    def iter_encoded_chunks(self,encoder):
        """Canonical C-encoded row groups with a conservative typed byte bound."""
        first=True
        for index,_ in enumerate(self.store.chunks):
            pending=[self.store._read_chunk(index)]
            while pending:
                self.store.check_cancelled();frame=pending.pop()
                if not frame.height:continue
                # Covers JSON escaping (including supplementary Unicode), scalar
                # punctuation and nested typed media/sequence representations.
                bound=12*frame.estimated_size()+32*frame.height*frame.width
                if frame.height>1 and (frame.height>256 or bound>MAX_ROW_BYTES):
                    middle=frame.height//2
                    pending.extend((frame.slice(middle),frame.head(middle)));continue
                rows=frame.rows()
                if self.store._json_columns:rows=[self.store.logical_row(row) for row in rows]
                encoded=encoder.encode(rows)
                if len(encoded)>MAX_BATCH_BYTES:raise ValueError('Canonical row group exceeds its bounded allocation.')
                if not first:yield ','
                yield encoded[1:-1];first=False
    def pair_at(self,index):
        """Read a row and its identity under one bounded chunk admission.

        This is for consumers that need both values. It avoids independent
        row/identity lookups without weakening the live check in _read_chunk.
        """
        if type(index) is not int or not 0<=index<len(self):raise IndexError('Columnar pair index out of range.')
        chunk=bisect_right(self.store.starts,index)-1
        frame=self.store._read_chunk(chunk)
        identity=self.store._identity_at(index)
        return self.store.logical_row(frame.row(index-self.store.starts[chunk])),identity

    def iter_selected_pairs(self,indices):
        """Checked bounded gather without per-match authority walks or SQL.

        Only the current immutable frame and one identity page are retained. Rows
        stream individually; checks occur on every frame change,256 rows or2MiB
        of processed JSON/identity work, whichever comes first.
        """
        self.store.check_cancelled()
        if self.store.closed:raise ValueError('Columnar row store is closed.')
        starts=self.store.starts;chunk=None;frame=None;count=0;processed=0
        for index in indices:
            if type(index) is not int or not 0<=index<len(self):raise IndexError('Columnar selection index out of range.')
            selected=bisect_right(starts,index)-1
            if selected!=chunk:
                frame=self.store._read_chunk(selected);chunk=selected;count=0;processed=0
            if self.store.closed:raise ValueError('Columnar row store is closed.')
            row=self.store.logical_row(frame.row(index-starts[chunk]));identity=self.store._identity_at(index)
            size=len(json.dumps(row,ensure_ascii=True,allow_nan=False,separators=(',',':')))+6*len(identity.encode('utf-8'))
            if count>=256 or processed+size>MAX_BATCH_BYTES:
                self.store.check_cancelled();count=0;processed=0
            count+=1;processed+=size
            yield row,identity

    def _get(self,index):
        chunk=bisect_right(self.store.starts,index)-1
        return self.store.logical_row(self.store._read_chunk(chunk).row(index-self.store.starts[chunk]))
    def __iter__(self):
        for index,(_,_,_) in enumerate(self.store.chunks):
            frame=self.store._read_chunk(index)
            for position,row in enumerate(frame.iter_rows(buffer_size=256)):
                if position%256==0:self.store.check_cancelled()
                yield self.store.logical_row(row)
    def iter_pairs(self):return self.store.iter_pairs()


class ColumnarIDs(_Sequence,StoredIDs):
    maximum_encoded_identity_bytes=6144
    def _get(self,index):
        return self.store._identity_at(index)
    def _iterate(self,order):
        self.store.check_cancelled()
        cursor=self.store._reader_cursor('SELECT identity FROM ids ORDER BY '+order)
        try:
            for i,(value,) in enumerate(cursor):
                if i%256==0:self.store.check_cancelled()
                yield self.store.identity_prefix+value
        finally:cursor.close()
    def __iter__(self):return self._iterate('position')
    def sorted_iter(self):return self._iterate('identity COLLATE BINARY')
    def contains(self,identity):
        if type(identity) is not str or not identity.startswith(self.store.identity_prefix):return False
        self.store.check_cancelled()
        return self.store._reader_cursor('SELECT 1 FROM ids WHERE identity=?',(identity[len(self.store.identity_prefix):],)).fetchone() is not None
    def __contains__(self,identity):return self.contains(identity)
