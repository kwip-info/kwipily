"""Native relational buffers for the existing table/identity contract.

Only owned, admitted scalar columns qualify. Payloads stay in native frames;
Python coordinates bounded identity/position maps. Legacy readers and algorithms
remain the fallback for unqualified representations and comparison semantics.
"""
from bisect import bisect_right
import polars as pl
if __package__:
    from . import dataflow as df
    from .dataflow_columnar import ColumnarStore,ColumnarRows,ColumnarIDs,DTYPES
    from .dataflow_shared import ColumnProjectionStore,supports_shared_table
else:
    import dataflow as df
    from dataflow_columnar import ColumnarStore,ColumnarRows,ColumnarIDs,DTYPES
    from dataflow_shared import ColumnProjectionStore,supports_shared_table

if __package__:from .native_frame_arena import resident_view,release_view
else:from native_frame_arena import resident_view,release_view

SCALARS=frozenset(('string','int64','float64','bool','date','datetime','time'))


def qualified(table):
    return supports_shared_table(table) and all(c['dtype'] in SCALARS for c in table.columns)


def identities(parent,start,count):
    cursor=parent._reader_cursor('SELECT identity FROM ids WHERE position>=? AND position<? ORDER BY position',(start,start+count))
    try:values=[parent.identity_prefix+identity for (identity,) in cursor]
    finally:cursor.close()
    if len(values)!=count:raise ValueError('Native batch identity population differs.')
    result=pl.Series('identity',values,dtype=pl.String)
    if count and result.str.len_chars().max()>200:raise ValueError('Input requires unique bounded observation identities.')
    return result


def check_parents(parents,check):
    # Share one operation guard within this boundary, never across boundaries.
    # Still inspect every lifetime, including additional computed/map owners.
    pending=list(parents);seen=set();guards=[]
    while pending:
        store=pending.pop()
        if id(store) in seen:continue
        seen.add(id(store))
        if store.closed:raise ValueError('Shared native parent is closed.')
        if isinstance(store,ColumnProjectionStore):
            guard=store._check
            pending.extend(getattr(store,'parents',(store.parent,)))
            for name in ('mapping','computed'):
                extra=getattr(store,name,None)
                if extra is not None:pending.append(extra)
        else:guard=store.check_cancelled
        if guard not in guards:guards.append(guard)
    if check not in guards:guards.append(check)
    for guard in guards:guard()


class NativeGatherStore(ColumnProjectionStore):
    def __init__(self,parent,mapping,check):
        self._arena=getattr(parent,'_arena',None)
        self.parent=parent;self.mapping=mapping;self._check=check
        self.closed=False;self.finished=True;self.columns=parent.columns
        self.count=mapping.count;self.identity_prefix=mapping.identity_prefix
        self._json_columns=();self.observed_nulls=parent.observed_nulls
        if getattr(parent,'membership_sha256',None) is not None:self.membership_sha256=parent.membership_sha256
    def check_cancelled(self):
        if self.closed:raise ValueError('Shared native order is closed.')
        check_parents((self.parent,self.mapping),self._check)
    @property
    def starts(self):self.check_cancelled();return self.mapping.starts
    @property
    def chunks(self):self.check_cancelled();return self.mapping.chunks
    def _reader_cursor(self,query,parameters=()):
        if self.closed:raise ValueError('Shared native order is closed.')
        return self.mapping._reader_cursor(query,parameters)
    def _identity_at(self,index):
        if self.closed:raise ValueError('Shared native order is closed.')
        return self.mapping._identity_at(index)
    @resident_view
    def _read_chunk(self,index):
        self.check_cancelled();positions=self.mapping._read_chunk(index).get_column('c0').to_list()
        groups={};starts=self.parent.starts
        for output,position in enumerate(positions):
            chunk=bisect_right(starts,position)-1
            if chunk<0 or position>=self.parent.count:raise ValueError('Native order position exceeds parent.')
            local,order=groups.setdefault(chunk,([],[]));local.append(position-starts[chunk]);order.append(output)
        frames=[]
        for chunk,(local,order) in groups.items():
            self.check_cancelled()
            frames.append(self.parent._read_chunk(chunk)[local].with_columns(pl.Series('_order',order)))
        if not frames:return pl.DataFrame(schema=[(f'c{i}',DTYPES[c['dtype']]) for i,c in enumerate(self.columns)])
        return pl.concat(frames).sort('_order').drop('_order')
    def close(self):
        if self.closed:return
        release_view(self);self.closed=True;self.mapping.close();self.parent=None
    def __del__(self):self.close()


class NativeUnionStore(NativeGatherStore):
    def __init__(self,parents,mapping,chunks,columns,check):
        super().__init__(parents[0],mapping,check)
        self.parents=tuple(parents);self.sources=tuple(chunks);self.columns=tuple(columns)
        self.observed_nulls=tuple(c['nullable'] for c in columns)
        if hasattr(self,'membership_sha256'):del self.membership_sha256
    def check_cancelled(self):
        if self.closed:raise ValueError('Shared native union is closed.')
        check_parents((*self.parents,self.mapping),self._check)
    @resident_view
    def _read_chunk(self,index):
        self.check_cancelled();parent,chunk,expressions,offset,count=self.sources[index]
        return self.parents[parent]._read_chunk(chunk).slice(offset,count).select(expressions)
    def close(self):
        if self.closed:return
        release_view(self);self.closed=True;self.mapping.close();self.parents=();self.parent=None


def union_table(args,c,columns,mappings,source,history,check):
    tables=[args[port] for port in c['inputs']]
    if not all(qualified(t) for t in tables):return None
    if any(index is not None and columns[i]['dtype']!=table.columns[index]['dtype'] for table,selected in zip(tables,mappings) for i,index in enumerate(selected)):return None
    parents=[t.rows.store for t in tables]
    mapping=ColumnarStore(({'name':'position','dtype':'int64','nullable':False},),check,budget=parents[0].budget)
    sources=[]
    try:
        for parent_index,(parent,selected,port) in enumerate(zip(parents,mappings,c['inputs'])):
            expressions=[(pl.col(f'c{index}') if index is not None else pl.lit(None,dtype=DTYPES[columns[i]['dtype']])).alias(f'c{i}') for i,index in enumerate(selected)]
            if c['sourceColumn']:expressions.append(pl.lit(port,dtype=pl.String).alias(f'c{len(columns)-1}'))
            for chunk,(start,count,_) in enumerate(parent.chunks):
                check_parents((parent,),check)
                ids=identities(parent,start,count)
                # Only bounded identity metadata crosses Python; never payloads.
                output_ids=pl.Series('identity',['union:'+df.digest(['kwip.union.row.v1',port,rid]) for rid in ids],dtype=pl.String)
                before=len(mapping.chunks);output_start=mapping.count
                if not mapping.append_frame(pl.int_range(0,count,eager=True,dtype=pl.Int64).rename('c0').to_frame(),output_ids):raise ValueError('Native union mapping exceeds its batch boundary.')
                mapping._flush();sources.extend((parent_index,chunk,expressions,start-output_start,count) for start,count,_ in mapping.chunks[before:])
        mapping.finish();store=NativeUnionStore(parents,mapping,sources,columns,check)
        return df.derive(df.Table(tuple(columns),ColumnarRows(store),ColumnarIDs(store),source,diagnostics=history))
    except BaseException:mapping.close();raise


def sort_table(table,c,check):
    if not qualified(table) or any(k['comparison']!='exact' for k in c['keys']):return None
    selected=[table.index(k['column']) for k in c['keys']];parent=table.rows.store
    if len(selected)>126:return None
    columns=[{**table.columns[i],'name':f'key{j}'} for j,i in enumerate(selected)]
    columns.extend(({'name':'position','dtype':'int64','nullable':False},{'name':'identity','dtype':'string','nullable':False}))
    keys=ColumnarStore(columns,check,budget=parent.budget,residency=False)
    mapping=ColumnarStore(({'name':'position','dtype':'int64','nullable':False},),check,budget=parent.budget)
    try:
        for chunk,(start,count,_) in enumerate(parent.chunks):
            check_parents((parent,),check);frame=parent._read_chunk(chunk);ids=identities(parent,start,count)
            data=frame.select([pl.col(f'c{i}').alias(f'c{j}') for j,i in enumerate(selected)])
            data=data.with_columns(pl.int_range(start,start+count,eager=True,dtype=pl.Int64).alias(f'c{len(selected)}'),ids.alias(f'c{len(selected)+1}'))
            if not keys.append_frame(data,ids):keys.close();mapping.close();return None
        keys.finish()
        paths=[str(path) for _,_,path in keys.chunks]
        if paths:
            query=pl.scan_ipc(paths).sort([f'c{i}' for i in range(len(selected))],descending=[k['direction']=='descending' for k in c['keys']],nulls_last=[k['missing']=='last' for k in c['keys']],maintain_order=True)
            batches=query.collect_batches(chunk_size=4096,maintain_order=True,engine='in-memory')
            try:
                for frame in batches:
                    check_parents((parent,),check)
                    if not mapping.append_frame(frame.select(pl.col(f'c{len(selected)}').alias('c0')),frame.get_column(f'c{len(selected)+1}')):raise ValueError('Native order map exceeds its batch boundary.')
            finally:del batches
        mapping.finish();store=NativeGatherStore(parent,mapping,check)
        diagnostic=dict(kind='sort',keys=c['keys'],inputRows=len(table.rows),outputRows=len(table.rows))
        return df.derive(table,rows=ColumnarRows(store),ids=ColumnarIDs(store),diagnostics=table.diagnostics+(diagnostic,))
    except BaseException:mapping.close();raise
    finally:keys.close()


def key_rows(table,indices):
    if not qualified(table):return None
    def values():
        parent=table.rows.store
        for chunk,_ in enumerate(parent.chunks):
            parent.check_cancelled()
            yield from parent._read_chunk(chunk).select([pl.col(f'c{i}') for i in indices]).iter_rows(buffer_size=256)
    return values()


def validate_owned(table):
    """Reuse producer type/uniqueness admission, retaining live owner/ID checks."""
    if not qualified(table):return False
    parent=table.rows.store
    for start,count,_ in parent.chunks:
        parent.check_cancelled();values=identities(parent,start,count)
        if values.len() and values.str.len_chars().max()>200:raise ValueError('Input requires unique bounded observation identities.')
    return True


def row_size_bounds(table,indices):
    """Conservative ASCII-escaped row sizes without Python payload conversion.

    Strings/integers are exact; float/bool scalar bounds may overestimate. The
    caller uses this only as a safe admission proof and falls back to the exact
    reference preflight when the conservative total reaches the artifact ceiling.
    """
    if not qualified(table):return None
    result=[];parent=table.rows.store
    for chunk,(_,count,_) in enumerate(parent.chunks):
        parent.check_cancelled();frame=parent._read_chunk(chunk);terms=[]
        for i in indices:
            col=pl.col(f'c{i}');dtype=table.columns[i]['dtype']
            if dtype=='string':
                size=(col.str.len_chars().cast(pl.UInt64)+2+
                    col.str.count_matches(r'["\\\x08\x09\x0a\x0c\x0d]').cast(pl.UInt64)+
                    col.str.count_matches(r'[\x00-\x07\x0b\x0e-\x1f]').cast(pl.UInt64)*5+
                    col.str.count_matches(r'[^\x00-\x7f]').cast(pl.UInt64)*5+
                    col.str.count_matches(r'[\x{10000}-\x{10ffff}]').cast(pl.UInt64)*6)
            elif dtype in ('int64','date','datetime','time'):size=col.cast(pl.String).str.len_chars().cast(pl.UInt64)
            else:size=pl.when(col.is_not_null()).then(pl.lit(32,dtype=pl.UInt64)).otherwise(None)
            terms.append(size.fill_null(4))
        if terms:result.extend(frame.select((pl.sum_horizontal(terms)+2+max(0,len(indices)-1)).alias('bytes')).get_column('bytes').to_list())
        else:result.extend([2]*count)
    return result


def _gather(parent,positions,columns):
    groups={};absent=[];starts=parent.starts
    for output,position in enumerate(positions):
        if position is None:absent.append(output);continue
        chunk=bisect_right(starts,position)-1
        if chunk<0 or position>=parent.count:raise ValueError('Native join position exceeds its parent.')
        local,order=groups.setdefault(chunk,([],[]));local.append(position-starts[chunk]);order.append(output)
    frames=[]
    for chunk,(local,order) in groups.items():
        parent.check_cancelled();frames.append(parent._read_chunk(chunk)[local].with_columns(pl.Series('_order',order,dtype=pl.Int64)))
    if absent:
        frame=pl.select([pl.repeat(None,len(absent),dtype=DTYPES[c['dtype']]).alias(f'c{i}') for i,c in enumerate(columns)])
        frames.append(frame.with_columns(pl.Series('_order',absent,dtype=pl.Int64)))
    return pl.concat(frames).sort('_order').drop('_order')


class NativeJoinStore(NativeUnionStore):
    def __init__(self,parents,mapping,columns,projection,check):
        super().__init__(parents,mapping,(),columns,check)
        self.projection=tuple(projection)
    @resident_view
    def _read_chunk(self,index):
        self.check_cancelled();mapping=self.mapping._read_chunk(index)
        a,b=self.parents
        left=_gather(a,mapping.get_column('c0').to_list(),a.columns)
        right=_gather(b,mapping.get_column('c1').to_list(),b.columns)
        return pl.DataFrame([(left.get_column(f'c{i}') if i<len(a.columns) else right.get_column(f'c{i-len(a.columns)}')).alias(f'c{target}') for target,i in enumerate(self.projection)])


def join_table(plan,check):
    if not qualified(plan.left) or not qualified(plan.right):return None
    parents=(plan.left.rows.store,plan.right.rows.store)
    columns=({'name':'left','dtype':'int64','nullable':True},{'name':'right','dtype':'int64','nullable':True})
    mapping=ColumnarStore(columns,check,budget=parents[0].budget)
    pending=[];ids=[]
    def flush():
        if not pending:return
        check_parents(parents,check)
        frame=pl.DataFrame(pending,schema=[('c0',pl.Int64),('c1',pl.Int64)],orient='row')
        if not mapping.append_frame(frame,pl.Series('identity',ids,dtype=pl.String)):raise ValueError('Native join map exceeds its batch boundary.')
        pending.clear();ids.clear()
    try:
        for a,b in plan.pairs():
            # Indexed identity metadata is bounded; no payload row is fetched.
            if len(pending)%128==0:check_parents(parents,check)
            left=parents[0]._identity_at(a) if a is not None else None
            right=parents[1]._identity_at(b) if b is not None else None
            pending.append((a,b));ids.append('join:'+df.digest(['kwip.join.row.v1',left,right]))
            if len(pending)>=4096:flush()
        flush();mapping.finish();store=NativeJoinStore(parents,mapping,plan.columns,plan.projection,check)
        return df.derive(df.Table(plan.columns,ColumnarRows(store),ColumnarIDs(store),plan.source_digest,diagnostics=plan.diagnostics))
    except BaseException:mapping.close();raise
