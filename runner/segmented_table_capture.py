"""Column-oriented retained table snapshots over immutable live segments.

Only the logical table passed by the capture owner is persisted: a required-only
selection never archives hidden parent columns or rejected parent rows. Equal
bounded column chunks can be reused by later explicitly requested snapshots.
"""
import json,re,sys
if __package__:
    from . import dataflow as df
    from .packed_ordered_index import IndexBuilder,OrderedIndex
    from .table_artifact_pack import encode_object
    from .table_storage import StoredRows,StoredIDs
    from .dataflow_disk import DiskStore
    from .workload_policy import current_policy
    from .execution_checkpoints import RowCheckpoints
else:
    import dataflow as df
    from packed_ordered_index import IndexBuilder,OrderedIndex
    from table_artifact_pack import encode_object
    from table_storage import StoredRows,StoredIDs
    from dataflow_disk import DiskStore
    from workload_policy import current_policy
    from execution_checkpoints import RowCheckpoints


class _RelativePack:
    def __init__(self,writer):self.writer=writer
    def add_json(self,value):return self.writer.relative(self.writer.add_json(value))


def _small_scalar_bound(row,rid):
    maximum=8+12*len(rid)+2*len(row)
    for value in row:
        if type(value) is str:maximum+=12*len(value)+2
        elif type(value) is int:maximum+=max(1,abs(value).bit_length()*30103//100000+2)
        elif type(value) in (float,bool,type(None)):maximum+=32
        else:return None
        if maximum>8192:return None
    return maximum if maximum<=8192 else None


def _lineage(table):
    return {'role':table.role,'sourceDigest':table.source_digest,'membershipDigest':df.membership_digest(table.row_ids),
            'dataVersion':table.data_version,'partitionDigest':table.partition_digest,'fittedStateDigests':list(table.fitted_states),'diagnostics':list(table.diagnostics)}


class _SnapshotIndexReader:
    """Bounded reuse of already range-verified immutable index metadata.

    Private to one snapshot-authoring scope. No cells, access grants or live
    authority decisions are cached. Cache misses use the original guarded reader;
    the author checks bounded work and final publication independently.
    """
    def __init__(self,resolver,owner=None,state=None):
        import collections
        self.resolver=resolver;self.owner=owner
        self.state=state if state is not None else [collections.OrderedDict(),0]
    def read_node(self,reference):
        if self.resolver.closed:raise ValueError('Snapshot index reader is closed.')
        absolute={'pack':reference['pack'] if reference['pack'] is not None else self.owner,'object':reference['object']}
        key=json.dumps(absolute,sort_keys=True,separators=(',',':'))
        cache=self.state[0];cached=cache.pop(key,None)
        if cached is None:
            value,context=self.resolver._read(reference,self.owner)
            owner=context.owner
            if type(value) is dict and value.get('kind')=='orderedIndex':
                cost=len(json.dumps(value,sort_keys=True,separators=(',',':')).encode())+len(key.encode())
                if cost<=2_097_152:
                    while cache and (len(cache)>=128 or self.state[1]+cost>2_097_152):
                        _,old=cache.popitem(last=False);self.state[1]-=old[2]
                    cache[key]=(value,owner,cost);self.state[1]+=cost
        else:
            value,owner,cost=cached;cache[key]=cached
        return value,_SnapshotIndexReader(self.resolver,owner,self.state)


def _shared_selection(writer,table):
    shared=sys.modules.get((__package__+'.' if __package__ else '')+'dataflow_shared')
    if shared is None or not shared.supports_shared_table(table):return None
    store=table.rows.store
    if type(store) is not shared.RowSelectionStore:return None
    prior=writer.captured_tables.get(store.parent)
    root=writer.storage._path
    if prior is None or root is None:return None
    metadata,owner=prior;depth=metadata.get('selectionDepth',0)+1
    if depth>32:return None # Bound retained reference chains; materialize this boundary.
    if metadata['columns']!=list(table.columns):raise ValueError('Selection snapshot columns differ from its parent.')
    if __package__:from .segmented_table_store import SegmentResolver
    else:from segmented_table_store import SegmentResolver
    def rooted(index):
        reference=dict(index['reference'])
        if reference['pack'] is None:reference['pack']=dict(owner['pack'])
        return {**index,'reference':reference}
    pack=_RelativePack(writer)
    builders=[IndexBuilder(pack) for _ in range(len(table.columns)+1)]
    with SegmentResolver(root,maximum_bytes=writer.maximum,check_cancelled=writer.storage.check_cancelled) as resolver:
        index_reader=_SnapshotIndexReader(resolver)
        indexes=[OrderedIndex(index_reader,rooted(index),maximum_count=metadata['rowCount']) for index in [*metadata['columnIndexes'],metadata['identityIndex']]]
        states=[None for _ in indexes];pending=[[] for _ in indexes];seen=0
        checkpoints=RowCheckpoints(writer.storage.check_cancelled);work=(None,)*len(indexes)
        def flush(column):
            if not pending[column]:return
            start,end,reference,span=states[column]
            value={'schemaVersion':1,'kind':'selectedColumn','source':reference,'sourceCount':span,'positions':pending[column]}
            builders[column].append(len(pending[column]),pack.add_json(value));pending[column]=[]
        for chunk,parent_chunk in enumerate(store.parent_chunks):
            writer.storage.check_cancelled()
            start=store.parent.starts[parent_chunk]
            for local in store.mapping._read_chunk(chunk).get_column('c0').to_list():
                checkpoints.admit(work)
                if type(local) is not int or not 0<=local<store.parent.chunks[parent_chunk][1]:raise ValueError('Selection position differs from its parent batch.')
                position=start+local
                for column,index in enumerate(indexes):
                    state=states[column]
                    if state is None or not state[0]<=position<state[1]:
                        flush(column)
                        reference,offset,context,span=index.locate_span(position)
                        reference=dict(reference)
                        if reference['pack'] is None:reference['pack']=dict(context.owner)
                        states[column]=(position-offset,position-offset+span,reference,span)
                    if len(pending[column])==256:flush(column)
                    pending[column].append(position-states[column][0])
                seen+=1
        checkpoints.finish()
        for column in range(len(indexes)):flush(column)
        if seen!=len(table.rows):raise ValueError('Selection snapshot population changed.')
    return {'schemaVersion':2,'kind':'columnTable','selectionDepth':depth,'columns':list(table.columns),'rowCount':seen,
            'columnIndexes':[b.finish() for b in builders[:-1]],'identityIndex':builders[-1].finish(),'lineage':_lineage(table)}


def _shared_projection(writer,table):
    # A portable NumPy-only bundle must not acquire a Polars dependency merely
    # to decode/capture a disk table. Shared IPC stores have loaded this module.
    shared=sys.modules.get((__package__+'.' if __package__ else '')+'dataflow_shared')
    if shared is None or not shared.supports_shared_table(table):return None
    store=table.rows.store
    if type(store) not in (shared.ColumnProjectionStore, shared.ComputedColumnsStore):return None
    prior=writer.captured_tables.get(store.parent)
    if prior is None:return None # Never retain an unrequested parent population.
    metadata,owner=prior
    if metadata['rowCount']!=len(table.rows):raise ValueError('Captured projection population differs from its parent.')
    def rooted(index,context):
        reference=dict(index['reference'])
        if reference['pack'] is None:reference['pack']=dict(context['pack'])
        return {**index,'reference':reference}
    if type(store) is shared.ColumnProjectionStore:
        indexes=[rooted(metadata['columnIndexes'][index],owner) for index in store.indices]
    else:
        # Only changed fields are captured. The parent is reusable solely because
        # the owner already explicitly requested and sealed that population.
        from_columns=store.computed.columns
        from_rows=shared.ColumnarRows(store.computed);from_ids=shared.ColumnarIDs(store.computed)
        version=df.digest({'columns':from_columns,'rows':from_rows,'ids':from_ids})
        changes=df.Table(from_columns,from_rows,from_ids,table.source_digest,table.role,version,
                         table.partition_digest,table.fitted_states,table.diagnostics)
        capture_table(writer,changes)
        computed,computed_owner=writer.captured_tables[store.computed]
        indexes=[rooted((metadata if source=='parent' else computed)['columnIndexes'][index],
                         owner if source=='parent' else computed_owner) for source,index in store.bindings]
    depth=max(metadata.get('selectionDepth',0),computed.get('selectionDepth',0) if type(store) is shared.ComputedColumnsStore else 0)
    return {'schemaVersion':2 if depth else 1,'kind':'columnTable','columns':list(table.columns),'rowCount':len(table.rows),
            'columnIndexes':indexes,'identityIndex':rooted(metadata['identityIndex'],owner),'lineage':_lineage(table),
            **({'selectionDepth':depth} if depth else {})}


def _publish_table(writer,table,metadata):
    result=writer.add_json(metadata);writer.seal();reference=result.reference()
    store=getattr(table.rows,'store',None)
    if store is not None and store is getattr(table.row_ids,'store',None):
        writer.captured_tables[store]=(metadata,reference)
    return reference


def capture_table(writer,table):
    if writer.closed or writer.failed or writer.index is None:
        raise ValueError('Retained snapshot requires an active unpoisoned segment writer.')
    check=writer.storage.check_cancelled;policy=current_policy()
    width=len(table.columns);count=len(table.rows)
    if not 1<=width<=policy.limit('tableColumns') or count>policy.limit('tableRows') or count*width>policy.limit('tableCells'):
        raise ValueError('Retained table exceeds the captured workload row/cell/column allowance.')
    if len(table.row_ids)!=count:raise ValueError('Retained table identities differ from its population.')
    # Repeated consumers (for example an Output followed by inspection) can
    # publish the same explicitly requested immutable snapshot. Deduplicating
    # only after encoding every column still scans/serializes the population.
    # Reuse its sealed reference instead, with fresh live ownership/lifetime
    # checks. A changed schema or scientific lineage must not inherit this pin.
    store=getattr(table.rows,'store',None)
    if store is not None and store is getattr(table.row_ids,'store',None):
        check();store.check_cancelled()
        prior=writer.captured_tables.get(store)
        if prior is not None and store.finished and not store.closed:
            metadata,reference=prior
            if (metadata['rowCount']==count and metadata['columns']==list(table.columns)
                and metadata['lineage']==_lineage(table)):
                return reference
    selected=_shared_selection(writer,table)
    if selected is not None:return _publish_table(writer,table,selected)
    shared=_shared_projection(writer,table)
    if shared is not None:return _publish_table(writer,table,shared)
    pack=_RelativePack(writer);columns=[IndexBuilder(pack) for _ in table.columns];identities=IndexBuilder(pack)
    vectors=[[] for _ in table.columns];ids=[];size=0;seen=0;admitted_work=0
    check()
    def flush():
        nonlocal vectors,ids,size
        if not ids:return
        for builder,values in zip(columns,vectors):builder.append(len(ids),pack.add_json(values))
        identities.append(len(ids),pack.add_json(ids))
        vectors=[[] for _ in table.columns];ids=[];size=0
    for row,rid in df.row_pairs(table):
        if len(row)!=width or type(rid) is not str:raise ValueError('Invalid retained table row/identity.')
        upper=_small_scalar_bound(row,rid)
        if seen%256==0 or upper is None or admitted_work+(upper or 0)>2_097_152:
            check();admitted_work=0
        # Encoding is bounded before buffering; account escaped bytes rather than
        # assuming strings, tensor references or UTF-8 have fixed expansion.
        # Small scalar rows already have a conservative escaped-size proof.
        # Charge that upper bound to the buffer; canonical JSON is generated
        # once for the published column chunks, not once again for each row.
        encoded=upper if upper is not None else len(encode_object([row,rid],check))
        admitted_work+=upper or encoded
        if ids and (len(ids)>=256 or size+encoded>1_048_576):flush()
        for vector,value in zip(vectors,row):vector.append(value)
        ids.append(rid);size+=encoded;seen+=1
    flush();check()
    if seen!=count:raise ValueError('Retained table population changed during capture.')
    lineage=_lineage(table)
    metadata={'schemaVersion':1,'kind':'columnTable','columns':list(table.columns),'rowCount':count,
              'columnIndexes':[builder.finish() for builder in columns],'identityIndex':identities.finish(),'lineage':lineage}
    return _publish_table(writer,table,metadata)


class _CapturedRows(StoredRows):
    def __init__(self,store):self.store=store
    def __len__(self):return self.store.count
    def __getitem__(self,index):
        if isinstance(index,slice):return tuple(self[i] for i in range(*index.indices(len(self))))
        if index<0:index+=len(self)
        return self.store.row(index)
    def __iter__(self):
        yield from self.store.iter_rows()


class _CapturedIDs(StoredIDs):
    def __init__(self,store):self.store=store
    def __len__(self):return self.store.count
    def __getitem__(self,index):return self.store.ids[index]
    def __iter__(self):return iter(self.store.ids)
    def sorted_iter(self):return self.store.ids.sorted_iter()
    def contains(self,identity):return self.store.ids.contains(identity)
    def __contains__(self,identity):return self.contains(identity)


class _CaptureStore:
    def __init__(self,reader,metadata,check):
        self.reader=reader;self.count=metadata['rowCount'];self.check=check;self.closed=False;self.finished=False;self.cache=None;self.identity_cache=None
        self.selection_depth=metadata.get('selectionDepth',0)
        self.columns=metadata['columns'];self.indexes=[OrderedIndex(reader,index,maximum_count=self.count) for index in metadata['columnIndexes']]
        self.identity_index=OrderedIndex(reader,metadata['identityIndex'],maximum_count=self.count)
        if any(index.count!=self.count for index in [*self.indexes,self.identity_index]):raise ValueError('Retained column populations differ.')
        self.id_store=DiskStore(check);self.ids=None;self.identity_prefix='';self.budget=self.id_store.budget
    def check_cancelled(self):
        self.check()
        if self.closed:raise ValueError('Retained table is closed.')
    def _values(self,index,position):
        reference,offset,owner,span=index.locate_span(position)
        def decode(context,reference,depth):
            value,children=context.read_node(reference)
            if type(value) is list:
                if len(value)>256:raise ValueError('Invalid retained column chunk geometry.')
                return value
            if (depth>=self.selection_depth or type(value) is not dict
                or set(value)!={'schemaVersion','kind','source','sourceCount','positions'}
                or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or value['kind']!='selectedColumn'
                or type(value['sourceCount']) is not int or not 1<=value['sourceCount']<=256
                or type(value['positions']) is not list or not 1<=len(value['positions'])<=256
                or any(type(i) is not int or not 0<=i<value['sourceCount'] for i in value['positions'])
                or len(set(value['positions']))!=len(value['positions'])):
                raise ValueError('Invalid retained selection geometry.')
            source=decode(children,value['source'],depth+1)
            if len(source)!=value['sourceCount']:raise ValueError('Retained selection source span differs.')
            return [source[i] for i in value['positions']]
        values=decode(owner,reference,0)
        if type(values) is not list or not 0<=offset<len(values) or len(values)!=span or len(values)>256:raise ValueError('Invalid retained column chunk geometry.')
        return values,offset
    def row(self,position):
        self.check()
        return self._row(position)
    def iter_rows(self):
        self.check_cancelled()
        checkpoints=RowCheckpoints(self.check_cancelled)
        for position in range(self.count):
            row=self._row(position)
            checkpoints.admit(row)
            yield row
        checkpoints.finish()
    def _row(self,position):
        if self.closed:raise ValueError('Retained table is closed.')
        if type(position) is not int or not 0<=position<self.count:raise IndexError('Retained row is outside its population.')
        if self.cache is None or not self.cache[0]<=position<self.cache[1]:
            chunks=[];start=0;end=self.count;buffer_bytes=0
            for index in self.indexes:
                values,offset=self._values(index,position)
                local=position-offset
                # Columns can reuse independent sealed parents/chunk sizes.
                # Cache only their common span without changing logical order.
                start=max(start,local);end=min(end,local+len(values))
                buffer_bytes+=len(encode_object(values,self.check))
                if buffer_bytes>2_097_152:raise ValueError('Retained column batch exceeds its aggregate decoder boundary.')
                chunks.append((local,values))
            if not start<=position<end:raise ValueError('Retained column spans do not cover the requested row.')
            self.cache=(start,end,[values[start-local:end-local] for local,values in chunks])
        return tuple(values[position-self.cache[0]] for values in self.cache[2])
    def close(self):
        if not self.closed:
            self.closed=True;self.cache=None;self.id_store.close()
            if hasattr(self,'owned_resolver'):self.owned_resolver.close()
    def __del__(self):
        if hasattr(self,'id_store'):self.close()


def _read_metadata(resolver,reference):
    metadata,reader=resolver.read_node(reference);policy=current_policy()
    fields={'schemaVersion','kind','columns','rowCount','columnIndexes','identityIndex','lineage'}
    if (type(metadata) is not dict or type(metadata.get('schemaVersion')) is not int or metadata['schemaVersion'] not in (1,2)
        or set(metadata)!=fields|({'selectionDepth'} if metadata['schemaVersion']==2 else set())
        or metadata['schemaVersion']==2 and (type(metadata['selectionDepth']) is not int or not 1<=metadata['selectionDepth']<=32)
        or metadata['kind']!='columnTable'
        or type(metadata['rowCount']) is not int or not 0<=metadata['rowCount']<=policy.limit('tableRows')
        or type(metadata['columns']) is not list or not 1<=len(metadata['columns'])<=policy.limit('tableColumns')
        or metadata['rowCount']*len(metadata['columns'])>policy.limit('tableCells')
        or type(metadata['columnIndexes']) is not list or len(metadata['columnIndexes'])!=len(metadata['columns'])):
        raise ValueError('Invalid retained table metadata or workload bounds.')
    df._fitted_module().schema_valid(metadata['columns'])
    lineage=metadata['lineage']
    if type(lineage) is not dict or set(lineage)!={'role','sourceDigest','membershipDigest','dataVersion','partitionDigest','fittedStateDigests','diagnostics'}:
        raise ValueError('Invalid retained table lineage.')
    if (lineage['role'] not in ('unassigned','training','validation','test')
        or any(type(lineage[name]) is not str for name in ('sourceDigest','membershipDigest','dataVersion','partitionDigest'))
        or type(lineage['fittedStateDigests']) is not list or len(lineage['fittedStateDigests'])>policy.limit('planNodes')
        or any(type(pin) is not str or re.fullmatch('[0-9a-f]{64}',pin) is None for pin in lineage['fittedStateDigests'])
        or type(lineage['diagnostics']) is not list or any(type(item) is not dict for item in lineage['diagnostics'])):
        raise ValueError('Invalid retained table lineage.')
    return metadata,reader


def read_capture(resolver,reference,*,check_cancelled=lambda:None):
    metadata,reader=_read_metadata(resolver,reference)
    lineage=metadata["lineage"]
    store=_CaptureStore(reader,metadata,check_cancelled)
    try:
        cached_ids=None
        for position in range(store.count):
            if position%256==0:check_cancelled()
            if cached_ids is None or not cached_ids[0]<=position<cached_ids[1]:
                values,offset=store._values(store.identity_index,position)
                cached_ids=(position-offset,position-offset+len(values),values)
            rid=cached_ids[2][position-cached_ids[0]]
            if type(rid) is not str or not 1<=len(rid.encode('utf-8'))<=200 or '\0' in rid:raise ValueError('Invalid retained observation identity.')
            store.id_store.append((),rid)
        _,store.ids=store.id_store.finish();store.finished=True
        rows=_CapturedRows(store);ids=_CapturedIDs(store)
        if df.membership_digest(ids)!=lineage['membershipDigest']:raise ValueError('Retained membership pin differs.')
        for row in rows:
            for value,column in zip(row,metadata['columns']):
                if value is None:
                    if not column['nullable']:raise ValueError('Retained null violates column schema.')
                elif not df._fitted_module().typed(value,column['dtype']):raise ValueError('Retained value violates its column dtype.')
        if df.digest({'columns':metadata['columns'],'rows':rows,'ids':ids})!=lineage['dataVersion']:raise ValueError('Retained table data version differs.')
        return df.Table(tuple(metadata['columns']),rows,ids,lineage['sourceDigest'],lineage['role'],lineage['dataVersion'],lineage['partitionDigest'],tuple(lineage['fittedStateDigests']),tuple(lineage['diagnostics']))
    except BaseException:store.close();raise


def envelope(table,reference):
    """Small table-file entry; segment files remain separately inventoried."""
    return {'schemaVersion':4,'kind':'segmentedTable','columns':list(table.columns),'table':reference}


def read_envelope(path,value,*,check_cancelled=lambda:None):
    if type(value) is not dict or set(value)!={'schemaVersion','kind','columns','table'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=4 or value['kind']!='segmentedTable':
        raise ValueError('Unsupported segmented table envelope.')
    if __package__:from .segmented_table_store import SegmentResolver
    else:from segmented_table_store import SegmentResolver
    from pathlib import Path
    resolver=SegmentResolver(Path(path).parent,maximum_bytes=current_policy().limit('retainedBytes'),check_cancelled=check_cancelled)
    try:
        table=read_capture(resolver,value['table'],check_cancelled=check_cancelled)
        if list(table.columns)!=value['columns']:
            table.rows.store.close();raise ValueError('Segmented table schema differs from its envelope.')
        resolver.audit()
        table.rows.store.owned_resolver=resolver
        return table
    except BaseException:resolver.close();raise


def read_page(path,expected_sha256,*,offset,limit,check_cancelled=lambda:None,render_row=None):
    """Selected pinned ranges only; not global membership/dataVersion validation.

    Never hand this result to training/replay as a scientifically verified Table.
    Full readers retain the complete validation/audit contract above.
    """
    import hashlib
    from pathlib import Path
    if __package__:
        from .dataflow_storage import bounded_regular_read
        from .segmented_table_store import SegmentResolver
    else:
        from dataflow_storage import bounded_regular_read
        from segmented_table_store import SegmentResolver
    if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=100:
        raise ValueError('Invalid retained table page.')
    check_cancelled()
    if Path(path).lstat().st_size>2_000_000:return None # Legacy full-table reader owns larger files.
    raw=bounded_regular_read(path,2_000_000)
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Retained table envelope pin differs.')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate retained table envelope field.')
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=unique)
    if type(value) is not dict or value.get('kind')!='segmentedTable':return None
    if set(value)!={'schemaVersion','kind','columns','table'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=4:
        raise ValueError('Unsupported segmented table envelope.')
    with SegmentResolver(Path(path).parent,maximum_bytes=current_policy().limit('retainedBytes'),check_cancelled=check_cancelled) as resolver:
        metadata,reader=_read_metadata(resolver,value['table'])
        if metadata['columns']!=value['columns']:raise ValueError('Segmented table schema differs from its envelope.')
        store=_CaptureStore(reader,metadata,check_cancelled)
        try:
            if __package__:from .table_transfer_policy import MAX_PAGE_BYTES
            else:from table_transfer_policy import MAX_PAGE_BYTES
            rows=[];identities=set();identity_cache=None;positions=[]
            consumed=len(json.dumps(metadata['columns'],ensure_ascii=True).encode())
            for position in range(offset,min(offset+limit,store.count)):
                check_cancelled();row=store.row(position)
                if any(not df._fitted_module().typed(v,c['dtype']) if v is not None else not c['nullable'] for v,c in zip(row,metadata['columns'])):
                    raise ValueError('Retained page value violates its column schema.')
                if identity_cache is None or not identity_cache[0]<=position<identity_cache[1]:
                    values,local=store._values(store.identity_index,position)
                    identity_cache=(position-local,position-local+len(values),values)
                identity=identity_cache[2][position-identity_cache[0]]
                if type(identity) is not str or not 1<=len(identity.encode('utf-8'))<=200 or '\0' in identity or identity in identities:
                    raise ValueError('Invalid retained page observation identity.')
                rendered=list(row) if render_row is None else render_row(row,metadata['columns'])
                size=len(json.dumps(rendered,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode())
                if consumed+size>MAX_PAGE_BYTES:
                    if not rows:raise ValueError('Table View row exceeds its display byte budget.')
                    break
                consumed+=size;identities.add(identity);rows.append(rendered);positions.append(position)
            check_cancelled()
            return {'columns':metadata['columns'],'rows':rows,'positions':positions,'totalRows':store.count,'verification':'pinnedRanges'}
        finally:store.close()
