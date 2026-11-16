"""Exact grouped reduction with one sequential input-row scan and bounded scratch.

This kernel stages values once, then visits each group's identity-ordered rows
without random IPC chunk reads. It retains reference group order, exact rational
and decimal arithmetic, membership IDs, null decisions and diagnostics. Ordered
text concatenation remains an explicit reference-kernel capability barrier.
"""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import sqlite3
if __package__:
    from . import dataflow as df, dataflow_aggregate as reference, formula_decimal
    from .dataflow_disk import DiskStore,current_check
    from .dataflow_ordering import scratch
    from .workload_policy import current_policy
else:
    import dataflow as df, dataflow_aggregate as reference, formula_decimal
    from dataflow_disk import DiskStore,current_check
    from dataflow_ordering import scratch
    from workload_policy import current_policy

SUPPORTED_REDUCERS=frozenset(reference.REDUCERS)-{'joinText'}


class _Reduction:
    def __init__(self,field,dtype):
        self.field=field;self.dtype=dtype;self.count=0;self.present=0
        self.total=Fraction();self.decimal=Decimal(0);self.extreme=None;self.error=None

    def observe(self,value):
        self.count+=1
        if value is None:return
        self.present+=1
        if self.error is not None:return
        op=self.field['op']
        try:
            if op in ('sum','mean'):self.total+=Fraction(value)
            elif op=='sumDecimal':
                parsed=formula_decimal.parse(value)
                if parsed is None:raise ValueError('Exact decimal sum received invalid or nonfinite text.')
                self.decimal=formula_decimal.exact(lambda:self.decimal+parsed)
            elif op in ('min','max'):
                if self.present==1 or (value<self.extreme if op=='min' else value>self.extreme):self.extreme=value
        except ValueError as error:self.error=error

    def value(self):
        op=self.field['op']
        if op=='countRows':return self.count
        if op=='countNonNull':return self.present
        if self.present!=self.count:
            if self.field['nullPolicy']=='error':raise ValueError('Resolve missing reducer values or choose an explicit null policy.')
            if self.field['nullPolicy']=='propagate':return None
        if not self.present:return None
        if self.error is not None:raise self.error
        if op=='sumDecimal':return formula_decimal.canonical(self.decimal)
        if op in ('min','max'):return self.extreme
        if op=='sum' and self.dtype=='int64':
            value=int(self.total)
            if not -(2**63)<=value<2**63:raise ValueError('Aggregate int64 sum overflow; explicitly cast or revise data.')
            return value
        try:value=float(self.total/self.present if op=='mean' else self.total)
        except OverflowError as error:raise ValueError('Aggregate numeric overflow; result is outside finite float64.') from error
        if not math.isfinite(value):raise ValueError('Aggregate numeric overflow; result is outside finite float64.')
        return value


def _qualified_native(table):
    # Keep the reference kernel standard-library-only.
    if not isinstance(table.rows,df.StoredRows) or not callable(getattr(table.rows.store,"_read_chunk",None)):return False
    if __package__:from .dataflow_native_relational import qualified
    else:from dataflow_native_relational import qualified
    return qualified(table)


def _native_projection(table,indices,check):
    """Only reducer/key scalars cross Python for exact reference reductions.

    The owned typed producer already admitted widths/types/nullability. Keep its
    lifetime and work guards live; no general table or untyped reader qualifies.
    """
    if not _qualified_native(table):return None
    parent=table.rows.store
    def rows():
        for number,(_,count,_) in enumerate(parent.chunks):
            check();frame=parent._read_chunk(number)
            if frame.height!=count:raise ValueError('Aggregate native population differs.')
            if indices:
                yield from frame.select([f'c{i}' for i in indices]).iter_rows()
            else:
                for _ in range(count):yield ()
    return rows()


def aggregate_table(table,config,check_cancelled=None):
    check=current_check() if check_cancelled is None else check_cancelled
    reference.validate_config(config)
    reference._validate_table(table,validate_data=False,maximum_rows=current_policy().limit('tableRows'))
    if any(f['op'] not in SUPPORTED_REDUCERS for f in config['aggregations']):
        raise ValueError('Ordered text concatenation requires the reference Aggregate kernel.')
    typed=df._fitted_module().typed
    with scratch(check) as (db,account):
        # Validate all IDs before row values, matching reference failure precedence.
        db.execute('CREATE TABLE inputs (position INTEGER PRIMARY KEY,identity TEXT UNIQUE COLLATE BINARY,payload TEXT,groupID INTEGER)')
        identifier_count=0
        try:
            for position,identity in enumerate(table.row_ids):
                identifier_count+=1
                if position%128==0:check()
                if type(identity) is not str or not 1<=len(identity)<=200:raise ValueError('Aggregate requires unique bounded observation identities.')
                db.execute('INSERT INTO inputs(position,identity) VALUES (?,?)',(position,identity))
                if position%128==0:account()
        except sqlite3.IntegrityError as error:raise ValueError('Aggregate requires unique bounded observation identities.') from error
        if identifier_count!=len(table.row_ids):raise ValueError('Aggregate requires unique bounded observation identities.')
        # Resolve projection only for producer-admitted typed stores. General
        # reference inputs retain complete validation and original error order.
        native_rows=None;selected=None
        if _qualified_native(table):
            indices,columns=reference.aggregate_schema(table,config)
            fields=config['aggregations'];original_fields=[table.index(f['column']) if 'column' in f else None for f in fields]
            selected=sorted(set(indices)|{i for i in original_fields if i is not None})
            native_rows=_native_projection(table,selected,check)
        row_count=0;pending=[];pending_bytes=0
        for position,row in enumerate(native_rows if native_rows is not None else table.rows):
            row_count+=1
            if position%128==0:check()
            if native_rows is None:
                if len(row)!=len(table.columns):raise ValueError('Aggregate input row width differs from schema.')
                for value,column in zip(row,table.columns):
                    if value is None:
                        if not column['nullable']:raise ValueError('Aggregate input null violates schema.')
                    elif not typed(value,column['dtype']):raise ValueError('Aggregate input violates finite column dtype.')
            encoded=reference.encoded(row)
            if pending and pending_bytes+len(encoded)>2_097_152:
                db.executemany('UPDATE inputs SET payload=? WHERE position=?',pending);pending.clear();pending_bytes=0;account()
            pending.append((encoded.decode('ascii'),position));pending_bytes+=len(encoded)
            if len(pending)==128:
                db.executemany('UPDATE inputs SET payload=? WHERE position=?',pending);pending.clear();pending_bytes=0;account()
        if pending:db.executemany('UPDATE inputs SET payload=? WHERE position=?',pending)
        if row_count!=len(table.rows):raise ValueError('Invalid aggregate input row bounds.')
        account()
        if native_rows is None:
            indices,columns=reference.aggregate_schema(table,config)
            fields=config['aggregations'];original_fields=[table.index(f['column']) if 'column' in f else None for f in fields]
        key_positions=[selected.index(i) for i in indices] if selected is not None else indices
        field_indices=[selected.index(i) if i is not None else None for i in original_fields] if selected is not None else original_fields
        field_dtypes=[table.columns[i]['dtype'] if i is not None else None for i in original_fields]
        groups={};dropped=0;null_counts=[0]*len(fields)
        for position,payload in db.execute('SELECT position,payload FROM inputs ORDER BY position'):
            if position%128==0:check()
            row=json.loads(payload);key=tuple(reference._key_value(row[position],table.columns[index]['dtype']) for position,index in zip(key_positions,indices))
            if None in key:
                if config['nullKeys']=='error':raise ValueError('Resolve missing group key values or choose an explicit missing-key policy.')
                if config['nullKeys']=='drop':dropped+=1;continue
            if key not in groups:
                if len(groups)>=config['maxRows']:raise ValueError('Aggregate exceeds its output row limit; revise grouping or budget.')
                groups[key]=len(groups)
            db.execute('UPDATE inputs SET groupID=? WHERE position=?',(groups[key],position))
            for i,index in enumerate(field_indices):null_counts[i]+=int(index is not None and row[index] is None)
        if not groups:raise ValueError('Aggregate produced no rows after dropping missing group keys.')
        if len(groups)*len(columns)>5_000_000:raise ValueError('Aggregate exceeds five million output cells.')
        db.execute('CREATE INDEX grouping ON inputs(groupID,identity COLLATE BINARY)');account();db.commit()
        diagnostic={'kind':'aggregate','inputRows':len(table.rows),'outputRows':len(groups),'droppedKeyRows':dropped,
            'sourceDigest':table.source_digest,'reducers':[{'output':f['output'],'op':f['op'],'nullInputRows':0 if f['op']=='countRows' else null_counts[i]} for i,f in enumerate(fields)]}
        diagnostics=[];seen=set();diagnostic_bytes=2
        for item in (*table.diagnostics,diagnostic):
            content=reference.encoded(item)
            if content in seen:continue
            diagnostic_bytes+=len(content)+1
            if len(diagnostics)>=reference.MAX_DIAGNOSTICS or diagnostic_bytes>reference.MAX_ARTIFACT_BYTES:raise ValueError('Aggregate diagnostic count or byte budget exceeded.')
            seen.add(content);diagnostics.append(item)
        source=df.digest(['kwip.aggregate.sources.v1',table.source_digest,config['groupBy'],config['nullKeys']])
        envelope={'schemaVersion':1,'kind':'table','columns':columns,'rows':[],
            'lineage':{'role':'unassigned','sourceDigest':source,'membershipDigest':'0'*64,'observationIDs':[],
                       'dataVersion':'0'*64,'partitionDigest':'','fittedStateDigests':[],'diagnostics':diagnostics}}
        size=len(reference.encoded(envelope));store=DiskStore(check)
        if __package__:from .execution_checkpoints import RowCheckpoints
        else:from execution_checkpoints import RowCheckpoints
        group_work=RowCheckpoints(check);member_work=RowCheckpoints(check)
        try:
            for group_position,(key,ordinal) in enumerate(groups.items()):
                group_work.admit((*key,*([None]*len(fields))))
                states=[_Reduction(f,dtype) for f,dtype in zip(fields,field_dtypes)]
                membership=hashlib.sha256(b'kwip.membership.v1\n')
                for member_position,(payload,identity) in enumerate(db.execute('SELECT payload,identity FROM inputs WHERE groupID=? ORDER BY identity COLLATE BINARY',(ordinal,))):
                    row=json.loads(payload);member_work.admit((*row,*([None]*len(fields))))
                    raw=identity.encode('utf-8');membership.update(str(len(raw)).encode()+b':'+raw)
                    for state,index in zip(states,field_indices):state.observe(row[index] if index is not None else 1)
                identity='aggregate:'+df.digest(['kwip.aggregate.row.v1',table.source_digest,
                    [[table.columns[i]['name'],table.columns[i]['dtype'],value] for i,value in zip(indices,key)],membership.hexdigest()])
                row=tuple(key)+tuple(state.value() for state in states)
                size+=len(reference.encoded(row))+len(reference.encoded(identity))+(2 if group_position else 0)
                if size>reference.MAX_ARTIFACT_BYTES:raise ValueError('Aggregate exceeds its 50 MB serialized byte budget.')
                store.append(row,identity)
            group_work.finish();member_work.finish();rows,ids=store.finish()
            return df.derive(df.Table(tuple(columns),rows,ids,source,diagnostics=tuple(diagnostics)))
        except BaseException:store.close();raise
