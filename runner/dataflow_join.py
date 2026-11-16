"""Bounded equijoins before partitioning, with explicit cardinality and null rules."""
from collections import Counter
from dataclasses import dataclass
import json

MAX_ARTIFACT_BYTES=50_000_000
MAX_DIAGNOSTICS=16_384

def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def _diagnostics(*histories):
    result=[];seen=set();size=2
    for history in histories:
        for item in history:
            content=_encoded(item)
            if content in seen:continue
            size+=len(content)+1
            if len(result)>=MAX_DIAGNOSTICS or size>MAX_ARTIFACT_BYTES:
                raise ValueError('Join diagnostic history exceeds its count or byte budget; simplify upstream branches.')
            seen.add(content);result.append(item)
    return tuple(result)

@dataclass(frozen=True)
class JoinPlan:
    left:object
    right:object
    mode:str
    left_keys:list
    right_keys:list
    left_counts:dict
    right_index:dict
    columns:tuple
    source_digest:str
    diagnostics:tuple
    output_rows:int
    artifact_bytes:int
    projection:tuple

    def pairs(self):
        for a,key in enumerate(self.left_keys):
            targets=self.right_index.get(key,()) if key is not None else ()
            for b in targets:yield a,b
            if not targets and self.mode in ('left','full'):yield a,None
        if self.mode in ('right','full'):
            for b,key in enumerate(self.right_keys):
                if key is None or key not in self.left_counts:yield None,b

    def iter_rows(self):
        # pairs() visits left positions monotonically, including repeated matches.
        # Scan that side once instead of decoding/indexing it for every output.
        # Retain just the current left row; right-side lookup remains checked for
        # every output, including large many-to-many expansions.
        left_rows=iter(df.row_pairs(self.left));left_position=-1;left_pair=None
        paired=getattr(self.right.rows,'pair_at',None)
        shared=getattr(self.right.row_ids,'store',None) is getattr(self.right.rows,'store',None)
        gather=getattr(self.right.rows,'iter_selected_pairs',None) if shared else None
        right_rows=gather(b for _,b in self.pairs() if b is not None) if gather is not None else None
        if not (paired and shared):
            paired=lambda index:(self.right.rows[index],self.right.row_ids[index])
        try:
            for a,b in self.pairs():
                if a is not None:
                    while left_position<a:
                        left_pair=next(left_rows);left_position+=1
                    left_row,left_identity=left_pair
                else:left_row,left_identity=(None,)*len(self.left.columns),None
                right_row,right_identity=(next(right_rows) if right_rows is not None else paired(b)) if b is not None else ((None,)*len(self.right.columns),None)
                row=left_row+right_row
                row=tuple(row[i] for i in self.projection)
                identity='join:'+df.digest(['kwip.join.row.v1',left_identity,right_identity])
                yield row,identity
        finally:
            close=getattr(left_rows,'close',None)
            if close is not None:close()
            if right_rows is not None:right_rows.close()

if __package__:
    from . import dataflow as df
else:
    import dataflow as df

FIELDS={'leftKeys','rightKeys','mode','nullPolicy','cardinality','leftPrefix','rightPrefix','maxRows'}


def validate_config(config):
    if not isinstance(config,dict) or not FIELDS.issubset(config) or set(config)-FIELDS-{'outputColumns','newColumns'}:
        raise ValueError('Join requires explicit keys, mode, null policy, cardinality, prefixes and row limit.')
    validate_projection(config)
    if not df.names(config['leftKeys']) or not df.names(config['rightKeys']) or len(config['leftKeys'])!=len(config['rightKeys']):
        raise ValueError('Join keys must be unique ordered column lists of equal length.')
    if config['mode'] not in ('inner','left','right','full') or config['nullPolicy'] not in ('neverMatch','match','error'):
        raise ValueError('Invalid join mode or null policy.')
    if config['cardinality'] not in ('oneToOne','oneToMany','manyToOne','manyToMany'):
        raise ValueError('Invalid join cardinality.')
    if type(config['maxRows']) is not int or not 1<=config['maxRows']<=df.MAX_ROWS:
        raise ValueError('Join row limit must be between 1 and 100000.')
    if any(not isinstance(config[key],str) or not config[key].strip() or len(config[key])>100 for key in ('leftPrefix','rightPrefix')):
        raise ValueError('Join column prefixes must be nonblank and at most 100 characters.')
    if config['leftPrefix']==config['rightPrefix']:
        raise ValueError('Join column prefixes must be distinct.')


def validate_projection(config):
    if ('outputColumns' in config) != ('newColumns' in config):
        raise ValueError('Supply output columns and new-column policy together.')
    if 'outputColumns' not in config:return
    rows=config['outputColumns']
    if config['newColumns'] not in ('keep','drop','error') or not isinstance(rows,list) or len(rows)>256:
        raise ValueError('Invalid join output grid or new-column policy.')
    seen=set()
    for row in rows:
        if (not isinstance(row,dict) or set(row)!={'side','source','name','included'} or
            row['side'] not in ('left','right') or type(row['included']) is not bool or
            any(not isinstance(row[k],str) or not row[k].strip() or len(row[k])>200 for k in ('source','name'))):
            raise ValueError('Join output rows require a side, exact source, output name and Keep choice.')
        key=(row['side'],row['source'])
        if key in seen:raise ValueError('A source column appears twice in the join output grid.')
        seen.add(key)


def project_columns(left,right,config):
    overlap={c['name'] for c in left.columns}&{c['name'] for c in right.columns}
    available={};index=0
    for side,table in (('left',left),('right',right)):
        for column in table.columns:
            item=dict(column)
            if item['name'] in overlap:item['name']=config[side+'Prefix']+item['name']
            if (side=='left' and config['mode'] in ('right','full')) or (side=='right' and config['mode'] in ('left','full')):item['nullable']=True
            available[(side,column['name'])]=(index,item);index+=1
    if 'outputColumns' not in config:chosen=list(available.values())
    else:
        chosen=[];seen=set()
        for row in config['outputColumns']:
            key=(row['side'],row['source']);seen.add(key)
            if key not in available:raise ValueError('A configured join output source column is missing; revise the grid.')
            if row['included']:
                i,column=available[key];chosen.append((i,{**column,'name':row['name']}))
        new=[value for key,value in available.items() if key not in seen]
        if new and config['newColumns']=='error':raise ValueError('New join source columns require explicit grid choices.')
        if config['newColumns']=='keep':chosen.extend(new)
    columns=tuple(column for _,column in chosen)
    if not df.names([c['name'] for c in columns]):raise ValueError('Join output names must be unique and nonempty, with 1–128 columns.')
    return columns,tuple(i for i,_ in chosen)


def _validate_table(table,*,use_native=True):
    if table.role!='unassigned' or table.partition_digest or table.fitted_states:
        raise ValueError('Join sources before splitting or fitting transforms; joining existing partitions would discard isolation evidence.')
    df._fitted_module().schema_valid(list(table.columns))
    if not 1<=len(table.rows)<=df.MAX_ROWS or len(table.rows)!=len(table.row_ids):
        raise ValueError('Join requires bounded rows with matching observation identities.')
    if __package__:from .dataflow_native_relational import validate_owned
    else:from dataflow_native_relational import validate_owned
    if use_native and validate_owned(table):return
    if any(not isinstance(i,str) or not 1<=len(i)<=200 for i in table.row_ids) or len(set(table.row_ids))!=len(table.row_ids):
        raise ValueError('Join requires unique bounded observation identities.')
    for row in table.rows:
        if len(row)!=len(table.columns):raise ValueError('Join input row width differs from column schema.')
        for value,column in zip(row,table.columns):
            if value is None:
                if not column['nullable']:raise ValueError('Join input null violates column schema.')
            elif not df._fitted_module().typed(value,column['dtype']):raise ValueError('Join input violates finite column dtype.')


def plan_join(left,right,config,*,use_native=True,_exact_sizes=False):
    validate_config(config)
    for table in (left,right):_validate_table(table,use_native=use_native)
    left_indices=[left.index(name) for name in config['leftKeys']]
    right_indices=[right.index(name) for name in config['rightKeys']]
    if any(table.columns[index]['dtype'] in ('image','audio','tensor') for table,indices in ((left,left_indices),(right,right_indices)) for index in indices):raise ValueError('Image columns cannot be join keys; join on explicit scalar file metadata instead.')
    if any(left.columns[a]['dtype']!=right.columns[b]['dtype'] for a,b in zip(left_indices,right_indices)):
        raise ValueError('Join key dtype differs; add an explicit Cast before joining.')
    columns,projection=project_columns(left,right,config)

    def keys(table,indices):
        result=[]
        if __package__:from .dataflow_native_relational import key_rows
        else:from dataflow_native_relational import key_rows
        native=key_rows(table,indices) if use_native else None
        values=native if native is not None else (tuple(row[i] for i in indices) for row in table.rows)
        for key in values:
            if None in key:
                if config['nullPolicy']=='error':raise ValueError('Resolve missing join key values or choose an explicit null policy.')
                if config['nullPolicy']=='neverMatch':key=None
            result.append(key)
        return result
    left_keys=keys(left,left_indices);right_keys=keys(right,right_indices)
    left_counts=Counter(k for k in left_keys if k is not None)
    right_counts=Counter(k for k in right_keys if k is not None)
    if ((config['cardinality'] in ('oneToOne','oneToMany') and any(n>1 for n in left_counts.values())) or
        (config['cardinality'] in ('oneToOne','manyToOne') and any(n>1 for n in right_counts.values()))):
        raise ValueError('Join cardinality does not match duplicate keys; inspect duplicates or explicitly choose the intended relationship.')
    unmatched_left=sum(1 for k in left_keys if k is None or k not in right_counts)
    unmatched_right=sum(1 for k in right_keys if k is None or k not in left_counts)
    matches=sum(n*right_counts.get(k,0) for k,n in left_counts.items())
    total=matches+(unmatched_left if config['mode'] in ('left','full') else 0)+(unmatched_right if config['mode'] in ('right','full') else 0)
    # Count the complete expansion before allocating any output pairs or rows.
    if total>config['maxRows']:raise ValueError('Join would exceed its output row limit; reduce inputs, repair duplicate keys or explicitly revise the budget.')
    if total*len(columns)>5_000_000:raise ValueError('Join would exceed five million output cells; select fewer columns or reduce input rows.')
    if not total:raise ValueError('Join produced no rows; inspect key values, types and null policy or choose an outer join.')
    right_index={}
    for index,key in enumerate(right_keys):
        if key is not None:right_index.setdefault(key,[]).append(index)
    source=df.digest(['kwip.join.sources.v1',left.source_digest,right.source_digest])
    diagnostic={'kind':'join','mode':config['mode'],'cardinality':config['cardinality'],'nullPolicy':config['nullPolicy'],
                'leftRows':len(left.rows),'rightRows':len(right.rows),'matchedPairs':matches,'leftUnmatched':unmatched_left,
                'rightUnmatched':unmatched_right,'outputRows':total,'leftSourceDigest':left.source_digest,'rightSourceDigest':right.source_digest}
    diagnostics=_diagnostics(left.diagnostics,right.diagnostics,(diagnostic,))
    # Exact writer-format accounting, including escaped Unicode/control characters.
    # Multiplicities account for expansion without constructing output pairs/rows.
    selected_left=[i for i in projection if i<len(left.columns)]
    selected_right=[i-len(left.columns) for i in projection if i>=len(left.columns)]
    if __package__:from .dataflow_native_relational import row_size_bounds
    else:from dataflow_native_relational import row_size_bounds
    left_sizes=row_size_bounds(left,selected_left) if use_native and not _exact_sizes else None
    right_sizes=row_size_bounds(right,selected_right) if use_native and not _exact_sizes else None
    if left_sizes is None:left_sizes=[len(_encoded(tuple(row[i] for i in selected_left))) for row in left.rows]
    if right_sizes is None:right_sizes=[len(_encoded(tuple(row[i] for i in selected_right))) for row in right.rows]
    null_left=len(_encoded((None,)*len(selected_left)))
    null_right=len(_encoded((None,)*len(selected_right)))
    join_adjustment=1 if selected_left and selected_right else 2
    row_bytes=sum(left_sizes[a]*right_counts.get(k,0) for a,k in enumerate(left_keys) if k is not None)
    row_bytes+=sum(right_sizes[b]*left_counts.get(k,0) for b,k in enumerate(right_keys) if k is not None)
    row_bytes-=join_adjustment*matches # Joining two JSON row arrays removes one delimiter byte.
    if config['mode'] in ('left','full'):
        row_bytes+=sum(left_sizes[a]+null_right-join_adjustment for a,k in enumerate(left_keys) if k is None or k not in right_counts)
    if config['mode'] in ('right','full'):
        row_bytes+=sum(null_left+right_sizes[b]-join_adjustment for b,k in enumerate(right_keys) if k is None or k not in left_counts)
    envelope={'schemaVersion':1,'kind':'table','columns':columns,'rows':[],
              'lineage':{'role':'unassigned','sourceDigest':source,'membershipDigest':'0'*64,
                         'observationIDs':[],'dataVersion':'0'*64,'partitionDigest':'',
                         'fittedStateDigests':[],'diagnostics':diagnostics}}
    size=len(_encoded(envelope))+row_bytes+(total-1)+total*len(_encoded('join:'+'0'*64))+(total-1)
    if size>MAX_ARTIFACT_BYTES and use_native and not _exact_sizes:
        return plan_join(left,right,config,use_native=use_native,_exact_sizes=True)
    if size>MAX_ARTIFACT_BYTES:
        raise ValueError('Join would exceed its 50 MB serialized byte budget; select fewer columns, shorten values or reduce matched rows.')
    return JoinPlan(left,right,config['mode'],left_keys,right_keys,left_counts,right_index,tuple(columns),source,diagnostics,total,size,projection)


def join_tables(left,right,config,*,use_native=True):
    plan=plan_join(left,right,config,use_native=use_native)
    if __package__:
        from .dataflow_native_relational import join_table
        from .dataflow_disk import current_check
    else:
        from dataflow_native_relational import join_table
        from dataflow_disk import current_check
    native=join_table(plan,current_check()) if use_native else None
    if native is not None:return native
    rows=[];identities=[]
    for row,identity in plan.iter_rows():rows.append(row);identities.append(identity)
    return df.derive(df.Table(plan.columns,tuple(rows),tuple(identities),plan.source_digest,diagnostics=plan.diagnostics))


def verify_join_output(left,right,config,output):
    """Validate a materialization against exact ordered parent pairs, without copying it."""
    plan=plan_join(left,right,config)
    if (output.columns!=plan.columns or output.source_digest!=plan.source_digest or
        output.role!='unassigned' or output.partition_digest or output.fitted_states or
        output.diagnostics!=plan.diagnostics or len(output.rows)!=plan.output_rows or
        len(output.row_ids)!=plan.output_rows):
        raise ValueError('Cached join schema or lineage mismatch')
    for (row,identity),actual_row,actual_identity in zip(plan.iter_rows(),output.rows,output.row_ids):
        if row!=actual_row or identity!=actual_identity:
            raise ValueError('Cached join parent-pair rows or identities mismatch')
