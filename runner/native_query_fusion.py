"""Qualified linear pure regions: one native query, shared logical output views.

Sources, effects, fitting, sampling and fan-out remain scheduler boundaries. No
intermediate payload/identity SQLite store is created. Unqualified or uncertain
regions return to the captured node kernels before any result is published.
"""
from copy import copy
import polars as pl
if __package__:
    from . import dataflow as df
    from .dataflow_columns import compile_selection
    from .dataflow_formula import compile_plan
    from .dataflow_shared import ColumnProjectionStore
    from .dataflow_columnar import ColumnarRows,ColumnarIDs,MAX_BATCH_BYTES
    from .dataflow_native_relational import qualified
    from .native_formula_expression import compile_native
    from .native_frame_arena import resident_view,release_view
else:
    import dataflow as df
    from dataflow_columns import compile_selection
    from dataflow_formula import compile_plan
    from dataflow_shared import ColumnProjectionStore
    from dataflow_columnar import ColumnarRows,ColumnarIDs,MAX_BATCH_BYTES
    from dataflow_native_relational import qualified
    from native_formula_expression import compile_native
    from native_frame_arena import resident_view,release_view

POLICY={'schemaVersion':1,'maximumNodesPerRegion':32,'mode':'pureLinearStatic'}
PHYSICAL={'string':pl.String,'int64':pl.Int64,'float64':pl.Float64,'bool':pl.Boolean}

class _Refused(Exception):pass


def _static(node):
    if node[0] in ('column','literal'):return True
    if node[0]=='call':return node[1] in ('coalesce','trim','length') and all(_static(x) for x in node[2])
    if node[0]=='binary':return node[1] in ('=','==','!=','<','<=','>','>=','AND','OR') and _static(node[2]) and _static(node[3])
    if node[0]=='isNull':return _static(node[2])
    if node[0]=='unary':return node[1]=='NOT' and _static(node[2])
    return False


def _compile(table,steps):
    columns=list(table.columns);physical=[dict(c,name=f'c{i}') for i,c in enumerate(columns)]
    bindings=list(range(len(columns)));operations=[];flags=[];views=[];literal_bytes=0;expressions=0
    for key,operator,config in steps:
        schema=df.Table(tuple(columns),(),(),'')
        if operator=='columns':
            columns,selections=compile_selection(schema,config)
            if any(kind not in ('keep','image') for _,_,kind in selections):return None
            bindings=[bindings[i] for i,_,_ in selections]
        elif operator=='formula':
            columns,plan=compile_plan(columns,config)
            for expression,targets,policy in plan:
                if len(targets)!=1 or not _static(expression.tree):return None
                target,dtype=targets[0]
                if dtype not in PHYSICAL:return None
                remapped=copy(expression)
                remapped.columns={name:(bindings[index],kind) for name,(index,kind) in expression.columns.items()}
                kernel=compile_native(remapped,expression.tree,None)
                if kernel is None:return None
                number=len(physical)
                if number>=128:return None  # Physical region fallback, never a workflow restriction.
                operations.append(kernel.cast(PHYSICAL[dtype]).alias(f'c{number}'));expressions+=1
                physical.append({'name':f'c{number}','dtype':dtype,'nullable':True})
                if target==len(bindings):bindings.append(number)
                else:bindings[target]=number
                value=pl.col(f'c{number}')
                if policy=='error':flags.append(value.is_null())
                if dtype=='string':flags.append(value.str.len_chars()>4096)
                # Literal text allocates a new array; source-derived text cannot
                # exceed the conservative source-sized-per-expression bound.
                def literals(node):
                    if node[0]=='literal':return len(node[1].encode('utf-8')) if type(node[1]) is str else 32
                    if node[0]=='call':return sum(literals(x) for x in node[2])
                    if node[0]=='binary':return literals(node[2])+literals(node[3])
                    if node[0] in ('unary','isNull'):return literals(node[2])
                    return 0
                literal_bytes+=literals(expression.tree)
                # Freeze each flag in its expression's namespace before a later
                # overwrite. Every physical value has a distinct column name.
        else:return None
        bounds=[pl.col(f'c{index}').str.len_bytes().cast(pl.UInt64).fill_null(0)*12+2 if c['dtype']=='string' else pl.lit(32,dtype=pl.UInt64) for c,index in zip(columns,bindings)]
        flags.append(pl.sum_horizontal(bounds)+2+2*len(columns)>1_048_576)
        for c,index in zip(columns,bindings):
            if not c['nullable']:flags.append(pl.col(f'c{index}').is_null())
        views.append((key,tuple(dict(c) for c in columns),tuple(bindings)))
    return physical,operations,flags,views,literal_bytes,expressions


class FusedRegionStore(ColumnProjectionStore):
    def __init__(self,parent,compiled,check):
        self.parent=parent;self._arena=parent._arena;self._check=check;self.closed=False;self.finished=True
        self.columns,self.operations,self.flags,self.views,self.literal_bytes,self.expressions=compiled
        self.columns=tuple(self.columns);self.count=parent.count;self.identity_prefix=parent.identity_prefix
        self._json_columns=();self.observed_nulls=[False]*len(self.columns);self.native_collects=0
        if getattr(parent,'membership_sha256',None) is not None:self.membership_sha256=parent.membership_sha256
    @resident_view
    def _read_chunk(self,index):
        self.check_cancelled();source=self.parent._read_chunk(index)
        allowance=min(MAX_BATCH_BYTES,self._arena.limit//4)
        if source.estimated_size()*(1+self.expressions)+(self.literal_bytes+32*self.expressions)*source.height>allowance:raise _Refused()
        query=source.lazy()
        for operation in self.operations:query=query.with_columns(operation)
        query=query.with_columns(pl.any_horizontal(self.flags).fill_null(False).alias('_invalid'))
        try:frame=query.collect()
        except (pl.exceptions.InvalidOperationError,pl.exceptions.ComputeError):raise _Refused() from None
        self.native_collects+=1;self.check_cancelled()
        if frame.get_column('_invalid').any():raise _Refused()
        values=[frame.get_column(f'c{i}') for i in range(len(self.columns))]
        for i,value in enumerate(values):self.observed_nulls[i]|=value.null_count()>0
        return pl.DataFrame(values)
    def close(self):
        if self.closed:return
        release_view(self);self.closed=True;self.parent=None


def try_fuse(table,steps,check):
    if (len(steps)<2 or len(steps)>POLICY['maximumNodesPerRegion'] or not qualified(table)
        or getattr(table.rows.store,'_arena',None) is None):return None
    check()
    try:compiled=_compile(table,steps)
    except (ValueError,KeyError,TypeError,IndexError):return None
    if compiled is None:return None
    store=FusedRegionStore(table.rows.store,compiled,check)
    try:
        for index,_ in enumerate(store.chunks):store._read_chunk(index)
        result={}
        for key,columns,indices in store.views:
            view=ColumnProjectionStore(store,columns,indices,check)
            result[key]={'table':df.derive(table,columns=columns,rows=ColumnarRows(view),ids=ColumnarIDs(view))}
        check();return result
    except _Refused:
        store.close();return None
    except BaseException:
        store.close();raise
