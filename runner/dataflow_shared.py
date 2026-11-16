"""Immutable column projections over an owned IPC population.

Only values already validated by ColumnarStore can enter this view. The root
store is retained by a normal strong reference; closing a view never closes its
siblings or root. No paths, authority grants or separate row copies are created.
"""
if __package__:
    from . import dataflow as df
    from .dataflow_columnar import ColumnarStore,ColumnarRows,ColumnarIDs
else:
    import dataflow as df
    from dataflow_columnar import ColumnarStore,ColumnarRows,ColumnarIDs


if __package__:from .native_frame_arena import resident_view,release_view
else:from native_frame_arena import resident_view,release_view


class ColumnProjectionStore:
    def __init__(self,parent,columns,indices,check):
        if parent.closed or not parent.finished:raise ValueError('Shared column parent is closed or incomplete.')
        self._arena=getattr(parent,'_arena',None)
        self.closed=False;self.finished=True;self._check=check
        self.columns=tuple(dict(c) for c in columns)
        self._json_columns=tuple(i for i,c in enumerate(self.columns) if c['dtype'] in ('audio','tensor'))
        self.observed_nulls=tuple(parent.observed_nulls[i] for i in indices)
        if type(parent) is ColumnProjectionStore:
            self.indices=tuple(parent.indices[i] for i in indices)
            self.parent=parent.parent
        else:self.indices=tuple(indices);self.parent=parent
        self.count=parent.count;self.identity_prefix=parent.identity_prefix
        if getattr(parent,'membership_sha256',None) is not None:self.membership_sha256=parent.membership_sha256

    def check_cancelled(self):
        # A view chain may hold the exact same operation guard at every level.
        # Invoke each distinct live guard once *within this checkpoint*, while
        # checking every retained parent lifetime. Nothing is cached across
        # checkpoints and a different parent/ready-group guard is never skipped.
        current=self;guards=[];seen=set()
        while isinstance(current,ColumnProjectionStore):
            if id(current) in seen:raise ValueError('Shared column owner chain is cyclic.')
            seen.add(id(current))
            if current._check not in guards:
                current._check();guards.append(current._check)
            if current.closed or current.parent is None or current.parent.closed:raise ValueError('Shared column store is closed.')
            current=current.parent
        guard=current.check_cancelled
        if guard not in guards:guard()

    @property
    def starts(self):self.check_cancelled();return self.parent.starts
    @property
    def chunks(self):self.check_cancelled();return self.parent.chunks
    @property
    def connection(self):self.check_cancelled();return self.parent.connection
    @property
    def budget(self):self.check_cancelled();return self.parent.budget
    def _reader_cursor(self,query,parameters=()):
        if self.closed:raise ValueError('Shared column store is closed.')
        return self.parent._reader_cursor(query,parameters)
    def _identity_at(self,index):
        if self.closed:raise ValueError('Shared column store is closed.')
        return self.parent._identity_at(index)

    @resident_view
    def _read_chunk(self,index):
        self.check_cancelled()
        frame=self.parent._read_chunk(index)
        # Polars column selection retains the immutable Arrow arrays. Aliases
        # keep the internal positional schema independent of user labels.
        return frame.select([frame.get_column(f'c{source}').alias(f'c{target}') for target,source in enumerate(self.indices)])

    def logical_row(self,row):
        import json
        if not self._json_columns:return row if type(row) is tuple else tuple(row)
        result=list(row)
        for index in self._json_columns:
            if result[index] is not None:result[index]=json.loads(result[index])
        return tuple(result)

    def iter_pairs(self):
        for index,(start,count,_) in enumerate(self.chunks):
            frame=self._read_chunk(index)
            cursor=self._reader_cursor('SELECT identity FROM ids WHERE position>=? AND position<? ORDER BY position',(start,start+count))
            try:
                for position,(row,(identity,)) in enumerate(zip(frame.iter_rows(buffer_size=256),cursor)):
                    if position%256==0:self.check_cancelled()
                    yield self.logical_row(row),self.identity_prefix+identity
            finally:cursor.close()

    def close(self):
        if getattr(self,'closed',True):return
        release_view(self);self.closed=True;self.parent=None
    def __del__(self):self.close()


def supports_shared_table(table):
    if (not isinstance(table.rows,ColumnarRows) or not isinstance(table.row_ids,ColumnarIDs)
        or table.rows.store is not table.row_ids.store
        or not isinstance(table.rows.store,(ColumnarStore,ColumnProjectionStore))):return False
    store=table.rows.store
    if len(table.columns)!=len(store.columns):return False
    for index,(public,physical) in enumerate(zip(table.columns,store.columns)):
        if public['dtype']!=physical['dtype'] or public.get('temporal')!=physical.get('temporal'):return False
        if not public['nullable'] and physical['nullable'] and store.observed_nulls[index]:return False
    return True


def projection_table(table,columns,indices,check):
    """Return a shared projection, or None for an unqualified storage adapter."""
    if not supports_shared_table(table):return None
    if (not indices or len(indices)!=len(columns) or len(set(indices))!=len(indices)
        or any(type(i) is not int or not 0<=i<len(table.columns) for i in indices)
        or any({k:v for k,v in selected.items() if k!='name'}!={k:v for k,v in table.columns[i].items() if k!='name'} for selected,i in zip(columns,indices))):
        raise ValueError('Shared column projection differs from its immutable parent schema.')
    if len(table.rows)*len(columns)>df.MAX_TABLE_CELLS:raise ValueError('Table exceeds the five million cell storage budget')
    check();store=ColumnProjectionStore(table.rows.store,columns,indices,check)
    try:return df.derive(table,rows=ColumnarRows(store),ids=ColumnarIDs(store),columns=columns)
    except BaseException:store.close();raise


class RowSelectionStore(ColumnProjectionStore):
    """A bounded position/identity map; source columns remain shared immutable IPC."""
    def __init__(self,parent,mapping,parent_chunks,check):
        self._arena=getattr(parent,'_arena',None)
        self.parent=parent;self.mapping=mapping;self.parent_chunks=tuple(parent_chunks)
        self.closed=False;self.finished=True;self._check=check
        self.columns=parent.columns;self.count=mapping.count;self.identity_prefix=mapping.identity_prefix
        self._json_columns=parent._json_columns
        self.observed_nulls=parent.observed_nulls
    @property
    def starts(self):self.check_cancelled();return self.mapping.starts
    @property
    def chunks(self):self.check_cancelled();return self.mapping.chunks
    @property
    def connection(self):self.check_cancelled();return self.mapping.connection
    def _reader_cursor(self,query,parameters=()):
        if self.closed:raise ValueError('Shared selection store is closed.')
        return self.mapping._reader_cursor(query,parameters)
    def _identity_at(self,index):
        if self.closed:raise ValueError('Shared selection store is closed.')
        return self.mapping._identity_at(index)
    @resident_view
    def _read_chunk(self,index):
        self.check_cancelled()
        positions=self.mapping._read_chunk(index).get_column('c0').to_list()
        source=self.parent._read_chunk(self.parent_chunks[index])
        return source[positions]
    def close(self):
        if self.closed:return
        release_view(self);self.closed=True;self.mapping.close();self.parent=None
    def __del__(self):self.close()


class ComputedColumnsStore(ColumnProjectionStore):
    """Changed fields have their own bounded store; untouched arrays stay shared."""
    def __init__(self,parent,computed,columns,bindings,check):
        from bisect import bisect_right
        self._arena=getattr(parent,'_arena',None)
        self.parent=parent;self.computed=computed;self.columns=tuple(columns);self.bindings=tuple(bindings)
        self.closed=False;self.finished=True;self._check=check
        self.count=parent.count;self.identity_prefix=parent.identity_prefix
        self._json_columns=tuple(i for i,c in enumerate(columns) if c['dtype'] in ('audio','tensor'))
        self.observed_nulls=tuple((parent if source=='parent' else computed).observed_nulls[index] for source,index in bindings)
        starts=sorted(set(parent.starts)|set(computed.starts));self._starts=tuple(starts)
        self._chunks=tuple((start,(starts[i+1] if i+1<len(starts) else self.count)-start,None) for i,start in enumerate(starts))
        self._locations=tuple((bisect_right(parent.starts,start)-1,bisect_right(computed.starts,start)-1) for start in starts)
        if getattr(parent,'membership_sha256',None) is not None:self.membership_sha256=parent.membership_sha256
    @property
    def starts(self):self.check_cancelled();return self._starts
    @property
    def chunks(self):self.check_cancelled();return self._chunks
    @property
    def connection(self):self.check_cancelled();return self.computed.connection
    def _reader_cursor(self,query,parameters=()):
        if self.closed:raise ValueError('Computed column store is closed.')
        return self.computed._reader_cursor(query,parameters)
    def _identity_at(self,index):
        if self.closed:raise ValueError('Computed column store is closed.')
        return self.computed._identity_at(index)
    @resident_view
    def _read_chunk(self,index):
        import polars as pl
        self.check_cancelled();start,count,_=self._chunks[index];left,right=self._locations[index]
        original=self.parent._read_chunk(left).slice(start-self.parent.starts[left],count)
        changed=self.computed._read_chunk(right).slice(start-self.computed.starts[right],count)
        return pl.DataFrame([(original if source=='parent' else changed).get_column(f'c{column}').alias(f'c{target}') for target,(source,column) in enumerate(self.bindings)])
    def close(self):
        if self.closed:return
        release_view(self);self.closed=True;self.computed.close();self.parent=None
    def __del__(self):self.close()


def computed_table(table,columns,computed,changed_indices,check):
    if computed.count!=len(table.rows):raise ValueError('Computed column population differs from its parent.')
    changed={target:position for position,target in enumerate(changed_indices)}
    bindings=[('computed',changed[i]) if i in changed else ('parent',i) for i in range(len(columns))]
    store=ComputedColumnsStore(table.rows.store,computed,columns,bindings,check)
    try:return df.derive(table,columns=columns,rows=ColumnarRows(store),ids=ColumnarIDs(store))
    except BaseException:store.close();raise


def _boolean_kernel(text,columns):
    """Compile the exact typed scalar subset; never coerce int64 columns to float.

    Native predicates consume immutable column batches. Unsupported arithmetic,
    functions and mixed numeric domains keep the scalar reference path.
    """
    if __package__:from .dataflow_expression import Expression
    else:from dataflow_expression import Expression
    import polars as pl
    try:expression=Expression(text,columns)
    except ValueError:return None
    if expression.dtype not in ('bool','null'):return None
    types={'bool':pl.Boolean,'int64':pl.Int64,'float64':pl.Float64,'string':pl.String,'null':pl.Null}
    def visit(node):
        kind=node[0];dtype=expression.type_of(node)
        if dtype not in types:return None
        if kind=='column':
            index,_=expression.columns[node[1]]
            return pl.col(f'c{index}')
        if kind=='literal':return pl.lit(node[1],dtype=types[dtype])
        if kind=='unary' and node[1]=='NOT':
            value=visit(node[2]);return None if value is None else ~value
        if kind=='isNull':
            value=visit(node[2]);return None if value is None else value.is_not_null() if node[1] else value.is_null()
        if kind=='binary':
            op=node[1]
            if op not in ('AND','OR','=','==','!=','<','<=','>','>='):return None
            a=visit(node[2]);b=visit(node[3])
            if a is None or b is None:return None
            left=expression.type_of(node[2]);right=expression.type_of(node[3])
            if {left,right}=={'int64','float64'}:
                # Python compares arbitrary int64 to float without rounding the
                # integer first. Polars supertype coercion does not. Only an
                # exactly representable integer literal may cross this boundary.
                integer=node[2] if left=='int64' else node[3]
                if integer[0]!='literal' or int(float(integer[1]))!=integer[1]:return None
            if op=='AND':return a&b
            if op=='OR':return a|b
            if op in ('=','=='):return a==b
            if op=='!=':return a!=b
            if op=='<':return a<b
            if op=='<=':return a<=b
            if op=='>':return a>b
            return a>=b
        return None
    return visit(expression.tree)


def filter_table(table,configuration,check):
    """Native boolean masks with bounded row-selection maps, or explicit fallback."""
    if configuration['mode']!='expression' or not supports_shared_table(table):return None
    parent=table.rows.store
    if not isinstance(parent,(ColumnarStore,ColumnProjectionStore)):return None
    expression=_boolean_kernel(configuration['expression'],table.columns)
    if expression is None:return None
    mappings={};parent_chunks={port:[] for port in ('matching','nonmatching')};outputs={}
    try:
        for port in parent_chunks:
            mappings[port]=ColumnarStore(({'name':'position','dtype':'int64','nullable':False},),check,budget=parent.budget,identity_prefix=parent.identity_prefix)
        for index,(start,count,_) in enumerate(parent.chunks):
            check();frame=parent._read_chunk(index)
            import polars as pl
            mask=frame.with_columns(expression.alias('_mask')).get_column('_mask')
            if len(mask)!=count:raise ValueError('Boolean selection population differs from parent.')
            if configuration['nullPolicy']=='error' and mask.null_count():
                raise ValueError(f'Filter row {start+mask.is_null().arg_true()[0]+1} is null; add a missing-value condition or choose nonmatching')
            cursor=parent._reader_cursor('SELECT identity FROM ids WHERE position>=? AND position<? ORDER BY position',(start,start+count))
            try:identities=pl.Series('identity',[parent.identity_prefix+identity for (identity,) in cursor],dtype=pl.String)
            finally:cursor.close()
            if len(identities)!=count:raise ValueError('Boolean selection identity population differs from parent.')
            mask=mask.fill_null(False);positions=pl.int_range(0,count,eager=True,dtype=pl.Int64).rename('c0')
            for port,mapping in mappings.items():
                check();selected=mask if port=='matching' else ~mask
                chosen=positions.filter(selected);chosen_ids=identities.filter(selected)
                before=len(mapping.chunks)
                if not mapping.append_frame(chosen.to_frame(),chosen_ids):
                    for local,identity in zip(chosen,chosen_ids):mapping.append((local,),identity)
                mapping._flush()
                parent_chunks[port].extend([index]*(len(mapping.chunks)-before))
        for port,mapping in mappings.items():
            mapping.finish()
            store=RowSelectionStore(parent,mapping,parent_chunks[port],check)
            outputs[port]=df.derive(table,rows=ColumnarRows(store),ids=ColumnarIDs(store))
        check();return outputs
    except BaseException:
        for mapping in mappings.values():mapping.close()
        raise


def native_formula_table(table,columns,plan,check,*,extended=True,deadline=None):
    """Qualified literal/copy/coalesce operations on immutable scalar batches.

    Unqualified expressions or conservative size/value bounds return to the
    existing reference implementation before publishing any computed result.
    """
    import polars as pl
    if not supports_shared_table(table) or any(c['dtype'] not in ('bool','string','int64','float64') for c in table.columns):return None
    physical={'bool':pl.Boolean,'string':pl.String,'int64':pl.Int64,'float64':pl.Float64,'null':pl.Null}
    def compile_expression(expression,node):
        dtype=expression.type_of(node)
        if dtype not in physical:return None
        if node[0]=='column':return pl.col(f'c{expression.columns[node[1]][0]}')
        if node[0]=='literal':return pl.lit(node[1],dtype=physical[dtype])
        if node[0]=='call' and node[1]=='coalesce':
            types={expression.type_of(n) for n in node[2]}-{'null'}
            if len(types)>1:return None
            values=[compile_expression(expression,n) for n in node[2]]
            if any(v is None for v in values):return None
            return pl.coalesce(values)
        return None
    operations=[]
    for expression,targets,policy in plan:
        kernel=compile_expression(expression,expression.tree)
        if __package__:from .native_formula_expression import supported
        else:from native_formula_expression import supported
        if len(targets)!=1 or (kernel is None and (not extended or not supported(expression.tree))):return None
        target,dtype=targets[0]
        if dtype not in physical:return None
        operations.append((expression,target,dtype,policy))
    parent=table.rows.store;changed=sorted({target for _,target,_,_ in operations})
    computed=ColumnarStore([columns[i] for i in changed],check,budget=parent.budget,identity_prefix=parent.identity_prefix)
    try:
        for chunk,(start,count,_) in enumerate(parent.chunks):
            if deadline is not None:
                import time
                if time.monotonic()>deadline:computed.close();return None
            check();frame=parent._read_chunk(chunk)
            for expression,target,dtype,policy in operations:
                if __package__:from .native_formula_expression import compile_native
                else:from native_formula_expression import compile_native
                kernel=compile_expression(expression,expression.tree)
                if kernel is None:kernel=compile_native(expression,expression.tree,frame)
                if kernel is None:computed.close();return None
                check();frame=frame.with_columns(kernel.cast(physical[dtype]).alias(f'c{target}'))
                values=frame.get_column(f'c{target}')
                if policy=='error' and values.null_count():computed.close();return None
                if dtype=='string' and (values.str.len_chars().max() or 0)>4096:computed.close();return None
                if dtype=='float64' and not values.drop_nulls().is_finite().all():computed.close();return None
            # Preserve the reference full-row byte contract. Uncertain oversized
            # rows use its exact scalar check, never an unchecked native shortcut.
            bounds=[pl.col(f'c{i}').str.len_bytes().cast(pl.UInt64).fill_null(0)*12+2 if c['dtype']=='string' else pl.lit(32,dtype=pl.UInt64) for i,c in enumerate(columns)]
            if count and frame.select((pl.sum_horizontal(bounds)+2+2*len(columns)).max()).item()>1_048_576:computed.close();return None
            selected=frame.select([pl.col(f'c{target}').alias(f'c{i}') for i,target in enumerate(changed)])
            cursor=parent._reader_cursor('SELECT identity FROM ids WHERE position>=? AND position<? ORDER BY position',(start,start+count))
            try:identities=pl.Series('identity',[parent.identity_prefix+identity for (identity,) in cursor],dtype=pl.String)
            finally:cursor.close()
            if not computed.append_frame(selected,identities):computed.close();return None
        computed.finish();check();return computed_table(table,columns,computed,changed,check)
    except BaseException:computed.close();raise
