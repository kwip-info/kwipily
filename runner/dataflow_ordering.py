"""Bounded scratch and exact key semantics shared by Sort and Unique."""
import json
import sqlite3
import tempfile
from contextlib import contextmanager
if __package__:
    from . import dataflow as df
    from .execution_checkpoints import RowCheckpoints
else:
    import dataflow as df
    from execution_checkpoints import RowCheckpoints

@contextmanager
def scratch(check):
    check()
    if __package__:
        from .dataflow_disk import current_scope,ScratchBudget
    else:
        from dataflow_disk import current_scope,ScratchBudget
    scope=current_scope();budget=scope[1] if scope else ScratchBudget()
    with tempfile.TemporaryDirectory(prefix='.kwip-ordering-',dir=scope[0] if scope else None) as directory:
        connection=sqlite3.connect(directory+'/keys.sqlite');reserved=0;cancellation=[]
        def progress():
            try:check();return 0
            except BaseException as error:cancellation.append(error);return 1
        def account():
            nonlocal reserved
            # Account every insertion; cooperative checks belong to the bounded
            # row iterator and SQLite's independent VM progress handler.
            current=connection.execute('PRAGMA page_count').fetchone()[0]*4096
            budget.resize(reserved,current);reserved=current
        try:
            connection.execute('PRAGMA page_size=4096')
            connection.execute('PRAGMA max_page_count='+str(max(3,(budget.limit-budget.used)//4096))) # Captured shared scratch allowance.
            connection.execute('PRAGMA cache_size=-8192')
            connection.execute('PRAGMA temp_store=FILE')
            connection.execute('PRAGMA journal_mode=OFF')
            if __package__:from .native_frame_arena import current_arena
            else:from native_frame_arena import current_arena
            arena=current_arena()
            # Bounded VM instruction work; every invocation still runs the live
            # guard. Entry/row/I/O/publication checks are unchanged.
            connection.set_progress_handler(progress,getattr(arena,'sqlite_instructions',1000))
            try:yield connection,account
            except sqlite3.Error as error:
                if cancellation:raise cancellation[0]
                if isinstance(error,sqlite3.IntegrityError):raise ValueError("Sort/Unique requires unique valid observation identities.") from error
                raise ValueError("Sort/Unique scratch storage failed or exceeded its storage budget.") from error
        finally:
            connection.close();budget.resize(reserved,0)

def key_columns(table,names):
    df._fitted_module().schema_valid(list(table.columns))
    if len(table.rows)!=len(table.row_ids):raise ValueError('Input rows and observation identities differ.')
    columns=[table.columns[table.index(name)] for name in names]
    if any(c['dtype'] in ('image','audio','tensor') for c in columns):raise ValueError('Images cannot be keys; select scalar columns.')
    # Temporal values use their canonical physical integers and unchanged schema metadata.
    if any(c['dtype'] not in ('string','int64','float64','bool','date','datetime','time') for c in columns):
        raise ValueError('Sort/Unique keys currently require text, integer, number or boolean; select a supported scalar key.')
    return [table.index(name) for name in names]

def checked_pairs(table,check):
    typed=df._fitted_module().typed
    checkpoints=RowCheckpoints(check)
    for position,(row,rid) in enumerate(df.row_pairs(table)):
        checkpoints.admit(row)
        if len(row)!=len(table.columns) or not isinstance(rid,str) or not 1<=len(rid)<=200:raise ValueError('Invalid input row or observation identity.')
        if any((v is None and not c['nullable']) or (v is not None and not typed(v,c['dtype'])) for v,c in zip(row,table.columns)):
            raise ValueError('Input violates declared types or nullability.')
        yield position,row,rid
    checkpoints.finish()

def encode(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def normalized(value,comparison):
    if isinstance(value,str) and comparison=='casefold':return value.casefold()
    # Float schema accepts exact integers too: normalize representations that compare equal.
    if type(value) is float and value.is_integer():return int(value)
    return value

def verify(expected,actual):
    from itertools import zip_longest
    if any(getattr(expected,k)!=getattr(actual,k) for k in ('columns','source_digest','role','partition_digest','fitted_states','diagnostics')) or len(expected.rows)!=len(actual.rows):
        raise ValueError('Cached ordering schema, lineage or count mismatch.')
    for a,b in zip_longest(df.row_pairs(expected),df.row_pairs(actual)):
        if a!=b:raise ValueError('Cached ordering rows or identities mismatch.')
