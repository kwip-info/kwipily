"""Filter rows: typed conditions, bounded expressions and supervised regex.

Both branches preserve captured observation identities and partition/fitted lineage.
No source reparsing, implicit casts, image decoding or Python expression execution.
"""
import math
import json
import time

COMPARISONS=('eq','ne','lt','le','gt','ge','contains','notContains','startsWith','endsWith','in','notIn','isMissing','isNotMissing')
MISSING={'isMissing','isNotMissing'}
MEMBERSHIP={'in','notIn'}
TEXT={'contains','notContains','startsWith','endsWith'}
MAX_PATTERN_BATCH_ROWS=8192
MAX_PATTERN_BATCH_BYTES=1_000_000


def validate_config(c):
    if c['mode']=='simple' and not c['conditions']:raise ValueError('Filter needs at least one condition')
    if c['mode']=='expression' and not c['expression'].strip():raise ValueError('Filter expression must not be blank')
    if c['mode']=='pattern' and (not c['column'].strip() or not c['pattern']):raise ValueError('Filter pattern needs a column and pattern')
    for condition in c['conditions']:
        if not condition['column'].strip():raise ValueError('Filter column must not be blank')
        op=condition['comparison'];value=condition['value']
        if op in MISSING:
            if value is not None:raise ValueError('Missing-value conditions require a null literal')
        elif op in MEMBERSHIP:
            if not isinstance(value,list) or not 1<=len(value)<=256:raise ValueError('Membership needs 1–256 typed values')
        elif value is None or isinstance(value,list):raise ValueError('Comparison needs one non-null typed literal')
        values=value if isinstance(value,list) else [value]
        if any(type(v) not in (str,int,float,bool,type(None)) or (type(v) is float and not math.isfinite(v)) for v in values):raise ValueError('Filter literals must be finite typed scalars')
        if op not in MISSING and any(v is None for v in values):raise ValueError('Use an explicit missing-value condition instead of a null comparison')


def _simple(table,c):
    predicates=[]
    for condition in c['conditions']:
        index=table.index(condition['column']);dtype=table.columns[index]['dtype']
        comparison=condition['comparison'];value=condition['value'];sensitive=condition['caseSensitive']
        if comparison not in MISSING:
            if dtype in ('image','audio','tensor'):raise ValueError('Image columns support only missing-value conditions; use an explicit image adapter')
            if comparison in TEXT and dtype!='string':raise ValueError('Text conditions require a string column; use Columns to cast explicitly')
            values=value if comparison in MEMBERSHIP else [value]
            for literal in values:
                valid=type(literal) is str if dtype=='string' else type(literal) is bool if dtype=='bool' else type(literal) in (int,float) if dtype in ('int64','float64') else False
                if not valid:raise ValueError('Filter literal type differs from column; no implicit conversion was applied')
            if dtype=='bool' and comparison not in ('eq','ne','in','notIn'):raise ValueError('Boolean columns support equality, membership and missing-value conditions')
            if dtype=='string' and not sensitive:value=[v.casefold() for v in value] if comparison in MEMBERSHIP else value.casefold()
            if comparison in MEMBERSHIP:value=frozenset(value)
        def predicate(row,index=index,comparison=comparison,value=value,dtype=dtype,sensitive=sensitive):
            cell=row[index]
            if comparison=='isMissing':return cell is None
            if comparison=='isNotMissing':return cell is not None
            if cell is None:return None
            if dtype=='string' and not sensitive:cell=cell.casefold()
            if comparison=='eq':return cell==value
            if comparison=='ne':return cell!=value
            if comparison=='lt':return cell<value
            if comparison=='le':return cell<=value
            if comparison=='gt':return cell>value
            if comparison=='ge':return cell>=value
            if comparison=='contains':return value in cell
            if comparison=='notContains':return value not in cell
            if comparison=='startsWith':return cell.startswith(value)
            if comparison=='endsWith':return cell.endswith(value)
            if comparison=='in':return cell in value
            if comparison=='notIn':return cell not in value
            raise ValueError('Unknown Filter comparison')
        predicates.append(predicate)
    def evaluate(row):
        values=[predicate(row) for predicate in predicates]
        if c['match']=='all':return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    return evaluate


def _expression(table,c):
    if __package__:from .dataflow_expression import Expression
    else:from dataflow_expression import Expression
    class FilterExpression(Expression):
        def type_of(self,node):
            if node[0]=='isNull' and node[2][0]=='column' and node[2][1] in self.columns and self.columns[node[2][1]][1]=='image':return 'bool'
            return super().type_of(node)
    expression=FilterExpression(c['expression'],table.columns)
    if expression.dtype not in ('bool','null'):raise ValueError('Filter expression must produce a boolean comparison')
    return expression.evaluate


def apply(table,c,check,*,share=False):
    if __package__:from . import dataflow as df
    else:import dataflow as df
    validate_config(c)
    evaluate=_simple(table,c) if c['mode']=='simple' else _expression(table,c) if c['mode']=='expression' else None
    if share:
        if __package__:from .dataflow_shared import filter_table
        else:from dataflow_shared import filter_table
        shared=filter_table(table,c,check)
        if shared is not None:return shared
    if c['mode']=='pattern':
        if __package__:from .dataflow_regex import supervised,TIMEOUT_SECONDS
        else:from dataflow_regex import supervised,TIMEOUT_SECONDS
        index=table.index(c['column'])
        if table.columns[index]['dtype']!='string':raise ValueError('Pattern needs a string column; image/scalar decoding is not supported')
        pattern_config={key:c[key] for key in ('column','pattern','flags','matchMode','nullPolicy')}
        deadline=time.monotonic()+TIMEOUT_SECONDS
        def pattern_result(batch):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise ValueError('Regex exceeded five-second total execution budget; no result accepted')
            return supervised('regexMatch',pattern_config,[row[index] for row,_ in batch],check,timeout=remaining)
    stores={port:df.row_store(table.columns,check,table.row_ids.store.identity_prefix if isinstance(table.row_ids,df.StoredIDs) else '') for port in ('matching','nonmatching')}
    def append(row,rid,result,position):
        if result is None and c['nullPolicy']=='error':raise ValueError(f'Filter row {position+1} is null; add a missing-value condition or choose nonmatching')
        stores['matching' if result is True else 'nonmatching'].append(row,rid)
    try:
        if evaluate is not None:
            for position,(row,rid) in enumerate(df.row_pairs(table)):
                if position%128==0:check()
                try:result=evaluate(row)
                except ValueError as error:raise ValueError(f'Filter row {position+1}: {error}') from error
                append(row,rid,result,position)
        else:
            batch=[];size=0;offset=0
            def flush():
                nonlocal offset
                for position,((row,rid),result) in enumerate(zip(batch,pattern_result(batch))):append(row,rid,result,offset+position)
                offset+=len(batch)
            for position,(row,rid) in enumerate(df.row_pairs(table)):
                if position%128==0:check()
                cell=row[index]
                if cell is None and c['nullPolicy']=='error':raise ValueError(f'Filter row {position+1} is null; choose nonmatching or resolve missing text')
                if cell is not None and (type(cell) is not str or len(cell)>16384):raise ValueError('Regex requires text cells of at most 16384 characters')
                length=len(json.dumps(cell,ensure_ascii=True).encode())
                if batch and (len(batch)>=MAX_PATTERN_BATCH_ROWS or size+length>MAX_PATTERN_BATCH_BYTES):flush();batch=[];size=0
                batch.append((row,rid));size+=length
            # Even an empty table validates regex syntax under supervision.
            flush()
        outputs={}
        for port,store in stores.items():
            rows,ids=store.finish();outputs[port]=df.derive(table,rows=rows,ids=ids)
        return outputs
    except BaseException:
        for store in stores.values():store.close()
        raise
