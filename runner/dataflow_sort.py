"""Stable multi-key external sort. Equal keys always retain incoming order."""
import json
from dataclasses import replace
if __package__:
    from .dataflow_ordering import df,scratch,key_columns,checked_pairs,encode,normalized,verify
    from .execution_checkpoints import RowCheckpoints
else:
    from dataflow_ordering import df,scratch,key_columns,checked_pairs,encode,normalized,verify
    from execution_checkpoints import RowCheckpoints

def validate_config(c):
    if not isinstance(c,dict) or set(c)!={'keys'} or not isinstance(c['keys'],list) or not 1<=len(c['keys'])<=128:raise ValueError('Sort requires 1–128 ordered keys.')
    names=[]
    for key in c['keys']:
        if not isinstance(key,dict) or set(key)!={'column','direction','missing','comparison'} or not isinstance(key['column'],str) or not key['column'] or len(key['column'])>200 or key['direction'] not in ('ascending','descending') or key['missing'] not in ('first','last') or key['comparison'] not in ('exact','casefold'):
            raise ValueError('Invalid Sort column, direction, missing placement or comparison.')
        names.append(key['column'])
    if len(set(names))!=len(names):raise ValueError('Sort keys must be unique.')

def sort_table(table,c,check_cancelled=lambda:None,*,use_native=True):
    validate_config(c);indices=key_columns(table,[k['column'] for k in c['keys']])
    if __package__:from .dataflow_native_relational import sort_table as native_sort
    else:from dataflow_native_relational import sort_table as native_sort
    native=native_sort(table,c,check_cancelled) if use_native else None
    if native is not None:return native
    def compare(left,right):
        a,b=json.loads(left),json.loads(right)
        for x,y,key in zip(a,b,c['keys']):
            if x is None or y is None:
                result=0 if x is y else (-1 if x is None else 1)*(1 if key['missing']=='first' else -1)
            else:
                result=(x>y)-(x<y)
                if key['direction']=='descending':result=-result
            if result:return result
        return 0
    with scratch(check_cancelled) as (db,account):
        db.create_collation('kwip_keys',compare)
        db.execute('CREATE TABLE rows (position INTEGER PRIMARY KEY, key TEXT, payload TEXT, rid TEXT UNIQUE)')
        db.execute('CREATE INDEX ordering ON rows (key COLLATE kwip_keys,position)')
        for position,row,rid in checked_pairs(table,check_cancelled):
            key=[normalized(row[i],k['comparison']) for i,k in zip(indices,c['keys'])]
            db.execute('INSERT INTO rows VALUES (?,?,?,?)',(position,encode(key),encode(row),rid))
            account()
        account();db.commit()
        diagnostic=dict(kind='sort',keys=c['keys'],inputRows=len(table.rows),outputRows=len(table.rows))
        base=replace(table,diagnostics=table.diagnostics+(diagnostic,))
        def pairs():
            checkpoints=RowCheckpoints(check_cancelled)
            for payload,rid in db.execute('SELECT payload,rid FROM rows ORDER BY key COLLATE kwip_keys,position'):
                row=tuple(json.loads(payload));checkpoints.admit(row);yield row,rid
            checkpoints.finish()
        return df.streaming_rows(base,pairs(),check_cancelled=check_cancelled)

def verify_output(table,c,output,check_cancelled=lambda:None):verify(sort_table(table,c,check_cancelled),output)

CONFIG_SCHEMA={'type':'object','required':['keys'],'additionalProperties':False,'properties':{'keys':{'type':'array','minItems':1,'maxItems':128,'items':{'type':'object','required':['column','direction','missing','comparison'],'additionalProperties':False,'properties':{'column':{'type':'string','minLength':1,'maxLength':200},'direction':{'type':'string','enum':['ascending','descending']},'missing':{'type':'string','enum':['first','last']},'comparison':{'type':'string','enum':['exact','casefold']}}}}}}
DEFAULT_CONFIG={'keys':[]}
def output_schema(table,c):
    validate_config(c);key_columns(table,[k['column'] for k in c['keys']]);return table.columns
