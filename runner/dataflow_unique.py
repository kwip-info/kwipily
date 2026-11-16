"""Spilled unique-key tracking with complementary source-ordered outputs."""
import json
from dataclasses import replace
if __package__:
    from .dataflow_ordering import df,scratch,key_columns,checked_pairs,encode,normalized,verify
else:
    from dataflow_ordering import df,scratch,key_columns,checked_pairs,encode,normalized,verify

def validate_config(c):
    if not isinstance(c,dict) or set(c)!={'keys','keep','missing','comparison'}:raise ValueError('Unique requires keys, retained member, missing and comparison policies.')
    keys=c['keys']
    if not isinstance(keys,list) or not 1<=len(keys)<=128 or any(not isinstance(k,str) or not k or len(k)>200 for k in keys) or len(set(keys))!=len(keys):raise ValueError('Unique requires 1–128 unique key columns.')
    if c['keep'] not in ('first','last') or c['missing'] not in ('equal','distinct','stop') or c['comparison'] not in ('exact','casefold'):raise ValueError('Invalid Unique keep, missing or comparison policy.')

def unique_table(table,c,check_cancelled=lambda:None):
    validate_config(c);indices=key_columns(table,c['keys'])
    with scratch(check_cancelled) as (db,account):
        db.execute('CREATE TABLE rows (position INTEGER PRIMARY KEY, key TEXT, payload TEXT, rid TEXT UNIQUE)')
        db.execute('CREATE TABLE retained (key TEXT PRIMARY KEY, position INTEGER) WITHOUT ROWID')
        for position,row,rid in checked_pairs(table,check_cancelled):
            values=[normalized(row[i],c['comparison']) for i in indices]
            missing=any(v is None for v in values)
            if missing and c['missing']=='stop':raise ValueError('Unique found a missing key. Choose Equal or Distinct rows to continue.')
            key=encode([position,values] if missing and c['missing']=='distinct' else [None,values])
            db.execute('INSERT INTO rows VALUES (?,?,?,?)',(position,key,encode(row),rid))
            db.execute('INSERT OR '+('IGNORE' if c['keep']=='first' else 'REPLACE')+' INTO retained VALUES (?,?)',(key,position))
            account()
        account();db.commit()
        count=db.execute('SELECT count(*) FROM retained').fetchone()[0]
        outputs={}
        try:
            for port,equal in [('unique','='),('duplicates','<>')]:
                diagnostic=dict(kind='unique',keys=c['keys'],keep=c['keep'],missing=c['missing'],comparison=c['comparison'],inputRows=len(table.rows),uniqueRows=count,duplicateRows=len(table.rows)-count,port=port)
                base=replace(table,diagnostics=table.diagnostics+(diagnostic,))
                def pairs():
                    for payload,rid in db.execute(f'SELECT r.payload,r.rid FROM rows r JOIN retained k ON r.key=k.key WHERE r.position{equal}k.position ORDER BY r.position'):
                        check_cancelled();yield tuple(json.loads(payload)),rid
                outputs[port]=df.streaming_rows(base,pairs(),check_cancelled=check_cancelled)
        except BaseException:
            for result in outputs.values():
                if hasattr(result.rows,'store'):result.rows.store.close()
            raise

        return outputs

def verify_output(table,c,outputs,check_cancelled=lambda:None):
    if set(outputs)!={'unique','duplicates'}:raise ValueError('Cached Unique requires both complementary outputs.')
    for port,expected in unique_table(table,c,check_cancelled).items():verify(expected,outputs[port])

CONFIG_SCHEMA={'type':'object','required':['keys','keep','missing','comparison'],'additionalProperties':False,'properties':{'keys':{'type':'array','minItems':1,'maxItems':128,'items':{'type':'string','minLength':1,'maxLength':200}},'keep':{'type':'string','enum':['first','last']},'missing':{'type':'string','enum':['equal','distinct','stop']},'comparison':{'type':'string','enum':['exact','casefold']}}}
DEFAULT_CONFIG={'keys':[],'keep':'first','missing':'equal','comparison':'exact'}
def output_schema(table,c):
    validate_config(c);key_columns(table,c['keys']);return {'unique':table.columns,'duplicates':table.columns}
