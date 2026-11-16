"""Standalone bounded learned-transform parameters and deterministic application.

The source digest is provenance, not authentication. Package inventory verification
and transform-chain ordering are the caller's responsibility.
"""
import json
import math
import re

PREFIX='kwip.data.'
SCALER='standardScalerFit'
IMPUTER='imputerFit'
ENCODER='oneHotEncoderFit'
MAX_ROWS=100_000
MAX_COLUMNS=128
MAX_BYTES=20_000_000
COMMON={'schemaVersion','kind','operator','columns','inputSchema','outputSchema','sourceContentDigest'}
EXTRA={SCALER:{'means','scales'},IMPUTER:{'strategy','fillValues'},ENCODER:{'vocabularies','unknownPolicy'}}


TEMPORAL_DTYPES=('date','datetime','time')
def _temporal_module():
    # Legacy standalone parameter imports need no additional modules until temporals appear.
    if __package__:from . import dataflow_temporal
    else:import dataflow_temporal
    return dataflow_temporal


def numeric(value):
    try:return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:return False


def typed(value,dtype):
    if dtype=='int64Sequence':
        if __package__:from .named_inputs import valid_sequence
        else:from named_inputs import valid_sequence
        return valid_sequence(value)
    if dtype in TEMPORAL_DTYPES:return _temporal_module().valid_value(value,dtype)
    if dtype in ('audio','tensor'):
        if __package__:from .media_artifacts import is_reference
        else:from media_artifacts import is_reference
        return is_reference(value,dtype)
    if dtype=='image':
        if __package__:from .image_artifacts import is_reference
        else:from image_artifacts import is_reference
        return is_reference(value)
    if dtype=='string':return type(value) is str
    if dtype=='bool':return type(value) is bool
    if dtype=='int64':return type(value) is int and -(2**63)<=value<2**63
    return dtype=='float64' and numeric(value)


def names(value):
    return isinstance(value,list) and 1<=len(value)<=MAX_COLUMNS and all(type(v) is str and bool(v.strip()) and len(v)<=200 for v in value) and len(set(value))==len(value)


def schema_valid(schema):
    if not isinstance(schema,list) or not names([v.get('name') if isinstance(v,dict) else None for v in schema]):raise ValueError('Invalid fitted table schema')
    for col in schema:
        if set(col)-{'name','dtype','nullable','displayLabel','temporal'} or col.get('dtype') not in ('float64','int64','string','bool','image','audio','tensor','int64Sequence')+TEMPORAL_DTYPES or type(col.get('nullable')) is not bool:raise ValueError('Invalid fitted column schema')
        if col['dtype'] in TEMPORAL_DTYPES:_temporal_module().validate_column(col)
        elif 'temporal' in col:raise ValueError('Temporal metadata requires a temporal dtype')
        if 'displayLabel' in col and (type(col['displayLabel']) is not str or len(col['displayLabel'])>400):raise ValueError('Invalid column display label')


def encoder_schema(columns,selected,vocabularies):
    vocabulary_by_name=dict(zip(selected,vocabularies));output=[];original={c['name'] for c in columns};generated=set()
    for col in columns:
        if col['name'] not in vocabulary_by_name:output.append(dict(col));continue
        for category in vocabulary_by_name[col['name']]:
            name='onehot:'+json.dumps([col['name'],category],ensure_ascii=False,separators=(',',':'))
            if name in original or name in generated:raise ValueError('Encoded output column collision with existing/source names')
            if len(name)>200:raise ValueError('Encoded category/column name exceeds the 200-character output bound')
            generated.add(name);output.append({'name':name,'displayLabel':f"{col['name']} = {category}",'dtype':'float64','nullable':False})
    if len(output)>MAX_COLUMNS:raise ValueError('Encoded output exceeds 128 total columns')
    return output


def validate(params):
    if isinstance(params,dict) and params.get('operator')==PREFIX+'preprocess':
        if __package__:from .preprocess_parameters import validate as check
        else:from preprocess_parameters import validate as check
        return check(params)
    try:
        if len(json.dumps(params,allow_nan=False).encode())>MAX_BYTES:raise ValueError('Parameters exceed 20 MB')
    except (TypeError,OverflowError,RecursionError) as error:raise ValueError('Invalid fitted parameters') from error
    if not isinstance(params,dict):raise ValueError('Parameters must be an object')
    operator=params.get('operator')
    if not isinstance(operator,str) or not operator.startswith(PREFIX):raise ValueError('Invalid fitted operator')
    operator=operator[len(PREFIX):]
    if operator not in EXTRA or set(params)!=COMMON|EXTRA[operator] or type(params['schemaVersion']) is not int or params['schemaVersion']!=1 or params['kind']!='portableFittedParameters':raise ValueError('Invalid fitted parameter fields')
    if not isinstance(params['sourceContentDigest'],str) or re.fullmatch('[0-9a-f]{64}',params['sourceContentDigest']) is None:raise ValueError('Invalid source state digest')
    if not names(params['columns']):raise ValueError('Invalid fitted columns')
    schema_valid(params['inputSchema']);schema_valid(params['outputSchema'])
    schema={c['name']:c for c in params['inputSchema']}
    if any(c not in schema for c in params['columns']):raise ValueError('Fitted columns absent from schema')
    if any(schema[c]['dtype'] in TEMPORAL_DTYPES for c in params['columns']):raise ValueError('Temporal fitted features require explicit Formula extraction before fitting.')
    expected=[dict(c) for c in params['inputSchema']]
    count=len(params['columns'])
    if operator==SCALER:
        if any(not isinstance(params[k],list) or len(params[k])!=count or any(not numeric(v) for v in params[k]) for k in ('means','scales')) or any(v<=0 for v in params['scales']):raise ValueError('Invalid scaler parameters')
        for col in expected:
            if col['name'] in params['columns']:
                if col['dtype'] not in ('float64','int64'):raise ValueError('Scaler requires numeric schema')
                col['dtype']='float64'
    elif operator==IMPUTER:
        if params['strategy'] not in ('mean','median','mostFrequent') or not isinstance(params['fillValues'],list) or len(params['fillValues'])!=count:raise ValueError('Invalid imputer parameters')
        for name,value in zip(params['columns'],params['fillValues']):
            col=next(c for c in expected if c['name']==name)
            if params['strategy'] in ('mean','median'):
                if col['dtype'] not in ('float64','int64') or not numeric(value):raise ValueError('Invalid numeric imputer statistic')
                col['dtype']='float64'
            elif not typed(value,col['dtype']):raise ValueError('Invalid imputer fill dtype')
            col['nullable']=False
    else:
        if params['unknownPolicy'] not in ('error','ignore') or not isinstance(params['vocabularies'],list) or len(params['vocabularies'])!=count:raise ValueError('Invalid encoder parameters')
        for name,vocab in zip(params['columns'],params['vocabularies']):
            if schema[name]['dtype']!='string' or not isinstance(vocab,list) or not 1<=len(vocab)<=MAX_COLUMNS or any(type(v) is not str or len(v)>200 for v in vocab) or len(set(vocab))!=len(vocab):raise ValueError('Invalid encoder vocabulary')
        expected=encoder_schema(params['inputSchema'],params['columns'],params['vocabularies'])
    if params['outputSchema']!=expected:raise ValueError('Fitted output schema differs from learned transform')
    return params


def apply(rows,input_schema,params):
    """Return transformed rows, derived columns, and transform-only diagnostics."""
    if isinstance(params,dict) and params.get('operator')==PREFIX+'preprocess':
        if __package__:from .preprocess_parameters import apply as run
        else:from preprocess_parameters import apply as run
        return run(rows,input_schema,params)
    validate(params)
    if input_schema!=params['inputSchema']:raise ValueError('Fitted input schema mismatch')
    if not isinstance(rows,(list,tuple)) or len(rows)>MAX_ROWS:raise ValueError('Rows exceed bound')
    if any(not isinstance(row,(list,tuple)) or len(row)!=len(input_schema) for row in rows):raise ValueError('Row width differs from schema')
    for row in rows:
        for value,column in zip(row,input_schema):
            if column['dtype'] in TEMPORAL_DTYPES and not (value is None and column['nullable'] or value is not None and typed(value,column['dtype'])):raise ValueError('Portable transform input violates temporal dtype/nullability')
    rows=[list(row) for row in rows];cols=[dict(c) for c in params['outputSchema']];diagnostics=[]
    indices=[next(i for i,c in enumerate(input_schema) if c['name']==name) for name in params['columns']]
    operator=params['operator'][len(PREFIX):]
    if operator==SCALER:
        for i,mean,scale in zip(indices,params['means'],params['scales']):
            for row in rows:
                if not numeric(row[i]):raise ValueError('Transform requires finite nonmissing data')
                row[i]=(row[i]-mean)/scale
                if not math.isfinite(row[i]):raise ValueError('Transform numerical overflow')
    elif operator==IMPUTER:
        for i,fill in zip(indices,params['fillValues']):
            count=0
            for row in rows:
                if row[i] is None:row[i]=fill;count+=1
                if not typed(row[i],cols[i]['dtype']):raise ValueError('Imputer transform encountered invalid typed value')
                if cols[i]['dtype']=='float64':row[i]=float(row[i])
            if count:diagnostics.append({'code':'missingValuesImputed','column':input_schema[i]['name'],'count':count,'stateDigest':params['sourceContentDigest']})
    else:
        vocab_by_index=dict(zip(indices,params['vocabularies']));unknown={i:0 for i in indices};converted=[]
        for row in rows:
            out=[]
            for i,value in enumerate(row):
                if i not in vocab_by_index:out.append(value);continue
                if not isinstance(value,str):raise ValueError('Resolve missing categories before encoding')
                vocab=vocab_by_index[i]
                if value not in vocab:
                    if params['unknownPolicy']=='error':raise ValueError(f'Unknown category in column {input_schema[i]["name"]}')
                    unknown[i]+=1
                out.extend(float(value==category) for category in vocab)
            converted.append(out)
        rows=converted
        for i,count in unknown.items():
            if count:diagnostics.append({'code':'unknownCategoryIgnored','column':input_schema[i]['name'],'count':count,'stateDigest':params['sourceContentDigest']})
    return rows,cols,diagnostics
