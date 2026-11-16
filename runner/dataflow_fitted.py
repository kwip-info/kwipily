"""Training-partition-only portable fitted transforms; no user code execution."""
import json
import math
if __package__:
    from . import dataflow as df
    from . import portable_fitted as portable
else:
    import dataflow as df
    import portable_fitted as portable

SCALER='standardScalerFit'
IMPUTER='imputerFit'
ENCODER='oneHotEncoderFit'
COMMON={'schemaVersion','operator','columns','inputSchema','fitEvidence','fitDataVersion','partitionDigest','graphDigest','contentDigest'}
MODERN={'outputSchema','inputFittedStateDigests'}
EXTRA={SCALER:{'means','scales'},IMPUTER:{'configuration','fillValues','observedCounts','emptyColumns'},ENCODER:{'configuration','vocabularies'}}


def numeric(value):return type(value) in (int,float) and math.isfinite(value)

def category_name(column,category):return 'onehot:'+json.dumps([column,category],ensure_ascii=False,separators=(',',':'))

if __package__:from . import dataflow_temporal
else:import dataflow_temporal

def schema_valid(columns):
    if not isinstance(columns,list) or not df.names([c.get('name') if isinstance(c,dict) else None for c in columns]):raise ValueError('Invalid fitted table schema')
    for col in columns:
        if set(col)-{'name','dtype','nullable','displayLabel','temporal'} or col.get('dtype') not in ('float64','int64','string','bool','image','audio','tensor','int64Sequence')+dataflow_temporal.DTYPES or type(col.get('nullable')) is not bool:raise ValueError('Invalid fitted column schema')
        dataflow_temporal.validate_column(col)
        if 'displayLabel' in col and (not isinstance(col['displayLabel'],str) or len(col['displayLabel'])>400):raise ValueError('Invalid column display label')


def typed(value,dtype):
    if dtype in ('audio','tensor'):
        if __package__:from .media_artifacts import is_reference
        else:from media_artifacts import is_reference
        return is_reference(value,dtype)
    if dtype=='int64Sequence':
        if __package__:from .named_inputs import valid_sequence
        else:from named_inputs import valid_sequence
        return valid_sequence(value)
    if dtype in dataflow_temporal.DTYPES:return dataflow_temporal.valid_value(value,dtype)
    if dtype=='image':
        if __package__:from .image_artifacts import is_reference
        else:from image_artifacts import is_reference
        return is_reference(value)
    if dtype=='string':return type(value) is str
    if dtype=='bool':return type(value) is bool
    if dtype=='int64':return type(value) is int and -(2**63)<=value<2**63
    return dtype=='float64' and numeric(value)


def fit(table,operator,config,graph_digest):
    raise ValueError('Retired fitted execution is unavailable; upgrade paired partition transforms to current Preprocess.')


def encoder_schema(columns,selected,vocabularies):
    return portable.encoder_schema(columns,selected,vocabularies)


def load(text):
    if not isinstance(text,str) or len(text.encode())>20_000_000:raise ValueError('State exceeds 20 MB')
    state=json.loads(text)
    if not isinstance(state,dict):raise ValueError('Fitted state must be an object')
    if state.get('contentDigest')!=df.digest({k:v for k,v in state.items() if k!='contentDigest'}):raise ValueError('State content digest mismatch')
    operator=state.get('operator','').removeprefix(df.PREFIX)
    if operator not in EXTRA or state.get('operator')!=df.PREFIX+operator or type(state.get('schemaVersion')) is not int or state['schemaVersion']!=1:raise ValueError('Invalid fitted state discriminator')
    allowed=COMMON|MODERN|EXTRA[operator]
    if set(state)-allowed or (COMMON|EXTRA[operator])-set(state):raise ValueError('Invalid fitted state fields')
    # Original scaler schema remains readable; new fitted types require full lineage/schema.
    if operator!=SCALER and MODERN-set(state):raise ValueError('Fitted state output/chain metadata missing')
    if not df.names(state['columns']):raise ValueError('Invalid fitted columns')
    schema_valid(state['inputSchema'])
    schema_by_name={c['name']:c for c in state['inputSchema']}
    if any(c not in schema_by_name for c in state['columns']):raise ValueError('Fitted columns absent from input schema')
    if 'outputSchema' in state:schema_valid(state['outputSchema'])
    ev=state['fitEvidence'];ids=ev['observationIDs']
    if not isinstance(ids,list) or not 1<=len(ids)<=df.MAX_ROWS or any(not isinstance(v,str) or len(v)>200 for v in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid observation identities')
    if ev['role']!='training' or ev['membershipDigest']!=df.membership_digest(ids):raise ValueError('Invalid fit membership')
    if 'inputFittedStateDigests' in state and (not isinstance(state['inputFittedStateDigests'],list) or len(state['inputFittedStateDigests'])>128 or any(not isinstance(v,str) or len(v)!=64 for v in state['inputFittedStateDigests'])):raise ValueError('Invalid fitted input state chain')
    if operator==SCALER:
        if len(state['means'])!=len(state['columns']) or len(state['scales'])!=len(state['columns']) or any(not numeric(v) for v in state['means']+state['scales']) or any(v<=0 for v in state['scales']):raise ValueError('Invalid scaler parameters')
        expected=[dict(c) for c in state['inputSchema']]
        for col in expected:
            if col['name'] in state['columns']:
                if col['dtype'] not in ('float64','int64'):raise ValueError('Scaler requires numeric schema')
                col['dtype']='float64'
    elif operator==IMPUTER:
        config=state['configuration'];df.schema_validate(config,df.DEFINITIONS[operator]['configSchema'])
        if config['columns']!=state['columns'] or len(state['fillValues'])!=len(state['columns']) or len(state['observedCounts'])!=len(state['columns']):raise ValueError('Invalid imputer state dimensions')
        if any(type(c) is not int or not 0<=c<=len(ids) for c in state['observedCounts']):raise ValueError('Invalid imputer observation counts')
        if state['emptyColumns']!=[name for name,count in zip(state['columns'],state['observedCounts']) if count==0]:raise ValueError('Invalid empty-column evidence')
        expected=[dict(c) for c in state['inputSchema']]
        for name,value in zip(state['columns'],state['fillValues']):
            col=next(c for c in expected if c['name']==name)
            if config['strategy'] in ('mean','median'):
                if col['dtype'] not in ('float64','int64') or not numeric(value):raise ValueError('Invalid numeric imputer statistic')
                col['dtype']='float64'
            elif not typed(value,col['dtype']):raise ValueError('Invalid imputer fill dtype')
            col['nullable']=False
    else:
        config=state['configuration'];df.schema_validate(config,df.DEFINITIONS[operator]['configSchema'])
        if config['columns']!=state['columns'] or len(state['vocabularies'])!=len(state['columns']):raise ValueError('Invalid encoder state dimensions')
        for name,vocab in zip(state['columns'],state['vocabularies']):
            if schema_by_name[name]['dtype']!='string' or not isinstance(vocab,list) or not 1<=len(vocab)<=config['maxCategories'] or any(not isinstance(v,str) for v in vocab) or len(set(vocab))!=len(vocab):raise ValueError('Invalid encoder vocabulary')
            if config['vocabularyOrder']=='lexicographic' and vocab!=sorted(vocab):raise ValueError('Encoder vocabulary order mismatch')
        expected=encoder_schema(state['inputSchema'],state['columns'],state['vocabularies'])
    if 'outputSchema' in state and state['outputSchema']!=expected:raise ValueError('Fitted output schema differs from learned transform')
    return state


def transform(table,state,graph_digest):
    state=df.load_state(json.dumps(state))
    if state['graphDigest']!=graph_digest or state['fitEvidence']['sourceDigest']!=table.source_digest or state['partitionDigest']!=table.partition_digest or state['inputSchema']!=list(table.columns):raise ValueError('Stale source, schema, partition or graph state')
    if 'inputFittedStateDigests' in state and state['inputFittedStateDigests']!=list(table.fitted_states):raise ValueError('Input fitted-state chain differs from fit input')
    fitids=set(state['fitEvidence']['observationIDs']);ids=set(table.row_ids)
    if table.role=='training':
        if not ids<=fitids:raise ValueError('Training rows outside fitted membership')
        if table.data_version!=state['fitDataVersion']:raise ValueError('Training data version differs from fitted observations')
    elif table.role not in ('validation','test') or ids&fitids:raise ValueError('Heldout fit leakage or missing partition role')
    rows,cols,diagnostics=portable.apply(table.rows,list(table.columns),_portable_parameters(state))
    return df.derive(table,rows=map(tuple,rows),columns=cols,fitted_states=table.fitted_states+(state['contentDigest'],),diagnostics=table.diagnostics+tuple(diagnostics))


def portable_parameters(state):
    """Validate full native authority evidence before projecting learned parameters."""
    if state.get('operator')==df.PREFIX+'preprocess':
        if __package__:from .dataflow_preprocess import load as load_preprocess,parameters
        else:from dataflow_preprocess import load as load_preprocess,parameters
        return parameters(load_preprocess(json.dumps(state)))
    return _portable_parameters(load(json.dumps(state)))


def _portable_parameters(state):
    operator=state['operator'].removeprefix(df.PREFIX)
    result={'schemaVersion':1,'kind':'portableFittedParameters','operator':state['operator'],
            'columns':state['columns'],'inputSchema':state['inputSchema'],
            'sourceContentDigest':state['contentDigest']}
    if operator==SCALER:
        result.update(means=state['means'],scales=state['scales'])
        output=[{**c,'dtype':'float64'} if c['name'] in state['columns'] else dict(c) for c in state['inputSchema']]
    elif operator==IMPUTER:
        result.update(strategy=state['configuration']['strategy'],fillValues=state['fillValues'])
        output=state['outputSchema']
    else:
        result.update(unknownPolicy=state['configuration']['unknownPolicy'],vocabularies=state['vocabularies'])
        output=state['outputSchema']
    result['outputSchema']=output
    return portable.validate(json.loads(json.dumps(result)))
