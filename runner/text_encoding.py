"""Deterministic local text encoding with Training-only vocabulary ownership.

No executable tokenizer imports, pretrained downloads, windowing or target shifts.
Sequences remain typed table cells; application to held-out data never refits.
"""
from collections import Counter
from itertools import zip_longest
import json
import re
import sqlite3
import unicodedata
if __package__:
    from . import dataflow as df
    from .named_inputs import MAX_SEQUENCE
else:
    import dataflow as df
    from named_inputs import MAX_SEQUENCE

DEFAULTS={'textColumn':'text','tokenizer':'character','unicodeNormalization':'none','letterCase':'preserve','whitespace':'preserve','missingText':'error','outputColumn':'tokens','keepSourceText':True,'vocabularySource':'training','vocabulary':[],'minimumFrequency':1,'vocabularyLimit':256,'idOrder':'frequency','unknownPolicy':'unk','pad':True,'bos':False,'eos':False}
if __package__:from .portable_text_encoding import special_tokens,tokens
else:from portable_text_encoding import special_tokens,tokens

CHOICES={'tokenizer':('character','word'),'unicodeNormalization':('none','NFC','NFKC'),'letterCase':('preserve','lower','casefold'),'whitespace':('preserve','trim','collapse'),'missingText':('error','empty'),'vocabularySource':('training','supplied'),'idOrder':('frequency','lexicographic'),'unknownPolicy':('unk','error')}
MAX_DISTINCT=100000


def validate_config(c):
    if type(c) is not dict or set(c)-{'mode','featureSettings'}!=set(DEFAULTS):raise ValueError('Text encoding requires its exact versioned configuration fields.')
    if c.get('mode','tokens') not in ('tokens','documentFeatures'):raise ValueError('Choose token IDs or document features.')
    if c.get('mode')=='documentFeatures':
        if 'featureSettings' not in c:raise ValueError('Document feature settings are required.')
        if __package__:from .document_feature_transform import validate_config as validate_features
        else:from document_feature_transform import validate_config as validate_features
        validate_features(c)
    elif 'featureSettings' in c:raise ValueError('Feature settings require document-feature mode.')
    for key,choices in CHOICES.items():
        if type(c[key]) is not str or c[key] not in choices:raise ValueError('Invalid Text encoding choice: '+key)
    for key in ('keepSourceText','pad','bos','eos'):
        if type(c[key]) is not bool:raise ValueError('Text encoding checkboxes require Boolean values.')
    for key in ('textColumn','outputColumn'):
        if type(c[key]) is not str or not c[key].strip() or len(c[key])>200 or '\0' in c[key]:raise ValueError('Text encoding column names must be nonblank and bounded.')
    for key,lo,hi in [('minimumFrequency',1,1000000),('vocabularyLimit',1,65536)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError('Text encoding count is outside its declared bounds.')
    v=c['vocabulary']
    if type(v) is not list or len(v)>65536 or any(type(t) is not str or not t or len(t.encode('utf-8'))>65536 for t in v) or len(set(v))!=len(v) or len(json.dumps(v,ensure_ascii=True))>4000000:raise ValueError('Supply unique bounded vocabulary tokens.')
    if c['vocabularySource']=='training' and v:raise ValueError('Training-fit mode cannot also supply a vocabulary.')
    if c['vocabularySource']=='supplied' and not v:raise ValueError('Supply a nonempty vocabulary before encoding.')
    if c['vocabularyLimit']<len(special_tokens(c)):raise ValueError('Vocabulary limit must include reserved special tokens.')
    return c








def ports(c,connected=None):
    if c['vocabularySource']=='supplied':return {'table'},{'table'},{'table','state','vocabulary'}
    return {'training','validation','test'},{'training'},{'training','state','vocabulary'}|({'validation','test'} if connected is None else set(connected)&{'validation','test'})


def _schema(table,c):
    index=table.index(c['textColumn'])
    if table.columns[index]['dtype']!='string':raise ValueError('Select an actual text column.')
    columns=[dict(x) for x in table.columns if c['keepSourceText'] or x['name']!=c['textColumn']]
    if c['outputColumn'] in [x['name'] for x in columns] or c['outputColumn']+'_length' in [x['name'] for x in columns]:raise ValueError('Token output names collide with existing columns; choose another name.')
    columns += [{'name':c['outputColumn'],'dtype':'int64Sequence','nullable':False},{'name':c['outputColumn']+'_length','dtype':'int64','nullable':False}]
    if len(columns)>df.MAX_COLUMNS:raise ValueError('Text encoding exceeds the 128-column table bound.')
    return tuple(columns),index


def apply(args,c,graph,check):
    validate_config(c);check()
    if c.get('mode')=='documentFeatures':
        if __package__:from .document_feature_transform import apply as apply_features
        else:from document_feature_transform import apply as apply_features
        return apply_features(args,c,graph,check)
    allowed,required,_=ports(c,args)
    if not required<=set(args)<=allowed:raise ValueError('Connect Training with optional held-out tables, or Table for a supplied vocabulary.')
    primary=args['training'] if c['vocabularySource']=='training' else args['table']
    columns,index=_schema(primary,c)
    if c['vocabularySource']=='training':
        if primary.role!='training' or not primary.partition_digest:raise ValueError('Fit vocabulary only on a verified Training partition.')
        membership=df.DiskStore(check)
        try:
            db=membership.connection;db.execute('CREATE TABLE populations (identity TEXT PRIMARY KEY COLLATE BINARY) WITHOUT ROWID')
            for role,table in args.items():
                if table.role!=role or table.source_digest!=primary.source_digest or table.partition_digest!=primary.partition_digest or table.columns!=primary.columns or table.fitted_states!=primary.fitted_states:raise ValueError('Text partitions must share source, schema, partition and fitted-state identity.')
                for i,rid in enumerate(table.row_ids):
                    if i%128==0:check();membership._allocation_limit();membership._account()
                    try:db.execute('INSERT INTO populations VALUES (?)',(rid,))
                    except sqlite3.IntegrityError as error:raise ValueError('Text partitions overlap or repeat observation identities.') from error
            db.commit();membership._account()
        finally:membership.close()
    frequencies=Counter();vocabulary_bytes=0
    if c['vocabularySource']=='training':
        for i,(row,_) in enumerate(df.row_pairs(primary)):
            if i%64==0:check()
            for token in tokens(row[index],c):
                if token not in frequencies:
                    vocabulary_bytes+=len(token.encode('utf-8'))
                    if vocabulary_bytes>8000000:raise ValueError('Vocabulary fitting exceeds its 8 MB token-text allowance; revise tokenization or supply a vocabulary.')
                frequencies[token]+=1
                if len(frequencies)>MAX_DISTINCT:raise ValueError('Vocabulary fitting exceeds 100000 distinct tokens; revise tokenization or supply a vocabulary.')
        learned=[t for t,n in frequencies.items() if n>=c['minimumFrequency']]
        learned.sort(key=(lambda t:(-frequencies[t],t)) if c['idOrder']=='frequency' else None)
    else:learned=list(c['vocabulary'])
    special=special_tokens(c)
    if set(special)&set(learned):raise ValueError('A literal vocabulary token collides with a reserved special token; disable that role or choose different tokenization.')
    if c['vocabularySource']=='supplied' and len(learned)+len(special)>c['vocabularyLimit']:raise ValueError('Supplied vocabulary exceeds the configured limit; no supplied tokens were discarded.')
    vocabulary=special+learned[:c['vocabularyLimit']-len(special)]
    if not vocabulary:raise ValueError('Text vocabulary is empty after frequency filtering.')
    state={'schemaVersion':1,'operator':df.PREFIX+'textEncode','configuration':dict(c),'columns':[c['textColumn']],'inputSchema':list(primary.columns),'outputSchema':list(columns),'vocabulary':vocabulary,'frequencies':[0]*len(special)+[frequencies[t] for t in vocabulary[len(special):]],'specialTokenCount':len(special),'fitEvidence':primary.evidence(),'fitDataVersion':primary.data_version,'partitionDigest':primary.partition_digest,'graphDigest':graph,'inputFittedStateDigests':list(primary.fitted_states),'unicodeVersion':unicodedata.unidata_version}
    # Materialize evidence into bounded, portable JSON; no row store escapes as state.
    prefix=primary.row_ids.store.identity_prefix if isinstance(primary.row_ids,df.StoredIDs) else ''
    ids=[];identity_bytes=len(prefix.encode())
    for i,rid in enumerate(primary.row_ids):
        if i%128==0:check()
        if not rid.startswith(prefix):raise ValueError('Text observation identity differs from its captured prefix.')
        suffix=rid[len(prefix):];identity_bytes+=len(json.dumps(suffix).encode())+1
        if identity_bytes>16000000:raise ValueError('Vocabulary fitting identities exceed their 16 MB state allowance.')
        ids.append(suffix)
    if __package__:from .dataflow_preprocess import _membership
    else:from dataflow_preprocess import _membership
    state['fitEvidence']={'role':primary.role,'sourceDigest':primary.source_digest,'membershipDigest':_membership(prefix,ids),**({'observationIDPrefix':prefix,'observationIDSuffixes':ids} if prefix else {'observationIDs':ids})}
    state['contentDigest']=df.digest(state)
    outputs={'state':state}
    for role,table in args.items():outputs[role]=transform(table,state,graph,check)
    vcols=tuple({'name':n,'dtype':t,'nullable':False} for n,t in [('id','int64'),('token','string'),('frequency','int64'),('special','bool')])
    outputs['vocabulary']=df.derive(df.Table(vcols,tuple((i,t,state['frequencies'][i],i<len(special)) for i,t in enumerate(vocabulary)),tuple(state['contentDigest']+':'+str(i) for i in range(len(vocabulary))),state['contentDigest']))
    return outputs


def transform(table,state,graph,check=lambda:None):
    if state.get('schemaVersion')==2:
        if __package__:from .document_feature_transform import transform as apply_features
        else:from document_feature_transform import transform as apply_features
        return apply_features(table,state,graph,check)
    state=load(json.dumps(state));c=state['configuration']
    if graph!=state['graphDigest'] or list(table.columns)!=state['inputSchema'] or list(table.fitted_states)!=state['inputFittedStateDigests'] or table.source_digest!=state['fitEvidence']['sourceDigest'] or table.partition_digest!=state['partitionDigest']:raise ValueError('Text source/schema/preparation identity differs from the fitted vocabulary.')
    if c['vocabularySource']=='training':
        if table.role=='training':
            if table.data_version!=state['fitDataVersion']:raise ValueError('Training text differs from the fitted vocabulary population.')
        elif table.role not in ('validation','test') or set(table.row_ids)&set(df.fit_observation_ids(state)):raise ValueError('Held-out text overlaps vocabulary fitting.')
    columns,index=_schema(table,c);lookup={t:i for i,t in enumerate(state['vocabulary'])};unknown=lookup.get('<unk>') if c['unknownPolicy']=='unk' else None
    def pairs():
        for i,(row,rid) in enumerate(df.row_pairs(table)):
            if i%64==0:check()
            ids=[]
            if c['bos']:ids.append(lookup['<bos>'])
            for t in tokens(row[index],c):
                if t not in lookup and unknown is None:raise ValueError(f'Unknown token in text record {i+1}; supply coverage or enable <unk>.')
                ids.append(lookup.get(t,unknown))
            if c['eos']:ids.append(lookup['<eos>'])
            values=[v for j,v in enumerate(row) if c['keepSourceText'] or j!=index]
            yield values+[ids,len(ids)],rid
    output=df.streaming_rows(table,pairs(),columns,check)
    # Supplied vocabulary is deterministic configuration, not training-fitted state.
    return df.derive(output,fitted_states=table.fitted_states+((state['contentDigest'],) if c['vocabularySource']=='training' else ()))


def load(text):
    if type(text) is not str or len(text.encode())>20000000:raise ValueError('Text vocabulary state exceeds 20 MB.')
    s=json.loads(text)
    if type(s) is dict and s.get('schemaVersion')==2:
        if __package__:from .document_feature_transform import load as load_features
        else:from document_feature_transform import load as load_features
        return load_features(s)
    keys={'schemaVersion','operator','configuration','columns','inputSchema','outputSchema','vocabulary','frequencies','specialTokenCount','fitEvidence','fitDataVersion','partitionDigest','graphDigest','inputFittedStateDigests','unicodeVersion','contentDigest'}
    if type(s) is not dict or set(s)!=keys or s['schemaVersion']!=1 or s['operator']!=df.PREFIX+'textEncode' or s['contentDigest']!=df.digest({k:v for k,v in s.items() if k!='contentDigest'}):raise ValueError('Text vocabulary state identity is invalid.')
    c=validate_config(s['configuration']);v=s['vocabulary'];special=special_tokens(c)
    if type(v) is not list or not 1<=len(v)<=c['vocabularyLimit'] or any(type(t) is not str or not t or len(t.encode())>65536 for t in v) or len(set(v))!=len(v) or v[:len(special)]!=special or s['specialTokenCount']!=len(special):raise ValueError('Text vocabulary ordering/special roles are invalid.')
    if type(s['frequencies']) is not list or len(s['frequencies'])!=len(v) or any(type(n) is not int or not 0<=n<2**63 for n in s['frequencies']):raise ValueError('Vocabulary frequencies are invalid.')
    df._fitted_module().schema_valid(s['inputSchema']);df._fitted_module().schema_valid(s['outputSchema'])
    ev=s['fitEvidence'];prefix=ev.get('observationIDPrefix','');ids=ev.get('observationIDSuffixes') if 'observationIDSuffixes' in ev else ev.get('observationIDs')
    if type(prefix) is not str or len(prefix)>1024:raise ValueError('Invalid vocabulary identity prefix.')
    if __package__:from .dataflow_preprocess import _membership
    else:from dataflow_preprocess import _membership
    if type(ids) is not list or len(ids)>df.MAX_TABLE_CELLS or any(type(i) is not str or not prefix+i or len(prefix+i)>1024 for i in ids) or len(set(ids))!=len(ids) or ev.get('membershipDigest')!=_membership(prefix,ids):raise ValueError('Vocabulary observation evidence is invalid.')
    if c['vocabularySource']=='training' and ev.get('role')!='training':raise ValueError('Vocabulary was not fitted on Training.')
    if s['unicodeVersion']!=unicodedata.unidata_version:raise ValueError('Tokenizer Unicode version differs from captured preparation.')
    return s


def decode(ids,state,skip_special=True):
    state=load(json.dumps(state));v=state['vocabulary'];special=state['specialTokenCount']
    if type(ids) not in (list,tuple) or len(ids)>MAX_SEQUENCE or any(type(i) is not int or not 0<=i<len(v) for i in ids):raise ValueError('Decode requires valid bounded vocabulary IDs.')
    return ''.join(v[i] for i in ids if not skip_special or i>=special)


def verify_output(args,c,graph,outputs,check):
    expected=apply(args,c,graph,check)
    if set(expected)!=set(outputs):raise ValueError('Cached text ports differ from the current configuration.')
    for port,value in expected.items():
        actual=outputs[port]
        if port=='state':
            if actual!=value:raise ValueError('Cached vocabulary differs from Training/configuration.')
        elif actual.columns!=value.columns or actual.data_version!=value.data_version or actual.fitted_states!=value.fitted_states or actual.role!=value.role or actual.partition_digest!=value.partition_digest:raise ValueError('Cached text output values or lineage differ.')
