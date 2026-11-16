"""Single-node, training-only preprocessing with bounded disk statistics and streamed outputs."""
import json
import hashlib
import sqlite3
import math
if __package__:
    from . import dataflow as df
    from . import preprocess_parameters as portable
else:
    import dataflow as df
    import preprocess_parameters as portable
rule_defaults=portable.rule_defaults
def validate_config(config):
    if type(config) is dict and config.get('mode')=='frozen':
        if set(config)!={'mode','columns'} or config['columns']!=[]:raise ValueError('Apply saved rules has no editable learned column rules.')
        return config
    return portable.validate_config(config)


def ports(config,connected=None):
    if config['mode'] in ('table','frozen'):return {'table'},{'table'},{'table'}
    inputs={'training','validation','test'}
    return inputs,{'training'}, {'state','training'}|({'validation','test'} if connected is None else set(connected)&{'validation','test'})


def _retained(row,schema,config):
    drop={r['column'] for r in config['columns'] if r['missing']=='drop'}
    rules={r['column']:r for r in config['columns']}
    return not any((portable.clean_text(v,rules.get(c['name'],{})) if c['dtype']=='string' else v) is None and c['name'] in drop for v,c in zip(row,schema))


def _fit(table,config,graph,check):
    schema=portable.validate_schema(list(table.columns),config)
    rules=[dict(r,fill=r['value'] if r['missing']=='constant' else None,vocabulary=[],mean=0.,scale=1.) for r in config['columns']]
    indices=[table.index(r['column']) for r in rules]
    statistics=[dict(n=0,mean=0.,m2=0.) for _ in rules]
    store=df.DiskStore(check);db=store.connection
    db.execute('CREATE TABLE values_for_median (col INTEGER, value REAL)')
    db.execute('CREATE INDEX median_order ON values_for_median(col,value)')
    db.execute('CREATE TABLE frequencies (col INTEGER, value TEXT COLLATE BINARY, count INTEGER, PRIMARY KEY(col,value)) WITHOUT ROWID')
    retained=0
    try:
        for position,(row,_) in enumerate(df.row_pairs(table)):
            if position%128==0:check();store._allocation_limit();store._account()
            if not _retained(row,schema,config):continue
            retained+=1
            for k,(i,r,stat) in enumerate(zip(indices,rules,statistics)):
                value=row[i]
                if value is None:continue
                if schema[i]['dtype'] not in ('image','audio','tensor','date','datetime','time') and not portable.typed(value,schema[i]['dtype']):raise ValueError('Preprocessing input violates finite scalar type: '+r['column'])
                stat['n']+=1
                if r['missing'] in ('mean','median'):
                    delta=float(value)-stat['mean'];stat['mean']+=delta/stat['n']
                    if not math.isfinite(stat['mean']):raise ValueError('Imputation statistic overflow.')
                    if r['missing']=='median':db.execute('INSERT INTO values_for_median VALUES (?,?)',(k,float(value)))
                elif r['missing']=='mostFrequent':
                    db.execute('INSERT INTO frequencies VALUES (?,?,1) ON CONFLICT(col,value) DO UPDATE SET count=count+1',(k,json.dumps(value,ensure_ascii=False,allow_nan=False)))
        check();db.commit();store._account()
        if not retained:raise ValueError('Preprocessing needs retained training observations; revise missing-row exclusions.')
        for k,(i,r,stat) in enumerate(zip(indices,rules,statistics)):
            strategy=r['missing']
            if strategy not in ('mean','median','mostFrequent'):continue
            if not stat['n']:
                if r['emptyPolicy']=='error':raise ValueError('Entirely missing training column: '+r['column'])
                fill=r['emptyValue']
            elif strategy=='mean':fill=stat['mean']
            elif strategy=='median':
                n=stat['n'];middle=[v[0] for v in db.execute('SELECT value FROM values_for_median WHERE col=? ORDER BY value LIMIT ? OFFSET ?',(k,2 if n%2==0 else 1,(n-1)//2))]
                fill=sum(v/len(middle) for v in middle)
            else:
                maximum=db.execute('SELECT MAX(count) FROM frequencies WHERE col=?',(k,)).fetchone()[0]
                ties=0;fill=None
                for (raw,) in db.execute('SELECT value FROM frequencies WHERE col=? AND count=?',(k,maximum)):
                    check();v=json.loads(raw);ties+=1
                    if fill is None or v<fill:fill=v
                if ties>1 and r['tiePolicy']=='error':raise ValueError('Most-frequent tie in training column: '+r['column'])
            r['fill']=float(fill) if strategy in ('mean','median') else fill
        # Learn vocabulary and population scaling only after row exclusion and imputation.
        statistics=[dict(n=0,mean=0.,m2=0.) for _ in rules]
        for position,(row,_) in enumerate(df.row_pairs(table)):
            if position%128==0:check()
            if not _retained(row,schema,config):continue
            for i,r,stat in zip(indices,rules,statistics):
                v=row[i]
                if v is None and r['missing'] not in ('keep','drop','error'):v=r['fill']
                if v is None and r['missing']=='error':raise ValueError('Missing training value: '+r['column'])
                if r['encoding']=='oneHot':
                    if type(v) is not str:raise ValueError('Resolve missing categories before encoding: '+r['column'])
                    if v not in r['vocabulary']:
                        r['vocabulary'].append(v)
                        if len(r['vocabulary'])>r['maxCategories']:raise ValueError('Training vocabulary exceeds maximum categories: '+r['column'])
                if r['scaling']=='standard':
                    if not portable.numeric(v):raise ValueError('Scaling requires finite nonmissing training data: '+r['column'])
                    stat['n']+=1;delta=float(v)-stat['mean'];stat['mean']+=delta/stat['n'];stat['m2']+=delta*(float(v)-stat['mean'])
        for r,stat in zip(rules,statistics):
            if r['vocabularyOrder']=='lexicographic':r['vocabulary'].sort()
            if r['scaling']=='standard':
                r['mean']=stat['mean'] if r['center'] else 0.
                r['scale']=(math.sqrt(max(0.,stat['m2']/stat['n'])) or 1.) if r['unitVariance'] else 1.
        columns=portable.output_schema(schema,config,rules)
        # Evidence includes exactly the retained training rows used for statistics.
        prefix=table.row_ids.store.identity_prefix if isinstance(table.row_ids,df.StoredIDs) else ''
        fit_ids=[];identity_bytes=len(prefix.encode())
        for row,rid in df.row_pairs(table):
            check()
            if _retained(row,schema,config):
                if not rid.startswith(prefix):raise ValueError('Training identity differs from its verified prefix.')
                suffix=rid[len(prefix):]
                identity_bytes+=len(json.dumps(suffix).encode())+1
                if identity_bytes>16_000_000:raise ValueError('Retained training identities exceed the bounded learned-state budget; use fewer training observations or shorter identities.')
                fit_ids.append(suffix)
        state=dict(schemaVersion=1,operator=df.PREFIX+'preprocess',columns=[r['column'] for r in rules],inputSchema=schema,outputSchema=columns,
                   configuration=config,rules=rules,fitEvidence={'role':'training','sourceDigest':table.source_digest,'membershipDigest':_membership(prefix,fit_ids),**({'observationIDPrefix':prefix,'observationIDSuffixes':fit_ids} if prefix else {'observationIDs':fit_ids})},
                   fitDataVersion=table.data_version,partitionDigest=table.partition_digest,graphDigest=graph,inputFittedStateDigests=list(table.fitted_states))
        state['contentDigest']=df.digest(state)
        return load(json.dumps(state,allow_nan=False))
    finally:store.close()


def _membership(prefix,suffixes):
    # A common prefix does not change lexical ordering; keep only compact suffixes while sorting.
    hasher=hashlib.sha256(b'kwip.membership.v1\n')
    for suffix in sorted(suffixes,key=lambda v:v.encode()):
        raw=(prefix+suffix).encode();hasher.update(str(len(raw)).encode()+b':'+raw)
    return hasher.hexdigest()


def parameters(state):
    return portable.validate(dict(schemaVersion=1,kind='portableFittedParameters',operator=df.PREFIX+'preprocess',columns=state['columns'],inputSchema=state['inputSchema'],outputSchema=state['outputSchema'],sourceContentDigest=state['contentDigest'],configuration=state['configuration'],rules=state['rules']))


def load(text):
    if len(text.encode())>20_000_000:raise ValueError('Preprocessing state exceeds 20 MB.')
    s=json.loads(text)
    expected={'schemaVersion','operator','columns','inputSchema','outputSchema','configuration','rules','fitEvidence','fitDataVersion','partitionDigest','graphDigest','inputFittedStateDigests','contentDigest'}
    if type(s) is not dict or set(s)!=expected or s['schemaVersion']!=1 or s['operator']!=df.PREFIX+'preprocess' or s['configuration']['mode']!='partitions' or s['contentDigest']!=df.digest({k:v for k,v in s.items() if k!='contentDigest'}):raise ValueError('Invalid learned preprocessing state/digest.')
    parameters(s);df._fitted_module().schema_valid(s['inputSchema']);df._fitted_module().schema_valid(s['outputSchema'])
    ev=s['fitEvidence'];prefix=ev.get('observationIDPrefix','');ids=ev.get('observationIDSuffixes') if 'observationIDSuffixes' in ev else ev.get('observationIDs')
    keys={'role','sourceDigest','membershipDigest'}|({'observationIDPrefix','observationIDSuffixes'} if 'observationIDSuffixes' in ev else {'observationIDs'})
    if set(ev)!=keys or ev['role']!='training' or type(prefix) is not str or '\0' in prefix or len(prefix.encode())>1024 or type(ids) is not list or not ids or len(ids)>df.MAX_TABLE_CELLS or any(type(v) is not str or not (prefix+v) or '\0' in v or len((prefix+v).encode())>1024 for v in ids) or len(set(ids))!=len(ids) or ev['membershipDigest']!=_membership(prefix,ids):raise ValueError('Invalid preprocessing training evidence.')
    if type(s['inputFittedStateDigests']) is not list or len(s['inputFittedStateDigests'])>127 or len(set(s['inputFittedStateDigests']))!=len(s['inputFittedStateDigests']):raise ValueError('Invalid prior preprocessing state chain.')
    for value in [s['fitDataVersion'],s['partitionDigest'],s['graphDigest'],ev['sourceDigest'],*s['inputFittedStateDigests']]:
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid preprocessing lineage digest.')
    return s


def _transform(table,config,rules,columns,check,state=None):
    counts={};retained=[0];input_schema=list(table.columns)
    def pairs():
        for position,(row,rid) in enumerate(df.row_pairs(table)):
            if position%128==0:check()
            try:value=portable.transform_row(row,input_schema,rules,counts)
            except ValueError as error:raise ValueError(f'{table.role} row {position+1}: {error}') from error
            if value is not None:retained[0]+=1;yield value,rid
    out=df.streaming_rows(table,pairs(),columns=columns,check_cancelled=check)
    diagnostics=[dict(code=code,column=column,count=count,**({'stateDigest':state['contentDigest']} if state else {})) for (code,column),count in counts.items()]
    if retained[0]!=len(table.rows):diagnostics.append(dict(code='missingRowsExcluded',count=len(table.rows)-retained[0]))
    return df.replace(out,fitted_states=table.fitted_states+((state['contentDigest'],) if state else ()),diagnostics=table.diagnostics+tuple(diagnostics))


def transform(table,state,graph,check=lambda:None):
    state=load(json.dumps(state));parameters(state)
    if state['graphDigest']!=graph or state['fitEvidence']['sourceDigest']!=table.source_digest or state['partitionDigest']!=table.partition_digest or state['inputSchema']!=list(table.columns) or state['inputFittedStateDigests']!=list(table.fitted_states):raise ValueError('Preprocessing source, schema, partition, graph or prior chain differs.')
    fit=set(df.fit_observation_ids(state))
    if table.role=='training':
        if table.data_version!=state['fitDataVersion']:raise ValueError('Training data differs from learned preprocessing.')
    elif table.role not in ('validation','test') or any(rid in fit for rid in table.row_ids):raise ValueError('Preprocessing held-out membership overlaps training.')
    return _transform(table,state['configuration'],state['rules'],state['outputSchema'],check,state)


def apply(args,config,graph,check,node=None):
    check();validate_config(config)
    allowed,required,_=ports(config,args)
    if not required<=set(args)<=allowed:raise ValueError('Connect only the inputs for the selected preprocessing mode.')
    if config['mode']=='frozen':
        if __package__:from . import prebuilt_preparation as frozen
        else:import prebuilt_preparation as frozen
        envelope,model=frozen.captured(node)
        return {'table':frozen.apply(args['table'],envelope,model,check)}
    if config['mode']=='table':
        table=args['table'];schema=portable.validate_schema(list(table.columns),config)
        rules=[dict(r,fill=r['value'] if r['missing']=='constant' else None,vocabulary=[],mean=0.,scale=1.) for r in config['columns']]
        return {'table':_transform(table,config,rules,portable.output_schema(schema,config,rules),check)}
    train=args['training']
    if train.role!='training' or not train.partition_digest:raise ValueError('Learn preprocessing only from a verified Training partition.')
    # Validate all connected populations before fitting or writing outputs.
    membership=df.DiskStore(check)
    try:
        db=membership.connection;db.execute('CREATE TABLE populations (id TEXT PRIMARY KEY COLLATE BINARY) WITHOUT ROWID')
        for role,table in args.items():
            if table.role!=role or table.source_digest!=train.source_digest or table.partition_digest!=train.partition_digest or table.columns!=train.columns or table.fitted_states!=train.fitted_states:raise ValueError('Preprocess inputs must share source, schema, partition and prior fitted chain, with their declared roles.')
            for position,rid in enumerate(table.row_ids):
                if position%128==0:check();membership._allocation_limit();membership._account()
                try:db.execute('INSERT INTO populations VALUES (?)',(rid,))
                except sqlite3.IntegrityError as error:raise ValueError('Preprocess partitions overlap or repeat identities.') from error
        db.commit();membership._account()
    finally:membership.close()
    state=_fit(train,config,graph,check);outputs={'state':state}
    for role,table in args.items():outputs[role]=_transform(table,config,state['rules'],state['outputSchema'],check,state)
    return outputs


def verify_output(args,config,graph,outputs,check):
    if set(outputs)!=ports(config,args)[2]:raise ValueError('Cached Preprocess ports differ.')
    state=load(json.dumps(outputs['state'])) if config['mode']=='partitions' else None
    if state:
        train=args['training']
        if state['partitionDigest']!=train.partition_digest or state['fitEvidence']['sourceDigest']!=train.source_digest:raise ValueError('Cached Preprocess source/partition differs.')
        if state['configuration']!=config or state['graphDigest']!=graph or state['fitDataVersion']!=train.data_version or state['inputSchema']!=list(train.columns) or state['inputFittedStateDigests']!=list(train.fitted_states):raise ValueError('Cached Preprocess training/configuration identity differs.')
        expected_ids=(rid for row,rid in df.row_pairs(train) if _retained(row,list(train.columns),config))
        from itertools import zip_longest
        if any(a!=b for a,b in zip_longest(expected_ids,df.fit_observation_ids(state))):raise ValueError('Cached Preprocess fit membership differs.')
    for role,source in args.items():
        result=outputs[role]
        columns=state['outputSchema'] if state else portable.output_schema(list(source.columns),config,[dict(r,fill=r['value'],vocabulary=[],mean=0.,scale=1.) for r in config['columns']])
        if result.columns!=tuple(columns) or result.source_digest!=source.source_digest or result.role!=source.role or result.partition_digest!=source.partition_digest or result.fitted_states!=source.fitted_states+((state['contentDigest'],) if state else ()):raise ValueError('Cached Preprocess output schema/lineage differs.')
        from itertools import zip_longest
        rules=state['rules'] if state else [dict(r,fill=r['value'] if r['missing']=='constant' else None,vocabulary=[],mean=0.,scale=1.) for r in config['columns']]
        def expected_pairs():
            for row,rid in df.row_pairs(source):
                check()
                transformed=portable.transform_row(row,list(source.columns),rules)
                if transformed is not None:yield transformed,rid
        for actual,expected in zip_longest(df.row_pairs(result),expected_pairs()):
            check()
            if actual!=expected:raise ValueError('Cached Preprocess output values/identity/order differ.')
