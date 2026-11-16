"""Text encoding's document mode: frozen processor, row-preserving summaries.

Keep source text with its processor identity rather than expanding vocabulary to table
columns. Estimator materializes bounded CSR only inside its isolated fitting operation.
"""
import json
if __package__:
    from . import dataflow as df,document_classification as dc
else:
    import dataflow as df
    import document_classification as dc


def validate_config(c):
    dc.validate_features(c['featureSettings'])
    dc.require(c.get('mode')=='documentFeatures' and c['vocabularySource']=='training' and c['keepSourceText'],
               'Document features fit on Training and retain source text for the frozen processor.')
    return c


def schema(table,c):
    index=table.index(c['textColumn']);dc.require(table.columns[index]['dtype']=='string','Select an actual text column.')
    name=c['outputColumn']+'_nonzero';dc.require(name not in [x['name'] for x in table.columns],'Feature summary name collides with a source column.')
    columns=tuple(table.columns)+({'name':name,'dtype':'int64','nullable':False},)
    dc.require(len(columns)<=df.MAX_COLUMNS,'Feature summary exceeds the table column budget.')
    return columns,index


def apply(args,c,graph,check):
    validate_config(c);check();dc.require('training' in args and set(args)<= {'training','validation','test'},'Connect Training and optional Validation/Test.')
    primary=args['training'];columns,index=schema(primary,c)
    dc.require(primary.role=='training' and primary.partition_digest,'Fit document features on a declared Training partition.')
    ids=set();contents={};texts=[]
    for role,table in args.items():
        dc.require(table.role==role and table.source_digest==primary.source_digest and table.partition_digest==primary.partition_digest and table.columns==primary.columns and table.fitted_states==primary.fitted_states,'Text feature partitions must share source, schema, partition and fitted state.')
        for i,(row,rid) in enumerate(df.row_pairs(table)):
            if i%64==0:check()
            dc.require(rid not in ids,'Document partitions overlap observation identities.');ids.add(rid)
            text=row[index];dc.documents([text],check)
            # Identical raw text in different partitions is leakage, irrespective of filename.
            pin=dc.digest(text);dc.require(pin not in contents or contents[pin]==role,'Identical duplicate document text crosses partitions; group duplicates before splitting.')
            contents[pin]=role
            if role=='training':texts.append(text)
    processor=dc.fit_features(texts,c['featureSettings'],check)
    train_ids=list(primary.row_ids)
    state=dict(schemaVersion=2,operator=df.PREFIX+'textEncode',configuration=dict(c),columns=[c['textColumn']],
               inputSchema=list(primary.columns),outputSchema=list(columns),processor=processor,
               fitEvidence={'role':'training','sourceDigest':primary.source_digest,'membershipDigest':df.membership_digest(train_ids),'observationIDs':train_ids},
               fitDataVersion=primary.data_version,partitionDigest=primary.partition_digest,graphDigest=graph,inputFittedStateDigests=list(primary.fitted_states))
    dc.require(len(df.json.dumps(state).encode())<20_000_000,'Fitted document evidence exceeds its retained state budget.')
    state['contentDigest']=df.digest(state)
    output={role:transform(table,state,graph,check) for role,table in args.items()}
    output['state']=state
    vcols=({'name':'feature','dtype':'int64','nullable':False},{'name':'term','dtype':'string','nullable':False},{'name':'idf','dtype':'float64','nullable':False})
    output['vocabulary']=df.derive(df.Table(vcols,tuple((i,t,processor['idf'][i]) for i,t in enumerate(processor['vocabulary'])),tuple(state['contentDigest']+':'+str(i) for i in range(len(processor['vocabulary']))),state['contentDigest']))
    return output


def load(s):
    keys={'schemaVersion','operator','configuration','columns','inputSchema','outputSchema','processor','fitEvidence','fitDataVersion','partitionDigest','graphDigest','inputFittedStateDigests','contentDigest'}
    dc.require(type(s) is dict and set(s)==keys and s['schemaVersion']==2 and s['operator']==df.PREFIX+'textEncode' and s['contentDigest']==df.digest({k:v for k,v in s.items() if k!='contentDigest'}),'Invalid document feature state identity.')
    validate_config(s['configuration']);dc.validate_processor(s['processor'])
    dc.require(s['processor']['configuration']==s['configuration']['featureSettings'],'Processor settings differ from captured text preparation.')
    df._fitted_module().schema_valid(s['inputSchema']);df._fitted_module().schema_valid(s['outputSchema'])
    ev=s['fitEvidence'];ids=ev.get('observationIDs')
    dc.require(type(ids) is list and ids and len(ids)<=1_000_000 and all(type(x) is str and x and len(x.encode())<=1024 for x in ids) and len(set(ids))==len(ids) and ev.get('role')=='training' and ev.get('membershipDigest')==df.membership_digest(ids),'Invalid document feature fitting membership.')
    for pin in [s['graphDigest'],s['partitionDigest'],ev.get('sourceDigest'),s['fitDataVersion']]+s['inputFittedStateDigests']:
        dc.require(type(pin) is str and len(pin)==64 and all(x in '0123456789abcdef' for x in pin),'Document features require exact provenance pins.')
    return s


def transform(table,state,graph,check=lambda:None):
    state=load(state);c=state['configuration'];columns,index=schema(table,c)
    dc.require(graph==state['graphDigest'] and list(table.columns)==state['inputSchema'] and table.partition_digest==state['partitionDigest'] and table.source_digest==state['fitEvidence']['sourceDigest'] and list(table.fitted_states)==state['inputFittedStateDigests'],'Text feature input identity differs from fitted preparation.')
    if table.role=='training':dc.require(table.data_version==state['fitDataVersion'],'Training population differs from fitted document features.')
    else:dc.require(table.role in ('validation','test') and set(table.row_ids).isdisjoint(state['fitEvidence']['observationIDs']),'Held-out document identities overlap feature fitting.')
    lookup={term:i for i,term in enumerate(state["processor"]["vocabulary"])}
    def pairs():
        batch=[]
        def emit():
            matrix=dc.transform([r[index] for r,_ in batch],state['processor'],check,_validated=True,_lookup=lookup)
            for i,(row,rid) in enumerate(batch):yield tuple(row)+(int(matrix.indptr[i+1]-matrix.indptr[i]),),rid
        for row,rid in df.row_pairs(table):
            batch.append((row,rid))
            if len(batch)>=32:yield from emit();batch=[]
        if batch:yield from emit()
    output=df.streaming_rows(table,pairs(),columns,check)
    return df.derive(output,fitted_states=table.fitted_states+(state['contentDigest'],))
