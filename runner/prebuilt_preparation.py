"""Inert, model-bound learned preparation. Never fits or discovers new categories."""
import hashlib
import json
from pathlib import Path
if __package__:
    from . import prebuilt_models as pb, preprocess_parameters as pp
    from .dataflow_storage import bounded_regular_read
else:
    import prebuilt_models as pb, preprocess_parameters as pp
    from dataflow_storage import bounded_regular_read

MAX_BYTES=20_000_000
MAX_CHAIN=16
KIND='kwipml.prebuilt-preparation.v1'
FIELDS={'schemaVersion','kind','modelSHA256','featureColumns','inputSchema','outputSchema','parameters','contentDigest'}

def digest(value):return hashlib.sha256(pb.canonical(value)).hexdigest()
def is_digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def validate(value,model_raw):
    pb.require(type(value) is dict and set(value)==FIELDS and type(value['schemaVersion']) is int and value['schemaVersion']==1 and value['kind']==KIND,'Unsupported learned-preparation envelope.')
    pb.require(len(pb.canonical(value))<=MAX_BYTES,'Learned preparation exceeds 20MB.')
    pb.require(is_digest(value['contentDigest']) and value['contentDigest']==digest({k:v for k,v in value.items() if k!='contentDigest'}),'Learned-preparation content digest differs.')
    pb.require(value['modelSHA256']==hashlib.sha256(model_raw).hexdigest(),'Learned preparation belongs to a different model.')
    model=pb.load(model_raw)
    pb.require(value['featureColumns']==model['featureColumns'],'Learned-preparation feature order differs from model.')
    chain=value['parameters'];pb.require(type(chain) is list and 1<=len(chain)<=MAX_CHAIN,'Learned preparation needs 1–16 complete supported steps.')
    previous=None;seen=set()
    for entry in chain:
        pp.validate(entry)
        pb.require(entry['configuration']['mode']=='partitions','Only Training-fitted Preprocess rules are reusable.')
        key=entry['sourceContentDigest'];pb.require(key not in seen,'Duplicate learned-preparation step.');seen.add(key)
        if previous is not None:
            by_name={c['name']:c for c in previous}
            pb.require(all(by_name.get(c['name'])==c for c in entry['inputSchema']),'Learned-preparation chain schema/order differs.')
        previous=entry['outputSchema']
    pb.require(value['inputSchema']==chain[0]['inputSchema'] and value['outputSchema']==previous,'Learned-preparation boundary differs.')
    schema={c['name']:c for c in previous}
    pb.require(all(n in schema and schema[n]['dtype'] in ('float64','int64') for n in model['featureColumns']),'Learned preparation does not produce every numeric model feature.')
    return value

def project(parameters,feature_columns):
    """Retain dependencies of model features, not training targets or unrelated columns."""
    required=set(feature_columns);result=[]
    for p in reversed(parameters):
        pp.validate(p);rules={r['column']:r for r in p['rules']};selected=[]
        for column in p['inputSchema']:
            rule=rules.get(column['name'])
            outputs=pp.output_schema([column],{'mode':'partitions','columns':[r for r in p['configuration']['columns'] if r['column']==column['name']]},[rule] if rule else [])
            if required & {c['name'] for c in outputs}:selected.append(column)
        names={c['name'] for c in selected}
        config={'mode':'partitions','columns':[r for r in p['configuration']['columns'] if r['column'] in names]}
        selected_rules=[r for r in p['rules'] if r['column'] in names]
        outputs=pp.output_schema(selected,config,selected_rules)
        pb.require(required<={c['name'] for c in outputs},'Saved model features are not produced by this preparation chain.')
        entry=dict(p,columns=[r['column'] for r in selected_rules],inputSchema=selected,outputSchema=outputs,configuration=config,rules=selected_rules)
        pp.validate(entry);result.append(entry);required=names
    return list(reversed(result))

def create(model_raw,parameters):
    pb.require(type(parameters) is list and parameters,'No learned preparation is available.')
    model=pb.load(model_raw);parameters=project(parameters,model['featureColumns'])
    value=dict(schemaVersion=1,kind=KIND,modelSHA256=hashlib.sha256(model_raw).hexdigest(),featureColumns=model['featureColumns'],inputSchema=parameters[0]['inputSchema'],outputSchema=parameters[-1]['outputSchema'],parameters=parameters)
    value['contentDigest']=digest(value)
    return validate(value,model_raw)

def load(path,model_raw):
    raw=bounded_regular_read(path,MAX_BYTES)
    return validate(json.loads(raw),model_raw)

def apply(table,value,model_raw,check=lambda:None):
    if __package__:from . import dataflow as df, dataflow_preprocess as dp
    else:import dataflow as df, dataflow_preprocess as dp
    value=validate(value,model_raw)
    chain=value['parameters'];keys=[p['sourceContentDigest'] for p in chain]
    pb.require(not set(keys+[value['contentDigest']])&set(table.fitted_states),'Saved preparation has already been applied or partially applied.')
    current=table
    for p in chain:
        check();actual=list(current.columns);by_name={c['name']:c for c in actual}
        pb.require(len(by_name)==len(actual),'Duplicate input column names.')
        # Display labels do not change scalar semantics. Required feature names,
        # types, nullability and temporal contracts must match; extras pass through.
        contract=lambda c:{k:v for k,v in c.items() if k not in ('displayLabel','nullable')}
        # A new population with no missing values satisfies a nullable input.
        # Never widen a saved nonnullable requirement or change scalar semantics.
        pb.require(all(c['name'] in by_name and contract(c)==contract(by_name[c['name']]) and
            (c['nullable'] or not by_name[c['name']]['nullable']) for c in p['inputSchema']),'Input schema differs from saved required features; prepare column types explicitly.')
        columns=pp.output_schema(actual,p['configuration'],p['rules'])
        current=dp._transform(current,p['configuration'],p['rules'],columns,check,{'contentDigest':p['sourceContentDigest']})
    return df.replace(current,fitted_states=current.fitted_states+(value['contentDigest'],))

# The native broker owns these copies. Graph configuration never contains paths.
from contextvars import ContextVar
from contextlib import contextmanager
_context=ContextVar('prebuilt_preparation_job',default=None)
@contextmanager
def job_scope(root,check):
    root=Path(root);manifest=root/'preparation-models.json'
    pins=json.loads(bounded_regular_read(manifest,1_000_000)) if manifest.exists() else {}
    pb.require(type(pins) is dict and len(pins)<=128,'Invalid captured preparation resources.')
    token=_context.set((root,pins,check))
    try:yield
    finally:_context.reset(token)

def captured(node):
    ctx=_context.get();pb.require(ctx is not None and node in ctx[1],'Connect a verified Load model with learned preparation.')
    root,pins,check=ctx;pin=pins[node]
    pb.require(type(pin) is dict and set(pin)=={'path','sha256','preparationPath','preparationSHA256'},'Invalid captured preparation pin.')
    def read(key,hashkey,limit):
        relative=Path(pin[key]);pb.require(not relative.is_absolute() and '..' not in relative.parts and len(relative.parts)==3 and relative.parts[0]=='preparation-models','Unsafe captured model path.')
        path=root/relative
        pb.require(not any(p.is_symlink() for p in [path,path.parent,path.parent.parent]),'Captured model resources cannot be symlinks.')
        check();raw=bounded_regular_read(path,limit);pb.require(hashlib.sha256(raw).hexdigest()==pin[hashkey],'Captured learned model resource changed.');return raw
    model=read('path','sha256',pb.MAX_STATE);raw=read('preparationPath','preparationSHA256',MAX_BYTES)
    return validate(json.loads(raw),model),model
