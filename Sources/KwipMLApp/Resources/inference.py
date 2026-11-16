"""Portable KwipML inference for already-prepared CPU tensors with an explicit input contract.

Run `python inference.py --example` for synthetic inputs, or --input prepared.json
for a JSON array of prepared feature rows. Raw CSV preprocessing is not included.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import torch

ROOT = Path(__file__).resolve().parent
ALLOWED = {'named_inputs.py','tokenizers.json','portable_text_encoding.py','model.py', 'checkpoint.pt', 'inference.py', 'input-contract.json', 'README.md', 'requirements.txt', 'training_recipe.py', 'rate-trace.json', 'objective_recipe.py', 'initialization_recipe.py', 'gradient_policy.py', 'trainability_recipe.py', 'data_loading_recipe.py'}
REQUIRED = {'model.py', 'checkpoint.pt', 'inference.py', 'input-contract.json'}


def read_regular(path, limit):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError('Export artifact must be a bounded regular file')
    with path.open('rb') as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Export artifact exceeds its size bound')
    return value


def verified_policy_metadata(manifest, files):
    """Admit public policy provenance without importing helpers or exposing derivative samples."""
    for field, helper, fields in (
        ('gradientPolicy', 'gradient_policy.py', {'source', 'recipeSHA256', 'accumulationSteps', 'gradientScale'}),
        ('trainability', 'trainability_recipe.py', {'source', 'recipeSHA256', 'owners'}),
    ):
        if field not in manifest:
            if helper in files:
                raise ValueError('Unselected portable policy helper')
            continue
        value = manifest[field]
        if helper not in files or not isinstance(value, dict) or set(value) != fields:
            raise ValueError('Invalid portable policy provenance')
        source = value['source']
        if not isinstance(source, dict) or set(source) != {'domain', 'localID'} or source['domain'] != 'control' or not isinstance(source['localID'], str) or not source['localID'] or '\x00' in source['localID']:
            raise ValueError('Invalid qualified policy source')
        if not isinstance(value['recipeSHA256'], str) or re.fullmatch('[0-9a-f]{64}', value['recipeSHA256']) is None:
            raise ValueError('Invalid selected recipe hash')
        if field == 'gradientPolicy':
            if type(value['accumulationSteps']) is not int or not 1 <= value['accumulationSteps'] <= 32 or value['gradientScale'] not in ('mean', 'sum'):
                raise ValueError('Invalid public accumulation settings')
        else:
            owners = value['owners']
            if not isinstance(owners, list) or len(owners) > 128:
                raise ValueError('Invalid portable trainability owners')
            seen = set()
            for owner in owners:
                if not isinstance(owner, dict) or set(owner) != {'node', 'parameter', 'mode'} or owner['mode'] not in ('inherit', 'trainable', 'frozen'):
                    raise ValueError('Invalid portable trainability owner')
                if any(not isinstance(owner[key], str) or not owner[key] or '\x00' in owner[key] for key in ('node', 'parameter')):
                    raise ValueError('Invalid portable native owner')
                identity = (owner['node'], owner['parameter'])
                if identity in seen:
                    raise ValueError('Duplicate portable native owner')
                seen.add(identity)


def verified_files():
    manifest = json.loads(read_regular(ROOT / 'export-manifest.json', 1_000_000))
    if manifest.get('schemaVersion') != 1 or manifest.get('kind') != 'kwipmlTrainedModel':
        raise ValueError('Unsupported trained export manifest')
    entries = manifest.get('files')
    if not isinstance(entries, list) or not 4 <= len(entries) <= len(ALLOWED):
        raise ValueError('Invalid export inventory')
    files = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Invalid export inventory entry')
        name = entry['path']
        if not isinstance(name, str) or name not in ALLOWED or name in files:
            raise ValueError('Invalid export artifact name')
        maximum = 100_000_000 if name == 'checkpoint.pt' else (131072 if name in ('objective_recipe.py', 'initialization_recipe.py', 'gradient_policy.py', 'trainability_recipe.py', 'data_loading_recipe.py') else (8192 if name == 'training_recipe.py' else (4_000_000 if name == 'rate-trace.json' else 2_000_000)))
        if type(entry['bytes']) is not int or not 1 <= entry['bytes'] <= maximum:
            raise ValueError('Invalid export artifact size')
        if not isinstance(entry['sha256'], str) or re.fullmatch('[0-9a-f]{64}', entry['sha256']) is None:
            raise ValueError('Invalid export artifact hash')
        value = read_regular(ROOT / name, maximum)
        if len(value) != entry['bytes'] or hashlib.sha256(value).hexdigest() != entry['sha256']:
            raise ValueError('Export artifact integrity mismatch: ' + name)
        if name in ('training_recipe.py','objective_recipe.py','initialization_recipe.py','gradient_policy.py','trainability_recipe.py','data_loading_recipe.py'):
            try:
                value.decode('utf-8')
            except UnicodeDecodeError as error:
                raise ValueError('Recipe helper must be UTF-8 text') from error
        files[name] = value
    if not REQUIRED <= files.keys():
        raise ValueError('Required export artifacts are missing')
    verified_policy_metadata(manifest, files)
    return files


MAX_ELEMENTS = 100_000_000
OBJECTIVES = {'loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin'}


def sample_elements(shape):
    count=1
    for value in shape:
        if type(value) is not int or not 1<=value<=4096 or count>MAX_ELEMENTS//value:
            raise ValueError('Model sample shape exceeds its element/dimension limit')
        count*=value
    return count


def objective_policy(contract,bag=False):
    """Structural portable contract admission; canonical settings were captured by KwipML.

    This does not import or execute objective_recipe.py. Predictions remain model outputs.
    """
    if set(contract)!={'schemaVersion','input','output','objective'}:
        raise ValueError('Invalid objective export contract fields')
    spec=contract['input'];output=contract['output'];objective=contract['objective']
    if not bag and (not isinstance(spec,dict) or set(spec)!={'dtype','shape','featureColumns','preprocessing'}):
        raise ValueError('Invalid objective input contract')
    if not isinstance(output,dict) or set(output)!={'kind','dtype','shape'} or output['kind']!='modelOutput' or output['dtype']!='float32':
        raise ValueError('Invalid objective model output contract')
    for value in ((output,) if bag else (spec,output)):
        shape=value.get('shape')
        if not isinstance(shape,list) or not 2<=len(shape)<=8 or shape[0]!='N':
            raise ValueError('Invalid objective observation-first shape')
        sample_elements(shape[1:])
    if bag:
        columns=spec['mapping']['indicesColumns']
    else:
        if not isinstance(spec,dict) or set(spec)!={'dtype','shape','featureColumns','preprocessing'}:
            raise ValueError('Invalid objective input contract')
        if spec['dtype'] not in ('float32','int64') or spec['preprocessing']!='alreadyPrepared':
            raise ValueError('Objective inference requires prepared float32 inputs')
        columns=spec['featureColumns']
        if not isinstance(columns,list) or len(columns)!=sample_elements(spec['shape'][1:]) or not 1<=len(columns)<=128 or any(not isinstance(c,str) or not c.strip() or len(c)>200 for c in columns) or len(set(columns))!=len(columns):
            raise ValueError('Input column mapping differs from model sample shape')
    if not isinstance(objective,dict) or set(objective)!={'recipe','source','targetMapping','scalarization'} or objective['scalarization'] not in ('mean','sum'):
        raise ValueError('Invalid objective export metadata')
    recipe=objective['recipe'];source=objective['source'];mapping=objective['targetMapping']
    if not isinstance(recipe,dict) or set(recipe)!={'schemaVersion','componentID','configurationVersion','config'} or type(recipe['schemaVersion']) is not int or recipe['schemaVersion']!=1 or type(recipe['configurationVersion']) is not int or recipe['configurationVersion']!=1 or recipe['componentID'] not in OBJECTIVES or not isinstance(recipe['config'],dict):
        raise ValueError('Invalid captured objective recipe')
    if len(json.dumps(recipe,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'))>32768:
        raise ValueError('Objective recipe exceeds its byte limit')
    if not isinstance(source,dict) or set(source)!={'domain','localID'} or source['domain']!='control' or not isinstance(source['localID'],str) or not source['localID'].strip() or len(source['localID'].encode('utf-8'))>1024:
        raise ValueError('Invalid qualified objective source')
    if not isinstance(mapping,dict) or set(mapping)!={'schemaVersion','componentID','inputColumns','roles','tripletSourceGroups'} or type(mapping['schemaVersion']) is not int or mapping['schemaVersion']!=1 or mapping['componentID']!=recipe['componentID'] or mapping['inputColumns']!=columns or not isinstance(mapping['roles'],list) or not isinstance(mapping['tripletSourceGroups'],list):
        raise ValueError('Invalid captured objective target mapping')
    roles=mapping['roles'];names=[];used=set(columns)
    for role in roles:
        if not isinstance(role,dict) or set(role)!={'role','columns','sampleShape','dtype'} or not isinstance(role['role'],str) or role['dtype'] not in ('float32','int64') or not isinstance(role['sampleShape'],list) or len(role['sampleShape'])>7:
            raise ValueError('Invalid target role contract')
        role_columns=role['columns']
        if not isinstance(role_columns,list) or len(role_columns)!=sample_elements(role['sampleShape']) or not 1<=len(role_columns)<=128 or any(not isinstance(c,str) or not c.strip() or len(c)>200 for c in role_columns) or len(set(role_columns))!=len(role_columns) or used.intersection(role_columns):
            raise ValueError('Invalid ordered target role columns')
        used.update(role_columns);names.append(role['role'])
    required={'target','input_lengths','target_lengths'} if recipe['componentID']=='loss_ctc' else {'target'}
    optional={'weight','pos_weight'} if recipe['componentID']=='loss_bce_with_logits' else set()
    if recipe['componentID']=='loss_triplet_margin':
        groups=mapping['tripletSourceGroups']
        if roles or len(groups)!=3 or any(not isinstance(g,list) or not g or len(g)!=len(groups[0]) for g in groups) or [c for g in groups for c in g]!=columns:
            raise ValueError('Triplet export requires three ordered input groups without target columns')
    elif mapping['tripletSourceGroups'] or len(names)!=len(set(names)) or not required<=set(names) or not set(names)<=required|optional:
        raise ValueError('Objective export target roles differ from its component')
    return None if bag else tuple(spec['shape'][1:]),torch.float32 if bag else getattr(torch,spec['dtype']),tuple(output['shape'][1:])


def bag_policy(contract):
    """Schema3 retains coordinate roles and whole groups, never a dense input shape."""
    objective='objective' in contract
    if type(contract.get('schemaVersion')) is not int or contract['schemaVersion']!=3 or set(contract)!=({'schemaVersion','input','output','objective'} if objective else {'schemaVersion','input','output'}):
        raise ValueError('Invalid bag export contract fields')
    spec=contract['input']
    if not isinstance(spec,dict) or set(spec)!={'kind','schemaVersion','sourceID','configuration','ports','mapping','numEmbeddings','preprocessing'}:
        raise ValueError('Invalid bag input contract')
    if spec['kind']!='bagPacket' or type(spec['schemaVersion']) is not int or spec['schemaVersion']!=1 or spec['preprocessing']!='alreadyPrepared':
        raise ValueError('Unsupported prepared bag packet contract')
    if not isinstance(spec['sourceID'],str) or not spec['sourceID'] or '\x00' in spec['sourceID'] or len(spec['sourceID'].encode('utf-8'))>1024:
        raise ValueError('Invalid bag source identity')
    if type(spec['numEmbeddings']) is not int or not 1<=spec['numEmbeddings']<=100_000_000:
        raise ValueError('Invalid bag vocabulary bound')
    config=spec['configuration']
    if not isinstance(config,dict) or set(config)!={'layout','maximum_items_per_bag','per_sample_weights','include_last_offset'}:
        raise ValueError('Invalid bag source configuration')
    if config['layout'] not in ('fixed','packed') or type(config['maximum_items_per_bag']) is not int or not 1<=config['maximum_items_per_bag']<=128 or type(config['per_sample_weights']) is not bool or type(config['include_last_offset']) is not bool:
        raise ValueError('Invalid bag source controls')
    ports=['indices']+(['offsets'] if config['layout']=='packed' else [])+(['per_sample_weights'] if config['per_sample_weights'] else [])
    if spec['ports']!={p:{'role':p,'dtype':'float32' if p=='per_sample_weights' else 'int64'} for p in ports}:
        raise ValueError('Bag named coordinate ports differ from source controls')
    mapping=spec['mapping'];keys={'schemaVersion','indicesColumns'}|({'lengthColumn'} if config['layout']=='packed' else set())|({'weightColumns'} if config['per_sample_weights'] else set())
    if not isinstance(mapping,dict) or set(mapping)!=keys or type(mapping.get('schemaVersion')) is not int or mapping['schemaVersion']!=1:
        raise ValueError('Invalid bag input column mapping')
    columns=[]
    for key in ('indicesColumns','weightColumns'):
        if key not in mapping:continue
        values=mapping[key]
        if not isinstance(values,list) or len(values)!=config['maximum_items_per_bag']:
            raise ValueError('Bag column mapping differs from item capacity')
        columns.extend(values)
    if 'lengthColumn' in mapping:columns.append(mapping['lengthColumn'])
    if any(not isinstance(c,str) or not c.strip() or len(c)>200 for c in columns) or len(set(columns))!=len(columns):
        raise ValueError('Bag columns must be distinct bounded names')
    if objective:
        _,_,shape=objective_policy(contract,bag=True)
        if any(set(role['columns']).intersection(columns) for role in contract['objective']['targetMapping']['roles']):
            raise ValueError('Bag input and objective target columns must be separate')
    else:
        if contract['output']!={'kind':'logits','shape':['N',2],'classIndices':[0,1]} or any(type(i) is not int for i in contract['output']['classIndices']):
            raise ValueError('Invalid binary bag output contract')
        shape=(2,)
    return spec,shape,objective


def validate_bag_rows(rows,spec,output_shape):
    """Check every group before tensor allocation or any model invocation."""
    if not isinstance(rows,list) or not 1<=len(rows)<=4096:
        raise ValueError('Bag inputs require 1..4096 whole groups')
    if len(rows)>2_000_000//sample_elements(output_shape):
        raise ValueError('Bag output exceeds its element budget before execution')
    config=spec['configuration'];weighted=config['per_sample_weights'];total=0
    for row in rows:
        if not isinstance(row,dict) or set(row)!=({'indices','per_sample_weights'} if weighted else {'indices'}):
            raise ValueError('Each bag needs exact indices and configured paired weights only')
        ids=row['indices']
        if not isinstance(ids,list) or len(ids)>config['maximum_items_per_bag'] or (config['layout']=='fixed' and len(ids)!=config['maximum_items_per_bag']):
            raise ValueError('Bag length differs from the captured layout and capacity')
        if any(type(i) is not int or not 0<=i<spec['numEmbeddings'] for i in ids):
            raise ValueError('Bag indices require exact integers within the captured vocabulary')
        total+=len(ids)
        if total>1_000_000:raise ValueError('Bag items exceed their allocation budget')
        if weighted:
            weights=row['per_sample_weights']
            if not isinstance(weights,list) or len(weights)!=len(ids) or any(type(w) not in (int,float) or not math.isfinite(w) or abs(w)>3.4028234663852886e38 for w in weights):
                raise ValueError('Bag weights require finite float32 values paired with every item')


class BagPredictor:
    """Verified CPU packet snapshot; batch boundaries never split a group."""
    def __init__(self,model,spec,output_shape,objective):
        self._model=model;self.bag_input=spec;self.output_shape=output_shape;self.objective=objective

    def predict(self,rows,batch_size=1024):
        validate_bag_rows(rows,self.bag_input,self.output_shape)
        if type(batch_size) is not int or not 1<=batch_size<=100_000:
            raise ValueError('batch_size must be an integer from 1 to 100000')
        config=self.bag_input['configuration'];outputs=[]
        with torch.inference_mode():
            for start in range(0,len(rows),batch_size):
                selected=rows[start:start+batch_size];weights=None;offsets=None
                if config['layout']=='fixed':
                    indices=torch.tensor([r['indices'] for r in selected],dtype=torch.int64)
                    if config['per_sample_weights']:weights=torch.tensor([r['per_sample_weights'] for r in selected],dtype=torch.float32)
                else:
                    boundaries=[0]
                    for row in selected:boundaries.append(boundaries[-1]+len(row['indices']))
                    indices=torch.tensor([i for r in selected for i in r['indices']],dtype=torch.int64)
                    offsets=torch.tensor(boundaries if config['include_last_offset'] else boundaries[:-1],dtype=torch.int64)
                    if config['per_sample_weights']:weights=torch.tensor([w for r in selected for w in r['per_sample_weights']],dtype=torch.float32)
                output=self._model(indices,offsets,weights)
                if not isinstance(output,torch.Tensor) or output.device.type!='cpu' or output.dtype!=torch.float32 or tuple(output.shape)!=(len(selected),*self.output_shape) or not torch.isfinite(output).all().item():
                    raise ValueError('Model output violates its finite captured shape contract')
                outputs.append(output)
        return torch.cat(outputs,dim=0)


def input_policy(contract):
    if type(contract.get('schemaVersion')) is int and contract['schemaVersion']==2:
        shape,dtype,_=objective_policy(contract)
        return shape,dtype
    spec=contract.get('input', {})
    shape=spec.get('shape')
    dtype=spec.get('dtype')
    if (type(contract.get('schemaVersion')) is not int or contract['schemaVersion']!=1
            or dtype not in ('float32','int64') or spec.get('preprocessing')!='alreadyPrepared'
            or not isinstance(shape,list) or not 2<=len(shape)<=8 or shape[0]!='N'
            or any(type(n) is not int or not 1<=n<=128 for n in shape[1:])):
        raise ValueError('Unsupported model input contract')
    count=1
    for n in shape[1:]:count*=n
    if count>128 or (dtype=='float32' and shape!=['N',2]):
        raise ValueError('Unsupported model sample shape')
    return tuple(shape[1:]),getattr(torch,dtype)


def validate_features(features,shape=(2,),dtype=torch.float32):
    if isinstance(features,torch.Tensor) and features.numel()>MAX_ELEMENTS:
        raise ValueError('Input exceeds its element budget')
    if (not isinstance(features, torch.Tensor) or features.layout != torch.strided
            or features.is_nested or features.device.type != 'cpu' or features.dtype != dtype
            or features.ndim != len(shape)+1 or not 1 <= features.shape[0] <= 100_000
            or tuple(features.shape[1:])!=shape or not torch.isfinite(features).all().item()):
        raise ValueError(f'Expected finite CPU {dtype} prepared inputs with sample shape {shape}, 1 <= N <= 100000')


def prepared_json(rows,shape,dtype):
    if not isinstance(rows,list) or not 1<=len(rows)<=100_000:
        raise ValueError('Input JSON requires 1..100000 prepared observations')
    if len(rows)>MAX_ELEMENTS//sample_elements(shape):
        raise ValueError('Input exceeds its element budget before allocation')
    def check(value,axes):
        if axes:
            if not isinstance(value,list) or len(value)!=axes[0]:raise ValueError('Input JSON does not match model sample shape')
            for item in value:check(item,axes[1:])
        elif dtype==torch.int64:
            if type(value) is not int or not -(2**63)<=value<2**63:raise ValueError('Input JSON requires exact signed 64-bit integer indices')
        elif type(value) not in (int,float):raise ValueError('Input JSON requires numeric prepared values')
    for row in rows:check(row,shape)
    return torch.tensor(rows,dtype=dtype)


class Predictor:
    """A verified loaded snapshot. Reload explicitly to adopt changed package files."""
    def __init__(self, model,shape=(2,),dtype=torch.float32,output_shape=(2,),objective=False):
        self._model = model
        self.input_shape=shape;self.input_dtype=dtype
        self.output_shape=output_shape;self.objective=objective
        sample_elements(output_shape)

    def predict(self, features, batch_size=1024):
        """Predict in bounded batches, preserving row order and returning CPU logits."""
        validate_features(features,self.input_shape,self.input_dtype)
        if type(batch_size) is not int or not 1 <= batch_size <= 100_000:
            raise ValueError('batch_size must be an integer from 1 to 100000')
        if features.shape[0]>MAX_ELEMENTS//sample_elements(self.output_shape):
            raise ValueError('Output exceeds its element budget before model execution')
        outputs = []
        with torch.inference_mode():
            for batch in features.split(batch_size):
                logits = self._model(batch)
                if (not isinstance(logits, torch.Tensor) or logits.device.type != 'cpu'
                        or logits.dtype != torch.float32 or tuple(logits.shape) != (batch.shape[0], *self.output_shape)
                        or not torch.isfinite(logits).all().item()):
                    raise ValueError('Model output violates its finite captured shape contract')
                outputs.append(logits)
            return torch.cat(outputs, dim=0)


class NamedPredictor:
    """One semantic name per aligned tensor; raw file authority never enters here."""
    def __init__(self,model,contract,files):
        self.model=model;spec=contract.get('input');out=contract.get('output')
        if type(spec) is not dict or set(spec)!={'kind','plan','namedValues','preprocessing'} or spec['kind']!='namedInputs' or type(spec['namedValues']) is not bool or spec['preprocessing']!='capturedPreparation':raise ValueError('Invalid named model input contract.')
        if type(out) is not dict or set(out)!={'kind','dtype','shape'} or out['kind'] not in ('logits','modelOutput') or out['dtype']!='float32' or type(out['shape']) is not list or not out['shape'] or out['shape'][0]!='N':raise ValueError('Invalid named model output contract.')
        self.output_shape=tuple(out['shape'][1:]);sample_elements(self.output_shape)
        self.objective=out['kind']=='modelOutput';self.named_values=spec['namedValues']
        if set(contract)!={'schemaVersion','input','output'}|({'objective'} if self.objective else set()) or contract['schemaVersion']!=4:raise ValueError('Named export has unknown or missing fields.')
        if 'named_inputs.py' not in files:raise ValueError('Named conversion helper is missing.')
        namespace={'__name__':'kwipml_named_inputs'}
        exec(compile(files['named_inputs.py'],str(ROOT/'named_inputs.py'),'exec'),namespace)
        self.plan=namespace['validate_plan'](spec['plan']);self.convert_rows=namespace['convert_rows']
        if not self.named_values and len(self.plan['inputs'])!=1:raise ValueError('An unnamed model requires exactly one input.')
        self.names={entry['name']:entry for entry in self.plan['inputs']}
        self.input_shape=tuple(self.plan['inputs'][0]['sampleShape']);self.input_dtype=getattr(torch,self.plan['inputs'][0]['dtype'])
        self.files=files
        self.tokenizers=json.loads(files.get('tokenizers.json',b'{}'))
        if type(self.tokenizers) is not dict or set(self.tokenizers)-set(self.names):raise ValueError('Tokenizer names differ from the captured model inputs.')
        self.encode_tokenizer=None
        if self.tokenizers:
            if 'portable_text_encoding.py' not in files:raise ValueError('Captured tokenizer helper is missing.')
            tokenizer_namespace={'__name__':'kwipml_text_encoding'}
            exec(compile(files['portable_text_encoding.py'],str(ROOT/'portable_text_encoding.py'),'exec'),tokenizer_namespace)
            self.encode_tokenizer=tokenizer_namespace['encode']
            for name,metadata in self.tokenizers.items():
                if self.names[name]['representation']!='tokens' or len(metadata.get('vocabulary',[]))!=self.names[name]['vocabularySize']:raise ValueError('Captured vocabulary differs from the tensor mapping.')
                self.encode_tokenizer('',metadata)
    def encode_text(self,texts,input_name):
        if input_name not in self.tokenizers or type(texts) is not list or not 1<=len(texts)<=4096:raise ValueError('Choose a captured text tokenizer and a bounded list of texts.')
        return [self.encode_tokenizer(text,self.tokenizers[input_name]) for text in texts]
    def prepare_records(self,records):
        if type(records) is not list or not 1<=len(records)<=100000 or any(type(r) is not dict for r in records):raise ValueError('Supply 1–100000 prepared records.')
        plan=json.loads(json.dumps(self.plan));plan['roles']=[];plan['schemaVersion']=1;plan.pop('sequenceObjective',None)
        columns=[];dtypes={}
        for e in plan['inputs']:
            for c in e['columns']:
                dtype='int64Sequence' if e['representation']=='tokens' else 'int64' if e['dtype']=='int64' else 'float64'
                if c in dtypes and dtypes[c]!=dtype:
                    if {dtypes[c],dtype}=={'int64','float64'}:dtype='int64'
                    else:raise ValueError('Shared column representations differ.')
                dtypes[c]=dtype
                if c not in columns:columns.append(c)
        if any(set(r)!=set(columns) for r in records):raise ValueError('Prepared record fields differ from the captured input columns.')
        schema=[{'name':c,'dtype':dtypes[c],'nullable':False} for c in columns]
        per_row=sum(sample_elements(e['sampleShape'])*(8 if e['dtype']=='int64' else 4) for e in plan['inputs'])
        if len(records)*per_row>134217728:raise ValueError('Prepared records exceed the 128 MiB conversion allowance.')
        count=min(4096,plan['maximumBatchBytes']//per_row)
        if count<1:raise ValueError('One record exceeds the configured batch allowance.')
        result={k:[] for k in self.names}
        for begin in range(0,len(records),count):
            rows=[[r[c] for c in columns] for r in records[begin:begin+count]]
            converted,_=self.convert_rows(plan,schema,rows)
            for name,value in converted.items():result[name].append(value)
        return {k:torch.cat(v,dim=0) for k,v in result.items()}
    def predict(self,features,batch_size=1024):
        if type(features) is list:features=self.prepare_records(features)
        if type(features) is not dict or set(features)!=set(self.names):raise ValueError('Provide exactly the captured named tensors.')
        if type(batch_size) is not int or not 1<=batch_size<=100000:raise ValueError('Batch size must be a positive bounded integer.')
        counts=set()
        for name,entry in self.names.items():
            value=features[name]
            if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.is_nested or value.device.type!='cpu' or value.dtype!=getattr(torch,entry['dtype']) or tuple(value.shape[1:])!=tuple(entry['sampleShape']) or not 1<=value.shape[0]<=100000 or not torch.isfinite(value).all().item():raise ValueError('Named input shape/type/device or finite values differ: '+name)
            counts.add(value.shape[0])
            if entry['representation']=='tokens' and (value.min().item()<0 or value.max().item()>=entry['vocabularySize']):raise ValueError('Token IDs lie outside the saved vocabulary.')
        if len(counts)!=1:raise ValueError('Named tensors have misaligned observation counts.')
        count=counts.pop();per_row=sum(sample_elements(e['sampleShape'])*(8 if e['dtype']=='int64' else 4) for e in self.names.values())
        if min(batch_size,count)*per_row>self.plan['maximumBatchBytes'] or count>MAX_ELEMENTS//max(sample_elements(self.output_shape),sum(sample_elements(e['sampleShape']) for e in self.names.values())):raise ValueError('Named prediction exceeds its input/output allocation allowance.')
        outputs=[]
        with torch.inference_mode():
            for start in range(0,count,batch_size):
                batch={k:v[start:start+batch_size] for k,v in features.items()}
                prediction=self.model(batch if self.named_values else next(iter(batch.values())))
                if not isinstance(prediction,torch.Tensor) or prediction.dtype!=torch.float32 or tuple(prediction.shape)!=(min(batch_size,count-start),*self.output_shape) or not torch.isfinite(prediction).all().item():raise ValueError('Named model output differs from its captured shape or finite contract.')
                outputs.append(prediction)
        return torch.cat(outputs,dim=0)


def validate_reviewed_normalization_buffers(model,state):
    """Admit only exact typed counters of reviewed configured tracked modules."""
    classes=(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.InstanceNorm1d,torch.nn.InstanceNorm2d)
    reviewed={}
    for name,module in model.named_modules():
        if type(module) not in classes or not module.track_running_stats:continue
        if not 1<=module.num_features<=512:raise ValueError('Exported normalization channels exceed the reviewed policy')
        prefix=name+'.' if name else ''
        reviewed[prefix+'running_mean']=('float32',(module.num_features,),'mean')
        reviewed[prefix+'running_var']=('float32',(module.num_features,),'variance')
        reviewed[prefix+'num_batches_tracked']=('int64',(),'instanceCounter' if type(module) in (torch.nn.InstanceNorm1d,torch.nn.InstanceNorm2d) else 'batchCounter')
    if not reviewed:return False
    expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('Checkpoint state keys differ from configured model parameters/buffers')
    for key,value in state.items():
        if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or not torch.isfinite(value).all().item():raise ValueError('Checkpoint state is not finite dense typed model state')
        dtype,shape,kind=reviewed.get(key,('float32',tuple(expected[key].shape),'parameter'))
        if str(value.dtype).removeprefix('torch.')!=dtype or tuple(value.shape)!=shape:raise ValueError('Checkpoint parameter/buffer shape or dtype differs from the reviewed model')
        if kind=='variance' and (value<0).any().item():raise ValueError('Checkpoint running variance is negative')
        if kind in ('instanceCounter','batchCounter') and (value.item()<0 or value.item()>1_000_000_000 or (kind=='instanceCounter' and value.item()!=0)):raise ValueError('Checkpoint tracking counter differs from the reviewed policy')
    return True


def load_model():
    """Verify package bytes and load weights once for repeated prepared inputs."""
    files = verified_files()
    contract = json.loads(files['input-contract.json'])
    named=type(contract.get('schemaVersion')) is int and contract['schemaVersion']==4
    bag=type(contract.get('schemaVersion')) is int and contract['schemaVersion']==3
    if named:
        objective=contract.get('output',{}).get('kind')=='modelOutput'
    elif bag:
        bag_spec,output_shape,objective=bag_policy(contract)
    else:
        shape,dtype=input_policy(contract)
        objective=contract['schemaVersion']==2
        output_shape=objective_policy(contract)[2] if objective else (2,)
    if objective and 'objective_recipe.py' not in files:
        raise ValueError('Required objective helper is missing')
    # Code is generated model source included in this export. Hashes check package
    # consistency, not authorship; review code before running a third-party export.
    namespace = {'__name__': 'kwipml_exported_model'}
    exec(compile(files['model.py'], str(ROOT / 'model.py'), 'exec'), namespace)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        model = namespace['Model']().to(dtype=torch.float32)
    checkpoint = torch.load(io.BytesIO(files['checkpoint.pt']), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('checkpointSchemaVersion') != 1:
        raise ValueError('Unsupported lifecycle checkpoint')
    state = checkpoint.get('modelState')
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint has no learned model state')
    if not validate_reviewed_normalization_buffers(model,state) and any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           or value.layout != torch.strided or not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError('Checkpoint model state is not finite dense float32')
    model.load_state_dict(state, strict=True)
    model.eval()
    return NamedPredictor(model,contract,files) if named else BagPredictor(model,bag_spec,output_shape,objective) if bag else Predictor(model,shape,dtype,output_shape,objective)


def predict(features, batch_size=1024):
    """One-shot prediction; use load_model() to retain a verified model across calls."""
    return load_model().predict(features, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--example', action='store_true')
    source.add_argument('--input', type=Path, help='JSON array of already-prepared feature rows')
    parser.add_argument('--batch-size', type=int, default=1024, help='Rows per prediction batch (1–100000; default 1024)')
    args = parser.parse_args()
    predictor=load_model()
    if args.example and isinstance(predictor,BagPredictor):
        config=predictor.bag_input['configuration'];count=config['maximum_items_per_bag'] if config['layout']=='fixed' else 1
        features=[{'indices':[0]*count,**({'per_sample_weights':[1.]*count} if config['per_sample_weights'] else {})} for _ in range(2)]
    elif args.example and isinstance(predictor,NamedPredictor):
        features={name:torch.zeros((2,*entry["sampleShape"]),dtype=getattr(torch,entry["dtype"])) for name,entry in predictor.names.items()}
    elif args.example:
        if 2>MAX_ELEMENTS//max(sample_elements(predictor.input_shape),sample_elements(predictor.output_shape)):
            raise ValueError('Synthetic example exceeds its element budget before allocation')
        features=(torch.zeros((2,*predictor.input_shape),dtype=predictor.input_dtype) if predictor.objective else
                  torch.tensor([[0.,0.],[1.,-1.]]) if predictor.input_dtype==torch.float32
                  else torch.zeros((2,*predictor.input_shape),dtype=torch.int64))
    else:
        rows=json.loads(read_regular(args.input,10_000_000))
        features=rows if isinstance(predictor,(BagPredictor,NamedPredictor)) else prepared_json(rows,predictor.input_shape,predictor.input_dtype)
    logits=predictor.predict(features,batch_size=args.batch_size)
    result={'inputKind':'synthetic prepared example' if args.example else 'provided prepared features'}
    if predictor.objective:result['modelOutput']=logits.tolist()
    else:result.update(logits=logits.tolist(),classIndices=logits.argmax(dim=1).tolist())
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
