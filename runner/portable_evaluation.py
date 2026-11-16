"""Standalone frozen inference. Hashes establish consistency, not authorship.

Run only packages from a source you trust: model.py is executable Python.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import zipfile

import torch

if __package__:
    from . import portable_fitted
    from .final_evaluation import evaluate_classifier
else:
    # -I deliberately omits the script directory; dependencies are adjacent.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import portable_fitted
    from final_evaluation import evaluate_classifier

LIMITS = {'model.py':2_000_000, 'weights.pt':100_000_000, 'input-contract.json':2_000_000,
          'fitted-parameters.json':20_000_000, 'saved-summary.json':1_000_000,
          'portable_evaluation.py':200_000, 'portable_fitted.py':200_000,
          'final_evaluation.py':200_000, 'execution_device.py':200_000,
          'requirements.txt':10_000, 'README.md':100_000}
OPTIONAL = {'test-input.json':50_000_000, 'saved-predictions.jsonl':100_000_000}
METRICS = {'schemaVersion','evaluatedCount','device','batchSize','lossSum','lossMean','accuracy','confusionMatrix','perClass'}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return sha(encoded(value))
def membership_digest(ids):
    h=hashlib.sha256(b'kwip.membership.v1\n')
    for row in sorted(x.encode() for x in ids): h.update(str(len(row)).encode()+b':'+row)
    return h.hexdigest()


def hash_valid(value): return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def parse(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result, 'Duplicate JSON field'); result[key]=value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def directory(path):
    absolute=os.path.abspath(os.fspath(path))
    if sys.platform=='darwin':
        # Sandbox permission may cover the leaf without granting read-data on
        # ancestors. Darwin validates every component without opening ancestors.
        return os.open(absolute,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW_ANY|os.O_CLOEXEC)
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    try:
        for part in Path(absolute).parts[1:]:
            next_fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=fd)
            os.close(fd);fd=next_fd
        return fd
    except BaseException: os.close(fd);raise


def read(fd,name,maximum):
    require(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name and '\0' not in name,'Invalid package path')
    handle=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    try:
        before=os.fstat(handle)
        require(stat.S_ISREG(before.st_mode) and before.st_size<=maximum,'Invalid package file or bound')
        with os.fdopen(os.dup(handle),'rb') as stream: raw=stream.read(maximum+1)
        after=os.fstat(handle)
        require(len(raw)<=maximum and len(raw)==before.st_size and (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Package file changed')
        return raw
    finally: os.close(handle)


def contract_valid(contract,params):
    fields={'schemaVersion','kind','columns','featureColumns','targetColumn','dtype','sampleShape','fittedStateDigests','appliedPrefix','testEndpoint','testCount','membershipDigest','selectedDataVersion'}
    require(isinstance(contract,dict) and type(contract.get('schemaVersion')) is int and
            (contract['schemaVersion']==1 and set(contract)==fields or contract['schemaVersion']==2 and set(contract)==fields|{'partitionRole'} and contract['partitionRole']=='validation'), 'Invalid input contract fields/version')
    require(contract['kind']=='portableEvaluationInput','Invalid input contract kind')
    portable_fitted.schema_valid(contract['columns'])
    shape=contract['sampleShape'];features=contract['featureColumns'];chain=contract['fittedStateDigests'];prefix=contract['appliedPrefix']
    require(contract['dtype'] in ('float32','int64') and isinstance(shape,list) and 1<=len(shape)<=7 and all(type(v) is int and 1<=v<=100_000_000 for v in shape) and math.prod(shape)<=128,'Invalid input tensor contract')
    require(portable_fitted.names(features) and len(features)==math.prod(shape) and isinstance(contract['targetColumn'],str) and contract['targetColumn'] not in features,'Invalid mapped columns')
    require(isinstance(chain,list) and len(chain)<=128 and all(hash_valid(v) for v in chain) and isinstance(prefix,list) and chain[:len(prefix)]==prefix,'Invalid ordered fitted chain')
    require(isinstance(params,list) and len(params)==len(chain),'Fitted parameter chain differs')
    for value,pin in zip(params,chain):
        portable_fitted.validate(value);require(value['sourceContentDigest']==pin,'Fitted parameter identity differs')
    # Already-applied transforms may have ordinary projection/rename steps
    # between them. Only the remaining frozen computation must start from the
    # selected endpoint schema and form an executable consecutive chain.
    current_schema=contract['columns']
    for state in params[len(prefix):]:
        require(state['inputSchema']==current_schema,'Remaining fitted schema chain differs from selected endpoint')
        current_schema=state['outputSchema']
    final=current_schema;names=[c['name'] for c in final]
    require(all(v in names for v in [*features,contract['targetColumn']]),'Missing mapped column')
    require(type(contract['testCount']) is int and 1<=contract['testCount']<=100000 and hash_valid(contract['membershipDigest']) and hash_valid(contract['selectedDataVersion']),'Invalid protected population identity')
    endpoint=contract['testEndpoint'];require(isinstance(endpoint,dict) and set(endpoint)=={'node','port'} and all(isinstance(v,str) and 1<=len(v)<=200 for v in endpoint.values()),'Invalid selected endpoint')


def table_valid(table,contract,original):
    require(isinstance(table,dict) and set(table)=={'schemaVersion','columns','rows','observationIDs'} and type(table['schemaVersion']) is int and table['schemaVersion']==1,'Invalid input table fields')
    require(table['columns']==contract['columns'],'Input schema differs from selected endpoint contract')
    rows=table['rows'];ids=table['observationIDs'];columns=table['columns']
    require(isinstance(rows,list) and 1<=len(rows)<=100000 and isinstance(ids,list) and len(ids)==len(rows) and all(isinstance(v,str) and 1<=len(v)<=200 for v in ids) and len(set(ids))==len(ids),'Invalid input row bounds or identities')
    for row in rows:
        require(isinstance(row,list) and len(row)==len(columns),'Input row width mismatch')
        for value,col in zip(row,columns):
            require(col['nullable'] if value is None else portable_fitted.typed(value,col['dtype']),'Input value violates column type')
    if original:
        require(len(rows)==contract['testCount'] and membership_digest(ids)==contract['membershipDigest'],'Protected membership differs')
        require(digest({'columns':columns,'rows':rows,'ids':ids})==contract['selectedDataVersion'],'Protected values or order differ')
    return rows,ids


def summary_valid(summary,expected_count=None):
    require(isinstance(summary,dict) and set(summary)==METRICS,'Invalid aggregate summary fields')
    require(type(summary['schemaVersion']) is int and summary['schemaVersion']==1 and type(summary['evaluatedCount']) is int and 1<=summary['evaluatedCount']<=100000,'Invalid summary count')
    count=summary['evaluatedCount']
    require(expected_count is None or count==expected_count,'Summary count differs from protected contract')
    require(summary['device'] in ('cpu','mps') and type(summary['batchSize']) is int and 1<=summary['batchSize']<=100000,'Invalid saved device/batch settings')
    require(all(portable_fitted.numeric(summary[k]) and summary[k]>=0 for k in ('lossSum','lossMean','accuracy')) and summary['accuracy']<=1,'Invalid aggregate numeric metric')
    require(math.isclose(summary['lossSum']/count,summary['lossMean'],rel_tol=1e-9,abs_tol=1e-10),'Loss mean differs from sum/count')
    confusion=summary['confusionMatrix']
    require(isinstance(confusion,list) and len(confusion)==2 and all(isinstance(row,list) and len(row)==2 and all(type(v) is int and 0<=v<=count for v in row) for row in confusion) and sum(map(sum,confusion))==count,'Invalid confusion counts')
    require(summary['accuracy']==sum(confusion[i][i] for i in range(2))/count,'Accuracy differs from confusion counts')
    classes=[]
    for i in range(2):
        support=sum(confusion[i]);predicted=sum(r[i] for r in confusion)
        classes.append(dict(classIndex=i,precision=confusion[i][i]/predicted if predicted else None,precisionUnavailableReason=None if predicted else 'noPredictedObservations',recall=confusion[i][i]/support if support else None,recallUnavailableReason=None if support else 'noActualObservations',support=support))
    observed=summary['perClass']
    require(isinstance(observed,list) and len(observed)==2,'Invalid per-class metrics')
    for value,expected in zip(observed,classes):
        require(isinstance(value,dict) and set(value)==set(expected) and type(value['classIndex']) is int and type(value['support']) is int,'Invalid per-class identity/count types')
        for key in ('precision','recall'):
            require(value[key] is None or portable_fitted.numeric(value[key]),'Invalid per-class ratio type')
        require(value==expected,'Per-class metrics differ')
    return summary


def predictions_valid(raw,summary,ids=None):
    summary_valid(summary)
    require(raw.endswith(b'\n'),'Predictions require final newline')
    lines=raw.split(b'\n',100001);require(len(lines)==summary['evaluatedCount']+1 and lines[-1]==b'','Prediction count differs')
    rows=[];losses=[];confusion=[[0,0],[0,0]];seen=set()
    for line in lines[:-1]:
        require(len(line)<=10000,'Prediction record exceeds bound');p=parse(line)
        require(isinstance(p,dict) and set(p)=={'observationID','logits','predicted','target'},'Invalid prediction fields')
        ident=p['observationID'];v=p['logits'];target=p['target'];prediction=p['predicted']
        require(isinstance(ident,str) and 1<=len(ident)<=200 and ident not in seen,'Invalid prediction identity');seen.add(ident)
        require(isinstance(v,list) and len(v)==2 and all(portable_fitted.numeric(x) for x in v) and type(target) is int and target in (0,1) and type(prediction) is int and prediction==(0 if v[0]>=v[1] else 1),'Invalid classifier prediction')
        peak=max(v);losses.append((peak-v[target])+math.log(math.exp(v[0]-peak)+math.exp(v[1]-peak)));confusion[target][prediction]+=1;rows.append(p)
    if ids is not None: require([p['observationID'] for p in rows]==ids,'Prediction membership/order differs')
    count=len(rows);loss=math.fsum(losses)
    for key,value in {'lossSum':loss,'lossMean':loss/count,'accuracy':sum(confusion[i][i] for i in range(2))/count}.items():
        require(portable_fitted.numeric(summary[key]) and math.isclose(summary[key],value,rel_tol=1e-9,abs_tol=1e-10),'Aggregate metric differs from predictions')
    require(summary['confusionMatrix']==confusion,'Confusion counts differ')
    return rows


def require_supported_evaluation(version,device):
    if version==1 and device=='mps':
        raise ValueError('Prior MPS evaluation used an affected host-conversion path. Saved evidence remains unchanged. Run a new evaluation with the updated app and export that result.')


def validate_package(root):
    fd=directory(root)
    try:
        manifest=parse(read(fd,'manifest.json',2_000_000))
        fields={'schemaVersion','kind','includesTestRows','identities','files'}
        require(isinstance(manifest,dict) and type(manifest.get('schemaVersion')) is int and
                (manifest['schemaVersion'] in (1,2) and set(manifest)==fields or manifest['schemaVersion']==3 and set(manifest)==fields|{'partitionRole'} and manifest['partitionRole']=='validation'), 'Invalid package manifest fields/version')
        require(manifest['kind']=='portableFrozenEvaluation' and type(manifest['includesTestRows']) is bool,'Unsupported package kind')
        identities=manifest['identities'];require(isinstance(identities,dict) and set(identities)=={'evaluationID','evaluationManifestSHA256','checkpointSHA256','stateDigest','snapshotReceiptSHA256'},'Invalid original identities')
        require(all(hash_valid(identities[k]) for k in identities if k!='evaluationID') and isinstance(identities['evaluationID'],str) and 1<=len(identities['evaluationID'])<=200,'Invalid identity hashes')
        limits={**LIMITS,**(OPTIONAL if manifest['includesTestRows'] else {})};entries=manifest['files']
        for dependency in ('preprocess_parameters.py','dataflow_temporal.py','model_output_configuration.py'):
            if isinstance(entries,list) and any(isinstance(e,dict) and e.get('path')==dependency for e in entries):limits[dependency]=200_000
        require(isinstance(entries,list) and len(entries)==len(limits),'Invalid package inventory count')
        require(set(os.listdir(fd))==set(limits)|{'manifest.json'},'Unexpected package files')
        files={};total=0
        for entry in entries:
            require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid inventory entry')
            name=entry['path'];require(isinstance(name,str) and name in limits and name not in files,'Invalid inventory path')
            require(type(entry['bytes']) is int and 0<=entry['bytes']<=limits[name] and hash_valid(entry['sha256']),'Invalid inventory bound')
            total+=entry['bytes'];require(total<=280_000_000,'Package exceeds total budget')
            raw=read(fd,name,limits[name]);require(len(raw)==entry['bytes'] and sha(raw)==entry['sha256'],'Package hash mismatch: '+name);files[name]=raw
        contract=parse(files['input-contract.json']);params=parse(files['fitted-parameters.json']);contract_valid(contract,params)
        require(contract.get('partitionRole','test')==manifest.get('partitionRole','test'),'Portable population role differs from input contract')
        require(not any(p.get('operator')=='kwip.data.preprocess' for p in params) or 'preprocess_parameters.py' in files,'Preprocess portable implementation is missing')
        summary=parse(files['saved-summary.json']);summary_valid(summary,contract['testCount'])
        require_supported_evaluation(manifest['schemaVersion'],summary['device'])
        if manifest['includesTestRows']:
            table=parse(files['test-input.json']);_,ids=table_valid(table,contract,True)
            predictions_valid(files['saved-predictions.jsonl'],summary,ids)
        return manifest,files,contract,params
    finally: os.close(fd)


def load_model(files,identity):
    raw=files['weights.pt']
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries=archive.infolist()
            require(1<=len(entries)<=10000 and len({e.filename for e in entries})==len(entries) and sum(e.file_size for e in entries)<=100_000_000 and all(e.compress_type==zipfile.ZIP_STORED and not e.flag_bits&1 for e in entries),'Invalid expanded weights archive')
    except zipfile.BadZipFile as error: raise ValueError('Invalid weights archive') from error
    saved=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    require(isinstance(saved,dict) and len(saved)<=10000,'Invalid weights state dictionary')
    namespace={}
    with torch.random.fork_rng(devices=[]):
        exec(compile(files['model.py'],'model.py','exec'),namespace)
        model=namespace['Model']()
    expected=model.state_dict();require(set(saved)==set(expected),'Weights parameter names differ')
    digest_state=hashlib.sha256()
    for name in sorted(saved):
        v=saved[name];reference=expected[name]
        require(isinstance(v,torch.Tensor) and not v.is_nested and v.layout==torch.strided and not v.is_quantized and v.shape==reference.shape and v.dtype==reference.dtype and torch.isfinite(v).all().item(),'Weights shape, dtype or value differs')
        value=v.detach().cpu().contiguous();digest_state.update(str(value.dtype).encode());digest_state.update(str(tuple(value.shape)).encode());
        octets=value.reshape(-1).view(torch.uint8)
        for offset in range(0,octets.numel(),65536): digest_state.update(bytes(octets[offset:offset+65536].tolist()))
    require(digest_state.hexdigest()==identity['stateDigest'],'Frozen state digest differs')
    model.load_state_dict(saved,strict=True);model.eval();model.requires_grad_(False);return model


def evaluate_package(root,*,replay=False,input_path=None,batch_size=1024,device='cpu'):
    require(type(replay) is bool and (replay != (input_path is not None)),'Choose replay or explicit external input')
    manifest,files,contract,params=validate_package(root)
    if replay:
        require(manifest['includesTestRows'],'Replay requires bundled test rows');table=parse(files['test-input.json'])
    else:
        path=Path(input_path);fd=directory(path.absolute().parent)
        try: table=parse(read(fd,path.name,50_000_000))
        finally: os.close(fd)
    rows,ids=table_valid(table,contract,replay);columns=contract['columns']
    for state in params[len(contract['appliedPrefix']):]:rows,columns,_=portable_fitted.apply(rows,columns,state)
    names=[c['name'] for c in columns];indices=[names.index(v) for v in contract['featureColumns']];target=names.index(contract['targetColumn'])
    require(columns[target]['dtype']=='int64' and all(columns[i]['dtype'] in (('int64',) if contract['dtype']=='int64' else ('int64','float64')) for i in indices),'Mapped column types incompatible')
    x=[];y=[]
    for row in rows:
        values=[row[i] for i in indices];label=row[target]
        require(type(label) is int and label in (0,1),'Targets must be binary int64')
        require(all(portable_fitted.typed(v,'int64') if contract['dtype']=='int64' else portable_fitted.numeric(v) and abs(v)<=1e10 for v in values),'Invalid mapped feature value')
        x.append(values);y.append(label)
    model=load_model(files,manifest['identities'])
    result=evaluate_classifier(model,torch.tensor(x,dtype=getattr(torch,contract['dtype'])).reshape(len(x),*contract['sampleShape']),torch.tensor(y,dtype=torch.int64),ids,batch_size=batch_size,device=device,preview_limit=0)
    if replay:
        saved=parse(files['saved-summary.json']);predictions=predictions_valid(files['saved-predictions.jsonl'],saved,ids)
        for key in ('evaluatedCount','confusionMatrix','perClass'):require(result[key]==saved[key],'Replay differs from saved '+key)
        for key in ('lossSum','lossMean','accuracy'):require(math.isclose(result[key],saved[key],rel_tol=1e-5,abs_tol=1e-6),'Replay metric differs')
        for a,b in zip(result['predictions'],predictions):
            require(all(a[k]==b[k] for k in ('observationID','predicted','target')) and all(math.isclose(x,y,rel_tol=1e-5,abs_tol=1e-6) for x,y in zip(a['logits'],b['logits'])),'Replay prediction differs')
    return {**result,'population':('originalValidation' if contract.get('partitionRole')=='validation' else 'originalProtectedTest') if replay else 'externalInput'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--replay',action='store_true');group.add_argument('--input',type=Path)
    parser.add_argument('--batch-size',type=int,default=1024);parser.add_argument('--device',choices=['cpu','mps','auto'],default='cpu')
    options=parser.parse_args();result=evaluate_package(Path(__file__).resolve().parent,replay=options.replay,input_path=options.input,batch_size=options.batch_size,device=options.device)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
