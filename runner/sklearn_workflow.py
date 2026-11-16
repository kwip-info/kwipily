"""sklearn lifecycle bridge to KwipML's verified preparation artifact contract.

No canvas/runtime availability is implied by importing this foundation module.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
if __package__:
    from . import sklearn_backend as backend
    from .prepared_data import load_prepared
else:
    import sklearn_backend as backend
    from prepared_data import load_prepared


def validate_request(request):
    if not isinstance(request,dict) or (not {'schemaVersion','backendID','configuration','prepared'}<=set(request) or not set(request)<={'schemaVersion','backendID','configuration','prepared','capturePredictions','model'}) or type(request['schemaVersion']) is not int or request['schemaVersion']!=1 or request['backendID']!='sklearn':
        raise ValueError('Unsupported sklearn prepared request.')
    if type(request.get('capturePredictions',False)) is not bool:raise ValueError('capturePredictions must be Boolean.')
    if request['configuration'].get('task')=='documentClassification':
        if __package__:from .document_classification import estimator_options
        else:from document_classification import estimator_options
        if not {'task','settings'}<=set(request['configuration'])<= {'task','settings','operation','reviewThreshold'}:raise ValueError('Invalid document classification configuration.')
        estimator_options(request['configuration']['settings'])
        threshold=request['configuration'].get('reviewThreshold')
        if threshold is not None and (type(threshold) not in (int,float) or not 0<threshold<=1):raise ValueError('Review threshold must be a probability in (0,1].')
    elif request['configuration'].get('task')=='prebuilt':
        if __package__:from .prebuilt_workflow import validate_request as validate_prebuilt
        else:from prebuilt_workflow import validate_request as validate_prebuilt
        validate_prebuilt(request)
    else:backend.validate_configuration(request['configuration'])
    prepared=request['prepared']
    keys={'manifest','manifestSHA256','graphDigest','trainingNode','trainingPort','validationNode','validationPort','featureColumns','targetColumn'}
    if not isinstance(prepared,dict) or set(prepared)!=keys:raise ValueError('Prepared request fields differ from the supported contract.')
    for key in keys-{'featureColumns'}:
        if key=='targetColumn' and request['configuration'].get('task')=='prebuilt' and prepared[key]=='':continue
        if not isinstance(prepared[key],str) or not 1<=len(prepared[key])<=4096:raise ValueError('Prepared request text is empty or oversized.')
    for key in ('manifestSHA256','graphDigest'):
        if re.fullmatch('[0-9a-f]{64}',prepared[key]) is None:raise ValueError('Prepared request requires exact lowercase SHA256 pins.')
    if not isinstance(prepared['featureColumns'],list) or (not 1<=len(prepared['featureColumns'])<=128 if request['configuration'].get('task')=='prebuilt' else len(prepared['featureColumns'])!=(1 if request['configuration'].get('task')=='documentClassification' else 2)) or any(not isinstance(v,str) or not 1<=len(v)<=128 for v in prepared['featureColumns']):
        raise ValueError('The prepared foundation requires exactly two feature names.')
    return prepared


def run_prepared(request, output, emit, *, check_cancelled=lambda:None):
    prepared=validate_request(request)
    if request['configuration'].get('task')=='documentClassification':
        if __package__:from .document_classification_workflow import run_prepared as run_documents
        else:from document_classification_workflow import run_prepared as run_documents
        return run_documents(request,output,emit,check_cancelled)
    if request['configuration'].get('task')=='prebuilt':
        if __package__:from .prebuilt_workflow import run_prepared as run_models
        else:from prebuilt_workflow import run_prepared as run_models
        return run_models(request,output,emit,check_cancelled)
    destination=Path(output)
    if destination.exists() or destination.is_symlink():raise FileExistsError('Run output already exists.')
    check_cancelled()
    data,provenance,files=load_prepared(prepared['manifest'],prepared['trainingNode'],prepared['trainingPort'],
        prepared['validationNode'],prepared['validationPort'],prepared['featureColumns'],prepared['targetColumn'],
        expected_manifest_sha256=prepared['manifestSHA256'],expected_graph_digest=prepared['graphDigest'])
    # Existing preparation's public float32 model boundary remains authoritative.
    # Explicit widening cannot recover precision that was already rounded there.
    train_x,train_y,val_x,val_y=[value.detach().cpu().numpy() for value in data]
    train_x=train_x.astype(np.float64);val_x=val_x.astype(np.float64)
    check_cancelled()
    emit({'event':'backendStarted','backendID':'sklearn','device':'cpu','trainingSamples':len(train_x),
          'validationSamples':len(val_x),'splitHash':provenance['splitHash']})
    model=backend.fit(request['configuration'],train_x,train_y,check_cancelled=check_cancelled)
    evaluation=backend.evaluate(model,val_x,val_y)
    stage=Path(tempfile.mkdtemp(prefix='.sklearn-run-',dir=destination.parent));owned=None
    try:
        backend.save(model,stage/'model-state.json')
        backend.export(model,stage/'portable',check_cancelled=check_cancelled)
        for name,raw in files.items():
            check_cancelled()
            # completed_inventory already enforces paths; retain a local guard at writes.
            if not isinstance(name,str) or name.startswith('/') or any(p in ('','.','..') for p in name.split('/')):
                raise ValueError('Unsafe archived preparation path.')
            path=stage/'preparation'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        inventory=[];total=0
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                raw=path.read_bytes();total+=len(raw)
                if len(raw)>50_000_000 or total>200_000_000:raise ValueError('Backend artifacts exceed storage limits.')
                inventory.append({'path':path.relative_to(stage).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
        manifest={'schemaVersion':1,'backendID':'sklearn','runtimeVersion':model['runtimeVersion'],'device':'cpu',
                  'task':'binaryClassification','stateFormat':backend.STATE_FORMAT,'configuration':model['configuration'],
                  'provenance':provenance,'inputConversion':'verified float32 preparation widened to float64',
                  'evaluation':evaluation,'solver':{'iterations':model['iterations'],'converged':model['converged']},
                  'artifacts':inventory}
        if request.get('capturePredictions',False):
            manifest['predictions']={**backend.predict(model,val_x[:8]),'rowIDs':provenance['validationRowIDs'][:8]}
        check_cancelled()
        # Reserve the destination exclusively. A marker is installed only after all
        # artifacts, so concurrent consumers cannot accept a partially published run.
        destination.mkdir(mode=0o700);info=destination.stat();owned=(info.st_dev,info.st_ino)
        for child in stage.iterdir():
            check_cancelled();os.rename(child,destination/child.name)
        check_cancelled()
        with (destination/'manifest.json').open('x') as stream:json.dump(manifest,stream,sort_keys=True,allow_nan=False)
        emit({'event':'backendCompleted',**manifest})
        check_cancelled()
        return manifest
    except BaseException:
        if owned is not None:
            try:
                info=destination.lstat()
                if (info.st_dev,info.st_ino)==owned:shutil.rmtree(destination)
            except FileNotFoundError:pass
        raise
    finally:
        shutil.rmtree(stage)


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Verified sklearn prepared-data foundation worker')
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if __package__:from .prepared_data import bounded_read
    else:from prepared_data import bounded_read
    def emit(event):print(json.dumps(event,allow_nan=False),flush=True)
    try:
        raw=bounded_read(args.request,100_000)
        if hashlib.sha256(raw).hexdigest()!=args.request_sha256:raise ValueError('Backend request hash differs from captured request.')
        def unique(pairs):
            result={}
            for key,value in pairs:
                if key in result:raise ValueError('Duplicate request field.')
                result[key]=value
            return result
        run_prepared(json.loads(raw,object_pairs_hook=unique),args.output,emit)
        return 0
    except Exception as error:
        emit({'event':'error','backendID':'sklearn','message':str(error)})
        return 1


if __name__=='__main__':raise SystemExit(main())
