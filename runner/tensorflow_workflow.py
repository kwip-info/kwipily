"""TensorFlow foundation consumes the existing pinned preparation lifecycle.

No source resplitting, source permissions, native run acceptance, or worker process
admission are invented here. The supervised caller supplies those boundaries.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
if __package__:
    from . import tensorflow_foundation as backend
    from .prepared_data import load_prepared
else:
    import tensorflow_foundation as backend
    from prepared_data import load_prepared

PREPARED_KEYS={'manifest','manifestSHA256','graphDigest','trainingNode','trainingPort','validationNode','validationPort','featureColumns','targetColumn'}

def run_prepared(request,output,emit,*,check_cancelled=lambda:None):
    check_cancelled()
    if not isinstance(request,dict) or not {'schemaVersion','backendID','configuration','prepared'}<=set(request)<= {'schemaVersion','backendID','configuration','prepared','capturePredictions'} or type(request['schemaVersion']) is not int or request['schemaVersion']!=1 or request['backendID']!='tensorflow':raise ValueError('Invalid TensorFlow workflow request.')
    capture=request.get('capturePredictions',False)
    if type(capture) is not bool:raise ValueError('capturePredictions must be boolean.')
    config=request['configuration'];selection=request['prepared']
    if not isinstance(config,dict) or set(config)!={'graph','settings'}:raise ValueError('TensorFlow configuration requires graph and settings.')
    backend.validate_graph(config['graph']);backend.validate_configuration(config['settings'])
    if not isinstance(selection,dict) or set(selection)!=PREPARED_KEYS:raise ValueError('Complete pinned preparation selection required.')
    for key in ('manifestSHA256','graphDigest'):
        if not isinstance(selection[key],str) or re.fullmatch('[0-9a-f]{64}',selection[key]) is None:raise ValueError('Preparation requires SHA-256 pins.')
    output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    arrays,provenance,files=load_prepared(selection['manifest'],selection['trainingNode'],selection['trainingPort'],selection['validationNode'],selection['validationPort'],selection['featureColumns'],selection['targetColumn'],expected_manifest_sha256=selection['manifestSHA256'],expected_graph_digest=selection['graphDigest'])
    check_cancelled();output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.tensorflow-run-',dir=output.parent));published=False
    try:
        runtime=backend.capabilities()['runtimeVersion']
        emit({'event':'backendStarted','backendID':'tensorflow','runtimeVersion':runtime,'device':'cpu'})
        x,y,v,w=[a.numpy() for a in arrays]
        model,history=backend.fit(config['graph'],x,y,v,w,config['settings'],check_cancelled=check_cancelled,event=lambda metrics:emit({'event':'backendEpoch','backendID':'tensorflow',**metrics}))
        check_cancelled();evaluation=backend.evaluate(model,v,w)
        backend.export(model,config['graph'],stage/'portable')
        for name,content in files.items():
            check_cancelled();path=stage/'preparation'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        (stage/'history.json').write_text(json.dumps(history,allow_nan=False))
        inventory=[{'path':str(p.relative_to(stage)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()]
        manifest={'schemaVersion':1,'backendID':'tensorflow','runtimeVersion':runtime,'device':'cpu','task':'binaryClassification','configuration':{'graph':backend.portable_graph(config['graph']),'settings':config['settings']},'provenance':provenance,'evaluation':evaluation,'artifacts':inventory}
        if capture:
            logits=backend.predict(model,v[:8])
            manifest['predictions']={'rowIDs':provenance['validationRowIDs'][:8],'logits':logits.tolist(),'classes':logits.argmax(axis=1).tolist()}
        (stage/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,allow_nan=False))
        check_cancelled()
        if output.exists() or output.is_symlink():raise FileExistsError(output)
        stage.rename(output);published=True
        emit({'event':'backendCompleted','backendID':'tensorflow','manifest':str(output/'manifest.json'),'evaluation':evaluation})
        return manifest
    except BaseException:
        shutil.rmtree(output if published else stage,ignore_errors=True);raise
