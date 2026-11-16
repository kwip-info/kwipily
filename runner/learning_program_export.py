"""Actual bounded executable-program lessons; no expected answers enter execution."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import torch
if __package__:
    from .program_export import build_package, reload_program
    from .program_export_configuration import normalize_program_export
else:
    from program_export import build_package, reload_program
    from program_export_configuration import normalize_program_export


def _canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _sha(raw): return hashlib.sha256(raw).hexdigest()

def _configuration(submission):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','taskID','recipe'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='programExportLearningAction' or submission['taskID'] not in ('program-export-computation','program-export-size-range'):
        raise ValueError('Use this versioned executable-program lesson, never authored answers or code.')
    recipe=submission['recipe']
    if type(recipe) is not dict or set(recipe)!={'schemaVersion','componentID','configurationVersion','config'} or any(type(recipe[k]) is not int or recipe[k]!=1 for k in ('schemaVersion','configurationVersion')) or recipe['componentID']!='program_export' or len(_canonical(recipe))>32768:
        raise ValueError('Use the bounded program export recipe.')
    return normalize_program_export(recipe['config'])

def _inputs(inputs,parse_tensor):
    if type(inputs) is not dict or set(inputs)!={'sourceRows','parameter.weight','parameter.bias'}:
        raise ValueError('Supply only source rows and the actual selected learned arrays.')
    result={}
    for key,raw in inputs.items():
        shape,values=parse_tensor(raw)
        if type(shape) not in (list,tuple) or not shape or any(type(n) is not int or not 1<=n<=4 for n in shape) or len(values)!=math.prod(shape) or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>16 for v in values):
            raise ValueError('Program lesson arrays must contain bounded finite numeric cells.')
        result[key]=(list(shape),values)
    if result['sourceRows'][0][1:]!=[2] or result['parameter.weight'][0]!=[2,2] or result['parameter.bias'][0]!=[2]:
        raise ValueError('Use two input fields and two result fields with aligned learned arrays.')
    return result

def execute(submission,fixtures,evidence,stopped,parse_tensor):
    configuration=_configuration(submission)
    for fixture in fixtures:
        if stopped():return
        parsed=_inputs(fixture['inputs'],parse_tensor)
        shape,values=parsed['sourceRows']
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],sum(len(v) for _,v in parsed.values())+configuration['maximum_batch_size']*8)
        if stopped():return
        device=configuration['device']
        # The trusted fixture supplies a selected snapshot. Saving it provides an
        # actual immutable parent byte check, not a claim of training provenance.
        model=torch.nn.Linear(2,2,device=device)
        with torch.no_grad():
            model.weight.copy_(torch.tensor(parsed['parameter.weight'][1],dtype=torch.float32,device=device).reshape(2,2))
            model.bias.copy_(torch.tensor(parsed['parameter.bias'][1],dtype=torch.float32,device=device))
        x=torch.tensor(values,dtype=torch.float32,device=device).reshape(shape)
        example=x[:1].repeat(configuration['example_batch_size'],1)
        def check_cancelled():
            if stopped():raise InterruptedError('Lesson execution cancelled.')
        with tempfile.TemporaryDirectory(prefix='kwipml-program-lesson-') as directory:
            root=Path(directory);parent=root/'selected-snapshot.pt'
            torch.save(model.state_dict(),parent);before=parent.read_bytes()
            snapshot=_sha(before)
            identity={'runID':str(uuid.uuid4()).upper(),'requestIdentity':_sha(_canonical(submission)),'modelManifestSHA256':snapshot,'exportManifestSHA256':_sha(_canonical(fixture['inputs']))}
            destination=root/'program'
            manifest=build_package(model,example,{'dtype':'float32','sampleShape':[2]},configuration,destination,source_identity=identity,check_cancelled=check_cancelled)
            raw=(destination/'program.pt2').read_bytes();pin=_sha(_canonical(manifest))
            restored=reload_program(raw,manifest,expected_manifest_sha256=pin)
            output=restored(x)
            checks=[]
            wanted_mode='static' if submission['taskID']=='program-export-computation' else 'bounded'
            wanted=(2,2,2) if wanted_mode=='static' else (2,1,4)
            requested_policy=(manifest['configuration']['batch_mode']==wanted_mode and tuple(manifest['configuration'][key] for key in ('example_batch_size','minimum_batch_size','maximum_batch_size'))==wanted)
            for count in (configuration['minimum_batch_size'],configuration['maximum_batch_size']):
                y=restored(torch.zeros(count,2,dtype=torch.float32,device=device));checks.append(int(list(y.shape)==[count,2]))
            for count in (configuration['minimum_batch_size']-1,configuration['maximum_batch_size']+1):
                try:restored(torch.zeros(count,2,dtype=torch.float32,device=device))
                except ValueError:checks.append(1)
                else:checks.append(0)
            checks[0] *= int(requested_policy)
            # Real isolated process has no repository path or original model
            # class source. Its output is checked against the parent process's
            # actual reload, never a private expected-answer field.
            request=root/'rows.json';request.write_bytes(_canonical({'shape':shape,'values':values,'device':device}))
            script=root/'portable_check.py'
            script.write_text("import sys,json,torch\nsys.path.insert(0,sys.argv[1])\nfrom inference import load_model\nr=json.load(open(sys.argv[2]))\nx=torch.tensor(r['values'],dtype=torch.float32,device=r['device']).reshape(r['shape'])\ny=load_model().predict(x)\nprint(json.dumps({'shape':list(y.shape),'values':y.cpu().reshape(-1).tolist()},allow_nan=False))\n")
            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
            check_cancelled()
            done=subprocess.run([sys.executable,'-I',str(script),str(destination),str(request)],cwd=root,env=env,capture_output=True,text=True,timeout=30)
            if done.returncode!=0:raise ValueError('The isolated portable program reload failed: '+done.stderr[-1000:])
            try:portable=json.loads(done.stdout)
            except (ValueError,TypeError):raise ValueError('The portable process returned an invalid measurement.') from None
            isolated=int(portable=={'shape':list(output.shape),'values':output.detach().cpu().reshape(-1).tolist()})
            artifacts=[isolated]
            for changed,signature in ((raw[:-1]+bytes([raw[-1]^1]),pin),(raw,'0'*64)):
                try:reload_program(changed,manifest,expected_manifest_sha256=signature)
                except ValueError:artifacts.append(1)
                else:artifacts.append(0)
            difference=sum(a!=b for a,b in zip(before,parent.read_bytes()))+abs(len(before)-parent.stat().st_size)
            check_cancelled()
            measured={'reloadedOutput':{'shape':list(output.shape),'values':output.detach().cpu().reshape(-1).tolist()},'constraintChecks':{'shape':[4],'values':checks},'artifactChecks':{'shape':[3],'values':artifacts},'parentByteDifference':{'shape':[1],'values':[difference]}}
            evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':measured,'partitions':{}})
