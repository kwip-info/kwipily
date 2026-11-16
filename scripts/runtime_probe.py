#!/usr/bin/env python3
"""Relocate payload, deny checkout/network access, exercise real worker lifecycle."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]

def fixture():
    specs=[('input','input',{'features':2}),('a','linear',{'in_features':2,'out_features':8}),('b','linear',{'in_features':2,'out_features':8}),('relu','relu',{}),('tanh','tanh',{}),('merge','concat',{}),('head','linear',{'in_features':16,'out_features':2}),('output','output',{})]
    wires=[('input','a','in'),('input','b','in'),('a','relu','in'),('b','tanh','in'),('relu','merge','a'),('tanh','merge','b'),('merge','head','in'),('head','output','in')]
    return {'schemaVersion':1,'name':'Runtime portability probe','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],'edges':[{'id':f'e{n}','source':s,'sourcePort':'out','target':t,'targetPort':p} for n,(s,t,p) in enumerate(wires)]}

def probe(source,report):
    parent=Path(tempfile.mkdtemp(prefix='KwipML-runtime-relocated-'))
    runtime=parent/'Relocated KwipML.app/Contents/Resources/KwipMLRuntime'
    runtime.parent.mkdir(parents=True);shutil.copytree(source,runtime,symlinks=True)
    graph=parent/'graph.json';graph.write_text(json.dumps(fixture()))
    csv=parent/'private-source.csv';csv.write_text('first,second,label\n'+'\n'.join(f'{-1+i*.04},{i*.017},{int(i>=30)}' for i in range(60)))
    manifest=json.loads((runtime/'runtime-manifest.json').read_text());python=runtime/manifest['pythonExecutable']
    denied=[ROOT,Path.home()/'.local/share/uv/python',Path('/Library/Frameworks/Python.framework'),Path('/opt/homebrew')]
    profile='(version 1) (allow default) (deny network*) '+' '.join(f'(deny file-read* (subpath {json.dumps(str(p.resolve()))}))' for p in denied)
    prefix=['/usr/bin/sandbox-exec','-p',profile,str(python),'-I','-B']
    env=dict(os.environ);env.update({'PATH':'/usr/bin:/bin','PYTHONPATH':'/unavailable/host-python','PYTHONHOME':'/unavailable/host-python'})
    def command(args,timeout=60):
        result=subprocess.run([*prefix,str(runtime/'launch.py'),*args],cwd=parent,env=env,capture_output=True,text=True,timeout=timeout)
        if result.returncode: raise RuntimeError(result.stdout+'\n'+result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]
    # Prove the restricted process cannot read the checkout, not just that our commands omit it.
    denial=subprocess.run([*prefix,'-c',f'from pathlib import Path;Path({str(ROOT/"AGENTS.md")!r}).read_text()'],cwd=parent,env=env,capture_output=True,text=True)
    assert denial.returncode!=0 and 'PermissionError' in denial.stderr,denial.stderr
    capabilities=command(['--capabilities'])[0];runs=[]
    for device in capabilities['devices']:
        output=parent/f'run-{device}'
        events=command(['--graph',str(graph),'--output',str(output),'--dataset-csv',str(csv),'--feature-columns','first,second','--target-column','label','--device',device,'--epochs','30'])
        completed=events[-1];assert completed['event']=='completed'
        assert len([e for e in events if e['event']=='epoch'])==30
        assert all((output/name).exists() for name in ['manifest.json','model.py','checkpoint.pt','metrics.json','graph.json'])
        assert completed['accuracy']>=.8,completed
        code=command(['--graph',str(graph),'--export-code'])[0];assert code['event']=='codeExport'
        preview=command(['--graph',str(graph),'--output',str(parent/f'preview-{device}'),'--preview','--device',device,'--checkpoint',str(output/'checkpoint.pt'),'--dataset-csv',str(csv),'--feature-columns','first,second','--target-column','label'])[-1]
        assert preview['weights']=='checkpoint'
        # Import standalone exported code using only bundled Python/PyTorch, then compare.
        parity_code=f'''import sys,json,torch,importlib.util
sys.path.insert(0,{str(runtime/'runner')!r})
from engine import GraphModel
spec=importlib.util.spec_from_file_location('exported',{str(output/'model.py')!r}); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
state=torch.load({str(output/'checkpoint.pt')!r},map_location='cpu',weights_only=True)['modelState']
a=module.Model();a.load_state_dict(state)
b=GraphModel(json.loads(open({str(graph)!r}).read()),{str(runtime/'components.json')!r});b.load_state_dict(state)
x=torch.randn(8,2);assert torch.equal(a(x),b(x));print('parity passed')'''
        parity=subprocess.run([*prefix,'-c',parity_code],cwd=parent,env=env,capture_output=True,text=True,timeout=30)
        assert parity.returncode==0,parity.stderr
        runs.append({'device':device,'epochs':30,'accuracy':completed['accuracy'],'validationLoss':completed['validationLoss'],'exportParity':True,'checkpointPreview':True})
    cancelled=parent/'cancelled'
    proc=subprocess.Popen([*prefix,str(runtime/'launch.py'),'--graph',str(graph),'--output',str(cancelled),'--epochs','10000','--device','cpu'],cwd=parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    first=json.loads(proc.stdout.readline());assert first['event']=='started',first
    proc.terminate();stdout,stderr=proc.communicate(timeout=15)
    assert proc.returncode!=0 and not (cancelled/'manifest.json').exists()
    assert not any(json.loads(line)['event']=='completed' for line in stdout.splitlines())
    result={'schemaVersion':1,'relocatedRuntime':str(runtime),'workingDirectory':str(parent),'checkoutAccessDenied':True,'networkPolicy':'deny network*','hostPythonEnvironmentIgnored':True,'capabilities':capabilities,'runs':runs,'cancellation':{'returnCode':proc.returncode,'completedEvent':False,'manifestPresent':False},'limitations':['Seatbelt command profile is not App Sandbox entitlement validation.','Same account and same machine; clean-account installation not tested.','Runtime payload was relocated inside app resource layout; native UI lifecycle is a separate test.']}
    Path(report).write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runtime');parser.add_argument('--report',required=True)
    args=parser.parse_args();print(json.dumps(probe(Path(args.runtime).resolve(),args.report),indent=2))
