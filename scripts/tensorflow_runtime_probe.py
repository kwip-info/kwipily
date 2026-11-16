#!/usr/bin/env python3
"""Read-only feasibility probe of an explicitly supplied development runtime.

Relocates its Python closure plus a model export, denies checkout/host Python and
network reads, and executes real TensorFlow weights reload/inference alongside
PyTorch. Does not claim App Sandbox, signing, or clean-machine acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def probe(runtime,model,report):
    runtime=Path(runtime).resolve();model=Path(model).resolve()
    root=Path(tempfile.mkdtemp(prefix='KwipML-TensorFlow-closure-'))
    shutil.copytree(runtime/'python',root/'python',symlinks=True)
    shutil.copytree(model,root/'model',symlinks=False)
    python=root/'python/bin/python3.13'
    denied=[ROOT,Path('/Library/Frameworks/Python.framework'),Path('/opt/homebrew'),Path.home()/'.local/share/uv/python']
    profile='(version 1) (allow default) (deny network*) '+' '.join(f'(deny file-read* (subpath {json.dumps(str(path))}))' for path in denied)
    env=dict(os.environ);env.update(PATH='/usr/bin:/bin',PYTHONPATH='/missing/host-python',PYTHONHOME='/missing/host-python')
    prefix=['/usr/bin/sandbox-exec','-p',profile,str(python),'-I','-B']
    denial=subprocess.run([*prefix,'-c',f'open({str(ROOT/"AGENTS.md")!r}).read()'],cwd=root,env=env,capture_output=True,text=True,timeout=30)
    if denial.returncode==0 or 'PermissionError' not in denial.stderr:raise RuntimeError('Checkout denial was not proved.')
    code='''import sys,json,importlib.metadata as metadata
from pathlib import Path
import tensorflow as tf,numpy as np,torch,sklearn
from sklearn.linear_model import LogisticRegression
assert sklearn.__version__=="1.7.2"
classifier=LogisticRegression(random_state=42).fit([[-2.,-1.],[-1.,-2.],[1.,2.],[2.,1.]],[0,0,1,1])
assert classifier.predict([[3.,2.]]).tolist()==[1]
sys.path.insert(0,str(Path.cwd()/"model"))
from tensorflow_foundation import load,predict
model=load(Path.cwd()/"model")
x=np.array([[0.,0.],[1.,-1.]],dtype=np.float32)
a=predict(model,x);b=predict(load(Path.cwd()/"model"),x)
assert np.array_equal(a,b) and np.isfinite(a).all()
assert torch.equal(torch.tensor([2.])*3,torch.tensor([6.]))
print(json.dumps({"tensorflow":tf.__version__,"keras":tf.keras.__version__,"numpy":np.__version__,"torch":torch.__version__,"sklearn":sklearn.__version__,"sklearnFitPredict":True,"logits":a.tolist(),"reloadParity":True,"packages":dict(sorted((d.metadata["Name"],d.version) for d in metadata.distributions()))}))
'''
    result=subprocess.run([*prefix,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    observed=json.loads(result.stdout)
    value={'schemaVersion':1,'relocatedDirectory':str(root),'checkoutDenied':True,'hostPythonEnvironmentIgnored':True,'networkDenied':True,'actual':observed,'limitations':['Same host/account; not App Sandbox or distribution signing acceptance.','Worker/bootstrap dispatch not part of this dependency closure probe.']}
    Path(report).write_text(json.dumps(value,indent=2)+'\n');return value

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runtime');parser.add_argument('model');parser.add_argument('--report',required=True)
    args=parser.parse_args();print(json.dumps(probe(args.runtime,args.model,args.report),indent=2))
