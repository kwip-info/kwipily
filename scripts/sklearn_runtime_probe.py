#!/usr/bin/env python3
"""Relocated sklearn closure fit/reload probe; no distribution acceptance claim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]


def probe(runtime,report):
    temporary=Path(tempfile.mkdtemp(prefix='KwipML-sklearn-closure-'))
    shutil.copytree(Path(runtime)/'python',temporary/'python',symlinks=True)
    shutil.copyfile(ROOT/'runner/sklearn_backend.py',temporary/'sklearn_backend.py')
    denied=[ROOT,Path('/Library/Frameworks/Python.framework'),Path('/opt/homebrew'),Path.home()/'.local/share/uv/python']
    profile='(version 1) (allow default) (deny network*) '+' '.join(f'(deny file-read* (subpath {json.dumps(str(path))}))' for path in denied)
    env=dict(os.environ);env.update(PATH='/usr/bin:/bin',PYTHONPATH='/missing',PYTHONHOME='/missing')
    prefix=['/usr/bin/sandbox-exec','-p',profile,str(temporary/'python/bin/python3.13'),'-I','-B']
    denial=subprocess.run([*prefix,'-c',f'open({str(ROOT/"AGENTS.md")!r}).read()'],cwd=temporary,env=env,capture_output=True,text=True,timeout=30)
    if denial.returncode==0 or 'PermissionError' not in denial.stderr:raise RuntimeError('Checkout denial was not proved.')
    code='''import sys,json,importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,sklearn,scipy
from sklearn_backend import fit,predict,save,load,export,default_configuration
x=np.array([[-3.,0.],[-2.,1.],[-1.,-1.],[1.,1.],[2.,-1.],[3.,0.]])
y=np.array([0,0,0,1,1,1],dtype=np.int64)
model=fit(default_configuration(),x,y)
save(model,'model.json')
a=predict(model,x);b=predict(load('model.json'),x)
assert a==b and a['classes']==y.tolist()
export(model,'portable')
print(json.dumps({'sklearn':sklearn.__version__,'scipy':scipy.__version__,'numpy':np.__version__,'fit':True,'reloadParity':True,'probabilities':a['probabilities']}))
'''
    result=subprocess.run([*prefix,'-c',code],cwd=temporary,env=env,capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    value={'schemaVersion':1,'relocatedDirectory':str(temporary),'checkoutDenied':True,'hostPythonEnvironmentIgnored':True,'networkDenied':True,'actual':json.loads(result.stdout),'limitations':['Same host/account, not App Sandbox signing or clean-machine acceptance.','Native supervision and UI not exercised by this closure probe.']}
    Path(report).write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runtime');parser.add_argument('--report',required=True)
    args=parser.parse_args();print(json.dumps(probe(args.runtime,args.report),indent=2))
