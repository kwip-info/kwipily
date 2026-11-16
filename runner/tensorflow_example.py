"""Run the bundled preparation → TensorFlow → portable inference foundation.

Development harness; does not bypass native operation admission or advertise an
unverified runtime as installed in the app. Outputs require a new directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .dataflow_cli import run_dataflow
from .tensorflow_workflow import run_prepared

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    args.directory.mkdir(parents=True,exist_ok=False)
    prepared=args.directory/'prepared'
    run_dataflow(root/'examples/dataflow/scaler.json',{'source':str(root/'runner/backend_examples/tensorflow_observations.csv')},prepared,lambda _:None)
    raw=(prepared/'manifest.json').read_bytes();example=json.loads((root/'runner/backend_examples/tensorflow_classifier.json').read_text())
    request={'schemaVersion':1,'backendID':'tensorflow','configuration':{'graph':example['graph'],'settings':example['settings']},'prepared':{'manifest':str(prepared/'manifest.json'),'manifestSHA256':hashlib.sha256(raw).hexdigest(),'graphDigest':json.loads(raw)['graphDigest'],'trainingNode':'training','trainingPort':'table','validationNode':'heldout','validationPort':'table','featureColumns':['feature','second'],'targetColumn':'group'}}
    run_prepared(request,args.directory/'run',lambda event:print(json.dumps(event),flush=True))

if __name__=='__main__':main()
