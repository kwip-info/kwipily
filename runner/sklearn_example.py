"""Run the bundled foundation through real preparation and the sklearn bridge."""
import hashlib
import json
from pathlib import Path
import sys
if __package__:
    from .dataflow_cli import run_dataflow
    from .sklearn_workflow import run_prepared
else:
    from dataflow_cli import run_dataflow
    from sklearn_workflow import run_prepared


def run_example(destination,emit=lambda _:None):
    root=Path(__file__).resolve().parents[1]
    example=json.loads((root/'runner/backend_examples/sklearn_logistic.json').read_text())
    destination=Path(destination);destination.mkdir(mode=0o700)
    prepared=destination/'prepared'
    run_dataflow(root/example['preparationExample'],{example['sourceNode']:str(root/example['sourceExample'])},prepared,emit)
    raw=(prepared/'manifest.json').read_bytes()
    selected={key:example[key] for key in ('trainingNode','trainingPort','validationNode','validationPort','featureColumns','targetColumn')}
    selected.update(manifest=str(prepared/'manifest.json'),manifestSHA256=hashlib.sha256(raw).hexdigest(),graphDigest=json.loads(raw)['graphDigest'])
    return run_prepared({'schemaVersion':1,'backendID':'sklearn','configuration':example['configuration'],'prepared':selected},destination/'trained',emit)


if __name__=='__main__':
    run_example(sys.argv[1],lambda item:print(json.dumps(item,allow_nan=False),flush=True))
