"""Explicit maintenance tool: capture upstream native CPU masks, never KwipML outputs."""
import hashlib
import json
from pathlib import Path
import torch

EXPECTED_RUNTIME='2.14.1'
if str(torch.__version__)!=EXPECTED_RUNTIME:raise RuntimeError('Refuse stale mask regeneration on another Torch runtime.')
entries=[]
for seed in (7,41,173,911):
    for rows in (2,3,4):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            output=torch.nn.Dropout(p=.5,inplace=False).train()(torch.ones((rows,2,3),device='cpu',dtype=torch.float64))
        values=output.reshape(-1).tolist()
        assert set(values)<=set((0.,2.))
        entries.append({'seed':seed,'shape':[rows,2,3],'mask':[int(v==2) for v in values]})
source=Path(__file__).read_bytes()
bank={'schemaVersion':1,'maskVersion':'kwip.dropout.cpu-float64.v1','torchVersion':EXPECTED_RUNTIME,'device':'cpu','dtype':'float64','p':.5,'schedule':'reset declared native seed once; one nn.Dropout training call on ones', 'source':'upstream torch.nn.Dropout, independent of KwipML adapter','generatorSHA256':hashlib.sha256(source).hexdigest(),'entries':entries}
raw=(json.dumps(bank,sort_keys=True,indent=2)+'\n').encode()
base=Path(__file__).resolve().parents[2]
for path in (base/'runner/fixtures/remaining_activation_dropout_masks_v1.json',base/'Sources/KwipMLCore/Resources/remaining_activation_dropout_masks_v1.json'):
    path.write_bytes(raw)
print(hashlib.sha256(raw).hexdigest())
