import json
from pathlib import Path
import subprocess
import sys
import torch
from runner.engine import capabilities,resolve_device


def assert_contract(event):
    c=event['executionContract']
    assert c['schemaVersion']==1
    assert c['adapterID']=='kwip.runner.product-execution.v1'
    assert c['runtimeVersion']==event['torchVersion']==str(torch.__version__)
    assert c['availableDevices']==event['devices']
    assert c['supportedDevices']==['cpu','mps']
    assert c['graphSchemaVersions']==[1,2] and c['biasModes']==[False,True]
    assert c['tensor']=={'rank':2,'dtype':'float32','layout':'denseStrided','noncontiguous':True,'minimumBatch':1,'preservesRows':True}
    assert c['task']=={'kind':'binaryClassification','inputFeatures':2,'outputFeatures':2}
    assert c['limits']=={'nodes':128,'edges':256,'parameters':2000000,'configuredWidth':512,'intermediateWidth':4096}
    assert c['operations']=={'inference':'boundedGraph','training':'experimentOwnedAdam','exactResume':'cpuEpochBoundary','fineTune':'weightsOnlyFreshOptimizer','export':'pythonSourceRawTorch'}
    assert c['finiteChecks']==['sourcePartitions','initialParameters','publicForwardInputsOutputs','inspectedStages','validationLogitsAndLoss','trainingLoss']
    assert c['rawModelSemantics']=='pytorch'

def test_actual_cli_contract_matches_discovery():
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,'-B',str(root/'runner/main.py'),'--capabilities'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert_contract(json.loads(result.stdout))

def test_unavailable_mps_and_returned_mutation_do_not_change_contract(monkeypatch):
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    e=capabilities();assert_contract(e);assert e['devices']==['cpu']
    e['executionContract']['supportedDevices'].clear()
    assert_contract(capabilities())
    assert resolve_device('auto')=='cpu'
