"""Exact native-shaped saved-rule resources through real operation admission."""
import json
import sys
import subprocess
from pathlib import Path

import pytest

from runner.tests.test_preparation_operation_authorization import preparation_operation
from runner.tests.test_operation_authorization import operation_authority, publish, claim
from runner.tests.test_preview_operation_authorization import inventory
from runner.tests.test_evaluation_authorization import encoded, digest
from runner.tests.test_prebuilt_preparation_export import fitted_request
from runner.prebuilt_workflow import run_prepared
from runner.tests.test_worker_bootstrap import entry, run_supervisor
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_lifetime_watchdog import libraries


@pytest.fixture
def frozen_preparation(preparation_operation, tmp_path):
    from runner.polars_csv_configuration import DEFAULTS
    f=preparation_operation;folder=f[0]/'input'
    fit=tmp_path/'fit-source';fit.mkdir();r=fitted_request(fit)
    out=fit/'model';run_prepared(r,out,lambda _:None)
    model=(out/'model-state.json').read_bytes();rules=(out/'portable-prepared/preparation.json').read_bytes()
    identity=digest(model+rules);resource=folder/'preparation-models'/identity;resource.mkdir(parents=True)
    (resource/'state.json').write_bytes(model);(resource/'preparation.json').write_bytes(rules)
    mapping={'frozen':dict(path='preparation-models/'+identity+'/state.json',sha256=digest(model),
        preparationPath='preparation-models/'+identity+'/preparation.json',preparationSHA256=digest(rules))}
    (folder/'preparation-models.json').write_bytes(encoded(mapping))
    raw=b'x,city\n9,a\n10,unseen\n';source=folder/'sources'/digest(raw)/'new.csv';source.parent.mkdir(parents=True);source.write_bytes(raw)
    # Remove the old source from the request; never admit an orphan capture.
    old=json.loads((folder/'bindings.json').read_bytes())['source'];Path(old).unlink()
    graph=dict(schemaVersion=1,nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={**DEFAULTS,'emptyField':'null','schemaOverrides':{'x':'Float64','city':'String'}}),
        dict(id='frozen',operator='kwip.data.preprocess',config=dict(mode='frozen',columns=[]))],
        edges=[{'from':dict(node='source',port='table'),'to':dict(node='frozen',port='table')}])
    (folder/'graph.json').write_bytes(encoded(graph));(folder/'bindings.json').write_bytes(encoded({'source':str(source)}))
    (folder/'node-labels.json').write_bytes(encoded({'source':'New rows','frozen':'Apply saved rules'}))
    inventory(f);return f,mapping


def test_frozen_resources_are_admitted_and_live_changes_revoke(frozen_preparation):
    f,mapping=frozen_preparation
    with claim(f) as authority:
        authority.check()
        resource=f[0]/'input'/mapping['frozen']['preparationPath']
        before=resource.stat();raw=resource.read_bytes();resource.write_bytes(raw+b' ')
        with pytest.raises(ValueError):authority.check()


def test_cold_admission_never_imports_data_or_model_engines(frozen_preparation):
    f,_=frozen_preparation
    script='''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from runner.preparation_operation_contract import verify_inputs
root=Path(sys.argv[2]);verify_inputs(json.loads((root/'request.json').read_bytes()),root,{})
assert not {'numpy','torch','sklearn','polars','runner.dataflow'} & set(sys.modules)
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[2]),str(f[0])],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('change',['orphan','missing','wrong-pin','escape','extra-key','unselected-graph','other-model','changed-rules'])
def test_invalid_frozen_resource_binding_never_claims(frozen_preparation,change):
    f,mapping=frozen_preparation;folder=f[0]/'input';pin=mapping['frozen']
    if change=='orphan':(folder/Path(pin['path']).parent/'extra.json').write_bytes(b'{}')
    elif change=='missing':(folder/pin['preparationPath']).unlink()
    elif change=='wrong-pin':pin['preparationSHA256']='0'*64
    elif change=='escape':pin['path']='../outside/state.json'
    elif change=='extra-key':pin['extra']='not allowed'
    elif change=='unselected-graph':
        graph=json.loads((folder/'graph.json').read_bytes());graph['nodes'].pop();graph['edges']=[];(folder/'graph.json').write_bytes(encoded(graph))
    elif change=='other-model':
        path=folder/pin['path'];model=json.loads(path.read_bytes());model['featureColumns'].reverse();raw=encoded(model);path.write_bytes(raw);pin['sha256']=digest(raw)
    elif change=='changed-rules':
        path=folder/pin['preparationPath'];rules=json.loads(path.read_bytes());rules['parameters'][0]['rules'][0]['mean']+=1;raw=encoded(rules);path.write_bytes(raw);pin['preparationSHA256']=digest(raw)
    (folder/'preparation-models.json').write_bytes(encoded(mapping));inventory(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
def test_real_guarded_frozen_preparation_executes_without_training(frozen_preparation,tmp_path,libraries):
    from runner.dataflow_storage import completed_inventory
    from runner.dataflow_table_reader import read_table_artifact
    f,mapping=frozen_preparation;guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    path.write_text(text.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,([e for e in events if e.get('event')=='error'],error)
    assert events[-1]['event']=='dataflowCompleted'
    manifest,files,_=completed_inventory(f[0]/'artifacts/manifest.json')
    artifact=next(a for a in manifest['artifacts'] if a.get('nodeID')=='frozen' and a.get('port')=='table')
    table=read_table_artifact(f[0]/'artifacts'/artifact['path'],artifact['sha256'])
    try:
        assert len(table.rows)==2 and len(table.columns)==3
        assert list(table.rows)[1][1:]==(0.,0.)
        assert table.role=='unassigned' and len(table.fitted_states)==2
    finally:table.rows.store.close();table.row_ids.store.close()


def test_real_frozen_prediction_requires_the_applied_model_bound_rules(frozen_preparation,tmp_path,monkeypatch):
    from runner.dataflow_cli import run_dataflow
    from runner import prebuilt_models as pb, prebuilt_preparation as frozen
    f,mapping=frozen_preparation;folder=f[0]/'input';prep=tmp_path/'b-prepared'
    run_dataflow(folder/'graph.json',json.loads((folder/'bindings.json').read_bytes()),prep,lambda _:None)
    manifest=(prep/'manifest.json').read_bytes();model=folder/mapping['frozen']['path'];rules=folder/mapping['frozen']['preparationPath']
    state=pb.load(model.read_bytes())
    request=dict(schemaVersion=1,backendID='sklearn',configuration=dict(task='prebuilt',operation='predict',algorithm='ridge',settings=state['configuration']['options']),
        prepared=dict(manifest=str(prep/'manifest.json'),manifestSHA256=digest(manifest),graphDigest=json.loads(manifest)['graphDigest'],trainingNode='frozen',trainingPort='table',validationNode='frozen',validationPort='table',featureColumns=state['featureColumns'],targetColumn=''),
        model=dict(path=str(model),sha256=digest(model.read_bytes()),preparationPath=str(rules),preparationSHA256=digest(rules.read_bytes())))
    monkeypatch.setattr(pb,'fit',lambda *a,**k:pytest.fail('Prediction must never train'))
    out=tmp_path/'prediction';receipt=run_prepared(request,out,lambda _:None)
    assert receipt['evaluation']=={'count':2}
    assert (out/'model-state.json').read_bytes()==model.read_bytes()
    envelope=json.loads(rules.read_bytes());envelope['parameters'][0]['rules'][0]['mean']+=1
    envelope['contentDigest']=frozen.digest({k:v for k,v in envelope.items() if k!='contentDigest'})
    changed=tmp_path/'changed-rules.json';changed.write_bytes(pb.canonical(envelope))
    request['model'].update(preparationPath=str(changed),preparationSHA256=digest(changed.read_bytes()))
    with pytest.raises(ValueError,match='applied'):
        run_prepared(request,tmp_path/'refused',lambda _:None)
    assert not (tmp_path/'refused').exists()
