"""Standalone preparation retains sources, labels and exact selected reuse bytes."""
import json
import os
from pathlib import Path
import shutil
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.tests.test_preview_operation_authorization import inventory
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries
ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def preparation_operation(operation_authority):
    f=operation_authority;v=f[1]
    for key in ('runID','directoryID','phase'):del v[key]
    folder=f[0]/'input';folder.mkdir()
    (folder/'graph.json').write_bytes((ROOT/'examples/dataflow/scaler.json').read_bytes())
    raw=(ROOT/'examples/dataflow/observations.csv').read_bytes()
    source=folder/'sources'/digest(raw)/'observations.csv';source.parent.mkdir(parents=True);source.write_bytes(raw)
    (folder/'bindings.json').write_bytes(encoded({'source':str(source)}))
    (folder/'node-labels.json').write_bytes(encoded({'source':'Original observations','columns':'Convert columns'}))
    v.update(schemaVersion=2,toolMode='dataPreparation',arguments=['--execute-dataflow',str(folder/'graph.json'),
        '--source-bindings',str(folder/'bindings.json'),'--dataflow-node-labels',str(folder/'node-labels.json'),
        '--output',str(f[0]/'artifacts'),'--dataflow-cache-policy','off'])
    inventory(f);return f


def test_preparation_claim_rechecks_owned_source(preparation_operation):
    f=preparation_operation
    with claim(f) as authority:
        source=Path(json.loads((f[0]/'input/bindings.json').read_bytes())['source']);before=source.stat()
        raw=source.read_bytes();source.write_bytes(b'x'+raw[1:]);os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns))
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['duplicate-option','registry','unknown-policy','orphan-reuse','wrong-output','wrong-graph',
    'extra-source-binding','missing-source-binding','source-escape','unknown-label','blank-label','extra-input','digest-directory',
    'bool-size','graph-schema','duplicate-node','source-symlink','input-replacement'])
def test_invalid_preparation_never_claims(preparation_operation,change):
    f=preparation_operation;v=f[1];folder=f[0]/'input';args=v['arguments']
    graph=json.loads((folder/'graph.json').read_bytes());bindings=json.loads((folder/'bindings.json').read_bytes())
    labels=json.loads((folder/'node-labels.json').read_bytes())
    if change=='duplicate-option':args+=['--output',str(f[0]/'artifacts')]
    elif change=='registry':args+=['--registry','/tmp/registry']
    elif change=='unknown-policy':args[-1]='unknown'
    elif change=='orphan-reuse':args+=['--dataflow-reuse-manifest','/tmp/manifest','--dataflow-reuse-manifest-sha256','0'*64]
    elif change=='wrong-output':args[args.index('--output')+1]='/tmp/escape'
    elif change=='wrong-graph':args[1]='/tmp/graph.json'
    elif change=='extra-source-binding':bindings['unknown']=bindings['source']
    elif change=='missing-source-binding':bindings={}
    elif change=='source-escape':bindings['source']='/tmp/source.csv'
    elif change=='unknown-label':labels['unknown']='Label'
    elif change=='blank-label':labels['source']=' '
    elif change=='graph-schema':graph['schemaVersion']=True
    elif change=='duplicate-node':graph['nodes'].append(dict(graph['nodes'][0]))
    elif change=='digest-directory':
        source=Path(bindings['source']);new=source.parent.with_name('0'*64);source.parent.rename(new);bindings['source']=str(new/source.name)
    (folder/'graph.json').write_bytes(encoded(graph));(folder/'bindings.json').write_bytes(encoded(bindings));(folder/'node-labels.json').write_bytes(encoded(labels))
    inventory(f)
    if change=='extra-input':v['inputFiles'].append(dict(path='input/other.json',bytes=0,sha256=digest(b'')))
    elif change=='bool-size':v['inputFiles'][0]['bytes']=True
    elif change=='source-symlink':
        path=Path(bindings['source']);path.rename(path.with_name('old'));path.symlink_to('old')
    elif change=='input-replacement':
        path=folder/'bindings.json';path.unlink();os.mkfifo(path)
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


def test_source_budget_preserves_more_than_experiment_256mb(preparation_operation):
    from runner.preparation_operation_contract import parse_contract
    f=preparation_operation
    controls=[entry for entry in f[1]['inputFiles'] if '/sources/' not in entry['path']]
    sources=[dict(path='input/sources/'+format(i,'064x')+'/source.csv',bytes=10000000,sha256=format(i,'064x')) for i in range(30)]
    f[1]['inputFiles']=controls+sources
    options,files=parse_contract(f[1],f[0])
    assert len(files)==33 and options['--dataflow-cache-policy']=='off'


def add_reuse(f,tmp_path):
    from runner.dataflow_cli import run_dataflow
    prior=tmp_path/'prior'
    run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},prior,lambda _:None,cache_policy='materialize')
    staged=f[0]/'input/reuse';shutil.copytree(prior,staged)
    args=f[1]['arguments'];args[-1]='reuse'
    args+=['--dataflow-reuse-manifest',str(staged/'manifest.json'),'--dataflow-reuse-manifest-sha256',digest((staged/'manifest.json').read_bytes())]
    inventory(f)


@pytest.mark.parametrize('change',['different-current-graph','missing-file','different-pin','extra-artifact'])
def test_selected_reuse_inventory_is_complete_but_not_current_graph(preparation_operation,tmp_path,change):
    f=preparation_operation;add_reuse(f,tmp_path)
    if change=='different-current-graph':
        path=f[0]/'input/graph.json';graph=json.loads(path.read_bytes());next(node for node in graph['nodes'] if node['id']=='filter')['config']['conditions'][0]['value']=1;path.write_bytes(encoded(graph));inventory(f)
    elif change=='missing-file':f[1]['inputFiles'].pop()
    elif change=='different-pin':f[1]['arguments'][-1]='0'*64
    else:
        (f[0]/'input/reuse/other').write_bytes(b'');inventory(f)
    publish(f)
    if change=='different-current-graph':
        with claim(f) as authority:authority.check()
    else:
        with pytest.raises((ValueError,OSError)):claim(f)


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('policy',['off','materialize','reuse'])
def test_real_guarded_preparation_policy_and_complete_artifacts(preparation_operation,tmp_path,libraries,policy):
    from runner.dataflow_storage import completed_inventory
    f=preparation_operation
    if policy=='reuse':add_reuse(f,tmp_path)
    else:f[1]['arguments'][-1]=policy;publish(f)
    guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    path.write_text(text.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,(events,error)
    assert events[0]['toolMode']=='dataPreparation' and events[-1]['event']=='dataflowCompleted'
    manifest,files,_=completed_inventory(f[0]/'artifacts/manifest.json')
    assert manifest['cache']['policy']==policy and files['prep.py']
    if policy=='reuse':assert manifest['cache']['reusedNodeIDs'] and 'source' in manifest['cache']['executedNodeIDs']


@pytest.mark.parametrize('policy',['materialize','reuse'])
@pytest.mark.parametrize('limit',['1','200000000'])
def test_cache_allowance_is_pinned_at_both_authorization_boundaries(preparation_operation,tmp_path,policy,limit):
    from runner.worker_authorization import validate_phase_arguments
    f=preparation_operation
    if policy=='reuse':add_reuse(f,tmp_path)
    else:f[1]['arguments'][-1]=policy
    f[1]['arguments']+=['--dataflow-cache-limit-bytes',limit]
    phase_args=list(f[1]['arguments']);index=phase_args.index('--dataflow-node-labels');del phase_args[index:index+2]
    assert validate_phase_arguments('preparation',phase_args)['--dataflow-cache-limit-bytes']==limit
    publish(f)
    with claim(f) as authority:authority.check()


@pytest.mark.parametrize('value',['0','200000001','-1','1.5','true','01','+1',' 1',True])
def test_invalid_cache_allowance_never_claims(preparation_operation,value):
    from runner.worker_authorization import validate_phase_arguments
    f=preparation_operation;f[1]['arguments'][-1]='materialize'
    f[1]['arguments']+=['--dataflow-cache-limit-bytes',value]
    phase_args=list(f[1]['arguments']);index=phase_args.index('--dataflow-node-labels');del phase_args[index:index+2]
    with pytest.raises(ValueError):validate_phase_arguments('preparation',phase_args)
    publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.parametrize('change',['off','missing-policy','repeat','training'])
def test_cache_allowance_scope_is_enforced(preparation_operation,change):
    from runner.worker_authorization import validate_phase_arguments
    f=preparation_operation;args=f[1]['arguments'];args[-1]='materialize'
    if change=='off':args[-1]='off'
    if change=='missing-policy':del args[-2:]
    args+=['--dataflow-cache-limit-bytes','1000']
    if change=='repeat':args+=['--dataflow-cache-limit-bytes','1000']
    phase_args=list(args);index=phase_args.index('--dataflow-node-labels');del phase_args[index:index+2]
    with pytest.raises(ValueError):validate_phase_arguments('training' if change=='training' else 'preparation',phase_args)
    if change!='training':
        publish(f)
        with pytest.raises(ValueError):claim(f)
        assert not (f[0]/'claim.json').exists()


def test_each_authority_check_parses_preparation_contract_once(preparation_operation,monkeypatch):
    from runner import operation_authorization as authority_module
    from runner import preparation_operation_contract as contract
    original=contract.parse_contract;calls=[]
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(authority_module,'parse_preparation',counted)
    monkeypatch.setattr(contract,'parse_contract',counted)
    with claim(preparation_operation) as authority:
        calls.clear();authority.check()
        assert len(calls)==1


def test_warm_authority_checks_still_reject_in_memory_type_changes(preparation_operation):
    with claim(preparation_operation) as authority:
        authority.check()
        authority.request['schemaVersion']=2.0
        with pytest.raises(ValueError):authority.check()


def test_warm_authority_checks_reject_identical_control_replacement(preparation_operation):
    with claim(preparation_operation) as authority:
        authority.check()
        path=preparation_operation[0]/'input/graph.json'
        replacement=path.with_name('replacement.json');replacement.write_bytes(path.read_bytes())
        os.replace(replacement,path)
        with pytest.raises(ValueError):authority.check()

@pytest.mark.parametrize('change',['none','missing-processor','wrong-node','wrong-pin','orphan'])
def test_audio_model_capture_is_exact_and_does_not_need_text_tokenizer(preparation_operation,change):
    f=preparation_operation;root,value=f[:2];folder=root/'input'
    graph=json.loads((folder/'graph.json').read_bytes())
    from runner.audio_classification_tables import DEFAULTS
    graph['nodes'].append(dict(id='audio',operator='kwip.data.classifyAudio',config=dict(DEFAULTS)))
    pins=[dict(path=name,bytes=2,sha256=digest(b'{}')) for name in ['config.json','preprocessor_config.json','model.safetensors'] if not(change=='missing-processor' and name=='preprocessor_config.json')]
    identity=digest(encoded(pins));model=folder/'audio-models'/identity;model.mkdir(parents=True)
    for p in pins:(model/p['path']).write_bytes(b'{}')
    manifest={'other' if change=='wrong-node' else 'audio':dict(identity=identity,files=pins)}
    if change=='wrong-pin':manifest['audio']['identity']='0'*64
    if change=='orphan':graph['nodes'].pop()
    (folder/'audio-models.json').write_bytes(encoded(manifest));(folder/'graph.json').write_bytes(encoded(graph));inventory(f)
    if change=='none':
        with claim(f) as authority:authority.check()
    else:
        with pytest.raises((ValueError,OSError)):claim(f)

def test_decision_model_allowance_is_separate_bounded_and_single(preparation_operation):
    from runner.preparation_operation_contract import parse_contract
    f=preparation_operation;v=f[1];identity='a'*64
    weights=dict(path='input/decision-models/'+identity+'/model.safetensors',bytes=1_685_218_500,sha256='b'*64)
    v['inputFiles'].append(weights)
    parse_contract(v,f[0])
    v['inputFiles'].append(dict(weights,path='input/decision-models/'+'c'*64+'/model.safetensors',bytes=1))
    with pytest.raises(ValueError,match='bounds'):parse_contract(v,f[0])
    v['inputFiles'].pop();weights['bytes']=2_000_000_001
    with pytest.raises(ValueError):parse_contract(v,f[0])


@pytest.mark.parametrize('enabled',['true','false'])
def test_measurements_are_an_explicit_claimed_boolean(preparation_operation,enabled):
    f=preparation_operation;f[1]['arguments']+=['--dataflow-measurements',enabled];publish(f)
    with claim(f) as authority:authority.check()


def test_measurement_flag_refuses_non_boolean_value(preparation_operation):
    f=preparation_operation;f[1]['arguments']+=['--dataflow-measurements','/private/unreviewed'];publish(f)
    with pytest.raises(ValueError):claim(f)


def test_unchanged_cancellation_checks_do_not_reparse_graph(preparation_operation, monkeypatch):
    """A row-level checkpoint must retain authority without decoding the DAG again."""
    from runner import preparation_operation_contract as contract
    calls = []
    original = contract.object_json
    def counted(raw):
        calls.append(len(raw))
        return original(raw)
    monkeypatch.setattr(contract, 'object_json', counted)
    with claim(preparation_operation) as authority:
        after_claim = len(calls)
        for _ in range(20):
            authority.check()
        assert after_claim > 0
        assert len(calls) == after_claim
        # Decoding reuse is never permission to accept changed pinned controls.
        path = preparation_operation[0] / 'input/graph.json'
        before = path.stat()
        raw = path.read_bytes()
        path.write_bytes(b' ' + raw[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(ValueError):
            authority.check()


def test_decoding_reuse_still_rejects_in_memory_request_mutation(preparation_operation):
    with claim(preparation_operation) as authority:
        authority.request['arguments'][1] = '/tmp/unowned-graph.json'
        with pytest.raises(ValueError):
            authority.check()


def test_unchanged_pinned_controls_are_not_reread_for_each_work_checkpoint(preparation_operation,monkeypatch):
    from runner import preparation_operation_contract as contract
    reads=[];read=contract.read_regular
    def counted(*args,**kwargs):
        reads.append(args[1]);return read(*args,**kwargs)
    monkeypatch.setattr(contract,'read_regular',counted)
    with claim(preparation_operation) as authority:
        assert reads
        reads.clear()
        for _ in range(20):authority.check()
        assert reads==[], 'Unchanged verified control files were reserialized at every work checkpoint'
        # Identity replacement and bytes changed under restored mtime remain
        # covered by the warm-authority tests above, independently of decoding.


def test_warm_preparation_reuses_only_immutable_semantic_validation(preparation_operation,monkeypatch):
    from runner import preparation_operation_contract as contract
    calls=[];read=contract._json
    def count(*a,**kw):calls.append(a[1]);return read(*a,**kw)
    monkeypatch.setattr(contract,'_json',count)
    with claim(preparation_operation) as authority:
        assert calls;calls.clear()
        for _ in range(20):authority.check()
        assert calls==[]


def test_semantic_reuse_keeps_destination_directory_checks_live(preparation_operation,tmp_path):
    f=preparation_operation;folder=f[0]/'input';graph=json.loads((folder/'graph.json').read_bytes())
    graph['nodes'].append({'id':'export','operator':'kwip.data.outputDelimited','config':{}})
    (folder/'graph.json').write_bytes(encoded(graph));destination=tmp_path/'export';destination.mkdir()
    (folder/'destinations.json').write_bytes(encoded({'export':str(destination)}))
    f[1]['arguments']+=['--destination-bindings',str(folder/'destinations.json')];inventory(f)
    with claim(f) as authority:
        authority.check();destination.rmdir()
        with pytest.raises((OSError,ValueError)):authority.check()


@pytest.mark.parametrize('selected',['true','false','invalid'])
def test_live_preview_choice_is_separate_and_pinned(preparation_operation,selected):
    f=preparation_operation;f[1]['arguments']+=['--dataflow-live-previews',selected,'--dataflow-capture-outputs','false'];publish(f)
    if selected=='invalid':
        with pytest.raises(ValueError):claim(f)
    else:
        with claim(f) as authority:authority.check()


@pytest.mark.parametrize('selected',[['training'],['unrequired'],['training','training'],[True]])
def test_model_input_format_is_bounded_by_captured_required_selection(preparation_operation,selected):
    f=preparation_operation
    f[1]['arguments']+=['--dataflow-required-nodes','["training"]','--dataflow-model-input-nodes',json.dumps(selected)]
    inventory(f)
    if selected==['training']:
        with claim(f) as authority:authority.check()
    else:
        with pytest.raises(ValueError):
            with claim(f):pass
