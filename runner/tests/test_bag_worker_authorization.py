"""Managed bag admission, distinct retained workflow and standalone paths."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_worker_authorization import launch_fixture,claim as claim_launch,uid,encoded,digest
from runner.tests.test_bag_cli import candidate,registry,canonical,invoke


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path.read_bytes()


def phase_candidate(f,tmp_path,registry,objective=False):
    command,_,graph,_,mapping=candidate(tmp_path/'candidate',registry,objective=objective)
    root=Path(f[1]['arguments'][1]).parent.parent
    phase=root/'training';phase.mkdir()
    write(phase/'request.json',graph)
    shutil.copytree(tmp_path/'candidate/source/prepared',root/'preparation/artifacts',dirs_exist_ok=True)
    args=command[2:]
    for flag,path in [('--graph',phase/'request.json'),('--output',phase/'artifacts'),('--prepared-dataflow',root/'preparation/artifacts/manifest.json')]:args[args.index(flag)+1]=str(path)
    registry_value=json.loads(registry.read_bytes())
    registry_raw=write(root/'model-registry.json',registry_value)
    evidence=write(root/'runtime-evidence.json',dict(schemaVersion=1,workerFiles={'components.json':digest(registry.read_bytes())}))
    pin=lambda name,raw:dict(path=name,bytes=len(raw),sha256=digest(raw))
    capture=dict(schemaVersion=12,compilerContractVersion=11,captureFormatVersion=7,
        projectID=f[1]['projectID'],directoryID=f[1]['directoryID'],runID=f[1]['runID'],requestIdentity=f[1]['requestIdentity'],
        modelRegistry=pin('model-registry.json',registry_raw),runtimeEvidence=pin('runtime-evidence.json',evidence),
        plan=dict(modelGraph=graph,bagInput=mapping,targetColumn='label',training=dict(node='p',port='training'),validation=dict(node='p',port='validation')))
    write(root/'capture.json',capture)
    if objective:
        capture['plan'].pop('targetColumn')
        capture['plan']['lossRecipe']=dict(configuration=json.loads(args[args.index('--objective-recipe-json')+1]),targetMapping=json.loads(args[args.index('--objective-target-mapping-json')+1]))
        capture['plan']['objectiveScalarization']='mean'
        write(root/'capture.json',capture)
    f[1].update(phase='training',arguments=args)
    return root,phase,graph,mapping,capture


@pytest.mark.parametrize('header',[(12,11,7),(13,12,8),(14,12,8),(15,13,8),(16,14,8),(17,15,8),(18,15,8),(13,16,8),(14,16,8),(15,16,8),(16,16,8),(17,16,8),(18,16,8),(19,16,8)])
@pytest.mark.parametrize('kind',['operation','launch'])
def test_actual_phase_authority_accepts_bag_mapping_and_rechecks_pins(operation_authority,launch_fixture,tmp_path,registry,kind,header):
    f=operation_authority if kind=='operation'else launch_fixture
    prior=copy.deepcopy(f[1]);root,phase,_,_,capture=phase_candidate(f,tmp_path,registry)
    capture.update(zip(('schemaVersion','compilerContractVersion','captureFormatVersion'),header))
    write(root/'capture.json',capture)
    if kind=='operation':publish(f)
    else:
        f[1]['nonce']=uid();raw=encoded(f[1]);(f[0]/f"launch-{f[1]['nonce']}.json").write_bytes(raw);(f[0]/f"phase-{f[1]['runID']}-training.json").write_bytes(raw)
        (f[0]/f"claim-{prior['nonce']}.json").write_bytes(encoded(dict(schemaVersion=1,nonce=prior['nonce'],intentSHA256=digest(encoded(prior)),state='claimed')))
    with (claim(f)if kind=='operation'else claim_launch(f))as authority:
        authority.check()
        registry.write_text('{}')
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('mutation',['mapping','missing_mapping','roles','ordinary','capture_format','future_schema','captured_graph','registry_pin','manifest_pin','source_control','target_overlap'])
def test_phase_invalid_bag_inputs_refuse_before_claim(operation_authority,tmp_path,registry,mutation):
    f=operation_authority;root,phase,graph,mapping,capture=phase_candidate(f,tmp_path,registry)
    if mutation=='mapping':f[1]['arguments'][f[1]['arguments'].index('--bag-input-mapping-json')+1]='{}'
    elif mutation=='missing_mapping':
        index=f[1]['arguments'].index('--bag-input-mapping-json');del f[1]['arguments'][index:index+2]
    elif mutation=='roles':graph['edges'][0]['sourcePort']='offsets';capture['plan']['modelGraph']=graph;write(phase/'request.json',graph)
    elif mutation=='ordinary':graph['nodes'][0]['type']='input';capture['plan']['modelGraph']=graph;write(phase/'request.json',graph)
    elif mutation=='capture_format':capture['captureFormatVersion']=6
    elif mutation=='future_schema':capture.update(schemaVersion=19,compilerContractVersion=15,captureFormatVersion=8)
    elif mutation=='captured_graph':capture['plan']['modelGraph']['nodes'][1]['config']['embedding_dim']=4
    elif mutation=='registry_pin':capture['modelRegistry']['sha256']='0'*64
    elif mutation=='manifest_pin':f[1]['arguments'][f[1]['arguments'].index('--prepared-manifest-sha256')+1]='0'*64
    elif mutation=='source_control':graph['nodes'][0]['config']['maximum_items_per_bag']=3;capture['plan']['modelGraph']=graph;write(phase/'request.json',graph)
    else:mapping['lengthColumn']='label';capture['plan']['bagInput']=mapping;f[1]['arguments'][f[1]['arguments'].index('--bag-input-mapping-json')+1]=canonical(mapping)
    write(root/'capture.json',capture);publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not(f[0]/'claim.json').exists()and not(phase/'artifacts').exists()


@pytest.mark.parametrize('mutation',[None,'duplicate','whitespace','schema','nonprepared','legacy_columns','oversize'])
def test_phase_argument_transport_is_strict_without_torch(mutation):
    raw='{"indicesColumns":["item"],"schemaVersion":1}'
    args=['--bag-input-mapping-json',raw,'--prepared-dataflow','/prepared/manifest.json','--target-column','target',
        '--prepared-manifest-sha256','a'*64,'--prepared-graph-digest','b'*64,'--prepared-training-node','p','--prepared-training-port','training','--prepared-validation-node','p','--prepared-validation-port','validation']
    if mutation=='duplicate':args+=args[:2]
    elif mutation=='whitespace':args[1]=' '+raw
    elif mutation=='schema':args[1]='{}'
    elif mutation=='nonprepared':del args[2:4]
    elif mutation=='legacy_columns':args+=['--feature-columns-json','["item"]']
    elif mutation=='oversize':args[1]='x'*32769
    script='import sys; from runner.worker_authorization import validate_phase_arguments; validate_phase_arguments("training",'+repr(args)+'); assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert(result.returncode==0)==(mutation is None),result.stderr


@pytest.mark.parametrize('mutation',[None,'recipe','scalarization','target_mapping'])
def test_objective_phase_binds_existing_recipe_and_targets(operation_authority,tmp_path,registry,mutation):
    f=operation_authority;root,phase,_,_,capture=phase_candidate(f,tmp_path,registry,objective=True)
    if mutation=='recipe':capture['plan']['lossRecipe']['configuration']['componentID']='loss_l1'
    elif mutation=='scalarization':capture['plan']['objectiveScalarization']='sum'
    elif mutation=='target_mapping':capture['plan']['lossRecipe']['targetMapping']['roles'][0]['columns']=['label']
    write(root/'capture.json',capture);publish(f)
    if mutation is None:
        with claim(f)as authority:authority.check()
    else:
        with pytest.raises(ValueError):claim(f)
        assert not(f[0]/'claim.json').exists()


def standalone_candidate(f,tmp_path,registry):
    command,_,graph,_,_=candidate(tmp_path/'standalone',registry)
    inputs=f[0]/'input';inputs.mkdir()
    write(inputs/'graph.json',graph)
    shutil.copytree(tmp_path/'standalone/source/prepared',inputs/'preparation')
    shutil.copyfile(registry,inputs/'model-registry.json')
    args=command[2:];index=args.index('--registry');del args[index:index+2]
    args+=['--seed','42','--learning-rate','0.03']
    for flag,path in [('--graph',inputs/'graph.json'),('--output',f[0]/'artifacts'),('--prepared-dataflow',inputs/'preparation/manifest.json')]:args[args.index(flag)+1]=str(path)
    for key in ('runID','directoryID','phase'):f[1].pop(key)
    f[1].update(schemaVersion=2,toolMode='modelTraining',arguments=args,
        inputFiles=[dict(path=str(path.relative_to(f[0])),bytes=path.stat().st_size,sha256=digest(path.read_bytes()))for path in sorted(inputs.rglob('*'))if path.is_file()])
    publish(f);return f


@pytest.mark.parametrize('mutation',[None,'missing_registry','registry_bytes','wrong_roles','mapping_capacity','foreign_pin'])
def test_actual_standalone_authority_bag_inputs(operation_authority,tmp_path,registry,mutation):
    f=standalone_candidate(operation_authority,tmp_path,registry)
    if mutation=='missing_registry':f[1]['inputFiles']=[entry for entry in f[1]['inputFiles']if entry['path']!='input/model-registry.json']
    elif mutation=='registry_bytes':(f[0]/'input/model-registry.json').write_text('{}')
    elif mutation in ('wrong_roles','mapping_capacity'):
        path=f[0]/'input/graph.json';graph=json.loads(path.read_bytes())
        if mutation=='wrong_roles':graph['edges'][0]['sourcePort']='offsets'
        else:graph['nodes'][0]['config']['maximum_items_per_bag']=3
        raw=write(path,graph);entry=next(entry for entry in f[1]['inputFiles']if entry['path']=='input/graph.json');entry.update(bytes=len(raw),sha256=digest(raw))
    elif mutation=='foreign_pin':f[1]['arguments'][f[1]['arguments'].index('--prepared-manifest-sha256')+1]='0'*64
    publish(f)
    if mutation is None:
        from runner.worker_bootstrap import _arguments,RuntimeContext
        with claim(f)as authority:
            authority.check()
            routed=_arguments(f[1],RuntimeContext(Path(__file__),Path('/not-opened'),registry))
            assert routed[-2:]==['--registry',str(registry)]
            (f[0]/'input/model-registry.json').write_text('{}')
            with pytest.raises(ValueError):authority.check()
    else:
        with pytest.raises((ValueError,OSError)):claim(f)
        assert not(f[0]/'claim.json').exists()


@pytest.mark.parametrize('kind',['foreign','missing','boolean_number','equal_formatting','equal_number_format'])
def test_bootstrap_captured_registry_parity_before_runtime_injection(operation_authority,tmp_path,registry,kind):
    from runner.worker_bootstrap import _arguments,RuntimeContext
    f=standalone_candidate(operation_authority,tmp_path,registry)
    selected=tmp_path/'selected.json';selected.write_text('{}'if kind=='foreign'else json.dumps(json.loads(registry.read_bytes()),indent=4))
    if kind in ('boolean_number','equal_number_format'):
        value=json.loads(selected.read_bytes())
        field=next(field for component in value['components']for field in component['fields']if field.get('defaultValue')==2 and type(field['defaultValue'])is int)
        field['defaultValue']=True if kind=='boolean_number' else 2.0
        selected.write_text(json.dumps(value))
    with claim(f):
        if kind in ('equal_formatting','equal_number_format'):assert _arguments(f[1],RuntimeContext(Path(__file__),Path('/not-opened'),selected))[-1]==str(selected)
        else:
            with pytest.raises(ValueError):_arguments(f[1],RuntimeContext(Path(__file__),Path('/not-opened'),None if kind=='missing'else selected))
    assert not(f[0]/'artifacts').exists()


def test_rehashed_forged_source_body_is_refused_after_guard_before_table_load_or_forward(operation_authority,tmp_path,registry,monkeypatch,capsys):
    f=operation_authority;root,phase,graph,_,capture=phase_candidate(f,tmp_path,registry)
    graph['name']='Bags'
    for node in graph['nodes']:node.update(name='',x=0,y=0)
    snapshot=dict(nativeExpansionContractVersion=2,sourceCompositeDocument=dict(schemaVersion=3,name='Bags',root=dict(configurationVersion=2,nodes=copy.deepcopy(graph['nodes']),edges=copy.deepcopy(graph['edges']),instances=[]),definitions=[]),expandedGraph=graph,nodeOrigins={node['id']:dict(instancePath=[],localNodeID=node['id'])for node in graph['nodes']},consumedDefinitions=[])
    snapshot['sourceCompositeDocument']['root']['nodes'][1]['config']['num_embeddings']=6
    raw=write(phase/'composite-source.json',snapshot);write(phase/'request.json',graph)
    capture['plan'].update(modelGraph=graph,modelProvenance=snapshot);write(root/'capture.json',capture)
    f[1]['arguments']+=['--composite-source',str(phase/'composite-source.json'),'--composite-source-sha256',digest(raw)];publish(f)
    # Preclaim establishes pinned identity/roles, deliberately not source re-expansion.
    with claim(f)as authority:
        authority.check()
        handlers={number:signal.getsignal(number)for number in (signal.SIGTERM,signal.SIGINT)}
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        try:
            import main
            import prepared_bag_loader
            import engine
            monkeypatch.setattr(prepared_bag_loader,'load_prepared_bags',lambda *a,**k:pytest.fail('Table load preceded independent source admission'))
            monkeypatch.setattr(engine,'run',lambda *a,**k:pytest.fail('Training preceded independent source admission'))
            monkeypatch.setattr(sys,'argv',['main.py',*f[1]['arguments']])
            assert main.main(check_cancelled=authority.check)==1
            assert 'independent source expansion'in capsys.readouterr().out
        finally:
            sys.path.pop(0)
            for number,handler in handlers.items():signal.signal(number,handler)
    assert not(phase/'artifacts').exists()


@pytest.mark.parametrize('layout',['fixed','packed'])
@pytest.mark.parametrize('weighted',[False,True])
def test_preclaim_mapping_parity_with_postguard_contract(layout,weighted,tmp_path,registry):
    from runner.worker_input_contract import validate_bag_arguments
    from runner.bag_input_mapping import decode_mapping_json
    command,_,graph,_,mapping=candidate(tmp_path,registry,layout,weighted)
    args=command[2:];options={args[index]:args[index+1]for index in range(0,len(args),2)}
    config=graph['nodes'][0]['config']
    assert validate_bag_arguments(options,config)==decode_mapping_json(options['--bag-input-mapping-json'],config,target_column='label')
    for invalid in ({**mapping,'unexpected':1},{**mapping,'schemaVersion':True},{**mapping,'indicesColumns':['i1','i1']}):
        raw=canonical(invalid)
        with pytest.raises(ValueError):validate_bag_arguments({**options,'--bag-input-mapping-json':raw},config)
        with pytest.raises(ValueError):decode_mapping_json(raw,config,target_column='label')


@pytest.mark.parametrize('header',[
    (12,12,7),(13,11,8),(14,13,8),(15,12,8),(16,13,8),(16,15,8),(17,14,8),(18,14,8),(19,15,8),
    (12,11,8),(13,12,7),(14,12,7),(15,13,7),
    (13.0,12,8),(13,12.0,8),(13,12,8.0),
    (True,12,8),(13,True,8),(13,12,True),
    ("13",12,8),(13,None,8),(13,12,"8"),
])
def test_bag_capture_header_rejects_crossed_unknown_or_noninteger_contracts(operation_authority,tmp_path,registry,header):
    f=operation_authority
    root,phase,_,_,capture=phase_candidate(f,tmp_path,registry)
    capture.update(zip(('schemaVersion','compilerContractVersion','captureFormatVersion'),header))
    write(root/'capture.json',capture);publish(f)
    with pytest.raises(ValueError,match='Bag workflow requires'):
        claim(f)
    assert not(f[0]/'claim.json').exists() and not(phase/'artifacts').exists()
