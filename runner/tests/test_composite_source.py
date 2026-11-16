import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner.tests.test_runner import graph as make_graph

ROOT=Path(__file__).resolve().parents[2]

def fixture(tmp_path):
    graph=make_graph()
    snapshot={'nativeExpansionContractVersion':1,'sourceCompositeDocument':{'schemaVersion':1,'name':'Source','root':{'nodes':graph['nodes'],'edges':graph['edges'],'instances':[]},'definitions':[]},'expandedGraph':graph,'nodeOrigins':{n['id']:{'instancePath':[],'localNodeID':n['id']} for n in graph['nodes']},'consumedDefinitions':[]}
    path=tmp_path/'source.json';raw=json.dumps(snapshot).encode();path.write_bytes(raw)
    return graph,path,hashlib.sha256(raw).hexdigest(),raw

def test_valid_source_is_archived_and_does_not_change_execution(tmp_path):
    from runner.composite_source import load_source
    from runner.engine import run
    import torch
    graph,path,sha,raw=fixture(tmp_path)
    source=load_source(path,sha,graph)
    run(graph,tmp_path/'plain',epochs=2,device='cpu')
    run(graph,tmp_path/'composite',epochs=2,device='cpu',composite_source=source)
    assert (tmp_path/'composite/composite/source.json').read_bytes()==raw
    manifest=json.loads((tmp_path/'composite/manifest.json').read_text())
    assert manifest['provenance']['composite']['sha256']==sha
    a=torch.load(tmp_path/'plain/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'composite/checkpoint.pt',weights_only=True)
    assert all(torch.equal(a['modelState'][k],b['modelState'][k]) for k in a['modelState'])

def test_stale_graph_and_corrupt_source_rejected_before_run(tmp_path):
    from runner.composite_source import load_source
    graph,path,sha,_=fixture(tmp_path)
    stale=copy.deepcopy(graph);stale['nodes'][1]['config']['out_features']=8
    with pytest.raises(ValueError,match='architecture'):load_source(path,sha,stale)
    path.write_text('{}')
    with pytest.raises(ValueError,match='hash|digest'):load_source(path,sha,graph)

def test_cosmetic_edits_and_execution_refs_do_not_stale_source(tmp_path):
    from runner.composite_source import load_source
    graph,path,sha,_=fixture(tmp_path)
    graph['name']='New label';graph['nodes'][1].update(name='Mine',x=999,y=123);graph['execution']={'compositeSource':{'path':'private'}}
    assert load_source(path,sha,graph)

def test_source_shape_origin_coverage_and_nonregular_paths_rejected(tmp_path):
    from runner.composite_source import load_source
    graph,path,_,raw=fixture(tmp_path)
    snapshot=json.loads(raw);snapshot['nodeOrigins'].pop('left');bad=json.dumps(snapshot).encode();path.write_bytes(bad)
    with pytest.raises(ValueError,match='origin'):load_source(path,hashlib.sha256(bad).hexdigest(),graph)
    target=tmp_path/'link';target.symlink_to(path)
    with pytest.raises(ValueError,match='regular'):load_source(target,hashlib.sha256(bad).hexdigest(),graph)

def test_cli_pair_required_and_valid_archive(tmp_path):
    graph,path,sha,_=fixture(tmp_path);request=tmp_path/'graph.json';request.write_text(json.dumps(graph))
    args=[sys.executable,str(ROOT/'runner/main.py'),'--graph',str(request),'--output',str(tmp_path/'run'),'--epochs','1','--device','cpu','--composite-source',str(path)]
    missing=subprocess.run(args,capture_output=True,text=True);assert missing.returncode==1
    result=subprocess.run(args+['--composite-source-sha256',sha],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'run/composite/source.json').exists()

def test_expanded_schema_boolean_is_not_a_valid_version(tmp_path):
    from runner.composite_source import load_source
    graph,path,_,raw=fixture(tmp_path)
    source=json.loads(raw);source['expandedGraph']['schemaVersion']=True
    raw=json.dumps(source).encode();path.write_bytes(raw)
    with pytest.raises(ValueError,match='architecture'):load_source(path,hashlib.sha256(raw).hexdigest(),graph)

def test_schema2_source_with_legacy_pinned_body_archives_exact_bytes(tmp_path):
    from runner.composite_source import load_source
    from runner.engine import run
    g,path,_,raw=fixture(tmp_path);source=json.loads(raw)
    source['sourceCompositeDocument']['schemaVersion']=2
    source['sourceCompositeDocument']['root']['configurationVersion']=2
    for n in source['sourceCompositeDocument']['root']['nodes']:
        if n['type']=='linear':n['config']['bias']=True
    source['expandedGraph']=copy.deepcopy(source['sourceCompositeDocument']['root']);source['expandedGraph']['schemaVersion']=2
    source['sourceCompositeDocument']['definitions']=[{'reference':{'id':'legacy','version':{'major':1,'minor':0,'patch':0}},'title':'Legacy','body':{'nodes':g['nodes'],'edges':g['edges'],'instances':[]},'inputs':[],'outputs':[],'parameters':[]}]
    raw=json.dumps(source).encode();path.write_bytes(raw)
    loaded=load_source(path,hashlib.sha256(raw).hexdigest(),source['expandedGraph'])
    run(source['expandedGraph'],tmp_path/'v2-run',epochs=1,device='cpu',composite_source=loaded)
    assert (tmp_path/'v2-run/composite/source.json').read_bytes()==raw

@pytest.mark.parametrize('version,bias',[(1,True),(2,1),(2,None),(True,True),(3,True)])
def test_source_body_config_version_is_strict(tmp_path,version,bias):
    from runner.composite_source import load_source
    g,path,_,raw=fixture(tmp_path);source=json.loads(raw)
    source['sourceCompositeDocument']['root']['configurationVersion']=version
    source['sourceCompositeDocument']['root']['nodes'][1]['config']['bias']=bias
    raw=json.dumps(source).encode();path.write_bytes(raw)
    with pytest.raises(ValueError):load_source(path,hashlib.sha256(raw).hexdigest(),g)

@pytest.mark.parametrize('value',[False,True,0,1,None,'false'])
def test_composite_boolean_override_requires_actual_boolean(tmp_path,value):
    from runner.composite_source import load_source
    g,path,_,raw=fixture(tmp_path);source=json.loads(raw)
    ref={'id':'legacy','version':{'major':1,'minor':0,'patch':0}}
    source['sourceCompositeDocument']['schemaVersion']=2
    source['sourceCompositeDocument']['definitions']=[{'reference':ref,'body':{'nodes':[],'instances':[],'edges':[]},'parameters':[{'field':{'key':'useBias','kind':'boolean','title':'Bias','defaultValue':True},'targets':[]}]}]
    source['sourceCompositeDocument']['root']['instances']=[{'id':'instance','definition':ref,'overrides':{'useBias':value}}]
    raw=json.dumps(source).encode();path.write_bytes(raw)
    if type(value) is bool:assert load_source(path,hashlib.sha256(raw).hexdigest(),g)
    else:
        with pytest.raises(ValueError,match='boolean'):load_source(path,hashlib.sha256(raw).hexdigest(),g)

def test_invalid_boolean_default_cannot_be_hidden_by_override(tmp_path):
    from runner.composite_source import load_source
    g,path,_,raw=fixture(tmp_path);source=json.loads(raw)
    ref={'id':'legacy','version':{'major':1,'minor':0,'patch':0}}
    source['sourceCompositeDocument']['definitions']=[{'reference':ref,'body':{'nodes':[],'instances':[],'edges':[]},'parameters':[{'field':{'key':'useBias','kind':'boolean','title':'Bias','defaultValue':123},'targets':[]}]}]
    source['sourceCompositeDocument']['root']['instances']=[{'id':'instance','definition':ref,'overrides':{'useBias':True}}]
    raw=json.dumps(source).encode();path.write_bytes(raw)
    with pytest.raises(ValueError,match='declaration'):load_source(path,hashlib.sha256(raw).hexdigest(),g)
