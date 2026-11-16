import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from runner.tests.test_prepared_bag_loader import fixture as preparation_fixture, options
from runner.tests.test_bag_source_graph import graph

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def registry(tmp_path):
    """Inject independent reviewed roles even after the production pair is published."""
    from runner import engine
    value=json.loads(engine.REGISTRY.read_text())
    value['components']=[entry for entry in value['components'] if entry['id'] not in ('bag_input','embedding_bag')]
    for entry in [dict(id='bag_input',inputs=[],outputs=['indices','offsets','per_sample_weights'],fields=[]),dict(id='embedding_bag',inputs=['indices','offsets','per_sample_weights'],outputs=['out'],fields=[])]:
        value['components'].append(entry)
    path=tmp_path/'registry.json';path.write_text(json.dumps(value));return path


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def candidate(tmp_path,registry,layout='packed',weighted=True,objective=False):
    prepared=preparation_fixture(tmp_path/'source')
    value=graph(layout,weighted);value['nodes'][0]['config']['maximum_items_per_bag']=2
    if objective:value['nodes'][1]['config']['embedding_dim']=1
    graphfile=tmp_path/'model.json';graphfile.write_text(json.dumps(value))
    settings=options(prepared);mapping=settings['mapping']
    if layout=='fixed':mapping.pop('lengthColumn')
    if not weighted:mapping.pop('weightColumns')
    out=tmp_path/'trained'
    command=[sys.executable,str(ROOT/'runner/main.py'),'--graph',str(graphfile),'--registry',str(registry),
             '--output',str(out),'--epochs','1','--batch-size','3','--device','cpu',
             '--prepared-dataflow',str(prepared/'manifest.json'),
             '--prepared-manifest-sha256',settings['expected_manifest_sha256'],
             '--prepared-graph-digest',settings['expected_graph_digest'],
             '--prepared-training-node','p','--prepared-training-port','training',
             '--prepared-validation-node','p','--prepared-validation-port','validation',
             '--bag-input-mapping-json',canonical(mapping)]
    if objective:
        from runner.loss_configuration import validate_recipe
        recipe=validate_recipe(dict(schemaVersion=1,componentID='loss_mse',configurationVersion=1,config={}))
        targets=dict(schemaVersion=1,componentID='loss_mse',inputColumns=['i1','i2'],roles=[dict(role='target',columns=['desired'],sampleShape=[1],dtype='float32')],tripletSourceGroups=[])
        command += ['--objective-recipe-json',canonical(recipe),'--objective-target-mapping-json',canonical(targets)]
    else:command += ['--target-column','label']
    return command,out,value,graphfile,mapping


def invoke(command):
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    events=[json.loads(line)for line in result.stdout.splitlines()if line.startswith('{')]
    return result,events


@pytest.mark.parametrize('layout,weighted,objective',[('packed',True,False),('fixed',True,False),('packed',False,False),('packed',True,True)])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_cli_prepared_bag_binary_and_objective_training(tmp_path,registry,layout,weighted,objective,device):
    if device=='mps'and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    command,out,_,_,mapping=candidate(tmp_path,registry,layout,weighted,objective)
    command[command.index('--device')+1]=device
    result,events=invoke(command)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert events[-1]['event']=='completed'and(out/'checkpoint.pt').is_file()
    provenance=json.loads((out/'manifest.json').read_bytes())['provenance']
    assert provenance['dataset']=='prepared-dataflow-bags-v1'
    assert provenance['modelInput']['mapping']==mapping and 'sampleShape'not in provenance['modelInput']
    # This shared fixture deliberately uses current three-way Partition: 32
    # source observations, with 16 training / 8 validation / 8 untouched test.
    # Keep an independent exact seed-7 membership oracle, rather than merely
    # accepting whichever counts or IDs the training manifest happens to report.
    expected_indices={
        'training':[0,2,3,4,6,15,17,18,19,22,23,25,26,29,30,31],
        'validation':[1,5,10,13,14,16,24,27],
        'test':[7,8,9,11,12,20,21,28],
    }
    prepared=Path(command[command.index('--prepared-dataflow')+1]).parent
    manifest=json.loads((prepared/'manifest.json').read_bytes())
    def table(node,port):
        artifact=next(entry for entry in manifest['artifacts']
                      if entry.get('nodeID')==node and entry.get('port')==port)
        return json.loads((prepared/artifact['path']).read_bytes())
    source=table('s','table')
    source_ids=source['lineage']['observationIDs']
    assert len(source_ids)==32 and len(set(source_ids))==32
    source_names=[column['name'] for column in source['columns']]
    assert [float(row[source_names.index('desired')]) for row in source['rows']]==[i/8 for i in range(32)]
    selected_ids={}
    for role,indices in expected_indices.items():
        selected=table('p',role)
        names=[column['name'] for column in selected['columns']]
        assert [int(float(row[names.index('desired')])*8) for row in selected['rows']]==indices
        selected_ids[role]=[source_ids[i] for i in indices]
        assert selected['lineage']['observationIDs']==selected_ids[role]
    assert [len(selected_ids[role]) for role in ('training','validation','test')]==[16,8,8]
    assert set().union(*map(set,selected_ids.values()))==set(source_ids)
    for a,b in [('training','validation'),('training','test'),('validation','test')]:
        assert set(selected_ids[a]).isdisjoint(selected_ids[b])
    assert provenance['trainingRowIDs']==selected_ids['training']
    assert provenance['validationRowIDs']==selected_ids['validation']


@pytest.mark.parametrize('extra,fragment',[
    (['--bag-input-mapping-json','{}'],'repeated'),
    (['--feature-columns','i1,i2'],'feature'),
    (['--feature-columns-json','["i1","i2"]'],'feature'),
    (['--dataset-csv','foreign.csv'],'CSV'),
    (['--preview'],'training'),(['--capabilities'],'training'),
])
def test_cli_illegal_transport_combinations_refuse_before_output(tmp_path,registry,extra,fragment):
    command,out,*_=candidate(tmp_path,registry)
    result,events=invoke(command+extra)
    assert result.returncode==1 and events[-1]['event']=='error'
    assert fragment.lower()in events[-1]['message'].lower()
    assert not out.exists()


@pytest.mark.parametrize('bad,fragment',[('{}','version1'),('{"indicesColumns":["i1","i2"],"lengthColumn":"count","schemaVersion":1,"schemaVersion":1,"weightColumns":["w1","w2"]}','duplicate'),
                               (' {"schemaVersion":1}','canonical'), ('x'*32769,'32768')])
def test_cli_strict_mapping_schema_duplicate_keys_canonical_bytes_and_boundaries(tmp_path,registry,bad,fragment):
    command,out,*_=candidate(tmp_path,registry)
    command[command.index('--bag-input-mapping-json')+1]=bad
    result,events=invoke(command)
    assert result.returncode==1 and events[-1]['event']=='error'and not out.exists()
    assert fragment in events[-1]['message']


def test_cli_ordinary_source_and_missing_bag_mapping_are_distinct_refusals(tmp_path,registry):
    command,out,value,graphfile,_=candidate(tmp_path,registry)
    value={'schemaVersion':2,'nodes':[dict(id='source',type='input',config={'features':2}),dict(id='projection',type='linear',config=dict(in_features=2,out_features=2,bias=True)),dict(id='result',type='output',config={})],
           'edges':[dict(id='enter',source='source',sourcePort='out',target='projection',targetPort='in'),dict(id='out',source='projection',sourcePort='out',target='result',targetPort='in')]}
    graphfile.write_text(json.dumps(value))
    result,events=invoke(command)
    assert result.returncode==1 and 'bag'in events[-1]['message'].lower()and not out.exists()
    command,out,value,graphfile,_=candidate(tmp_path/'missing',registry)
    pos=command.index('--bag-input-mapping-json');del command[pos:pos+2]
    result,events=invoke(command)
    assert result.returncode==1 and 'mapping'in events[-1]['message'].lower()and not out.exists()


@pytest.mark.parametrize('flag',['--prepared-manifest-sha256','--prepared-graph-digest'])
def test_cli_foreign_preparation_pins_are_refused_before_publication(tmp_path,registry,flag):
    command,out,*_=candidate(tmp_path,registry)
    command[command.index(flag)+1]='0'*64
    result,events=invoke(command)
    assert result.returncode==1 and 'prepared'in events[-1]['message'].lower()and not out.exists()


@pytest.mark.parametrize('device',['cpu','mps'])
def test_cli_all_consumers_allow_different_valid_vocabulary_and_width(tmp_path,registry,device):
    if device=='mps'and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    from runner import engine
    from runner.bag_input_mapping import graph_admission
    command,out,value,path,_=candidate(tmp_path,registry,weighted=False)
    second=copy.deepcopy(value['nodes'][1]);second['id']='second'
    second['config'].update(num_embeddings=7,embedding_dim=3,mode='max')
    value['nodes'][1]['config']['mode']='mean'
    value['nodes'].insert(2,second)
    value['nodes'].insert(3,dict(id='join',type='concat',config={'dim':-1}))
    value['nodes'].insert(4,dict(id='head',type='linear',config=dict(in_features=5,out_features=2,bias=True)))
    value['edges']=[edge for edge in value['edges']if edge['target']!='result']
    value['edges'] += [dict(id='second-'+edge['id'],source='source',sourcePort=edge['sourcePort'],target='second',targetPort=edge['targetPort'])for edge in list(value['edges'])]
    value['edges'] += [dict(id='firstjoin',source='bag',sourcePort='out',target='join',targetPort='a'),dict(id='secondjoin',source='second',sourcePort='out',target='join',targetPort='b'),dict(id='head',source='join',sourcePort='out',target='head',targetPort='in'),dict(id='result',source='head',sourcePort='out',target='result',targetPort='in')]
    path.write_text(json.dumps(value))
    _,_,layouts,_=engine.validate(value,registry,include_types=True,bundled_dataset=False)
    admission=graph_admission(value,layouts)
    assert admission['num_embeddings']==5 and admission['embedding_dim']==3 and admission['mode']=='sum'
    command[command.index('--device')+1]=device
    result,events=invoke(command)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert events[-1]['event']=='completed'


@pytest.mark.parametrize('mutation',[None,'stale','foreign_pin','rehashed_invalid_origin'])
def test_cli_independent_source_authority_precedes_bag_table_loading(tmp_path,registry,mutation):
    command,out,value,graphfile,_=candidate(tmp_path,registry)
    value['name']='Bags'
    for node in value['nodes']:node.update(name='',x=0,y=0)
    graphfile.write_text(json.dumps(value))
    document=dict(schemaVersion=3,name='Bags',root=dict(configurationVersion=2,nodes=copy.deepcopy(value['nodes']),edges=copy.deepcopy(value['edges']),instances=[]),definitions=[])
    snapshot=dict(nativeExpansionContractVersion=2,sourceCompositeDocument=document,expandedGraph=copy.deepcopy(value),nodeOrigins={node['id']:dict(instancePath=[],localNodeID=node['id'])for node in value['nodes']},consumedDefinitions=[])
    if mutation=='stale':snapshot['sourceCompositeDocument']['root']['nodes'][1]['config']['num_embeddings']=6
    if mutation=='rehashed_invalid_origin':snapshot['nodeOrigins']['bag']['localNodeID']='foreign'
    raw=json.dumps(snapshot).encode();source=tmp_path/'source.json';source.write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest()if mutation!='foreign_pin'else'0'*64
    command += ['--composite-source',str(source),'--composite-source-sha256',sha]
    # A missing table pin is deliberately later than independent source admission.
    if mutation is not None:command[command.index('--prepared-manifest-sha256')+1]='0'*64
    result,events=invoke(command)
    if mutation is None:
        assert result.returncode==0,(result.stdout,result.stderr)
        assert (out/'composite/source.json').read_bytes()==raw
    else:
        assert result.returncode==1 and not out.exists()
        assert any(word in events[-1]['message'].lower()for word in ('source','origin','re-expand','expansion'))


def test_shared_mapping_strict_utf8_and_exact_source_variants():
    from runner.bag_input_mapping import decode_mapping_json,validate_mapping
    config=dict(layout='packed',maximum_items_per_bag=2,per_sample_weights=True,include_last_offset=True)
    mapping=dict(schemaVersion=1,indicesColumns=['élève','第二'],weightColumns=['importance1','importance2'],lengthColumn='count')
    assert decode_mapping_json(canonical(mapping),config)==mapping
    assert decode_mapping_json(canonical(mapping),config,target_column='target')==mapping
    for key in mapping:
        invalid=dict(mapping);invalid.pop(key)
        with pytest.raises(ValueError):decode_mapping_json(canonical(invalid),config)
    with pytest.raises(ValueError):decode_mapping_json(json.dumps(mapping,sort_keys=True,separators=(',',':')),config)
    with pytest.raises(ValueError):validate_mapping(mapping,config,target_column='count')
    with pytest.raises(ValueError):validate_mapping(mapping,{**config,'maximum_items_per_bag':1})
    with pytest.raises(ValueError):validate_mapping(mapping,{**config,'layout':'fixed'})
    with pytest.raises(ValueError):validate_mapping(mapping,{**config,'per_sample_weights':False})
