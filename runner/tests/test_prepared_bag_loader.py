import hashlib
import json

import pytest

from runner import dataflow as df
from runner.preprocess_parameters import rule_defaults
from runner.dataflow_cli import run_dataflow
from runner.prepared_bag_loader import load_prepared_bags


def fixture(tmp_path, fitted=False,segmented=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv = tmp_path / 'bags.csv'
    csv.write_text('i1,i2,w1,w2,count,label,desired\n'+''.join(
        f'{i%5},{(i+1)%5},{i+.5},{i+1.5},{i%3},{i%2},{i/8}\n' for i in range(32)))
    graph = dict(schemaVersion=1, nodes=[
        dict(id='s',operator='kwip.data.sourceCSV',config={'separator':','}),
        dict(id='c',operator='kwip.data.columns',config={'rows':[dict(source=c,included=True,outputName=c,targetType=t) for c,t in [('i1','int64'),('i2','int64'),('w1','float64'),('w2','float64'),('count','int64'),('label','int64'),('desired','float64')]],'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}),
        dict(id='p',operator='kwip.data.partition',config={'strategy':'random','seed':7,'validationPercent':25,'testPercent':25,'stratifyColumn':'label'})],
        edges=[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},
               {'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}])
    if fitted:
        graph['nodes'] += [dict(id='fit',operator='kwip.data.preprocess',config={'mode':'partitions','columns':[dict(rule_defaults('w1'),missing='error',scaling='standard')]}),
                           dict(id='training',operator='kwip.data.view',config={}),
                           dict(id='heldout',operator='kwip.data.view',config={})]
        for a,ap,b,bp in [('p','training','fit','training'),('p','validation','fit','validation'),
                          ('fit','training','training','table'),('fit','validation','heldout','table')]:
            graph['edges'].append({'from':{'node':a,'port':ap},'to':{'node':b,'port':bp}})
    graphfile=tmp_path/'flow.json';graphfile.write_text(json.dumps(graph))
    output=tmp_path/'prepared'
    from runner.workload_policy import LEGACY
    run_dataflow(graphfile, {'s':str(csv)}, output, lambda _:None,workload_policy=LEGACY.to_document() if segmented else None)
    return output


def manifest(path): return json.loads((path/'manifest.json').read_bytes())


def options(path, fitted=False):
    value=manifest(path)
    return dict(source_config={'layout':'packed','maximum_items_per_bag':2,'per_sample_weights':True,'include_last_offset':True},
                mapping={'schemaVersion':1,'indicesColumns':['i1','i2'],'weightColumns':['w1','w2'],'lengthColumn':'count'},
                num_embeddings=5,embedding_dim=2,mode='sum',target_column='label',
                expected_manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),
                expected_graph_digest=value['graphDigest'])


def load(path, fitted=False, **changes):
    args=options(path);args.update(changes)
    return load_prepared_bags(path/'manifest.json',*(('training','table','heldout','table') if fitted else ('p','training','p','validation')),**args)


def update_artifact(path, value, entry, artifact):
    raw=json.dumps(artifact).encode();(path/entry['path']).write_bytes(raw)
    entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def reseal_table(table):
    l=table['lineage'];l['membershipDigest']=df.membership_digest(l['observationIDs'])
    l['dataVersion']=df.digest({'columns':table['columns'],'rows':table['rows'],'ids':l['observationIDs']})


def selected_table(path, value, role):
    entry=next(e for e in value['artifacts'] if e.get('nodeID')=='p' and e.get('port')==role)
    return entry,json.loads((path/entry['path']).read_bytes())


@pytest.mark.parametrize('fitted',[False,True])
def test_real_completed_preparation_empty_bags_targets_order_and_exact_provenance(tmp_path,fitted):
    path=fixture(tmp_path,fitted)
    data, provenance, files=load(path,fitted)
    train,targets,val,held=data
    assert len(targets)==16 and len(held)==8
    assert provenance['dataset']=='prepared-dataflow-bags-v1'
    assert 'sampleShape' not in provenance['modelInput']
    assert provenance['modelInput']['kind']=='embeddingBagPacket'
    assert provenance['preparation']['manifestSHA256']==options(path)['expected_manifest_sha256']
    assert bool(provenance['preparation']['fittedStateDigests']) is fitted
    assert set(provenance['trainingRowIDs']).isdisjoint(provenance['validationRowIDs'])
    assert files['manifest.json']==(path/'manifest.json').read_bytes()
    for packet,labels,role in [(train,targets,'training'),(val,held,'validation')]:
        identity=provenance['bagSources'][role]
        entry=next(e for e in manifest(path)['artifacts'] if e.get('nodeID')==('training' if fitted and role=='training' else 'heldout' if fitted else 'p') and e.get('port')==('table' if fitted else role))
        table=json.loads(files[entry['path']]);names=[c['name']for c in table['columns']]
        expected_ids=[];starts=[0];importance=[]
        for row in table['rows']:
            count=row[names.index('count')]
            expected_ids += [row[names.index(n)]for n in ['i1','i2']][:count]
            importance += [row[names.index(n)]for n in ['w1','w2']][:count]
            starts.append(len(expected_ids))
        assert packet.indices.tolist()==expected_ids and packet.offsets.tolist()==starts
        assert packet.per_sample_weights.tolist()==pytest.approx(importance)
        assert labels.tolist()==[row[names.index('label')]for row in table['rows']]
        assert identity['observationIDs']==table['lineage']['observationIDs']
        assert identity['tableDigest']==df.digest(table)
        assert any(a==b for a,b in zip(starts,starts[1:]))


@pytest.mark.parametrize('key,bad',[('expected_manifest_sha256','0'*64),('expected_graph_digest','0'*64),
                                  ('expected_manifest_sha256',None),('expected_graph_digest','A'*64)])
def test_required_exact_pins_refuse_before_tensor_allocation(tmp_path,monkeypatch,key,bad):
    path=fixture(tmp_path)
    monkeypatch.setattr('torch.tensor',lambda *a,**k:pytest.fail('Pin refusal reached allocation'))
    with pytest.raises(ValueError):load(path,**{key:bad})


@pytest.mark.parametrize('status',['failed','cancelled','running'])
def test_partial_preparation_is_never_accepted_even_when_rehashed(tmp_path,status):
    path=fixture(tmp_path);value=manifest(path);value['status']=status
    (path/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):load(path)


@pytest.mark.parametrize('change',['schema','source','partition','overlap','fit_chain'])
def test_rehashed_paired_partition_tampering_cannot_bypass_lineage(tmp_path,change):
    path=fixture(tmp_path);value=manifest(path)
    te,train=selected_table(path,value,'training');ve,val=selected_table(path,value,'validation')
    if change=='schema': val['columns'][-1]['name']='other'
    elif change=='source': val['lineage']['sourceDigest']='d'*64
    elif change=='partition': val['lineage']['partitionDigest']='d'*64
    elif change=='overlap': val['lineage']['observationIDs'][0]=train['lineage']['observationIDs'][0]
    else: val['lineage']['fittedStateDigests']=['d'*64]
    reseal_table(val);update_artifact(path,value,ve,val)
    (path/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):load(path)


def test_rehashed_invalid_source_graph_is_structurally_rejected(tmp_path):
    path=fixture(tmp_path);value=manifest(path)
    entry=next(e for e in value['artifacts']if e['path']=='dataflow.json')
    graph=json.loads((path/'dataflow.json').read_bytes());graph['nodes'][0]['operator']='kwip.data.not-real'
    update_artifact(path,value,entry,graph);value['graphDigest']=df.digest(graph)
    (path/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):load(path)


def test_selected_artifact_must_name_a_declared_source_graph_output(tmp_path):
    path=fixture(tmp_path);value=manifest(path)
    for entry in value['artifacts']:
        if entry.get('nodeID')=='p' and entry.get('port')=='training':entry['nodeID']='invented'
    (path/'manifest.json').write_text(json.dumps(value))
    args=options(path)
    with pytest.raises(ValueError):load_prepared_bags(path/'manifest.json','invented','training','p','validation',**args)


def test_actual_fitted_state_with_heldout_fit_member_is_rejected_after_rehash(tmp_path):
    path=fixture(tmp_path,True);value=manifest(path)
    state_entry=next(e for e in value['artifacts']if e.get('kind')=='state')
    state=json.loads((path/state_entry['path']).read_bytes())
    held_entry=next(e for e in value['artifacts']if e.get('nodeID')=='heldout')
    held=json.loads((path/held_entry['path']).read_bytes())
    old=state['contentDigest'];state['fitEvidence']['observationIDs'].append(held['lineage']['observationIDs'][0])
    state['fitEvidence']['membershipDigest']=df.membership_digest(state['fitEvidence']['observationIDs'])
    state['contentDigest']=df.digest({k:v for k,v in state.items()if k!='contentDigest'})
    update_artifact(path,value,state_entry,state)
    for entry in value['artifacts']:
        if entry.get('kind')!='table':continue
        table=json.loads((path/entry['path']).read_bytes())
        if old in table['lineage']['fittedStateDigests']:
            table['lineage']['fittedStateDigests']=[state['contentDigest']if x==old else x for x in table['lineage']['fittedStateDigests']]
            update_artifact(path,value,entry,table)
    (path/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='nontraining|training observations'):load(path,True)


@pytest.mark.parametrize('layout,terminal',[('fixed',False),('fixed',True),('packed',False)])
def test_actual_source_variants_preserve_observation_count_and_explicit_packet_identity(tmp_path,layout,terminal):
    path=fixture(tmp_path);args=options(path)
    args['source_config'].update(layout=layout,include_last_offset=terminal,per_sample_weights=False)
    args['mapping'].pop('weightColumns')
    if layout=='fixed':args['mapping'].pop('lengthColumn')
    data,provenance,_=load(path,**args)
    for packet,targets in [(data[0],data[1]),(data[2],data[3])]:
        assert packet.layout==layout and packet.include_last_offset is terminal
        assert packet.per_sample_weights is None
        if layout=='fixed':assert packet.indices.shape==(len(targets),2) and packet.offsets is None
        else:assert len(packet.offsets)==len(targets)
    assert provenance['modelInput']['layout']==layout
    assert provenance['modelInput']['weightDType'] is None
    assert 'sampleShape'not in provenance['modelInput']


def test_real_objective_preparation_keeps_continuous_targets_and_captured_mapping(tmp_path):
    from runner.loss_configuration import validate_recipe
    path=fixture(tmp_path)
    recipe=validate_recipe({'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':{}})
    mapping={'schemaVersion':1,'componentID':'loss_mse','inputColumns':['i1','i2'],
             'roles':[{'role':'target','columns':['desired'],'sampleShape':[1],'dtype':'float32'}],
             'tripletSourceGroups':[]}
    data,provenance,files=load(path,target_column=None,objective_mapping=mapping,objective_recipe=recipe,prediction_shape=[1])
    assert data[1]['target'].shape==(16,1) and data[3]['target'].shape==(8,1)
    assert provenance['objectiveMapping']==mapping and provenance['objectiveRecipe']==recipe
    assert 'targetColumn'not in provenance
    for targets,role in [(data[1],'training'),(data[3],'validation')]:
        entry=next(e for e in manifest(path)['artifacts']if e.get('nodeID')=='p'and e.get('port')==role)
        table=json.loads(files[entry['path']]);column=next(i for i,c in enumerate(table['columns'])if c['name']=='desired')
        assert targets['target'].flatten().tolist()==[row[column]for row in table['rows']]


def test_manifest_and_artifact_regular_file_rules_cannot_be_bypassed(tmp_path):
    path=fixture(tmp_path);args=options(path)
    (path/'manifest.json').rename(path/'saved-manifest.json')
    (path/'manifest.json').symlink_to(path/'saved-manifest.json')
    with pytest.raises(ValueError):load(path,**args)
    other=fixture(tmp_path/'other');args=options(other);value=manifest(other)
    entry=next(e for e in value['artifacts']if e.get('kind')=='table')
    artifact=other/entry['path'];saved=other/('saved-'+entry['path'])
    artifact.rename(saved);artifact.symlink_to(saved)
    with pytest.raises(ValueError):load(other,**args)


def test_artifact_bytes_cannot_change_under_a_pinned_manifest(tmp_path):
    path=fixture(tmp_path);args=options(path)
    entry=next(e for e in manifest(path)['artifacts']if e.get('kind')=='table')
    (path/entry['path']).write_text('{}')
    with pytest.raises(ValueError):load(path,**args)


def test_rehashed_duplicate_port_and_missing_referenced_state_are_refused(tmp_path):
    path=fixture(tmp_path);value=manifest(path)
    entry=next(e for e in value['artifacts']if e.get('nodeID')=='p'and e.get('port')=='training')
    duplicate=dict(entry,path='duplicate-table.json')
    (path/duplicate['path']).write_bytes((path/entry['path']).read_bytes())
    value['artifacts'].append(duplicate);(path/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Duplicate'):load(path)
    other=fixture(tmp_path/'other',True);value=manifest(other)
    value['artifacts']=[e for e in value['artifacts']if e.get('kind')!='state']
    (other/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='missing'):load(other,True)


def test_segmented_bags_preserve_fitted_weights_empty_bags_and_exact_targets(tmp_path):
    import torch
    old=fixture(tmp_path/'legacy',fitted=True)
    new=fixture(tmp_path/'segmented',fitted=True,segmented=True)
    expected,old_provenance,_=load(old,True)
    actual,new_provenance,_=load(new,True)
    for index in [0,2]:
        for key in ['indices','offsets','per_sample_weights']:
            assert torch.equal(getattr(expected[index],key),getattr(actual[index],key))
    assert torch.equal(expected[1],actual[1]) and torch.equal(expected[3],actual[3])
    assert old_provenance['trainingRowIDs']==new_provenance['trainingRowIDs']
    assert old_provenance['preparation']['fittedStateDigests']==new_provenance['preparation']['fittedStateDigests']
