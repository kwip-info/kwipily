"""Actual preparation/cache/standalone replay, not only window helper tests."""
import hashlib
import json
from pathlib import Path
import pytest
from runner import dataflow as df
from runner.sequence_windows import DEFAULTS
from runner.text_encoding import DEFAULTS as TEXT
from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_portable_dependencies import replay
from runner.tests.test_columns import config as columns_config,item


def fixture(tmp_path):
    source=tmp_path/'fragments.csv';source.write_text('doc,order,text\na,10,de\na,2,abc\nb,0,abcdef\n')
    nodes=[('s','sourceCSV',{}),('columns','columns',columns_config([item('order',dtype='int64')])),('text','textEncode',{**TEXT,'vocabularySource':'supplied','vocabulary':list('abcdef')}),('w','sequenceWindows',{**DEFAULTS,'contextLength':3,'stride':3,'scope':'document','documentColumn':'doc','orderColumn':'order','incomplete':'pad'}),('v','view',{})]
    graph={'schemaVersion':1,'nodes':[{'id':i,'operator':'kwip.data.'+op,'config':c} for i,op,c in nodes],'edges':[{'from':{'node':a,'port':'table'},'to':{'node':b,'port':'table'}} for a,b in [('s','columns'),('columns','text'),('text','w'),('w','v')]]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));return graph,path,{'s':str(source)}

@pytest.mark.parametrize('engine',['disk','polars'])
def test_real_pipeline_cache_portable_replay_and_source_change(tmp_path,engine):
    graph,path,bindings=fixture(tmp_path)
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize',engine=engine)
    item=next(e for e in manifest['artifacts'] if e.get('nodeID')=='w' and e.get('port')=='table')
    value=json.loads((first/item['path']).read_text());names=[c['name'] for c in value['columns']]
    assert [r[names.index('input_ids')] for r in value['rows']]==[[2,3,4],[5,1,1],[2,3,4],[5,6,1]]
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin,engine=engine)
    assert 'w' in cached['cache']['reusedNodeIDs'] and cached['artifacts']==manifest['artifacts']
    private=tmp_path/'private.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'unrelated';cwd.mkdir()
    assert replay(first,private,tmp_path/'replayed',cwd)['artifacts']==manifest['artifacts']
    Path(bindings['s']).write_text('doc,order,text\na,0,abcdef\n')
    changed=run_dataflow(path,bindings,tmp_path/'changed',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin,engine=engine)
    assert 'w' not in changed['cache']['reusedNodeIDs']

@pytest.mark.parametrize('operator',['partition','sequenceWindows'])
def test_unsafe_partition_or_rewindow_rejected_before_source_reads(tmp_path,operator):
    graph,_,_=fixture(tmp_path)
    c={'strategy':'random','seed':7,'validationPercent':20,'testPercent':20} if operator=='partition' else DEFAULTS
    graph['nodes'].append({'id':'unsafe','operator':'kwip.data.'+operator,'config':c})
    graph['edges'].append({'from':{'node':'w','port':'table'},'to':{'node':'unsafe','port':'table'}})
    with pytest.raises(ValueError,match='[Ww]indow'):df.execute(graph,{'s':'/absent.csv'})


def test_real_training_fitted_partition_windows_do_not_cross_roles(tmp_path):
    graph,path,bindings=fixture(tmp_path)
    Path(bindings['s']).write_text('doc,order,text\n'+''.join(f'd{i},0,abcdefabcdef\n' for i in range(20)))
    graph['nodes'].insert(2,{'id':'p','operator':'kwip.data.partition','config':{'strategy':'random','seed':7,'validationPercent':20,'testPercent':20}})
    text=next(n for n in graph['nodes'] if n['id']=='text');text['config'].update(vocabularySource='training',vocabulary=[])
    next(n for n in graph['nodes'] if n['id']=='w')['config']['mode']='partitions'
    graph['edges']=[e for e in graph['edges'] if (e['from']['node'],e['to']['node']) not in [('columns','text'),('text','w'),('w','v')]]
    graph['edges'].append({'from':{'node':'columns','port':'table'},'to':{'node':'p','port':'table'}})
    for role in ('training','validation','test'):
        for a,b in [('p','text'),('text','w')]:graph['edges'].append({'from':{'node':a,'port':role},'to':{'node':b,'port':role}})
    graph['edges'].append({'from':{'node':'w','port':'validation'},'to':{'node':'v','port':'table'}})
    result=df.execute(graph,bindings,engine='polars')
    state=result['text']['state'];populations=[]
    for role in ('training','validation','test'):
        out=result['w'][role]
        assert out.role==role and out.fitted_states[-1]==state['contentDigest']
        assert out.partition_digest==result['p'][role].partition_digest
        populations.append(set(out.row_ids))
    assert all(a.isdisjoint(b) for i,a in enumerate(populations) for b in populations[i+1:])
    assert len(result['w']['training'].rows)==48 and len(result['w']['validation'].rows)==16
