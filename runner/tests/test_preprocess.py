import json
import math
import pytest
from runner import dataflow as df
from runner.dataflow_preprocess import apply, rule_defaults, load


def table(rows,role='training',ids=None):
    columns=({'name':'x','dtype':'float64','nullable':True},{'name':'city','dtype':'string','nullable':True},{'name':'target','dtype':'int64','nullable':False})
    return df.derive(df.Table(columns,tuple(map(tuple,rows)),tuple(ids or [f'{role}-{i}' for i in range(len(rows))]),'a'*64,role,partition_digest='b'*64))


def config(mode='partitions'):
    return {'mode':mode,'columns':[dict(rule_defaults('x'),missing='mean',scaling='standard'),dict(rule_defaults('city'),missing='mostFrequent',encoding='oneHot',unknownPolicy='ignore',tiePolicy='smallest')]}


def test_fit_training_only_and_same_rules_across_three_roles():
    train=table([[1.,'a',0],[3.,'a',1],[None,'b',0]])
    val=table([[100.,'heldout',1],[None,None,0]],'validation')
    test=table([[5.,'b',1]],'test')
    out=apply({'training':train,'validation':val,'test':test},config(),'c'*64,lambda:None)
    assert set(out)=={'training','validation','test','state'}
    assert [c['name'] for c in out['training'].columns]==['x','onehot:["city","a"]','onehot:["city","b"]','target']
    assert list(out['validation'].rows)[0][1:3]==(0.,0.)
    assert list(out['validation'].rows)[1][0]==0.
    assert list(out['test'].rows)[0][0]==pytest.approx(3/math.sqrt(2/3))
    assert all(out[p].fitted_states==(out['state']['contentDigest'],) for p in ('training','validation','test'))
    assert out['state']['fitEvidence']['observationIDs']==list(train.row_ids)
    assert load(json.dumps(out['state']))==out['state']


def test_optional_partitions_are_absent_not_invented():
    out=apply({'training':table([[1.,'a',0],[3.,'a',1]])},config(),'c'*64,lambda:None)
    assert set(out)=={'training','state'}


def test_fixed_rules_drop_and_typed_fill_preserve_identities():
    c={'mode':'table','columns':[dict(rule_defaults('x'),missing='constant',value=7.),dict(rule_defaults('city'),missing='drop')]}
    source=table([[None,'a',0],[2.,None,1]],'unassigned')
    out=apply({'table':source},c,'c'*64,lambda:None)['table']
    assert list(out.rows)==[(7.,'a',0)]
    assert list(out.row_ids)==['unassigned-0']
    assert not out.fitted_states
    with pytest.raises(ValueError):apply({'table':source},config('table'),'c'*64,lambda:None)


@pytest.mark.parametrize('kind',['role','source','schema','overlap','partition'])
def test_mismatched_heldout_refuses(kind):
    train=table([[1.,'a',0],[3.,'a',1]])
    val=table([[2.,'b',0]],'validation')
    if kind=='role':val=df.replace(val,role='training')
    if kind=='source':val=df.replace(val,source_digest='d'*64)
    if kind=='schema':val=df.replace(val,columns=tuple(reversed(val.columns)))
    if kind=='overlap':val=df.replace(val,row_ids=('training-0',))
    if kind=='partition':val=df.replace(val,partition_digest='d'*64)
    with pytest.raises(ValueError):apply({'training':train,'validation':val},config(),'c'*64,lambda:None)


def test_median_ties_empty_unknown_and_cancel():
    c={'mode':'partitions','columns':[dict(rule_defaults('x'),missing='median')]}
    out=apply({'training':table([[9.,'a',0],[1.,'b',1],[3.,'a',0],[None,'a',1]])},c,'c'*64,lambda:None)
    assert list(out['training'].rows)[-1][0]==3.
    with pytest.raises(ValueError):apply({'training':table([[None,'a',0]])},c,'c'*64,lambda:None)
    c['columns'][0].update(emptyPolicy='constant',emptyValue=11.)
    assert list(apply({'training':table([[None,'a',0]])},c,'c'*64,lambda:None)['training'].rows)[0][0]==11.
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):apply({'training':table([[1.,'a',0]])},c,'c'*64,cancel)


def practice_graph():
    from runner.polars_csv_configuration import DEFAULTS
    source={'id':'s','operator':'kwip.data.sourceCSV','config':{**DEFAULTS,'emptyField':'null','schemaOverrides':{'time':'Int64','x':'Float64','target':'Int64'}}}
    partition={'id':'p','operator':'kwip.data.partition','config':{'strategy':'time','timeColumn':'time','timeEncoding':'epochMilliseconds','trainingEnd':5000,'validationEnd':8000,'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'}}
    prep={'id':'prep','operator':'kwip.data.preprocess','config':config()}
    edges=[{'from':{'node':'s','port':'table'},'to':{'node':'p','port':'table'}}]
    edges += [{'from':{'node':'p','port':role},'to':{'node':'prep','port':role}} for role in ('training','validation','test')]
    return {'schemaVersion':1,'nodes':[source,partition,prep],'edges':edges}


@pytest.mark.parametrize('engine',['disk','polars'])
def test_real_capture_cache_history_portable_replay_and_frozen_test(tmp_path,engine,monkeypatch):
    from pathlib import Path
    from runner.dataflow_cli import run_dataflow, inspect_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    from runner.prepared_data import load_prepared,load_test_partition
    from runner.dataflow_fitted import portable_parameters
    from runner.portable_fitted import apply as portable_apply
    import hashlib
    graph=practice_graph();path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/preprocess-practice-observations.csv'
    output=tmp_path/'capture';m=run_dataflow(path,{'s':str(source)},output,lambda _:None,engine=engine,cache_policy='materialize')
    entries={e['port']:e for e in m['artifacts'] if e.get('nodeID')=='prep'}
    values={p:json.loads((output/e['path']).read_text()) for p,e in entries.items()}
    assert [len(values[p]['rows']) for p in ('training','validation','test')]==[6,3,3]
    assert values['state']['rules'][0]['fill']==3.6
    assert values['state']['rules'][1]['vocabulary']==['a','b']
    state=values['state'];params=portable_parameters(state)
    raw=[e for e in m['artifacts'] if e.get('nodeID')=='p' and e.get('port')=='validation'][0]
    raw=json.loads((output/raw['path']).read_text())
    transformed,columns,_=portable_apply(raw['rows'],raw['columns'],params)
    assert transformed==values['validation']['rows'];assert columns==values['validation']['columns']
    pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    inspect_dataflow(output/'manifest.json',pin,lambda _:None)
    binding=tmp_path/'bindings.json';binding.write_text(json.dumps({'s':str(source)}));cwd=tmp_path/'elsewhere';cwd.mkdir()
    reproduced=replay(output,binding,tmp_path/'replay',cwd)
    assert m['artifacts']==reproduced['artifacts']
    # Cache reuse must not fit again.
    from runner import dataflow_preprocess as prep
    monkeypatch.setattr(prep,'_fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('refit forbidden')))
    reused=run_dataflow(path,{'s':str(source)},tmp_path/'reused',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=output/'manifest.json',reuse_manifest_sha256=pin)
    assert 'prep' in reused['cache']['reusedNodeIDs']
    tensors,provenance,_=load_prepared(output/'manifest.json','prep','training','prep','validation',['x'],'target',expected_manifest_sha256=pin,sample_shape=[1])
    result,_,_=load_test_partition(output/'manifest.json','prep','test',provenance,expected_manifest_sha256=pin,sample_shape=[1])
    assert result[0].shape==(3,1)
    assert tensors[0].shape==(6,1)


def test_mode_ports_and_missing_optional_output_refuse():
    graph=practice_graph();graph['edges']=[e for e in graph['edges'] if e['to']['port']!='test']
    nodes,inputs,_=df.validate(graph)
    assert df.configured_ports(*nodes['prep'],inputs['prep'])[2]=={'training','validation','state'}
    graph['nodes'].append({'id':'v','operator':'kwip.data.view','config':{}})
    graph['edges'].append({'from':{'node':'prep','port':'test'},'to':{'node':'v','port':'table'}})
    with pytest.raises(ValueError,match='corresponding'):df.validate(graph)


@pytest.mark.parametrize('center,unit',[(True,True),(True,False),(False,True),(False,False)])
def test_scaling_switches_population_variance_and_constant_columns(center,unit):
    c={'mode':'partitions','columns':[dict(rule_defaults('x'),scaling='standard',center=center,unitVariance=unit)]}
    out=apply({'training':table([[1.,'a',0],[5.,'b',1]])},c,'c'*64,lambda:None)
    assert [row[0] for row in out['training'].rows]==[(1.-(3. if center else 0.))/(2. if unit else 1.),(5.-(3. if center else 0.))/(2. if unit else 1.)]
    constant=apply({'training':table([[3.,'a',0],[3.,'b',1]])},c,'c'*64,lambda:None)
    assert all(math.isfinite(row[0]) for row in constant['training'].rows)


@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('count',[100001,500001])
def test_large_streamed_fit_exceeds_old_cap(engine,count):
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_disk import disk_scope
    columns=({'name':'x','dtype':'float64','nullable':False},)
    with disk_scope(),preparation_engine(engine):
        prefix='source:'+'a'*64+':'
        writer=df.row_store(columns,lambda:None,identity_prefix=prefix)
        for i in range(count):writer.append((float(i),),prefix+str(i))
        rows,ids=writer.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64,'training',partition_digest='b'*64))
        out=apply({'training':source},{'mode':'partitions','columns':[dict(rule_defaults('x'),scaling='standard')]},'c'*64,lambda:None)
        assert len(out['training'].rows)==count
        assert isinstance(out['training'].rows,df.StoredRows)
        assert out['state']['rules'][0]['mean']==(count-1)/2
        assert out['state']['fitEvidence']['observationIDPrefix']==prefix
        assert len(json.dumps(out['state']).encode())<6_000_000
        assert list(df.fit_observation_ids(out['state']))[:2]==[prefix+'0',prefix+'1']
        assert abs(sum(row[0] for row in out['training'].rows))<1e-6


def test_policy_combinations_typed_literals_and_training_exclusions():
    train=table([[1.,'a',0],[None,'b',1],[9.,None,0]])
    c={'mode':'partitions','columns':[dict(rule_defaults('x'),missing='mean'),dict(rule_defaults('city'),missing='drop')]}
    out=apply({'training':train},c,'c'*64,lambda:None)
    assert [r[0] for r in out['training'].rows]==[1.,1.]
    assert out['state']['fitEvidence']['observationIDs']==['training-0','training-1']
    assert out['training'].diagnostics[-1]=={'code':'missingRowsExcluded','count':1}
    c={'mode':'partitions','columns':[dict(rule_defaults('city'),missing='mostFrequent')]}
    with pytest.raises(ValueError,match='tie'):apply({'training':table([[1.,'a',0],[3.,'b',1]])},c,'c'*64,lambda:None)
    c['columns'][0]['tiePolicy']='smallest'
    assert apply({'training':table([[1.,'a',0],[3.,'b',1],[4.,None,0]])},c,'c'*64,lambda:None)['state']['rules'][0]['fill']=='a'
    c={'mode':'table','columns':[dict(rule_defaults('x'),missing='constant',value='bad')]}
    with pytest.raises(ValueError,match='type'):apply({'table':train},c,'c'*64,lambda:None)
    c={'mode':'partitions','columns':[dict(rule_defaults('city'),encoding='oneHot',vocabularyOrder='firstSeen')]}
    out=apply({'training':table([[1.,'b',0],[3.,'a',1]])},c,'c'*64,lambda:None)
    assert out['state']['rules'][0]['vocabulary']==['b','a']
    with pytest.raises(ValueError,match='Unknown'):apply({'training':table([[1.,'b',0],[3.,'a',1]]),'test':table([[4.,'c',1]],'test')},c,'c'*64,lambda:None)
    c['columns'][0]['maxCategories']=1
    with pytest.raises(ValueError,match='maximum'):apply({'training':train},c,'c'*64,lambda:None)


@pytest.mark.parametrize('engine',['disk','polars'])
def test_images_pass_through_and_cannot_be_scalar_preprocessed(engine):
    from runner.tests.test_image_tables import table as image_table
    from runner.polars_preparation import preparation_engine
    source,_,reference=image_table()
    with preparation_engine(engine):
        out=apply({'table':source},{'mode':'table','columns':[]},'c'*64,lambda:None)['table']
        assert out.columns==source.columns
        assert list(out.rows)==list(source.rows)
        assert list(apply({'table':source},{'mode':'table','columns':[rule_defaults('image')]},'c'*64,lambda:None)['table'].rows)==list(source.rows)
        with pytest.raises(ValueError,match='Image'):apply({'table':source},{'mode':'table','columns':[dict(rule_defaults('image'),missing='constant')]},'c'*64,lambda:None)


def test_frozen_real_weight_export_replays_preprocess_without_refitting(tmp_path,monkeypatch):
    import hashlib
    from pathlib import Path
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_portable_evaluation import captured_evaluation
    from runner.tests.test_runner import graph as model_graph
    from runner.portable_evaluation_export import build_package
    from runner.portable_evaluation import evaluate_package
    from runner import dataflow_preprocess
    source=tmp_path/'observations.csv';source.write_bytes((Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/preprocess-practice-observations.csv').read_bytes())
    path=tmp_path/'graph.json';path.write_text(json.dumps(practice_graph()))
    preparation=tmp_path/'prepared';run_dataflow(path,{'s':str(source)},preparation,lambda _:None)
    mapping=('prep','training','prep','validation',['x','time'],'target')
    root,pin=captured_evaluation(tmp_path,preparation/'manifest.json',model_graph(),mapping,('prep','test'))
    source.unlink()
    monkeypatch.setattr(dataflow_preprocess,'_fit',lambda *a,**k:pytest.fail('Frozen export must not refit'))
    destination=tmp_path/'portable';build_package(root,destination,pin,include_test_rows=True)
    assert (destination/'preprocess_parameters.py').is_file()
    assert evaluate_package(destination,replay=True,batch_size=2)['evaluatedCount']==3


@pytest.mark.parametrize('mode',['table','partitions'])
def test_real_optional_capture_and_historical_inspection(tmp_path,mode):
    from pathlib import Path
    from runner.polars_csv_configuration import DEFAULTS
    from runner.dataflow_cli import run_dataflow,inspect_dataflow
    import hashlib
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/preprocess-practice-observations.csv'
    graph=practice_graph()
    if mode=='partitions':graph['edges']=[e for e in graph['edges'] if e['to']['port'] not in ('validation','test')]
    else:
        graph['nodes']=[graph['nodes'][0],{'id':'prep','operator':'kwip.data.preprocess','config':{'mode':'table','columns':[dict(rule_defaults('x'),missing='constant',value=9.)]}}]
        graph['edges']=[{'from':{'node':'s','port':'table'},'to':{'node':'prep','port':'table'}}]
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));output=tmp_path/'capture'
    manifest=run_dataflow(path,{'s':str(source)},output,lambda _:None)
    ports={e['port'] for e in manifest['artifacts'] if e.get('nodeID')=='prep'}
    assert ports==({'table'} if mode=='table' else {'training','state'})
    pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    inspect_dataflow(output/'manifest.json',pin,lambda _:None)


def test_compact_fit_membership_rejects_tampering_and_legacy_dev_state_stays_readable():
    from runner.dataflow_disk import disk_scope
    with disk_scope():
        cols=({'name':'x','dtype':'float64','nullable':False},)
        prefix='source:'+'a'*64+':'
        writer=df.row_store(cols,lambda:None,identity_prefix=prefix)
        writer.append((1.,),prefix+'0');writer.append((3.,),prefix+'1')
        rows,ids=writer.finish();table=df.derive(df.Table(cols,rows,ids,'a'*64,'training',partition_digest='b'*64))
        outputs=apply({'training':table},{'mode':'partitions','columns':[dict(rule_defaults('x'),scaling='standard')]},'c'*64,lambda:None)
        state=outputs['state'];assert list(df.fit_observation_ids(state))==[prefix+'0',prefix+'1']
        forged=json.loads(json.dumps(state));forged['fitEvidence']['observationIDPrefix']='other:'
        forged['contentDigest']=df.digest({k:v for k,v in forged.items() if k!='contentDigest'})
        with pytest.raises(ValueError,match='evidence'):load(json.dumps(forged))
        legacy=json.loads(json.dumps(state));evidence=legacy['fitEvidence']
        evidence['observationIDs']=list(df.fit_observation_ids(legacy));del evidence['observationIDPrefix'];del evidence['observationIDSuffixes']
        legacy['contentDigest']=df.digest({k:v for k,v in legacy.items() if k!='contentDigest'})
        assert list(df.fit_observation_ids(load(json.dumps(legacy))))==[prefix+'0',prefix+'1']


@pytest.mark.parametrize('engine',['disk','polars'])
def test_fixed_text_cleanup_order_nulls_ids_and_cache_verification(engine):
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_preprocess import verify_output
    columns=({'name':'text','dtype':'string','nullable':True},)
    source=df.derive(df.Table(columns,((' \tＥ\u0301!１２\n A\x00 ',),(' !1 ',),(None,),('',)),('a','b','c','d'),'a'*64,'unassigned'))
    rule=dict(rule_defaults('text'),unicodeNormalization='NFKC',removeControls=True,removePunctuation=True,removeDigits=True,whitespace='collapse',case='lower',emptyAfterCleanup='missing',missing='constant',value='fixed')
    c={'mode':'table','columns':[rule]}
    with preparation_engine(engine):
        out=apply({'table':source},c,'c'*64,lambda:None)
        assert list(out['table'].rows)==[('é a',),('fixed',),('fixed',),('fixed',)]
        assert list(out['table'].row_ids)==['a','b','c','d']
        assert not out['table'].fitted_states
        verify_output({'table':source},c,'c'*64,out,lambda:None)
        c['columns'][0]['missing']='drop'
        dropped=apply({'table':source},c,'c'*64,lambda:None)
        assert list(dropped['table'].row_ids)==['a']
        verify_output({'table':source},c,'c'*64,dropped,lambda:None)


@pytest.mark.parametrize('options,expected',[
    ({},' \tE\u0301!١\n A\x00 '),({'unicodeNormalization':'NFC'},' \tÉ!١\n A\x00 '),
    ({'removeControls':True},' E\u0301!١ A '),({'removePunctuation':True},' \tE\u0301١\n A\x00 '),
    ({'removeDigits':True},' \tE\u0301!\n A\x00 '),({'whitespace':'trim'},'E\u0301!١\n A\x00'),
    ({'whitespace':'collapse'},'E\u0301!١ A\x00'),({'case':'lower'},' \te\u0301!١\n a\x00 '),
    ({'case':'upper'},' \tE\u0301!١\n A\x00 ')])
def test_each_fixed_cleanup_option(options,expected):
    source=df.derive(df.Table(({'name':'s','dtype':'string','nullable':False},),((' \tE\u0301!١\n A\x00 ',),),('id',),'a'*64,'unassigned'))
    c={'mode':'table','columns':[dict(rule_defaults('s'),**options)]}
    assert list(apply({'table':source},c,'c'*64,lambda:None)['table'].rows)==[(expected,)]


def test_cleanup_rejects_nontext_and_partitioned_and_projects_nullable():
    source=table([[1.,'a',0]],'unassigned')
    for c in [{'mode':'table','columns':[dict(rule_defaults('x'),case='lower')]},{'mode':'partitions','columns':[dict(rule_defaults('city'),case='lower')]}]:
        with pytest.raises(ValueError,match='Text|Table'):apply({'table':source} if c['mode']=='table' else {'training':table([[1.,'a',0]])},c,'c'*64,lambda:None)
    cols=({'name':'s','dtype':'string','nullable':False},)
    source=df.derive(df.Table(cols,(('',),),('id',),'a'*64,'unassigned'))
    c={'mode':'table','columns':[dict(rule_defaults('s'),emptyAfterCleanup='missing')]}
    out=apply({'table':source},c,'c'*64,lambda:None)['table']
    assert out.columns[0]['nullable'] and list(out.rows)==[(None,)]


def test_legacy_rules_portable_over_100k_and_cached_value_tampering():
    from runner import preprocess_parameters as portable
    from runner.dataflow_preprocess import verify_output
    legacy={k:v for k,v in rule_defaults('s').items() if k not in portable.CLEANUP_DEFAULTS}
    c={'mode':'table','columns':[legacy]}
    columns=[{'name':'s','dtype':'string','nullable':False}]
    rules=[dict(legacy,fill=None,vocabulary=[],mean=0.,scale=1.)]
    params=dict(schemaVersion=1,kind='portableFittedParameters',operator='kwip.data.preprocess',columns=['s'],inputSchema=columns,outputSchema=columns,sourceContentDigest='a'*64,configuration=c,rules=rules)
    rows=[['unchanged']]*100001
    values,schema,_=portable.apply(rows,columns,params)
    assert len(values)==100001 and schema==columns and values[-1]==['unchanged']
    source=df.derive(df.Table(tuple(columns),(('original',),),('id',),'a'*64,'unassigned'))
    outputs=apply({'table':source},c,'c'*64,lambda:None)
    outputs['table']=df.replace(outputs['table'],rows=(('forged',),))
    with pytest.raises(ValueError,match='values'):verify_output({'table':source},c,'c'*64,outputs,lambda:None)


@pytest.mark.parametrize('dtype,value',[('int64',7),('float64',1.25),('bool',True),('string','inserted!')])
def test_typed_fixed_replacements_never_cast_or_reclean(dtype,value):
    columns=({'name':'v','dtype':dtype,'nullable':True},)
    source=df.derive(df.Table(columns,((None,),),('id',),'a'*64,'unassigned'))
    c={'mode':'table','columns':[dict(rule_defaults('v'),missing='constant',value=value,**({'removePunctuation':True} if dtype=='string' else {}))]}
    assert list(apply({'table':source},c,'c'*64,lambda:None)['table'].rows)==[(value,)]
    c['columns'][0]['value']='wrong' if dtype!='string' else 3
    with pytest.raises(ValueError,match='type'):apply({'table':source},c,'c'*64,lambda:None)


@pytest.mark.parametrize('dtype,value',[('date',0),('datetime',1000001),('time',1000001)])
def test_temporal_preservation_missing_policy_and_metadata(dtype,value):
    from runner.dataflow_temporal import metadata
    columns=({'name':'v','dtype':dtype,'nullable':True,'temporal':metadata(dtype)},)
    source=df.derive(df.Table(columns,((value,),(None,)),('a','b'),'a'*64,'unassigned'))
    c={'mode':'table','columns':[rule_defaults('v')]}
    out=apply({'table':source},c,'c'*64,lambda:None)['table']
    assert out.columns==columns and list(out.rows)==[(value,),(None,)]
    c['columns'][0]['missing']='drop'
    assert list(apply({'table':source},c,'c'*64,lambda:None)['table'].row_ids)==['a']
    c['columns'][0]['missing']='error'
    with pytest.raises(ValueError,match='Missing'):apply({'table':source},c,'c'*64,lambda:None)
    c['columns'][0]['missing']='constant'
    with pytest.raises(ValueError,match='temporal'):apply({'table':source},c,'c'*64,lambda:None)


@pytest.mark.parametrize('engine',['disk','polars'])
def test_cleanup_practice_capture_standalone_replay_and_cache(tmp_path,engine):
    from pathlib import Path
    import hashlib
    from runner.polars_csv_configuration import DEFAULTS
    from runner.dataflow_cli import run_dataflow,inspect_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    rule=dict(rule_defaults('name'),unicodeNormalization='NFKC',removeControls=True,removePunctuation=True,removeDigits=True,whitespace='collapse',case='lower',emptyAfterCleanup='missing',missing='drop')
    c={'mode':'table','columns':[rule,dict(rule_defaults('amount'),missing='constant',value=0.)]}
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{**DEFAULTS,'emptyField':'null','schemaOverrides':{'name':'String','amount':'Float64'}}},{'id':'prep','operator':'kwip.data.preprocess','config':c}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'prep','port':'table'}}]}
    graph_path=tmp_path/'graph.json';graph_path.write_text(json.dumps(graph))
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/preprocess-cleanup-observations.csv'
    output=tmp_path/'capture';manifest=run_dataflow(graph_path,{'s':str(source)},output,lambda _:None,engine=engine,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='prep' and e.get('port')=='table')
    table=json.loads((output/entry['path']).read_text())
    assert table['rows']==[['alice brown',125.],['bobsmith',0.],['chén wei',78.5]]
    def identities(payload):
        evidence=payload['lineage']
        return evidence['observationIDs'] if 'observationIDs' in evidence else [evidence['observationIDPrefix']+suffix for suffix in evidence['observationIDSuffixes']]
    source_entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='s' and e.get('port')=='table')
    captured_source=json.loads((output/source_entry['path']).read_text())
    assert identities(table)==identities(captured_source)[:3]
    pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    inspect_dataflow(output/'manifest.json',pin,lambda _:None)
    binding=tmp_path/'bindings.json';binding.write_text(json.dumps({'s':str(source)}));cwd=tmp_path/'elsewhere';cwd.mkdir()
    reproduced=replay(output,binding,tmp_path/'replay',cwd)
    assert manifest['artifacts']==reproduced['artifacts']
    reused=run_dataflow(graph_path,{'s':str(source)},tmp_path/'reuse',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=output/'manifest.json',reuse_manifest_sha256=pin)
    assert 'prep' in reused['cache']['reusedNodeIDs']


@pytest.mark.parametrize('missing,expected',[('keep',[(None,),('kept',)]),('constant',[('fixed!',),('kept',)]),('drop',[('kept',)]),('error',None)])
def test_empty_after_cleanup_all_missing_policies(missing,expected):
    columns=({'name':'s','dtype':'string','nullable':False},)
    source=df.derive(df.Table(columns,((' !١ ',),('kept',)),('empty','kept'),'a'*64,'unassigned'))
    c={'mode':'table','columns':[dict(rule_defaults('s'),removePunctuation=True,removeDigits=True,whitespace='trim',emptyAfterCleanup='missing',missing=missing,value='fixed!')]}
    if expected is None:
        with pytest.raises(ValueError,match='Missing'):apply({'table':source},c,'c'*64,lambda:None)
    else:
        out=apply({'table':source},c,'c'*64,lambda:None)['table']
        assert list(out.rows)==expected
        assert list(out.row_ids)==(['kept'] if missing=='drop' else ['empty','kept'])
        assert out.columns[0]['nullable']==(missing=='keep')


def test_unicode_character_categories_case_expansion_and_keep_empty():
    from runner import preprocess_parameters as portable
    # Format characters (Cf), letter numbers (Nl), superscripts (No) are not Cc/P/Nd.
    rule=dict(rule_defaults('s'),removeControls=True,removePunctuation=True,removeDigits=True)
    assert portable.clean_text('\x00—١\u200dⅣ²',rule)=='\u200dⅣ²'
    assert portable.clean_text('Straße',dict(rule_defaults('s'),case='upper'))=='STRASSE'
    assert portable.clean_text('İ',dict(rule_defaults('s'),case='lower'))=='i\u0307'
    assert portable.clean_text('',rule)==''
    assert portable.clean_text(None,rule) is None
    for options in [{'unicodeNormalization':'NFD'},{'removeControls':1},{'case':'fold'},{'whitespace':'guess'},{'emptyAfterCleanup':'guess'}]:
        with pytest.raises(ValueError,match='cleanup'):portable.validate_config({'mode':'table','columns':[dict(rule_defaults('s'),**options)]})
    duplicate={'mode':'table','columns':[rule,rule]}
    with pytest.raises(ValueError,match='once'):portable.validate_config(duplicate)
