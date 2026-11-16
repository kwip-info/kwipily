"""Retirement upgrades are pure; ambiguous graphs never run an old executor."""
import copy
import json
from pathlib import Path
import pytest
from runner import dataflow as df
from runner.retired_data_migration import RETIRED,upgrade_graph


def node(id,op,config):return dict(id=id,operator='kwip.data.'+op,config=config)
def edge(a,aport,b,bport):return {'from':{'node':a,'port':aport},'to':{'node':b,'port':bport}}
def graph(nodes,edges=()):return dict(schemaVersion=1,nodes=nodes,edges=list(edges))


def test_current_registry_has_no_retired_execution_and_decoder_is_bounded():
    current={x['id'].removeprefix(df.PREFIX) for x in df.REGISTRY['operators']}
    assert not current & RETIRED
    assert set(df.RETIRED_DEFINITIONS)==RETIRED
    for op in RETIRED:
        with pytest.raises(ValueError,match='Retired data execution'):df._execute_node(op,{}, {},'a'*64,None,lambda:None)


@pytest.mark.parametrize('op,config',[
 ('split',dict(seed=42,validationPercent=20)),('cast',dict(columns={'x':'float64'})),
 ('missing',dict(columns=['x'],policy='fill',value='0')),('missing',dict(columns=['x'],policy='drop')),
 ('filter',dict(column='x',comparison='eq',value=1)),
 ('calculate',dict(expression='[x]+1',outputColumn='x',dtype='float64',replaceExisting=True,nullPolicy='preserve')),
 ('transform',{})])
def test_ambiguous_upgrade_refuses_without_mutating_source(op,config):
    original=graph([node('tool',op,config)]);before=copy.deepcopy(original)
    with pytest.raises(ValueError,match='Original source is unchanged'):upgrade_graph(original)
    assert original==before


def test_exact_column_upgrade_preserves_order_names_types_and_policy():
    old=graph([node('cast','cast',dict(columns={'x':'float64'},missingPolicy='preserve'))])
    schema=[dict(name='id',dtype='string',nullable=False),dict(name='x',dtype='string',nullable=True),dict(name='tag',dtype='string',nullable=False)]
    new=upgrade_graph(old,{'cast':schema});assert new==upgrade_graph(new)
    c=new['nodes'][0]['config'];assert [r['source'] for r in c['rows']]==['id','x','tag']
    assert [r['targetType'] for r in c['rows']]==['keep','float64','keep']
    table=df.derive(df.Table(tuple(schema),(('001','2.5','a'),('002',None,'b')),('a','b'),'a'*64))
    result=df._execute_node('columns',c,{'table':table},'b'*64,None,lambda:None)['table']
    assert [c['name'] for c in result.columns]==['id','x','tag']
    assert list(result.rows)==[('001',2.5,'a'),('002',None,'b')]
    assert list(result.row_ids)==['a','b']
    selected=upgrade_graph(graph([node('n','selectRename',dict(columns=['tag','id'],rename={'tag':'label'}))]))
    assert selected['nodes'][0]['config']['rows']==[dict(source='tag',included=True,outputName='label',targetType='keep'),dict(source='id',included=True,outputName='id',targetType='keep')]


@pytest.mark.parametrize('op,c,expression,outputs',[
 ('calculate',dict(expression='[x]*2',outputColumn='twice',dtype='float64',replaceExisting=False,nullPolicy='error'),'[x]*2',['twice']),
 ('regexReplace',dict(column='name',pattern=' +',flags=[],nullPolicy='preserve',replacement=' ',replaceMode='all',outputColumn='clean',replaceExisting=False),'regex_replace([name], " +", " ", "all", "")',['clean']),
 ('regexExtract',dict(column='name',pattern='(?P<a>.*)',flags=['ignoreCase'],nullPolicy='preserve',groups=[dict(group='a',output='part')],noMatchPolicy='null',replaceExisting=False),'regex_extract([name], "(?P<a>.*)", "search", "null", "ignoreCase", "a")',['part'])])
def test_formula_upgrade_uses_current_ordered_expression_contract(op,c,expression,outputs):
    result=upgrade_graph(graph([node('tool',op,c)]));operation=result['nodes'][0]['config']['operations'][0]
    assert result['nodes'][0]['operator']=='kwip.data.formula'
    assert operation['expression']==expression
    assert [x['name'] for x in operation['outputs']]==outputs
    assert {x['write'] for x in operation['outputs']}=={'new'}


def test_filter_and_partition_exact_modes_and_missing_literals():
    expression=upgrade_graph(graph([node('n','filterExpression',dict(expression='[x]>1',nullPolicy='nonmatching'))]))
    assert expression['nodes'][0]['config']['mode']=='expression'
    pattern=upgrade_graph(graph([node('n','regexMatch',dict(column='name',pattern='x',flags=[],nullPolicy='error',matchMode='full'))]))
    assert pattern['nodes'][0]['config']['mode']=='pattern'
    entity=upgrade_graph(graph([node('p','groupSplit',dict(groupBy=['entity'],nullKeys='error',allocationUnit='groups',seed=42,validationPercent=20,testPercent=20))]))
    assert entity['nodes'][0]['config']==dict(strategy='entity',groupBy=['entity'],seed=42,validationPercent=20,testPercent=20)
    missing=graph([node('n','missing',dict(columns=['x'],policy='fill',value=7))])
    new=upgrade_graph(missing,{'n':[dict(name='x',dtype='int64',nullable=True)]})
    rule=new['nodes'][0]['config']['columns'][0]
    assert rule['missing']=='constant' and rule['value']==7


def fitted_graph():
    from runner.polars_csv_configuration import DEFAULTS
    nodes=[node('s','sourceCSV',{**DEFAULTS,'emptyField':'null','schemaOverrides':{'time':'Int64','x':'Float64'}}),node('p','temporalSplit',dict(timeColumn='time',timeEncoding='epochMilliseconds',trainingEnd=1,validationEnd=3,embargoMilliseconds=0,entityColumns=[],entityPolicy='allowAcrossTime')),node('fit','standardScalerFit',dict(columns=['x']))]
    edges=[edge('s','table','p','table'),edge('p','training','fit','training')]
    for role in ('training','validation','test'):
        nodes += [node(role,'transform',{}),node(role+'-view','view',{})]
        edges += [edge('p',role,role,'table'),edge('fit','state',role,'state'),edge(role,'table',role+'-view','table')]
    return graph(nodes,edges)


@pytest.mark.parametrize('engine',['disk','polars'])
def test_fit_apply_fusion_runs_current_preprocess_with_exact_roles_ids_and_values(tmp_path,engine):
    old=fitted_graph();new=upgrade_graph(old)
    assert not {n['operator'].removeprefix(df.PREFIX) for n in new['nodes']} & RETIRED
    assert new==upgrade_graph(new)
    source=tmp_path/'source.csv';source.write_text('time,x\n0,1\n1,3\n2,100\n3,101\n4,200\n5,201\n')
    result=df.execute(new,{'s':str(source)},engine=engine)
    for role,values in [('training',[-1.,1.]),('validation',[98.,99.]),('test',[198.,199.])]:
        assert result[role+'-view']['table'].values('x')==values
        assert result[role+'-view']['table'].role==role
        assert list(result[role+'-view']['table'].row_ids)==list(result['p'][role].row_ids)
    assert result['fit']['state']['fitEvidence']['role']=='training'
    assert result['fit']['state']['rules'][0]['mean']==2.


def test_unpaired_or_cross_family_fit_is_explicit_repair():
    for edit in ('stateConsumer','missingTraining','otherFamily'):
        old=fitted_graph()
        if edit=='stateConsumer':old['edges'].append(edge('fit','state','training-view','other'))
        elif edit=='missingTraining':old['edges']=[e for e in old['edges'] if e['to']['node']!='training']
        else:next(e for e in old['edges'] if e['to']==dict(node='validation',port='table'))['from']['node']='s'
        before=copy.deepcopy(old)
        with pytest.raises(ValueError,match='Original source is unchanged'):upgrade_graph(old)
        assert old==before


def two_stage_fitted_graph():
    old=fitted_graph()
    first=next(n for n in old['nodes'] if n['id']=='fit')
    first.update(operator=df.PREFIX+'imputerFit',config=dict(columns=['x'],strategy='mean',emptyPolicy='error'))
    next(n for n in old['nodes'] if n['id']=='p')['config'].update(trainingEnd=2,validationEnd=4)
    old['nodes'].append(node('scale','standardScalerFit',dict(columns=['x'])))
    old['edges'].append(edge('training','table','scale','training'))
    for role in ('training','validation','test'):
        old['nodes'] += [node(role+'-scaled','transform',{}),node(role+'-final','view',{})]
        old['edges'] += [edge(role,'table',role+'-scaled','table'),edge('scale','state',role+'-scaled','state'),edge(role+'-scaled','table',role+'-final','table')]
    return old


def semantic_graph(value):
    return dict(schemaVersion=value['schemaVersion'],nodes=sorted(value['nodes'],key=lambda n:n['id']),
                edges=sorted(value['edges'],key=lambda e:json.dumps(e,sort_keys=True)))


@pytest.mark.parametrize('engine',['disk','polars'])
def test_reversed_two_stage_fit_order_has_identical_upgrade_and_scientific_outputs(tmp_path,engine):
    import math
    ordered=two_stage_fitted_graph();reversed_graph=copy.deepcopy(ordered)
    reversed_graph['nodes'].reverse();before=copy.deepcopy(reversed_graph)
    expected=upgrade_graph(ordered);actual=upgrade_graph(reversed_graph)
    assert semantic_graph(actual)==semantic_graph(expected)
    assert reversed_graph==before and upgrade_graph(actual)==actual
    assert [n['id'] for n in actual['nodes']]==[n['id'] for n in before['nodes'] if n['operator']!=df.PREFIX+'transform']
    source=tmp_path/'source.csv';source.write_text('time,x\n0,1\n1,\n2,3\n3,100\n4,101\n5,200\n6,201\n')
    reference=df.execute(expected,{'s':str(source)},engine=engine)
    result=df.execute(actual,{'s':str(source)},engine=engine)
    scale=math.sqrt(2/3)
    for role,values in [('training',[-1/scale,0.,1/scale]),('validation',[98/scale,99/scale]),('test',[198/scale,199/scale])]:
        table=result[role+'-final']['table'];other=reference[role+'-final']['table']
        assert table.values('x')==pytest.approx(values,rel=1e-12,abs=1e-12)
        assert list(table.rows)==list(other.rows) and list(table.row_ids)==list(other.row_ids)
        assert table.columns==other.columns and table.role==other.role==role
        assert list(table.row_ids)==list(result['p'][role].row_ids)
    assert result['fit']['state']['rules'][0]['fill']==2.
    assert result['scale']['state']['rules'][0]['mean']==2.
    assert result['scale']['state']['rules'][0]['scale']==pytest.approx(scale)
    for ident in ('fit','scale'):
        assert list(df.fit_observation_ids(result[ident]['state']))==list(result['p']['training'].row_ids)


@pytest.mark.parametrize('invalid',['cycle','unprovable'])
def test_two_stage_cycle_or_unprovable_family_refuses_without_partial_upgrade(invalid):
    old=two_stage_fitted_graph()
    incoming=next(e for e in old['edges'] if e['to']==dict(node='fit',port='training'))
    incoming['from']=dict(node='training-scaled' if invalid=='cycle' else 's',port='table')
    old['nodes'].reverse();before=copy.deepcopy(old)
    with pytest.raises(ValueError,match='Original source is unchanged'):upgrade_graph(old)
    assert old==before
