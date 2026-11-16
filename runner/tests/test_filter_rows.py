"""Additive Filter partitions bounded tables without altering legacy operators."""
import copy
import json
from pathlib import Path
import pytest
from runner import dataflow as df
from runner.polars_preparation import preparation_engine

DEFAULT={'mode':'simple','conditions':[{'column':'feature','comparison':'gt','value':0,'caseSensitive':True}], 'match':'all','expression':'[feature] > 0','column':'feature','pattern':'.+','matchMode':'search','flags':[],'nullPolicy':'error'}
def config(**kw):return {**copy.deepcopy(DEFAULT),**kw}
def condition(column='number',comparison='ge',value=2,caseSensitive=True):return dict(column=column,comparison=comparison,value=value,caseSensitive=caseSensitive)
def table():return df.Table(({'name':'number','dtype':'int64','nullable':True},{'name':'text','dtype':'string','nullable':True},{'name':'flag','dtype':'bool','nullable':False}), ((1,'Alpha',True),(2,'beta',False),(3,'ALPHA',True),(None,None,False)),('a','b','c','d'),'source',role='validation',partition_digest='partition',fitted_states=('fit',))
def apply(c,t=None):return df._execute_node('filterRows',c,{'table':t or table()},'graph',None,lambda:None)
def graph(c):return {'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{}},{'id':'f','operator':'kwip.data.filterRows','config':c}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'f','port':'table'}}]}
def ids(out,port='matching'):return list(out[port].row_ids)

def test_exact_registry_and_default():
    df.validate(graph(config()))
    assert df.DEFINITIONS['filterRows']['defaultConfig']==DEFAULT
    assert [p['id'] for p in df.DEFINITIONS['filterRows']['outputs']]==['matching','nonmatching']

@pytest.mark.parametrize('comparison,value,expected',[
 ('eq',2,['b']),('ne',2,['a','c']),('lt',2,['a']),('le',2,['a','b']),('gt',2,['c']),('ge',2,['b','c']),
 ('in',[1,3],['a','c']),('notIn',[1,3],['b']),('isMissing',None,['d']),('isNotMissing',None,['a','b','c']),
])
def test_numeric_and_missing_comparisons(comparison,value,expected):
    out=apply(config(conditions=[condition(comparison=comparison,value=value)],nullPolicy='nonmatching'))
    assert ids(out)==expected
    assert ids(out,'nonmatching')==[i for i in ['a','b','c','d'] if i not in expected]

@pytest.mark.parametrize('comparison,value,expected',[
 ('contains','ph',['a','c']),('notContains','ph',['b']),('startsWith','al',['a','c']),('endsWith','TA',['b']),
 ('eq','alpha',['a','c']),('ne','alpha',['b']),('in',['alpha','beta'],['a','b','c']),('notIn',['alpha'],['b']),
])
def test_case_insensitive_text_and_membership(comparison,value,expected):
    assert ids(apply(config(conditions=[condition('text',comparison,value,False)],nullPolicy='nonmatching')))==expected

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('stored',[False,True])
def test_modes_preserve_row_values_ids_schema_roles(engine,stored):
    with preparation_engine(engine):
        source=table()
        if stored:source=df.streaming_rows(source,df.row_pairs(source))
        out=apply(config(conditions=[condition()],nullPolicy='nonmatching'),source)
        assert ids(out)==['b','c'] and ids(out,'nonmatching')==['a','d']
        for result in out.values():
            assert result.columns==source.columns
            assert (result.role,result.partition_digest,result.fitted_states,result.source_digest)==('validation','partition',('fit',),'source')
        assert list(out['matching'].rows)==[source.rows[1],source.rows[2]]

def test_three_valued_group_reduces_before_null_policy():
    # Missing number AND false flag is known false; OR true flag is known true.
    t=df.Table(table().columns,((None,'x',False),(None,'x',True)),('no','yes'),'source')
    a=condition();b=condition('flag','eq',True)
    assert ids(apply(config(conditions=[a,b],match='all',nullPolicy='nonmatching'),t))==[]
    assert ids(apply(config(conditions=[a,b],match='any',nullPolicy='nonmatching'),t))==['yes']
    false_only=df.Table(t.columns,(t.rows[0],),('no',),'source')
    assert ids(apply(config(conditions=[a,b],match='all'),false_only))==[]
    true_only=df.Table(t.columns,(t.rows[1],),('yes',),'source')
    assert ids(apply(config(conditions=[a,b],match='any'),true_only))==['yes']
    with pytest.raises(ValueError,match='null|missing'):apply(config(conditions=[a]))

@pytest.mark.parametrize('expression,expected',[('[number] >= 2',['b','c']),('[number] IS NULL OR [flag] = true',['a','c','d']),('false AND [number] > 0',[])])
def test_expression_uses_bounded_dsl_and_final_null_policy(expression,expected):
    out=apply(config(mode='expression',expression=expression,nullPolicy='nonmatching'))
    assert ids(out)==expected

@pytest.mark.parametrize('changes,expected',[
 ({'pattern':'alpha','flags':['ignoreCase'],'matchMode':'full'},['a','c']),
 ({'pattern':'ph','flags':[],'matchMode':'search'},['a']),
 ({'pattern':'ph','flags':[],'matchMode':'full'},[]),
])
def test_pattern_mode_uses_supervised_regex(changes,expected):
    out=apply(config(mode='pattern',column='text',nullPolicy='nonmatching',**changes))
    assert ids(out)==expected

@pytest.mark.parametrize('c',[
 config(conditions=[]),config(conditions=[condition(value=True)]),config(conditions=[condition('flag','lt',True)]),
 config(conditions=[condition('text','contains',1)]),config(conditions=[condition(comparison='in',value=[])]),
 config(conditions=[condition(comparison='isMissing',value=1)]),config(conditions=[condition('missing')]),
 config(mode='expression',expression='__import__("os")'),config(mode='expression',expression='[number] / 0 > 1'),
 config(mode='pattern',column='number'),config(mode='pattern',column='text',pattern='('),
])
def test_invalid_predicates_do_not_silently_coerce_or_run_arbitrary_code(c):
    with pytest.raises(ValueError):apply(c)

@pytest.mark.parametrize('mutation',[
 lambda c:c.update(extra=True),lambda c:c.pop('match'),lambda c:c.update(match='xor'),
 lambda c:c.update(conditions=[condition()]*33),lambda c:c.update(conditions=[condition(value=[1]*257)]),
 lambda c:c.update(conditions=[condition(caseSensitive=1)]),lambda c:c.update(flags=['dotAll','dotAll']),
 lambda c:c.update(pattern='x'*513),lambda c:c.update(expression='x'*4097),
])
def test_configuration_rejected_before_source_read(mutation,tmp_path):
    c=config();mutation(c)
    with pytest.raises(ValueError):df.execute(graph(c),{'s':str(tmp_path/'absent.csv')})


def test_image_content_is_retained_and_only_missing_tests_are_permitted():
    from runner.tests.test_image_tables import table as image_table
    source,_,ref=image_table()
    out=apply(config(conditions=[condition('image','isNotMissing',None)]),source)
    assert list(out['matching'].rows)==[(1,ref)] and list(out['nonmatching'].rows)==[(2,None)]
    assert ids(apply(config(mode='expression',expression='[image] IS NULL'),source))==['a:1']
    with pytest.raises(ValueError,match='[Ii]mage'):apply(config(conditions=[condition('image','eq','x')]),source)
    with pytest.raises(ValueError,match='[Ii]mage'):apply(config(mode='expression',expression='[image] = null'),source)

@pytest.mark.parametrize('changes,rows,expected',[
 ({'pattern':'^tail$','flags':['multiline']},['head\ntail','tailX'],['0']),
 ({'pattern':'a.b','flags':['dotAll'],'matchMode':'full'},['a\nb','aXb','ab'],['0','1']),
 ({'pattern':'alpha','flags':[],'matchMode':'full'},['Alpha','alpha'],['1']),
])
def test_all_pattern_flags_and_case_sensitive_default(changes,rows,expected):
    t=df.Table(({'name':'text','dtype':'string','nullable':False},),tuple((s,) for s in rows),tuple(str(i) for i in range(len(rows))),'source')
    assert ids(apply(config(mode='pattern',column='text',**changes),t))==expected

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('mode',['simple','expression','pattern'])
def test_large_stored_table_both_branches_stream_past_eager_limit(engine,mode):
    source=df.Table(table().columns,tuple((i%4,'ready' if i%2 else 'hold',bool(i%2)) for i in range(100_005)),tuple('id:'+str(i) for i in range(100_005)),'source')
    with preparation_engine(engine):
        source=df.streaming_rows(source,df.row_pairs(source))
        out=apply(config(mode=mode,conditions=[condition(value=2)],expression='[number] >= 2',column='text',pattern='^ready$',matchMode='full'),source)
        assert all(isinstance(result.rows,df.StoredRows) for result in out.values())
        assert sum(len(result.rows) for result in out.values())==100_005
        assert len(out['matching'].rows)==(50_002 if mode=='pattern' else 50_002)
        assert ids(out)[:2]==(['id:1','id:3'] if mode=='pattern' else ['id:2','id:3'])
        assert ids(out,'nonmatching')[-1]=='id:100004'


def test_filter_cancellation_is_observed_before_output_and_regex_timeout_reaps_worker():
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):df._execute_node('filterRows',config(conditions=[condition()]),{'table':table()},'graph',None,stop)
    from runner import dataflow_filter
    from runner.dataflow_regex import supervised
    with pytest.raises(ValueError,match='budget'):
        supervised('regexMatch',{'pattern':'(a+)+$','flags':[],'matchMode':'full','nullPolicy':'nonmatching'},['a'*5000+'!'],lambda:None,timeout=.1)

@pytest.mark.parametrize('mode,expected',[('simple',['A01','A03','A07','A08']),('expression',['A01','A03','A07','A08']),('pattern',['A01','A03','A04','A06','A08'])])
def test_real_shipped_filter_practice_and_external_replay(mode,expected,tmp_path):
    from runner.polars_csv_configuration import DEFAULTS as csv_defaults
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/filter-observations.csv'
    c=config(mode=mode,conditions=[condition('amount','ge',10)],expression='[amount] >= 10',column='status',pattern='^ready$',matchMode='full',flags=['ignoreCase'],nullPolicy='nonmatching')
    document=graph(c);document['nodes'][0]['config']={**copy.deepcopy(csv_defaults),'emptyField':'null','schemaOverrides':{'amount':'Float64'}}
    out=df.execute(document,{'s':str(source)})
    assert out['f']['matching'].values('observation_id')==expected
    assert out['f']['nonmatching'].values('observation_id')==[f'A{i:02}' for i in range(1,9) if f'A{i:02}' not in expected]
    assert sum(len(branch.rows) for branch in out['f'].values())==8
    path=tmp_path/'graph.json';path.write_text(json.dumps(document));binding=tmp_path/'bindings.json';binding.write_text(json.dumps({'s':str(source)}))
    export=tmp_path/'export';manifest=run_dataflow(path,{'s':str(source)},export,lambda _:None)
    assert (export/'dataflow_filter.py').is_file()
    other=tmp_path/'other';other.mkdir()
    reproduced=replay(export,binding,tmp_path/'replayed',other)
    assert manifest['artifacts']==reproduced['artifacts']

@pytest.mark.parametrize('condition_value',[condition(value='2'),condition(comparison='in',value=[1,True]),condition('text','in',[1,'text']),condition('flag','eq',1)])
def test_typed_membership_and_boolean_literals_are_never_coerced(condition_value):
    with pytest.raises(ValueError,match='type'):apply(config(conditions=[condition_value]))


def test_empty_branches_and_both_empty_table_retain_schema():
    out=apply(config(conditions=[condition(value=20)],nullPolicy='nonmatching'))
    assert len(out['matching'].rows)==0 and out['matching'].columns==table().columns
    source=df.Table(table().columns,(),(),'source')
    for mode in ('simple','expression','pattern'):
        out=apply(config(mode=mode,conditions=[condition()],expression='true',column='text'),source)
        assert all(len(branch.rows)==0 and branch.columns==source.columns for branch in out.values())


def test_pattern_missing_error_uses_global_row_index_across_batches(monkeypatch):
    from runner import dataflow_filter
    monkeypatch.setattr(dataflow_filter,'MAX_PATTERN_BATCH_ROWS',2)
    t=df.Table(({'name':'text','dtype':'string','nullable':True},),(('a',),('b',),('c',),(None,)),('0','1','2','3'),'source')
    with pytest.raises(ValueError,match='row 4 is null'):apply(config(mode='pattern',column='text'),t)
