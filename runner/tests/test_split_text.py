"""Independent option fixtures for literal extraction, identity and standalone replay."""
import copy
import json
from pathlib import Path
import pytest
from runner import dataflow as df
from runner import dataflow_split_text as st
from runner.polars_preparation import preparation_engine

def field(name='part',dtype='string',start=1,width=1):return dict(name=name,type=dtype,start=start,width=width)
def cfg(**kw):return {**st.defaults(),'source':'text',**kw}
def table(values,**kw):return df.Table((dict(name='id',dtype='int64',nullable=False),dict(name='text',dtype='string',nullable=True)),tuple((i+1,v) for i,v in enumerate(values)),tuple('row'+str(i) for i in range(len(values))),'source',**kw)
def rows(values,c=None,**kw):return list(st.apply(table(values),c or cfg(**kw)).rows)
def graph(c):return {'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{}},{'id':'split','operator':'kwip.data.splitText','config':c}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'split','port':'table'}}]}

@pytest.mark.parametrize('engine',['disk','polars'])
def test_columns_quoted_multichar_delimiters_and_lineage(engine):
    c=cfg(delimiter='::',fields=[field('city'),field('number','int64'),field('flag','bool')])
    t=table(['"New::York"::12::true','"Say ""yes"""::-3::false'],role='training',partition_digest='partition',fitted_states=('fitted',))
    with preparation_engine(engine):out=st.apply(t,c)
    assert list(out.rows)==[(1,'New::York',12,True),(2,'Say "yes"',-3,False)]
    assert [x['name'] for x in out.columns]==['id','city','number','flag']
    assert list(out.row_ids)==list(t.row_ids) and out.evidence()==t.evidence()
    assert (out.role,out.partition_digest,out.fitted_states)==('training','partition',('fitted',))
    st.verify_output(t,c,out)

@pytest.mark.parametrize('c,text,expected',[
    (cfg(delimiter='💠💠',quote="'",escaping='escape',escape='~',fields=[field('x'),field('y')]), "'a💠💠b~'c'💠💠z",[(1,"a💠💠b'c",'z')]),
    (cfg(delimiter='|',quoted=False,escaping='escape',fields=[field('x'),field('y')]),'a\\|b|c',[(1,'a|b','c')]),
    (cfg(escaping='none',fields=[field('x'),field('y')]),'"a,b",c',[(1,'a,b','c')]),
    (cfg(quoted=False,fields=[field('x'),field('y'),field('z')]),'"a,b",c',[(1,'"a','b"','c')]),
    (cfg(extraFields='drop',fields=[field('x')]),'a,b,c',[(1,'a')]),
    (cfg(extraFields='remainder',fields=[field('x'),field('y')]),'a,"b,c",d',[(1,'a','b,c,d')]),
    (cfg(shortFields='missing',fields=[field('x'),field('y')]),'a',[(1,'a',None)]),
    (cfg(shortFields='empty',fields=[field('x'),field('y')]),'a',[(1,'a','')]),
    (cfg(fields=[field('x'),field('y'),field('z')]),',a,',[(1,'','a','')]),
    (cfg(retainSource=True), 'a', [(1,'a','a')]),
    (cfg(trim=True,fields=[field('x','int64')]),' 12 ',[(1,12)]),
    (cfg(conversionFailure='missing',fields=[field('x','int64')]),'1.5',[(1,None)]),
    (cfg(fields=[field('x','float64')]),'12.5',[(1,12.5)]),
])
def test_delimited_options(c,text,expected):assert rows([text],c)==expected

@pytest.mark.parametrize('config,text,message',[
    (cfg(),'a,b','extra'),(cfg(fields=[field('x'),field('y')],shortFields='error'),'a','short'),
    (cfg(),None,'unused'),(cfg(missingSource='error'),None,'missing'),
    (cfg(),'"unterminated','Unclosed'),(cfg(),'"a"oops','Unexpected'),
    (cfg(escaping='escape'),'a\\','Trailing'),(cfg(fields=[field('x','bool')]),'TRUE','Boolean'),
    (cfg(fields=[field('x','float64')]),'nan','finite'),(cfg(fields=[field('x','int64')]),'9223372036854775808','int64'),
])
def test_delimited_errors(config,text,message):
    if message=='unused':assert rows([text],config)==[(1,None)]
    else:
        with pytest.raises(ValueError,match=message):rows([text],config)

@pytest.mark.parametrize('empty,expected,ordinals',[('keep',['a','','b'],[1,2,3]),('drop',['a','b'],[1,3])])
def test_rows_empty_tokens_ordinal_and_missing(empty,expected,ordinals):
    c=cfg(outputMode='rows',emptyTokens=empty,ordinal=True,fields=[field('token')])
    t=table(['a,,b',None]);out=st.apply(t,c)
    assert list(out.rows)==[(1,v,i) for v,i in zip(expected,ordinals)]+[(2,None,1)]
    assert len(set(out.row_ids))==len(out.row_ids)
    assert list(out.row_ids)==['split:'+df.digest(['kwip.splitText.token.v1',t.source_digest,'row0',i]) for i in ordinals]+['split:'+df.digest(['kwip.splitText.token.v1',t.source_digest,'row1',1])]
    assert out.source_digest==st.source_digest(t,c) and out.evidence()!=t.evidence()
    st.verify_output(t,c,out)

@pytest.mark.parametrize('kwargs',[dict(role='training'),dict(partition_digest='partition'),dict(fitted_states=('fit',))])
def test_rows_refuses_partition_or_fitted_input(kwargs):
    with pytest.raises(ValueError,match='before Partition'):st.apply(table(['a,b'],**kwargs),cfg(outputMode='rows'))

@pytest.mark.parametrize('engine',['disk','polars'])
def test_fixed_width_unicode_codepoints_overlaps_and_types(engine):
    c=cfg(mode='fixedWidth',fields=[field('accent',start=1,width=1),field('overlap',start=1,width=2),field('n','int64',3,2),field('tail',start=4,width=9)])
    with preparation_engine(engine):out=st.apply(table(['é界12']),c)
    assert list(out.rows)==[(1,'é','é界',12,None)]
    assert list(out.row_ids)==['row0']

def test_fixed_width_utf8_strict_decode_and_short_separate_from_conversion():
    c=cfg(mode='fixedWidth',measurement='utf8',fields=[field('accent',start=1,width=2),field('cjk',start=3,width=3),field('n','int64',6,2)])
    assert rows(['é界12'],c)==[(1,'é','界',12)]
    bad={**c,'fields':[field('partial',start=1,width=1)]}
    with pytest.raises(ValueError,match='conversion'):rows(['é界12'],bad)
    assert rows(['é界12'],{**bad,'conversionFailure':'missing'})==[(1,None)]
    with pytest.raises(ValueError,match='short fixed'):rows(['é'],{**c,'shortRecord':'error'})
    assert rows([None],c)==[(1,None,None,None)]
    assert rows([' 12 '],cfg(mode='fixedWidth',trim=True,fields=[field('n','int64',1,4)]))==[(1,12)]

@pytest.mark.parametrize('change',[
 dict(delimiter=''),dict(delimiter='"|'),dict(escaping='escape',escape='"'),dict(escaping='escape',delimiter='\\|'),dict(quote='ab'),dict(escape=''),dict(mode='unknown'),dict(retainSource=1),dict(extra=True),dict(fields=[]),dict(fields=[field('x'),field('x')]),dict(fields=[field('x','image')]),dict(fields=[field(start=0)]),dict(fields=[field(width=0)]),dict(fields=[field(' ')]),
])
def test_invalid_config(change):
    with pytest.raises(ValueError):st.validate_config(cfg(**change))

def test_schema_collisions_unknown_and_nontext():
    for c in [cfg(fields=[field('id')]),cfg(retainSource=True,fields=[field('text')]),cfg(outputMode='rows',ordinal=True,ordinalName='id'),cfg(source='gone'),cfg(source='id')]:
        with pytest.raises(ValueError):st.apply(table(['a']),c)

def test_cancellation_inside_long_cell_and_output_budget(monkeypatch):
    calls=[]
    def check():
        calls.append(1)
        if len(calls)>5:raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):st.apply(table(['a'*50_000]),cfg(),check)
    assert len(calls)==6
    monkeypatch.setattr(st,'MAX_CELL_BYTES',4)
    with pytest.raises(ValueError,match='string budget'):rows(['abcde'])
    monkeypatch.setattr(st,'MAX_OUTPUT_BYTES',3)
    with pytest.raises(ValueError,match='output text budget'):rows(['abcd'])

def test_over_100k_rows_and_bounded_expansion():
    c=cfg(outputMode='rows',ordinal=True)
    t=table(['x,y']*50_003)
    out=st.apply(t,c)
    assert len(out.rows)==100_006 and out.rows[-1]==(50_003,'y',2)

@pytest.mark.parametrize('tamper',['values','ids','schema','lineage','source'])
def test_cache_verifier_rejects_tamper(tamper):
    t=table(['a,b']);c=cfg(outputMode='rows');out=st.apply(t,c)
    bad=df.Table(out.columns,tuple(out.rows),tuple(out.row_ids),out.source_digest)
    if tamper=='values':bad=df.derive(bad,rows=((1,'forged'),(1,'b')))
    if tamper=='ids':bad=df.derive(bad,ids=('forged','row'))
    if tamper=='schema':bad=df.derive(bad,columns=(bad.columns[0],{**bad.columns[1],'name':'changed'}))
    if tamper=='lineage':bad=df.derive(bad,role='training')
    if tamper=='source':bad=df.derive(bad,source_digest='forged')
    with pytest.raises(ValueError,match='Cached Split'):st.verify_output(t,c,bad)


def test_practice_and_standalone_export_replay(tmp_path):
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    from runner.polars_csv_configuration import DEFAULTS
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/split-text-observations.csv'
    c=cfg(source='address',delimiter='|',fields=[field('street'),field('city'),field('postal_code')])
    document=graph(c);document['nodes'][0]['config']=copy.deepcopy(DEFAULTS)
    actual=df.execute(document,{'s':str(source)})['split']['table']
    assert list(actual.rows)==[('001','1220231250.50','123 Main St','New York','10001'),('002','3420220987.00','456 | Oak Ave','Los Angeles','90001'),('003','5620210043.20','789 Pine Rd','Chicago','60601')]
    path=tmp_path/'graph.json';path.write_text(json.dumps(document));bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'s':str(source)}))
    exported=tmp_path/'export';before=run_dataflow(path,{'s':str(source)},exported,lambda _:None)
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();after=replay(exported,bindings,tmp_path/'replay',elsewhere)
    assert before['artifacts']==after['artifacts']
    assert (exported/'dataflow_split_text.py').is_file()

@pytest.mark.parametrize('c,source,expected',[
    (cfg(source='text',outputMode='rows',ordinal=True,emptyTokens='drop'), 'id,text\n001,"a,,b"\n002,z\n', [['001','a',1],['001','b',3],['002','z',1]]),
    (cfg(source='text',mode='fixedWidth',measurement='utf8',fields=[field('accent',start=1,width=2),field('number','int64',3,2)]), 'id,text\n001,é12\n002,é34\n', [['001','é',12],['002','é',34]]),
])
def test_other_modes_standalone_export_replay(tmp_path,c,source,expected):
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    from runner.polars_csv_configuration import DEFAULTS
    source_path=tmp_path/'source.csv';source_path.write_text(source)
    document=graph(c);document['nodes'][0]['config']=copy.deepcopy(DEFAULTS)
    path=tmp_path/'graph.json';path.write_text(json.dumps(document));bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'s':str(source_path)}))
    exported=tmp_path/'export';before=run_dataflow(path,{'s':str(source_path)},exported,lambda _:None)
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();out=tmp_path/'replay';after=replay(exported,bindings,out,elsewhere)
    assert before['artifacts']==after['artifacts']
    entry=next(e for e in after['artifacts'] if e.get('nodeID')=='split' and e.get('port')=='table')
    assert json.loads((out/entry['path']).read_text())['rows']==expected


def test_working_population_and_view_complete_demand(tmp_path):
    from runner.polars_csv_configuration import DEFAULTS
    source=tmp_path/'many.csv';source.write_text('id,text\n001,"a,b"\n002,"c,d"\n003,"e,f"\n')
    document=graph(cfg(outputMode='rows'));document['nodes'][0]['config']=copy.deepcopy(DEFAULTS)
    from runner.preparation_demand import build_demand_plan
    sampled=df.execute(document,{'s':str(source)},demand_plan=build_demand_plan(document,{'mode':'rowCount','rowCount':2}))
    assert list(sampled['split']['table'].rows)==[('001','a'),('001','b'),('002','c'),('002','d')]
    document['nodes'].append({'id':'view','operator':'kwip.data.view','config':{}})
    document['edges'].append({'from':{'node':'split','port':'table'},'to':{'node':'view','port':'table'}})
    complete=df.execute(document,{'s':str(source)},demand_plan=build_demand_plan(document,{'mode':'rowCount','rowCount':2}))
    assert len(complete['split']['table'].rows)==6


def test_images_and_temporal_metadata_pass_through_without_scalar_conversion():
    temporal={'version':1,'unit':'day'}
    columns=(dict(name='text',dtype='string',nullable=False),dict(name='date',dtype='date',nullable=False,temporal=temporal))
    t=df.Table(columns,(('value',0),),('id',),'source')
    out=st.apply(t,cfg())
    assert out.columns[0]==columns[1] and list(out.rows)==[(0,'value')]
    with pytest.raises(ValueError,match='text column'):st.apply(t,cfg(source='date'))
