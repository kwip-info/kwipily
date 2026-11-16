"""Columns is an additive, exact, streaming selection/rename/cast operator."""
import copy
import json
import subprocess
import sys
import pytest
from runner import dataflow as df
from runner.polars_preparation import preparation_engine

DEFAULT={'rows':[], 'castFailure':'error','trimText':False,'newColumns':'keep','missingPolicy':'preserve'}
def config(rows=(),**kw):return {**DEFAULT,'rows':list(rows),**kw}
def item(source,output=None,dtype='keep',included=True):return {'source':source,'included':included,'outputName':source if output is None else output,'targetType':dtype}
def table():return df.Table(({'name':'a','dtype':'string','nullable':True},{'name':'b','dtype':'string','nullable':False},{'name':'drop','dtype':'string','nullable':False}), ((' 12 ','true','unused'),(None,'false','unused')),('first','second'),'source',role='training',partition_digest='partition',fitted_states=('fit',))
def apply(t,c):return df._execute_node('columns',c,{'table':t},'graph',None,lambda:None)['table']
def graph(c):return {'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{}},{'id':'c','operator':'kwip.data.columns','config':c}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}}]}

def test_columns_is_registered_and_accepts_exact_configuration():
    df.validate(graph(config()))
    assert df.DEFINITIONS['columns']['defaultConfig']==DEFAULT

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('stored',[False,True])
def test_order_selection_simultaneous_names_casts_and_lineage(engine,stored):
    with preparation_engine(engine):
        source=table()
        if stored:source=df.streaming_rows(source,df.row_pairs(source))
        out=apply(source,config([item('b','a','bool'),item('a','b','int64'),item('drop',included=False)],trimText=True))
        assert [c['name'] for c in out.columns]==['a','b']
        assert list(out.rows)==[(True,12),(False,None)]
        assert [c['dtype'] for c in out.columns]==['bool','int64']
        assert [c['nullable'] for c in out.columns]==[False,True]
        assert list(out.row_ids)==list(source.row_ids)
        assert out.evidence()==source.evidence()
        assert (out.role,out.partition_digest,out.fitted_states)==('training','partition',('fit',))

@pytest.mark.parametrize('policy,expected',[('keep',['b','a','drop']),('drop',['b'])])
def test_unconfigured_columns_policy(policy,expected):
    out=apply(table(),config([item('b')],newColumns=policy))
    assert [c['name'] for c in out.columns]==expected

def test_unchanged_columns_preserve_schema_and_text():
    out=apply(table(),config([item('a')],trimText=True))
    assert out.columns==table().columns
    assert list(out.rows)==list(table().rows)

@pytest.mark.parametrize('c,message',[
    (config([item('gone',included=False)]),'Unknown column'),
    (config([item('a','b')]),'duplicate'),
    (config([item('a')],newColumns='error'),'Unconfigured'),
    (config([item('a',included=False),item('b',included=False),item('drop',included=False)]),'at least one'),
    (config([item('a',dtype='int64')]),'row 1'),
    (config([item('a',dtype='int64')],trimText=True,missingPolicy='error'),'missing'),
])
def test_runtime_rejects_stale_schema_collisions_and_unsafe_casts(c,message):
    with pytest.raises(ValueError,match=message):apply(table(),c)

def test_invalid_cast_can_be_explicit_null_but_null_input_policy_still_applies():
    c=config([item('a',dtype='int64')],castFailure='null')
    out=apply(table(),c)
    assert list(out.rows)==[(None,'true','unused'),(None,'false','unused')]
    assert out.columns[0]['nullable'] is True
    with pytest.raises(ValueError,match='missing'):apply(table(),{**c,'missingPolicy':'error'})

@pytest.mark.parametrize('mutate',[
    lambda c:c.update(extra=True),lambda c:c.pop('trimText'),lambda c:c.update(trimText=1),
    lambda c:c.update(rows=[item('a'),item('a')]),lambda c:c.update(rows=[item('a',' ')]),
    lambda c:c.update(rows=[item('a','x'),item('b','x')]),lambda c:c.update(rows=[{**item('a'),'extra':1}]),
    lambda c:c.update(rows=[item('a',dtype='unknown')]),lambda c:c.update(rows=[item('a')]*129),
])
def test_invalid_config_rejected_before_source_read(mutate,tmp_path):
    c=config();mutate(c)
    with pytest.raises(ValueError):df.execute(graph(c),{'s':str(tmp_path/'absent.csv')})

def test_image_keep_is_preserved_and_scalar_cast_forbidden_even_null_policy():
    t=df.Table(({'name':'image','dtype':'image','nullable':False},), (({'kind':'image'},),), ('id',),'source')
    assert list(apply(t,config()).rows)==list(t.rows)
    with pytest.raises(ValueError,match='Image'):apply(t,config([item('image',dtype='string')],castFailure='null'))

@pytest.mark.parametrize('dtype,value,expected',[
 ('float64','2.5',2.5),('int64','-12',-12),('bool','false',False),('string',12,'12'),
])
def test_every_cast_type(dtype,value,expected):
    t=df.Table(({'name':'v','dtype':'string' if isinstance(value,str) else 'int64','nullable':False},),((value,),),('id',),'source')
    out=apply(t,config([item('v',dtype=dtype)]))
    assert list(out.rows)==[(expected,)] and out.columns[0]['dtype']==dtype

@pytest.mark.parametrize('engine',['disk','polars'])
def test_over_eager_limit_rows_stream_without_materializing_source(engine,tmp_path,monkeypatch):
    source=tmp_path/'many.csv';source.write_text('a,b,drop\n'+'12,true,unused\n'*100_005)
    c=config([item('a','number','int64'),item('drop',included=False)],newColumns='keep')
    with preparation_engine(engine):
        result=df.execute(graph(c),{'s':str(source)},engine=engine)
    out=result['c']['table'];before=result['s']['table']
    assert isinstance(out.rows,df.StoredRows)
    assert len(out.rows)==100_005 and out.rows[100_004]==(12,'true')
    assert [column['name'] for column in out.columns]==['number','b']
    assert out.row_ids[100_004]==before.row_ids[100_004]


def test_cancellation_is_observed_during_columns_stream():
    t=table();calls=[]
    def cancelled():calls.append(True);raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        df._execute_node('columns',config(),{'table':t},'graph',None,cancelled)
    assert calls


def test_columns_export_replays_outside_repo_with_exact_artifact_hash(tmp_path):
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    source=tmp_path/'source.csv';source.write_text('a,b,drop\n 12 ,true,x\n9,false,y\n')
    c=config([item('b','flag','bool'),item('a','value','int64'),item('drop',included=False)],trimText=True)
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph(c)))
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'s':str(source)}))
    export=tmp_path/'export';original=run_dataflow(path,{'s':str(source)},export,lambda _:None)
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    out=tmp_path/'replay';reproduced=replay(export,bindings,out,elsewhere)
    assert original['artifacts']==reproduced['artifacts']
    assert (export/'dataflow_columns.py').is_file()
    entry=next(e for e in reproduced['artifacts'] if e.get('nodeID')=='c' and e.get('port')=='table')
    actual=json.loads((out/entry['path']).read_text())
    assert actual['rows']==[[True,12],[False,9]]


def test_shipped_columns_practice_fixture_real_polars_replay(tmp_path):
    """Independent authored counterpart to ColumnsPracticeExample.configuration.

    Uses the shipped native six-row CSV, never a synthetic replacement fixture.
    The CSV profile defaults mirror CSVParsingConfiguration.newDefaults.
    """
    from pathlib import Path
    from runner.polars_csv_configuration import DEFAULTS as csv_defaults
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/columns-observations.csv'
    columns=config([item('score_text','score','float64'),item('person_id'),
                    item('active_text','active','bool'),item('notes',included=False)],newColumns='drop')
    document=graph(columns)
    document['nodes'][0]['config']=copy.deepcopy(csv_defaults)
    document['nodes'].append({'id':'view','operator':'kwip.data.view','config':{}})
    document['edges'].append({'from':{'node':'c','port':'table'},'to':{'node':'view','port':'table'}})
    actual=df.execute(document,{'s':str(source)})
    for node in ('c','view'):
        output=actual[node]['table']
        assert [(c['name'],c['dtype']) for c in output.columns]==[('score','float64'),('person_id','string'),('active','bool')]
        assert len(output.rows)==6 and output.rows[0]==(12.5,'001',True)
        assert list(output.rows)==[(12.5,'001',True),(7.25,'002',False),(18.0,'003',True),(0.0,'004',False),(9.5,'005',True),(21.75,'006',False)]
        assert list(output.row_ids)==list(actual['s']['table'].row_ids)
    graph_path=tmp_path/'practice.json';graph_path.write_text(json.dumps(document))
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'s':str(source)}))
    exported=tmp_path/'exported';manifest=run_dataflow(graph_path,{'s':str(source)},exported,lambda _:None)
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    reproduced=replay(exported,bindings,tmp_path/'replayed',elsewhere)
    assert manifest['artifacts']==reproduced['artifacts']

@pytest.mark.parametrize('engine',['disk','polars'])
def test_explicit_image_type_is_identity_only_retaining_nulls_and_metadata(engine):
    from runner.tests.test_image_tables import table as image_table
    source,_,reference=image_table()
    c=config([item('image','pixels','image'),item('key','record_id','keep')],trimText=True,missingPolicy='error',castFailure='null')
    df.validate(graph(c))
    with preparation_engine(engine):out=apply(source,c)
    assert out.columns==({'name':'pixels','dtype':'image','nullable':True},{'name':'record_id','dtype':'int64','nullable':False})
    assert list(out.rows)==[(reference,1),(None,2)]
    assert list(out.row_ids)==list(source.row_ids)

@pytest.mark.parametrize('failure',['error','null'])
def test_scalar_to_image_refused_even_with_null_cast_failure(failure):
    with pytest.raises(ValueError,match='[Ii]mage'):
        apply(table(),config([item('a',dtype='image')],castFailure=failure))


def test_explicit_image_column_preserves_real_pixel_inventory_and_external_replay(tmp_path):
    from PIL import Image
    from runner.file_input_options import defaults as file_defaults
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    sources=tmp_path/'images';sources.mkdir()
    Image.new('RGB',(2,1),(7,11,13)).save(sources/'sample.png')
    cfg=file_defaults();cfg['sourceMode']='directory';cfg['extensions']=['png']
    columns=config([item('image_content','pixels','image'),item('name','filename')],newColumns='drop',missingPolicy='error')
    document=graph(columns)
    document['nodes'][0].update(operator='kwip.data.sourceFiles',config=cfg)
    graph_path=tmp_path/'graph.json';graph_path.write_text(json.dumps(document))
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'s':str(sources)}))
    export=tmp_path/'export';original=run_dataflow(graph_path,{'s':str(sources)},export,lambda _:None)
    images=[entry for entry in original['artifacts'] if entry['kind']=='imagePixels']
    assert len(images)==1
    artifact=next(entry for entry in original['artifacts'] if entry.get('nodeID')=='c' and entry.get('port')=='table')
    captured=json.loads((export/artifact['path']).read_text())
    assert [(c['name'],c['dtype']) for c in captured['columns']]==[('pixels','image'),('filename','string')]
    assert captured['rows'][0]==[images[0]['image'],'sample.png']
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    repeated=replay(export,bindings,tmp_path/'repeated',elsewhere)
    assert repeated['artifacts']==original['artifacts']


def test_file_reader_helpers_and_library_versions_participate_in_cache_identity(monkeypatch):
    import importlib.metadata
    from runner.dataflow_cache import implementation_digest
    before=implementation_digest()
    original=importlib.metadata.version
    for package in ('Pillow','pypdf','python-docx','striprtf'):
        def changed(name,selected=package):
            return 'changed-test-version' if name==selected else original(name)
        monkeypatch.setattr(importlib.metadata,'version',changed)
        assert implementation_digest()!=before,package
        monkeypatch.setattr(importlib.metadata,'version',original)
