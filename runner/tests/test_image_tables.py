import hashlib
import json
import pytest
from runner import dataflow as df
from runner import image_artifacts as ia
from runner.dataflow_join import join_tables
from runner.polars_preparation import preparation_engine
from runner.table_view import build_index,query_index


def table():
    content,ref=ia.encode_pixels(bytes(6),width=2,height=1,channels=3,dtype='uint8',layout='HWC',policy={'orientation':'stored','colorMode':'preserve','palette':'expandRGBA','actualMode':'RGB'})
    columns=({'name':'key','dtype':'int64','nullable':False},{'name':'image','dtype':'image','nullable':True})
    return df.derive(df.Table(columns,((1,ref),(2,None)),('a:0','a:1'),'a'*64)),content,ref



def columns(rows):
    return {'rows':[dict(source=source,included=True,outputName=name,targetType=dtype)for source,name,dtype in rows],
            'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}


def filter_rows(column,value):
    return {'mode':'simple','conditions':[dict(column=column,comparison='eq',value=value,caseSensitive=True)],
            'match':'all','expression':'','column':column,'pattern':'.+','matchMode':'search','flags':[],'nullPolicy':'error'}

@pytest.mark.parametrize('engine',['disk','polars'])
def test_row_preparation_preserves_structured_reference_and_row_identity(engine):
    source,_,ref=table()
    with preparation_engine(engine):
        selected=df._execute_node('columns',columns([('image','pixels','image'),('key','key','keep')]),{'table':source},'',None,lambda:None)['table']
        assert selected.rows[0][0]==ref and selected.columns[0]['dtype']=='image' and tuple(selected.row_ids)==tuple(source.row_ids)
        result=df._execute_node('filterRows',filter_rows('key',1),{'table':selected},'',None,lambda:None)['matching']
        assert result.rows[0][0]==ref and tuple(result.row_ids)==('a:0',)
        store=df.row_store(source.columns,lambda:None)
        for row,rid in zip(source.rows,source.row_ids):store.append(row,rid)
        try:
            rows,ids=store.finish();assert rows[0][1]==ref and rows[1][1] is None
        finally:store.close()


def test_nonkey_join_preserves_image_and_image_keys_refuse_before_hashing():
    left,_,ref=table();right=df.derive(df.Table(({'name':'key','dtype':'int64','nullable':False},),((1,),),('b:0',),'b'*64))
    config={'leftKeys':['key'],'rightKeys':['key'],'mode':'inner','nullPolicy':'neverMatch','cardinality':'manyToMany','leftPrefix':'left_','rightPrefix':'right_','maxRows':100}
    result=join_tables(left,right,config)
    assert result.rows[0][result.index('image')]==ref
    with pytest.raises(ValueError,match='Image'):join_tables(left,left,{**config,'leftKeys':['image'],'rightKeys':['image']})


@pytest.mark.parametrize('operation,config',[
    ('columns',columns([('image','image','string')])),
    ('filterRows',filter_rows('image','anything')),
    ('formula',{'operations':[dict(id='image-to-scalar',expression='[image]',outputs=[dict(name='converted',dtype='string',write='new')],nullPolicy='error')]}),
    ('partition',{'strategy':'random','seed':1,'validationPercent':25,'testPercent':25,'stratifyColumn':'image'}),
])
def test_no_implicit_scalar_conversion_or_image_key(operation,config):
    source,_,ref=table()
    with pytest.raises(ValueError,match='[Ii]mage'):df._execute_node(operation,config,{'table':source},'',None,lambda:None)
    with pytest.raises(ValueError,match='[Ii]mage'):df.cast_value(ref,'string')


def test_view_preserves_typed_cells_searches_display_metadata_and_refuses_image_sort(tmp_path):
    source,_,ref=table();path=tmp_path/'view.sqlite3';pin='a'*64
    receipt=build_index(source,path,source_sha256=pin)
    page=query_index(path,receipt['index']['sha256'],source_sha256=pin)
    assert page['rows'][0][1]==ref and page['rows'][1][1] is None
    assert query_index(path,receipt['index']['sha256'],source_sha256=pin,search='uint8')['matchedRows']==1
    with pytest.raises(ValueError,match='[Ii]mage'):query_index(path,receipt['index']['sha256'],source_sha256=pin,sort_column=1)


def test_image_tables_use_new_artifact_version_and_frozen_reader_rejects_old_version(tmp_path):
    from runner.dataflow_table_reader import read_table_artifact
    source,_,ref=table()
    value={'schemaVersion':2,'kind':'table','columns':source.columns,'rows':source.rows,'lineage':{**source.evidence(),'dataVersion':source.data_version,'partitionDigest':'','fittedStateDigests':[],'diagnostics':[]}}
    path=tmp_path/'table.json';path.write_text(json.dumps(value,sort_keys=True,separators=(',',':')))
    loaded=read_table_artifact(path,hashlib.sha256(path.read_bytes()).hexdigest())
    try:assert loaded.rows[0][1]==ref
    finally:loaded.rows.store.close();loaded.row_ids.store.close()
    value['schemaVersion']=1;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='[Vv]ersion'):read_table_artifact(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_cli_owns_pixel_inventory_and_frozen_history_rejects_missing_or_changed_assets(tmp_path,monkeypatch):
    from runner.dataflow_cli import run_dataflow,inspect_dataflow
    from runner.dataflow_storage import completed_inventory
    source,content,ref=table();original=df._execute_node
    def produce(op,*args,**kwargs):
        if op!='sourceCSV':return original(op,*args,**kwargs)
        ia.publish_encoded(content,ref)
        return {'table':source}
    monkeypatch.setattr(df,'_execute_node',produce)
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}],'edges':[]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));output=tmp_path/'prepared'
    manifest=run_dataflow(path,{'source':'explicit-test-binding'},output,lambda _:None,cache_policy='materialize')
    asset=next(entry for entry in manifest['artifacts'] if entry['kind']=='imagePixels')
    assert asset['image']==ref and asset['path']=='image-'+ref['sha256']+'.kwipimg'
    selected=next(entry for entry in manifest['artifacts'] if entry['kind']=='table')
    assert json.loads((output/selected['path']).read_bytes())['schemaVersion']==2
    completed_inventory(output/'manifest.json',lazy=True)
    events=[];inspect_dataflow(output/'manifest.json',hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),events.append)
    preview=next(event for event in events if event['event']=='dataflowPreview')
    assert preview['outputs']['table']['preview']['rows'][0][1]==ref
    manifest['artifacts'].remove(asset);(output/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='missing.*inventory'):completed_inventory(output/'manifest.json',lazy=True)
    manifest['artifacts'].append(asset);(output/'manifest.json').write_text(json.dumps(manifest))
    (output/asset['path']).write_bytes(content[:-1]+b'1')
    with pytest.raises(ValueError,match='mismatch|differs'):completed_inventory(output/'manifest.json',lazy=True)


def test_portable_reader_dependencies_follow_selected_file_formats_without_importing_readers():
    from runner.table_source_options import required_reader_packages
    def packages(**config):return required_reader_packages({'nodes':[{'operator':'kwip.data.sourceFiles','config':config}]})
    base={'polars','polars-runtime-32'}
    assert packages(extensions=['txt'])==base
    assert packages(extensions=['jpg'])==base|{'Pillow'}
    assert packages(extensions=['pdf','docx'])==base|{'pypdf','python-docx','lxml','typing-extensions'}
    assert packages(readMode='metadata')==base
