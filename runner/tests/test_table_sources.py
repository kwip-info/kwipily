import json
from pathlib import Path
import pytest
import polars as pl
from runner.table_sources import source_file,preview_file,normalize_options


def test_json_typed_values_nulls_ids_and_preview(tmp_path):
    path=tmp_path/'rows.json';path.write_text('[{"name":"A","value":2,"flag":true},{"name":null,"value":3,"flag":false}]')
    config={'format':'json','options':{}}
    table=source_file(path,config)
    try:
        assert table.rows[:]==(('A',2,True),(None,3,False))
        assert [c['dtype'] for c in table.columns]==['string','int64','bool']
        assert table.row_ids[0]==table.source_digest+':0'
        preview=preview_file(path,config)
        assert preview['rows']==[['A',2,True],[None,3,False]]
    finally:table.rows.store.close()


def test_parquet_projection_and_native_types(tmp_path):
    path=tmp_path/'rows.parquet';pl.DataFrame({'a':[1,2],'b':['x','y']}).write_parquet(path)
    table=source_file(path,{'format':'parquet','options':{'columns':['b'],'n_rows':1}})
    try:assert table.rows[:]==(('x',),) and table.columns[0]['name']=='b'
    finally:table.rows.store.close()


def test_unknown_or_remote_options_refused_before_source_read(tmp_path):
    for config in [{'format':'json','options':{'evil':True}},{'format':'parquet','options':{'storage_options':{'key':'secret'}}}]:
        with pytest.raises(ValueError):source_file(tmp_path/'missing',config)

@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_excel_real_workbook_each_public_engine(tmp_path,engine):
    from openpyxl import Workbook
    path=tmp_path/'data.xlsx';workbook=Workbook();sheet=workbook.active;sheet.title='Selected';sheet.append(['name','amount']);sheet.append(['Alpha',2]);sheet.append(['Beta',3]);workbook.save(path)
    table=source_file(path,{'format':'excel','options':{'sheet_name':'Selected','engine':engine}})
    try:assert list(table.rows)==[('Alpha',2),('Beta',3)]
    finally:table.rows.store.close()


def test_jsonl_schema_override_prefix_and_stable_full_hash(tmp_path):
    path=tmp_path/'data.jsonl';path.write_text(''.join(json.dumps({'x':i,'text':str(i)})+'\n' for i in range(30)))
    import hashlib
    config={'format':'jsonl','options':{'schema_overrides':{'x':'Float64'},'row_index_name':'row','row_index_offset':10,'n_rows':3}}
    table=source_file(path,config)
    try:
        assert list(table.rows)==[(10,0.0,'0'),(11,1.0,'1'),(12,2.0,'2')]
        assert table.source_digest==hashlib.sha256(path.read_bytes()).hexdigest()
    finally:table.rows.store.close()


def test_markdown_real_tables_escapes_links_and_explicit_types(tmp_path):
    path=tmp_path/'table.md';path.write_text('| Name | Value |\n|---|---|\n| [Alpha](https://example.com) | 2 |\n| escaped \\| pipe | 3 |\n')
    table=source_file(path,{'format':'markdown','options':{'link_mode':'url','schema_overrides':{'Value':'Int64'}}})
    try:assert list(table.rows)==[('https://example.com',2),('escaped | pipe',3)]
    finally:table.rows.store.close()


def test_xml_fields_namespaces_null_and_no_implicit_flatten(tmp_path):
    path=tmp_path/'rows.xml';path.write_text('<root xmlns:n="urn:test"><row id="a"><n:value>2</n:value></row><row id="b"/></root>')
    table=source_file(path,{'format':'xml','options':{'fields':{'id':'@id','value':'n:value/text()'},'namespaces':{'n':'urn:test'},'schema_overrides':{'value':'Int64'}}})
    try:assert list(table.rows)==[('a',2),('b',None)]
    finally:table.rows.store.close()
    path.write_text('<root><row><nested><value>1</value></nested></row></root>')
    with pytest.raises(ValueError,match='Nested'):source_file(path,{'format':'xml','options':{}})


def test_html_hidden_rows_links_and_span_policy(tmp_path):
    path=tmp_path/'table.html';path.write_text('<table><tr><th>Name</th><th>Value</th></tr><tr hidden><td>hidden</td><td>9</td></tr><tr><td><a href="https://example.com">Alpha</a></td><td>2</td></tr></table>')
    table=source_file(path,{'format':'html','options':{'link_mode':'both','schema_overrides':{'Value':'Int64'}}})
    try:assert list(table.rows)==[('Alpha (https://example.com)',2)]
    finally:table.rows.store.close()
    path.write_text('<table><tr><th>a</th><th>b</th></tr><tr><td colspan="2">x</td></tr></table>')
    with pytest.raises(ValueError,match='spans'):source_file(path,{'format':'html','options':{}})
    table=source_file(path,{'format':'html','options':{'span_policy':'expand'}})
    try:assert list(table.rows)==[('x','x')]
    finally:table.rows.store.close()


def test_pyarrow_parquet_engine_bounded_batch_path(tmp_path):
    path=tmp_path/'table.parquet';pl.DataFrame({'x':[1,2,3],'v':['a','b','c']}).write_parquet(path)
    table=source_file(path,{'format':'parquet','options':{'use_pyarrow':True,'n_rows':2,'columns':['v']}})
    try:assert list(table.rows)==[('a',),('b',)]
    finally:table.rows.store.close()


def test_internal_database_snapshot_ipc_helpers(tmp_path):
    from runner.table_sources import source_typed_snapshot,preview_typed_snapshot
    path=tmp_path/'snapshot.arrow';pl.DataFrame({'x':[1,2],'y':['a',None]}).write_ipc(path)
    table=source_typed_snapshot(path)
    try:assert list(table.rows)==[(1,'a'),(2,None)]
    finally:table.rows.store.close()
    assert preview_typed_snapshot(path)[:2]==(['x','y'],[[1,'a'],[2,None]])
    with pytest.raises(ValueError):source_file(path,{'format':'ipc','options':{}})


def test_source_changes_cancellation_symlink_and_nested_refuse(tmp_path):
    path=tmp_path/'data.json';path.write_text('[{"x":1}]');link=tmp_path/'link.json';link.symlink_to(path)
    with pytest.raises(ValueError,match='symlink'):source_file(link,{'format':'json','options':{}})
    path.write_text('[{"x":[1,2]}]')
    with pytest.raises(ValueError,match='scalar'):source_file(path,{'format':'json','options':{}})
    path.write_text('[{"x":1}]');calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==5:path.write_text('[{"x":2}]')
    with pytest.raises(ValueError,match='changed'):source_file(path,{'format':'json','options':{}},check)

@pytest.mark.parametrize('fmt',['parquet','json','jsonl','excel','markdown','xml','html'])
def test_real_cli_portable_replay_full_exports_and_pinned_dependencies(tmp_path,fmt):
    import subprocess,sys
    from runner.dataflow_cli import run_dataflow
    source=tmp_path/('source.'+('xlsx' if fmt=='excel' else fmt));options={}
    if fmt=='parquet':pl.DataFrame({'name':['A','B'],'value':[2,3]}).write_parquet(source)
    elif fmt=='json':source.write_text('[{"name":"A","value":2},{"name":"B","value":3}]')
    elif fmt=='jsonl':source.write_text('{"name":"A","value":2}\n{"name":"B","value":3}\n')
    elif fmt=='excel':
        from openpyxl import Workbook
        workbook=Workbook();sheet=workbook.active;sheet.append(['name','value']);sheet.append(['A',2]);sheet.append(['B',3]);workbook.save(source)
    elif fmt=='markdown':source.write_text('| name | value |\n|---|---|\n| A | 2 |\n| B | 3 |\n');options={'schema_overrides':{'value':'Int64'}}
    elif fmt=='xml':source.write_text('<root><row><name>A</name><value>2</value></row><row><name>B</name><value>3</value></row></root>');options={'schema_overrides':{'value':'Int64'}}
    else:source.write_text('<table><tr><th>name</th><th>value</th></tr><tr><td>A</td><td>2</td></tr><tr><td>B</td><td>3</td></tr></table>');options={'schema_overrides':{'value':'Int64'}}
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{'format':fmt,'options':options}},{'id':'export','operator':'kwip.data.outputDelimited','config':{'filename':'table.csv','separator':',','overwrite':False,'columns':['name','value']}}], 'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'export','port':'table'}}]}
    graph_path=tmp_path/'graph.json';graph_path.write_text(json.dumps(graph));destination=tmp_path/'first-export';destination.mkdir();capture=tmp_path/'capture'
    first=run_dataflow(graph_path,{'source':str(source)},capture,lambda _:None,destination_bindings={'export':str(destination)})
    expected=(destination/'table.csv').read_bytes();assert expected==b'name,value\nA,2\nB,3\n'
    dependencies=(capture/'requirements-preparation.txt').read_text()
    assert 'polars==' in dependencies
    if fmt=='excel':assert 'fastexcel==' in dependencies
    if fmt=='parquet':assert 'pyarrow==' in dependencies
    if fmt=='markdown':assert 'markdown-it-py==' in dependencies
    if fmt in ('xml','html'):assert 'lxml==' in dependencies
    other=tmp_path/'second-export';other.mkdir();bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}));outputs=tmp_path/'outputs.json';outputs.write_text(json.dumps({'export':str(other)}))
    subprocess.run([sys.executable,str(capture/'prep.py'),'--source-bindings',str(bindings),'--destination-bindings',str(outputs),'--output',str(tmp_path/'replayed')],cwd=tmp_path,check=True,capture_output=True,text=True)
    assert (other/'table.csv').read_bytes()==expected
    replay=json.loads((tmp_path/'replayed/manifest.json').read_text())
    assert [a['sha256'] for a in first['artifacts'] if a.get('kind')=='table']==[a['sha256'] for a in replay['artifacts'] if a.get('kind')=='table']


def test_excel_named_table_and_public_whitespace_controls(tmp_path):
    from openpyxl import Workbook
    from openpyxl.worksheet.table import Table
    path=tmp_path/'table.xlsx';workbook=Workbook();sheet=workbook.active;sheet.append(['Name','Value']);sheet.append(['A',2]);sheet.append(['B',3]);sheet.add_table(Table(displayName='Values',ref='A1:B3'));workbook.save(path)
    table=source_file(path,{'format':'excel','options':{'table_name':'Values'}})
    try:assert [c['name'] for c in table.columns]==['Name','Value'] and list(table.rows)==[('A',2),('B',3)]
    finally:table.rows.store.close()
    sheet['A2']='  ';workbook.save(path)
    table=source_file(path,{'format':'excel','options':{'read_options':{'whitespace_as_null':True,'skip_whitespace_tail_rows':True}}})
    try:assert list(table.rows)==[(None,2),('B',3)]
    finally:table.rows.store.close()


def test_xml_entities_refused_in_utf8_and_utf16_without_external_read(tmp_path):
    source='<!DOCTYPE root [<!ENTITY secret SYSTEM "file:///private/not-a-source">]><root><row><x>&secret;</x></row></root>'
    for encoding in ('utf-8','utf-16'):
        path=tmp_path/(encoding+'.xml');path.write_bytes(source.encode(encoding))
        with pytest.raises(ValueError,match='DTDs'):source_file(path,{'format':'xml','options':{'encoding':encoding}})
    path=tmp_path/'literal.xml';path.write_text('<root><row><x><![CDATA[<!DOCTYPE example>]]></x></row></root>')
    table=source_file(path,{'format':'xml','options':{}})
    try:assert list(table.rows)==[('<!DOCTYPE example>',)]
    finally:table.rows.store.close()


def test_json_schema_inference_override_and_invalid_strings_are_explicit(tmp_path):
    path=tmp_path/'table.json';path.write_text('[{"value":1},{"value":1.5}]')
    table=source_file(path,{'format':'json','options':{'infer_schema_length':None}})
    try:assert list(table.rows)==[(1.0,),(1.5,)] and table.columns[0]['dtype']=='float64'
    finally:table.rows.store.close()
    path=tmp_path/'table.jsonl';path.write_text('{"value":1}\n{"value":"bad"}\n')
    with pytest.raises(ValueError,match='parse'):source_file(path,{'format':'jsonl','options':{'schema':{'value':'Int64'}}})
    table=source_file(path,{'format':'jsonl','options':{'schema':{'value':'Int64'},'ignore_errors':True}})
    try:assert list(table.rows)==[(1,),(None,)]
    finally:table.rows.store.close()


def test_reader_schema_and_xml_field_maps_have_canonical_order(tmp_path):
    xml=tmp_path/'table.xml';xml.write_text('<root><row><x>1</x><y>2</y></row></root>')
    a=source_file(xml,{'format':'xml','options':{'fields':{'z':'x/text()','a':'y/text()'}}})
    b=source_file(xml,{'format':'xml','options':{'fields':{'a':'y/text()','z':'x/text()'}}})
    try:assert a.data_version==b.data_version and [c['name'] for c in a.columns]==['a','z']
    finally:a.rows.store.close();b.rows.store.close()


def test_count_sampling_is_pushed_before_columnar_storage_but_preserves_full_source_digest(tmp_path,monkeypatch):
    import hashlib
    from runner import table_sources
    path=tmp_path/'table.jsonl';path.write_text(''.join('{"x":'+str(i)+'}\n' for i in range(10000)))
    original=table_sources.ColumnarStore.append;seen=[]
    def append(self,row,identity):seen.append(identity);return original(self,row,identity)
    monkeypatch.setattr(table_sources.ColumnarStore,'append',append)
    table=source_file(path,{'format':'jsonl','options':{}},sampling={'schemaVersion':1,'mode':'rowCount','rowCount':5})
    try:assert len(seen)==5 and len(table.rows)==5 and table.source_digest==hashlib.sha256(path.read_bytes()).hexdigest()
    finally:table.rows.store.close()

@pytest.mark.parametrize('selection',[{'columns':[1]},{'read_options':{'use_columns':'B'}}])
def test_named_excel_table_column_indexes_and_ranges_are_not_dropped(tmp_path,selection):
    from openpyxl import Workbook
    from openpyxl.worksheet.table import Table
    path=tmp_path/'table.xlsx';workbook=Workbook();sheet=workbook.active;sheet.append(['Name','Value']);sheet.append(['A',2]);sheet.append(['B',3]);sheet.add_table(Table(displayName='Values',ref='A1:B3'));workbook.save(path)
    table=source_file(path,{'format':'excel','options':{'table_name':'Values',**selection}})
    try:assert [c['name'] for c in table.columns]==['Value'] and list(table.rows)==[(2,),(3,)]
    finally:table.rows.store.close()


def test_parquet_explicit_missing_columns_are_inserted_with_declared_type(tmp_path):
    path=tmp_path/'table.parquet';pl.DataFrame({'a':[1,2]}).write_parquet(path)
    table=source_file(path,{'format':'parquet','options':{'schema':{'a':'Int64','missing':'String'},'columns':['missing'],'missing_columns':'insert'}})
    try:assert list(table.rows)==[(None,),(None,)] and table.columns[0]=={'name':'missing','dtype':'string','nullable':True}
    finally:table.rows.store.close()

@pytest.mark.parametrize('fmt,content',[('xml','<root><row><flag>true</flag></row><row><flag>false</flag></row></root>'),('markdown','| flag |\n|---|\n| true |\n| false |\n'),('html','<table><tr><th>flag</th></tr><tr><td>true</td></tr><tr><td>false</td></tr></table>')])
def test_markup_explicit_boolean_reader_type_is_supported(tmp_path,fmt,content):
    path=tmp_path/('table.'+fmt);path.write_text(content)
    table=source_file(path,{'format':fmt,'options':{'schema_overrides':{'flag':'Boolean'}}})
    try:assert list(table.rows)==[(True,),(False,)] and table.columns[0]['dtype']=='bool'
    finally:table.rows.store.close()


def test_markup_boolean_declarations_preserve_null_and_refuse_unrecognized_text(tmp_path):
    path=tmp_path/'table.xml';path.write_text('<root><row><flag>true</flag></row><row/></root>')
    config={'format':'xml','options':{'schema_overrides':{'flag':'Boolean'}}}
    table=source_file(path,config)
    try:assert list(table.rows)==[(True,),(None,)] and table.columns[0]['nullable']
    finally:table.rows.store.close()
    path.write_text('<root><row><flag>TRUE</flag></row></root>')
    with pytest.raises(ValueError,match='parse'):source_file(path,config)


def test_pyarrow_memory_map_is_one_top_level_control_only():
    from runner.table_source_options import normalize_options
    with pytest.raises(ValueError,match='Unknown or unsafe'):
        normalize_options({'format':'parquet','options':{'use_pyarrow':True,'pyarrow_options':{'memory_map':True}}})


@pytest.mark.parametrize('options',[
    {'read_options':{'use_columns':[{}]}},
    {'read_options':{'dtypes':{'':'int'}}},
    {'engine':'xlsx2csv','read_options':{'null_values':[{}]}},
    {'engine':'xlsx2csv','read_options':{'quote_char':'é'}},
    {'engine':'xlsx2csv','read_options':{'new_columns':['x','x']}},
])
def test_nested_excel_options_refuse_invalid_content_before_file_access(tmp_path,options):
    with pytest.raises(ValueError):source_file(tmp_path/'does-not-exist.xlsx',{'format':'excel','options':options})


def test_parquet_explicit_extra_column_policy_is_honored(tmp_path):
    path=tmp_path/'table.parquet';pl.DataFrame({'a':[1],'extra':[2]}).write_parquet(path)
    base={'schema':{'a':'Int64'}}
    with pytest.raises(ValueError):source_file(path,{'format':'parquet','options':base})
    table=source_file(path,{'format':'parquet','options':{**base,'extra_columns':'ignore'}})
    try:assert list(table.rows)==[(1,)] and [c['name'] for c in table.columns]==['a']
    finally:table.rows.store.close()


@pytest.mark.parametrize('source_type,target_type,policy,value',[
    (pl.Int32,'Int64',{'integer_cast':'upcast'},7),
    (pl.Int32,'Float64',{'integer_cast':'allow-float'},7.0),
    (pl.Float32,'Float64',{'float_cast':'upcast'},1.25),
    (pl.Float64,'Float32',{'float_cast':'downcast'},1.25),
    (pl.Float32,'Float64',{'float_cast':['downcast','upcast']},1.25),
])
def test_parquet_explicit_scalar_cast_policies_apply_at_reader_boundary(tmp_path,source_type,target_type,policy,value):
    path=tmp_path/'table.parquet';pl.DataFrame({'x':pl.Series([int(value) if source_type.is_integer() else value],dtype=source_type)}).write_parquet(path)
    base={'schema':{'x':target_type}}
    with pytest.raises(ValueError):source_file(path,{'format':'parquet','options':base})
    table=source_file(path,{'format':'parquet','options':{**base,'cast_options':policy}})
    try:assert list(table.rows)==[(value,)]
    finally:table.rows.store.close()


def test_parquet_scalar_cast_configuration_is_strict_and_native_only():
    from runner.table_source_options import normalize_options
    invalid=[{'float_cast':['forbid','upcast']},{'float_cast':['upcast','upcast']},{'integer_cast':'downcast'},{'datetime_cast':'upcast'},{'extra_struct_fields':'ignore'}]
    for cast in invalid:
        with pytest.raises(ValueError):normalize_options({'format':'parquet','options':{'cast_options':cast}})
    with pytest.raises(ValueError):normalize_options({'format':'parquet','options':{'use_pyarrow':True,'cast_options':{'integer_cast':'upcast'}}})


def test_parquet_explicit_categorical_string_boundary_is_honored(tmp_path):
    path=tmp_path/'table.parquet';pl.DataFrame({'x':pl.Series(['a','b'],dtype=pl.Categorical)}).write_parquet(path)
    base={'schema':{'x':'String'}}
    with pytest.raises(ValueError):source_file(path,{'format':'parquet','options':base})
    table=source_file(path,{'format':'parquet','options':{**base,'cast_options':{'categorical_to_string':'allow'}}})
    try:assert list(table.rows)==[('a',),('b',)] and table.columns[0]['dtype']=='string'
    finally:table.rows.store.close()


@pytest.mark.parametrize('extension,expected_sha',[
    ('ods','1aef869b5e34ad735010ca37a27fcf081824fd45d2af351542a74d03e3fed01d'),
    ('xls','efb61c9a60b826b18b251cd74bb80819fcf8eb566e74ef6130709dcd2f7e0590'),
    ('xlsb','ad645571475f1e30875610fc51ad7726d8b7af296850135bc593e15ead0e28c6'),
])
def test_official_legacy_excel_fixture_reader_smoke(extension,expected_sha):
    import os,hashlib
    directory=os.environ.get('KWIPML_LEGACY_EXCEL_FIXTURES_DIR')
    if not directory:pytest.skip('Opt-in pinned official calamine legacy fixture receipt')
    path=Path(directory)/('any_sheets.'+extension)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha
    config={'format':'excel','options':{'sheet_name':'Visible','read_options':{'dtype_coercion':'coerce'}}}
    table=source_file(path,config);again=source_file(path,config)
    try:
        assert list(table.rows)==[('3',4),('5',6),('This workbook contains 4 sheets: Visible, Hidden, VeryHidden and Chart',None)]
        assert [c['dtype'] for c in table.columns]==['string','int64']
        assert table.data_version==again.data_version
        assert table.source_digest==expected_sha
        assert preview_file(path,config)['rows']==[list(row) for row in table.rows]
        # A contradictory explicit strict declaration must fail clearly rather than coerce.
        with pytest.raises(ValueError,match='Table Input excel'):
            source_file(path,{'format':'excel','options':{'sheet_name':'Visible','read_options':{'dtype_coercion':'strict'}}})
    finally:table.rows.store.close();again.rows.store.close()
