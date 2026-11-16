"""Independent synthetic option semantics for JSON and local markup readers."""
import hashlib
from pathlib import Path
import pytest
from runner.table_sources import source_file,preview_file

FIXTURES=Path(__file__).parent/'fixtures'/'table_options'/'markup_json'
RECORDS=[('café',1,True),('B',2,False),(None,3,None)]


def read(fmt,name=None,options=None,path=None):
    path=path or FIXTURES/(name or ('records.'+fmt))
    table=source_file(path,{'format':fmt,'options':options or {}})
    try:
        return dict(columns=list(table.columns),rows=list(table.rows),ids=list(table.row_ids),digest=table.source_digest,version=table.data_version)
    finally:table.rows.store.close()


@pytest.mark.parametrize('fmt',['json','jsonl'])
def test_json_schema_declares_exact_columns_types_and_sorted_map_order(fmt):
    result=read(fmt,options={'schema':{'value':'Float64','name':'String','flag':'Boolean'}})
    assert [c['name'] for c in result['columns']]==['flag','name','value']
    assert [c['dtype'] for c in result['columns']]==['bool','string','float64']
    assert result['rows']==[(True,'café',1.0),(False,'B',2.0),(None,None,3.0)]
    subset=read(fmt,options={'schema':{'value':'Int64'}})
    assert subset['rows']==[(1,),(2,),(3,)] and [c['name'] for c in subset['columns']]==['value']


@pytest.mark.parametrize('fmt',['json','jsonl'])
@pytest.mark.parametrize('length',[2,100,None])
def test_json_inference_length_changes_late_type_discovery(fmt,length):
    result=read(fmt,'late_type.'+fmt,{'infer_schema_length':length})
    assert result['rows']==[('1',),('later',)] and result['columns'][0]['dtype']=='string'
    with pytest.raises(ValueError):read(fmt,'late_type.'+fmt,{'infer_schema_length':1})


@pytest.mark.parametrize('fmt',['json','jsonl'])
def test_json_explicit_override_controls_type_instead_of_inference(fmt):
    result=read(fmt,'late_type.'+fmt,{'infer_schema_length':1,'schema_overrides':{'x':'String'}})
    assert result['rows']==[('1',),('later',)] and result['columns'][0]['dtype']=='string'
    with pytest.raises(ValueError):read(fmt,'late_type.'+fmt,{'schema_overrides':{'x':'Int64'}})


@pytest.mark.parametrize('fmt',['json','jsonl'])
@pytest.mark.parametrize('dtype,family',[(x,'int64') for x in ('Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64')]+[(x,'float64') for x in ('Float32','Float64')]+[('String','string'),('Boolean','bool')])
@pytest.mark.parametrize('field',['schema','schema_overrides'])
def test_json_all_exposed_scalar_schema_choices(fmt,dtype,family,field):
    column='flag' if dtype=='Boolean' else 'name' if dtype=='String' else 'value'
    result=read(fmt,options={field:{column:dtype}})
    expected=[(row[{'name':0,'value':1,'flag':2}[column]],) for row in RECORDS] if field=='schema' else RECORDS
    assert result['rows']==expected
    assert next(c for c in result['columns'] if c['name']==column)['dtype']==family


@pytest.mark.parametrize('batch_size',[1,2,1024])
@pytest.mark.parametrize('low_memory',[False,True])
def test_jsonl_batch_size_and_memory_strategy_preserve_exact_data(batch_size,low_memory):
    result=read('jsonl',options={'batch_size':batch_size,'low_memory':low_memory})
    assert result['rows']==RECORDS
    assert [c['dtype'] for c in result['columns']]==['string','int64','bool']
    assert result['digest']==hashlib.sha256((FIXTURES/'records.jsonl').read_bytes()).hexdigest()
    assert result['ids']==[result['digest']+':'+str(i) for i in range(3)]
    assert result['version']==read('jsonl')['version']


@pytest.mark.parametrize('limit',[1,2,None])
@pytest.mark.parametrize('offset',[0,17])
def test_jsonl_read_limit_and_index_name_offset(limit,offset):
    result=read('jsonl',options={'n_rows':limit,'row_index_name':'record','row_index_offset':offset})
    selected=RECORDS[:limit]
    assert result['rows']==[(offset+i,*row) for i,row in enumerate(selected)]
    assert [c['name'] for c in result['columns']]==['record','name','value','flag']
    assert result['ids']==[result['digest']+':'+str(i) for i in range(len(selected))]


def test_jsonl_ignore_errors_is_explicit_null_replacement_not_row_loss():
    with pytest.raises(ValueError):read('jsonl','late_type.jsonl',{'schema':{'x':'Int64'},'ignore_errors':False})
    result=read('jsonl','late_type.jsonl',{'schema':{'x':'Int64'},'ignore_errors':True})
    assert result['rows']==[(1,),(None,)] and result['columns'][0]['nullable']


@pytest.mark.parametrize('fmt',['markdown','xml','html'])
@pytest.mark.parametrize('dtype,family',[(x,'int64') for x in ('Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64')]+[(x,'float64') for x in ('Float32','Float64')]+[('String','string'),('Boolean','bool')])
def test_markup_all_exposed_scalar_override_choices(fmt,dtype,family):
    options={'schema_overrides':{'flag' if dtype=='Boolean' else 'name' if dtype=='String' else 'value':dtype},'n_rows':2}
    name='records.xml' if fmt=='xml' else 'tables.md' if fmt=='markdown' else 'tables.html'
    result=read(fmt,name,options)
    numeric=8 if fmt!='html' else 10
    if dtype=='Boolean':expected=[('café','7',True),('B',str(numeric),False)]
    elif dtype=='String':expected=[('café','7','true'),('B',str(numeric),'false')]
    else:expected=[('café',7,'true'),('B',numeric,'false')]
    assert result['rows']==expected
    column='flag' if dtype=='Boolean' else 'name' if dtype=='String' else 'value'
    assert next(c for c in result['columns'] if c['name']==column)['dtype']==family


@pytest.mark.parametrize('fmt,name',[('markdown','tables.md'),('xml','records.xml'),('html','tables.html')])
@pytest.mark.parametrize('encoding',['utf-8','utf-8-sig','utf-16','cp1252'])
def test_all_exposed_markup_encodings_decode_non_ascii_cells(tmp_path,fmt,name,encoding):
    path=tmp_path/name;path.write_bytes((FIXTURES/name).read_text().encode(encoding))
    result=read(fmt,options={'encoding':encoding,'n_rows':1},path=path)
    assert result['rows'][0]==('café','7','true')
    assert result['digest']==hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('fmt,name',[('markdown','tables.md'),('html','tables.html')])
@pytest.mark.parametrize('columns',[['value','name'],[1,0]])
def test_markup_name_and_index_projections_preserve_requested_order(fmt,name,columns):
    result=read(fmt,name,{'columns':columns,'n_rows':1})
    assert [c['name'] for c in result['columns']]==['value','name'] and result['rows']==[('7','café')]


def test_markdown_table_selection_header_and_limit_choices():
    assert read('markdown','tables.md',{'table_index':1,'n_rows':1})['rows']==[('Ada','bold and code')]
    result=read('markdown','tables.md',{'has_header':False,'n_rows':1})
    assert [c['name'] for c in result['columns']]==['column_1','column_2','column_3']
    assert result['rows']==[('name','value','flag')]
    with pytest.raises(ValueError,match='index'):read('markdown','tables.md',{'table_index':2})


@pytest.mark.parametrize('mode,expected',[('label',[('Ada','bold and code'),('photo','left|right')]),('url',[('https://example.invalid/a','bold and code'),('https://example.invalid/p.png','left|right')]),('both',[('Ada (https://example.invalid/a)','bold and code'),('photo (https://example.invalid/p.png)','left|right')])])
def test_markdown_link_representation_all_choices(mode,expected):
    assert read('markdown','tables.md',{'table_index':1,'link_mode':mode})['rows']==expected


def test_markdown_raw_cells_preserve_source_markup():
    result=read('markdown','tables.md',{'table_index':1,'cell_mode':'markdown'})
    assert result['rows']==[('[Ada](https://example.invalid/a)','**bold** and `code`'),('![photo](https://example.invalid/p.png)','left|right')]


@pytest.mark.parametrize('mode,expected',[('newline','first\nsecond\nthird\nfourth'),('space','first second third fourth')])
def test_markdown_line_break_representation_changes_common_br_cells(mode,expected):
    assert read('markdown','linebreaks.md',{'line_breaks':mode})['rows']==[('café',expected)]


def test_xml_attribute_toggle_and_custom_prefix_change_schema():
    base=read('xml',options={'n_rows':1});assert [c['name'] for c in base['columns']]==['name','value','flag']
    result=read('xml',options={'include_attributes':True,'attribute_prefix':'attr:','n_rows':1})
    assert [c['name'] for c in result['columns']]==['attr:id','name','value','flag']
    assert result['rows']==[('r1','café','7','true')]
    default=read('xml',options={'include_attributes':True,'n_rows':1})
    assert default['columns'][0]['name']=='@id'


def test_xml_record_xpath_namespaces_and_explicit_fields():
    options={'row_xpath':'./t:record','namespaces':{'t':'urn:synthetic:table'},'fields':{'z':'t:detail/t:leaf/text()','name':'t:label/text()','id':'@id'},'n_rows':1}
    result=read('xml','namespaces.xml',options)
    assert [c['name'] for c in result['columns']]==['id','name','z']
    assert result['rows']==[('r1','café','inner')]
    with pytest.raises(ValueError):read('xml','namespaces.xml',{'row_xpath':'./t:record'})
    with pytest.raises(ValueError,match='Nested'):read('xml','namespaces.xml',{'row_xpath':'./t:record','namespaces':options['namespaces']})
    with pytest.raises(ValueError,match='multiple'):read('xml','namespaces.xml',{**options,'fields':{'x':'t:*/text()'}})


@pytest.mark.parametrize('mode,expected',[('label','Ada'),('url','https://example.invalid/a'),('both','Ada (https://example.invalid/a)')])
def test_html_links_all_choices_and_text_cell_contents(mode,expected):
    result=read('html','tables.html',{'table_index':1,'link_mode':mode})
    assert result['rows']==[(expected,'bold and code\nend')]


def test_html_raw_cell_mode_preserves_markup_without_rendering():
    result=read('html','tables.html',{'table_index':1,'cell_mode':'html'})
    assert result['rows']==[('<a href="https://example.invalid/a">Ada</a>','<b>bold</b> and <code>code</code><br>end<script>ignore me</script>')]


@pytest.mark.parametrize('include,expected',[ (False,['café','B']),(True,['café','hidden','styled','B'])])
def test_html_stored_hidden_row_toggle(include,expected):
    assert [r[0] for r in read('html','tables.html',{'include_hidden':include})['rows']]==expected


def test_html_header_index_generated_names_and_skip_rows():
    assert read('html','headers.html',{'header':1})['rows']==[('café','7'),('B','8')]
    assert read('html','headers.html',{'skip_rows':1})['rows']==[('café','7'),('B','8')]
    result=read('html','headers.html',{'header':None,'skip_rows':1,'n_rows':1})
    assert [c['name'] for c in result['columns']]==['column_1','column_2'] and result['rows']==[('name','value')]
    with pytest.raises(ValueError,match='header'):read('html','headers.html',{'header':4})
    with pytest.raises(ValueError,match='selected rows'):read('html','headers.html',{'skip_rows':4})


def test_html_span_policies_reject_or_expand_exact_rows():
    with pytest.raises(ValueError,match='Expand'):read('html','spans.html',{'span_policy':'reject'})
    assert read('html','spans.html',{'span_policy':'expand'})['rows']==[('a','b'),('a','c'),('d','d')]


@pytest.mark.parametrize('fmt,name',[('markdown','tables.md'),('xml','records.xml'),('html','tables.html')])
def test_markup_row_limit_is_exact_with_original_row_identity(fmt,name):
    result=read(fmt,name,{'n_rows':1})
    assert len(result['rows'])==1 and result['ids']==[result['digest']+':0']
    assert len(preview_file(FIXTURES/name,{'format':fmt,'options':{}})['rows'])<=8


@pytest.mark.parametrize('fmt',['json','jsonl','markdown','xml','html'])
def test_unsupported_reader_options_and_nested_types_refuse_before_source_access(tmp_path,fmt):
    for options in ({'schema_overrides':{'x':'Datetime'}},{'unsafe_callback':'x'}):
        with pytest.raises(ValueError):read(fmt,options=options,path=tmp_path/'absent')


def test_jsonl_narrow_integer_declaration_refuses_overflow(tmp_path):
    path=tmp_path/'overflow.jsonl';path.write_text('{"x":128}\n')
    for field in ('schema','schema_overrides'):
        with pytest.raises(ValueError):read('jsonl',options={field:{'x':'Int8'}},path=path)


OPTION_CASES={
 'json.schema':['test_json_schema_declares_exact_columns_types_and_sorted_map_order','test_json_all_exposed_scalar_schema_choices'],
 'json.schema_overrides':['test_json_explicit_override_controls_type_instead_of_inference','test_json_all_exposed_scalar_schema_choices'],
 'json.infer_schema_length':['test_json_inference_length_changes_late_type_discovery'],
 'jsonl.schema':['test_json_schema_declares_exact_columns_types_and_sorted_map_order','test_json_all_exposed_scalar_schema_choices','test_jsonl_narrow_integer_declaration_refuses_overflow'],
 'jsonl.schema_overrides':['test_json_explicit_override_controls_type_instead_of_inference','test_json_all_exposed_scalar_schema_choices','test_jsonl_narrow_integer_declaration_refuses_overflow'],
 'jsonl.infer_schema_length':['test_json_inference_length_changes_late_type_discovery'],
 'jsonl.batch_size':['test_jsonl_batch_size_and_memory_strategy_preserve_exact_data'],
 'jsonl.n_rows':['test_jsonl_read_limit_and_index_name_offset'],
 'jsonl.low_memory':['test_jsonl_batch_size_and_memory_strategy_preserve_exact_data'],
 'jsonl.row_index_name':['test_jsonl_read_limit_and_index_name_offset'],
 'jsonl.row_index_offset':['test_jsonl_read_limit_and_index_name_offset'],
 'jsonl.ignore_errors':['test_jsonl_ignore_errors_is_explicit_null_replacement_not_row_loss'],
 'markdown.table_index':['test_markdown_table_selection_header_and_limit_choices'],
 'markdown.has_header':['test_markdown_table_selection_header_and_limit_choices'],
 'markdown.cell_mode':['test_markdown_raw_cells_preserve_source_markup','test_markdown_link_representation_all_choices'],
 'markdown.link_mode':['test_markdown_link_representation_all_choices'],
 'markdown.line_breaks':['test_markdown_line_break_representation_changes_common_br_cells'],
 'markdown.columns':['test_markup_name_and_index_projections_preserve_requested_order'],
 'markdown.encoding':['test_all_exposed_markup_encodings_decode_non_ascii_cells'],
 'markdown.n_rows':['test_markup_row_limit_is_exact_with_original_row_identity'],
 'markdown.schema_overrides':['test_markup_all_exposed_scalar_override_choices'],
 'xml.row_xpath':['test_xml_record_xpath_namespaces_and_explicit_fields'],
 'xml.fields':['test_xml_record_xpath_namespaces_and_explicit_fields'],
 'xml.include_attributes':['test_xml_attribute_toggle_and_custom_prefix_change_schema'],
 'xml.attribute_prefix':['test_xml_attribute_toggle_and_custom_prefix_change_schema'],
 'xml.namespaces':['test_xml_record_xpath_namespaces_and_explicit_fields'],
 'xml.encoding':['test_all_exposed_markup_encodings_decode_non_ascii_cells'],
 'xml.n_rows':['test_markup_row_limit_is_exact_with_original_row_identity'],
 'xml.schema_overrides':['test_markup_all_exposed_scalar_override_choices'],
 'html.table_index':['test_html_links_all_choices_and_text_cell_contents'],
 'html.header':['test_html_header_index_generated_names_and_skip_rows'],
 'html.cell_mode':['test_html_raw_cell_mode_preserves_markup_without_rendering','test_html_links_all_choices_and_text_cell_contents'],
 'html.link_mode':['test_html_links_all_choices_and_text_cell_contents'],
 'html.include_hidden':['test_html_stored_hidden_row_toggle'],
 'html.columns':['test_markup_name_and_index_projections_preserve_requested_order'],
 'html.span_policy':['test_html_span_policies_reject_or_expand_exact_rows'],
 'html.skip_rows':['test_html_header_index_generated_names_and_skip_rows'],
 'html.encoding':['test_all_exposed_markup_encodings_decode_non_ascii_cells'],
 'html.n_rows':['test_markup_row_limit_is_exact_with_original_row_identity'],
 'html.schema_overrides':['test_markup_all_exposed_scalar_override_choices'],
}

# Controlled combinations, not missing tests: raw Markdown/HTML cell modes retain
# markup, so link/line-break presentation controls apply to text mode only.
# XML include_attributes/attribute_prefix apply to automatic fields; explicit
# fields own their XPath selection. JSON/JSONL have no encoding field (UTF-8).
# Resource options preserve records rather than altering table semantics.
INAPPLICABLE_COMBINATIONS={
 'markdown.cell_mode=markdown':['link_mode','line_breaks'],
 'html.cell_mode=html':['link_mode'],
 'xml.fields=nonempty':['include_attributes','attribute_prefix'],
}


@pytest.mark.parametrize('dtype,invalid', [('Int8',128),('Int16',32768),('UInt8',-1),('UInt16',65536)])
def test_jsonl_narrow_overflow_error_policy_nulls_only_conflicting_cells(tmp_path,dtype,invalid):
    path=tmp_path/'narrow.jsonl';path.write_text('{"x":'+str(invalid)+'}\n{"x":7}\n')
    with pytest.raises(ValueError):read('jsonl',options={'schema':{'x':dtype},'ignore_errors':False},path=path)
    result=read('jsonl',options={'schema':{'x':dtype},'ignore_errors':True},path=path)
    assert result['rows']==[(None,),(7,)] and result['columns'][0]['nullable']
    assert result['ids']==[result['digest']+':0',result['digest']+':1']


def test_jsonl_schema_override_takes_precedence_over_declared_schema(tmp_path):
    path=tmp_path/'precedence.jsonl';path.write_text('{"x":128}\n')
    result=read('jsonl',options={'schema':{'x':'Int8'},'schema_overrides':{'x':'Int64'}},path=path)
    assert result['rows']==[(128,)] and result['columns'][0]['dtype']=='int64'
    with pytest.raises(ValueError):read('jsonl',options={'schema':{'x':'Int64'},'schema_overrides':{'x':'Int8'}},path=path)
    assert read('jsonl',options={'schema':{'x':'Int64'},'schema_overrides':{'x':'Int8'},'ignore_errors':True},path=path)['rows']==[(None,)]


@pytest.mark.parametrize('mode',['newline','space'])
def test_markdown_raw_mode_preserves_br_markup_for_both_linebreak_settings(mode):
    assert read('markdown','linebreaks.md',{'cell_mode':'markdown','line_breaks':mode})['rows']==[('café','first<br>second<br/>third<br />fourth')]


OPTION_CASES['jsonl.ignore_errors'].append('test_jsonl_narrow_overflow_error_policy_nulls_only_conflicting_cells')
OPTION_CASES['jsonl.schema'].append('test_jsonl_schema_override_takes_precedence_over_declared_schema')
OPTION_CASES['jsonl.schema_overrides'].append('test_jsonl_schema_override_takes_precedence_over_declared_schema')
OPTION_CASES['markdown.cell_mode'].append('test_markdown_raw_mode_preserves_br_markup_for_both_linebreak_settings')
