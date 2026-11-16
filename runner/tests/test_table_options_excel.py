"""Behavioral workbook option matrix; expected cells are independently authored.

Operational flags have explicit semantic-invariance checks, not claimed memory wins.
"""
from pathlib import Path
import importlib.util
import pytest
from runner.table_sources import source_file


@pytest.fixture(scope='module')
def books(tmp_path_factory):
    path=Path(__file__).parent/'fixtures/table_options/excel/generate.py'
    spec=importlib.util.spec_from_file_location('table_options_excel_generator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.generate(tmp_path_factory.mktemp('table-options-excel'))


def read(books, fixture='basic', engine='calamine', **options):
    if engine=='calamine':options['read_options']={'dtype_coercion':'coerce',**options.get('read_options',{})}
    table=source_file(books/(fixture+'.xlsx'),{'format':'excel','options':{'engine':engine,**options}})
    try:return ([column['name'] for column in table.columns],list(table.rows),[column['dtype'] for column in table.columns])
    finally:table.rows.store.close()


@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_common_engine_sheet_name_and_sheet_id(books,engine):
    for selection in ({'sheet_id':2},{'sheet_name':'Second','sheet_id':None}):
        names,rows,types=read(books,engine=engine,**selection)
        assert names==['name','amount','flag'] and rows==[('Other',9,False)]
        assert types==['string','int64','bool']
    with pytest.raises(ValueError):read(books,engine=engine,sheet_name='Absent')


@pytest.mark.parametrize('columns,expected', [(['flag','name'],['flag','name']),([2,0],['flag','name']),([],['name','amount','flag'])])
@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_common_columns_preserve_selected_order(books,engine,columns,expected):
    names,rows,_=read(books,engine=engine,columns=columns)
    assert names==expected
    assert rows[0]==((True,'Alpha') if columns else ('Alpha',1,True))


def test_common_named_table_and_invalid_engine_conditional(books):
    assert read(books,table_name='Records',sheet_id=None)[1]==[('Alpha',1,True),('Beta',2,False),('Gamma',3,True)]
    for engine in ('openpyxl','xlsx2csv'):
        with pytest.raises(ValueError,match='Named Excel tables require calamine'):
            read(books,engine=engine,table_name='Records')


@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_common_header_and_no_header_have_distinct_observations(books,engine):
    names,rows,_=read(books,'types',engine,has_header=False,infer_schema_length=None)
    assert len(rows)==3 and tuple(rows[0])==('number','flag','decimal')
    assert names[0]!='number'
    assert read(books,'types',engine,has_header=True)[0]==['number','flag','decimal']


@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_common_empty_rows_and_columns(books,engine):
    names,rows,_=read(books,'layout',engine,drop_empty_rows=False,drop_empty_cols=False)
    assert names==['name','amount','unused'] and rows==[('A',1,None),(None,None,None),('B',2,None)]
    names,rows,_=read(books,'layout',engine,drop_empty_rows=True,drop_empty_cols=True)
    assert names==['name','amount','unused'] and rows==[('A',1,None),('B',2,None)]  # Named columns are preserved even when all data cells are null.


@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_common_schema_override_and_inference_optional(books,engine):
    names,rows,types=read(books,'types',engine,schema_overrides={'number':'Int64'})
    assert names[0]=='number' and [row[0] for row in rows]==[1,2] and types[0]=='int64'
    for limit in (1,100,None):
        assert read(books,'basic',engine,infer_schema_length=limit)[1][0]==('Alpha',1,True)


@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
@pytest.mark.parametrize('raise_if_empty',[True,False])
def test_common_header_only_is_a_named_zero_row_table(books,engine,raise_if_empty):
    names,rows,_=read(books,'empty',engine,raise_if_empty=raise_if_empty)
    assert names==['name','amount'] and rows==[]


@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_common_blank_workbook_strict_empty_refusal(books,engine):
    with pytest.raises(ValueError):read(books,'blank',engine,raise_if_empty=True)


def test_calamine_header_row_optional_and_column_names(books):
    assert read(books,'headers',read_options={'header_row':1})[:2]==(['name','amount'],[('A',1),('B',2)])
    names,rows,_=read(books,'types',read_options={'header_row':None,'column_names':['n','b','d'],'dtypes':'string'})
    assert names==['n','b','d'] and rows[0]==('number','flag','decimal')


@pytest.mark.parametrize('skip,expected',[(0,[('Alpha',1,True),('Beta',2,False),('Gamma',3,True)]),(1,[('Beta',2,False),('Gamma',3,True)]),([0,2],[('Beta',2,False)])])
def test_calamine_skip_rows_count_and_indexes(books,skip,expected):
    assert read(books,read_options={'skip_rows':skip})[1]==expected


@pytest.mark.parametrize('limit,count',[(None,3),(0,0),(1,1),(2,2)])
def test_calamine_row_limit_optional_zero_and_positive(books,limit,count):
    assert len(read(books,read_options={'n_rows':limit},raise_if_empty=False)[1])==count


@pytest.mark.parametrize('selection', [['amount'],[1],'B'])
def test_calamine_use_columns_names_indexes_range(books,selection):
    assert read(books,read_options={'use_columns':selection})[:2]==(['amount'],[(1,),(2,),(3,)])


def test_calamine_strict_and_coerce_type_conflicts(books):
    with pytest.raises(ValueError):read(books,'mixed',read_options={'dtype_coercion':'strict'})
    assert read(books,'mixed',read_options={'dtype_coercion':'coerce','dtypes':{'number':'int'}})[1]==[(1,'A'),(None,'B'),(3,'C')]


def test_calamine_sparse_defaults_match_visible_coerce_policy(books):
    table=source_file(books/'mixed.xlsx',{'format':'excel','options':{}})
    # Without an explicit dtype, coerce preserves all cells as text; an explicit
    # numeric override's null conversion is covered by the strict/coerce test.
    try:assert list(table.rows)==[('1','A'),('oops','B'),('3','C')]
    finally:table.rows.store.close()


@pytest.mark.parametrize('dtype,expected,logical',[('string','1','string'),('int',1,'int64'),('float',1.0,'float64')])
def test_calamine_scalar_dtype_selection(books,dtype,expected,logical):
    _,rows,types=read(books,columns=['amount'],read_options={'dtypes':dtype})
    assert rows==[(expected,),(str(2) if dtype=='string' else float(2) if dtype=='float' else 2,),(str(3) if dtype=='string' else float(3) if dtype=='float' else 3,)]
    assert types==['int64' if dtype=='float' else logical]  # Polars refines integral reader floats back to integers.


def test_calamine_boolean_map_and_absent_dtype_default(books):
    assert read(books,columns=['flag'],read_options={'dtypes':{'flag':'boolean'}})[1]==[(True,),(False,),(True,)]
    assert read(books,read_options={'dtypes':None})[1][0]==('Alpha',1,True)


def test_calamine_whitespace_null_and_tail_flags(books):
    names,rows,_=read(books,'whitespace',drop_empty_rows=False,read_options={'whitespace_as_null':False,'skip_whitespace_tail_rows':False,'dtypes':'string'})
    assert rows==[('A','1'),('   ','  '),('\t','   ')]
    assert read(books,'whitespace',drop_empty_rows=False,read_options={'whitespace_as_null':True,'skip_whitespace_tail_rows':False,'dtypes':'string'})[1]==[('A','1'),(None,None),(None,None)]
    assert read(books,'whitespace',drop_empty_rows=False,read_options={'whitespace_as_null':False,'skip_whitespace_tail_rows':True,'dtypes':'string'})[1]==[('A','1')]


@pytest.mark.parametrize('read_only',[True,False])
@pytest.mark.parametrize('rich_text',[True,False])
def test_openpyxl_operational_flags_preserve_plain_cells(books,read_only,rich_text):
    assert read(books,engine='openpyxl',engine_options={'read_only':read_only,'rich_text':rich_text})[1]==[('Alpha',1,True),('Beta',2,False),('Gamma',3,True)]


@pytest.mark.parametrize('quoting',[0,1,2])
def test_xlsx2csv_quoting_roundtrip_delimiter_and_quote_cells(books,quoting):
    assert read(books,'text','xlsx2csv',engine_options={'quoting':quoting})[1]==[('hello,world',1),('line\nbreak',2),('quote"text',3),('café🙂',4)]


def test_xlsx2csv_date_time_and_float_formats(books):
    _,rows,_=read(books,'dates','xlsx2csv',engine_options={'dateformat':'%Y/%m/%d','timeformat':'%H:%M','floatformat':'%.2f'},read_options={'infer_schema':False})
    assert rows==[('2024/02/03','13:14','1.23')]


def test_xlsx2csv_no_line_breaks_and_escape_strings(books):
    plain=read(books,'text','xlsx2csv',read_options={'infer_schema':False})[1]
    assert plain[1][0]=='line\nbreak'
    changed=read(books,'text','xlsx2csv',engine_options={'no_line_breaks':True},read_options={'infer_schema':False})[1]
    assert '\n' not in changed[1][0]
    escaped=read(books,'shared-text','xlsx2csv',engine_options={'escape_strings':True},read_options={'infer_schema':False})[1]
    assert escaped[1][0]=='line\\nbreak'


def test_xlsx2csv_hyperlink_stored_reference_only(books):
    assert read(books,'hyperlinks','xlsx2csv',engine_options={'hyperlinks':False})[1][0][0]=='label'
    assert 'https://example.invalid/only-a-stored-reference' in read(books,'hyperlinks','xlsx2csv',engine_options={'hyperlinks':True})[1][0][0]


def test_xlsx2csv_hidden_sheet_and_row_policies(books):
    assert len(read(books,'hidden','xlsx2csv',engine_options={'skip_hidden_rows':False})[1])==2
    assert read(books,'hidden','xlsx2csv',engine_options={'skip_hidden_rows':True})[1]==[('shown',1)]
    assert read(books,'hidden','xlsx2csv',sheet_name='Hidden',engine_options={'exclude_hidden_sheets':False})[1]==[('secret-synthetic',9)]
    with pytest.raises(ValueError):read(books,'hidden','xlsx2csv',sheet_name='Hidden',engine_options={'exclude_hidden_sheets':True})


def test_xlsx2csv_merge_cells(books):
    assert read(books,'merged','xlsx2csv',engine_options={'merge_cells':False})[1]==[('A',1),(None,2)]
    assert read(books,'merged','xlsx2csv',engine_options={'merge_cells':True})[1]==[('A',1),('A',2)]


def test_xlsx2csv_converted_skip_modes(books):
    assert read(books,'headers','xlsx2csv',read_options={'skip_rows':1})[:2]==(['name','amount'],[('A',1),('B',2)])
    assert read(books,'headers','xlsx2csv',read_options={'skip_lines':1})[:2]==(['name','amount'],[('A',1),('B',2)])
    assert read(books,'basic','xlsx2csv',read_options={'skip_rows_after_header':1})[1]==[('Beta',2,False),('Gamma',3,True)]


def test_xlsx2csv_converted_comment_and_null_policies(books):
    assert read(books,'comments','xlsx2csv',read_options={'comment_prefix':'#'})[1]==[('keep',2)]
    assert read(books,'nulls','xlsx2csv',read_options={'null_values':['NA','NULL'],'infer_schema':False})[1][0][0] is None
    assert read(books,'nulls','xlsx2csv',read_options={'null_values':{'name':'NA'},'infer_schema':False})[1][-1][0]=='NULL'
    assert read(books,'nulls','xlsx2csv',read_options={'empty_string_is_null':False,'infer_schema':False})[1][1][1]==''
    assert read(books,'nulls','xlsx2csv',read_options={'empty_string_is_null':True,'infer_schema':False})[1][1][1] is None


@pytest.mark.parametrize('encoding',['utf8','utf8-lossy'])
@pytest.mark.parametrize('low_memory',[True,False])
def test_xlsx2csv_operational_decoding_and_memory_preserve_valid_unicode(books,encoding,low_memory):
    assert read(books,'text','xlsx2csv',read_options={'encoding':encoding,'low_memory':low_memory})[1][-1]==('café🙂',4)


def test_xlsx2csv_converted_limit_index_names_and_decimal_comma(books):
    names,rows,types=read(books,'basic','xlsx2csv',read_options={'n_rows':1,'row_index_name':'record','row_index_offset':7,'new_columns':['label','count','enabled']})
    assert names==['record','label','count','enabled'] and rows==[(7,'Alpha',1,True)] and types[0]=='int64'
    assert read(books,'comma-decimal','xlsx2csv',read_options={'decimal_comma':True})[1]==[(1.25,),(2.5,)]
    assert read(books,'comma-decimal','xlsx2csv',read_options={'decimal_comma':False})[1]==[('1,25',),('2,50',)]


def test_xlsx2csv_invalid_character_policy(books):
    from xlsx2csv import XlsxValueError
    with pytest.raises((ValueError,XlsxValueError)):read(books,'invalid-date','xlsx2csv',engine_options={'ignore_invalid_char_data':False})
    assert read(books,'invalid-date','xlsx2csv',engine_options={'ignore_invalid_char_data':True})[1]==[('2024-02-03',1),(None,2)]


@pytest.mark.parametrize('engine',['calamine','openpyxl','xlsx2csv'])
def test_common_drop_unnamed_empty_columns_or_refuse_unusable_header(books,engine):
    assert read(books,'trim',engine,drop_empty_cols=True)[:2]==(['name','amount'],[('A',1),('B',2)])
    if engine=='calamine':
        # Calamine's used range already excludes the all-empty terminal column.
        assert read(books,'trim',engine,drop_empty_cols=False)[0]==['name','amount']
    else:
        with pytest.raises(ValueError,match='nonblank'):read(books,'trim',engine,drop_empty_cols=False)


def test_calamine_invalid_range_and_projection_conflicts_refuse(books):
    with pytest.raises(ValueError):read(books,read_options={'use_columns':'B:B'})
    with pytest.raises(ValueError):read(books,columns=['name'],read_options={'use_columns':[1]})


def test_calamine_fractional_float_dtype_is_retained(books):
    assert read(books,'types',columns=['decimal'],read_options={'dtypes':'float'})[1:]==([(1.25,),(2.5,)],['float64'])


def test_calamine_scalar_boolean_dtype(books):
    assert read(books,columns=['flag'],read_options={'dtypes':'boolean'})[1:]==([(True,),(False,),(True,)],['bool'])


def test_xlsx2csv_quote_none_and_no_quoting_refuse_unescaped_delimiters(books):
    import csv
    assert read(books,engine='xlsx2csv',engine_options={'quoting':3},read_options={'quote_char':None})[1][0]==('Alpha',1,True)
    with pytest.raises(csv.Error):read(books,'text','xlsx2csv',engine_options={'quoting':3})


def test_xlsx2csv_empty_lines_and_trailing_columns(books):
    assert len(read(books,'layout','xlsx2csv',drop_empty_rows=False,engine_options={'skip_empty_lines':False})[1])==3
    assert len(read(books,'layout','xlsx2csv',drop_empty_rows=False,engine_options={'skip_empty_lines':True})[1])==2
    with pytest.raises(ValueError):read(books,'trim','xlsx2csv',drop_empty_cols=False,engine_options={'skip_trailing_columns':False})
    assert read(books,'trim','xlsx2csv',drop_empty_cols=False,engine_options={'skip_trailing_columns':True})[0]==['name','amount']


def test_xlsx2csv_scientific_notation_and_ignored_date_format(books):
    plain=read(books,'scientific','xlsx2csv',engine_options={'scifloat':False,'floatformat':'%.4f'},read_options={'infer_schema':False})[1]
    changed=read(books,'scientific','xlsx2csv',engine_options={'scifloat':True,'floatformat':'%.4f'},read_options={'infer_schema':False})[1]
    assert plain[0][0]=='1.23456e-12' and float(changed[0][0])==0.0
    import datetime
    serial=(datetime.date(2024,2,3)-datetime.date(1899,12,30)).days
    assert read(books,'dates','xlsx2csv',engine_options={'ignore_formats':['date']},read_options={'infer_schema':False})[1][0][0]==str(serial)


def test_xlsx2csv_inference_error_and_date_contract(books):
    assert read(books,'basic','xlsx2csv',read_options={'infer_schema':False})[1][0]==('Alpha','1','TRUE')
    with pytest.raises(ValueError):read(books,'mixed','xlsx2csv',infer_schema_length=1,read_options={'ignore_errors':False})
    assert read(books,'mixed','xlsx2csv',infer_schema_length=1,read_options={'ignore_errors':True})[1]==[(1,'A'),(None,'B'),(3,'C')]
    assert read(books,'dates','xlsx2csv',read_options={'try_parse_dates':False})[2][0]=='string'
    with pytest.raises(ValueError,match='scalar'):read(books,'dates','xlsx2csv',read_options={'try_parse_dates':True})


@pytest.mark.parametrize('limit,count',[(None,3),(0,0),(2,2)])
def test_xlsx2csv_optional_read_limit(books,limit,count):
    assert len(read(books,engine='xlsx2csv',read_options={'n_rows':limit},raise_if_empty=False)[1])==count


def test_xlsx2csv_converted_ragged_policy(books):
    with pytest.raises(ValueError):read(books,'delimiter-only','xlsx2csv',read_options={'quote_char':None,'truncate_ragged_lines':False})
    assert read(books,'delimiter-only','xlsx2csv',read_options={'quote_char':None,'truncate_ragged_lines':True})[:2]==(['name','amount'],[('"hello','world"')])


def test_xlsx2csv_optional_date_time_formats_and_default_quote(books):
    row=read(books,'dates','xlsx2csv',engine_options={'dateformat':None,'timeformat':None},read_options={'infer_schema':False})[1][0]
    assert row[0]=='2024-02-03' and row[1]=='13:14' and float(row[2])==1.23456
    assert read(books,'comments','xlsx2csv',read_options={'comment_prefix':None})[1]==[('#omit',1),('keep',2)]


def test_openpyxl_rich_text_cells_keep_visible_text(books):
    for read_only in (True,False):
        for rich_text in (True,False):
            assert read(books,'richtext','openpyxl',engine_options={'read_only':read_only,'rich_text':rich_text})[1]==[('plain bold',1)]


def test_xlsx2csv_merge_populates_rows_without_physical_cells_and_keeps_calls_isolated(books):
    import xlsx2csv
    original=xlsx2csv.Sheet
    assert read(books,'merged-gaps','xlsx2csv',engine_options={'merge_cells':True})[1]==[('A',1),('A',None),('A',None)]
    assert read(books,'merged-gaps','xlsx2csv',engine_options={'merge_cells':False})[1]==[('A',1)]
    assert xlsx2csv.Sheet is original


def test_xlsx2csv_invalid_date_policy_does_not_leak_between_calls(books):
    from xlsx2csv import XlsxValueError
    assert read(books,'invalid-date','xlsx2csv',engine_options={'ignore_invalid_char_data':True})[1][-1]==(None,2)
    with pytest.raises((ValueError,XlsxValueError)):read(books,'invalid-date','xlsx2csv',engine_options={'ignore_invalid_char_data':False})


def test_xlsx2csv_index_offset_zero_and_optional_reader_names(books):
    assert read(books,engine='xlsx2csv',read_options={'row_index_name':'record','row_index_offset':0})[1][0]==(0,'Alpha',1,True)
    assert read(books,engine='xlsx2csv',read_options={'row_index_name':None,'new_columns':[]})[0]==['name','amount','flag']


def test_xlsx2csv_overlapping_merged_ranges_refuse(books):
    with pytest.raises(ValueError,match='must not overlap'):
        read(books,fixture='merged-overlap',engine='xlsx2csv',engine_options={'merge_cells':True})


# Qualified field IDs correspond to the Swift-derived active configuration fields.
# Each function checks actual output/refusal; operational checks are labeled below.
_COMMON={
 'sheet_id':['test_common_engine_sheet_name_and_sheet_id'],
 'sheet_name':['test_common_engine_sheet_name_and_sheet_id'],
 'has_header':['test_common_header_and_no_header_have_distinct_observations'],
 'columns':['test_common_columns_preserve_selected_order'],
 'drop_empty_rows':['test_common_empty_rows_and_columns'],
 'drop_empty_cols':['test_common_empty_rows_and_columns','test_common_drop_unnamed_empty_columns_or_refuse_unusable_header'],
 'engine':['test_common_engine_sheet_name_and_sheet_id'],
 'schema_overrides':['test_common_schema_override_and_inference_optional'],
 'infer_schema_length':['test_common_schema_override_and_inference_optional'],
 'raise_if_empty':['test_common_header_only_is_a_named_zero_row_table','test_common_blank_workbook_strict_empty_refusal'],
}
OPTION_CASES={f'excel.{engine}.{key}':list(tests) for engine in ('calamine','openpyxl','xlsx2csv') for key,tests in _COMMON.items()}
OPTION_CASES.update({
 'excel.calamine.table_name':['test_common_named_table_and_invalid_engine_conditional'],
 'excel.calamine.read_options.header_row':['test_calamine_header_row_optional_and_column_names'],
 'excel.calamine.read_options.column_names':['test_calamine_header_row_optional_and_column_names'],
 'excel.calamine.read_options.skip_rows':['test_calamine_skip_rows_count_and_indexes'],
 'excel.calamine.read_options.n_rows':['test_calamine_row_limit_optional_zero_and_positive'],
 'excel.calamine.read_options.dtype_coercion':['test_calamine_strict_and_coerce_type_conflicts','test_calamine_sparse_defaults_match_visible_coerce_policy'],
 'excel.calamine.read_options.use_columns':['test_calamine_use_columns_names_indexes_range','test_calamine_invalid_range_and_projection_conflicts_refuse'],
 'excel.calamine.read_options.dtypes':['test_calamine_scalar_dtype_selection','test_calamine_boolean_map_and_absent_dtype_default','test_calamine_fractional_float_dtype_is_retained','test_calamine_scalar_boolean_dtype'],
 'excel.calamine.read_options.skip_whitespace_tail_rows':['test_calamine_whitespace_null_and_tail_flags'],
 'excel.calamine.read_options.whitespace_as_null':['test_calamine_whitespace_null_and_tail_flags'],
 'excel.openpyxl.engine_options.read_only':['test_openpyxl_operational_flags_preserve_plain_cells','test_openpyxl_rich_text_cells_keep_visible_text'],
 'excel.openpyxl.engine_options.rich_text':['test_openpyxl_rich_text_cells_keep_visible_text','test_openpyxl_operational_flags_preserve_plain_cells'],
 'excel.xlsx2csv.engine_options.quoting':['test_xlsx2csv_quoting_roundtrip_delimiter_and_quote_cells','test_xlsx2csv_quote_none_and_no_quoting_refuse_unescaped_delimiters'],
 'excel.xlsx2csv.engine_options.dateformat':['test_xlsx2csv_date_time_and_float_formats','test_xlsx2csv_optional_date_time_formats_and_default_quote'],
 'excel.xlsx2csv.engine_options.timeformat':['test_xlsx2csv_date_time_and_float_formats','test_xlsx2csv_optional_date_time_formats_and_default_quote'],
 'excel.xlsx2csv.engine_options.floatformat':['test_xlsx2csv_date_time_and_float_formats','test_xlsx2csv_scientific_notation_and_ignored_date_format'],
 'excel.xlsx2csv.engine_options.scifloat':['test_xlsx2csv_scientific_notation_and_ignored_date_format'],
 'excel.xlsx2csv.engine_options.skip_empty_lines':['test_xlsx2csv_empty_lines_and_trailing_columns'],
 'excel.xlsx2csv.engine_options.skip_trailing_columns':['test_xlsx2csv_empty_lines_and_trailing_columns'],
 'excel.xlsx2csv.engine_options.escape_strings':['test_xlsx2csv_no_line_breaks_and_escape_strings'],
 'excel.xlsx2csv.engine_options.no_line_breaks':['test_xlsx2csv_no_line_breaks_and_escape_strings'],
 'excel.xlsx2csv.engine_options.hyperlinks':['test_xlsx2csv_hyperlink_stored_reference_only'],
 'excel.xlsx2csv.engine_options.exclude_hidden_sheets':['test_xlsx2csv_hidden_sheet_and_row_policies'],
 'excel.xlsx2csv.engine_options.merge_cells':['test_xlsx2csv_merge_cells','test_xlsx2csv_merge_populates_rows_without_physical_cells_and_keeps_calls_isolated','test_xlsx2csv_overlapping_merged_ranges_refuse'],
 'excel.xlsx2csv.engine_options.ignore_formats':['test_xlsx2csv_scientific_notation_and_ignored_date_format'],
 'excel.xlsx2csv.engine_options.skip_hidden_rows':['test_xlsx2csv_hidden_sheet_and_row_policies'],
 'excel.xlsx2csv.engine_options.ignore_invalid_char_data':['test_xlsx2csv_invalid_character_policy','test_xlsx2csv_invalid_date_policy_does_not_leak_between_calls'],
 'excel.xlsx2csv.read_options.quote_char':['test_xlsx2csv_quoting_roundtrip_delimiter_and_quote_cells','test_xlsx2csv_quote_none_and_no_quoting_refuse_unescaped_delimiters'],
 'excel.xlsx2csv.read_options.comment_prefix':['test_xlsx2csv_converted_comment_and_null_policies','test_xlsx2csv_optional_date_time_formats_and_default_quote'],
 'excel.xlsx2csv.read_options.skip_rows':['test_xlsx2csv_converted_skip_modes'],
 'excel.xlsx2csv.read_options.skip_lines':['test_xlsx2csv_converted_skip_modes'],
 'excel.xlsx2csv.read_options.skip_rows_after_header':['test_xlsx2csv_converted_skip_modes'],
 'excel.xlsx2csv.read_options.null_values':['test_xlsx2csv_converted_comment_and_null_policies'],
 'excel.xlsx2csv.read_options.empty_string_is_null':['test_xlsx2csv_converted_comment_and_null_policies'],
 'excel.xlsx2csv.read_options.infer_schema':['test_xlsx2csv_inference_error_and_date_contract'],
 'excel.xlsx2csv.read_options.ignore_errors':['test_xlsx2csv_inference_error_and_date_contract'],
 'excel.xlsx2csv.read_options.n_rows':['test_xlsx2csv_optional_read_limit'],
 'excel.xlsx2csv.read_options.encoding':['test_xlsx2csv_operational_decoding_and_memory_preserve_valid_unicode'],
 'excel.xlsx2csv.read_options.low_memory':['test_xlsx2csv_operational_decoding_and_memory_preserve_valid_unicode'],
 'excel.xlsx2csv.read_options.try_parse_dates':['test_xlsx2csv_inference_error_and_date_contract'],
 'excel.xlsx2csv.read_options.row_index_name':['test_xlsx2csv_converted_limit_index_names_and_decimal_comma','test_xlsx2csv_index_offset_zero_and_optional_reader_names'],
 'excel.xlsx2csv.read_options.row_index_offset':['test_xlsx2csv_converted_limit_index_names_and_decimal_comma','test_xlsx2csv_index_offset_zero_and_optional_reader_names'],
 'excel.xlsx2csv.read_options.new_columns':['test_xlsx2csv_converted_limit_index_names_and_decimal_comma','test_xlsx2csv_index_offset_zero_and_optional_reader_names'],
 'excel.xlsx2csv.read_options.decimal_comma':['test_xlsx2csv_converted_limit_index_names_and_decimal_comma'],
 'excel.xlsx2csv.read_options.truncate_ragged_lines':['test_xlsx2csv_converted_ragged_policy'],
})
OPTION_CLASSIFICATION={key:('operational-invariance' if key.endswith(('.read_only','.encoding','.low_memory','.quoting')) else 'effect-or-explicit-refusal') for key in OPTION_CASES}
