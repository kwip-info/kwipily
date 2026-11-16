"""Dedicated real-file option fixtures for native/compatibility CSV and Parquet.

Value assertions are independent fixture expectations. Reader strategy/resource knobs
also verify exact dispatch while the real reader parses; these are not speed claims.
"""
from pathlib import Path
import hashlib
import pytest
import polars as pl
import pyarrow.parquet as pq
from runner import polars_csv as csv
from runner import table_sources as sources
from runner.dataflow import source_csv as compatibility_csv

FIXTURES = Path(__file__).parent / 'fixtures' / 'table_options' / 'csv_parquet'


def values(table):
    try:
        return tuple(c['name'] for c in table.columns), list(table.rows), tuple(c['dtype'] for c in table.columns)
    finally:
        if hasattr(table.rows, 'store'): table.rows.store.close()


def native(name, options=None):
    return values(csv.source_csv(FIXTURES / name, options))


def temporary(tmp_path, data):
    path=tmp_path/'case.csv';path.write_bytes(data);return path


@pytest.mark.parametrize('delimiter', [',','\t',';','|','^'])
def test_csv_separator_choices(tmp_path, delimiter):
    path=temporary(tmp_path, f'a{delimiter}b\n001{delimiter}two\n'.encode())
    assert values(csv.source_csv(path,{'separator':delimiter}))[:2]==(('a','b'),[('001','two')])
    assert csv.preview_csv(path,{'separator':delimiter})==(['a','b'],[['001','two']])


@pytest.mark.parametrize('name,quote,expected', [('quoted.csv','"',[('1','two\nlines'),('2','a "quote"')]),('single.csv',"'",[('1','two;parts')]),('single.csv','',[('1',"'two","parts'")])])
def test_csv_quote_choices(name, quote, expected):
    options={'separator':';','quoteChar':quote}
    if not quote:
        with pytest.raises(ValueError):native(name,options)
        return
    assert native(name,options)[1]==expected
    assert csv.preview_csv(FIXTURES/name,options)[1]==[list(row) for row in expected]


def test_csv_no_quote_has_literal_quotes(tmp_path):
    path=temporary(tmp_path,b'a,b\n1,"quoted"\n')
    assert values(csv.source_csv(path,{'quoteChar':''}))[1]==[('1','"quoted"')]


@pytest.mark.parametrize('encoding,name,expected', [('utf-8-sig','bom.csv','café'),('utf8-lossy','lossy.csv','\ufffd')])
def test_csv_encoding_choices(encoding,name,expected):
    assert native(name,{'encoding':encoding})[1]==[(expected,)]
    assert csv.preview_csv(FIXTURES/name,{'encoding':encoding})[1]==[[expected]]


def test_csv_strict_utf8_refuses_bad_bytes():
    with pytest.raises(ValueError):native('lossy.csv',{'encoding':'utf-8-sig'})


@pytest.mark.parametrize('mode,names,rows', [('firstRecord',('a','b'),[('001','2'),('003','4'),('005','6')]),('generated',('column_0','column_1'),[('a','b'),('001','2'),('003','4'),('005','6')])])
def test_csv_header_mode(mode,names,rows):
    assert native('basic.csv',{'headerMode':mode})[:2]==(names,rows)


@pytest.mark.parametrize('key,payload,options,expected', [
 ('skipRecords',b'"ignore\nlogical",record\na,b\nx,y\n',{'skipRecords':1},[('x','y')]),
 ('skipLines',b'ignored physical\na,b\nx,y\n',{'skipLines':1},[('x','y')]),
 ('skipRowsAfterHeader',b'a,b\ndiscard,me\nx,y\n',{'skipRowsAfterHeader':1},[('x','y')]),
 ('commentPrefix',b'## ignored\na,b\nx,y\n',{'commentPrefix':'##'},[('x','y')]),
])
def test_csv_skip_and_comment_options(tmp_path,key,payload,options,expected):
    path=temporary(tmp_path,payload)
    assert values(csv.source_csv(path,options))[1]==expected
    assert csv.preview_csv(path,options)[1]==[list(row) for row in expected]


@pytest.mark.parametrize('policy,expected', [('text',[('','NA'),('NULL','')]),('null',[(None,'NA'),('NULL',None)])])
def test_csv_empty_field_choices(policy,expected):
    assert native('nulls.csv',{'emptyField':policy})[1]==expected


def test_csv_global_and_per_column_null_markers():
    assert native('nulls.csv',{'nullMarkers':['NA','NULL']})[1]==[('',None),(None,'')]
    assert native('nulls.csv',{'columnNullValues':{'a':'NULL','b':'NA'}})[1]==[('',None),(None,'')]
    assert native('nulls.csv',{'columnNullValues':{'a':''}})[1]==[('','NA'),('NULL','')]
    assert native('quoted_nulls.csv',{'columnNullValues':{'a':''}})[1]==[(None,'NA'),('NULL','')]
    with pytest.raises(ValueError):native('nulls.csv',{'nullMarkers':['NA'],'columnNullValues':{'b':'NA'}})


@pytest.mark.parametrize('columns,names,rows', [([],('a','b'),[('001','2'),('003','4'),('005','6')]),(['b','a'],('b','a'),[('2','001'),('4','003'),('6','005')]),([1],('b',),[('2',),('4',),('6',)])])
def test_csv_columns_choices(columns,names,rows):
    assert native('basic.csv',{'columns':columns})[:2]==(names,rows)


@pytest.mark.parametrize('dtype', ['String','Boolean','Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64','Float32','Float64'])
@pytest.mark.parametrize('key', ['schema','schemaOverrides'])
def test_csv_scalar_schema_choices(tmp_path,key,dtype):
    raw='true' if dtype=='Boolean' else '1.5' if dtype.startswith('Float') else '7'
    expected=True if dtype=='Boolean' else 1.5 if dtype.startswith('Float') else '7' if dtype=='String' else 7
    path=temporary(tmp_path,('value\n'+raw+'\n').encode())
    result=values(csv.source_csv(path,{key:{'value':dtype}}))
    assert result[1]==[(expected,)]
    assert result[2]==(('bool',) if dtype=='Boolean' else ('float64',) if dtype.startswith('Float') else ('string',) if dtype=='String' else ('int64',))


@pytest.mark.parametrize('infer,expected,dtype', [(False,'001','string'),(True,1,'int64')])
def test_csv_infer_schema_choice(infer,expected,dtype):
    result=native('basic.csv',{'inferSchema':infer})
    assert result[1][0][0]==expected and result[2][0]==dtype


@pytest.mark.parametrize('length,expected', [(0,[('1',),('bad',)]),(None,[('1',),('bad',)])])
def test_csv_inference_optional_and_zero(length,expected):
    assert native('errors.csv',{'inferSchema':True,'inferSchemaLength':length})[1]==expected


def test_csv_short_inference_window_exposes_later_type_error():
    with pytest.raises(ValueError):native('errors.csv',{'inferSchema':True,'inferSchemaLength':1})


@pytest.mark.parametrize('count,expected', [(None,[('001','2'),('003','4'),('005','6')]),(2,[('001','2'),('003','4')]),(0,[])])
def test_csv_n_rows_optional_and_zero(count,expected):
    assert native('basic.csv',{'nRows':count,'raiseIfEmpty':False})[1]==expected


@pytest.mark.parametrize('enabled',[False,True])
def test_csv_low_memory_real_read_and_dispatch(monkeypatch,enabled):
    original=csv.pl.scan_csv;seen=[]
    def traced(*args,**kwargs):seen.append(kwargs['low_memory']);return original(*args,**kwargs)
    monkeypatch.setattr(csv.pl,'scan_csv',traced)
    assert native('basic.csv',{'lowMemory':enabled})[1][0]==('001','2')
    assert seen==[enabled]


def test_csv_date_parse_choice_and_typed_boundary():
    assert native('dates.csv',{'tryParseDates':False,'inferSchema':True})[1]==[('2026-10-13',)]
    with pytest.raises(ValueError,match='scalar'):native('dates.csv',{'tryParseDates':True,'inferSchema':True})


@pytest.mark.parametrize('ignore,expected', [(True,[(1,),(None,)])])
def test_csv_ignore_errors_choice(ignore,expected):
    assert native('errors.csv',{'schemaOverrides':{'a':'Int64'},'ignoreErrors':ignore})[1]==expected
    with pytest.raises(ValueError):native('errors.csv',{'schemaOverrides':{'a':'Int64'},'ignoreErrors':False})


def test_csv_decimal_comma_and_rename_index():
    assert native('decimal.csv',{'separator':';','decimalComma':True,'schemaOverrides':{'a':'Float64'}})[1]==[(1.25,'2')]
    assert native('decimal.csv',{'separator':';','decimalComma':False})[1]==[('1,25','2')]
    names,rows,_=native('basic.csv',{'newColumns':['number','amount'],'rowIndexName':'row','rowIndexOffset':10,'nRows':1})
    assert names==('row','number','amount') and rows==[(10,'001','2')]
    assert native('basic.csv',{'rowIndexName':'row','rowIndexOffset':0,'nRows':1})[1]==[(0,'001','2')]


@pytest.mark.parametrize('eol', ['\n','\r','~'])
def test_csv_record_terminator_choices(tmp_path,eol):
    path=temporary(tmp_path,('a,b'+eol+'1,2'+eol).encode())
    assert values(csv.source_csv(path,{'eolChar':eol}))[1]==[('1','2')]
    assert csv.preview_csv(path,{'eolChar':eol})[1]==[['1','2']]


def test_csv_extra_missing_and_empty_choices(tmp_path):
    assert native('ragged.csv',{'extraColumns':'ignore'})[1]==[('1','2'),('3','4')]
    with pytest.raises(ValueError):native('ragged.csv',{'extraColumns':'raise'})
    path=temporary(tmp_path,b'a\n1\n')
    assert values(csv.source_csv(path,{'schema':{'a':'Int64','b':'String'},'missingColumns':'insert'}))[1]==[(1,'')]
    assert values(csv.source_csv(path,{'schema':{'a':'Int64','b':'String'},'missingColumns':'insert','emptyField':'null'}))[1]==[(1,None)]
    with pytest.raises(ValueError):csv.source_csv(path,{'schema':{'a':'Int64','b':'String'},'missingColumns':'raise'})
    assert native('header_only.csv',{'raiseIfEmpty':False})[1]==[]
    with pytest.raises(ValueError):native('header_only.csv',{'raiseIfEmpty':True})


def test_csv_native_profile_and_incompatible_engine_are_explicit():
    assert native('basic.csv',{'parserProfile':'polars','usePyarrow':False})[1][0]==('001','2')
    with pytest.raises(ValueError):native('basic.csv',{'usePyarrow':True})


@pytest.mark.parametrize('encoding,name', [('cp1252','cp1252.csv'),('utf-16','utf16.csv'),('utf-8-sig','bom.csv')])
def test_compatibility_csv_encodings(encoding,name):
    assert values(compatibility_csv(FIXTURES/name,options={'encoding':encoding}))[1]==[('café',)]


@pytest.mark.parametrize('mode,name,character', [('custom','legacy_custom.csv','~'),('backslash','legacy_backslash.csv','\\')])
def test_compatibility_csv_escape_modes(mode,name,character):
    assert values(compatibility_csv(FIXTURES/name,options={'escapeMode':mode,'escapeChar':character}))[1]==[('1','x"y')]


@pytest.mark.parametrize('mode', ['double','none'])
def test_compatibility_csv_double_or_disabled_escaping(tmp_path,mode):
    path=temporary(tmp_path,b'a,b\n1,"plain"\n')
    assert values(compatibility_csv(path,options={'escapeMode':mode}))[1]==[('1','plain')]


@pytest.mark.parametrize('ending,payload', [('auto',b'a,b\n1,2\n'),('lf',b'a,b\n1,2\n'),('crlf',b'a,b\r\n1,2\r\n'),('cr',b'a,b\r1,2\r')])
def test_compatibility_csv_record_endings(tmp_path,ending,payload):
    path=temporary(tmp_path,payload)
    assert values(compatibility_csv(path,options={'recordEndings':ending}))[1]==[('1','2')]


@pytest.mark.parametrize('allow',[True,False])
def test_compatibility_csv_quoted_newlines(allow):
    options={'separator':';','allowQuotedNewlines':allow}
    if allow:assert values(compatibility_csv(FIXTURES/'quoted.csv',options=options))[1][0]==('1','two\nlines')
    else:
        with pytest.raises(ValueError):compatibility_csv(FIXTURES/'quoted.csv',options=options)


@pytest.mark.parametrize('skip',[True,False])
def test_compatibility_csv_blank_records(tmp_path,skip):
    path=temporary(tmp_path,b'a,b\n\n1,2\n')
    if skip:assert values(compatibility_csv(path,options={'skipBlankRecords':skip}))[1]==[('1','2')]
    else:
        with pytest.raises(ValueError):compatibility_csv(path,options={'skipBlankRecords':skip})


@pytest.mark.parametrize('skip,expected',[ (False,' x'),(True,'x') ])
def test_compatibility_csv_initial_spaces(tmp_path,skip,expected):
    path=temporary(tmp_path,b'a,b\n1, x\n')
    assert values(compatibility_csv(path,options={'skipInitialSpace':skip}))[1]==[('1',expected)]


def parquet(options=None,name='basic.parquet'):
    return values(sources.source_file(FIXTURES/name,{'format':'parquet','options':options or {}}))


@pytest.mark.parametrize('strategy',['auto','columns','row_groups','none','prefiltered'])
def test_parquet_parallel_real_read_dispatch(monkeypatch,strategy):
    original=sources.pl.scan_parquet;seen=[]
    def traced(*args,**kwargs):seen.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(sources.pl,'scan_parquet',traced)
    assert parquet({'parallel':strategy})[1]==[(1,'x'),(2,'y'),(3,'z')]
    assert seen[0]['parallel']==strategy and seen[0]['glob'] is False and seen[0]['hive_partitioning'] is False


@pytest.mark.parametrize('key',['use_statistics','low_memory'])
@pytest.mark.parametrize('enabled',[False,True])
def test_parquet_native_tuning_real_read_dispatch(monkeypatch,key,enabled):
    original=sources.pl.scan_parquet;seen=[]
    def traced(*args,**kwargs):seen.append(kwargs[key]);return original(*args,**kwargs)
    monkeypatch.setattr(sources.pl,'scan_parquet',traced)
    assert parquet({key:enabled})[1]==[(1,'x'),(2,'y'),(3,'z')]
    assert seen==[enabled]


@pytest.mark.parametrize('arrow',[False,True])
@pytest.mark.parametrize('columns',[[],['b','a'],[1]])
def test_parquet_reader_and_projection_choices(arrow,columns):
    names,rows,_=parquet({'use_pyarrow':arrow,'columns':columns})
    expected = (('a','b'),[(1,'x'),(2,'y'),(3,'z')]) if not columns else (('b','a'),[('x',1),('y',2),('z',3)]) if columns==['b','a'] else (('b',),[('x',),('y',),('z',)])
    assert (names,rows)==expected


@pytest.mark.parametrize('arrow',[False,True])
@pytest.mark.parametrize('limit',[None,0,2])
def test_parquet_optional_limits_and_index(arrow,limit):
    names,rows,_=parquet({'use_pyarrow':arrow,'n_rows':limit,'row_index_name':'row','row_index_offset':5})
    assert names==('row','a','b') and rows==[(5+i,a,b) for i,(a,b) in enumerate([(1,'x'),(2,'y'),(3,'z')][:limit])]


def test_parquet_schema_missing_and_extra_choices():
    assert parquet({'schema':{'a':'Int64','missing':'String'},'missing_columns':'insert','extra_columns':'ignore'})[1]==[(1,None),(2,None),(3,None)]
    for options in [{'schema':{'a':'Int64','missing':'String'},'missing_columns':'raise','extra_columns':'ignore'},{'schema':{'a':'Int64'},'extra_columns':'raise'}]:
        with pytest.raises(ValueError):parquet(options)
    assert parquet({'schema':{'a':'Int64'},'extra_columns':'ignore'})[1]==[(1,),(2,),(3,)]


@pytest.mark.parametrize('integer',['forbid','upcast','allow-float'])
def test_parquet_integer_cast_choices(integer):
    target='Float64' if integer=='allow-float' else 'Int64'
    options={'columns':['a'],'schema':{'a':target},'extra_columns':'ignore','cast_options':{'integer_cast':integer}}
    if integer=='forbid':
        with pytest.raises(ValueError):parquet(options,'widths.parquet')
    else:
        _,rows,dtypes=parquet(options,'widths.parquet')
        assert rows==([(1.0,),(2.0,)] if integer=='allow-float' else [(1,),(2,)])
        assert dtypes==(('float64',) if integer=='allow-float' else ('int64',))


@pytest.mark.parametrize('float_cast',['forbid','upcast','downcast',['downcast','upcast']])
def test_parquet_float_cast_choices(float_cast):
    options={'columns':['b'],'schema':{'b':'Float64'},'extra_columns':'ignore','cast_options':{'float_cast':float_cast}}
    if float_cast in ('forbid','downcast'):
        with pytest.raises(ValueError):parquet(options,'widths.parquet')
    else:assert parquet(options,'widths.parquet')[1]==[(1.5,),(2.5,)]


@pytest.mark.parametrize('cast',['forbid','allow'])
def test_parquet_categorical_cast_choices(cast):
    options={'schema':{'category':'String'},'cast_options':{'categorical_to_string':cast}}
    if cast=='forbid':
        with pytest.raises(ValueError):parquet(options,'category.parquet')
    else:assert parquet(options,'category.parquet')[1]==[('red',),('blue',)]


@pytest.mark.parametrize('key,setting', [
 ('memory_map',True),('memory_map',False),('use_threads',True),('use_threads',False),
 ('use_pandas_metadata',True),('use_pandas_metadata',False),('pre_buffer',True),('pre_buffer',False),
 ('buffer_size',0),('buffer_size',8192),('coerce_int96_timestamp_unit',''),('coerce_int96_timestamp_unit','ns'),
 ('coerce_int96_timestamp_unit','us'),('coerce_int96_timestamp_unit','ms'),
 ('thrift_string_size_limit',None),('thrift_string_size_limit',1000000),
 ('thrift_container_size_limit',None),('thrift_container_size_limit',100000),
 ('schema_depth_limit',None),('schema_depth_limit',16),
 ('page_checksum_verification',False),('page_checksum_verification',True),
])
def test_parquet_arrow_options_real_read_and_dispatch(monkeypatch,key,setting):
    original=pq.ParquetFile;constructors=[];batches=[]
    original_batches=original.iter_batches
    def traced_batches(self,*args,**kwargs):batches.append(kwargs);yield from original_batches(self,*args,**kwargs)
    monkeypatch.setattr(original,'iter_batches',traced_batches)
    def traced(*args,**kwargs):
        reader=original(*args,**kwargs);constructors.append((kwargs,str(reader.schema_arrow.field('stamp').type) if 'stamp' in reader.schema_arrow.names else None));return reader
    monkeypatch.setattr(pq,'ParquetFile',traced)
    options={'use_pyarrow':True,'columns':['a']}
    if key=='memory_map':options[key]=setting
    else:options['pyarrow_options']={key:setting}
    assert parquet(options,'int96.parquet')[1]==[(1,),(2,)]
    actual=[entry for entry in constructors if 'memory_map' in entry[0]][0]
    if key in ('use_threads','use_pandas_metadata'):assert batches[0][key]==setting
    elif setting in (None,''):assert key not in actual[0]
    else:assert actual[0][key]==setting
    if key=='coerce_int96_timestamp_unit':assert actual[1]==f'timestamp[{setting or "ns"}]'


def test_parquet_arrow_metadata_limit_refusal_is_real():
    with pytest.raises(Exception):parquet({'use_pyarrow':True,'pyarrow_options':{'thrift_string_size_limit':1}})
    with pytest.raises(Exception):parquet({'use_pyarrow':True,'pyarrow_options':{'thrift_container_size_limit':1}})


@pytest.mark.parametrize('delimiter', [',','\t',';','|','^'])
def test_compatibility_csv_separator_choices(tmp_path,delimiter):
    path=temporary(tmp_path,('a'+delimiter+'b\n001'+delimiter+'text\n').encode())
    assert values(compatibility_csv(path,options={'separator':delimiter}))[1]==[('001','text')]


@pytest.mark.parametrize('quote,payload,expected', [('"',b'a;b\n1;"x;y"\n','x;y'),("'",b"a;b\n1;'x;y'\n",'x;y'),('',b'a;b\n1;"literal"\n','"literal"')])
def test_compatibility_csv_quote_choices(tmp_path,quote,payload,expected):
    path=temporary(tmp_path,payload)
    assert values(compatibility_csv(path,options={'separator':';','quoteChar':quote}))[1]==[('1',expected)]


def test_compatibility_csv_utf8_preserves_bom_and_generated_header_skip():
    names,rows,_=values(compatibility_csv(FIXTURES/'bom.csv',options={'encoding':'utf-8'}))
    assert names==('\ufeffa',) and rows==[('café',)]
    names,rows,_=values(compatibility_csv(FIXTURES/'basic.csv',options={'headerMode':'generated','skipRecords':1}))
    assert names==('column_1','column_2') and rows==[('001','2'),('003','4'),('005','6')]


@pytest.mark.parametrize('policy,expected',[('text',[('','NA'),('NULL','')]),('null',[(None,'NA'),('NULL',None)])])
def test_compatibility_csv_empty_field_and_null_markers(policy,expected):
    assert values(compatibility_csv(FIXTURES/'nulls.csv',options={'emptyField':policy}))[1]==expected
    assert values(compatibility_csv(FIXTURES/'nulls.csv',options={'nullMarkers':['NA','NULL'],'emptyField':policy}))[1]==[(None if policy=='null' else '',None),(None,None if policy=='null' else '')]


def test_compatibility_csv_escape_disabled_preserves_extra_quotes():
    assert values(compatibility_csv(FIXTURES/'quoted.csv',options={'separator':';','escapeMode':'double'}))[1][1]==('2','a "quote"')
    assert values(compatibility_csv(FIXTURES/'quoted.csv',options={'separator':';','escapeMode':'none'}))[1][1]==('2','a "quote"""')


def test_parquet_checksum_choice_detects_corrupted_page():
    assert parquet({'use_pyarrow':True,'pyarrow_options':{'page_checksum_verification':True}},'checksum.parquet')[1]==[(1,),(2,),(3,)]
    assert parquet({'use_pyarrow':True,'pyarrow_options':{'page_checksum_verification':False}},'checksum_corrupt.parquet')[1]==[(1,),(20,),(3,)]
    with pytest.raises(Exception,match='CRC|checksum'):
        parquet({'use_pyarrow':True,'pyarrow_options':{'page_checksum_verification':True}},'checksum_corrupt.parquet')


@pytest.mark.parametrize('float_cast',['forbid','upcast','downcast',['downcast','upcast']])
def test_parquet_float_narrowing_has_measured_numeric_effect(float_cast):
    options={'schema':{'b':'Float32'},'cast_options':{'float_cast':float_cast}}
    if float_cast in ('forbid','upcast'):
        with pytest.raises(ValueError):parquet(options,'float64.parquet')
    else:
        assert parquet(options,'float64.parquet')[1]==[(16777216.0,),(1.5,)]


@pytest.mark.parametrize('dtype',['String','Boolean','Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64','Float32','Float64'])
def test_parquet_schema_scalar_choices(dtype):
    expected='seven' if dtype=='String' else True if dtype=='Boolean' else 1.5 if dtype.startswith('Float') else 7
    names,rows,types=parquet({'columns':[dtype],'schema':{dtype:dtype},'extra_columns':'ignore'},'scalar_types.parquet')
    assert names==(dtype,) and rows==[(expected,)]
    assert types==(('string',) if dtype=='String' else ('bool',) if dtype=='Boolean' else ('float64',) if dtype.startswith('Float') else ('int64',))


@pytest.mark.parametrize('arrow',[False,True])
def test_parquet_no_row_index_and_zero_index_offset(arrow):
    assert parquet({'use_pyarrow':arrow,'row_index_name':'','row_index_offset':0})[0]==('a','b')
    assert parquet({'use_pyarrow':arrow,'row_index_name':'row','row_index_offset':0})[1][0]==(0,1,'x')


@pytest.mark.parametrize('ending,payload', [('lf',b'a,b\r\n1,2\r\n'),('crlf',b'a,b\n1,2\n'),('cr',b'a,b\n1,2\n')])
def test_compatibility_csv_strict_record_endings_refuse_other_endings(tmp_path,ending,payload):
    path=temporary(tmp_path,payload)
    assert values(compatibility_csv(path,options={'recordEndings':'auto'}))[1]==[('1','2')]
    with pytest.raises(ValueError):compatibility_csv(path,options={'recordEndings':ending})


OPTION_CASES = {
    'csv.compatibility.separator': ['test_compatibility_csv_separator_choices'],
    'csv.compatibility.quoteChar': ['test_compatibility_csv_quote_choices'],
    'csv.compatibility.escapeMode': ['test_compatibility_csv_escape_modes','test_compatibility_csv_double_or_disabled_escaping','test_compatibility_csv_escape_disabled_preserves_extra_quotes'],
    'csv.compatibility.escapeChar': ['test_compatibility_csv_escape_modes'],
    'csv.compatibility.encoding': ['test_compatibility_csv_encodings','test_compatibility_csv_utf8_preserves_bom_and_generated_header_skip'],
    'csv.compatibility.recordEndings': ['test_compatibility_csv_record_endings','test_compatibility_csv_strict_record_endings_refuse_other_endings'],
    'csv.compatibility.allowQuotedNewlines': ['test_compatibility_csv_quoted_newlines'],
    'csv.compatibility.headerMode': ['test_compatibility_csv_utf8_preserves_bom_and_generated_header_skip','test_compatibility_csv_separator_choices'],
    'csv.compatibility.skipRecords': ['test_compatibility_csv_utf8_preserves_bom_and_generated_header_skip','test_compatibility_csv_separator_choices'],
    'csv.compatibility.skipBlankRecords': ['test_compatibility_csv_blank_records'],
    'csv.compatibility.skipInitialSpace': ['test_compatibility_csv_initial_spaces'],
    'csv.compatibility.emptyField': ['test_compatibility_csv_empty_field_and_null_markers'],
    'csv.compatibility.nullMarkers': ['test_compatibility_csv_empty_field_and_null_markers'],
    'csv.polars.separator': ['test_csv_separator_choices'],
    'csv.polars.quoteChar': ['test_csv_quote_choices','test_csv_no_quote_has_literal_quotes'],
    'csv.polars.encoding': ['test_csv_encoding_choices','test_csv_strict_utf8_refuses_bad_bytes'],
    'csv.polars.headerMode': ['test_csv_header_mode'],
    'csv.polars.skipRecords': ['test_csv_skip_and_comment_options','test_csv_header_mode'],
    'csv.polars.skipLines': ['test_csv_skip_and_comment_options','test_csv_header_mode'],
    'csv.polars.skipRowsAfterHeader': ['test_csv_skip_and_comment_options','test_csv_header_mode'],
    'csv.polars.commentPrefix': ['test_csv_skip_and_comment_options','test_csv_header_mode'],
    'csv.polars.emptyField': ['test_csv_empty_field_choices'],
    'csv.polars.nullMarkers': ['test_csv_global_and_per_column_null_markers','test_csv_empty_field_choices'],
    'csv.polars.columnNullValues': ['test_csv_global_and_per_column_null_markers'],
    'csv.polars.columns': ['test_csv_columns_choices'],
    'csv.polars.schema': ['test_csv_scalar_schema_choices'],
    'csv.polars.schemaOverrides': ['test_csv_scalar_schema_choices'],
    'csv.polars.inferSchema': ['test_csv_infer_schema_choice'],
    'csv.polars.inferSchemaLength': ['test_csv_inference_optional_and_zero','test_csv_short_inference_window_exposes_later_type_error','test_csv_infer_schema_choice'],
    'csv.polars.nRows': ['test_csv_n_rows_optional_and_zero'],
    'csv.polars.lowMemory': ['test_csv_low_memory_real_read_and_dispatch'],
    'csv.polars.tryParseDates': ['test_csv_date_parse_choice_and_typed_boundary'],
    'csv.polars.ignoreErrors': ['test_csv_ignore_errors_choice'],
    'csv.polars.decimalComma': ['test_csv_decimal_comma_and_rename_index'],
    'csv.polars.newColumns': ['test_csv_decimal_comma_and_rename_index','test_csv_header_mode'],
    'csv.polars.rowIndexName': ['test_csv_decimal_comma_and_rename_index','test_csv_header_mode'],
    'csv.polars.rowIndexOffset': ['test_csv_decimal_comma_and_rename_index'],
    'csv.polars.eolChar': ['test_csv_record_terminator_choices'],
    'csv.polars.extraColumns': ['test_csv_extra_missing_and_empty_choices'],
    'csv.polars.missingColumns': ['test_csv_extra_missing_and_empty_choices'],
    'csv.polars.raiseIfEmpty': ['test_csv_extra_missing_and_empty_choices'],
    'parquet.native.columns': ['test_parquet_reader_and_projection_choices'],
    'parquet.native.n_rows': ['test_parquet_optional_limits_and_index'],
    'parquet.native.row_index_name': ['test_parquet_optional_limits_and_index','test_parquet_no_row_index_and_zero_index_offset'],
    'parquet.native.row_index_offset': ['test_parquet_optional_limits_and_index','test_parquet_no_row_index_and_zero_index_offset'],
    'parquet.native.parallel': ['test_parquet_parallel_real_read_dispatch'],
    'parquet.native.use_statistics': ['test_parquet_native_tuning_real_read_dispatch'],
    'parquet.native.low_memory': ['test_parquet_native_tuning_real_read_dispatch'],
    'parquet.native.schema': ['test_parquet_schema_scalar_choices','test_parquet_schema_missing_and_extra_choices'],
    'parquet.native.missing_columns': ['test_parquet_schema_missing_and_extra_choices'],
    'parquet.native.extra_columns': ['test_parquet_schema_missing_and_extra_choices'],
    'parquet.native.cast_options.integer_cast': ['test_parquet_integer_cast_choices'],
    'parquet.native.cast_options.float_cast': ['test_parquet_float_cast_choices','test_parquet_float_narrowing_has_measured_numeric_effect'],
    'parquet.native.cast_options.categorical_to_string': ['test_parquet_categorical_cast_choices'],
    'parquet.native.use_pyarrow': ['test_parquet_reader_and_projection_choices'],
    'parquet.pyarrow.columns': ['test_parquet_reader_and_projection_choices'],
    'parquet.pyarrow.n_rows': ['test_parquet_optional_limits_and_index'],
    'parquet.pyarrow.row_index_name': ['test_parquet_optional_limits_and_index','test_parquet_no_row_index_and_zero_index_offset'],
    'parquet.pyarrow.row_index_offset': ['test_parquet_optional_limits_and_index','test_parquet_no_row_index_and_zero_index_offset'],
    'parquet.pyarrow.use_pyarrow': ['test_parquet_reader_and_projection_choices'],
    'parquet.pyarrow.memory_map': ['test_parquet_arrow_options_real_read_and_dispatch'],
    'parquet.pyarrow.pyarrow_options.use_threads': ['test_parquet_arrow_options_real_read_and_dispatch'],
    'parquet.pyarrow.pyarrow_options.use_pandas_metadata': ['test_parquet_arrow_options_real_read_and_dispatch'],
    'parquet.pyarrow.pyarrow_options.buffer_size': ['test_parquet_arrow_options_real_read_and_dispatch'],
    'parquet.pyarrow.pyarrow_options.pre_buffer': ['test_parquet_arrow_options_real_read_and_dispatch'],
    'parquet.pyarrow.pyarrow_options.coerce_int96_timestamp_unit': ['test_parquet_arrow_options_real_read_and_dispatch'],
    'parquet.pyarrow.pyarrow_options.thrift_string_size_limit': ['test_parquet_arrow_options_real_read_and_dispatch','test_parquet_arrow_metadata_limit_refusal_is_real'],
    'parquet.pyarrow.pyarrow_options.thrift_container_size_limit': ['test_parquet_arrow_options_real_read_and_dispatch','test_parquet_arrow_metadata_limit_refusal_is_real'],
    'parquet.pyarrow.pyarrow_options.schema_depth_limit': ['test_parquet_arrow_options_real_read_and_dispatch'],
    'parquet.pyarrow.pyarrow_options.page_checksum_verification': ['test_parquet_arrow_options_real_read_and_dispatch','test_parquet_checksum_choice_detects_corrupted_page'],
}

# These strategies were checked by exact dispatch plus a real parse, not a speed,
# RSS, cache-hit, thread-count or pandas-index reconstruction benchmark.
DISPATCH_ONLY_OPTIONS = {
    'csv.polars.lowMemory', 'parquet.native.parallel', 'parquet.native.use_statistics',
    'parquet.native.low_memory', 'parquet.pyarrow.memory_map',
    'parquet.pyarrow.pyarrow_options.use_threads', 'parquet.pyarrow.pyarrow_options.use_pandas_metadata',
    'parquet.pyarrow.pyarrow_options.buffer_size', 'parquet.pyarrow.pyarrow_options.pre_buffer',
    'parquet.pyarrow.pyarrow_options.schema_depth_limit',
}


def test_option_case_references_are_local_real_tests():
    assert OPTION_CASES and DISPATCH_ONLY_OPTIONS <= OPTION_CASES.keys()
    assert all(names and all(name.startswith('test_') and callable(globals().get(name)) for name in names) for names in OPTION_CASES.values())


def test_parquet_arrow_schema_is_explicitly_refused():
    assert parquet({'use_pyarrow':True,'schema':{}})[1]==[(1,'x'),(2,'y'),(3,'z')]
    with pytest.raises(ValueError,match='schema'):
        parquet({'use_pyarrow':True,'schema':{'a':'Int64','b':'String'}})
