"""Native Polars CSV profile: explicit dialect, stable rows and bounded storage."""
import hashlib
from pathlib import Path
import pytest
pytest.importorskip('polars')
from runner.polars_csv import csv_arguments, source_csv, preview_csv
from runner.dataflow_columnar import ColumnarRows
from runner.dataflow_disk import disk_scope


def test_native_fields_map_without_inference_or_remote_globbing():
    args = csv_arguments({'parserProfile':'polars','separator':';','quoteChar':'',
                         'encoding':'utf8-lossy','headerMode':'generated','skipRecords':2,
                         'skipLines':0,'skipRowsAfterHeader':3,'commentPrefix':'#',
                         'emptyField':'text','nullMarkers':['NA']})
    assert args['separator']==';' and args['quote_char'] is None
    assert args['encoding']=='utf8-lossy' and args['has_header'] is False
    assert args['skip_rows']==2 and args['skip_lines']==0 and args['skip_rows_after_header']==3
    assert args['comment_prefix']=='#' and args['null_values']==['NA']
    assert args['empty_string_is_null'] is False and args['infer_schema'] is False
    assert args['glob'] is False and args['ignore_errors'] is False


@pytest.mark.parametrize('option', [ {'separator':'§'}, {'quoteChar':'§'}, {'escapeMode':'backslash'},
    {'escapeMode':'custom'}, {'recordEndings':'crlf'}, {'allowQuotedNewlines':False},
    {'skipInitialSpace':True}, {'encoding':'cp1252'}, {'unknown':True}, {'skipRecords':True},
    {'skipRecords':1,'skipLines':1}])
def test_unsupported_configuration_is_explicitly_rejected(option):
    with pytest.raises(ValueError): csv_arguments(option)


def test_source_native_values_string_types_nulls_bom_and_ids(tmp_path):
    payload=b'\xef\xbb\xbfa;b;c\n001;"two\nlines";NA\n002;"a ""quote""";\n'
    path=tmp_path/'native.csv';path.write_bytes(payload)
    result=source_csv(path, {'separator':';','nullMarkers':['NA'],'emptyField':'text'})
    assert isinstance(result.rows,ColumnarRows)
    assert list(result.rows)==[('001','two\nlines',None),('002','a "quote"','')]
    assert result.columns==({'name':'a','dtype':'string','nullable':False},
                           {'name':'b','dtype':'string','nullable':False},
                           {'name':'c','dtype':'string','nullable':True})
    digest=hashlib.sha256(payload).hexdigest()
    assert result.source_digest==digest and list(result.row_ids)==[digest+':0',digest+':1']
    assert result.data_version
    result.rows.store.close()


def test_native_comment_skip_and_generated_headers(tmp_path):
    path=tmp_path/'native.csv';path.write_text('ignored physical\nignored;record\n# comment\na;b\ndiscard;me\nx;y\n')
    result=source_csv(path, {'separator':';','skipLines':2,
                            'skipRowsAfterHeader':1,'commentPrefix':'#'})
    assert [column['name'] for column in result.columns]==['a','b']
    assert list(result.rows)==[('x','y')]
    result.rows.store.close()
    result=source_csv(path, {'separator':';','skipRecords':2,
                            'headerMode':'generated','commentPrefix':'#'})
    assert [column['name'] for column in result.columns]==['column_0','column_1']
    assert list(result.rows)==[('a','b'),('discard','me'),('x','y')]
    result.rows.store.close()


def test_native_blank_records_and_duplicate_headers_are_polars_semantics(tmp_path):
    path=tmp_path/'native.csv';path.write_text('a;a\n\nx;y\n\n')
    result=source_csv(path, {'separator':';','skipBlankRecords':False})
    assert [column['name'] for column in result.columns]==['a','a_duplicated_0']
    # Native all-string parsing retains blank records with the chosen empty policy.
    assert list(result.rows)==[('', ''),('x','y'),('', '')]
    result.rows.store.close()


def test_utf8_lossy_is_explicit_and_legacy_bom_preservation_refused(tmp_path):
    path=tmp_path/'invalid.csv';path.write_bytes(b'a\n\xff\n')
    with pytest.raises(ValueError): source_csv(path)
    result=source_csv(path, {'encoding':'utf8-lossy'})
    assert list(result.rows)==[('\ufffd',)]
    result.rows.store.close()
    path.write_bytes(b'\xef\xbb\xbfa\nx\n')
    with pytest.raises(ValueError,match='BOM'): source_csv(path,{'encoding':'utf-8'})


def test_source_large_rows_no_100k_cap_order_and_cleanup(tmp_path):
    path=tmp_path/'large.csv';path.write_text('a;b\n'+'1;2\n'*100_001)
    with disk_scope(directory=tmp_path) as budget:
        result=source_csv(path,{'separator':';'})
        assert len(result.rows)==100_001 and result.rows[-1]==('1','2')
        assert result.row_ids[-1].endswith(':100000') and budget.used>0
        assert not list(tmp_path.glob('.polars-csv-*'))
        result.rows.store.close()
        assert budget.used==0


def test_source_shared_scratch_refusal_and_cancellation_clean_up(tmp_path):
    path=tmp_path/'large.csv';path.write_text('a;b\n'+'1;2\n'*5000)
    with disk_scope(directory=tmp_path,budget=20_000):
        with pytest.raises(ValueError,match='scratch'): source_csv(path,{'separator':';'})
    assert not list(tmp_path.glob('.polars-csv-*')) and not list(tmp_path.glob('.table-columns-*'))
    calls=0
    def cancelled():
        nonlocal calls
        calls+=1
        if calls==12:raise RuntimeError('cancelled')
    with disk_scope(directory=tmp_path):
        with pytest.raises(RuntimeError,match='cancelled'):source_csv(path,{'separator':';'},check_cancelled=cancelled)
    assert not list(tmp_path.glob('.polars-csv-*')) and not list(tmp_path.glob('.table-columns-*'))


def test_source_regular_local_only_and_mutation_refusal(tmp_path):
    path=tmp_path/'native.csv';path.write_text('a\nx\n')
    link=tmp_path/'link.csv';link.symlink_to(path)
    for candidate in (link,tmp_path,'https://example.com/data.csv',str(tmp_path/'*.csv')):
        with pytest.raises(ValueError):source_csv(candidate)
    calls=0
    def changed():
        nonlocal calls
        calls+=1
        if calls==8:path.write_text('a\ny\n')
    with pytest.raises(ValueError,match='changed'):source_csv(path,check_cancelled=changed)


def test_preview_bounded_rows_and_truncated_tail_does_not_invalidate_first_eight(tmp_path):
    path=tmp_path/'prefix.csv';path.write_bytes(b'a;b\n'+b'1;2\n'*8+b'"truncated')
    header,rows=preview_csv(path,{'separator':';'})
    assert header==['a','b'] and rows==[['1','2']]*8
    path.write_bytes(b'a;b\n1;"two\nlines"\n2;last\n')
    assert preview_csv(path,{'separator':';'})==(['a','b'],[['1','two\nlines'],['2','last']])
    path.write_bytes(b'a;b\n'+b'1;2\n'*8+b'\xff'*1_100_000)
    assert preview_csv(path,{'separator':';'})[1]==[['1','2']]*8


def test_configuration_admission_imports_only_standard_library():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',
        "import sys; from runner.polars_csv_configuration import normalize_options; "
        "assert normalize_options({'parserProfile':'polars'})['encoding']=='utf-8-sig'; "
        "assert 'polars' not in sys.modules and 'numpy' not in sys.modules"],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr


def test_source_byte_column_and_cell_limits_refuse_without_retained_scratch(tmp_path,monkeypatch):
    import runner.polars_csv as adapter
    path=tmp_path/'wide.csv'
    path.write_text(';'.join('c'+str(i) for i in range(129))+'\n'+';'.join('v' for _ in range(129))+'\n')
    with disk_scope(directory=tmp_path):
        with pytest.raises(ValueError,match='128 column'):source_csv(path,{'separator':';'})
    path.write_text('a;b;c\n1;2;3\n4;5;6\n')
    monkeypatch.setattr(adapter,'MAX_CELLS',5)
    with disk_scope(directory=tmp_path):
        with pytest.raises(ValueError,match='cell'):source_csv(path,{'separator':';'})
    with path.open('wb') as stream:stream.truncate(10_000_001)
    with pytest.raises(ValueError,match='10 MB'):source_csv(path)
    assert not list(tmp_path.glob('.polars-csv-*')) and not list(tmp_path.glob('.table-columns-*'))


def test_preview_native_comment_bom_and_quoted_newlines_are_same_as_source(tmp_path):
    path=tmp_path/'prefix.csv'
    path.write_bytes(b'\xef\xbb\xbf# comment "not a quoted CSV field\na;b\n1;"x\ny"\n'+b'2;z\n'*7+b'"truncated')
    header,rows=preview_csv(path,{'separator':';','commentPrefix':'#'})
    assert header==['a','b'] and rows==[['1','x\ny']]+[['2','z']]*7


def test_preview_comment_prefix_budget_uses_one_native_parse(tmp_path,monkeypatch):
    import runner.polars_csv as adapter
    original=adapter.pl.read_csv;calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(adapter.pl,'read_csv',counted)
    path=tmp_path/'comments.csv'
    path.write_bytes(b'a;b\n'+b'# comment\n'*100_000+b'1;2\n'*8+b'"truncated')
    assert preview_csv(path,{'separator':';','commentPrefix':'#'})==(['a','b'],[['1','2']]*8)
    assert len(calls)==1
    calls.clear()
    path.write_bytes(b'a;b\n'+b'# comment\n'*110_000)
    assert preview_csv(path,{'separator':';','commentPrefix':'#'})==(['a','b'],[])
    assert len(calls)==1


def test_native_encoding_aliases_normalize_to_ui_profile_choices():
    from runner.polars_csv_configuration import normalize_options
    assert normalize_options({'encoding':'utf8'})['encoding']=='utf-8-sig'
    assert normalize_options({'encoding':'utf-8'})['encoding']=='utf-8-sig'


def test_preview_hash_belongs_to_same_bounded_buffer_as_rows(tmp_path):
    import hashlib
    path=tmp_path/'source.csv';raw=b'a\n'+b'x\n'*600000;path.write_bytes(raw)
    header,rows,pin=preview_csv(path,include_source_hash=True)
    assert header==['a'] and rows==[['x']]*8
    assert pin==hashlib.sha256(raw[:1048576]).hexdigest()


def test_literal_local_filename_with_brackets_is_not_a_glob(tmp_path):
    path=tmp_path/'observations [export].csv';path.write_text('a\nx\n')
    assert tuple(source_csv(path).rows)==(('x',),)


def test_count_sample_limits_native_scan_but_hashes_complete_source(tmp_path,monkeypatch):
    import runner.polars_csv as adapter
    native=adapter.pl.scan_csv;calls=[]
    def observed(*args,**kwargs):
        calls.append(kwargs.get('n_rows'))
        return native(*args,**kwargs)
    monkeypatch.setattr(adapter.pl,'scan_csv',observed)
    payload=b'a;b\n'+b'1;2\n'*100_001
    path=tmp_path/'large.csv';path.write_bytes(payload)
    sampled=source_csv(path,{'separator':';'},sampling={'mode':'rowCount','rowCount':100})
    assert len(sampled.rows)==100 and calls==[100]
    digest=hashlib.sha256(payload).hexdigest()
    assert sampled.source_digest==digest and sampled.row_ids[-1]==digest+':99'
    sampled.rows.store.close()


def test_percentage_sample_uses_full_cardinality_but_retains_original_ids(tmp_path):
    payload=b'a\n'+b'x\n'*101
    path=tmp_path/'rows.csv';path.write_bytes(payload)
    sampled=source_csv(path,sampling={'mode':'percentage','percentage':10})
    assert len(sampled.rows)==11
    assert sampled.row_ids[-1]==hashlib.sha256(payload).hexdigest()+':10'
    sampled.rows.store.close()
