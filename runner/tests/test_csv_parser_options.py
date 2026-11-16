import pytest
from runner.dataflow import source_csv
from runner.data import inspect_data


def source(tmp_path, raw, **options):
    p=tmp_path/'table.csv';p.write_bytes(raw)
    return source_csv(p, options=options)


def test_custom_separator_quotes_multiline_and_text(tmp_path):
    t=source(tmp_path,b"id^note^empty\r\n001^'hello\r\nworld'^\r\n",separator='^',quoteChar="'",emptyField='text')
    assert t.rows==(('001','hello\r\nworld',''),)


def test_headerless_skip_encoding_nulls(tmp_path):
    t=source(tmp_path,'ignore\n001; café;NA\n'.encode('cp1252'),separator=';',encoding='cp1252',headerMode='generated',skipRecords=1,skipInitialSpace=True,nullMarkers=['NA'])
    assert [c['name'] for c in t.columns]==['column_1','column_2','column_3']
    assert t.rows==( ('001','café',None), )


def test_inspection_execution_parity(tmp_path):
    p=tmp_path/'table.csv';p.write_text('a,b\n,NA\n')
    opts={'emptyField':'text','nullMarkers':['NA']}
    t=source_csv(p,options=opts);preview=inspect_data(p,options=opts)
    assert preview['rows']==[['','']]
    assert preview['nullCells']==[[0,1]]
    assert preview['parsing']['validatedRows']==1 and preview['parsing']['previewRows']==1


@pytest.mark.parametrize('options,raw,code',[
 ({'recordEndings':'crlf'},b'a,b\n1,2\n','recordEnding'),
 ({'allowQuotedNewlines':False},b'a,b\n1,"two\nlines"\n','quotedNewline'),
 ({},b'a,b\n1\n','rowWidth'),
 ({},b'a,b\n1,"unfinished\n','syntax'),
])
def test_actionable_diagnostics(tmp_path,options,raw,code):
    with pytest.raises(ValueError) as e:source(tmp_path,raw,**options)
    assert e.value.code==code
    assert e.value.line>=1
    assert 'unfinished' not in str(e.value)


def test_no_quote_backslash_and_blanks(tmp_path):
    t=source(tmp_path,b'a,b\n\n"x",hello\\,world\n',quoteChar='',escapeMode='backslash',skipBlankRecords=True)
    assert t.rows==(('"x"','hello,world'),)


def test_record_endings_do_not_rewrite_embedded_newlines(tmp_path):
    assert source(tmp_path,b'a,b\r\n1,"x\ny"\r\n',recordEndings='crlf').rows==( ('1','x\ny'), )


def test_bounded_and_cancelled(tmp_path):
    from runner.csv_parser import parse_csv
    p=tmp_path/'table.csv';p.write_text('a,b\n1,2\n3,4\n')
    for kwargs in ({'max_bytes':2},{'max_rows':1},{'max_columns':1}):
        with pytest.raises(ValueError):parse_csv(p,**kwargs)
    def cancel():raise InterruptedError()
    with pytest.raises(InterruptedError):parse_csv(p,check_cancelled=cancel)


@pytest.mark.parametrize('raw',['null','[]','{"separator":",","separator":";"}','{"skipRecords":true}','{"quoteChar":","}'])
def test_reject_invalid_option_transport(raw):
    from runner.csv_parser import parse_options_json
    with pytest.raises(ValueError):parse_options_json(raw)


def test_custom_unicode_quote_and_doubled_quotes(tmp_path):
    t=source(tmp_path,'id,note\n001,§hello, §§world§§§\n'.encode(),quoteChar='§')
    assert t.rows==( ('001','hello, §world§'), )


def test_legacy_preparation_defaults_preserved(tmp_path):
    p=tmp_path/'table.csv';p.write_text('a,b\n1,\n')
    assert source_csv(p).rows==( ('1',None), )
    p.write_text('a,b\n\n1,2\n')
    with pytest.raises(ValueError,match='row width mismatch'):source_csv(p)


@pytest.mark.parametrize('options',[{'nullMarkers':['']},{'nullMarkers':['NA','NA']},{'quoteChar':'\\','escapeMode':'backslash'},{'quoteChar':'\ud800'}])
def test_reject_ambiguous_settings(tmp_path,options):
    with pytest.raises(ValueError):source(tmp_path,b'a,b\n1,2\n',**options)


def test_legacy_exact_source_and_table_identity(tmp_path):
    import hashlib
    from runner.dataflow import digest, membership_digest
    raw=b'\xef\xbb\xbf id ,note,empty\r\n001,"two\r\nlines",\r\n002,"quoted ""word""",x\r\n'
    p=tmp_path/'legacy.csv';p.write_bytes(raw);table=source_csv(p)
    source_hash=hashlib.sha256(raw).hexdigest()
    expected_rows=(('001','two\r\nlines',None),('002','quoted "word"','x'))
    expected_columns=tuple({'name':name,'dtype':'string','nullable':name=='empty'} for name in [' id ','note','empty'])
    expected_ids=tuple(f'{source_hash}:{i}' for i in range(2))
    assert table.rows==expected_rows and table.columns==expected_columns
    assert table.source_digest==source_hash and table.row_ids==expected_ids
    assert table.data_version==digest({'columns':expected_columns,'rows':expected_rows,'ids':expected_ids})
    assert table.evidence()['membershipDigest']==membership_digest(expected_ids)
    assert source_csv(p,options={})==table


@pytest.mark.parametrize('delimiter',['\ud800','\r','\n','\0'])
def test_reject_surrogate_and_record_control_delimiters(tmp_path,delimiter):
    with pytest.raises(ValueError):source(tmp_path,b'a,b\n1,2\n',separator=delimiter)


def test_custom_control_delimiter_matches_swift_contract(tmp_path):
    assert source(tmp_path,b'a\x1fb\n1\x1f2\n',separator='\x1f').rows==( ('1','2'), )


def test_custom_escape_inspection_execution(tmp_path):
    p=tmp_path/'custom.csv'
    p.write_text('id,note\n001,"say ~"hello~""\n002,hello~,world\n')
    opts=dict(escapeMode='custom',escapeChar='~')
    assert source_csv(p,options=opts).rows == (('001','say "hello"'),('002','hello,world'))
    assert inspect_data(p,options=opts)['rows'] == [['001','say "hello"'],['002','hello,world']]


@pytest.mark.parametrize('char',['', 'xx', '\n', '\0', ',', '"'])
def test_custom_escape_invalid_characters(tmp_path,char):
    with pytest.raises(ValueError):source(tmp_path,b'a,b\n1,2\n',escapeMode='custom',escapeChar=char)


def test_preview_stops_before_invalid_tail_and_large_row_count(tmp_path):
    p=tmp_path/'large.csv';p.write_bytes(b'a,b\n'+b'1,2\n'*500001+b'broken\n')
    preview=inspect_data(p,options={})
    assert len(preview['rows'])==8
    assert preview['rowCount']==8
    assert preview['parsing']['sampleOnly'] is True


def test_preview_byte_budget_is_not_a_truncated_record(tmp_path):
    from runner.csv_parser import parse_csv
    p=tmp_path/'large.csv';p.write_bytes(b'a,b\n1,"'+b'x'*2000000)
    with pytest.raises(ValueError) as error:
        parse_csv(p,{},preview=True)
    assert error.value.code=='sampleBudget'


def test_preview_reads_fixed_prefix_only(tmp_path,monkeypatch):
    from pathlib import Path
    from runner.csv_parser import parse_csv
    p=tmp_path/'huge.csv';p.write_bytes(b'a,b\n'+b'1,2\n'*3_000_000)
    original=Path.open; observed=[]
    class Observed:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def read(self,count):
            block=self.stream.read(count);observed.append(len(block));return block
    monkeypatch.setattr(Path,'open',lambda path,*args,**kwargs:Observed(original(path,*args,**kwargs)))
    _,rows,_,_=parse_csv(p,{},preview=True)
    assert len(rows)==8
    assert sum(observed)==1_048_576


def test_preview_does_not_validate_records_after_sample(tmp_path):
    p=tmp_path/'tail.csv';p.write_bytes(b'a,b\n'+b'1,2\n'*8+b'bad_width\n')
    assert len(inspect_data(p,options={})['rows'])==8
    with pytest.raises(ValueError,match='width'):source_csv(p,options={})
