"""Independent real documents, strict options and bounded extraction evidence."""
import importlib.util
from pathlib import Path
import pytest
from runner.file_input_documents import extract_document

spec=importlib.util.spec_from_file_location('document_fixtures',Path(__file__).parent/'fixtures/file_input/documents/generate.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

def read(raw,fmt,**options):return list(extract_document(raw,fmt,options,lambda:None))
def texts(rows):return [row['text_content'] for row in rows]

def test_text_default_bom_and_file_unit():
    assert read(b'\xef\xbb\xbfA\r\nB','text')==[{'text_content':'A\r\nB','unit_index':0}]

@pytest.mark.parametrize('encoding', ['utf-8','utf-16','utf-32','cp1252','latin-1'])
def test_text_encoding_choices(encoding):
    assert texts(read('café'.encode(encoding),'text',encoding=encoding))==['café']

@pytest.mark.parametrize('errors,expected',[('replace','A�'),('ignore','A')])
def test_text_bad_bytes_policy(errors,expected):
    assert texts(read(b'A\xff','text',errors=errors))==[expected]
    with pytest.raises(UnicodeDecodeError):read(b'A\xff','text',errors='strict')

def test_text_line_units_newline_and_blank_policy():
    assert texts(read(b'A\r\n\rB\n','text',newline='lf'))==['A\n\nB\n']
    assert texts(read(b'A\r\n\rB\n','text',newline='crlf'))==['A\r\n\r\nB\r\n']
    rows=read(b'A\r\n\rB\n','text',units='line');assert texts(rows)==['A','','B']
    assert [row['unit_index'] for row in rows]==[0,1,2]
    assert texts(read(b'A\n\nB','text',units='line',keep_empty_lines=False))==['A','B']

@pytest.mark.parametrize('count',[100000,100001,200000])
def test_line_count_has_no_arbitrary_cutoff(count):
    seen=0
    for unit in extract_document(b'x\n'*count,'text',{'units':'line'}):
        assert unit=={'text_content':'x','unit_index':seen}
        seen+=1
    assert seen==count

def test_many_units_still_obey_bytes_and_cancellation(monkeypatch):
    from runner import file_input_documents as documents
    monkeypatch.setattr(documents,'MAX_TEXT_BYTES',10)
    with pytest.raises(ValueError,match='byte budget'):
        list(extract_document(b'xx\n'*6,'text',{'units':'line'}))
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):next(extract_document(b'x\n'*200000,'text',{'units':'line'},cancel))

def test_pdf_file_page_and_selected_order():
    raw=fixtures.pdf_bytes();assert texts(read(raw,'pdf'))==['First page\n\nSecond page']
    rows=read(raw,'pdf',units='page',pages=[2,1]);assert texts(rows)==['Second page','First page']
    assert [x['page_number'] for x in rows]==[2,1]
    assert [x['unit_index'] for x in rows]==[0,1]

@pytest.mark.parametrize('mode',['plain','layout'])
def test_pdf_real_plain_layout_configuration(mode):
    options={'extraction_mode':mode,'strict':True}
    if mode=='plain':options.update(orientations=[0],space_width=100)
    else:options.update(layout_space_vertically=False,layout_scale_weight=1.0,layout_strip_rotated=False,layout_font_height_weight=1.5)
    assert 'First page' in texts(read(fixtures.pdf_bytes(),'pdf',**options))[0]

def test_pdf_encryption_malformed_and_selection_refusal():
    with pytest.raises(ValueError,match='[Ee]ncrypt'):read(fixtures.pdf_bytes(True),'pdf')
    with pytest.raises(ValueError):read(b'not pdf','pdf')
    with pytest.raises(ValueError):read(fixtures.pdf_bytes(),'pdf',pages=[3])


def test_docx_document_order_nested_table_and_paragraphs():
    rows=read(fixtures.docx_bytes(),'docx',units='paragraph')
    assert texts(rows)==['First paragraph','Left','Nested','Right','Last paragraph']
    assert texts(read(fixtures.docx_bytes(),'docx',include_tables=False))==['First paragraph\n\nLast paragraph']

@pytest.mark.parametrize('headers,footers,expected',[(True,False,'Header'),(False,True,'Footer'),(True,True,'Header')])
def test_docx_headers_and_footers_are_explicit(headers,footers,expected):
    output=texts(read(fixtures.docx_bytes(),'docx',include_headers=headers,include_footers=footers))[0]
    assert expected in output
    assert ('Footer' in output)==footers and ('Header' in output)==headers


def test_rtf_real_encoding_unicode_and_paragraph_units():
    raw=b"{\\rtf1\\ansi Caf\\'e9\\par Next}"
    assert texts(read(raw,'rtf'))==['Café\n\nNext']
    assert texts(read(raw,'rtf',units='paragraph',encoding='cp1252',errors='strict'))==['Café','Next']
    assert texts(read(b"{\\rtf1\\ansi \\'ff}",'rtf',encoding='ascii',errors='replace'))==['�']
    assert texts(read(b"{\\rtf1\\ansi \\'ff}",'rtf',encoding='ascii',errors='ignore'))==['']


def test_odt_order_whitespace_lists_and_table_policy():
    assert texts(read(fixtures.odt_bytes(),'odt',units='paragraph'))==['Title','A   B\tC\nD','List item','Cell','End']
    assert texts(read(fixtures.odt_bytes(),'odt',include_tables=False))==['Title\n\nA   B\tC\nD\n\nList item\n\nEnd']

@pytest.mark.parametrize('fmt',['docx','odt'])
def test_zip_safety_entities_and_expansion_budget(fmt,monkeypatch):
    import runner.file_input_documents as module
    with pytest.raises(ValueError):read(b'not zip',fmt)
    raw=fixtures.docx_bytes() if fmt=='docx' else fixtures.odt_bytes()
    monkeypatch.setattr(module,'MAX_EXPANDED_BYTES',128)
    with pytest.raises(ValueError,match='expanded'):read(raw,fmt)


def test_odt_dtd_entities_external_and_repeat_bombs_refused():
    evil='<!DOCTYPE x [<!ENTITY a SYSTEM "file:///etc/passwd">]><x>&a;</x>'
    with pytest.raises(ValueError,match='entit|DTD'):read(fixtures.odt_bytes(evil),'odt')
    repeat='<x xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><text:p><text:s text:c="999999999"/></text:p></x>'
    with pytest.raises(ValueError):read(fixtures.odt_bytes(repeat),'odt')

@pytest.mark.parametrize('fmt,options',[('text',{'errors':'silent'}),('pdf',{'password':'x'}),('pdf',{'pages':[0]}),('pdf',{'orientations':[45]}),('docx',{'include_tables':1}),('odt',{'include_tables':'yes'}),('rtf',{'unknown':True})])
def test_unknown_and_invalid_reader_options_refused(fmt,options):
    with pytest.raises(ValueError):read(b'',fmt,**options)


def test_capture_output_units_and_cancellation_bounded(monkeypatch):
    import runner.file_input_documents as module
    monkeypatch.setattr(module,'MAX_CAPTURE_BYTES',3)
    with pytest.raises(ValueError,match='capture'):read(b'abcd','text')
    monkeypatch.setattr(module,'MAX_CAPTURE_BYTES',64*1024*1024)
    monkeypatch.setattr(module,'MAX_TEXT_BYTES',3)
    with pytest.raises(ValueError,match='text'):read('éé'.encode(),'text')
    monkeypatch.setattr(module,'MAX_TEXT_BYTES',64*1024*1024)
    assert texts(read(b'A\nB','text',units='line'))==['A','B']
    def cancel():raise RuntimeError('cancelled independently')
    with pytest.raises(RuntimeError,match='cancelled'):list(extract_document(b'A','text',{},cancel))

@pytest.mark.parametrize('encoding',['utf-8-sig','utf-16','utf-32'])
def test_text_auto_detects_only_bom_encodings(encoding):
    assert texts(read('café'.encode(encoding),'text',encoding='auto'))==['café']
    with pytest.raises(UnicodeDecodeError):read('café'.encode('cp1252'),'text',encoding='auto')


def test_pdf_orientation_selection_and_layout_rotated_policy():
    raw=fixtures.pdf_bytes(rotated=True)
    assert texts(read(raw,'pdf',orientations=[0]))==['\n\n']
    assert texts(read(raw,'pdf',orientations=[90]))==['First page\n\nSecond page']
    assert texts(read(raw,'pdf',extraction_mode='layout',layout_strip_rotated=True))==['\n\n']
    assert 'First page' in texts(read(raw,'pdf',extraction_mode='layout',layout_strip_rotated=False))[0]


@pytest.mark.parametrize('fmt',['pdf','docx','rtf','odt'])
def test_document_custom_separator(fmt):
    raw=fixtures.pdf_bytes() if fmt=='pdf' else fixtures.docx_bytes() if fmt=='docx' else fixtures.odt_bytes() if fmt=='odt' else b'{\\rtf1 A\\par B}'
    value=texts(read(raw,fmt,separator=' | '))[0]
    assert ' | ' in value and '\n\n' not in value


def test_docx_nested_tables_can_be_excluded_independently():
    output=texts(read(fixtures.docx_bytes(),'docx',units='paragraph',include_nested_tables=False))
    assert output==['First paragraph','Left','Right','Last paragraph']


def test_docx_zip_xml_dtd_refused_before_parser():
    import zipfile,io
    raw=fixtures.docx_bytes();output=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as original,zipfile.ZipFile(output,'w') as changed:
        for name in original.namelist():
            data=original.read(name)
            if name=='word/document.xml':data=b'<!DOCTYPE x [<!ENTITY test "private">]>'+data
            changed.writestr(name,data)
    with pytest.raises(ValueError,match='DTD'):read(output.getvalue(),'docx')


def test_zip_unsafe_duplicate_members_refused():
    import io,zipfile,warnings
    for names in (['../escape'],['content.xml','content.xml']):
        output=io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(output,'w') as archive:
                for name in names:archive.writestr(name,b'x')
        with pytest.raises(ValueError,match='Unsafe|duplicate'):read(output.getvalue(),'odt')


@pytest.mark.parametrize('option,value',[('strict',1),('pages',[1,1]),('space_width',0),('layout_scale_weight',float('nan')),('layout_font_height_weight',float('inf')),('layout_space_vertically','no'),('separator','x'*1025)])
def test_pdf_strict_invalid_options(option,value):
    with pytest.raises(ValueError):read(b'','pdf',**{option:value})


def test_pdf_aggregate_expansion_and_page_budget(monkeypatch):
    import runner.file_input_documents as module
    monkeypatch.setattr(module,'MAX_EXPANDED_BYTES',70)
    with pytest.raises(ValueError,match='expanded'):read(fixtures.pdf_bytes(),'pdf')
    monkeypatch.setattr(module,'MAX_PAGES',1)
    with pytest.raises(ValueError,match='page budget'):read(fixtures.pdf_bytes(),'pdf')


def test_odt_repeated_space_expansion_is_admitted_before_allocation(monkeypatch):
    import runner.file_input_documents as module
    raw=fixtures.odt_bytes().replace(b'never-matches',b'')
    monkeypatch.setattr(module,'MAX_TEXT_BYTES',6)
    with pytest.raises(ValueError,match='text'):read(raw,'odt')


def test_required_document_package_closure_is_reader_specific():
    from runner.file_input_documents import required_document_packages
    assert required_document_packages(['text'])==set()
    assert required_document_packages(['pdf'])=={'pypdf'}
    assert required_document_packages(['docx','rtf','odt'])=={'python-docx','lxml','typing-extensions','striprtf','defusedxml'}


OPTION_CASES={
 'text.encoding':['test_text_encoding_choices','test_text_auto_detects_only_bom_encodings'],
 'text.errors':['test_text_bad_bytes_policy'],
 'text.newline':['test_text_line_units_newline_and_blank_policy'],
 'text.units':['test_text_line_units_newline_and_blank_policy','test_text_default_bom_and_file_unit'],
 'text.keep_empty_lines':['test_text_line_units_newline_and_blank_policy'],
 'pdf.units':['test_pdf_file_page_and_selected_order'],
 'pdf.pages':['test_pdf_file_page_and_selected_order','test_pdf_encryption_malformed_and_selection_refusal'],
 'pdf.strict':['test_pdf_real_plain_layout_configuration'],
 'pdf.extraction_mode':['test_pdf_real_plain_layout_configuration','test_pdf_orientation_selection_and_layout_rotated_policy'],
 'pdf.orientations':['test_pdf_orientation_selection_and_layout_rotated_policy'],
 'pdf.space_width':['test_pdf_real_plain_layout_configuration'],
 'pdf.layout_space_vertically':['test_pdf_real_plain_layout_configuration'],
 'pdf.layout_scale_weight':['test_pdf_real_plain_layout_configuration'],
 'pdf.layout_strip_rotated':['test_pdf_orientation_selection_and_layout_rotated_policy'],
 'pdf.layout_font_height_weight':['test_pdf_real_plain_layout_configuration'],
 'pdf.separator':['test_document_custom_separator'],
 'docx.units':['test_docx_document_order_nested_table_and_paragraphs'],
 'docx.include_tables':['test_docx_document_order_nested_table_and_paragraphs'],
 'docx.include_nested_tables':['test_docx_nested_tables_can_be_excluded_independently'],
 'docx.include_headers':['test_docx_headers_and_footers_are_explicit'],
 'docx.include_footers':['test_docx_headers_and_footers_are_explicit'],
 'docx.separator':['test_document_custom_separator'],
 'rtf.encoding':['test_rtf_real_encoding_unicode_and_paragraph_units'],
 'rtf.errors':['test_rtf_real_encoding_unicode_and_paragraph_units'],
 'rtf.units':['test_rtf_real_encoding_unicode_and_paragraph_units'],
 'rtf.separator':['test_document_custom_separator'],
 'odt.units':['test_odt_order_whitespace_lists_and_table_policy'],
 'odt.include_tables':['test_odt_order_whitespace_lists_and_table_policy'],
 'odt.separator':['test_document_custom_separator'],
}
# These tested numeric/strictness controls affect parsing tolerance/heuristics;
# the simple independent document does not demonstrate each visual layout change.
OPERATIONAL_OPTIONS={'pdf.strict','pdf.space_width','pdf.layout_space_vertically','pdf.layout_scale_weight','pdf.layout_font_height_weight'}


def test_pdf_layout_spacing_parameters_change_real_geometry():
    raw=fixtures.pdf_bytes(layout=True)
    def output(**options):return texts(read(raw,'pdf',units='page',pages=[1],extraction_mode='layout',**options))[0]
    default=output()
    assert default=='Left                        Right\n\n\n\n\n                             Down'
    assert output(layout_space_vertically=False)=='Left                        Right\n                             Down'
    assert output(layout_font_height_weight=2.0)=='Left                        Right\n\n                             Down'
    assert output(layout_scale_weight=0.5)=='Left                        Right\n\n\n\n\n           Down'
    assert output(layout_scale_weight=2.0)=='Left                        Right\n\n\n\n\n                                              Down'


def test_pdf_strictness_controls_header_recovery():
    raw=fixtures.pdf_bytes();bad_header=b'XXXXX'+raw[5:]
    with pytest.raises(ValueError,match='Malformed PDF'):read(bad_header,'pdf',strict=True)
    assert texts(read(bad_header,'pdf',strict=False))==['First page\n\nSecond page']


OPTION_CASES['pdf.strict'].append('test_pdf_strictness_controls_header_recovery')
for key in ('pdf.layout_space_vertically','pdf.layout_scale_weight','pdf.layout_font_height_weight'):
    OPTION_CASES[key].append('test_pdf_layout_spacing_parameters_change_real_geometry')
OPERATIONAL_OPTIONS={'pdf.space_width'}


def test_docx_uppercase_relationship_xml_is_entity_admitted_before_parser():
    import io,zipfile
    original=fixtures.docx_bytes();output=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source,zipfile.ZipFile(output,'w') as target:
        for name in source.namelist():
            raw=source.read(name)
            if name=='word/document.xml':
                name='word/CONTENT.XML'
                end=raw.index(b'?>')+2
                raw=raw[:end]+b'<!DOCTYPE x [<!ENTITY forbidden "fixture">]>'+raw[end:]
            elif name=='_rels/.rels':raw=raw.replace(b'word/document.xml',b'word/CONTENT.XML')
            elif name=='[Content_Types].xml':raw=raw.replace(b'/word/document.xml',b'/word/CONTENT.XML')
            target.writestr(name,raw)
    with pytest.raises(ValueError,match='DTD/entities'):read(output.getvalue(),'docx')
