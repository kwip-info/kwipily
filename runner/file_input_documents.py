"""Offline, bounded document extraction from an already captured byte string.

No paths, external relationship targets, macros, passwords or callbacks in options.
Each yielded unit contains text and ordinal evidence, never a live source handle.
"""
import codecs
import io
import math
import re
import zipfile

MAX_CAPTURE_BYTES=64*1024*1024
MAX_EXPANDED_BYTES=128*1024*1024
MAX_TEXT_BYTES=64*1024*1024
MAX_PAGES=10000
ENCODINGS=('auto','utf-8','utf-8-sig','utf-16','utf-16-le','utf-16-be','utf-32','utf-32-le','utf-32-be','ascii','latin-1','cp1252')
DEFAULTS={
 'text':{'encoding':'auto','errors':'strict','newline':'preserve','units':'file','keep_empty_lines':True},
 'pdf':{'units':'file','pages':[],'strict':False,'extraction_mode':'plain','orientations':[0,90,180,270],'space_width':200.0,'layout_space_vertically':True,'layout_scale_weight':1.25,'layout_strip_rotated':True,'layout_font_height_weight':1.0,'separator':'\n\n'},
 'docx':{'units':'file','include_tables':True,'include_nested_tables':True,'include_headers':False,'include_footers':False,'separator':'\n\n'},
 'rtf':{'encoding':'cp1252','errors':'strict','units':'file','separator':'\n\n'},
 'odt':{'units':'file','include_tables':True,'separator':'\n\n'},
}


def require(condition,message):
    if not condition:raise ValueError(message)


def normalize_document_options(format,options):
    if format=='pdf':options={k:v for k,v in options.items() if k not in ('renderMode','renderDPI','maximumRenderPages')}
    require(format in DEFAULTS,'Unsupported document format')
    require(type(options) is dict and set(options)<=set(DEFAULTS[format]),'Unknown document reader option')
    selected={**DEFAULTS[format],**options}
    for key,value in selected.items():
        default=DEFAULTS[format][key]
        if type(default) is bool:require(type(value) is bool,'Document '+key+' must be boolean')
    if 'encoding' in selected:require(selected['encoding'] in ENCODINGS,'Unsupported text encoding')
    if 'errors' in selected:require(selected['errors'] in ('strict','replace','ignore'),'Unsupported decoding error policy')
    allowed=('file','line') if format=='text' else ('file','page') if format=='pdf' else ('file','paragraph')
    require(selected['units'] in allowed,'Unsupported document unit mode')
    if 'separator' in selected:require(type(selected['separator']) is str and len(selected['separator'].encode())<=1024,'Document separator exceeds its bound')
    if format=='text':require(selected['newline'] in ('preserve','lf','crlf'),'Unsupported newline policy')
    if format=='pdf':
        pages=selected['pages'];require(type(pages) is list and len(pages)<=MAX_PAGES and all(type(p) is int and 1<=p<=MAX_PAGES for p in pages) and len(set(pages))==len(pages),'PDF pages require distinct positive 1-based numbers')
        orientations=selected['orientations'];require(type(orientations) is list and bool(orientations) and all(type(p) is int and p in (0,90,180,270) for p in orientations) and len(set(orientations))==len(orientations),'Unsupported PDF text orientation')
        require(selected['extraction_mode'] in ('plain','layout'),'Unsupported PDF extraction mode')
        for key in ('space_width','layout_scale_weight','layout_font_height_weight'):
            value=selected[key];require(type(value) in (int,float) and math.isfinite(value) and 0<value<=10000,'PDF '+key+' must be positive and bounded')
    return selected


def required_document_packages(formats):
    packages=set()
    for format in formats:
        require(format in DEFAULTS,'Unsupported document format')
        packages|={'pypdf'} if format=='pdf' else {'python-docx','lxml','typing-extensions'} if format=='docx' else {'striprtf'} if format=='rtf' else {'defusedxml'} if format=='odt' else set()
    return packages


def _decode(raw,encoding,errors):
    if encoding=='auto':
        encoding='utf-32' if raw.startswith((codecs.BOM_UTF32_LE,codecs.BOM_UTF32_BE)) else 'utf-16' if raw.startswith((codecs.BOM_UTF16_LE,codecs.BOM_UTF16_BE)) else 'utf-8-sig'
    return raw.decode(encoding,errors)


def _archive(raw,check):
    try:archive=zipfile.ZipFile(io.BytesIO(raw))
    except (zipfile.BadZipFile,OSError) as error:raise ValueError('Malformed document ZIP container') from error
    try:
        entries=archive.infolist()
        require(len(entries)<=10000,'Document archive exceeds its entry budget')
        require(len({e.filename for e in entries})==len(entries),'Document archive has duplicate members')
        require(sum(e.file_size for e in entries)<=MAX_EXPANDED_BYTES,'Document expanded content exceeds its byte budget')
        for entry in entries:
            check();require(not entry.flag_bits&1,'Encrypted document ZIP is unsupported')
            require(entry.file_size<=MAX_CAPTURE_BYTES,'Document archive member exceeds its byte budget')
            require(not entry.filename.startswith(('/','\\')) and '\\' not in entry.filename and all(part not in ('..','') for part in entry.filename.rstrip('/').split('/')),'Unsafe document archive member')
            if entry.filename.lower().endswith(('.xml','.rels')):
                with archive.open(entry) as stream:
                    tail=b''
                    while True:
                        check();block=stream.read(65536)
                        if not block:break
                        tokens=(tail+block).replace(b'\0',b'').upper()
                        require(b'<!DOCTYPE' not in tokens and b'<!ENTITY' not in tokens,'Document XML DTD/entities are unsupported')
                        tail=tokens[-64:]
        return archive
    except BaseException:
        archive.close();raise


def _docx(raw,options,check):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    with _archive(raw,check):pass
    try:document=Document(io.BytesIO(raw))
    except Exception as error:raise ValueError('Malformed DOCX document') from error
    def blocks(container,depth=0):
        require(depth<=32,'DOCX nested tables exceed their depth budget')
        for block in container.iter_inner_content():
            check()
            if isinstance(block,Paragraph):
                if block.text:yield block.text
            elif isinstance(block,Table) and options['include_tables'] and (depth==0 or options['include_nested_tables']):
                seen=set()
                for row in block.rows:
                    for cell in row.cells:
                        # Merged cells repeat wrappers over the same XML element.
                        if cell._tc in seen:continue
                        seen.add(cell._tc);yield from blocks(cell,depth+1)
    seen_parts=set()
    if options['include_headers']:
        for section in document.sections:
            for name in ('header','first_page_header','even_page_header'):
                header=getattr(section,name)
                if header.part.partname not in seen_parts:
                    seen_parts.add(header.part.partname);yield from blocks(header)
    yield from blocks(document)
    if options['include_footers']:
        for section in document.sections:
            for name in ('footer','first_page_footer','even_page_footer'):
                footer=getattr(section,name)
                if footer.part.partname not in seen_parts:
                    seen_parts.add(footer.part.partname);yield from blocks(footer)


def _odt(raw,options,check):
    from defusedxml.ElementTree import fromstring
    text='urn:oasis:names:tc:opendocument:xmlns:text:1.0'
    table='urn:oasis:names:tc:opendocument:xmlns:table:1.0'
    with _archive(raw,check) as archive:
        require('content.xml' in archive.namelist(),'ODT lacks content.xml')
        require('mimetype' in archive.namelist(),'ODT lacks its mimetype')
        require(archive.read('mimetype')==b'application/vnd.oasis.opendocument.text','Document is not an ODT text container')
        try:root=fromstring(archive.read('content.xml'))
        except Exception as error:raise ValueError('Malformed ODT XML') from error
    office='urn:oasis:names:tc:opendocument:xmlns:office:1.0'
    require(root.tag=='{'+office+'}document-content','Malformed ODT document root')
    body=root.find('{'+office+'}body/{'+office+'}text')
    require(body is not None,'ODT lacks its text body')
    def contents(element,depth=0):
        check();require(depth<=128,'ODT text exceeds its depth budget')
        parts=[];size=0
        def append(value):
            nonlocal size
            size+=len(value.encode());require(size<=MAX_TEXT_BYTES,'ODT text exceeds its byte budget');parts.append(value)
        append(element.text or '')
        for child in element:
            if child.tag=='{'+text+'}s':
                try:count=int(child.get('{'+text+'}c','1'))
                except ValueError as error:raise ValueError('ODT space repeat must be an integer') from error
                require(1<=count<=MAX_TEXT_BYTES-size,'ODT repeated spaces exceed the text budget');append(' '*count)
            elif child.tag=='{'+text+'}tab':append('\t')
            elif child.tag=='{'+text+'}line-break':append('\n')
            else:append(contents(child,depth+1))
            append(child.tail or '')
        result=''.join(parts);require(len(result.encode())<=MAX_TEXT_BYTES,'ODT text exceeds its byte budget');return result
    def walk(element,depth=0):
        check();require(depth<=128,'ODT structure exceeds its depth budget')
        if element.tag=='{'+table+'}table' and not options['include_tables']:return
        # Repeated table coordinates change semantic text multiplicity. Do not
        # silently collapse them until a bounded expansion policy is provided.
        for key in ('number-columns-repeated','number-rows-repeated'):
            require(element.get('{'+table+'}'+key,'1')=='1','Repeated ODT table coordinates are unsupported')
        if element.tag in ('{'+text+'}p','{'+text+'}h'):
            value=contents(element)
            if value:yield value
            return
        for child in element:yield from walk(child,depth+1)
    yield from walk(body)


def _pdf(raw,options,check):
    from pypdf import PdfReader
    try:reader=PdfReader(io.BytesIO(raw),strict=options['strict'],root_object_recovery_limit=10000)
    except Exception as error:raise ValueError('Malformed PDF document') from error
    try:
        require(not reader.is_encrypted,'Encrypted PDF extraction is unsupported')
        require(len(reader.pages)<=MAX_PAGES,'PDF exceeds its page budget')
        pages=options['pages'] or list(range(1,len(reader.pages)+1))
        require(all(number<=len(reader.pages) for number in pages),'PDF selected page does not exist')
        expanded=0
        for number in pages:
            check();page=reader.pages[number-1]
            content=page.get_contents()
            if content is not None:
                expanded+=len(content.get_data());require(expanded<=MAX_EXPANDED_BYTES,'PDF expanded pages exceed their content budget')
            if options['extraction_mode']=='plain':
                value=page.extract_text(extraction_mode='plain',orientations=tuple(options['orientations']),space_width=options['space_width'])
            else:
                value=page.extract_text(extraction_mode='layout',layout_mode_space_vertically=options['layout_space_vertically'],layout_mode_scale_weight=options['layout_scale_weight'],layout_mode_strip_rotated=options['layout_strip_rotated'],layout_mode_font_height_weight=options['layout_font_height_weight'])
            yield value,number
    finally:reader.close()


def extract_document(raw,format,options,check=lambda:None):
    selected=normalize_document_options(format,options)
    require(type(raw) is bytes and len(raw)<=MAX_CAPTURE_BYTES,'Document capture exceeds its byte budget')
    check()
    if format=='text':
        value=_decode(raw,selected['encoding'],selected['errors'])
        if selected['newline']!='preserve':
            value=value.replace('\r\n','\n').replace('\r','\n')
            if selected['newline']=='crlf':value=value.replace('\n','\r\n')
        # StringIO iterates without constructing an unbounded split-lines list.
        lines=(line.rstrip('\r\n') for line in io.StringIO(value,newline=None))
        values=[value] if selected['units']=='file' else (line for line in lines if selected['keep_empty_lines'] or line)
        units=((text,None) for text in values)
    elif format=='pdf':units=_pdf(raw,selected,check)
    elif format=='docx':units=((text,None) for text in _docx(raw,selected,check))
    elif format=='odt':units=((text,None) for text in _odt(raw,selected,check))
    else:
        from striprtf.striprtf import rtf_to_text
        value=_decode(raw,'latin-1','strict')
        require(value.lstrip().startswith('{\\rtf'),'Malformed RTF document')
        text=rtf_to_text(value,encoding=selected['encoding'] if selected['encoding']!='auto' else 'cp1252',errors=selected['errors'])
        units=((part.rstrip('\r\n'),None) for part in io.StringIO(text,newline=None))
    total=0;joined=[];count=0
    for text,page in units:
        check();require(type(text) is str,'Document extractor returned non-text')
        total+=len(text.encode('utf-8'));require(total<=MAX_TEXT_BYTES,'Document text exceeds its byte budget')
        if selected['units']=='file' and format!='text':joined.append(text)
        else:
            unit={'text_content':text,'unit_index':count}
            if page is not None:unit['page_number']=page
            yield unit
        count+=1
    if selected['units']=='file' and format!='text':
        result=selected['separator'].join(joined)
        require(len(result.encode())<=MAX_TEXT_BYTES,'Joined document text exceeds its byte budget')
        yield {'text_content':result,'unit_index':0}
