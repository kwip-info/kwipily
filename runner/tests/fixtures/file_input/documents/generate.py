"""Synthetic document fixtures; expected strings are authored in the tests."""
import io
import zipfile


def pdf_bytes(encrypted=False,rotated=False,layout=False):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
    writer=PdfWriter()
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
    for text in ('First page','Second page'):
        page=writer.add_blank_page(width=300,height=300)
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        location='0 1 -1 0 260 30 Tm' if rotated else '30 260 Td'
        content=f'BT /F1 12 Tf {location} ({text}) Tj ET'.encode()
        if layout:content=b'BT /F1 12 Tf 30 260 Td (Left) Tj 100 0 Td (Right) Tj 0 -60 Td (Down) Tj ET'
        stream=DecodedStreamObject();stream.set_data(content)
        page[NameObject('/Contents')]=writer._add_object(stream)
    if encrypted:writer.encrypt('synthetic-fixture-only')
    output=io.BytesIO();writer.write(output);return output.getvalue()


def docx_bytes():
    from docx import Document
    document=Document();document.add_paragraph('First paragraph')
    table=document.add_table(rows=1,cols=2);table.cell(0,0).text='Left';table.cell(0,1).text='Right'
    nested=table.cell(0,0).add_table(rows=1,cols=1);nested.cell(0,0).text='Nested'
    document.add_paragraph('Last paragraph')
    document.sections[0].header.paragraphs[0].text='Header'
    document.sections[0].footer.paragraphs[0].text='Footer'
    output=io.BytesIO();document.save(output);return output.getvalue()


def odt_bytes(content=None,extra=None):
    xml=content or '''<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"><office:body><office:text><text:h>Title</text:h><text:p>A<text:s text:c="3"/>B<text:tab/>C<text:line-break/>D</text:p><text:list><text:list-item><text:p>List item</text:p></text:list-item></text:list><table:table><table:table-row><table:table-cell><text:p>Cell</text:p></table:table-cell></table:table-row></table:table><text:p>End</text:p></office:text></office:body></office:document-content>'''
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('mimetype','application/vnd.oasis.opendocument.text');archive.writestr('content.xml',xml)
        for name,data in (extra or {}).items():archive.writestr(name,data)
    return output.getvalue()
