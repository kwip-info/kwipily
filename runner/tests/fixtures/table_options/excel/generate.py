"""Reproducible public synthetic workbooks for the Table Input option audit.

No binary assets, customer data, macros or external network resources are needed.
Run with a destination directory, or call generate(directory) from the tests.
"""
from pathlib import Path
import datetime
import zipfile
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.worksheet.table import Table
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont


def generate(directory):
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    def book(name, rows):
        workbook = Workbook(); sheet = workbook.active; sheet.title = 'Data'
        for row in rows: sheet.append(row)
        return workbook, sheet, root / (name + '.xlsx')
    w,s,p=book('basic',[['name','amount','flag'],['Alpha',1,True],['Beta',2,False],['Gamma',3,True]])
    second=w.create_sheet('Second');second.append(['name','amount','flag']);second.append(['Other',9,False])
    s.add_table(Table(displayName='Records',ref='A1:C4'));w.save(p)
    w,s,p=book('layout',[['name','amount','unused'],['A',1,None],[None,None,None],['B',2,None]])
    s['C1'].font=Font(bold=True);w.save(p)
    w,s,p=book('headers',[['preamble','ignored'],['name','amount'],['A',1],['B',2]]);w.save(p)
    w,s,p=book('whitespace',[['name','amount'],['A',1],['   ','  '],['\t','   ']]);w.save(p)
    w,s,p=book('types',[['number','flag','decimal'],['01',True,1.25],['02',False,2.5]]);w.save(p)
    w,s,p=book('mixed',[['number','name'],[1,'A'],['oops','B'],[3,'C']]);w.save(p)
    w,s,p=book('text',[['name','amount'],['hello,world',1],['line\nbreak',2],['quote"text',3],['café🙂',4]]);w.save(p)
    w,s,p=book('dates',[['date','time','fraction'],[datetime.date(2024,2,3),datetime.time(13,14,15),1.23456]])
    s['A2'].number_format='yyyy-mm-dd';s['B2'].number_format='hh:mm:ss';w.save(p)
    w,s,p=book('scientific',[['number'],[1.23456e-12],[1.23456e12]])
    s['A2'].number_format='0.0000';s['A3'].number_format='0.0000';w.save(p)
    w,s,p=book('hyperlinks',[['name','amount'],['label',1],['ordinary',2]])
    s['A2'].hyperlink='https://example.invalid/only-a-stored-reference';w.save(p)
    w,s,p=book('hidden',[['name','amount'],['shown',1],['hidden-row',2]])
    s.row_dimensions[3].hidden=True;hidden=w.create_sheet('Hidden');hidden.append(['name','amount']);hidden.append(['secret-synthetic',9]);hidden.sheet_state='hidden';w.save(p)
    w,s,p=book('merged',[['group','amount'],['A',1],[None,2]])
    s.merge_cells('A2:A3');w.save(p)
    w,s,p=book('merged-gaps',[['group','amount'],['A',1]])
    s.merge_cells('A2:A4');w.save(p)
    w,s,p=book('merged-overlap',[['group','amount'],['A',1],[None,2],[None,3]])
    s.merge_cells('A2:A3');s.merge_cells('A3:A4');w.save(p)
    w,s,p=book('empty',[['name','amount']]);w.save(p)
    w,s,p=book('blank',[]);w.save(p)
    w,s,p=book('nulls',[['name','amount'],['NA','1'],['A',''],['NULL','3']]);w.save(p)
    w,s,p=book('comma-decimal',[['amount'],['1,25'],['2,50']]);w.save(p)
    w,s,p=book('delimiter-only',[['name','amount'],['hello,world',1]]);w.save(p)
    w,s,p=book('comments',[['name','amount'],['#omit',1],['keep',2]]);w.save(p)
    w,s,p=book('richtext',[['name','amount'],[None,1]])
    s['A2']=CellRichText('plain ',TextBlock(InlineFont(b=True),'bold'));w.save(p)
    # A legitimate XML text entity is replaced by an invalid literal control
    # for the xlsx2csv strict/ignore character-data option, in a private fixture.
    w,s,p=book('trim',[['name','amount',None],['A',1,None],['B',2,None]])
    s['C1'].font=Font(bold=True);w.save(p)
    w,s,p=book('ragged',[['name',None],['A',None,'extra']]);w.save(p)
    w,s,p=book('invalid-date',[['date','amount'],[datetime.date(2024,2,3),1],[datetime.date(2024,2,4),2]])
    w.save(p)
    original=p.read_bytes()
    import io,re
    with zipfile.ZipFile(io.BytesIO(original)) as source,zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as target:
        for entry in source.infolist():
            data=source.read(entry.filename)
            if entry.filename=='xl/worksheets/sheet1.xml':data=re.sub(rb'(<c r="A3"[^>]*><v>)[^<]+',rb'\g<1>999999999999999',data)
            target.writestr(entry,data)
    # Convert inline strings into a standards-compliant shared-string fixture.
    # xlsx2csv's escape_strings option applies to this storage representation.
    text_path=root/'text.xlsx';shared_path=root/'shared-text.xlsx';strings=[]
    import xml.etree.ElementTree as ET
    from xml.sax.saxutils import escape
    with zipfile.ZipFile(text_path) as source,zipfile.ZipFile(shared_path,'w',zipfile.ZIP_DEFLATED) as target:
        for entry in source.infolist():
            data=source.read(entry.filename)
            if entry.filename.startswith('xl/worksheets/'):
                def replace(match):
                    value=''.join(ET.fromstring(b'<is>'+match.group(3)+b'</is>').itertext());index=len(strings);strings.append(value)
                    return b'<c'+match.group(1)+match.group(2)+b' t="s"><v>'+str(index).encode()+b'</v></c>'
                data=re.sub(rb'<c([^>]*?) t="inlineStr"([^>]*)><is>(.*?)</is></c>',replace,data,flags=re.S)
            if entry.filename=='[Content_Types].xml':data=data.replace(b'</Types>',b'<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>')
            if entry.filename=='xl/_rels/workbook.xml.rels':data=data.replace(b'</Relationships>',b'<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml" Id="rIdShared"/></Relationships>')
            target.writestr(entry,data)
        xml='<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="'+str(len(strings))+'" uniqueCount="'+str(len(strings))+'">'+''.join('<si><t xml:space="preserve">'+escape(x)+'</t></si>' for x in strings)+'</sst>'
        target.writestr('xl/sharedStrings.xml',xml)
    w,s,p=book('invalid-char',[['name','amount'],['beforeAFTER',1]]);w.save(p)
    original=p.read_bytes()
    import io
    with zipfile.ZipFile(io.BytesIO(original)) as source,zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as target:
        for entry in source.infolist():
            data=source.read(entry.filename)
            if entry.filename=='xl/worksheets/sheet1.xml':data=data.replace(b'beforeAFTER',b'before\x01AFTER')
            target.writestr(entry,data)
    return root


if __name__=='__main__':
    import sys
    generate(sys.argv[1])
