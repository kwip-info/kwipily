"""Local markup table readers: explicit cells and table/record selection only."""
from pathlib import Path
from collections import OrderedDict
import re
import polars as pl
if __package__:
    from .table_source_options import dtype_map,require
else:
    from table_source_options import dtype_map,require

MAX_CELLS=5_000_000


def _frame(headers,rows,options):
    require(1<=len(headers)<=128 and all(type(x) is str and x.strip() and len(x)<=200 for x in headers) and len(set(headers))==len(headers),'Table requires 1–128 unique nonblank header names; choose generated headers or explicit fields')
    require(len(rows)<=MAX_CELLS//len(headers),'Markup table exceeds its five million cell budget')
    require(all(len(row)==len(headers) for row in rows),'Markup table has inconsistent row widths')
    frame=pl.DataFrame(rows,schema=[(name,pl.String) for name in headers],orient='row')
    columns=options.get('columns')
    if columns:frame=frame.select([headers[x] if type(x) is int else x for x in columns])
    if options.get('n_rows') is not None:frame=frame.head(options['n_rows'])
    if options.get('schema_overrides'):
        overrides=dtype_map(options['schema_overrides'])
        # Markup cells start as text; Boolean is an explicit exact lexical declaration.
        for name,dtype in overrides.items():
            if dtype==pl.Boolean:
                frame=frame.with_columns(pl.col(name).replace_strict({'true':True,'false':False},return_dtype=pl.Boolean))
        frame=frame.cast(overrides,strict=True)
    return frame


def _md_cell(token,options):
    if options['cell_mode']=='markdown':return token.content
    result=[];links=[]
    for child in token.children or []:
        if child.type=='link_open':links.append((len(result),child.attrGet('href') or ''))
        elif child.type=='link_close' and links:
            start,url=links.pop();label=''.join(result[start:])
            if options['link_mode']=='url':result[start:]=[url]
            elif options['link_mode']=='both':result.append(' ('+url+')')
        elif child.type=='image':
            label=child.content;url=child.attrGet('src') or ''
            result.append(url if options['link_mode']=='url' else label+' ('+url+')' if options['link_mode']=='both' else label)
        elif child.type in ('softbreak','hardbreak') or (child.type=='html_inline' and re.fullmatch(r'<br\s*/?>',child.content,re.I)):
            result.append('\n' if options['line_breaks']=='newline' else ' ')
        elif child.type in ('text','code_inline','html_inline'):result.append(child.content)
    return ''.join(result)


def markdown(path,options,check):
    from markdown_it import MarkdownIt
    text=Path(path).read_bytes().decode(options['encoding']);check()
    tokens=MarkdownIt('commonmark').enable('table').parse(text)
    tables=[];rows=None;row=None;cell=False
    for index,token in enumerate(tokens):
        if index%256==0:check()
        if token.type=='table_open':rows=[]
        elif token.type=='tr_open' and rows is not None:row=[]
        elif token.type in ('th_open','td_open') and row is not None:cell=True
        elif token.type=='inline' and cell:row.append(_md_cell(token,options));cell=False
        elif token.type=='tr_close' and row is not None:rows.append(row);row=None
        elif token.type=='table_close':tables.append(rows);rows=None
    require(options['table_index']<len(tables),'Markdown table index does not exist')
    records=tables[options['table_index']];require(bool(records),'Markdown table is empty')
    headers=records[0] if options['has_header'] else [f'column_{i+1}' for i in range(len(records[0]))]
    return _frame(headers,records[1:] if options['has_header'] else records,options)


def xml(path,options,check):
    from lxml import etree
    raw=Path(path).read_bytes()
    parser=etree.XMLParser(resolve_entities=False,load_dtd=False,no_network=True,recover=False,huge_tree=False,encoding='utf-8' if options['encoding']=='utf-8-sig' else options['encoding'])
    root=etree.fromstring(raw,parser);check()
    require(not root.getroottree().docinfo.doctype,'XML DTDs and entities are unsupported')
    records=root.xpath(options['row_xpath'],namespaces=options['namespaces'])
    require(isinstance(records,list) and records and all(isinstance(row,etree._Element) for row in records),'XML row XPath must select element records')
    fields=dict(sorted(options['fields'].items()));rows=[];headers=list(fields)
    if not fields:
        first=records[0];headers=[]
        if options['include_attributes']:headers += [options['attribute_prefix']+key for key in first.attrib]
        for child in first:
            require(isinstance(child.tag,str) and not len(child),'Nested XML requires explicit field XPath; no automatic flattening')
            headers.append(child.tag)
        require(len(set(headers))==len(headers),'Repeated XML fields require explicit field XPath')
    for index,record in enumerate(records):
        if index%256==0:check()
        if options['n_rows'] is not None and len(rows)>=options['n_rows']:break
        row=[]
        if fields:
            for expression in fields.values():
                value=record.xpath(expression,namespaces=options['namespaces'])
                if isinstance(value,list):
                    require(len(value)<=1,'XML field XPath selected multiple values; choose one explicit scalar field')
                    value=value[0] if value else None
                if isinstance(value,etree._Element):
                    require(not len(value),'XML field selected nested content; select text() or string() explicitly')
                    value=value.text
                require(value is None or isinstance(value,(str,int,float,bool)),'XML field must select a scalar')
                row.append(None if value is None else str(value))
        else:
            values={}
            if options['include_attributes']:values.update((options['attribute_prefix']+key,value) for key,value in record.attrib.items())
            for child in record:
                require(isinstance(child.tag,str) and not len(child) and child.tag not in values,'Nested/repeated XML field requires explicit XPath')
                values[child.tag]=child.text
            require(not set(values)-set(headers),'XML records introduce undeclared fields; choose explicit field XPath')
            row=[values.get(name) for name in headers]
        rows.append(row)
        require(len(rows)<=MAX_CELLS//max(1,len(headers)),'XML exceeds its five million cell budget')
    return _frame(headers,rows,options)


def _hidden(element):
    return any('hidden' in node.attrib or re.search(r'(?:display\s*:\s*none|visibility\s*:\s*hidden)',node.get('style',''),re.I) for node in [element,*element.iterancestors()])


def _html_cell(cell,options):
    from lxml import etree
    if options['cell_mode']=='html':return (cell.text or '')+''.join(etree.tostring(child,encoding='unicode',method='html') for child in cell)
    def text(element):
        if element.tag in ('script','style'):return ''
        value=element.text or ''
        for child in element:
            if child.tag=='br':content='\n'
            else:content=text(child)
            if child.tag=='a':
                url=child.get('href','')
                if options['link_mode']=='url':content=url
                elif options['link_mode']=='both':content+=' ('+url+')'
            value+=content+(child.tail or '')
        return value
    return text(cell).strip()


def html(path,options,check):
    from lxml import etree
    parser=etree.HTMLParser(no_network=True,recover=True,huge_tree=False,encoding='utf-8' if options['encoding']=='utf-8-sig' else options['encoding'])
    root=etree.fromstring(Path(path).read_bytes(),parser);check()
    tables=root.xpath('//table');require(options['table_index']<len(tables),'HTML table index does not exist')
    table=tables[options['table_index']];records=[];spans={};cell_count=0
    for index,tr in enumerate(table.xpath('.//tr')):
        if index%256==0:check()
        if next(iter(tr.iterancestors('table')),None) is not table:continue
        if not options['include_hidden'] and _hidden(tr):continue
        values={column:value for column,(value,remaining) in spans.items()};spans={column:(value,remaining-1) for column,(value,remaining) in spans.items() if remaining>1}
        column=0
        for cell in tr.xpath('./th|./td'):
            if not options['include_hidden'] and _hidden(cell):continue
            while column in values:column+=1
            try:width=int(cell.get('colspan','1'));height=int(cell.get('rowspan','1'))
            except ValueError:raise ValueError('HTML cell spans must be positive bounded integers') from None
            require(1<=width<=128 and 1<=height<=100000,'HTML span exceeds table budget')
            require(options['span_policy']=='expand' or (width==1 and height==1),'HTML spans require explicit Expand span policy')
            value=_html_cell(cell,options)
            for offset in range(width):
                require(column+offset not in values,'HTML spans overlap')
                values[column+offset]=value
                if height>1:spans[column+offset]=(value,height-1)
            column+=width
        if values:
            require(max(values)<128,'HTML table exceeds 128 columns')
            records.append([values.get(i) for i in range(max(values)+1)])
            cell_count+=len(records[-1])
        require(cell_count<=MAX_CELLS,'HTML exceeds five million cells')
    records=records[options['skip_rows']:];require(bool(records),'HTML table has no selected rows')
    header=options['header']
    if header is None:headers=[f'column_{i+1}' for i in range(len(records[0]))]
    else:
        require(header<len(records),'HTML header row does not exist');headers=records[header];records=records[header+1:]
    return _frame(headers,records,options)
