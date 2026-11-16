"""File Table Input adapters into the shared typed-table/owned-columnar boundary.

Every reader consumes a private immutable capture of exactly one local grant.
No credentials, paths, URLs, globs or executable conversion callbacks enter config.
"""
import hashlib
from pathlib import Path
import tempfile
import zipfile
import polars as pl
if __package__:
    from .table_source_options import normalize_options,dtype_map,require
    from .dataflow_disk import disk_scope,current_scope
    from .dataflow_columnar import ColumnarStore
    from .polars_csv import _open_source,_verify_source
    from .dataflow_temporal import metadata as temporal_metadata
else:
    from table_source_options import normalize_options,dtype_map,require
    from dataflow_disk import disk_scope,current_scope
    from dataflow_columnar import ColumnarStore
    from polars_csv import _open_source,_verify_source
    from dataflow_temporal import metadata as temporal_metadata

MAX_SOURCE_BYTES=10_000_000
MAX_EXPANDED_BYTES=100_000_000
MAX_CELLS=5_000_000
BATCH_ROWS=4096


def _parquet_admission(path,options,check=lambda:None):
    import pyarrow.parquet as pq
    reader=pq.ParquetFile(path,arrow_extensions_enabled=False,thrift_string_size_limit=10_000_000,thrift_container_size_limit=1_000_000)
    try:
        columns=options['columns'];file_names=reader.schema_arrow.names
        names=sorted(options['schema']) if options.get('schema') else file_names
        if columns:columns=[names[c] if type(c) is int else c for c in columns]
        require(all(c in names for c in columns),'Selected Parquet column does not exist')
        require(options['missing_columns']=='insert' or all(c in file_names for c in columns),'Missing Parquet column requires an explicit Insert policy and schema')
        count=reader.metadata.num_rows if options['n_rows'] is None else min(reader.metadata.num_rows,options['n_rows'])
        require(count<=MAX_CELLS//max(1,len(columns) or len(names)),'Parquet source exceeds five million selected cells; choose columns or a row read limit')
        expanded=0;rows=0
        for i in range(reader.metadata.num_row_groups):
            check()
            if rows>=count:break
            group=reader.metadata.row_group(i);rows+=group.num_rows
            for j in range(group.num_columns):
                column=group.column(j)
                if not columns or column.path_in_schema.split('.')[0] in columns:expanded+=column.total_uncompressed_size
        require(expanded<=MAX_EXPANDED_BYTES,'Selected Parquet row groups exceed the 100 MB expanded-data budget')
        return names
    finally:reader.close()


def _excel_admission(path,check=lambda:None):
    # XLSX/XLSM are ZIP containers; eager shared strings must be admitted before
    # any engine opens the workbook. Other calamine-supported binaries stay 10 MB.
    if not zipfile.is_zipfile(path):return
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();require(len(entries)<=10000,'Workbook exceeds its archive entry budget')
        require(sum(e.file_size for e in entries)<=MAX_EXPANDED_BYTES,'Workbook exceeds its 100 MB expanded archive budget')
        for entry in entries:
            check()
            require(entry.file_size<=MAX_EXPANDED_BYTES and not entry.flag_bits&1,'Encrypted or oversized workbook content is unsupported')
            # DTD/entity declarations are not needed by OOXML, and rejecting
            # them protects all three engines rather than relying on a parser flag.
            if entry.filename.endswith(('.xml','.rels')):
                with archive.open(entry) as stream:
                    tail=b''
                    while True:
                        check()
                        block=stream.read(65536)
                        if not block:break
                        chunk=(tail+block).upper();tokens=chunk.replace(b'\0',b'')
                        require(b'<!DOCTYPE' not in tokens and b'<!ENTITY' not in tokens,'Workbook XML entities/DTDs are unsupported')
                        tail=chunk[-64:]


def _excel_openpyxl(path,args,check):
    import io
    import openpyxl
    from openpyxl.cell.rich_text import CellRichText
    from polars.io.spreadsheet.functions import _get_read_options,_read_spreadsheet_openpyxl
    parser=openpyxl.load_workbook(io.BytesIO(path.read_bytes()),data_only=True,**args['engine_options'])
    try:
        name=args['sheet_name'] or parser.sheetnames[(args['sheet_id'] or 1)-1]
        sheet=parser[name]
        require(sheet.max_row*sheet.max_column<=MAX_CELLS,'Workbook selected worksheet exceeds five million cells')
        if not args['engine_options'].get('read_only'):
            for index,row in enumerate(sheet.iter_rows()):
                if index%256==0:check()
                for cell in row:
                    if isinstance(cell.value,CellRichText):cell.value=str(cell.value)
        # Upstream drops the header when there are no body columns to transpose.
        # Preserve a named empty schema, as the other workbook engines do.
        if args['has_header'] and sheet.max_row==1:
            header=next(sheet.iter_rows(),None)
            require(header is not None,'Empty worksheet has no readable header')
            names=[cell.value for cell in header]
            require(all(type(name) is str and name.strip() for name in names),'Empty worksheet requires named columns')
            schema=args['schema_overrides'] or {}
            frame=pl.DataFrame(schema={name:schema.get(name,pl.String) for name in names})
            if args['columns']:frame=frame.select([names[c] if type(c) is int else c for c in args['columns']])
            return frame
        read=_get_read_options(args['read_options'],engine='openpyxl',columns=args['columns'],has_header=args['has_header'],infer_schema_length=args['infer_schema_length'])
        return _read_spreadsheet_openpyxl(parser,sheet_name=name,read_options=read,schema_overrides=args['schema_overrides'],columns=args['columns'],table_name=None,drop_empty_rows=args['drop_empty_rows'],drop_empty_cols=args['drop_empty_cols'],raise_if_empty=args['raise_if_empty'])
    finally:parser.close()


def _excel_xlsx2csv(path,args,check):
    import io,re,types
    import xml.etree.ElementTree as ET
    import xlsx2csv
    def column_index_from_string(text):
        number=0
        for character in text:number=number*26+ord(character)-64
        return number
    def get_column_letter(number):
        text=''
        while number:number,remainder=divmod(number-1,26);text=chr(65+remainder)+text
        return text
    def range_boundaries(text):
        match=re.fullmatch(r'([A-Z]+)([1-9][0-9]*):([A-Z]+)([1-9][0-9]*)',text)
        require(match is not None,'Workbook merge range must use bounded cell coordinates')
        return column_index_from_string(match[1]),int(match[2]),column_index_from_string(match[3]),int(match[4])
    from polars.io.spreadsheet.functions import _get_read_options,_read_spreadsheet_xlsx2csv
    selected=dict(args['engine_options'])
    parser=xlsx2csv.Xlsx2csv(io.BytesIO(path.read_bytes()),**selected)
    class Sheet(xlsx2csv.Sheet):
        def __init__(self,*positional,**keywords):
            super().__init__(*positional,**keywords)
            # The pinned converter consults this missing field on conversion errors.
            self.options=selected
            self._last_input_row=0
            self._merge_rows={}
        def set_merge_cells(self,enabled):
            if not enabled:return
            if not self.filedata:self.filedata=self.filehandle.read()
            root=ET.fromstring(self.filedata)
            for item in root.findall('.//{*}mergeCell'):
                left,top,right,bottom=range_boundaries(item.attrib['ref'])
                require(1<=left<=right<=128 and 1<=top<=bottom and (right-left+1)*(bottom-top+1)+len(self.mergeCells)<=MAX_CELLS,'Workbook merged cells exceed the table budget')
                origin=get_column_letter(left)+str(top)
                for row in range(top,bottom+1):
                    if row%256==0:check()
                    for column in range(left,right+1):
                        address=get_column_letter(column)+str(row)
                        require(address not in self.mergeCells,'Workbook merged ranges must not overlap')
                        self.mergeCells[address]={'copyFrom':origin}
                        self._merge_rows.setdefault(row,[]).append(address)
        def _synthetic_row(self,number):
            if number%256==0:check()
            super().handleStartElement('row',{'r':str(number)})
            self.handleEndElement('row')
        def handleStartElement(self,name,attributes):
            if name.split(':')[-1]=='row':
                number=int(attributes.get('r',self._last_input_row+1))
                for missing in sorted(row for row in self._merge_rows if self._last_input_row<row<number):self._synthetic_row(missing)
                self._last_input_row=number
            super().handleStartElement(name,attributes)
        def handleEndElement(self,name):
            if name.split(':')[-1]=='row' and self.in_row:
                for address in self._merge_rows.get(int(self.rowNum),[]):
                    entry=self.mergeCells[address]
                    origin=self.mergeCells.get(entry['copyFrom'],{})
                    column=column_index_from_string(re.match(r'[A-Z]+',address).group())-1
                    if column not in self.columns and 'value' in origin:self.columns[column]=origin['value']
            if name.split(':')[-1]=='sheetData':
                for missing in sorted(row for row in self._merge_rows if row>self._last_input_row):self._synthetic_row(missing)
            super().handleEndElement(name)
    # Use the trusted pinned converter code with a private Sheet factory. No
    # global class replacement can affect another read or concurrent operation.
    original=xlsx2csv.Xlsx2csv._convert
    local=types.FunctionType(original.__code__,{**original.__globals__,'Sheet':Sheet},original.__name__,original.__defaults__,original.__closure__)
    parser._convert=types.MethodType(local,parser)
    try:
        name=args['sheet_name'] or parser.workbook.sheets[(args['sheet_id'] or 1)-1]['name']
        read=_get_read_options(args['read_options'],engine='xlsx2csv',columns=args['columns'],has_header=args['has_header'],infer_schema_length=args['infer_schema_length'])
        return _read_spreadsheet_xlsx2csv(parser,sheet_name=name,read_options=read,schema_overrides=args['schema_overrides'],columns=args['columns'],table_name=None,drop_empty_rows=args['drop_empty_rows'],drop_empty_cols=args['drop_empty_cols'],raise_if_empty=args['raise_if_empty'])
    except xlsx2csv.XlsxException as error:
        raise ValueError('The selected worksheet could not be read by xlsx2csv: '+str(error)) from error
    finally:parser.close()


def _excel(path,options,check=lambda:None):
    _excel_admission(path,check)
    if options['engine']=='xlsx2csv' and options['engine_options'].get('exclude_hidden_sheets'):
        # xlsx2csv applies this flag only to its all-sheets mode. This reader
        # selects one worksheet, so enforce the choice before conversion too.
        from xml.etree import ElementTree
        with zipfile.ZipFile(path) as archive:
            workbook=ElementTree.fromstring(archive.read('xl/workbook.xml'))
        sheets=workbook.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheets/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet')
        selected=next((sheet for sheet in sheets if sheet.get('name')==options['sheet_name']),None) if options['sheet_name'] else sheets[(options['sheet_id'] or 1)-1] if 0<(options['sheet_id'] or 1)<=len(sheets) else None
        require(selected is not None,'Selected workbook sheet does not exist')
        require(selected.get('state','visible') not in ('hidden','veryHidden'),'Selected worksheet is hidden; disable Exclude hidden worksheets to read it explicitly')
    args={k:v for k,v in options.items() if k not in ('schema_overrides','columns')}
    args['schema_overrides']=dtype_map(options['schema_overrides']);args['columns']=options['columns'] or None
    args['sheet_name']=options['sheet_name'] or None;args['table_name']=options['table_name'] or None
    if args['sheet_name'] or args['table_name']:args['sheet_id']=None
    read=dict(args['read_options']);engine=args['engine']
    if engine=='calamine':
        for key in ('column_names','use_columns','dtypes'):
            if not read.get(key):read.pop(key,None)
        if not args['has_header']:read['header_row']=None
        elif args['table_name'] and read.get('header_row') is None:read.pop('header_row',None)
        selected_columns=args['columns'] or read.get('use_columns')
        if (not args['table_name'] and (read.get('skip_whitespace_tail_rows') or read.get('whitespace_as_null') or isinstance(read.get('skip_rows'),list))) or (args['table_name'] and selected_columns):
            # Polars 2.0 calls fastexcel's legacy load_sheet_eager wrapper, which
            # lacks two public options accepted by load_sheet(eager=True).
            # Adapt only that call, retaining Polars' shared schema/drop logic.
            import fastexcel
            from polars.io.spreadsheet.functions import _read_spreadsheet_calamine
            reader=fastexcel.read_excel(path.read_bytes())
            class Reader:
                def load_sheet_eager(self,*positional,**keywords):
                    # Eager fastexcel ignores selected row indexes and trailing
                    # whitespace handling. Its lazy sheet materialization honors
                    # those authored choices before the Arrow conversion.
                    return reader.load_sheet(*positional,**keywords,eager=False).to_arrow()
                def load_table(self,*positional,**keywords):
                    # Polars' table helper assumes every use_columns list is
                    # names; restore public fastexcel index/range selection.
                    if selected_columns:keywords['use_columns']=selected_columns
                    return reader.load_table(*positional,**keywords)
            name=args['sheet_name'] if args['table_name'] else args['sheet_name'] or reader.sheet_names[(args['sheet_id'] or 1)-1]
            read['schema_sample_rows']=args['infer_schema_length']
            return _read_spreadsheet_calamine(Reader(),sheet_name=name,read_options=read,schema_overrides=args['schema_overrides'],columns=args['columns'],table_name=args['table_name'],drop_empty_rows=args['drop_empty_rows'],drop_empty_cols=args['drop_empty_cols'],raise_if_empty=args['raise_if_empty'])
        for key in ('skip_whitespace_tail_rows','whitespace_as_null'):
            if not read.get(key):read.pop(key,None)
    elif engine=='openpyxl':args['engine_options']={**args['engine_options'],'keep_vba':False,'keep_links':False}
    else:
        for key in ('null_values','new_columns'):
            if not read.get(key):read.pop(key,None)
        for key in ('quote_char','comment_prefix','row_index_name'):
            if key in read and not read[key]:read[key]=None
        args['engine_options']={k:v for k,v in args['engine_options'].items() if v not in ('',None)}
        args['engine_options'].update(delimiter=',',lineterminator='\n',outputencoding='utf-8')
    args['read_options']=read
    if engine=='openpyxl':return _excel_openpyxl(path,args,check)
    if engine=='xlsx2csv':return _excel_xlsx2csv(path,args,check)
    frame=pl.read_excel(path.read_bytes(),**args)
    require(isinstance(frame,pl.DataFrame),'Choose exactly one worksheet or named table')
    return frame


def _batches(path,fmt,options,check):
    narrow_schema={}
    if fmt=='parquet':
        names=_parquet_admission(path,options,check)
        if options['use_pyarrow']:
            import pyarrow.parquet as pq
            arrow=dict(options['pyarrow_options'])
            for key in ('use_threads','use_pandas_metadata'):arrow.pop(key,None)
            arrow={key:value for key,value in arrow.items() if value not in (None,'')}
            reader=pq.ParquetFile(path,memory_map=options['memory_map'],arrow_extensions_enabled=False,**arrow)
            columns=options['columns'] or None
            if columns:columns=[names[c] if type(c) is int else c for c in columns]
            count=0
            try:
                for batch in reader.iter_batches(batch_size=BATCH_ROWS,columns=columns,use_threads=options['pyarrow_options']['use_threads'],use_pandas_metadata=options['pyarrow_options']['use_pandas_metadata']):
                    check();frame=pl.from_arrow(batch)
                    if options['n_rows'] is not None:frame=frame.head(max(0,options['n_rows']-count))
                    if options['row_index_name']:frame=frame.with_row_index(options['row_index_name'],options['row_index_offset']+count)
                    count+=frame.height
                    yield frame
                    if options['n_rows'] is not None and count>=options['n_rows']:break
            finally:reader.close()
            return
        args={k:options[k] for k in ('n_rows','row_index_offset','parallel','use_statistics','low_memory','missing_columns','extra_columns')}
        args.update(row_index_name=options['row_index_name'] or None,schema=dtype_map(options['schema']),glob=False,hive_partitioning=False,credential_provider=None,cast_options=pl.ScanCastOptions(**options['cast_options'],datetime_cast='forbid',missing_struct_fields='raise',extra_struct_fields='raise'))
        lazy=pl.scan_parquet(path,**args)
        if options['columns']:
            selected=[names[c] if type(c) is int else c for c in options['columns']]
            if options['row_index_name']:selected=[options['row_index_name'],*selected]
            lazy=lazy.select(selected)
    elif fmt=='jsonl':
        args={**options,'schema':dtype_map(options['schema']),'schema_overrides':dtype_map(options['schema_overrides']),'row_index_name':options['row_index_name'] or None}
        # Polars' streaming NDJSON decoder cannot construct narrow integer
        # columns and may terminate its async task without yielding an error.
        # Decode into supported widths, then enforce the authored width on each
        # bounded batch. Never turn a decoder failure into a successful empty table.
        wider={pl.Int8:pl.Int32,pl.Int16:pl.Int32,pl.UInt8:pl.UInt32,pl.UInt16:pl.UInt32}
        for key in ('schema','schema_overrides'):
            if args[key]:
                for name,dtype in args[key].items():
                    if dtype in wider:narrow_schema[name]=dtype
                    else:narrow_schema.pop(name,None)
                args[key]={name:wider.get(dtype,dtype) for name,dtype in args[key].items()}
        lazy=pl.scan_ndjson(path,**args,credential_provider=None)
    elif fmt=='ipc':
        lazy=pl.scan_ipc(path,n_rows=options['n_rows'],glob=False,hive_partitioning=False,credential_provider=None)
        if options['columns']:
            names=lazy.collect_schema().names();lazy=lazy.select([names[c] if type(c) is int else c for c in options['columns']])
    else:
        if fmt=='json':frame=pl.read_json(path,schema=dtype_map(options['schema']),schema_overrides=dtype_map(options['schema_overrides']),infer_schema_length=options['infer_schema_length'])
        elif fmt=='excel':frame=_excel(path,options,check)
        else:
            if __package__:from . import table_text_readers
            else:import table_text_readers
            frame=getattr(table_text_readers,fmt)(path,options,check)
        require(frame.width<=128 and frame.height<=MAX_CELLS//max(1,frame.width),'Table exceeds 128 columns or five million cells')
        yield from frame.iter_slices(BATCH_ROWS)
        if not frame.height:yield frame
        return
    batches=lazy.collect_batches(chunk_size=BATCH_ROWS,maintain_order=True,engine='streaming');seen=False
    try:
        for batch in batches:
            seen=True
            yield batch.cast(narrow_schema,strict=not options.get('ignore_errors',False)) if narrow_schema else batch
        if not seen:
            frame=pl.DataFrame(schema=lazy.collect_schema())
            yield frame.cast(narrow_schema) if narrow_schema else frame
    finally:del batches


def _column_dtype(dtype,*,allow_temporal=False):
    if dtype in (pl.String,pl.Null):return 'string'
    if allow_temporal and isinstance(dtype,pl.Datetime) and dtype.time_unit=='us':return 'datetime'
    if dtype==pl.Boolean:return 'bool'
    if dtype.is_integer():return 'int64'
    if dtype.is_float():return 'float64'
    raise ValueError('Table Input supports scalar string, Boolean, integer and floating columns. Select supported columns or choose explicit source types; nested, temporal, decimal and binary values are not flattened or converted automatically.')


def source_file(path,configuration,check_cancelled=lambda:None,*,sampling=None,_internal=False,_allow_temporal=False):
    fmt,options=normalize_options(configuration,internal=_internal)
    # Prefix count demand can be pushed into readers without changing source
    # capture/hash or observation ordinals. Percentage needs known cardinality.
    if sampling is not None:
        if __package__:from .preparation_demand_configuration import sampling_row_limit,normalize_sampling
        else:from preparation_demand_configuration import sampling_row_limit,normalize_sampling
        policy=normalize_sampling(sampling)
        limit=sampling_row_limit(policy) if policy['mode']!='percentage' else None
        if limit is not None:
            if 'n_rows' in options:options['n_rows']=min(options['n_rows'],limit) if options['n_rows'] is not None else limit
            if fmt=='excel' and options['engine']=='calamine':
                read=dict(options['read_options']);read['n_rows']=min(read['n_rows'],limit) if read['n_rows'] is not None else limit;options['read_options']=read
    if options.get('use_pyarrow'):
        require(not options.get('schema'),'PyArrow reading does not support the native schema option')
        require(options['parallel']=='auto' and options['use_statistics'] and not options['low_memory'] and options['missing_columns']=='raise','Native-only Parquet strategy/schema controls cannot configure PyArrow')
    original,stream,identity=_open_source(path);store=None;reserved=0
    with stream,disk_scope(check_cancelled=check_cancelled):
        directory,budget,_=current_scope()
        with tempfile.TemporaryDirectory(prefix='.table-source-',dir=directory) as temporary:
            capture=Path(temporary)/('source.'+('xlsx' if fmt=='excel' else fmt));hasher=hashlib.sha256();size=0
            try:
                with capture.open('xb') as target:
                    while True:
                        check_cancelled();block=stream.read(262144)
                        if not block:break
                        size+=len(block);require(size<=MAX_SOURCE_BYTES,'Table source exceeds its 10 MB byte budget')
                        budget.resize(reserved,size);reserved=size;target.write(block);hasher.update(block)
                source=hasher.hexdigest();columns=None;nulls=None;count=0
                for frame in _batches(capture,fmt,options,check_cancelled):
                    check_cancelled()
                    if columns is None:
                        require(1<=frame.width<=128 and all(name.strip() and len(name)<=200 for name in frame.columns),'Table requires 1–128 nonblank bounded columns')
                        columns=tuple({'name':name,'dtype':_column_dtype(dtype,allow_temporal=_internal and _allow_temporal),'nullable':True,**({'temporal':temporal_metadata('datetime',dtype.time_zone or 'UTC')} if isinstance(dtype,pl.Datetime) and _internal and _allow_temporal else {})} for name,dtype in frame.schema.items())
                        store=ColumnarStore(columns,check_cancelled,identity_prefix=source+':');nulls=[False]*frame.width
                    require(frame.columns==[c['name'] for c in columns],'Table schema changed between source batches')
                    require(count+frame.height<=MAX_CELLS//len(columns),'Table source exceeds five million cells')
                    for index,column in enumerate(frame):nulls[index] |= column.null_count()>0
                    temporal_names=[c['name'] for c in columns if c['dtype']=='datetime']
                    if temporal_names:frame=frame.with_columns(pl.col(name).cast(pl.Int64) for name in temporal_names)
                    for row in frame.iter_rows(buffer_size=256):store.append(row,f'{source}:{count}');count+=1
                require(store is not None,'Table source has no readable schema or records')
                _verify_source(original,stream,identity,source,check_cancelled)
                rows,ids=store.finish()
                if __package__:from .dataflow import Table,derive
                else:from dataflow import Table,derive
                result=derive(Table(tuple({**column,'nullable':nulls[i]} for i,column in enumerate(columns)),rows,ids,source))
                if sampling is not None:
                    if __package__:from .preparation_demand import sample_table
                    else:from preparation_demand import sample_table
                    result=sample_table(result,sampling,check_cancelled)
                check_cancelled();return result
            except BaseException as error:
                if store is not None:store.close()
                if isinstance(error,(pl.exceptions.PolarsError,TypeError,KeyError,IndexError,zipfile.BadZipFile)) or type(error).__module__.startswith(('lxml.','_fastexcel','fastexcel.')):
                    message=str(error).replace(str(capture),'[private source capture]')[:1000]
                    raise ValueError(f'Table Input {fmt} could not parse the selected source/options: {message}') from error
                raise
            finally:
                if reserved:budget.resize(reserved,0)


def preview_file(path,configuration,check_cancelled=lambda:None,*,_internal=False):
    fmt,options=normalize_options(configuration,internal=_internal)
    if 'n_rows' in options:options['n_rows']=min(options['n_rows'],8) if options['n_rows'] is not None else 8
    if fmt=='excel' and options['engine']=='calamine':
        read=dict(options['read_options']);read['n_rows']=min(read['n_rows'],8) if read['n_rows'] is not None else 8;options['read_options']=read
    table=source_file(path,{'format':fmt,'options':options},check_cancelled,_internal=_internal)
    try:return {'columns':list(table.columns),'rows':[list(row) for row in table.rows[:8]],'sourceSHA256':table.source_digest,'sampleOnly':True}
    finally:table.rows.store.close()


def preview_table(path,configuration,check_cancelled=lambda:None):
    result=preview_file(path,configuration,check_cancelled)
    return [column['name'] for column in result['columns']],result['rows'],result['sourceSHA256']


def source_typed_snapshot(path,check_cancelled=lambda:None,*,sampling=None,allow_temporal=False):
    return source_file(path,{'format':'ipc','options':{}},check_cancelled,sampling=sampling,_internal=True,_allow_temporal=allow_temporal)


def preview_typed_snapshot(path,check_cancelled=lambda:None):
    result=preview_file(path,{'format':'ipc','options':{}},check_cancelled,_internal=True)
    return [column['name'] for column in result['columns']],result['rows'],result['sourceSHA256']
