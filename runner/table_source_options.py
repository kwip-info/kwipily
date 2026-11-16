"""Strict local Table Input reader options, separate from permissions and parsing."""
MAX_INTEGER=2**31-1
DTYPES={name:name for name in ('String','Boolean','Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64','Float32','Float64')}
DTYPES.update({'string':'String','bool':'Boolean','int64':'Int64','float64':'Float64'})
DEFAULTS={
 'parquet':dict(columns=[],n_rows=None,row_index_name='',row_index_offset=0,parallel='auto',use_statistics=True,low_memory=False,schema={},missing_columns='raise',extra_columns='raise',cast_options={},use_pyarrow=False,memory_map=True,pyarrow_options={}),
 'json':dict(schema={},schema_overrides={},infer_schema_length=100),
 'jsonl':dict(schema={},schema_overrides={},infer_schema_length=100,batch_size=1024,n_rows=None,low_memory=False,row_index_name='',row_index_offset=0,ignore_errors=False),
 'excel':dict(sheet_id=1,sheet_name='',table_name='',engine='calamine',engine_options={},read_options={},has_header=True,columns=[],schema_overrides={},infer_schema_length=100,drop_empty_rows=True,drop_empty_cols=True,raise_if_empty=True),
 'markdown':dict(table_index=0,encoding='utf-8-sig',has_header=True,columns=[],n_rows=None,schema_overrides={},cell_mode='text',link_mode='label',line_breaks='newline'),
 'xml':dict(row_xpath='./row',namespaces={},fields={},encoding='utf-8',include_attributes=False,attribute_prefix='@',n_rows=None,schema_overrides={}),
 'html':dict(table_index=0,encoding='utf-8',header=0,skip_rows=0,columns=[],n_rows=None,span_policy='reject',schema_overrides={},cell_mode='text',link_mode='label',include_hidden=False),
 'ipc':dict(columns=[],n_rows=None),
}
PARQUET_ARROW_DEFAULTS=dict(use_threads=True,pre_buffer=True,buffer_size=0,use_pandas_metadata=False,coerce_int96_timestamp_unit='ns',thrift_string_size_limit=10_000_000,thrift_container_size_limit=1_000_000,schema_depth_limit=64,page_checksum_verification=True)
PARQUET_CAST_DEFAULTS=dict(integer_cast='forbid',float_cast='forbid',categorical_to_string='forbid')
CALAMINE_READ=dict(header_row=0,column_names=[],skip_rows=0,n_rows=None,dtype_coercion='coerce',use_columns=[],dtypes={},skip_whitespace_tail_rows=False,whitespace_as_null=False)
XLSX_ENGINE=dict(quoting=0,dateformat='',timeformat='',floatformat='',scifloat=False,skip_empty_lines=False,skip_trailing_columns=False,escape_strings=False,no_line_breaks=False,hyperlinks=False,exclude_hidden_sheets=False,merge_cells=False,ignore_formats=[],skip_hidden_rows=False,ignore_invalid_char_data=False)
# CSV read options which affect converted XLSX content; top-level header/schema/projection
# own their fields, and separator/encoding are fixed by the converter's transport.
XLSX_READ=dict(quote_char='"',comment_prefix='',skip_rows=0,skip_lines=0,skip_rows_after_header=0,null_values=[],empty_string_is_null=True,infer_schema=True,ignore_errors=False,try_parse_dates=False,n_rows=None,low_memory=False,row_index_name='',row_index_offset=0,eol_char='\n',truncate_ragged_lines=False,decimal_comma=False,encoding='utf8',new_columns=[])


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_dtype_map(value):
    require(isinstance(value,dict) and len(value)<=128,'Reader schema must map up to 128 names to supported scalar dtypes')
    result={}
    for name,dtype in sorted(value.items()):
        require(type(name) is str and name.strip() and len(name)<=200 and type(dtype) is str and dtype in DTYPES,'Reader schema has an unsupported name or dtype')
        result[name]=DTYPES[dtype]
    return result or None


def dtype_map(value):
    names=validate_dtype_map(value)
    if not names:return None
    import polars as pl
    return {name:getattr(pl,dtype) for name,dtype in names.items()}


def _validate_fields(values,defaults):
    require(isinstance(values,dict) and not set(values)-set(defaults),'Unknown or unsafe local reader option')
    result={**defaults,**values}
    for key,value in result.items():
        default=defaults[key]
        if value is None and key in ('row_index_name','sheet_name','table_name','dateformat','timeformat','comment_prefix','quote_char'):
            result[key]='';continue
        if value is None and key=='dtypes':result[key]={};continue
        if value is None and key in ('thrift_string_size_limit','thrift_container_size_limit','schema_depth_limit'):
            continue
        if key in ('sheet_id','header','header_row','infer_schema_length') and value is None:continue
        if key=='skip_rows' and isinstance(value,list):
            require(len(value)<=100000 and all(type(x) is int and 0<=x<=MAX_INTEGER for x in value),'Invalid skipped row indexes');continue
        if key=='use_columns' and isinstance(value,str):
            require(0<len(value)<=4096 and '\0' not in value,'Invalid Excel column selection');continue
        if key=='dtypes' and isinstance(value,str):
            require(value in ('int','float','string','boolean'),'Unsupported calamine dtype');continue
        if key=='null_values' and isinstance(value,dict):
            require(len(value)<=128 and all(type(k) is str and k and len(k)<=200 and type(v) is str and len(v)<=200 for k,v in value.items()),'Invalid per-column null markers');continue
        if type(default) is bool:require(type(value) is bool,'Reader '+key+' requires a Boolean')
        elif type(default) is int:require(type(value) is int and 0<=value<=MAX_INTEGER,'Reader '+key+' requires a bounded integer')
        elif type(default) is str:require(type(value) is str and len(value)<=4096 and '\0' not in value,'Reader '+key+' requires bounded text')
        elif default is None:require(value is None or (type(value) is int and 0<=value<=MAX_INTEGER),'Reader '+key+' requires a bounded integer or null')
        elif isinstance(default,list):require(isinstance(value,list) and len(value)<=128,'Reader '+key+' requires a bounded list')
        elif isinstance(default,dict):require(isinstance(value,dict) and len(value)<=128,'Reader '+key+' requires a bounded mapping')
    return result


def normalize_options(configuration,*,internal=False):
    require(isinstance(configuration,dict) and set(configuration)<={'format','options'} and 'format' in configuration,'Table Input requires format/options only')
    fmt=configuration['format'];require(fmt in DEFAULTS and (fmt!='ipc' or internal),'Unsupported local Table Input format')
    options=_validate_fields(configuration.get('options',{}),DEFAULTS[fmt])
    for key in ('schema','schema_overrides'):
        if key in options:validate_dtype_map(options[key])
    if 'columns' in options:
        columns=options['columns'];require(not columns or (all(type(x) is str and x and len(x)<=200 for x in columns) or all(type(x) is int and 0<=x<128 for x in columns)),'Columns must be all names or all zero-based indexes')
        require(len(set(columns))==len(columns),'Selected columns must be unique')
    if 'infer_schema_length' in options:require(options['infer_schema_length'] is None or 0<=options['infer_schema_length']<=MAX_INTEGER,'Invalid schema inference length')
    if 'encoding' in options:require(options['encoding'] in ('utf-8','utf-8-sig','utf-16','cp1252'),'Unsupported local text encoding')
    if 'link_mode' in options:require(options['link_mode'] in ('label','url','both'),'Unsupported link representation')
    if fmt=='parquet':
        require(options['parallel'] in ('auto','columns','row_groups','none','prefiltered'),'Unsupported Parquet parallel strategy')
        require(options['missing_columns'] in ('raise','insert'),'Unsupported missing columns policy')
        require(options['extra_columns'] in ('raise','ignore'),'Unsupported extra columns policy')
        cast=options['cast_options']
        require(isinstance(cast,dict) and not set(cast)-set(PARQUET_CAST_DEFAULTS),'Unknown or unsupported scalar Parquet cast option')
        cast={**PARQUET_CAST_DEFAULTS,**cast}
        require(cast['integer_cast'] in ('forbid','upcast','allow-float'),'Unsupported integer cast policy')
        require(cast['categorical_to_string'] in ('forbid','allow'),'Unsupported categorical cast policy')
        floats=cast['float_cast']
        if isinstance(floats,list):
            require(bool(floats) and len(floats)<=2 and all(type(x) is str and x in ('upcast','downcast') for x in floats) and len(set(floats))==len(floats),'Float cast list permits unique upcast/downcast only')
            cast['float_cast']=sorted(floats)
        else:require(type(floats) is str and floats in ('forbid','upcast','downcast'),'Unsupported float cast policy')
        options['cast_options']=cast
        if options['use_pyarrow']:require(cast==PARQUET_CAST_DEFAULTS and options['extra_columns']=='raise','Native schema evolution/cast options require the native Polars engine')
        require(options['use_pyarrow'] or options['memory_map'],'Memory mapping is configurable only with the PyArrow engine')
        options['pyarrow_options']=_validate_fields(options['pyarrow_options'],PARQUET_ARROW_DEFAULTS)
        require(options['pyarrow_options']['coerce_int96_timestamp_unit'] in ('','ns','us','ms'),'Unsupported INT96 time unit')
        if options['pyarrow_options']!=PARQUET_ARROW_DEFAULTS:require(options['use_pyarrow'],'PyArrow options require the PyArrow engine')
    if fmt=='excel':
        require(options['engine'] in ('calamine','openpyxl','xlsx2csv'),'Unsupported Excel engine')
        require(options['sheet_id'] is None or options['sheet_id']>0,'Select one worksheet; sheet_id=0/multiple sheets are unsupported')
        if options['sheet_name']:require(options['sheet_id'] in (None,1),'Choose worksheet name or ID, not both')
        require(not (options['sheet_name'] and options['table_name']),'Choose worksheet name or named table')
        if options['table_name']:require(options['engine']=='calamine','Named Excel tables require calamine')
        if options['engine']=='calamine':
            require(not options['engine_options'],'Calamine has no public workbook engine options')
            supplied_read=options['read_options']
            options['read_options']=_validate_fields(options['read_options'],CALAMINE_READ)
            if options['table_name'] and 'header_row' not in supplied_read:options['read_options']['header_row']=None
            read=options['read_options'];require(read['dtype_coercion'] in ('strict','coerce'),'Unsupported Excel dtype coercion')
            if read['column_names']:require(all(type(x) is str and x and len(x)<=200 for x in read['column_names']),'Invalid Excel column names')
            if read['use_columns']:
                require(not options['columns'],'Choose top-level columns or calamine use_columns')
                if isinstance(read['use_columns'],list):require(all(type(x) is int and 0<=x<128 for x in read['use_columns']) or all(type(x) is str and x and len(x)<=200 for x in read['use_columns']),'Invalid calamine selected columns')
            if isinstance(read['dtypes'],dict):require(all(type(k) is str and k and len(k)<=200 and type(v) is str and v in ('int','float','string','boolean') for k,v in read['dtypes'].items()),'Unsupported calamine dtypes')
            else:require(not options['schema_overrides'],'Scalar calamine dtypes cannot combine with schema overrides')
            if options['table_name']:require(type(read['skip_rows']) is int,'Named Excel tables require an integer skip_rows')
        elif options['engine']=='openpyxl':
            options['engine_options']=_validate_fields(options['engine_options'],dict(read_only=True,rich_text=False))
            require(not options['read_options'],'openpyxl read_options are controlled by top-level header and inference options')
        else:
            options['engine_options']=_validate_fields(options['engine_options'],XLSX_ENGINE)
            options['read_options']=_validate_fields(options['read_options'],XLSX_READ)
            require(options['engine_options']['quoting'] in (0,1,2,3),'Unsupported CSV quoting mode')
            read=options['read_options']
            require(len(read['quote_char'].encode('utf-8'))<=1 and len(read['eol_char'].encode('utf-8'))==1,'Converted CSV quote and EOL require single-byte characters')
            if isinstance(read['null_values'],list):require(all(type(x) is str and len(x)<=200 for x in read['null_values']),'Invalid converted CSV null markers')
            require(all(type(x) is str and x and len(x)<=200 for x in read['new_columns']) and len(set(read['new_columns']))==len(read['new_columns']),'Invalid converted CSV replacement names')
            require(all(type(x) is str and len(x)<=200 for x in options['engine_options']['ignore_formats']),'Invalid ignored cell formats')
    if fmt=='markdown':
        require(options['cell_mode'] in ('text','markdown') and options['line_breaks'] in ('newline','space'),'Unsupported Markdown cell representation')
    if fmt=='html':
        require(options['cell_mode'] in ('text','html') and options['span_policy'] in ('reject','expand'),'Unsupported HTML table representation')
    if fmt=='xml':
        require(options['row_xpath'] and len(options['row_xpath'])<=2000,'XML requires a bounded row XPath')
        for mapping in ('namespaces','fields'):require(all(type(k) is str and k and len(k)<=200 and type(v) is str and v and len(v)<=2000 for k,v in options[mapping].items()),'Invalid XML '+mapping)
    return fmt,options


def required_reader_packages(graph):
    """Distributions actually needed by portable local readers, without imports."""
    packages=set()
    for node in graph.get('nodes',[]):
        if node.get('operator')=='kwip.data.sourceFiles':
            if __package__:
                from .file_input_options import validate_configuration,SPEC
                from .file_input_documents import required_document_packages
            else:
                from file_input_options import validate_configuration,SPEC
                from file_input_documents import required_document_packages
            config=validate_configuration(node.get('config',{}))
            if config['readMode']=='content':
                extensions=set(config['extensions'])
                formats={name for name,values in SPEC['extensions'].items() if not extensions or extensions.intersection(values)}
                packages|=required_document_packages(formats-{'image','audio'})
                if 'image' in formats:packages.add('Pillow')
            packages|={'polars','polars-runtime-32'}
            continue
        if node.get('operator')!='kwip.data.sourceCSV':continue
        config=node.get('config',{});fmt=config.get('format','csv');options=config.get('options',{})
        if fmt=='csv':
            if config.get('parserProfile')=='polars' or options.get('parserProfile')=='polars':packages|={'polars','polars-runtime-32'}
            continue
        packages|={'polars','polars-runtime-32'}
        if fmt=='parquet' or fmt=='database':packages.add('pyarrow')
        if fmt=='excel':
            engine=options.get('engine','calamine')
            packages|={'fastexcel','pyarrow'} if engine=='calamine' else {'openpyxl','defusedxml'} if engine=='openpyxl' else {'xlsx2csv'}
        if fmt=='markdown':packages.add('markdown-it-py')
        if fmt in ('xml','html'):packages.add('lxml')
    return packages


def reader_versions(graph):
    import importlib.metadata
    result={}
    for name in sorted(required_reader_packages(graph)):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:raise ValueError('Install the captured Table Input reader dependency: '+name) from error
    return result


OPTION_COVERAGE={
 'parquet':{'publicReader':'polars.scan_parquet / pyarrow.parquet.ParquetFile','exposed':sorted(DEFAULTS['parquet']), 'controlled':{'glob':'False; exactly one local literal grant','hive_partitioning':'False; no dataset directories','credential_provider':'None; no network authority','arrow_extensions_enabled':'False; no extension loading','memory_map':'Native streaming reader chooses its memory strategy; explicit toggle available only for PyArrow'},'inapplicable':{'storage_options':'Remote filesystems are outside local Table Input','hive_schema/try_parse_hive_dates':'No directory/hive input','include_file_paths':'Would expose private managed capture paths','hidden_file_prefix/cache':'Single immutable file with explicit operation ownership','use_pyarrow schema':'PyArrow engine has no native schema override; selected schema is not silently coerced','cast_options datetime/struct fields':'Datetime casts fixed forbid, missing/extra struct fields fixed raise; table schema is scalar only','cast_options precision':'Explicit allow-float and float downcast may reduce precision; never inferred or silently enabled','map ordering':'Schema/field name maps are canonically sorted; ordered columns selections preserve their explicit order','filters':'Filtering belongs to canvas Filter, not source parsing','decryption_properties/filesystem':'Credentials and filesystem capabilities do not belong in portable source options','binary_type/list_type/read_dictionary':'Typed tables currently preserve four scalar families, not binary/list/categorical metadata','_private_parameters':'Not supported public API'}},
 'json':{'publicReader':'polars.read_json','exposed':sorted(DEFAULTS['json']),'inapplicable':{}},
 'jsonl':{'publicReader':'polars.scan_ndjson','exposed':sorted(DEFAULTS['jsonl']),'controlled':{'credential_provider':'None; no network authority'},'inapplicable':{'storage_options/include_file_paths':'Remote/private-capture paths are not portable source data'}},
 'excel':{'publicReader':'polars.read_excel + fastexcel.load_sheet public option bridge','exposed':sorted(DEFAULTS['excel']),'engineOptions':{'calamine':[],'openpyxl':['read_only','rich_text'],'xlsx2csv':sorted(XLSX_ENGINE)},'readOptions':{'calamine':sorted(CALAMINE_READ),'openpyxl':[],'xlsx2csv':sorted(XLSX_READ)},'controlled':{'openpyxl keep_vba/keep_links':'False; no macro preservation or external workbook links','openpyxl data_only':'True; native Polars reads cached formula values','xlsx2csv delimiter/lineterminator/outputencoding':'Comma/LF/UTF-8 converter transport; reader options control resulting cell parse','schema_sample_rows/infer_schema_length':'Top-level inference owns the same setting; duplicate nested option refused'},'inapplicable':{'sheet_id=0/sheet sequences':'One Table Input output is one selected sheet/table','include_file_paths':'Managed capture path is not portable data','xlsx2csv sheetdelimiter/include/exclude_sheet_pattern':'Already one explicitly selected sheet','callable use_columns/skip_rows/converters':'No executable Python in source configuration','n_threads/batch_size/rechunk':'Removed from converted CSV public API in Polars 2.0'}},
 'markdown':{'publicReader':'markdown-it-py CommonMark table extension + explicit cell extraction','exposed':sorted(DEFAULTS['markdown']),'controlled':{'parser':'CommonMark with table rule; no rendering, URL fetch or executable extensions'}},
 'xml':{'publicReader':'lxml.etree.XMLParser/XPath; explicit record and scalar field selection','exposed':sorted(DEFAULTS['xml']),'controlled':{'resolve_entities/load_dtd/no_network/huge_tree':'False/False/True/False; DTD/entity declarations refused'},'inapplicable':{'stylesheet/XSLT/network/external entities':'Source parsing has no execution or external resource authority','automatic nested flattening':'Choose explicit scalar field XPath instead'}},
 'html':{'publicReader':'lxml.etree.HTMLParser; local stored table DOM','exposed':sorted(DEFAULTS['html']),'controlled':{'no_network/huge_tree':'True/False; no JS, stylesheet or remote-page execution'},'inapplicable':{'browser rendering/CSS stylesheet visibility':'Only stored hidden attributes and inline style are available locally'}},
}
