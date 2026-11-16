"""Standard-library-only admission for the native Polars CSV profile.

Do not import Polars here: source-inspection admission runs before worker authority.
Legacy-only dialect switches are refused unless they are default-compatible.
"""
DEFAULTS = {'parserProfile':'polars', 'separator':',', 'quoteChar':'"',
            'encoding':'utf-8-sig', 'headerMode':'firstRecord', 'skipRecords':0,
            'skipLines':0, 'skipRowsAfterHeader':0, 'commentPrefix':'',
            'emptyField':'text', 'nullMarkers':[], 'columnNullValues':{},
            'columns':[], 'schema':{}, 'schemaOverrides':{}, 'inferSchema':False,
            'inferSchemaLength':100, 'nRows':None, 'lowMemory':True,
            'tryParseDates':False, 'ignoreErrors':False, 'decimalComma':False,
            'newColumns':[], 'rowIndexName':'', 'rowIndexOffset':0, 'eolChar':'\n',
            'extraColumns':'raise', 'missingColumns':'raise', 'raiseIfEmpty':True,
            'usePyarrow':False}
_COMPATIBLE = {'escapeMode':'double', 'escapeChar':'\\', 'recordEndings':'auto',
               'allowQuotedNewlines':True, 'skipInitialSpace':False}


def normalize_options(options=None):
    if options is None: options = {}
    if not isinstance(options, dict): raise ValueError('Polars CSV options must be an object')
    if set(options) - set(DEFAULTS) - set(_COMPATIBLE) - {'skipBlankRecords'}:
        raise ValueError('Unknown Polars CSV option')
    for key, default in _COMPATIBLE.items():
        if key in options and (type(options[key]) is not type(default) or options[key] != default):
            raise ValueError('Polars CSV profile does not support the selected '+key+'; use a native CSV setting')
    if 'skipBlankRecords' in options and type(options['skipBlankRecords']) is not bool:
        raise ValueError('Invalid CSV skipBlankRecords')
    result = {**DEFAULTS, **{key:value for key,value in options.items() if key in DEFAULTS}}
    if result['parserProfile'] != 'polars': raise ValueError('Invalid Polars CSV parser profile')
    for key in ('separator','quoteChar'):
        value=result[key]
        if key=='quoteChar' and value is None: value='';result[key]=''
        if not isinstance(value,str) or (key=='separator' and not value) or len(value)>1 or value in ('\r','\n','\0'):
            raise ValueError('Polars CSV '+key+' must be a single byte'+(' or empty' if key=='quoteChar' else ''))
        try: encoded=value.encode('utf-8')
        except UnicodeError as error: raise ValueError('Invalid Polars CSV '+key) from error
        if len(encoded)>1: raise ValueError('Polars CSV '+key+' must be a single UTF-8 byte')
    if result['separator']==result['quoteChar']: raise ValueError('CSV delimiter must differ from quote')
    if result['encoding'] not in ('utf-8-sig','utf8','utf-8','utf8-lossy'):
        raise ValueError('Polars CSV supports UTF-8 or explicit UTF-8 lossy decoding')
    if result['encoding'] in ('utf8','utf-8'):result['encoding']='utf-8-sig'
    if result['headerMode'] not in ('firstRecord','generated'): raise ValueError('Invalid CSV header mode')
    if result['emptyField'] not in ('text','null'): raise ValueError('Invalid CSV empty field policy')
    for key in ('skipRecords','skipLines','skipRowsAfterHeader'):
        if type(result[key]) is not int or not 0<=result[key]<=100000:
            raise ValueError('CSV '+key+' must be an integer from 0 through 100000')
    if result['skipRecords'] and result['skipLines']:
        raise ValueError('Polars CSV supports skipped records or skipped physical lines, not both together')
    comment=result['commentPrefix']
    if not isinstance(comment,str) or len(comment)>200 or any(c in comment for c in '\r\n\0'):
        raise ValueError('CSV comment prefix must be at most 200 characters without newline or NUL')
    try: comment.encode('utf-8')
    except UnicodeError as error: raise ValueError('Invalid CSV comment prefix') from error
    markers=result['nullMarkers']
    if not isinstance(markers,list) or len(markers)>32 or any(not isinstance(x,str) or not x or len(x)>200 or '\0' in x for x in markers):
        raise ValueError('CSV null markers require at most 32 nonempty strings of 200 characters')
    try:
        for marker in markers: marker.encode('utf-8')
    except UnicodeError as error: raise ValueError('Invalid CSV null marker') from error
    if len(set(markers))!=len(markers): raise ValueError('CSV null markers must be unique')
    result['nullMarkers']=list(markers)
    for key in ('inferSchema','lowMemory','tryParseDates','ignoreErrors','decimalComma','raiseIfEmpty','usePyarrow'):
        if type(result[key]) is not bool:raise ValueError('CSV '+key+' requires a Boolean')
    if result['usePyarrow']:raise ValueError('CSV PyArrow reading cannot preserve the bounded streaming and explicit inference contract; use the native reader')
    for key in ('inferSchemaLength','nRows'):
        if result[key] is not None and (type(result[key]) is not int or not 0<=result[key]<=5_000_000):raise ValueError('CSV '+key+' requires a bounded nonnegative integer or null')
    if type(result['rowIndexOffset']) is not int or not 0<=result['rowIndexOffset']<=2**32-1:raise ValueError('Invalid CSV row index offset')
    if type(result['rowIndexName']) is not str or len(result['rowIndexName'])>200 or '\0' in result['rowIndexName']:raise ValueError('Invalid CSV row index name')
    if result['rowIndexOffset'] and not result['rowIndexName']:raise ValueError('CSV row index offset requires a row index name')
    if result['extraColumns'] not in ('raise','ignore') or result['missingColumns'] not in ('raise','insert'):raise ValueError('Invalid CSV column-width policy')
    eol=result['eolChar']
    if type(eol) is not str or len(eol)!=1 or len(eol.encode('utf-8'))!=1 or eol=='\0' or eol in (result['separator'],result['quoteChar']):raise ValueError('CSV record terminator must be one distinct UTF-8 byte')
    if result['decimalComma'] and result['separator']==',':raise ValueError('Decimal comma requires a non-comma delimiter')
    types={name:name for name in ('String','Boolean','Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64','Float32','Float64')}
    types.update({'string':'String','bool':'Boolean','int64':'Int64','float64':'Float64'})
    def named(value):return type(value) is str and bool(value.strip()) and len(value)<=200 and '\0' not in value
    for key in ('schema','schemaOverrides'):
        mapping=result[key]
        if type(mapping) is not dict or len(mapping)>128 or any(not named(k) or type(v) is not str or v not in types for k,v in mapping.items()):raise ValueError('CSV '+key+' requires up to 128 named supported scalar types')
        result[key]={k:types[mapping[k]] for k in sorted(mapping)}
    if result['schema'] and result['schemaOverrides']:raise ValueError('Choose full CSV schema or schema overrides, not both')
    columns=result['columns']
    if type(columns) is not list or len(columns)>128 or (columns and not (all(named(x) for x in columns) or all(type(x) is int and 0<=x<128 for x in columns))) or len(set(columns))!=len(columns):raise ValueError('CSV columns require distinct names or indexes, not a mixture')
    renamed=result['newColumns']
    if type(renamed) is not list or len(renamed)>128 or any(not named(x) for x in renamed) or len(set(renamed))!=len(renamed):raise ValueError('CSV new column names must be distinct bounded names')
    nulls=result['columnNullValues']
    if type(nulls) is not dict or len(nulls)>128 or any(not named(k) or type(v) is not str or len(v)>200 or '\0' in v for k,v in nulls.items()):raise ValueError('Invalid CSV per-column null markers')
    if nulls and result['nullMarkers']:raise ValueError('Choose global null markers or per-column null markers')
    result['columns']=list(columns);result['newColumns']=list(renamed);result['columnNullValues']=dict(nulls)
    return result
