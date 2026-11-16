"""Local native CSV ingestion into owned typed columnar tables.

Polars owns value parsing and native blank-record/header behavior. Sources are
captured incrementally, then scanned with the native streaming engine. No URLs,
globs, arbitrary Python, or whole-table Python lists are admitted. Inference is
explicitly enabled by the captured parser configuration.
"""
import hashlib
import io
import math
import os
from pathlib import Path
import stat
import tempfile

import polars as pl

if __package__:
    from .polars_csv_configuration import normalize_options
    from .dataflow_columnar import ColumnarStore
    from .dataflow_disk import disk_scope, current_scope
else:
    from polars_csv_configuration import normalize_options
    from dataflow_columnar import ColumnarStore
    from dataflow_disk import disk_scope, current_scope

MAX_SOURCE_BYTES=10_000_000
MAX_PREVIEW_BYTES=1_048_576
MAX_CELLS=5_000_000
MAX_COLUMNS=128
BATCH_ROWS=4096


def csv_arguments(options=None):
    c=normalize_options(options)
    types={name:getattr(pl,name) for name in ('String','Boolean','Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64','Float32','Float64')}
    return dict(has_header=c['headerMode']=='firstRecord',separator=c['separator'],
                quote_char=c['quoteChar'] or None,encoding='utf8-lossy' if c['encoding']=='utf8-lossy' else 'utf8',
                skip_rows=c['skipRecords'],skip_lines=c['skipLines'],skip_rows_after_header=c['skipRowsAfterHeader'],
                comment_prefix=c['commentPrefix'] or None,null_values=c['columnNullValues'] or c['nullMarkers'] or None,
                empty_string_is_null=c['emptyField']=='null',infer_schema=c['inferSchema'],
                infer_schema_length=c['inferSchemaLength'],n_rows=c['nRows'],
                schema={k:types[v] for k,v in c['schema'].items()} or None,
                schema_overrides={k:types[v] for k,v in c['schemaOverrides'].items()} or None,
                ignore_errors=c['ignoreErrors'],low_memory=c['lowMemory'],glob=False,try_parse_dates=c['tryParseDates'],
                extra_columns=c['extraColumns'],missing_columns=c['missingColumns'],raise_if_empty=c['raiseIfEmpty'],
                decimal_comma=c['decimalComma'],new_columns=c['newColumns'] or None,
                row_index_name=c['rowIndexName'] or None,row_index_offset=c['rowIndexOffset'],eol_char=c['eolChar'])


def _project(lazy,config):
    columns=config['columns']
    if not columns:return lazy
    names=lazy.collect_schema().names()
    if type(columns[0]) is int:
        if any(index>=len(names) for index in columns):raise ValueError('CSV column index is outside the parsed schema')
        columns=[names[index] for index in columns]
    if any(name not in names for name in columns):raise ValueError('CSV selected column does not exist')
    return lazy.select(columns)


def _schema_columns(schema,nullable=True):
    if __package__:from .table_sources import _column_dtype
    else:from table_sources import _column_dtype
    return tuple({'name':name,'dtype':_column_dtype(dtype),'nullable':nullable} for name,dtype in schema.items())



def _identity(value):
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


def _open_source(path):
    if not isinstance(path,(str,os.PathLike)):
        raise ValueError('CSV source must be a local regular file')
    text=os.fspath(path)
    if not isinstance(text,str) or not text or '\0' in text or '://' in text:
        raise ValueError('CSV source must be one local file; URLs and globs are unsupported')
    path=Path(text)
    try:
        descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    except OSError as error:
        raise ValueError('CSV source must be an accessible local regular file without a symlink') from error
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise ValueError('CSV source must be a regular file without a symlink')
        if info.st_size>MAX_SOURCE_BYTES: raise ValueError('CSV source exceeds its 10 MB byte budget')
        return path,os.fdopen(descriptor,'rb'),_identity(info)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_source(path,stream,identity,expected,check):
    check(); stream.seek(0); hasher=hashlib.sha256();size=0
    while True:
        check();block=stream.read(262144)
        if not block:break
        size+=len(block)
        if size>MAX_SOURCE_BYTES:raise ValueError('CSV source changed while parsing; capture and prepare it again')
        hasher.update(block)
    try:
        current=path.lstat()
        stable=_identity(os.fstat(stream.fileno()))==identity==_identity(current) and stat.S_ISREG(current.st_mode)
    except OSError: stable=False
    if not stable or hasher.hexdigest()!=expected:
        raise ValueError('CSV source changed while parsing; capture and prepare it again')
    check()


def source_csv(path,options=None,check_cancelled=lambda:None,*,sampling=None):
    legacy_bom_preserving=isinstance(options,dict) and options.get('encoding')=='utf-8'
    c=normalize_options(options);arguments=csv_arguments(c);check_cancelled()
    if __package__:from .preparation_demand import normalize_sampling,sampling_row_limit
    else:from preparation_demand import normalize_sampling,sampling_row_limit
    policy=normalize_sampling(sampling)
    if policy['mode']=='rowCount':arguments['n_rows']=min(arguments['n_rows'],sampling_row_limit(policy)) if arguments['n_rows'] is not None else sampling_row_limit(policy)
    original,stream,identity=_open_source(path)
    store=None;reserved=0
    # Inherit the execution's single budget, including the immutable input copy.
    with stream, disk_scope(check_cancelled=check_cancelled):
        directory,budget,_=current_scope()
        with tempfile.TemporaryDirectory(prefix='.polars-csv-',dir=directory) as temporary:
            capture=Path(temporary)/'source.csv';hasher=hashlib.sha256();size=0
            try:
                with capture.open('xb') as target:
                    while True:
                        check_cancelled();block=stream.read(262144)
                        if not block:break
                        if size==0 and legacy_bom_preserving and block.startswith(b'\xef\xbb\xbf'):
                            raise ValueError('Polars consumes a UTF-8 BOM; choose UTF-8 (BOM accepted) instead of the legacy BOM-preserving UTF-8 option')
                        size+=len(block)
                        if size>MAX_SOURCE_BYTES:raise ValueError('CSV source exceeds its 10 MB byte budget')
                        budget.resize(reserved,size);reserved=size
                        target.write(block);hasher.update(block)
                check_cancelled();source=hasher.hexdigest()
                lazy=_project(pl.scan_csv(capture,**arguments),c)
                schema=lazy.collect_schema();header=schema.names()
                if policy['mode']=='percentage' and policy['percentage']<100:
                    # Percentage needs an honest full cardinality scan. Native
                    # streaming count avoids a Python row list or table capture.
                    check_cancelled()
                    total=lazy.select(pl.len().alias('__count')).collect(engine='streaming').item()
                    check_cancelled();lazy=lazy.head(sampling_row_limit(policy,total))
                if not 1<=len(header)<=MAX_COLUMNS:raise ValueError('CSV source exceeds its 128 column budget')
                if any(not name.strip() or len(name)>200 for name in header):raise ValueError('CSV headers require nonblank names of at most 200 characters')
                columns=_schema_columns(schema)
                store=ColumnarStore(columns,check_cancelled,identity_prefix=source+':')
                nulls=[False]*len(header);count=0
                batches=lazy.collect_batches(chunk_size=BATCH_ROWS,maintain_order=True,engine='streaming')
                try:
                    for batch in batches:
                        check_cancelled()
                        if batch.width!=len(header) or count+batch.height>MAX_CELLS//len(header):
                            raise ValueError('Table exceeds the five million cell storage budget')
                        for index,column in enumerate(batch):nulls[index] |= column.null_count()>0
                        physical=batch.select([pl.col(name).alias(f'c{index}') for index,name in enumerate(header)])
                        identities=pl.Series('identity',[f'{source}:{i}' for i in range(count,count+batch.height)],dtype=pl.String)
                        if store.append_frame(physical,identities):count+=batch.height
                        else:
                            for row in batch.iter_rows(buffer_size=256):
                                store.append(row,f'{source}:{count}');count+=1
                        check_cancelled()
                finally:del batches
                if not count and c['raiseIfEmpty']:raise ValueError('CSV source requires at least one data record')
                _verify_source(original,stream,identity,source,check_cancelled)
                rows,ids=store.finish()
                if __package__:from .dataflow import Table,derive
                else:from dataflow import Table,derive
                columns=tuple(dict(column,nullable=nulls[index]) for index,column in enumerate(columns))
                result=derive(Table(columns,rows,ids,source))
                check_cancelled()
                return result
            except pl.exceptions.PolarsError as error:
                if store is not None:store.close()
                raise ValueError('Polars could not parse this CSV with the selected settings; check its encoding, delimiter, quotes and record widths') from error
            except BaseException:
                if store is not None:store.close()
                raise
            finally:
                if reserved:budget.resize(reserved,0)


def _record_boundaries(raw,config,*,include_eof=True):
    """Locate complete record boundaries only; Polars still parses every value.

    Cutting before later records prevents Polars' read-ahead from invalidating an
    eight-row prefix because a later quoted record was truncated by the byte cap.
    """
    quote=ord(config['quoteChar']) if config['quoteChar'] else None
    delimiter=ord(config['separator']);comment=config['commentPrefix'].encode();eol=ord(config['eolChar'])
    i=0;start=0;quoted=False;field_start=True;physical=0;skip=config['skipLines']
    if raw.startswith(b'\xef\xbb\xbf'):i=3;start=3
    while i<len(raw):
        value=raw[i]
        if physical<skip:
            if value==eol:physical+=1;start=i+1;field_start=True;yield i+1
            i+=1;continue
        if i==start and comment and raw.startswith(comment,i):
            end=raw.find(bytes([eol]),i)
            if end<0:return
            i=end+1;start=i;field_start=True;yield i;continue
        if quoted:
            if value==quote:
                if i+1<len(raw) and raw[i+1]==quote:i+=2;continue
                quoted=False
        elif quote is not None and value==quote and field_start:
            quoted=True;field_start=False
        elif value==delimiter:field_start=True
        elif value==eol:
            i+=1;start=i;field_start=True;yield i;continue
        elif value!=13:field_start=False
        i+=1
    if include_eof and not quoted and start<len(raw):yield len(raw)


def preview_csv(path,options=None,max_rows=8,check_cancelled=lambda:None,*,include_source_hash=False,include_column_schema=False):
    if type(include_column_schema) is not bool:raise ValueError('CSV preview schema selection must be boolean')
    if type(include_source_hash) is not bool:raise ValueError('CSV preview hash selection must be boolean')
    if type(max_rows) is not int or not 1<=max_rows<=8:raise ValueError('CSV preview is bounded to eight rows')
    legacy_bom_preserving=isinstance(options,dict) and options.get('encoding')=='utf-8'
    c=normalize_options(options);args=csv_arguments(c);check_cancelled()
    original,stream,identity=_open_source(path)
    with stream:
        raw=stream.read(MAX_PREVIEW_BYTES)
        if legacy_bom_preserving and raw.startswith(b'\xef\xbb\xbf'):
            raise ValueError('Polars consumes a UTF-8 BOM; choose UTF-8 (BOM accepted)')
        # A source-inspection capture can itself be a truncated prefix. Never
        # require its trailing record after the requested sample was captured.
        expected=c['skipRecords']+c['skipRowsAfterHeader']+max_rows+(c['headerMode']=='firstRecord')
        count=0;boundary=None;previous=0;physical=c['skipLines'];started=False
        comment=c['commentPrefix'].encode()
        try:
            for end in _record_boundaries(raw,c,include_eof=len(raw)<MAX_PREVIEW_BYTES):
                check_cancelled();record=raw[previous:end];previous=end;boundary=end
                if physical:physical-=1;continue
                if record.startswith(b'\xef\xbb\xbf'):record=record[3:]
                if comment and record.startswith(comment):continue
                # Native parsing ignores leading blank records, while blanks
                # inside a table are real empty/null observations.
                if not started and not record.strip(b'\r\n'):continue
                started=True;count+=1
                if count>=expected:break
            if boundary is None:raise ValueError('CSV preview has no complete record in its 1 MiB sample budget')
            # Exactly one native parse, even for a byte budget of comment lines.
            args['n_rows']=min(max_rows,args['n_rows']) if args['n_rows'] is not None else max_rows
            last=_project(pl.read_csv(io.BytesIO(raw[:boundary]),**args).lazy(),c).collect(engine='streaming')
            column_schema=list(_schema_columns(last.schema,nullable=None))
            if last.width>MAX_COLUMNS:raise ValueError('CSV source exceeds its 128 column budget')
            if any(not name.strip() or len(name)>200 for name in last.columns):
                raise ValueError('CSV headers require nonblank names of at most 200 characters')
            check_cancelled()
            if _identity(os.fstat(stream.fileno()))!=identity or _identity(original.lstat())!=identity:
                raise ValueError('CSV source changed while inspecting; refresh the source')
            rows=[list(row) for row in last.iter_rows()]
            if any((type(value) is int and not -(2**63)<=value<2**63) or (type(value) is float and not math.isfinite(value)) for row in rows for value in row):
                raise ValueError('CSV preview has a numeric value outside the finite signed scalar storage contract')
            result=(list(last.columns),rows)
            if include_source_hash:result=(*result,hashlib.sha256(raw).hexdigest())
            if include_column_schema:result=(*result,column_schema)
            return result
        except pl.exceptions.PolarsError as error:
            raise ValueError('Polars could not parse the CSV preview with the selected settings') from error
