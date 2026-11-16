"""Pinned, disk-backed table exploration. Never opens source bindings or executes SQL from callers."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time

if __package__:from . import dataflow_temporal
else:import dataflow_temporal

if __package__:from .table_transfer_policy import MAX_ARTIFACT_BYTES, MAX_PAGE_BYTES, MAX_PAGE_ROWS, MAX_COLUMNS, MAX_ROWS, MAX_CELLS
else:from table_transfer_policy import MAX_ARTIFACT_BYTES, MAX_PAGE_BYTES, MAX_PAGE_ROWS, MAX_COLUMNS, MAX_ROWS, MAX_CELLS

MAX_INDEX_BYTES = MAX_ARTIFACT_BYTES
MAX_CELL_CHARACTERS = 2000


def _capacity(name,legacy):
    if __package__:from .workload_policy import captured_policy
    else:from workload_policy import captured_policy
    policy=captured_policy()
    return legacy if policy is None else policy.limit(name)


def index_allowance():return _capacity('retainedBytes',MAX_INDEX_BYTES)



def _cell(value,column):
    if value is None:return column['nullable']
    dtype=column['dtype']
    if dtype in dataflow_temporal.DTYPES:return dataflow_temporal.valid_value(value,dtype)
    if dtype=='int64Sequence':
        if __package__:from .named_inputs import valid_sequence
        else:from named_inputs import valid_sequence
        return valid_sequence(value)
    if dtype in ('audio','tensor'):
        if __package__:from .media_artifacts import is_reference
        else:from media_artifacts import is_reference
        return is_reference(value,dtype)
    if dtype=='image':
        if __package__:from .image_artifacts import is_reference
        else:from image_artifacts import is_reference
        return is_reference(value)
    if dtype=='string':return type(value) is str
    if dtype=='bool':return type(value) is bool
    if dtype=='int64':return type(value) is int and -(2**63)<=value<2**63
    return dtype=='float64' and type(value) in (int,float) and math.isfinite(value)


def _display(value,column=None):
    if column is not None and column['dtype'] in dataflow_temporal.DTYPES:
        return dataflow_temporal.format_value(value,column['dtype'],timezone=column['temporal'].get('timezone','UTC')) or ''
    if type(value) is dict and value.get('kind') in ('audioSamplesRef','preparedTensorRef'):
        return ('Audio' if value['kind']=='audioSamplesRef' else 'Tensor')+' '+'×'.join(map(str,value['shape']))+' · '+value['dtype']+' · '+' × '.join(value['axes'])
    if type(value) is dict:
        return f"Image {value['width']}×{value['height']} · {value['channels']} channels · {value['dtype']} · {value['layout']}"
    return '' if value is None else str(value)


def _json(value):
    # Live column/image dictionaries and canonical cache reloads may have
    # different insertion order. Object order is not table semantics: pin the
    # same bytes for both while preserving ordered columns and row arrays.
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':'))


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _pin(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Table View requires an exact artifact SHA-256 pin.')


def _open_directory(path):
    # Reject traversal before Path/abspath can normalize away explicit dots.
    raw = os.fsdecode(os.fspath(path))
    if any(part in ('.', '..') for part in raw.split('/')):
        raise ValueError('Table View requires a canonical artifact path.')
    absolute = os.path.abspath(raw)
    if sys.platform == 'darwin':
        # Seatbelt may authorize only the leaf. Opening ancestors asks for
        # additional read-data authority; the kernel can instead reject symlinks
        # in every component atomically while opening the authorized leaf.
        return os.open(absolute, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW_ANY | os.O_CLOEXEC)
    parts = Path(absolute).parts
    parent = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in parts[1:]:
            if part in ('.', '..'):
                raise ValueError('Table View requires a canonical artifact path.')
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            os.close(parent)
            parent = child
        result = parent
        parent = None
        return result
    finally:
        if parent is not None:
            os.close(parent)


def _open_regular(path,maximum=None):
    path = Path(path).absolute()
    parent = _open_directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= (index_allowance() if maximum is None else maximum):
            os.close(descriptor)
            raise ValueError('Table View index must be a regular file within its captured allowance.')
        return descriptor, info
    finally:
        os.close(parent)


def build_index(table, destination, *, source_sha256, check_cancelled=lambda: None, scratch_budget=None, retention_budget=None):
    """Publish one immutable index; caller charges its receipt to the artifact inventory.

    The table is already materialized by execution. It is consumed as bounded batches;
    this function does not regenerate a table or follow any external source path.
    """
    _pin(source_sha256)
    for column in table.columns:dataflow_temporal.validate_column(column)
    if __package__:
        from .table_batches import table_batches
    else:
        from table_batches import table_batches
    check_cancelled()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Table View index destination must be new.')
    parent_fd = _open_directory(destination.parent)
    parent_identity = os.fstat(parent_fd)
    directory = Path(tempfile.mkdtemp(prefix='.table-view-', dir=destination.parent))
    directory_identity = directory.stat()
    directory_fd = _open_directory(directory)
    database = directory / 'index.sqlite3'
    connection = None
    reserved = 0;retained_reserved=0;published=False
    if scratch_budget is None:
        if __package__:from .dataflow_disk import current_scope
        else:from dataflow_disk import current_scope
        scope=current_scope()
        if scope is not None:scratch_budget=scope[1]
    def account():
        nonlocal reserved,retained_reserved
        current=connection.execute('PRAGMA page_count').fetchone()[0]*4096
        if scratch_budget is not None:scratch_budget.resize(reserved,current);reserved=current
        if retention_budget is not None:
            retention_budget.reserve(current-retained_reserved);retained_reserved=current
    try:
        connection = sqlite3.connect(database)
        connection.execute('PRAGMA page_size=4096')
        allowance=min(index_allowance(),scratch_budget.limit-scratch_budget.used if scratch_budget is not None else index_allowance(),
            retention_budget.maximum_bytes-retention_budget.used_bytes if retention_budget is not None else index_allowance())
        if allowance<3*4096:raise ValueError('Table View index exceeds its shared storage allowance.')
        connection.execute(f'PRAGMA max_page_count={allowance // 4096}')
        connection.execute('PRAGMA cache_size=-2048')
        connection.execute('PRAGMA temp_store=FILE')
        connection.execute('PRAGMA mmap_size=0')
        connection.execute('PRAGMA journal_mode=OFF')
        connection.execute('CREATE TABLE metadata (value TEXT NOT NULL)')
        connection.execute('CREATE TABLE records (position INTEGER PRIMARY KEY, values_json TEXT NOT NULL, search_text TEXT)')
        count = 0
        for batch in table_batches(table, check_cancelled=check_cancelled):
            for row in batch.rows:
                if len(row)!=len(table.columns) or any(not _cell(value,column) for value,column in zip(row,table.columns)):raise ValueError('Table View contains an invalid typed cell.')
                text = _json(row)
                search = '\n'.join(_display(value,column) for value,column in zip(row,table.columns)).casefold()
                # Large cells already live in values_json. Avoid doubling the
                # artifact merely to search them; schema2 computes their search
                # text one bounded row at a time, retaining the small-row fast path.
                connection.execute('INSERT INTO records VALUES (?,?,?)', (count, text, search if len(search.encode()) <= 4096 else None))
                count += 1
            check_cancelled()
            account()
            if database.stat().st_size > index_allowance():
                raise ValueError('Table View index exceeds its captured artifact budget.')
        metadata = {'schemaVersion': 2, 'kind': 'tableViewIndex', 'sourceSHA256': source_sha256,
                    'columns': list(table.columns), 'rowCount': count}
        if len(_json(metadata).encode()) > 100_000:
            raise ValueError('Table View metadata exceeds its budget.')
        connection.execute('INSERT INTO metadata VALUES (?)', (_json(metadata),))
        connection.commit()
        account()
        connection.close()
        connection = None
        check_cancelled()
        descriptor, info = _open_regular(database)
        try:
            digest = hashlib.sha256()
            while block := os.read(descriptor, 65536):
                check_cancelled()
                digest.update(block)
            if _identity(os.fstat(descriptor)) != _identity(info):
                raise ValueError('Table View index changed before publication.')
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # Hard-link creation is atomic and refuses replacement of any existing object.
        current_parent = os.stat(destination.parent, follow_symlinks=False)
        if (current_parent.st_dev, current_parent.st_ino) != (parent_identity.st_dev, parent_identity.st_ino):
            raise ValueError('Table View output directory changed before publication.')
        os.link('index.sqlite3', destination.name, src_dir_fd=directory_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        published=True
        return {**metadata, 'index': {'bytes': info.st_size, 'sha256': digest.hexdigest()}}
    except sqlite3.OperationalError as error:
        if 'full' in str(error).lower():raise ValueError('Table exploration exceeds its index storage byte budget; reduce the captured output size.') from error
        raise
    finally:
        if connection is not None:
            connection.close()
        if scratch_budget is not None and reserved:scratch_budget.resize(reserved,0)
        if retention_budget is not None and not published and retained_reserved:retention_budget.release(retained_reserved)
        os.close(directory_fd)
        os.close(parent_fd)
        if directory.exists() and (directory.lstat().st_dev, directory.lstat().st_ino) == (directory_identity.st_dev, directory_identity.st_ino):
            shutil.rmtree(directory)


MAX_SORT_BYTES = 100_000_000


def _sorted_positions(connection, directory, columns, predicate, parameters, column, descending, offset, limit, check):
    """A bounded disk B-tree avoids SQLite's unaccounted external-sort files.

    Its primary key IS the requested order; no CREATE INDEX or ORDER BY sort is
    needed. page_count is enforced by SQLite on this separate writable database.
    """
    sorting=sqlite3.connect(directory/'sorting.sqlite3')
    try:
        sorting.execute('PRAGMA page_size=4096')
        sorting.execute('PRAGMA max_page_count='+str(_capacity('scratchBytes',MAX_SORT_BYTES)//4096))
        sorting.execute('PRAGMA cache_size=-2048')
        sorting.execute('PRAGMA mmap_size=0')
        sorting.execute('PRAGMA journal_mode=OFF')
        sorting.execute('PRAGMA synchronous=OFF')
        sorting.execute('PRAGMA trusted_schema=OFF')
        sorting.execute('CREATE TABLE ordering (absent INTEGER NOT NULL, value NOT NULL, position INTEGER NOT NULL, PRIMARY KEY(absent ASC,value '+('DESC' if descending else 'ASC')+',position ASC)) WITHOUT ROWID')
        for position,raw in connection.execute('SELECT position,values_json FROM records'+predicate,parameters):
            check()
            values=json.loads(raw)
            if type(values) is not list or len(values)!=len(columns):raise ValueError('Invalid Table View sorting row.')
            value=values[column]
            if value is not None and (type(value) not in (str,int,float,bool) or type(value) is float and not math.isfinite(value)):
                raise ValueError('Invalid Table View sorting value.')
            sorting.execute('INSERT INTO ordering VALUES(?,?,?)',(int(value is None),'' if value is None else value,position))
        sorting.commit();check()
        # Scanning the WITHOUT ROWID primary key requires no temporary sorter.
        statement='SELECT position FROM ordering ORDER BY absent ASC,value '+('DESC' if descending else 'ASC')+',position ASC LIMIT ? OFFSET ?'
        if any('TEMP B-TREE' in row[3] for row in sorting.execute('EXPLAIN QUERY PLAN '+statement,(limit,offset))):
            raise ValueError('Table View sort unexpectedly requires unbounded temporary storage.')
        return [row[0] for row in sorting.execute(statement,(limit,offset))]
    finally:sorting.close()


def query_index(path, expected_sha256, *, offset=0, limit=100, search='', source_sha256=None,
                check_cancelled=lambda: None, scratch_directory=None, sort_column=None, sort_descending=False, full_cell_column=None, full_cell_start=0):
    """Return a bounded read-only page from an exact private copy of the pinned index.

    Search is a literal case-insensitive substring across displayed scalar values;
    it never accepts SQL or reparses source files. Offset refers to filtered rows.
    """
    _pin(expected_sha256)
    if source_sha256 is not None:
        _pin(source_sha256)
    if type(offset) is not int or not 0 <= offset <= _capacity('tableRows',MAX_ROWS) or type(limit) is not int or not 1 <= limit <= MAX_PAGE_ROWS:
        raise ValueError('Table View page must request 1–100 rows and a bounded offset.')
    if type(search) is not str or len(search) > 256 or '\0' in search:
        raise ValueError('Table View search must be at most 256 characters.')
    if (sort_column is not None and (type(sort_column) is not int or not 0<=sort_column<_capacity('tableColumns',MAX_COLUMNS))) or type(sort_descending) is not bool:
        raise ValueError('Invalid Table View sort selection.')
    if full_cell_column is not None and (type(full_cell_column) is not int or not 0<=full_cell_column<_capacity('tableColumns',MAX_COLUMNS) or limit!=1 or search or sort_column is not None or sort_descending):
        raise ValueError('Cell inspection requires one original row and a bounded column.')
    if type(full_cell_start) is not int or not 0<=full_cell_start<=1_048_576 or (full_cell_column is None and full_cell_start):
        raise ValueError('Invalid cell preview position.')
    check_cancelled()
    descriptor, initial = _open_regular(path)
    directory = Path(tempfile.mkdtemp(prefix='kwip-table-query-',dir=scratch_directory))
    snapshot = directory / 'index.sqlite3'
    connection = None
    try:
        digest = hashlib.sha256()
        with snapshot.open('xb') as output:
            size = 0
            while block := os.read(descriptor, 65536):
                check_cancelled()
                size += len(block)
                if size > index_allowance():
                    raise ValueError('Table View index grew beyond its byte budget.')
                output.write(block)
                digest.update(block)
        if size != initial.st_size or digest.hexdigest() != expected_sha256 or _identity(os.fstat(descriptor)) != _identity(initial):
            raise ValueError('Table View index integrity check failed.')
        check_cancelled()
        connection = sqlite3.connect(snapshot.as_uri() + '?mode=ro&immutable=1', uri=True)
        if hasattr(connection, 'enable_load_extension'):
            connection.enable_load_extension(False)
        if hasattr(connection, 'setlimit'):
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_100_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 10_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 256)
        connection.execute('PRAGMA trusted_schema=OFF')
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA cache_size=-2048')
        connection.execute('PRAGMA mmap_size=0')
        started = time.monotonic()
        interruption = []
        def progress():
            try:
                check_cancelled()
                if time.monotonic() - started > 5:
                    raise TimeoutError('Table View search exceeded its five-second query budget. Narrow the search.')
            except BaseException as error:
                interruption.append(error)
                return 1
            return 0
        connection.set_progress_handler(progress, 1000)
        schema = connection.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        if schema != [('table', 'metadata'), ('table', 'records')]:
            raise ValueError('Table View index schema is unsupported.')
        raw = connection.execute('SELECT value FROM metadata LIMIT 2').fetchall()
        if len(raw) != 1 or len(raw[0][0].encode()) > 100_000:
            raise ValueError('Table View index metadata is invalid.')
        metadata = json.loads(raw[0][0])
        if set(metadata) != {'schemaVersion','kind','sourceSHA256','columns','rowCount'} or metadata['schemaVersion'] not in (1,2) or metadata['kind'] != 'tableViewIndex':
            raise ValueError('Table View index metadata is unsupported.')
        _pin(metadata['sourceSHA256'])
        if source_sha256 is not None and source_sha256 != metadata['sourceSHA256']:
            raise ValueError('Table View index belongs to another table artifact.')
        columns = metadata['columns']
        if type(columns) is not list or not 1 <= len(columns) <= _capacity('tableColumns',MAX_COLUMNS) or type(metadata['rowCount']) is not int or not 0 <= metadata['rowCount'] <= min(_capacity('tableRows',MAX_ROWS),_capacity('tableCells',MAX_CELLS) // len(columns)):
            raise ValueError('Table View index dimensions are invalid.')
        for column in columns:
            if type(column) is not dict or type(column.get('name')) is not str or not 1 <= len(column['name']) <= 4096 or column.get('dtype') not in ('string','int64','float64','bool','image','audio','tensor','int64Sequence')+dataflow_temporal.DTYPES or type(column.get('nullable')) is not bool:
                raise ValueError('Table View index columns are invalid.')
            dataflow_temporal.validate_column(column)
        if len({column['name'] for column in columns}) != len(columns):
            raise ValueError('Table View index column names are duplicated.')
        if full_cell_column is not None and (full_cell_column>=len(columns) or columns[full_cell_column]['dtype'] not in ('string','int64','float64','bool')+dataflow_temporal.DTYPES or offset>=metadata['rowCount']):
            raise ValueError('Select an existing scalar cell for text inspection.')
        def row_search(raw):
            try:
                check_cancelled()
                if time.monotonic()-started>5:raise TimeoutError('Table View search exceeded its five-second query budget. Narrow the search.')
                if type(raw) is not str or len(raw.encode())>1_048_576:raise ValueError('Table View row exceeds its byte budget.')
                values=json.loads(raw)
                if type(values) is not list or len(values)!=len(columns) or any(not _cell(v,c) for v,c in zip(values,columns)):
                    raise ValueError('Table View contains an invalid search row.')
                return '\n'.join(_display(v,c) for v,c in zip(values,columns)).casefold()
            except BaseException as error:
                interruption.append(error);raise
        if metadata['schemaVersion']==2:
            connection.create_function('kwip_row_search',1,row_search)
        search_value='coalesce(search_text,kwip_row_search(values_json))' if metadata['schemaVersion']==2 else 'search_text'
        predicate = ' WHERE instr('+search_value+',?)>0' if search else ''
        parameters = [search.casefold()] if search else []
        try:
            matched = connection.execute('SELECT count(*) FROM records' + predicate, parameters).fetchone()[0]
            direction = 'DESC' if sort_descending else 'ASC'
            if sort_column is None:
                cursor=connection.execute('SELECT position,values_json FROM records'+predicate+' ORDER BY position '+direction+' LIMIT ? OFFSET ?',parameters+[limit,offset])
            else:
                if sort_column>=len(columns):raise ValueError('Table View sort column is outside its schema.')
                if columns[sort_column]['dtype'] in ('image','audio','tensor','int64Sequence'):raise ValueError('Image and token-sequence columns cannot be sorted as scalar values.')
                def sort_check():
                    check_cancelled()
                    if time.monotonic()-started>5:raise TimeoutError('Table View sorting exceeded its five-second query budget. Narrow the search.')
                selected=_sorted_positions(connection,directory,columns,predicate,parameters,sort_column,sort_descending,offset,limit,sort_check)
                cursor=(connection.execute('SELECT position,values_json FROM records WHERE position=?',(position,)).fetchone() for position in selected)
            rows, positions, consumed, truncated = [], [], len(_json(columns).encode()), False
            cell_total=0
            for position, raw_row in cursor:
                check_cancelled()
                if len(raw_row.encode()) > 1_048_576:
                    raise ValueError('Table View row exceeds its byte budget.')
                values = json.loads(raw_row)
                if type(values) is not list or len(values) != len(columns):
                    raise ValueError('Table View row does not match its schema.')
                if any(not _cell(value,column) for value,column in zip(values,columns)):
                    raise ValueError('Table View contains an unsupported cell value.')
                if full_cell_column is not None:
                    if full_cell_column>=len(columns) or columns[full_cell_column]['dtype'] not in ('string','int64','float64','bool')+dataflow_temporal.DTYPES:
                        raise ValueError('Select a scalar cell for text inspection.')
                    value=values[full_cell_column]
                    if isinstance(value,str):
                        cell_total=len(value)
                        if full_cell_start>cell_total:raise ValueError('Cell preview position exceeds the captured text.')
                        value=value[full_cell_start:full_cell_start+16384]
                    elif full_cell_start:raise ValueError('Only text cells support paging.')
                    rendered=[value]
                else:rendered = [value[:MAX_CELL_CHARACTERS] if isinstance(value, str) else value for value in values]
                truncated = truncated or (full_cell_column is None and rendered != values)
                row_bytes = len(_json(rendered).encode())
                if consumed + row_bytes > MAX_PAGE_BYTES:
                    if not rows:
                        raise ValueError('Table View row exceeds its display byte budget.')
                    break
                rows.append(rendered)
                positions.append(position)
                consumed += row_bytes
        except sqlite3.OperationalError:
            if interruption:
                raise interruption[0]
            raise
        check_cancelled()
        result={'schemaVersion': 1, 'columns': [columns[full_cell_column]] if full_cell_column is not None else columns, 'rows': rows, 'positions': positions,
                'offset': offset, 'nextOffset': offset + len(rows) if full_cell_column is None and offset + len(rows) < matched else None,
                'totalRows': metadata['rowCount'], 'matchedRows': matched, 'cellTruncated': truncated,
                'sourceSHA256': metadata['sourceSHA256']}
        if full_cell_column is not None:result.update(cellOffset=full_cell_start,cellTotalCharacters=cell_total)
        return result
    finally:
        os.close(descriptor)
        if connection is not None:
            connection.close()
        shutil.rmtree(directory)


def validate_query(value,*,workload=None):
    if workload is not None:
        if __package__:from .workload_policy import workload_scope
        else:from workload_policy import workload_scope
        with workload_scope(workload):return validate_query(value)
    base={'schemaVersion','offset','limit','search','sourceSHA256','indexSHA256'}
    if type(value) is not dict or not base<=set(value) or set(value)-base-{'sortColumn','sortDescending','inputKind','formula','fullCellColumn','fullCellStart'} or type(value['schemaVersion']) is not int or value['schemaVersion'] not in (1,2,3,4,5):
        raise ValueError('Unsupported Table View query request.')
    if value['schemaVersion']==5:
        start=value.get('fullCellStart',0)
        if type(start) is not int or not 0<=start<=1_048_576:raise ValueError('Invalid cell preview position')
        column=value.get('fullCellColumn')
        if type(column) is not int or not 0<=column<_capacity('tableColumns',MAX_COLUMNS) or value['limit']!=1 or value['search'] or value.get('sortColumn') is not None or value.get('sortDescending',False) or 'formula' in value or value.get('inputKind') not in (None,'index','table'):raise ValueError('Invalid full-cell inspection request')
    elif 'fullCellColumn' in value or 'fullCellStart' in value:raise ValueError('Full-cell inspection requires schema 5')
    if value['schemaVersion']==4:
        if __package__:from .dataflow_formula import validate_config
        else:from dataflow_formula import validate_config
        validate_config(value.get('formula'))
        if type(value['offset']) is not int or type(value['limit']) is not int or value['offset']!=0 or not 1<=value['limit']<=8 or value['search'] or value.get('sortColumn') is not None or value.get('sortDescending',False) or value.get('inputKind') not in (None,'index','table'):raise ValueError('Formula samples require first 1–8 captured rows without sorting/search')
    elif 'formula' in value:raise ValueError('Formula draft requires query schema 4')
    if value['schemaVersion']==1 and set(value)!=base:raise ValueError('Legacy Table View query cannot carry sort fields.')
    if value['schemaVersion']<3 and 'inputKind' in value:raise ValueError('Legacy Table View query cannot select another input kind.')
    if value['schemaVersion']==3 and value.get('inputKind') not in ('table','index'):raise ValueError('Unsupported Table View input kind.')
    if value.get('inputKind')=='table' and value['sourceSHA256']!=value['indexSHA256']:raise ValueError('Table input must pin its exact source artifact.')
    column=value.get('sortColumn');descending=value.get('sortDescending',False)
    if (column is not None and (type(column) is not int or not 0<=column<_capacity('tableColumns',MAX_COLUMNS))) or type(descending) is not bool:raise ValueError('Invalid Table View sort selection.')
    _pin(value['sourceSHA256']);_pin(value['indexSHA256'])
    if type(value['offset']) is not int or not 0 <= value['offset'] <= _capacity('tableRows',MAX_ROWS) or type(value['limit']) is not int or not 1 <= value['limit'] <= MAX_PAGE_ROWS:
        raise ValueError('Table View query page is outside its bounds.')
    if type(value['search']) is not str or len(value['search']) > 256 or '\0' in value['search']:
        raise ValueError('Table View query search exceeds its bounds.')
    return value


def parse_contract(value, base=None):
    if __package__:
        from .worker_authorization import require, hash_value
    else:
        from worker_authorization import require, hash_value
    args=value['arguments'];files=value['inputFiles']
    if __package__:from .workload_policy import WorkloadPolicy,LEGACY
    else:from workload_policy import WorkloadPolicy,LEGACY
    captured=value.get('executionBudget',{}).get('resources',{}).get('workloadPolicy')
    policy=LEGACY if captured is None else WorkloadPolicy.from_document(captured)
    def segment(name):return type(name) is str and re.fullmatch(r'input/table-pack-[0-9a-f]{32}\.kwipdata',name) is not None
    require(type(args) is list and len(args)==2 and args[0]=='--query-table-view' and type(args[1]) is str and '\0' not in args[1] and len(args[1].encode())<=4096,'Invalid Table View query arguments')
    require(type(files) is list and 2<=len(files)<=(2 if captured is None else 62500),'Table View query exceeds its captured inventory allowance')
    inventory={}
    for file in files:
        require(type(file) is dict and set(file)=={'path','bytes','sha256'},'Invalid Table View captured file')
        require((file['path'] in ('input/view-query.json','input/view.sqlite3','input/view.table.json') or captured is not None and segment(file['path'])) and file['path'] not in inventory,'Invalid Table View input path')
        maximum=256_000 if file['path']=='input/view-query.json' else policy.limit('retainedBytes') if segment(file['path']) or captured is not None and file['path']=='input/view.sqlite3' else MAX_INDEX_BYTES
        require(type(file['bytes']) is int and 0<file['bytes']<=maximum,'Table View captured input exceeds its budget')
        hash_value(file['sha256']);inventory[file['path']]=file
    ordinary={name for name in inventory if not segment(name)}
    require(ordinary in ({'input/view-query.json','input/view.sqlite3'},{'input/view-query.json','input/view.table.json'}) and (len(inventory)==2 or 'input/view.table.json' in ordinary),'Table View requires exactly one captured table or index')
    require(sum(entry['bytes'] for entry in inventory.values())<=policy.limit('retainedBytes')+256_000,'Table View captured inputs exceed their aggregate allowance')
    path=Path(args[1])
    root=Path(base) if base is not None else path.parent.parent
    require(path.is_absolute() and str(path)==args[1] and '..' not in path.parts and args[1]==str(root/'input/view-query.json'),'Table View query escapes its retained operation')
    return inventory


def verify_inputs(value, root, cache):
    if __package__:
        from .worker_input_contract import verify_pinned_file
    else:
        from worker_input_contract import verify_pinned_file
    inventory=parse_contract(value,root)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
    descriptor,info=_open_regular(Path(root)/'input/view-query.json',256_000)
    try:
        if info.st_size>256_000:raise ValueError('Table View request exceeds its byte budget.')
        raw=os.read(descriptor,256_001)
        if len(raw)!=info.st_size or hashlib.sha256(raw).hexdigest()!=inventory['input/view-query.json']['sha256'] or _identity(os.fstat(descriptor))!=_identity(info):
            raise ValueError('Table View request changed after admission.')
        if __package__:from .preparation_artifact_limits import request_policy
        else:from preparation_artifact_limits import request_policy
        request=validate_query(json.loads(raw),workload=request_policy(value))
    finally:os.close(descriptor)
    input_path='input/view.table.json' if request.get('inputKind')=='table' else 'input/view.sqlite3'
    if input_path not in inventory or request['indexSHA256'] != inventory[input_path]['sha256']:
        raise ValueError('Table View query input differs from its captured inventory.')


def run_query(request_path, emit, check_cancelled=lambda:None, scratch_directory=None):
    """CLI entry; managed-worker admission has already pinned both owned input files."""
    check_cancelled()
    request_path=Path(request_path)
    descriptor,info=_open_regular(request_path,256_000)
    try:
        if info.st_size>256_000:raise ValueError('Table View query request exceeds its budget.')
        raw=os.read(descriptor,256_001)
        if len(raw)!=info.st_size or _identity(os.fstat(descriptor))!=_identity(info):raise ValueError('Table View query request changed.')
    finally:os.close(descriptor)
    request=validate_query(json.loads(raw))
    if (request.get('inputKind')=='table' and not request['search']
        and request.get('sortColumn') is None and not request.get('sortDescending',False)
        and request.get('formula') is None):
        if __package__:
            from .segmented_table_capture import read_page
            from .dataflow_disk import disk_scope
        else:
            from segmented_table_capture import read_page
            from dataflow_disk import disk_scope
        cell_column=request.get('fullCellColumn');cell_start=request.get('fullCellStart',0)
        state={'truncated':False,'cellTotal':0}
        def render(row,columns):
            if cell_column is not None:
                if cell_column>=len(columns) or columns[cell_column]['dtype'] not in ('string','int64','float64','bool')+dataflow_temporal.DTYPES:
                    raise ValueError('Select an existing scalar cell for text inspection.')
                value=row[cell_column]
                if type(value) is str:
                    state['cellTotal']=len(value)
                    if cell_start>len(value):raise ValueError('Cell preview position exceeds the captured text.')
                    value=value[cell_start:cell_start+16384]
                elif cell_start:raise ValueError('Only text cells support paging.')
                return [value]
            values=[v[:MAX_CELL_CHARACTERS] if type(v) is str else v for v in row]
            state['truncated'] |= values!=list(row)
            return values
        with disk_scope(directory=scratch_directory,check_cancelled=check_cancelled):
            page=read_page(request_path.parent/'view.table.json',request['indexSHA256'],
                           offset=request['offset'],limit=request['limit'],check_cancelled=check_cancelled,render_row=render)
        if page is not None:
            if cell_column is not None:
                if not page['rows']:raise ValueError('Select an existing scalar cell for text inspection.')
                page['columns']=[page['columns'][cell_column]]
                page.update(cellOffset=cell_start,cellTotalCharacters=state['cellTotal'])
            count=page['totalRows'];following=request['offset']+len(page['rows'])
            page.update(schemaVersion=1,offset=request['offset'],nextOffset=following if cell_column is None and following<count else None,
                        matchedRows=count,cellTruncated=state['truncated'],sourceSHA256=request['sourceSHA256'])
            check_cancelled();emit({'event':'tableViewPage',**page});return
    index_path=request_path.parent/'view.sqlite3';index_pin=request['indexSHA256'];materialized=None
    if request.get('inputKind')=='table':
        if __package__:
            from .dataflow_disk import disk_scope
            from .dataflow_table_reader import read_table_artifact
        else:
            from dataflow_disk import disk_scope
            from dataflow_table_reader import read_table_artifact
        artifacts=request_path.parent.parent/'artifacts'
        artifacts.mkdir(exist_ok=True)
        # Fixed operation-owned output; no caller supplied source or destination path.
        index_path=artifacts/'view.sqlite3'
        table=None
        with disk_scope(directory=scratch_directory,check_cancelled=check_cancelled) as budget:
            try:
                table=read_table_artifact(request_path.parent/'view.table.json',request['indexSHA256'],check_cancelled)
                receipt=build_index(table,index_path,source_sha256=request['sourceSHA256'],check_cancelled=check_cancelled,scratch_budget=budget)
                index_pin=receipt['index']['sha256']
                materialized={'path':'artifacts/view.sqlite3',**receipt['index']}
            finally:
                if table is not None:
                    stores={id(store):store for value in (table.rows,table.row_ids) if (store:=getattr(value,'store',None)) is not None}
                    for store in stores.values():store.close()
    try:
        result=query_index(index_path,index_pin,offset=request['offset'],limit=request['limit'],search=request['search'],source_sha256=request['sourceSHA256'],check_cancelled=check_cancelled,scratch_directory=scratch_directory,sort_column=request.get('sortColumn'),sort_descending=request.get('sortDescending',False),full_cell_column=request.get('fullCellColumn'),full_cell_start=request.get('fullCellStart',0))
        if request.get('formula') is not None:
            if result['cellTruncated']:raise ValueError('Captured display cells were truncated; Formula draft test refuses incomplete values')
            if __package__:
                from . import dataflow
                from .dataflow_formula import apply
            else:
                import dataflow
                from dataflow_formula import apply
            table=dataflow.derive(dataflow.Table(tuple(result['columns']),tuple(tuple(row) for row in result['rows']),tuple('sample:'+str(i) for i in range(len(result['rows']))),request['sourceSHA256']))
            sample=apply(table,request['formula'],check_cancelled)
            try:
                result.update(columns=list(sample.columns),rows=[list(row) for row in sample.rows],positions=list(range(len(sample.rows))),totalRows=len(sample.rows),matchedRows=len(sample.rows),offset=0,nextOffset=None,cellTruncated=False)
            finally:sample.rows.store.close()
        if materialized is not None:result['materializedIndex']=materialized
        check_cancelled();emit({'event':'tableViewPage',**result})
    except BaseException:
        if materialized is not None:index_path.unlink(missing_ok=True)
        raise
