"""Verify and decode v1 table artifacts without retaining complete JSON or row arrays."""
import codecs
import hashlib
import json
import os
import re
import stat

if __package__:
    from .dataflow_disk import DiskStore, MAX_ROW_BYTES, disk_scope, current_check
else:
    from dataflow_disk import DiskStore, MAX_ROW_BYTES, disk_scope, current_check

if __package__:from .table_transfer_policy import MAX_ARTIFACT_BYTES, MAX_ROWS
else:from table_transfer_policy import MAX_ARTIFACT_BYTES, MAX_ROWS

MAX_FILE_BYTES = MAX_ARTIFACT_BYTES
MAX_METADATA_BYTES = 2_000_000


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate table artifact JSON key.')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('Nonfinite table artifact JSON value.')


class _Reader:
    def __init__(self, descriptor, check_cancelled):
        self.descriptor = descriptor
        self.check_cancelled = check_cancelled
        self.buffer = ''
        self.offset = 0
        self.eof = False
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.json = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_constant)
        self.sha256 = hashlib.sha256()
        self.size = 0
        self.metadata_bytes = 0

    def more(self):
        if self.eof:
            return False
        self.check_cancelled()
        raw = os.read(self.descriptor, 65536)
        self.size += len(raw)
        if self.size > MAX_FILE_BYTES:
            raise ValueError('Table artifact exceeds its 50 MB file budget.')
        self.sha256.update(raw)
        self.buffer = self.buffer[self.offset:] + self.decoder.decode(raw, final=not raw)
        self.offset = 0
        if not raw:
            self.eof = True
        return bool(raw)

    def peek(self):
        while True:
            while self.offset < len(self.buffer) and self.buffer[self.offset] in ' \t\r\n':
                self.offset += 1
            if self.offset < len(self.buffer):
                return self.buffer[self.offset]
            if not self.more():
                return ''

    def expect(self, character):
        if self.peek() != character:
            raise ValueError('Malformed table artifact JSON.')
        self.offset += 1

    def value(self, limit, metadata=False):
        self.peek()
        while True:
            try:
                value, end = self.json.raw_decode(self.buffer, self.offset)
                # A scalar at the current buffer end might continue in the next read.
                if end == len(self.buffer) and not self.eof:
                    self.more()
                    continue
                size = len(self.buffer[self.offset:end].encode('utf-8'))
                if size > limit:
                    raise ValueError('Table artifact JSON value exceeds its byte budget.')
                if metadata:
                    self.metadata_bytes += size
                    if self.metadata_bytes > MAX_METADATA_BYTES:
                        raise ValueError('Table artifact metadata exceeds its 2 MB budget.')
                self.offset = end
                return value
            except json.JSONDecodeError as error:
                if len(self.buffer[self.offset:].encode('utf-8')) > limit or self.eof:
                    raise ValueError('Malformed or oversized table artifact JSON value.') from error
                self.more()

    def object(self):
        self.expect('{')
        seen = set()
        if self.peek() == '}':
            self.offset += 1
            return
        while True:
            key = self.value(512, metadata=True)
            if type(key) is not str or key in seen:
                raise ValueError('Invalid or duplicate table artifact JSON key.')
            seen.add(key)
            self.expect(':')
            yield key
            marker = self.peek()
            if marker == '}':
                self.offset += 1
                return
            self.expect(',')

    def array(self, limit):
        self.expect('[')
        if self.peek() == ']':
            self.offset += 1
            return
        count = 0
        while True:
            if count % 256 == 0:
                self.check_cancelled()
            yield self.value(limit)
            count += 1
            if self.peek() == ']':
                self.offset += 1
                return
            self.expect(',')


def _pass(descriptor, check_cancelled, row_store=None, id_store=None):
    reader = _Reader(descriptor, check_cancelled)
    metadata = {}
    lineage = None
    prefix = None
    row_count = id_count = 0
    saw_rows = saw_ids = False
    for key in reader.object():
        if key == 'rows':
            saw_rows = True
            for row in reader.array(MAX_ROW_BYTES):
                if type(row) is not list or len(row) > 128:
                    raise ValueError('Invalid cached row bounds.')
                if row_store is not None:
                    row_store.append(row, str(row_count))
                row_count += 1
                # At least one column is mandatory, so no valid table can exceed this.
                if row_count > MAX_ROWS:
                    raise ValueError('Table artifact exceeds its cell budget.')
        elif key == 'lineage':
            lineage = {}
            for field in reader.object():
                if field == 'observationIDs':
                    saw_ids = True
                    for rid in reader.array(2048):
                        if type(rid) is not str or not 1 <= len(rid) <= 200 or len(rid.encode('utf-8')) > 200 or '\0' in rid:
                            raise ValueError('Invalid cached observation identities.')
                        if id_store is not None:
                            id_store.append((), rid)
                        prefix = rid if prefix is None else os.path.commonprefix((prefix, rid))
                        id_count += 1
                        if id_count > MAX_ROWS:
                            raise ValueError('Table artifact exceeds its cell budget.')
                else:
                    lineage[field] = reader.value(MAX_METADATA_BYTES, metadata=True)
        elif key in ('columns', 'kind', 'schemaVersion'):
            metadata[key] = reader.value(MAX_METADATA_BYTES, metadata=True)
        else:
            raise ValueError('Invalid cached table artifact fields.')
    if reader.peek():
        raise ValueError('Trailing table artifact JSON content.')
    if not saw_rows or not saw_ids or lineage is None or set(metadata) != {'columns', 'kind', 'schemaVersion'}:
        raise ValueError('Incomplete cached table artifact.')
    if row_count != id_count:
        raise ValueError('Invalid cached observation identities.')
    return metadata, lineage, prefix or '', row_count, reader.sha256.hexdigest(), reader.size


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_table_artifact(path, expected_sha256, check_cancelled):
    """Read two bounded passes on one pinned regular descriptor, preserving v1 identity.

    The first pass derives a common ID prefix without collecting IDs. The second
    uses that prefix to keep the unique identity index compact on disk. Files are
    hashed during both passes; mutation never produces an accepted mixed snapshot.
    """
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('Table artifact requires an exact SHA-256 pin.')
    if __package__:
        from . import dataflow as df
    else:
        import dataflow as df
    check_cancelled()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise ValueError('Table artifact must be a readable regular file.') from error
    row_store = id_store = None
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > MAX_FILE_BYTES:
            raise ValueError('Table artifact must be a regular file within 50 MB.')
        # New small envelopes reference separately pinned immutable segments.
        # Historical row JSON keeps the existing two-pass path unchanged.
        if initial.st_size<=MAX_METADATA_BYTES:
            raw=os.read(descriptor,MAX_METADATA_BYTES+1)
            if hashlib.sha256(raw).hexdigest()!=expected_sha256 or _identity(os.fstat(descriptor))!=_identity(initial):
                raise ValueError('Table artifact integrity check failed.')
            try:candidate=json.loads(raw,object_pairs_hook=_unique,parse_constant=_constant)
            except json.JSONDecodeError:candidate=None
            if type(candidate) is dict and candidate.get('kind')=='segmentedTable':
                if __package__:from .segmented_table_capture import read_envelope
                else:from segmented_table_capture import read_envelope
                return read_envelope(path,candidate,check_cancelled=check_cancelled)
            os.lseek(descriptor,0,os.SEEK_SET)
        first = _pass(descriptor, check_cancelled)
        if first[4] != expected_sha256 or first[5] != initial.st_size or _identity(os.fstat(descriptor)) != _identity(initial):
            raise ValueError('Table artifact integrity check failed.')
        row_store = DiskStore(check_cancelled)
        id_store = DiskStore(check_cancelled, identity_prefix=first[2])
        os.lseek(descriptor, 0, os.SEEK_SET)
        metadata, lineage, _, count, pin, size = _pass(descriptor, check_cancelled, row_store, id_store)
        if pin != expected_sha256 or size != initial.st_size or _identity(os.fstat(descriptor)) != _identity(initial):
            raise ValueError('Table artifact changed while reading.')
        rows, _ = row_store.finish()
        _, ids = id_store.finish()
        if metadata['kind'] != 'table':
            raise ValueError('Invalid cached table artifact.')
        columns = metadata['columns']
        df._fitted_module().schema_valid(columns)
        if __package__:from .image_artifacts import validate_table_version
        else:from image_artifacts import validate_table_version
        validate_table_version(metadata['schemaVersion'],columns)
        if count > df.MAX_TABLE_CELLS // len(columns):
            raise ValueError('Invalid cached row bounds.')
        if lineage.get('membershipDigest') != df.membership_digest(ids) or lineage.get('role') not in ('unassigned', 'training', 'validation', 'test'):
            raise ValueError('Invalid cached partition membership.')
        if lineage.get('dataVersion') != df.digest({'columns': columns, 'rows': rows, 'ids': ids}):
            raise ValueError('Cached table data version mismatch.')
        fitted = lineage.get('fittedStateDigests', [])
        diagnostics = lineage.get('diagnostics', [])
        if type(fitted) is not list or len(fitted) > 128 or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in fitted):
            raise ValueError('Invalid cached fitted-state chain.')
        if type(diagnostics) is not list or len(diagnostics) > 16384 or any(type(v) is not dict for v in diagnostics):
            raise ValueError('Invalid cached diagnostics.')
        for index, row in enumerate(rows):
            if index % 256 == 0:
                check_cancelled()
            if len(row) != len(columns):
                raise ValueError('Invalid cached row bounds.')
            for value, column in zip(row, columns):
                if value is None:
                    if not column['nullable']:
                        raise ValueError('Cached null violates column schema.')
                elif not df._fitted_module().typed(value, column['dtype']):
                    raise ValueError('Cached value violates column dtype.')
        if type(lineage.get('sourceDigest')) is not str or type(lineage.get('partitionDigest')) is not str:
            raise ValueError('Invalid cached source identity.')
        return df.Table(tuple(columns), rows, ids, lineage['sourceDigest'], lineage['role'], lineage['dataVersion'], lineage['partitionDigest'], tuple(fitted), tuple(diagnostics))
    except BaseException:
        if row_store is not None:
            row_store.close()
        if id_store is not None:
            id_store.close()
        raise
    finally:
        os.close(descriptor)


def read_table_artifact(path, expected_sha256, check_cancelled=None):
    check=current_check() if check_cancelled is None else check_cancelled
    # Both row and ID stores share a ceiling even for standalone reader callers.
    # Managed workers inherit the operation directory/budget/authority from main.
    with disk_scope(check_cancelled=check):
        return _read_table_artifact(path,expected_sha256,check)
