"""Source-independent bounded interchange: explicit schema, values and observation IDs.

Sources own parsing/access and scientific lineage. Engines consume this boundary;
no source format, credential or permission is encoded in a batch.
"""
from dataclasses import dataclass
import json

MAX_BATCH_ROWS=4096
MAX_BATCH_BYTES=2_097_152


@dataclass(frozen=True)
class TypedBatch:
    columns: tuple
    rows: tuple
    observation_ids: tuple
    encoded_bytes: int


def table_batches(table, *, max_rows=MAX_BATCH_ROWS, max_bytes=MAX_BATCH_BYTES, check_cancelled=lambda:None):
    if type(max_rows) is not int or not 1<=max_rows<=MAX_BATCH_ROWS:
        raise ValueError('Invalid typed batch row budget')
    if type(max_bytes) is not int or not 1<=max_bytes<=MAX_BATCH_BYTES:
        raise ValueError('Invalid typed batch byte budget')
    if len(table.rows)!=len(table.row_ids):raise ValueError('Typed table row/identity count differs')
    rows=[];ids=[];size=0
    pairs=table.rows.iter_pairs() if hasattr(table.rows,'iter_pairs') and table.rows.store is getattr(table.row_ids,'store',None) else zip(table.rows,table.row_ids)
    for index,(row,rid) in enumerate(pairs):
        if index%256==0:check_cancelled()
        if len(row)!=len(table.columns):raise ValueError('Typed table row width differs from schema')
        encoded=len(json.dumps([row,rid],ensure_ascii=True,allow_nan=False,separators=(',',':')).encode('utf-8'))
        if encoded>max_bytes:raise ValueError('A typed record exceeds the batch byte budget')
        if rows and (len(rows)>=max_rows or size+encoded>max_bytes):
            check_cancelled()
            yield TypedBatch(tuple(table.columns),tuple(rows),tuple(ids),size)
            rows=[];ids=[];size=0
        rows.append(tuple(row));ids.append(rid);size+=encoded
    if rows:
        check_cancelled()
        yield TypedBatch(tuple(table.columns),tuple(rows),tuple(ids),size)
    check_cancelled()
