"""Offline API snapshot validation. No network entry is reachable from this module."""
import hashlib,json
from pathlib import Path
if __package__:
    from .api_tables import options,APIError
else:
    from api_tables import options,APIError

def identity(c):return hashlib.sha256(json.dumps(options(c),sort_keys=True,separators=(',',':')).encode()).hexdigest().encode()
def verified(path,c):
    import pyarrow.ipc as ipc
    with Path(path).open('rb') as stream:
        reader=ipc.open_file(stream)
        if (reader.schema.metadata or {}).get(b'kwip.api.selection.v1')!=identity(c):raise APIError('API settings changed. Fetch a new snapshot before using this source.')

def source_table(path,configuration,check_cancelled=lambda:None,sampling=None):
    if __package__:from .table_sources import source_typed_snapshot
    else:from table_sources import source_typed_snapshot
    verified(path,configuration['options']);return source_typed_snapshot(path,check_cancelled=check_cancelled,sampling=sampling,allow_temporal=True)
def preview_evidence(path,configuration,check_cancelled=lambda:None):
    table=source_table(path,configuration,check_cancelled,sampling={'schemaVersion':1,'mode':'rowCount','rowCount':8})
    import pyarrow.ipc as ipc
    with Path(path).open('rb') as stream:metadata=ipc.open_file(stream).schema.metadata or {}
    fetch=json.loads(metadata.get(b'kwip.api.fetch.v1',b'null'))
    try:return {'apiFetch':fetch,'columns':list(table.columns),'rows':[list(row) for row in table.rows[:8]],'sourceSHA256':table.source_digest,'sampleOnly':True}
    finally:table.rows.store.close()
