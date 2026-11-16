#!/usr/bin/env python3
"""Fixed authenticated broker entry. Private frame; input IPC or output receipt only."""
import hashlib,json,os,signal,stat,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='1'

def main():
    receipt=dict(schemaVersion=1,kind='apiWrite',status='unknown',message='API operation interrupted; inspect delivery before retrying.')
    def stop(*args):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        length=int.from_bytes(sys.stdin.buffer.read(4),'big')
        if not 1<=length<=65536:raise ValueError('Invalid API frame')
        raw=sys.stdin.buffer.read(length)
        if len(raw)!=length:raise ValueError('Incomplete API frame')
        value=json.loads(raw)
        if type(value) is not dict or type(value.get('schemaVersion')) is not int or value['schemaVersion']!=1:raise ValueError('Unsupported API frame')
        credentials=value.get('credentials')
        if type(credentials) is not dict or set(credentials)-{'username','password'} or any(type(v) is not str or len(v.encode())>8192 or any(ch in v for ch in '\r\n\0') for v in credentials.values()):raise ValueError('Invalid private credentials')
        from database_source_capture import _arm_lifetime
        guard=_arm_lifetime()
        import threading,time,resource
        memory=value.pop('memoryLimit',None);seconds=value.pop('operationSeconds',300)
        if memory is not None and (type(memory) is not int or not 1_000_000<=memory<=2**63-1):raise ValueError('Invalid memory budget')
        if type(seconds) is not int or not 1<=seconds<=86400:raise ValueError('Invalid operation budget')
        started=time.monotonic()
        def enforce():
            while True:
                peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                if memory is not None and peak>memory or time.monotonic()-started>seconds:os._exit(137)
                time.sleep(.05)
        threading.Thread(target=enforce,daemon=True).start()
        from api_tables import options,fetch,deliver
        c=options(value['options']);secret=value['credentials']
        if c['direction']=='input':
            if set(value)!={'schemaVersion','options','credentials','preview'} or value['preview'] is not False:raise ValueError('Capture a complete source')
            import pyarrow as pa,pyarrow.ipc as ipc
            pa.set_cpu_count(1);pa.set_io_thread_count(1)
            from api_table_snapshot import identity
            columns,rows,fetch_receipt=fetch(c,secret)
            types=dict(string=pa.string(),int64=pa.int64(),float64=pa.float64(),bool=pa.bool_(),datetime=pa.timestamp('us',tz='UTC'))
            fields=[pa.field(col['name'],types[col['dtype']],nullable=True) for col in columns]
            schema=pa.schema(fields,metadata={b'kwip.api.selection.v1':identity(c),b'kwip.api.fetch.v1':json.dumps(fetch_receipt,separators=(',',':')).encode()})
            import datetime
            arrays=[pa.array([datetime.datetime.fromisoformat(r[i]) if columns[i]['dtype']=='datetime' and r[i] is not None else r[i] for r in rows],type=fields[i].type) for i in range(len(fields))]
            sink=pa.BufferOutputStream()
            with ipc.new_file(sink,schema) as writer:writer.write_table(pa.Table.from_arrays(arrays,schema=schema))
            data=sink.getvalue().to_pybytes()
            if len(data)>10_000_000:raise ValueError('API IPC snapshot exceeds budget')
            sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
            receipt=dict(schemaVersion=1,kind='databaseCapture',format='ipc',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),rows=len(rows),columns=len(columns),sampleOnly=False,apiFetch=fetch_receipt)
        else:
            if set(value)!={'schemaVersion','action','options','operationID','credentials','tableSHA256','sqlitePath'}:raise ValueError('Unsupported API delivery frame')
            import uuid
            uuid.UUID(value['operationID'])
            if type(value['tableSHA256']) is not str or len(value['tableSHA256'])!=64 or any(ch not in '0123456789abcdef' for ch in value['tableSHA256']):raise ValueError('Invalid table identity')
            if value['sqlitePath'] is not None:raise ValueError('API delivery does not accept filesystem destinations')
            receipt['operationID']=value['operationID']
            if value['action']!='write':raise ValueError('API output requires explicit run')
            with os.fdopen(os.dup(1),'rb') as stream:
                info=os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=50_000_000:raise ValueError('Invalid input')
                data=stream.read(50_000_001)
            if len(data)!=info.st_size or hashlib.sha256(data).hexdigest()!=value['tableSHA256']:raise ValueError('Input changed')
            table=json.loads(data)
            if table.get('kind')!='table':raise ValueError('Use captured table')
            receipt=deliver(c,secret,table['columns'],table['rows'],value['operationID']);receipt['sourceSHA256']=value['tableSHA256']
        encoded=json.dumps(receipt,allow_nan=False,separators=(',',':'))
        if len(encoded.encode())>65536:raise ValueError('Receipt exceeds budget')
        sys.stderr.write(encoded+'\n');return 0
    except BaseException:
        sys.stderr.write(json.dumps(receipt,separators=(',',':'))+'\n');return 1 if 'preview' in locals().get('value',{}) else 0
if __name__=='__main__':sys.exit(main())
