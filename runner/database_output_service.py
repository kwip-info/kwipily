#!/usr/bin/env python3
"""Database broker child; framed private control on stdin, captured table on FD1.

FD1 is deliberately a read-only regular file, not a text stream. Receipt only on
stderr. Ordinary workers never dispatch this entry and cannot gain network rights.
"""
import hashlib,json,os,signal,stat,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[_key]='1'

def main():
    from database_source_capture import _arm_lifetime,Cancelled
    from database_output import inspect,write,options,OutputError
    def cancelled(*_):raise Cancelled()
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
    receipt=dict(schemaVersion=1,kind='databaseWrite',status='unknown',message='Database broker interrupted. Inspect the destination before retrying.')
    try:
        header=sys.stdin.buffer.read(4)
        if len(header)!=4:raise OutputError('Missing database operation frame.')
        length=int.from_bytes(header,'big')
        if not 1<=length<=65_536:raise OutputError('Database operation frame exceeds its bound.')
        raw=sys.stdin.buffer.read(length)
        if len(raw)!=length:raise OutputError('Incomplete database operation frame.')
        value=json.loads(raw);_guard=_arm_lifetime()
        if set(value)!={'schemaVersion','action','options','credentials','operationID','tableSHA256','sqlitePath'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or value['action'] not in ('inspect','write'):raise OutputError('Invalid database operation.')
        c=options(value['options'],require_mapping=value['action']=='write');deadline=time.monotonic()+c['timeoutSeconds']
        def check():
            if time.monotonic()>deadline:raise Cancelled()
        if value['action']=='inspect':receipt=inspect(c,value['credentials'],sqlite_path=value['sqlitePath'])
        else:
            fd=os.dup(1)
            with os.fdopen(fd,'rb',closefd=True) as stream:
                info=os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=50_000_000:raise OutputError('Captured input must be a bounded regular table.')
                digest=hashlib.sha256();blocks=[];count=0
                while True:
                    check();block=stream.read(min(1_048_576,50_000_001-count))
                    if not block:break
                    blocks.append(block);count+=len(block);digest.update(block)
                    if count>50_000_000:raise OutputError('Captured input exceeds 50 MB.')
                if count!=info.st_size or digest.hexdigest()!=value['tableSHA256']:raise OutputError('Captured input changed before database dispatch.')
                table=json.loads(b''.join(blocks));del blocks
            if type(table) is not dict or table.get('kind')!='table' or type(table.get('columns')) is not list or type(table.get('rows')) is not list:raise OutputError('Use a verified complete captured table.')
            receipt=write(c,value['credentials'],table['columns'],table['rows'],operation_id=value['operationID'],sqlite_path=value['sqlitePath'],check_cancelled=check)
            receipt['sourceSHA256']=value['tableSHA256']
        encoded=json.dumps(receipt,allow_nan=False,separators=(',',':'))
        if len(encoded.encode())>65_536:raise OutputError('Database receipt exceeds its bound.')
    except BaseException:
        encoded=json.dumps(receipt,separators=(',',':'))
    sys.stderr.write(encoded+'\n');sys.stderr.flush()
    return 0
if __name__=='__main__':sys.exit(main())
