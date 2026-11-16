#!/usr/bin/env python3
"""Private DB broker child: credentials only on stdin; IPC only on stdout.

This entry is intentionally separate from normal network-denied worker dispatch.
A signed, authenticated native broker supplies exactly one scoped capture job.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
# This dedicated source service has a fixed, conservative CPU allocation.
# Set before importing any native engine, including the watchdog bootstrap.
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'

def _constrain_arrow():
    import pyarrow
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
def _arm_lifetime():
    import ctypes,os
    from worker_bootstrap import RuntimeContext,_verify_native
    runner=Path(__file__).resolve().parent
    bundled=runner/'native/lifetime_watchdog.dylib'
    if bundled.is_file():runtime=RuntimeContext(Path(__file__).resolve(),bundled,None)
    else:
        native=runner.parent/'.local/native-watchdog/lifetime_watchdog.dylib'
        runtime=RuntimeContext(Path(__file__).resolve(),native,None,native_manifest=native.with_suffix('.json'),native_source=runner/'native/lifetime_watchdog.c')
    _verify_native(runtime)
    library=ctypes.PyDLL(str(runtime.native_library));arm=library.kwip_lifetime_arm
    arm.argtypes=[ctypes.c_int];arm.restype=ctypes.c_int
    descriptor=os.dup(sys.stdin.fileno())
    try:
        if arm(descriptor)!=0:raise ValueError('Database lifetime guard refused its parent pipe.')
    finally:os.close(descriptor)
    return library


class Cancelled(Exception):pass

def main():
    deadline=None
    def cancel(*_):raise Cancelled()
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def check():
        if deadline is not None and time.monotonic()>deadline:raise Cancelled()
    try:
        header=sys.stdin.buffer.read(4)
        if len(header)!=4:raise ValueError('Missing database capture frame.')
        size=int.from_bytes(header,'big')
        if not 1<=size<=65_536:raise ValueError('Database capture message exceeds its private IPC budget.')
        raw=sys.stdin.buffer.read(size)
        if len(raw)!=size:raise ValueError('Incomplete database capture frame.')
        native_library=_arm_lifetime()
        _constrain_arrow()
        from database_source import capture,options,credentials,DatabaseSourceError
        value=json.loads(raw)
        if type(value) is not dict or set(value)!={'schemaVersion','options','credentials','preview'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['preview']) is not bool:raise DatabaseSourceError('Invalid database capture message.')
        config=options(value['options']);secret=credentials(value['credentials'])
        if config['backend']=='sqlite':raise DatabaseSourceError('Local SQLite capture uses its file-authorized adapter.')
        deadline=time.monotonic()+config['timeoutSeconds']
        receipt=capture(config,secret,sys.stdout.buffer,check_cancelled=check,preview=value['preview'])
        check();sys.stdout.buffer.flush()
        sys.stderr.write(json.dumps(receipt,separators=(',',':'))+'\n');sys.stderr.flush()
        return 0
    except BaseException as error:
        # Never relay driver text, tracebacks, private settings, or credentials.
        reason='cancelled' if isinstance(error,(Cancelled,KeyboardInterrupt)) else 'capture_failed'
        sys.stderr.write(json.dumps({'schemaVersion':1,'error':reason})+'\n');sys.stderr.flush()
        return 130 if reason=='cancelled' else 1

if __name__=='__main__':sys.exit(main())
