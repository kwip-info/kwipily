"""Exercise the actual prepared-artifact → read-only FD → broker boundary."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from runner.dataflow_cli import run_dataflow
from runner.workload_policy import WorkloadPolicy

ROOT = Path(__file__).resolve().parents[2]


def prepared(tmp_path, destination, options, retain):
    source = tmp_path / 'source.csv'
    source.write_text('id,amount\n1,1.25\n2,2.50\n3,3.75\n')
    cast = dict(rows=[dict(source=n, outputName=n, included=True, targetType=t)
                      for n, t in [('id', 'int64'), ('amount', 'float64')]],
                newColumns='drop', castFailure='error', missingPolicy='preserve', trimText=False)
    graph = dict(schemaVersion=1, nodes=[
        dict(id='source', operator='kwip.data.sourceCSV', config={}),
        dict(id='typed', operator='kwip.data.columns', config=cast),
        dict(id='save', operator='kwip.data.outputDelimited', config=dict(
            destination=destination, **{destination: options}, filename='unused.csv',
            columns=[], separator=',', overwrite=False))], edges=[
        dict(**{'from': dict(node=a, port='table'), 'to': dict(node=b, port='table')})
        for a, b in [('source', 'typed'), ('typed', 'save')]])
    path = tmp_path / 'graph.json'
    path.write_text(json.dumps(graph))
    archive = tmp_path / 'archive'
    manifest = run_dataflow(path, {'source': str(source)}, archive, lambda _: None,
                           engine='polars', workload_policy=WorkloadPolicy('balanced').to_document(),
                           retain_outputs=retain, defer_database_writes=True,
                           sampling=dict(mode='rowCount', rowCount=1))
    tables = [entry for entry in manifest['artifacts'] if entry['kind'] == 'table']
    if retain:
        # Compatibility is local to the external boundary, not all intermediates.
        upstream = next(entry for entry in tables if entry['nodeID'] == 'typed')
        assert json.loads((archive / upstream['path']).read_bytes())['kind'] == 'segmentedTable'
    entry = next(entry for entry in tables if entry['nodeID'] == 'save')
    assert not manifest.get('outputs')  # Preparation never performs the write.
    return archive / entry['path'], entry['sha256']


def broker(entry, table, digest, options, sqlite_path=None):
    operation = str(uuid.uuid4()).upper()
    raw = json.dumps(dict(schemaVersion=1, action='write', options=options,
                         credentials={}, operationID=operation, tableSHA256=digest,
                         sqlitePath=sqlite_path)).encode()
    with table.open('rb') as read:
        child = subprocess.Popen([sys.executable, '-I', '-B', str(ROOT / 'runner' / entry)],
                                 stdin=subprocess.PIPE, stdout=read, stderr=subprocess.PIPE)
        try:
            child.stdin.write(len(raw).to_bytes(4, 'big') + raw)
            child.stdin.flush()  # Keep lifetime authority open until the child exits.
            child.wait(timeout=20)
            receipt = json.loads(child.stderr.read(65_537))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdin.close()
            child.stderr.close()
    return receipt, operation


@pytest.mark.parametrize('retain', [False, True])
@pytest.mark.parametrize('tamper', [False, True])
def test_captured_workload_reaches_actual_sqlite_broker(tmp_path, retain, tamper):
    options = dict(backend='sqlite', table='orders', mode='create', keys=['id'], mapping=[
        dict(source=n, target=n, type=t, nullable=False)
        for n, t in [('id', 'INTEGER'), ('amount', 'REAL')]])
    table, digest = prepared(tmp_path, 'database', options, retain)
    target = tmp_path / 'output.sqlite'
    sqlite3.connect(target).close()
    if tamper:
        table.write_bytes(table.read_bytes() + b' ')
    before = hashlib.sha256(table.read_bytes()).hexdigest()
    receipt, operation = broker('database_output_service.py', table, digest, options, str(target))
    with sqlite3.connect(target) as con:
        if tamper:
            assert receipt['status'] != 'committed'
            assert con.execute("SELECT count(*) FROM sqlite_master WHERE name='orders'").fetchone()[0] == 0
        else:
            assert receipt['status'] == 'committed' and receipt['operationID'] == operation
            assert receipt['sourceSHA256'] == digest
            assert con.execute('SELECT * FROM orders ORDER BY id').fetchall() == [(1, 1.25), (2, 2.5), (3, 3.75)]
    assert hashlib.sha256(table.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('retain', [False, True])
@pytest.mark.parametrize('tamper', [False, True])
def test_captured_workload_reaches_actual_api_broker(tmp_path, retain, tamper):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(202)
            self.end_headers()
            self.wfile.write(b'{"accepted":true}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        options = dict(direction='output', method='POST', allowLoopback=True,
                       url=f'http://127.0.0.1:{server.server_port}/predictions',
                       delivery='batch', batchRows=2, arrayPath='/predictions', mapping=[
                           dict(source=n, target='/' + n, type=t)
                           for n, t in [('id', 'int64'), ('amount', 'float64')]])
        table, digest = prepared(tmp_path, 'api', options, retain)
        assert calls == []
        if tamper:
            table.write_bytes(table.read_bytes() + b' ')
        before = hashlib.sha256(table.read_bytes()).hexdigest()
        receipt, operation = broker('api_table_service.py', table, digest, options)
        if tamper:
            assert receipt['status'] != 'committed' and calls == []
        else:
            assert receipt['status'] == 'committed' and receipt['operationID'] == operation
            assert receipt['sourceSHA256'] == digest and receipt['acceptedRows'] == 3
            assert calls == [dict(predictions=[dict(id=1, amount=1.25), dict(id=2, amount=2.5)]),
                             dict(predictions=[dict(id=3, amount=3.75)])]
        assert hashlib.sha256(table.read_bytes()).hexdigest() == before
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
