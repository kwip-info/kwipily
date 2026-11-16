"""Independent option-effect fixtures; all databases are disposable and local.

See fixtures/table_options/database/COVERAGE.md for applicability and trust limits.
No native app, Keychain, production account, global trust or service is changed.
"""
import io
import json
import sqlite3
import time
from pathlib import Path
import pytest
from runner import database_source as db
from runner.tests.test_database_source import postgres, mysql

FIXTURES=Path(__file__).parent/'fixtures/table_options/database'
CASES=json.loads((FIXTURES/'selection.json').read_text())

# Consumed by the native-authoritative exposed-option inventory guard.
OPTION_CASES={'database.mysql.backend': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.binaryMode': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.columns': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.database': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.dateTimeMode': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.decimalMode': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.fetchBatchRows': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.host': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.orderBy': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.port': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.rowLimit': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.rowOffset': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.schema': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.table': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.timeoutSeconds': ['test_mysql_query_batching_type_and_timeout_options'],
 'database.mysql.tlsMode': ['test_mysql_tls_verified_success_and_certificate_hostname_refusal',
                            'test_insecure_transport_requires_literal_loopback_and_bounded_options'],
 'database.postgresql.backend': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.binaryMode': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.columns': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.database': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.dateTimeMode': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.decimalMode': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.fetchBatchRows': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.host': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.orderBy': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.port': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.rowLimit': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.rowOffset': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.schema': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.table': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.timeoutSeconds': ['test_postgres_query_batching_type_and_timeout_options'],
 'database.postgresql.tlsMode': ['test_postgres_tls_verified_success_and_certificate_hostname_refusal',
                                 'test_insecure_transport_requires_literal_loopback_and_bounded_options'],
 'database.sqlite.backend': ['test_sqlite_every_selected_table_query_option'],
 'database.sqlite.binaryMode': ['test_sqlite_binary_policies_and_native_text_date_numeric_semantics'],
 'database.sqlite.columns': ['test_sqlite_every_selected_table_query_option'],
 'database.sqlite.dateTimeMode': ['test_sqlite_binary_policies_and_native_text_date_numeric_semantics'],
 'database.sqlite.decimalMode': ['test_sqlite_binary_policies_and_native_text_date_numeric_semantics'],
 'database.sqlite.fetchBatchRows': ['test_sqlite_batching_is_real_and_lossless'],
 'database.sqlite.orderBy': ['test_sqlite_every_selected_table_query_option'],
 'database.sqlite.rowLimit': ['test_sqlite_every_selected_table_query_option'],
 'database.sqlite.rowOffset': ['test_sqlite_every_selected_table_query_option'],
 'database.sqlite.schema': ['test_sqlite_every_selected_table_query_option'],
 'database.sqlite.table': ['test_sqlite_every_selected_table_query_option'],
 'database.sqlite.timeoutSeconds': ['test_sqlite_timeout_limits_busy_lock_wait']}


def capture(config,secret=None,path=None):
    import pyarrow.ipc as ipc
    sink=io.BytesIO()
    receipt=db.capture(config,secret or {},sink,sqlite_path=path)
    reader=ipc.open_file(sink.getvalue());table=reader.read_all()
    return table.column_names,[list(row) for row in zip(*(column.to_pylist() for column in table.columns))],receipt,reader.num_record_batches


@pytest.fixture
def sqlite_options(tmp_path):
    path=tmp_path/'options.sqlite'
    with sqlite3.connect(path) as connection:connection.executescript((FIXTURES/'selection.sql').read_text())
    return path


def assert_cases(config,secret=None,path=None):
    for case in CASES:
        names,rows,_,_=capture({**config,**case['options']},secret,path)
        assert names==case['columns'],case['name']
        assert rows==case['rows'],case['name']


def test_sqlite_every_selected_table_query_option(sqlite_options):
    assert_cases({'backend':'sqlite','table':'samples'},path=sqlite_options)
    for schema in ('main',''):
        assert capture({'table':'samples','schema':schema,'columns':['id'],'orderBy':['+id']},path=sqlite_options)[1]==[[1],[2],[3],[4]]
    with pytest.raises(db.DatabaseSourceError):capture({'table':'samples','schema':'other'},path=sqlite_options)
    for change in ({'table':'missing'},{'columns':['absent']},{'orderBy':['+absent']}):
        with pytest.raises(db.DatabaseSourceError):capture({'table':'samples',**change},path=sqlite_options)


def test_sqlite_batching_is_real_and_lossless(sqlite_options):
    config={'table':'samples','columns':['id'],'orderBy':['+id']}
    one=capture({**config,'fetchBatchRows':1},path=sqlite_options)
    large=capture({**config,'fetchBatchRows':8192},path=sqlite_options)
    assert one[1]==large[1]==[[1],[2],[3],[4]]
    assert one[3]==4 and large[3]==1


def test_sqlite_binary_policies_and_native_text_date_numeric_semantics(sqlite_options):
    config={'table':'binary_values','columns':['payload']}
    assert capture({**config,'binaryMode':'hex'},path=sqlite_options)[1]==[['00ff41']]
    assert capture({**config,'binaryMode':'base64'},path=sqlite_options)[1]==[['AP9B']]
    with pytest.raises(db.DatabaseSourceError,match='Binary'):capture(config,path=sqlite_options)
    # sqlite3 exposes DATE as text and NUMERIC as a native float. Policies do
    # not invent Decimal/date types or parse those values a second time.
    for decimal in ('text','error'):
        for date in ('iso8601','error'):
            assert capture({'table':'text_dates','decimalMode':decimal,'dateTimeMode':date},path=sqlite_options)[1]==[[1,'2026-10-13',1.25]]


def test_sqlite_timeout_limits_busy_lock_wait(sqlite_options):
    connection=sqlite3.connect(sqlite_options);connection.execute('BEGIN EXCLUSIVE')
    try:
        started=time.monotonic()
        with pytest.raises(db.DatabaseSourceError):capture({'table':'samples','timeoutSeconds':1},path=sqlite_options)
        assert .7<=time.monotonic()-started<4
    finally:connection.rollback();connection.close()



def test_postgres_query_batching_type_and_timeout_options(postgres):
    config,secret=postgres
    import psycopg
    connection=psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username'])
    try:
        with connection.cursor() as cursor:
            cursor.execute('CREATE TABLE samples (id BIGINT, cohort TEXT, name TEXT, value DOUBLE PRECISION)')
            cursor.execute("INSERT INTO samples VALUES (4,'B','delta',4.5),(2,'A','beta',2.5),(1,'A','alpha',1.5),(3,'B','gamma',NULL)")
            cursor.execute('CREATE TABLE alternative (LIKE samples)');cursor.execute("INSERT INTO alternative VALUES (99,'Z','other-table',99.5)")
            cursor.execute("CREATE TABLE binary_values AS SELECT decode('00ff41','hex') AS payload")
            cursor.execute('CREATE SCHEMA selected_schema');cursor.execute('CREATE TABLE selected_schema.samples (id BIGINT)');cursor.execute('INSERT INTO selected_schema.samples VALUES (88)')
            cursor.execute('CREATE VIEW slowly AS SELECT id, pg_sleep(0.4)::text AS delay FROM samples')
            cursor.execute("CREATE TABLE temporal AS SELECT TIMESTAMP '2026-10-13 08:09:10' AS moment, TIME '08:09:10' AS clock")
            cursor.execute("CREATE TABLE calendar_interval AS SELECT INTERVAL '1 month' AS duration")
        connection.commit()
    finally:connection.close()
    config={**config,'table':'samples'};assert_cases(config,secret)
    with pytest.raises(db.DatabaseSourceError,match='unsupported database value type'):
        capture({**config,'table':'calendar_interval','dateTimeMode':'iso8601'},secret)
    assert capture({**config,'schema':'selected_schema'},secret)[1]==[[88]]
    for schema in ('public','main',''):
        assert capture({**config,'schema':schema,'columns':['id'],'orderBy':['+id']},secret)[1]==[[1],[2],[3],[4]]
    for batch,expected in ((1,4),(8192,1)):
        result=capture({**config,'fetchBatchRows':batch,'columns':['id'],'orderBy':['+id']},secret)
        assert result[1]==[[1],[2],[3],[4]] and result[3]==expected
    assert capture({**config,'table':'temporal','dateTimeMode':'iso8601'},secret)[1]==[['2026-10-13T08:09:10','08:09:10']]
    with pytest.raises(db.DatabaseSourceError):capture({**config,'table':'temporal','dateTimeMode':'error'},secret)
    typed={**config,'table':'chosen','columns':['amount','moment'],'orderBy':['+id']}
    assert capture(typed,secret)[1]==[['1.25','2026-10-13'],['2.50','2026-10-14']]
    for option in ('decimalMode','dateTimeMode'):
        with pytest.raises(db.DatabaseSourceError):capture({**typed,option:'error'},secret)
    for policy,expected in (('hex','00ff41'),('base64','AP9B')):
        assert capture({**config,'table':'binary_values','binaryMode':policy},secret)[1]==[[expected]]
    with pytest.raises(db.DatabaseSourceError):capture({**config,'table':'binary_values'},secret)
    for change in ({'database':'missing_database'},{'schema':'missing_schema'},{'table':'missing_table'},{'columns':['absent']},{'orderBy':['-absent']},{'port':1},{'host':'127.0.0.2'}):
        with pytest.raises(db.DatabaseSourceError):capture({**config,**change,'timeoutSeconds':1},secret)
    started=time.monotonic()
    with pytest.raises(db.DatabaseSourceError):capture({**config,'table':'slowly','timeoutSeconds':1},secret)
    assert .7<=time.monotonic()-started<5


def test_mysql_query_batching_type_and_timeout_options(mysql):
    import pymysql
    config,secret=mysql
    with pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=secret['username'],password='',ssl_disabled=True) as selected:
        with selected.cursor() as cursor:cursor.execute('SELECT @@socket');socket=cursor.fetchone()[0]
    with pymysql.connect(unix_socket=socket,user='root',password='',database=config['database'],ssl_disabled=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute('CREATE TABLE samples (id BIGINT, cohort VARCHAR(10), name VARCHAR(40), value DOUBLE)')
            cursor.execute("INSERT INTO samples VALUES (4,'B','delta',4.5),(2,'A','beta',2.5),(1,'A','alpha',1.5),(3,'B','gamma',NULL)")
            cursor.execute('CREATE TABLE alternative LIKE samples');cursor.execute("INSERT INTO alternative VALUES (99,'Z','other-table',99.5)")
            cursor.execute('CREATE TABLE binary_values (payload BLOB)');cursor.execute("INSERT INTO binary_values VALUES (X'00FF41')")
            cursor.execute('CREATE VIEW slowly AS SELECT id, SLEEP(0.4) AS delay FROM samples')
            cursor.execute('CREATE TABLE temporal (moment DATETIME, clock TIME)')
            cursor.execute("INSERT INTO temporal VALUES ('2026-10-13 08:09:10','08:09:10')")
            cursor.execute("GRANT SELECT ON learning.* TO 'kwip_test'@'127.0.0.1'")
        connection.commit()
    config={**config,'table':'samples'};assert_cases(config,secret)
    for schema in ('learning','main',''):
        assert capture({**config,'schema':schema,'columns':['id'],'orderBy':['+id']},secret)[1]==[[1],[2],[3],[4]]
    for batch,expected in ((1,4),(8192,1)):
        result=capture({**config,'fetchBatchRows':batch,'columns':['id'],'orderBy':['+id']},secret)
        assert result[1]==[[1],[2],[3],[4]] and result[3]==expected
    assert capture({**config,'table':'temporal','columns':['moment'],'dateTimeMode':'iso8601'},secret)[1]==[['2026-10-13T08:09:10']]
    with pytest.raises(db.DatabaseSourceError):capture({**config,'table':'temporal','columns':['moment'],'dateTimeMode':'error'},secret)
    assert capture({**config,'table':'temporal','columns':['clock'],'dateTimeMode':'iso8601'},secret)[1]==[['PT8H9M10S']]
    typed={**config,'table':'chosen','columns':['amount','moment'],'orderBy':['+id']}
    assert capture(typed,secret)[1]==[['1.25','2026-10-13'],['2.50','2026-10-14']]
    for option in ('decimalMode','dateTimeMode'):
        with pytest.raises(db.DatabaseSourceError):capture({**typed,option:'error'},secret)
    for policy,expected in (('hex','00ff41'),('base64','AP9B')):
        assert capture({**config,'table':'binary_values','binaryMode':policy},secret)[1]==[[expected]]
    with pytest.raises(db.DatabaseSourceError):capture({**config,'table':'binary_values'},secret)
    for change in ({'database':'missing_database','schema':'main'},{'schema':'other_schema'},{'table':'missing_table'},{'columns':['absent']},{'orderBy':['-absent']},{'port':1},{'host':'127.0.0.2'}):
        with pytest.raises(db.DatabaseSourceError):capture({**config,**change,'timeoutSeconds':1},secret)
    started=time.monotonic()
    with pytest.raises(db.DatabaseSourceError):capture({**config,'table':'slowly','timeoutSeconds':1},secret)
    assert .7<=time.monotonic()-started<5


def certificates(folder,*,wrong_name=False):
    import subprocess
    folder.mkdir(exist_ok=True)
    openssl=Path('/opt/homebrew/opt/openssl@3/bin/openssl')
    if not openssl.is_file():openssl=Path('/usr/bin/openssl')
    ca_path=folder/'ca.pem';ca_key=folder/'ca.key';cert_path=folder/'server.crt';key_path=folder/'server.key';request=folder/'server.csr';extensions=folder/'extensions.cnf'
    def run(arguments):subprocess.run([str(openssl),*map(str,arguments)],check=True,capture_output=True,timeout=15)
    run(['req','-x509','-newkey','rsa:2048','-nodes','-days','1','-sha256','-subj','/CN=KWIP disposable fixture CA','-addext','basicConstraints=critical,CA:TRUE','-addext','keyUsage=critical,keyCertSign,cRLSign','-out',ca_path,'-keyout',ca_key])
    run(['req','-new','-newkey','rsa:2048','-nodes','-subj','/CN=fixture.invalid' if wrong_name else '/CN=localhost','-out',request,'-keyout',key_path])
    extensions.write_text('basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nauthorityKeyIdentifier=keyid,issuer\nsubjectKeyIdentifier=hash\nextendedKeyUsage=serverAuth\nsubjectAltName='+('DNS:fixture.invalid' if wrong_name else 'DNS:localhost,IP:127.0.0.1')+'\n')
    run(['x509','-req','-in',request,'-CA',ca_path,'-CAkey',ca_key,'-CAcreateserial','-out',cert_path,'-days','1','-sha256','-extfile',extensions])
    key_path.chmod(0o600);ca_key.chmod(0o600)
    return ca_path,cert_path,key_path


def test_postgres_tls_verified_success_and_certificate_hostname_refusal(postgres,tmp_path,monkeypatch):
    import psycopg
    config,secret=postgres
    ca,cert,key=certificates(tmp_path/'tls')
    with psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username']) as connection:
        datadir=connection.execute('SHOW data_directory').fetchone()[0]
    with (Path(datadir)/'postgresql.conf').open('a') as stream:
        stream.write("\nssl=on\nssl_cert_file='"+str(cert)+"'\nssl_key_file='"+str(key)+"'\n")
    def reload_server():
        with psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username'],sslmode='disable') as connection:
            connection.execute('SELECT pg_reload_conf()')
        time.sleep(.2)
    reload_server()
    secure={**config,'tlsMode':'verifyFull','columns':['id'],'orderBy':['+id']}
    monkeypatch.setenv('SSL_CERT_FILE',str(ca))
    assert capture(secure,secret)[1]==[[1],[2]]
    # A different private CA is trusted only in this pytest process, never OS trust.
    other_ca,_,_=certificates(tmp_path/'other-ca')
    monkeypatch.setenv('SSL_CERT_FILE',str(other_ca))
    with pytest.raises(db.DatabaseSourceError):capture(secure,secret)
    wrong_ca,_,_=certificates(tmp_path/'tls',wrong_name=True)
    monkeypatch.setenv('SSL_CERT_FILE',str(wrong_ca));reload_server()
    # Prove the new chain is valid before isolating hostname refusal.
    with psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username'],sslmode='verify-ca',sslrootcert=str(wrong_ca)) as chain_verified:
        assert chain_verified.execute('SELECT 1').fetchone()==(1,)
    with pytest.raises(db.DatabaseSourceError):capture(secure,secret)
    # Explicit loopback-only plaintext remains independent of certificate state.
    assert capture({**secure,'tlsMode':'disabledLoopback'},secret)[1]==[[1],[2]]


@pytest.mark.parametrize('backend',['postgresql','mysql'])
def test_insecure_transport_requires_literal_loopback_and_bounded_options(backend):
    config={'backend':backend,'host':'127.0.0.1','database':'learning','table':'samples','tlsMode':'disabledLoopback'}
    assert db.options(config)['tlsMode']=='disabledLoopback'
    for host in ('localhost','example.com','192.0.2.1','127.0.0.1/uri'):
        with pytest.raises(db.DatabaseSourceError):db.options({**config,'host':host})
    for key,invalid in (('port',0),('port',65536),('rowLimit',-1),('rowLimit',5_000_001),('rowOffset',-1),('fetchBatchRows',0),('fetchBatchRows',8193),('timeoutSeconds',0),('timeoutSeconds',301)):
        with pytest.raises(db.DatabaseSourceError):db.options({**config,key:invalid})


def test_mysql_tls_verified_success_and_certificate_hostname_refusal(mysql,tmp_path,monkeypatch):
    import pymysql,shutil
    config,secret=mysql
    with pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=secret['username'],password='',ssl_disabled=True) as selected:
        with selected.cursor() as cursor:cursor.execute('SELECT @@socket,@@datadir');socket,datadir=cursor.fetchone()
    ca,cert,key=certificates(tmp_path/'tls-mysql')
    def install(ca,cert,key):
        for source,name in ((ca,'ca.pem'),(cert,'server-cert.pem'),(key,'server-key.pem')):
            target=Path(datadir)/name;shutil.copyfile(source,target);target.chmod(0o600)
        with pymysql.connect(unix_socket=socket,user='root',password='',ssl_disabled=True) as connection:
            with connection.cursor() as cursor:cursor.execute('ALTER INSTANCE RELOAD TLS')
    install(ca,cert,key)
    import ssl
    monkeypatch.setenv('SSL_CERT_FILE',str(ca))
    with pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=secret['username'],password='',ssl=ssl.create_default_context()) as verified:
        with verified.cursor() as cursor:cursor.execute('SELECT 1');assert cursor.fetchone()==(1,)
    secure={**config,'tlsMode':'verifyFull','columns':['id'],'orderBy':['+id']}
    monkeypatch.setenv('SSL_CERT_FILE',str(ca))
    assert capture(secure,secret)[1]==[[1],[2]]
    other_ca,_,_=certificates(tmp_path/'other-mysql-ca')
    monkeypatch.setenv('SSL_CERT_FILE',str(other_ca))
    with pytest.raises(db.DatabaseSourceError):capture(secure,secret)
    ca,cert,key=certificates(tmp_path/'tls-mysql',wrong_name=True);install(ca,cert,key)
    monkeypatch.setenv('SSL_CERT_FILE',str(ca))
    chain_only=ssl.create_default_context();chain_only.check_hostname=False
    with pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=secret['username'],password='',ssl=chain_only) as chain_verified:
        with chain_verified.cursor() as cursor:cursor.execute('SELECT 1');assert cursor.fetchone()==(1,)
    with pytest.raises(db.DatabaseSourceError):capture(secure,secret)
    assert capture({**secure,'tlsMode':'disabledLoopback'},secret)[1]==[[1],[2]]


def test_mysql_time_duration_policy_preserves_sign_days_zero_and_microseconds(mysql):
    import pymysql
    config,secret=mysql
    with pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=secret['username'],password='',ssl_disabled=True) as selected:
        with selected.cursor() as cursor:cursor.execute('SELECT @@socket');socket=cursor.fetchone()[0]
    with pymysql.connect(unix_socket=socket,user='root',password='',database=config['database'],ssl_disabled=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute('CREATE TABLE duration_values (id INTEGER, duration TIME(6))')
            cursor.execute("INSERT INTO duration_values VALUES (1,'32:09:10.123456'),(2,'-32:09:10.123456'),(3,'00:00:00'),(4,'00:00:00.000001')")
            cursor.execute("GRANT SELECT ON learning.duration_values TO 'kwip_test'@'127.0.0.1'")
        connection.commit()
    selected={**config,'table':'duration_values','columns':['duration'],'orderBy':['+id'],'dateTimeMode':'iso8601'}
    assert capture(selected,secret)[1]==[['P1DT8H9M10.123456S'],['-P1DT8H9M10.123456S'],['PT0S'],['PT0.000001S']]
    with pytest.raises(db.DatabaseSourceError):capture({**selected,'dateTimeMode':'error'},secret)

OPTION_CASES['database.mysql.dateTimeMode'].append('test_mysql_time_duration_policy_preserves_sign_days_zero_and_microseconds')


def test_mysql_first_fresh_caching_sha2_password_capture_over_literal_loopback(mysql,monkeypatch):
    """No prior login/TLS warming: actual initial RSA exchange for SELECT-only role."""
    import secrets
    import pymysql
    config,fixture_secret=mysql
    with pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=fixture_secret['username'],password='',ssl_disabled=True) as fixture_role:
        with fixture_role.cursor() as cursor:cursor.execute('SELECT @@socket');socket=cursor.fetchone()[0]
    username='fresh_rsa_'+secrets.token_hex(6);password=secrets.token_urlsafe(24)
    with pymysql.connect(unix_socket=socket,user='root',password='',database=config['database'],ssl_disabled=True,autocommit=True) as admin:
        with admin.cursor() as cursor:
            cursor.execute("CREATE USER %s@'127.0.0.1' IDENTIFIED WITH caching_sha2_password BY %s",(username,password))
            cursor.execute("GRANT SELECT ON learning.chosen TO %s@'127.0.0.1'",(username,))
            cursor.execute("SELECT plugin FROM mysql.user WHERE user=%s AND host='127.0.0.1'",(username,))
            assert cursor.fetchone()==('caching_sha2_password',)
    # This is the role's first connection. database_source constructs the real
    # PyMySQL client; password stays in memory and never appears in argv/env.
    original_connect=pymysql.connect;diagnostics=[]
    def observed_connect(*args,**kwargs):
        try:return original_connect(*args,**kwargs)
        except Exception as error:
            diagnostics.append(str(error).replace(password,'[synthetic password redacted]').replace(username,'[synthetic role]'))
            raise
    monkeypatch.setattr(pymysql,'connect',observed_connect)
    try:
        names,rows,receipt,batches=capture({**config,'tlsMode':'disabledLoopback','columns':['id','label'],'orderBy':['+id']},{'username':username,'password':password})
    except db.DatabaseSourceError:
        pytest.fail('Fresh fixture authentication refused: '+repr(diagnostics),pytrace=False)
    assert names==['id','label'] and rows==[[1,'one'],[2,'two']]
    assert all(type(row[0]) is int and type(row[1]) is str for row in rows)
    assert receipt['rows']==2 and batches==1
    # Check genuine privilege restriction after first-auth capture, independently
    # of the adapter's own read-only transaction setting.
    with pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=username,password=password,ssl_disabled=True,autocommit=True) as role:
        with role.cursor() as cursor:
            with pytest.raises(pymysql.err.OperationalError) as refusal:
                cursor.execute("INSERT INTO chosen (id,label) VALUES (999,'must-not-write')")
            assert refusal.value.args[0]==1142
            cursor.execute('SELECT COUNT(*) FROM chosen');assert cursor.fetchone()==(2,)

OPTION_CASES['database.mysql.backend'].append('test_mysql_first_fresh_caching_sha2_password_capture_over_literal_loopback')
OPTION_CASES['database.mysql.tlsMode'].append('test_mysql_first_fresh_caching_sha2_password_capture_over_literal_loopback')
