"""Unified Partition. Assignment indexes and outputs use owned bounded disk stores.

Legacy split/groupSplit/temporalSplit semantics remain untouched. Exclusions are
an inspection-only audit table, never a fourth model partition or a canvas wire.
"""
import hashlib
import json
import sqlite3
from contextlib import contextmanager

if __package__:
    from . import dataflow as df
    from . import dataflow_partition as legacy
else:
    import dataflow as df
    import dataflow_partition as legacy

ROLES=('training','validation','test')
PORTS=(*ROLES,'exclusions')


def legacy_config(c):
    strategy=c['strategy']
    values={k:v for k,v in c.items() if k!='strategy'}
    if strategy=='entity':values.update(nullKeys='error',allocationUnit='groups')
    return ('groupSplit' if strategy=='entity' else 'temporalSplit'),values


def validate_config(c):
    if type(c) is not dict or c.get('strategy') not in ('random','entity','time'):raise ValueError('Choose Random, By entity or By time.')
    strategy=c['strategy']
    fields={'strategy','seed','validationPercent','testPercent'}
    if strategy=='random':
        if set(c) not in (fields,fields|{'stratifyColumn'}):raise ValueError('Random Partition accepts only allocation, seed and optional stratification.')
        if 'stratifyColumn' in c and not df.names([c['stratifyColumn']]):raise ValueError('Choose one exact stratification column.')
        if type(c['seed']) is not int or not 0<=c['seed']<=2**32-1:raise ValueError('Seed must be an integer from 0 through 4,294,967,295.')
        if any(type(c[k]) is not int or not 1<=c[k]<=98 for k in ('validationPercent','testPercent')) or c['validationPercent']+c['testPercent']>=100:raise ValueError('Validation and Test percentages must leave positive Training, Validation and Test allocations.')
    else:
        expected=fields|{'groupBy'} if strategy=='entity' else {'strategy','timeColumn','timeEncoding','trainingEnd','validationEnd','embargoMilliseconds','entityColumns','entityPolicy'}
        if set(c)!=expected:raise ValueError('Use only the selected Partition strategy controls.')
        legacy.validate_config(*legacy_config(c))


def _counts(n,c):
    proportions=(100-c['validationPercent']-c['testPercent'],c['validationPercent'],c['testPercent'])
    values=[n*p//100 for p in proportions]
    for i in sorted(range(3),key=lambda i:(-(n*proportions[i]%100),i))[:n-sum(values)]:values[i]+=1
    if not all(values):raise ValueError('Partition requires three nonempty allocations in each stratum; add observations/entities or revise percentages/stratification.')
    return values


def _key(table,row,indices):
    values=[]
    for i in indices:
        column=table.columns[i];value=row[i]
        if column['dtype'] in ('image','audio','tensor'):raise ValueError('Partition keys must be scalar metadata, not images.')
        if value is None:raise ValueError('Resolve missing entity, stratification or time values before Partition.')
        if column['dtype']=='float64' and (type(value) is int or value.is_integer()):value=int(value)
        values.append([column['name'],column['dtype'],value])
    return values


@contextmanager
def _plan(table,c,check):
    check();validate_config(c)
    if table.role!='unassigned' or table.partition_digest or table.fitted_states:raise ValueError('Partition requires an unassigned table before fitted transforms; existing partitions cannot be replaced.')
    df._fitted_module().schema_valid(list(table.columns))
    n=len(table.rows)
    if not n or n!=len(table.row_ids) or n*len(table.columns)>df.MAX_TABLE_CELLS:raise ValueError('Partition input exceeds its typed table/identity/cell bounds.')
    store=df.DiskStore(check_cancelled=check);db=store.connection;cancel=[]
    store.partition_identity_prefix=table.row_ids.store.identity_prefix if isinstance(table.row_ids,df.StoredIDs) else ''
    def progress():
        try:check();return 0
        except BaseException as error:cancel.append(error);return 1
    db.set_progress_handler(progress,4096)
    try:
        db.execute('CREATE TABLE members (position INTEGER PRIMARY KEY, rid TEXT NOT NULL UNIQUE COLLATE BINARY, key TEXT NOT NULL COLLATE BINARY, rank BLOB NOT NULL, micros INTEGER, role TEXT, body TEXT)')
        db.execute('CREATE INDEX member_rank ON members(key,rank,rid)')
        db.execute('CREATE INDEX member_time ON members(micros,rid)')
        db.execute('CREATE TABLE entities (key TEXT PRIMARY KEY COLLATE BINARY, rank BLOB NOT NULL, role TEXT) WITHOUT ROWID')
        db.execute('CREATE INDEX entity_rank ON entities(rank,key)')
        strategy=c['strategy'];keys=c['groupBy'] if strategy=='entity' else [c['stratifyColumn']] if strategy=='random' and 'stratifyColumn' in c else c['entityColumns'] if strategy=='time' else []
        indices=[table.index(name) for name in keys]
        time_index=table.index(c['timeColumn']) if strategy=='time' else None
        if time_index is not None:
            expected='int64' if c['timeEncoding']=='epochMilliseconds' else 'string'
            if table.columns[time_index]['dtype']!=expected:raise ValueError('Time column dtype differs from its declared encoding; cast explicitly before Partition.')
            first=legacy._time(c['trainingEnd'],c['timeEncoding']);second=legacy._time(c['validationEnd'],c['timeEncoding']);gap=c['embargoMilliseconds']*1000
        for position,(row,rid) in enumerate(df.row_pairs(table)):
            if position%128==0:check()
            if len(row)!=len(table.columns) or not isinstance(rid,str) or not rid or '\0' in rid or len(rid.encode())>1024:raise ValueError('Partition row width or observation identity is invalid.')
            for value,column in zip(row,table.columns):
                if value is None:
                    if not column['nullable']:raise ValueError('Partition input null violates its schema.')
                elif not df._fitted_module().typed(value,column['dtype']):raise ValueError('Partition input violates its finite typed schema.')
            typed=_key(table,row,indices);key=json.dumps(typed,sort_keys=True,separators=(',',':'),allow_nan=False)
            if len(key.encode())>1_048_576:raise ValueError('Partition key exceeds the bounded record budget.')
            micros=role=body=None
            if strategy=='time':
                micros=legacy._time(row[time_index],c['timeEncoding'])
                role='training' if micros<=first else 'validation' if first+gap<micros<=second else 'test' if micros>second+gap else 'exclusions'
                body=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
                if len(body.encode())>1_048_576:raise ValueError('Partition row exceeds the bounded record budget.')
                rank=b''
            elif strategy=='entity':rank=df.digest(['kwip.groupSplit.rank.v1',c['seed'],typed])
            else:rank=df.digest(['kwip.partition.random.v1',c['seed'],typed,rid])
            store._allocation_limit()
            db.execute('INSERT INTO members VALUES (?,?,?,?,?,?,?)',(position,rid[len(store.partition_identity_prefix):] if rid.startswith(store.partition_identity_prefix) else rid,key,bytes.fromhex(rank) if isinstance(rank,str) else rank,micros,role,body))
            if strategy=='entity':db.execute('INSERT OR IGNORE INTO entities(key,rank) VALUES (?,?)',(key,bytes.fromhex(rank) if isinstance(rank,str) else rank))
            store._account()
        if db.execute('SELECT COUNT(*) FROM members').fetchone()[0]!=n:raise ValueError('Partition input row/identity count changed.')
        if strategy=='random':
            for key,count in db.execute('SELECT key,COUNT(*) FROM members GROUP BY key'):
                training,validation,_=_counts(count,c)
                db.execute('WITH ranks AS MATERIALIZED (SELECT position,row_number() OVER (ORDER BY rank,rid) AS r FROM members WHERE key=?) UPDATE members SET role=(SELECT CASE WHEN r<=? THEN "training" WHEN r<=? THEN "validation" ELSE "test" END FROM ranks WHERE ranks.position=members.position) WHERE key=?',(key,training,training+validation,key))
        elif strategy=='entity':
            training,validation,_=_counts(db.execute('SELECT COUNT(*) FROM entities').fetchone()[0],c)
            db.execute('WITH ranks AS MATERIALIZED (SELECT key,row_number() OVER (ORDER BY rank,key) AS r FROM entities) UPDATE entities SET role=(SELECT CASE WHEN r<=? THEN "training" WHEN r<=? THEN "validation" ELSE "test" END FROM ranks WHERE ranks.key=entities.key)',(training,training+validation))
            db.execute('UPDATE members SET role=(SELECT role FROM entities WHERE entities.key=members.key)')
        else:
            if c['entityPolicy']=='requireDisjoint' and db.execute('SELECT 1 FROM members WHERE role!="exclusions" GROUP BY key HAVING COUNT(DISTINCT role)>1 LIMIT 1').fetchone():raise ValueError('An entity crosses retained chronological partitions; revise cutoffs or explicitly allow entities across time.')
        counts=dict(db.execute('SELECT role,COUNT(*) FROM members GROUP BY role'))
        if any(not counts.get(role) for role in ROLES):raise ValueError('Partition requires three nonempty retained populations; revise cutoffs, embargo or allocations.')
        db.commit();store._account();check()
        yield store
    except sqlite3.IntegrityError as error:raise ValueError('Partition requires unique observation identities.') from error
    except sqlite3.OperationalError as error:
        if cancel:raise cancel[0]
        raise ValueError('Partition assignment exceeded its bounded scratch storage or database capacity.') from error
    finally:store.close()


def _membership(db,role,check,prefix):
    h=hashlib.sha256(b'kwip.membership.v1\n')
    for index,(identity,) in enumerate(db.execute('SELECT rid FROM members WHERE role=? ORDER BY rid COLLATE BINARY',(role,))):
        if index%256==0:check()
        raw=(prefix+identity).encode();h.update(str(len(raw)).encode()+b':'+raw)
    return h.hexdigest()


def _diagnostic(table,c,store,check):
    db=store.connection;parts=[]
    for role in ROLES:
        count,minimum,maximum=db.execute('SELECT COUNT(*),MIN(micros),MAX(micros) FROM members WHERE role=?',(role,)).fetchone()
        part={'role':role,'rows':count,'membershipDigest':_membership(db,role,check,store.partition_identity_prefix)}
        if c['strategy']=='entity':part['groups']=db.execute('SELECT COUNT(*) FROM entities WHERE role=?',(role,)).fetchone()[0]
        if c['strategy']=='time':part.update(minTimeMicroseconds=minimum,maxTimeMicroseconds=maximum)
        parts.append(part)
    return {'kind':'partition','algorithm':'kwip.partition.'+c['strategy']+'.v1','configDigest':df.digest(c),
            'inputRows':len(table.rows),'inputSourceDigest':table.source_digest,'inputDataVersion':table.data_version,'inputMembershipDigest':df.membership_digest(table.row_ids),
            'partitions':parts,'excludedRows':db.execute('SELECT COUNT(*) FROM members WHERE role="exclusions"').fetchone()[0],
            'excludedMembershipDigest':_membership(db,'exclusions',check,store.partition_identity_prefix)}


def _expected_pairs(table,c,store,role,check):
    db=store.connection
    if c['strategy']=='time':
        for position,(body,rid) in enumerate(db.execute('SELECT body,rid FROM members WHERE role=? ORDER BY micros,rid COLLATE BINARY',(role,))):
            if position%128==0:check()
            yield tuple(json.loads(body)),store.partition_identity_prefix+rid
    else:
        cursor=db.execute('SELECT rid,role FROM members ORDER BY position')
        for position,((row,rid),(expected,assigned)) in enumerate(zip(df.row_pairs(table),cursor)):
            if position%128==0:check()
            if rid!=store.partition_identity_prefix+expected:raise ValueError('Partition input identities changed during assignment.')
            if assigned==role:yield row,rid


def apply(table,c,check):
    results={}
    with _plan(table,c,check) as store:
        diagnostic=_diagnostic(table,c,store,check);partition=df.digest(diagnostic);diagnostics=legacy._merge_diagnostics(table.diagnostics,diagnostic)
        try:
            if c['strategy']=='time':
                for role in PORTS:
                    results[role]=df.streaming_rows(table,_expected_pairs(table,c,store,role,check),check_cancelled=check)
            else:
                writers={}
                try:
                    for role in PORTS:writers[role]=df.row_store(table.columns,check,store.partition_identity_prefix)
                    cursor=store.connection.execute('SELECT rid,role FROM members ORDER BY position')
                    for position,((row,rid),(expected,assigned)) in enumerate(zip(df.row_pairs(table),cursor)):
                        if position%128==0:check()
                        if rid!=store.partition_identity_prefix+expected:raise ValueError('Partition input identities changed during assignment.')
                        writers[assigned].append(tuple(row),rid)
                    for role,writer in writers.items():
                        rows,ids=writer.finish();results[role]=df.derive(table,rows=rows,ids=ids)
                except BaseException:
                    for writer in writers.values():writer.close()
                    raise
            for role,result in results.items():
                results[role]=df.replace(result,role=role if role in ROLES else 'unassigned',partition_digest=partition,fitted_states=(),diagnostics=diagnostics)
            return results
        except BaseException:
            for result in results.values():result.rows.store.close()
            raise


def verify_output(table,c,outputs,check):
    if set(outputs)!=set(PORTS):raise ValueError('Cached Partition outputs differ from the captured roles/audit table.')
    with _plan(table,c,check) as store:
        diagnostic=_diagnostic(table,c,store,check);partition=df.digest(diagnostic);diagnostics=legacy._merge_diagnostics(table.diagnostics,diagnostic)
        for role in PORTS:
            output=outputs[role]
            count=store.connection.execute('SELECT COUNT(*) FROM members WHERE role=?',(role,)).fetchone()[0]
            if (not isinstance(output,df.Table) or output.columns!=table.columns or output.source_digest!=table.source_digest or output.role!=(role if role in ROLES else 'unassigned') or output.partition_digest!=partition or output.fitted_states or output.diagnostics!=diagnostics or len(output.rows)!=count or len(output.row_ids)!=count):raise ValueError('Cached Partition counts, schema or lineage differ.')
            for expected,actual in zip(_expected_pairs(table,c,store,role,check),df.row_pairs(output)):
                if expected!=(tuple(actual[0]),actual[1]):raise ValueError('Cached Partition values, order or observation identities differ.')
