"""Bounded old-partition evidence verifier and shared canonical clock/config helpers.

Current Partition owns execution. The old plan only verifies frozen captured evidence.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re

if __package__:
    from . import dataflow as df
else:
    import dataflow as df

ROLES=('training','validation','test')
MAX_ARTIFACT_BYTES=50_000_000
MAX_DIAGNOSTICS=16_384
EPOCH=datetime(1970,1,1)
ISO=re.compile(r'([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,6}))?(Z|[+-][0-9]{2}:[0-9]{2})\Z')


def _encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def _micros(value):
    delta=value-EPOCH
    return (delta.days*86400+delta.seconds)*1_000_000+delta.microseconds

MIN_TIME=_micros(datetime.min)
MAX_TIME=_micros(datetime.max)


def _time(value,encoding):
    if encoding=='epochMilliseconds':
        if type(value) is not int or not -(2**63)<=value<2**63 or not MIN_TIME<=value*1000<=MAX_TIME:
            raise ValueError('Epoch milliseconds require an int64 within UTC calendar years 0001 through 9999.')
        return value*1000
    match=ISO.fullmatch(value) if isinstance(value,str) else None
    if match is None:raise ValueError('Time requires strict ISO8601 with an explicit timezone offset and at most six fractional digits.')
    year,month,day,hour,minute,second,fraction,zone=match.groups()
    try:
        instant=datetime(*map(int,(year,month,day,hour,minute,second)),int((fraction or '').ljust(6,'0')))
        if zone!='Z':
            hours=int(zone[1:3]);minutes=int(zone[4:6])
            if hours>23 or minutes>59:raise ValueError('Invalid timezone offset')
            offset=timedelta(hours=hours,minutes=minutes)
            instant=instant-offset if zone[0]=='+' else instant+offset
        return _micros(instant)
    except (ValueError,OverflowError) as error:raise ValueError('Time is outside valid UTC calendar years 0001 through 9999, or contains an invalid date/time.') from error


def validate_config(operator,config):
    group={'groupBy','nullKeys','seed','allocationUnit','validationPercent','testPercent'}
    temporal={'timeColumn','timeEncoding','trainingEnd','validationEnd','embargoMilliseconds','entityColumns','entityPolicy'}
    fields=group if operator=='groupSplit' else temporal if operator=='temporalSplit' else None
    if fields is None or not isinstance(config,dict) or set(config)!=fields:raise ValueError('Partition requires its exact explicit configuration fields.')
    if operator=='groupSplit':
        if not df.names(config['groupBy']) or config['nullKeys']!='error' or config['allocationUnit']!='groups':raise ValueError('Group split requires unique grouping columns, missing-key error policy and group allocation units.')
        if type(config['seed']) is not int or not 0<=config['seed']<=2**32-1:raise ValueError('Group split seed must be a uint32 integer.')
        if any(type(config[k]) is not int or not 1<=config[k]<=98 for k in ('validationPercent','testPercent')) or config['validationPercent']+config['testPercent']>=100:
            raise ValueError('Validation and test percentages must leave a positive training proportion.')
    else:
        if not df.names([config['timeColumn']]) or config['timeEncoding'] not in ('iso8601Offset','epochMilliseconds'):raise ValueError('Invalid chronological time column or encoding.')
        if not isinstance(config['entityColumns'],list) or (config['entityColumns'] and not df.names(config['entityColumns'])):raise ValueError('Entity columns must be ordered unique names.')
        if config['entityPolicy'] not in ('allowAcrossTime','requireDisjoint') or (config['entityPolicy']=='requireDisjoint' and not config['entityColumns']):raise ValueError('Disjoint entity policy requires explicit entity columns.')
        if type(config['embargoMilliseconds']) is not int or not 0<=config['embargoMilliseconds']<=86_400_000_000:raise ValueError('Embargo must be integer milliseconds from zero through 1000 days.')
        start=_time(config['trainingEnd'],config['timeEncoding']);end=_time(config['validationEnd'],config['timeEncoding'])
        if start>=end:raise ValueError('Training cutoff must precede validation cutoff.')


def _validate_table(table):
    if table.role!='unassigned' or table.partition_digest or table.fitted_states:
        raise ValueError('Prepare partitions before splitting or fitting transforms; existing partition evidence cannot be replaced.')
    df._fitted_module().schema_valid(list(table.columns))
    if len(table.rows)*len(table.columns)>5_000_000:raise ValueError('Partition exceeds five million table cells.')
    if not 1<=len(table.rows)<=df.MAX_ROWS or len(table.rows)!=len(table.row_ids):raise ValueError('Partition input row bounds or identities differ.')
    if any(not isinstance(i,str) or not 1<=len(i)<=200 for i in table.row_ids) or len(set(table.row_ids))!=len(table.row_ids):raise ValueError('Partition requires unique bounded observation identities.')
    for row in table.rows:
        if len(row)!=len(table.columns):raise ValueError('Partition input row width differs from schema.')
        for value,column in zip(row,table.columns):
            if value is None:
                if not column['nullable']:raise ValueError('Partition input null violates schema.')
            elif not df._fitted_module().typed(value,column['dtype']):raise ValueError('Partition input violates finite schema dtype.')


def _key(table,row,indices):
    values=[]
    for i in indices:
        value=row[i]
        if value is None:raise ValueError('Resolve missing group/entity key values before partitioning.')
        if table.columns[i]['dtype']=='float64' and (isinstance(value,int) or value.is_integer()):value=int(value)
        values.append(value)
    return tuple(values)


def _merge_diagnostics(prior,diagnostic):
    values=[];seen=set();size=2
    for item in (*prior,diagnostic):
        content=_encoded(item)
        if content in seen:continue
        size+=len(content)+1
        if len(values)>=MAX_DIAGNOSTICS or size>MAX_ARTIFACT_BYTES:raise ValueError('Partition diagnostic count or byte budget exceeded.')
        seen.add(content);values.append(item)
    return tuple(values)


@dataclass(frozen=True)
class PartitionPlan:
    indices:dict
    excluded:tuple
    partition_digest:str
    diagnostics:tuple


def plan_partition(table,operator,config):
    validate_config(operator,config);_validate_table(table)
    keys=config['groupBy'] if operator=='groupSplit' else [config['timeColumn'],*config['entityColumns']]
    if any(table.columns[table.index(name)]['dtype'] in ('image','audio','tensor') for name in keys):raise ValueError('Image columns cannot be partition keys; choose scalar metadata columns.')
    assigned={role:[] for role in ROLES};excluded=[];info=[]
    if operator=='groupSplit':
        columns=[table.index(name) for name in config['groupBy']];groups={}
        for i,row in enumerate(table.rows):groups.setdefault(_key(table,row,columns),[]).append(i)
        proportions=[100-config['validationPercent']-config['testPercent'],config['validationPercent'],config['testPercent']]
        counts=[len(groups)*percent//100 for percent in proportions]
        remainder=len(groups)-sum(counts)
        priority=sorted(range(3),key=lambda i:(-(len(groups)*proportions[i]%100),i))
        for i in priority[:remainder]:counts[i]+=1
        if not all(counts):raise ValueError('Group allocation requires three nonempty partitions; add groups or revise proportions.')
        def rank(key):
            typed=[[table.columns[i]['name'],table.columns[i]['dtype'],value] for i,value in zip(columns,key)]
            return df.digest(['kwip.groupSplit.rank.v1',config['seed'],typed]),_encoded(typed)
        ordered=sorted(groups,key=rank);cursor=0
        for role,count in zip(ROLES,counts):
            for key in ordered[cursor:cursor+count]:assigned[role].extend(groups[key])
            assigned[role].sort();cursor+=count
            info.append({'role':role,'groups':count})
    else:
        index=table.index(config['timeColumn'])
        expected='string' if config['timeEncoding']=='iso8601Offset' else 'int64'
        if table.columns[index]['dtype']!=expected:raise ValueError('Time column dtype differs from declared encoding; cast explicitly before splitting.')
        entity_columns=[table.index(name) for name in config['entityColumns']]
        first=_time(config['trainingEnd'],config['timeEncoding']);second=_time(config['validationEnd'],config['timeEncoding']);gap=config['embargoMilliseconds']*1000
        times=[];entities={}
        for i,row in enumerate(table.rows):
            if row[index] is None:raise ValueError('Resolve missing time values before chronological splitting.')
            instant=_time(row[index],config['timeEncoding']);times.append(instant)
            entity=_key(table,row,entity_columns) if entity_columns else None
            role='training' if instant<=first else 'validation' if first+gap<instant<=second else 'test' if instant>second+gap else None
            if role is None:excluded.append(i);continue
            if entity_columns and config['entityPolicy']=='requireDisjoint':
                previous=entities.setdefault(entity,role)
                if previous!=role:raise ValueError('An entity crosses retained chronological partitions; revise cutoffs or explicitly allow entities across time.')
            assigned[role].append(i)
        if not all(assigned.values()):raise ValueError('Chronological cutoffs and embargo must leave three nonempty partitions.')
        for role in ROLES:
            assigned[role].sort(key=lambda i:(times[i],table.row_ids[i]))
            info.append({'role':role,'minTimeMicroseconds':times[assigned[role][0]],'maxTimeMicroseconds':times[assigned[role][-1]]})
    # Establish exact exhaustive assignment, including embargo exclusions.
    all_indices=[i for role in ROLES for i in assigned[role]]+excluded
    if len(all_indices)!=len(table.rows) or len(set(all_indices))!=len(table.rows):raise ValueError('Partition assignment is not disjoint and exhaustive.')
    for item in info:
        ids=[table.row_ids[i] for i in assigned[item['role']]]
        item.update(rows=len(ids),membershipDigest=df.membership_digest(ids))
    diagnostic={'kind':'partition','algorithm':'kwip.'+operator+'.v1','configDigest':df.digest(config),
        'inputRows':len(table.rows),'inputSourceDigest':table.source_digest,'inputDataVersion':table.data_version,'inputMembershipDigest':df.membership_digest(table.row_ids),
        'partitions':info,'excludedRows':len(excluded),'excludedMembershipDigest':df.membership_digest([table.row_ids[i] for i in excluded])}
    partition=df.digest(diagnostic)
    diagnostics=_merge_diagnostics(table.diagnostics,diagnostic)
    for role in ROLES:
        envelope={'schemaVersion':1,'kind':'table','columns':table.columns,'rows':[],
            'lineage':{'role':role,'sourceDigest':table.source_digest,'membershipDigest':next(x['membershipDigest'] for x in info if x['role']==role),
            'observationIDs':[],'dataVersion':'0'*64,'partitionDigest':partition,'fittedStateDigests':[],'diagnostics':diagnostics}}
        size=len(_encoded(envelope))
        for n,i in enumerate(assigned[role]):
            size+=len(_encoded(table.rows[i]))+len(_encoded(table.row_ids[i]))+(2 if n else 0)
            if size>MAX_ARTIFACT_BYTES:raise ValueError('Partition exceeds its 50 MB serialized byte budget.')
    return PartitionPlan({role:tuple(indices) for role,indices in assigned.items()},tuple(excluded),partition,diagnostics)


def partition_table(table,operator,config):
    raise ValueError('Retired partition execution is unavailable; use current Partition or explicitly repair a legacy random split.')


def verify_partition_output(table,operator,config,outputs):
    plan=plan_partition(table,operator,config)
    if set(outputs)!=set(ROLES):raise ValueError('Cached partition output roles differ.')
    for role,indices in plan.indices.items():
        value=outputs[role]
        if (not isinstance(value,df.Table) or value.columns!=table.columns or value.source_digest!=table.source_digest or
            value.role!=role or value.partition_digest!=plan.partition_digest or value.fitted_states or value.diagnostics!=plan.diagnostics or
            len(value.rows)!=len(indices) or len(value.row_ids)!=len(indices)):
            raise ValueError('Cached partition schema or lineage differs.')
        for index,row,identity in zip(indices,value.rows,value.row_ids):
            if table.rows[index]!=row or table.row_ids[index]!=identity:raise ValueError('Cached partition rows or membership differ.')
