"""Standard-library-only sampling admission before data-engine authorization."""
import json
import math

ALL={'schemaVersion':1,'mode':'all'}
MAX_SAMPLE_ROWS=5_000_000


def normalize_sampling(policy=None):
    if policy is None:return dict(ALL)
    if not isinstance(policy,dict):raise ValueError('Sampling policy must be an object')
    version=policy.get('schemaVersion',1)
    if type(version) is not int or version!=1:raise ValueError('Unsupported sampling policy version')
    mode=policy.get('mode')
    keys={'schemaVersion','mode'}
    if mode=='rowCount':
        keys.add('rowCount');value=policy.get('rowCount')
        if type(value) is not int or not 1<=value<=MAX_SAMPLE_ROWS:raise ValueError('Sample row count must be an integer from 1 through 5000000')
        result={'schemaVersion':1,'mode':mode,'rowCount':value}
    elif mode=='percentage':
        keys.add('percentage');value=policy.get('percentage')
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=100:raise ValueError('Sample percentage must be greater than zero and at most 100')
        result={'schemaVersion':1,'mode':mode,'percentage':value}
    elif mode=='all':result=dict(ALL)
    else:raise ValueError('Sampling mode must be all, rowCount or percentage')
    if set(policy)-keys:raise ValueError('Unknown or inactive sampling policy fields')
    return result


def is_complete_sampling(policy):
    c=normalize_sampling(policy)
    return c['mode']=='all' or (c['mode']=='percentage' and c['percentage']==100)


def sampling_row_limit(policy,total_rows=None):
    c=normalize_sampling(policy)
    if is_complete_sampling(c):return None
    if c['mode']=='rowCount':return c['rowCount']
    if type(total_rows) is not int or total_rows<0:
        raise ValueError('Percentage sampling requires a complete source cardinality scan or a verified row count')
    return math.ceil(total_rows*c['percentage']/100)



def normalize_complete_nodes(nodes):
    if not isinstance(nodes,list) or len(nodes)>128 or any(not isinstance(x,str) or not 1<=len(x)<=128 or '\0' in x for x in nodes) or len(set(nodes))!=len(nodes):
        raise ValueError('Complete nodes require at most 128 distinct bounded node IDs')
    try:
        for node in nodes:node.encode('utf-8')
    except UnicodeError as error:raise ValueError('Complete node IDs must be valid UTF-8') from error
    return list(nodes)


def _parse(text):
    if not isinstance(text,str):raise ValueError('Demand option requires JSON text')
    try:size=len(text.encode('utf-8'))
    except UnicodeError as error:raise ValueError('Demand option must be UTF-8 JSON') from error
    if size>4096:raise ValueError('Demand option exceeds its 4096 byte bound')
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate demand option JSON field')
            result[key]=value
        return result
    def nonfinite(value):raise ValueError('Demand JSON must be finite')
    try:return json.loads(text,object_pairs_hook=object_pairs,parse_constant=nonfinite)
    except (ValueError,RecursionError) as error:raise ValueError('Invalid demand option JSON') from error


def parse_sampling_json(text):
    value=_parse(text)
    if not isinstance(value,dict):raise ValueError('Sampling option requires a JSON object')
    return normalize_sampling(value)


def parse_complete_nodes_json(text):return normalize_complete_nodes(_parse(text))
