"""Bounded typed-table DAG with explicit lineage and operator diagnostics."""
from dataclasses import dataclass, replace
from contextlib import nullcontext
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
if __package__:
    from .dataflow_disk import DiskStore, DiskRows, DiskIDs, iter_json, disk_scope
else:
    from dataflow_disk import DiskStore, DiskRows, DiskIDs, iter_json, disk_scope

if __package__:from .table_storage import StoredRows,StoredIDs
else:from table_storage import StoredRows,StoredIDs

if __package__:from .table_transfer_policy import MAX_CELLS, MAX_COLUMNS
else:from table_transfer_policy import MAX_CELLS, MAX_COLUMNS

MAX_TABLE_CELLS=MAX_CELLS
EAGER_ROW_LIMIT=100_000
EAGER_CELL_LIMIT=5_000_000
SPILL_ROWS=8192

PREFIX='kwip.data.'
MAX_BYTES=10_000_000
# Existing eager operator/config bounds; source admission uses MAX_TABLE_CELLS.
MAX_ROWS=EAGER_ROW_LIMIT
REGISTRY_PATH=Path(__file__).with_name('dataflow-operators.json')
REGISTRY=json.loads(REGISTRY_PATH.read_text())
# Retired descriptors are decode-only: no current registry entry or executor.
_RETIRED_REGISTRY=json.loads(Path(__file__).with_name('retired-data-operators-v1.json').read_text())
RETIRED_DEFINITIONS={op['id'][len(PREFIX):]:op for op in _RETIRED_REGISTRY['operators']}
DEFINITIONS={**RETIRED_DEFINITIONS,**{op['id'][len(PREFIX):]:op for op in REGISTRY['operators']}}
OPS={name:({p['id'] for p in op['inputs']},{p['id'] for p in op['outputs']},set(op['configSchema']['properties']),set(op['configSchema']['required'])) for name,op in DEFINITIONS.items()}


def schema_validate(value,schema):
    """Small validator for the exact JSON Schema vocabulary in our bundled registry."""
    if 'type' in schema:
        allowed=schema['type'] if isinstance(schema['type'],list) else [schema['type']]
        actual='boolean' if type(value) is bool else 'integer' if type(value) is int else 'number' if type(value) is float else 'string' if isinstance(value,str) else 'array' if isinstance(value,list) else 'object' if isinstance(value,dict) else 'null'
        if actual not in allowed and not(actual=='integer' and 'number' in allowed):raise ValueError('Invalid configuration value type')
    if 'enum' in schema and value not in schema['enum']:raise ValueError('Invalid configuration choice')
    if 'const' in schema and value!=schema['const']:raise ValueError('Invalid configuration constant')
    if type(value) in (int,float):
        if not math.isfinite(value) or value<schema.get('minimum',-math.inf) or value>schema.get('maximum',math.inf):raise ValueError('Configuration number outside bounds')
    if isinstance(value,str) and not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',MAX_BYTES):raise ValueError('Configuration string outside bounds')
    if isinstance(value,list):
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',MAX_ROWS):raise ValueError('Configuration array outside bounds')
        if schema.get('uniqueItems') and len({digest(v) for v in value})!=len(value):raise ValueError('Configuration items must be unique')
        for item in value:schema_validate(item,schema.get('items',{}))
    if isinstance(value,dict):
        if set(schema.get('required',[]))-set(value):raise ValueError('Required configuration field missing')
        if not schema.get('minProperties',0)<=len(value)<=schema.get('maxProperties',MAX_COLUMNS):raise ValueError('Configuration object outside bounds')
        for key,item in value.items():
            props=schema.get('properties',{})
            if key in props:schema_validate(item,props[key])
            elif schema.get('additionalProperties') is False:raise ValueError('Unknown configuration field')
            elif isinstance(schema.get('additionalProperties'),dict):schema_validate(item,schema['additionalProperties'])
    if 'oneOf' in schema:
        successes=0
        for candidate in schema['oneOf']:
            try:schema_validate(value,candidate);successes+=1
            except ValueError:pass
        if successes!=1:raise ValueError('Configuration must match exactly one conditional schema')
    if 'not' in schema:
        try:schema_validate(value,schema['not'])
        except ValueError:return
        raise ValueError('Configuration matches prohibited schema')


def digest(value):
    # Join/Union identities are tiny flat metadata vectors. The C encoder emits
    # exactly the same canonical bytes without one recursive Python generator
    # per token. Complete populations/structured objects retain the bounded stream.
    if type(value) in (list,tuple) and len(value)<=16 and all(
        (type(item) is str and len(item)<=4096) or
        (type(item) is int and item.bit_length()<=64) or
        type(item) in (float,bool,type(None)) for item in value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=True).encode('ascii')).hexdigest()
    h=hashlib.sha256()
    for block in iter_json(value):h.update(block)
    return h.hexdigest()

def membership_digest(ids):
    store=ids.store if isinstance(ids,StoredIDs) else None
    if store is not None:
        store.check_cancelled()
        if store.closed:raise ValueError('Stored observation identities are closed')
        cached=getattr(store,'membership_sha256',None)
        if store.finished and cached is not None:return cached
    h=hashlib.sha256(b'kwip.membership.v1\n')
    ordered=(x.encode() for x in ids.sorted_iter()) if isinstance(ids,StoredIDs) else sorted(x.encode() for x in ids)
    for row in ordered:h.update(str(len(row)).encode()+b':'+row)
    value=h.hexdigest()
    if store is not None and store.finished:store.membership_sha256=value
    return value

def names(value):
    return isinstance(value,list) and bool(value) and len(value)<=MAX_COLUMNS and all(isinstance(x,str) and bool(x.strip()) and 0<len(x)<=200 for x in value) and len(set(value))==len(value)

def _table_tool_module(operator):
    import importlib
    module={'sort':'dataflow_sort','unique':'dataflow_unique','reshapeTable':'dataflow_reshape','splitText':'dataflow_split_text'}[operator]
    return importlib.import_module('.'+module,__package__) if __package__ else importlib.import_module(module)


def upgrade_graph(graph,known_input_schemas=None):
    if __package__:from .retired_data_migration import upgrade_graph as upgrade
    else:from retired_data_migration import upgrade_graph as upgrade
    return upgrade(graph,known_input_schemas)


def configured_ports(operator,config,connected=None):
    if operator=='union':return set(config['inputs']),set(config['inputs']),{'table'}
    if operator=='sequenceWindows':
        if __package__:from .sequence_windows import ports
        else:from sequence_windows import ports
        return ports(config,connected)
    if operator=='textEncode':
        if __package__:from .text_encoding import ports
        else:from text_encoding import ports
        return ports(config,connected)
    if operator=='preprocess':
        if __package__:from .dataflow_preprocess import ports
        else:from dataflow_preprocess import ports
        return ports(config,connected)
    spec=OPS[operator]
    # Model resources are required in authoring/discovery, and admitted through the
    # owned resource manifest rather than pretending they are table-DAG values.
    required={p['id'] for p in DEFINITIONS[operator]['inputs'] if p['required'] and p['kind']!='model'}
    return spec[0],required,spec[1]


def validate(graph,*,historical=False):
    if not historical:graph=upgrade_graph(graph)
    if not isinstance(graph,dict) or set(graph)!={'schemaVersion','nodes','edges'} or type(graph['schemaVersion']) is not int or graph['schemaVersion']!=1:raise ValueError('Invalid dataflow schema')
    if not isinstance(graph['nodes'],list) or not 1<=len(graph['nodes'])<=128 or not isinstance(graph['edges'],list) or len(graph['edges'])>256:raise ValueError('Graph bounds exceeded')
    nodes={};inputs={};deps={}
    for n in graph['nodes']:
        if not isinstance(n,dict) or set(n)!={'id','operator','config'}:raise ValueError('Invalid node')
        key=n['id'];op=n['operator'];c=n['config']
        if not isinstance(key,str) or not 1<=len(key)<=128 or key in nodes:raise ValueError('Invalid or duplicate node ID')
        if not isinstance(op,str) or not op.startswith(PREFIX) or op[len(PREFIX):] not in OPS:raise ValueError('Unknown operator')
        op=op[len(PREFIX):];spec=OPS[op]
        schema_validate(c,DEFINITIONS[op]['configSchema'])
        if not isinstance(c,dict) or set(c)-spec[2] or spec[3]-set(c):raise ValueError('Invalid operator configuration')
        if op=='sourceFiles':
            if __package__:from .file_input_options import validate_configuration
            else:from file_input_options import validate_configuration
            c=validate_configuration(c)
        if op in ('selectRename','standardScalerFit','missing') and not names(c['columns']):raise ValueError('Invalid columns configuration')
        if op=='selectRename' and (not isinstance(c['rename'],dict) or set(c['rename'])-set(c['columns']) or not names([c['rename'].get(x,x) for x in c['columns']])):raise ValueError('Invalid rename configuration')
        if op=='filterRows':
            if __package__:from .dataflow_filter import validate_config
            else:from dataflow_filter import validate_config
            validate_config(c)
        if op=='sequenceWindows':
            if __package__:from .sequence_windows import validate_config
            else:from sequence_windows import validate_config
            validate_config(c)
        if op=='textEncode':
            if __package__:from .text_encoding import validate_config
            else:from text_encoding import validate_config
            validate_config(c)
        if op=='preprocess':
            if __package__:from .dataflow_preprocess import validate_config
            else:from dataflow_preprocess import validate_config
            validate_config(c)
        if op=='gameRules':
            if __package__:from .game_rules_tables import validate as validate_rules
            else:from game_rules_tables import validate as validate_rules
            validate_rules(c)
        if op=='decide':
            if __package__:from .decision_tables import validate
            else:from decision_tables import validate
            validate(c)
        if op=='classifyAudio':
            if __package__:from .audio_classification_tables import validate
            else:from audio_classification_tables import validate
            validate(c)
        if op=='transcribeAudio':
            if __package__:from .speech_tables import validate
            else:from speech_tables import validate
            validate(c)
        if op=='extractText':
            if __package__:from .ocr import validate
            else:from ocr import validate
            validate(c)
        if op=='prepareMedia':
            if __package__:from .prepare_media import validate_config
            else:from prepare_media import validate_config
            validate_config(c)
        if op=='formula':
            if __package__:from .dataflow_formula import validate_config
            else:from dataflow_formula import validate_config
            validate_config(c)
        if op=='columns':
            if __package__:from .dataflow_columns import validate_config
            else:from dataflow_columns import validate_config
            validate_config(c)
        if op=='cast' and (not isinstance(c['columns'],dict) or not names(list(c['columns'])) or any(v not in ('float64','int64','string','bool') for v in c['columns'].values())):raise ValueError('Invalid cast configuration')
        if op=='filter' and (not isinstance(c['column'],str) or c['comparison'] not in ('eq','ne','lt','le','gt','ge') or type(c['value']) not in (str,int,float,bool)):raise ValueError('Invalid filter configuration')
        if op=='missing' and (c['policy'] not in ('error','drop','fill') or ('value' in c)!=(c['policy']=='fill') or (c['policy']=='fill' and type(c['value']) not in (str,int,float,bool))):raise ValueError('Invalid missing configuration')
        if op=='split' and (type(c['seed']) is not int or not 0<=c['seed']<=2**32-1 or type(c['validationPercent']) is not int or not 1<=c['validationPercent']<=50 or ('stratifyColumn' in c and not isinstance(c['stratifyColumn'],str))):raise ValueError('Invalid split configuration')
        if op in ('sort','unique','reshapeTable','splitText'):
            _table_tool_module(op).validate_config(c)
        if op=='union':
            if __package__:from .dataflow_union import validate_config
            else:from dataflow_union import validate_config
            validate_config(c)
        if op=='outputDelimited' and c.get('destination')=='api':
            if __package__:from .api_tables import options as api_options
            else:from api_tables import options as api_options
            a=api_options(c.get('api',{}))
            if a['direction']!='output':raise ValueError('API output requires output configuration')
        if op=='outputDelimited' and c.get('destination')=='database':
            if __package__:from .database_output import options as database_options
            else:from database_output import options as database_options
            database_options(c.get('database',{}))
        if op=='outputDelimited' and c.get('destination','file')=='file':
            if __package__:
                from .table_output_writers import options
                from .dataflow_output import safe_name
            else:
                from table_output_writers import options
                from dataflow_output import safe_name
            options(c);safe_name(c['filename'])
        if op=='join':_join_module().validate_config(c)
        if op=='aggregate':_aggregate_module().validate_config(c)
        if op in ('groupSplit','temporalSplit'):_partition_module().validate_config(op,c)
        if op=='partition':
            if __package__:from .dataflow_partition_tool import validate_config
            else:from dataflow_partition_tool import validate_config
            validate_config(c)
        nodes[key]=(op,c);inputs[key]={};deps[key]=set()
    for edge in graph['edges']:
        if not isinstance(edge,dict) or set(edge)!={'from','to'}:raise ValueError('Invalid edge')
        a,b=edge['from'],edge['to']
        if any(not isinstance(e,dict) or set(e)!={'node','port'} or not all(isinstance(v,str) for v in e.values()) for e in (a,b)):raise ValueError('Invalid endpoint')
        if a['node'] not in nodes or b['node'] not in nodes:raise ValueError('Unknown endpoint node')
        if a['port'] not in configured_ports(*nodes[a['node']])[2] or b['port'] not in configured_ports(*nodes[b['node']])[0]:raise ValueError('Unknown port')
        if any(p.get('inspectionOnly') and p['id']==a['port'] for p in DEFINITIONS[nodes[a['node']][0]]['outputs']):raise ValueError('Inspection-only audit tables cannot be connected to the canvas.')
        if (a['port']=='state')!=(b['port']=='state'):raise ValueError('Incompatible port kinds')
        if b['port'] in inputs[b['node']]:raise ValueError('Input already connected')
        inputs[b['node']][b['port']]=(a['node'],a['port']);deps[b['node']].add(a['node'])
    for key,(op,c) in nodes.items():
        allowed,required,outputs=configured_ports(op,c,inputs[key])
        if not required<=set(inputs[key])<=allowed:raise ValueError('Required input not connected')
        for edge in graph['edges']:
            if edge['from']['node']==key and edge['from']['port'] not in outputs:raise ValueError('Connect the corresponding Preprocess partition input before using its output.')
    order=[]
    while len(order)<len(nodes):
        ready=[k for k in nodes if k not in order and deps[k]<=set(order)]
        if not ready:raise ValueError('Dataflow cycle')
        order.extend(ready)
    ancestry={}
    for node in order:
        parents=set().union(*(ancestry[p]|{p} for p in deps[node])) if deps[node] else set()
        ancestry[node]=parents
        if nodes[node][0]=='sequenceWindows' and any(nodes[p][0]=='sequenceWindows' for p in parents):
            raise ValueError('Do not create sequence windows from already windowed examples.')
        if nodes[node][0] in ('partition','split','groupSplit','temporalSplit') and any(nodes[p][0]=='sequenceWindows' for p in parents):
            raise ValueError('Partition source documents or contiguous regions before creating sequence windows; overlapping windows cannot be split safely afterward.')
    digest(graph) # Reject unsupported/nonfinite JSON before source allocation.
    return nodes,inputs,order

@dataclass(frozen=True)
class Table:
    columns:tuple
    rows:tuple
    row_ids:tuple
    source_digest:str
    role:str='unassigned'
    data_version:str=''
    partition_digest:str=''
    fitted_states:tuple=()
    diagnostics:tuple=()

    def __getattribute__(self,name):
        value=object.__getattribute__(self,name)
        if name=='data_version':
            if __package__:from .table_identity import CanonicalTableIdentity
            else:from table_identity import CanonicalTableIdentity
            if isinstance(value,CanonicalTableIdentity):return value.resolve()
        return value

    def index(self,name):
        for i,c in enumerate(self.columns):
            if c['name']==name:return i
        raise ValueError(f'Unknown column: {name}')
    def values(self,name):return [r[self.index(name)] for r in self.rows]
    def evidence(self):return {'role':self.role,'sourceDigest':self.source_digest,'membershipDigest':membership_digest(self.row_ids),'observationIDs':self.row_ids if isinstance(self.row_ids,StoredIDs) else list(self.row_ids)}
    def preview(self,*,include_version=True):return {'columns':list(self.columns),'rows':[[v[:200] if isinstance(v,str) else v for v in row] for row in self.rows[:8]],'rowCount':len(self.rows),'partition':self.role,'dataVersion':self.data_version if include_version else None,'diagnostics':list(self.diagnostics[-16:]),'diagnosticCount':len(self.diagnostics)}


def fit_observation_ids(state):
    evidence=state['fitEvidence']
    if 'observationIDSuffixes' in evidence:
        return (evidence['observationIDPrefix']+suffix for suffix in evidence['observationIDSuffixes'])
    return iter(evidence['observationIDs'])


def derive(table,rows=None,ids=None,columns=None,**changes):
    rows=table.rows if rows is None else rows if isinstance(rows,StoredRows) else tuple(rows);ids=table.row_ids if ids is None else ids if isinstance(ids,StoredIDs) else tuple(ids);columns=table.columns if columns is None else tuple(columns)
    if __package__:from .table_identity import version
    else:from table_identity import version
    return replace(table,rows=rows,row_ids=ids,columns=columns,data_version=version(table,columns,rows,ids),**changes)


def row_store(columns,check_cancelled,identity_prefix=''):
    if __package__:from .polars_preparation import current_engine
    else:from polars_preparation import current_engine
    if current_engine()=='polars':
        if __package__:from .dataflow_columnar import ColumnarStore
        else:from dataflow_columnar import ColumnarStore
        return ColumnarStore(columns,check_cancelled,identity_prefix=identity_prefix)
    return DiskStore(check_cancelled,identity_prefix=identity_prefix)


def source_csv(path,separator=",",*,options=None,check_cancelled=lambda:None,sampling=None):
    if isinstance(options,dict) and options.get('format')=='api':
        if __package__:from .api_table_snapshot import source_table
        else:from api_table_snapshot import source_table
        return source_table(path,options,check_cancelled,sampling=sampling)
    if isinstance(options,dict) and options.get('format')=='database':
        if __package__:from .database_source import source_table
        else:from database_source import source_table
        return source_table(path,options,check_cancelled,sampling=sampling)
    if isinstance(options,dict) and options.get('format') not in (None,'csv','database'):
        if __package__:from .table_sources import source_file
        else:from table_sources import source_file
        return source_file(path,options,check_cancelled,sampling=sampling)
    if isinstance(options,dict) and options.get('format')=='csv':
        flattened={key:value for key,value in options.items() if key not in ('format','options')}
        nested=options.get('options',{})
        if not isinstance(nested,dict) or set(flattened)&set(nested):raise ValueError('CSV settings must not be duplicated between flat and nested options')
        options={**flattened,**nested}
    if isinstance(options,dict) and options.get('parserProfile')=='polars':
        if __package__:
            from .polars_preparation import engine_identity
            from .polars_csv import source_csv as native_csv
        else:
            from polars_preparation import engine_identity
            from polars_csv import source_csv as native_csv
        engine_identity('polars')
        return native_csv(path,options,check_cancelled,sampling=sampling)
    if __package__:from .csv_parser import CSVStream
    else:from csv_parser import CSVStream
    # Hash incrementally before assigning stable observation IDs. The second pass
    # verifies the same bytes while parsing the private immutable source capture.
    hasher=hashlib.sha256();size=0
    with Path(path).open('rb') as stream:
        while True:
            check_cancelled();block=stream.read(262144)
            if not block:break
            size+=len(block)
            if size>MAX_BYTES:raise ValueError('CSV source exceeds its 10 MB byte budget')
            hasher.update(block)
    source=hasher.hexdigest();rows=[];store=None;nullable=None
    try:
        with CSVStream(path,options,separator,check_cancelled=check_cancelled,max_bytes=MAX_BYTES,max_columns=MAX_COLUMNS) as parsed:
            nullable=[False]*len(parsed.header)
            for index,row in enumerate(parsed):
                if (index+1)>MAX_TABLE_CELLS//len(row):raise ValueError('Table exceeds the five million cell storage budget; select fewer columns or reduce source size')
                for i,value in enumerate(row):nullable[i] |= value is None
                if store is None and (index>=SPILL_ROWS or (index+1)*len(row)>65536):
                    store=row_store(tuple({'name':n,'dtype':'string','nullable':True} for n in parsed.header),check_cancelled,source+':')
                    for previous,value in enumerate(rows):store.append(value,f'{source}:{previous}')
                    rows=[]
                if store is None:rows.append(row)
                else:store.append(row,f'{source}:{index}')
            if parsed.source_sha256!=source:raise ValueError('Source changed while parsing; capture and prepare it again')
            columns=tuple({'name':n,'dtype':'string','nullable':nullable[i]} for i,n in enumerate(parsed.header))
        if store is None:rows=tuple(rows);ids=tuple(f'{source}:{i}' for i in range(len(rows)))
        else:rows,ids=store.finish()
        result=derive(Table(columns,rows,ids,source))
        if sampling is not None:
            if __package__:from .preparation_demand import sample_table
            else:from preparation_demand import sample_table
            result=sample_table(result,sampling,check_cancelled)
        return result
    except BaseException:
        if store is not None:store.close()
        raise


def row_pairs(table):
    if hasattr(table.rows,'iter_pairs') and table.rows.store is getattr(table.row_ids,'store',None):
        return table.rows.iter_pairs()
    return zip(table.rows,table.row_ids)


def streaming_rows(table, pairs, columns=None, check_cancelled=lambda:None, *, identity_prefix=None):
    """Transform once into owned disk rows, retaining original observation identities."""
    columns=table.columns if columns is None else tuple(columns)
    store=row_store(columns,check_cancelled,(table.row_ids.store.identity_prefix if isinstance(table.row_ids,StoredIDs) else '') if identity_prefix is None else identity_prefix)
    try:
        for index,(row,rid) in enumerate(pairs):
            if index>=MAX_TABLE_CELLS//len(columns):raise ValueError('Table exceeds the five million cell storage budget')
            store.append(tuple(row),rid)
        rows,ids=store.finish()
        return derive(table,rows=rows,ids=ids,columns=columns)
    except BaseException:store.close();raise


def admit_eager_table(table, operator):
    if len(table.rows)>EAGER_ROW_LIMIT or len(table.rows)*len(table.columns)>EAGER_CELL_LIMIT:
        raise ValueError(f'{DEFINITIONS[operator]["title"]} currently requires an in-memory table within 100,000 rows and five million cells. Add Row Sample before this component. Streaming support for this operation is not available yet.')


def cast_value(value,dtype):
    if isinstance(value,dict):raise ValueError('Image references cannot be cast to scalar values; use an explicit image adapter.')
    if value is None:raise ValueError('Resolve missing values explicitly before casting')
    if dtype=='string':return str(value)
    if dtype=='bool':
        if type(value) is bool:return value
        if value in ('true','false'):return value=='true'
        raise ValueError('Boolean cast requires true or false')
    if type(value) is bool:raise ValueError('Boolean is not implicitly numeric')
    try:
        if dtype=='int64':
            if isinstance(value,str):
                if not value or value.strip()!=value or any(c not in '+-0123456789' for c in value):raise ValueError('Lossy integer cast')
                result=int(value)
            else:
                result=int(value)
                if result!=value:raise ValueError('Lossy integer cast')
            if not -(2**63)<=result<2**63:raise ValueError('Integer outside int64')
            return result
        result=float(value)
        if not math.isfinite(result):raise ValueError('Numeric values must be finite')
        if isinstance(value,int) and int(result)!=value:raise ValueError('Lossy float cast')
        return result
    except (OverflowError,TypeError) as e:raise ValueError('Invalid numeric cast') from e


def _partition_module():
    if __package__:
        from . import dataflow_partition
    else:
        import dataflow_partition
    return dataflow_partition


def _aggregate_module():
    if __package__:
        from . import dataflow_aggregate
    else:
        import dataflow_aggregate
    return dataflow_aggregate


def _join_module():
    if __package__:
        from . import dataflow_join
    else:
        import dataflow_join
    return dataflow_join


def _fitted_module():
    if __package__:
        from . import dataflow_fitted
    else:
        import dataflow_fitted
    return dataflow_fitted


def fit_scaler(table,columns,graph_digest):
    return _fitted_module().fit(table,'standardScalerFit',{'columns':columns},graph_digest)


def load_state(text):
    try:
        if json.loads(text).get('operator')==PREFIX+'textEncode':
            if __package__:from .text_encoding import load
            else:from text_encoding import load
            return load(text)
        if json.loads(text).get('operator')==PREFIX+'preprocess':
            if __package__:from .dataflow_preprocess import load
            else:from dataflow_preprocess import load
            return load(text)
        return _fitted_module().load(text)
    except (TypeError,KeyError,AttributeError,OverflowError) as error:raise ValueError('Malformed fitted state') from error


def transform(table,state,graph_digest):
    if state.get('operator')==PREFIX+'textEncode':
        if __package__:from .text_encoding import transform as apply
        else:from text_encoding import transform as apply
        return apply(table,state,graph_digest)
    if state.get('operator')==PREFIX+'preprocess':
        if __package__:from .dataflow_preprocess import transform as apply
        else:from dataflow_preprocess import transform as apply
        return apply(table,state,graph_digest)
    return _fitted_module().transform(table,state,graph_digest)


class OperatorFailure(ValueError):
    def __init__(self,code,cause,suggestion):
        super().__init__(cause);self.code=code;self.suggestion=suggestion


class DataFlowExecutionError(ValueError):
    def __init__(self,node,operator,args,cause,label=None):
        tables=[value for value in args.values() if isinstance(value,Table)]
        count=len(tables[0].rows) if len(tables)==1 else None
        title=DEFINITIONS[operator]['title']
        suggestion=getattr(cause,'suggestion','Inspect this block’s input columns and configuration, then correct the indicated condition and prepare again. No settings were changed automatically.')
        self.detail={'schemaVersion':1,'code':getattr(cause,'code','nodeExecution'),'nodeID':node,'nodeLabel':label,'operatorID':PREFIX+operator,'componentTitle':title,'inputRowCount':count,'cause':str(cause)[:500],'suggestion':suggestion}
        display=' '.join((label or title).split())
        rows=f', {count} input rows' if count is not None else ''
        super().__init__(f'{display} [{node}] ({title}{rows}): {self.detail["cause"]}. {suggestion}')


def validate_node_labels(labels,node_ids):
    if not isinstance(labels,dict) or len(labels)>128 or set(labels)-set(node_ids) or any(not isinstance(value,str) or not value.strip() or len(value)>200 for value in labels.values()):raise ValueError('Dataflow node labels require at most 128 known node IDs mapped to nonblank strings of at most 200 characters')


def _execute_node(op,c,args,graph_digest,source_path,check_cancelled,*,node_id=None,output_transaction=None,sampling=None,kernel=None):
    # Retired descriptors exist only for bounded migration and historical reading.
    if op in RETIRED_DEFINITIONS:raise ValueError('Retired data execution requires an explicit current-tool upgrade: '+op)
    if op=='union':
        if __package__:from .dataflow_union import union_tables
        else:from dataflow_union import union_tables
        return {'table':union_tables(args,c,check_cancelled,use_native=kernel=='union.nativeConcatOrReference.v1')}
    if op in ('sort','unique','reshapeTable','splitText'):
        module=_table_tool_module(op)
        if op=='sort':return {'table':module.sort_table(args['table'],c,check_cancelled,use_native=kernel=='sort.nativeOrderOrReference.v1')}
        if op=='unique':return module.unique_table(args['table'],c,check_cancelled)
        if op=='reshapeTable':return {'table':module.reshape_table(args['table'],c,check_cancelled)}
        return {'table':module.apply(args['table'],c,check_cancelled)}
    if op=='sequenceWindows':
        if __package__:from .sequence_windows import apply
        else:from sequence_windows import apply
        return apply(args,c,graph_digest,check_cancelled)
    if op=='textEncode':
        if __package__:from .text_encoding import apply
        else:from text_encoding import apply
        return apply(args,c,graph_digest,check_cancelled)
    table=args.get('table')
    if op=='gameRules':
        if __package__:from .game_rules_tables import apply as apply_rules
        else:from game_rules_tables import apply as apply_rules
        return apply_rules(c,check_cancelled)
    if op=='decide':
        if __package__:from .decision_tables import apply
        else:from decision_tables import apply
        return apply(table,c,check_cancelled,node_id)
    if op=='classifyAudio':
        if __package__:from .audio_classification_tables import apply
        else:from audio_classification_tables import apply
        return apply(table,c,check_cancelled,node_id)
    if op=='transcribeAudio':
        if __package__:from .speech_tables import apply
        else:from speech_tables import apply
        return apply(table,c,check_cancelled,node_id)
    if op=='extractText':
        if __package__:from .ocr_tables import apply
        else:from ocr_tables import apply
        return apply(table,c,check_cancelled,getattr(output_transaction,'image_reader',None))
    if op=='prepareMedia':
        if __package__:from .prepare_media import apply
        else:from prepare_media import apply
        return {'table':apply(table,c,check_cancelled,getattr(output_transaction,'image_reader',None))}
    if op=='preprocess':
        if __package__:from .dataflow_preprocess import apply
        else:from dataflow_preprocess import apply
        return apply(args,c,graph_digest,check_cancelled,node_id)
    if op=='formula':
        if __package__:from .dataflow_formula import apply
        else:from dataflow_formula import apply
        return {'table':apply(table,c,check_cancelled,share=kernel in ('formula.sharedColumns.v1','formula.nativeExpressionsOrReference.v2'),native_extended=kernel=='formula.nativeExpressionsOrReference.v2')}
    if op=='partition':
        if __package__:from .dataflow_partition_tool import apply
        else:from dataflow_partition_tool import apply
        return apply(table,c,check_cancelled)
    if op=='filterRows':
        if __package__:from .dataflow_filter import apply
        else:from dataflow_filter import apply
        return apply(table,c,check_cancelled,share=kernel=='filter.booleanSelectionOrReference.v1')
    if op=='columns':
        if __package__:from .dataflow_columns import apply
        else:from dataflow_columns import apply
        return {'table':apply(table,c,check_cancelled,share=kernel=='columns.sharedProjection.v1')}
    if op=='view':out={'table':table}
    elif op=='outputDelimited':
        try:output_transaction.stage(node_id,table,c)
        except OSError as error:raise ValueError('Output destination access failed; renew the write permission or choose another directory') from error
        out={'table':table}
    elif op=='sourceCSV':out={'table':source_csv(source_path,options=c,check_cancelled=check_cancelled,sampling=sampling)}
    elif op=='capturedTable':
        if __package__:
            from .dataflow_table_reader import read_table_artifact
            from .preparation_demand import sample_table
        else:
            from dataflow_table_reader import read_table_artifact
            from preparation_demand import sample_table
        loaded=read_table_artifact(source_path,c['sha256'],check_cancelled)
        out={'table':sample_table(derive(loaded,source_digest=c['sha256']),sampling,check_cancelled)}
    elif op=='sourceFiles':
        if __package__:from .file_input_source import source_files
        else:from file_input_source import source_files
        out={'table':source_files(source_path,c,check_cancelled,sampling=sampling)}
    elif op=='join':out={'table':_join_module().join_tables(args['left'],args['right'],c,use_native=kernel=='join.nativePairsOrReference.v1')}
    elif op=='aggregate':
        if kernel=='aggregate.exactStreaming.v1':
            if __package__:from .dataflow_aggregate_streaming import aggregate_table
            else:from dataflow_aggregate_streaming import aggregate_table
            out={'table':aggregate_table(args['table'],c,check_cancelled)}
        else:out={'table':_aggregate_module().aggregate_table(args['table'],c)}
    elif op=='sampleRows':
        if __package__:
            from .dataflow_sample import sample_rows
        else:
            from dataflow_sample import sample_rows
        out={'table':sample_rows(table,c['count'],c['seed'])}
    else:raise ValueError('Unsupported current data operator: '+op)
    return out


def _execute_graph(graph,source_bindings,on_node=None,materialization=None,node_labels=None,check_cancelled=lambda:None,*,retain_results=True,on_node_started=None,destination_bindings=None,on_output_saved=None,output_transaction=None,demand_plan=None,measurements=None,kernel_plan=None):
    """Return every output by default; callback consumers may release ports after their last use.

    This streams node lifetimes, not CSV rows. The cell bound includes live inputs and
    newly computed outputs; it does not measure Python/NumPy temporary allocations or RSS.
    """
    graph=upgrade_graph(graph)
    if type(retain_results) is not bool:raise ValueError('retain_results must be a boolean')
    check_cancelled()
    try:nodes,inputs,order=validate(graph)
    except (TypeError,KeyError,AttributeError) as error:raise ValueError('Malformed dataflow configuration') from error
    if __package__:from .preparation_demand import build_demand_plan,validate_demand_plan,sample_table
    else:from preparation_demand import build_demand_plan,validate_demand_plan,sample_table
    demand=build_demand_plan(graph) if demand_plan is None else validate_demand_plan(demand_plan,graph)
    if kernel_plan is not None:
        if __package__:from .preparation_kernels import validate_kernel_plan
        else:from preparation_kernels import validate_kernel_plan
        kernel_plan=validate_kernel_plan(kernel_plan,nodes)
    node_labels={} if node_labels is None else node_labels
    validate_node_labels(node_labels,nodes)
    result={};graph_digest=digest(graph);retained_cells=0;node_keys={}
    consumers={}
    for incoming in inputs.values():
        for endpoint in incoming.values():consumers[endpoint]=consumers.get(endpoint,0)+1
    def cells(value):return len(value.rows)*len(value.columns) if isinstance(value,Table) and not isinstance(value.rows,StoredRows) else 0
    def release(node,port):
        nonlocal retained_cells
        retained_cells-=cells(result[node].pop(port))
        if not result[node]:del result[node]
    if set(source_bindings)!={k for k,(op,_) in nodes.items() if op in ('sourceCSV','sourceFiles','capturedTable')}:raise ValueError('Source bindings must match source node IDs exactly')
    def phase(node,name):return measurements.phase(node,name) if measurements is not None else nullcontext()
    if __package__:from .preparation_scheduler import StableReadyQueue,maximum_native_tasks,can_overlap,run_ready_group
    else:from preparation_scheduler import StableReadyQueue,maximum_native_tasks,can_overlap,run_ready_group
    from contextlib import ExitStack
    maximum=maximum_native_tasks() if kernel_plan is not None else 1
    if maximum>1:
        import threading
        check_lock=threading.RLock();original_check=check_cancelled
        def check_cancelled():
            with check_lock:original_check()
    queue=StableReadyQueue(order,{node:{parent for parent,_ in incoming.values()} for node,incoming in inputs.items()})
    def eligible(key):
        if maximum==1:return False
        op,c=nodes[key]
        arguments={port:result[a][b] for port,(a,b) in inputs[key].items()}
        record=kernel_plan['nodes'][key]
        return record['parallel'] and can_overlap(op,c,arguments,record['kernelID'])
    prefetched={}
    def fused_region(key,args):
        if (maximum!=1 or not kernel_plan or kernel_plan['schemaVersion']!=4
            or (materialization is not None and materialization.policy!='off')
            or set(args)!={'table'} or not isinstance(args['table'],Table)):return None
        if __package__:from .native_frame_arena import current_arena
        else:from native_frame_arena import current_arena
        if current_arena() is None:return None
        steps=[];current=key
        while len(steps)<kernel_plan['fusion']['maximumNodesPerRegion']:
            op,config=nodes[current];choice=kernel_plan['nodes'][current]['kernelID']
            if (not demand.is_complete(current) or op not in ('formula','columns')
                or choice not in ('formula.nativeExpressionsOrReference.v2','columns.sharedProjection.v1')):break
            steps.append((current,op,config))
            if consumers.get((current,'table'))!=1:break
            following=[node for node,incoming in inputs.items() if incoming=={'table':(current,'table')}]
            if len(following)!=1:break
            current=following[0]
        if len(steps)<2:return None
        if __package__:from .native_query_fusion import try_fuse
        else:from native_query_fusion import try_fuse
        # Longest qualified prefix; a barrier never makes later nodes disappear.
        for end in range(len(steps),1,-1):
            outputs=try_fuse(args['table'],steps[:end],check_cancelled)
            if outputs is not None:return outputs
        return None

    while not queue.finished:
        group=queue.take(eligible,maximum)
        if not group:raise ValueError('Preparation ready queue made no progress.')
        with ExitStack() as lifetimes:
            tasks=[]
            for key in group:
                check_cancelled()
                node_lifetime=lifetimes.enter_context(ExitStack())
                if measurements is not None:node_lifetime.enter_context(measurements.node(key))
                with phase(key,"dispatch"):
                    if on_node_started:on_node_started(key)
                op,c=nodes[key];args={port:result[a][b] for port,(a,b) in inputs[key].items()}
                policy=demand.policy_for(key)
                coverage={'completeness':'complete' if demand.is_complete(key) else 'sampled','sampling':policy}
                if not demand.is_complete(key):
                    args={port:sample_table(value,policy,check_cancelled) if isinstance(value,Table) and demand.is_complete(inputs[key][port][0]) else value for port,value in args.items()}
                predecessors={port:node_keys.get(source) for port,(source,_) in inputs[key].items()}
                try:
                    with phase(key,"cacheKey"):
                        cache_key=materialization.key(key,op,c,args,predecessors,graph_digest,coverage=coverage) if materialization else None
                    with phase(key,"cacheLookup"):
                        out=materialization.lookup(key,cache_key,op,args,graph_digest,config=c) if materialization and op!='outputDelimited' else None
                except ValueError as error:raise DataFlowExecutionError(key,op,args,error,node_labels.get(key)) from error
                reused=out is not None
                if measurements is not None:measurements.cache_hit(key,reused)
                tasks.append({'key':key,'op':op,'config':c,'args':args,'policy':policy,'coverage':coverage,'predecessors':predecessors,'cacheKey':cache_key,'reused':reused,'out':out,'lifetime':node_lifetime})
            def compute(task,check):
                if task['reused']:return task['out']
                key=task['key']
                try:
                    with phase(key,"execution"):
                        if key in prefetched:return prefetched.pop(key)
                        fused=fused_region(key,task['args'])
                        if fused is not None:
                            output=fused.pop(key);prefetched.update(fused);return output
                        return _execute_node(task['op'],task['config'],task['args'],graph_digest,source_bindings.get(key),check,node_id=key,output_transaction=output_transaction,sampling=task['policy'],kernel=kernel_plan['nodes'][key]['kernelID'] if kernel_plan else None)
                except ValueError as error:raise DataFlowExecutionError(key,task['op'],task['args'],error,node_labels.get(key)) from error
            active=[task for task in tasks if not task['reused']]
            overlap=len(active)>1 and all(kernel_plan['nodes'][task['key']]['parallel'] and can_overlap(task['op'],task['config'],task['args'],kernel_plan['nodes'][task['key']]['kernelID']) for task in active)
            if overlap:
                computed=run_ready_group([(task['key'],lambda check,task=task:compute(task,check)) for task in active],check_cancelled)
                for task,out in zip(active,computed):task['out']=out
                computed=None
            else:
                for task in active:task['out']=compute(task,check_cancelled)
            for task in tasks:
                key=task['key'];op=task['op'];c=task['config'];args=task['args'];out=task['out'];cache_key=task['cacheKey']
                try:
                    retained_cells+=sum(cells(value) for value in out.values())
                    if retained_cells>5_000_000:raise ValueError('Dataflow exceeds five million retained table cells; reduce rows or branches')
                    if materialization:
                        with phase(key,"cacheKey"):
                            if op in ('sourceCSV','sourceFiles','capturedTable'):cache_key=materialization.key(key,op,c,args,task['predecessors'],graph_digest,source=out['table'],coverage=task['coverage'])
                        node_keys[key]=cache_key
                        materialization.record(key,cache_key,task['reused'],source=op in ('sourceCSV','sourceFiles','capturedTable'))
                except ValueError as error:raise DataFlowExecutionError(key,op,args,error,node_labels.get(key)) from error
                check_cancelled();result[key]=out if retain_results else dict(out)
                with phase(key,"observation"):
                    if on_node:on_node(key,out)
                check_cancelled()
                with phase(key,"release"):
                    if not retain_results:
                        for endpoint in inputs[key].values():
                            consumers[endpoint]-=1
                            if consumers[endpoint]==0:release(*endpoint)
                        for port in tuple(out):
                            if consumers.get((key,port),0)==0:release(key,port)
                        args.clear();task['out']=None;out=None
                queue.complete(key)
                task['lifetime'].close()
            tasks.clear()
    return result


def _execute_owned(graph,source_bindings,on_node=None,materialization=None,node_labels=None,check_cancelled=lambda:None,*,retain_results=True,on_node_started=None,destination_bindings=None,on_output_saved=None,output_transaction=None,demand_plan=None,measurements=None,kernel_plan=None):
    """Execute a graph with explicit, deferred output destinations; cached sinks never write."""
    if __package__:from .dataflow_output import OutputTransaction
    else:from dataflow_output import OutputTransaction
    check_cancelled()
    try:nodes,_,_=validate(graph)
    except (TypeError,KeyError,AttributeError) as error:raise ValueError('Malformed dataflow configuration') from error
    destinations={} if destination_bindings is None else destination_bindings
    sinks={node for node,(op,c) in nodes.items() if op=='outputDelimited' and c.get('destination','file')=='file'}
    if not isinstance(destinations,dict) or set(destinations)!=sinks:raise ValueError('Destination bindings must match output node IDs exactly')
    owned=output_transaction is None
    transaction=OutputTransaction(destinations,check_cancelled) if owned else output_transaction
    try:
        result=_execute_graph(graph,source_bindings,on_node,materialization,node_labels,check_cancelled,retain_results=retain_results,on_node_started=on_node_started,output_transaction=transaction,demand_plan=demand_plan,measurements=measurements,kernel_plan=kernel_plan)
        if owned:
            for receipt in transaction.commit():
                if on_output_saved:on_output_saved(receipt)
        return result
    except ValueError as error:
        if owned:transaction.rollback()
        if hasattr(error,'node'):raise DataFlowExecutionError(error.node,'outputDelimited',{},error,(node_labels or {}).get(error.node)) from error
        raise
    except BaseException:
        if owned:transaction.rollback()
        raise
    finally:
        if owned:transaction.close()


def selected_engine(graph,engine=None):
    if engine is not None:return engine
    nodes=graph.get('nodes',[]) if isinstance(graph,dict) else []
    if not isinstance(nodes,list):return 'disk'
    return 'polars' if any(isinstance(node,dict) and isinstance(node.get('config'),dict) and (node['config'].get('parserProfile')=='polars' or node['config'].get('format') in ('parquet','json','jsonl','excel','markdown','xml','html','database','api')) for node in nodes) else 'disk'


def execute(graph,source_bindings,*args,engine=None,**kwargs):
    if __package__:from .polars_preparation import preparation_engine
    else:from polars_preparation import preparation_engine
    active_engine=selected_engine(graph,engine)
    if __package__:from .native_frame_arena import native_buffer_scope
    else:from native_frame_arena import native_buffer_scope
    captured=kwargs.get('kernel_plan')
    resident=(active_engine=='polars' and isinstance(captured,dict) and captured.get('schemaVersion') in (3,4))
    if __package__:from .table_identity import deferred_identity_scope
    else:from table_identity import deferred_identity_scope
    deferred=resident and captured.get('schemaVersion')==4
    vm_grain=1000
    if deferred:
        boundary=captured.get('workBoundaries')
        if not isinstance(boundary,dict) or type(boundary.get('sqliteInstructions')) is not int or boundary['sqliteInstructions']!=32768:
            raise ValueError('Unsupported captured native VM work boundary.')
        vm_grain=boundary['sqliteInstructions']
    with preparation_engine(active_engine), disk_scope(), (native_buffer_scope(sqlite_instructions=vm_grain) if resident else nullcontext()), (deferred_identity_scope() if deferred else nullcontext()):
        return _execute_owned(graph,source_bindings,*args,**kwargs)
