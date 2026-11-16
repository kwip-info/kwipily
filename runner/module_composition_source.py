"""Independent bounded source expansion; never constructs or executes a module."""
import copy
import hashlib
import json
import math
from pathlib import Path
import stat
if __package__:
    from .graph_semantics import effective_config, semantic_graph
    from .checkpoint_lifecycle import digest
    from .module_composition_policy import validate_policy, _KEY, _RESERVED, nn
    from .module_composition_plan import container_id, validate_plan
else:
    from graph_semantics import effective_config, semantic_graph
    from checkpoint_lifecycle import digest
    from module_composition_policy import validate_policy, _KEY, _RESERVED, nn
    from module_composition_plan import container_id, validate_plan

MAX_BYTES=2_000_000

def _config(node,version):
    try:return effective_config(node,version)
    except (KeyError,TypeError,OverflowError) as e:raise ValueError('Malformed or unsupported source primitive configuration.') from e

def _fail(message): raise ValueError(message)
def _exact(value,required,optional=()):
    if type(value) is not dict or not set(required)<=set(value)<=set(required)|set(optional):_fail('Composition source has missing or unknown fields.')

def _name(value):
    if type(value) is not str or not 1<=len(value)<=128 or any(0xD800<=ord(c)<=0xDFFF for c in value):_fail('Composition source requires bounded Unicode identities.')
    return value

def _list(value,maximum,minimum=0):
    if type(value) is not list or not minimum<=len(value)<=maximum:_fail('Composition source collection exceeds its bounds.')
    return value

def _finite(value):
    try:return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:return False

def _integer(value):
    return _finite(value) and -(2**63)<=value<2**63 and int(value)==value

def _ref(value):
    _exact(value,('id','version'));_name(value['id']);v=value['version'];_exact(v,('major','minor','patch'))
    if any(type(x) is not int or not 0<=x<=2**31-1 for x in v.values()):_fail('Invalid pinned definition revision.')
    return value['id'],v['major'],v['minor'],v['patch']

def _target(value,field=False):
    key='field' if field else 'portID';_exact(value,('componentID',key))
    return _name(value['componentID']),_name(value[key])

def _accepts(field,value):
    kind=field.get('kind','integer')
    if value is None:return field.get('allowsNull',False) is True and kind in ('integer','number','integerList','numberList')
    if kind=='boolean':return type(value) is bool
    if kind=='choice':return type(value) is str and value in [c['value'] for c in field['choices']]
    if kind=='text':return type(value) is str and field['minItems']<=len(value)<=field['maxItems']
    if kind in ('integerList','numberList'):
        check=_integer if kind=='integerList' else _finite
        return type(value) is list and field['minItems']<=len(value)<=field['maxItems'] and all(check(v) and field['min']<=v<=field['max'] for v in value)
    if kind in ('integer','number'):
        return (_integer(value) if kind=='integer' else _finite(value)) and field['min']<=value<=field['max']
    return False

def _field(field):
    _exact(field,('key','title','defaultValue'),('kind','min','max','minItems','maxItems','allowsNull','choices','enabledValue'))
    _name(field['key'])
    if type(field['title']) is not str:_fail('Field title must be text.')
    kind=field.get('kind','integer');nullable=field.get('allowsNull',False)
    if nullable is None:nullable=False
    if type(nullable) is not bool or nullable and kind not in ('integer','number','integerList','numberList'):_fail('Invalid optional field declaration.')
    if kind!='choice' and field.get('choices') is not None:_fail('Only choice fields declare choices.')
    if kind=='choice':
        choices=_list(field.get('choices'),64,1)
        for c in choices:
            _exact(c,('value','title'))
            if any(type(c[k]) is not str or not c[k].strip() or len(c[k])>128 for k in c):_fail('Invalid field choice.')
        if len({c['value'] for c in choices})!=len(choices):_fail('Duplicate field choice.')
    if kind in ('choice','boolean','text') and any(field.get(k) is not None for k in ('min','max')):_fail('Unexpected numeric field bounds.')
    if kind in ('choice','boolean','integer','number') and any(field.get(k) is not None for k in ('minItems','maxItems')):_fail('Unexpected field item bounds.')
    if kind in ('integer','number','integerList','numberList'):
        check=_integer if kind in ('integer','integerList') else _finite
        if not check(field.get('min')) or not check(field.get('max')) or field['min']>field['max']:_fail('Invalid field numeric bounds.')
    if kind in ('integerList','numberList','text'):
        low,high=field.get('minItems'),field.get('maxItems')
        if type(low) is not int or type(high) is not int or not (1 if kind=='text' else 0)<=low<=high<=4096:_fail('Invalid field item bounds.')
    if kind=='text' and nullable:_fail('Text fields cannot be null.')
    if not _accepts(field,field['defaultValue']):_fail('Invalid field default.')
    if field.get('enabledValue') is not None and (not nullable or not _accepts(field,field['enabledValue'])):_fail('Invalid enabled field value.')
    return field

def _can_expose(field,target):
    if field.get('kind','integer')!=target.get('kind','integer') or field.get('allowsNull',False) and not target.get('allowsNull',False):return False
    kind=field.get('kind','integer')
    if kind=='boolean':return True
    if kind=='choice':return {c['value'] for c in field['choices']}<={c['value'] for c in target['choices']}
    if kind in ('text','integerList','numberList') and not target['minItems']<=field['minItems']<=field['maxItems']<=target['maxItems']:return False
    return kind=='text' or target['min']<=field['min']<=field['max']<=target['max']

def _registry(path):
    if not isinstance(path,(str,Path)):_fail('Native composition source requires its captured registry path.')
    path=Path(path);info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_BYTES:_fail('Captured registry must be a bounded regular file.')
    value=json.loads(path.read_bytes());_exact(value,('schemaVersion','components'))
    if type(value['schemaVersion']) is not int or value['schemaVersion']!=2:_fail('Unsupported captured registry.')
    definitions={}
    for d in _list(value['components'],1024,1):
        if type(d) is not dict:_fail('Malformed captured registry component.')
        ident=_name(d.get('id'))
        if ident in definitions:_fail('Duplicate registry component.')
        definitions[ident]=d
    return definitions

def _scoped(kind,path,ident):
    if not path:return ident
    # JSONEncoder's legacy default escapes forward slashes but emits UTF8 Unicode.
    wire=json.dumps([kind,*path,ident],ensure_ascii=False,separators=(',',':')).replace('/','\\/').encode('utf8')
    return kind+'-'+hashlib.sha256(wire).hexdigest()

def reexpand(document, registry, *, objective_context=None):
    """Reconstruct source3 using exact captured registry fields and native-free math."""
    _exact(document,('schemaVersion','name','root','definitions'),('execution','namedInputs'))
    if type(document['schemaVersion']) is not int or document['schemaVersion']!=3 or type(document['name']) is not str:_fail('Composition expansion requires source schema3.')
    try:raw=json.dumps(document,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
    except (ValueError,TypeError,UnicodeError) as e:raise ValueError('Invalid finite composition source.') from e
    if len(raw)>MAX_BYTES:_fail('Composition source exceeds 2 MB.')
    if document.get('execution') is not None:
        execution=document['execution']
        allowed={'device','epochs','datasetCSV','featureColumns','targetColumn','batchSize','sampleLimit','seed','splitSeed','validationPercent','learningRate','preparedTraining','compositeSource','gradientCapture','trainingControl'}
        if type(execution) is not dict or not set(execution)<=allowed:_fail('Unknown captured execution metadata.')
        if 'trainingControl' in execution:
            if __package__:from .training_control import validate_policy as validate_training_control
            else:from training_control import validate_policy as validate_training_control
            validate_training_control(execution['trainingControl'])
    declared=_registry(registry);definitions={}
    for definition in _list(document['definitions'],64):
        if 'appearance' in definition and definition['appearance'] not in ('parallel_features','separable_conv2d','transformer'): raise ValueError('Unknown container appearance identity.')
        _exact(definition,('reference','title','body','inputs','outputs','parameters'),('integerProducts','composition','appearance'))
        ref=_ref(definition['reference'])
        if ref in definitions:_fail('Duplicate pinned definition revision.')
        if type(definition['title']) is not str:_fail('Definition title must be text.')
        definitions[ref]=definition
        seen=set()
        for parameter in _list(definition['parameters'],128):
            _exact(parameter,('field','targets'));field=_field(parameter['field'])
            if field['key'] in seen:_fail('Duplicate exposed field.')
            seen.add(field['key'])
            for target in _list(parameter['targets'],256,1):_target(target,True)
        if sum(len(p['targets']) for p in definition['parameters'])>256:_fail('Parameter target budget exceeded.')
        for i in _list(definition['inputs'],128):
            _exact(i,('id','targets'));_name(i['id'])
            for t in _list(i['targets'],256,1):_target(t)
        for o in _list(definition['outputs'],128,1):_exact(o,('id','source'));_name(o['id']);_target(o['source'])
        if len({i['id'] for i in definition['inputs']})!=len(definition['inputs']) or len({o['id'] for o in definition['outputs']})!=len(definition['outputs']):_fail('Duplicate Block boundary.')
        if 'composition' in definition and definition['composition'] is None:_fail('Explicit null composition is refused.')
    def body(value,inside):
        _exact(value,('nodes','instances','edges'),('configurationVersion',))
        version=value.get('configurationVersion',1)
        if version is None:version=1
        if type(version) is not int or version not in (1,2):_fail('Invalid source configuration version.')
        nodes=_list(value['nodes'],128);instances=_list(value['instances'],128);edges=_list(value['edges'],256)
        if len(nodes)+len(instances)>128:_fail('Block child limit exceeded.')
        ids=[]
        for n in nodes:
            _exact(n,('id','type','name','x','y','config'));ids.append(_name(n['id']));_name(n['type'])
            if n['type'] not in declared or inside and n['type'] in ('input','index_input','bag_input','output'):_fail('Unknown or internal source sentinel primitive.')
            if type(n['name']) is not str or not all(_finite(n[k]) for k in ('x','y')) or type(n['config']) is not dict or len(n['config'])>128:_fail('Invalid primitive metadata.')
            _config(n,version)
        for i in instances:
            _exact(i,('id','definition','name','x','y','overrides'));ids.append(_name(i['id']))
            if _ref(i['definition']) not in definitions:_fail('Missing pinned nested definition.')
            if type(i['name']) is not str or not all(_finite(i[k]) for k in ('x','y')) or type(i['overrides']) is not dict or len(i['overrides'])>128:_fail('Invalid instance metadata.')
        if len(set(ids))!=len(ids):_fail('Duplicate local child identity.')
        edge_ids=set()
        for e in edges:
            _exact(e,('id','source','sourcePort','target','targetPort'))
            for v in e.values():_name(v)
            if e['id'] in edge_ids:_fail('Duplicate local wire identity.')
            edge_ids.add(e['id'])
        return version
    body(document['root'],False)
    for d in definitions.values():
        body(d['body'],True)
        if 'composition' in d:
            policy=d['composition'];_exact(policy,('schemaVersion','kind','children'))
            if type(policy['schemaVersion']) is not int or policy['schemaVersion']!=1 or policy['kind'] not in ('graph','sequential','orderedList','namedMap'):_fail('Invalid authored composition policy.')
            children=_list(policy['children'],128,1);ids=[]
            for c in children:
                _exact(c,('localID','key') if policy['kind']=='namedMap' else ('localID',));ids.append(_name(c['localID']))
                if policy['kind']=='namedMap':
                    key=c['key']
                    if type(key) is not str or _KEY.fullmatch(key) is None or key in _RESERVED or hasattr(nn.Module,key) or hasattr(nn.ModuleDict,key):_fail('Named child requires a native-safe key.')
            if policy['kind']=='namedMap' and len({c['key'] for c in children})!=len(children):_fail('Duplicate named child key.')
            if len(set(ids))!=len(ids) or set(ids)!={n['id'] for n in d['body']['nodes']}|{i['id'] for i in d['body']['instances']}:_fail('Policy must list every actual child once.')

    completed=set()
    def cycles(ref,active,depth):
        if depth>8 or ref in active:_fail('Recursive or over-depth Block definition.')
        if (ref,depth) in completed:return
        for instance in definitions[ref]['body']['instances']:cycles(_ref(instance['definition']),active|{ref},depth+1)
        completed.add((ref,depth))
    for ref in definitions:cycles(ref,set(),1)
    nodes=[];edges=[];origins={};consumed=[];containers={};policy_checks=[];occupied=set();edge_ids=set();selected=False
    reached=set()
    def reachable(ref):
        nonlocal selected
        if ref in reached:return
        reached.add(ref)
        d=definitions[ref]
        if 'composition' in d:selected=True
        for i in d['body']['instances']:reachable(_ref(i['definition']))
    for i in document['root']['instances']:reachable(_ref(i['definition']))
    def resolved(d,overrides):
        value=copy.deepcopy(d['body']);version=value.get('configurationVersion') or 1
        for n in value['nodes']:n['config']=_config(n,version)
        value['configurationVersion']=2
        fields={p['field']['key']:p['field'] for p in d['parameters']}
        if not set(overrides)<=set(fields):_fail('Undeclared Block override.')
        assigned=set();values={}
        def destination(address):
            ident,key=address
            n=next((n for n in value['nodes'] if n['id']==ident),None)
            if n is not None:
                f=next((f for f in declared[n['type']]['fields'] if f['key']==key),None)
            else:
                i=next((i for i in value['instances'] if i['id']==ident),None)
                nested=definitions[_ref(i['definition'])] if i else None
                f=next((p['field'] for p in nested['parameters'] if p['field']['key']==key),None) if nested else None
            if f is None:_fail('Configuration target field does not exist.')
            return _field(f)
        def assign(address,v):
            if address in assigned:_fail('Duplicate direct or product configuration target.')
            assigned.add(address);ident,key=address
            n=next((n for n in value['nodes'] if n['id']==ident),None)
            if n is not None:n['config'][key]=v
            else:next(i for i in value['instances'] if i['id']==ident)['overrides'][key]=v
        for p in d['parameters']:
            f=p['field'];v=overrides.get(f['key'],f['defaultValue']);values[f['key']]=v
            if not _accepts(f,v):_fail('Block override is outside its declared field bounds.')
            for t in p['targets']:
                address=_target(t,True);target=destination(address)
                if not _can_expose(f,target) or not _accepts(target,v):_fail('Exposed field range or value is incompatible with its target.')
                assign(address,v)
        for product in _list([] if d.get('integerProducts') is None else d['integerProducts'],128):
            _exact(product,('lhs','rhs','targets'));operands=[]
            for name in ('lhs','rhs'):
                f=fields.get(product[name])
                if f is None or f.get('kind','integer')!='integer' or f.get('allowsNull',False):_fail('Integer products require declared nonoptional integer fields.')
                operands.append(values[product[name]])
            v=int(operands[0])*int(operands[1])
            if not -(2**63)<=v<2**63:_fail('Integer product exceeds Int64.')
            for t in _list(product['targets'],256,1):
                address=_target(t,True);target=destination(address)
                if target.get('kind','integer')!='integer' or target.get('allowsNull',False) or not _accepts(target,v):_fail('Integer product target is not an admitted integer field.')
                assign(address,v)
        return value
    def expand(value,path,definition=None,x=0,y=0):
        if len(path)>8:_fail('Source expansion exceeds depth8.')
        version=value.get('configurationVersion') or 1
        primitive={};nested={}
        for n in value['nodes']:
            flat=copy.deepcopy(n);flat['id']=_scoped('n',path,n['id']);flat['config']=_config(n,version);flat['x']+=x;flat['y']+=y
            if not _finite(flat['x']) or not _finite(flat['y']) or flat['id'] in origins:_fail('Expanded node identity or position is invalid.')
            primitive[n['id']]=flat['id'];nodes.append(flat)
            origin=dict(instancePath=list(path),localNodeID=n['id'])
            if definition:origin['definition']=copy.deepcopy(definition['reference'])
            origins[flat['id']]=origin
            if len(nodes)>128:_fail('Expanded model exceeds128nodes.')
        for i in value['instances']:
            d=definitions[_ref(i['definition'])]
            if d['reference'] not in consumed:consumed.append(copy.deepcopy(d['reference']))
            nested[i['id']]=expand(resolved(d,i['overrides']),[*path,i['id']],d,x+i['x'],y+i['y'])
        def endpoint(address,output=False):
            ident,port=address
            if ident in primitive:
                n=next(n for n in value['nodes'] if n['id']==ident)
                allowed=declared[n['type']]['outputs' if output else 'inputs']
                if port in allowed:return (primitive[ident],port) if output else [(primitive[ident],port)]
            elif ident in nested and port in nested[ident][1 if output else 0]:return nested[ident][1 if output else 0][port]
            _fail('Source boundary or connection names a missing child port.')
        ins={};outs={};bound=set()
        if definition:
            for i in definition['inputs']:
                targets=[]
                for t in i['targets']:
                    for target in endpoint(_target(t)):
                        if target in bound or target in occupied:_fail('Ambiguous or multiply bound Block input.')
                        bound.add(target);targets.append(target)
                ins[i['id']]=targets
            for o in definition['outputs']:outs[o['id']]=endpoint(_target(o['source']),True)
        for e in value['edges']:
            source=endpoint((e['source'],e['sourcePort']),True)
            for index,target in enumerate(endpoint((e['target'],e['targetPort']))):
                if target in occupied or target in bound:_fail('A child input has multiple internal or boundary sources.')
                occupied.add(target)
                edge_id=_scoped('e',path if index==0 else [*path,'$fanout',str(index)],e['id'])
                if edge_id in edge_ids:_fail('Expanded wire identity collision.')
                edge_ids.add(edge_id);edges.append(dict(id=edge_id,source=source[0],sourcePort=source[1],target=target[0],targetPort=target[1]))
                if len(edges)>256:_fail('Expanded model exceeds256wires.')
        if selected:
            policy=definition.get('composition') if definition else None
            ordered=policy['children'] if policy else [dict(localID=n['id']) for n in value['nodes'] if n['type'] not in ('input','index_input','bag_input','output')]+[dict(localID=i['id']) for i in value['instances']]
            children=[]
            for index,c in enumerate(ordered):
                ident=c['localID'];key=c.get('key') if policy and policy.get('kind')=='namedMap' else str(index)
                if ident in primitive:children.append(dict(kind='node',id=primitive[ident],key=key))
                elif ident in nested:children.append(dict(kind='container',id=container_id([*path,ident]),key=key))
                else:_fail('Composition policy has a foreign child.')
            ep=lambda t:dict(node=t[0],port=t[1])
            if definition:
                pi=[dict(id=i['id'],targets=[ep(t) for t in ins[i['id']]]) for i in definition['inputs']]
                po=[dict(id=o['id'],source=ep(outs[o['id']])) for o in definition['outputs']]
            else:
                pi=[dict(id=n['id'],targets=[dict(node=e['target'],port=e['targetPort']) for e in edges if e['source']==n['id']]) for n in value['nodes'] if n['type'] in ('input','index_input','bag_input')]
                po=[]
                for n in value['nodes']:
                    if n['type']=='output':
                        es=[e for e in edges if e['target']==n['id']]
                        if len(es)!=1:_fail('Root output needs exactly one source.')
                        po.append(dict(id=n['id'],source=dict(node=es[0]['source'],port=es[0]['sourcePort'])))
            containers[container_id(path)]=dict(id=container_id(path),kind=policy['kind'] if policy else 'graph',path=list(path),children=children,inputs=pi,outputs=po)
            if policy:policy_checks.append((policy,value,definition,primitive))
        return ins,outs
    # Validate retained defaults, targets and unused nested overrides as typed
    # authoring data too; only actual reachable instances become native owners.
    for d in definitions.values():resolved(d,{})
    for value in [document['root'],*[d['body'] for d in definitions.values()]]:
        for instance in value['instances']:resolved(definitions[_ref(instance['definition'])],instance['overrides'])
    expand(document['root'],[])
    graph=dict(schemaVersion=2,name=document['name'],nodes=nodes,edges=edges)
    if document.get('execution') is not None:graph['execution']=copy.deepcopy(document['execution'])
    if document.get('namedInputs') is not None:graph['namedInputs']=copy.deepcopy(document['namedInputs'])
    if selected:
        preorder=[];seen=set()
        def append(ident):
            if ident not in containers or ident in seen:_fail('Native container ownership is missing or aliased.')
            seen.add(ident);row=containers[ident];preorder.append(row)
            for c in row['children']:
                if c['kind']=='container':append(c['id'])
        append('root');graph['compositionPlan']=dict(schemaVersion=1,root='root',containers=preorder)
    # Imports occur only after all source data/recursion/budget checks. Validation
    # derives actual configured layouts and allocates no native parameters.
    if __package__:from .engine import validate
    else:from engine import validate
    order,_,layouts,_=validate(graph,registry,include_types=True,bundled_dataset=False,objective_context=objective_context,training_required=False)
    ports={}
    for n in nodes:
        inputs=[e['targetPort'] for e in edges if e['target']==n['id']]
        outputs=list(layouts[n['id']]) if type(layouts[n['id']]) is dict else ([] if n['type']=='output' else ['out'])
        ports[n['id']]=dict(inputs=inputs,outputs=outputs)
    for policy,value,d,primitive in policy_checks:
        local={name:ports[ident] for name,ident in primitive.items()}
        for i in value['instances']:
            child=definitions[_ref(i['definition'])];local[i['id']]=dict(inputs=[p['id'] for p in child['inputs']],outputs=[p['id'] for p in child['outputs']])
        validate_policy(policy,local,value['edges'],d['inputs'],d['outputs'])
    if selected:validate_plan(graph['compositionPlan'],graph,ports)
    return dict(expandedGraph=graph,nodeOrigins=origins,consumedDefinitions=consumed)

def checked_source_v2(raw,sha,graph,registry,*,objective_context=None):
    def pairs(entries):
        result={}
        for k,v in entries:
            if k in result:_fail('Duplicate composition source JSON field.')
            result[k]=v
        return result
    try:source=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:_fail('Nonfinite source JSON.'))
    except (UnicodeError,TypeError,json.JSONDecodeError) as e:raise ValueError('Malformed composition source JSON.') from e
    _exact(source,('nativeExpansionContractVersion','sourceCompositeDocument','expandedGraph','nodeOrigins','consumedDefinitions'))
    if type(source['nativeExpansionContractVersion']) is not int or source['nativeExpansionContractVersion']!=2:_fail('Composition source requires expansion version2.')
    actual=reexpand(source['sourceCompositeDocument'],registry,objective_context=objective_context)
    if any(actual[k]!=source[k] for k in actual):_fail('Captured composition graph, plan, origins or consumed pins differ from independent source expansion.')
    normalized=semantic_graph(actual['expandedGraph'])
    if normalized!=semantic_graph(graph):_fail('Composition source architecture or plan is stale.')
    return dict(path='composite/source.json',sha256=sha,bytes=len(raw),nativeExpansionContractVersion=2,expandedSemanticDigest=digest(normalized),consumedDefinitions=actual['consumedDefinitions'])
