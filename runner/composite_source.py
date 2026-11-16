"""Bounded non-executable native composite source provenance for an expanded graph."""
import hashlib
import json
import math
from pathlib import Path
import stat
if __package__:
    from .graph_semantics import effective_config
    from .checkpoint_lifecycle import semantic_graph,digest
else:
    from graph_semantics import effective_config
    from checkpoint_lifecycle import semantic_graph,digest

MAX_BYTES=2_000_000

def checked_source(raw,expected_sha256,graph,registry=None,*,objective_context=None):
    if not isinstance(raw,bytes) or len(raw)>MAX_BYTES:raise ValueError('Composite source exceeds 2 MB bound')
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(c not in '0123456789abcdefABCDEF' for c in expected_sha256):raise ValueError('Composite source requires a 64-character hexadecimal digest')
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=expected_sha256.lower():raise ValueError('Composite source hash/digest changed after selection')
    def reject_constant(value):raise ValueError('Composite source JSON requires finite values')
    source=json.loads(raw,parse_constant=reject_constant)
    if isinstance(source,dict) and type(source.get('nativeExpansionContractVersion')) is int and source['nativeExpansionContractVersion']==2:
        if __package__:
            from .module_composition_source import checked_source_v2
        else:
            from module_composition_source import checked_source_v2
        return checked_source_v2(raw,sha,graph,registry,objective_context=objective_context)
    keys={'nativeExpansionContractVersion','sourceCompositeDocument','expandedGraph','nodeOrigins','consumedDefinitions'}
    if not isinstance(source,dict) or set(source)!=keys or type(source['nativeExpansionContractVersion']) is not int or source['nativeExpansionContractVersion']!=1:raise ValueError('Unsupported composite source envelope')
    document=source['sourceCompositeDocument']
    if not isinstance(document,dict) or type(document.get('schemaVersion')) is not int or document['schemaVersion'] not in (1,2) or not isinstance(document.get('root'),dict) or not isinstance(document.get('definitions'),list) or len(document['definitions'])>64:raise ValueError('Invalid composite source document')
    if 'compositionPlan' in graph or isinstance(source.get('expandedGraph'),dict) and 'compositionPlan' in source['expandedGraph'] or any(isinstance(d,dict) and 'composition' in d for d in document['definitions']):
        raise ValueError('Native composition cannot use the legacy source expansion contract')
    validate_bodies(document)
    expanded=source['expandedGraph']
    if document.get('namedInputs')!=expanded.get('namedInputs'):raise ValueError('Composite named input declarations differ from the expanded model.')
    if not isinstance(expanded,dict) or type(expanded.get('schemaVersion')) is not int or expanded.get('schemaVersion') not in (1,2) or not isinstance(expanded.get('nodes'),list) or not 1<=len(expanded['nodes'])<=128 or not isinstance(expanded.get('edges'),list) or len(expanded['edges'])>256:raise ValueError('Invalid expanded architecture')
    try:
        normalized=semantic_graph(expanded)
        if normalized!=semantic_graph(graph):raise ValueError('Composite source architecture is stale; adopt the edited composite again')
        ids=[n['id'] for n in expanded['nodes']]
    except (KeyError,TypeError) as error:raise ValueError('Malformed expanded architecture') from error
    origins=source['nodeOrigins']
    if len(set(ids))!=len(ids) or not isinstance(origins,dict) or set(origins)!=set(ids):raise ValueError('Composite node origin map must cover every expanded node exactly')
    def name(value):return isinstance(value,str) and 1<=len(value)<=128
    def reference(value):
        return isinstance(value,dict) and set(value)=={'id','version'} and name(value['id']) and isinstance(value['version'],dict) and set(value['version'])=={'major','minor','patch'} and all(type(v) is int and 0<=v<=2**31-1 for v in value['version'].values())
    for origin in origins.values():
        if not isinstance(origin,dict) or set(origin)-{'instancePath','localNodeID','definition'} or not name(origin.get('localNodeID')) or not isinstance(origin.get('instancePath'),list) or len(origin['instancePath'])>8 or not all(name(v) for v in origin['instancePath']) or (origin.get('definition') is not None and not reference(origin['definition'])):raise ValueError('Invalid composite node origin')
    pins=source['consumedDefinitions']
    if not isinstance(pins,list) or len(pins)>64 or not all(reference(pin) for pin in pins) or len({json.dumps(p,sort_keys=True) for p in pins})!=len(pins):raise ValueError('Invalid consumed composite definition pins')
    return {'path':'composite/source.json','sha256':sha,'bytes':len(raw),'nativeExpansionContractVersion':1,'expandedSemanticDigest':digest(normalized),'consumedDefinitions':pins}

def load_source(path,expected_sha256,graph,registry=None,*,objective_context=None):
    path=Path(path);info=path.lstat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('Composite source must be a regular file, not a symbolic link')
    if info.st_size>MAX_BYTES:raise ValueError('Composite source exceeds 2 MB bound')
    with path.open('rb') as stream:raw=stream.read(MAX_BYTES+1)
    checked_source(raw,expected_sha256,graph,registry,objective_context=objective_context)
    return (raw,expected_sha256)


def validate_bodies(document):
    """Check typed source data; native expander remains owner of expansion semantics."""
    definitions={}
    for definition in document['definitions']:
        if not isinstance(definition,dict) or not isinstance(definition.get('reference'),dict):raise ValueError('Invalid composite definition')
        key=json.dumps(definition['reference'],sort_keys=True)
        if key in definitions:raise ValueError('Duplicate composite definition')
        definitions[key]=definition
        parameters=definition.get('parameters',[])
        if not isinstance(parameters,list):raise ValueError('Invalid composite parameter declaration')
        declared=set()
        for parameter in parameters:
            field=parameter.get('field') if isinstance(parameter,dict) else None
            validate_field(field)
            if field['key'] in declared:raise ValueError('Duplicate composite parameter declaration')
            declared.add(field['key'])
    for definition in definitions.values():
        validate_integer_products(definition,definitions,{})
    for body in [document['root']]+[d.get('body') for d in definitions.values()]:
        if not isinstance(body,dict):raise ValueError('Invalid composite body')
        version=body.get('configurationVersion',1)
        if type(version) is not int or version not in (1,2):raise ValueError('Unsupported composite configuration version')
        nodes=body.get('nodes');instances=body.get('instances')
        if not isinstance(nodes,list) or len(nodes)>128 or not isinstance(instances,list) or len(instances)>128:raise ValueError('Invalid composite body components')
        for node in nodes:
            if not isinstance(node,dict):raise ValueError('Invalid composite primitive')
            effective_config(node,version)
        for instance in instances:
            if not isinstance(instance,dict) or not isinstance(instance.get('overrides'),dict):raise ValueError('Invalid composite overrides')
            definition=definitions.get(json.dumps(instance.get('definition'),sort_keys=True))
            if definition is None:raise ValueError('Missing pinned composite definition')
            parameters=definition.get('parameters',[])
            if not isinstance(parameters,list):raise ValueError('Invalid composite parameters')
            fields={}
            for parameter in parameters:
                field=parameter.get('field') if isinstance(parameter,dict) else None
                if not isinstance(field,dict) or not isinstance(field.get('key'),str) or field['key'] in fields:raise ValueError('Invalid composite parameter field')
                fields[field['key']]=field
            if not set(instance['overrides']).issubset(fields):raise ValueError('Undeclared composite override')
            validate_integer_products(definition,definitions,instance['overrides'])
            for key,field in fields.items():
                value=instance['overrides'].get(key,field.get('defaultValue'))
                if not accepts_field_value(field,value):
                    raise ValueError(f"Composite {field.get('kind','integer')} override is outside its declared contract")


def finite_number(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value)
    except OverflowError:return False


def native_integer(value):
    return finite_number(value) and -(2**63)<=value<2**63 and int(value)==value


def accepts_field_value(field,value):
    """Value check shared by defaults and overrides; declaration validated first."""
    kind=field.get('kind','integer')
    if value is None:return field.get('allowsNull') is True and kind in ('integer','number')
    if kind=='boolean':return type(value) is bool
    if kind=='choice':return type(value) is str and value in [c['value'] for c in field['choices']]
    if kind=='integerList':
        return (type(value) is list and field['minItems']<=len(value)<=field['maxItems']
                and all(native_integer(v) and field['min']<=v<=field['max'] for v in value))
    if kind in ('integer','number'):
        return (native_integer(value) if kind=='integer' else finite_number(value)) and field['min']<=value<=field['max']
    return False


def validate_field(field):
    """Mirror native ComponentField definition rules for retained source data."""
    def invalid():raise ValueError('Invalid composite parameter declaration')
    if (not isinstance(field,dict) or not isinstance(field.get('key'),str)
            or not 1<=len(field['key'])<=128 or 'defaultValue' not in field):invalid()
    kind=field.get('kind','integer')
    nullable=field.get('allowsNull',False)
    # Native optional Bool decoding also treats JSON null as absent.
    if nullable is None:nullable=False
    if type(nullable) is not bool or (nullable and kind not in ('integer','number')):invalid()
    if kind!='choice' and field.get('choices') is not None:invalid()
    if kind in ('choice','boolean'):
        if any(field.get(k) is not None for k in ('min','max','minItems','maxItems')):invalid()
        if kind=='choice':
            choices=field.get('choices')
            if type(choices) is not list or not 1<=len(choices)<=64:invalid()
            for choice in choices:
                if (type(choice) is not dict or any(type(choice.get(k)) is not str or not choice[k].strip()
                        or len(choice[k])>128 for k in ('value','title'))):invalid()
            if len({c['value'] for c in choices})!=len(choices):invalid()
    elif kind in ('integer','integerList','number'):
        low,high=field.get('min'),field.get('max')
        check=finite_number if kind=='number' else native_integer
        if not check(low) or not check(high) or low>high:invalid()
        if kind=='integerList':
            lower,upper=field.get('minItems'),field.get('maxItems')
            if not native_integer(lower) or not native_integer(upper) or not 1<=lower<=upper<=64:invalid()
        elif any(field.get(k) is not None for k in ('minItems','maxItems')):invalid()
    else:invalid()
    if not accepts_field_value(field,field['defaultValue']):invalid()


def validate_integer_products(definition,definitions,overrides):
    """Validate typed product metadata; native expansion still owns graph lowering."""
    bindings=definition.get('integerProducts')
    if bindings is None:return
    def invalid():raise ValueError('Invalid composite integer product binding')
    if type(bindings) is not list or len(bindings)>128:invalid()
    fields={p['field']['key']:p['field'] for p in definition.get('parameters',[])}
    occupied=set()
    for parameter in definition.get('parameters',[]):
        for target in parameter.get('targets',[]):
            if (type(target) is not dict or any(type(target.get(k)) is not str for k in ('componentID','field'))):invalid()
            occupied.add((target['componentID'],target['field']))
    body=definition.get('body')
    if type(body) is not dict:invalid()
    nodes=body.get('nodes');instances=body.get('instances')
    if type(nodes) is not list or type(instances) is not list:invalid()
    for binding in bindings:
        if type(binding) is not dict or set(binding)!={'lhs','rhs','targets'}:invalid()
        operands=[]
        for key in ('lhs','rhs'):
            name=binding[key]
            if type(name) is not str or name not in fields:invalid()
            field=fields[name]
            if field.get('kind','integer')!='integer' or field.get('allowsNull') is True:invalid()
            value=overrides.get(name,field['defaultValue'])
            if not accepts_field_value(field,value):invalid()
            operands.append(int(value))
        product=operands[0]*operands[1]
        if not -(2**63)<=product<2**63:invalid()
        targets=binding['targets']
        if type(targets) is not list or not 1<=len(targets)<=256:invalid()
        for target in targets:
            if (type(target) is not dict or set(target)!={'componentID','field'}
                    or any(type(target[k]) is not str or not 1<=len(target[k])<=128 for k in target)):invalid()
            address=(target['componentID'],target['field'])
            if address in occupied:invalid()
            occupied.add(address)
            node=next((n for n in nodes if isinstance(n,dict) and n.get('id')==address[0]),None)
            if node is not None:
                config=effective_config(node,body.get('configurationVersion',1))
                if address[1] not in config or not native_integer(config[address[1]]):invalid()
            else:
                instance=next((n for n in instances if isinstance(n,dict) and n.get('id')==address[0]),None)
                nested=definitions.get(json.dumps(instance.get('definition'),sort_keys=True)) if instance else None
                if nested is None:invalid()
                field=next((p['field'] for p in nested.get('parameters',[]) if p['field']['key']==address[1]),None)
                if field is None or field.get('kind','integer')!='integer' or field.get('allowsNull') is True or not accepts_field_value(field,product):invalid()
