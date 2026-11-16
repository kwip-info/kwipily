"""Strict immutable composition plans and real native container execution.

The flat schema-2 DAG supplies math; the separately admitted plan supplies native
registration and boundary invocation. No native module is allocated by admission.
"""
from dataclasses import dataclass
import hashlib
import json
import weakref
from torch import nn
if __package__:
    from .module_composition_policy import _exact, _name, _array, _KEY, _RESERVED
else:
    from module_composition_policy import _exact, _name, _array, _KEY, _RESERVED

@dataclass(frozen=True)
class AdmittedPlan:
    wire: str
    schedules: tuple
    memberships: tuple
    def to_dict(self): return json.loads(self.wire)

def container_id(path):
    return 'root' if not path else 'c-'+hashlib.sha256(b''.join(str(len(v.encode('utf8'))).encode('ascii')+b':'+v.encode('utf8') for v in path)).hexdigest()

def validate_plan(plan, graph, child_ports):
    _exact(plan, ('schemaVersion','root','containers'), 'Composition plan')
    if type(plan['schemaVersion']) is not int or plan['schemaVersion']!=1 or plan['root']!='root':
        raise ValueError('Composition plan requires schema 1 and the root container.')
    if graph.get('schemaVersion')!=2:raise ValueError('Composition plans require primitive graph schema 2.')
    try: wire=json.dumps(plan,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError,UnicodeError) as exc:raise ValueError('Composition plan must contain finite bounded JSON.') from exc
    try: size=len(wire.encode('utf8'))
    except UnicodeError as exc:raise ValueError('Composition plan requires valid Unicode.') from exc
    if size>131072:raise ValueError('Composition plan exceeds 131072 UTF8 bytes.')
    rows=_array(plan['containers'],1,129,'Composition containers');by_id={};child_count=0;endpoint_count=0
    nodes={n['id']:n for n in graph['nodes']};compute={i for i,n in nodes.items() if n['type'] not in ('input','index_input','bag_input','output')}
    source_nodes=[i for i,n in nodes.items() if n['type'] in ('input','index_input','bag_input')];sink_nodes=[i for i,n in nodes.items() if n['type']=='output']
    named=graph.get('namedInputs')
    if len(sink_nodes)!=1 or (len(source_nodes)!=1 if named is None else type(named) is not dict or not 1<=len(source_nodes)<=8 or set(named.values())!=set(source_nodes)):raise ValueError('Composition needs its exact declared input and output sentinels.')
    incoming={(e['target'],e['targetPort']):(e['source'],e['sourcePort']) for e in graph['edges']}
    def endpoint(value,direction):
        _exact(value,('node','port'),'Composition boundary endpoint');node=_name(value['node'],'Boundary node');port=_name(value['port'],'Boundary port')
        if node not in compute or node not in child_ports or port not in child_ports[node][direction]:raise ValueError('Composition boundary must reference an actual configured compute port.')
        return node,port
    for row in rows:
        _exact(row,('id','kind','path','children','inputs','outputs'),'Composition container')
        ident=_name(row['id'],'Container identity');path=_array(row['path'],0,8,'Container path')
        for segment in path:_name(segment,'Instance path segment')
        if ident!=container_id(path) or ident in by_id:raise ValueError('Container identities must match unique length-delimited instance paths.')
        if row['kind'] not in ('graph','sequential','orderedList','namedMap'):raise ValueError('Unsupported composition container kind.')
        if ident=='root' and (path or row['kind']!='graph'):raise ValueError('The synthetic root must be an owning graph container.')
        children=_array(row['children'],1,128,'Container children');child_count+=len(children);keys=set();refs=set()
        for index,child in enumerate(children):
            _exact(child,('kind','id','key'),'Composition child')
            if child['kind'] not in ('node','container'):raise ValueError('Composition children have typed node or container references.')
            _name(child['id'],'Child identity');key=child['key']
            ref=child['kind'],child['id']
            if ref in refs:raise ValueError('Composition child aliases are refused.')
            refs.add(ref)
            if row['kind']=='namedMap':
                if type(key) is not str or _KEY.fullmatch(key) is None or key in _RESERVED or hasattr(nn.Module,key) or hasattr(nn.ModuleDict,key):raise ValueError('Named collection key is invalid or collides with a native attribute.')
            elif type(key) is not str or key!=str(index):raise ValueError('Graph, sequence and list storage keys must be exact positional indices.')
            if key in keys:raise ValueError('Native child storage keys must be unique.')
            keys.add(key)
        names=set()
        for entry in _array(row['inputs'],0,128,'Container inputs'):
            _exact(entry,('id','targets'),'Container input');name=_name(entry['id'],'Container input')
            if name in names:raise ValueError('Container input names must be unique.')
            names.add(name)
            targets=_array(entry['targets'],1,256,'Container input targets');endpoint_count+=len(targets)
            for target in targets:endpoint(target,'inputs')
        names=set()
        for entry in _array(row['outputs'],1,128,'Container outputs'):
            _exact(entry,('id','source'),'Container output');name=_name(entry['id'],'Container output')
            if name in names:raise ValueError('Container output names must be unique.')
            names.add(name);endpoint(entry['source'],'outputs');endpoint_count+=1
        by_id[ident]=row
    if child_count>256 or endpoint_count>4096:raise ValueError('Composition plan exceeds child or boundary endpoint limits.')
    if 'root' not in by_id:raise ValueError('Composition root is missing.')
    preorder=[];seen_nodes=set();members={}
    def visit(ident,ancestry):
        if ident not in by_id or ident in preorder or ident in ancestry:raise ValueError('Composition containers are missing, cyclic or aliased.')
        row=by_id[ident];preorder.append(ident);leaves=set()
        for child in row['children']:
            kind,node=child['kind'],child['id']
            if kind=='node':
                if node not in compute or node in seen_nodes:raise ValueError('Each actual compute node must be registered exactly once.')
                seen_nodes.add(node);leaves.add(node)
            else:
                if node not in by_id or len(by_id[node]['path'])!=len(row['path'])+1 or by_id[node]['path'][:-1]!=row['path']:raise ValueError('Nested container path must be the direct actual instance path.')
                leaves.update(visit(node,ancestry+(ident,)))
        members[ident]=leaves;return leaves
    visit('root',())
    if preorder!=[r['id'] for r in rows] or seen_nodes!=compute:raise ValueError('Composition requires exact preorder and complete primitive coverage.')
    schedules=[]
    for row in rows:
        ident=row['id'];inside=members[ident];claimed=set()
        for entry in row['inputs']:
            addresses=[]
            for e in entry['targets']:
                target=e['node'],e['port']
                if target[0] not in inside or target in claimed or target not in incoming or incoming[target][0] in inside:raise ValueError('Composition inputs must exactly describe external incoming wires.')
                claimed.add(target);addresses.append(incoming[target])
            root_bag=ident=='root' and nodes[source_nodes[0]]['type']=='bag_input' and entry['id']==source_nodes[0]
            if root_bag:
                enabled=child_ports[source_nodes[0]]['outputs']
                if any(node!=source_nodes[0] or port not in enabled for node,port in addresses):raise ValueError('Root bag boundary requires enabled roles from its actual single source.')
            elif len(set(addresses))!=1:raise ValueError('One boundary input cannot combine distinct external sources.')
        crossings={target for target,source in incoming.items() if target[0] in inside and source[0] not in inside}
        if claimed!=crossings:raise ValueError('Composition input boundaries omit or forge flat DAG crossings.')
        outputs={(e['source']['node'],e['source']['port']) for e in row['outputs']}
        if any(node not in inside for node,port in outputs):raise ValueError('Container outputs must belong to that container.')
        external_outputs={source for target,source in incoming.items() if source[0] in inside and target[0] not in inside}
        if not external_outputs<=outputs:raise ValueError('Composition outputs omit an external flat DAG crossing.')
        if ident=='root':
            expected_targets={target for target,source in incoming.items() if source[0] in source_nodes}
            if [e['id'] for e in row['inputs']]!=source_nodes or claimed!=expected_targets or len(row['outputs'])!=1 or row['outputs'][0]['id']!=sink_nodes[0] or outputs!={incoming[(sink_nodes[0],'in')]}:raise ValueError('Root boundaries must exactly match the actual input/output sentinels.')
            for entry in row['inputs']:
                if {(e['node'],e['port']) for e in entry['targets']}!={target for target,source in incoming.items() if source[0]==entry['id']}:raise ValueError('Root named input binding merges different sources.')
        direct={}
        for index,child in enumerate(row['children']):
            leaves={child['id']} if child['kind']=='node' else members[child['id']]
            for node in leaves:direct[node]=index
        deps=[set() for _ in row['children']]
        for target,source in incoming.items():
            if target[0] in direct and source[0] in direct and direct[target[0]]!=direct[source[0]]:deps[direct[target[0]]].add(direct[source[0]])
        order=[];remaining=set(range(len(deps)))
        while remaining:
            ready=[i for i in sorted(remaining) if deps[i]<=set(order)]
            if not ready:raise ValueError('Container invocation graph contains a cycle.')
            order.extend(ready);remaining.difference_update(ready)
        if row['kind']=='sequential':
            if len(row['inputs'])!=1 or len(row['outputs'])!=1:raise ValueError('Sequence exposes exactly one input and output.')
            for index,child in enumerate(row['children']):
                sig=child_ports[child['id']] if child['kind']=='node' else by_id[child['id']]
                if len(sig['inputs'])!=1 or len(sig['outputs'])!=1:raise ValueError('Sequence children require one configured input/output without auxiliary outputs.')
                if deps[index] != ({index-1} if index else set()):raise ValueError('Sequence order must match the actual complete wired chain.')
            first_targets={direct[e['node']] for e in row['inputs'][0]['targets']}
            if first_targets!={0} or direct[row['outputs'][0]['source']['node']]!=len(row['children'])-1:raise ValueError('Sequence boundaries must name the first and last child.')
            if any(source[0] not in inside for target,source in incoming.items() if target[0] in inside and direct[target[0]]>0):raise ValueError('Sequence later children cannot consume another external source.')
        schedules.append((ident,tuple(order)))
    return AdmittedPlan(wire,tuple(schedules),tuple((i,tuple(sorted(members[i]))) for i in preorder))

@dataclass(frozen=True)
class Invocation:
    values: tuple
    frame: object

class ExecutionFrame:
    def __init__(self,x,observer):self.x=x;self.observer=observer;self.values={}

class PrimitiveInvocation(nn.Module):
    def __init__(self,owner,node,operation=None):
        super().__init__();self._owner=weakref.ref(owner);self.node=node
        if operation is not None:self.operation=operation
    def forward(self,call):return self._owner()._composition_leaf(self.node,call)

class GraphContainer(nn.Module):
    def __init__(self,owner,ident,children):
        super().__init__();self._owner=weakref.ref(owner);self.ident=ident;self.blocks=nn.ModuleDict(children)
    def forward(self,call):return self._owner()._composition_container(self.ident,call)

class OrderedContainer(nn.ModuleList):
    # A collection's explicit admitted DAG supplies invocation; no automatic fold.
    def __init__(self,owner,ident,children):
        super().__init__(children);self._owner=weakref.ref(owner);self.ident=ident
    def forward(self,call):return self._owner()._composition_container(self.ident,call)

class NamedContainer(nn.ModuleDict):
    def __init__(self,owner,ident,children):
        super().__init__(children);self._owner=weakref.ref(owner);self.ident=ident
    def forward(self,call):return self._owner()._composition_container(self.ident,call)
