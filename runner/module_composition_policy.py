"""Immutable explicit Block container policy (foundation; no runtime integration)."""
from dataclasses import dataclass
import re
from torch import nn

@dataclass(frozen=True)
class CompositionChild:
    local_id: str
    key: str | None = None

@dataclass(frozen=True)
class CompositionPolicy:
    schema_version: int
    kind: str
    children: tuple[CompositionChild, ...]

    def to_dict(self):
        return {'schemaVersion': self.schema_version, 'kind': self.kind,
                'children': [dict(localID=c.local_id, **({'key': c.key} if c.key is not None else {})) for c in self.children]}


# Instance attributes created by nn.Module.__init__, plus installed native class
# attributes below. This admits safe underscore keys without permitting internal
# registration/state/hook attributes. No module is instantiated by admission.
_RESERVED = frozenset({
    'training', '_parameters', '_buffers', '_non_persistent_buffers_set',
    '_backward_pre_hooks', '_backward_hooks', '_is_full_backward_hook',
    '_forward_hooks', '_forward_hooks_with_kwargs', '_forward_hooks_always_called',
    '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_state_dict_hooks',
    '_state_dict_pre_hooks', '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks',
    '_modules',
})
_KEY = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,63}', flags=re.ASCII)


def _exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + ' has missing or unknown fields.')


def _name(value, label):
    # Local identities preserve the existing source contract: Unicode scalars,
    # no lossy sanitization or reinterpretation as native registration keys.
    if type(value) is not str or not 1 <= len(value) <= 128 or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError(label + ' requires a nonempty Unicode identity of at most 128 scalars.')
    return value


def _array(value, minimum, maximum, label):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError(label + f' requires {minimum}–{maximum} entries.')
    return value


def _port_reference(value, ports, direction):
    _exact(value, ('componentID', 'portID'), 'Boundary endpoint')
    node = _name(value['componentID'], 'Boundary child')
    port = _name(value['portID'], 'Boundary port')
    if node not in ports or port not in ports[node][direction]:
        raise ValueError('Boundary endpoint must name an actual configured child port.')
    return node, port


def validate_policy(policy, child_ports, edges, inputs, outputs):
    """Admit explicit immutable policy against caller-resolved source ports.

    child_ports is the exact primitive/instance local-ID map; configured port
    lookup and nested-definition/revision/shape validation belong to the caller.
    edges/inputs/outputs use CompositeGraph/CompositeInput/CompositeOutput wire
    fields. This function neither constructs modules nor infers tensor layouts.
    """
    _exact(policy, ('schemaVersion', 'kind', 'children'), 'Composition policy')
    if type(policy['schemaVersion']) is not int or policy['schemaVersion'] != 1:
        raise ValueError('Composition policy requires exact integer schema version 1.')
    kind = policy['kind']
    if type(kind) is not str or kind not in ('graph', 'sequential', 'orderedList', 'namedMap'):
        raise ValueError('Choose Graph, Sequence, Ordered collection or Named collection.')
    if type(child_ports) is not dict or not 1 <= len(child_ports) <= 128:
        raise ValueError('Composition requires 1–128 actual body primitives or instances.')
    ports = {}
    for node, declaration in child_ports.items():
        _name(node, 'Body child')
        _exact(declaration, ('inputs', 'outputs'), 'Configured child ports')
        entry = {}
        for direction in ('inputs', 'outputs'):
            values = _array(declaration[direction], 0, 128, 'Configured ports')
            for value in values: _name(value, 'Configured port')
            if len(set(values)) != len(values):
                raise ValueError('Configured port names must be unique.')
            entry[direction] = tuple(values)
        ports[node] = entry
    children = []; seen = set(); keys = set()
    for child in _array(policy['children'], 1, 128, 'Composition children'):
        _exact(child, ('localID', 'key') if kind == 'namedMap' else ('localID',), 'Composition child')
        local_id = _name(child['localID'], 'Composition child')
        if local_id not in ports or local_id in seen:
            raise ValueError('List each actual body child exactly once; aliases and foreign children are refused.')
        seen.add(local_id)
        key = child.get('key')
        if kind == 'namedMap':
            if type(key) is not str or _KEY.fullmatch(key) is None:
                raise ValueError('Named collection keys use 1–64 ASCII letters, digits or underscores, starting with a letter or underscore.')
            if key in _RESERVED or hasattr(nn.Module, key) or hasattr(nn.ModuleDict, key):
                raise ValueError('Named collection key collides with a native module attribute. Choose another name.')
            if key in keys:
                raise ValueError('Named collection keys must be unique.')
            keys.add(key)
        children.append(CompositionChild(local_id, key))
    if seen != set(ports):
        raise ValueError('Composition must include every actual body child exactly once.')

    admitted_edges = []; edge_ids = set(); occupied = set()
    for edge in _array(edges, 0, 256, 'Body connections'):
        _exact(edge, ('id', 'source', 'sourcePort', 'target', 'targetPort'), 'Body connection')
        for field in ('id', 'source', 'sourcePort', 'target', 'targetPort'): _name(edge[field], 'Connection ' + field)
        source = edge['source'], edge['sourcePort']; target = edge['target'], edge['targetPort']
        if edge['id'] in edge_ids or source[0] not in ports or target[0] not in ports or source[1] not in ports[source[0]]['outputs'] or target[1] not in ports[target[0]]['inputs']:
            raise ValueError('Connections require unique identities and actual configured child ports.')
        if target in occupied:
            raise ValueError('Each child input accepts exactly one internal or boundary connection.')
        edge_ids.add(edge['id']); occupied.add(target); admitted_edges.append((source, target))
    admitted_inputs = []; boundary_ids = set()
    for boundary in _array(inputs, 0, 128, 'Boundary inputs'):
        _exact(boundary, ('id', 'targets'), 'Boundary input'); name = _name(boundary['id'], 'Boundary input')
        if name in boundary_ids: raise ValueError('Boundary input identities must be unique.')
        boundary_ids.add(name); targets = []
        for endpoint in _array(boundary['targets'], 1, 256, 'Boundary input targets'):
            target = _port_reference(endpoint, ports, 'inputs')
            if target in occupied: raise ValueError('Each child input accepts exactly one internal or boundary connection.')
            occupied.add(target); targets.append(target)
        admitted_inputs.append(tuple(targets))
    admitted_outputs = []; boundary_ids = set()
    for boundary in _array(outputs, 1, 128, 'Boundary outputs'):
        _exact(boundary, ('id', 'source'), 'Boundary output'); name = _name(boundary['id'], 'Boundary output')
        if name in boundary_ids: raise ValueError('Boundary output identities must be unique.')
        boundary_ids.add(name); admitted_outputs.append(_port_reference(boundary['source'], ports, 'outputs'))

    if kind == 'sequential':
        if any(len(p['inputs']) != 1 or len(p['outputs']) != 1 for p in ports.values()):
            raise ValueError('Sequence children require exactly one actual input and output, without auxiliary outputs.')
        if len(admitted_inputs) != 1 or len(admitted_inputs[0]) != 1 or len(admitted_outputs) != 1:
            raise ValueError('A sequence exposes one input with one target and one output.')
        ids = [c.local_id for c in children]
        first = ids[0], ports[ids[0]]['inputs'][0]
        last = ids[-1], ports[ids[-1]]['outputs'][0]
        expected = {((a, ports[a]['outputs'][0]), (b, ports[b]['inputs'][0])) for a, b in zip(ids, ids[1:])}
        if admitted_inputs[0][0] != first or admitted_outputs[0] != last or len(admitted_edges) != len(ids)-1 or set(admitted_edges) != expected:
            raise ValueError('Sequence order must match the complete single wired chain from its first input to its last output.')
    return CompositionPolicy(1, kind, tuple(children))
