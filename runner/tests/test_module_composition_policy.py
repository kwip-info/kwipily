import copy
from dataclasses import FrozenInstanceError
import pytest
import torch
from runner.module_composition_policy import validate_policy


def fixture(kind='graph', ids=('left', 'right')):
    ports = {ident: {'inputs': ['in'], 'outputs': ['out']} for ident in ids}
    children = [dict(localID=ident, **({'key': 'stage_' + str(i)} if kind == 'namedMap' else {})) for i, ident in enumerate(ids)]
    edges = [dict(id='wire'+str(i), source=a, sourcePort='out', target=b, targetPort='in') for i, (a,b) in enumerate(zip(ids, ids[1:]))]
    inputs = [dict(id='values', targets=[dict(componentID=ids[0], portID='in')])]
    outputs = [dict(id='result', source=dict(componentID=ids[-1], portID='out'))]
    return dict(schemaVersion=1, kind=kind, children=children), ports, edges, inputs, outputs


@pytest.mark.parametrize('kind', ['graph', 'sequential', 'orderedList', 'namedMap'])
def test_exact_nondefault_policy_and_deep_immutability(kind):
    args = fixture(kind); actual = validate_policy(*args)
    assert actual.to_dict() == args[0]
    with pytest.raises(FrozenInstanceError): actual.kind = 'graph'
    with pytest.raises(FrozenInstanceError): actual.children[0].local_id = 'foreign'
    args[0]['children'][0]['localID'] = 'changed'
    assert actual.children[0].local_id == 'left'
    exported = actual.to_dict(); exported['children'].clear()
    assert len(actual.children) == 2


@pytest.mark.parametrize('change', [
    {'schemaVersion': True}, {'schemaVersion': 1.0}, {'schemaVersion': 0}, {'schemaVersion': 2},
    {'schemaVersion': 2**63}, {'kind': 'ModuleDict'}, {'kind': None}, {'extra': 1},
    {'children': []}, {'children': None}, {'children': [{'localID': 'left'}]},
    {'children': [{'localID': 'left'}, {'localID': 'left'}]},
    {'children': [{'localID': 'left'}, {'localID': 'foreign'}]},
    {'children': [{'localID': 'left', 'extra': 1}, {'localID': 'right'}]},
    {'children': [{'localID': 'left', 'key': None}, {'localID': 'right'}]},
])
def test_strict_policy_envelope_and_exact_body_members(change):
    args = list(fixture()); args[0].update(change)
    with pytest.raises(ValueError): validate_policy(*args)


@pytest.mark.parametrize('kind', ['graph', 'sequential', 'orderedList'])
def test_keys_are_for_named_map_only(kind):
    args = fixture(kind); args[0]['children'][0]['key'] = 'valid_key'
    with pytest.raises(ValueError): validate_policy(*args)


@pytest.mark.parametrize('key', ['', 'a.b', 'has space', 'é', '1first', 'a\x00b', 'x'*65, None, True,
                                'training', 'forward', 'items', 'update', '_modules', '_parameters', '_buffers', '__dict__'])
def test_native_map_keys_refuse_invalid_names_and_collisions(key):
    args = fixture('namedMap'); args[0]['children'][0]['key'] = key
    with pytest.raises(ValueError): validate_policy(*args)


def test_named_map_requires_complete_unique_keys():
    args = fixture('namedMap'); del args[0]['children'][0]['key']
    with pytest.raises(ValueError): validate_policy(*args)
    args = fixture('namedMap'); args[0]['children'][1]['key'] = args[0]['children'][0]['key']
    with pytest.raises(ValueError): validate_policy(*args)


def test_allow_unicode_body_identity_and_safe_underscore_map_key():
    args = fixture('namedMap', ('測定_α', 'é'*128))
    args[0]['children'][0]['key'] = '_stage'
    assert validate_policy(*args).children[0].key == '_stage'


@pytest.mark.parametrize('ident', ['', 'x'*129, None, True])
def test_body_identity_bounds(ident):
    args = list(fixture()); args[1] = {ident: args[1]['left'], 'right': args[1]['right']}
    args[0]['children'][0]['localID'] = ident
    with pytest.raises(ValueError): validate_policy(*args)


def test_body_resource_limit_and_maximum_chain():
    args = fixture('sequential', tuple('node'+str(i) for i in range(128)))
    assert len(validate_policy(*args).children) == 128
    with pytest.raises(ValueError): validate_policy(*fixture('sequential', tuple('node'+str(i) for i in range(129))))


@pytest.mark.parametrize('mutation', ['reorder','missingEdge','reverseEdge','branch','multipleInputs','multipleTargets','multipleOutputs','wrongBoundaryStart','wrongBoundaryEnd','auxiliaryOutput','multiInput','wrongSourcePort','wrongTargetPort','duplicateEdgeID','unknownEdgeField'])
def test_sequence_requires_actual_ordered_unary_chain(mutation):
    args = list(fixture('sequential', ('a','b','c'))); policy, ports, edges, inputs, outputs = args
    if mutation == 'reorder': policy['children'].reverse()
    elif mutation == 'missingEdge': edges.pop()
    elif mutation == 'reverseEdge': edges[0]['source'],edges[0]['target']='b','a'
    elif mutation == 'branch': edges[1]['source']='a'
    elif mutation == 'multipleInputs': inputs.append(dict(id='other',targets=[dict(componentID='b',portID='in')]))
    elif mutation == 'multipleTargets': inputs[0]['targets'].append(dict(componentID='b',portID='in'))
    elif mutation == 'multipleOutputs': outputs.append(dict(id='other',source=dict(componentID='b',portID='out')))
    elif mutation == 'wrongBoundaryStart': inputs[0]['targets'][0]['componentID']='b'
    elif mutation == 'wrongBoundaryEnd': outputs[0]['source']['componentID']='b'
    elif mutation == 'auxiliaryOutput': ports['b']['outputs'].append('metadata')
    elif mutation == 'multiInput': ports['b']['inputs'].append('other')
    elif mutation == 'wrongSourcePort': edges[0]['sourcePort']='wrong'
    elif mutation == 'wrongTargetPort': edges[0]['targetPort']='wrong'
    elif mutation == 'duplicateEdgeID': edges[1]['id']=edges[0]['id']
    elif mutation == 'unknownEdgeField': edges[0]['extra']=1
    with pytest.raises(ValueError): validate_policy(*args)


def test_single_child_sequence_and_arbitrary_actual_port_names():
    args = list(fixture('sequential', ('only',))); args[1]['only']={'inputs':['values'],'outputs':['result']}
    args[3][0]['targets'][0]['portID']='values';args[4][0]['source']['portID']='result'
    assert validate_policy(*args).kind == 'sequential'


@pytest.mark.parametrize('kind', ['graph','orderedList','namedMap'])
def test_branched_dag_has_storage_order_without_fake_sequential_requirement(kind):
    args = list(fixture(kind, ('a','b','merge')))
    args[1]['merge']={'inputs':['a','b'],'outputs':['out','metadata']}
    args[2]=[dict(id='a-merge',source='a',sourcePort='out',target='merge',targetPort='a'),dict(id='b-merge',source='b',sourcePort='out',target='merge',targetPort='b')]
    args[3][0]['targets'].append(dict(componentID='b',portID='in'))
    args[0]['children'].reverse()
    assert validate_policy(*args).children[0].local_id == 'merge'


def test_native_attribute_collisions_checked_without_constructing_modules(monkeypatch):
    keys={key for key in dir(torch.nn.ModuleDict()) if key and len(key)<=64 and key.isascii() and (key[0].isalpha() or key[0]=='_') and all(c.isalnum() or c=='_' for c in key)}
    def forbidden(*a, **k): raise AssertionError('Policy validation must not allocate a module')
    monkeypatch.setattr(torch.nn.ModuleDict, '__init__', forbidden)
    monkeypatch.setattr(torch.nn.Module, '__init__', forbidden)
    for key in keys:
        args=fixture('namedMap');args[0]['children'][0]['key']=key
        with pytest.raises(ValueError): validate_policy(*args)
    assert validate_policy(*fixture('namedMap')).kind == 'namedMap'


@pytest.mark.parametrize('change', ['unknownPortsField','duplicatePorts','nonListPorts','foreignBoundary','emptyBoundary','unknownBoundaryField','emptyOutputs','tooManyEdges'])
def test_source_contracts_are_bounded_exact_and_known(change):
    args=list(fixture())
    if change=='unknownPortsField':args[1]['left']['extra']=1
    elif change=='duplicatePorts':args[1]['left']['inputs']=['in','in']
    elif change=='nonListPorts':args[1]['left']['outputs']='out'
    elif change=='foreignBoundary':args[3][0]['targets'][0]['componentID']='foreign'
    elif change=='emptyBoundary':args[3][0]['targets']=[]
    elif change=='unknownBoundaryField':args[4][0]['extra']=1
    elif change=='emptyOutputs':args[4]=[]
    elif change=='tooManyEdges':args[2]=args[2]*257
    with pytest.raises(ValueError):validate_policy(*args)
