import json
from pathlib import Path
import pytest
import torch
from runner.output_routing import OutputAddress, select_output
from runner.engine import GraphModel, validate, export_source, REGISTRY
from runner.tests.test_runner import graph


def test_named_outputs_do_not_fall_back_and_keep_tensor_identity():
    value=torch.tensor([1.],requires_grad=True);indices=torch.tensor([0],dtype=torch.int64)
    outputs={'node':{'out':value,'indices':indices},'legacy':value}
    assert select_output(outputs,OutputAddress('node','out')) is value
    assert select_output(outputs,OutputAddress('node','indices')) is indices
    assert select_output(outputs,OutputAddress('legacy','out')) is value
    for address in [OutputAddress('node','missing'),OutputAddress('legacy','indices'),OutputAddress('missing','out')]:
        with pytest.raises(ValueError,match='output'):select_output(outputs,address)
    select_output(outputs,OutputAddress('node','out')).sum().backward()
    assert torch.equal(value.grad,torch.ones_like(value))


def test_validation_refuses_declared_but_unresolved_named_output(tmp_path):
    source=graph();registry=json.loads(Path(REGISTRY).read_text())
    node=next(n for n in source['nodes'] if n['type']=='linear')
    next(c for c in registry['components'] if c['id']=='linear')['outputs'].append('indices')
    next(e for e in source['edges'] if e['source']==node['id'])['sourcePort']='indices'
    path=tmp_path/'registry.json';path.write_text(json.dumps(registry))
    with pytest.raises(ValueError,match='output'):validate(source,path)


def test_model_retains_source_port_and_export_preserves_forward_gradient():
    source=graph();model=GraphModel(source)
    for edge in source['edges']:
        assert model.incoming[edge['target']][edge['targetPort']]==OutputAddress(edge['source'],edge['sourcePort'])
    ns={};exec(export_source(model),ns);exported=ns['Model']();exported.load_state_dict(model.state_dict())
    x=torch.randn(3,2,requires_grad=True)
    assert torch.equal(model(x),exported(x))
    assert torch.equal(torch.autograd.grad(model(x).sum(),x)[0],torch.autograd.grad(exported(x).sum(),x)[0])
