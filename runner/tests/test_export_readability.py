import ast
import copy
import hashlib
import keyword
import torch
import pytest
from runner.engine import GraphModel,export_source
from runner.tests.test_runner import graph


def compiled(source):
    namespace={};exec(compile(source,'exported-model.py','exec'),namespace)
    return namespace['Model']()

def hashed_graph(labels):
    value=graph();mapping={n['id']:hashlib.sha256(n['id'].encode()).hexdigest() for n in value['nodes']}
    for node,label in zip(value['nodes'],labels):node['id']=mapping[node['id']];node['name']=label
    for edge in value['edges']:edge['source']=mapping[edge['source']];edge['target']=mapping[edge['target']]
    return value

def local_assignments(source):
    model=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef))
    forward=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    return [n.targets[0] for n in forward.body if isinstance(n,ast.Assign)]

@pytest.mark.parametrize("merge",["concat","add"])
def test_hashed_ids_become_readable_tensor_locals_without_parameter_key_changes(merge):
    value=hashed_graph(['Features','Left projection','Right projection','Left activation','Right activation','Join branches','Classifier','Prediction'])
    if merge=='add':
        next(n for n in value['nodes'] if n['type']=='concat')['type']='add'
        next(n for n in value['nodes'] if n['name']=='Classifier')['config']['in_features']=16
    model=GraphModel(value);source=export_source(model)
    locals=local_assignments(source)
    assert all(isinstance(n,ast.Name) for n in locals)
    assert {'features','left_projection','right_projection','join_branches','prediction'}<={n.id for n in locals}
    exported=compiled(source);exported.load_state_dict(model.state_dict(),strict=True)
    assert list(exported.state_dict())==list(model.state_dict())
    x=torch.randn(8,2,requires_grad=True)
    assert torch.equal(model(x),exported(x))
    assert torch.equal(torch.autograd.grad(model(x).sum(),x)[0],torch.autograd.grad(exported(x).sum(),x)[0])
    assert export_source(model)==source

def test_duplicate_reserved_code_like_and_unicode_labels_are_safe_and_deterministic():
    labels=['self','torch','class','same label','same-label','123\n__import__("os").system("bad")','日本語','x']
    model=GraphModel(hashed_graph(labels));source=export_source(model);names=[n.id for n in local_assignments(source) if isinstance(n,ast.Name)]
    assert len(names)==len(model.order) and len(set(names))==len(names)
    assert all(n.isidentifier() and not keyword.iskeyword(n) and n not in {'self','x','torch','nn'} for n in names)
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='__import__' for n in ast.walk(tree))
    exported=compiled(source);exported.load_state_dict(model.state_dict())
    assert torch.equal(model(torch.ones(2,2)),exported(torch.ones(2,2)))
    assert export_source(model)==source

def test_label_changes_leave_existing_checkpoint_keys_and_outputs_compatible():
    original=graph();model=GraphModel(original);old_checkpoint=copy.deepcopy(model.state_dict())
    for node in original['nodes']:node['name']='Readable user name'
    renamed=GraphModel(original);exported=compiled(export_source(renamed));exported.load_state_dict(old_checkpoint,strict=True)
    x=torch.randn(3,2);assert torch.equal(exported(x),model(x))
    assert set(exported.state_dict())==set(old_checkpoint)
