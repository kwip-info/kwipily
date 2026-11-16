"""Actual relocated standalone selected loading/model exports, no runner imports."""
import importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import pytest,torch
from runner import engine
from runner.tests.test_dataset_loader_configuration import recipe
from runner.tests.test_dataset_loader_engine import setup
from runner.dataset_loader_portable import data_loading_recipe_code

@pytest.fixture(autouse=True)
def policy():
    before=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(before,warn_only=warn)


def load_module(path):
    spec=importlib.util.spec_from_file_location('data_loading_recipe',path);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def test_generated_legacy_size_and_bounded_source_worker_semantics(tmp_path):
    file=tmp_path/'data_loading_recipe.py';file.write_text(data_loading_recipe_code(recipe(kind='subset',indices=[4,1,0]),None,3000))
    module=load_module(file);x=torch.arange(10.).reshape(5,2).requires_grad_()
    source=module.create_training_source(x,{'target':torch.arange(5)},list(range(5)))
    with module.open_training_epoch(source,0) as batches:
        row=next(batches);assert row.row_indices==(4,1,0)
        row.input.sum().backward()
    assert x.grad.tolist()==[[1.,1.],[1.,1.],[0.,0.],[0.,0.],[1.,1.]]
    assert len(file.read_bytes())<=131072
    assert 'from runner' not in file.read_text() and 'import runner' not in file.read_text()

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('domain',['tensor','whole_bags'])
def test_real_saved_package_isolated_native_loading_predictions_and_fresh_update(tmp_path,device,domain):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    if domain=='tensor':
        graph,kwargs=setup(tmp_path);data=kwargs['prepared'];source=data[0][0]
        payload=dict(inputs=source.tolist(),roles={k:v.tolist() for k,v in data[0][1].items()},ids=data[1]['trainingRowIDs'])
    else:
        from runner.tests.test_prepared_bag_loader import fixture,load
        from runner.tests.test_bag_source_graph import graph as bag_graph
        data=load(fixture(tmp_path/'source'));graph=bag_graph();graph['nodes'][0]['config']['maximum_items_per_bag']=2
        kwargs=dict(prepared=data,seed=41,device=device);source=data[0][0]
        payload=dict(indices=source.indices.tolist(),offsets=source.offsets.tolist(),weights=source.per_sample_weights.tolist(),roles={'target':data[0][1].tolist()},ids=data[1]['trainingRowIDs'])
    kwargs['device']=device
    dr=recipe(kind='subset',indices=[4,1,0,7,3]);lr=recipe('data_loader',batch_size=2,shuffle=True,num_workers=1,prefetch_factor=1,timeout=10,collation='whole_bags' if domain=='whole_bags' else 'fixed_stack')
    engine.run(graph,tmp_path/'run',epochs=1,batch_size=2,dataset_recipe=dr,loader_recipe=lr,**kwargs)
    state=torch.load(tmp_path/'run/checkpoint.pt',weights_only=True)['modelState'];package=tmp_path/'isolated';package.mkdir()
    for name in ('model.py','checkpoint.pt','data_loading_recipe.py','objective_recipe.py','training_recipe.py'):
        if (tmp_path/'run'/name).exists():shutil.copyfile(tmp_path/'run'/name,package/name)
    shutil.rmtree(tmp_path/'run');shutil.rmtree(tmp_path/'source')
    (package/'input.json').write_text(json.dumps(payload))
    script='''import json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import model,data_loading_recipe as loading

def main():
    device=DEVICE
    values=json.loads(Path(__file__).with_name('input.json').read_text())
    native=model.Model().to(device)
    native.load_state_dict(torch.load(Path(__file__).with_name('checkpoint.pt'),map_location=device,weights_only=True)['modelState'])
    if DOMAIN=='tensor':
        inputs=torch.tensor(values['inputs'])
        roles={k:torch.tensor(v,dtype=torch.float32) for k,v in values['roles'].items()}
    else:
        inputs=loading.EmbeddingBagPacket('packed',torch.tensor(values['indices']),torch.tensor(values['offsets']),torch.tensor(values['weights']),True)
        roles={'target':torch.tensor(values['roles']['target'])}
    source=loading.create_training_source(inputs,roles,values['ids'])
    outputs=[];indices=[]
    with loading.open_training_epoch(source,1) as batches:
        pids=list(batches.worker_pids)
        for row in batches:
            if DOMAIN=='tensor':
                row=loading.TrainingBatch(row.input.to(device),{k:v.to(device) for k,v in row.roles.items()},row.row_indices,row.row_ids)
            else:
                p=row.input;row=loading.TrainingBatch(loading.EmbeddingBagPacket(p.layout,p.indices.to(device),p.offsets.to(device),p.per_sample_weights.to(device),p.include_last_offset),{k:v.to(device) for k,v in row.roles.items()},row.row_indices,row.row_ids)
            output=loading.forward_batch(native,row)
            outputs.extend(output.detach().cpu().tolist());indices.extend(row.row_indices)
    result=dict(outputs=outputs,indices=indices,workerPIDs=pids)
    if DOMAIN=='tensor':
        # A real fresh SGD step using exported objective/optimizer helpers.
        import objective_recipe,training_recipe
        optimizer=training_recipe.create_optimizer(native)
        objective=objective_recipe.create_training_objective(device)
        positions=indices[:2];x=torch.tensor(values['inputs'],device=device)[positions]
        target=torch.tensor(values['roles']['target'],device=device)[positions]
        optimizer.zero_grad();measurement=objective_recipe.evaluate_training_objective(objective,native(x),{'target':target})
        measurement['scalar'].backward();optimizer.step()
        result['updated']={k:v.detach().cpu().tolist() for k,v in native.state_dict().items()}
        result['firstRows']=positions
    print(json.dumps(result))
if __name__=='__main__':main()
'''.replace('DEVICE',repr(device)).replace('DOMAIN',repr(domain))
    (package/'check.py').write_text(script)
    result=subprocess.run([sys.executable,'-I',str(package/'check.py')],cwd=package,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    actual=json.loads(result.stdout);positions=actual['indices'];assert sorted(positions)==[0,1,3,4,7]
    expected=[]
    if domain=='tensor':
        w=state['layers.learn.weight'].tolist();b=state['layers.learn.bias'].tolist()
        for i in positions:expected.append([bias+sum(a*x for a,x in zip(row,payload['inputs'][i])) for row,bias in zip(w,b)])
        chosen=actual['firstRows'];dw=[[0.,0.] for _ in range(2)];db=[0.,0.]
        for i in chosen:
            for o in range(2):
                pred=b[o]+sum(a*x for a,x in zip(w[o],payload['inputs'][i]));grad=2*(pred-payload['roles']['target'][i][o])/(len(chosen)*2)
                db[o]+=grad
                for j in range(2):dw[o][j]+=grad*payload['inputs'][i][j]
        torch.testing.assert_close(torch.tensor(actual['updated']['layers.learn.weight']),torch.tensor([[w[o][j]-.02*dw[o][j] for j in range(2)] for o in range(2)]),rtol=3e-5,atol=3e-6)
        assert actual['updated']['layers.learn.bias']==pytest.approx([b[o]-.02*db[o] for o in range(2)])
    else:
        table=state['layers.bag.weight'].tolist()
        for i in positions:
            start,end=payload['offsets'][i:i+2]
            expected.append([sum(table[payload['indices'][j]][o]*payload['weights'][j] for j in range(start,end) if payload['indices'][j]!=0) for o in range(2)])
    torch.testing.assert_close(torch.tensor(actual['outputs']),torch.tensor(expected),rtol=3e-5,atol=3e-6)
