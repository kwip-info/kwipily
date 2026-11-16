"""Real prepared packet training publishes bounded derivatives without changing updates."""
import json
import pytest
import torch
from runner import engine
from runner.tests.test_bag_source_graph import registry, graph
from runner.tests.test_prepared_bag_loader import fixture, load
from runner.tests.test_gradient_engine import policy, preserve
from runner.tests.test_bag_requested_gradients import q

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('capture',[False,True])
def test_saved_bag_derivatives_match_weights_and_keep_training_state(tmp_path,registry,device,capture):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    source=fixture(tmp_path/'source');prepared=load(source)
    value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2
    request=q();request['row_count']=4
    common=dict(registry=registry,prepared=prepared,device=device,epochs=1,batch_size=3)
    output=tmp_path/'run'
    if not capture:
        with pytest.raises(ValueError,match='[Ee]nable derivative'):
            engine.run(value,output,gradient_policy=policy(requests=[request]),derivative_capture=dict(schemaVersion=1,values=False),**common)
        assert not (output/'checkpoint.pt').exists()
        return
    engine.run(value,output,gradient_policy=policy(requests=[request]),derivative_capture=dict(schemaVersion=1,values=True),**common)
    manifest=json.loads((output/'manifest.json').read_bytes())
    result=json.loads((output/'derivative-results.json').read_bytes())
    assert manifest['derivativeResults']['schemaVersion']==result['schemaVersion']==2
    assert result['modelInput']['kind']=='bagPacket'
    selected=result['requests'][0]
    packet=prepared[0][2];starts=packet.offsets.tolist()[:5]
    assert selected['sample']['bagItemCounts']==[b-a for a,b in zip(starts,starts[1:])]
    targets=selected['targets'];assert [x['cause'] for x in targets[:2]]==['integerInput','integerInput']
    state=torch.load(output/'checkpoint.pt',weights_only=True,map_location='cpu')
    weights=state['modelState']['layers.bag.weight']
    expected=[0. if int(i)==0 else sum(weights[int(i)].tolist()) for i in packet.indices[:starts[-1]]]
    if capture:assert targets[2]['values']==pytest.approx(expected,rel=2e-5,abs=2e-5)
    else:assert targets[2].get('values') is None
    assert (output/'gradient_policy.py').exists()
    if device=='cpu':
        import shutil,subprocess,sys
        isolated=tmp_path/'relocated';isolated.mkdir()
        for name in ('model.py','gradient_policy.py'):shutil.copyfile(output/name,isolated/name)
        torch.save(dict(state=state['modelState'],indices=packet.indices,offsets=packet.offsets,weights=packet.per_sample_weights,expected=expected),isolated/'input.pt')
        (isolated/'request.json').write_text(json.dumps(request))
        program="""import sys;sys.path.insert(0,'.')
import torch,json,model,gradient_policy
saved=torch.load('input.pt',weights_only=True,map_location='cpu')
m=model.Model();m.load_state_dict(saved['state'],strict=True)
x=model.EmbeddingBagPacket('packed',saved['indices'],saved['offsets'],saved['weights'],True)
# The standalone helper analyzes the selected observations, using its sibling packet class.
n=4;end=int(x.offsets[n]);selected=model.EmbeddingBagPacket('packed',x.indices[:end],x.offsets[:n+1],x.per_sample_weights[:end],True)
actual=gradient_policy.analyze(m,selected,json.load(open('request.json')),device='cpu')['targets'][2]['values']
torch.testing.assert_close(torch.tensor(actual),torch.tensor(saved['expected']),rtol=2e-5,atol=2e-5)
"""
        child=subprocess.run([sys.executable,'-I','-c',program],cwd=isolated,capture_output=True,text=True,timeout=30)
        assert child.returncode==0,child.stdout+child.stderr
    engine.run(value,tmp_path/'control',gradient_policy=policy(),**common)
    control=torch.load(tmp_path/'control/checkpoint.pt',weights_only=True,map_location='cpu')
    for key,actual in state['modelState'].items():torch.testing.assert_close(actual,control['modelState'][key],rtol=0,atol=0)
    assert state['optimizerState']['param_groups']==control['optimizerState']['param_groups']
    for key,entry in state['optimizerState']['state'].items():
        for field,actual in entry.items():torch.testing.assert_close(actual,control['optimizerState']['state'][key][field],rtol=0,atol=0)


def test_invalid_bag_derivative_role_refuses_before_training(tmp_path,registry):
    prepared=load(fixture(tmp_path/'source'));value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2
    request=q();request['targets'][0]['port']='out';events=[]
    with pytest.raises(ValueError):engine.run(value,tmp_path/'bad',registry=registry,prepared=prepared,device='cpu',epochs=1,gradient_policy=policy(requests=[request]),derivative_capture=dict(schemaVersion=1,values=True),emit=events.append)
    assert not any(e['event'] in ('started','epoch','completed') for e in events)
    assert not (tmp_path/'bad/checkpoint.pt').exists()
