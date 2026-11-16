import json
from pathlib import Path
import torch
import pytest
from runner.engine import run
from runner.tests.test_runner import graph


def state(path):return torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)

def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested_equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:assert a==b


def test_exact_resume_matches_uninterrupted_cpu_weights_optimizer_and_metrics(tmp_path):
    full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    def consume_rng(event):
        if event['event']=='epoch':torch.rand(7)
    run(graph(),full,epochs=12,device='cpu',batch_size=32,emit=consume_rng)
    run(graph(),first,epochs=5,device='cpu',batch_size=32,emit=consume_rng)
    original=(first/'checkpoint.pt').read_bytes();events=[]
    run(graph(),resumed,epochs=12,device='cpu',batch_size=32,resume_checkpoint=first/'checkpoint.pt',emit=lambda e:(events.append(e),consume_rng(e)))
    a,b=state(full),state(resumed)
    for name in ('modelState','optimizerState','rngState'):assert_nested_equal(a[name],b[name])
    assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
    manifest=json.loads((resumed/'manifest.json').read_text());parent=json.loads((first/'manifest.json').read_text())
    assert manifest['lineage']['mode']=='exactResume' and manifest['lineage']['parentRunID']==parent['runID']
    assert manifest['runID']!=parent['runID'] and events[0]['initialEpoch']==5
    assert (first/'checkpoint.pt').read_bytes()==original


def _independent_prepared_captures(tmp_path):
    from runner.dataflow_cli import run_dataflow
    from runner.prepared_data import load_prepared
    from runner.workload_policy import LEGACY
    root=Path(__file__).resolve().parents[2]
    captures=[]
    for name in ('first-preparation','second-preparation'):
        folder=tmp_path/name
        run_dataflow(root/'examples/dataflow/scaler.json',{'source':str(root/'examples/dataflow/observations.csv')},folder,lambda _:None,workload_policy=LEGACY.to_document())
        captures.append(load_prepared(folder/'manifest.json','training','table','heldout','table',['feature','second'],'group'))
    return captures


def test_exact_prepared_resume_ignores_only_independently_verified_storage_layout(tmp_path):
    first,second=_independent_prepared_captures(tmp_path)
    assert first[1]['preparation']['manifestSHA256']!=second[1]['preparation']['manifestSHA256']
    assert all(torch.equal(a,b) for a,b in zip(first[0],second[0]))
    parent=tmp_path/'parent';full=tmp_path/'full';resumed=tmp_path/'resumed'
    run(graph(),parent,epochs=2,device='cpu',batch_size=8,prepared=first)
    original=(parent/'checkpoint.pt').read_bytes()
    run(graph(),full,epochs=4,device='cpu',batch_size=8,prepared=first)
    run(graph(),resumed,epochs=4,device='cpu',batch_size=8,prepared=second,resume_checkpoint=parent/'checkpoint.pt')
    a,b=state(full),state(resumed)
    for name in ('modelState','optimizerState','rngState'):assert_nested_equal(a[name],b[name])
    assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
    assert (parent/'checkpoint.pt').read_bytes()==original


@pytest.mark.parametrize('change',['tensor','source','split','graph','trainingVersion','validationVersion','fittedState','endpoint','unknownField'])
def test_prepared_resume_changed_scientific_identity_still_refused(tmp_path,change):
    import copy
    first,second=_independent_prepared_captures(tmp_path)
    parent=tmp_path/'parent'
    run(graph(),parent,epochs=2,device='cpu',batch_size=8,prepared=first)
    original=(parent/'checkpoint.pt').read_bytes()
    data=tuple(t.clone() for t in second[0]);provenance=copy.deepcopy(second[1])
    preparation=provenance['preparation'];different='f'*64
    if change=='tensor':data[0][0,0]+=.125
    elif change=='source':provenance['sourceHash']=different
    elif change=='split':provenance['splitHash']=different
    elif change=='graph':preparation['graphDigest']=different
    elif change=='trainingVersion':preparation['trainingDataVersion']=different
    elif change=='validationVersion':preparation['validationDataVersion']=different
    elif change=='fittedState':preparation['fittedStateDigests']=[different]
    elif change=='endpoint':preparation['trainingNode']='different-training'
    else:preparation['futureSemanticField']=True
    with pytest.raises(ValueError,match='Exact resume requires'):
        run(graph(),tmp_path/'refused',epochs=4,device='cpu',batch_size=8,prepared=(data,provenance,second[2]),resume_checkpoint=parent/'checkpoint.pt')
    assert not (tmp_path/'refused/manifest.json').exists()
    assert (parent/'checkpoint.pt').read_bytes()==original


def test_prepared_resume_parent_transport_pin_cannot_disagree_with_provenance(tmp_path):
    first,second=_independent_prepared_captures(tmp_path)
    parent=tmp_path/'parent'
    run(graph(),parent,epochs=2,device='cpu',batch_size=8,prepared=first)
    def alter(checkpoint,manifest):
        checkpoint['contract']['data']['preparation']['manifestSHA256']='f'*64
        manifest['checkpoint']['contract']=checkpoint['contract']
    rewrite_checkpoint(parent,alter)
    with pytest.raises(ValueError,match='preparation contract differs from parent provenance'):
        run(graph(),tmp_path/'refused',epochs=4,device='cpu',batch_size=8,prepared=second,resume_checkpoint=parent/'checkpoint.pt')
    assert not (tmp_path/'refused/manifest.json').exists()


@pytest.mark.parametrize('change', [{'batch_size':16},{'learning_rate':.01},{'split_seed':43},{'seed':43},{'epochs':2}])
def test_resume_incompatible_recipe_or_nonadvancing_epoch_rejected(tmp_path,change):
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu',batch_size=32)
    options={'epochs':4,'device':'cpu','batch_size':32,**change}
    with pytest.raises(ValueError):run(graph(),tmp_path/'bad',resume_checkpoint=first/'checkpoint.pt',**options)
    assert not (tmp_path/'bad/manifest.json').exists()


def test_finetune_is_fresh_optimizer_epoch_rng_and_new_data(tmp_path):
    first=tmp_path/'first';next_run=tmp_path/'fine';run(graph(),first,epochs=3,device='cpu',batch_size=32)
    run(graph(),next_run,epochs=1,device='cpu',batch_size=1000,seed=9,split_seed=17,learning_rate=.01,finetune_checkpoint=first/'checkpoint.pt')
    checkpoint=state(next_run);manifest=json.loads((next_run/'manifest.json').read_text())
    assert checkpoint['epoch']==1 and checkpoint['seed']==9
    assert all(v['step'].item()==1 for v in checkpoint['optimizerState']['state'].values())
    assert manifest['lineage']['mode']=='weightsOnlyFineTune'
    assert manifest['provenance']['splitSeed']==17


def test_tampered_checkpoint_and_legacy_are_not_resumable(tmp_path):
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu')
    with (first/'checkpoint.pt').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='integrity'):run(graph(),tmp_path/'bad',epochs=4,device='cpu',resume_checkpoint=first/'checkpoint.pt')
    legacy=tmp_path/'legacy';legacy.mkdir();torch.save({'modelState':{}},legacy/'checkpoint.pt');(legacy/'manifest.json').write_text('{"status":"completed"}')
    with pytest.raises(ValueError,match='versioned'):run(graph(),tmp_path/'bad2',epochs=4,device='cpu',finetune_checkpoint=legacy/'checkpoint.pt')


def rewrite_checkpoint(path,mutate):
    import hashlib
    checkpoint=state(path);manifest=json.loads((path/'manifest.json').read_text())
    mutate(checkpoint,manifest)
    torch.save(checkpoint,path/'checkpoint.pt');raw=(path/'checkpoint.pt').read_bytes()
    manifest['checkpoint'].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    (path/'manifest.json').write_text(json.dumps(manifest))


def test_adam_hyperparameters_cannot_disagree_with_contract(tmp_path):
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu')
    rewrite_checkpoint(first,lambda c,m:c['optimizerState']['param_groups'][0].update(lr=.5))
    with pytest.raises(ValueError,match='hyperparameters'):run(graph(),tmp_path/'next',epochs=3,device='cpu',resume_checkpoint=first/'checkpoint.pt')


def test_runtime_graph_and_symlink_constraints(tmp_path):
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu')
    altered=graph();altered['nodes'][3]['type']='tanh'
    with pytest.raises(ValueError,match='architecture'):run(altered,tmp_path/'bad',epochs=3,device='cpu',resume_checkpoint=first/'checkpoint.pt')
    renamed=graph();renamed['name']='Cosmetic rename';renamed['nodes'][0]['x']=123
    run(renamed,tmp_path/'renamed',epochs=3,device='cpu',resume_checkpoint=first/'checkpoint.pt')
    def runtime_change(c,m):
        c['contract']['runtime']['torch']='other-version';m['checkpoint']['contract']=c['contract']
    rewrite_checkpoint(first,runtime_change)
    with pytest.raises(ValueError,match='runtime'):run(graph(),tmp_path/'runtime',epochs=3,device='cpu',resume_checkpoint=first/'checkpoint.pt')
    linked=tmp_path/'linked';linked.mkdir();(linked/'manifest.json').symlink_to(first/'manifest.json');(linked/'checkpoint.pt').symlink_to(first/'checkpoint.pt')
    with pytest.raises(ValueError,match='regular files'):run(graph(),tmp_path/'linkrun',epochs=3,device='cpu',resume_checkpoint=linked/'checkpoint.pt')


def test_cli_exact_resume_and_cancel_preserve_parent(tmp_path):
    import subprocess,sys
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu',batch_size=32)
    before={p.name:p.read_bytes() for p in first.iterdir() if p.is_file()}
    model=tmp_path/'graph.json';model.write_text(json.dumps(graph()));out=tmp_path/'cancelled'
    main=Path(__file__).resolve().parents[1]/'main.py'
    args=[sys.executable,str(main),'--graph',str(model),'--output',str(out),'--device','cpu','--batch-size','32','--epochs','10000','--resume-checkpoint',str(first/'checkpoint.pt')]
    process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    started=json.loads(process.stdout.readline());assert started['initialEpoch']==2
    process.terminate();stdout,stderr=process.communicate(timeout=10)
    assert process.returncode==130 and not (out/'manifest.json').exists()
    assert not any(json.loads(line)['event']=='completed' for line in stdout.splitlines())
    assert before=={p.name:p.read_bytes() for p in first.iterdir() if p.is_file()}
    result=subprocess.run(args+['--finetune-checkpoint',str(first/'checkpoint.pt')],capture_output=True,text=True)
    assert result.returncode==1 and 'not allowed' in result.stdout


def test_optimizer_parameter_order_cannot_silently_swap_branch_moments(tmp_path):
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu')
    reordered=graph();reordered['edges'][0],reordered['edges'][1]=reordered['edges'][1],reordered['edges'][0]
    with pytest.raises(ValueError,match='schedule'):run(reordered,tmp_path/'bad',epochs=3,device='cpu',resume_checkpoint=first/'checkpoint.pt')
    run(reordered,tmp_path/'fine',epochs=1,device='cpu',finetune_checkpoint=first/'checkpoint.pt')


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable on this host')
def test_mps_finetune_and_explicit_cpu_only_exact_resume(tmp_path):
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu')
    with pytest.raises(ValueError,match='CPU only'):run(graph(),tmp_path/'not-exact',epochs=3,device='mps',resume_checkpoint=first/'checkpoint.pt')
    run(graph(),tmp_path/'fine-mps',epochs=1,device='mps',finetune_checkpoint=first/'checkpoint.pt')
    manifest=json.loads((tmp_path/'fine-mps/manifest.json').read_text())
    assert manifest['runtime']['device']=='mps' and manifest['checkpoint']['contract']['guarantee']=='weightsOnly'


def test_checkpoint_cannot_silently_cast_model_dtype(tmp_path):
    first=tmp_path/'first';run(graph(),first,epochs=2,device='cpu')
    def change_dtype(c,m):
        key=next(iter(c['modelState']));c['modelState'][key]=c['modelState'][key].double()
    rewrite_checkpoint(first,change_dtype)
    with pytest.raises(ValueError,match='dtype'):run(graph(),tmp_path/'next',epochs=3,device='cpu',resume_checkpoint=first/'checkpoint.pt')


def test_resumed_export_loads_existing_modelstate_format(tmp_path):
    from runner.engine import GraphModel
    first=tmp_path/'first';out=tmp_path/'next';run(graph(),first,epochs=2,device='cpu')
    run(graph(),out,epochs=3,device='cpu',resume_checkpoint=first/'checkpoint.pt')
    namespace={};exec((out/'model.py').read_text(),namespace)
    exported=namespace['Model']();reference=GraphModel(graph());weights=state(out)['modelState']
    exported.load_state_dict(weights);reference.load_state_dict(weights)
    x=torch.tensor([[.2,.7],[-.5,1.2]])
    assert torch.equal(exported(x),reference(x))
