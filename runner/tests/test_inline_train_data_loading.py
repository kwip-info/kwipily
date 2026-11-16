"""Train-owned loading normalizes to unchanged worker recipes and replay flags."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
import torch
from runner.workflow_code import generate_package,execute_snapshot,pytorch_arguments,validate_snapshot
from runner.tests.test_workflow_code import training_snapshot
from runner.tests.test_dataset_loader_configuration import recipe
from runner.tests.test_checkpoint_lifecycle import assert_nested_equal


def inline_snapshot():
    snapshot=training_snapshot()
    dataset=recipe(kind='subset',indices=[4,1,0,7,3])
    loader=recipe('data_loader',batch_size=2,shuffle=True,seed=99,collation='fixed_stack')
    snapshot['document']['nodes'][-1]['payload']['config']={'datasetAccess':copy.deepcopy(dataset['config']),'batching':copy.deepcopy(loader['config'])}
    training=snapshot['trainings'][0]
    training['recipes'].update(dataset_recipe=dataset,loader_recipe=loader)
    training['recipeSources'].update(dataset_recipe={'domain':'control','localID':'train'},loader_recipe={'domain':'control','localID':'train'})
    training['settings']['batchSize']=2
    return snapshot


def test_inline_origin_projection_keeps_exact_versioned_flags_and_no_implicit_source():
    snapshot=validate_snapshot(inline_snapshot());training=snapshot['trainings'][0]
    arguments=pytorch_arguments(training,'graph.json','registry.json',{'manifest':'prepared.json','manifestSHA256':'a'*64,'graphDigest':'b'*64},'output',{})
    assert json.loads(arguments[arguments.index('--dataset-recipe-json')+1])==training['recipes']['dataset_recipe']
    assert json.loads(arguments[arguments.index('--loader-recipe-json')+1])==training['recipes']['loader_recipe']
    assert arguments[arguments.index('--batch-size')+1]=='2'
    assert '--dataset-csv' not in arguments
    assert training['recipeSources']['dataset_recipe']==training['recipeSources']['loader_recipe']=={'domain':'control','localID':'train'}


def test_real_inline_train_export_replay_and_exact_continuation(tmp_path):
    before=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        snapshot=inline_snapshot();root=Path(__file__).resolve().parents[2]
        bindings={'sources':{'source':str(root/'examples/dataflow/observations.csv')}}
        execute_snapshot(snapshot,bindings,tmp_path/'full',lambda _:None)
        manifest=json.loads((tmp_path/'full/training-000/manifest.json').read_text())
        assert manifest['dataLoading']['originalTrainingRows']==24
        assert manifest['dataLoading']['selectedRows']==5 and manifest['dataLoading']['retainedRows']==5
        assert manifest['dataLoading']['batchSize']==2 and manifest['dataLoading']['batchesPerEpoch']==3
        assert manifest['evaluation']['count']==8
        assert manifest['settings']['datasetRecipe']==snapshot['trainings'][0]['recipes']['dataset_recipe']
        assert manifest['settings']['loaderRecipe']==snapshot['trainings'][0]['recipes']['loader_recipe']
        parent=copy.deepcopy(snapshot);parent['trainings'][0]['settings']['epochs']=1
        execute_snapshot(parent,bindings,tmp_path/'parent',lambda _:None)
        child=copy.deepcopy(snapshot);child['trainings'][0]['trainingMode']='exactResume'
        child['document']['nodes'][-1]['payload']['config']['trainingMode']='exactResume'
        child_bindings={**bindings,'checkpoints':{'train':str(tmp_path/'parent/training-000/checkpoint.pt')}}
        execute_snapshot(child,child_bindings,tmp_path/'resumed',lambda _:None)
        full=torch.load(tmp_path/'full/training-000/checkpoint.pt',weights_only=True)
        resumed=torch.load(tmp_path/'resumed/training-000/checkpoint.pt',weights_only=True)
        for key in ('modelState','optimizerState','rngState','dataLoadingState'):
            assert_nested_equal(full[key],resumed[key])
        changed=copy.deepcopy(child);changed['trainings'][0]['recipes']['loader_recipe']['config']['seed']=100
        with pytest.raises(subprocess.CalledProcessError):execute_snapshot(changed,child_bindings,tmp_path/'changed',lambda _:None)
        package=generate_package(snapshot);destination=tmp_path/'package';destination.mkdir()
        for name,content in package['files'].items():(destination/name).write_text(content)
        runtime=destination/'runner';runtime.mkdir()
        for source in root.joinpath('runner').iterdir():
            if source.is_file() and (source.suffix in ('.py','.json') or source.name=='laya-LICENSE.txt'):shutil.copyfile(source,runtime/source.name)
        external=tmp_path/'bindings.json';external.write_text(json.dumps(bindings))
        elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
        result=subprocess.run([sys.executable,str(destination/'workflow.py'),'--bindings',str(external),'--output',str(tmp_path/'replayed')],cwd=elsewhere,text=True,capture_output=True,timeout=90)
        assert result.returncode==0,result.stderr+result.stdout
        replayed=torch.load(tmp_path/'replayed/training-000/checkpoint.pt',weights_only=True)
        assert_nested_equal(full['modelState'],replayed['modelState'])
        assert (tmp_path/'full/training-000/metrics.json').read_bytes()==(tmp_path/'replayed/training-000/metrics.json').read_bytes()
    finally:torch.use_deterministic_algorithms(before,warn_only=warn)
