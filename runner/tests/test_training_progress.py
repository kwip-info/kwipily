import math
import pytest
from runner.training_progress import TrainingProgress, validate_policy


def test_weighted_intervals_and_distinct_accumulated_updates():
    events=[];clock=[0.]
    progress=TrainingProgress('owner',events.append,dict(everyBatches=2,everySeconds=1),clock=lambda:clock[0])
    progress.batch(epoch=1,epoch_batch=1,epoch_batches=3,numerator=6,denominator=2,updated=False)
    clock[0]=1.1
    progress.batch(epoch=1,epoch_batch=2,epoch_batches=3,numerator=3,denominator=1,updated=True)
    progress.batch(epoch=1,epoch_batch=3,epoch_batches=3,numerator=8,denominator=4,updated=True)
    assert [x['trainingLoss'] for x in events]==[3,3,2]
    assert [x['updates'] for x in events]==[0,1,2]
    assert [x['batches'] for x in events]==[1,2,3]
    progress.validation(1,2.4,1.2,.5)
    assert events[-1]['kind']=='validation' and events[-1]['validationLoss']==1.2


def test_cadence_time_and_terminal_flush_and_sum():
    events=[];clock=[0.]
    p=TrainingProgress('owner',events.append,dict(everyBatches=100,everySeconds=2),clock=lambda:clock[0],reduction='sum')
    for batch,time in [(1,0),(2,1),(3,2),(4,2.1)]:
        clock[0]=time;p.batch(epoch=1,epoch_batch=batch,epoch_batches=4,numerator=2,denominator=8,updated=True)
    assert [e['batches'] for e in events]==[1,3,4]
    assert [e['trainingLoss'] for e in events]==[2,4,2]


@pytest.mark.parametrize('value',[None,{},dict(everyBatches=True,everySeconds=1),dict(everyBatches=1,everySeconds=math.nan),dict(everyBatches=0,everySeconds=1),dict(everyBatches=1,everySeconds=0)])
def test_strict_policy(value):
    with pytest.raises(ValueError):validate_policy(value)


def test_engine_progress_before_epoch_and_reporting_does_not_change_weights(tmp_path):
    import torch
    from runner.engine import run
    from runner.tests.test_runner import graph
    events=[]
    run(graph(),tmp_path/'a',epochs=1,batch_size=64,device='cpu',emit=events.append,training_progress=dict(everyBatches=1,everySeconds=1))
    run(graph(),tmp_path/'b',epochs=1,batch_size=64,device='cpu',training_progress=dict(everyBatches=1000,everySeconds=60))
    assert next(i for i,e in enumerate(events) if e['event']=='trainingProgress')<next(i for i,e in enumerate(events) if e['event']=='epoch')
    a=torch.load(tmp_path/'a/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'b/checkpoint.pt',weights_only=True)
    for key,value in a['modelState'].items():assert torch.equal(value,b['modelState'][key])
    import json
    assert json.loads((tmp_path/'a/metrics.json').read_bytes())==json.loads((tmp_path/'b/metrics.json').read_bytes())
    final=[e for e in events if e['event']=='trainingProgress'][-1]
    assert final['kind']=='validation' and final['batches']==13 and final['updates']==13


def test_masked_sequence_uneven_batches_and_accumulation_use_actual_denominators(tmp_path):
    import json
    from runner.engine import run
    from runner.tests.test_sequence_training import fixture
    from runner.tests.test_sequence_objective import recipe
    from runner import gradient_policy
    prepared,model,target,_,_=fixture(tmp_path)
    events=[]
    policy=dict(schemaVersion=1,configurationVersion=1,componentID='gradient_policy',config=gradient_policy.canonical_configuration({'accumulation_steps':3}))
    run(model,tmp_path/'sequence',epochs=1,device='cpu',batch_size=7,prepared=prepared,objective_recipe=recipe(),target_mapping=target,gradient_policy=policy,emit=events.append,training_progress=dict(everyBatches=1,everySeconds=1))
    rows=[e for e in events if e['event']=='trainingProgress' and e['kind']=='training']
    assert [r['updates'] for r in rows]==[0,0,1,1,1,2,3]
    # Padding varies by window; uneven batches use actual valid-token counts.
    targets=prepared[0][1]['target']
    assert [r['denominator'] for r in rows]==[int((targets[start:start+7]!=-100).sum()) for start in range(0,48,7)]
    epoch=json.loads((tmp_path/'sequence/metrics.json').read_bytes())[0]
    assert sum(r['numerator'] for r in rows)/sum(r['denominator'] for r in rows)==pytest.approx(epoch['trainLoss'])
    assert rows[-1]['batches']==7 and rows[-1]['updates']==3


def test_reporting_policy_admitted_only_by_training_authorities():
    import json
    from runner.worker_authorization import validate_phase_arguments
    policy=json.dumps(dict(everyBatches=5,everySeconds=.5))
    assert validate_phase_arguments('training',['--training-progress-json',policy])['--training-progress-json']==policy
    with pytest.raises(ValueError):validate_phase_arguments('preparation',['--training-progress-json',policy])
    with pytest.raises(ValueError):validate_phase_arguments('training',['--training-progress-json','{}'])
