import copy
import math
import pytest
import torch
from runner.named_inputs import validate_plan,convert_rows
from runner.loss_configuration import canonical_configuration
from runner.loss_objectives import create_objective,evaluate_objective
from runner.loss_partition import objective_partition_contribution,finish_objective_partition


def plan():
    tensor=dict(representation='tokens',dtype='int64',sampleShape=[3],lengthPolicy='error',paddingValue=1,vocabularySize=5)
    return dict(schemaVersion=2,inputs=[dict(tensor,name='tokens',nodeID='source',columns=['x'])],roles=[dict(tensor,name='target',columns=['y'])],maximumBatchBytes=1048576,sequenceObjective=dict(schemaVersion=1,inputName='tokens',targetName='target',lengthColumn='n',maskColumn='m'))


def recipe():return dict(schemaVersion=1,configurationVersion=1,componentID='loss_cross_entropy',config=canonical_configuration('loss_cross_entropy',{}))


def schema():return [dict(name=n,dtype=t,nullable=False) for n,t in [('x','int64Sequence'),('y','int64Sequence'),('n','int64'),('m','int64Sequence')]]


def test_mask_conversion_exact_valid_token_mean_gradients_and_batch_aggregation():
    p=plan();validate_plan(p)
    x,y=convert_rows(p,schema(),[[[2,3,1],[3,4,1],2,[1,1,0]],[[4,1,1],[2,1,1],1,[1,0,0]]])
    assert x['tokens'].tolist()==[[2,3,1],[4,1,1]]
    assert y['target'].tolist()==[[3,4,-100],[2,-100,-100]]
    logits=torch.tensor([[[.1,.2,.3,.4,.5],[1.,2.,3.,4.,5.],[9.,8.,7.,6.,5.]],[[.5,.4,.3,.2,.1],[5.,6.,7.,8.,9.],[1.,1.,1.,1.,1.]]],requires_grad=True)
    objective=create_objective(recipe())
    result=evaluate_objective(objective,logits,y)
    expected=torch.nn.functional.cross_entropy(logits.transpose(1,2),y['target'],ignore_index=-100)
    torch.testing.assert_close(result['scalar'],expected)
    result['scalar'].backward();assert torch.equal(logits.grad[y['target']==-100],torch.zeros_like(logits.grad[y['target']==-100]))
    parts=[objective_partition_contribution(objective,logits[i:i+1],{'target':y['target'][i:i+1]}) for i in range(2)]
    total=finish_objective_partition(parts,recipe())
    torch.testing.assert_close(total['scalar'],expected)
    assert total['metadata']['aggregationDenominator']==3 and total['metadata']['ignoredCount']==3
    assert math.exp(total['scalar'].item())==pytest.approx(math.exp(expected.item()))

@pytest.mark.parametrize('change',[
    lambda r:r[0].__setitem__(2,0),lambda r:r[0].__setitem__(3,[0,0,0]),
    lambda r:r[0].__setitem__(3,[1,0,1]),lambda r:r[0].__setitem__(3,[1,2,0]),
    lambda r:r[0].__setitem__(1,[4,3,1]),lambda r:r[0].__setitem__(0,[2,3,0]),
    lambda r:r[0].__setitem__(1,[3,5,1]),lambda r:r[0].__setitem__(2,True)])
def test_bad_masks_lengths_shift_padding_or_vocabulary_are_not_repaired(change):
    rows=[[[2,3,1],[3,4,1],2,[1,1,0]]];change(rows)
    with pytest.raises(ValueError):convert_rows(plan(),schema(),rows)


def test_sequence_contract_cannot_hide_mask_in_model_features_or_resize_targets():
    for mutation in [lambda p:p['inputs'][0].update(lengthPolicy='pad'),lambda p:p['roles'][0].update(sampleShape=[2]),lambda p:p['sequenceObjective'].update(maskColumn='x'),lambda p:p.update(schemaVersion=1),lambda p:p['sequenceObjective'].update(extra=True)]:
        p=plan();mutation(p)
        with pytest.raises(ValueError):validate_plan(p)
