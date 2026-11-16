"""Independent row selection/alignment oracles for real native data sources."""
import copy
import pytest
from runner.learning_data_loading import execute
from runner.dataset_loader_configuration import DATASET_DEFAULTS,LOADER_DEFAULTS

TASKS=['dataset-indexed-observations','dataset-aligned-tensors','dataset-selected-view','dataset-finite-stream','loader-complete-and-short-groups','loader-repeatable-mixing','loader-aligned-sequences']
def tensor(shape,values):return dict(shape=shape,values=values)
def submission(task):
    dataset=dict(DATASET_DEFAULTS);loader=dict(LOADER_DEFAULTS,batch_size=4,collation='fixed_stack')
    if task==TASKS[1]:dataset.update(kind='tensor')
    if task==TASKS[2]:dataset.update(kind='subset',indices=[4,1,0])
    if task==TASKS[3]:dataset.update(kind='boundedIterable',max_rows=5)
    if task==TASKS[5]:loader.update(batch_size=3,shuffle=True,collation='auto')
    if task==TASKS[6]:loader.update(collation='padded_sequence')
    return dict(schemaVersion=1,kind='dataLoadingLearningGraph',taskID=task,graph=dict(schemaVersion=2,name='Access rows',nodes=[dict(id='d',type='dataset_source',name='Rows',x=0,y=0,config=dataset),dict(id='l',type='data_loader',name='Groups',x=0,y=1,config=loader),dict(id='t',type='train',name='Train',x=1,y=0,config={})],edges=[dict(id='d-t',source='d',sourcePort='out',target='t',targetPort='dataset'),dict(id='l-t',source='l',sourcePort='out',target='t',targetPort='loader')]))
def fixture(task):
    rows=[v for i in range(9) for v in (i,-i)]
    f=dict(id='private',inputs=dict(sourceRows=tensor([9,2],rows),desiredRows=tensor([9,1],[i/2 for i in range(9)]),rowIDs=tensor([9],list(range(101,110))),seed=tensor([1],[17])))
    if task==TASKS[6]:f['inputs'].update(sourceRows=tensor([9,3,1],[v for i in range(9) for v in (i,0,-i)]),sourceLengths=tensor([9],[i%4 for i in range(9)]))
    return f

def measure(s,f,stop=lambda:False):
    evidence=dict(maximumTensorElements=0,fixtures=[])
    execute(s,[f],evidence,stop,lambda x:(x['shape'],x['values']))
    return evidence

@pytest.mark.parametrize('index,selected',[(0,list(range(9))),(1,list(range(9))),(2,[4,1,0]),(3,list(range(5)))])
def test_actual_source_native_kind_and_independent_ordered_membership(index,selected):
    task=TASKS[index];observed=measure(submission(task),fixture(task))['fixtures'][0]['tensors']
    assert observed['sourceKind']['values']==[index]
    assert observed['selectedRowIDs']['values']==[101+i for i in selected]
    assert observed['selectedInputs']['values']==[v for i in selected for v in (i,-i)]
    assert observed['selectedTargets']['values']==[i/2 for i in selected]

def test_real_short_last_group_and_independent_drop_last_comparison():
    task=TASKS[4];t=measure(submission(task),fixture(task))['fixtures'][0]['tensors']
    assert t['batchRowIDs']==dict(shape=[3,4],values=list(range(101,110))+[-1]*3)
    assert t['batchTargets']['values']==[i/2 for i in range(9)]+[0]*3
    assert t['batchLengths']['values']==[4,4,1]
    assert t['dropLastRowIDs']==dict(shape=[2,4],values=list(range(101,109)))

def test_real_shuffle_repeat_changed_seed_membership_and_alignment():
    task=TASKS[5];t=measure(submission(task),fixture(task))['fixtures'][0]['tensors']
    assert t['shuffleChecks']['values']==[1]*4
    assert t['selectionDifference']['values']==[0]*9
    assert t['alignmentDifference']['values']==[0]*27
    wrong=submission(task);wrong['graph']['nodes'][1]['config']['shuffle']=False
    assert measure(wrong,fixture(task))['fixtures'][0]['tensors']['shuffleChecks']['values']!=[1]*4

def test_actual_padded_sequence_values_lengths_and_empty_rows_aligned():
    task=TASKS[6];t=measure(submission(task),fixture(task))['fixtures'][0]['tensors']
    assert t['batchedInputs']==dict(shape=[3,4,3,1],values=[v for i in range(9) for v in (i,0,-i)]+[0]*9)
    assert t['sequenceLengths']['values']==[i%4 for i in range(9)]+[0]*3
    assert t['batchRowIDs']['values']==list(range(101,110))+[-1]*3

@pytest.mark.parametrize('mutation',[lambda s:s.update(expected={}),lambda s:s['graph']['edges'][0].update(targetPort='data'),lambda s:s['graph']['nodes'][0].update(type='input'),lambda s:s['graph']['nodes'][2]['config'].update(epochs=10)])
def test_wrong_wiring_or_authored_answers_refused_before_source(mutation,monkeypatch):
    task=TASKS[0];s=submission(task);mutation(s)
    monkeypatch.setattr('runner.learning_data_loading.make_source',lambda *a,**k:pytest.fail('source before admission'))
    with pytest.raises(ValueError):measure(s,fixture(task))

@pytest.mark.parametrize('role,value',[('sourceRows',float('nan')),('desiredRows',True),('rowIDs',101.5),('rowIDs',101)])
def test_malformed_private_row_cells_refuse_before_allocation(role,value,monkeypatch):
    task=TASKS[0];f=fixture(task);f['inputs'][role]['values'][-1]=value
    monkeypatch.setattr('torch.tensor',lambda *a,**k:pytest.fail('allocated before checking cells'))
    with pytest.raises(ValueError):measure(submission(task),f)

def test_cancel_before_execution():
    task=TASKS[0];assert not measure(submission(task),fixture(task),lambda:True)['fixtures']

def test_real_checkpoint_whole_bags_two_readers_continuation_not_helper_marker():
    task='loader-continue-grouping';s=submission(TASKS[0]);s['taskID']=task
    s['graph']['nodes'][1]['config'].update(shuffle=True,num_workers=2,prefetch_factor=1,timeout=5.,collation='whole_bags')
    f=fixture(TASKS[0]);f['inputs'].update(sourceRows=tensor([9,3],[v for i in range(9) for v in (i%5,(i+1)%5,(i+2)%5)]),sourceLengths=tensor([9],[i%4 for i in range(9)]),sourceWeights=tensor([9,3],[.5,1.,-.25]*9))
    measured=measure(s,f)['fixtures'][0]['tensors']
    assert measured['continuationChecks']['values']==[1]*4
    assert measured['readerPolicy']['values']==[2,1]
    assert measured['parentByteDifference']['values']==[0]
    assert measured['epochOrderDifference']==dict(shape=[9],values=[0]*9)

@pytest.mark.parametrize('changed',[{'batch_size':2},{'shuffle':False},{'seed':43}])
def test_real_continuation_cannot_grade_a_different_public_grouping_policy(changed):
    s=submission(TASKS[0]);s['taskID']='loader-continue-grouping'
    s['graph']['nodes'][1]['config'].update(shuffle=True,num_workers=2,prefetch_factor=1,timeout=5.,collation='whole_bags')
    s['graph']['nodes'][1]['config'].update(changed)
    f=fixture(TASKS[0]);f['inputs'].update(sourceRows=tensor([9,3],[v for i in range(9) for v in (i%5,(i+1)%5,(i+2)%5)]),sourceLengths=tensor([9],[i%4 for i in range(9)]),sourceWeights=tensor([9,3],[.5,1.,-.25]*9))
    measured=measure(s,f)['fixtures'][0]['tensors']
    assert measured['continuationChecks']['values'][1]==0
    assert measured['parentByteDifference']['values']==[0]
