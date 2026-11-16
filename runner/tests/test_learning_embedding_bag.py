"""Direct executor tests; shared evidence dispatch is a later publication seam."""
import copy
import pytest
from runner.learning_embedding_bag import execute
from runner.embedding_bag_configuration import DEFAULTS


def submission(layout='packed',mode='sum',weighted=False,terminal=False,**changes):
    source=dict(layout=layout,maximum_items_per_bag=3,per_sample_weights=weighted,include_last_offset=terminal)
    config=dict(DEFAULTS,num_embeddings=4,embedding_dim=2,layout=layout,mode=mode,per_sample_weights=weighted,include_last_offset=terminal,**changes)
    roles=['indices']+(['offsets'] if layout=='packed' else [])+(['per_sample_weights'] if weighted else [])
    return dict(schemaVersion=1,kind='embeddingBagGraph',graph=dict(schemaVersion=2,name='Groups',nodes=[dict(id='s',type='bag_input',name='Rows',x=0,y=0,config=source),dict(id='b',type='embedding_bag',name='Combine',x=1,y=0,config=config),dict(id='o',type='output',name='Result',x=2,y=0,config={})],edges=[dict(id=r,source='s',sourcePort=r,target='b',targetPort=r) for r in roles]+[dict(id='out',source='b',sourcePort='out',target='o',targetPort='in')]))


def tensor(shape,values):return dict(shape=shape,values=values)
def fixture(weighted=False,layout='packed'):
    inputs=dict(sourceRows=tensor([3,3],[1,2,1,0,0,0,3,2,0]),table=tensor([4,2],[0,0,2,-1,1,3,-2,4]),upstream=tensor([3,2],[1,2,3,4,5,6]))
    if layout=='packed':inputs['lengths']=tensor([3],[3,0,1])
    if weighted:inputs['sourceWeights']=tensor([3,3],[.5,2,-1,9,8,7,2,3,4])
    return dict(id='private',inputs=inputs)


def measure(s,f,stop=lambda:False):
    evidence=dict(maximumTensorElements=0,fixtures=[])
    execute(s,[f],evidence,stop,lambda x:(x['shape'],x['values']))
    return evidence

@pytest.mark.parametrize('terminal',[False,True])
@pytest.mark.parametrize('mode,expected,gradient',[('sum',[5,1,0,0,-2,4],[0,0,2,4,1,2,5,6]),('mean',[5/3,1/3,0,0,-2,4],[0,0,2/3,4/3,1/3,2/3,5,6]),('max',[2,3,0,0,-2,4],[0,0,1,0,0,2,5,6])])
def test_real_packed_modes_repeat_and_empty(mode,expected,gradient,terminal):
    t=measure(submission(mode=mode,terminal=terminal),fixture())['fixtures'][0]['tensors']
    assert t['output']['shape']==[3,2]
    assert t['output']['values']==pytest.approx(expected)
    assert t['tableGradient']['values']==pytest.approx(gradient)
    assert not {'inputGradient','indicesGradient','offsetsGradient'} & t.keys()

@pytest.mark.parametrize('layout',['fixed','packed'])
def test_real_weight_gradient_and_dormant_cells(layout):
    t=measure(submission(layout=layout,weighted=True),fixture(weighted=True,layout=layout))['fixtures'][0]['tensors']
    assert t['output']['values'][:2]==pytest.approx([1,6.5])
    assert t['sourceWeightsGradient']['values'][:3]==pytest.approx([0,7,0])
    if layout=='packed':
        assert t['sourceWeightsGradient']['values'][3:]==pytest.approx([0,0,0,14,0,0])
        assert t['tableGradient']['values']==pytest.approx([0,0,-.5,-1,2,4,10,12])


def test_all_empty_packet_and_source_weights_remain_real():
    f=fixture(True);f['inputs']['lengths']['values']=[0,0,0]
    t=measure(submission(weighted=True,terminal=True),f)['fixtures'][0]['tensors']
    assert t['output']['values']==[0]*6
    assert t['sourceWeightsGradient']['values']==[0]*9
    assert t['tableGradient']['values']==[0]*8


def test_padding_norm_and_frozen_table_measurements():
    t=measure(submission(padding_idx=1,max_norm=1.,freeze=True),fixture())['fixtures'][0]['tensors']
    assert 'tableGradient' not in t
    assert t['output']['values'][:2]==pytest.approx([1/(10**.5+1e-7),3/(10**.5+1e-7)])
    assert t['tableAfterForward']['values'][2:4]==pytest.approx([2/(5**.5+1e-7),-1/(5**.5+1e-7)])

@pytest.mark.parametrize('mutation',[lambda s:s.update(expected={}),lambda s:s['graph']['edges'][0].update(sourcePort='offsets'),lambda s:s['graph']['nodes'][0].update(type='input'),lambda s:s['graph']['nodes'][0]['config'].update(include_last_offset=True),lambda s:s['graph']['nodes'].append(copy.deepcopy(s['graph']['nodes'][0]))])
def test_authored_answers_wrong_roles_and_duplicate_sources_reject(mutation):
    s=submission();mutation(s)
    with pytest.raises(ValueError):measure(s,fixture())

@pytest.mark.parametrize('role,value',[('sourceRows',1.5),('sourceRows',4),('sourceRows',True),('sourceWeights',float('nan')),('lengths',4),('lengths',1.5)])
def test_every_source_cell_checked_including_dormant(role,value,monkeypatch):
    f=fixture(True);f['inputs'][role]['values'][-1]=value
    def forbidden(*a,**k):raise AssertionError('allocated before source validation')
    monkeypatch.setattr('torch.tensor',forbidden)
    with pytest.raises(ValueError):measure(submission(weighted=True),f)


def test_cancellation_and_budget_before_execution():
    assert not measure(submission(),fixture(),lambda:True)['fixtures']
    e=dict(maximumTensorElements=0,fixtures=[])
    execute(submission(),[fixture()],e,lambda:e['maximumTensorElements']>1,lambda x:(x['shape'],x['values']))
    assert e['maximumTensorElements']>1 and not e['fixtures']

@pytest.mark.parametrize('mode',['sum','mean','max'])
@pytest.mark.parametrize('terminal',[False,True])
def test_all_empty_unweighted_modes(mode,terminal):
    f=fixture();f['inputs']['lengths']['values']=[0,0,0]
    t=measure(submission(mode=mode,terminal=terminal),f)['fixtures'][0]['tensors']
    assert t['output']['values']==[0]*6
    assert t['tableGradient']['values']==[0]*8

@pytest.mark.parametrize('mode',['sum','mean','max'])
def test_seeded_fixed_forward_matches_independent_row_loop(mode):
    import random
    for seed in (3,17,91):
        rng=random.Random(seed)
        f=fixture(layout='fixed')
        rows=[rng.randrange(4) for _ in range(9)]
        table=[rng.randint(-8,8)/4 for _ in range(8)]
        f['inputs']['sourceRows']['values']=rows;f['inputs']['table']['values']=table
        expected=[]
        for r in range(3):
            for c in range(2):
                values=[table[rows[r*3+j]*2+c] for j in range(3)]
                expected.append(max(values) if mode=='max' else sum(values)/(3 if mode=='mean' else 1))
        t=measure(submission(layout='fixed',mode=mode),f)['fixtures'][0]['tensors']
        assert t['output']['values']==pytest.approx(expected)


def test_labels_layout_and_ids_do_not_change_semantics():
    s=submission();f=fixture();expected=measure(s,f)
    for node in s['graph']['nodes']:
        old=node['id'];node['id']='renamed-'+old;node['name']='My label';node['x']+=100
        for edge in s['graph']['edges']:
            for k in ('source','target'):
                if edge[k]==old:edge[k]=node['id']
    s['graph']['nodes'].reverse();s['graph']['edges'].reverse()
    assert measure(s,f)==expected


def test_sparse_table_is_measured_dense_without_optimizer_claim():
    result=measure(submission(sparse=True),fixture())['fixtures'][0]['tensors']
    assert result['tableGradient']['values']==[0,0,2,4,1,2,5,6]
