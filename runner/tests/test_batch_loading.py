import pytest
import torch
from runner.dataset_sources import make_source
from runner.batch_loading import plan_epoch,open_epoch,epoch_seed
from runner.tests.test_dataset_loader_configuration import recipe

def source(kind='indexed',**kw):
    return make_source(torch.arange(10,dtype=torch.float32).reshape(5,2),{'target':torch.arange(5)},list('abcde'),recipe(kind=kind,**kw))

@pytest.mark.parametrize('kind',['indexed','tensor','subset','boundedIterable'])
@pytest.mark.parametrize('drop',[False,True])
def test_order_short_tail_and_global_drop(kind,drop):
    s=source(kind,**({'indices':[4,1,3]} if kind=='subset' else {}));r=recipe('data_loader',batch_size=2,drop_last=drop)
    plans=plan_epoch(s,r,0);expected=list(s.positions)
    if drop:expected=expected[:len(expected)//2*2]
    assert [i for p in plans for i in p.row_indices]==expected
    with open_epoch(s,r,0) as batches:
        result=list(batches)
    assert [i for b in result for i in b.row_indices]==expected
    for b in result:
        assert b.roles['target'].tolist()==list(b.row_indices)
        assert b.input[:,0].tolist()==[2*i for i in b.row_indices]

def test_shuffle_private_seed_repeat_and_epoch_change():
    s=source();r=recipe('data_loader',batch_size=2,shuffle=True)
    rng=torch.get_rng_state().clone();p=plan_epoch(s,r,0)
    assert torch.equal(rng,torch.get_rng_state())
    assert p==plan_epoch(s,r,0)
    assert epoch_seed(42,0)!=epoch_seed(42,1)
    assert any(p!=plan_epoch(s,r,e) for e in range(1,8))
    assert sorted(i for b in p for i in b.row_indices)==list(range(5))

def test_domain_and_drop_all_refusal():
    for r in [recipe('data_loader',collation='whole_bags'),recipe('data_loader',batch_size=6,drop_last=True)]:
        with pytest.raises(ValueError):plan_epoch(source(),r,0)

import os
import random
import time
from dataclasses import replace
from torch.utils.data import Dataset
from runner.embedding_bag_packet import EmbeddingBagPacket
import runner.batch_loading as loading


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

@pytest.mark.parametrize('kind',['indexed','tensor','subset','boundedIterable'])
@pytest.mark.parametrize('workers',[1,2])
def test_actual_spawn_order_rows_and_shutdown(kind,workers):
    s=source(kind,**({'indices':[4,0,2]} if kind=='subset' else {}))
    r=recipe('data_loader',batch_size=2,shuffle=True,num_workers=workers,prefetch_factor=1,timeout=10)
    torch_rng=torch.get_rng_state().clone();python_rng=random.getstate()
    with open_epoch(s,r,2) as batches:
        pids=batches.worker_pids
        assert len(pids)==workers and all(p!=os.getpid() and alive(p) for p in pids)
        rows=list(batches)
        assert tuple(b.row_indices for b in rows)==tuple(p.row_indices for p in plan_epoch(s,r,2))
        assert all(b.roles['target'].tolist()==list(b.row_indices) for b in rows)
        assert all(b.input[:,0].tolist()==[i*2 for i in b.row_indices] for b in rows)
    assert not any(alive(p) for p in pids)
    assert torch.equal(torch_rng,torch.get_rng_state()) and random.getstate()==python_rng

@pytest.mark.parametrize('kind',['indexed','boundedIterable'])
def test_actual_spawn_global_short_tail_drop(kind):
    r=recipe('data_loader',batch_size=2,drop_last=True,num_workers=2,prefetch_factor=1,timeout=10)
    with open_epoch(source(kind),r,0) as batches:rows=list(batches)
    assert [b.row_indices for b in rows]==[(0,1),(2,3)]


def test_worker_cancellation_joins_live_processes():
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls>2:raise InterruptedError('cancelled')
    r=recipe('data_loader',batch_size=2,num_workers=2,prefetch_factor=1,timeout=10)
    with pytest.raises(InterruptedError):
        with open_epoch(source(),r,0,check_cancelled=cancel) as batches:
            pids=batches.worker_pids
            list(batches)
    assert not any(alive(p) for p in pids)


class BrokenDataset(Dataset):
    def __len__(self):return 5
    def __getitem__(self,index):raise ValueError('intentional native worker failure')


class SlowDataset(Dataset):
    def __len__(self):return 5
    def __getitem__(self,index):
        time.sleep(2)
        return index

@pytest.mark.parametrize('native,error',[('broken','intentional native worker failure'),('slow','timed out')])
def test_worker_failure_or_timeout_joins(native,error):
    s=replace(source(),native=BrokenDataset() if native=='broken' else SlowDataset())
    r=recipe('data_loader',batch_size=2,num_workers=1,prefetch_factor=1,timeout=1 if native=='slow' else 10)
    with pytest.raises((ValueError,RuntimeError),match=error):
        with open_epoch(s,r,0) as batches:
            pids=batches.worker_pids
            next(batches)
    assert not any(alive(p) for p in pids)

@pytest.mark.parametrize('layout',['fixed','packed'])
def test_whole_bags_keep_empty_groups_targets_and_source_weight_gradient(layout):
    weights=torch.tensor([.5,2.,-1.],requires_grad=True)
    packet=EmbeddingBagPacket('packed',torch.tensor([1,2,3]),torch.tensor([0,2,2,3]),weights,True)
    if layout=='fixed':packet=EmbeddingBagPacket('fixed',torch.tensor([[1],[2],[3]]),None,weights.reshape(3,1),False)
    s=make_source(packet,{'target':torch.tensor([10,20,30])},['a','b','c'],recipe(kind='subset',indices=[2,1,0]))
    r=recipe('data_loader',batch_size=2,collation='whole_bags')
    with open_epoch(s,r,0) as batches:rows=list(batches)
    assert [b.row_indices for b in rows]==[(2,1),(0,)]
    assert rows[0].roles['target'].tolist()==[30,20]
    if layout=='packed':
        assert rows[0].input.offsets.tolist()==[0,1,1]
        assert rows[0].input.indices.tolist()==[3]
    sum((b.input.per_sample_weights*3).sum() for b in rows).backward()
    torch.testing.assert_close(weights.grad,torch.full((3,),3.))
    with pytest.raises(ValueError,match='TensorDataset'):make_source(packet,{},[0,1,2],recipe(kind='tensor'))
    with pytest.raises(ValueError,match='gradients'):
        plan_epoch(s,recipe('data_loader',num_workers=1,prefetch_factor=1,timeout=10),0)


def test_padded_sequence_native_collation_preserves_actual_typed_lengths():
    x=torch.arange(30,dtype=torch.float32).reshape(5,3,2)
    s=make_source(x,{'input_lengths':torch.tensor([3,2,1,3,2]),'target':torch.tensor([[1,2],[2,0],[1,0],[2,1],[1,0]])},list(range(5)),recipe(kind='subset',indices=[4,1]))
    with open_epoch(s,recipe('data_loader',batch_size=2,collation='padded_sequence'),0) as batches:b=next(batches)
    torch.testing.assert_close(b.input,x[[4,1]])
    assert b.roles['input_lengths'].tolist()==[2,2]
    assert b.roles['target'].tolist()==[[1,0],[2,0]]
    with pytest.raises(ValueError,match='padded'):plan_epoch(source(),recipe('data_loader',collation='padded_sequence'),0)


def test_bounded_resource_guard_before_native_creation(monkeypatch):
    monkeypatch.setattr(loading,'MAX_WORKING_BYTES',1)
    monkeypatch.setattr(loading,'DataLoader',lambda *a,**k:pytest.fail('native loader allocated before admission'))
    with pytest.raises(ValueError,match='256 MiB'):
        with open_epoch(source(),recipe('data_loader'),0):pass


def test_mutated_source_values_and_geometry_refuse_before_loading():
    s=source();s.indexed.inputs[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):plan_epoch(s,recipe('data_loader'),0)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_literal_affine_first_update_and_epoch_boundary_order_reconstruction(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    s=source();r=recipe('data_loader',batch_size=2,shuffle=True)
    w=torch.tensor(.25,device=device,requires_grad=True)
    expected=.25
    with open_epoch(s,r,0) as batches:
        for b in batches:
            xs=[2*i for i in b.row_indices];ys=list(b.row_indices)
            gradient=sum(2*(expected*x-y)*x for x,y in zip(xs,ys))/len(xs)
            loss=((b.input[:,0].to(device)*w-b.roles['target'].to(device))**2).mean()
            loss.backward()
            assert w.grad.item()==pytest.approx(gradient,abs=1e-5)
            with torch.no_grad():w-=.001*w.grad
            w.grad=None;expected-=.001*gradient
    assert w.item()==pytest.approx(expected,abs=1e-6)
    # Restoring only a completed-epoch position recreates the same next plan.
    restored_source=source()
    assert plan_epoch(s,r,1)==plan_epoch(restored_source,r,1)


def test_absent_loader_uses_exact_legacy_batch_bound_without_selected_recipe():
    s=source()
    with open_epoch(s,None,0,legacy_batch_size=3000) as batches:
        row=next(batches)
        assert row.row_indices==(0,1,2,3,4)
    for bad in (0,100001,True):
        with pytest.raises(ValueError):plan_epoch(s,None,0,legacy_batch_size=bad)
    with pytest.raises(ValueError):plan_epoch(s,recipe('data_loader',batch_size=2),0,legacy_batch_size=3)
