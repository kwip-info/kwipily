"""Immutable epoch order with real bounded native DataLoader execution.

Only complete epoch boundaries are resumable. Iterators/workers are never saved.
Iterable workers yield single rows; the parent collates the planned global groups,
so a short tail/drop_last applies once rather than once per worker.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import torch
from torch.utils.data import DataLoader
if __package__:
    from .dataset_loader_configuration import validate_recipe, LOADER_DEFAULTS
    from .dataset_sources import RestartableTrainingIterable, make_source
    from .embedding_bag_packet import EmbeddingBagPacket
else:
    from dataset_loader_configuration import validate_recipe, LOADER_DEFAULTS
    from dataset_sources import RestartableTrainingIterable, make_source
    from embedding_bag_packet import EmbeddingBagPacket

ORDER_VERSION='kwip.data-loader.epoch-order.v1'
MAX_WORKING_BYTES=256*1024*1024


@dataclass(frozen=True)
class BatchPlan:
    ordinal: int
    row_indices: tuple
    row_ids: tuple


@dataclass(frozen=True)
class TrainingBatch:
    input: object
    roles: dict
    row_indices: tuple
    row_ids: tuple


def epoch_seed(seed,epoch):
    if type(seed) is not int or not 0<=seed<2**63 or type(epoch) is not int or not 0<=epoch<2**63:
        raise ValueError('Loader seed and zero-based epoch require nonnegative Int64 integers.')
    raw=json.dumps([ORDER_VERSION,seed,epoch],ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big') & (2**63-1)


def _configuration(source,recipe,legacy_batch_size=None):
    absent=recipe is None
    if absent:
        if type(legacy_batch_size) is not int or not 1<=legacy_batch_size<=100000:
            raise ValueError('Absent loader uses the exact captured legacy batch size in 1..100000.')
        recipe=dict(schemaVersion=1,configurationVersion=1,componentID="data_loader",config=dict(LOADER_DEFAULTS))
    result=validate_recipe(recipe)
    if absent:result['config']['batch_size']=legacy_batch_size
    elif legacy_batch_size is not None and legacy_batch_size!=result['config']['batch_size']:
        raise ValueError('Captured batch size differs from selected loader recipe; mirror the selected value.')
    if result['componentID']!='data_loader':raise ValueError('Choose a DataLoader recipe for loading.')
    config=result['config'];collation=config['collation']
    if source.domain=='whole_bags':
        if collation not in ('auto','whole_bags'):raise ValueError('Bag sources require auto or whole_bags collation; keep item weights paired.')
    else:
        if collation=='whole_bags':raise ValueError('whole_bags requires a typed bag source.')
        if collation=='padded_sequence' and source.indexed.inputs.ndim<3:
            raise ValueError('padded_sequence requires an already padded [rows,time,...] tensor source.')
    # Admission is conservative: source copies plus bounded prefetched batches.
    # Count each role separately even if caller storage aliases another role.
    source_bytes=sum(t.numel()*t.element_size() for t in source.tensors)
    workers=config['num_workers']
    if workers and any(t.device.type!='cpu' or t.requires_grad for t in source.tensors):
        raise ValueError('Spawned loading requires CPU prepared tensors without gradients; choose zero workers to preserve floating-source derivatives.')
    max_row_bytes=source_bytes # safe bound even for one unusually large packed bag
    prefetched=min(source_bytes,max_row_bytes*config['batch_size'])
    peak=(1+workers)*source_bytes+workers*(config['prefetch_factor'] or 0)*prefetched
    if peak>MAX_WORKING_BYTES:
        raise ValueError('DataLoader exceeds the 256 MiB source/worker/prefetch admission bound; reduce data or workers.')
    checked=make_source(source.indexed.inputs,source.indexed.roles,source.row_ids,source.recipe)
    if checked.positions!=source.positions:
        raise ValueError('Source view changed; reconstruct the selected training source before loading.')
    return config


def plan_epoch(source,recipe,epoch,*,legacy_batch_size=None):
    config=_configuration(source,recipe,legacy_batch_size)
    seed=epoch_seed(config['seed'],epoch);positions=source.positions
    if config['shuffle']:
        generator=torch.Generator(device='cpu').manual_seed(seed)
        positions=tuple(positions[i] for i in torch.randperm(len(positions),generator=generator).tolist())
    batch=config['batch_size'];limit=len(positions)
    if config['drop_last']:limit=limit//batch*batch
    if limit==0:raise ValueError('drop_last would remove every training row; reduce batch_size or retain the final group.')
    return tuple(BatchPlan(i//batch,tuple(positions[i:min(i+batch,limit)]),
        tuple(source.row_ids[p] for p in positions[i:min(i+batch,limit)])) for i in range(0,limit,batch))


def _identity(record):return record


def _bags(records):
    packets=[r.input for r in records];first=packets[0]
    if any((p.layout,p.include_last_offset,p.per_sample_weights is not None)!=(first.layout,first.include_last_offset,first.per_sample_weights is not None) for p in packets):
        raise ValueError('Whole-bag collation cannot mix packet contracts.')
    indices=torch.cat([p.indices for p in packets],dim=0)
    weights=torch.cat([p.per_sample_weights for p in packets],dim=0) if first.per_sample_weights is not None else None
    offsets=None
    if first.layout=='packed':
        starts=[];cursor=0
        for packet in packets:
            starts.append(cursor);cursor+=packet.indices.numel()
        if first.include_last_offset:starts.append(cursor)
        offsets=torch.tensor(starts,dtype=torch.int64,device=indices.device)
    return EmbeddingBagPacket(first.layout,indices,offsets,weights,first.include_last_offset)


@dataclass(frozen=True)
class _Collator:
    mode: str
    def __call__(self,records):
        first=records[0]
        if type(first.input) is dict:
            if any(type(r.input) is not dict or set(r.input)!=set(first.input) for r in records):raise ValueError('Named batch inputs differ.')
            value={k:torch.cat([r.input[k] for r in records],dim=0) for k in first.input}
        elif type(first.input) is EmbeddingBagPacket:value=_bags(records)
        elif self.mode=='padded_sequence':
            value=torch.nn.utils.rnn.pad_sequence([r.input.squeeze(0) for r in records],batch_first=True)
        else:value=torch.cat([r.input for r in records],dim=0)
        roles={k:torch.cat([r.roles[k] for r in records],dim=0) for k in first.roles}
        return TrainingBatch(value,roles,tuple(r.position for r in records),tuple(r.row_id for r in records))


class EpochBatches:
    """Owned iterator; context exit joins native workers on success/error/cancel."""
    def __init__(self,source,config,plans,loader,iterator,check_cancelled):
        self.source=source;self.config=config;self.plans=plans;self.loader=loader;self.iterator=iterator
        self.check_cancelled=check_cancelled;self.cursor=0;self.closed=False
        self.worker_pids=tuple(w.pid for w in getattr(iterator,'_workers',()))
    def __iter__(self):return self
    def __next__(self):
        if self.closed or self.cursor==len(self.plans):raise StopIteration
        if self.check_cancelled:self.check_cancelled()
        plan=self.plans[self.cursor]
        if self.source.recipe['config']['kind']=='boundedIterable':
            records=[]
            for expected in plan.row_indices:
                if self.check_cancelled:self.check_cancelled()
                record=next(self.iterator)
                if record.position!=expected:raise ValueError('Native iterable returned rows outside the planned global order.')
                records.append(record)
            batch=_Collator(self.config['collation'])(records)
        else:batch=next(self.iterator)
        if batch.row_indices!=plan.row_indices or batch.row_ids!=plan.row_ids:
            raise ValueError('Native DataLoader returned different prepared row identities/order.')
        self.cursor+=1
        return batch
    def close(self):
        if self.closed:return
        self.closed=True
        # PyTorch exposes no public iterator close. This bounded lifecycle hook
        # is used only to terminate owned native worker processes, never math.
        shutdown=getattr(self.iterator,'_shutdown_workers',None)
        if shutdown is not None:shutdown()
        self.iterator=None;self.loader=None


@contextmanager
def open_epoch(source,recipe,epoch,*,check_cancelled=None,legacy_batch_size=None):
    config=_configuration(source,recipe,legacy_batch_size);plans=plan_epoch(source,recipe,epoch,legacy_batch_size=legacy_batch_size)
    if check_cancelled:check_cancelled()
    generator=torch.Generator(device='cpu').manual_seed(epoch_seed(config['seed'],epoch))
    options=dict(num_workers=config['num_workers'],timeout=config['timeout'],generator=generator,
        persistent_workers=False,pin_memory=False,in_order=True)
    if config['num_workers']:
        options.update(multiprocessing_context='spawn',prefetch_factor=config['prefetch_factor'])
    if source.recipe['config']['kind']=='boundedIterable':
        positions=tuple(p for b in plans for p in b.row_indices)
        native=RestartableTrainingIterable(source.indexed,positions)
        loader=DataLoader(native,batch_size=None,collate_fn=_identity,**options)
    else:
        lookup={p:i for i,p in enumerate(source.positions)}
        batches=[[lookup[p] for p in b.row_indices] for b in plans]
        loader=DataLoader(source.native,batch_sampler=batches,collate_fn=_Collator(config['collation']),**options)
    owned=None
    try:
        iterator=iter(loader)
        owned=EpochBatches(source,config,plans,loader,iterator,check_cancelled)
        yield owned
    finally:
        if owned is not None:owned.close()
