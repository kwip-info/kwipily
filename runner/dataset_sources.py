"""Real native sources over already verified training partitions.

No callbacks, file reads, preprocessing or unbounded external generators. Tensor
storage is retained deliberately so zero-worker floating inputs remain connected.
"""
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, IterableDataset, TensorDataset, Subset, get_worker_info
if __package__:
    from .dataset_loader_configuration import validate_recipe, MAX_ROWS
    from .model_input import observation_count, select_observations, canonical_input_tensors
    from .embedding_bag_packet import EmbeddingBagPacket
else:
    from dataset_loader_configuration import validate_recipe, MAX_ROWS
    from model_input import observation_count, select_observations, canonical_input_tensors
    from embedding_bag_packet import EmbeddingBagPacket


@dataclass(frozen=True)
class SourceRecord:
    input: object
    roles: dict
    position: int
    row_id: str | int


class IndexedTrainingDataset(Dataset):
    def __init__(self,inputs,roles,row_ids):
        self.inputs=inputs;self.roles=roles;self.row_ids=row_ids
    def __len__(self):return len(self.row_ids)
    def __getitem__(self,index):
        return SourceRecord(select_observations(self.inputs,[index]),
            {k:v[index:index+1] for k,v in self.roles.items()},index,self.row_ids[index])


class TrainingTensorDataset(TensorDataset):
    def __init__(self,inputs,roles,row_ids):
        self.role_names=tuple(roles);self.row_ids=row_ids
        super().__init__(inputs,*(roles[k] for k in self.role_names))
    def __getitem__(self,index):
        tensors=super().__getitem__(index)
        return SourceRecord(tensors[0].unsqueeze(0),
            {k:v.unsqueeze(0) for k,v in zip(self.role_names,tensors[1:])},index,self.row_ids[index])


class RestartableTrainingIterable(IterableDataset):
    def __init__(self,indexed,positions):self.indexed=indexed;self.positions=tuple(positions)
    def __iter__(self):
        worker=get_worker_info();start=worker.id if worker else 0;step=worker.num_workers if worker else 1
        for position in self.positions[start::step]:yield self.indexed[position]


@dataclass(frozen=True)
class TrainingSource:
    native: object
    indexed: IndexedTrainingDataset
    positions: tuple
    recipe: dict
    row_ids: tuple
    tensors: tuple
    @property
    def domain(self):return 'whole_bags' if type(self.indexed.inputs) is EmbeddingBagPacket else 'tensor'


def make_source(inputs, roles, row_ids, recipe):
    recipe=validate_recipe(recipe)
    if recipe['componentID']!='dataset_source':raise ValueError('Choose a Dataset recipe for the source.')
    count=observation_count(inputs)
    if count>MAX_ROWS:raise ValueError('Training source exceeds 100000 prepared rows.')
    if type(row_ids) not in (list,tuple) or len(row_ids)!=count:
        raise ValueError('Training row identities must exactly match the verified partition.')
    if any(type(v) not in (str,int) or (type(v) is str and (not v or '\0' in v or len(v.encode())>1024)) for v in row_ids) or len(set(row_ids))!=count:
        raise ValueError('Training row identities must be bounded and unique.')
    if type(roles) is not dict or len(roles)>32 or any(type(k) is not str or not k or '\0' in k or len(k.encode())>128 for k in roles):
        raise ValueError('Supply bounded named typed target roles.')
    tensors=[v for _,v in canonical_input_tensors(inputs)]
    for name,value in roles.items():
        if not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[0]!=count:
            raise ValueError(f'Target role {name} must align with every original training row.')
        tensors.append(value)
    if sum(t.numel() for t in tensors)>100000000:
        raise ValueError('Training source exceeds the 100M-element data bound.')
    for tensor in tensors:
        if tensor.is_nested or tensor.layout!=torch.strided or tensor.device.type not in ('cpu','mps') or tensor.dtype not in (torch.float32,torch.float64,torch.int64):
            raise ValueError('Training sources require dense typed CPU/MPS tensors.')
        if not torch.isfinite(tensor).all().item():raise ValueError('Training source values must be finite.')
    config=recipe['config'];kind=config['kind'];positions=tuple(range(count))
    indexed=IndexedTrainingDataset(inputs,dict(roles),tuple(row_ids))
    if kind=='subset':
        positions=tuple(config['indices'])
        if any(i>=count for i in positions):raise ValueError('Subset index lies outside the original training partition.')
        native=Subset(indexed,positions)
    elif kind=='boundedIterable':
        positions=positions[:config['max_rows']];native=RestartableTrainingIterable(indexed,positions)
    elif kind=='tensor':
        if type(inputs) is EmbeddingBagPacket or type(inputs) is dict:raise ValueError('TensorDataset requires a single tensor; choose indexed, subset or boundedIterable for named inputs or bags.')
        native=TrainingTensorDataset(inputs,roles,tuple(row_ids))
    else:native=indexed
    return TrainingSource(native,indexed,positions,recipe,tuple(row_ids),tuple(tensors))
