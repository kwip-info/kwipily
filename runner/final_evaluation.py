"""Bounded, inference-only binary classifier metrics over every supplied row.

This helper does not establish protected-test provenance or checkpoint authority;
callers must verify those independently. Confusion rows are actual classes and
columns predicted classes. Tied logits select class zero. CPU float64 loss
reduction keeps metric aggregation independent of uneven inference batches.
"""
import math

import torch

if __package__:
    from .execution_device import resolve_device
else:
    from execution_device import resolve_device


def _bounded_integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} through {high}.')


def _tensors(model):
    # named_buffers includes nonpersistent state that state_dict omits.
    return dict([('parameter:' + key, value) for key, value in model.named_parameters()] +
                [('buffer:' + key, value) for key, value in model.named_buffers()])


def _snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in _tensors(model).items()}


def _unchanged(model, before):
    current = _tensors(model)
    if current.keys() != before.keys():
        return False
    for key, original in before.items():
        value = current[key].detach().cpu()
        if value.shape != original.shape or value.dtype != original.dtype:
            return False
        # Bitwise comparison catches signed-zero and NaN payload changes too.
        if not torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                           original.contiguous().reshape(-1).view(torch.uint8)):
            return False
    return True


def evaluate_classifier(model, x, y, observation_ids, *, batch_size, device, preview_limit=8, on_batch=None):
    if on_batch is not None and not callable(on_batch):
        raise TypeError('Evaluation progress callback must be callable.')
    _bounded_integer(batch_size, 1, 100000, 'Batch size')
    _bounded_integer(preview_limit, 0, 100, 'Preview limit')
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Evaluation requires a torch module.')
    features=list(x.values()) if type(x) is dict else [x]
    if not 1 <= len(features) <= 8 or (type(x) is dict and any(not isinstance(k,str) or not k for k in x)):
        raise ValueError("Evaluation requires one through eight named feature tensors.")
    if any(not isinstance(v,torch.Tensor) or not 2<=v.ndim<=8 or v.dtype not in (torch.float32,torch.float64,torch.int64) for v in features):
        raise ValueError('Features must be rank 2 through 8 float32, float64 or int64 tensors.')
    count=len(features[0])
    if any(len(v)!=count for v in features):
        raise ValueError("Named feature tensors must have aligned observation counts.")
    if not 1<=count<=100000 or any(any(size<1 for size in v.shape) for v in features) or sum(v.numel() for v in features)>100000000:
        raise ValueError('Evaluation requires 1 through 100000 nonempty rows and at most 100000000 feature elements.')
    if any(not torch.isfinite(v).all().item() for v in features):raise ValueError('Evaluation features must be finite.')
    if not isinstance(y, torch.Tensor) or y.dtype != torch.int64 or y.ndim != 1 or len(y) != count:
        raise ValueError('Targets must be one int64 label per observation.')
    if not ((y == 0) | (y == 1)).all().item():
        raise ValueError('Binary targets must be zero or one.')
    if (not isinstance(observation_ids, list) or len(observation_ids) != count or
            any(not isinstance(value, str) or not 1 <= len(value) <= 200 for value in observation_ids) or
            len(set(observation_ids)) != count):
        raise ValueError('Observation identities must be unique nonempty strings of at most 200 characters.')
    if not isinstance(device, str) or device not in ('cpu', 'mps', 'auto'):
        raise ValueError('Device must be auto, cpu or mps.')
    actual_device = resolve_device(device)
    tensors = _tensors(model)
    devices = {value.device for value in tensors.values()}
    if len(devices) > 1:
        raise ValueError('Frozen model state must reside on one device before evaluation.')
    original_device = next(iter(devices), torch.device('cpu'))
    before = _snapshot(model)
    modes = [(module, module.training) for module in model.modules()]
    predictions = []
    losses = []
    confusion = [[0, 0], [0, 0]]
    try:
        model.to(actual_device)
        model.eval()
        if on_batch is not None: on_batch(0, count)
        with torch.inference_mode():
            for offset in range(0, count, batch_size):
                end = min(offset + batch_size, count)
                # Never expose caller-owned feature storage to forward mutation.
                batch={k:v[offset:end].to(actual_device) for k,v in x.items()} if type(x) is dict else x[offset:end].to(actual_device)
                logits=model({k:v.clone() for k,v in batch.items()} if type(batch) is dict else batch.clone())
                if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or
                        tuple(logits.shape) != (end - offset, 2) or not logits.is_floating_point() or
                        not torch.isfinite(logits).all().item()):
                    raise ValueError('Classifier logits must be finite floating-point values with shape [batch, 2].')
                # Transfer first: some MPS runtimes silently zero a combined
                # device-transfer/float64 conversion. Promote on the CPU only.
                values = logits.detach().to(device='cpu').to(dtype=torch.float64)
                targets = y[offset:end].detach().cpu()
                row_losses = -torch.log_softmax(values, dim=1).gather(1, targets[:, None]).squeeze(1)
                if not torch.isfinite(row_losses).all().item():
                    raise ValueError('Cross-entropy loss is not finite.')
                losses.extend(row_losses.tolist())
                selected = values.argmax(dim=1).tolist()
                for local, (pair, predicted, target) in enumerate(zip(values.tolist(), selected, targets.tolist())):
                    confusion[target][predicted] += 1
                    predictions.append(dict(observationID=observation_ids[offset + local],
                                            logits=pair, predicted=predicted, target=target))
                if on_batch is not None: on_batch(end, count)
    finally:
        unchanged = _unchanged(model, before)
        model.to(original_device)
        for module, training in modes:
            module.training = training
        if not unchanged:
            raise ValueError('Model parameters or buffers changed during frozen evaluation; no metrics are valid.')
    loss_sum = math.fsum(losses)
    if not math.isfinite(loss_sum):
        raise ValueError('Summed cross-entropy loss is not finite.')
    per_class = []
    for index in range(2):
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        true_positive = confusion[index][index]
        per_class.append(dict(classIndex=index,
                              precision=true_positive / predicted_count if predicted_count else None,
                              precisionUnavailableReason=None if predicted_count else 'noPredictedObservations',
                              recall=true_positive / support if support else None,
                              recallUnavailableReason=None if support else 'noActualObservations', support=support))
    return dict(schemaVersion=1, evaluatedCount=count, device=actual_device, batchSize=batch_size,
                lossSum=loss_sum, lossMean=loss_sum / count,
                accuracy=(confusion[0][0] + confusion[1][1]) / count,
                confusionMatrix=confusion, perClass=per_class, predictions=predictions,
                preview=predictions[:preview_limit])


def evaluate_frozen_objective(model,x,roles,observation_ids,*,recipe,scalarization='mean',batch_size,device,preview_limit=8,on_batch=None):
    """Measure the archived training objective with the same partition reducer.

    This is inference only: no optimizer, fitting, target coercion or new metric
    convention. Parameters/buffers and every module's mode are restored/checked.
    """
    if __package__:
        from .loss_objectives import create_objective
        from .loss_partition import objective_partition_contribution,finish_objective_partition
    else:
        from loss_objectives import create_objective
        from loss_partition import objective_partition_contribution,finish_objective_partition
    _bounded_integer(batch_size,1,100000,'Batch size');_bounded_integer(preview_limit,0,100,'Preview limit')
    if not isinstance(model,torch.nn.Module) or type(roles) is not dict:raise ValueError('Frozen objective requires a model and typed target roles')
    features=list(x.values()) if type(x) is dict else [x]
    if not 1<=len(features)<=8 or any(not isinstance(v,torch.Tensor) or not 2<=v.ndim<=8 or v.dtype not in (torch.float32,torch.int64) for v in features):raise ValueError('Frozen objective requires finite declared feature tensors')
    count=len(features[0])
    if not 1<=count<=100000 or any(len(v)!=count or not torch.isfinite(v).all().item() for v in features) or sum(v.numel() for v in features)>100000000:raise ValueError('Frozen objective feature population exceeds bounds or differs')
    if any(not isinstance(v,torch.Tensor) or v.ndim<1 or len(v)!=count or v.numel()>100000000 or not torch.isfinite(v).all().item() for v in roles.values()):raise ValueError('Frozen objective target population differs')
    if not isinstance(observation_ids,list) or len(observation_ids)!=count or len(set(observation_ids))!=count or any(type(v) is not str or not 1<=len(v.encode())<=200 for v in observation_ids):raise ValueError('Frozen objective identities differ')
    if device not in ('cpu','mps','auto') or on_batch is not None and not callable(on_batch):raise ValueError('Invalid evaluation execution settings')
    actual=resolve_device(device);objective=create_objective(recipe,actual)
    devices={v.device for v in _tensors(model).values()}
    if len(devices)>1:raise ValueError('Frozen model state must reside on one device')
    original=next(iter(devices),torch.device('cpu'));before=_snapshot(model);modes=[(m,m.training) for m in model.modules()]
    parts=[];outputs=[];rows=[];elements=0
    try:
        model.to(actual);model.eval()
        if on_batch:on_batch(0,count)
        with torch.inference_mode():
            for begin in range(0,count,batch_size):
                end=min(begin+batch_size,count)
                batch={k:v[begin:end].to(actual).clone() for k,v in x.items()} if type(x) is dict else x[begin:end].to(actual).clone()
                predicted=model(batch)
                if not isinstance(predicted,torch.Tensor) or predicted.dtype!=torch.float32 or not 2<=predicted.ndim<=8 or len(predicted)!=end-begin:raise ValueError('Frozen objective prediction geometry differs')
                elements+=predicted.numel()
                if elements>100000000:raise ValueError('Frozen objective output exceeds retained tensor budget')
                targets={k:v[begin:end].to(actual).clone() for k,v in roles.items()}
                parts.append(objective_partition_contribution(objective,predicted,targets,scalarization))
                values=predicted.detach().cpu();outputs.append(values)
                for i in range(end-begin):rows.append(dict(observationID=observation_ids[begin+i],output=values[i].tolist(),targets={k:v[i].detach().cpu().tolist() for k,v in targets.items()}))
                if on_batch:on_batch(end,count)
            measured=finish_objective_partition(parts,recipe,scalarization)
    finally:
        unchanged=_unchanged(model,before);model.to(original)
        for module,training in modes:module.training=training
        if not unchanged:raise ValueError('Model parameters or buffers changed during frozen evaluation; no metrics are valid.')
    metadata={k:v for k,v in measured['metadata'].items() if k!='recipe'}
    return dict(schemaVersion=2,evaluatedCount=count,device=actual,batchSize=batch_size,objective=metadata,predictions=rows,preview=rows[:preview_limit])
