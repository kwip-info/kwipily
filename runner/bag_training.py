"""Whole-observation worker operations and honest packet prediction capture."""
import json
import torch
if __package__:
    from .embedding_bag_packet import EmbeddingBagPacket
    from .model_input import observation_count,select_observations,move_to_device
else:
    from embedding_bag_packet import EmbeddingBagPacket
    from model_input import observation_count,select_observations,move_to_device


def input_slice(value,start,stop,device):
    if isinstance(value,torch.Tensor):return value[start:stop].to(device)
    count=observation_count(value)
    if type(start) is not int or type(stop) is not int or not 0<=start<stop<=count:
        raise ValueError('Select a nonempty bounded range of whole observations.')
    return move_to_device(select_observations(value,list(range(start,stop))),device)


def prediction_rows(packet,outputs,targets,*,header,objective=False,raw=None,decode_axis=None,byte_limit=1_000_000):
    count=min(200,observation_count(packet))
    if type(packet) is not EmbeddingBagPacket:raise ValueError('Packet capture requires a typed bag input.')
    bounds=packet.offsets.detach().cpu().tolist() if packet.offsets is not None else None
    result=[]
    # Count JSON header and separators; all rows must fit, never silently truncate.
    used=len(json.dumps({**header,'predictionRows':[]},sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    refusal='Prediction preview exceeds its 1 MB artifact budget. Turn off prediction-row capture or reduce the output shape.'
    for position in range(count):
        if bounds is None:
            ids=packet.indices[position];weights=packet.per_sample_weights[position] if packet.per_sample_weights is not None else None
        else:
            begin=bounds[position];end=bounds[position+1] if position+1<len(bounds) else packet.indices.numel()
            ids=packet.indices[begin:end];weights=packet.per_sample_weights[begin:end] if packet.per_sample_weights is not None else None
        required=ids.numel()+(weights.numel() if weights is not None else 0)
        if objective:
            required+=outputs[position].numel()+sum(value[position].numel() for value in targets.values())
            if raw is not None:required+=raw[position].numel()
        if required>byte_limit-used:raise ValueError(refusal)
        source={'indices':ids.detach().cpu().tolist()}
        if weights is not None:source['per_sample_weights']=weights.detach().cpu().tolist()
        row={'input':source}
        if objective:
            row['output']=outputs[position].detach().cpu().tolist()
            row['targets']={key:value[position].detach().cpu().tolist() for key,value in targets.items()}
            if raw is not None:row['objectiveOutput']=raw[position].detach().cpu().tolist()
            if decode_axis is not None:row['decoded']=outputs[position].argmax(decode_axis).detach().cpu().tolist()
        else:row.update(label=int(targets[position]),predicted=int(outputs[position].argmax()))
        encoded=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        used+=len(encoded)+int(position>0)
        if used>byte_limit:raise ValueError(refusal)
        result.append(row)
    return result
