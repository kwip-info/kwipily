"""Complete held-out prediction tables; previews never define their population."""
import json
import re
import torch
if __package__:
    from . import dataflow as df
else:
    import dataflow as df


def prediction_table(outputs, targets, observation_ids, role, source_sha256, *, classification=False, check_cancelled=lambda:None):
    if role not in ('validation','test') or not isinstance(source_sha256,str) or re.fullmatch('[0-9a-f]{64}',source_sha256) is None:
        raise ValueError('Prediction tables require a verified held-out population and source pin.')
    ids=list(observation_ids)
    if not ids or len(ids)!=len(outputs) or len(set(ids))!=len(ids) or any(type(i) is not str or not 1<=len(i.encode())<=200 for i in ids):
        raise ValueError('Prediction table observation identities differ from the complete population.')
    outputs=outputs.detach().cpu()
    if not torch.isfinite(outputs).all():raise ValueError('Prediction table contains nonfinite outputs.')
    targets={k:v.detach().cpu() for k,v in targets.items()} if isinstance(targets,dict) else targets.detach().cpu()
    if any(len(v)!=len(ids) for v in targets.values()) if isinstance(targets,dict) else len(targets)!=len(ids):
        raise ValueError('Prediction targets differ from the complete population.')
    if classification and (outputs.ndim!=2 or outputs.shape[1]>120 or isinstance(targets,dict) or targets.ndim!=1):
        raise ValueError('Scalar classification columns require a bounded class vector and scalar targets.')
    def col(name,dtype):return dict(name=name,dtype=dtype,nullable=False)
    columns=[col('observation_id','string'),col('partition','string'),col('target','int64' if classification else 'string')]
    if classification:columns+=[col('predicted','int64')]+[col(f'probability_{i}','float64') for i in range(outputs.shape[1])]
    columns+=[col('output','string')]
    if len(ids)*len(columns)>df.MAX_TABLE_CELLS:raise ValueError('Prediction table exceeds its five million cell budget.')
    rows=[]
    probabilities=torch.softmax(outputs.to(torch.float64),dim=1) if classification else None
    for i,ident in enumerate(ids):
        if i%256==0:check_cancelled()
        target={k:v[i].tolist() for k,v in targets.items()} if isinstance(targets,dict) else targets[i].tolist()
        row=[ident,role,int(target) if classification else json.dumps(target,separators=(',',':'),allow_nan=False)]
        if classification:row+=[int(outputs[i].argmax())]+probabilities[i].tolist()
        row+=[json.dumps(outputs[i].tolist(),separators=(',',':'),allow_nan=False)]
        rows.append(row)
    return dict(schemaVersion=1,kind='table',columns=columns,rows=rows,
        lineage=dict(role=role,sourceDigest=source_sha256,membershipDigest=df.membership_digest(ids),observationIDs=ids,
            dataVersion=df.digest(dict(columns=columns,rows=rows,ids=ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[]))
