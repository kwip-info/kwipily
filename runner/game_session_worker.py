"""Owned frozen-policy episodes, publishing complete standard tables under the task cap."""
import hashlib,json
from pathlib import Path
if __package__:
    from .game_session_operation_contract import verify_inputs
    from .game_episodes import FrozenGraphPolicy,run_episodes,configuration,encoded,digest,require
    from .game_rules import Definition
    from .worker_resources import configure_torch
else:
    from game_session_operation_contract import verify_inputs
    from game_episodes import FrozenGraphPolicy,run_episodes,configuration,encoded,digest,require
    from game_rules import Definition
    from worker_resources import configure_torch

def table(rows,pin):
    # Nested authored structures are explicit JSON text, handled by the shared paged cell viewer.
    if __package__:from . import dataflow as df
    else:import dataflow as df
    names=list(rows[0]) if rows else ['status'];types={}
    for n in names:
        vals=[r[n] for r in rows if r[n] is not None]
        types[n]='bool' if vals and all(type(v) is bool for v in vals) else 'int64' if vals and all(type(v) is int for v in vals) else 'float64' if vals and all(type(v) in (int,float) for v in vals) else 'string'
    columns=[dict(name=n,dtype=types[n],nullable=any(r[n] is None for r in rows)) for n in names]
    data=[[encoded(r[n]).decode() if type(r[n]) in (dict,list) else r[n] for n in names] for r in rows]
    ids=[str(i) for i in range(len(rows))]
    return dict(schemaVersion=1,kind='table',columns=columns,rows=data,lineage=dict(role='unassigned',sourceDigest=pin,membershipDigest=df.membership_digest(ids),observationIDs=ids,dataVersion=df.digest(dict(columns=columns,rows=data,ids=ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[]))

def execute(authority,emit,check):
    import torch
    configure_torch(torch)
    root=Path(authority._path);request=authority.request;inner=verify_inputs(request,root,{})
    c=configuration(inner['configuration']);policies={}
    for role,p in inner['policies'].items():
        check();state=None
        if 'checkpointSHA256' in p:
            # Only native verified History checkpoints reach this owned, rehashed path.
            checkpoint=torch.load(root/'input'/f'{role}.pt',map_location='cpu',weights_only=True)
            require(type(checkpoint) is dict and isinstance(checkpoint.get('modelState'),dict),'Unsupported saved game policy checkpoint.')
            state=checkpoint['modelState']
        policies[role]=FrozenGraphPolicy(p['graph'],seed=c['policies'][role]['initializationSeed'],input_name=c['policies'][role]['inputName'],device=c['device'],state=state,initialization=p.get('initialization'),check_cancelled=check)
    if inner['validateOnly']:
        # Same admission as episodes, but never takes a transition or runs the surrounding workflow.
        d=Definition(inner['definition'],check_cancelled=check);tests=d.run_tests()['tests']
        require(all(t['status']=='passed' for t in tests),'Authored game tests failed.')
        import itertools
        actions=list(itertools.product(*(v for _,v in d.action_values)));s=d.initial();d.remaining=2_000_000
        observation=d.observation(s,d.env(s)['actor']);d.legal_actions(s)
        for role,p in c['policies'].items():
            if p['source']=='random':continue
            model=policies.get('A' if p['source']=='sameA' else role);require(model is not None,'Missing policy '+role)
            before=model.snapshot_digest();values=model.logits(observation)
            require(len(values)==len(actions) and all(__import__('math').isfinite(v) for v in values),'Policy output width/values do not match the action catalogue.')
            require(before==model.snapshot_digest(),'Policy mutated during validation.')
        out=dict(schemaVersion=1,kind='kwip.game-session.v1',identity=dict(definitionSHA256=d.identity,configurationSHA256=digest(c)),summary=dict(status='Bindings validated',observations=len(observation),actions=len(actions),tests=len(tests)),episodes=[],moves=[])
    else:out=run_episodes(inner['definition'],c,policies,check_cancelled=check,progress=lambda p:emit(dict(event='gameSessionProgress',**p)))
    values={'session.json':out,'summary.table.json':table([out['summary']],request['requestIdentity']),'episodes.table.json':table(out['episodes'],request['requestIdentity']),'moves.table.json':table(out['moves'],request['requestIdentity'])}
    raws={n:encoded(v) for n,v in values.items()}
    require(sum(map(len,raws.values()))<=8_000_000,'Game session exceeds the shared 8 MB publication allowance; reduce episodes/moves or decision capture.')
    check();verify_inputs(request,root,{})
    output=root/'artifacts';output.mkdir(mode=0o700);files=[]
    for name,raw in raws.items():
        check()
        with (output/name).open('xb') as f:f.write(raw)
        files.append(dict(path='artifacts/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    check();emit(dict(event='gameSessionCompleted',operationID=request['operationID'],requestIdentity=request['requestIdentity'],files=files))
