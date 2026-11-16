"""Managed Self-play; standard tables remain under 8 MB, required replay has a separate admitted cap."""
import hashlib,json,shutil,stat,io,time,os
from pathlib import Path
if __package__:
    from .self_play_operation_contract import verify_inputs,replay_limit
    from .self_play_training import train,TrainingSettings
    from .self_play_model import PolicyValueModel
    from .self_play_search import Search,SearchSettings
    from .game_episodes import require,encoded
    from .game_rules import Definition
    from .game_session_worker import table
    from .worker_resources import configure_torch
else:
    from self_play_operation_contract import verify_inputs,replay_limit
    from self_play_training import train,TrainingSettings
    from self_play_model import PolicyValueModel
    from self_play_search import Search,SearchSettings
    from game_episodes import require,encoded
    from game_rules import Definition
    from game_session_worker import table
    from worker_resources import configure_torch

def execute(authority,emit,check):
    import torch
    configure_torch(torch)
    root=Path(authority._path);request=authority.request;inner=verify_inputs(request,root,{})
    output=root/'artifacts';output.mkdir(mode=0o700,exist_ok=True);replay=output/'replay'
    grant=None
    if (root/'input/campaign-grant.json').exists():
        if __package__:from .self_play_campaign_grant import campaign_directory
        else:from self_play_campaign_grant import campaign_directory
        grant=json.loads((root/'input/campaign-grant.json').read_bytes());campaign_root=campaign_directory(root,grant);replay=campaign_root/'replay'
    if inner['resume']:
        require(not replay.exists(),'Resume destination must be new.')
        if __package__:from .worker_authorization import directory,read_regular
        else:from worker_authorization import directory,read_regular
        replay.mkdir(mode=0o700)
        for entry in request['inputFiles']:
            if not entry['path'].startswith('input/resume/'):continue
            source=root/entry['path'];fd=directory(str(source.parent))
            try:raw=read_regular(fd,source.name,entry['bytes'])
            finally:os.close(fd)
            require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],'Resume artifact changed during capture.')
            target=replay/entry['path'].removeprefix('input/resume/');target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(raw)
    opponents={}
    selected=inner['configuration'].get('evaluation',{}).get('opponents',[])
    for name in selected:
        if not name.startswith('checkpoint:'):continue
        identifier=name.removeprefix('checkpoint:');base=root/'input/opponents'/identifier
        metadata=json.loads((base/'identity.json').read_bytes());model_bytes=(base/'model.pt').read_bytes();graph_bytes=(base/'model.json').read_bytes()
        require(metadata['kind']=='kwip.game-policy-value.inference.v1' and metadata['sourcePins']['definitionSHA256']==Definition(inner['definition']).identity,'Evaluation checkpoint belongs to incompatible rules.')
        for filename,raw in [('model.pt',model_bytes),('model.json',graph_bytes)]:require(any(f['path']==filename and f['sha256']==hashlib.sha256(raw).hexdigest() for f in metadata['files']),'Evaluation model content pin differs.')
        saved=torch.load(io.BytesIO(model_bytes),map_location='cpu',weights_only=True)
        model=PolicyValueModel(json.loads(graph_bytes),state=saved['modelState']).frozen();opponents[name]=model
    if __package__:from .self_play_evaluation import validate_opponents
    else:from self_play_evaluation import validate_opponents
    validate_opponents(TrainingSettings(**inner['configuration']),opponents)
    last_report=[0.]
    def report(p):
        now=time.monotonic()
        if p['phase']=='checkpoint' or now-last_report[0]>=.2 or p.get('update') in (1,p.get('updates',-1)):
            last_report[0]=now;emit(dict(event='selfPlayProgress',**p))
    if inner['validateOnly']:
        c=TrainingSettings(**inner['configuration']);d=Definition(inner['definition'],check_cancelled=check)
        require(all(t['status']=='passed' for t in d.run_tests()['tests']),'Authored game tests failed.')
        if __package__:from .self_play_devices import qualify_devices
        else:from self_play_devices import qualify_devices
        qualified=qualify_devices(inner['graph'],c,inner['initialization'],inner['trainability'])
        model=PolicyValueModel(inner['graph'],seed=c.initializationSeed,device=qualified['learning'],initialization=inner['initialization'],trainability=inner['trainability']);search=Search(d,model,SearchSettings(**c.search),check_cancelled=check)
        for opening in c.evaluation.get('openings',[[]]):
            state=d.initial()
            for action in opening:
                check();require(action in d.legal_actions(state),'Held-out opening contains an illegal action.');state=d.step(state,action)['state']
        state=d.initial();out=search.run(state,__import__('random').Random(c.seed),noise=False)
        result=dict(kind='kwip.self-play.v1',summary=dict(status='Bindings validated',actions=len(search.catalog),devices=qualified),iterations=[],episodes=[],moves=[],evaluationGames=[])
    else:
        if grant:
            if __package__:from .self_play_campaign import run_slice
            else:from self_play_campaign import run_slice
            def pause_requested():
                if not (root/'pause-at-boundary').exists():return False
                if __package__:from .self_play_replay import read_owned
                else:from self_play_replay import read_owned
                return read_owned(root,'pause-at-boundary',64)==b'pause'
            result=run_slice(campaign_root,inner['definition'],inner['graph'],inner['configuration'],action=grant['action'],expected_index=grant['expectedIndexSHA256'],initialization=inner['initialization'],trainability=inner['trainability'],check_cancelled=check,progress=report,opponents=opponents,parent_root=campaign_directory(root,grant['parent']) if grant['parent'] else None,parent_index=grant['parent']['expectedIndexSHA256'] if grant['parent'] else None,pause_requested=pause_requested)
        else:
            require(not inner['configuration'].get('campaign',{}).get('enabled'),'Campaign requires a captured native store grant.')
            result=train(inner['definition'],inner['graph'],inner['configuration'],replay,resume=inner['resume'],initialization=inner['initialization'],trainability=inner['trainability'],check_cancelled=check,progress=report,opponents=opponents)
        result['summary']=dict(status='Completed',iterations=len(result['iterations']),weightsSHA256=result['weightsSHA256'],checkpoint=result['checkpoint']['iteration'])
        if grant:result['summary']['campaign']=result['campaign']
    result['parentOperationID']=inner['parentOperationID']
    tables={'iterations':result['iterations'],'episodes':result['episodes'],'moves':result['moves'],'evaluation':result['evaluationGames']}
    if inner['configuration'].get('campaign',{}).get('enabled'):
        tables['campaign']=result.get('phases',[])
        tables['performance']=result.get('performance',[])
        tables['opponents']=[dict(iteration=row['iteration'],**summary) for row in result['iterations'] for summary in row.get('evaluation',[])]
    # Replay remains owned chunks. Do not duplicate its inventory/state into a huge overview.
    result.pop('checkpoint',None)
    captured_routing=result.pop('routingTables',{})
    has_moe=any(n.get('type')=='mixture_experts' for n in inner['graph'].get('nodes',[]))
    raws={'self-play.json':encoded(result),**{name+'.table.json':encoded(table(rows,request['requestIdentity'])) for name,rows in tables.items()}}
    if has_moe:
        for port in ('routing','routingSample'):
            value=captured_routing.get(port,table([],request['requestIdentity']))
            raws[port+'.table.json']=encoded(value)
    require(sum(map(len,raws.values()))<=8_000_000,'Self-play tables exceed the 8 MB publication allowance; reduce run limits.')
    check();verify_inputs(request,root,{})
    for name,raw in raws.items():
        with (output/name).open('xb') as f:f.write(raw)
    if not inner['validateOnly']:
        if __package__:from .engine import export_source
        else:from engine import export_source
        from_checkpoint=replay/'checkpoints'/f"iteration-{result['summary']['checkpoint']:02}"/'model.pt'
        state=torch.load(from_checkpoint,map_location='cpu',weights_only=True)
        model=PolicyValueModel(inner['graph'],state=state)
        stream=io.BytesIO();torch.save(dict(modelState=state,graph=inner['graph'],task='gamePolicyValue',taskVersion=1,resumable=False),stream)
        inference={'model.pt':stream.getvalue(),'model.py':export_source(model.model).encode(),'model.json':encoded(inner['graph'])}
        manifest=dict(kind='kwip.game-policy-value.inference.v1',taskVersion=1,resumable=False,sourcePins=result['pins'],iteration=result['summary']['checkpoint'],weightsSHA256=result['weightsSHA256'],files=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in inference.items()])
        inference['inference-manifest.json']=encoded(manifest)
        for name,raw in inference.items():
            with (output/name).open('xb') as f:f.write(raw)
    files=[];total=0
    for path in sorted(output.rglob('*')):
        if path.is_dir():require(not path.is_symlink(),'Output directory is a link.');continue
        relative=path.relative_to(output).as_posix()
        if relative=='replay/writer.lock':continue
        limit=replay_limit(relative.removeprefix('replay/')) if relative.startswith('replay/') else 100_000_000 if relative=='model.pt' else 8_000_000
        require(stat.S_ISREG(path.lstat().st_mode) and path.stat().st_nlink==1 and path.stat().st_size<=limit,'Output file type/size differs.')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):check();h.update(block)
        size=path.stat().st_size;total+=size;files.append(dict(path='artifacts/'+relative,bytes=size,sha256=h.hexdigest()))
    require(total<=1_000_000_000 and len(files)<=605,'Self-play output exceeds its admitted storage.')
    emit(dict(event='selfPlayCompleted',operationID=request['operationID'],requestIdentity=request['requestIdentity'],files=files))
