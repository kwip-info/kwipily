"""Bounded frozen-collection → joint update → committed resume boundary → independent evaluation."""
import copy,io,json,math,random,time,hashlib,sys
from pathlib import Path
from dataclasses import dataclass,asdict,field
import torch
if __package__:from .game_episodes import require,digest,encoded
else:from game_episodes import require,digest,encoded
if __package__:from .game_rules import Definition
else:from game_rules import Definition
if __package__:from .self_play_model import PolicyValueModel,joint_loss
else:from self_play_model import PolicyValueModel,joint_loss
if __package__:from .self_play_search import Search,SearchSettings,SearchLimit
else:from self_play_search import Search,SearchSettings,SearchLimit
if __package__:from .self_play_replay import ReplayStore,CapacityError,read_pinned
else:from self_play_replay import ReplayStore,CapacityError,read_pinned

@dataclass(frozen=True)
class TrainingSettings:
    iterations:int=8
    games:int=8
    updates:int=16
    batch:int=64
    maxMoves:int=128
    seed:int=4100
    initializationSeed:int=17
    learningRate:float=.001
    optimizer:str='adam'
    scheduler:str='constant'
    finalLearningRate:float=.0001
    momentum:float=0.
    beta1:float=.9
    beta2:float=.999
    epsilon:float=1e-8
    weightDecay:float=0.
    policyWeight:float=1.
    valueWeight:float=1.
    gradientClip:float=1.
    temperature:float=1.
    temperatureMoves:int=12
    replayPositions:int=25000
    replayBytes:int=134217728
    replayPolicy:str='stop'
    checkpoints:int=3
    evaluationPairs:int=4
    evaluationEvery:int=1
    seconds:float=600.
    device:str='cpu'
    search:dict=field(default_factory=lambda:SearchSettings().value())
    campaign:dict=field(default_factory=dict)
    evaluation:dict=field(default_factory=dict)
    compute:dict=field(default_factory=dict)
    def __post_init__(self):
        for key,lo,hi in [('iterations',1,1000000 if self.campaign.get('enabled') is True else 99),('games',2,256),('updates',1,4096),('batch',1,1024),('maxMoves',1,4096),('seed',0,2147483647),('initializationSeed',0,2147483647),('temperatureMoves',0,4096),('evaluationPairs',1,128),('evaluationEvery',1,99),('replayPositions',1,100000),('replayBytes',65536,2147483648 if self.campaign.get('enabled') is True else 536870912),('checkpoints',1,10)]:
            v=getattr(self,key);require(type(v) is int and lo<=v<=hi,'Invalid Self-play '+key+'.')
        for key,lo,hi in [('learningRate',1e-7,1),('weightDecay',0,1),('finalLearningRate',1e-7,1),('momentum',0,.999),('beta1',0,.9999),('beta2',0,.999999),('epsilon',1e-12,1),('policyWeight',0,100),('valueWeight',0,100),('gradientClip',.001,1000),('temperature',.01,10),('seconds',.01,3600)]:
            v=getattr(self,key);require(type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi,'Invalid Self-play '+key+'.')
        slices=self.campaign.get('sliceIterations',1) if self.campaign.get('enabled') is True else self.iterations
        require(type(slices) is int and 1<=slices<=99,'Invalid campaign slice.')
        require(slices*self.games*self.maxMoves<=32768,'Planned episode moves exceed the bounded 32768-move run allowance.')
        require(slices*self.updates<=4096,'Retained update measurements are limited to 4096; reduce iterations or updates.')
        require(self.games%2==0 and self.policyWeight+self.valueWeight>0,'Use balanced collection games and at least one joint loss.')
        require(self.optimizer in ('adam','adamw','sgd') and self.scheduler in ('constant','linear','cosine') and self.device in ('cpu','mps','auto') and self.replayPolicy in ('stop','rolling'),'Unsupported Self-play optimizer/device/retention.')
        if self.campaign:
            require(set(self.campaign)=={'enabled','sliceIterations','maximumSeconds','maximumBytes'} and type(self.campaign['enabled']) is bool,'Invalid campaign configuration.')
            require(type(self.campaign['maximumSeconds']) is int and 1<=self.campaign['maximumSeconds']<=604800 and type(self.campaign['maximumBytes']) is int and 67108864<=self.campaign['maximumBytes']<=8589934592,'Invalid campaign time/storage budget.')
        require(set(self.compute)<= {'activeGames','inferenceDevice','batchSize','batchWait','cacheBytes'} and type(self.compute) is dict,'Invalid compute configuration.')
        for k,lo,hi in [('activeGames',1,8),('batchSize',1,256),('cacheBytes',0,268435456)]:
            v=self.compute.get(k,lo);require(type(v) is int and lo<=v<=hi,'Invalid compute '+k)
        require(self.compute.get('inferenceDevice','cpu') in ('cpu','mps','auto'),'Invalid inference device.')
        require(type(self.compute.get('batchWait',.002)) in (int,float) and 0<=self.compute.get('batchWait',.002)<=.05,'Invalid batch wait.')
        require(set(self.evaluation)<= {'opponents','openings','seed','checkpoints'} and type(self.evaluation) is dict,'Invalid evaluation suite.')
        require(type(self.search) is dict,'Search requires bounded settings.');SearchSettings(**self.search)
    def value(self):return asdict(self)

def _bytes(value):
    out=io.BytesIO();torch.save(value,out);return out.getvalue()
def _tuples(x):return tuple(_tuples(v) for v in x) if isinstance(x,list) else x

def play(definition,models,settings,rng,check,*,search=True,noise=False,temperature=0.,opening=None,policy_modes=None):
    d=definition;d.remaining=2_000_000;state=d.initial();positions=[];moves=[];catalog=None
    for action in opening or []:
        check();d.remaining=2_000_000;require(not d.outcome(state)['terminal'],'Opening continued after terminal state.');state=d.step(state,action)['state']
    solvers={}
    for turn in range(settings.maxMoves):
        check();d.remaining=2_000_000;end=d.outcome(state)
        if end['terminal']:break
        actor=d.env(state)['actor'];model=models[actor]
        if actor not in solvers:solvers[actor]=Search(d,model,SearchSettings(**settings.search),check_cancelled=check)
        solver=solvers[actor]
        catalog=solver.catalog;observation=d.observation(state,actor)
        policy_mode=(policy_modes or {}).get(actor,'search' if search else 'policy-only')
        search_started=time.monotonic();search_nodes=simulations=0
        if policy_mode=='search' or policy_mode=='search-assisted':
            try:result=solver.run(state,rng,noise=noise)
            except SearchLimit as e:return dict(status='failed',winner=None,moves=len(moves),error=str(e)),[],moves
            pi=result['probabilities'];mask=result['legalMask'];search_nodes=result['nodes'];simulations=sum(result['visits'])
        else:
            logits,_=model.predict(observation);legal={digest(a) for a in d.legal_actions(state)};mask=[digest(a) in legal for a in catalog]
            require(len(logits)==len(catalog),'Policy catalogue width differs.');best=rng.choice([i for i in range(len(catalog)) if mask[i]]) if policy_mode=='random' else max((i for i in range(len(catalog)) if mask[i]),key=lambda i:logits[i]);pi=[float(i==best) for i in range(len(catalog))]
        if temperature>0 and turn<settings.temperatureMoves:
            weights=[p**(1/temperature) for p in pi];action=rng.choices(range(len(catalog)),weights=weights,k=1)[0]
        else:action=max(range(len(pi)),key=lambda i:pi[i])
        positions.append(dict(observation=observation,actor=actor,legalMask=mask,policy=pi))
        d.remaining=2_000_000;step=d.step(state,catalog[action]);state=step['state']
        for current in solvers.values():current.advance(action,state)
        moves.append(dict(move=turn+1,actor=actor,actionIndex=action,action=catalog[action],rule=step['rule'],searchNodes=search_nodes,simulations=simulations,decisionSeconds=time.monotonic()-search_started))
    d.remaining=2_000_000;end=d.outcome(state)
    status='terminal' if end['terminal'] else 'truncated'
    for p in positions:p['outcome']=0. if end['winner'] is None else 1. if p['actor']==end['winner'] else -1.
    return dict(status=status,winner=end['winner'],moves=len(moves),error=''),positions if status=='terminal' else [],moves

def paired_interval(scores,seed):
    if not scores:return None
    r=random.Random(seed);means=sorted(sum(r.choices(scores,k=len(scores)))/len(scores) for _ in range(2000))
    return [means[49],means[1949]]

def evaluate(definition,model,baseline,c,iteration,check):
    if c.compute:
        if __package__:from .self_play_acceleration import CachedRules,CachedPolicy
        else:from self_play_acceleration import CachedRules,CachedPolicy
        definition=CachedRules(definition,c.compute.get('cacheBytes',16777216));model=CachedPolicy(model);baseline=CachedPolicy(baseline)
    rows=[];aggregate=[]
    for mode in ('policy-only','search-assisted'):
        scores=[];wins=draws=losses=truncated=failed=0
        before=model.snapshot_digest();baseline_pin=baseline.snapshot_digest()
        for pair in range(c.evaluationPairs):
            pair_scores=[]
            for seat in (0,1):
                seed=c.seed+1_000_000+iteration*10000+pair;players=[baseline,baseline];players[seat]=model
                try:
                    outcome,_,_=play(definition,players,c,random.Random(seed),check,search=mode=='search-assisted')
                    winner=outcome['winner'];score=None if outcome['status']!='terminal' else .5 if winner is None else float(winner==seat)
                    truncated+=int(outcome['status']=='truncated');failed+=int(outcome['status']=='failed')
                    if score is not None:wins+=int(score==1);draws+=int(score==.5);losses+=int(score==0);pair_scores.append(score)
                    rows.append(dict(iteration=iteration,mode=mode,pair=pair,seat=seat,seed=seed,status=outcome['status'],winner=winner,score=score,error=outcome['error']))
                except SearchLimit as e:
                    failed+=1;rows.append(dict(iteration=iteration,mode=mode,pair=pair,seat=seat,seed=seed,status='failed',winner=None,score=None,error=str(e)))
            if len(pair_scores)==2:scores.append(sum(pair_scores)/2)
        terminal=wins+draws+losses
        require(before==model.snapshot_digest() and baseline_pin==baseline.snapshot_digest(),'Evaluation mutated frozen weights.')
        aggregate.append(dict(mode=mode,wins=wins,draws=draws,losses=losses,terminal=terminal,truncated=truncated,failed=failed,attempts=len(rows)-sum(a['attempts'] for a in aggregate),completePairs=len(scores),score=None if not terminal else (wins+.5*draws)/terminal,pairedInterval=paired_interval(scores,c.seed+iteration),opponent='initial-snapshot',opponentSHA256=baseline_pin))
    return rows,aggregate

def train(definition,graph,configuration,root,*,resume=False,check_cancelled=lambda:None,progress=lambda x:None,initialization=None,trainability=None,campaign_context=None,initial_state=None,opponents=None,pause_requested=lambda:False):
    c=TrainingSettings(**configuration)
    if __package__:from .self_play_evaluation import validate_opponents
    else:from self_play_evaluation import validate_opponents
    validate_opponents(c,opponents)
    if __package__:from .self_play_devices import qualify_devices
    else:from self_play_devices import qualify_devices
    devices=qualify_devices(graph,c,initialization,trainability)
    c=__import__('dataclasses').replace(c,device=devices['learning'])
    d=Definition(definition,check_cancelled=check_cancelled)
    require(all(t['status']=='passed' for t in d.run_tests()['tests']),'Authored game tests failed.')
    model=PolicyValueModel(graph,seed=c.initializationSeed,device=c.device,initialization=initialization,trainability=trainability,state=initial_state);baseline=model.frozen()
    solver=Search(d,model,SearchSettings(**c.search));d.remaining=2_000_000;obs=d.observation(d.initial(),d.env(d.initial())['actor']);logits,_=model.predict(obs)
    require(len(logits)==len(solver.catalog),'Policy width must equal the complete fixed action catalogue.')
    # Scientific model pins exclude labels, positions and unrelated run presentation.
    semantic_graph=copy.deepcopy(graph);semantic_graph.pop('name',None);semantic_graph.pop('execution',None)
    for node in semantic_graph['nodes']:
        for key in ('name','x','y'):node.pop(key,None)
    sources=['engine.py','game_rules.py','self_play_model.py','self_play_training.py','self_play_search.py','self_play_replay.py','model_output_configuration.py','initialization.py','trainability.py','self_play_acceleration.py','self_play_devices.py','self_play_campaign.py','self_play_evaluation.py','self_play_campaign_grant.py','moe.py']
    implementation=digest({name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in sources})
    scientific=c.value()
    if campaign_context:
        for key in ('iterations','seconds','campaign'):scientific.pop(key,None)
    pins=dict(initialWeightsSHA256=model.snapshot_digest(),definitionSHA256=d.identity,graphSHA256=digest(semantic_graph),configurationSHA256=digest(scientific),encodingSHA256=digest(dict(observations=len(obs),actions=solver.catalog)),runtime=dict(torch=str(torch.__version__),python=sys.version.split()[0],cpuThreads=torch.get_num_threads(),device=c.device,implementationSHA256=implementation),initializationSHA256=digest(initialization),trainabilitySHA256=digest(trainability))
    if __package__:from .moe import RoutedExperts,objective_statistics as moe_stats,captured_tables as moe_tables
    else:from moe import RoutedExperts,objective_statistics as moe_stats,captured_tables as moe_tables
    for module in model.model.modules():
        if isinstance(module,RoutedExperts):module.reset_capture()
    options=dict(momentum=c.momentum) if c.optimizer=='sgd' else dict(betas=(c.beta1,c.beta2),eps=c.epsilon)
    optimizer=({'adam':torch.optim.Adam,'adamw':torch.optim.AdamW,'sgd':torch.optim.SGD}[c.optimizer])([p for p in model.model.parameters() if p.requires_grad],lr=c.learningRate,weight_decay=c.weightDecay,**options)
    store=ReplayStore(root,maximum_positions=c.replayPositions,maximum_bytes=c.replayBytes,retention=c.replayPolicy,checkpoints=c.checkpoints,maximum_chunks=1200 if campaign_context else 500)
    rng=random.Random(c.seed);manifest=None;start=1;history=[];episodes=[];moves=[];evaluations=[];measurements=[];performance=[];started=time.monotonic();elapsed_base=0.;deadline=started+c.seconds
    def check():
        check_cancelled()
        if time.monotonic()>deadline:raise TimeoutError('Self-play run time budget reached; last committed iteration remains resumable.')
    with store.exclusive(),torch.random.fork_rng(devices=[]):
        torch.manual_seed(c.seed)
        if resume:
            checkpoint,manifest=store.checkpoint();require(manifest['pins']==pins,'Resume pins differ: begin a new run for incompatible rules, model, settings or runtime.')
            files={f['path']:f for f in manifest['files']}
            load=lambda n:torch.load(io.BytesIO(read_pinned(checkpoint,files[n],100_000_000)),map_location=c.device,weights_only=True)
            model.model.load_state_dict(load('model.pt'),strict=True);optimizer.load_state_dict(load('optimizer.pt'));torch.set_rng_state(load('rng.pt').cpu())
            if c.device=='mps':torch.mps.set_rng_state(torch.tensor(manifest['state']['mpsRNG'],dtype=torch.uint8))
            state=manifest['state'];rng.setstate(_tuples(state['random']));history=state['iterations'];episodes=state['episodes'];moves=state['moves'];evaluations=state['evaluationGames'];start=manifest['iteration']+1;measurements=state.get('measurements',[]);elapsed_base=measurements[-1]['elapsedSeconds'] if measurements else 0.
        if campaign_context:
            history=[];episodes=[];moves=[];evaluations=[];measurements=[]
        end=min(c.iterations,start+c.campaign.get('sliceIterations',1)-1) if campaign_context else c.iterations
        for iteration in range(start,end+1):
            check();frozen=model.frozen();collection_pin=frozen.snapshot_digest();positions=[];iteration_episodes=[];iteration_moves=[]
            if __package__:from .self_play_acceleration import CachedRules,InferenceQueue
            else:from self_play_acceleration import CachedRules,InferenceQueue
            frozen.device=devices['inference'];frozen.model.to(frozen.device)
            active=c.compute.get('activeGames',1);queue=InferenceQueue(frozen,batch_size=c.compute.get('batchSize',16),wait_seconds=c.compute.get('batchWait',.002) if active>1 else 0,cache_bytes=c.compute.get('cacheBytes',16777216),check=check) if c.compute else None
            seeds=[rng.randrange(2**31) for _ in range(c.games)]
            def collect(game):
                local=copy.copy(d);local.check_cancelled=check;local.remaining=2_000_000
                if c.compute:local=CachedRules(local,c.compute.get('cacheBytes',16777216)//max(1,active))
                policy=queue if queue else frozen
                outcome,replay,trace=play(local,[policy,policy],c,random.Random(seeds[game]),check,noise=True,temperature=c.temperature)
                return game,outcome,replay,trace
            pool=__import__('concurrent.futures',fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor(max_workers=active) if active>1 else None
            try:
                results=pool.map(collect,range(c.games)) if pool else map(collect,range(c.games))
                for game,outcome,replay,trace in results:
                    for p in replay:p.update(iteration=iteration,game=game,collectionSHA256=collection_pin,definitionSHA256=pins['definitionSHA256'],encodingSHA256=pins['encodingSHA256'])
                    positions.extend(replay);iteration_episodes.append(dict(iteration=iteration,episode=game,seed=seeds[game],**outcome))
                    iteration_moves.extend(dict(iteration=iteration,episode=game,**m) for m in trace)
                    progress(dict(phase='collection',iteration=iteration,game=game+1,games=c.games))
            finally:
                if pool:pool.shutdown(wait=True,cancel_futures=True)
                if queue:
                    queue.close();performance.append(dict(iteration=iteration,stage='collection inference',device=devices['inference'],forwards=queue.calls,observations=queue.rows,inferenceSeconds=queue.seconds,cacheHits=queue.cache.hits,cacheBytes=queue.cache.bytes,activeGames=active))
            require(frozen.snapshot_digest()==collection_pin,'Collection snapshot mutated.')
            # Admit capacity before changing weights. Pending rows are sampled in bounded batches.
            old,_,_=store.admit(positions,manifest);total_old=sum(x['count'] for x in old)
            sampling_manifest=dict(replay=old)
            require(total_old+len(positions)>0,'No terminal games: no invented draw targets or update was performed.')
            last={};model.model.train()
            for update in range(c.updates):
                check();batch=[]
                schedule_start=(campaign_context or {}).get('phaseStart',1)
                fraction=((iteration-schedule_start)*c.updates+update)/max(1,(c.iterations-schedule_start+1)*c.updates-1)
                blend=1. if c.scheduler=='constant' else 1-fraction if c.scheduler=='linear' else .5*(1+math.cos(math.pi*fraction))
                learning_rate=c.learningRate if c.scheduler=='constant' else c.finalLearningRate+(c.learningRate-c.finalLearningRate)*blend
                for group in optimizer.param_groups:group['lr']=learning_rate
                for _ in range(c.batch):
                    index=rng.randrange(total_old+len(positions))
                    batch.append(positions[index-total_old] if index>=total_old else store.sample(sampling_manifest,1,rng)[0])
                tensor=lambda key,dtype:torch.tensor([p[key] for p in batch],dtype=dtype,device=c.device)
                x=tensor('observation',torch.float32).reshape(c.batch,*model.sample_shape);out=model.forward(x)
                loss,last=joint_loss(out,tensor('legalMask',torch.bool),tensor('policy',torch.float32),tensor('outcome',torch.float32).reshape(-1,1),policy_weight=c.policyWeight,value_weight=c.valueWeight)
                if __package__:from .moe import auxiliary_loss as moe_auxiliary_loss
                else:from moe import auxiliary_loss as moe_auxiliary_loss
                stats=moe_stats(model.model)
                if stats:last.update(stats,trainingObjective=last['loss']+stats['balancePenalty']+stats['routerZPenalty'])
                loss=loss+moe_auxiliary_loss(model.model)
                require(torch.isfinite(loss).item(),'Joint loss became nonfinite.');optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),c.gradientClip,error_if_nonfinite=True);optimizer.step()
                measurement=dict(phase='learning',iteration=iteration,update=update+1,updates=c.updates,totalUpdates=(iteration-1)*c.updates+update+1,elapsedSeconds=elapsed_base+time.monotonic()-started,learningRate=learning_rate,**last,gradientNorm=float(norm))
                measurements.append(measurement);progress(measurement)
            # Independent evaluation never enters the replay inventory.
            if c.evaluation:
                if __package__:from .self_play_evaluation import evaluate_suite
                else:from self_play_evaluation import evaluate_suite
                evaluated,summary=evaluate_suite(d,model.frozen(),baseline,c,iteration,check,opponents=opponents) if iteration%c.evaluationEvery==0 else ([],[])
            else:evaluated,summary=evaluate(d,model.frozen(),baseline,c,iteration,check) if iteration%c.evaluationEvery==0 else ([],[])
            evaluations.extend(evaluated);episodes.extend(iteration_episodes);moves.extend(iteration_moves)
            history.append(dict(iteration=iteration,collectionSHA256=collection_pin,checkpointSHA256=model.snapshot_digest(),positions=len(positions),terminal=sum(e['status']=='terminal' for e in iteration_episodes),truncated=sum(e['status']=='truncated' for e in iteration_episodes),failed=sum(e['status']=='failed' for e in iteration_episodes),updates=c.updates,**last,evaluation=summary))
            state=dict(random=rng.getstate(),iterations=history,episodes=episodes,moves=moves,evaluationGames=evaluations,measurements=measurements)
            require(len(encoded(state))<=6_000_000,'Run tables exceed their bounded retention; reduce episodes or moves.')
            artifacts={'model.pt':_bytes({k:v.detach().cpu() for k,v in model.model.state_dict().items()}),'optimizer.pt':_bytes(optimizer.state_dict()),'rng.pt':_bytes(torch.get_rng_state())}
            if c.device=='mps':
                torch.mps.synchronize();state['mpsRNG']=torch.mps.get_rng_state().tolist()
            state['devices']=devices
            if campaign_context:state['campaignActiveSeconds']=campaign_context.get('activeSeconds',0.)+time.monotonic()-started
            check();manifest=store.commit(iteration,pins,positions,artifacts,state,manifest)
            progress(dict(phase='checkpoint',iteration=iteration,checkpointSHA256=model.snapshot_digest()))
            if pause_requested():break
    return dict(routingTables=moe_tables(model.model,'selfPlay',pins['graphSHA256']) if any(isinstance(m,RoutedExperts) for m in model.model.modules()) else {},kind='kwip.self-play.v1',pins=pins,iterations=history,episodes=episodes,moves=moves,evaluationGames=evaluations,measurements=measurements,performance=performance,devices=devices,checkpoint=manifest,weightsSHA256=model.snapshot_digest())
