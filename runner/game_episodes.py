"""Versioned frozen-policy episode orchestration over the shared typed rule evaluator.

No game-name dispatch, gradients, search, training, downloads or executable rule source.
The managed worker separately owns input capture, resource admission and publication.
"""
import copy
import hashlib
import itertools
import json
import math
import random
import time
try:
    from .game_rules import Definition, RuleError, RuleBudgetError
except ImportError:
    from game_rules import Definition, RuleError, RuleBudgetError

class SessionError(ValueError):pass
class EpisodeTimeLimit(SessionError):pass
class SessionTimeLimit(SessionError):pass

def require(condition,message):
    if not condition:raise SessionError(message)

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()

def configuration(raw):
    require(type(raw) is dict,'Game session configuration must be an object.')
    defaults=dict(version=1,episodes=8,seats='alternate',aSeat=0,seed=4100,maxMoves=128,maximumTotalMoves=32768,episodeSeconds=5.,sessionSeconds=60.,captureDecisions=False,device='cpu',policies={})
    require(not set(raw)-set(defaults),'Unknown game session configuration fields.')
    c={**defaults,**copy.deepcopy(raw)}
    require(type(c['version']) is int and c['version']==1,'Unsupported game session version.')
    for key,lo,hi in [('episodes',1,256),('aSeat',0,1),('seed',0,2**31-1),('maxMoves',1,4096),('maximumTotalMoves',1,32768)]:
        require(type(c[key]) is int and lo<=c[key]<=hi,'Invalid '+key+'.')
    require(c['seats'] in ('fixed','alternate'),'Choose fixed or alternate seats.')
    require(c['seats']!='alternate' or c['episodes']%2==0,'Alternating seats requires an even episode count.')
    require(type(c['captureDecisions']) is bool and c['device'] in ('cpu','mps'),'Invalid capture or device setting.')
    for key,maximum in [('episodeSeconds',3600),('sessionSeconds',3600)]:
        require(type(c[key]) in (int,float) and math.isfinite(c[key]) and 0<c[key]<=maximum,'Invalid '+key+'.')
    require(type(c['policies']) is dict and not set(c['policies'])-{'A','B'},'Define only Policy A and B.')
    policies={}
    for role in ('A','B'):
        p=dict(source='canvas',initializationSeed=17 if role=='A' else 18,selection='greedy',temperature=1.,inputName='input')
        supplied=c['policies'].get(role,{})
        require(type(supplied) is dict and not set(supplied)-set(p),'Unknown policy fields.')
        p.update(supplied)
        require(p['source'] in ('canvas','saved','random','sameA') and (role=='B' or p['source']!='sameA'),'Choose a supported policy source.')
        require(type(p['initializationSeed']) is int and 0<=p['initializationSeed']<=2**31-1,'Invalid initialization seed.')
        require(p['selection'] in ('greedy','sampling'),'Choose greedy or seeded sampling.')
        require(type(p['temperature']) in (int,float) and math.isfinite(p['temperature']) and 0.01<=p['temperature']<=100,'Temperature must be 0.01…100.')
        require(type(p['inputName']) is str and 1<=len(p['inputName'])<=128,'Choose one model input name.')
        policies[role]=p
    require(not (policies['B']['source']=='sameA' and policies['A']['source']=='random'),'Same snapshot requires a model in Policy A.')
    c['policies']=policies
    return c

class FrozenGraphPolicy:
    """Only the bundled GraphModel architecture; safe tensor checkpoints, no source exec."""
    def __init__(self,graph,*,seed=17,input_name='input',registry=None,device='cpu',state=None,initialization=None,check_cancelled=lambda:None):
        import torch
        try:
            from .engine import GraphModel
            from . import initialization as initializer
        except ImportError:
            from engine import GraphModel
            import initialization as initializer
        require(device in ('cpu','mps') and (device!='mps' or torch.backends.mps.is_available()),'Requested game policy device is unavailable.')
        check_cancelled()
        # Snapshot construction must not perturb another caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model=GraphModel(graph,registry,bundled_dataset=False,architecture_only=True)
            model.materialize_lazy_parameters()
            if state is not None:
                require(isinstance(state,dict) and all(type(k) is str and isinstance(v,torch.Tensor) for k,v in state.items()),'Saved policy must contain only a verified model-state tensor dictionary.')
                expected=model.state_dict()
                require(set(state)==set(expected),'Saved policy tensor inventory differs.')
                require(all(v.layout==torch.strided and v.dtype==expected[k].dtype and tuple(v.shape)==tuple(expected[k].shape) and torch.isfinite(v).all().item() for k,v in state.items()),'Saved policy tensor shape/dtype or finite-value contract differs.')
                model.load_state_dict(state,strict=True)
            elif initialization is not None:initializer.apply_initialization(model,initialization,check_cancelled=check_cancelled)
        require(len([n for n in model.nodes.values() if n['type'] in ('input','index_input','bag_input')])==1,'Game policy requires exactly one float32 numeric input.')
        require(all(n['type'] not in ('index_input','bag_input') for n in model.nodes.values()),'Token and bag policies are not numeric game adapters.')
        require(not model.named_inputs or set(model.named_inputs)=={input_name},'Bind exactly the declared named model input.')
        require(model.named_inputs or input_name=='input','Unnamed canvas policy input is called input.')
        require(sum(p.numel()*p.element_size() for p in model.parameters())<=100_000_000,'Game policy exceeds the bounded snapshot allowance.')
        self.model=model.to(device).eval();self.model.requires_grad_(False)
        self.input_name=input_name;self.device=device;self.graph=copy.deepcopy(graph)
        semantic=copy.deepcopy(graph)
        for node in semantic.get('nodes',[]):
            for field in ('name','x','y'):node.pop(field,None)
        semantic.pop('execution',None)
        self.identity=digest(dict(adapter='kwip.game-policy.v1',graph=semantic,weights=self.snapshot_digest(),inputName=input_name,device=device))
        check_cancelled()
    def snapshot_digest(self):
        import torch
        h=hashlib.sha256()
        for name,value in sorted(self.model.state_dict().items()):
            h.update(encoded(dict(name=name,shape=list(value.shape),dtype=str(value.dtype))))
            value=value.detach().cpu().contiguous()
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def logits(self,observation):
        import torch
        tensor=torch.tensor([observation],dtype=torch.float32,device=self.device)
        require(torch.isfinite(tensor).all().item(),'Observation cannot be encoded as finite float32.')
        with torch.inference_mode():out=self.model({self.input_name:tensor} if self.model.named_inputs else tensor)
        if type(out) is dict and set(out)=={'policy','value'}:out=out['policy']
        require(isinstance(out,torch.Tensor) and out.dtype==torch.float32 and out.ndim==2 and out.shape[0]==1,'Policy must return one batch of float32 logits.')
        return out.detach().cpu().tolist()[0]

def _checked_logits(policy,observation,width):
    values=policy.logits(observation)
    require(type(values) is list and len(values)==width and all(type(v) in (int,float) and math.isfinite(v) for v in values),'Policy output must contain exactly one finite action logit per candidate action.')
    return values

def _select(logits,mask,p,rng):
    legal=[i for i,allowed in enumerate(mask) if allowed]
    require(bool(legal),'Nonterminal state has no legal actions.')
    if logits is None:
        probabilities=[1/len(legal) if allowed else 0 for allowed in mask]
        index=legal[rng.randrange(len(legal))]
    elif p['selection']=='greedy':
        index=max(legal,key=lambda i:logits[i]);probabilities=[int(i==index) for i in range(len(mask))]
    else:
        maximum=max(logits[i] for i in legal)
        weights=[math.exp((logits[i]-maximum)/p['temperature']) if mask[i] else 0. for i in range(len(mask))]
        total=sum(weights);probabilities=[v/total for v in weights]
        sample=rng.random();cumulative=0.;index=legal[-1]
        for i in legal:
            cumulative+=probabilities[i]
            if sample<cumulative:index=i;break
    return index,probabilities

class GameEnvironment:
    """One versioned episode. Stale or duplicate commands cannot advance its state."""
    def __init__(self,definition,episode_id):
        require(type(episode_id) is str and 1<=len(episode_id)<=128,'Choose a bounded episode identity.')
        self.definition=definition;self.episode_id=episode_id;self.revision=0;self._state=definition.initial()
        self.catalog=[dict(zip([k for k,_ in definition.action_values],values)) for values in itertools.product(*(v for _,v in definition.action_values))]
    def snapshot(self):
        return dict(schemaVersion=1,kind='kwip.game-environment.v1',episodeID=self.episode_id,definitionSHA256=self.definition.identity,revision=self.revision,stateSHA256=digest(self._state),state=copy.deepcopy(self._state))
    def step(self,index,*,expected_revision,expected_state_sha256):
        require(type(expected_revision) is int and expected_revision==self.revision and expected_state_sha256==digest(self._state),'Stale or duplicate game step; current revision/state is required.')
        require(type(index) is int and 0<=index<len(self.catalog),'Action index is outside the fixed catalogue.')
        result=self.definition.step(self._state,self.catalog[index])
        self.definition.check_cancelled() # Check time/cancellation before committing this candidate.
        self._state=copy.deepcopy(result['state']);self.revision+=1
        return result
    @staticmethod
    def verify_replay(source,rows,check_cancelled=lambda:None):
        definition=Definition(source,check_cancelled=check_cancelled);environment=GameEnvironment(definition,'replay')
        for revision,row in enumerate(rows):
            check_cancelled();definition.remaining=2_000_000
            require(row['turn']==revision and row['before']==environment.snapshot()['state'],'Replay before-state/revision differs.')
            result=environment.step(row['actionIndex'],expected_revision=revision,expected_state_sha256=digest(row['before']))
            require(environment.catalog[row['actionIndex']]==row['action'] and result['state']==row['after'] and result['rewards']==row['rewards'] and result['terminal']==row['terminal'],'Replay transition/outcome differs.')
        return environment.snapshot()

def run_episodes(source,config,policies,*,check_cancelled=lambda:None,clock=time.monotonic,progress=lambda value:None,maximum_bytes=8_000_000):
    c=configuration(config);check_cancelled();started=clock();deadline=started+c['sessionSeconds'];active_episode_deadline=[None]
    def session_check():
        check_cancelled()
        now=clock()
        if now>=deadline:raise SessionTimeLimit('Session time limit reached.')
        if active_episode_deadline[0] is not None and now>=active_episode_deadline[0]:raise EpisodeTimeLimit('Episode time limit reached.')
    definition=Definition(source,check_cancelled=session_check)
    def rule(method,*args):
        # Every pure operation is independently bounded; outer time/move caps never reset.
        session_check();definition.remaining=2_000_000
        try:return method(*args)
        except RuleError as e:raise SessionError(str(e)) from e
    tests=rule(definition.run_tests)['tests']
    require(all(t['status']=='passed' for t in tests),'Authored game tests failed; repair the definition before running episodes.')
    catalog=[dict(zip([k for k,_ in definition.action_values],values)) for values in itertools.product(*(v for _,v in definition.action_values))]
    catalog_index={digest(action):i for i,action in enumerate(catalog)}
    initial=definition.initial();observation=rule(definition.observation,initial,rule(definition.env,initial)['actor'])
    bound={};snapshots={}
    for role,p in c['policies'].items():
        if p['source']=='random':bound[role]=None;continue
        policy=bound.get('A') if p['source']=='sameA' else policies.get(role)
        require(policy is not None,'Connect and verify Policy '+role+'; no random fallback is performed.')
        require(type(policy.identity) is str and len(policy.identity)==64,'Policy snapshot identity is missing.')
        snapshots[role]=policy.snapshot_digest();bound[role]=policy
        if not rule(definition.outcome,initial)['terminal']:_checked_logits(policy,observation,len(catalog))
    # Validate the initial legal set before publishing a run or evaluating a terminal model.
    rule(definition.legal_actions,initial)
    identity=dict(schemaVersion=1,definitionSHA256=definition.identity,actionCatalogSHA256=digest(catalog),observationEncodingSHA256=digest(dict(version=1,expressions=source['observations'],dtype='float32',sampleShape=[len(observation)])),configurationSHA256=digest(c),policies={r:p.identity if p else 'random-legal.v1' for r,p in bound.items()})
    episodes=[];moves=[];used=len(encoded(identity))+4096;stop=False
    def reserve(value):
        nonlocal used
        used+=len(encoded(value))+8
        require(used<=maximum_bytes-2048,'Game session artifact budget exhausted; no complete-output claim.')
    for episode in range(c['episodes']):
        try:session_check()
        except SessionTimeLimit:break
        a_seat=c['aSeat'] if c['seats']=='fixed' else (c['aSeat']+episode)%2
        seats={a_seat:'A',1-a_seat:'B'};seed=(c['seed']+episode)%(2**31)
        rngs={r:random.Random(int(digest(dict(version=1,seed=seed,policy=r,seat=a_seat if r=='A' else 1-a_seat))[:16],16)) for r in ('A','B')}
        environment=GameEnvironment(definition,str(episode+1));state=environment.snapshot()['state'];count=0;rewards=[0.,0.];episode_start=clock();episode_deadline=episode_start+c['episodeSeconds'];active_episode_deadline[0]=episode_deadline
        status='moveLimit';reason='Move limit reached.';winner=None;terminal=False
        try:
            result=rule(definition.outcome,state)
            if result['terminal']:
                terminal=True;winner=result['winner'];status='completed';reason='Initial state is terminal.'
                rewards=[rule(definition.reward.value,{**rule(definition.env,state,None,p),'outcome':result},definition) for p in (0,1)]
                require(all(type(r) in (int,float) and math.isfinite(r) and -1<=r<=1 for r in rewards) and sum(rewards)==0,'Invalid terminal reward.')
            while not terminal and count<c['maxMoves']:
                session_check()
                if len(moves)>=c['maximumTotalMoves']:status='totalMoveLimit';reason='Session move limit reached.';stop=True;break
                if clock()>=episode_deadline:raise EpisodeTimeLimit('Episode time limit reached.')
                actor=rule(definition.env,state)['actor'];role=seats[actor];policy=bound[role]
                observation=rule(definition.observation,state,actor)
                legal=rule(definition.legal_actions,state);legal_set={catalog_index[digest(a)] for a in legal};mask=[i in legal_set for i in range(len(catalog))]
                logits=_checked_logits(policy,observation,len(catalog)) if policy else None
                session_check()
                if clock()>=episode_deadline:raise EpisodeTimeLimit('Episode time limit reached after inference.')
                index,probabilities=_select(logits,mask,c['policies'][role],rngs[role]);before=state
                result=rule(lambda:environment.step(index,expected_revision=count,expected_state_sha256=digest(state)))
                row=dict(episode=episode+1,turn=count,revisionBefore=count,revisionAfter=count+1,beforeSHA256=digest(before),afterSHA256=digest(result['state']),player=actor,entrant=role,actionIndex=index,action=copy.deepcopy(catalog[index]),before=before,after=result['state'],rewards=result['rewards'],terminal=result['terminal'],status='applied',policySHA256=identity['policies'][role])
                if c['captureDecisions']:row['decision']=dict(observation=observation,legalMask=mask,logits=logits,probabilities=probabilities)
                reserve(row);moves.append(row);state=result['state'];count+=1
                rewards=[a+b for a,b in zip(rewards,result['rewards'])];terminal=result['terminal'];winner=result['winner']
                if terminal:status='completed';reason=result['rule']
            if not terminal:session_check()
        except EpisodeTimeLimit as e:status='timeLimit';reason=str(e)
        except SessionTimeLimit as e:status='sessionTimeLimit';reason=str(e);stop=True
        except (SessionError,RuleError) as e:
            if 'artifact budget' in str(e):raise
            status='failed';reason=str(e);stop=True
        active_episode_deadline[0]=None
        record=dict(episode=episode+1,seed=seed,aSeat=a_seat,status=status,reason=reason,winner=seats[winner] if terminal and winner is not None else 'draw' if terminal else None,gameWinner=winner if terminal else None,terminal=terminal,moves=count,rewards=rewards,durationSeconds=max(0,clock()-episode_start))
        reserve(record);episodes.append(record);progress(dict(completed=len(episodes),requested=c['episodes'],moves=len(moves),status=status))
        if stop:break
    for role,policy in bound.items():
        if policy is not None:require(policy.snapshot_digest()==snapshots[role],'Frozen policy weights or buffers changed during inference.')
    completed=[e for e in episodes if e['status']=='completed']
    summary=dict(requested=c['episodes'],attempted=len(episodes),terminal=len(completed),truncated=sum(e['status'] in ('moveLimit','timeLimit','sessionTimeLimit','totalMoveLimit') for e in episodes),failed=sum(e['status']=='failed' for e in episodes),notStarted=c['episodes']-len(episodes),aWins=sum(e['winner']=='A' for e in completed),bWins=sum(e['winner']=='B' for e in completed),draws=sum(e['winner']=='draw' for e in completed),moves=len(moves))
    out=dict(schemaVersion=1,kind='kwip.game-session.v1',identity=identity,summary=summary,episodes=episodes,moves=moves)
    require(len(encoded(out))<=maximum_bytes,'Game session artifact budget exhausted; no complete-output claim.')
    return out
