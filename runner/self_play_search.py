"""Bounded PUCT over the existing typed rule interpreter; no game-specific dispatch."""
import copy,itertools,math,time
from dataclasses import dataclass,asdict
if __package__:from .game_episodes import require,digest
else:from game_episodes import require,digest

class SearchLimit(ValueError):pass

@dataclass(frozen=True)
class SearchSettings:
    simulations:int=32
    exploration:float=1.5
    maximumDepth:int=128
    maximumNodes:int=2048
    seconds:float=3.
    noiseAlpha:float=.3
    noiseMix:float=.25
    reuseTree:bool=False
    def __post_init__(self):
        require(type(self.reuseTree) is bool,'Invalid tree reuse setting.')
        for key,lo,hi in [('simulations',1,4096),('maximumDepth',1,512),('maximumNodes',2,10000)]:
            v=getattr(self,key);require(type(v) is int and lo<=v<=hi,'Invalid search '+key+'.')
        for key,lo,hi in [('exploration',0,100),('seconds',.001,60),('noiseAlpha',.001,10),('noiseMix',0,1)]:
            v=getattr(self,key);require(type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi,'Invalid search '+key+'.')
    def value(self):return asdict(self)

class Node:
    def __init__(self,state,actor,outcome):
        self.state=copy.deepcopy(state);self.actor=actor;self.outcome=outcome
        self.priors=None;self.n={};self.w={};self.children={}

class Search:
    def __init__(self,definition,policy,settings,*,check_cancelled=lambda:None,clock=time.monotonic):
        self.root=None
        self.definition=definition;self.policy=policy;self.settings=settings;self.check_cancelled=check_cancelled;self.clock=clock
        self.catalog=[dict(zip([k for k,_ in definition.action_values],v)) for v in itertools.product(*(v for _,v in definition.action_values))]
        self.indices={digest(v):i for i,v in enumerate(self.catalog)}
        require(1<=len(self.catalog)<=256,'Search requires 1…256 stable action indices.')
    def run(self,state,rng,*,noise=False):
        started=self.clock();deadline=started+self.settings.seconds;definition=self.definition;original=definition.check_cancelled
        def check():
            self.check_cancelled();original()
            if self.clock()>=deadline:raise SearchLimit('Search time limit reached; no unsearched move was substituted.')
        def rule(method,*args):
            check();definition.remaining=2_000_000;return method(*args)
        def node(s):return Node(s,rule(definition.env,s)['actor'],rule(definition.outcome,s))
        def terminal(n):return 0. if n.outcome['winner'] is None else 1. if n.outcome['winner']==n.actor else -1.
        def probe(n,expand):
            if n.outcome['terminal']:return terminal(n)
            observation=rule(definition.observation,n.state,n.actor)
            logits,value=self.policy.predict(observation);check()
            require(type(logits) is list and len(logits)==len(self.catalog) and all(type(v) in (int,float) and math.isfinite(v) for v in logits),'Policy must emit one finite logit per fixed action.')
            require(type(value) in (int,float) and math.isfinite(value) and -1<=value<=1,'Value estimate must be finite in −1…1.')
            legal=[self.indices[digest(a)] for a in rule(definition.legal_actions,n.state)]
            require(bool(legal),'Nonterminal search state has no legal actions.')
            if expand:
                maximum=max(logits[i] for i in legal);weights={i:math.exp(logits[i]-maximum) for i in legal};total=sum(weights.values())
                n.priors={i:w/total for i,w in weights.items()};n.base_priors=dict(n.priors);n.n={i:0 for i in legal};n.w={i:0. for i in legal}
            return float(value)
        definition.check_cancelled=check
        try:
            root=self.root if self.settings.reuseTree and self.root is not None and digest(self.root.state)==digest(state) else node(state)
            self.root=root
            if root.outcome['terminal']:return dict(terminal=True,winner=root.outcome['winner'],visits=[0]*len(self.catalog),nodes=1)
            if root.priors is None:probe(root,True);root.base_priors=dict(root.priors)
            else:root.priors=dict(root.base_priors)
            old_visits=dict(root.n);nodes=1
            if self.settings.reuseTree:
                pending=[root]
                while pending:
                    item=pending.pop();pending.extend(item.children.values());nodes+=len(item.children)
            if noise and self.settings.noiseMix:
                samples={i:rng.gammavariate(self.settings.noiseAlpha,1) for i in root.priors};total=sum(samples.values())
                require(total>0 and math.isfinite(total),'Root exploration sample is invalid.')
                root.priors={i:(1-self.settings.noiseMix)*p+self.settings.noiseMix*samples[i]/total for i,p in root.priors.items()}
            for _ in range(self.settings.simulations):
                check();current=root;path=[];depth=0
                while not current.outcome['terminal'] and current.priors is not None and depth<self.settings.maximumDepth:
                    total=max(1,sum(current.n.values()))
                    action=max(current.priors,key=lambda i:(current.w[i]/current.n[i] if current.n[i] else 0.)+self.settings.exploration*current.priors[i]*math.sqrt(total)/(1+current.n[i]))
                    path.append((current,action));depth+=1
                    if action not in current.children:
                        candidate=rule(definition.step,current.state,self.catalog[action])['state'];child=node(candidate)
                        if nodes<self.settings.maximumNodes:current.children[action]=child;nodes+=1
                        current=child;break
                    current=current.children[action]
                value=terminal(current) if current.outcome['terminal'] else probe(current,nodes<self.settings.maximumNodes and depth<self.settings.maximumDepth)
                leaf_actor=current.actor
                for parent,action in path:
                    parent.n[action]+=1;parent.w[action]+=value if parent.actor==leaf_actor else -value
            visits=[root.n.get(i,0)-old_visits.get(i,0) for i in range(len(self.catalog))];total=sum(visits)
            require(total==self.settings.simulations,'Search visit ledger differs from its admitted simulations.')
            best=max(root.n,key=lambda i:visits[i])
            return dict(terminal=False,actionIndex=best,visits=visits,probabilities=[n/total for n in visits],legalMask=[i in root.priors for i in range(len(visits))],q=[root.w.get(i,0)/root.n[i] if root.n.get(i) else 0. for i in range(len(visits))],nodes=nodes)
        finally:definition.check_cancelled=original

    def advance(self,action,state):
        if not self.settings.reuseTree:self.root=None;return
        child=self.root.children.get(action) if self.root is not None else None
        self.root=child if child is not None and digest(child.state)==digest(state) else None
