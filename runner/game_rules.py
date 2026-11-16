"""Versioned pure game-rule data evaluator. Never eval/exec/import user source.

Expressions are parsed Python expression *syntax*, interpreted by this module's explicit
whitelist. No arbitrary Python runtime is exposed. Every node and iteration is budgeted.
"""
import ast
import copy
import hashlib
import itertools
import json
import math
import re

class RuleError(ValueError):pass
class RuleBudgetError(RuleError):pass

def value_cells(value):
    if type(value) is list:
        count=1
        for item in value:
            count+=value_cells(item)
            if count>4096:raise RuleBudgetError("Expression allocation budget exceeds 4096 cells.")
        return count
    return 1

def require(value,message):
    if not value:raise RuleError(message)

def load_definition(text):
    require(isinstance(text,str) and len(text.encode())<=131072,'Definition must be UTF-8 JSON within 128 KiB.')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key: '+key);result[key]=value
        return result
    try:return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(RuleError('Nonfinite JSON value.')))
    except (ValueError,RecursionError) as e:raise RuleError('Definition JSON: '+str(e)) from e

I='integer';B='boolean';F='number';S='string';N='null'
NAME=re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,47}$')
FUNCTIONS={'range','len','sum','min','max','all','any','abs','set_at','repeat'}
BIN={ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod}
CMP={ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.In,ast.NotIn}

def compatible(actual,wanted):return actual==wanted or wanted==F and actual==I or isinstance(wanted,tuple) and wanted[0]=='optional' and (actual==N or compatible(actual,wanted[1]))
def kind(spec):
    t=spec['type']
    if t=='array':return ('array',kind(spec['items']))
    if t=='optional':return ('optional',kind(spec['value']))
    return t

def spec_valid(s,depth=0):
    require(type(s) is dict and depth<=3 and 'type' in s,'Invalid field type.')
    t=s['type'];allowed={'type','initial'}
    if t in (I,F):
        allowed|={'minimum','maximum'}
        require(type(s.get('minimum')) in (int,float) and type(s.get('maximum')) in (int,float) and math.isfinite(s['minimum']) and math.isfinite(s['maximum']) and -1e9<=s['minimum']<=s['maximum']<=1e9,'Numeric fields require finite bounds within ±1e9.')
    elif t==S:
        allowed|={'values'};require(type(s.get('values')) is list and 1<=len(s['values'])<=64 and all(type(x) is str and len(x)<=200 for x in s['values']) and len(set(s['values']))==len(s['values']),'String fields require a finite unique vocabulary.')
    elif t=='array':
        allowed|={'length','items'};require(type(s.get('length')) is int and 1<=s['length']<=256,'Array length must be 1…256.');spec_valid(s['items'],depth+1)
    elif t=='optional':allowed|={'value'};spec_valid(s['value'],depth+1)
    else:require(t==B,'Unsupported field type: '+str(t))
    require(not set(s)-allowed,'Unknown field constraint.')

def check_value(v,s,path):
    t=s['type']
    if t=='optional':
        if v is not None:check_value(v,s['value'],path)
    elif t=='array':
        require(type(v) is list and len(v)==s['length'],path+': wrong array length.')
        for i,x in enumerate(v):check_value(x,s['items'],path+f'[{i}]')
    elif t in (I,F):require(type(v) is int if t==I else type(v) in (int,float),path+': wrong numeric type.');require(math.isfinite(v) and s['minimum']<=v<=s['maximum'],path+': number outside declared bounds.')
    elif t==B:require(type(v) is bool,path+': expected Boolean.')
    elif t==S:require(type(v) is str and v in s['values'],path+': unknown enum value.')

class Expression:
    def __init__(self,text,scope,wanted=None):
        require(type(text) is str and 0<len(text)<=4096,'Rule expression must be 1…4096 characters.')
        try:self.tree=ast.parse(text,mode='eval').body
        except (SyntaxError,RecursionError) as e:raise RuleError('Expression syntax: '+str(e)) from e
        require(sum(1 for _ in ast.walk(self.tree))<=1000,'Expression exceeds 1000 syntax nodes.')
        self.type=self.infer(self.tree,scope,0)
        if wanted is not None:require(compatible(self.type,wanted),f'Expression type {self.type} does not match {wanted}.')
    def infer(self,n,scope,d):
        require(d<=30,'Expression nesting exceeds 30.');infer=lambda x:self.infer(x,scope,d+1)
        if isinstance(n,ast.Constant):
            require(n.value is None or type(n.value) in (int,float,bool,str),'Unsupported literal.')
            if type(n.value) in (int,float):require(math.isfinite(n.value) and abs(n.value)<=1e9,'Literal outside finite bounds.')
            if type(n.value) is str:require(len(n.value)<=200,'Text literal too long.')
            return N if n.value is None else B if type(n.value) is bool else I if type(n.value) is int else F if type(n.value) is float else S
        if isinstance(n,ast.Name):require(n.id in scope,'Unknown field or variable: '+n.id);return scope[n.id]
        if isinstance(n,ast.Attribute):
            base=infer(n.value);require(type(base) is dict and n.attr in base,'Unknown record field: '+n.attr);return base[n.attr]
        if isinstance(n,ast.Subscript):
            base=infer(n.value);index=infer(n.slice);require(type(base) is tuple and base[0]=='array' and index==I,'Array index must be integer.');return base[1]
        if isinstance(n,(ast.List,ast.Tuple)):
            require(1<=len(n.elts)<=256,'Literal arrays need 1…256 values.');types=[infer(x) for x in n.elts];require(all(x==types[0] for x in types),'Array types must agree.');return ('array',types[0])
        if isinstance(n,ast.UnaryOp):
            t=infer(n.operand)
            if isinstance(n.op,ast.Not):require(t==B,'Not requires Boolean.');return B
            require(type(n.op) in (ast.USub,ast.UAdd) and t in (I,F),'Invalid numeric unary operation.');return t
        if isinstance(n,ast.BinOp):
            a,b=infer(n.left),infer(n.right);require(type(n.op) in BIN and a in (I,F) and b in (I,F),'Only bounded numeric arithmetic is supported.');return F if isinstance(n.op,ast.Div) or F in (a,b) else I
        if isinstance(n,ast.BoolOp):require(type(n.op) in (ast.And,ast.Or) and all(infer(x)==B for x in n.values),'Boolean logic needs Boolean terms.');return B
        if isinstance(n,ast.Compare):
            prev=infer(n.left)
            for op,x in zip(n.ops,n.comparators):
                t=infer(x);require(type(op) in CMP,'Unsupported comparison.')
                if isinstance(op,(ast.In,ast.NotIn)):require(type(t) is tuple and t[0]=='array' and compatible(prev,t[1]),'Membership type mismatch.')
                elif isinstance(op,(ast.Eq,ast.NotEq)):require(compatible(prev,t) or compatible(t,prev) or N in (prev,t),'Equality type mismatch.')
                else:require(prev in (I,F) and t in (I,F),'Ordering needs numeric values.')
                prev=t
            return B
        if isinstance(n,ast.IfExp):
            require(infer(n.test)==B,'Conditional needs Boolean.');a,b=infer(n.body),infer(n.orelse)
            if a==b:return a
            if N in (a,b):return ('optional',b if a==N else a)
            require(a in (I,F) and b in (I,F),'Conditional branch types must agree.');return F
        if isinstance(n,(ast.ListComp,ast.GeneratorExp)):
            require(1<=len(n.generators)<=3,'At most three bounded generators.');local=dict(scope)
            for g in n.generators:
                require(not g.is_async and isinstance(g.target,ast.Name) and NAME.fullmatch(g.target.id) and g.target.id not in local,'Generator must bind a fresh simple variable.')
                seq=self.infer(g.iter,local,d+1);require(type(seq) is tuple and seq[0]=='array','Generator needs a finite sequence.');local[g.target.id]=seq[1]
                require(all(self.infer(x,local,d+1)==B for x in g.ifs),'Generator filters need Boolean.')
            return ('array',self.infer(n.elt,local,d+1))
        if isinstance(n,ast.Call):
            require(isinstance(n.func,ast.Name) and n.func.id in FUNCTIONS and not n.keywords,'Only declared rule functions are supported.')
            f=n.func.id;ts=[infer(x) for x in n.args]
            if f=='range':require(1<=len(ts)<=3 and all(t==I for t in ts),'range needs integer bounds.');return ('array',I)
            if f=='repeat':require(len(ts)==2 and ts[1]==I,'repeat(value, count) requires integer count.');return ('array',ts[0])
            if f=='set_at':require(len(ts)==3 and type(ts[0]) is tuple and ts[0][0]=='array' and ts[1]==I and compatible(ts[2],ts[0][1]),'set_at(array, index, value) type mismatch.');return ts[0]
            if f=='abs':require(len(ts)==1 and ts[0] in (I,F),'abs needs one number.');return ts[0]
            if f=='len':require(len(ts)==1 and type(ts[0]) is tuple and ts[0][0]=='array','len needs an array.');return I
            if f in ('all','any'):require(ts==[('array',B)],f+' needs Boolean values.');return B
            if f in ('sum','min','max'):
                if len(ts)==1 and type(ts[0]) is tuple and ts[0][0]=='array':t=ts[0][1];require(t in (I,F),f+' needs numeric values.');return t
                require(f!='sum' and len(ts)>=2 and all(t in (I,F) for t in ts),'min/max need numeric arguments.');return F if F in ts else I
        raise RuleError('Unsupported expression operation: '+type(n).__name__)
    def value(self,env,job):return self.evaluate(self.tree,env,job)
    def evaluate(self,n,e,j):
        j.tick();ev=lambda x:self.evaluate(x,e,j)
        try:
            if isinstance(n,ast.Constant):return n.value
            if isinstance(n,ast.Name):return e[n.id]
            if isinstance(n,ast.Attribute):return ev(n.value)[n.attr]
            if isinstance(n,ast.Subscript):
                a,i=ev(n.value),ev(n.slice);require(0<=i<len(a),'Array index outside bounds.');return a[i]
            if isinstance(n,(ast.List,ast.Tuple)):
                out=[];cells=1
                for x in n.elts:
                    v=ev(x);cells+=value_cells(v)
                    if cells>4096:raise RuleBudgetError('Expression allocation budget exceeds 4096 cells.')
                    out.append(v)
                return out
            if isinstance(n,ast.UnaryOp):return not ev(n.operand) if isinstance(n.op,ast.Not) else -ev(n.operand) if isinstance(n.op,ast.USub) else ev(n.operand)
            if isinstance(n,ast.BinOp):
                a,b=ev(n.left),ev(n.right)
                result={ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b,ast.Div:lambda:a/b,ast.FloorDiv:lambda:a//b,ast.Mod:lambda:a%b}[type(n.op)]()
                require(math.isfinite(result) and abs(result)<=1e12,'Arithmetic exceeds finite bounds.');return result
            if isinstance(n,ast.BoolOp):return all(ev(x) for x in n.values) if isinstance(n.op,ast.And) else any(ev(x) for x in n.values)
            if isinstance(n,ast.Compare):
                a=ev(n.left)
                for op,x in zip(n.ops,n.comparators):
                    b=ev(x);yes={ast.Eq:lambda:a==b,ast.NotEq:lambda:a!=b,ast.Lt:lambda:a<b,ast.LtE:lambda:a<=b,ast.Gt:lambda:a>b,ast.GtE:lambda:a>=b,ast.In:lambda:a in b,ast.NotIn:lambda:a not in b}[type(op)]()
                    if not yes:return False
                    a=b
                return True
            if isinstance(n,ast.IfExp):return ev(n.body) if ev(n.test) else ev(n.orelse)
            if isinstance(n,(ast.ListComp,ast.GeneratorExp)):
                out=[];cells=1
                def collect(index,scope):
                    nonlocal cells
                    if index==len(n.generators):
                        require(len(out)<4096,'Comprehension exceeds 4096 results.')
                        v=self.evaluate(n.elt,scope,j);cells+=value_cells(v)
                        if cells>4096:raise RuleBudgetError('Expression allocation budget exceeds 4096 cells.')
                        out.append(v);return
                    g=n.generators[index]
                    for value in self.evaluate(g.iter,scope,j):
                        j.tick();local={**scope,g.target.id:value}
                        if all(self.evaluate(x,local,j) for x in g.ifs):collect(index+1,local)
                collect(0,e);return out
            if isinstance(n,ast.Call):
                args=[ev(x) for x in n.args];f=n.func.id
                if f=='range':
                    r=range(*args);require(len(r)<=256,'range exceeds 256 values.');return list(r)
                if f=='repeat':
                    require(0<=args[1]<=256,'repeat exceeds 256 values.')
                    if value_cells(args[0])*args[1]+1>4096:raise RuleBudgetError('Expression allocation budget exceeds 4096 cells.')
                    return [copy.deepcopy(args[0]) for _ in range(args[1])]
                if f=='set_at':
                    a,i,v=args;require(0<=i<len(a),'Update index outside bounds.');a=copy.deepcopy(a);a[i]=v;return a
                if f in ('all','any'):return all(args[0]) if f=='all' else any(args[0])
                if f=='sum':return sum(args[0])
                if f=='abs':return abs(args[0])
                return (min if f=='min' else max)(args[0]) if len(args)==1 else (min if f=='min' else max)(args)
        except (ArithmeticError,KeyError,TypeError,IndexError) as error:raise RuleError('Rule evaluation: '+str(error)) from error
        raise RuleError('Unimplemented rule operation.')

class Definition:
    def __init__(self,source,*,maximum_operations=2_000_000,check_cancelled=lambda:None):
        check_cancelled();self.remaining=maximum_operations;self.check_cancelled=check_cancelled;self.tick()
        self.source=load_definition(json.dumps(source,allow_nan=False));x=self.source
        keys={'version','name','state','actions','currentPlayer','legal','transitions','terminal','observations','reward','invariants','tests','parameters'}
        require(type(x) is dict and set(x)==keys and type(x['version']) is int and x['version']==1,'Unsupported definition fields or version.')
        require(type(x['name']) is str and 1<=len(x['name'])<=120,'Definition name is required.')
        require(type(x['parameters']) is dict and len(x['parameters'])<=16 and all(NAME.fullmatch(k) and type(v) is int and 0<=v<=256 for k,v in x['parameters'].items()),'Parameters must be named bounded integers.')
        self.schemas={}
        for key in ('state','actions'):
            fields=x[key];require(type(fields) is dict and 1<=len(fields)<=32 and all(NAME.fullmatch(k) for k in fields),'Invalid '+key+' schema.')
            for k,s in fields.items():spec_valid(s)
            self.schemas[key]={k:kind(s) for k,s in fields.items()}
        self.scope={'state':self.schemas['state'],'action':self.schemas['actions'],'actor':I,'player':I,'params':{k:I for k in x['parameters']}}
        self.initial_state={k:copy.deepcopy(s.get('initial')) for k,s in x['state'].items()};self.validate_state(self.initial_state)
        require(sum(value_cells(v) for v in self.initial_state.values())<=4096,'State exceeds 4096 cells.')
        make=lambda t,w=None,scope=None:Expression(t,scope or self.scope,w)
        state_scope={k:v for k,v in self.scope.items() if k in ('state','params')}
        move_scope={k:v for k,v in self.scope.items() if k in ('state','params','action','actor')}
        self.current=make(x['currentPlayer'],I,state_scope);self.legal=make(x['legal'],B,move_scope)
        require(type(x['transitions']) is list and 1<=len(x['transitions'])<=16,'Define 1…16 transitions.')
        self.transitions=[]
        for t in x['transitions']:
            require(type(t) is dict and set(t)=={'name','when','updates'} and type(t['name']) is str and 1<=len(t['name'])<=120,'Invalid transition fields.')
            require(type(t['updates']) is dict and 1<=len(t['updates'])<=32 and not set(t['updates'])-set(x['state']),'Transition update fields unknown.')
            updates={};pending=dict(t['updates']);ns={**move_scope,'next':{}}
            while pending:
                ready=[]
                for field,text in pending.items():
                    try:tree=ast.parse(text,mode='eval')
                    except (SyntaxError,TypeError,RecursionError) as e:raise RuleError('Update expression syntax: '+str(e)) from e
                    refs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='next'}
                    require(not refs-set(t['updates']),'next references an unwritten field.')
                    if refs<=set(updates):ready.append(field)
                require(ready,'Cyclic next-state dependencies.')
                for field in sorted(ready):
                    updates[field]=make(pending.pop(field),self.schemas['state'][field],ns);ns['next']={**ns['next'],field:self.schemas['state'][field]}
            self.transitions.append((t['name'],make(t['when'],B,move_scope),updates))
        require(type(x['terminal']) is list and 1<=len(x['terminal'])<=16,'Define 1…16 terminal rules.')
        self.terminals=[]
        for t in x['terminal']:
            require(type(t) is dict and set(t)=={'name','when','winner'} and type(t['name']) is str,'Invalid terminal rule.')
            self.terminals.append((t['name'],make(t['when'],B,state_scope),make(t['winner'],('optional',I),state_scope)))
        require(type(x['invariants']) is list and len(x['invariants'])<=16,'Too many invariants.')
        self.invariants=[make(t,B,state_scope) for t in x['invariants']]
        require(type(x['observations']) is list and 1<=len(x['observations'])<=256,'Define 1…256 numeric observation expressions.')
        self.observations=[make(t,F,{**state_scope,'player':I}) for t in x['observations']]
        self.reward=make(x['reward'],F,{**state_scope,'player':I,'outcome':{'terminal':B,'winner':('optional',I)}})
        require(type(x['tests']) is list and 1<=len(x['tests'])<=32,'Define 1…32 tests.')
        self.action_values=[]
        for k,s in sorted(x['actions'].items()):
            require(s['type'] in (I,B,S),'Actions must be finite integer, Boolean or enum fields.')
            require(s['type']!=I or type(s['minimum']) is int and type(s['maximum']) is int and s['maximum']-s['minimum']<256,'Action domain exceeds 256 values.')
            vals=list(range(s['minimum'],s['maximum']+1)) if s['type']==I and type(s['minimum']) is int and type(s['maximum']) is int else [False,True] if s['type']==B else s.get('values',[])
            require(1<=len(vals)<=256,'Action domain exceeds 256 values.');self.action_values.append((k,vals))
        require(math.prod(len(v) for _,v in self.action_values)<=256,'Action product exceeds 256 candidates.')
        self.canonical=json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
        # Next-field dependencies determine evaluation order, independent of JSON key order.
        self.identity=hashlib.sha256(self.canonical.encode()).hexdigest()
        self.invariant_check(self.initial_state)
    def tick(self):
        self.remaining-=1
        if self.remaining<0:raise RuleBudgetError('Rule operation budget exhausted.')
        if self.remaining%64==0:self.check_cancelled()
        if self.remaining==1_999_999:self.check_cancelled()
    def validate_state(self,state):
        require(type(state) is dict and set(state)==set(self.source['state']),'State fields must match the definition exactly.')
        for k,s in self.source['state'].items():check_value(state[k],s,'state.'+k)
    def env(self,state,action=None,player=0):
        e={'state':state,'action':action or {},'params':self.source['parameters'],'actor':0,'player':player}
        actor=self.current.value(e,self);require(type(actor) is int and actor in (0,1),'currentPlayer must be 0 or 1.');e['actor']=actor;return e
    def invariant_check(self,state):
        require(all(t.value(self.env(state),self) for t in self.invariants),'State invariant failed.')
    def initial(self):return copy.deepcopy(self.initial_state)
    def outcome(self,state):
        e=self.env(state);matches=[(name,w.value(e,self)) for name,c,w in self.terminals if c.value(e,self)]
        require(len(matches)<=1,'Terminal rules are ambiguous; make conditions mutually exclusive.')
        winner=matches[0][1] if matches else None;require(winner is None or type(winner) is int and winner in (0,1),'Winner must be player 0/1 or null for draw.')
        return {'terminal':bool(matches),'winner':winner}
    def legal_actions(self,state):
        self.validate_state(state);self.invariant_check(state)
        if self.outcome(state)['terminal']:return []
        keys=[k for k,_ in self.action_values];actions=[]
        for vs in itertools.product(*(v for _,v in self.action_values)):
            self.tick();a=dict(zip(keys,vs))
            if self.legal.value(self.env(state,a),self):actions.append(a)
        require(bool(actions),'Nonterminal state has no legal actions; define its terminal or pass rule explicitly.');return actions
    def observation(self,state,player):
        values=[t.value(self.env(state,player=player),self) for t in self.observations]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in values),'Observation must be finite numeric values.');return values
    def step(self,state,action):
        self.check_cancelled();self.validate_state(state);self.invariant_check(state)
        require(not self.outcome(state)['terminal'],'Cannot step a terminal state.')
        require(type(action) is dict and set(action)==set(self.source['actions']),'Action fields must match the definition exactly.')
        try:
            for k,s in self.source['actions'].items():check_value(action[k],s,'action.'+k)
        except RuleError as e:raise RuleError('illegal action: '+str(e)) from e
        e=self.env(state,action);require(self.legal.value(e,self),'illegal action: predicate refused the move.')
        matches=[(name,updates) for name,c,updates in self.transitions if c.value(e,self)]
        require(len(matches)==1,'Transition is ambiguous or missing for this legal action.')
        name,updates=matches[0];candidate=copy.deepcopy(state);next_values={}
        for k,t in updates.items():next_values[k]=t.value({**e,'next':next_values},self);candidate[k]=next_values[k]
        self.validate_state(candidate);self.invariant_check(candidate);o=self.outcome(candidate)
        rewards=[self.reward.value({**self.env(candidate,player=p),'outcome':o},self) for p in (0,1)]
        require(all(type(r) in (int,float) and math.isfinite(r) and -1<=r<=1 for r in rewards) and sum(rewards)==0,'Rewards must be finite, zero-sum and within −1…1.')
        return dict(state=candidate,**o,rewards=rewards,rule=name,observation=self.observation(candidate,self.env(candidate)['actor']))
    def run_tests(self):
        tests=[];trace=[];trace_bytes=0
        require(len({x.get('name') for x in self.source['tests'] if type(x) is dict})==len(self.source['tests']),'Test names must be unique.')
        for i,test in enumerate(self.source['tests']):
            self.check_cancelled();require(type(test) is dict and set(test)<= {'name','state','actions','expect','expectError'} and type(test.get('name')) is str and len(test['name'])<=120,'Invalid test fields.')
            require(type(test.get('actions')) is list and len(test['actions'])<=128,'Test actions exceed 128 moves.')
            state=copy.deepcopy(test.get('state',self.initial()));self.validate_state(state);self.invariant_check(state)
            error=None;out=dict(state=state,**self.outcome(state));start=len(trace)
            for move,action in enumerate(test['actions']):
                require(len(trace)<2048,'Test trace exceeds 2048 moves.');before=copy.deepcopy(state)
                try:
                    out=self.step(state,action);state=out['state'];trace.append(dict(test=test['name'],move=move+1,before=before,action=action,after=copy.deepcopy(state),status='applied',rule=out['rule'],terminal=out['terminal'],winner=out['winner'],rewards=out['rewards']))
                except RuleError as e:
                    if isinstance(e,RuleBudgetError):raise
                    error=str(e);trace.append(dict(test=test['name'],move=move+1,before=before,action=action,after=copy.deepcopy(state),status='refused',rule=error,terminal=out['terminal'],winner=out['winner'],rewards=[]))
                    trace_bytes+=len(json.dumps(trace[-1],allow_nan=False).encode())
                    if trace_bytes>8_000_000:raise RuleBudgetError('Trace exceeds its 8 MB retention budget.')
                    break
                trace_bytes+=len(json.dumps(trace[-1],allow_nan=False).encode())
                if trace_bytes>8_000_000:raise RuleBudgetError('Trace exceeds its 8 MB retention budget.')
            expect=test.get('expect',{});require(type(expect) is dict and set(expect)<= {'state','terminal','winner','rewards'},'Unknown test expectation.')
            require(bool(expect) or type(test.get('expectError')) is str and bool(test['expectError']),'Every test needs a nonempty expectation or expected error.')
            ok=bool(error and test.get('expectError') and test['expectError'] in error) if test.get('expectError') else error is None
            for k,v in expect.items():
                if k=='state':ok=ok and type(v) is dict and bool(v) and not set(v)-set(state) and all(state[f]==a and (type(state[f]) is not bool or type(a) is bool) for f,a in v.items())
                else:ok=ok and out.get(k)==v and (k!='terminal' or type(v) is bool) and (k!='winner' or v is None or type(v) is int)
            tests.append(dict(test=test['name'],status='passed' if ok else 'failed',detail=error or ('Expectations matched.' if ok else 'Expected state/outcome differs.'),moves=len(trace)-start,definition=self.identity))
        return dict(tests=tests,trace=trace,identity=self.identity,definition=self.canonical)
