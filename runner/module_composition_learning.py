"""Real, source-qualified tiny container lessons; scalar expectations are independent."""
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path
if __package__:
    from .module_composition_reference import AffineParameters, evaluate, evaluate_outer, left_sgd_step
    from .module_composition_source import reexpand
else:
    from module_composition_reference import AffineParameters, evaluate, evaluate_outer, left_sgd_step
    from module_composition_source import reexpand

TASKS = ('composition-build-reuse', 'composition-order-name')
SHAPES = {
 TASKS[0]:dict(branchOutputs=[3,3],nestedInfluences=[11],privateAdjustment=[8],modePropagation=[3,2],savedReuse=[9]),
 TASKS[1]:dict(sequentialInfluences=[10],orderedOutputs=[3,3],namedOutputs=[3,2],structuralIdentity=[1],savedReuse=[3,3])}

def _node(i,t,config=None):return dict(id=i,type=t,name=i,x=0,y=0,config=config or {})
def _edge(i,s,t,sp='out',tp='in'):return dict(id=i,source=s,sourcePort=sp,target=t,targetPort=tp)
def _ref(i):return dict(id=i,version=dict(major=1,minor=0,patch=0))
def _instance(i,definition):return dict(id=i,definition=_ref(definition),name=i,x=0,y=0,overrides={})
def _input(*targets):return dict(id='in',targets=[dict(componentID=n,portID=p) for n,p in targets])
def _output(i,n,p='out'):return dict(id=i,source=dict(componentID=n,portID=p))
def _policy(kind,children):return dict(schemaVersion=1,kind=kind,children=[dict(localID=i,**({'key':k} if k is not None else {})) for i,k in children])
def _definition(i,nodes,instances,edges,inputs,outputs,policy):return dict(reference=_ref(i),title=i,body=dict(configurationVersion=2,nodes=nodes,instances=instances,edges=edges),inputs=inputs,outputs=outputs,parameters=[],composition=policy)
def _linear(i,width=1):return _node(i,'linear',dict(in_features=width,out_features=1,bias=True))

def source_template(task, *, complete=True):
    if task not in TASKS:raise ValueError('Choose a current module composition task.')
    if task==TASKS[0]:
        inner=_definition('inner',[_linear('a'),_linear('b'),_node('join','add',dict(alpha=1.,sample_broadcast=False)),_node('probe','dropout',dict(p=1.))],[],[_edge('ab','a','b'),_edge('aj','a','join',tp='a'),_edge('bj','b','join',tp='b'),_edge('probe','join','probe')],[_input(('a','in'))],[_output('out','join'),_output('modeProbe','probe')],_policy('graph',[(i,None) for i in ('a','b','join','probe')]))
        outer=_definition('outer',[_linear('negate'),_node('pair','concat',dict(dim=-1)),_linear('difference',2),_node('modeLeft','add',dict(alpha=0.,sample_broadcast=False)),_node('modeRight','add',dict(alpha=0.,sample_broadcast=False))],[_instance('left','inner'),_instance('right','inner')],[_edge('nr','negate','right'),_edge('lp','left','pair',tp='a'),_edge('rp','right','pair',tp='b'),_edge('pd','pair','difference'),_edge('dm','difference','modeLeft',tp='a'),_edge('lm','left','modeLeft',sp='modeProbe',tp='b'),_edge('mm','modeLeft','modeRight',tp='a'),_edge('rm','right','modeRight',sp='modeProbe',tp='b')],[_input(('left','in'),('negate','in'))],[_output('out','modeRight'),_output('left','left'),_output('right','right'),_output('modeProbe','left','modeProbe')],_policy('graph',[(i,None) for i in ('negate','left','right','pair','difference','modeLeft','modeRight')]))
        definitions=[inner,outer];instances=[_instance('block','outer')];nodes=[_node('input','input',dict(features=1)),_node('output','output')];edges=[_edge('enter','input','block',tp='in'),_edge('leave','block','output')]
    else:
        definitions=[]
        for ident,kind,children in [('sequence','sequential',[('a',None),('b',None)]),('ordered','orderedList',[('b',None),('a',None)]),('named','namedMap',[('b','finish'),('a','start')])]:
            definitions.append(_definition(ident,[_linear('a'),_linear('b')],[],[_edge('ab','a','b')],[_input(('a','in'))],[_output('out','b')],_policy(kind,children)))
        instances=[_instance(i,i) for i in ('sequence','ordered','named')]
        nodes=[_node('input','input',dict(features=1)),_node('pair','concat',dict(dim=-1)),_node('triple','concat',dict(dim=-1)),_node('output','output')]
        edges=[*[_edge('in-'+i,'input',i) for i in ('sequence','ordered','named')],_edge('sp','sequence','pair',tp='a'),_edge('op','ordered','pair',tp='b'),_edge('pt','pair','triple',tp='a'),_edge('nt','named','triple',tp='b'),_edge('out','triple','output')]
    source=dict(schemaVersion=3,name=task,root=dict(configurationVersion=2,nodes=nodes,instances=instances,edges=edges),definitions=definitions)
    if not complete:source['root']['edges']=[]
    return source

def _objective(task):
    if __package__:from .loss_configuration import canonical_configuration
    else:from loss_configuration import canonical_configuration
    width=1 if task==TASKS[0] else 3
    return dict(recipe=dict(schemaVersion=1,configurationVersion=1,componentID='loss_mse',config=canonical_configuration('loss_mse',{})),targetMapping=dict(schemaVersion=1,componentID='loss_mse',inputColumns=['x'],roles=[dict(role='target',columns=['target'+str(i) for i in range(width)],sampleShape=[width],dtype='float32')],tripletSourceGroups=[]))

def configuration(submission, *, task, registry):
    if task not in TASKS or not isinstance(submission,dict) or set(submission)!={'schemaVersion','kind','taskID','source'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='compositionLearningGraph' or submission['taskID']!=task:raise ValueError('Use the current task-qualified authored composition source only.')
    if len(json.dumps(submission,ensure_ascii=False).encode())>2_000_000:raise ValueError('Composition lesson source exceeds its byte budget.')
    source=submission['source'];baseline=source_template(task)
    if not isinstance(source,dict) or set(source)!=set(baseline) or type(source['schemaVersion']) is not int or source['schemaVersion']!=3 or not isinstance(source['definitions'],list) or not 1<=len(source['definitions'])<=64:raise ValueError('Keep the supplied source version and bounded private block definitions.')
    def reference(value):
        if not isinstance(value,dict) or set(value)!={'id','version'} or not isinstance(value['id'],str) or not value['id'] or len(value['id'])>128:raise ValueError('Use bounded pinned private definition references.')
        v=value['version']
        if not isinstance(v,dict) or set(v)!={'major','minor','patch'} or any(type(v[k]) is not int or not 0<=v[k]<=2**31-1 for k in v):raise ValueError('Use exact definition revisions.')
        return value['id'],v['major'],v['minor'],v['patch']
    definitions={}
    for definition in source['definitions']:
        key=reference(definition['reference'])
        if key in definitions:raise ValueError('Definition references must resolve uniquely.')
        definitions[key]=definition
    original_definitions={reference(d['reference']):d for d in baseline['definitions']}
    def protected(body,original,stack):
        if not isinstance(body,dict) or set(body)!=set(original) or body['configurationVersion']!=2 or len(body['nodes'])!=len(original['nodes']) or len(body['instances'])!=len(original['instances']):raise ValueError('Keep the supplied block membership.')
        for a,b in zip(body['nodes'],original['nodes']):
            if set(a)!=set(b) or any(a[k]!=b[k] for k in ('id','type','config')):raise ValueError('Keep supplied model widths and behaviors; edit the connections and container policy.')
        for a,b in zip(body['instances'],original['instances']):
            if set(a)!=set(b) or any(a[k]!=b[k] for k in ('id','overrides')):raise ValueError('Keep every supplied private instance role.')
            key=reference(a['definition'])
            if key not in definitions or key in stack or len(stack)>=3:raise ValueError('Private roles must resolve without cycles within their supplied depth.')
            protected_definition(definitions[key],original_definitions[reference(b['definition'])],stack+(key,))
    def protected_definition(actual,expected,stack):
        if set(actual)!=set(expected) or actual['parameters']!=expected['parameters']:raise ValueError('Keep protected definition controls and membership.')
        protected(actual['body'],expected['body'],stack)
    protected(source['root'],baseline['root'],())
    # Copy-on-write retains previous revisions. None may conceal a foreign role.
    for key,definition in definitions.items():
        matched=False
        for expected in baseline['definitions']:
            try:protected_definition(definition,expected,(key,))
            except (ValueError,KeyError,TypeError):continue
            matched=True;break
        if not matched:raise ValueError('Every retained definition must preserve a supplied protected role.')
    return reexpand(source,registry,objective_context=_objective(task))

def _fixture(inputs):
    if set(inputs)!=set(('coefficients','rows','upstream','targets')):raise ValueError('Use the supplied coefficient, observation, influence and target tables.')
    shapes=dict(coefficients=[4],rows=[3,1],upstream=[3,1],targets=[3,1])
    import math
    for k,shape in shapes.items():
        t=inputs[k]
        if set(t)!=set(('shape','values')) or t['shape']!=shape or len(t['values'])!=math.prod(shape) or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>100 for v in t['values']):raise ValueError('Keep complete bounded finite fixture tables.')
    import struct
    if len({struct.pack('f',v) for v in inputs['rows']['values']})!=3:raise ValueError('Observation rows must remain distinct at float32 precision.')
    p=AffineParameters(*inputs['coefficients']['values'])
    if p.c*p.b+p.d==p.a*p.d+p.b:raise ValueError('Use supplied noncommuting affine steps.')
    return p,inputs['rows']['values'],inputs['upstream']['values'],inputs['targets']['values']

def measure(submission, inputs, *, task, registry, device='cpu'):
    import torch
    if __package__:
        from .engine import GraphModel, export_source
        from .native_ownership import native_owners
    else:
        from engine import GraphModel, export_source
        from native_ownership import native_owners
    expansion=configuration(submission,task=task,registry=registry)
    p,rows,upstream,targets=_fixture(inputs)
    graph=expansion['expandedGraph'];origins=expansion['nodeOrigins'];plan=graph['compositionPlan']
    model=GraphModel(graph,registry=registry,bundled_dataset=False,objective_context=_objective(task)).to(device)
    owners=native_owners(model);values=dict(a=(p.a,p.b),b=(p.c,p.d));qualified={}
    with torch.no_grad():
        for meta,param in owners:
            origin=origins[meta['node']];local=origin['localNodeID'];qualified[(tuple(origin['instancePath']),local,meta['parameter'])]=param
            if local in values:value=values[local][meta['parameter']=='bias']
            elif local=='negate':value=-1. if meta['parameter']=='weight' else 0.
            elif local=='difference':value=[1.,-1.] if meta['parameter']=='weight' else 0.
            else:raise ValueError('Unexpected learned owner in lesson source.')
            param.copy_(torch.tensor(value,device=device).expand_as(param) if not isinstance(value,list) else torch.tensor(value,device=device).reshape(param.shape))
            if local in ('negate','difference'):param.requires_grad_(False)
    independent=len({v.untyped_storage().data_ptr() for _,v in owners})==len(owners)
    if not independent:raise ValueError('Private instances must own distinct native parameter storage.')
    def node(path,local):return next(k for k,v in origins.items() if v['instancePath']==path and v['localNodeID']==local)
    calls=[];hooks=[]
    for module in model.modules():
        if isinstance(module,torch.nn.Sequential):hooks.append(module.register_forward_hook(lambda *args:calls.append('sequential')))
    x=torch.tensor(rows,device=device).reshape(3,1).requires_grad_();u=torch.tensor(upstream,device=device).reshape(3,1)
    operations=[]
    model.eval()
    try:out,stages=model(x,inspect=True,operation_observer=lambda phase,n:operations.append((phase,n)))
    finally:
        for h in hooks:h.remove()
    result={k:dict(shape=s,values=[0.]*__import__('math').prod(s)) for k,s in SHAPES[task].items()}
    def difference(key,actual,expected):
        actual=torch.as_tensor(actual).detach().cpu().flatten().tolist() if isinstance(actual,torch.Tensor) else actual
        if len(actual)!=len(expected):raise ValueError('Measured role shape differs from the reviewed lesson role.')
        result[key]['values']=[float(a)-float(b) for a,b in zip(actual,expected)]
    def inventory_matches():
        expected=source_template(task);gold=reexpand(expected,registry,objective_context=_objective(task))['expandedGraph']['compositionPlan']
        return plan==gold
    expected_owners=[];native_structure=[]
    graph_nodes={n['id']:n for n in graph['nodes']};by_container={c['id']:c for c in plan['containers']}
    def inspect_native(ident,path):
        container=by_container[ident];native=model.get_submodule(path)
        kind=container['kind'];cls={'graph':torch.nn.Module,'sequential':torch.nn.Sequential,'orderedList':torch.nn.ModuleList,'namedMap':torch.nn.ModuleDict}[kind]
        valid=isinstance(native,cls)
        children=native.blocks if kind=='graph' else native
        valid=valid and list(children._modules)==[c['key'] for c in container['children']]
        native_structure.append(dict(path=path,kind=kind,keys=list(children._modules),matches=valid))
        for child in container['children']:
            childpath=path+('.blocks.' if kind=='graph' else '.')+child['key']
            if child['kind']=='container':inspect_native(child['id'],childpath)
            elif graph_nodes[child['id']]['type']=='linear':
                cfg=graph_nodes[child['id']]['config']
                expected_owners.extend([dict(node=child['id'],parameter='weight',name=childpath+'.operation.weight',shape=[cfg['out_features'],cfg['in_features']]),dict(node=child['id'],parameter='bias',name=childpath+'.operation.bias',shape=[cfg['out_features']])])
    inspect_native('root','composition_root')
    actual_owners=[{k:m[k] for k in ('node','parameter','name','shape')} for m,_ in owners]
    structural=0. if inventory_matches() and all(r['matches'] for r in native_structure) and expected_owners==actual_owners else 1.
    if task==TASKS[0]:
        left=node(['block','left'],'join');right=node(['block','right'],'join')
        scalar=evaluate('branch',p,rows,upstream);other=evaluate('branch',p,[-r for r in rows],upstream);outer=evaluate_outer(p,p,rows,upstream)
        actual=torch.cat([out,stages[left],stages[right]],dim=1)
        expected=[v for triple in zip(outer.output,scalar.output,other.output) for v in triple]
        difference('branchOutputs',actual,expected)
        selected=[qualified[(('block',side),local,param)] for side in ('left','right') for local,param in [('a','weight'),('a','bias'),('b','weight'),('b','bias')]]
        grads=torch.autograd.grad((out*u).sum(),[x,*selected])
        difference('nestedInfluences',torch.cat([g.flatten() for g in grads]),[*outer.inputInfluence,*[v for _,v in outer.parameterInfluence]])
        oldright=[t.detach().clone() for t in selected[4:]]
        optimizer=torch.optim.SGD(selected[:4],lr=.05)
        if [id(p) for g in optimizer.param_groups for p in g['params']]!=[id(p) for p in selected[:4]]:raise ValueError('Optimizer must own exactly the left private settings once.')
        optimizer.zero_grad(set_to_none=True)
        _,training=model(x.detach(),inspect=True);loss=(training[left]-torch.tensor(targets,device=device).reshape(3,1)).square().mean();loss.backward();optimizer.step()
        expected_update=left_sgd_step(p,p,rows,targets)
        difference('privateAdjustment',[t.item() for t in selected],[expected_update.left.a,expected_update.left.b,expected_update.left.c,expected_update.left.d,p.a,p.b,p.c,p.d])
        if any(not torch.equal(a,b) for a,b in zip(oldright,selected[4:])):result['privateAdjustment']['values'][4]+=1.
        # Mode assay uses starting coefficients again, separate from the adjustment.
        with torch.no_grad():
            for t,v in zip(selected,[p.a,p.b,p.c,p.d]*2):t.fill_(v)
        frozen=selected[0];frozen.requires_grad_(False);before=frozen.detach().clone();probe=node(['block','left'],'probe')
        model.train();_,trainstages=model(x.detach(),inspect=True);train_flags=all(m.training for m in model.modules());model.eval();_,evalstages=model(x.detach(),inspect=True)
        difference('modePropagation',torch.cat([trainstages[probe],evalstages[probe]],dim=1),[v for value in scalar.output for v in (0.,value)])
        if not train_flags or frozen.requires_grad or not torch.equal(frozen,before) or any(m.training for m in model.modules()):result['modePropagation']['values'][0]+=1.
        # A mismatching native structure uses an explicit positive failure marker,
        # never an additive term that could cancel a numerical discrepancy.
        if structural:result['branchOutputs']['values'][0]=1.+max(abs(v) for v in result['branchOutputs']['values'])
        saved_expected=expected
    else:
        expected=evaluate('sequenceAB',p,rows,upstream);reverse=evaluate('sequenceBA',p,rows,upstream)
        sequence=next(c for c in plan['containers'] if c['path']==['sequence'])['outputs'][0]['source']['node'];selected=[qualified[(('sequence',),local,param)] for local,param in [('a','weight'),('a','bias'),('b','weight'),('b','bias')]]
        grads=torch.autograd.grad((stages[sequence]*u).sum(),[x,*selected])
        difference('sequentialInfluences',torch.cat([stages[sequence].flatten(),*[g.flatten() for g in grads]]),[*expected.output,*expected.inputInfluence,*[v for _,v in expected.parameterInfluence]])
        if len(calls)!=1:result['sequentialInfluences']['values'][0]+=1.
        def variant(change,local):
            value=copy.deepcopy(submission);change(value['source'])
            expanded=configuration(value,task=task,registry=registry);m=GraphModel(expanded['expandedGraph'],registry=registry,bundled_dataset=False,objective_context=_objective(task)).to(device)
            with torch.no_grad():
                for meta,param in native_owners(m):
                    o=expanded['nodeOrigins'][meta['node']];param.fill_(values[o['localNodeID']][meta['parameter']=='bias'])
            m.eval();_,s=m(x.detach(),inspect=True);return s[node([local],'b' if local!='rewired' else 'a')]
        def reorder(s):s['definitions'][1]['composition']['children'].reverse()
        reordered=variant(reorder,'ordered')
        changed=copy.deepcopy(submission);d=changed['source']['definitions'][1];d['body']['edges'][0].update(source='b',target='a');d['inputs'][0]['targets'][0]['componentID']='b';d['outputs'][0]['source']['componentID']='a'
        expanded=configuration(changed,task=task,registry=registry);m=GraphModel(expanded['expandedGraph'],registry=registry,bundled_dataset=False,objective_context=_objective(task)).to(device)
        with torch.no_grad():
            for meta,param in native_owners(m):param.fill_(values[expanded['nodeOrigins'][meta['node']]['localNodeID']][meta['parameter']=='bias'])
        m.eval();_,rewired=m(x.detach(),inspect=True)
        difference('orderedOutputs',torch.cat([stages[node(['ordered'],'b')],reordered,rewired[node(['ordered'],'a')]],dim=1),[v for t in zip(expected.output,expected.output,reverse.output) for v in t])
        def rename(s):s['definitions'][2]['composition']['children'][1]['key']='prepare'
        renamed=variant(rename,'named')
        difference('namedOutputs',torch.cat([stages[node(['named'],'b')],renamed],dim=1),[v for y in expected.output for v in (y,y)])
        result['structuralIdentity']['values']=[structural]
        saved_expected=[v for y in expected.output for v in (y,y,y)]
    # Genuine exact bytes and a new native hierarchy, then independent standalone source.
    payload=io.BytesIO();torch.save(model.state_dict(),payload);payload.seek(0)
    fresh=GraphModel(graph,registry=registry,bundled_dataset=False,objective_context=_objective(task)).to(device);fresh.load_state_dict(torch.load(payload,weights_only=True,map_location=device),strict=True);fresh.eval()
    flags={name:p.requires_grad for name,p in model.named_parameters()}
    for name,p in fresh.named_parameters():p.requires_grad_(flags[name])
    if __package__:from .trainability import validate_recipe as validate_trainability
    else:from trainability import validate_recipe as validate_trainability
    flags_recipe=validate_trainability(dict(schemaVersion=1,configurationVersion=1,componentID='trainability',config=dict(owners=[dict(node=m['node'],parameter=m['parameter'],mode='trainable' if p.requires_grad else 'frozen') for m,p in native_owners(model)])))
    source=export_source(model,trainability_recipe=flags_recipe)
    with tempfile.TemporaryDirectory(prefix='kwipml-composition-lesson-') as folder:
        path=Path(folder)/'model.py';path.write_text(source)
        spec=importlib.util.spec_from_file_location('lesson_saved_model',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        portable=module.Model().to(device);portable.load_state_dict(fresh.state_dict(),strict=True);portable.eval()
        if {name:p.requires_grad for name,p in portable.named_parameters()}!=flags:raise ValueError('Saved standalone model lost selected fixed-setting flags.')
        full=portable(x.detach());micro=torch.cat([portable(row[None]) for row in x.detach()])
        if not torch.equal(full,micro):
            if not torch.allclose(full,micro,atol=2e-6,rtol=0):raise ValueError('Saved model full and microbatch results disagree.')
        if task==TASKS[0]:
            _,saved=portable(x.detach(),inspect=True);actual=torch.cat([full,saved[left],saved[right]],dim=1)
        else:actual=full
        difference('savedReuse',actual,saved_expected)
        # Isolated saved inference imports only generated model source and saved state.
        import subprocess,sys
        state_path=Path(folder)/'state.pt';torch.save(model.state_dict(),state_path)
        script="import importlib.util,json,sys,torch;s=importlib.util.spec_from_file_location('saved',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);x=m.Model();x.load_state_dict(torch.load(sys.argv[2],weights_only=True,map_location='cpu'),strict=True);x.eval();print(json.dumps(x(torch.tensor(json.loads(sys.argv[3]),dtype=torch.float32).reshape(3,1)).detach().tolist()))"
        run=subprocess.run([sys.executable,'-I','-c',script,str(path),str(state_path),json.dumps(rows)],cwd=folder,text=True,capture_output=True,timeout=15)
        if run.returncode:raise ValueError('Saved standalone inference refused: '+run.stderr[-300:])
        cli=torch.tensor(json.loads(run.stdout),device=device)
        if cli.shape!=full.shape or not torch.allclose(cli,full,atol=2e-5,rtol=0):result['savedReuse']['values'][0]+=1.
        if list(portable.state_dict())!=list(model.state_dict()) or list(fresh.state_dict())!=list(model.state_dict()) or any(not torch.equal(v,fresh.state_dict()[k]) or not torch.equal(v,portable.state_dict()[k]) for k,v in model.state_dict().items()):result['savedReuse']['values'][0]+=1.
    return dict(tensors=result,metadata=dict(nativeOwners=[dict(m,requiresGrad=bool(p.requires_grad)) for m,p in owners],nativeContainers=native_structure,invocations=operations,savedStateKeys=list(model.state_dict()),independentStorage=independent,sourceDigest=hashlib.sha256(json.dumps(submission['source'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()))


def _public_metadata(raw):
    """Copy only the bounded public structural record; never serialize checker-local values."""
    def text(value,limit=128):
        if not isinstance(value,str) or not value or '\0' in value or len(value.encode('utf-8'))>limit:raise ValueError('Native composition measurement identity is malformed.')
        return value
    def boolean(value):
        if type(value) is not bool:raise ValueError('Native composition measurement flags must be Boolean.')
        return value
    def array(value,limit):
        if not isinstance(value,(list,tuple)) or len(value)>limit:raise ValueError('Native composition measurement collection exceeds its bound.')
        return value
    owners=[]
    for owner in array(raw['nativeOwners'],256):
        shape=list(array(owner['shape'],7));product=1
        if not shape:raise ValueError('Native learned parameters need their actual shape.')
        for n in shape:
            if type(n) is not int or not 1<=n<=4096 or product>200000//n:raise ValueError('Native learned parameter shape exceeds its bound.')
            product*=n
        if owner['dtype']!='float32':raise ValueError('Lesson native owners must remain float32.')
        owners.append(dict(node=text(owner['node']),parameter=text(owner['parameter']),name=text(owner['name'],4096),shape=shape,dtype='float32',requiresGrad=boolean(owner['requiresGrad'])))
    containers=[]
    for container in array(raw['nativeContainers'],129):
        keys=[text(key) for key in array(container['keys'],128)]
        if not keys or len(set(keys))!=len(keys) or container['kind'] not in ('graph','sequential','orderedList','namedMap'):raise ValueError('Native container keys or kind are malformed.')
        containers.append(dict(path=text(container['path'],4096),kind=container['kind'],keys=keys,matches=boolean(container['matches'])))
    invocations=[]
    for pair in array(raw['invocations'],256):
        if len(array(pair,2))!=2 or pair[0] not in ('started','completed'):raise ValueError('Native invocation requires its actual phase and node.')
        invocations.append([pair[0],text(pair[1])])
    states=[text(key,4096) for key in array(raw['savedStateKeys'],512)]
    if not containers or len({o['name'] for o in owners})!=len(owners) or len({c['path'] for c in containers})!=len(containers) or len(set(states))!=len(states):raise ValueError('Native composition measurements cannot repeat ownership or state.')
    result=dict(schemaVersion=1,nativeOwners=owners,nativeContainers=containers,invocations=invocations,savedStateKeys=states,independentStorage=boolean(raw['independentStorage']))
    if len(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8'))>131072:raise ValueError('Native composition measurements exceed their public byte budget.')
    return result


def execute(submission,fixtures,evidence,stopped,tensor,*,task,registry=None):
    """Measured differences and bounded public structural records; no checker-local value arrays."""
    if not isinstance(fixtures,list) or len(fixtures)!=4 or len({f.get('id') for f in fixtures})!=4:raise ValueError('Use all four uniquely identified composition fixtures.')
    configuration(submission,task=task,registry=registry)
    for fixture in fixtures:
        if stopped():raise ValueError('Composition lesson cancelled or exceeded its budget.')
        measured=measure(submission,fixture['inputs'],task=task,registry=registry)
        if stopped():raise ValueError('Composition lesson cancelled or exceeded its budget.')
        tables=measured['tensors'];count=sum(len(v['values']) for v in tables.values())
        if count>4096:raise ValueError('Measured composition exceeds its tiny fixture budget.')
        metadata=_public_metadata(measured['metadata'])
        evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),count)
        evidence['fixtures'].append(dict(fixtureID=fixture['id'],tensors=tables,partitions={},composition=metadata))
