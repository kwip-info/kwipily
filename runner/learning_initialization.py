"""Small real starting-settings lessons; saved values are not training continuation.

The submitted recipe initializes an actual GraphModel. Expected arrays use
independent random primitives and scalar math, never nn.init or helper outputs.
"""
import copy
import hashlib
import io
import json
import math
import struct
import torch
if __package__:
    from . import initialization
    from .engine import GraphModel
    from .optimizer_configuration import canonical_configuration as optimizer_configuration
    from .loss_configuration import validate_recipe as objective_recipe
else:
    import initialization
    from engine import GraphModel
    from optimizer_configuration import canonical_configuration as optimizer_configuration
    from loss_configuration import validate_recipe as objective_recipe

TASKS=('initialization-starting-values','initialization-starting-spread')
WIDTHS=((2,3,2),(3,4,2),(4,3,2),(2,5,3))
SGD={'lr':.05,'momentum':0.,'dampening':0.,'weight_decay':0.,'nesterov':False,'maximize':False}
KEYS=('firstWeight','firstBias','secondWeight','secondBias')
PARAMETERS=('layers.first.weight','layers.first.bias','layers.second.weight','layers.second.bias')


def task_recipe(task):
    if task not in TASKS:raise ValueError('Choose a supported starting-settings lesson.')
    def rule(node,parameter,algorithm,settings):return {'owner':{'node':node,'parameter':parameter},'algorithm':algorithm,'settings':settings}
    rules=[rule('first','bias','zeros',{}),rule('first','weight','normal',{'mean':.15,'std':.2})] if task==TASKS[0] else [rule('first','weight','xavier_uniform',{'gain':.7}),rule('second','weight','kaiming_uniform',{'a':.2,'mode':'fan_out','nonlinearity':'leaky_relu'})]
    return {'schemaVersion':1,'configurationVersion':1,'componentID':'initialization','config':{'seed':42,'rules':rules}}


def _submission(submission,task):
    if task not in TASKS or type(submission) is not dict or set(submission)!={'schemaVersion','kind','taskID','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='initializationLearningGraph' or submission['taskID']!=task:
        raise ValueError('Submit the starting-settings graph, never answers, files or saved state.')
    try:size=len(json.dumps(submission,ensure_ascii=False,allow_nan=False).encode('utf-8'))
    except (TypeError,ValueError,UnicodeError):raise ValueError('Lesson submission requires finite portable JSON.') from None
    if size>2000000:raise ValueError('Lesson submission exceeds its 2 MB limit.')
    graph=submission['graph']
    if type(graph) is not dict or set(graph)!={'schemaVersion','name','nodes','edges'} or type(graph['schemaVersion']) is not int or graph['schemaVersion']!=9 or type(graph['name']) is not str or len(graph['name'])>256 or type(graph['nodes']) is not list or len(graph['nodes'])!=3 or type(graph['edges']) is not list or len(graph['edges'])!=2:
        raise ValueError('Connect Starting settings and supplied SGD to the unchanged Train context.')
    by={}
    for node in graph['nodes']:
        if type(node) is not dict or set(node)!={'id','type','name','x','y','config'} or type(node['id']) is not str or not 1<=len(node['id'])<=128 or '\0' in node['id'] or node['id'] in by or type(node['name']) is not str or len(node['name'])>256 or any(type(node[key]) not in (int,float) or not math.isfinite(node[key]) for key in ('x','y')):
            raise ValueError('Malformed starting-settings lesson block.')
        by[node['id']]=node
    initializers=[node for node in by.values() if node['type']=='initialization']
    optimizers=[node for node in by.values() if node['type']=='optimizer_sgd']
    trains=[node for node in by.values() if node['type']=='train' and node['config']=={}]
    if len(initializers)!=1 or len(optimizers)!=1 or len(trains)!=1:raise ValueError('Keep one initializer, one supplied SGD and the supplied Train context.')
    supplied=optimizers[0]['config']
    if type(supplied) is not dict or set(supplied)!=set(SGD) or optimizer_configuration('optimizer_sgd',supplied)!=SGD:
        raise ValueError('Keep supplied SGD at rate0.05 with no momentum or decay.')
    expected={(initializers[0]['id'],'out',trains[0]['id'],'initialization'),(optimizers[0]['id'],'out',trains[0]['id'],'optimizer')}
    actual=set();ids=set()
    for edge in graph['edges']:
        if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'} or type(edge['id']) is not str or not 1<=len(edge['id'])<=128 or edge['id'] in ids or any(type(edge[key]) is not str for key in ('source','sourcePort','target','targetPort')):raise ValueError('Malformed starting-settings lesson wire.')
        ids.add(edge['id']);actual.add((edge['source'],edge['sourcePort'],edge['target'],edge['targetPort']))
    if actual!=expected:raise ValueError('Connect initializer out to Train.initialization and SGD out to Train.optimizer.')
    recipe=initialization.validate_recipe({'schemaVersion':1,'configurationVersion':1,'componentID':'initialization','config':initializers[0]['config']})
    return recipe,optimizer_configuration('optimizer_sgd',supplied)


def configuration(submission,task):
    return _submission(submission,task)[0]


def _f32(value):
    try:result=struct.unpack('f',struct.pack('f',value))[0]
    except (OverflowError,struct.error):raise ValueError('Supplied lesson value exceeds finite float32.') from None
    if not math.isfinite(result):raise ValueError('Supplied lesson values must remain finite.')
    return result


def _fixtures(fixtures,tensor):
    if type(fixtures) is not list or not 1<=len(fixtures)<=4:raise ValueError('Use up to four bounded supplied starting-settings fixtures.')
    result=[];ids=set()
    for fixture in fixtures:
        if type(fixture) is not dict or set(fixture)!={'id','inputs'} or type(fixture['id']) is not str or not 1<=len(fixture['id'])<=128 or fixture['id'] in ids or type(fixture['inputs']) is not dict or set(fixture['inputs'])!={'x','target',*KEYS}:
            raise ValueError('Use only supplied input/desired rows and the four learned arrays.')
        vals={key:tensor(value) for key,value in fixture['inputs'].items()}
        for shape,values in vals.values():
            if type(shape) is not list or any(type(v) is not int or v<=0 for v in shape) or type(values) is not list or math.prod(shape)!=len(values) or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e6 for v in values):raise ValueError('Malformed bounded starting-settings fixture.')
        xshape=vals['x'][0];wshape=vals['firstWeight'][0];oshape=vals['secondWeight'][0]
        if len(xshape)!=2 or len(wshape)!=2 or len(oshape)!=2:raise ValueError('Starting-settings rows and stored weights require two directions.')
        i,h,o=xshape[1],wshape[0],oshape[0]
        shapes={'x':[6,i],'target':[4,o],'firstWeight':[h,i],'firstBias':[h],'secondWeight':[o,h],'secondBias':[o]}
        if (i,h,o) not in WIDTHS or any(vals[key][0]!=shape for key,shape in shapes.items()) or any(math.prod(vals[key][0])>=16 for key in KEYS):raise ValueError('Fixture exceeds the four reviewed width choices or small-array oracle range.')
        ids.add(fixture['id']);result.append((fixture['id'],(i,h,o),{key:([*shape],[_f32(v) for v in values]) for key,(shape,values) in vals.items()}))
    return result


def _model(widths,vals,registry=None):
    i,h,o=widths
    nodes=[{'id':'input','type':'input','config':{'features':i}}, {'id':'first','type':'linear','config':{'in_features':i,'out_features':h,'bias':True}}, {'id':'bend','type':'leaky_relu','config':{'negative_slope':.2}}, {'id':'second','type':'linear','config':{'in_features':h,'out_features':o,'bias':True}}, {'id':'output','type':'output','config':{}}]
    graph={'schemaVersion':2,'nodes':nodes,'edges':[{'id':'e'+str(index),'source':a['id'],'sourcePort':'out','target':b['id'],'targetPort':'in'} for index,(a,b) in enumerate(zip(nodes,nodes[1:]))]}
    recipe=objective_recipe({'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':{}})
    mapping={'schemaVersion':1,'componentID':'loss_mse','inputColumns':['x'+str(index) for index in range(i)],'roles':[{'role':'target','columns':['target'+str(index) for index in range(o)],'sampleShape':[o],'dtype':'float32'}],'tripletSourceGroups':[]}
    with torch.random.fork_rng(devices=[]):model=GraphModel(graph,registry=registry,bundled_dataset=False,objective_context={'recipe':recipe,'targetMapping':mapping}).cpu()
    with torch.no_grad():
        state=model.state_dict()
        for key,name in zip(KEYS,PARAMETERS):state[name].copy_(torch.tensor(vals[key][1],dtype=torch.float32).reshape(vals[key][0]))
    return model


def _owner_seed(seed,node,parameter):
    # Intentionally independent of initialization.SEED_VERSION / helper hashing.
    raw=json.dumps(['kwip.initialization.owner-seed.v1',seed,node,parameter],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**63)


def _oracle_array(shape,rule,seed):
    if math.prod(shape)>=16:raise ValueError('Independent lesson oracle requires fewer than16 values per learned array.')
    algorithm=rule['algorithm'];settings=rule['settings']
    generator=torch.Generator(device='cpu').manual_seed(_owner_seed(seed,rule['owner']['node'],rule['owner']['parameter']))
    if algorithm=='zeros':return [0.]*math.prod(shape)
    if algorithm=='normal':return (torch.randn(shape,dtype=torch.float64,generator=generator)*settings['std']+settings['mean']).float().flatten().tolist()
    fan_in=shape[1]*math.prod(shape[2:]);fan_out=shape[0]*math.prod(shape[2:])
    if algorithm=='xavier_uniform':bound=settings['gain']*math.sqrt(6/(fan_in+fan_out))
    else:
        gain=math.sqrt(2.) if settings['nonlinearity']=='relu' else math.sqrt(2/(1+settings['a']**2))
        bound=math.sqrt(3)*gain/math.sqrt(fan_in if settings['mode']=='fan_in' else fan_out)
    low,high=_f32(-bound),_f32(bound)
    unit=torch.rand(shape,dtype=torch.float32,generator=generator)
    return (unit.double()*(high-low)+low).float().flatten().tolist()


def _reference_parameters(task,vals):
    values={name:list(vals[key][1]) for key,name in zip(KEYS,PARAMETERS)}
    shapes={name:vals[key][0] for key,name in zip(KEYS,PARAMETERS)}
    expected=task_recipe(task)
    for rule in expected['config']['rules']:
        name='layers.'+rule['owner']['node']+'.'+rule['owner']['parameter']
        values[name]=_oracle_array(shapes[name],rule,expected['config']['seed'])
    return values


def _scalar(values,widths,vals,rate=.05):
    i,h,o=widths;w1,b1,w2,b2=[values[name] for name in PARAMETERS]
    rows=[vals['x'][1][n*i:(n+1)*i] for n in range(6)]
    z=[[sum(w1[a*i+c]*row[c] for c in range(i))+b1[a] for a in range(h)] for row in rows]
    active=[[value if value>0 else .2*value for value in row] for row in z]
    predictions=[[sum(w2[b*h+a]*active[n][a] for a in range(h))+b2[b] for b in range(o)] for n in range(6)]
    dy=[[2*(predictions[n][b]-vals['target'][1][n*o+b])/(4*o) for b in range(o)] for n in range(4)]
    dz=[[sum(dy[n][b]*w2[b*h+a] for b in range(o))*(1 if z[n][a]>0 else .2) for a in range(h)] for n in range(4)]
    gradients={PARAMETERS[0]:[sum(dz[n][a]*rows[n][c] for n in range(4)) for a in range(h) for c in range(i)],PARAMETERS[1]:[sum(dz[n][a] for n in range(4)) for a in range(h)],PARAMETERS[2]:[sum(dy[n][b]*active[n][a] for n in range(4)) for b in range(o) for a in range(h)],PARAMETERS[3]:[sum(dy[n][b] for n in range(4)) for b in range(o)]}
    updated={name:[value-rate*gradient for value,gradient in zip(values[name],gradients[name])] for name in PARAMETERS}
    return [value for row in predictions for value in row],gradients,updated


def _flat(state):return [float(value) for name in PARAMETERS for value in state[name].detach().reshape(-1).tolist()]
def _flat_reference(state):return [value for name in PARAMETERS for value in state[name]]
def _difference(a,b):return [x-y for x,y in zip(a,b)]


def execute(submission,fixtures,evidence,stopped,tensor,*,task,registry=None):
    def check():
        if stopped():raise ValueError('Starting-settings lesson cancelled or exceeded its budget.')
    check();recipe,optimizer_config=_submission(submission,task);validated=_fixtures(fixtures,tensor)
    for ident,widths,vals in validated:
        check();model=_model(widths,vals,registry=registry);base=copy.deepcopy(model);rng=torch.get_rng_state().clone()
        initialization.apply_initialization(model,recipe,check_cancelled=check)
        starting=_flat(model.state_dict())
        repeat=copy.deepcopy(base);initialization.apply_initialization(repeat,recipe,check_cancelled=check)
        reordered=copy.deepcopy(recipe);reordered['config']['rules'].reverse();reverse=copy.deepcopy(base);initialization.apply_initialization(reverse,reordered,check_cancelled=check)
        rng_changes=sum(a!=b for a,b in zip(rng.tolist(),torch.get_rng_state().tolist()))
        repeat_values=_difference(_flat(repeat.state_dict()),starting)+_difference(_flat(reverse.state_dict()),starting)+[float(rng_changes)]
        expected=_reference_parameters(task,vals);prediction_ref,gradient_ref,updated_ref=_scalar(expected,widths,vals,optimizer_config['lr'])
        x=torch.tensor(vals['x'][1],dtype=torch.float32).reshape(vals['x'][0]);target=torch.tensor(vals['target'][1],dtype=torch.float32).reshape(vals['target'][0])
        predictions=model(x);loss=torch.nn.functional.mse_loss(predictions[:4],target,reduction='mean')
        opt=torch.optim.SGD(model.parameters(),**optimizer_config,foreach=False,fused=False,differentiable=False)
        opt.zero_grad(set_to_none=True);loss.backward();gradients=[float(value) for name,parameter in model.named_parameters() for value in parameter.grad.flatten().tolist()]
        if tuple(name for name,_ in model.named_parameters())!=PARAMETERS:raise ValueError('Supplied model parameter order differs from the checked owners.')
        opt.step();updated=_flat(model.state_dict());check()
        saved=io.BytesIO();torch.save(model.state_dict(),saved);original=saved.getvalue()
        if len(original)>100000000:raise ValueError('Saved learned values exceed their checkpoint limit.')
        restored=copy.deepcopy(base);saved.seek(0);restored.load_state_dict(torch.load(saved,map_location='cpu',weights_only=True),strict=True);restored.eval()
        reuse_prediction=restored(x).detach().flatten().tolist();post_ref=_scalar(updated_ref,widths,vals,optimizer_config['lr'])[0]
        byte_changes=sum(a!=b for a,b in zip(original,saved.getvalue()))+abs(len(original)-len(saved.getvalue()))
        p=len(starting);o=widths[2]
        tables={'startingSettings':{'shape':[p],'values':_difference(starting,_flat_reference(expected))},'predictions':{'shape':[6,o],'values':_difference(predictions.detach().flatten().tolist(),prediction_ref)},'firstAdjustment':{'shape':[2,p],'values':_difference(gradients,_flat_reference(gradient_ref))+_difference(updated,_flat_reference(updated_ref))},'repeatIsolation':{'shape':[2*p+1],'values':repeat_values},'savedReuse':{'shape':[p+6*o+1],'values':_difference(_flat(restored.state_dict()),updated)+_difference(reuse_prediction,post_ref)+[float(byte_changes)]}}
        if any(not math.isfinite(value) for table in tables.values() for value in table['values']):raise ValueError('Starting choices produced nonfinite predictions or adjustments; reduce their spread or center.')
        evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),max(len(table['values']) for table in tables.values()))
        evidence['fixtures'].append({'fixtureID':ident,'tensors':tables,'partitions':{}})
