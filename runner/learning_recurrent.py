"""Measured recurrent lessons. Trusted fixtures, independent gold supplied by Core."""
import math
import torch
if __package__:
 from .recurrent_configuration import normalize_recurrent,parameter_shapes,configured_input_ports
else:
 from recurrent_configuration import normalize_recurrent,parameter_shapes,configured_input_ports
TASKS={'rnn-sequence-memory':'rnn','lstm-separate-memory':'lstm','gru-reset-memory':'gru','rnn-cell-step':'rnn_cell','lstm-cell-step':'lstm_cell','gru-cell-step':'gru_cell'}
def target(op):
 c=dict(input_size=2,hidden_size=2,bias=True,state_mode='supplied')
 if not op.endswith('_cell'):c.update(num_layers=1,bidirectional=False,dropout=0.)
 if op.startswith('rnn'):c['nonlinearity']='relu'
 if op=='lstm':c['proj_size']=0
 return c
def roles(op):return ['input','hidden','cell'] if op.startswith('lstm') else ['input','hidden']
def outputs(op):return ['result','hidden','cell'] if op.startswith('lstm') else ['result','hidden']
def configuration(submission,task):
 if task not in TASKS or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='recurrentGraph':raise ValueError('Use the recurrent graph; weights and answers belong to the checker.')
 op=TASKS[task];g=submission['graph']
 if type(g) is not dict or set(g)!={'schemaVersion','name','nodes','edges'} or type(g['schemaVersion']) is not int or g['schemaVersion']!=2 or type(g['name']) is not str or len(g['name'])>256 or type(g['nodes']) is not list or len(g['nodes'])!=len(roles(op))+len(outputs(op))+1 or type(g['edges']) is not list:raise ValueError('Connect the supplied input and state to one recurrent block and each named result.')
 nodes={};sources={};sinks={};operator=None
 for node in g['nodes']:
  if type(node) is not dict or set(node)!={'id','type','name','x','y','config'} or type(node['id']) is not str or not 1<=len(node['id'])<=128 or node['id'] in nodes or type(node['name']) is not str or len(node['name'])>256 or any(type(node[k]) not in (int,float) or not math.isfinite(node[k]) for k in ('x','y')):raise ValueError('Malformed recurrent lesson block.')
  nodes[node['id']]=node;c=node['config']
  if node['type']=='input':
   if type(c) is not dict or set(c)!={'fixture'} or c['fixture'] not in roles(op) or c['fixture'] in sources:raise ValueError('Use one Input for each supplied role.')
   sources[c['fixture']]=node['id']
  elif node['type']=='output':
   if type(c) is not dict or set(c)!={'role'} or c['role'] not in outputs(op) or c['role'] in sinks:raise ValueError('Use a separate Output for every result/state role.')
   sinks[c['role']]=node['id']
  elif node['type']==op and operator is None:operator=node
  else:raise ValueError('Use only the requested recurrent block and role boundaries.')
 if operator is None or set(sources)!=set(roles(op)) or set(sinks)!=set(outputs(op)):raise ValueError('Complete every source and result role.')
 c=normalize_recurrent(op,operator['config'])
 if c['state_mode']!='supplied' or c['input_size']!=2 or c['hidden_size']!=2 or c.get('num_layers',1)!=1 or c.get('bidirectional',False) or c.get('dropout',0)!=0 or c.get('proj_size',0)!=0:raise ValueError('This exercise uses supplied state, width2 and one forward layer with no dropout or projection.')
 expected={(ident,'out',operator['id'],'in' if role=='input' else role) for role,ident in sources.items()}|{(operator['id'],'out' if role=='result' else role,ident,'in') for role,ident in sinks.items()};actual=set();ids=set()
 for edge in g['edges']:
  if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'} or type(edge['id']) is not str or not 1<=len(edge['id'])<=128 or edge['id'] in ids:raise ValueError('Malformed recurrent role connection.')
  ids.add(edge['id']);actual.add((edge['source'],edge['sourcePort'],edge['target'],edge['targetPort']))
 if actual!=expected or len(g['edges'])!=len(expected):raise ValueError('Connect each fixture role to its matching input and each named result to its matching Output.')
 return op,c

def execute(submission,fixtures,evidence,stopped,tensor,*,task):
 if stopped():raise ValueError('Recurrent lesson cancelled.')
 op,c=configuration(submission,task);cell=op.endswith('_cell');ps=parameter_shapes(op,target(op));shapes={p['name']:list(p['shape']) for p in ps};required=set(roles(op))|{'parameter.'+k for k in shapes}|{'upstream.'+r for r in outputs(op)}
 if type(fixtures) is not list or not 1<=len(fixtures)<=32:raise ValueError('Use one to32 bounded supplied fixtures.')
 checked=[];ids=set()
 for f in fixtures:
  if type(f) is not dict or set(f)!={'id','inputs'} or type(f['id']) is not str or not f['id'] or f['id'] in ids or type(f['inputs']) is not dict or set(f['inputs'])!=required:raise ValueError('Fixture roles, parameters and output influence must be checker-owned.')
  ids.add(f['id']);values={};n=None
  for key,raw in f['inputs'].items():
   shape,flat=tensor(raw)
   if not isinstance(shape,(list,tuple)) or any(type(i) is not int or i<=0 for i in shape) or math.prod(shape)>256 or not isinstance(flat,(list,tuple)) or len(flat)!=math.prod(shape) or any(type(v) not in (int,float) or not math.isfinite(v) for v in flat):raise ValueError('Use bounded finite supplied arrays.')
   if key=='input':n=shape[0]
   values[key]=(list(shape),list(flat))
  if n not in (1,2,3):raise ValueError('Use one to three independent observations.')
  layout={r:([n,2] if cell else [n,3,2] if r=='input' else [n,1,2]) for r in roles(op)}
  outs={'result':layout['input'],'hidden':layout['hidden'],**({'cell':layout['cell']} if 'cell' in layout else {})}
  expected={**layout,**{'parameter.'+k:v for k,v in shapes.items()},**{'upstream.'+k:v for k,v in outs.items()}}
  if any(values[k][0]!=shape for k,shape in expected.items()):raise ValueError('A supplied input, state, weight or result-influence shape differs from this exercise.')
  checked.append((f['id'],values))
 for ident,values in checked:
  if stopped():raise ValueError('Recurrent lesson cancelled.')
  args=dict(input_size=2,hidden_size=2,bias=c['bias'])
  if op.startswith('rnn'):args['nonlinearity']=c['nonlinearity']
  if not cell:args.update(num_layers=1,batch_first=True,bidirectional=False,dropout=0.)
  module=getattr(torch.nn,{'rnn':'RNN','lstm':'LSTM','gru':'GRU','rnn_cell':'RNNCell','lstm_cell':'LSTMCell','gru_cell':'GRUCell'}[op])(**args).to(dtype=torch.float64).eval()
  active=dict(module.named_parameters())
  with torch.no_grad():
   for name,p in active.items():p.copy_(torch.tensor(values['parameter.'+name][1],dtype=torch.float64).reshape(values['parameter.'+name][0]))
  x={r:torch.tensor(values[r][1],dtype=torch.float64).reshape(values[r][0]).requires_grad_(True) for r in roles(op)}
  state=x['hidden'] if cell else x['hidden'].transpose(0,1).contiguous()
  if op.startswith('lstm'):state=(state,x['cell'] if cell else x['cell'].transpose(0,1).contiguous())
  result=module(x['input'],state)
  if cell:
   out=result[0] if op.startswith('lstm') else result;measured={'result':out,'hidden':out,**({'cell':result[1]} if op.startswith('lstm') else {})}
  else:
   out,state=result;h,cs=state if op.startswith('lstm') else (state,None);measured={'result':out,'hidden':h.transpose(0,1),**({'cell':cs.transpose(0,1)} if cs is not None else {})}
  loss=sum((value*torch.tensor(values['upstream.'+r][1],dtype=torch.float64).reshape(values['upstream.'+r][0])).sum() for r,value in measured.items());loss.backward()
  for r,value in x.items():measured[r+'Gradient']=value.grad
  for name,shape in shapes.items():measured['parameter.'+name+'Gradient']=active[name].grad if name in active else torch.zeros(shape,dtype=torch.float64)
  if any(v is None or not torch.isfinite(v).all().item() for v in measured.values()):raise ValueError('The recurrent measurement produced unavailable or nonfinite influence.')
  tables={k:dict(shape=list(v.shape),values=v.detach().reshape(-1).tolist()) for k,v in measured.items()};evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),max(v.numel() for v in measured.values()));evidence['fixtures'].append(dict(fixtureID=ident,tensors=tables,partitions={}))
