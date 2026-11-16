"""Strict allocation-free recurrent controls, named state geometry and ownership."""
import math
if __package__:from .tensor_layout import BatchedLayout
else:from tensor_layout import BatchedLayout
OPERATIONS=frozenset({'rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell'})
MAXIMUM=100_000_000

def normalize_recurrent(operation,config):
 if operation not in OPERATIONS or type(config) is not dict:raise ValueError('Choose a supported recurrent component and declared controls.')
 c={'input_size':2,'hidden_size':2,'bias':True,'state_mode':'zero'}
 if not operation.endswith('_cell'):c.update(num_layers=1,bidirectional=False,dropout=0.)
 if operation in ('rnn','rnn_cell'):c['nonlinearity']='tanh'
 if operation=='lstm':c['proj_size']=0
 if set(config)-set(c):raise ValueError('Recurrent controls contain undeclared fields; state and weights belong to their explicit connections.')
 c.update(config)
 bounds={'input_size':(1,4096),'hidden_size':(1,512),'num_layers':(1,8),'proj_size':(0,511)}
 for key,(low,high) in bounds.items():
  if key in c:
   value=c[key]
   if type(value) not in (int,float) or not math.isfinite(value) or value!=int(value) or not low<=value<=high:raise ValueError(f'{key} requires an exact integer from {low} to {high}.')
   c[key]=int(value)
 for key in ('bias','bidirectional'):
  if key in c and type(c[key]) is not bool:raise ValueError(f'{key} requires a Boolean switch.')
 if type(c['state_mode']) is not str or c['state_mode'] not in ('zero','supplied'):raise ValueError('Choose zero-reset or explicitly supplied state.')
 if 'nonlinearity' in c and (type(c['nonlinearity']) is not str or c['nonlinearity'] not in ('tanh','relu')):raise ValueError('Choose tanh or relu for this recurrent component.')
 if 'dropout' in c:
  v=c['dropout']
  if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('Recurrent dropout must be a finite probability from zero to one.')
  c['dropout']=float(v)
  if v and c['num_layers']==1:raise ValueError('Inter-layer dropout needs two or more layers; choose two layers or dropout zero.')
 if c.get('proj_size',0)>=c['hidden_size']:raise ValueError('LSTM projection width must be below hidden width; zero disables projection.')
 return c

def configured_input_ports(operation,config):
 c=normalize_recurrent(operation,config)
 return ('in',) if c['state_mode']=='zero' else ('in','hidden','cell') if operation.startswith('lstm') else ('in','hidden')

def parameter_shapes(operation,config):
 c=normalize_recurrent(operation,config);h=c['hidden_size'];p=c.get('proj_size',0) or h;g=4 if operation.startswith('lstm') else 3 if operation.startswith('gru') else 1;cell=operation.endswith('_cell');d=2 if c.get('bidirectional',False) else 1
 rows=[]
 for layer in range(c.get('num_layers',1)):
  width=c['input_size'] if layer==0 else d*p
  for direction in range(d):
   suffix='' if cell else '_l'+str(layer)+('_reverse' if direction else '')
   for name,shape in [('weight_ih',(g*h,width)),('weight_hh',(g*h,p))]+([('bias_ih',(g*h,)),('bias_hh',(g*h,))] if c['bias'] else [])+([('weight_hr',(p,h))] if c.get('proj_size',0) else []):rows.append({'name':name+suffix,'shape':shape})
 return rows

def resolve_recurrent(operation,config,layouts):
 c=normalize_recurrent(operation,config);ports=configured_input_ports(operation,c);cell=operation.endswith('_cell')
 if type(layouts) not in (list,tuple) or len(layouts)!=len(ports) or any(not isinstance(x,BatchedLayout) for x in layouts):raise ValueError('Connect exactly the declared recurrent input and supplied-state ports.')
 sample=layouts[0].sample;h=c['hidden_size'];p=c.get('proj_size',0) or h;l=c.get('num_layers',1);d=2 if c.get('bidirectional',False) else 1;g=4 if operation.startswith('lstm') else 3 if operation.startswith('gru') else 1
 if len(sample)!=(1 if cell else 2) or sample[-1]!=c['input_size'] or (not cell and not 1<=sample[0]<=4096):raise ValueError('Recurrent input requires its exact feature width and one to 4096 sequence positions.')
 t=1 if cell else sample[0];hidden=(h,) if cell else (l*d,p);cell_shape=(h,) if cell else (l*d,h)
 if len(layouts)>1 and layouts[1].sample!=hidden:raise ValueError('Supplied hidden state must match observations, layers/directions and hidden/projection width.')
 if len(layouts)>2 and layouts[2].sample!=cell_shape:raise ValueError('Supplied cell state must match observations, layers/directions and cell width.')
 outputs={'out':BatchedLayout((h,) if cell else (t,d*p)),'hidden':BatchedLayout(hidden)}
 if operation.startswith('lstm'):outputs['cell']=BatchedLayout(cell_shape)
 parameters=sum(math.prod(x['shape']) for x in parameter_shapes(operation,c))
 physical=sum(math.prod(v.sample) for key,v in outputs.items() if not(cell and key=='hidden'))
 gate=t*l*d*g*h
 work=t*sum(g*h*((c['input_size'] if layer==0 else d*p)+p)+(p*h if c.get('proj_size',0) else 0) for layer in range(l) for _ in range(d))
 if parameters>2_000_000 or any(v>MAXIMUM for v in [physical,gate,work,*[math.prod(x.sample) for x in layouts]]):raise ValueError('Recurrent parameters, tensor outputs, gate storage or recurrence work exceed the declared allocation budget.')
 return {'config':c,'outputs':outputs,'parameterCount':parameters,'outputElementsPerObservation':physical,'gateElementsPerObservation':gate,'workPerObservation':work}

def capability(operation):
 if operation not in OPERATIONS:raise ValueError('Unknown recurrent component.')
 cell=operation.endswith('_cell');lstm=operation.startswith('lstm')
 return {'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.'+operation,'graphSchemaVersions':[2],'preservesBatch':True,'graphDTypes':['float32'],'inputSampleRank':1 if cell else 2,'stateSampleRank':1 if cell else 2,'inputMinimum':1,'inputMaximum':4096,'hiddenMinimum':1,'hiddenMaximum':512,'layerMinimum':1,'layerMaximum':1 if cell else 8,'timeMinimum':1,'timeMaximum':1 if cell else 4096,'maximumParameters':2_000_000,'maximumTensorElements':MAXIMUM,'maximumGateElements':MAXIMUM,'maximumWorkElements':MAXIMUM,'stateModes':['zero','supplied'],'inputPorts':['in','hidden','cell'] if lstm else ['in','hidden'],'outputPorts':['out','hidden','cell'] if lstm else ['out','hidden'],'stateLayout':'observationFirst','nativeStateLayout':'observationFirst' if cell else 'layerDirectionObservationWidth','statePersistence':'explicitConnectionsOnly','cellHiddenAlias':cell,'dropoutPolicy':'none' if cell else 'betweenLayersTrainingOnly','parameterOrder':'nativeLayerDirectionGateOrder','projectionPolicy':'cpuOnly' if operation=='lstm' else 'notApplicable','projectionMaximum':511 if operation=='lstm' else 0}
