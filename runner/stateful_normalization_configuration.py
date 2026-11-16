"""Torch-free strict channel/reduction/state geometry for remaining normalization."""
import math
if __package__:
 from .tensor_layout import BatchedLayout
else:
 from tensor_layout import BatchedLayout
OPERATIONS=frozenset({'batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm'})
EPS_MIN=1.1754943508222875e-38
EPS_MAX=3.4028234663852886e38
MAXIMUM=100_000_000
COUNTER_MAXIMUM=1_000_000_000

def normalize_stateful_normalization(operation,config):
 if type(operation) is not str or operation not in OPERATIONS or type(config) is not dict:raise ValueError('Choose a reviewed normalization with declared controls.')
 defaults={'eps':1e-5,'affine':not operation.startswith('instance'),'bias':True}
 if operation=='group_norm':defaults.update(num_channels=2,num_groups=1)
 else:
  defaults.update(num_features=2,momentum=.1,track_running_stats=operation.startswith('batch'))
  if operation.startswith('batch'):defaults['averaging']='moving'
 if set(config)-set(defaults):raise ValueError('Normalization accepts only its declared channel, statistics and affine controls.')
 c={**defaults,**config}
 for key in ('num_features','num_channels','num_groups'):
  if key in c:
   value=c[key]
   if type(value) not in (int,float) or not 1<=value<=512 or value!=int(value):raise ValueError('Normalization channels/groups require exact integers from1 to512.')
   c[key]=int(value)
 for key in ('affine','bias','track_running_stats'):
  if key in c and type(c[key]) is not bool:raise ValueError('Normalization switches require Boolean values.')
 for key,lo,hi in [('eps',EPS_MIN,EPS_MAX),('momentum',0.,1.)]:
  if key in c:
   value=c[key]
   if type(value) not in (int,float) or not lo<=value<=hi:raise ValueError('Normalization epsilon/update weight is outside its finite supported range.')
   c[key]=float(value)
 if 'averaging' in c and (type(c['averaging']) is not str or c['averaging'] not in ('moving','cumulative')):raise ValueError('Choose moving or cumulative running statistics.')
 if operation=='group_norm' and c['num_channels']%c['num_groups']:raise ValueError('Groups must divide channels exactly.')
 return c

def resolve_stateful_normalization(operation,config,layout):
 c=normalize_stateful_normalization(operation,config);ranks={'batch_norm1d':(1,2),'batch_norm2d':(3,),'instance_norm1d':(2,),'instance_norm2d':(3,),'group_norm':tuple(range(1,8))}
 if not isinstance(layout,BatchedLayout) or len(layout.sample) not in ranks[operation]:raise ValueError('Normalization input rank differs from its reviewed batched channel/spatial layout.')
 channels=c['num_channels'] if operation=='group_norm' else c['num_features']
 if layout.sample[0]!=channels or any(not 1<=v<=4096 for v in layout.sample[1:]):raise ValueError('Normalization requires matching channels and spatial extents from1 to4096.')
 spatial=math.prod(layout.sample[1:]);members=spatial*channels//c['num_groups'] if operation=='group_norm' else spatial
 if operation=='group_norm' and members<=1:raise ValueError('Each group needs more than one member per observation; reduce groups or add spatial positions for batch-one inference.')
 buffers={}
 if c.get('track_running_stats',False):buffers={'running_mean':{'dtype':'float32','shape':[channels]},'running_var':{'dtype':'float32','shape':[channels]},'num_batches_tracked':{'dtype':'int64','shape':[]}}
 return {'config':c,'layout':layout,'parameterCount':channels*(1+int(c['bias'])) if c['affine'] else 0,'buffers':buffers,'spatialElements':spatial,'groupMembers':members}

def capability(operation):
 if operation not in OPERATIONS:raise ValueError('Unknown normalization operator.')
 ranks={'batch_norm1d':[1,2],'batch_norm2d':[3],'instance_norm1d':[2],'instance_norm2d':[3],'group_norm':list(range(1,8))}
 d={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.'+operation,'graphSchemaVersions':[2],'inputSampleRanks':ranks[operation],'channelAxis':1,'channelMinimum':1,'channelMaximum':512,'spatialMinimum':1,'spatialMaximum':4096,'preservesBatch':True,'graphDTypes':['float32'],'epsilonMinimum':EPS_MIN,'epsilonMaximum':EPS_MAX,'affineModes':[False,True],'biasModes':[False,True],'forwardVarianceCorrection':0,'maximumTensorElements':MAXIMUM}
 if operation=='group_norm':d.update(groupMinimum=1,groupMaximum=512,requiresChannelDivisibility=True,minimumGroupMembers=2,statisticsScope='observationGroupSpatial',usesCurrentStatistics=True)
 else:
  d.update(momentumMinimum=0.,momentumMaximum=1.,trackingModes=[False,True],runningVarianceCorrection=1,runningMeanDType='float32',runningVarianceDType='float32',counterDType='int64',counterMaximum=COUNTER_MAXIMUM,statisticsScope='batchSpatial' if operation.startswith('batch') else 'observationChannelSpatial',evaluationUsesRunningStatisticsWhenTracked=True)
  if operation.startswith('batch'):d.update(averagingModes=['moving','cumulative'],counterBehavior='trainingCalls',minimumCurrentStatisticsMembers=2)
  else:d.update(counterBehavior='constantZero',minimumCurrentStatisticsSpatialElements=2)
 return d
