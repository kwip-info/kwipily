"""Allocation-free strict paired spatial extraction and overlap-add geometry."""
if __package__:
 from .tensor_layout import BatchedLayout
 from .convolution_geometry import resolve_axis
else:
 from tensor_layout import BatchedLayout
 from convolution_geometry import resolve_axis
OPERATIONS=frozenset({'unfold','fold'})
MAXIMUM=100_000_000

def normalize_spatial_patch(operation,config):
 if type(operation) is not str or operation not in OPERATIONS or type(config) is not dict:raise ValueError('Choose a reviewed spatial patch operator and declared paired controls.')
 defaults={'kernel_size':[2,2],'stride':[1,1],'dilation':[1,1],'padding':[0,0]}
 if operation=='fold':defaults['output_size']=[2,2]
 if set(config)-set(defaults):raise ValueError('Spatial patches accept only declared paired geometry controls.')
 result={}
 for key,value in {**defaults,**config}.items():
  lo=0 if key=='padding' else 1
  if type(value) is not list or len(value)!=2 or any(type(v) not in (int,float) or not lo<=v<=4096 or v!=int(v) for v in value):raise ValueError(f'{operation} {key} needs two exact integers from {lo} to 4096.')
  result[key]=[int(v) for v in value]
 return result

def _count(sample):
 count=1
 for extent in sample:
  if count>MAXIMUM//extent:raise ValueError('Spatial patches exceed the 100-million element or extraction work budget.')
  count*=extent
 return count

def resolve_spatial_patch(operation,config,layout):
 c=normalize_spatial_patch(operation,config)
 if not isinstance(layout,BatchedLayout) or len(layout.sample)!=(3 if operation=='unfold' else 2):raise ValueError('Unfold needs channel/height/width; Fold needs packed patch rows/window columns.')
 _count(layout.sample);area=c['kernel_size'][0]*c['kernel_size'][1]
 if operation=='unfold':channels=layout.sample[0];spatial=layout.sample[1:]
 else:
  if layout.sample[0]%area:raise ValueError('Fold packed rows must be divisible by patch area.')
  channels=layout.sample[0]//area;spatial=c['output_size']
 if not 1<=channels<=512:raise ValueError('Spatial patches require 1–512 channels.')
 windows=[resolve_axis(extent,k,s,d,p,'explicit','zeros',operation=operation)[0] for extent,k,s,d,p in zip(spatial,c['kernel_size'],c['stride'],c['dilation'],c['padding'])]
 columns=windows[0]*windows[1]
 if operation=='unfold':sample=(channels*area,columns);work=_count(sample)
 else:
  if layout.sample[1]!=columns:raise ValueError('Fold window columns differ from the configured output grid geometry.')
  sample=(channels,*spatial);work=_count(layout.sample)
 _count(sample)
 return {'config':c,'layout':BatchedLayout(tuple(sample)),'workPerObservation':work}

def capability(operation):
 if operation not in OPERATIONS:raise ValueError('Unknown spatial patch operator.')
 d={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.'+operation,'graphSchemaVersions':[2],'preservesBatch':True,'graphDTypes':['float32'],'parameterless':True,'inputSampleRank':3 if operation=='unfold' else 2,'outputSampleRank':2 if operation=='unfold' else 3,'channelMinimum':1,'channelMaximum':512,'spatialMinimum':1,'spatialMaximum':4096,'kernelMinimum':1,'kernelMaximum':4096,'strideMinimum':1,'strideMaximum':4096,'dilationMinimum':1,'dilationMaximum':4096,'paddingMinimum':0,'paddingMaximum':4096,'paddingMode':'zeros','maximumTensorElements':MAXIMUM,'maximumWorkElements':MAXIMUM}
 if operation=='unfold':d.update(packedOrder='channelKernelHeightKernelWidth',windowOrder='heightWidth')
 else:d.update(overlapMode='add',uncoveredValue=0)
 return d
