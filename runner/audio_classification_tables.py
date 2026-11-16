"""Complete, source-linked window predictions and class-score tables."""
from contextvars import ContextVar
from contextlib import contextmanager
import json,math
from pathlib import Path
import numpy as np
if __package__:
 from . import dataflow as df,media_artifacts as media
 from .audio_classification_models import AST,require
else:
 import dataflow as df,media_artifacts as media
 from audio_classification_models import AST,require
DEFAULTS=dict(inputColumn='prepared',outputPrefix='audio_classification',windowSeconds=10,hopSeconds=8,finalWindow='pad',minimumScore=.5,reportedLabels=5,batchWindows=1,maximumWindows=256,maximumScoreRows=150000)
def validate(c):
 require(type(c) is dict and set(c)==set(DEFAULTS),'Classify audio requires its complete configuration.')
 require(all(type(c[k]) is str and 0<len(c[k].strip())<=160 for k in ('inputColumn','outputPrefix')) and c['finalWindow'] in ('pad','refuse'),'Choose waveform, output prefix and final-window policy.')
 require(all(type(c[k]) in (int,float) and math.isfinite(c[k]) and lo<=c[k]<=hi for k,lo,hi in [('windowSeconds',.025,10),('hopSeconds',.025,10),('minimumScore',0,1)]) and c['hopSeconds']<=c['windowSeconds'],'Audio window, hop or score bounds are invalid.')
 require(all(type(c[k]) is int and lo<=c[k]<=hi for k,lo,hi in [('reportedLabels',1,1024),('batchWindows',1,8),('maximumWindows',1,4096),('maximumScoreRows',1,150000)]),'Classification output/compute allowance exceeded.')
 return c
_context=ContextVar('audio_classification_job',default=None)
@contextmanager
def job_scope(root,check):
 root=Path(root);path=root/'audio-models.json';pins=json.loads(path.read_bytes()) if path.exists() else {}
 token=_context.set(dict(root=root,models=pins,adapters={},check=check))
 try:yield
 finally:_context.reset(token)
def adapter(node):
 ctx=_context.get();require(ctx is not None and node in ctx['models'],'Connect and verify an AST Load model before classification.')
 identity=ctx['models'][node]['identity']
 if identity not in ctx['adapters']:
  engine=AST(ctx['root']/'audio-models'/identity,ctx['check']);require(engine.info['identity']==identity,'Captured AST identity changed.');ctx['adapters'][identity]=engine
 return ctx['adapters'][identity]
def windows(length,size,hop):
 require(length>0 and size>0 and 0<hop<=size,'Invalid audio windows.')
 starts=[0]
 while starts[-1]+size<length:starts.append(starts[-1]+hop)
 return [(start,min(start+size,length)) for start in starts]
def apply(table,c,check,node,classifier=None):
 c=validate(c);index=table.index(c['inputColumn']);kind=table.columns[index]['dtype'];require(kind in ('audio','tensor'),'Choose a captured waveform column.')
 engine=classifier or adapter(node);labels=engine.info['labels'];require(1<=len(labels)<=1024,'Class vocabulary exceeds admission.')
 prefix=c['outputPrefix'];extra=[prefix+'_'+n for n in ('window','start_seconds','end_seconds','label','score','reported','status','model')]
 require(not set(extra)&{x['name'] for x in table.columns},'Classification output names collide with source columns.')
 predictions=[];scores=[];seen=0
 for row,identity in df.row_pairs(table):
  check();ref=row[index];require(ref is not None and ref['axes'] in (['samples','channels'],['channels','samples']),'Classification needs waveform samples, not generic spectra.')
  _,raw=media.read_asset(ref,check);array=np.frombuffer(raw,dtype=np.dtype(ref['dtype']).newbyteorder('<')).reshape(ref['shape'])
  if ref['axes']==['samples','channels']:array=array.T
  require(array.shape[0]==1 and ref['dtype']=='float32' and ref['metadata']['sampleRate']==16000,'Use Prepare media for mono float32 at16000 Hz.')
  count=ref['metadata'].get('validSamples',array.shape[1]);require(type(count) is int and 0<count<=array.shape[1],'Invalid valid audio length.')
  audio=array[0,:count];require(np.isfinite(audio).all(),'Waveform contains nonfinite samples.')
  spans=windows(count,round(c['windowSeconds']*16000),round(c['hopSeconds']*16000))
  require(seen+len(spans)<=c['maximumWindows'],'Complete classification exceeds the window allowance.')
  require((seen+len(spans))*len(labels)<=c['maximumScoreRows'],'Complete score row allowance exceeded.')
  require(c['finalWindow']=='pad' or spans[-1][1]-spans[-1][0]==round(c['windowSeconds']*16000),'Short final window is refused by policy.')
  offset=ref['metadata'].get('sourceStart',0)/ref['metadata'].get('sourceRate',16000)
  for batch in range(0,len(spans),c['batchWindows']):
   check();subset=spans[batch:batch+c['batchWindows']];logits=engine.predict([audio[a:b] for a,b in subset],16000)
   require(logits.shape==(len(subset),len(labels)) and np.isfinite(logits).all(),'Invalid class logits.')
   for bi,(start,end) in enumerate(subset):
    check();values=[1/(1+math.exp(-float(x))) if x>=0 else math.exp(float(x))/(1+math.exp(float(x))) for x in logits[bi]]
    order=sorted(range(len(labels)),key=lambda i:(-values[i],i));ranks={i:n+1 for n,i in enumerate(order)}
    reported={i for i in order if values[i]>=c['minimumScore']};reported=set([i for i in order if i in reported][:c['reportedLabels']])
    window=batch+bi;begin=offset+start/16000;finish=offset+end/16000;model=engine.info['identity'];wid=df.digest([identity,window,begin,finish,model])
    top=order[0];status='classified' if values[top]>=c['minimumScore'] else 'noLabelAboveThreshold'
    predictions.append(([*row,window,begin,finish,labels[top],values[top],len(reported),status,model],wid))
    for i,label in enumerate(labels):scores.append(([identity,window,begin,finish,label,float(logits[bi,i]),values[i],ranks[i],values[i]>=c['minimumScore'],i in reported,ref,model,'windowCoverage'],df.digest([wid,i])))
  seen+=len(spans)
 schema=list(table.columns)+[dict(name=n,dtype=t,nullable=False) for n,t in zip(extra,['int64','float64','float64','string','float64','int64','string','string'])]
 score_schema=[dict(name=n,dtype=t,nullable=False) for n,t in [('source_observation','string'),('window','int64'),('start_seconds','float64'),('end_seconds','float64'),('label','string'),('logit','float64'),('score','float64'),('rank','int64'),('meets_threshold','bool'),('reported','bool'),('source_audio',kind),('model_identity','string'),('timestamp_coverage','string')]]
 return dict(predictions=df.streaming_rows(table,iter(predictions),schema,check,identity_prefix=''),scores=df.streaming_rows(table,iter(scores),score_schema,check,identity_prefix=''))
