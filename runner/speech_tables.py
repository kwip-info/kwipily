"""Complete source-linked transcription tables; models are captured resources, never paths in graphs."""
from contextvars import ContextVar
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
if __package__:
 from . import dataflow as df,media_artifacts as media
 from .speech_models import Whisper,require
else:
 import dataflow as df,media_artifacts as media
 from speech_models import Whisper,require
DEFAULTS=dict(inputColumn='prepared',outputColumn='transcript',language='auto',chunkSeconds=30,maximumTokens=256,beamWidth=1,maximumCharacters=1000000,maximumSegments=10000)

def validate(c):
 require(type(c) is dict and set(c)==set(DEFAULTS),'Transcribe audio requires its complete configuration.')
 require(all(type(c[k]) is str and 0<len(c[k])<=200 for k in ('inputColumn','outputColumn')) and c['language'] in ('auto','en','es','fr','de','it','pt','ja','ko','zh'),'Choose audio/text columns and a supported speech language.')
 require(all(type(c[k]) is int and lo<=c[k]<=hi for k,lo,hi in [('chunkSeconds',5,30),('maximumTokens',16,440),('beamWidth',1,5),('maximumCharacters',1,1000000),('maximumSegments',1,10000)]),'Transcription processing limits exceed supported bounds.')
 return c
_context=ContextVar('speech_job',default=None)
@contextmanager
def job_scope(root,check):
 root=Path(root);manifest=root/'speech-models.json';models=json.loads(manifest.read_bytes()) if manifest.exists() else {}
 token=_context.set(dict(root=root,models=models,adapters={},check=check))
 try:yield
 finally:_context.reset(token)

def adapter(node):
 ctx=_context.get();require(ctx is not None and node in ctx['models'],'Connect and verify a Whisper Load model resource before transcription.')
 identity=ctx['models'][node]['identity']
 if identity not in ctx['adapters']:
  value=Whisper(ctx['root']/'speech-models'/identity,ctx['check']);require(value.info['identity']==identity,'Captured speech model identity changed.');ctx['adapters'][identity]=value
 return ctx['adapters'][identity]

def apply(table,c,check,node,recognizer=None):
 c=validate(c);index=table.index(c['inputColumn']);kind=table.columns[index]['dtype'];require(kind in ('audio','tensor'),'Choose a captured audio or waveform tensor column.')
 extra=[c['outputColumn'],'speech_status','speech_model','speech_chunks'];require(not set(extra)&{x['name'] for x in table.columns},'Transcription output names collide with source columns.')
 engine=recognizer or adapter(node);transcripts=[];segments=[];characters=0
 for row,identity in df.row_pairs(table):
  check();ref=row[index];require(ref is not None,'Missing audio: filter failed/missing source rows explicitly.')
  require(ref['axes'] in (['samples','channels'],['channels','samples']),'Transcription requires waveform samples; generic spectral tensors are not a Whisper processor input.')
  _,raw=media.read_asset(ref,check);array=np.frombuffer(raw,dtype=np.dtype(ref['dtype']).newbyteorder('<')).reshape(ref['shape'])
  if ref['axes']==['samples','channels']:array=array.T
  require(array.shape[0]==1 and ref['dtype']=='float32' and ref['metadata']['sampleRate']==16000,'Use Prepare media to explicitly normalize PCM, mix/select mono and resample to 16000 Hz.')
  count=ref['metadata'].get('validSamples',array.shape[1]);audio=array[0,:count]
  chunks=engine.transcribe(audio,16000,c);text=' '.join(x['text'] for x in chunks if x['text']);characters+=len(text)
  require(characters<=c['maximumCharacters'],'Complete transcription exceeds the character allowance; shorten the input or raise its explicit limit.')
  model=engine.info['identity'];offset=ref['metadata'].get('sourceStart',0)/ref['metadata'].get('sourceRate',16000)
  transcripts.append(([*row,text,'tokenLimit' if any(x.get('truncated',False) for x in chunks) else 'transcribed' if text else 'noSpeech',model,len(chunks)],identity))
  for chunk in chunks:
   for start,end,value in chunk['segments']:
    check();require(0<=start<end<=len(audio)/16000+0.001,'Speech timestamps exceed actual unpadded audio.')
    require(len(segments)<c['maximumSegments'],'Complete transcript segments exceed the explicit allowance.')
    segments.append(([identity,len(segments),offset+start,offset+end,value,ref,model,chunk['timestampCoverage']],df.digest([identity,start,end,value])))
 schema=list(table.columns)+[dict(name=n,dtype=t,nullable=False) for n,t in zip(extra,['string','string','string','int64'])]
 segment_schema=[dict(name=n,dtype=t,nullable=False) for n,t in [('source_observation','string'),('segment','int64'),('start_seconds','float64'),('end_seconds','float64'),('text','string'),('source_audio',kind),('speech_model','string'),('timestamp_coverage','string')]]
 return dict(transcripts=df.streaming_rows(table,iter(transcripts),schema,check),segments=df.streaming_rows(table,iter(segments),segment_schema,check,identity_prefix=''))
