import numpy as np,pytest
from runner import speech_tables as st,media_artifacts as media,dataflow as df

class Recognizer:
 info={'identity':'a'*64}
 def transcribe(self,a,rate,c):
  assert a.shape==(160,) and rate==16000
  return [dict(start=0,end=.01,text='A private note.',segments=[(0,.01,'A private note.')],timestampCoverage='modelSegments')]

def table():
 raw,ref=media.encode_asset(np.zeros((1,320),dtype='<f4').tobytes(),kind='tensor',shape=[1,320],axes=['channels','samples'],dtype='float32',metadata=dict(sampleRate=16000,sourceRate=48000,sourceStart=48000,sourceEnd=48480,validSamples=160))
 t=df.Table(columns=[dict(name='prepared',dtype='tensor',nullable=False)],rows=[[ref]],row_ids=['source-1'],source_digest='b'*64)
 return raw,ref,t

def test_transcripts_segments_complete_source_offsets_padding_and_schema():
 raw,ref,t=table()
 with media.artifact_scope(lambda *_:None,lambda *_:raw):
  out=st.apply(t,dict(st.DEFAULTS),lambda:None,'speech',Recognizer())
 assert list(out['transcripts'].rows[0][1:]) == ['A private note.','transcribed','a'*64,1]
 assert list(out['segments'].rows[0][:5]) == ['source-1',0,1.0,1.01,'A private note.']
 assert out['segments'].rows[0][5]==ref
 assert out['segments'].columns[5]['dtype']=='tensor'


def test_explicit_limits_cancel_and_no_resource_admission():
 raw,ref,t=table()
 with media.artifact_scope(lambda *_:None,lambda *_:raw):
  c=dict(st.DEFAULTS,maximumCharacters=3)
  with pytest.raises(ValueError,match='character allowance'):st.apply(t,c,lambda:None,'speech',Recognizer())
  with pytest.raises(ValueError,match='Connect and verify'):st.apply(t,dict(st.DEFAULTS),lambda:None,'missing')
  def cancel():raise InterruptedError()
  with pytest.raises(InterruptedError):st.apply(t,dict(st.DEFAULTS),cancel,'speech',Recognizer())
 for key,value in [('language','xx'),('beamWidth',0),('chunkSeconds',31),('maximumTokens',True)]:
  with pytest.raises(ValueError):st.validate(dict(st.DEFAULTS,**{key:value}))


def test_truncation_is_visible_and_generic_spectra_are_refused():
 raw,ref,t=table()
 class Limited(Recognizer):
  def transcribe(self,*args):
   rows=super().transcribe(*args);rows[0]['truncated']=True;return rows
 with media.artifact_scope(lambda *_:None,lambda *_:raw):
  assert st.apply(t,dict(st.DEFAULTS),lambda:None,'speech',Limited())['transcripts'].rows[0][2]=='tokenLimit'
 from runner.preparation_operation_contract import file_limit
 assert file_limit('input/speech-models/'+'a'*64+'/preprocessor_config.json')==8000000
 for name in ['input/speech-models/'+'a'*64+'/model.py','input/speech-models/../config.json','input/speech-models/'+'a'*64+'/model.bin']:
  with pytest.raises(ValueError):file_limit(name)
