import math
import numpy as np
import pytest
from runner import media_artifacts as media
from runner.tests.test_speech_tables import table
from runner import audio_classification_tables as ac

class Classifier:
 info={'identity':'a'*64,'labels':['Speech','Music','Other']}
 def predict(self,waves,rate):
  assert rate==16000
  return np.array([[2.0,1.0,-2.0] for _ in waves],dtype=np.float32)

def test_complete_multilabel_scores_and_source_offsets():
 raw,ref,t=table()
 with media.artifact_scope(lambda *_:None,lambda *_:raw):
  out=ac.apply(t,dict(ac.DEFAULTS),lambda:None,'classify',Classifier())
 assert len(out['predictions'].rows)==1 and len(out['scores'].rows)==3
 score=out['scores'];assert [r[score.index('label')] for r in score.rows]==['Speech','Music','Other']
 assert score.rows[0][score.index('score')]==pytest.approx(1/(1+math.exp(-2)))
 assert sum(r[score.index('score')] for r in score.rows)>1
 assert score.rows[0][score.index('start_seconds')]==1.0
 assert sum(r[score.index('meets_threshold')] for r in score.rows)==2
 assert out['predictions'].rows[0][out['predictions'].index('audio_classification_reported')]==2

def test_threshold_never_truncates_scores_and_bound_admission():
 raw,ref,t=table()
 with media.artifact_scope(lambda *_:None,lambda *_:raw):
  out=ac.apply(t,dict(ac.DEFAULTS,minimumScore=1.0,reportedLabels=1),lambda:None,'classify',Classifier())
  assert len(out['scores'].rows)==3
  assert out['predictions'].rows[0][out['predictions'].index('audio_classification_status')]=='noLabelAboveThreshold'
  with pytest.raises(ValueError,match='score row'):ac.apply(t,dict(ac.DEFAULTS,maximumScoreRows=2),lambda:None,'classify',Classifier())
  with pytest.raises(ValueError,match='final window'):ac.apply(t,dict(ac.DEFAULTS,finalWindow='refuse'),lambda:None,'classify',Classifier())
  def cancel():raise InterruptedError()
  with pytest.raises(InterruptedError):ac.apply(t,dict(ac.DEFAULTS),cancel,'classify',Classifier())
 for key,value in [('windowSeconds',0),('hopSeconds',11),('minimumScore',float('nan')),('batchWindows',True)]:
  with pytest.raises(ValueError):ac.validate(dict(ac.DEFAULTS,**{key:value}))

def test_windows_cover_final_samples_without_duplicate_empty_tail():
 assert ac.windows(160000,160000,128000)==[(0,160000)]
 assert ac.windows(288000,160000,128000)==[(0,160000),(128000,288000)]
 assert ac.windows(288001,160000,128000)[-1]==(256000,288001)
