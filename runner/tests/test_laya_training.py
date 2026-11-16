import json
import pytest
import torch
from safetensors.torch import load_file
from runner.tests.test_laya_models import model_folder
from runner.tests.test_typed_decisions import QUESTIONS
from runner.laya_training import fine_tune,DEFAULTS,fit_calibration

def rows(prefix,count):return [dict(case_id=prefix+str(i),text='technical crash',team='technical') for i in range(count)]
def settings():
 q=dict(QUESTIONS,questions=[QUESTIONS['questions'][0]])
 return dict(DEFAULTS,questionSet=q,targetColumns={'team':'team'},epochs=8,batchSize=2,headLearningRate=.001,maximumResidentBytes=1000000000)
def test_real_head_training_freezes_encoder_and_reload_keeps_questions(model_folder,tmp_path):
 original=load_file(model_folder/'model.safetensors');events=[];c=settings();out=tmp_path/'trained'
 report=fine_tune(model_folder,out,rows('train-',8),rows('valid-',4),c,emit=events.append)
 after=load_file(out/'model/model.safetensors')
 assert all(torch.equal(v,after[k]) for k,v in original.items() if k.startswith(('encoder.','act_head.')))
 assert any(not torch.equal(v,after[k]) for k,v in original.items() if k.startswith('scorer.'))
 assert report['metrics'][-1]['trainingLoss']<report['metrics'][0]['trainingLoss']
 assert report['continuation']=='weightsOnly' and report['task']=='typedDecisions'
 assert any(e.get('event')=='trainingProgress' for e in events)
 package=json.loads((out/'model/kwip-decisions.json').read_text());assert package['questionSet']==c['questionSet']
 assert not (out/'model/training-data.json').exists()
 assert (out/'result-manifest.json').exists()
def test_full_training_updates_encoder_and_rejects_leakage(model_folder,tmp_path):
 c=settings();c.update(mode='full',epochs=1,encoderLearningRate=.001)
 before=load_file(model_folder/'model.safetensors')
 fine_tune(model_folder,tmp_path/'full',rows('a',4),rows('b',2),c)
 after=load_file(tmp_path/'full/model/model.safetensors')
 assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('encoder.'))
 with pytest.raises(ValueError,match='group'):fine_tune(model_folder,tmp_path/'overlap',rows('same',4),rows('same',2),c)
 c['device']='mps'
 with pytest.raises(ValueError,match='CPU'):fine_tune(model_folder,tmp_path/'gpu',rows('x',4),rows('y',2),c)
def test_calibration_is_bounded_and_missing_support_never_accepts():
 q=QUESTIONS['questions'][0]
 logits=[[2.,0.]]*20;targets=[[1.,0.]]*20
 got=fit_calibration(logits,targets,.1,groups=[str(i) for i in range(20)])
 assert .5<=got['temperature']<=5 and got['support']==20
 assert got['acceptEnabled'] is False
 with pytest.raises(ValueError,match='support'):fit_calibration(logits[:2],targets[:2],.1,groups=['a','b'])
