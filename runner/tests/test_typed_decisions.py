"""Task-specific, complete decision contracts; never next-token generation."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from runner.typed_decisions import question_set, target_distribution, split_groups, decode

QUESTIONS={'schemaVersion':1,'id':'ticket-triage','revision':1,'name':'Ticket triage','questions':[
 {'id':'team','type':'choice','instructions':'Which team handles this ticket?','options':[{'id':'billing','label':'Billing'},{'id':'technical','label':'Technical'}]},
 {'id':'urgency','type':'score','instructions':'How urgent is it?','options':[{'id':'low','label':'Low'},{'id':'medium','label':'Medium'},{'id':'high','label':'High'}]},
 {'id':'review','type':'noul','instructions':'Does this ticket need review?','options':[{'id':'false','label':'No'},{'id':'true','label':'Yes'}]}]}

def test_immutable_question_revision_and_stable_option_order():
 got=question_set(QUESTIONS);assert got['hash']==question_set(copy.deepcopy(QUESTIONS))['hash']
 edited=copy.deepcopy(QUESTIONS);edited['questions'][0]['options'].reverse()
 assert got['hash']!=question_set(edited)['hash']
 assert got['questions'][1]['options'][1]['id']=='medium'
 for value in [dict(QUESTIONS,schemaVersion=True),dict(QUESTIONS,revision=True),dict(QUESTIONS,id='../unsafe')]:
  with pytest.raises(ValueError):question_set(value)
 bad=copy.deepcopy(QUESTIONS);bad['questions'][0]['options'][1]['id']='billing'
 with pytest.raises(ValueError):question_set(bad)

def test_targets_are_exact_per_question_distributions_not_token_labels():
 options=QUESTIONS['questions'][0]['options']
 np.testing.assert_array_equal(target_distribution('technical',options),[0,1])
 np.testing.assert_allclose(target_distribution({'technical':.8,'billing':.2},options),[.2,.8])
 for invalid in [None,True,'missing',{'billing':.5},{'billing':float('nan'),'technical':0},{'billing':-.1,'technical':1.1},{'billing':.5,'technical':.4}]:
  with pytest.raises(ValueError):target_distribution(invalid,options)

def test_calibration_split_preserves_case_groups_and_never_touches_validation():
 rows=[{'case':str(i//3),'x':i} for i in range(30)]
 a,b=split_groups(rows,'case',.2,17)
 assert len(a)==24 and len(b)==6
 assert not {r['case'] for r in a}&{r['case'] for r in b}
 assert split_groups(rows,'case',.2,17)==(a,b)
 with pytest.raises(ValueError):split_groups([{'case':'a'}],'case',.5,1)

def test_decoding_uses_top_probability_and_complete_ordinal_distribution():
 q=QUESTIONS['questions'][1];result=decode(q,np.log([.1,.2,.7]),1.,None)
 assert result['answer']=='high' and result['confidence']==pytest.approx(.7)
 assert result['expectedIndex']==pytest.approx(1.6)
 assert result['status']=='uncalibrated' and len(result['probabilities'])==3
 assert 'act_probability' not in result
 for invalid in [[1.,float('nan'),2.],[1.,2.]]:
  with pytest.raises(ValueError):decode(q,np.array(invalid),1.,None)
