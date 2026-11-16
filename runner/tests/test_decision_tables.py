import copy
import pytest
from runner import decision_tables as dt,dataflow as df
from runner.tests.test_typed_decisions import QUESTIONS
class Engine:
 info={'identity':'a'*64};task={}
 def predict(self,states,questions,check=None):
  assert [s['text'] for s in states]==['refund']
  return [[dict(question=q['id'],answer=q['options'][0]['id'],confidence=.7,expectedIndex=.4 if q['type']=='score' else None,yesProbability=.3 if q['type']=='noul' else None,status='uncalibrated',temperature=1.,probabilities=[.7]+[.3/(len(q['options'])-1)]*(len(q['options'])-1)) for q in questions['questions']]]
def table():
 return df.Table([dict(name='text',dtype='string',nullable=False)],[['refund']],['case-a'],'source','unassigned')
def test_complete_tables_and_missing_evidence_visible():
 t=table();c=dict(dt.DEFAULTS,questionSet=copy.deepcopy(QUESTIONS));out=dt.apply(t,c,lambda:None,'decision',Engine())
 assert len(out['decisions'].rows)==3 and len(out['scores'].rows)==7
 assert all(row[out['decisions'].index('needs_review')] for row in out['decisions'].rows)
 assert set(r[out['scores'].index('option_key')] for r in out['scores'].rows)>={'billing','technical','low','medium','high','true','false'}
 assert all(r[out['decisions'].index('source_observation')]=='case-a' for r in out['decisions'].rows)
def test_limits_and_unavailable_model_questions_never_execute_silently():
 t=table();c=dict(dt.DEFAULTS,questionSet=copy.deepcopy(QUESTIONS))
 with pytest.raises(ValueError,match='score'):dt.apply(t,dict(c,maximumScoreRows=2),lambda:None,'d',Engine())
 with pytest.raises(ValueError,match='question'):dt.apply(t,dict(c,questionsFromModel=True),lambda:None,'d',Engine())
 def cancel():raise InterruptedError()
 with pytest.raises(InterruptedError):dt.apply(t,c,cancel,'d',Engine())

def test_columnar_source_prefix_is_not_inherited_by_expanded_decision_ids():
 from runner.dataflow_columnar import ColumnarStore
 t=table();store=ColumnarStore(t.columns,identity_prefix='source:')
 store.append(['refund'],'source:case-a');rows,ids=store.finish()
 stored=df.Table(t.columns,rows,ids,'source','validation')
 out=dt.apply(stored,dict(dt.DEFAULTS,questionSet=copy.deepcopy(QUESTIONS)),lambda:None,'d',Engine())
 assert len(out['decisions'].rows)==3 and len(set(out['decisions'].row_ids))==3
 assert all(r[out['decisions'].index('source_observation')]=='source:case-a' for r in out['decisions'].rows)
