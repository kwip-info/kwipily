"""Complete typed-decision tables through the existing dataflow/worker authority."""
import json,math
from pathlib import Path
from contextvars import ContextVar
from contextlib import contextmanager
if __package__:
 from . import dataflow as df
 from .typed_decisions import require,question_set
 from .laya_models import Laya
else:
 import dataflow as df
 from typed_decisions import require,question_set
 from laya_models import Laya
DEFAULT_QUESTIONS={'schemaVersion':1,'id':'ticket-triage','revision':1,'name':'Ticket triage','questions':[{'id':'category','type':'choice','instructions':'Which team should handle this ticket?','options':[{'id':'billing','label':'Billing and payments'},{'id':'technical','label':'Technical problems'},{'id':'account','label':'Account access'}]}]}
DEFAULTS=dict(stateColumns=['text'],identityColumn='',questionSet=DEFAULT_QUESTIONS,questionsFromModel=False,batchRows=1,maximumRows=1000,maximumScoreRows=100000)
DECISION_COLUMNS=[('source_observation','string',False),('source_key','string',False),('question','string',False),('type','string',False),('answer','string',False),('confidence','float64',False),('expected_index','float64',True),('yes_probability','float64',True),('status','string',False),('needs_review','bool',False),('model_identity','string',False),('question_set','string',False)]
SCORE_COLUMNS=[('source_observation','string',False),('source_key','string',False),('question','string',False),('option_key','string',False),('option_label','string',False),('option_index','int64',False),('probability','float64',False),('temperature','float64',False),('model_identity','string',False),('question_set','string',False)]
def schema(columns):return [dict(name=n,dtype=t,nullable=v) for n,t,v in columns]
def validate(c):
 require(type(c) is dict and set(c)==set(DEFAULTS),'Decide needs its complete versioned configuration.')
 require(type(c['stateColumns']) is list and 1<=len(c['stateColumns'])<=16 and len(set(c['stateColumns']))==len(c['stateColumns']) and all(type(n) is str and 0<len(n)<=160 for n in c['stateColumns']),'Choose unique bounded state columns.')
 require(type(c['identityColumn']) is str and len(c['identityColumn'])<=160 and type(c['questionsFromModel']) is bool,'Choose source identity and question source.')
 for key,lo,hi in [('batchRows',1,8),('maximumRows',1,10000),('maximumScoreRows',1,200000)]:require(type(c[key]) is int and lo<=c[key]<=hi,'Decide compute/output limits exceed admission.')
 question_set(c['questionSet']);return c
_context=ContextVar('typed_decision_job',default=None)
@contextmanager
def job_scope(root,check):
 root=Path(root);path=root/'decision-models.json';pins=json.loads(path.read_bytes()) if path.exists() else {};token=_context.set(dict(root=root,models=pins,adapters={},check=check))
 try:yield
 finally:_context.reset(token)
def adapter(node):
 ctx=_context.get();require(ctx is not None and node in ctx['models'],'Connect a verified typed-decision Load model or completed Train model.')
 pin=ctx['models'][node];key=pin['identity']
 if key not in ctx['adapters']:
  engine=Laya(ctx['root']/'decision-models'/key,ctx['check']);require(engine.info['identity']==key,'Captured decision-model identity changed.');ctx['adapters'][key]=engine
 return ctx['adapters'][key]
def apply(table,c,check,node,engine=None):
 c=validate(c);check();engine=engine or adapter(node)
 questions=engine.task.get('questionSet') if c['questionsFromModel'] else c['questionSet'];require(questions is not None,'The model has no question set; choose a project question set.');qset=question_set(questions)
 require(len(table.rows)<=c['maximumRows'],'Complete decision row allowance exceeded.')
 score_count=sum(len(q['options']) for q in qset['questions'])*len(table.rows);require(score_count<=c['maximumScoreRows'],'Complete score row allowance exceeded; increase the explicit limit or reduce input.')
 indices=[table.index(n) for n in c['stateColumns']]
 require(all(table.columns[i]['dtype'] in ('string','int64','float32','float64','bool') for i in indices),'State columns must be scalar text/numeric/Boolean values, not media or tensor references.')
 key_index=table.index(c['identityColumn']) if c['identityColumn'] else None
 decisions=[];scores=[];batch=[];keys=[];seen_keys=set()
 def flush():
  check()
  if not batch:return
  predicted=engine.predict([x[0] for x in batch],questions,check=check);require(len(predicted)==len(batch),'Decision batch row count changed.')
  for (state,source,key),answers in zip(batch,predicted):
   require(len(answers)==len(qset['questions']),'Decision question count changed.')
   for q,result in zip(qset['questions'],answers):
    require(result['question']==q['id'] and len(result['probabilities'])==len(q['options']),'Decision probabilities differ from the declared question set.')
    row=[source,key,q['id'],q['type'],result['answer'],result['confidence'],result['expectedIndex'],result['yesProbability'],result['status'],result['status']!='accepted',engine.info['identity'],qset['hash']]
    ident=df.digest([source,qset['hash'],q['id'],engine.info['identity']]);decisions.append((row,ident))
    for i,(option,probability) in enumerate(zip(q['options'],result['probabilities'])):
     require(type(probability) in (float,int) and math.isfinite(probability) and 0<=probability<=1,'Invalid option probability.')
     scores.append(([source,key,q['id'],option['id'],option['label'],i,probability,result['temperature'],engine.info['identity'],qset['hash']],df.digest([ident,option['id']])))
  batch.clear()
 for row,source in df.row_pairs(table):
  check();state={name:row[i] for name,i in zip(c['stateColumns'],indices)}
  require(all(v is not None for v in state.values()) and any(str(v).strip() for v in state.values()),'Decision state has missing/empty required fields.')
  key=str(row[key_index]) if key_index is not None else source
  require(key not in seen_keys and key not in ('','None'),'Source keys must be present and unique.');seen_keys.add(key)
  batch.append((state,source,key))
  if len(batch)>=c['batchRows']:flush()
 flush()
 return dict(decisions=df.streaming_rows(table,iter(decisions),schema(DECISION_COLUMNS),check,identity_prefix=''),scores=df.streaming_rows(table,iter(scores),schema(SCORE_COLUMNS),check,identity_prefix=''))
