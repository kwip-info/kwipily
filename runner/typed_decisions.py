"""Versioned, inert question/target contracts shared by Train and Decide."""
import hashlib,json,math,random,re
import numpy as np
CALIBRATION_METHOD='group-selection-independent-audit-Wilson-upper95-v2'

def require(ok,message):
 if not ok:raise ValueError(message)
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def name(value):return type(value) is str and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}',value) is not None
def question_set(value):
 require(type(value) is dict and set(value)=={'schemaVersion','id','revision','name','questions'},'Choose a complete versioned question set.')
 require(type(value['schemaVersion']) is int and value['schemaVersion']==1 and name(value['id']) and type(value['revision']) is int and 1<=value['revision']<=1000000,'Invalid question-set identity/revision.')
 require(type(value['name']) is str and 1<=len(value['name'].strip())<=160 and type(value['questions']) is list and 1<=len(value['questions'])<=16,'A question set needs a name and1–16 questions.')
 ids=[]
 for q in value['questions']:
  require(type(q) is dict and set(q)=={'id','type','instructions','options'} and name(q['id']),'Question needs stable ID, type, instructions and options.')
  require(q['type'] in ('choice','score','noul') and type(q['instructions']) is str and 1<=len(q['instructions'].strip())<=2000,'Choose a typed question and bounded instructions.')
  opts=q['options'];require(type(opts) is list and 2<=len(opts)<=32,'Each question needs2–32 explicit options.')
  for o in opts:require(type(o) is dict and set(o)=={'id','label'} and name(o['id']) and type(o['label']) is str and 1<=len(o['label'].strip())<=240,'Use stable option keys and bounded labels.')
  require(len({o['id'] for o in opts})==len(opts) and len({o['label'] for o in opts})==len(opts),'Question options must have unique keys and labels.')
  if q['type']=='noul':require([o['id'] for o in opts]==['false','true'],'Yes/no semantic order must be false then true.')
  ids.append(q['id'])
 require(len(set(ids))==len(ids),'Question IDs must be unique.')
 return dict(value,hash=identity(value))
def internal_question(q):
 return {'t':q['type'],'ins':q['instructions'],'crit':{o['id']:o['label'] for o in q['options']} if q['type']=='choice' else [o['label'] for o in q['options']] if q['type']=='score' else {'false':q['options'][0]['label'],'true':q['options'][1]['label']}}
def target_distribution(value,options):
 keys=[o['id'] for o in options]
 if type(value) is str:
  require(value in keys,'Target label is not a declared option.');return np.array([float(k==value) for k in keys],dtype=np.float32)
 require(type(value) is dict and set(value)==set(keys),'Soft target needs exactly one probability per declared option.')
 require(all(type(value[k]) in (int,float) and math.isfinite(value[k]) and 0<=value[k]<=1 for k in keys),'Target probabilities must be finite and nonnegative.')
 require(abs(sum(value.values())-1)<=1e-6,'Target probabilities must sum to one.')
 return np.array([value[k] for k in keys],dtype=np.float32)
def split_groups(rows,group,fraction,seed):
 require(type(fraction) in (int,float) and math.isfinite(fraction) and 0<=fraction<1 and type(seed) is int,'Invalid held-out calibration split.')
 for r in rows:require(group in r and type(r[group]) in (str,int) and str(r[group]),'Every observation needs its source group identity.')
 groups=sorted({str(r[group]) for r in rows});n=int(len(groups)*fraction)
 if fraction>0:require(n>=1 and n<len(groups),'Calibration needs enough independent source groups; choose a larger population or disable fitting.')
 random.Random(seed).shuffle(groups);held=set(groups[:n])
 return ([r for r in rows if str(r[group]) not in held],[r for r in rows if str(r[group]) in held])
def decode(q,logits,temperature,calibration):
 z=np.asarray(logits,dtype=np.float64);require(z.shape==(len(q['options']),) and np.isfinite(z).all(),'Decision logits do not match the declared finite options.')
 require(type(temperature) in (int,float) and math.isfinite(temperature) and .5<=temperature<=5,'Unsupported decision temperature.')
 z=z/temperature;p=np.exp(z-z.max());p/=p.sum();i=int(p.argmax());confidence=float(p[i]);status='uncalibrated'
 if calibration is not None:
  require(type(calibration) is dict and set(calibration)>={'threshold','support'} and type(calibration['support']) is int and calibration['support']>=10 and type(calibration['threshold']) in (int,float) and math.isfinite(calibration['threshold']) and 0<=calibration['threshold']<=1,'Calibration evidence is insufficient or invalid.')
  status='accepted' if calibration.get('acceptEnabled',True) and confidence>=calibration['threshold'] else 'needsReview'
 return dict(answer=q['options'][i]['id'],confidence=confidence,expectedIndex=float(np.arange(len(p))@p) if q['type']=='score' else None,yesProbability=float(p[1]) if q['type']=='noul' else None,status=status,probabilities=[float(x) for x in p])
