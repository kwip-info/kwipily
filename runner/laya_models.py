"""Offline reviewed ModernBERT/Laya adapter. Assets are inert; no SDK/server/remote code."""
import hashlib,json,math,os,shutil,stat
from pathlib import Path
import numpy as np
if __package__:
 from .typed_decisions import require,question_set,internal_question,decode,identity,CALIBRATION_METHOD
else:from typed_decisions import require,question_set,internal_question,decode,identity,CALIBRATION_METHOD
NAMES={'rl_agent_config.json','model.safetensors','encoder/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json','tokenizer/added_tokens.json','kwip-decisions.json','config.json','README.md','LICENSE','LICENSE.txt','NOTICE','NOTICE.txt'}
REQUIRED={'rl_agent_config.json','model.safetensors','encoder/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json'}

def _read(folder,relative,maximum,check):
 require(relative in NAMES,'Unsupported model asset path.')
 try:
  parent=os.open(folder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:
   parts=relative.split('/')
   if len(parts)==2:
    nested=os.open(parts[0],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);os.close(parent);parent=nested
   fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
   try:
    before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=maximum,'Model needs bounded regular files, never symbolic links.')
    chunks=[];h=hashlib.sha256();total=0
    while True:
     check();part=os.read(fd,1048576)
     if not part:break
     total+=len(part);require(total<=maximum,'Model asset exceeds byte budget.');h.update(part)
     if not relative.endswith('.safetensors'):chunks.append(part)
    after=os.fstat(fd);named=os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and (after.st_dev,after.st_ino)==(named.st_dev,named.st_ino) and total==before.st_size,'Model asset changed during inspection.')
    return dict(path=relative,bytes=total,sha256=h.hexdigest()),b''.join(chunks)
   finally:os.close(fd)
  finally:os.close(parent)
 except OSError as e:raise ValueError('Model assets must be readable regular files, never symbolic links.') from e

def _json(raw):
 def pairs(items):
  out={}
  for k,v in items:require(k not in out,'Duplicate model metadata key.');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite model metadata.')))
def inventory(folder,maximum_bytes=2_000_000_000,check=lambda:None):
 folder=Path(folder);require(not folder.is_symlink() and folder.is_dir(),'Choose a regular model directory.')
 files=[];documents={};total=0
 for name in sorted(NAMES):
  if not (folder/name).exists() and not (folder/name).is_symlink():continue
  entry,raw=_read(folder,name,maximum_bytes if name.endswith('.safetensors') else 8_000_000,check);total+=entry['bytes'];require(total<=maximum_bytes,'Model source exceeds byte budget.');files.append(entry)
  if name.endswith('.json'):documents[name]=_json(raw)
 require(REQUIRED<={f['path'] for f in files},'Laya needs its complete encoder, tokenizer, decision config and safe weights.')
 for doc in documents.values():require(type(doc) is dict and not doc.get('auto_map') and not doc.get('custom_pipelines'),'Repository code is not supported.')
 return files,documents,total

def build(config,encoder_config,empty=False):
 import torch
 from transformers import ModernBertConfig,ModernBertModel
 from transformers.initialization import no_init_weights
 if __package__:from .laya_architecture import DecisionModel
 else:from laya_architecture import DecisionModel
 # Explicit architecture, never AutoModel dispatch from untrusted metadata.
 with torch.random.fork_rng(),no_init_weights():
  c=ModernBertConfig.from_dict(encoder_config);c._attn_implementation='sdpa'
  if empty:
   with torch.device('meta'):model=DecisionModel(ModernBertModel(c),config['head_layers'],len(config['act_costs'])+1)
  else:model=DecisionModel(ModernBertModel(c),config['head_layers'],len(config['act_costs'])+1,no_init=True)
 return model

def inspect_model(folder,maximum_bytes=2_000_000_000,check=lambda:None):
 from safetensors import safe_open
 require(type(maximum_bytes) is int and 0<maximum_bytes<=2_000_000_000,'Invalid model byte budget.')
 files,docs,total=inventory(folder,maximum_bytes,check);c=docs['rl_agent_config.json'];e=docs['encoder/config.json'];t=docs['tokenizer/tokenizer_config.json']
 require(e.get('model_type')=='modernbert' and c.get('encoder') in ('answerdotai/ModernBERT-large','answerdotai/ModernBERT-base') and c.get('option_layout','sequential')=='sequential','This Laya adapter supports reviewed ModernBERT sequential-option checkpoints only.')
 require(not e.get('quantization_config') and not e.get('is_encoder_decoder') and type(c.get('head_layers')) is int and 1<=c['head_layers']<=2 and type(c.get('act_costs')) is dict and len(c['act_costs'])==1,'Unsupported Laya architecture/head configuration.')
 for key,lo,hi in [('hidden_size',64,1024),('num_hidden_layers',1,28),('num_attention_heads',1,16),('intermediate_size',64,4096),('vocab_size',8,50368),('max_position_embeddings',128,8192)]:require(type(e.get(key)) is int and lo<=e[key]<=hi,'Laya encoder geometry exceeds reviewed bounds.')
 require(e['hidden_size']%64==0 and e['hidden_size']%e['num_attention_heads']==0,'Invalid Laya attention geometry.')
 require(type(c.get('max_len')) is int and 32<=c['max_len']<=1024 and c['max_len']<=e['max_position_embeddings'] and type(c.get('head_max_len')) is int and 16<=c['head_max_len']<=256 and c['head_max_len']<c['max_len'],'Laya sequence budgets exceed admission.')
 require(t.get('tokenizer_class') in ('PreTrainedTokenizerFast','TokenizersBackend') and all(t.get(k)==v for k,v in [('cls_token','[CLS]'),('sep_token','[SEP]'),('mask_token','[MASK]'),('pad_token','[PAD]')]),'Laya needs its exact matching fast tokenizer and special tokens.')
 if 'config.json' in docs:require(docs['config.json'].get('model_type') in ('laya',None),'Laya root config conflicts with its decision task.')
 shapes={};parameters=0;weight_bytes=0;dtypes=set()
 with safe_open(Path(folder)/'model.safetensors',framework='pt',device='cpu') as f:
  require(len(f.keys())<=4096,'Laya weight inventory exceeds the tensor-count limit.')
  for key in f.keys():
   check();v=f.get_slice(key);shape=tuple(v.get_shape());dtype=v.get_dtype();require(dtype in ('F16','F32','BF16') and all(type(n) is int and n>0 for n in shape),'Invalid Laya safe weight shape/type.')
   shapes[key]=shape;count=math.prod(shape);parameters+=count;weight_bytes+=count*(4 if dtype=='F32' else 2);dtypes.add(dtype)
 require(parameters<=430_000_000,'Laya weight parameters exceed reviewed admission.')
 model=build(c,e,empty=True);expected={k:tuple(v.shape) for k,v in model.state_dict().items()};require(shapes==expected,'Laya weights do not match the complete encoder/head architecture.');del model
 from transformers import PreTrainedTokenizerFast
 tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(folder)/'tokenizer/tokenizer.json'),**{k:t[k] for k in ('cls_token','sep_token','mask_token','pad_token','unk_token')})
 require(len(tok)<=e['vocab_size'] and len({tok.cls_token_id,tok.sep_token_id,tok.mask_token_id,tok.pad_token_id,tok.unk_token_id})==5,'Laya tokenizer vocabulary/special token IDs disagree.')
 temperatures=c.get('temperature',[1.,1.,1.]);buckets=c.get('temperature_by_options',{});require(type(temperatures) is list and len(temperatures)==3 and type(buckets) is dict,'Invalid Laya temperature metadata.')
 warnings=[]
 for key,value in list(enumerate(temperatures))+list(buckets.items()):
  require(type(value) in (int,float) and math.isfinite(value) and value>0,'Laya temperature must be finite and positive.')
  if not .5<=value<=5:warnings.append(str(key)+' clamped to supported0.5–5; domain calibration unqualified')
 task=docs.get('kwip-decisions.json')
 if task is not None:
  require(type(task) is dict and task.get('schemaVersion')==1,'Unknown typed-decision package metadata.');questions=question_set(task['questionSet'])
  calibration=task.get('calibration')
  if calibration is not None:
   require(type(calibration) is dict and calibration.get('questionHash')==questions['hash'] and type(calibration.get('questions')) is dict,'Calibration must match the complete question revision.')
   require(set(calibration['questions'])=={q['id'] for q in questions['questions']},'Calibration must cover every declared question.')
   for fit in calibration['questions'].values():
    require(type(fit) is dict and type(fit.get('temperature')) in (int,float) and math.isfinite(fit['temperature']) and .5<=fit['temperature']<=5,'Invalid fitted temperature.')
    require(type(fit.get('threshold')) in (int,float) and math.isfinite(fit['threshold']) and 0<=fit['threshold']<=1 and type(fit.get('acceptEnabled')) is bool,'Invalid fitted review policy.')
    require(type(fit.get('support')) is int and fit['support']>=10 and type(fit.get('acceptedSupport')) is int and 0<=fit['acceptedSupport']<=fit['support'],'Invalid calibration support.')
    require(not fit['acceptEnabled'] or fit['acceptedSupport']>=10,'Acceptance requires independent support.')
    if fit.get('method')==CALIBRATION_METHOD:
     require(fit.get('supportUnit')=='independentSourceGroup' and fit.get('errorUnit')=='anyIncorrectAcceptedDecisionInSourceGroup','Unknown calibration audit unit.')
     require(all(type(fit.get(k)) is int and fit[k]>=0 for k in ('selectionSupport','auditSupport','rowSupport','splitSeed')) and fit['selectionSupport']+fit['auditSupport']==fit['support'] and fit['acceptedSupport']<=fit['auditSupport'] and fit['rowSupport']>=fit['support'],'Invalid independent calibration support.')
     require(all(type(fit.get(k)) is str and len(fit[k])==64 and all(c in '0123456789abcdef' for c in fit[k]) for k in ('selectionGroupsDigest','auditGroupsDigest')) and fit['selectionGroupsDigest']!=fit['auditGroupsDigest'],'Calibration selection and audit identities must be distinct.')
     require(type(fit.get('targetError')) in (int,float) and math.isfinite(fit['targetError']) and 0<=fit['targetError']<=.5,'Invalid calibration error objective.')
     if fit['acceptEnabled']:
      require(fit.get('acceptanceReason')=='qualifiedIndependentAudit' and all(type(fit.get(k)) in (int,float) and math.isfinite(fit[k]) and 0<=fit[k]<=1 for k in ('observedError','errorUpperBound')) and fit['observedError']<=fit['errorUpperBound']<=fit['targetError'],'Automatic acceptance needs a successful independent audit.')
    else:warnings.append('Legacy calibration has no independent group audit; newly computed decisions require review.')
 return dict(schemaVersion=1,architecture='laya-modernbert',task='typedDecisions',capabilities=['inspect','typedDecisions','export','typedTrainHead','typedTrainFull'],identity=identity(files),files=files,parameterCount=parameters,weightBytes=weight_bytes,totalBytes=total,dtypes=sorted(dtypes),vocabularySize=e['vocab_size'],maximumTokens=c['max_len'],decisionHeadTokens=c['head_max_len'],specialTokens={},calibrationWarnings=warnings,questionSet=task.get('questionSet') if task else None,calibration=task.get('calibration') if task else None,training={'supportedModes':['head','full'],'supportedObjectives':['softCE'],'resume':'weightsOnly','devices':['cpu']})

class Laya:
 def __init__(self,folder,check=lambda:None,maximum_resident_bytes=None,maximum_tokens=None):
  import torch
  from transformers import PreTrainedTokenizerFast
  from safetensors.torch import load_file
  self.folder=Path(folder);self.info=inspect_model(folder,check=check);_,docs,_=inventory(folder,check=check);self.config=docs['rl_agent_config.json'];self.task=docs.get('kwip-decisions.json',{});self.check=check
  require(maximum_tokens is None or type(maximum_tokens) is int and maximum_tokens>=1,'Invalid task token allowance.')
  self.maximum_tokens=min(maximum_tokens or self.config['max_len'],self.config['max_len'])
  if maximum_resident_bytes is not None:require(self.info['parameterCount']*4+500_000_000<=maximum_resident_bytes,'Laya resident model exceeds the run memory budget.')
  t=docs['tokenizer/tokenizer_config.json'];self.tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(self.folder/'tokenizer/tokenizer.json'),**{k:t[k] for k in ('cls_token','sep_token','mask_token','pad_token','unk_token')})
  check();self.model=build(self.config,docs['encoder/config.json']);self.model.load_state_dict(load_file(self.folder/'model.safetensors'),strict=True);self.model=self.model.float().cpu().eval();check()
 def prepare(self,states,questions,check=None):
  if __package__:from . import laya_architecture as architecture
  else:import laya_architecture as architecture
  check=check or self.check;schema=question_set(questions);require(type(states) is list and 1<=len(states)<=8,'Decide accepts1–8 states per CPU batch.')
  groups=[];metadata=[]
  for state in states:
   require(type(state) in (str,dict,list) and len(json.dumps(state,ensure_ascii=False,allow_nan=False).encode())<=262144,'State must be bounded inert text or structured data.')
   require(self.tokenizer.mask_token not in json.dumps(state,ensure_ascii=False),'State contains the reserved model mask token; remove or encode it explicitly.')
   items=[]
   for q in schema['questions']:
    check();require(self.tokenizer.mask_token not in q['instructions'] and all(self.tokenizer.mask_token not in o['label'] and self.tokenizer.mask_token not in o['id'] for o in q['options']),'Question contains the reserved model mask token.');internal=internal_question(q);options=architecture.render_options(internal)
    lengths=[len(self.tokenizer(' '+o.replace(self.tokenizer.mask_token,' '),add_special_tokens=False)['input_ids']) for o in options]
    instructions=self.tokenizer(q['type']+' question: '+q['instructions'].replace(self.tokenizer.mask_token,' '),add_special_tokens=False)['input_ids']
    require(max(lengths)<=48 and len(instructions)<=max(8,self.config['head_max_len']-sum(n+1 for n in lengths)) and sum(n+1 for n in lengths)<=self.config['head_max_len']-16,'Question/options exceed the exact token budget; shorten them instead of silently dropping text.')
    ids,markers,stats,coverage=architecture.build_sequence(self.tokenizer,state,internal,self.config['max_len'],self.config['head_max_len'],return_stats=True,return_truncation_stats=True)
    require(not coverage['truncated'] and len(markers)==len(q['options']) and stats['options_distinct']==len(q['options']),'State/options exceed the model token budget or collapse after encoding.')
    require(len(ids)<=self.maximum_tokens,'Encoded state/questions exceed the configured task token budget.')
    items.append(dict(ids=ids,markers=markers,qtype=architecture.QTYPES[q['type']]));metadata.append(q)
   groups.append(items)
  b=architecture.collate_items(groups,self.tokenizer.pad_token_id)
  return {k:b[k] for k in ('input_ids','attention_mask','marker_pos','marker_mask','qtype')},metadata
 def logits(self,states,questions,check=None):
  import torch
  check=check or self.check;b,meta=self.prepare(states,questions,check);check()
  with torch.inference_mode():raw,_=self.model(**b)
  check();require(torch.isfinite(raw).all().item(),'Laya produced nonfinite logits.');return raw,meta
 def predict(self,states,questions,check=None):
  raw,meta=self.logits(states,questions,check);schema=question_set(questions);calibration=self.task.get('calibration')
  valid_calibration=calibration if calibration and calibration.get('questionHash')==schema['hash'] else None
  answers=[]
  for i,q in enumerate(meta):
   k=len(q['options']);bucket=q['type']+(':'+('2' if k<=2 else '3-5' if k<=5 else '6-10' if k<=10 else '11+'));type_index={'choice':0,'score':1,'noul':2}[q['type']]
   active=valid_calibration.get('questions',{}).get(q['id']) if valid_calibration else None
   # Existing packages and archived tables remain readable. A newly computed
   # decision cannot inherit automatic acceptance from the unaudited v1 fit.
   if active is not None and active.get('method')!=CALIBRATION_METHOD:active=dict(active,acceptEnabled=False)
   temperature=active['temperature'] if active else self.config.get('temperature_by_options',{}).get(bucket,self.config.get('temperature',[1.,1.,1.])[type_index]);temperature=min(5.,max(.5,temperature))
   result=decode(q,raw[i,:k].detach().cpu().numpy(),temperature,active);result['question']=q['id'];result['temperature']=temperature
   if i%len(schema['questions'])==0:answers.append([])
   answers[-1].append(result)
  return answers

def export_model(folder,destination,check=lambda:None):
 files,_,_=inventory(folder,check=check);destination=Path(destination);require(not destination.exists(),'Model destination already exists; choose a new folder.')
 # Publication is staged beside the destination, then renamed without an overwrite.
 import tempfile
 temporary=Path(tempfile.mkdtemp(prefix='.kwip-laya-',dir=destination.parent))
 try:
  for f in files:
   check();src=Path(folder)/f['path'];out=temporary/f['path'];out.parent.mkdir(parents=True,exist_ok=True)
   with src.open('rb') as source,out.open('xb') as target:
    h=hashlib.sha256()
    while part:=source.read(1048576):check();target.write(part);h.update(part)
   require(h.hexdigest()==f['sha256'],'Model source changed during export.')
  info=inspect_model(temporary,check=check);require(info['files']==files,'Export inventory differs from source.')
  if __package__:from .dataflow_output import atomic_rename
  else:from dataflow_output import atomic_rename
  fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:atomic_rename(fd,temporary.name,destination.name)
  finally:os.close(fd)
 except BaseException:
  shutil.rmtree(temporary);raise
 return info
