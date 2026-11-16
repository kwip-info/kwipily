"""A campaign is an owned journal of bounded slices, not an unbounded worker.

Replay is shared once; each committed iteration has one immutable result segment.
Recovery reconciles an atomic replay boundary that preceded its journal publication.
"""
import copy,contextlib,fcntl,hashlib,json,os,time,uuid
from pathlib import Path
if __package__:
 from .game_episodes import require,encoded,digest
 from .self_play_replay import ReplayStore,read_owned,write_file
 from .self_play_training import TrainingSettings,train
else:
 from game_episodes import require,encoded,digest
 from self_play_replay import ReplayStore,read_owned,write_file
 from self_play_training import TrainingSettings,train

FIELDS=('iterations','episodes','moves','evaluationGames','measurements')
def read_campaign(root):
 value=json.loads(read_owned(Path(root),'campaign.json',2_000_000))
 require(value.get('kind')=='kwip.self-play.campaign.v1' and type(value.get('iteration')) is int and 0<=value['iteration']<=1000000 and type(value.get('phase')) is int and 1<=value['phase']<=1000 and type(value.get('segments')) is list and len(value['segments'])<=256,'Invalid campaign journal.')
 return value
def atomic(root,name,value):
 temporary='.'+uuid.uuid4().hex;write_file(root,temporary,encoded(value));os.replace(Path(root)/temporary,Path(root)/name)
 fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
def immutable_json(root,name,value):
 raw=encoded(value)
 if (Path(root)/name).exists():require(read_owned(root,name,2_000_000)==raw,'Conflicting immutable campaign phase.')
 else:write_file(root,name,raw)
def scientific(definition,graph,configuration,initialization,trainability):
 c=copy.deepcopy(configuration)
 for k in ('iterations','seconds','campaign'):c.pop(k,None)
 g=copy.deepcopy(graph)
 for k in ('name','execution'):g.pop(k,None)
 for node in g['nodes']:
  for k in ('name','x','y'):node.pop(k,None)
 return digest([definition,g,c,initialization,trainability])
def _bytes(root):
 total=0
 for path in Path(root).rglob('*'):
  require(not path.is_symlink(),'Campaign store contains a link.')
  if path.is_file():require(path.stat().st_nlink==1,'Campaign store contains a hard link.');total+=path.stat().st_size
 return total

def run_slice(root,definition,graph,configuration,*,action='resume',check_cancelled=lambda:None,progress=lambda p:None,initialization=None,trainability=None,opponents=None,expected_index='unrestricted-test',parent_root=None,parent_index=None,pause_requested=lambda:False):
 root=Path(root);root.mkdir(mode=0o700,parents=True,exist_ok=True);require(not root.is_symlink(),'Campaign store is a link.')
 c=TrainingSettings(**configuration);require(c.campaign.get('enabled') is True,'Enable a finite campaign before using campaign actions.')
 require(action in ('start','resume','extend','fork'),'Choose Start, Resume or Extend. Changed scientific settings require a Fork.')
 fingerprint=scientific(definition,graph,c.value(),initialization,trainability)
 fd=os.open(root/'campaign.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  exists=(root/'campaign.json').exists()
  if expected_index!='unrestricted-test':
   actual=hashlib.sha256(read_owned(root,'campaign.json',2_000_000)).hexdigest() if exists else None
   require(expected_index==actual,'Campaign changed after capture; reload before continuing.')
  if exists:
   index=read_campaign(root);require(action!='start','Campaign already exists; Resume or Extend it.')
   require(index['scientificSHA256']==fingerprint,'Fork required: rules, model, evaluation or learning/search profile changed.')
   require(index['maximumBytes']==c.campaign['maximumBytes'],'Fork required: campaign disk allowance changed.')
  else:
   require(action in ('start','fork'),'No campaign exists; choose Start.')
   for name in ('segments','phases'):
    (root/name).mkdir(mode=0o700,exist_ok=True);require(not (root/name).is_symlink(),'Campaign metadata directory is a link.')
   initial=None
   if action=='fork':
    require(parent_root is not None,'Fork requires a captured parent campaign.')
    require(Path(parent_root).resolve()!=root.resolve(),'A fork needs a distinct owned campaign.')
    parent_fd=os.open(Path(parent_root)/'campaign.lock',os.O_RDWR|os.O_NOFOLLOW)
    try:
     fcntl.flock(parent_fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
     parent=read_campaign(parent_root)
     require(hashlib.sha256(read_owned(parent_root,'campaign.json',2_000_000)).hexdigest()==parent_index,'Parent campaign changed after capture.')
     identity=scientific(definition,graph,{},None,None)
     require(parent.get('gameModelSHA256')==identity,'Fork requires compatible rules and model construction; replay is never guessed compatible.')
     cp,m=ReplayStore(Path(parent_root)/'replay').checkpoint();entry=next(f for f in m['files'] if f['path']=='model.pt')
     if __package__:from .self_play_replay import read_pinned
     else:from self_play_replay import read_pinned
     raw=read_pinned(cp,entry,100_000_000);initial=write_file(root,'initial-model.pt',raw)
    finally:os.close(parent_fd)
   index=dict(gameModelSHA256=scientific(definition,graph,{},None,None),initial=initial,parent=None if parent_root is None else dict(id=parent['id'],iteration=parent['iteration'],mode='weights only; optimizer/replay reset'),kind='kwip.self-play.campaign.v1',id=str(uuid.uuid4()).upper(),scientificSHA256=fingerprint,iteration=0,target=c.iterations,phase=1,phaseStart=1,activeSeconds=0.,maximumSeconds=c.campaign['maximumSeconds'],maximumBytes=c.campaign['maximumBytes'],segments=[],segmentCount=0)
   immutable_json(root/'phases','phase-00000001.json',dict(configuration=configuration,initialization=initialization,trainability=trainability,sourceSHA256=fingerprint))
   atomic(root,'campaign.json',index)
  def publish_boundary():
   nonlocal index
   replay=ReplayStore(root/'replay');_,m=replay.checkpoint();it=m['iteration']
   if it<=index['iteration']:return
   require(it==index['iteration']+1,'Campaign boundary skipped an iteration.')
   state=m['state'];segment={name:[r for r in state[name] if r['iteration']==it] for name in FIELDS}
   segment.update(iteration=it,pins=m['pins'],weightsSHA256=segment['iterations'][-1]['checkpointSHA256'])
   raw=encoded(segment);require(len(raw)<=8_000_000,'Iteration segment exceeds the8MB publication allowance.')
   name=f'iteration-{it:08}.json';path=root/'segments'/name
   if path.exists():require(read_owned(root/'segments',name,8_000_000)==raw,'Conflicting immutable campaign segment.')
   else:write_file(root/'segments',name,raw)
   entry=dict(path='segments/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),iteration=it,phase=index['phase'])
   next_index=copy.deepcopy(index);next_index['iteration']=it;next_index['segmentCount']=it;next_index['segments']=(index['segments']+[entry])[-256:]
   next_index['activeSeconds']=state.get('campaignActiveSeconds',index['activeSeconds'])
   require(_bytes(root)<=index['maximumBytes'],'Campaign disk budget reached; required boundaries and pins were not deleted.')
   atomic(root,'campaign.json',next_index);index=next_index
  if (root/'replay/latest.json').exists():publish_boundary()
  if index.get('pendingSlice'):
   pending=index.pop('pendingSlice');index['activeSeconds']=max(index['activeSeconds'],pending['base']+pending['allowance']);atomic(root,'campaign.json',index)
  if action=='extend':
   require(c.iterations>=index['target'] and (c.iterations>index['target'] or c.campaign['maximumSeconds']>index['maximumSeconds']) and index['iteration']<=index['target'],'Invalid committed campaign boundary.')
   index['phase']+=1;index['phaseStart']=index['iteration']+1;index['target']=c.iterations;index['maximumSeconds']=c.campaign['maximumSeconds']
   immutable_json(root/'phases',f"phase-{index['phase']:08}.json",dict(configuration=configuration,parentIteration=index['iteration'],kind='budget-extension',schedule='new explicit segment',scientificSHA256=fingerprint))
   atomic(root,'campaign.json',index)
  else:require(c.iterations==index['target'],'Extend required for a changed total iteration budget.')
  require(index['iteration']<index['target'],'Campaign complete; Extend it to add learning budget.')
  require(index['activeSeconds']<index['maximumSeconds'],'Campaign active-time budget exhausted; explicitly extend the budget.')
  # Reserve a bounded next slice, including temporary state and result publication, before updates.
  require(_bytes(root)+c.replayBytes+32_000_000<=index['maximumBytes'],'Campaign needs space for the next bounded slice; review retention/allowance.')
  started=time.monotonic();base=index['activeSeconds']
  index['pendingSlice']=dict(base=base,allowance=min(c.seconds,index['maximumSeconds']-base));atomic(root,'campaign.json',index)
  def check():
   check_cancelled()
   require(base+time.monotonic()-started<index['maximumSeconds'],'Campaign active-time budget exhausted; last commit retained.')
  def report(event):
   if event['phase']=='checkpoint':publish_boundary()
   progress(event)
  initial_state=None
  if index.get('initial'):
   if __package__:from .self_play_replay import read_pinned
   else:from self_play_replay import read_pinned
   initial_state=__import__('torch').load(__import__('io').BytesIO(read_pinned(root,index['initial'],100_000_000)),weights_only=True,map_location='cpu')
  result=train(definition,graph,configuration,root/'replay',resume=(root/'replay/latest.json').exists(),check_cancelled=check,progress=report,initialization=initialization,trainability=trainability,campaign_context=dict(phaseStart=index['phaseStart'],activeSeconds=base),initial_state=initial_state,opponents=opponents,pause_requested=pause_requested)
  publish_boundary();index.pop('pendingSlice',None);index['activeSeconds']=base+time.monotonic()-started;atomic(root,'campaign.json',index)
  phase_files=sorted((root/'phases').glob('phase-*.json'));require(len(phase_files)<=1000,'Campaign phase inventory limit reached.')
  phases=[]
  for path in phase_files:
   phase=json.loads(read_owned(root/'phases',path.name,2000000));settings=phase.get('configuration',{});phases.append(dict(phase=int(path.stem.split('-')[1]),parentIteration=phase.get('parentIteration',0),target=settings.get('iterations'),learningRate=settings.get('learningRate'),kind=phase.get('kind','initial'),schedule=phase.get('schedule',settings.get('scheduler','constant'))))
  result['phases']=phases;result['campaign']=dict(id=index['id'],phase=index['phase'],iteration=index['iteration'],target=index['target'],complete=index['iteration']>=index['target'],activeSeconds=index['activeSeconds'],retainedBytes=_bytes(root),segmentCount=index['segmentCount'],reproducibility=result['devices']['reproducibility'])
  return result
 finally:
  if 'started' in locals() and (root/'campaign.json').exists():
   current=read_campaign(root);current.pop('pendingSlice',None);current['activeSeconds']=max(current['activeSeconds'],base+time.monotonic()-started);atomic(root,'campaign.json',current)
  os.close(fd)
