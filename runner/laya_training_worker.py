"""Typed task dispatch inside the existing owned model operation, not another engine."""
import hashlib,json
from pathlib import Path
if __package__:
 from . import laya_training as training
 from . import dataflow as df
 from .dataflow_table_reader import read_table_artifact
 from .typed_decisions import require
 from .decision_tables import DECISION_COLUMNS,schema
else:
 import laya_training as training,dataflow as df
 from dataflow_table_reader import read_table_artifact
 from typed_decisions import require
 from decision_tables import DECISION_COLUMNS,schema

def execute(authority,inner,info,output,files,write,emit,check):
 root=Path(authority._path);source=root/'source';action=inner['action'];c=dict(inner['settings']);role=c.pop('evaluationRole',None)
 require(c.get('task')=='typedDecisions' and c.get('resumeMode') is None,'Choose typed-decision settings; exact resume is not qualified.')
 budget=(authority.request['executionBudget'].get('resources') or {}).get('maximumResidentBytes') or c['maximumResidentBytes'];c['maximumResidentBytes']=min(budget,c['maximumResidentBytes']);c=training.settings(c)
 values={};ids={};sources={};roles=('training','evaluation','evaluationPopulation') if action=='evaluate' else ('training','validation')
 names=list(dict.fromkeys(c['stateColumns']+list(c['targetColumns'].values())+[c['groupColumn']]))
 for name in roles:
  path='input/'+name+'.table.json';pin=next(x['sha256'] for x in authority.request['inputFiles'] if x['path']==path);table=read_table_artifact(root/path,pin,check)
  try:
   expected=role if name.startswith('evaluation') else name
   require(table.role==expected or (action!='evaluate' and table.role=='unassigned'),'Use the declared Training/Validation partitions, never protected Test for fitting.')
   require(1<=len(table.rows)<=10000,'Typed decisions accept complete populations up to10000 rows.');ids[name]=set(table.row_ids);sources[name]=table.source_digest
   if name=='evaluationPopulation':continue
   indices=[table.index(n) for n in names];require(all(table.columns[i]['dtype'] in ('string','int64','float32','float64','bool') for i in indices),'Decision training mappings require scalar columns.')
   values[name]=[dict(zip(names,[row[i] for i in indices]),__observation_id=identity) for row,identity in df.row_pairs(table)]
  finally:table.rows.store.close();table.row_ids.store.close()
 other='evaluation' if action=='evaluate' else 'validation';require(not ids['training']&ids[other],'Training and held-out observations overlap.')
 training.require_disjoint_groups(values['training'],values[other],c['groupColumn'])
 if action=='prepare':write('tokens.json',training.prepared_preview(source,values['training'],values['validation'],c,check));return
 def table_result(rows,model_identity,partition):
  columns=schema(DECISION_COLUMNS);rowids=[df.digest([r[0],r[2],model_identity]) for r in rows]
  write('table.json',dict(schemaVersion=1,kind='table',columns=columns,rows=rows,lineage=dict(role=partition,sourceDigest=model_identity,membershipDigest=df.membership_digest(rowids),observationIDs=rowids,dataVersion=df.digest(dict(columns=columns,rows=rows,ids=rowids)),partitionDigest='',fittedStateDigests=[],diagnostics=[])))
 if action=='evaluate':
  require(role in ('validation','test') and ids['evaluationPopulation']==ids['evaluation'] and sources['evaluationPopulation']==sources['evaluation'],'Evaluation must score the complete originating protected population.')
  report=training.evaluate(source,values['evaluation'],c,check);rows=report.pop('rows');report.update(role=role,observationIDs=sorted(ids['evaluation']),sourceDigest=sources['evaluation'],modelIdentity=info['identity']);write('tokens.json',report);table_result(rows,info['identity'],role);return
 require(action=='train','Unsupported typed-decision operation.')
 def pause():
  if not (root/'pause-after-epoch').exists():return False
  if __package__:from .worker_authorization import directory,read_regular
  else:from worker_authorization import directory,read_regular
  import os
  fd=directory(str(root))
  try:return read_regular(fd,'pause-after-epoch',64)==b'pause'
  finally:os.close(fd)
 training.fine_tune(source,output/'training',values['training'],values['validation'],c,emit=emit,check_cancelled=check,run_id=inner['operationID'],pause_requested=pause)
 for path in sorted((output/'training').rglob('*')):
  if not path.is_file():continue
  check();h=hashlib.sha256()
  with path.open('rb') as f:
   while part:=f.read(1048576):check();h.update(part)
  files.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h.hexdigest()))
 if __package__:from .laya_models import inspect_model
 else:from laya_models import inspect_model
 metadata=inspect_model(output/'training/model',check=check);write('model-metadata.json',metadata)
 if c['capturePredictions']:table_result(training.prediction_rows(output/'training/model',values['validation'],c,check),metadata['identity'],'validation')
