import copy
import pytest
from runner import learning_gradients as lesson

TASKS=lesson.TASKS

def request(input_only=False):
 targets=[{'kind':'input','node':'input','port':'out'}]
 if not input_only:targets.append({'kind':'parameter','node':'first','parameter':'weight'})
 return {'id':'input-effect' if input_only else 'selected-effect','output':{'node':'second','port':'out'},'partition':'validation','row_start':0,'row_count':3,'reduction':'mean','upstream':None,'targets':targets,'unavailable':'error'}

def submission(task):
 gradient={'accumulation_steps':2 if task==TASKS[1] else 1,'gradient_scale':'mean','requests':[request(task==TASKS[2])] if task in (TASKS[0],TASKS[2]) else []}
 owners=[{'node':'first','parameter':p,'mode':'frozen' if task==TASKS[2] else 'trainable'}for p in ('weight','bias')] if task in TASKS[2:] else []
 optimizer={'lr':.05,'momentum':.5 if task==TASKS[3] else 0.,'dampening':0.,'weight_decay':0.,'nesterov':False,'maximize':False}
 nodes=[{'id':ident,'type':kind,'name':kind,'x':0,'y':0,'config':config}for ident,kind,config in [('gradient','gradient_policy',gradient),('owners','trainability',{'owners':owners}),('optimizer','optimizer_sgd',optimizer),('train','train',{})]]
 edges=[{'id':str(i),'source':source,'sourcePort':'out','target':'train','targetPort':port}for i,(source,port)in enumerate([('gradient','gradientPolicy'),('owners','trainability'),('optimizer','optimizer')])]
 return {'schemaVersion':1,'kind':'gradientLearningGraph','taskID':task,'graph':{'schemaVersion':10,'name':task,'nodes':nodes,'edges':edges}}

@pytest.mark.parametrize('task',TASKS)
def test_actual_control_namespace_accepts_correct_complete_constructions(task):
 result=lesson.configuration(submission(task),task)
 assert set(result)=={'gradient','trainability','optimizer'}
 assert result['gradient']['componentID']=='gradient_policy'
 if task==TASKS[2]:assert result['gradient']['config']['requests'][0]['targets']==[{'kind':'input','node':'input','port':'out'}]

@pytest.mark.parametrize('change',[lambda s:s.update(expected=[0]),lambda s:s['graph']['nodes'][3]['config'].update(seed=7),lambda s:s['graph']['nodes'][2]['config'].update(lr=.9),lambda s:s['graph']['edges'][0].update(targetPort='optimizer'),lambda s:s['graph']['nodes'][0]['config'].update(gradient_scale='sum'),lambda s:s['graph']['nodes'][1]['config'].update(owners=[{'node':'foreign','parameter':'weight','mode':'frozen'}])])
def test_foreign_answers_context_changes_and_wrong_wires_refuse(change):
 s=submission(TASKS[0]);change(s)
 with pytest.raises(ValueError):lesson.configuration(s,TASKS[0])

def test_frozen_keep_task_rejects_foreign_parameter_request():
 s=submission(TASKS[2]);s['graph']['nodes'][0]['config']['requests'][0]['targets'].append({'kind':'parameter','node':'first','parameter':'weight'})
 with pytest.raises(ValueError):lesson.configuration(s,TASKS[2])

def test_valid_wrong_gradient_choice_is_measured_not_solution_matched():
 s=submission(TASKS[1]);s['graph']['nodes'][0]['config']['accumulation_steps']=3
 assert lesson.configuration(s,TASKS[1])['gradient']['config']['accumulation_steps']==3

def fixture(widths,index=0):
 i,h,o=widths
 x=[[((r*3+c*5+index)%19-9)/16+r/1024 for c in range(i)]for r in range(8)]
 y=[[sum((c+1)*(k+1)*v/8 for c,v in enumerate(row))+.125 for k in range(o)]for row in x]
 def tensor(rows):return {'shape':[len(rows),len(rows[0])],'values':sum(rows,[])}
 return {'id':f'fixture-{index}','inputs':{'trainingInputs':tensor(x[:5]),'trainingTargets':tensor(y[:5]),'heldoutInputs':tensor(x[5:]),'heldoutTargets':tensor(y[5:]),'seed':{'shape':[1],'values':[41+index]}}}

def parse_tensor(value):return value['shape'],value['values']

@pytest.mark.parametrize('task',TASKS)
def test_real_engine_four_fixture_assessment_and_private_safe_difference_roles(task):
 import time
 started=time.monotonic();evidence={'fixtures':[],'maximumTensorElements':0}
 lesson.execute(submission(task),[fixture(w,n)for n,w in enumerate(lesson.WIDTHS)],evidence,lambda:False,parse_tensor,task=task)
 assert time.monotonic()-started<60
 assert len(evidence['fixtures'])==4
 for n,result in enumerate(evidence['fixtures']):
  assert result['fixtureID']==f'fixture-{n}'
  assert result['partitions']=={}
  assert len(result['tensors'])==5
  assert not any(k in result for k in ('expected','seed','path','modelState','optimizerState'))
  for key,tensor in result['tensors'].items():
   if key=='windowSchedule':assert tensor=={'shape':[4,4],'values':[1,2,4,1,1,1,1,2,2,2,4,3,2,1,1,4]}
   elif key=='recordingAvailability':assert tensor=={'shape':[3],'values':[1,0,-1]}
   else:assert max(map(abs,tensor['values']))<2e-6,(task,n,key,tensor)

@pytest.mark.parametrize('shape',[[],[5],[5,2,1]])
def test_malformed_fixture_rank_refuses_before_engine(shape,monkeypatch):
 from runner import engine
 monkeypatch.setattr(engine,'run',lambda *a,**k:pytest.fail('Invalid fixture reached training.'))
 f=fixture(lesson.WIDTHS[0]);f['inputs']['trainingInputs']['shape']=shape
 with pytest.raises(ValueError,match='shape|geometry|rank'):
  lesson.execute(submission(TASKS[0]),[f],{'fixtures':[]},lambda:False,parse_tensor,task=TASKS[0])

def test_heldout_overlap_refuses_without_weakening_real_preparation(monkeypatch):
 from runner import engine
 monkeypatch.setattr(engine,'run',lambda *a,**k:pytest.fail('Leaking fixture reached training.'))
 f=fixture(lesson.WIDTHS[0]);f['inputs']['heldoutInputs']['values'][0]=f['inputs']['trainingInputs']['values'][0]
 with pytest.raises(ValueError,match='overlap'):
  lesson.execute(submission(TASKS[0]),[f],{'fixtures':[]},lambda:False,parse_tensor,task=TASKS[0])

@pytest.mark.parametrize('task',TASKS[:3])
def test_real_wrong_control_choices_cannot_receive_all_matching_comparisons(task):
 s=submission(task)
 if task==TASKS[0]:s['graph']['nodes'][0]['config']['requests'][0]['reduction']='sum'
 elif task==TASKS[1]:s['graph']['nodes'][0]['config']['accumulation_steps']=3
 else:s['graph']['nodes'][1]['config']['owners']=[]
 evidence={'fixtures':[]}
 lesson.execute(s,[fixture(lesson.WIDTHS[0])],evidence,lambda:False,parse_tensor,task=task)
 tables=evidence['fixtures'][0]['tensors']
 if task==TASKS[1]:assert tables['windowSchedule']['shape']!=[4,4]
 else:assert any(abs(v)>2e-6 for table in tables.values()for v in table['values'])

def test_non_influence_task_cannot_grant_itself_derivative_capture():
 s=submission(TASKS[1]);s['graph']['nodes'][0]['config']['requests']=[request()]
 with pytest.raises(ValueError,match='permission|capture|Capture|derivative'):
  lesson.execute(s,[fixture(lesson.WIDTHS[0])],{'fixtures':[]},lambda:False,parse_tensor,task=TASKS[1])

def test_real_finetune_detects_nested_saved_parent_preparation_mutation(monkeypatch):
 from runner import engine
 original=engine.run
 def altered(graph,output,*a,**k):
  result=original(graph,output,*a,**k)
  if output.name=='child':
   parent=k['finetune_checkpoint'].parent
   path=next(p for p in (parent/'preparation').rglob('*')if p.is_file())
   path.write_bytes(path.read_bytes()+b'\n')
  return result
 monkeypatch.setattr(engine,'run',altered)
 e={'fixtures':[]};lesson.execute(submission(TASKS[3]),[fixture(lesson.WIDTHS[0])],e,lambda:False,parse_tensor,task=TASKS[3])
 assert e['fixtures'][0]['tensors']['parentByteDifference']['values'][0]>0

def test_parent_snapshot_retains_many_small_segments_and_detects_changed_bytes(tmp_path):
 parent=tmp_path/'parent';parent.mkdir()
 for index in range(257):
  (parent/f'segment-{index}.json').write_bytes(b'{"values":[1]}')
 before=lesson._parent_snapshot(parent)
 assert len(before)==257 and all(value==b'{"values":[1]}' for value in before.values())
 (parent/'segment-256.json').write_bytes(b'{"values":[2]}')
 after=lesson._parent_snapshot(parent)
 assert set(before)==set(after)
 assert [name for name in before if before[name]!=after[name]]==['segment-256.json']

def test_parent_snapshot_accounts_for_empty_entry_metadata_budget(tmp_path,monkeypatch):
 parent=tmp_path/'parent';parent.mkdir()
 for index in range(10):(parent/f'empty-{index}').touch()
 monkeypatch.setattr(lesson,'_PARENT_SNAPSHOT_MAX_BYTES',2048)
 with pytest.raises(ValueError,match='bounded file inventory'):lesson._parent_snapshot(parent)

def test_parent_snapshot_refuses_payload_beyond_combined_byte_budget(tmp_path):
 parent=tmp_path/'parent';parent.mkdir()
 with (parent/'oversized').open('wb')as stream:stream.truncate(32*1024*1024)
 with pytest.raises(ValueError,match='bounded file inventory'):lesson._parent_snapshot(parent)

@pytest.mark.parametrize('link_directory',[False,True])
def test_parent_snapshot_refuses_external_file_or_directory_symlinks(tmp_path,link_directory):
 parent=tmp_path/'parent';parent.mkdir();outside=tmp_path/'outside'
 if link_directory:outside.mkdir();(outside/'secret').write_text('not a parent file')
 else:outside.write_text('not a parent file')
 (parent/'alias').symlink_to(outside,target_is_directory=link_directory)
 with pytest.raises(ValueError,match='symlink'):lesson._parent_snapshot(parent)


@pytest.mark.parametrize('widths,index',[(widths,index)for index,widths in enumerate(lesson.WIDTHS)])
def test_current_preparation_retains_supplied_five_three_order_and_isolates_neutral_probe(tmp_path,widths,index):
 import hashlib,json,torch
 from runner import engine
 from runner.loss_configuration import canonical_configuration
 raw=fixture(widths,index)['inputs'];inputs={name:[values['values'][offset:offset+values['shape'][-1]]for offset in range(0,len(values['values']),values['shape'][-1])]for name,values in raw.items()if name!='seed'}
 objective={'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':canonical_configuration('loss_mse',{})}
 mapping,prepared=lesson._prepared(tmp_path,inputs,41+index,objective,engine.REGISTRY)
 data,provenance,files=prepared
 matrices=(data[0],data[1]['target'],data[2],data[3]['target'])
 for actual,name in zip(matrices,('trainingInputs','trainingTargets','heldoutInputs','heldoutTargets')):
  assert torch.equal(actual,torch.tensor(inputs[name],dtype=torch.float32))
 graph=json.loads((tmp_path/'flow.json').read_text())
 assert {node['operator']for node in graph['nodes']}=={'kwip.data.sourceCSV','kwip.data.columns','kwip.data.partition'}
 manifest=json.loads((tmp_path/'prepared/manifest.json').read_text());tables={entry['port']:json.loads((tmp_path/'prepared'/entry['path']).read_text())for entry in manifest['artifacts']if entry.get('nodeID')=='p'}
 assert [len(tables[role]['rows'])for role in ('training','validation','test','exclusions')]==[5,3,1,0]
 assert [row[-2]for row in tables['training']['rows']]==list(range(5))
 assert [row[-2]for row in tables['validation']['rows']]==list(range(5,8))
 assert tables['test']['rows']==[[0.]*(widths[0]+widths[2])+[8,'reserved-test-probe']]
 ids=[set(tables[role]['lineage']['observationIDs'])for role in ('training','validation','test')]
 assert len(set.union(*ids))==9 and all(not a&b for index,a in enumerate(ids)for b in ids[index+1:])
 assert set(provenance['trainingRowIDs'])==ids[0]and set(provenance['validationRowIDs'])==ids[1]
 assert not ids[2]&(set(provenance['trainingRowIDs'])|set(provenance['validationRowIDs']))
 assert provenance['preparation']['manifestSHA256']==hashlib.sha256((tmp_path/'prepared/manifest.json').read_bytes()).hexdigest()


def test_actual_requested_derivative_selects_only_three_supplied_heldout_rows(monkeypatch):
 import torch
 from runner import requested_gradients
 supplied=fixture(lesson.WIDTHS[0]);heldout=supplied['inputs']['heldoutInputs'];expected=torch.tensor(heldout['values'],dtype=torch.float32).reshape(heldout['shape'])
 original=requested_gradients.analyze;observed=[]
 def analyze(model,rows,*args,**kwargs):
  assert torch.equal(rows,expected)
  observed.append(rows.clone())
  return original(model,rows,*args,**kwargs)
 monkeypatch.setattr(requested_gradients,'analyze',analyze)
 evidence={'fixtures':[]}
 lesson.execute(submission(TASKS[0]),[supplied],evidence,lambda:False,parse_tensor,task=TASKS[0])
 assert observed and len(evidence['fixtures'])==1
 assert all(max(map(abs,tensor['values']))<2e-6 for tensor in evidence['fixtures'][0]['tensors'].values())
