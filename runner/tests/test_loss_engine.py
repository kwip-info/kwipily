"""Actual sealed typed partitions train all eight objectives without binary coercion."""
import hashlib,json,math
from pathlib import Path
import pytest,torch
from runner import engine,dataflow
from runner.dataflow_cli import run_dataflow
from runner.loss_prepared_data import load_prepared_objective
from runner.tests.test_loss_prepared_data import fixture,CASES
from runner.loss_configuration import validate_recipe

@pytest.fixture(autouse=True)
def restore_runtime_policy():
 old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 yield
 torch.use_deterministic_algorithms(old,warn_only=warn)

def model(op):
 p=CASES[op][0];features=6 if op=='loss_triplet_margin' else 2
 nodes=[dict(id='input',type='input',config={'features':features})]
 def node(ident,typ,c):nodes.append(dict(id=ident,type=typ,config=c))
 if op=='loss_triplet_margin':
  node('shape','reshape',{'shape':[3,2]});node('learn','linear',dict(in_features=2,out_features=2,bias=True))
 else:
  node('learn','linear',dict(in_features=features,out_features=math.prod(p),bias=True))
  if len(p)>1:node('shape','reshape',{'shape':p})
  if op in ('loss_nll','loss_kl_div','loss_ctc'):node('log','log_softmax',{'dim':-1})
 node('output','output',{})
 edges=[dict(id=f'e{i}',source=a['id'],sourcePort='out',target=b['id'],targetPort='in') for i,(a,b) in enumerate(zip(nodes,nodes[1:]))]
 return dict(schemaVersion=2,nodes=nodes,edges=edges)

def prepared(tmp,op,config=None):
 tmp.mkdir(parents=True,exist_ok=True);table,m,r,p=fixture(op)
 if config:r=validate_recipe({**r,'config':{**r['config'],**config}})
 columns=[*table['columns'],{'name':'fixture_order','dtype':'int64','nullable':False}];rows=[]
 for i in range(16):rows.append([v+i*.1 if j<len(m['inputColumns']) else v for j,v in enumerate(table['rows'][i%4])]+[i])
 csv=tmp/'source.csv';csv.write_text(','.join(c['name'] for c in columns)+'\n'+''.join(','.join(str(v) for v in row)+'\n' for row in rows))
 graph=dict(schemaVersion=1,nodes=[dict(id='s',operator='kwip.data.sourceCSV',config={'separator':','}),dict(id='c',operator='kwip.data.columns',config={'rows':[dict(source=c['name'],included=True,outputName=c['name'],targetType=c['dtype'])for c in columns],'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}),dict(id='p',operator='kwip.data.partition',config={'strategy':'time','timeColumn':'fixture_order','timeEncoding':'epochMilliseconds','trainingEnd':7,'validationEnd':11,'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'})],edges=[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},{'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}])
 graph_path=tmp/'dataflow.json';graph_path.write_text(json.dumps(graph));out=tmp/'preparation';run_dataflow(graph_path,{'s':str(csv)},out,lambda _:None)
 sha=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
 result=load_prepared_objective(out/'manifest.json','p','training','p','validation',m,r,expected_manifest_sha256=sha,expected_graph_digest=dataflow.digest(graph),sample_shape=[len(m['inputColumns'])],prediction_shape=p)
 return model(op),r,m,result

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',CASES)
def test_real_eight_objective_prepared_training_actual_outputs_history_and_checkpoint(tmp_path,op,device):
 graph,r,m,data=prepared(tmp_path/'source',op);events=[]
 engine.run(graph,tmp_path/'run',epochs=2,batch_size=5,seed=41,device=device,prepared=data,objective_recipe=r,target_mapping=m,emit=events.append)
 manifest=json.loads((tmp_path/'run/manifest.json').read_text());history=json.loads((tmp_path/'run/metrics.json').read_text());saved=torch.load(tmp_path/'run/checkpoint.pt',weights_only=True)
 assert manifest['settings']['objectiveRecipe']==r and manifest['settings']['targetMapping']==m
 assert manifest['checkpoint']['contract']['objective']['recipe']==r and saved['objectiveState']=={}
 assert len(history)==2 and all(math.isfinite(e['trainLoss']) and math.isfinite(e['validationLoss']) for e in history)
 assert ('accuracy' in history[0])==(op in ('loss_cross_entropy','loss_nll'))
 assert manifest['evaluation']['objective']['observations']==4
 assert next(e for e in events if e['event']=='completed')['evaluationCount']==4
 assert set(data[0][1])=={v['role'] for v in m['roles']}

@pytest.mark.parametrize('op',CASES)
def test_actual_context_admits_only_compatible_output_and_legacy_stays_binary(op):
 _,m,r,p=fixture(op);graph=model(op)
 if p!=[2]:
  with pytest.raises(ValueError,match='Classification output'):engine.validate(graph,bundled_dataset=False)
 context={'recipe':r,'targetMapping':m};engine.validate(graph,bundled_dataset=False,objective_context=context)
 bad=json.loads(json.dumps(context));bad['targetMapping']['componentID']='foreign'
 with pytest.raises(ValueError):engine.validate(graph,bundled_dataset=False,objective_context=bad)

@pytest.mark.parametrize('op',CASES)
def test_cpu_exact_resume_all_objectives_restores_real_identity_state_and_history(tmp_path,op):
 graph,r,m,data=prepared(tmp_path/'source',op)
 kwargs=dict(seed=41,batch_size=5,device='cpu',prepared=data,objective_recipe=r,target_mapping=m)
 engine.run(graph,tmp_path/'full',epochs=4,**kwargs);engine.run(graph,tmp_path/'parent',epochs=2,**kwargs)
 engine.run(graph,tmp_path/'resumed',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kwargs)
 full=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);resumed=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
 assert full['modelState'].keys()==resumed['modelState'].keys()
 assert all(torch.equal(full['modelState'][k],resumed['modelState'][k]) for k in full['modelState'])
 assert torch.equal(full['rngState'],resumed['rngState']) and full['objectiveState']==resumed['objectiveState']
 assert json.loads((tmp_path/'full/metrics.json').read_text())==json.loads((tmp_path/'resumed/metrics.json').read_text())
 lineage=json.loads((tmp_path/'resumed/manifest.json').read_text())['lineage'];assert 'objective' in lineage['restoredState']

@pytest.mark.parametrize('op',CASES)
def test_mps_weights_only_all_objectives_and_exact_refusal(tmp_path,op):
 graph,r,m,data=prepared(tmp_path/'source',op);kwargs=dict(seed=41,batch_size=5,device='mps',prepared=data,objective_recipe=r,target_mapping=m)
 engine.run(graph,tmp_path/'parent',epochs=2,**kwargs)
 with pytest.raises(ValueError,match='CPU'):engine.run(graph,tmp_path/'exact',epochs=3,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kwargs)
 engine.run(graph,tmp_path/'fresh',epochs=1,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**kwargs)
 lineage=json.loads((tmp_path/'fresh/manifest.json').read_text())['lineage'];assert 'objective' in lineage['resetState'] and 'optimizer' in lineage['resetState']

@pytest.mark.parametrize('op',CASES)
def test_actual_exported_objective_helper_is_delivered_separately_from_model(tmp_path,op):
 graph,r,m,data=prepared(tmp_path/'source',op);engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,device='cpu',prepared=data,objective_recipe=r,target_mapping=m)
 path=tmp_path/'run/objective_recipe.py';assert path.is_file() and len(path.read_bytes())<=131072
 assert 'loss_' not in (tmp_path/'run/model.py').read_text()
 import importlib.util
 spec=importlib.util.spec_from_file_location('standalone_objective',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 assert module.OBJECTIVE_RECIPE==r


def test_rehashed_foreign_objective_state_is_refused_even_for_weights_only(tmp_path):
 graph,r,m,data=prepared(tmp_path/'source','loss_cross_entropy');kwargs=dict(seed=41,batch_size=5,device='cpu',prepared=data,objective_recipe=r,target_mapping=m)
 engine.run(graph,tmp_path/'parent',epochs=1,**kwargs)
 checkpoint=tmp_path/'parent/checkpoint.pt';value=torch.load(checkpoint,weights_only=True);value['objectiveState']={'foreign':torch.tensor(1.)};torch.save(value,checkpoint)
 manifest=tmp_path/'parent/manifest.json';doc=json.loads(manifest.read_text());raw=checkpoint.read_bytes();doc['checkpoint'].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));manifest.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match='Objective|objective'):engine.run(graph,tmp_path/'fresh',epochs=1,finetune_checkpoint=checkpoint,**kwargs)

@pytest.mark.parametrize('op',CASES)
@pytest.mark.parametrize('activation,rows',[(True,True),(False,True),(True,False),(False,False)])
def test_none_outputs_actual_shapes_and_independent_capture_permissions(tmp_path,op,activation,rows):
 graph,r,m,data=prepared(tmp_path/'source',op,{'reduction':'none'});events=[]
 engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,device='cpu',prepared=data,objective_recipe=r,target_mapping=m,sample_capture={'schemaVersion':1,'activationValues':activation,'predictionRows':rows},emit=events.append)
 completed=next(e for e in events if e['event']=='completed');manifest=json.loads((tmp_path/'run/manifest.json').read_text())
 from runner.loss_objectives import create_objective,evaluate_objective
 from runner.engine import GraphModel
 model_instance=GraphModel(graph,bundled_dataset=False,objective_context={'recipe':r,'targetMapping':m})
 model_instance.load_state_dict(torch.load(tmp_path/'run/checkpoint.pt',weights_only=True)['modelState']);model_instance.eval()
 with torch.no_grad():expected=evaluate_objective(create_objective(r),model_instance(data[0][2]),data[0][3])['raw']
 preview=completed['objective']['rawPreview'];assert preview==manifest['evaluation']['objective']['rawPreview']
 assert preview['shape']==list(expected.shape) and preview['dtype']=='float32'
 assert preview['values']==(expected.flatten()[:8].tolist() if activation else [])
 assert preview['captureStatus']==('captured' if activation else 'omittedByPolicy')
 assert completed['predictionRows']==json.loads((tmp_path/'run/predictions.json').read_text())['predictionRows']
 assert len(completed['predictionRows'])==(len(expected) if rows else 0)
 for i,row in enumerate(completed['predictionRows']):assert row['objectiveOutput']==expected[i].tolist()
 json.dumps(completed,allow_nan=False)

@pytest.mark.parametrize('op',CASES)
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('reduction,scalarization',[('sum','mean'),('none','mean'),('none','sum')])
def test_actual_partition_reductions_and_exported_native_objective_replay(tmp_path,op,device,reduction,scalarization):
 graph,r,m,data=prepared(tmp_path/'source',op,{'reduction':reduction});events=[]
 engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,device=device,prepared=data,objective_recipe=r,target_mapping=m,objective_scalarization=scalarization,emit=events.append)
 import importlib.util
 def imported(name,path):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
 exported=imported('exported_model',tmp_path/'run/model.py');objective=imported('exported_objective',tmp_path/'run/objective_recipe.py')
 instance=exported.Model();instance.load_state_dict(torch.load(tmp_path/'run/checkpoint.pt',weights_only=True)['modelState']);instance.to(device).eval()
 with torch.no_grad():
  predicted=instance(data[0][2].to(device))
  result=objective.evaluate_training_objective(objective.create_training_objective(device),predicted,{k:v.to(device) for k,v in data[0][3].items()})
 event=next(e for e in events if e['event']=='completed')
 assert event['validationLoss']==pytest.approx(result['scalar'].item(),rel=2e-5,abs=2e-5)
 assert event['objective']['rawShape']==list(result['raw'].shape)
 if reduction=='sum':assert all('objectiveOutput' not in row for row in event['predictionRows'])

@pytest.mark.parametrize('config',[{'reduction':'sum'},{'class_weights':[.5,1.,2.]},{'label_smoothing':.15}])
def test_cpu_exact_resume_refuses_changed_objective_but_weights_only_resets_it(tmp_path,config):
 graph,r,m,data=prepared(tmp_path/'original','loss_cross_entropy');kw=dict(device='cpu',batch_size=5,seed=41)
 engine.run(graph,tmp_path/'parent',epochs=1,prepared=data,objective_recipe=r,target_mapping=m,**kw)
 next_graph,next_r,next_m,next_data=prepared(tmp_path/'changed','loss_cross_entropy',config)
 with pytest.raises(ValueError,match='Exact resume requires matching'):
  engine.run(next_graph,tmp_path/'exact',epochs=2,prepared=next_data,objective_recipe=next_r,target_mapping=next_m,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)
 engine.run(next_graph,tmp_path/'reset',epochs=1,prepared=next_data,objective_recipe=next_r,target_mapping=next_m,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)
 checkpoint=torch.load(tmp_path/'reset/checkpoint.pt',weights_only=True)
 assert checkpoint['contract']['objective']['recipe']==next_r
 if next_r['config']['class_weights']:assert torch.equal(checkpoint['objectiveState']['class_weights'],torch.tensor(next_r['config']['class_weights']))

@pytest.mark.parametrize('op',['loss_mse','loss_l1','loss_triplet_margin'])
def test_plateau_accuracy_unavailable_refuses_before_forward_or_optimizer(tmp_path,op,monkeypatch):
 graph,r,m,data=prepared(tmp_path/'source',op)
 from runner.optimizer_configuration import validate_recipe as optimizer_recipe
 from runner.scheduler_configuration import validate_recipe as scheduler_recipe
 def forbidden(*args,**kwargs):raise AssertionError('Undefined accuracy must be rejected before model construction')
 monkeypatch.setattr(engine,'GraphModel',forbidden)
 with pytest.raises(ValueError,match='accuracy|Accuracy'):
  engine.run(graph,tmp_path/'run',epochs=1,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer_recipe(dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config={})),scheduler_recipe=scheduler_recipe(dict(schemaVersion=1,componentID='scheduler_reduce_on_plateau',configurationVersion=1,config={'metric':'validation_accuracy'})))

@pytest.mark.parametrize('op,config',[
 ('loss_cross_entropy',{'class_weights':[.5,1.,2.],'label_smoothing':.1}),
 ('loss_bce_with_logits',{'pos_weight_mode':'configured','pos_weight_shape':[2],'pos_weight_values':[1.,2.]}),
 ('loss_nll',{'class_weights':[.5,1.,2.]})])
def test_fixed_objective_buffers_cpu_exact_resume_and_strict_state(tmp_path,op,config):
 graph,r,m,data=prepared(tmp_path/'source',op,config);kw=dict(device='cpu',batch_size=5,seed=41,prepared=data,objective_recipe=r,target_mapping=m)
 engine.run(graph,tmp_path/'full',epochs=4,**kw);engine.run(graph,tmp_path/'parent',epochs=2,**kw)
 engine.run(graph,tmp_path/'resumed',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)
 full=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);resume=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
 assert full['objectiveState'] and full['objectiveState'].keys()==resume['objectiveState'].keys()
 assert all(torch.equal(full['objectiveState'][k],resume['objectiveState'][k]) for k in full['objectiveState'])
 assert all(torch.equal(full['modelState'][k],resume['modelState'][k]) for k in full['modelState'])
 assert json.loads((tmp_path/'full/metrics.json').read_text())==json.loads((tmp_path/'resumed/metrics.json').read_text())

@pytest.mark.parametrize('op',CASES)
def test_current_objective_fixture_preserves_explicit_population_values(tmp_path,op):
 _,_,mapping,loaded=prepared(tmp_path,op)
 table,_,_,_=fixture(op)
 feature_indices=[next(i for i,c in enumerate(table['columns'])if c['name']==name)for name in mapping['inputColumns']]
 # Independent supplied four-row pattern plus the declared per-observation offset.
 for selected,positions in [(loaded[0][0],range(8)),(loaded[0][2],range(8,12))]:
  expected=[[table['rows'][i%4][j]+i*.1 for j in feature_indices]for i in positions]
  torch.testing.assert_close(selected,torch.tensor(expected,dtype=torch.float32),rtol=0,atol=0)
 assert loaded[1]['preparation']['fittedStateDigests']==[]
 graph=json.loads((tmp_path/'dataflow.json').read_text())
 assert [n['operator']for n in graph['nodes']]==['kwip.data.sourceCSV','kwip.data.columns','kwip.data.partition']
 manifest=json.loads((tmp_path/'preparation/manifest.json').read_text());ids=[]
 for role,positions in [('training',range(8)),('validation',range(8,12)),('test',range(12,16))]:
  entry=next(a for a in manifest['artifacts']if a.get('nodeID')=='p'and a.get('port')==role)
  artifact=json.loads((tmp_path/'preparation'/entry['path']).read_text())
  assert [row[-1]for row in artifact['rows']]==list(positions)
  assert artifact['lineage']['role']==role
  ids.append(set(artifact['lineage']['observationIDs']))
 assert all(ids[i].isdisjoint(ids[j])for i in range(3)for j in range(i))
