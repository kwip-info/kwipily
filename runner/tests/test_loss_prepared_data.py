import copy,hashlib,json
from pathlib import Path
import pytest,torch
from runner import dataflow as df
from runner.dataflow_cli import run_dataflow
from runner.loss_configuration import validate_recipe
from runner.tests.test_columns import config as columns_config,item
from runner.loss_prepared_data import validate_target_mapping,table_objective_tensors,load_prepared_objective

CASES={
 'loss_cross_entropy':([3],[('target',[], 'int64',[2])],{}),
 'loss_nll':([3],[('target',[], 'int64',[2])],{}),
 'loss_bce_with_logits':([2],[('target',[2],'float32',[.2,.8]),('weight',[2],'float32',[.5,2.])],{'weight_mode':'mapped'}),
 'loss_mse':([2],[('target',[2],'float32',[-2.,1.])],{}),
 'loss_l1':([2],[('target',[2],'float32',[-2.,1.])],{}),
 'loss_kl_div':([2],[('target',[2],'float32',[.4,.6])],{}),
 'loss_ctc':([3,3],[('target',[2],'int64',[1,2]),('input_lengths',[],'int64',[3]),('target_lengths',[],'int64',[2])],{'target_layout':'packed'}),
 'loss_triplet_margin':([3,2],[],{})}

def fixture(op,role='training'):
 p,roles,c=CASES[op];features=[f'x{i}'for i in range(6 if op=='loss_triplet_margin'else 2)]
 spec=[{'role':name,'columns':[name+str(i)for i in range(len(v))],'sampleShape':shape,'dtype':dtype}for name,shape,dtype,v in roles]
 mapping={'schemaVersion':1,'componentID':op,'inputColumns':features,'roles':spec,'tripletSourceGroups':[features[:2],features[2:4],features[4:]]if op=='loss_triplet_margin'else []}
 columns=[{'name':x,'dtype':'float64','nullable':False}for x in features]+[{'name':name+str(i),'dtype':'int64'if dtype=='int64'else 'float64','nullable':False}for name,shape,dtype,v in roles for i in range(len(v))]
 rows=[[float(i+j)for j in range(len(features))]+[v for _,_,_,values in roles for v in values]for i in range(4)]
 ids=[role+str(i)for i in range(4)];lineage={'role':role,'observationIDs':ids,'membershipDigest':df.membership_digest(ids),'sourceDigest':'a'*64,'partitionDigest':'b'*64,'dataVersion':df.digest({'columns':columns,'rows':rows,'ids':ids}),'fittedStateDigests':[],'diagnostics':[]}
 return {'schemaVersion':1,'kind':'table','columns':columns,'rows':rows,'lineage':lineage},mapping,validate_recipe({'schemaVersion':1,'componentID':op,'configurationVersion':1,'config':c}),p

@pytest.mark.parametrize('op',CASES)
def test_eight_real_typed_roles_row_order_and_no_dummy_triplet_target(op):
 item,m,r,p=fixture(op);before=copy.deepcopy(item)
 x,roles,lineage,columns=table_objective_tensors(item,'training',m,r,'float32',[len(m['inputColumns'])],p)
 assert x.dtype==torch.float32 and x.shape==(4,len(m['inputColumns']))and x[2].tolist()==item['rows'][2][:len(m['inputColumns'])]
 assert set(roles)=={v['role']for v in m['roles']}
 for descriptor in m['roles']:
  value=roles[descriptor['role']];assert value.dtype==getattr(torch,descriptor['dtype'])and value.shape==(4,*descriptor['sampleShape'])
 assert item==before and lineage==item['lineage']and columns==item['columns']

@pytest.mark.parametrize('change',[lambda m:m.update(schemaVersion=True),lambda m:m.update(componentID='loss_l1'),lambda m:m.update(extra=1),lambda m:m['inputColumns'].append('x0'),lambda m:m['roles'][0].update(columns=['x0']),lambda m:m['roles'][0].update(sampleShape=[2]),lambda m:m['roles'][0].update(dtype='bool'),lambda m:m['roles'].append(copy.deepcopy(m['roles'][0]))])
def test_mapping_tampering_refuses_before_table_allocation(change):
 _,m,r,_=fixture('loss_cross_entropy');change(m)
 with pytest.raises(ValueError):validate_target_mapping(m,r)

@pytest.mark.parametrize('change',[lambda t:t['rows'][0].__setitem__(0,True),lambda t:t['rows'][0].__setitem__(2,2.5),lambda t:t['lineage'].update(role='validation'),lambda t:t['lineage']['observationIDs'].__setitem__(1,'training0'),lambda t:t['columns'][2].update(dtype='float64')])
def test_rehashed_bad_values_dtype_or_lineage_refused(change):
 t,m,r,p=fixture('loss_cross_entropy');change(t);t['lineage']['membershipDigest']=df.membership_digest(t['lineage']['observationIDs']);t['lineage']['dataVersion']=df.digest({'columns':t['columns'],'rows':t['rows'],'ids':t['lineage']['observationIDs']})
 with pytest.raises(ValueError):table_objective_tensors(t,'training',m,r,'float32',[2],p)

@pytest.mark.parametrize("segmented",[False,True])
def test_real_preparation_inventory_typed_targets_pins_and_legacy_refusal(tmp_path,segmented):
 from runner.workload_policy import LEGACY
 source=tmp_path/'rows.csv';source.write_text('x0,x1,target0\n'+''.join(f'{i},{i+.5},2\n'for i in range(16)))
 graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{'separator':','}},{'id':'c','operator':'kwip.data.columns','config':columns_config([item('x0',dtype='float64'),item('x1',dtype='float64'),item('target0',dtype='int64')])},{'id':'p','operator':'kwip.data.partition','config':{'strategy':'random','seed':7,'validationPercent':25,'testPercent':25}}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},{'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}]}
 graph_file=tmp_path/'flow.json';graph_file.write_text(json.dumps(graph));out=tmp_path/'prepared';run_dataflow(graph_file,{'s':str(source)},out,lambda _:None,workload_policy=LEGACY.to_document() if segmented else None)
 raw=(out/'manifest.json').read_bytes();h=hashlib.sha256(raw).hexdigest();_,m,r,p=fixture('loss_cross_entropy')
 kwargs=dict(expected_manifest_sha256=h,expected_graph_digest=df.digest(graph),input_dtype='float32',sample_shape=[2],prediction_shape=p)
 values,provenance,files=load_prepared_objective(out/'manifest.json','p','training','p','validation',m,r,**kwargs)
 # This fixture explicitly authors current three-population membership; it is not a legacy Split upgrade.
 x,roles,v,held=values;assert len(x)==8 and len(v)==4 and set(roles['target'].tolist())=={2}and set(held['target'].tolist())=={2}
 assert x.dtype==v.dtype==torch.float32 and roles['target'].dtype==held['target'].dtype==torch.int64
 for features in (x,v):
  assert torch.equal(features[:,1],features[:,0]+.5)
  assert features[:,0].tolist()==sorted(features[:,0].tolist())
 assert provenance['objectiveMapping']==m and 'targetColumn'not in provenance and provenance['preparation']['manifestSHA256']==h and 'dataflow.json'in files
 assert set(provenance['trainingRowIDs']).isdisjoint(provenance['validationRowIDs'])
 from runner.prepared_data import load_prepared
 with pytest.raises(ValueError,match='0 or 1'):load_prepared(out/'manifest.json','p','training','p','validation',['x0','x1'],'target0',**{k:v for k,v in kwargs.items()if k!='prediction_shape'})
 with pytest.raises(ValueError):load_prepared_objective(out/'manifest.json','p','training','p','validation',m,r,**{**kwargs,'expected_manifest_sha256':'0'*64})

@pytest.mark.parametrize('op',['loss_mse','loss_l1','loss_kl_div','loss_bce_with_logits'])
@pytest.mark.parametrize('value',[None,True,float('inf'),1e39,1e-50])
def test_floating_target_conversion_never_silently_repairs(op,value):
 t,m,r,p=fixture(op);t['rows'][0][2]=value
 # Nonfinite JSON cannot be resealed; the decoder must refuse it as supplied.
 if value != float('inf'):
  t['lineage']['dataVersion']=df.digest({'columns':t['columns'],'rows':t['rows'],'ids':t['lineage']['observationIDs']})
 with pytest.raises(ValueError):table_objective_tensors(t,'training',m,r,'float32',[2],p)

@pytest.mark.parametrize('op',CASES)
def test_prediction_geometry_is_checked_before_tensor_allocation(op):
 t,m,r,p=fixture(op)
 with pytest.raises(ValueError):table_objective_tensors(t,'training',m,r,'float32',[len(m['inputColumns'])],[True])

@pytest.mark.parametrize('change',[
 lambda m:m['tripletSourceGroups'].__setitem__(2,['x0','x1']),
 lambda m:m['tripletSourceGroups'].__setitem__(1,['x2']),
 lambda m:m.update(roles=[{'role':'target','columns':['y'],'sampleShape':[],'dtype':'int64'}]),
 lambda m:m.update(tripletSourceGroups=list(reversed(m['tripletSourceGroups'])))])
def test_triplet_source_groups_are_disjoint_ordered_and_not_dummy_labels(change):
 _,m,r,_=fixture('loss_triplet_margin');change(m)
 with pytest.raises(ValueError):validate_target_mapping(m,r)

def test_packed_ctc_preparation_keeps_full_observation_owned_suffixes():
 t,m,r,p=fixture('loss_ctc');t['rows'][0][-1]=1;t['rows'][0][3]=-99
 t['lineage']['dataVersion']=df.digest({'columns':t['columns'],'rows':t['rows'],'ids':t['lineage']['observationIDs']})
 _,roles,_,_=table_objective_tensors(t,'training',m,r,'float32',[2],p)
 assert roles['target'].shape==(4,2)and roles['target'][0].tolist()==[1,-99]
 assert roles['target_lengths'].tolist()==[1,2,2,2]


def test_class_shares_and_integer_inputs_have_distinct_explicit_dtypes():
 t,m,r,p=fixture('loss_cross_entropy')
 r=validate_recipe({**r,'config':{**r['config'],'target_mode':'probabilities'}})
 m['roles']=[{'role':'target','columns':['p0','p1','p2'],'sampleShape':[3],'dtype':'float32'}]
 t['columns']=t['columns'][:2]+[{'name':f'p{i}','dtype':'float64','nullable':False}for i in range(3)]
 t['rows']=[row[:2]+[.2,.3,.5]for row in t['rows']]
 t['lineage']['dataVersion']=df.digest({'columns':t['columns'],'rows':t['rows'],'ids':t['lineage']['observationIDs']})
 x,roles,_,_=table_objective_tensors(t,'training',m,r,'float32',[2],p)
 assert torch.equal(roles['target'],torch.tensor([[.2,.3,.5]]*4))
 with pytest.raises(ValueError,match='explicit int64'):table_objective_tensors(t,'training',m,r,'int64',[2],p)
 for c in t['columns'][:2]:c['dtype']='int64'
 for row in t['rows']:row[:2]=[int(v)for v in row[:2]]
 t['lineage']['dataVersion']=df.digest({'columns':t['columns'],'rows':t['rows'],'ids':t['lineage']['observationIDs']})
 x,roles,_,_=table_objective_tensors(t,'training',m,r,'int64',[2],p)
 assert x.dtype==torch.int64 and roles['target'].dtype==torch.float32
