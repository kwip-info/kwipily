import numpy as np
import pytest
import torch
from runner.media_artifacts import encode_asset
from runner.named_inputs import convert_rows,validate_plan


def plan():
    return dict(schemaVersion=3,inputs=[dict(name='image',nodeID='pixels',representation='prepared',dtype='float32',sampleShape=[1,2,2],sampleAxes=['channels','height','width'],columns=['prepared'])],roles=[dict(name='target',representation='numeric',dtype='int64',sampleShape=[],columns=['label'])],maximumBatchBytes=1048576)


def test_prepared_tensor_exact_axes_dtype_and_owned_reader():
    array=np.array([[[1,2],[3,4]]],dtype='<f4')
    raw,ref=encode_asset(array.tobytes(),kind='tensor',shape=[1,2,2],axes=['channels','height','width'],dtype='float32',metadata={})
    columns=[dict(name='prepared',dtype='tensor',nullable=False),dict(name='label',dtype='int64',nullable=False)]
    p=plan()
    with pytest.raises(ValueError):convert_rows(p,columns,[[ref,1]])
    x,y=convert_rows(p,columns,[[ref,1]],media_reader=lambda _:raw)
    assert torch.equal(x['image'],torch.tensor(array)[None]) and y['target'].tolist()==[1]
    p['inputs'][0]['sampleAxes']=['height','channels','width']
    with pytest.raises(ValueError,match='axes'):convert_rows(p,columns,[[ref,1]],media_reader=lambda _:raw)
    p=plan();p['inputs'][0]['dtype']='int64'
    with pytest.raises(ValueError,match='dtype'):convert_rows(p,columns,[[ref,1]],media_reader=lambda _:raw)
    with pytest.raises(ValueError):convert_rows(plan(),columns,[[ref,1]],media_reader=lambda _:raw[:-1])


def test_audio_mask_is_an_explicit_exact_int64_input_without_substitution():
    p=plan();p['inputs']=[dict(name='audio',nodeID='a',representation='prepared',dtype='float32',sampleShape=[1,4],sampleAxes=['channels','samples'],columns=['audio']),dict(name='mask',nodeID='m',representation='prepared',dtype='int64',sampleShape=[4],sampleAxes=['samples'],columns=['mask'])]
    _,audio=encode_asset(np.array([[1,2,0,0]],dtype='<f4').tobytes(),kind='tensor',shape=[1,4],axes=['channels','samples'],dtype='float32',metadata={})
    audio_raw,_=encode_asset(np.array([[1,2,0,0]],dtype='<f4').tobytes(),kind='tensor',shape=[1,4],axes=['channels','samples'],dtype='float32',metadata={})
    mask_raw,mask=encode_asset(np.array([1,1,0,0],dtype='<i8').tobytes(),kind='tensor',shape=[4],axes=['samples'],dtype='int64',metadata={})
    schema=[dict(name=n,dtype=d,nullable=False) for n,d in [('audio','tensor'),('mask','tensor'),('label','int64')]]
    assets={audio['sha256']:audio_raw,mask['sha256']:mask_raw}
    x,y=convert_rows(p,schema,[[audio,mask,1]],media_reader=lambda ref:assets[ref['sha256']])
    assert x['audio'].shape==(1,1,4) and x['mask'].dtype==torch.int64 and x['mask'].tolist()==[[1,1,0,0]]
    p['maximumBatchBytes']=1
    with pytest.raises(ValueError):convert_rows(p,schema,[[audio,mask,1]],media_reader=lambda ref:assets[ref['sha256']])


@pytest.mark.parametrize('capture',[True,False])
def test_real_image_fit_frozen_reload_and_unseen_inference(tmp_path,capture):
    import json,hashlib
    from PIL import Image
    from runner import dataflow as df
    from runner.prepare_media import defaults as prepare_defaults
    from runner.file_input_options import defaults as file_defaults
    from runner.dataflow_cli import run_dataflow
    from runner.named_prepared_data import load_named_prepared
    from runner.engine import run,REGISTRY
    from runner.frozen_checkpoint import load_frozen
    source=tmp_path/'images';source.mkdir()
    for i in range(40):Image.fromarray(np.full((2,2),255 if i%2 else 0,dtype=np.uint8)).save(source/(str(i%2)+'-'+str(i)+'.png'))
    def default(name):return next(o['defaultConfig'] for o in df.REGISTRY['operators'] if o['id']=='kwip.data.'+name)
    read=file_defaults();read.update(sourceMode='directory',extensions=['png'])
    formula={'operations':[dict(id='label',expression='regex_extract([name], "^([01])-", "search", "error", "", "1")',outputs=[dict(name='label',dtype='string',write='new')],nullPolicy='error')]}
    columns={**default('columns'),'rows':[dict(source='label',outputName='label',targetType='int64',included=True)]}
    partition={**default('partition'),'stratifyColumn':'label'}
    c=prepare_defaults();c.update(color='L',sampleFormat='float32',scaling='unit',layout='CHW')
    graph=dict(schemaVersion=1,nodes=[dict(id='source',operator='kwip.data.sourceFiles',config=read),dict(id='labels',operator='kwip.data.formula',config=formula),dict(id='cast',operator='kwip.data.columns',config=columns),dict(id='partition',operator='kwip.data.partition',config=partition)]+[dict(id=role,operator='kwip.data.prepareMedia',config=c) for role in ('training','validation','test')],edges=[])
    for a,ap,b in [('source','table','labels'),('labels','table','cast'),('cast','table','partition')]+[('partition',role,role) for role in ('training','validation','test')]:graph['edges'].append({'from':dict(node=a,port=ap),'to':dict(node=b,port='table')})
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));out=tmp_path/'prepared'
    run_dataflow(path,{'source':str(source)},out,lambda _:None,retain_outputs=capture,required_nodes=['partition','training','validation','test'])
    def node(i,t,c):return dict(id=i,type=t,config=c,name='',x=0,y=0)
    model=dict(schemaVersion=2,name='Tiny image classifier',namedInputs={'image':'pixels'},nodes=[node('pixels','input',dict(shape=[1,2,2])),node('flatten','flatten',dict(start_dim=1,end_dim=-1)),node('linear','linear',dict(in_features=4,out_features=2,bias=True)),node('out','output',{})],edges=[])
    for i,(a,b) in enumerate([('pixels','flatten'),('flatten','linear'),('linear','out')]):model['edges'].append(dict(id=str(i),source=a,sourcePort='out',target=b,targetPort='in'))
    p=plan();sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prepared=load_named_prepared(out/'manifest.json','training','table','validation','table',p,model,expected_manifest_sha256=sha(out/'manifest.json'),expected_graph_digest=df.digest(graph))
    trained=tmp_path/'trained'
    run(model,trained,epochs=2,device='cpu',batch_size=4,prepared=prepared,**({} if capture else {'sample_capture':dict(schemaVersion=3,activationValues=False,predictionRows=False,performanceMeasurements=False,retainComponentOutputs=False,retainDiagnosticDetails=False)}))
    frozen=load_frozen(trained/'checkpoint.pt',expected_manifest_sha256=sha(trained/'manifest.json'),expected_checkpoint_sha256=sha(trained/'checkpoint.pt'),registry_path=REGISTRY,expected_registry_sha256=sha(REGISTRY))
    assert frozen.input_contract=={'namedInputPlan':p}
    unseen={'image':torch.tensor([[[[0.,.25],[.5,1.]]]],dtype=torch.float32)}
    with torch.no_grad():assert frozen.model(unseen).shape==(1,2)
    assert (trained/'preparation'/'manifest.json').is_file()
    from runner.prepared_data import load_evaluation_partition
    from runner.final_evaluation import evaluate_classifier
    import shutil
    shutil.rmtree(source)
    (test_x,test_y),provenance,_=load_evaluation_partition(out/'manifest.json','test','table',frozen.training_provenance,partition_role='test',expected_manifest_sha256=sha(out/'manifest.json'),named_plan=p,model_graph=model)
    result=evaluate_classifier(frozen.model,test_x,test_y,provenance['testRowIDs'],batch_size=3,device='cpu')
    assert result['evaluatedCount']==len(test_y)>0
    if not capture:assert not (trained/'predictions.json').exists()


def test_prepared_audio_can_use_16000_samples_within_existing_byte_allowance():
    p=plan();p['inputs'][0].update(sampleShape=[1,16000],sampleAxes=['channels','samples'])
    raw,ref=encode_asset(bytes(64000),kind='tensor',shape=[1,16000],axes=['channels','samples'],dtype='float32',metadata={})
    schema=[dict(name='prepared',dtype='tensor',nullable=False),dict(name='label',dtype='int64',nullable=False)]
    x,_=convert_rows(p,schema,[[ref,0]],media_reader=lambda _:raw)
    assert x['image'].shape==(1,1,16000)


def test_actual_audio_attention_forward_uses_explicit_blocked_mask_and_axes():
    from runner.engine import GraphModel
    from runner.attention_configuration import normalize_attention
    audio_raw,audio=encode_asset(np.array([[1,2,0,0]],dtype='<f4').tobytes(),kind='tensor',shape=[1,4],axes=['channels','samples'],dtype='float32',metadata={})
    mask_raw,mask=encode_asset(np.array([0,0,1,1],dtype='<i8').tobytes(),kind='tensor',shape=[4],axes=['samples'],dtype='int64',metadata={'maskConvention':'blocked'})
    p=plan();p['roles']=[];p['inputs']=[dict(name='audio',nodeID='audio',representation='prepared',dtype='float32',sampleShape=[1,4],sampleAxes=['channels','samples'],columns=['audio']),dict(name='mask',nodeID='mask',representation='prepared',dtype='int64',sampleShape=[4],sampleAxes=['samples'],columns=['mask'])]
    schema=[dict(name=n,dtype='tensor',nullable=False) for n in ['audio','mask']]
    assets={audio['sha256']:audio_raw,mask['sha256']:mask_raw}
    x,_=convert_rows(p,schema,[[audio,mask]],media_reader=lambda ref:assets[ref['sha256']])
    def node(i,t,c):return dict(id=i,type=t,config=c,name='',x=0,y=0)
    configuration=normalize_attention('multihead_attention',dict(embed_dim=1,num_heads=1,bias=False,padding_mask_mode='supplied',return_weights=False))
    graph=dict(schemaVersion=2,name='Audio binding proof',namedInputs=dict(audio='audio',mask='mask'),nodes=[node('audio','input',dict(shape=[1,4])),node('mask','index_input',dict(shape=[4])),node('axes','transpose',dict(dim0=1,dim1=2)),node('attn','multihead_attention',configuration),node('out','output',{})],edges=[])
    for i,(a,b,port) in enumerate([('audio','axes','in'),('axes','attn','query'),('axes','attn','key'),('axes','attn','value'),('mask','attn','key_padding_mask'),('attn','out','in')]):graph['edges'].append(dict(id=str(i),source=a,sourcePort='out',target=b,targetPort=port))
    model=GraphModel(graph,bundled_dataset=False,architecture_only=True).eval()
    with torch.no_grad():
        model.layers['attn'].in_proj_weight.fill_(1);model.layers['attn'].out_proj.weight.fill_(1)
        actual=model(x)
    import math
    expected=torch.tensor([[[(math.exp(q)+2*math.exp(2*q))/(math.exp(q)+math.exp(2*q))] for q in [1,2,0,0]]])
    torch.testing.assert_close(actual,expected)
