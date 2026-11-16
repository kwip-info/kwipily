import copy
import pytest
import torch
from runner.named_inputs import validate_plan, convert_rows
from runner import model_input


def plan():
    return {'schemaVersion':1,'inputs':[
        {'name':'measurements','nodeID':'numeric','representation':'numeric','dtype':'float32','sampleShape':[2],'columns':['a','b']},
        {'name':'tokens','nodeID':'words','representation':'tokens','dtype':'int64','sampleShape':[3],'columns':['tokens'],'lengthPolicy':'pad','paddingValue':0,'vocabularySize':10}],
        'roles':[{'name':'target','representation':'numeric','dtype':'int64','sampleShape':[],'columns':['label']}], 'maximumBatchBytes':1048576}


def test_named_numeric_token_batch_alignment_and_bounds():
    schema=[{'name':n,'dtype':t,'nullable':False} for n,t in [('a','float64'),('b','int64'),('tokens','int64Sequence'),('label','int64')]]
    x,y=convert_rows(plan(),schema,[[1.5,2,[2,3],1],[4.,5,[8,9,1],0]])
    assert list(x)==['measurements','tokens']
    assert torch.equal(x['tokens'],torch.tensor([[2,3,0],[8,9,1]]))
    assert x['measurements'].tolist()==[[1.5,2],[4.,5]]
    assert y['target'].tolist()==[1,0]
    assert model_input.observation_count(x)==2
    assert model_input.select_observations(x,[1])['tokens'].tolist()==[[8,9,1]]
    assert model_input.move_to_device(x,'cpu')['tokens'].dtype==torch.int64
    assert model_input.model_inputs_digest([x])!=model_input.model_inputs_digest([{**x,'tokens':x['tokens'].flip(0)}])


@pytest.mark.parametrize('mutation',[
    lambda p:p['inputs'].append(copy.deepcopy(p['inputs'][0])),
    lambda p:p['inputs'][0].update(sampleShape=[0]),
    lambda p:p['inputs'][1].update(paddingValue=10),
    lambda p:p['inputs'][0].update(representation='image'),
    lambda p:p.update(maximumBatchBytes=True)])
def test_invalid_or_unimplemented_contract_is_refused(mutation):
    p=plan();mutation(p)
    with pytest.raises(ValueError):validate_plan(p)


def test_bad_token_length_range_and_boolean_refused_without_silent_cast():
    p=plan();schema=[{'name':n,'dtype':t,'nullable':False} for n,t in [('a','float64'),('b','int64'),('tokens','int64Sequence'),('label','int64')]]
    for tokens in ([1,2,3,4],[10],[True]):
        with pytest.raises(ValueError):convert_rows(p,schema,[[1.,2,tokens,0]])
    p['maximumBatchBytes']=8
    with pytest.raises(ValueError):convert_rows(p,schema,[[1.,2,[1],0]])


def named_graph():
    from runner.engine import GraphModel, validate
    from runner.product_execution import ProductExecutionAdapter
    def node(i,t,c):return {'id':i,'type':t,'config':c,'name':'','x':0,'y':0}
    def edge(a,b,p='in'):return {'id':a+'-'+b,'source':a,'sourcePort':'out','target':b,'targetPort':p}
    graph={'schemaVersion':2,'name':'named','namedInputs':{'left':'x','right':'y'},'nodes':[node('x','input',{'features':2}),node('y','input',{'features':2}),node('l','linear',{'in_features':2,'out_features':2,'bias':True}),node('r','linear',{'in_features':2,'out_features':2,'bias':True}),node('sum','add',{}),node('out','output',{})],'edges':[edge('x','l'),edge('y','r'),edge('l','sum','a'),edge('r','sum','b'),edge('sum','out')]}
    return graph


def test_multiple_model_boundaries_execute_with_real_gradients():
    from runner.engine import GraphModel,validate
    from runner.product_execution import ProductExecutionAdapter
    graph=named_graph()
    validate(graph,bundled_dataset=False)
    model=GraphModel(graph,bundled_dataset=False)
    x={'left':torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True),'right':torch.ones(3,2,requires_grad=True)}
    out=ProductExecutionAdapter(model).forward(x)
    assert out.shape==(3,2)
    out.sum().backward()
    assert x['left'].grad is not None and x['right'].grad is not None
    with pytest.raises(ValueError):model({'left':x['left']})



def test_named_training_checkpoint_prediction_and_export(tmp_path):
    from runner.engine import run,GraphModel,export_source
    graph=named_graph()
    x={'left':torch.tensor([[0.,1.],[1.,0.],[0.,2.],[2.,0.]]),'right':torch.ones(4,2)}
    y=torch.tensor([0,1,0,1])
    provenance={'sourceHash':'a'*64,'splitHash':'b'*64,'trainingRowIDs':['a','b','c','d'],'validationRowIDs':['e','f','g','h'],'featureColumns':['a','b'],'targetColumn':'label'}
    result=run(graph,tmp_path/'run',epochs=1,device='cpu',batch_size=2,prepared=((x,y,x,y),provenance,{}))
    assert result['evaluationCount']==4
    import json
    capture=json.loads((tmp_path/'run'/'predictions.json').read_text())
    assert capture['schemaVersion']==4
    assert capture['predictionRows'][0]['input']=={'left':[0.,1.],'right':[1.,1.]}
    model=GraphModel(graph,bundled_dataset=False)
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    assert torch.allclose(exported(x),model(x))


def test_named_inputs_preserve_native_container_invocation_and_export():
    from runner.engine import GraphModel,export_source
    graph=named_graph()
    graph['compositionPlan']={'schemaVersion':1,'root':'root','containers':[{'id':'root','kind':'graph','path':[],'children':[{'kind':'node','id':n,'key':str(i)} for i,n in enumerate(['l','r','sum'])],'inputs':[{'id':'x','targets':[{'node':'l','port':'in'}]},{'id':'y','targets':[{'node':'r','port':'in'}]}],'outputs':[{'id':'out','source':{'node':'sum','port':'out'}}]}]}
    model=GraphModel(graph,bundled_dataset=False)
    calls=[];model.composition_root.register_forward_hook(lambda *args:calls.append('root'))
    x={'left':torch.randn(3,2,requires_grad=True),'right':torch.randn(3,2,requires_grad=True)}
    result=model(x);result.sum().backward()
    assert calls==['root']
    assert x['left'].grad is not None and x['right'].grad is not None
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    torch.testing.assert_close(exported(x),model(x))
    tampered=copy.deepcopy(graph);tampered['compositionPlan']['containers'][0]['inputs'][0]['targets']=[{'node':'r','port':'in'}]
    with pytest.raises(ValueError):GraphModel(tampered,bundled_dataset=False)


def test_named_worker_authority_and_nontraining_mode_refusal(tmp_path):
    import json,subprocess,sys
    from runner.worker_authorization import validate_phase_arguments
    p=plan();args=['--graph','model.json','--output','out','--prepared-dataflow','manifest.json','--named-input-plan-json',json.dumps(p)]
    assert validate_phase_arguments('training',args)['--named-input-plan-json']==json.dumps(p)
    with pytest.raises(ValueError):validate_phase_arguments('preparation',args)
    result=subprocess.run([sys.executable,'runner/main.py','--capabilities','--named-input-plan-json',json.dumps(p)],capture_output=True,text=True)
    assert result.returncode!=0 and 'prepared training only' in result.stdout


@pytest.mark.parametrize('policy,source,expected',[
    ('error',[1,2,3],[1,2,3]),('pad',[1],[1,0,0]),
    ('truncate',[1,2,3,4],[1,2,3]),('padTruncate',[],[0,0,0]),('padTruncate',[1,2,3,4],[1,2,3])])
def test_each_explicit_sequence_length_policy(policy,source,expected):
    p=plan();p['inputs'][1]['lengthPolicy']=policy
    schema=[{'name':n,'dtype':t,'nullable':False} for n,t in [('a','float64'),('b','int64'),('tokens','int64Sequence'),('label','int64')]]
    x,_=convert_rows(p,schema,[[1.,2,source,0]])
    assert x['tokens'].tolist()==[expected]


def test_conversion_has_no_source_alias_and_rejects_invalid_numeric_or_misaligned_sources():
    p=plan();schema=[{'name':n,'dtype':t,'nullable':False} for n,t in [('a','float64'),('b','int64'),('tokens','int64Sequence'),('label','int64')]]
    source=[[1.,2,[1,2,3],0]];x,_=convert_rows(p,schema,source);x['tokens'][0,0]=9
    assert source[0][2]==[1,2,3]
    for value in (None,True,float('nan'),float('inf'),1e100,1e-100):
        with pytest.raises(ValueError):convert_rows(p,schema,[[value,2,[1],0]])
    with pytest.raises(ValueError):model_input.observation_count({'a':torch.zeros(2,1),'b':torch.zeros(3,1)})


@pytest.mark.parametrize('device',['cpu','mps'])
def test_mixed_numeric_and_token_sources_have_real_forward_gradient_and_export(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    from runner.engine import GraphModel,export_source
    from runner.product_execution import ProductExecutionAdapter
    g=named_graph();g['nodes'][0].update(type='index_input',config={'shape':[3]})
    g['nodes'][2]['config']['in_features']=6
    g['nodes'].extend([{'id':'embed','type':'embedding','name':'','x':0,'y':0,'config':{'num_embeddings':10,'embedding_dim':2,'padding_idx':0,'max_norm':None,'norm_type':2,'scale_grad_by_freq':False,'sparse':False,'freeze':False}}, {'id':'flat','type':'flatten','name':'','x':0,'y':0,'config':{'start_dim':1,'end_dim':-1}}])
    g['edges'][0].update(target='embed')
    g['edges'].extend([{'id':'embed-flat','source':'embed','sourcePort':'out','target':'flat','targetPort':'in'},{'id':'flat-l','source':'flat','sourcePort':'out','target':'l','targetPort':'in'}])
    m=GraphModel(g,bundled_dataset=False).to(device)
    x={'left':torch.tensor([[1,2,0],[3,4,5]],device=device),'right':torch.ones(2,2,device=device,requires_grad=True)}
    out=ProductExecutionAdapter(m).forward(x);out.sum().backward()
    assert x['right'].grad is not None
    assert m.layers['embed'].weight.grad is not None
    namespace={};exec(export_source(m),namespace)
    exported=namespace['Model']().to(device);exported.load_state_dict(m.state_dict())
    torch.testing.assert_close(exported(x),m(x))
    with pytest.raises(ValueError):ProductExecutionAdapter(m).forward({**x,'left':x['left'].float()})
