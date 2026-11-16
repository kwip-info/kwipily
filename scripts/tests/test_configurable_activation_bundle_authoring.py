import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('configurable_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
LIMIT=3.4028234663852886e38
LESSONS={'silu':'assessment.silu-values','leaky_relu':'assessment.leaky-relu-slope','elu':'assessment.elu-alpha'}
def source():
    v=json.loads((ROOT/'components/activations/bundle.json').read_text());v['components']=[r for r in v['components'] if r['definition']['id'] not in LESSONS]
    for ident,title,key,default in [('silu','SiLU',None,None),('leaky_relu','LeakyReLU','negative_slope',0.01),('elu','ELU','alpha',1)]:
        r=copy.deepcopy(v['components'][0]);r['definition']={'id':ident,'title':title,'category':'Activations','inputs':['in'],'outputs':['out'],'fields':[] if key is None else [{'key':key,'title':'Negative slope' if key=='negative_slope' else 'Alpha','kind':'number','defaultValue':default,'min':-LIMIT,'max':LIMIT}]}
        r['adapterID']='kwip.runner.graph-v2.'+ident;r['shapeRuleID']='kwip.shape.'+ident+'.v1'
        r['presentation'].update(icon=ident+'-v2',shortTitle=title)
        r['runtimeContract']={'version':{'major':1,'minor':0,'patch':0},'adapterID':r['adapterID'],'graphSchemaVersions':[2],'graphDTypes':['float32'],'preservesBatch':True,'preservesShape':True,'configuration':{} if key is None else {key:{'default':default,'minimum':-LIMIT,'maximum':LIMIT}},'outputDTypes':{'out':'float32'},'maximumRank':8,'parameterless':True,'deterministic':True,'inplace':False}
        r['learning'].update(exampleID='prepared-'+ident+'-classifier',lessonIDs=[LESSONS[ident]])
        v['components'].append(r)
    return v

def test_three_new_activations_preserve_existing_family_and_distinct_factories():
    v=source();bundles.validate(v,ROOT);s=bundles.render_swift(v,'activations')
    assert 'public let lessonIDs: [String]?' in s
    for ident,suffix in [('silu','SiLU'),('leaky_relu','LeakyReLU'),('elu','ELU')]:
        assert f'BuiltInCapabilityBridge.{ident.replace("leaky_relu","leakyReLU")}Declaration' in s
        assert f'WorkflowExamples.prepared{suffix}Classifier(device: $0)' in s

@pytest.mark.parametrize('ident',['silu','leaky_relu','elu'])
@pytest.mark.parametrize('mutation',['field','bound','typedDefault','inplace','version','schema','dtype','rank','preserveShape','lesson','missingLesson','icon','extraConfiguration'])
def test_configurable_activation_contract_drift_refused(ident,mutation):
    v=source();r=next(row for row in v['components'] if row['definition']['id']==ident);c=r['runtimeContract']
    if mutation=='field':r['definition']['fields'].append({'key':'inplace'})
    elif mutation=='bound':c['configuration']['unreviewed']={'minimum':0}
    elif mutation=='typedDefault':c['version']['major']=True
    elif mutation=='inplace':c['inplace']=True
    elif mutation=='version':c['version']['minor']=1
    elif mutation=='schema':c['graphSchemaVersions']=[1,2]
    elif mutation=='dtype':c['graphDTypes']=['float64']
    elif mutation=='rank':c['maximumRank']=9
    elif mutation=='preserveShape':c['preservesShape']=False
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.other']
    elif mutation=='missingLesson':r['learning'].pop('lessonIDs')
    elif mutation=='icon':r['presentation']['icon']='relu-v2'
    elif mutation=='extraConfiguration':c['configuration']['inplace']={'default':False}
    with pytest.raises(ValueError):bundles.validate(v,ROOT)

@pytest.mark.parametrize('ident',['leaky_relu','elu'])
@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),3.5e38])
def test_numeric_fields_cannot_advertise_boolean_nonfinite_or_overflow_defaults(ident,value):
    v=source();next(row for row in v['components'] if row['definition']['id']==ident)['definition']['fields'][0]['defaultValue']=value
    with pytest.raises(ValueError):bundles.validate(v,ROOT)
