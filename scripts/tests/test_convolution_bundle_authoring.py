import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('convolution_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source():
    rows=[]
    definitions=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components']
    for dimension in [1,2,3]:
        ident=f'conv{dimension}d'
        row=copy.deepcopy(json.loads((ROOT/'components/tensor-math/bundle.json').read_text())['components'][0])
        row['definition']=next(r for r in definitions if r['id']==ident)
        row['adapterID']='kwip.runner.graph-v2.'+ident;row['shapeRuleID']='kwip.shape.'+ident+'.v1'
        row['presentation'].update(theme='rose',icon=ident+'-v2',shortTitle=row['definition']['title'])
        contract={'version':{'major':1,'minor':0,'patch':0},'adapterID':row['adapterID'],'graphSchemaVersions':[2],'sampleRank':dimension+1,'preservesBatch':True,'graphDTypes':['float32'],'channelMinimum':1,'channelMaximum':512,'lengthMinimum':1,'lengthMaximum':4096,'kernelMinimum':1,'kernelMaximum':4096,'strideMinimum':1,'strideMaximum':4096,'dilationMinimum':1,'dilationMaximum':4096,'groupMinimum':1,'groupMaximum':512,'paddingMinimum':0,'paddingMaximum':4096,'maximumParameters':2000000,'maximumMultiplyAdds':128000000,'paddingPolicies':['explicit','valid','same'],'paddingModes':['zeros','reflect','replicate','circular'],'biasModes':[False,True]}
        if dimension>1:contract.update(spatialParameterLength=dimension,maximumSampleElements=100000000,maximumPaddedSampleElements=100000000)
        row['runtimeContract']=contract
        row['sources']={'native':f'Sources/KwipMLCore/Conv{dimension}dComponent.swift','worker':f'runner/conv{dimension}d_configuration.py','tests':f'runner/tests/test_conv{dimension}d_graph.py','evidence':f'docs/production/TORCH_CONV{dimension}D.md'}
        row['learning'].update(exampleID=f'prepared-conv{dimension}d-classifier',exampleSource='Sources/KwipMLCore/Conv'+str(dimension)+'dWorkflowExample.swift')
        rows.append(row)
    return {'schemaVersion':1,'id':'kwip.convolution','version':1,'backend':'pytorch','components':rows}

def test_all_three_convolutions_preserve_factories_and_declarations():
    value=source();bundles.validate(value,ROOT);s=bundles.render_swift(value,'convolution')
    assert 'BundledConvolutionComponents' in s
    for dim in [1,2,3]:
        assert f'BuiltInCapabilityBridge.conv{dim}dDeclaration' in s
        assert f'WorkflowExamples.preparedConv{dim}dClassifier(device: $0)' in s

@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('mutation',['field','spatialDefault','paddingChoice','rank','dtype','groups','bias','budget','schema','typedVersion','adapter','icon','theme','source','resource','example','missing'])
def test_exact_convolution_settings_and_provenance_drift_refused(dimension,mutation):
    v=source();r=v['components'][dimension-1];d=r['runtimeContract']
    if mutation=='field':r['definition']['fields'].pop()
    elif mutation=='spatialDefault':r['definition']['fields'][2]['defaultValue']=5
    elif mutation=='paddingChoice':r['definition']['fields'][7]['choices'].pop()
    elif mutation=='rank':d['sampleRank']+=1
    elif mutation=='dtype':d['graphDTypes']=['float64']
    elif mutation=='groups':d['groupMaximum']=513
    elif mutation=='bias':d['biasModes']=[True]
    elif mutation=='budget':d['maximumMultiplyAdds']+=1
    elif mutation=='schema':d['graphSchemaVersions']=[1,2]
    elif mutation=='typedVersion':d['version']['major']=True
    elif mutation=='adapter':r['adapterID']='kwip.runner.graph-v2.conv'
    elif mutation=='icon':r['presentation']['icon']='linear-v2'
    elif mutation=='theme':r['presentation']['theme']='blue'
    elif mutation=='source':r['sources']['worker']='../outside.py'
    elif mutation=='resource':r['learning']['sourceResource']='embedding-observations'
    elif mutation=='example':r['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='missing':v['components'].pop()
    with pytest.raises(ValueError):bundles.validate(v,ROOT)
