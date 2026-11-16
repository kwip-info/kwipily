import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('transpose_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source():
    v=json.loads((ROOT/'components/convolution/bundle.json').read_text());v['components']=[r for r in v['components'] if r['definition']['id'] in ['conv1d','conv2d','conv3d']]
    registry=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components']
    for dim in [1,2,3]:
        ident=f'conv_transpose{dim}d';r=copy.deepcopy(v['components'][dim-1]);r['definition']=next(r for r in registry if r['id']==ident)
        r['adapterID']='kwip.runner.graph-v2.'+ident;r['shapeRuleID']='kwip.shape.'+ident+'.v1'
        r['presentation'].update(icon=ident+'-v2',shortTitle=r['definition']['title'])
        c=r['runtimeContract'];c['adapterID']=r['adapterID'];c.pop('paddingPolicies');c.pop('paddingModes');c.pop('maximumPaddedSampleElements',None)
        c.update(spatialParameterKind=['scalar','pair','triple'][dim-1],maximumSampleElements=100000000,maximumUncroppedSampleElements=100000000,outputPaddingMinimum=0,outputPaddingMaximum=4096,configurationKeys=['in_channels','out_channels','kernel_size','stride','padding','output_padding','dilation','groups','bias'],paddingMode='zeros',outputPaddingRule='lessThanStrideOrDilation',supportsDynamicOutputSize=False)
        r['sources']={'native':f'Sources/KwipMLCore/ConvTranspose{dim}dComponent.swift','worker':f'runner/conv_transpose{dim}d_configuration.py','tests':f'runner/tests/test_conv_transpose{dim}d_graph.py','evidence':f'docs/production/TORCH_CONV_TRANSPOSE{dim}D.md'}
        r['learning'].update(exampleID='prepared-'+ident.replace('conv_transpose','conv-transpose')+'-classifier',exampleSource=f'Sources/KwipMLCore/ConvTranspose{dim}dWorkflowExample.swift')
        v['components'].append(r)
    return v

def test_existing_convolutions_and_all_transposed_factories_share_exact_family():
    v=source();bundles.validate(v,ROOT);s=bundles.render_swift(v,'convolution')
    for dim in [1,2,3]:
        assert f'BuiltInCapabilityBridge.convTranspose{dim}dDeclaration' in s
        assert f'WorkflowExamples.preparedConvTranspose{dim}dClassifier(device: $0)' in s
    bundles.validate({**v,'components':v['components'][:3]},ROOT)

@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('mutation',['field','outputDefault','outputRule','dynamic','paddingMode','kind','rank','budget','dtype','typedVersion','adapter','icon','source','example','partial'])
def test_transposed_contract_drift_or_partial_family_refused(dimension,mutation):
    v=source();r=v['components'][dimension+2];c=r['runtimeContract']
    if mutation=='field':r['definition']['fields'].append({'key':'output_size'})
    elif mutation=='outputDefault':r['definition']['fields'][5]['defaultValue']=1
    elif mutation=='outputRule':c['outputPaddingRule']='lessThanStrideAndDilation'
    elif mutation=='dynamic':c['supportsDynamicOutputSize']=True
    elif mutation=='paddingMode':c['paddingMode']='reflect'
    elif mutation=='kind':c['spatialParameterKind']='list'
    elif mutation=='rank':c['sampleRank']+=1
    elif mutation=='budget':c['maximumUncroppedSampleElements']+=1
    elif mutation=='dtype':c['graphDTypes']=['float64']
    elif mutation=='typedVersion':c['version']['major']=True
    elif mutation=='adapter':r['adapterID']='kwip.runner.graph-v2.convtranspose'
    elif mutation=='icon':r['presentation']['icon']='conv1d-v2'
    elif mutation=='source':r['sources']['worker']='../outside.py'
    elif mutation=='example':r['learning']['exampleID']='prepared-conv1d-classifier'
    elif mutation=='partial':v['components'].pop()
    with pytest.raises(ValueError):bundles.validate(v,ROOT)
