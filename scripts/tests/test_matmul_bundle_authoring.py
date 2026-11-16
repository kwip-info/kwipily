import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('matrix_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)

def source():
    row=copy.deepcopy(json.loads((ROOT/'components/layers/bundle.json').read_text())['components'][0])
    row['definition']={'id':'matmul','title':'Matrix multiply','category':'Tensor math','inputs':['a','b'],'outputs':['out'],'fields':[]}
    row['adapterID']='kwip.runner.graph-v2.matrixMultiply';row['shapeRuleID']='kwip.shape.matmul.v1'
    row['presentation']={'theme':'blue','icon':'matmul-v2','shortTitle':'Matrix multiply','purpose':'Combine rows from one array with columns from another.'}
    row['runtimeContract']={'version':{'major':1,'minor':0,'patch':0},'adapterID':row['adapterID'],'graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True,'minimumSampleRank':2,'sampleBroadcasting':True,'maximumOutputElements':100000000,'graphDTypes':['float32']}
    row['sources']={'native':'Sources/KwipMLCore/MatrixMultiplyComponent.swift','worker':'runner/matrix_multiply.py','tests':'runner/tests/test_matrix_graph.py','evidence':'docs/production/TORCH_MATMUL_CLOSURE.md'}
    row['learning'].update(exampleID='prepared-matrix-classifier',exampleSource='Sources/KwipMLCore/MatrixExample.swift')
    return {'schemaVersion':1,'id':'kwip.tensor-math','version':1,'backend':'pytorch','components':[row]}

def test_parameterless_matrix_bundle_preserves_actual_factory_and_bridge():
    value=source();bundles.validate(value,ROOT);rendered=bundles.render_swift(value,'tensor-math')
    assert 'BundledTensorMathComponents' in rendered
    assert 'BuiltInCapabilityBridge.matrixDeclaration' in rendered
    assert 'WorkflowExamples.preparedMatrixClassifier(device: $0)' in rendered

@pytest.mark.parametrize('mutation',['field','port','rank','sampleRank','broadcast','batch','budget','schema','dtype','typedVersion','adapter','icon','theme','source','example','date','resource','extra'])
def test_matrix_contract_or_provenance_drift_refused(mutation):
    value=source();row=value['components'][0]
    if mutation=='field':row['definition']['fields']=[{'key':'dim'}]
    elif mutation=='port':row['definition']['inputs']=['in']
    elif mutation=='rank':row['runtimeContract']['maximumRank']=9
    elif mutation=='sampleRank':row['runtimeContract']['minimumSampleRank']=1
    elif mutation=='broadcast':row['runtimeContract']['sampleBroadcasting']=False
    elif mutation=='batch':row['runtimeContract']['preservesBatch']=False
    elif mutation=='budget':row['runtimeContract']['maximumOutputElements']=100000001
    elif mutation=='schema':row['runtimeContract']['graphSchemaVersions']=[1,2]
    elif mutation=='dtype':row['runtimeContract']['graphDTypes']=['float64']
    elif mutation=='typedVersion':row['runtimeContract']['version']['major']=True
    elif mutation=='adapter':row['adapterID']='kwip.runner.graph-v2.matmul'
    elif mutation=='icon':row['presentation']['icon']='linear-v2'
    elif mutation=='theme':row['presentation']['theme']='rose'
    elif mutation=='source':row['sources']['worker']='../outside.py'
    elif mutation=='example':row['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='date':row['learning']['updatedAt']='tomorrow'
    elif mutation=='resource':row['learning']['sourceResource']='nonexistent'
    elif mutation=='extra':value['components'].append(copy.deepcopy(row))
    with pytest.raises(ValueError):bundles.validate(value,ROOT)
