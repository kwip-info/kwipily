import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('embedding_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source():
    rows=[]
    base=json.loads((ROOT/'components/registry-base.json').read_text())['components']
    if not any(r['id']=='embedding' for r in base):base=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components']
    for ident in ['embedding','index_input']:
        row=copy.deepcopy(json.loads((ROOT/'components/tensor-math/bundle.json').read_text())['components'][0])
        row['definition']=next(r for r in base if r['id']==ident)
        row['adapterID']='kwip.runner.graph-v2.'+ident;row['shapeRuleID']='kwip.shape.'+ident+'.v1'
        row['presentation'].update(theme='purple' if ident=='embedding' else 'taupe',icon=ident+'-v2',shortTitle=row['definition']['title'])
        row['runtimeContract']={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.embedding','graphSchemaVersions':[2],'inputDType':'int64','weightDType':'float32','maximumInputSampleRank':6,'maximumOutputRank':8,'maximumPreparedColumns':128,'maximumParameters':2000000,'preservesBatch':True,'outOfRangePolicy':'reject','paddingGradient':'suppressed','frequencyScaling':True,'frozenWeightsMayRenormalize':True,'trainingOptimizer':'Adam','trainableSparseGradients':False,'previewCacheSchemaForMaxNorm':2,'mpsMaxNormExecution':'cpuFallback'}
        row['sources']={'native':'Sources/KwipMLCore/EmbeddingComponent.swift','worker':'runner/embedding.py','tests':'runner/tests/test_embedding_contract.py','evidence':'docs/production/TORCH_EMBEDDING_CLOSURE.md'}
        row['learning'].update(exampleID='prepared-embedding-classifier',exampleSource='Sources/KwipMLCore/WorkflowExamples.swift',sourceResource='embedding-observations')
        rows.append(row)
    return {'schemaVersion':1,'id':'kwip.embeddings','version':1,'backend':'pytorch','components':rows}

def test_embedding_and_integer_boundary_preserve_factory_and_shared_descriptor():
    value=source();bundles.validate(value,ROOT);s=bundles.render_swift(value,'embeddings')
    assert 'BundledEmbeddingComponents' in s
    assert s.count('BuiltInCapabilityBridge.embeddingDeclaration')==2
    assert s.count('WorkflowExamples.preparedEmbeddingClassifier(device: $0)')==2

@pytest.mark.parametrize('mutation',['field','default','nullable','normBound','sourceShape','dtype','rank','sparse','fallback','frozen','cache','schema','typedVersion','adapter','icon','theme','source','resource','example','missing','extra'])
def test_embedding_settings_support_boundary_and_provenance_drift_refused(mutation):
    v=source();r=v['components'][0];d=r['runtimeContract']
    if mutation=='field':r['definition']['fields'].pop()
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=17
    elif mutation=='nullable':r['definition']['fields'][2]['allowsNull']=False
    elif mutation=='normBound':r['definition']['fields'][3]['min']=0
    elif mutation=='sourceShape':v['components'][1]['definition']['fields'][0]['maxItems']=6
    elif mutation=='dtype':d['inputDType']='float32'
    elif mutation=='rank':d['maximumInputSampleRank']=7
    elif mutation=='sparse':d['trainableSparseGradients']=True
    elif mutation=='fallback':d['mpsMaxNormExecution']='native'
    elif mutation=='frozen':d['frozenWeightsMayRenormalize']=False
    elif mutation=='cache':d['previewCacheSchemaForMaxNorm']=1
    elif mutation=='schema':d['graphSchemaVersions']=[1,2]
    elif mutation=='typedVersion':d['version']['major']=True
    elif mutation=='adapter':r['adapterID']='kwip.runner.graph-v2.lookup'
    elif mutation=='icon':r['presentation']['icon']='linear-v2'
    elif mutation=='theme':r['presentation']['theme']='rose'
    elif mutation=='source':r['sources']['worker']='../outside.py'
    elif mutation=='resource':r['learning']['sourceResource']='workflow-observations'
    elif mutation=='example':r['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='missing':v['components'].pop()
    elif mutation=='extra':v['components'].append(copy.deepcopy(r))
    with pytest.raises(ValueError):bundles.validate(v,ROOT)
