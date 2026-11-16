from scripts.tests.registry_history import historical_consolidation_copy
from scripts.tests.registry_history import historical_output_presentation,historical_pooling_output_prose,POOLING_HISTORY_PATH
"""Artifact lifecycle metadata must never become a tensor component or grant unverified support."""
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('lifecycle_authoring',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source():return json.loads((ROOT/'components/lifecycle/bundle.json').read_text())
def test_exact_artifact_identity_and_original_modes():
    value=source();assert value['kind']=='artifactLifecycle' and value['id']=='kwip.lifecycle'
    assert 'components' not in value
    entry=value['capabilities'][0]
    assert entry['id']=='checkpoint_lifecycle' and entry['coverageID']=='pytorch.torch.save'
    assert entry['modes']==['weights-save','continuation-save','weights-restore','continuation-restore','incompatible-artifact-refusal']
    assert 'definition' not in entry and 'shapeRuleID' not in entry
    bundles.validate(value,ROOT)
def test_all_prior_generated_outputs_are_byte_exact():
    outputs=bundles.render(ROOT)
    for name,value in outputs.items():assert (ROOT/name).read_text()==value,name
    # The later initialization family appends one reviewed control and two new outputs.
    # Reconstruct this test's original 83-definition snapshot; never refresh its frozen hash.
    prior={k:v for k,v in outputs.items() if k not in ('Sources/KwipMLCore/BundledLifecycleCapabilities.generated.swift','runner/lifecycle_capabilities_generated.py','Sources/KwipMLCore/BundledInitializationComponents.generated.swift','runner/initialization_adapters_generated.py','Sources/KwipMLCore/BundledGradientControlComponents.generated.swift','runner/gradient_controls_generated.py','Sources/KwipMLCore/BundledDataLoadingComponents.generated.swift','runner/data_loading_capabilities_generated.py','Sources/KwipMLCore/BundledProgramExportCapabilities.generated.swift','runner/program_export_capabilities_generated.py')}
    registry_path='Sources/KwipMLCore/Resources/components.json'
    registry=json.loads(prior[registry_path])
    registry['components']=[r for r in registry['components'] if r['id'] not in ('rms_norm','rotary_position','mixture_experts','gradient_policy','trainability',*('bag_input','embedding_bag','dataset_source','data_loader'))]
    # Project the original pair and pre-Bag optional lessonIDs field for the frozen receipt.
    old_embedding=json.loads((ROOT / 'Tests/historical/d13a57e/components/embeddings/bundle.json').read_bytes())
    prior['Sources/KwipMLCore/BundledEmbeddingComponents.generated.swift']=bundles.render_swift(old_embedding,'embeddings').replace('        public let lessonIDs: [String]?\n', '')
    initialization=json.loads((ROOT/'components/initialization/bundle.json').read_text())['components'][0]
    if initialization['published']:
        assert len(registry['components'])==84
        assert registry['components'].pop()==initialization['definition']
    registry['components']=historical_output_presentation(registry['components'])
    prior=historical_consolidation_copy(prior)
    prior[POOLING_HISTORY_PATH]=historical_pooling_output_prose(prior[POOLING_HISTORY_PATH])
    prior[registry_path]=json.dumps(registry,indent=2,ensure_ascii=False)+'\n'
    assert hashlib.sha256(json.dumps(prior,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='09835c9b17e39ff1df3ea6a3e3f47865df6b8688e9ffe1ecf25fd84c40300263'
    assert len(registry['components'])==83
def test_metadata_projection_has_no_forward_or_canvas_registration():
    value=source();text=bundles.render_lifecycle_swift(value);worker=bundles.render_lifecycle_capabilities(value)
    assert 'BundledLifecycleCapabilities' in text and 'ComponentDefinition' not in text
    assert 'CAPABILITIES' in worker and 'NORMALIZERS' not in worker and 'RESOLVERS' not in worker
    table={};exec(worker,table)
    assert table['CAPABILITIES']=={'checkpoint_lifecycle':value['capabilities'][0]['runtimeContract']}
@pytest.mark.parametrize('mutation',['kind','identity','unknown','definition','mode','action','training_mode','descriptor','bool_integer','source','missing_source','icon','learning','publication'])
def test_unreviewed_or_unsafe_metadata_is_rejected(mutation,tmp_path):
    value=source();r=value['capabilities'][0];bundles.validate(value,ROOT)
    if mutation=='kind':value['kind']='tensor'
    elif mutation=='identity':r['id']='linear'
    elif mutation=='unknown':value['plugin']='arbitrary'
    elif mutation=='definition':r['definition']={'inputs':['in'],'outputs':['out']}
    elif mutation=='mode':r['modes'].pop()
    elif mutation=='action':r['bindings'][0]['action']='executePython'
    elif mutation=='training_mode':r['bindings'][2]['trainingMode']='exactResume'
    elif mutation=='descriptor':r['runtimeContract']['externalImport']=True
    elif mutation=='bool_integer':r['runtimeContract']['checkpointSchemaVersions']=[True]
    elif mutation=='source':r['sources']['worker']='../escape.py'
    elif mutation=='missing_source':r['sources']['worker']='runner/does-not-exist.py'
    elif mutation=='icon':r['presentation']['icon']='docs/no-icon.png'
    elif mutation=='learning':r['published']=True;r['learning']=None
    else:r['published']=1
    with pytest.raises(ValueError):bundles.validate(value,ROOT)
def test_publication_requires_actual_learning_sources(tmp_path):
    value=source();r=value['capabilities'][0];r['published']=True
    r['learning']={'contentVersion':1,'updatedAt':'2026-10-09','lessonIDs':['assessment.checkpoint-restore-weights','assessment.checkpoint-continue-training'],'exampleIDs':['prepared-checkpoint-restore-weights','prepared-checkpoint-continue-training'],'sources':['Sources/KwipMLCore/NotWritten.swift'],'paragraphs':['Saved learned settings can be reused.']}
    with pytest.raises(ValueError):bundles.validate(value,ROOT)
def test_exact_runtime_literal_matches_frozen_contract():
    value=source()['capabilities'][0]['runtimeContract']
    contract=(ROOT/'docs/production/TORCH_SAVE_CLOSURE.md').read_text().split('The exact descriptor is:')[1].split('```json')[1].split('```')[0]
    assert json.dumps(value,sort_keys=True)==json.dumps(json.loads(contract),sort_keys=True)

def test_runtime_discovery_literal_matches_actual_loader():
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from checkpoint_lifecycle import capability
    assert source()['capabilities'][0]['runtimeContract']==capability()

def test_teaching_publication_does_not_control_actual_runtime_discovery():
    value=source();value['capabilities'][0]['published']=False
    table={};exec(bundles.render_lifecycle_capabilities(value),table)
    assert table['CAPABILITIES']['checkpoint_lifecycle']==source()['capabilities'][0]['runtimeContract']

def test_published_teaching_binds_real_checkpoint_exercises():
    value=source();row=value['capabilities'][0]
    assert row['published'] is True
    learning=row['learning']
    assert learning['lessonIDs']==['assessment.checkpoint-restore-weights','assessment.checkpoint-continue-training']
    assert learning['exampleIDs']==['prepared-checkpoint-restore-weights','prepared-checkpoint-continue-training']
    assert learning['contentVersion']==1 and learning['updatedAt']=='2026-10-09'
    for ref in learning['sources']:assert (ROOT/ref).is_file(),ref
    examples=(ROOT/'Sources/KwipMLApp/CheckpointLearningExamples.swift').read_text()
    for identity in learning['exampleIDs']:assert identity in examples
    bundles.validate(value,ROOT)

def test_other_example_identities_cannot_publish_as_checkpoint_worked_examples():
    value=source();row=value['capabilities'][0];row['published']=True
    row['learning']={'contentVersion':1,'updatedAt':'2026-10-09','lessonIDs':['assessment.checkpoint-restore-weights','assessment.checkpoint-continue-training'],'exampleIDs':['prepared-parallel-classifier','prepared-sgd-classifier'],'sources':['Sources/KwipMLCore/CheckpointLearningTask.swift'],'paragraphs':['Weights are learned settings.']}
    with pytest.raises(ValueError):bundles.validate(value,ROOT)
