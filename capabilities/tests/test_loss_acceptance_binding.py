"""Eight objective capabilities require frozen, current evidence for all five modes."""
from capabilities.tests.artwork_inventory_compatibility import historical_artwork_entries
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
BASELINE='e9254e9'
RECEIPT='docs/production/TORCH_LOSS_ACCEPTANCE_20261009.md'
MODES=['training-objective','evaluation-objective','target-contract','reduction','input-gradients']
MEMBERS={
 'pytorch.torch.nn.CrossEntropyLoss':('loss_cross_entropy','cross-entropy-class-choice','prepared-cross-entropy-classifier'),
 'pytorch.torch.nn.BCEWithLogitsLoss':('loss_bce_with_logits','bce-independent-labels','prepared-bce-multilabel'),
 'pytorch.torch.nn.MSELoss':('loss_mse','mse-error-size','prepared-mse-regression'),
 'pytorch.torch.nn.L1Loss':('loss_l1','l1-absolute-error','prepared-l1-regression'),
 'pytorch.torch.nn.NLLLoss':('loss_nll','nll-log-probability-choice','prepared-nll-classifier'),
 'pytorch.torch.nn.KLDivLoss':('loss_kl_div','kl-distribution-comparison','prepared-kl-distribution'),
 'pytorch.torch.nn.CTCLoss':('loss_ctc','ctc-sequence-alignment','prepared-ctc-sequence'),
 'pytorch.torch.nn.TripletMarginLoss':('loss_triplet_margin','triplet-relative-distance','prepared-triplet-distance'),
}
spec=importlib.util.spec_from_file_location('loss_binding_coverage',ROOT/'scripts/coverage_report.py')
coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)


def catalog():return json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
def bundle():return json.loads((ROOT/'components/losses/bundle.json').read_text())
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


@pytest.fixture(params=MEMBERS)
def binding(request):
    c=catalog();return c,next(u for u in c['capabilities']if u['id']==request.param),request.param


def complete_binding(binding):
    c,u,key=binding
    assert u['implementation_status']=='delivered'
    assert u['configuration_contract']=='verified' and u['mode_scope_status']=='reviewed'
    assert u['backend_matrix']['status']=='verified'
    assert set(u['backend_matrix']['cases'])=={'cpu-float32','mps-float32'}
    assert u['backend_matrix']['reference']==RECEIPT
    assert key in coverage.report_catalog(c,ROOT)['complete_ids']
    return c,u,key


def bound_rows(unit,dimension,path):
    rows=[r for r in unit['evidence'].get(dimension,[])if r['path']==path]
    assert {unit['id']+':'+mode for mode in MODES}<={r.get('scope')for r in rows},(dimension,path)
    assert rows and all(r.get('status')=='verified'and r.get('sha256')==digest(path)for r in rows),(dimension,path)
    return rows


def changed_unit(c,u):
    changed=copy.deepcopy(c);return changed,next(x for x in changed['capabilities']if x['id']==u['id'])


def test_eight_original_identities_modes_and_control_recipe_bundles(binding):
    c,u,key=complete_binding(binding)
    assert len(c['capabilities'])==88
    assert u['canonical_symbol']==key.removeprefix('pytorch.') and u['family']=='loss'
    assert u['mode_scope']==MODES
    op,task,example=MEMBERS[key]
    entry=next(e for e in bundle()['components']if e['definition']['id']==op)
    assert entry['published'] is True and entry['configurationVersion']==1
    assert entry['definition']['inputs']==[] and entry['definition']['outputs']==['out']
    assert u['lesson_id']=='assessment.'+task and u['lesson_ids']==[u['lesson_id']]
    assert entry['learning']['lessonIDs']==u['lesson_ids'] and entry['learning']['exampleID']==u['example_id']==example
    assert (ROOT/('Sources/KwipMLApp/Resources/'+entry['learning']['sourceResource']+'.csv')).is_file()
    assert entry['presentation']['icon']==op+'-v2' and entry['presentation']['theme']=='green'
    contract=entry['runtimeContract']
    assert contract['adapterID']==entry['adapterID']=='kwip.runner.objective.'+op
    assert contract['recipeSchemaVersions']==[1] and contract['configurationVersion']==1
    assert contract['predictionDTypes']==['float32'] and set(contract['devices'])=={'cpu','mps'}
    assert contract['predictionRanks']==[2,8] and contract['sampleDimensionMaximum']==4096
    assert contract['fixedWeights']=='buffers' and contract['mappedWeights']=='preparedData' and contract['batchBroadcast'] is False
    assert contract['recipeMaximumBytes']==32768
    assert contract['maximumTensorElements']==contract['maximumWorkElements']==100_000_000
    assert contract['scalarizations']==['mean','sum']
    assert set(contract['reductions'])==({'none','mean','sum','batchmean'}if op=='loss_kl_div'else {'none','mean','sum'})
    assert contract['targetRoles']==(['target','weight','pos_weight']if op=='loss_bce_with_logits'else ['target','input_lengths','target_lengths']if op=='loss_ctc'else []if op=='loss_triplet_margin'else ['target'])
    # The generated worker advertises recipes, not fictitious tensor-forward operations.
    tree=ast.parse((ROOT/'runner/loss_adapters_generated.py').read_text())
    assignments={target.id:node.value for node in tree.body if isinstance(node,ast.Assign)for target in node.targets if isinstance(target,ast.Name)}
    assert ast.literal_eval(assignments['CAPABILITIES'])[op]==contract
    assert 'SUPPORTED'not in assignments


def test_acceptance_is_current_for_all_ten_dimensions_and_five_modes(binding):
    _,u,_=complete_binding(binding)
    assert len(coverage.REQUIRED_EVIDENCE)==10 and (ROOT/RECEIPT).is_file()
    for dimension in coverage.REQUIRED_EVIDENCE:bound_rows(u,dimension,RECEIPT)


def test_critical_runtime_target_lesson_export_and_art_sources_are_current(binding):
    c,u,key=complete_binding(binding);op=MEMBERS[key][0]
    required={
      'configuration':['components/losses/bundle.json','Sources/KwipMLCore/LossConfiguration.swift','Sources/KwipMLCore/TypedConfiguration.swift','scripts/component_bundles.py'],
      'execution':['runner/engine.py','runner/loss_configuration.py','runner/loss_objectives.py','runner/loss_prepared_data.py','runner/loss_adapters_generated.py','runner/tests/test_loss_engine.py','Sources/KwipMLCore/LossRuntimeCapabilities.swift','Sources/KwipMLApp/WorkflowExecutionSession.swift','Tests/KwipMLAppTests/LossRuntimeAdmissionTests.swift'],
      'validation':['Sources/KwipMLCore/LossTargetMapping.swift','Sources/KwipMLCore/PreparedObjectiveSelection.swift','Sources/KwipMLCore/WorkflowCompiler.swift','Sources/KwipMLCore/LossCapabilityBridge.swift','runner/loss_target_mapping.py','Tests/KwipMLCoreTests/LossWorkflowCompilerTests.swift'],
      'lesson':['Sources/KwipMLCore/LossLessonReference.swift','Sources/KwipMLCore/LossLearningSubmission.swift','Sources/KwipMLCore/TensorLearningReceipt.swift','Sources/KwipMLApp/TensorLearningStore.swift','Sources/KwipMLApp/TensorLearningCanvasAdapter.swift','runner/learning_losses.py','runner/tests/test_learning_losses.py','Tests/KwipMLAppTests/LossLearningObjectivePortTests.swift'],
      'example':['Sources/KwipMLCore/WorkflowLossExamples.swift','Sources/KwipMLCore/PreparedWorkflowExample.swift','Sources/KwipMLApp/LearnCatalogContent.swift','Tests/KwipMLCoreTests/LossGenericExampleRoutingTests.swift'],
      'diagnostics':['Sources/KwipMLApp/WorkflowObjectiveArtifacts.swift','Sources/KwipMLApp/ObjectiveResultSectionsView.swift','Sources/KwipMLApp/WorkflowObjectiveIdentity.swift','Tests/KwipMLAppTests/ObjectiveExplicitCaptureTests.swift'],
      'ui':['Sources/KwipMLApp/ComponentArtworkCatalog.swift','Sources/KwipMLApp/LossTargetMappingInspector.swift','Sources/KwipMLApp/Resources/ComponentIcons/'+op+'-v2.png','docs/design/component-icons-v2/loss-provenance-20261008.json'],
      'persistence':['runner/checkpoint_lifecycle.py','Sources/KwipMLApp/WorkflowArchiveContracts.swift','Sources/KwipMLApp/WorkflowHistoryReader.swift','Tests/KwipMLAppTests/LossHistoryReadTests.swift'],
      'export':['runner/loss_export.py','Sources/KwipMLApp/Resources/inference.py','Sources/KwipMLApp/TrainedModelInputContract.swift','Tests/KwipMLAppTests/ObjectiveSavedExportTests.swift','runner/tests/test_objective_portable_inference.py'],
      'failure_handling':['runner/loss_partition.py','runner/tests/test_loss_objectives.py','Tests/KwipMLAppTests/LossCompletedArtifactTests.swift'],
    }
    for dimension,paths in required.items():
        for path in paths:
            rows=bound_rows(u,dimension,path)
            changed,target=changed_unit(c,u)
            next(r for r in target['evidence'][dimension]if r['path']==rows[0]['path'])['sha256']='0'*64
            assert key not in coverage.report_catalog(changed,ROOT)['complete_ids'],(dimension,path)
    art=json.loads((ROOT/'docs/design/component-icons-v2/loss-provenance-20261008.json').read_text())
    assert {a['componentID']for a in art['assets']}=={member[0]for member in MEMBERS.values()}
    asset=next(a for a in art['assets']if a['componentID']==op)
    path='Sources/KwipMLApp/Resources/ComponentIcons/'+op+'-v2.png'
    assert asset['destination']==path and asset['sha256']==digest(path)
    assert asset['tool']=='built-in image_gen' and asset['visuallyReviewed'] is True
    assert (ROOT/path).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_missing_modes_dimensions_and_unverified_rows_withhold_credit(binding):
    c,u,key=complete_binding(binding)
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            changed,target=changed_unit(c,u)
            target['evidence'][dimension]=[r for r in target['evidence'][dimension]if r.get('scope')!=key+':'+mode]
            assert key not in coverage.report_catalog(changed,ROOT)['complete_ids'],(dimension,mode)
        for disposition in ['missing','unverified']:
            changed,target=changed_unit(c,u)
            if disposition=='missing':del target['evidence'][dimension]
            else:
                for row in target['evidence'][dimension]:row['status']='unverified'
            assert key not in coverage.report_catalog(changed,ROOT)['complete_ids'],(dimension,disposition)


def test_pending_configuration_backend_and_stale_acceptance_cannot_earn_credit(binding):
    c,u,key=complete_binding(binding)
    for field,value in [('configuration_contract','pending'),('mode_scope_status','pending-operation-boundary-review'),('backend_matrix',{'status':'pending','cases':['cpu-float32','mps-float32']}),('backend_matrix',{'status':'verified','cases':[]})]:
        changed,target=changed_unit(c,u);target[field]=value
        assert key not in coverage.report_catalog(changed,ROOT)['complete_ids'],field
    changed,target=changed_unit(c,u);target['backend_matrix']['cases']=['cpu-float32']
    with pytest.raises(AssertionError):complete_binding((changed,target,key))
    for mode in MODES:
        changed,target=changed_unit(c,u)
        row=next(r for r in target['evidence']['execution']if r['path']==RECEIPT and r.get('scope')==key+':'+mode)
        row['sha256']='f'*64
        assert key not in coverage.report_catalog(changed,ROOT)['complete_ids']


def test_runtime_native_history_and_saved_export_proofs_are_bound_separately(binding):
    _,u,_=complete_binding(binding)
    requirements={
      'execution':['docs/production/TORCH_LOSS_ENGINE_RUNTIME_20261008.md'],
      'example':['docs/production/TORCH_LOSS_EXAMPLES_PUBLICATION_20261008.md'],
      'lesson':['docs/production/TORCH_LOSS_LEARNING_20261008.md'],
      'ui':['docs/production/TORCH_LOSS_NATIVE_INTEGRATION_20261009.md'],
      'persistence':['docs/production/TORCH_LOSS_ARCHIVE_20261009.md','docs/production/TORCH_LOSS_HISTORY_PHASE_20261009.md'],
      'export':['docs/production/TORCH_LOSS_SAVED_EXPORT_20261009.md','docs/production/TORCH_LOSS_NATIVE_INTEGRATION_20261009.md'],
    }
    for dimension,paths in requirements.items():
        for path in paths:bound_rows(u,dimension,path)


def test_prior_70_recursive_non_sha_semantics_and_original_receipts_icons_preserved():
    baseline=json.loads(subprocess.check_output(['git','show',BASELINE+':capabilities/catalogs/v1/pytorch.json'],cwd=ROOT))
    current=catalog()
    prior=[u for u in baseline['capabilities']if u.get('configuration_contract')=='verified'and u.get('mode_scope_status')=='reviewed'and u.get('backend_matrix',{}).get('status')=='verified']
    assert len(prior)==70 and len(current['capabilities'])==88
    def without_sha(value):
        if isinstance(value,dict):return {k:without_sha(v)for k,v in value.items()if k!='sha256'}
        if isinstance(value,list):return [without_sha(v)for v in value]
        return value
    for old in prior:
        new=next(u for u in current['capabilities']if u['id']==old['id'])
        assert without_sha(new)==without_sha(old),old['id']
        for rows in old.get('evidence',{}).values():
            for row in rows:
                path=row['path']
                if path.startswith('docs/')and path!='docs/design/component-icons-v2/catalog.json':assert digest(path)==row['sha256'],path
    # Preserve all original v1/v2 assets and original design receipts, not only icons
    # happened to be referenced by an individual acceptance row.
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE,'Sources/KwipMLApp/Resources/ComponentIcons','docs/design/component-icons-v2'],cwd=ROOT,text=True).splitlines()
    for path in tracked:
        original_icon=path.startswith('Sources/KwipMLApp/Resources/ComponentIcons/')and path.endswith('.png')
        frozen_design=path.startswith('docs/design/component-icons-v2/')and ('RECEIPT' in Path(path).name or 'provenance' in Path(path).name)
        if original_icon or frozen_design:
            original=subprocess.check_output(['git','show',BASELINE+':'+path],cwd=ROOT)
            assert hashlib.sha256(original).hexdigest()==digest(path),path
    old_art=json.loads(subprocess.check_output(['git','show',BASELINE+':docs/design/component-icons-v2/catalog.json'],cwd=ROOT))
    new_art=json.loads((ROOT/'docs/design/component-icons-v2/catalog.json').read_text())
    assert historical_artwork_entries(new_art['components'])[:len(old_art['components'])]==old_art['components']
