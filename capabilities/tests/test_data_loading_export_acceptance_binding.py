"""Original three consumer units require sealed current-source evidence; prior84 stay immutable."""
from capabilities.tests.artwork_inventory_compatibility import historical_artwork_entries
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
BASELINE='ded5dac889882c0751a7b989bd23e6b07274f3e3'
CATALOG='capabilities/catalogs/v1/pytorch.json'
ACCEPTANCE='docs/production/TORCH_DATA_LOADING_EXPORT_ACCEPTANCE_20261009.md'
NATIVE='docs/production/TORCH_DATA_LOADING_EXPORT_NATIVE_20261009.md'
TARGETS={
 'pytorch.torch.utils.data.Dataset':('dataset_source',['indexed-source','bounded-iterable-source','tensor-source','selected-index-view'],'dataset-v1',['dataset-indexed-observations','dataset-aligned-tensors','dataset-selected-view','dataset-finite-stream']),
 'pytorch.torch.utils.data.DataLoader':('data_loader',['batch-order','shuffle-seed','short-final-batch','drop-last','collation','bounded-workers','continuation-order'],'data-loader-v1',['loader-complete-and-short-groups','loader-repeatable-mixing','loader-aligned-sequences','loader-continue-grouping']),
 'pytorch.torch.export.export':('program_export',['program-export','shape-constraints','program-reload-execution'],'program-export-v1',['program-export-computation','program-export-size-range']),
}
spec=importlib.util.spec_from_file_location('loading_export_coverage',ROOT/'scripts/coverage_report.py')
coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)
def read(path):return json.loads((ROOT/path).read_text())
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def baseline(path):return subprocess.check_output(['git','show',BASELINE+':'+path],cwd=ROOT)
def without_sha(value):
 if isinstance(value,dict):return {k:without_sha(v) for k,v in value.items() if k!='sha256'}
 if isinstance(value,list):return [without_sha(v) for v in value]
 return value

def source_row(ident):
 component=TARGETS[ident][0]
 if component=='program_export':return read('components/program-export/bundle.json')['capabilities'][0]
 return next(r for r in read('components/training-data/bundle.json')['components'] if r['definition']['id']==component)

COMMON={
 'configuration':['Sources/KwipMLCore/WorkflowCompiler.swift','Sources/KwipMLApp/WorkflowControlNodeContext.swift'],
 'execution':['runner/engine.py','runner/main.py'],
 'export':['Sources/KwipMLApp/WorkflowHistoryReader.swift','Sources/KwipMLApp/Resources/inference.py'],
 'validation':['Sources/KwipMLCore/WorkflowCodec.swift','runner/worker_authorization.py'],
 'diagnostics':['Sources/KwipMLApp/WorkflowPhaseArtifacts.swift'],
 'lesson':['Sources/KwipMLCore/TensorLearningTask.swift','runner/learning_evidence.py','docs/production/TORCH_DATA_PROGRAM_LEARN_INTEGRATION_20261009.md'],
 'example':['Tests/KwipMLAppTests/WorkflowDataLoadingIntegrationTests.swift'],
 'ui':['Sources/KwipMLApp/ComponentArtworkCatalog.swift','Sources/KwipMLApp/CapabilityActions.swift'],
 'persistence':['Sources/KwipMLApp/WorkflowArchiveContracts.swift','Sources/KwipMLApp/WorkflowExecutionSession.swift'],
 'failure_handling':['capabilities/tests/test_data_loading_export_acceptance_binding.py'],
}
SPECIFIC={
 'dataset_source':{'configuration':['Sources/KwipMLCore/DatasetConfiguration.swift','components/training-data/bundle.json'],'execution':['runner/dataset_sources.py','runner/tests/test_dataset_sources.py'],'persistence':['Sources/KwipMLApp/WorkflowLifecycleParentRecord.swift','Tests/KwipMLAppTests/WorkflowDataLoadingParentIdentityTests.swift'],'ui':['Sources/KwipMLApp/OptimizerRecipeInspectorView.swift'],'export':['runner/dataset_loader_portable.py','runner/tests/test_dataset_loader_portable.py']},
 'data_loader':{'configuration':['Sources/KwipMLCore/DataLoaderConfiguration.swift','components/training-data/bundle.json'],'execution':['runner/dataset_sources.py','runner/tests/test_dataset_loader_engine.py'],'persistence':['Sources/KwipMLApp/WorkflowLifecycleParentRecord.swift','Tests/KwipMLAppTests/WorkflowDataLoadingParentIdentityTests.swift'],'ui':['Sources/KwipMLApp/OptimizerRecipeInspectorView.swift'],'export':['runner/dataset_loader_portable.py','runner/tests/test_dataset_loader_portable.py']},
 'program_export':{'configuration':['Sources/KwipMLCore/ProgramExportConfiguration.swift','components/program-export/bundle.json'],'execution':['runner/program_export.py','runner/program_export_cli.py','runner/tests/test_program_export.py','runner/tests/test_program_export_operation.py'],'export':['Sources/KwipMLApp/WorkflowProgramExportManagedTransport.swift','Tests/KwipMLAppTests/WorkflowProgramExportIntegrationTests.swift'],'ui':['Sources/KwipMLApp/WorkflowHistoryView.swift','Sources/KwipMLApp/WorkflowHistoryStore.swift','Sources/KwipMLApp/WorkflowProgramExportView.swift'],'persistence':['Sources/KwipMLApp/WorkflowProgramExportPackage.swift','Tests/KwipMLAppTests/WorkflowProgramExportManagedTests.swift'],'failure_handling':['runner/tests/test_program_export_cli.py','Tests/KwipMLAppTests/WorkflowProgramExportTests.swift']},
}

def required(ident,dimension):
 component=TARGETS[ident][0]
 result=set(COMMON[dimension])|set(SPECIFIC[component].get(dimension,[]))|{ACCEPTANCE}
 if dimension in ['configuration','lesson','example','ui','persistence']:result.add(NATIVE)
 if component=='program_export' and dimension=='example':result.discard('Tests/KwipMLAppTests/WorkflowDataLoadingIntegrationTests.swift');result.add('Tests/KwipMLAppTests/WorkflowProgramExportIntegrationTests.swift')
 if dimension=='ui':
  asset=TARGETS[ident][2]+'.png'
  result|={'docs/design/data-loading-export-icons-v1/'+asset,'Sources/KwipMLApp/Resources/ComponentIcons/'+asset,'docs/design/data-loading-export-icons-v1/provenance-v1.json','docs/design/component-icons-v2/catalog.json'}
 return result

def delivered(ident,catalog=None):
 catalog=read(CATALOG) if catalog is None else catalog
 unit=next(u for u in catalog['capabilities'] if u['id']==ident)
 assert len(catalog['capabilities'])==88
 assert unit['implementation_status']=='delivered' and unit['configuration_contract']=='verified' and unit['mode_scope_status']=='reviewed'
 assert unit['mode_scope']==TARGETS[ident][1]
 assert unit['backend_matrix']['status']=='verified' and unit['backend_matrix']['reference']==ACCEPTANCE
 assert all(any(c.startswith(device) for c in unit['backend_matrix']['cases']) for device in ['cpu','mps'])
 assert ident in coverage.report_catalog(catalog,ROOT)['complete_ids']
 return catalog,unit

@pytest.mark.parametrize('ident',TARGETS)
def test_original_modes_ten_dimensions_and_actual_sealed_proofs(ident):
 _,unit=delivered(ident)
 for receipt in [ACCEPTANCE,NATIVE]:
  status=next(line for line in (ROOT/receipt).read_text().splitlines() if line.startswith('Status:'))
  assert 'accepted' in status.lower() and 'pending' not in status.lower()
 for dimension in coverage.REQUIRED_EVIDENCE:
  for path in required(ident,dimension):
   rows=[r for r in unit['evidence'][dimension] if r['path']==path]
   assert {ident+':'+m for m in TARGETS[ident][1]} <= {r.get('scope') for r in rows}
   assert all(r['status']=='verified' and r['sha256']==digest(path) for r in rows)

@pytest.mark.parametrize('ident',TARGETS)
def test_published_authored_lessons_examples_and_original_generated_art(ident):
 _,unit=delivered(ident);component,modes,icon,tasks=TARGETS[ident];row=source_row(ident)
 assert row['published'] is True and row['configurationVersion']==1 and row['modes']==modes
 assert unit['lesson_ids']==row['learning']['lessonIDs']==['assessment.'+t for t in tasks]
 assert unit['example_ids']==row['learning']['exampleIDs']==['prepared-'+t for t in tasks]
 provenance=read('docs/design/data-loading-export-icons-v1/provenance-v1.json')
 asset='docs/design/data-loading-export-icons-v1/'+icon+'.png'
 pin=next(r for r in provenance['assets'] if r['path']==asset)
 assert pin['generator']=='built-in image_gen' and pin['transparentBackground'] is True and pin['inspected'] is True
 assert digest(asset)==pin['sha256']==digest('Sources/KwipMLApp/Resources/ComponentIcons/'+icon+'.png')
 design=next(r for r in read('docs/design/component-icons-v2/catalog.json')['components'] if r['id']==component)
 assert design['asset']==icon+'.png'
 if component=='program_export':
  assert row['bindings']==[{'id':'program-export','surface':'history','action':'exportProgramPackage'}]
  assert component not in {r['id'] for r in read('Sources/KwipMLCore/Resources/components.json')['components']}
 else:
  registry={r['id']:r for r in read('Sources/KwipMLCore/Resources/components.json')['components']}
  assert registry[component]==row['definition']

@pytest.mark.parametrize('ident',TARGETS)
@pytest.mark.parametrize('dimension',coverage.REQUIRED_EVIDENCE)
def test_missing_mode_unverified_foreign_scope_stale_or_missing_proof_withholds_credit(ident,dimension):
 catalog,unit=delivered(ident)
 for mutation in ['missing-mode','unverified','foreign','stale','missing']:
  modified=copy.deepcopy(catalog);target=next(u for u in modified['capabilities'] if u['id']==ident);rows=target['evidence'][dimension]
  if mutation=='missing-mode':target['evidence'][dimension]=[r for r in rows if r.get('scope')!=ident+':'+TARGETS[ident][1][-1]]
  elif mutation=='unverified':
   for r in rows:r['status']='unverified'
  elif mutation=='foreign':
   for r in rows:r['scope']=ident+':foreign'
  elif mutation=='stale':rows[0]['sha256']='0'*64
  else:rows[0]['path']='.local/nonexistent-data-loading-export-proof'
  assert ident not in coverage.report_catalog(modified,ROOT)['complete_ids']

@pytest.mark.parametrize('ident',TARGETS)
@pytest.mark.parametrize('field',['implementation','configuration','review','backend','cpu','mps'])
def test_pending_status_and_missing_device_refuse_credit(ident,field):
 catalog,_=delivered(ident);modified=copy.deepcopy(catalog);target=next(u for u in modified['capabilities'] if u['id']==ident)
 if field=='implementation':target['implementation_status']='planned'
 elif field=='configuration':target['configuration_contract']='pending'
 elif field=='review':target['mode_scope_status']='pending-operation-boundary-review'
 elif field=='backend':target['backend_matrix']['status']='pending'
 else:target['backend_matrix']['cases']=[v for v in target['backend_matrix']['cases'] if not v.startswith(field)]
 with pytest.raises(AssertionError):delivered(ident,modified)

def test_original88_prior84_semantics_frozen_receipts_and_art_unchanged():
 old=json.loads(baseline(CATALOG));current=read(CATALOG)
 assert [u['id'] for u in current['capabilities']]==[u['id'] for u in old['capabilities']] and len(current['capabilities'])==88
 # The owner's later named release deferral is the only permitted top-level delta.
 expected_deferrals={'status':'owner-approved','reference':'docs/production/TORCH_V1_DEFERRAL_20261009.md','units':[{'id':'pytorch.torch.optim.LBFGS','reason':'Owner deferred the last optimizer to a later release to prioritize LEARN and release closure on 2026-10-09.'}],'maximum_units':8,'required':'Every incomplete reviewed unit must retain an explicit owner-approved rationale; required families and original denominator remain unchanged.'}
 assert current['deferrals']==expected_deferrals
 assert (ROOT/current['deferrals']['reference']).is_file()
 assert without_sha({k:v for k,v in current.items() if k not in {'capabilities','deferrals'}})==without_sha({k:v for k,v in old.items() if k not in {'capabilities','deferrals'}})
 prior=[u for u in old['capabilities'] if u['configuration_contract']=='verified' and u['mode_scope_status']=='reviewed' and u['backend_matrix']['status']=='verified'];assert len(prior)==84
 for unit in old['capabilities']:
  now=next(u for u in current['capabilities'] if u['id']==unit['id'])
  if unit['id'] not in TARGETS:assert without_sha(now)==without_sha(unit)
  else:
   for key in ['id','canonical_symbol','family','mode_scope','source_members','required_semantics','consolidation','scope_reference']:assert now.get(key)==unit.get(key)
 for unit in prior:
  for rows in unit['evidence'].values():
   for row in rows:
    if row['path'].startswith('docs/') and row['path']!='docs/design/component-icons-v2/catalog.json':assert digest(row['path'])==row['sha256']
 tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE,'docs/design','Sources/KwipMLApp/Resources/ComponentIcons'],cwd=ROOT,text=True).splitlines()
 for path in tracked:
  if path.endswith('.png') or 'provenance' in Path(path).name or 'RECEIPT' in Path(path).name:assert (ROOT/path).read_bytes()==baseline(path)
 oldart=json.loads(baseline('docs/design/component-icons-v2/catalog.json'))['components'];nowart=historical_artwork_entries(read('docs/design/component-icons-v2/catalog.json')['components'])
 assert nowart[:len(oldart)]==oldart
 oldregistry=json.loads(baseline('Sources/KwipMLCore/Resources/components.json'))['components'];nowregistry={r['id']:r for r in read('Sources/KwipMLCore/Resources/components.json')['components']}
 assert all(nowregistry[r['id']]==r for r in oldregistry)
