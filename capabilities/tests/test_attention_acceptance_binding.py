"""Attention credit binds every original mode to measured masks, ownership and frozen proof."""
import copy,hashlib,importlib.util,json,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
RECEIPT='docs/production/TORCH_ATTENTION_ACCEPTANCE_20261008.md'
MODES=['inference','training','input-gradients','state-roundtrip']
MEMBERS={
 'pytorch.torch.nn.MultiheadAttention':('multihead_attention','multihead-attention-connections'),
 'pytorch.torch.nn.Transformer':('transformer','transformer-source-target'),
 'pytorch.torch.nn.TransformerEncoder':('transformer_encoder','transformer-encoder-context'),
 'pytorch.torch.nn.TransformerDecoder':('transformer_decoder','transformer-decoder-memory'),
 'pytorch.torch.nn.functional.scaled_dot_product_attention':('scaled_dot_product_attention','scaled-attention-similarity')}
spec=importlib.util.spec_from_file_location('attention_binding_coverage',ROOT/'scripts/coverage_report.py');coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)
def catalog():return json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
def bundle():return json.loads((ROOT/'components/attention/bundle.json').read_text())
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
@pytest.fixture(params=MEMBERS)
def binding(request):
 c=catalog();unit=next(u for u in c['capabilities']if u['id']==request.param);return c,unit,request.param

def complete_binding(binding):
 c,u,key=binding
 # Actual planned catalog must RED here before a future receipt can imply credit.
 assert u['implementation_status']=='delivered'
 assert u['mode_scope_status']=='reviewed' and u['configuration_contract']=='verified'
 assert u['id']in coverage.report_catalog(c,ROOT)['complete_ids']
 return c,u,key

def changed_unit(c,u):
 changed=copy.deepcopy(c);return changed,next(x for x in changed['capabilities']if x['id']==u['id'])
def bound_rows(u,dim,path):
 rows=[r for r in u['evidence'].get(dim,[])if r['path']==path]
 assert {u['id']+':'+mode for mode in MODES}<={r.get('scope')for r in rows},(dim,path)
 assert rows and all(r['status']=='verified'and r['sha256']==digest(path)for r in rows),(dim,path)
 return rows

def test_original_five_identities_modes_backends_and_exact_bundle_contract(binding):
 c,u,key=complete_binding(binding);r=coverage.report_catalog(c,ROOT)
 assert len(c['capabilities'])==r['capability_count']==88 and r['complete_count']>=70 and r['minimum_complete_units']==80
 assert u['family']=='attention'and u['canonical_symbol']==key.removeprefix('pytorch.')and u['mode_scope']==MODES
 assert u['backend_matrix']['status']=='verified'and set(u['backend_matrix']['cases'])=={'cpu-float32','mps-float32'}and u['backend_matrix']['reference']==RECEIPT
 op,task=MEMBERS[key];entry=next(e for e in bundle()['components']if e['definition']['id']==op)
 assert u['lesson_ids']==['assessment.'+task]and u['lesson_id']==u['lesson_ids'][0]
 assert u['example_id']=='prepared-'+op.replace('_','-')+'-classifier'
 assert entry['learning']['lessonIDs']==u['lesson_ids']and entry['learning']['exampleID']==u['example_id']
 assert entry['presentation']['icon']==op+'-v2'and entry['presentation']['theme']=='rose'and entry['definition']['category']=='Layers'
 fields={f['key']:f for f in entry['definition']['fields']};contract=entry['runtimeContract']
 assert contract['graphSchemaVersions']==[2]and contract['graphDTypes']==['float32']and contract['preservesBatch']is True
 assert contract['maskDType']=='int64'and contract['maskConvention']=='oneBlockedZeroAllowed'and contract['maskBroadcast']=='perObservationAcrossHeadsOnly'
 assert contract['statePersistence']=='none'and contract['parameterOrder']=='nativeModuleOrder'and contract['maximumParameters']==2_000_000
 for bound in ['maximumTensorElements','maximumScoreElements','maximumMaskElements','maximumWorkElements']:assert contract[bound]==100_000_000
 assert contract['headMaximum']==32 and contract['sdpaGroupedQuery']is False and contract['dropoutPolicy']=='trainingOnly'
 if op=='multihead_attention':
  assert entry['definition']['inputs']==['query','key','value','attn_mask','key_padding_mask']and entry['definition']['outputs']==['out','weights']
  assert entry['inputConditions']=={'attn_mask':'attention_mask_mode=supplied','key_padding_mask':'padding_mask_mode=supplied'}and entry['outputConditions']=={'weights':'return_weights=true'}
  for name in ['kdim','vdim']:assert fields[name]['defaultValue']is None and fields[name]['allowsNull']is True and fields[name]['enabledValue']==4
  assert fields['embed_dim']['defaultValue']==4 and fields['num_heads']['defaultValue']==2 and fields['return_weights']['defaultValue']is True
 elif op=='scaled_dot_product_attention':
  assert entry['definition']['inputs']==['query','key','value','mask']and entry['definition']['outputs']==['out']and entry['inputConditions']=={'mask':'mask_mode=supplied'}
  assert contract['sdpaMPSDropoutExecution']=='deviceLocalPublicOps'
  assert fields['scale']['defaultValue']is None and fields['scale']['allowsNull']is True and fields['scale']['min']==1.401298464324817e-45
 else:
  assert contract['sdpaMPSDropoutExecution']=='notApplicable'and fields['d_model']['defaultValue']==4 and fields['nhead']['defaultValue']==2
  assert fields['layer_norm_eps']['min']==1.401298464324817e-45 and fields['dropout']['defaultValue']==0
  if op!='transformer':assert fields['final_norm']['defaultValue']is False

def test_frozen_acceptance_binds_all_ten_dimensions_and_original_modes(binding):
 _,u,_=complete_binding(binding)
 assert (ROOT/RECEIPT).is_file(),'Only root-frozen measured acceptance can grant credit'
 assert 'Frozen finite acceptance'in (ROOT/RECEIPT).read_text()
 assert len(coverage.REQUIRED_EVIDENCE)==10
 for dim in coverage.REQUIRED_EVIDENCE:bound_rows(u,dim,RECEIPT)

def test_critical_actual_source_mask_icon_lesson_and_checkpoint_staleness_withholds_credit(binding):
 c,u,key=complete_binding(binding);op=MEMBERS[key][0]
 dependencies=[('configuration','components/attention/bundle.json'),('configuration','Sources/KwipMLCore/AttentionConfiguration.swift'),('configuration','Sources/KwipMLCore/TypedConfiguration.swift'),('execution','runner/attention_execution.py'),('execution','runner/attention_adapters_generated.py'),('execution','runner/tests/test_attention_runtime.py'),('validation','Sources/KwipMLCore/GraphValidator.swift'),('validation','Tests/KwipMLCoreTests/AttentionGraphIntegrationTests.swift'),('lesson','Sources/KwipMLCore/AttentionLessonReference.swift'),('lesson','runner/learning_attention.py'),('lesson','Sources/KwipMLApp/LessonAudienceGuide.swift'),('lesson','Tests/KwipMLAppTests/AttentionLearningIntegrationTests.swift'),('example','Sources/KwipMLCore/AttentionWorkflowExample.swift'),('example','Tests/KwipMLCoreTests/AttentionCanonicalExampleTests.swift'),('diagnostics','Sources/KwipMLApp/WorkflowModelTrainingExecution.swift'),('diagnostics','Tests/KwipMLAppTests/AttentionTrainingPresentationTests.swift'),('ui','Sources/KwipMLApp/Resources/ComponentIcons/'+op+'-v2.png'),('ui','Sources/KwipMLApp/Resources/ComponentIcons/attention_mask-v2.png'),('ui','docs/design/component-icons-v2/attention-provenance-20261008.json'),('persistence','runner/checkpoint_lifecycle.py'),('persistence','Sources/KwipMLApp/WorkflowLifecycleVerifier.swift'),('failure_handling','Tests/KwipMLAppTests/AttentionDiscoveryTests.swift'),('failure_handling','Tests/KwipMLAppTests/AttentionExecutionDisclosureTests.swift'),('export','Sources/KwipMLApp/Resources/inference.py'),('export','docs/production/TORCH_ATTENTION_NATIVE_EXPORT_20261008.md')]
 for dim,path in dependencies:
  bound_rows(u,dim,path);changed,target=changed_unit(c,u);next(r for r in target['evidence'][dim]if r['path']==path)['sha256']='0'*64
  assert u['id']not in coverage.report_catalog(changed,ROOT)['complete_ids'],(dim,path)
 art=json.loads((ROOT/'docs/design/component-icons-v2/attention-provenance-20261008.json').read_text());assert {a['componentID']for a in art['assets']}=={member[0]for member in MEMBERS.values()}|{'attention_mask'}
 for component in [op,'attention_mask']:
  asset=next(a for a in art['assets']if a['componentID']==component);path='Sources/KwipMLApp/Resources/ComponentIcons/'+component+'-v2.png'
  assert asset['destination']==path and asset['sha256']==digest(path)and asset['tool']=='built-in image_gen'and asset['visuallyReviewed']is True
  assert (ROOT/path).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')

def test_missing_mode_dimension_and_unverified_evidence_never_earn_credit(binding):
 c,u,_=complete_binding(binding)
 for dim in coverage.REQUIRED_EVIDENCE:
  for mode in MODES:
   changed,target=changed_unit(c,u);target['evidence'][dim]=[r for r in target['evidence'][dim]if r.get('scope')!=u['id']+':'+mode]
   assert u['id']not in coverage.report_catalog(changed,ROOT)['complete_ids'],(dim,mode)
  for disposition in ['unverified','missing']:
   changed,target=changed_unit(c,u)
   if disposition=='missing':del target['evidence'][dim]
   else:
    for row in target['evidence'][dim]:row['status']='unverified'
   assert u['id']not in coverage.report_catalog(changed,ROOT)['complete_ids'],(dim,disposition)

def test_planned_configuration_pending_device_and_stale_receipt_cannot_earn_credit(binding):
 c,u,_=complete_binding(binding)
 for field,value in [('mode_scope_status','reviewed-proposal-pending-acceptance'),('configuration_contract','proposed-not-accepted'),('backend_matrix',{'status':'pending-acceptance','cases':['cpu-float32','mps-float32']})]:
  changed,target=changed_unit(c,u);target[field]=value;assert u['id']not in coverage.report_catalog(changed,ROOT)['complete_ids'],field
 for mode in MODES:
  changed,target=changed_unit(c,u);rows=[r for r in target['evidence']['execution']if r['path']==RECEIPT and r.get('scope')==u['id']+':'+mode];assert rows;rows[0]['sha256']='f'*64
  assert u['id']not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_runtime_canonical_prior_native_and_saved_export_are_separate_bound_proofs(binding):
 _,u,_=complete_binding(binding)
 names=[('execution','RUNTIME'),('execution','PRIOR_REPLAY'),('example','CANONICAL'),('ui','NATIVE'),('export','NATIVE_EXPORT')]
 for dim,name in names:bound_rows(u,dim,'docs/production/TORCH_ATTENTION_'+name+'_20261008.md')
 runtime=(ROOT/'docs/production/TORCH_ATTENTION_RUNTIME_20261008.md').read_text();canonical=(ROOT/'docs/production/TORCH_ATTENTION_CANONICAL_20261008.md').read_text();prior=(ROOT/'docs/production/TORCH_ATTENTION_PRIOR_REPLAY_20261008.md').read_text()
 assert '190 new attention checks'in runtime and 'scalar'in runtime.lower()and 'device-local'in runtime.lower()
 assert 'ten completed training-device cases'in canonical and 'never represented as saved native UI exports'in canonical
 assert '78 files'in prior and '65 accepted units'in prior
 assert len({digest('docs/production/TORCH_ATTENTION_'+name+'_20261008.md')for _,name in names})==5

def test_mask_utility_is_teachable_and_real_but_never_a_sixth_credited_unit():
 c=catalog();assert len(c['capabilities'])==88 and not any(u['canonical_symbol'].endswith('attention_mask')for u in c['capabilities'])
 utility=next(e for e in bundle()['components']if e['definition']['id']=='attention_mask');mha=next(e for e in bundle()['components']if e['definition']['id']=='multihead_attention')
 assert utility['learning']['lessonIDs']==mha['learning']['lessonIDs']==['assessment.multihead-attention-connections']
 assert utility['learning']['exampleID']=='prepared-attention-mask-classifier'and utility['definition']['outputs']==['mask','padding']
 assert utility['runtimeContract']['dropoutPolicy']=='notApplicable'and utility['runtimeContract']['statePersistence']=='none'
 assert utility['runtimeContract']['utilityMaximumItems']==4096 and utility['runtimeContract']['utilityCoordinateMaximum']==4095
 for field in utility['definition']['fields']:
  if field['kind']=='integerList':assert field['defaultValue']==[] and (field['minItems'],field['maxItems'],field['min'],field['max'])==(0,4096,0,4095)

def test_prior_65_recursive_non_sha_semantics_and_frozen_receipts_preserved():
 baseline=json.loads(subprocess.check_output(['git','show','a3d99c3:capabilities/catalogs/v1/pytorch.json'],cwd=ROOT));current=catalog()
 prior=[u for u in baseline['capabilities']if u.get('configuration_contract')=='verified'and u.get('mode_scope_status')=='reviewed'and u.get('backend_matrix',{}).get('status')=='verified'];assert len(prior)==65
 def without_sha(v):
  if isinstance(v,dict):return {k:without_sha(x)for k,x in v.items()if k!='sha256'}
  if isinstance(v,list):return [without_sha(x)for x in v]
  return v
 for old in prior:
  new=next(u for u in current['capabilities']if u['id']==old['id']);assert without_sha(old)==without_sha(new),old['id']
  for rows in old.get('evidence',{}).values():
   for row in rows:
    path=row['path']
    if path.startswith('docs/')and path!='docs/design/component-icons-v2/catalog.json':assert digest(path)==row['sha256'],path
