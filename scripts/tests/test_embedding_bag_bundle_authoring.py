"""Strict authored bag pair, independent of runtime discovery and native acceptance."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('bag_embedding_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
# Reviewed authored proposal snapshot: tests do not depend on ignored .local files.
BAG_ROWS = [{'definition': {'id': 'bag_input',
                 'title': 'Bag input',
                 'category': 'Boundaries',
                 'inputs': [],
                 'outputs': ['indices', 'offsets', 'per_sample_weights'],
                 'fields': [{'key': 'layout',
                             'title': 'Group layout',
                             'kind': 'choice',
                             'defaultValue': 'fixed',
                             'choices': [{'value': 'fixed', 'title': 'Fixed rows'},
                                         {'value': 'packed', 'title': 'Packed groups'}]},
                            {'key': 'maximum_items_per_bag',
                             'title': 'Maximum items per group',
                             'kind': 'integer',
                             'defaultValue': 4,
                             'min': 1,
                             'max': 128},
                            {'key': 'per_sample_weights',
                             'title': 'Use item weights',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'include_last_offset',
                             'title': 'Include final boundary',
                             'kind': 'boolean',
                             'defaultValue': False}]},
  'published': True,
  'configurationVersion': 1,
  'adapterID': 'kwip.runner.graph-v2.bag_input',
  'shapeRuleID': 'kwip.shape.bag_input.v1',
  'presentation': {'theme': 'taupe',
                   'icon': 'bag-input-v1',
                   'shortTitle': 'Bag input',
                   'purpose': 'Supply whole observation groups with exact row selectors and optional real '
                              'item weights.'},
  'outputConditions': {},
  'sources': {'native': 'Sources/KwipMLCore/BagSourceConfiguration.swift',
              'worker': 'runner/bag_source.py',
              'tests': 'runner/tests/test_bag_source_semantics.py',
              'evidence': 'docs/production/TORCH_EMBEDDING_BAG_LEARNING_CANVAS_20261009.md'},
  'runtimeContract': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                      'adapterID': 'kwip.runner.graph-v2.bag_input',
                      'graphSchemaVersions': [2],
                      'packetSchemaVersion': 1,
                      'layouts': ['fixed', 'packed'],
                      'coordinateDTypes': {'indices': 'int64',
                                           'offsets': 'int64',
                                           'per_sample_weights': 'float32'},
                      'maximumItemsPerBag': 128,
                      'maximumBags': 4096,
                      'maximumItems': 1000000,
                      'maximumOutputElements': 2000000,
                      'integerDerivatives': 'unavailable',
                      'sourceWeights': 'realPairedInput',
                      'batching': 'wholeBags',
                      'emptyPackedBags': 'preserve'},
  'learning': {'contentVersion': 1,
               'updatedAt': '2026-10-09',
               'lessonIDs': ['assessment.bag-fixed-sum',
                             'assessment.bag-packed-mean',
                             'assessment.bag-packed-max',
                             'assessment.bag-fixed-weighted-sum',
                             'assessment.bag-packed-weighted-sum',
                             'assessment.bag-padding-norm'],
               'exampleID': 'prepared-embedding-bag-classifier',
               'sourceResource': 'bag-observations',
               'title': 'Combine categories for one observation',
               'subtitle': 'Turn a group of category numbers into one learned vector.',
               'disclosureTitle': 'From category groups to learned vectors',
               'formula': 'Select one table row for each category, then combine the selected rows for each '
                          'observation.',
               'paragraphs': ['A group belongs to one observation. Fixed rows have the same number of items; '
                              'packed groups use boundaries, called offsets, to show where each group '
                              'starts. Equal boundaries preserve empty groups.',
                              'Mean averages selected vectors, Sum adds them, and Maximum keeps the largest '
                              'value in each vector position. Real item weights work with Sum only.',
                              'Whole-number categories and boundaries have no derivatives. A gradient '
                              'describes how a small change affects the result; real item weights and '
                              'trainable table rows can have gradients.',
                              'Padding excludes a table row from the combination and its learning changes. '
                              'Maximum norm limits selected row lengths, including frozen rows. Use CPU for '
                              'maximum norm or trainable frequency scaling. Sparse gradients are unavailable '
                              'with the current Adam training recipe.'],
               'buttonTitle': 'Open grouped-category classifier',
               'exampleSource': 'Sources/KwipMLCore/EmbeddingBagWorkflowExample.swift'}},
 {'definition': {'id': 'embedding_bag',
                 'title': 'Embedding bag',
                 'category': 'Embeddings',
                 'inputs': ['indices', 'offsets', 'per_sample_weights'],
                 'outputs': ['out'],
                 'fields': [{'key': 'num_embeddings',
                             'title': 'Table rows',
                             'kind': 'integer',
                             'defaultValue': 16,
                             'min': 1,
                             'max': 100000000},
                            {'key': 'embedding_dim',
                             'title': 'Vector dimensions',
                             'kind': 'integer',
                             'defaultValue': 4,
                             'min': 1,
                             'max': 4096},
                            {'key': 'mode',
                             'title': 'Combine selected vectors',
                             'kind': 'choice',
                             'defaultValue': 'mean',
                             'choices': [{'value': 'mean', 'title': 'Mean'},
                                         {'value': 'sum', 'title': 'Sum'},
                                         {'value': 'max', 'title': 'Maximum'}]},
                            {'key': 'padding_idx',
                             'title': 'Padding row',
                             'kind': 'integer',
                             'defaultValue': None,
                             'min': -100000000,
                             'max': 99999999,
                             'allowsNull': True},
                            {'key': 'max_norm',
                             'title': 'Maximum norm',
                             'kind': 'number',
                             'defaultValue': None,
                             'min': 1.1754943508222875e-38,
                             'max': 3.4028234663852886e+38,
                             'allowsNull': True},
                            {'key': 'norm_type',
                             'title': 'Norm order',
                             'kind': 'number',
                             'defaultValue': 2,
                             'min': 1.1754943508222875e-38,
                             'max': 3.4028234663852886e+38},
                            {'key': 'scale_grad_by_freq',
                             'title': 'Scale gradients by frequency',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'sparse',
                             'title': 'Sparse gradients',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'include_last_offset',
                             'title': 'Include final boundary',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'freeze',
                             'title': 'Freeze learned weights',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'layout',
                             'title': 'Group layout',
                             'kind': 'choice',
                             'defaultValue': 'fixed',
                             'choices': [{'value': 'fixed', 'title': 'Fixed rows'},
                                         {'value': 'packed', 'title': 'Packed groups'}]},
                            {'key': 'per_sample_weights',
                             'title': 'Use item weights',
                             'kind': 'boolean',
                             'defaultValue': False}]},
  'published': True,
  'configurationVersion': 1,
  'adapterID': 'kwip.runner.graph-v2.embedding_bag',
  'shapeRuleID': 'kwip.shape.embedding_bag.v1',
  'presentation': {'theme': 'purple',
                   'icon': 'embedding-bag-v1',
                   'shortTitle': 'Embedding bag',
                   'purpose': 'Combine selected table vectors into one vector per observation group.'},
  'outputConditions': {},
  'sources': {'native': 'Sources/KwipMLCore/EmbeddingBagConfiguration.swift',
              'worker': 'runner/embedding_bag.py',
              'tests': 'runner/tests/test_embedding_bag.py',
              'evidence': 'docs/production/TORCH_EMBEDDING_BAG_LEARNING_CANVAS_20261009.md'},
  'runtimeContract': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                      'adapterID': 'kwip.runner.graph-v2.embedding_bag',
                      'graphSchemaVersions': [2],
                      'packetSchemaVersion': 1,
                      'layouts': ['fixed', 'packed'],
                      'coordinateDTypes': {'indices': 'int64',
                                           'offsets': 'int64',
                                           'per_sample_weights': 'float32'},
                      'maximumItemsPerBag': 128,
                      'maximumBags': 4096,
                      'maximumItems': 1000000,
                      'maximumOutputElements': 2000000,
                      'integerDerivatives': 'unavailable',
                      'sourceWeights': 'realPairedInput',
                      'batching': 'wholeBags',
                      'emptyPackedBags': 'preserve',
                      'modes': ['mean', 'sum', 'max'],
                      'weightedModes': ['sum'],
                      'outputDType': 'float32',
                      'maximumParameters': 2000000,
                      'maximumStaticTableElements': 100000000,
                      'paddingGradient': 'suppressed',
                      'frozenWeightsMayRenormalize': True,
                      'trainableSparseAdam': False,
                      'mpsMaxNormExecution': 'reject',
                      'mpsTrainableFrequencyScaling': 'reject',
                      'sourceWeightsDerivatives': 'sumOnly'},
  'learning': {'contentVersion': 1,
               'updatedAt': '2026-10-09',
               'lessonIDs': ['assessment.bag-fixed-sum',
                             'assessment.bag-packed-mean',
                             'assessment.bag-packed-max',
                             'assessment.bag-fixed-weighted-sum',
                             'assessment.bag-packed-weighted-sum',
                             'assessment.bag-padding-norm'],
               'exampleID': 'prepared-embedding-bag-classifier',
               'sourceResource': 'bag-observations',
               'title': 'Combine categories for one observation',
               'subtitle': 'Turn a group of category numbers into one learned vector.',
               'disclosureTitle': 'From category groups to learned vectors',
               'formula': 'Select one table row for each category, then combine the selected rows for each '
                          'observation.',
               'paragraphs': ['A group belongs to one observation. Fixed rows have the same number of items; '
                              'packed groups use boundaries, called offsets, to show where each group '
                              'starts. Equal boundaries preserve empty groups.',
                              'Mean averages selected vectors, Sum adds them, and Maximum keeps the largest '
                              'value in each vector position. Real item weights work with Sum only.',
                              'Whole-number categories and boundaries have no derivatives. A gradient '
                              'describes how a small change affects the result; real item weights and '
                              'trainable table rows can have gradients.',
                              'Padding excludes a table row from the combination and its learning changes. '
                              'Maximum norm limits selected row lengths, including frozen rows. Use CPU for '
                              'maximum norm or trainable frequency scaling. Sparse gradients are unavailable '
                              'with the current Adam training recipe.'],
               'buttonTitle': 'Open grouped-category classifier',
               'exampleSource': 'Sources/KwipMLCore/EmbeddingBagWorkflowExample.swift'}}]

def source():
 value=json.loads((ROOT/'components/embeddings/bundle.json').read_text())
 value['version']=2
 value['components']=value['components'][:2]+copy.deepcopy(BAG_ROWS)
 return value

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()

def test_exact_four_member_authoring_and_distinct_dispatch():
 value=source();bundles.validate(value,ROOT)
 generated=bundles.render_swift(value,'embeddings')
 assert generated==bundles.render_swift(value,'embeddings')
 assert generated.count('BuiltInCapabilityBridge.embeddingDeclaration')==2
 assert generated.count('EmbeddingBagCapabilityBridge.declaration')==2
 assert generated.count('WorkflowExamples.preparedEmbeddingBagClassifier(device: $0)')==2
 assert 'public let lessonIDs: [String]?' in generated
 assert [row['configurationVersion'] for row in value['components']]==[2,2,1,1]
 assert all(row['outputConditions']=={} for row in value['components'][2:])

def test_existing_runtime_rows_preserved_with_model_input_lesson_revision():
 value=json.loads((ROOT/'components/embeddings/bundle.json').read_text())
 rows=value['components'][:2]
 # Current inspector artwork/teaching changes are excluded explicitly; runtime stays frozen.
 import subprocess
 previous=json.loads((ROOT / 'Tests/historical/6ad54807/components/embeddings/bundle.json').read_bytes())['components'][:2]
 for row, old in zip(rows,previous):
  assert {k:v for k,v in row.items() if k not in ('learning','presentation')}=={k:v for k,v in old.items() if k not in ('learning','presentation')}
 rows=[dict(row,presentation=old['presentation']) for row,old in zip(rows,previous)]
 assert [hashlib.sha256(canonical({k:v for k,v in row.items() if k!='learning'})).hexdigest() for row in rows]==['91c82a2810518c242838166601a39a329e0ebf9b8ce5b01a800c93a26d9cd63a', '645fd87594286418b5b56eb6015cc897a37ab9e9856ef2c148a2039fa491b120']
 assert [row['learning']['contentVersion'] for row in rows]==[2,2]
 assert all(row['learning']['updatedAt']=='2026-10-13' for row in rows)
 assert all('Model definition' in str(row['learning']) for row in rows)

@pytest.mark.parametrize('mutation',['half-source','half-operator','duplicate','published','configuration-bool','configuration-two','field','extra-field','default','bounds','mode-choice','dtype','budget','fallback','weighted-modes','terminal-fiction','input-condition','icon','theme','source-path','missing-source','example','resource','lessons','date'])
def test_exact_bag_contract_drift_and_half_pair_refused(mutation):
 value=source();src=value['components'][2];row=value['components'][3]
 if mutation=='half-source':value['components'].remove(src)
 elif mutation=='half-operator':value['components'].remove(row)
 elif mutation=='duplicate':value['components'].append(copy.deepcopy(row))
 elif mutation=='published':src['published']=False
 elif mutation=='configuration-bool':row['configurationVersion']=True
 elif mutation=='configuration-two':row['configurationVersion']=2
 elif mutation=='field':src['definition']['fields'].pop()
 elif mutation=='extra-field':row['definition']['fields'].append(dict(key='fake',defaultValue=1))
 elif mutation=='default':row['definition']['fields'][0]['defaultValue']=17
 elif mutation=='bounds':src['definition']['fields'][1]['max']=129
 elif mutation=='mode-choice':row['definition']['fields'][2]['choices'].pop()
 elif mutation=='dtype':src['runtimeContract']['coordinateDTypes']['indices']='float32'
 elif mutation=='budget':row['runtimeContract']['maximumParameters']=100000000
 elif mutation=='fallback':row['runtimeContract']['mpsMaxNormExecution']='cpuFallback'
 elif mutation=='weighted-modes':row['runtimeContract']['weightedModes']=['mean','sum']
 elif mutation=='terminal-fiction':src['outputConditions']['offsets']='include_last_offset'
 elif mutation=='input-condition':row['inputConditions']={'offsets':'include_last_offset'}
 elif mutation=='icon':row['presentation']['icon']='embedding-v2'
 elif mutation=='theme':src['presentation']['theme']='purple'
 elif mutation=='source-path':row['sources']['worker']='../outside.py'
 elif mutation=='missing-source':src['sources']['native']='Sources/missing.swift'
 elif mutation=='example':row['learning']['exampleID']='prepared-embedding-classifier'
 elif mutation=='resource':src['learning']['sourceResource']='embedding-observations'
 elif mutation=='lessons':row['learning']['lessonIDs'].pop()
 else:row['learning']['updatedAt']='not-a-date'
 with pytest.raises(ValueError):bundles.validate(value,ROOT)

def test_independent_runtime_producers_and_approved_icon_copies_match_authored_pair():
 from runner.bag_source import capability as source_capability
 from runner.embedding_bag import capability as operator_capability
 expected=source()['components'][2:]
 assert source_capability()==expected[0]['runtimeContract']
 assert operator_capability()==expected[1]['runtimeContract']
 for row in expected:
  icon=row['presentation']['icon']+'.png'
  assert (ROOT/'docs/design/component-icons-v2'/icon).read_bytes()==(ROOT/'Sources/KwipMLApp/Resources/ComponentIcons'/icon).read_bytes()

def test_canonical_registry_and_generated_bag_dispatch_are_current():
 outputs=bundles.render(ROOT)
 registry=json.loads(outputs['Sources/KwipMLCore/Resources/components.json'])
 actual={r['id']:r for r in registry['components']}
 for row in source()['components']:
  assert actual[row['definition']['id']]==row['definition']
 generated=outputs['Sources/KwipMLCore/BundledEmbeddingComponents.generated.swift']
 assert generated.count('EmbeddingBagCapabilityBridge.declaration')==2
 assert (ROOT/'Sources/KwipMLCore/BundledEmbeddingComponents.generated.swift').read_text()==generated
 assert (ROOT/'Sources/KwipMLCore/Resources/components.json').read_text()==outputs['Sources/KwipMLCore/Resources/components.json']
