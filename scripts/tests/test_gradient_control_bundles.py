from scripts.tests.registry_history import historical_consolidation_copy
from scripts.tests.registry_history import historical_output_presentation,historical_pooling_output_prose,POOLING_HISTORY_PATH
"""Gradient controls are independently pinned recipe metadata, never tensor adapters."""
import ast
import copy
import hashlib
import importlib.util
import json
import pytest
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gradient_bundle_authoring', ROOT / 'scripts/component_bundles.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
SWIFT = 'Sources/KwipMLCore/BundledGradientControlComponents.generated.swift'
WORKER = 'runner/gradient_controls_generated.py'

def source(): return json.loads((ROOT / 'components/gradient_controls/bundle.json').read_text())

def unpublished_source():
    value=source()
    for row in value['components']:row['published']=False;row['learning']=None
    return value

def test_actual_valid_unpublished_controls_are_accepted_and_match_frozen_helpers():
    value = source()
    b.validate(value, ROOT)
    assert value['kind'] == 'trainingControl' and value['id'] == 'kwip.gradient-controls'
    assert [row['definition']['id'] for row in value['components']] == ['gradient_policy', 'trainability']
    for row in value['components']:
        ident = row['definition']['id']
        tree = ast.parse((ROOT / f'runner/{ident}.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'capability')
        assert row['runtimeContract'] == ast.literal_eval(next(n.value for n in function.body if isinstance(n, ast.Return)))
        assert row['published'] is True and row['learning'] is not None
        assert row['definition']['inputs'] == [] and row['definition']['outputs'] == ['out']
        assert len(json.dumps(row['runtimeContract'], separators=(',', ':')).encode()) <= 4096

def test_generation_is_additive_prior84_and_all_existing_output_bytes_are_exact():
    outputs = b.render(ROOT)
    assert SWIFT in outputs and WORKER in outputs
    prior_outputs={path:text for path,text in outputs.items() if path not in (SWIFT, WORKER,'Sources/KwipMLCore/BundledDataLoadingComponents.generated.swift','runner/data_loading_capabilities_generated.py','Sources/KwipMLCore/BundledProgramExportCapabilities.generated.swift','runner/program_export_capabilities_generated.py')}
    old_registry=(ROOT / 'Tests/historical/d13a57e/Sources/KwipMLCore/Resources/components.json').read_bytes().decode()
    current=json.loads(prior_outputs['Sources/KwipMLCore/Resources/components.json'])
    current['components']=[r for r in current['components'] if r['id'] not in ('rms_norm','rotary_position','mixture_experts','bag_input','embedding_bag','dataset_source','data_loader')]
    assert historical_output_presentation(current['components'][:84])==json.loads(old_registry)['components']
    assert [r['id'] for r in current['components'][84:]]==['gradient_policy','trainability']
    prior_outputs['Sources/KwipMLCore/Resources/components.json']=old_registry
    # Project the original pair and pre-Bag optional lessonIDs field for the frozen receipt.
    old_embedding=json.loads((ROOT / 'Tests/historical/d13a57e/components/embeddings/bundle.json').read_bytes())
    prior_outputs['Sources/KwipMLCore/BundledEmbeddingComponents.generated.swift']=b.render_swift(old_embedding,'embeddings').replace('        public let lessonIDs: [String]?\n', '')
    prior_outputs=historical_consolidation_copy(prior_outputs)
    prior_outputs[POOLING_HISTORY_PATH]=historical_pooling_output_prose(prior_outputs[POOLING_HISTORY_PATH])
    prior = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in prior_outputs.items()}
    assert len(prior) == 32
    assert hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == 'cc8195075ff466c4320f3efcd1b483d34688a1489ee85c718c304ce65b9162b2'
    assert len([row for row in json.loads(outputs['Sources/KwipMLCore/Resources/components.json'])['components'] if row['id'] not in ('rms_norm','rotary_position','mixture_experts')]) == 90
    for path, text in outputs.items(): assert (ROOT / path).read_text() == text, path

@pytest.mark.parametrize('path,replacement', [
    (('kind',), 'layer'), (('version',), True), (('backend',), 'portable'),
    (('components', 0, 'definition', 'inputs'), ['in']),
    (('components', 0, 'definition', 'outputs'), ['out', 'gradient']),
    (('components', 0, 'definition', 'fields', 0, 'max'), 1024),
    (('components', 0, 'definition', 'fields', 1, 'choices'), ['mean', 'sum']),
    (('components', 0, 'configurationVersion'), True),
    (('components', 0, 'editorExtension'), 'jsonEditor'),
    (('components', 1, 'editorExtension'), 'gradientRequestsV1'),
    (('components', 0, 'runtimeContract', 'derivativeOrder'), 2),
    (('components', 0, 'runtimeContract', 'requestMaximum'), True),
    (('components', 1, 'runtimeContract', 'emptyTrainingOwnership'), 'allow'),
    (('components', 0, 'modes'), ['requested-gradients']),
    (('components', 1, 'presentation', 'theme'), 'purple'),
    (('components', 0, 'presentation', 'icon'), 'run-setup-v2'),
    (('components', 0, 'sources', 'worker'), '/tmp/gradient_policy.py'),
    (('components', 1, 'sources', 'native'), '../../outside.swift'),
    (('components', 0, 'sources', 'tests'), 'runner/tests/nonexistent.py'),
    (('components', 0, 'published'), True),
    (('components', 1, 'learning'), {}),
])
def test_independent_metadata_mutations_refuse(path, replacement):
    value=unpublished_source(); b.validate(value, ROOT)
    target=value
    for key in path[:-1]: target=target[key]
    target[path[-1]]=replacement
    with pytest.raises(ValueError): b.validate(value, ROOT)

def test_order_unknown_keys_and_forward_adapter_claims_refuse():
    value=source(); value['components'].reverse()
    with pytest.raises(ValueError): b.validate(value, ROOT)
    value=source(); value['components'][0]['resolver']='tensor_forward'
    with pytest.raises(ValueError): b.validate(value, ROOT)
    value=source(); value['unexpected']=True
    with pytest.raises(ValueError): b.validate(value, ROOT)

def test_generated_table_has_only_exact_capabilities_and_no_execution_fiction():
    outputs=b.render(ROOT)
    namespace={}; exec(compile(outputs[WORKER], WORKER, 'exec'), namespace)
    assert namespace['CAPABILITIES']=={r['definition']['id']:r['runtimeContract'] for r in source()['components']}
    assert not {'NORMALIZERS','RESOLVERS','CONFIGURED_INPUTS','FORWARDS'}.intersection(namespace)
    assert '"gradient_policy": GradientControlCapabilityBridge.declaration' in outputs[SWIFT]
    assert '"trainability": GradientControlCapabilityBridge.declaration' in outputs[SWIFT]
    assert 'gradientRequestsV1' in outputs[SWIFT] and 'trainabilityOwnersV1' in outputs[SWIFT]

def test_original_artwork_and_registered_bytes_match_independent_provenance():
    provenance=json.loads((ROOT/'docs/design/gradient-control-icons-v1/provenance-20261009.json').read_text())
    expected={'gradient-policy-v1':'944f02d2fb730ccd28052bcd57b675656524ca4bef87f9872bb697052a095661','trainability-v1':'3e7ebe9a25da4d738a94a384e1220a0abe70d5f32158a0a17d25fd63810c6e2d'}
    for asset in provenance['assets']:
        name=asset['assetID']; original=(ROOT/asset['destination']).read_bytes()
        assert hashlib.sha256(original).hexdigest()==expected[name]==asset['sha256']
        assert (ROOT/f'Sources/KwipMLApp/Resources/ComponentIcons/{name}.png').read_bytes()==original

def test_choice_objects_match_actual_swift_required_title_and_value_schema():
    choices=source()['components'][0]['definition']['fields'][1]['choices']
    assert choices==[{'title':'Mean','value':'mean'},{'title':'Sum','value':'sum'}]
    assert all(set(choice)=={'title','value'} for choice in choices)
