"""Parameterless activation authoring preserves canonical semantics and provenance."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('activation_bundle_authoring', ROOT / 'scripts/component_bundles.py')
bundles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundles)


def source():
    row = copy.deepcopy(json.loads((ROOT / 'components/pooling/bundle.json').read_text())['components'][0])
    row.update(definition={'id':'sigmoid','title':'Sigmoid','category':'Activations','inputs':['in'],'outputs':['out'],'fields':[]},
               published=False, configurationVersion=2, adapterID='kwip.runner.graph-v2.sigmoid',
               shapeRuleID='kwip.shape.sigmoid.v1', presentation={'theme':'amber','icon':'sigmoid-v2','shortTitle':'Sigmoid','purpose':'Map each value to a smooth score between zero and one.'},
               outputConditions={}, learning=None,
               runtimeContract={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.sigmoid','graphSchemaVersions':[2],'graphDTypes':['float32'],'preservesBatch':True,'preservesShape':True,'configuration':{},'outputDTypes':{'out':'float32'},'maximumRank':8,'parameterless':True,'deterministic':True})
    return {'schemaVersion':1,'id':'kwip.activations','version':1,'backend':'pytorch','components':[row]}


def test_activation_contract_accepts_unpublished_parameterless_definition():
    bundles.validate(source(), ROOT)


@pytest.mark.parametrize('mutation', ['field','ports','adapter','shape','icon','source','publication','duplicate','backend','extra'])
def test_activation_contract_refuses_semantic_or_provenance_drift(mutation):
    value = source(); row = value['components'][0]
    if mutation == 'field': row['definition']['fields'] = [{'key':'temperature','title':'Temperature','defaultValue':1}]
    elif mutation == 'ports': row['definition']['outputs'] += ['weights']
    elif mutation == 'adapter': row['adapterID'] = 'arbitrary.eval'
    elif mutation == 'shape': row['runtimeContract']['preservesShape'] = False
    elif mutation == 'icon': row['presentation']['icon'] = '../other'
    elif mutation == 'source': row['sources']['worker'] = '../outside.py'
    elif mutation == 'publication': row['published'] = True
    elif mutation == 'duplicate': value['components'].append(copy.deepcopy(row))
    elif mutation == 'backend': value['backend'] = 'tensorflow'
    elif mutation == 'extra': row['unexpected'] = True
    with pytest.raises(ValueError): bundles.validate(value, ROOT)


def test_family_generation_keeps_unpublished_activation_out_of_registry(tmp_path):
    pooling = json.loads((ROOT / 'components/pooling/bundle.json').read_text())
    activation = source()
    references = {'components/registry-base.json', 'Sources/KwipMLApp/Resources/ComponentIcons/sigmoid-v2.png'}
    for bundle in (pooling, activation):
        for row in bundle['components']:
            references.update(row['sources'].values())
            references.add('Sources/KwipMLApp/Resources/ComponentIcons/' + row['presentation']['icon'] + '.png')
            if row['learning']:
                references.add(row['learning']['exampleSource'])
                references.add('Sources/KwipMLApp/Resources/' + row['learning']['sourceResource'] + '.csv')
    for relative in references:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    for relative, value in [('components/pooling/bundle.json', pooling), ('components/activations/bundle.json', activation)]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    outputs = bundles.render(tmp_path)
    assert 'BundledActivationComponents' in outputs['Sources/KwipMLCore/BundledActivationComponents.generated.swift']
    registry = json.loads(outputs['Sources/KwipMLCore/Resources/components.json'])
    assert 'sigmoid' not in {row['id'] for row in registry['components']}
    assert outputs == bundles.render(tmp_path)
    # A base definition cannot shadow the bundle's stable identity, even unpublished.
    base = json.loads((tmp_path / 'components/registry-base.json').read_text())
    base['components'].append(activation['components'][0]['definition'])
    (tmp_path / 'components/registry-base.json').write_text(json.dumps(base))
    with pytest.raises(ValueError, match='Duplicate'):
        bundles.render(tmp_path)


@pytest.mark.parametrize('mutation', ['schemaBoolean', 'shapeInteger', 'versionFloat'])
def test_activation_contract_rejects_equal_but_wrong_json_types(mutation):
    value = source()
    if mutation == 'schemaBoolean': value['schemaVersion'] = True
    elif mutation == 'shapeInteger': value['components'][0]['runtimeContract']['preservesShape'] = 1
    elif mutation == 'versionFloat': value['components'][0]['runtimeContract']['version']['major'] = 1.0
    with pytest.raises(ValueError): bundles.validate(value, ROOT)


@pytest.mark.parametrize('mutation', [None, 'schema', 'version', 'adapter', 'gain', 'icon', 'duplicate'])
def test_tanh_bundle_preserves_reviewed_legacy_contract(mutation):
    value = json.loads((ROOT / 'components/activations/bundle.json').read_text())
    row = next(r for r in value['components'] if r['definition']['id'] == 'tanh')
    if mutation == 'schema': row['runtimeContract']['graphSchemaVersions'] = [2]
    elif mutation == 'version': row['runtimeContract']['version']['minor'] = 0
    elif mutation == 'adapter': row['adapterID'] = 'kwip.runner.graph-v1.tanh'
    elif mutation == 'gain': row['definition']['fields'] = [{'key': 'gain'}]
    elif mutation == 'icon': row['presentation']['icon'] = 'sigmoid-v2'
    elif mutation == 'duplicate': value['components'].append(copy.deepcopy(row))
    if mutation is None:
        bundles.validate(value, ROOT)
        rendered = bundles.render(ROOT)
        registry = json.loads(rendered['Sources/KwipMLCore/Resources/components.json'])
        assert [r['id'] for r in registry['components']].count('tanh') == 1
        assert 'tanhDeclaration' in rendered['Sources/KwipMLCore/BundledActivationComponents.generated.swift']
    else:
        with pytest.raises(ValueError): bundles.validate(value, ROOT)


@pytest.mark.parametrize('mutation', [None, 'schema', 'version', 'inplace', 'gain', 'icon'])
def test_relu_bundle_preserves_out_of_place_legacy_contract(mutation):
    value = json.loads((ROOT / 'components/activations/bundle.json').read_text())
    row = next(r for r in value['components'] if r['definition']['id'] == 'relu')
    if mutation == 'schema': row['runtimeContract']['graphSchemaVersions'] = [2]
    elif mutation == 'version': row['runtimeContract']['version']['minor'] = 0
    elif mutation == 'inplace': row['runtimeContract']['inplace'] = True
    elif mutation == 'gain': row['definition']['fields'] = [{'key': 'gain'}]
    elif mutation == 'icon': row['presentation']['icon'] = 'tanh-v2'
    if mutation is None:
        bundles.validate(value, ROOT)
        rendered = bundles.render(ROOT)
        registry = json.loads(rendered['Sources/KwipMLCore/Resources/components.json'])
        assert [r['id'] for r in registry['components']].count('relu') == 1
        assert 'reluDeclaration' in rendered['Sources/KwipMLCore/BundledActivationComponents.generated.swift']
    else:
        with pytest.raises(ValueError): bundles.validate(value, ROOT)


def gelu_source():
    value = source()
    row = value['components'][0]
    row.update(definition={'id':'gelu','title':'GELU','category':'Activations','inputs':['in'],'outputs':['out'],'fields':[{'key':'approximate_tanh','title':'Tanh approximation','kind':'boolean','defaultValue':False}]}, adapterID='kwip.runner.graph-v2.gelu', shapeRuleID='kwip.shape.gelu.v1', presentation={'theme':'amber','icon':'gelu-v2','shortTitle':'GELU','purpose':'Smoothly weight each value; select exact or tanh approximation.'}, runtimeContract={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.gelu','graphSchemaVersions':[2],'approximateTanhModes':[False,True]})
    return value


def test_gelu_authoring_preserves_both_modes_and_existing_boolean_control():
    value = gelu_source()
    bundles.validate(value, ROOT)
    rendered = bundles.render_swift(value, 'activations')
    assert 'geluDeclaration' not in rendered  # unpublished metadata cannot admit execution


@pytest.mark.parametrize('mutation', ['default','kind','extraField','schema','mode','numericModes','version','icon'])
def test_gelu_authoring_rejects_mode_and_configuration_drift(mutation):
    value = gelu_source(); row = value['components'][0]
    if mutation == 'default': row['definition']['fields'][0]['defaultValue'] = True
    elif mutation == 'kind': row['definition']['fields'][0]['kind'] = 'integer'
    elif mutation == 'extraField': row['definition']['fields'].append({'key':'gain'})
    elif mutation == 'schema': row['runtimeContract']['graphSchemaVersions'] = [1,2]
    elif mutation == 'mode': row['runtimeContract']['approximateTanhModes'] = [False]
    elif mutation == 'numericModes': row['runtimeContract']['approximateTanhModes'] = [0,1]
    elif mutation == 'version': row['runtimeContract']['version']['minor'] = 1
    elif mutation == 'icon': row['presentation']['icon'] = 'tanh-v2'
    with pytest.raises(ValueError): bundles.validate(value, ROOT)


def test_published_gelu_bundle_migrates_once_without_losing_mode_control():
    value = json.loads((ROOT / 'components/activations/bundle.json').read_text())
    row = next(r for r in value['components'] if r['definition']['id'] == 'gelu')
    assert row['published'] is True
    assert row['definition']['fields'] == gelu_source()['components'][0]['definition']['fields']
    bundles.validate(value, ROOT)
    rendered = bundles.render(ROOT)
    registry = json.loads(rendered['Sources/KwipMLCore/Resources/components.json'])
    assert [r['id'] for r in registry['components']].count('gelu') == 1
    assert 'geluDeclaration' in rendered['Sources/KwipMLCore/BundledActivationComponents.generated.swift']
    assert 'preparedGELUClassifier' in rendered['Sources/KwipMLCore/BundledActivationComponents.generated.swift']


def softmax_source():
    value = source(); row = value['components'][0]
    definition = {'id':'softmax','title':'Softmax','category':'Activations','inputs':['in'],'outputs':['out'],'fields':[{'key':'dim','title':'Normalization axis','kind':'integer','defaultValue':-1,'min':-8,'max':7}]}
    row.update(definition=definition, adapterID='kwip.runner.graph-v2.softmax', shapeRuleID='kwip.shape.softmax.v1', presentation={'theme':'amber','icon':'softmax-v2','shortTitle':'Softmax','purpose':'Normalize each chosen group into shares that sum to one.'}, runtimeContract={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.softmax','graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True})
    return value


def test_softmax_authoring_preserves_explicit_integer_axis_and_discovery():
    bundles.validate(softmax_source(), ROOT)


@pytest.mark.parametrize('mutation', ['axisDefault','axisBoolean','axisRange','axisTitle','extraField','batchFlag','rank','schema','version','icon'])
def test_softmax_authoring_rejects_axis_or_discovery_drift(mutation):
    value = softmax_source(); row = value['components'][0]
    if mutation == 'axisDefault': row['definition']['fields'][0]['defaultValue'] = 1
    elif mutation == 'axisBoolean': row['definition']['fields'][0]['defaultValue'] = True
    elif mutation == 'axisRange': row['definition']['fields'][0]['min'] = -9
    elif mutation == 'axisTitle': row['definition']['fields'][0]['title'] = 'Batch axis'
    elif mutation == 'extraField': row['definition']['fields'].append({'key':'temperature'})
    elif mutation == 'batchFlag': row['runtimeContract']['preservesBatch'] = 1
    elif mutation == 'rank': row['runtimeContract']['maximumRank'] = 7
    elif mutation == 'schema': row['runtimeContract']['graphSchemaVersions'] = [1,2]
    elif mutation == 'version': row['runtimeContract']['version']['minor'] = 1
    elif mutation == 'icon': row['presentation']['icon'] = 'gelu-v2'
    with pytest.raises(ValueError): bundles.validate(value, ROOT)
