"""Model tasks inherit the retained worker's identity, file pins and isolation."""
import json
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest

@pytest.fixture
def model_operation(operation_authority):
    f=operation_authority;root,value=f[:2]
    for key in ('runID','directoryID','phase'):value.pop(key,None)
    source=root/'source';source.mkdir(mode=0o700)
    for name in ['config.json','tokenizer.json','model.safetensors']:(source/name).write_bytes(b'{}')
    inputs=root/'input';inputs.mkdir(mode=0o700)
    inner=dict(schemaVersion=1,projectID=value['projectID'],operationID=value['operationID'],requestIdentity=value['requestIdentity'],action='inspect',settings={},sourceIdentity='a'*64)
    (inputs/'model-request.json').write_bytes(encoded(inner))
    value.update(schemaVersion=2,toolMode='modelAsset',arguments=['--model-asset',str(inputs/'model-request.json')])
    value['inputFiles']=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in sorted(root.rglob('*')) if p.is_file() and (p.parent==inputs or p.parent==source)]
    publish(f);return f

def test_actual_model_authority_rechecks_changed_source(model_operation):
    f=model_operation
    with claim(f) as authority:
        authority.check();(f[0]/'source/config.json').write_bytes(b'changed')
        with pytest.raises(ValueError):authority.check()

@pytest.mark.parametrize('change',['code','pickle','traversal','wrong-owner','registry','duplicate','oversized','bool-size'])
def test_unreviewed_model_authority_never_claims(model_operation,change):
    f=model_operation;v=f[1]
    if change in ('code','pickle','traversal'):
        v['inputFiles'].append(dict(path={'code':'source/model.py','pickle':'source/pytorch_model.bin','traversal':'source/../secret'}[change],bytes=2,sha256=digest(b'{}')))
    elif change=='wrong-owner':v['arguments'][1]='/tmp/input/model-request.json'
    elif change=='registry':v['arguments']+=['--registry','/tmp/components.json']
    elif change=='duplicate':v['inputFiles'].append(v['inputFiles'][0])
    elif change=='oversized':v['inputFiles'][0]['bytes']=2_100_000_000
    else:v['inputFiles'][0]['bytes']=True
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


def expanded_request(f,action,extras,settings=None):
    root,value=f[:2];path=root/'input/model-request.json'
    inner=json.loads(path.read_text());inner.update(schemaVersion=3,action=action,settings=settings or {})
    path.write_bytes(encoded(inner))
    for name in extras:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'{}')
    value['inputFiles']=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in sorted(root.rglob('*')) if p.is_file() and (str(p.relative_to(root)).startswith('source/') or str(p.relative_to(root)).startswith('input/'))]
    return root,value


def test_segmented_capture_has_explicit_version_and_rechecks_pack_pin(model_operation):
    from runner.model_asset_operation_contract import verify_inputs
    pack='input/table-pack-'+'a'*32+'.kwipdata'
    root,value=expanded_request(model_operation,'prepare',['input/training.table.json','input/validation.table.json',pack])
    path=root/'input/model-request.json';inner=json.loads(path.read_bytes());inner['schemaVersion']=4
    path.write_bytes(encoded(inner))
    entry=next(f for f in value['inputFiles'] if f['path']=='input/model-request.json')
    entry.update(bytes=path.stat().st_size,sha256=digest(path.read_bytes()))
    assert verify_inputs(value,root,{})['schemaVersion']==4
    (root/pack).write_bytes(b'changed')
    with pytest.raises(ValueError):verify_inputs(value,root,{})


@pytest.mark.parametrize('version,action,pack',[
    (1,'train','input/table-pack-'+'a'*32+'.kwipdata'),
    (2,'train','input/table-pack-'+'a'*32+'.kwipdata'),
    (3,'prepare','input/table-pack-'+'a'*32+'.kwipdata'),
    (4,'inspect','input/table-pack-'+'a'*32+'.kwipdata'),
    (4,'prepare','input/../table-pack-'+'a'*32+'.kwipdata'),
    (4,'prepare','input/table-pack-'+'a'*31+'.kwipdata'),
])
def test_segments_do_not_expand_historical_or_other_action_authority(model_operation,version,action,pack):
    from runner.model_asset_operation_contract import verify_inputs
    root,value=expanded_request(model_operation,action,['input/training.table.json','input/validation.table.json'])
    if '..' not in pack:
        (root/pack).write_bytes(b'{}')
    value['inputFiles'].append(dict(path=pack,bytes=2,sha256=digest(b'{}')))
    path=root/'input/model-request.json';inner=json.loads(path.read_bytes());inner['schemaVersion']=version
    path.write_bytes(encoded(inner));entry=next(f for f in value['inputFiles'] if f['path']=='input/model-request.json')
    entry.update(bytes=path.stat().st_size,sha256=digest(path.read_bytes()))
    with pytest.raises(ValueError,match='Segmented model tables|Invalid expanded model action|Unreviewed model path'):verify_inputs(value,root,{})


def test_expanded_evaluation_requires_original_training_and_only_selected_population(model_operation):
    from runner.model_asset_operation_contract import verify_inputs
    root,value=expanded_request(model_operation,'evaluate',['input/training.table.json','input/evaluation.table.json','input/evaluationPopulation.table.json'],{'evaluationRole':'test'})
    assert verify_inputs(value,root,{})['action']=='evaluate'
    value['inputFiles']=[f for f in value['inputFiles'] if f['path']!='input/training.table.json']
    with pytest.raises(ValueError,match='Missing expanded'):verify_inputs(value,root,{})


@pytest.mark.parametrize('change',['unused-resume','unneeded-table','missing-terminal','wrong-mode','adapter'])
def test_expanded_actions_refuse_unnecessary_or_incomplete_authorities(model_operation,change):
    from runner.model_asset_operation_contract import verify_inputs
    extras=['input/training.table.json','input/validation.table.json'];settings={};action='prepare'
    if change=='unused-resume':extras+=['input/resume/training.json','input/resume/result-manifest.json']
    if change=='unneeded-table':extras+=['input/evaluation.table.json']
    if change=='missing-terminal':action='train';settings={'resumeMode':'exact'};extras+=['input/resume/training.json','input/resume/result-manifest.json']
    if change=='wrong-mode':action='train';settings={'resumeMode':'weights'};extras+=['input/resume/training.json','input/resume/result-manifest.json','input/resume/latest/kwip-continuation.json']
    if change=='adapter':extras+=['input/adapter/adapter_config.json']
    root,value=expanded_request(model_operation,action,extras,settings)
    with pytest.raises(ValueError,match='authority|resume'):verify_inputs(value,root,{})

@pytest.mark.parametrize('ast',[False,True])
def test_audio_processor_authority_does_not_replace_text_tokenizer(model_operation,ast):
    from runner.model_asset_operation_contract import verify_inputs
    f=model_operation;root,value=f[:2]
    (root/'source/tokenizer.json').unlink()
    (root/'source/preprocessor_config.json').write_bytes(b'{}')
    (root/'source/config.json').write_bytes(encoded(dict(model_type='audio-spectrogram-transformer',architectures=['ASTForAudioClassification']) if ast else dict(model_type='gpt2')))
    value['inputFiles']=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in sorted(root.rglob('*')) if p.is_file() and p.parent in (root/'source',root/'input')]
    if ast:assert verify_inputs(value,root,{})['action']=='inspect'
    else:
        with pytest.raises(ValueError,match='task-matched'):verify_inputs(value,root,{})
