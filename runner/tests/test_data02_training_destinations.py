"""Actual managed training preparation authority must retain destination sidecars."""
from pathlib import Path
import pytest
from runner.tests.test_operation_authorization import operation_authority, publish, claim
from runner.worker_authorization import validate_phase_arguments

def test_phase_destination_option_is_preparation_only(tmp_path):
    args=['--execute-dataflow',str(tmp_path/'request.json'),'--source-bindings',str(tmp_path/'bindings.json'),'--output',str(tmp_path/'artifacts'),'--destination-bindings',str(tmp_path/'destinations.json')]
    assert validate_phase_arguments('preparation',args)['--destination-bindings']==str(tmp_path/'destinations.json')
    with pytest.raises(ValueError):validate_phase_arguments('training',args)

@pytest.mark.parametrize('change',['valid','outside','missing','linked'])
def test_ordinary_phase_claim_requires_owned_destination_control(operation_authority,tmp_path,change):
    f=operation_authority
    phase=Path(f[1]['arguments'][1]).parent
    path=phase/'destinations.json'
    path.write_text('{}')
    if change=='outside':path=tmp_path/'outside.json';path.write_text('{}')
    elif change=='missing':path.unlink()
    elif change=='linked':path.unlink();target=tmp_path/'target';target.write_text('{}');path.symlink_to(target)
    f[1]['arguments']+=['--destination-bindings',str(path)];publish(f)
    if change=='valid':
        with claim(f) as authority:authority.check()
    else:
        with pytest.raises((ValueError,OSError)):claim(f)

def test_destination_control_cannot_change_after_authority_claim(operation_authority):
    f=operation_authority
    phase=Path(f[1]['arguments'][1]).parent
    path=phase/'destinations.json';path.write_text('{}')
    f[1]['arguments']+=['--destination-bindings',str(path)];publish(f)
    with claim(f) as authority:
        path.write_text('{"save":"/tmp/ungranted"}')
        with pytest.raises(ValueError):authority.check()

from runner.tests.test_preparation_operation_authorization import preparation_operation
from runner.tests.test_preparation_operation_authorization import inventory as preparation_inventory

def test_standalone_preparation_destination_inventory_is_pinned(preparation_operation,tmp_path):
    import json
    from runner.preparation_operation_contract import verify_inputs
    f=preparation_operation;root=f[0]
    graph=json.loads((root/'input/graph.json').read_text())
    graph['nodes'].append({'id':'save','operator':'kwip.data.outputDelimited','config':{'filename':'out.csv','separator':',','columns':[],'overwrite':False}})
    graph['edges'].append({'from':{'node':'source','port':'table'},'to':{'node':'save','port':'table'}})
    (root/'input/graph.json').write_text(json.dumps(graph))
    destination=tmp_path/'saved';destination.mkdir()
    path=root/'input/destinations.json';path.write_text(json.dumps({'save':str(destination)}))
    f[1]['arguments']+=['--destination-bindings',str(path)]
    preparation_inventory(f)
    verify_inputs(f[1],root,{})
    path.write_text(json.dumps({'save':'/tmp/ungranted'}))
    with pytest.raises((ValueError,OSError)):verify_inputs(f[1],root,{})
