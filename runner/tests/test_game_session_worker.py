import copy,json,uuid,types,hashlib
from pathlib import Path
import pytest
from runner.game_rule_examples import take_away
from runner.game_episodes import encoded,configuration,digest
from runner.game_session_operation_contract import parse_contract,verify_inputs
from runner.game_session_worker import execute
from runner.dataflow_table_reader import read_table_artifact
from runner.tests.test_game_episodes import numeric_graph

@pytest.fixture
def captured(tmp_path):
    root=tmp_path/'workflow-operations'/str(uuid.uuid4()).upper()/str(uuid.uuid4()).upper();(root/'input').mkdir(parents=True)
    inner=dict(schemaVersion=1,definition=take_away(),configuration=configuration({'episodes':2,'policies':{'A':{'source':'canvas'},'B':{'source':'sameA'}}}),policies={'A':dict(graph=numeric_graph(),sourceIdentity='a'*64)},validateOnly=False)
    raw=encoded(inner);(root/'input/session.json').write_bytes(raw)
    request=dict(operationID=root.name,projectID=root.parent.name,requestIdentity='b'*64,arguments=['--game-session',str(root/'input/session.json')],inputFiles=[dict(path='input/session.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
    return root,inner,request

def publish(f):
    root,inner,r=f;raw=encoded(inner);(root/'input/session.json').write_bytes(raw);r['inputFiles'][0].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def test_real_frozen_worker_publishes_tables_readable_by_shared_reader(captured):
    root,inner,r=captured;events=[]
    execute(types.SimpleNamespace(_path=root,request=r),events.append,lambda:None)
    assert events[-1]['event']=='gameSessionCompleted'
    assert len(events[-1]['files'])==4
    tables={}
    for f in events[-1]['files']:
        raw=(root/f['path']).read_bytes();assert digest(json.loads(raw))==f['sha256']
        if '.table.' in f['path']:
            tables[f['path']]=read_table_artifact(root/f['path'],f['sha256'],lambda:None)
    assert len(tables['artifacts/episodes.table.json'].rows)==2
    assert len(tables['artifacts/moves.table.json'].rows)>0
    assert sum(f['bytes'] for f in events[-1]['files'])<=8_000_000

def test_validation_runs_no_episodes_and_no_moves(captured):
    root,inner,r=captured;inner['validateOnly']=True;publish(captured);events=[]
    execute(types.SimpleNamespace(_path=root,request=r),events.append,lambda:None)
    out=json.loads((root/'artifacts/session.json').read_bytes());assert out['episodes']==out['moves']==[] and out['summary']['tests']==3

@pytest.mark.parametrize('mutation',['escape','duplicate','extra','size','tamper','symlink'])
def test_scoped_contract_refuses_unowned_or_changed_inputs(captured,tmp_path,mutation):
    root,inner,r=captured
    if mutation=='escape':r['arguments'][1]=str(tmp_path/'session.json')
    elif mutation=='duplicate':r['inputFiles']*=2
    elif mutation=='extra':r['inputFiles'][0]['path']='input/user.py'
    elif mutation=='size':r['inputFiles'][0]['bytes']=2_000_001
    elif mutation=='tamper':(root/'input/session.json').write_text('{}')
    elif mutation=='symlink':
        p=root/'input/session.json';raw=p.read_bytes();p.unlink();target=tmp_path/'outside';target.write_bytes(raw);p.symlink_to(target)
    with pytest.raises((ValueError,OSError)):verify_inputs(r,root,{})

def test_unsupported_model_width_is_not_random_fallback(captured):
    root,inner,r=captured;inner['definition']=__import__('runner.game_rule_examples',fromlist=['board_game']).board_game(3,3,3,False);publish(captured)
    with pytest.raises((ValueError,RuntimeError)):execute(types.SimpleNamespace(_path=root,request=r),lambda e:None,lambda:None)
    assert not (root/'artifacts').exists()

def test_actual_safe_checkpoint_is_restored_without_random_reinitialization(captured):
    import torch
    from runner.game_episodes import FrozenGraphPolicy
    root,inner,r=captured;p=FrozenGraphPolicy(numeric_graph(),seed=204)
    path=root/'input/A.pt';torch.save({'modelState':p.model.state_dict()},path);raw=path.read_bytes();pin=hashlib.sha256(raw).hexdigest()
    inner['policies']['A']['checkpointSHA256']=pin;inner['configuration']['policies']['A']['source']='saved'
    r['inputFiles'].append(dict(path='input/A.pt',bytes=len(raw),sha256=pin));publish(captured)
    execute(types.SimpleNamespace(_path=root,request=r),lambda e:None,lambda:None)
    result=json.loads((root/'artifacts/session.json').read_bytes())
    assert result['identity']['policies']['A']==p.identity
    assert path.read_bytes()==raw
