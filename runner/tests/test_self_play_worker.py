import hashlib,json,types,uuid
import pytest
from runner.game_rule_examples import take_away
from runner.game_episodes import encoded
from runner.self_play_worker import execute
from runner.self_play_operation_contract import verify_inputs
from runner.tests.test_self_play import paired_graph,tiny_settings
from runner.dataflow_table_reader import read_table_artifact

def capture(root,inner):
    (root/'input').mkdir(parents=True);document=encoded({'synthetic':'worker-only test'});inner['parentOperationID']=None;inner['trainAddress']={'domain':'control','localID':'train'};inner['documentSHA256']=hashlib.sha256(document).hexdigest();(root/'input/document.json').write_bytes(document);raw=encoded(inner);(root/'input/self-play.json').write_bytes(raw)
    return dict(operationID=root.name,projectID=root.parent.name,requestIdentity='b'*64,arguments=['--self-play',str(root/'input/self-play.json')],inputFiles=[dict(path='input/document.json',bytes=len(document),sha256=inner['documentSHA256']),dict(path='input/self-play.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])

def test_actual_worker_publishes_standard_tables_and_safe_paired_inference(tmp_path):
    import torch
    root=tmp_path/'workflow-operations'/str(uuid.uuid4()).upper()/str(uuid.uuid4()).upper()
    inner=dict(schemaVersion=1,definition=take_away(),graph=paired_graph(),configuration=tiny_settings(),initialization=None,trainability=None,resume=False,validateOnly=False)
    request=capture(root,inner);events=[];execute(types.SimpleNamespace(_path=root,request=request),events.append,lambda:None)
    done=events[-1];assert done['event']=='selfPlayCompleted'
    for f in done['files']:
        raw=(root/f['path']).read_bytes();assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256']
        if '.table.' in f['path']:read_table_artifact(root/f['path'],f['sha256'],lambda:None)
    model=torch.load(root/'artifacts/model.pt',weights_only=True)
    assert model['task']=='gamePolicyValue' and not model['resumable']
    from runner.game_episodes import FrozenGraphPolicy
    assert len(FrozenGraphPolicy(model['graph'],state=model['modelState']).logits([.4]))==3
    # Required resume state is independent from the portable inference export.
    assert (root/'artifacts/replay/latest.json').exists()
    request['arguments'][1]=str(tmp_path/'self-play.json')
    with pytest.raises(ValueError):verify_inputs(request,root,{})


def test_single_run_uses_captured_archived_opponent(tmp_path):
    import shutil
    project=tmp_path/'workflow-operations'/str(uuid.uuid4()).upper()
    def inner(config,validate=False):
        return dict(schemaVersion=1,definition=take_away(),graph=paired_graph(),configuration=config,initialization=None,trainability=None,resume=False,validateOnly=validate)
    source=project/str(uuid.uuid4()).upper();c=tiny_settings();c.update(iterations=1,updates=1)
    execute(types.SimpleNamespace(_path=source,request=capture(source,inner(c))),lambda e:None,lambda:None)
    opponent='checkpoint:'+source.name
    c['evaluation']={'opponents':[opponent],'openings':[[]],'seed':8301}
    for validate in (True,False):
        root=project/str(uuid.uuid4()).upper();request=capture(root,inner(c,validate))
        folder=root/'input/opponents'/source.name;folder.mkdir(parents=True)
        for original,name in [('model.pt','model.pt'),('model.json','model.json'),('inference-manifest.json','identity.json')]:
            target=folder/name;shutil.copyfile(source/'artifacts'/original,target);raw=target.read_bytes()
            request['inputFiles'].append(dict(path=target.relative_to(root).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        events=[];execute(types.SimpleNamespace(_path=root,request=request),events.append,lambda:None)
        assert events[-1]['event']=='selfPlayCompleted'
        if not validate:
            result=json.loads((root/'artifacts/self-play.json').read_text())
            assert len(result['evaluationGames'])==4
            assert {r['opponent'] for r in result['evaluationGames']}=={opponent}
            assert all(r['status']=='terminal' for r in result['evaluationGames'])


def test_missing_opponent_refused_before_any_learning(tmp_path):
    from runner.self_play_training import train
    c=tiny_settings();c['evaluation']={'opponents':['checkpoint:missing']}
    events=[]
    with pytest.raises(ValueError,match='verified compatible frozen opponent'):
        train(take_away(),paired_graph(),c,tmp_path,progress=events.append)
    assert events==[] and not (tmp_path/'latest.json').exists()
