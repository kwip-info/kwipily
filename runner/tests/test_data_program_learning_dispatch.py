"""Ten lesson routes through the existing strict pinned worker input boundary."""
import copy
import hashlib
import json
import pytest
from runner import learning_evidence as worker
from runner.tests.test_learning_data_loading import TASKS,submission as data_submission,fixture as data_fixture,tensor
from runner.tests.test_learning_program_export import submission as program_submission,fixture as program_fixture

DATA=TASKS+['loader-continue-grouping']
TASKS_ALL=DATA+['program-export-computation','program-export-size-range']

def case(task):
    if task in DATA:
        construction=data_submission(task if task in TASKS else TASKS[0]);construction['taskID']=task
        fixture=data_fixture(task if task in TASKS else TASKS[0])
        if task==DATA[-1]:
            construction['graph']['nodes'][1]['config'].update(shuffle=True,num_workers=2,prefetch_factor=1,timeout=5.,collation='whole_bags')
            fixture['inputs'].update(sourceRows=tensor([9,3],[v for i in range(9) for v in (i%5,(i+1)%5,(i+2)%5)]),sourceLengths=tensor([9],[i%4 for i in range(9)]),sourceWeights=tensor([9,3],[.5,1.,-.25]*9))
        capabilities={key:dict(major=1,minor=0,patch=0) for key in ('dataset_source','data_loader')}
    else:
        construction=program_submission(task.endswith('size-range'));fixture=program_fixture(3 if task.endswith('size-range') else 2)
        capabilities={'program_export':dict(major=1,minor=0,patch=0)}
    request=dict(assessmentID='assessment.'+task,assessmentVersion=dict(major=1,minor=0,patch=0),fixtureDigest='a'*64,runtime=dict(backend='pytorch',version=worker.runtime_versions()['pytorch'],capabilities=capabilities),budget=dict(milliseconds=60000,maximumTensorElements=100000),fixtures=[fixture])
    return request,construction

@pytest.mark.parametrize('task',TASKS_ALL)
def test_ten_real_lesson_routes_use_pinned_files_and_actual_measurement(task,tmp_path,monkeypatch):
    captured=[]
    if task==DATA[-1]:
        from runner import dataflow_cli
        original=dataflow_cli.run_dataflow
        def observe(graph_path,bindings,output,*args,**kwargs):
            graph=json.loads(graph_path.read_text())
            result=original(graph_path,bindings,output,*args,**kwargs)
            tables={entry['port']:json.loads((output/entry['path']).read_text()) for entry in result['artifacts'] if entry.get('nodeID')=='p'}
            captured.append((graph,tables))
            return result
        monkeypatch.setattr(dataflow_cli,'run_dataflow',observe)
    request,construction=case(task);raw=json.dumps(construction,sort_keys=True,separators=(',',':')).encode()
    r=tmp_path/'request.json';s=tmp_path/'submission.json';r.write_text(json.dumps(request));s.write_bytes(raw)
    pin=hashlib.sha256(raw).hexdigest();result=worker.execute_request(r,s,pin)
    assert result['submissionDigest']==pin and result['runtime']==request['runtime']
    assert not result.get('backendError'),result.get('backendError')
    assert not result['cancelled'] and len(result['fixtures'])==1
    observed=result['fixtures'][0]['tensors'];assert observed and not {'expected','inputGradient'}&set(observed)
    if task==DATA[-1]:
        assert observed['continuationChecks']['values']==[1]*4
        assert observed['readerPolicy']['values']==[2,1]
        assert observed['parentByteDifference']['values']==[0]
        assert observed['epochOrderDifference']==tensor([9],[0]*9)
        assert len(captured)==1
        graph,tables=captured[0]
        assert {node['operator'] for node in graph['nodes']}=={'kwip.data.sourceCSV','kwip.data.columns','kwip.data.partition'}
        assert [len(tables[role]['rows']) for role in ('training','validation','test','exclusions')]==[5,2,2,0]
        identities=[set(tables[role]['lineage']['observationIDs']) for role in ('training','validation','test')]
        assert len(set.union(*identities))==9 and all(not left&right for i,left in enumerate(identities) for right in identities[i+1:])
        inputs=request['fixtures'][0]['inputs'];expected={identity:index for index,identity in enumerate(inputs['rowIDs']['values'])}
        for role in ('training','validation','test'):
            table=tables[role];assert table['lineage']['role']==role
            assert [column['dtype'] for column in table['columns']]==['int64']*3+['float64']*3+['int64','float64','int64']
            assert [row[-1] for row in table['rows']]==sorted(row[-1] for row in table['rows'])
            for row in table['rows']:
                index=expected[row[-1]]
                assert row[:3]==inputs['sourceRows']['values'][index*3:index*3+3]
                assert row[3:6]==inputs['sourceWeights']['values'][index*3:index*3+3]
                assert row[6:8]==[inputs['sourceLengths']['values'][index],inputs['desiredRows']['values'][index]]
    if task.startswith('program-export-'):assert observed['artifactChecks']['values']==[1]*3
    with pytest.raises(ValueError,match='digest changed'):worker.execute_request(r,s,'0'*64)

@pytest.mark.parametrize('task',TASKS_ALL)
def test_task_binding_cancel_and_runtime_mismatch_cannot_gain_success(task):
    request,construction=case(task)
    stopped=worker.generate_evidence(request,construction,submission_digest='b'*64,cancel=lambda:True)
    assert stopped['cancelled'] and not stopped['fixtures']
    stale=copy.deepcopy(request);stale['runtime']['version']='0.0.0'
    assert not worker.generate_evidence(stale,construction,submission_digest='b'*64)['fixtures']
    foreign=copy.deepcopy(construction);foreign['taskID']='foreign'
    result=worker.generate_evidence(request,foreign,submission_digest='b'*64)
    assert not result['fixtures'] and result.get('backendError')


def test_native_standalone_import_route(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    runner=Path(worker.__file__).parent
    result=subprocess.run([sys.executable,'-I','-c',"import sys; sys.path.insert(0,sys.argv[1]); import learning_data_loading, learning_program_export" ,str(runner)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
