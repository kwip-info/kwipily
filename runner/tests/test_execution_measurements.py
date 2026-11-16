"""Real engine measurement must separate execution from observation, without cells/prose."""
import json
import pytest
from runner import dataflow


def test_real_dag_measurement_keeps_results_and_excludes_private_payload(tmp_path):
    from runner.execution_measurements import ExecutionMeasurements
    source=tmp_path/'private-customer.csv';source.write_text('secret\nPRIVATE_CELL_SENTINEL\n')
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}},{'id':'copy','operator':'kwip.data.formula','config':{'operations':[{'id':'copy','expression':'[secret]','outputs':[{'name':'duplicate','dtype':'string','write':'new'}],'nullPolicy':'preserve'}]}}], 'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'copy','port':'table'}}]}
    clock=[0]
    def now():return clock[0]
    def observe(node,outputs):clock[0]+=7_000_000
    measurements=ExecutionMeasurements(['source','copy'],clock=now)
    baseline=dataflow.execute(graph,{'source':source})
    result=dataflow.execute(graph,{'source':source},on_node=observe,measurements=measurements)
    assert result==baseline
    report=measurements.report()
    assert [r['nodeID'] for r in report['nodes']]==['source','copy']
    assert all(r['status']=='completed' and r['milliseconds']['observation']==7 for r in report['nodes'])
    assert all(r['milliseconds']['execution']==0 and r['cacheHit'] is False for r in report['nodes'])
    wire=json.dumps(report)
    for private in ('PRIVATE_CELL_SENTINEL','private-customer.csv','secret','duplicate','expression'):assert private not in wire
    # Report mutation does not change the collector's retained finite aggregates.
    report['nodes'][0]['milliseconds']['observation']=999
    assert measurements.report()['nodes'][0]['milliseconds']['observation']==7


def test_observation_failure_and_interruption_keep_only_phase_and_status(tmp_path):
    from runner.execution_measurements import ExecutionMeasurements
    source=tmp_path/'data.csv';source.write_text('value\n1\n')
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}],'edges':[]}
    for exception,status in [(ValueError('PRIVATE_PATH /secret'), 'failed'),(KeyboardInterrupt('PRIVATE_SECRET'), 'interrupted')]:
        measurements=ExecutionMeasurements(['source'])
        def fail(*args):raise exception
        with pytest.raises(type(exception)):
            dataflow.execute(graph,{'source':source},on_node=fail,measurements=measurements)
        report=measurements.report();entry=report['nodes'][0]
        assert entry['status']==status and entry['failurePhase']=='observation'
        assert 'PRIVATE' not in json.dumps(report) and len(report['nodes'])==1


def test_collector_rejects_foreign_and_duplicate_nodes_and_unreviewed_phases():
    from runner.execution_measurements import ExecutionMeasurements
    measurements=ExecutionMeasurements(['known'])
    with pytest.raises(ValueError):
        with measurements.node('foreign'):pass
    with measurements.node('known'):
        with pytest.raises(ValueError):
            with measurements.phase('known','arbitrary private field'):pass
    with pytest.raises(ValueError):
        with measurements.node('known'):pass
    assert len(measurements.report()['nodes'])==1


def test_independent_threads_own_distinct_finite_node_records():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from runner.execution_measurements import ExecutionMeasurements
    barrier=threading.Barrier(2,timeout=2);emitted=[]
    measurements=ExecutionMeasurements(['left','right'],on_node=emitted.append)
    def work(node):
        with measurements.node(node):
            with measurements.phase(node,'execution'):barrier.wait()
            measurements.cache_hit(node,False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(work,node) for node in ('left','right')]
        for future in futures:future.result()
    report=measurements.report()
    assert [n['nodeID'] for n in report['nodes']]==['left','right']
    assert len(emitted)==2 and all(n['status']=='completed' for n in emitted)
    assert all(n['milliseconds']['execution']>0 for n in report['nodes'])


def test_profiled_archive_equals_unprofiled_without_implicit_telemetry_artifacts(tmp_path):
    from pathlib import Path
    from runner.execution_measurements import ExecutionMeasurements
    from runner.dataflow_cli import run_dataflow
    root=Path(__file__).resolve().parents[2]
    graph=root/'examples/dataflow/scaler.json';bindings={'source':str(root/'examples/dataflow/observations.csv')}
    ids=[n['id'] for n in json.loads(graph.read_text())['nodes']]
    collector=ExecutionMeasurements(ids)
    baseline=run_dataflow(graph,bindings,tmp_path/'baseline',lambda e:None)
    profiled=run_dataflow(graph,bindings,tmp_path/'profiled',lambda e:None,measurements=collector)
    assert profiled==baseline
    assert [n['nodeID'] for n in collector.report()['nodes']]==ids
    for entry in collector.report()['nodes']:
        assert entry['wallMilliseconds']>=sum(entry['milliseconds'].values())-0.001
    assert not list((tmp_path/'profiled').glob('*measurement*.json'))


def test_cli_measurement_opt_in_emits_fixed_per_node_metadata_only(tmp_path):
    import subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    source=tmp_path/'data.csv';source.write_text('secret\nPRIVATE_CELL_SENTINEL\n')
    graph=tmp_path/'graph.json';graph.write_text(json.dumps({'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}],'edges':[]}))
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}))
    for enabled in ('true','false'):
        result=subprocess.run([sys.executable,str(root/'runner/main.py'),'--execute-dataflow',str(graph),'--source-bindings',str(bindings),'--output',str(tmp_path/enabled),'--dataflow-measurements',enabled],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        events=[json.loads(line) for line in result.stdout.splitlines()]
        timings=[e for e in events if e['event']=='dataflowNodeTiming']
        assert len(timings)==(1 if enabled=='true' else 0)
        if timings:
            assert timings[0]['measurement']['nodeID']=='source'
            wire=json.dumps(timings[0]);assert 'PRIVATE_CELL_SENTINEL' not in wire and str(source) not in wire
        assert events[0]['event']=='dataflowStarted' and events[-1]['event']=='dataflowCompleted'
