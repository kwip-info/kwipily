"""Finite, public synthetic data; fresh real native pools and owned branch drain."""
from pathlib import Path
import sys,json,threading,csv
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
from runner.worker_resources import apply_verified_resources
from runner.workload_policy import WorkloadPolicy
workload=WorkloadPolicy('balanced')
branches=int(sys.argv[3]) if len(sys.argv)>3 else 2
apply_verified_resources({'executionBudget':{'schemaVersion':2,'requestedSeconds':60,'maximumRunSeconds':60,'source':'application','resources':{'schemaVersion':2,'cpuThreads':2 if branches==2 else 8,'workloadPolicy':workload.to_document()}}})
from runner import dataflow as df
from runner.preparation_plan import build_plan,execute_plan
from runner.execution_measurements import ExecutionMeasurements
from runner.tests.test_filter_rows import config
import polars as pl
directory=Path(sys.argv[1]);failure=bool(int(sys.argv[2]))
source=directory/'source.csv';source.write_text('value\n'+''.join(('yes' if i%2==0 else 'no')+'\n' for i in range(6000)))
def node(ident,op,c):return {'id':ident,'operator':'kwip.data.'+op,'config':c}
def edge(a,b,port='table'):return {'from':{'node':a,'port':port},'to':{'node':b,'port':'table'}}
output=df.DEFINITIONS['outputDelimited']['defaultConfig']
graph={'schemaVersion':1,'nodes':[
    node('source','sourceCSV',{}),
    node('flag','formula',{'operations':[{'id':'flag','expression':'[value] = "yes"','outputs':[{'name':'keep','dtype':'bool','write':'new'}],'nullPolicy':'preserve'}]}),
    node('left','filterRows',config(mode='expression',expression='[keep]')),
    node('right','filterRows',config(mode='expression',expression='NOT [keep]')),
    node('left-write','outputDelimited',{**output,'filename':'left.csv'}),
    node('right-write','outputDelimited',{**output,'filename':'right.csv'})],
    'edges':[edge('source','flag'),edge('flag','left'),edge('flag','right'),edge('left','left-write','matching'),edge('right','right-write','matching')]}
branch_ids=['left','right']
if branches==4:
    for key in ('third','fourth'):
        branch_ids.append(key)
        graph['nodes'].insert(-2,node(key,'filterRows',config(mode='expression',expression='[keep]')))
        graph['nodes'].append(node(key+'-write','outputDelimited',{**output,'filename':key+'.csv'}))
        graph['edges'] += [edge('flag',key),edge(key,key+'-write','matching')]
sinks=[key+'-write' for key in branch_ids]
reference_dir=directory/'reference';reference_dir.mkdir()
baseline=df.execute(graph,{'source':str(source)},engine='polars',destination_bindings={key:str(reference_dir) for key in sinks})
plan=build_plan(graph,engine='polars',engine_version='2.0.0',workload_policy=workload)
actual_dir=directory/'actual';actual_dir.mkdir()
barrier=threading.Barrier(branches,timeout=5);lock=threading.Lock();active=0;maximum=0
original=df._execute_node
def observed(op,c,args,*pos,**kw):
    global active,maximum
    key=kw.get('node_id')
    if key not in branch_ids:return original(op,c,args,*pos,**kw)
    with lock:active+=1;maximum=max(maximum,active)
    try:
        barrier.wait()
        if failure and key=='right':raise ValueError('Injected native-branch failure')
        return original(op,c,args,*pos,**kw)
    finally:
        with lock:active-=1
df._execute_node=observed
measurements=ExecutionMeasurements([n['id'] for n in graph['nodes']])
try:
    result=execute_plan(plan,{'source':str(source)},destination_bindings={key:str(actual_dir) for key in sinks},measurements=measurements)
except df.DataFlowExecutionError as error:
    assert failure and error.detail['nodeID']=='right',error.detail
else:
    assert not failure
    assert result==baseline
    for filename,expected in [(key+'.csv','no' if key=='right' else 'yes') for key in branch_ids]:
        assert (actual_dir/filename).read_bytes()==(reference_dir/filename).read_bytes()
        with (actual_dir/filename).open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==3000 and all(row['value']==expected for row in rows)
assert active==0 and not any(t.name.startswith('kwip-native') for t in threading.enumerate())
records=measurements.report()['nodes']
assert len(records)==(2+branches if failure else 2+branches*2)
assert all(record['wallMilliseconds']>=sum(record['milliseconds'].values())-0.001 for record in records)
print(json.dumps({'maximumConcurrentNodeLifetimes':maximum,'activeAfterExit':active,'polarsPoolThreads':pl.thread_pool_size(),'capturedParallelTasks':workload.limit('parallelTasks'),'failureInjected':failure,'externalFiles':len(list(actual_dir.iterdir()))}))
