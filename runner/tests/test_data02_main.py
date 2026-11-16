"""Actual runner CLI admission, independent of direct adapter invocation."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_main_executes_explicit_destination_binding(tmp_path):
    destination=tmp_path/'saved';destination.mkdir()
    graph=ROOT/'runner/dataflow_examples/data02-output.json'
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(ROOT/'runner/dataflow_examples/data02-observations.csv')}))
    destinations=tmp_path/'destinations.json';destinations.write_text(json.dumps({'tool':str(destination)}))
    p=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--execute-dataflow',str(graph),'--source-bindings',str(bindings),'--destination-bindings',str(destinations),'--output',str(tmp_path/'artifacts')],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr+p.stdout
    rows=[json.loads(x) for x in p.stdout.splitlines()]
    assert next(x for x in rows if x['event']=='dataflowOutputSaved')['rowCount']==8
    assert (destination/'prepared-orders.csv').read_text().splitlines()[0]=='order_id;quantity;unit_price'

def test_destination_bindings_refused_without_dataflow(tmp_path):
    f=tmp_path/'destinations.json';f.write_text('{}')
    p=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--dataflow-registry','--destination-bindings',str(f)],capture_output=True,text=True,timeout=30)
    assert p.returncode!=0
    assert 'require --execute-dataflow' in p.stdout+p.stderr
