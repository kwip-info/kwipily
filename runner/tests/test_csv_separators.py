import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner import dataflow

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('separator',[',','\t',';','|'])
def test_inspection_and_preparation_match_quoted_fields_and_exact_headers(tmp_path,separator):
    stream=io.StringIO();writer=csv.writer(stream,delimiter=separator)
    writer.writerow([' exact name','category','measurement']);writer.writerows([['a'+separator+'b','quote "here"','1'],['second','multi\nline','2']])
    source=tmp_path/'source.csv';source.write_text(stream.getvalue())
    p=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--inspect-data',str(source),'--csv-separator',separator],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    preview=json.loads(p.stdout)
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{'separator':separator}}],'edges':[]}
    table=dataflow.execute(graph,{'s':str(source)})['s']['table']
    assert preview['columns']==[c['name'] for c in table.columns]
    assert preview['rows']==list(map(list,table.rows))
    assert table.source_digest==hashlib.sha256(source.read_bytes()).hexdigest()

@pytest.mark.parametrize('separator',['','::','\n','"','x'])
def test_invalid_separator_rejected_by_inspection_and_preparation(tmp_path,separator):
    source=tmp_path/'source.csv';source.write_text('x,y\n1,2\n')
    p=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--inspect-data',str(source),'--csv-separator',separator],capture_output=True,text=True)
    assert p.returncode==1 and json.loads(p.stdout)['event']=='error'
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{'separator':separator}}],'edges':[]}
    with pytest.raises(ValueError):dataflow.execute(graph,{'s':str(source)})

def test_separator_flag_cannot_silently_change_direct_training():
    p=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--capabilities','--csv-separator',';'],capture_output=True,text=True)
    assert p.returncode==1 and '--inspect-data' in json.loads(p.stdout)['message']

def test_semicolon_preparation_export_replays_separator_and_source_identity(tmp_path):
    from runner.dataflow_cli import run_dataflow
    source=tmp_path/'source.csv';source.write_text('x;category\n1;"a;b"\n2;plain\n')
    graph=tmp_path/'graph.json';graph.write_text(json.dumps({'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{'separator':';'}}],'edges':[]}))
    bindings={'s':str(source)};output=tmp_path/'first'
    manifest=run_dataflow(graph,bindings,output,lambda e:None)
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings))
    replay=tmp_path/'replay'
    p=subprocess.run([sys.executable,str(output/'prep.py'),'--source-bindings',str(private),'--output',str(replay)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr+p.stdout
    assert manifest['artifacts']==json.loads((replay/'manifest.json').read_text())['artifacts']
    assert json.loads((output/'node-000.json').read_text())['rows']==[['1','a;b'],['2','plain']]
