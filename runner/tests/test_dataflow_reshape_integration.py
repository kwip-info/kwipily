"""Real dispatch, cache and generated standalone Code for both table-reshape modes."""
import hashlib
import json
from pathlib import Path
import pytest
from runner.dataflow_cli import run_dataflow
from runner.dataflow_reshape import defaults
from runner.polars_csv_configuration import DEFAULTS
from runner.tests.test_dataflow_portable_dependencies import replay


def fixture(tmp_path,wide):
    source=tmp_path/'source.csv'
    source.write_text('customer,month,sales,units\nAda,feb,12,2\nAda,jan,10,1\nBen,jan,30,3\n' if wide else 'customer,jan,feb\nAda,10,12\nBen,20,\n')
    c=defaults()
    if wide:c.update(mode='wide',rowKeys=['customer'],columnKey='month',values=[dict(column='sales',aggregation='sum',prefix='sales_',nullPolicy='skip'),dict(column='units',aggregation='max',prefix='units_',nullPolicy='skip')])
    else:c.update(idColumns=['customer'],valueColumns=['jan','feb'])
    source_config={**DEFAULTS,'emptyField':'null','schemaOverrides':{n:'Int64' for n in (['sales','units'] if wide else ['jan','feb'])}}
    graph=dict(schemaVersion=1,nodes=[dict(id='source',operator='kwip.data.sourceCSV',config=source_config),dict(id='reshape',operator='kwip.data.reshapeTable',config=c)],edges=[{'from':dict(node='source',port='table'),'to':dict(node='reshape',port='table')}])
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));return path,{'source':str(source)}

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('wide',[False,True])
def test_real_cache_code_export_and_working_complete_authority(tmp_path,engine,wide):
    path,bindings=fixture(tmp_path,wide);first=tmp_path/'first'
    manifest=run_dataflow(path,bindings,first,lambda _:None,engine=engine,cache_policy='materialize')
    entry=next(item for item in manifest['artifacts'] if item.get('nodeID')=='reshape')
    output=json.loads((first/entry['path']).read_text())
    expected=[['Ada',12,10,2,1],['Ben',None,30,None,3]] if wide else [['Ada','jan',10],['Ada','feb',12],['Ben','jan',20],['Ben','feb',None]]
    assert output['rows']==expected
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['cache']['reusedNodeIDs']==['reshape']
    assert cached['artifacts']==manifest['artifacts']
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'cwd';cwd.mkdir()
    assert replay(first,private,tmp_path/'replay',cwd)['artifacts']==manifest['artifacts']
    sampled=run_dataflow(path,bindings,tmp_path/'sample',lambda _:None,engine=engine,sampling={'mode':'rowCount','rowCount':1})
    assert sampled['executionCoverage']['status']!='complete'
    graph=json.loads(path.read_text());graph['nodes'].append(dict(id='view',operator='kwip.data.view',config={}));graph['edges'].append({'from':dict(node='reshape',port='table'),'to':dict(node='view',port='table')});path.write_text(json.dumps(graph))
    complete=run_dataflow(path,bindings,tmp_path/'complete',lambda _:None,engine=engine,sampling={'mode':'rowCount','rowCount':1})
    assert complete['executionCoverage']['status']=='complete'
    entry=next(item for item in complete['artifacts'] if item.get('nodeID')=='reshape')
    assert json.loads((tmp_path/'complete'/entry['path']).read_text())['rows']==expected
