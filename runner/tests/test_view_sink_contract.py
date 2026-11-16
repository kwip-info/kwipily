"""Canvas sink presentation must not erase retained View execution evidence."""
import json
import pytest
from pathlib import Path
from runner.dataflow import REGISTRY,execute


def test_view_declares_canvas_sink_but_retains_technical_table_output():
    view=next(op for op in REGISTRY['operators'] if op['id']=='kwip.data.view')
    assert view['canvasRole']=='sink'
    assert [p['id'] for p in view['inputs']]==['table']
    assert [p['id'] for p in view['outputs']]==['table']


def test_saved_legacy_view_outbound_connection_still_executes(tmp_path):
    source=tmp_path/'values.csv';source.write_text('value\n7\n8\n')
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'view','operator':'kwip.data.view','config':{}},
        {'id':'cast','operator':'kwip.data.columns','config':{'rows':[dict(source='value',included=True,outputName='value',targetType='int64')],'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}}],
        'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'view','port':'table'}},
                 {'from':{'node':'view','port':'table'},'to':{'node':'cast','port':'table'}}]}
    result=execute(graph,{'source':str(source)})
    assert result['view']['table'].data_version==result['source']['table'].data_version
    assert list(result['cast']['table'].rows)==[(7,),(8,)]
    assert list(result['cast']['table'].row_ids)==list(result['view']['table'].row_ids)


def test_saved_view_unknown_output_is_rejected_without_borrowing_technical_table_port(tmp_path):
    source=tmp_path/'values.csv';source.write_text('value\n7\n8\n')
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'view','operator':'kwip.data.view','config':{}},
        {'id':'target','operator':'kwip.data.columns','config':{'rows':[],'castFailure':'error','trimText':False,'newColumns':'keep','missingPolicy':'preserve'}}],
        'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'view','port':'table'}},
                 {'from':{'node':'view','port':'out'},'to':{'node':'target','port':'table'}}]}
    with pytest.raises(ValueError,match='port|output'):
        execute(graph,{'source':str(source)})
