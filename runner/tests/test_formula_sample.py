import json
import pytest
from runner import table_view as view
from runner.tests.test_table_view import index
from runner.tests.test_formula import operation


def test_draft_sample_is_bounded_pinned_and_does_not_rewrite_index(tmp_path):
    root=tmp_path/'input';root.mkdir()
    path,receipt=index(root,30);before=path.read_bytes()
    request={'schemaVersion':4,'offset':0,'limit':8,'search':'','sourceSHA256':'a'*64,'indexSHA256':receipt['index']['sha256'],'formula':{'operations':[operation('[id] * 2',('double','int64'))]}}
    query=root/'view-query.json';query.write_text(json.dumps(request));events=[]
    view.run_query(query,events.append)
    page=events[0]
    assert page['totalRows']==page['matchedRows']==8 and page['nextOffset'] is None
    assert [r[-1] for r in page['rows']]==list(range(0,16,2))
    assert path.read_bytes()==before
    request['indexSHA256']='b'*64;query.write_text(json.dumps(request))
    with pytest.raises(ValueError):view.run_query(query,lambda _:None)


@pytest.mark.parametrize('change',[{'limit':9},{'offset':1},{'search':'x'},{'sortColumn':0},{'schemaVersion':3}])
def test_sample_cannot_expand_scope_or_enter_legacy_queries(change):
    request={'schemaVersion':4,'offset':0,'limit':8,'search':'','sourceSHA256':'a'*64,'indexSHA256':'b'*64,'formula':{'operations':[operation('0',('x','int64'))]}}
    request.update(change)
    with pytest.raises(ValueError):view.validate_query(request)


def test_draft_sample_refuses_truncated_display_cells(tmp_path):
    from types import SimpleNamespace
    root=tmp_path/'input';root.mkdir();path=root/'view.sqlite3'
    table=SimpleNamespace(columns=({'name':'text','dtype':'string','nullable':False},),rows=[('a'*2001,)],row_ids=['one'])
    receipt=view.build_index(table,path,source_sha256='a'*64)
    query=root/'view-query.json';query.write_text(json.dumps({'schemaVersion':4,'offset':0,'limit':8,'search':'','sourceSHA256':'a'*64,'indexSHA256':receipt['index']['sha256'],'formula':{'operations':[operation('length([text])',('n','int64'))]}}))
    with pytest.raises(ValueError,match='truncated'):view.run_query(query,lambda _:None)
