"""Formula stores changed fields and retains immutable parent columns."""
from pathlib import Path
import pytest
from runner import dataflow as df
from runner.dataflow_columnar import ColumnarStore
from runner.dataflow_disk import disk_scope
from runner.dataflow_formula import apply


@pytest.fixture(autouse=True)
def qualified_engine():
    from runner.polars_preparation import preparation_engine
    with preparation_engine('polars'):yield


def operation(name,expression,dtype,write='new'):
    return {'id':name+'-'+write,'expression':expression,'outputs':[{'name':name,'dtype':dtype,'write':write}],'nullPolicy':'preserve'}


@pytest.mark.parametrize('operations',[
    [operation('twice','[n] * 2','int64')],
    [operation('n','[n] + 1','int64','replace'),operation('twice','[n] * 2','int64')],
    [operation('new','[text]','string'),operation('upper','upper([new])','string')],
    [operation('keep','[n] > 10','bool')],
    [operation('hit','regex_match([text], "^1", "search", "")','bool')]])
def test_mixed_replacements_and_new_columns_equal_reference(tmp_path,operations):
    columns=({'name':'n','dtype':'int64','nullable':False},{'name':'text','dtype':'string','nullable':True})
    with disk_scope(directory=tmp_path) as budget:
        store=ColumnarStore(columns)
        for i in range(9000):store.append((i,None if i%7==0 else str(i)),str(i))
        rows,ids=store.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64))
        expected=actual=None
        try:
            expected=apply(source,{'operations':operations},lambda:None)
            actual=apply(source,{'operations':operations},lambda:None,share=True)
            assert actual==expected
            assert actual.rows.store.parent is store
            changed={output['name'] for op in operations for output in op['outputs']}
            assert len(actual.rows.store.computed.columns)==len(changed)
            actual.rows.store.close();assert source.rows[1]==(1,'1')
        finally:
            for table in (expected,actual):
                if table is not None:table.rows.store.close()
            store.close()
        assert budget.used==0 and list(tmp_path.iterdir())==[]


def test_shared_formula_preserves_full_row_byte_refusal(tmp_path):
    columns=({'name':'text','dtype':'string','nullable':False},)
    store=ColumnarStore(columns,directory=tmp_path);store.append(('x'*1_047_000,),'one')
    rows,ids=store.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64))
    config={'operations':[operation('new','"'+'a'*3000+'"','string')]}
    try:
        with pytest.raises(ValueError) as reference:apply(source,config,lambda:None)
        with pytest.raises(ValueError) as shared:apply(source,config,lambda:None,share=True)
        assert str(shared.value)==str(reference.value)
    finally:store.close()


def test_regex_worker_cannot_change_unmapped_input_fields(tmp_path,monkeypatch):
    from runner import dataflow_formula
    columns=({'name':'text','dtype':'string','nullable':False},)
    store=ColumnarStore(columns,directory=tmp_path);store.append(('original',),'one')
    rows,ids=store.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64))
    # Lookahead deliberately stays on the supervised reference path. Simple
    # ASCII-prefix patterns now qualify for native execution and have no worker.
    config={'operations':[operation('hit','regex_match([text], "^(?=o)", "search", "")','bool')]}
    monkeypatch.setattr(dataflow_formula,'_supervised',lambda *args,**kwargs:[['altered',True]])
    try:
        for shared in (False,True):
            with pytest.raises(ValueError,match='unmapped input'):apply(source,config,lambda:None,share=shared)
    finally:store.close()


def test_literal_copy_and_coalesce_formula_stays_in_native_batches(tmp_path,monkeypatch):
    from runner import dataflow_formula
    columns=({'name':'n','dtype':'int64','nullable':True},{'name':'text','dtype':'string','nullable':True})
    source_store=ColumnarStore(columns,directory=tmp_path)
    for i in range(9000):source_store.append((None if i%2 else i,None if i%3 else 'é😀'),str(i))
    rows,ids=source_store.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64))
    config={'operations':[operation('constant','"purchase"','string'),operation('copied','[text]','string'),operation('filled','coalesce([n], 0)','int64')]}
    expected=apply(source,config,lambda:None)
    def refuse(*a,**kw):raise AssertionError('Qualified Formula returned to Python payload rows')
    monkeypatch.setattr(dataflow_formula,'evaluate_row',refuse)
    monkeypatch.setattr(ColumnarStore,'append',refuse)
    actual=None
    try:
        actual=apply(source,config,lambda:None,share=True)
        assert actual==expected and actual.rows.store.parent is source_store
    finally:
        if actual:actual.rows.store.close()
        expected.rows.store.close();source_store.close()
