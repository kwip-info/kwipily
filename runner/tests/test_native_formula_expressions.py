"""Qualified native expressions preserve exact decimal text and regex semantics."""
from dataclasses import replace
import pytest
from runner import dataflow as df,dataflow_formula as formula
from runner.dataflow_columnar import ColumnarStore
from runner.dataflow_disk import disk_scope

@pytest.mark.parametrize('expression,dtype',[
 ('decimal_parse([value])','string'),('decimal_is_finite([value])','bool'),
 ('decimal_is_integer([value])','bool'),('decimal_abs([value])','string'),
 ('decimal_multiply([value], "1.95")','string'),('decimal_add([value], "0.025")','string'),
 ('decimal_subtract([value], "0.025")','string'),('decimal_compare([value], "2")','int64'),
 ('length(trim([value])) = 1','bool'),
 ('regex_match([value], "^[0-9]", "search", "")','bool'),
 ('regex_extract([value], "^([0-9]{1,4})", "search", "null", "", "1")','string'),
])
def test_native_expression_matches_reference_without_row_evaluation(tmp_path,monkeypatch,expression,dtype):
    values=['1.95','2.00','-0.00','0.001','1234','',None]
    if expression.startswith(('decimal_multiply','decimal_abs','decimal_add','decimal_subtract','decimal_compare')):values=values[:5]+[None]
    with disk_scope(directory=tmp_path):
        columns=({'name':'value','dtype':'string','nullable':True},)
        store=ColumnarStore(columns,directory=tmp_path)
        for i,value in enumerate(values):store.append((value,),f'r:{i}')
        rows,ids=store.finish();table=df.derive(df.Table(columns,rows,ids,'a'*64))
        config={'operations':[{'id':'one','expression':expression,'outputs':[{'name':'result','dtype':dtype,'write':'new'}],'nullPolicy':'preserve'}]}
        expected=formula.apply(replace(table,rows=tuple(table.rows),row_ids=tuple(table.row_ids)),config,lambda:None)
        def forbidden(*a,**k):raise AssertionError('Qualified expression evaluated Python rows')
        monkeypatch.setattr(formula,'evaluate_row',forbidden);monkeypatch.setattr(formula,'_supervised',forbidden)
        actual=formula.apply(table,config,lambda:None,share=True)
        assert tuple(actual.rows)==tuple(expected.rows)
        assert actual.data_version==expected.data_version
        actual.rows.store.close();store.close();expected.rows.store.close()


def test_native_decimal_seeded_values_match_exact_reference():
    import random
    import polars as pl
    from runner.dataflow_formula import FormulaExpression as Expression
    from runner.native_formula_expression import compile_native
    randomizer=random.Random(70821)
    values=[]
    for _ in range(500):
        scale=randomizer.randrange(0,9);digits=str(randomizer.randrange(-10**14,10**14))
        if scale:digits=digits[:-scale]+'.'+digits[-scale:] if len(digits.lstrip('-'))>scale else '0.'+str(abs(int(digits))).zfill(scale)
        values.append(digits)
    columns=({'name':'value','dtype':'string','nullable':True},);frame=pl.DataFrame({'c0':values})
    for text in ('decimal_parse([value])','decimal_multiply([value], "-1.95")','decimal_add([value], "0.000000025")','decimal_compare([value], "0")'):
        expression=Expression(text,columns);native=compile_native(expression,expression.tree,frame)
        assert native is not None
        assert frame.select(native.alias('r')).get_column('r').to_list()==[expression.evaluate((value,)) for value in values]


@pytest.mark.parametrize('value',['1e3','١٢٣.٤','123456789012345678901234567890123456789','0.'+'0'*38+'1'])
def test_broad_decimal_inputs_keep_reference_fallback(value):
    import polars as pl
    from runner.dataflow_formula import FormulaExpression as Expression
    from runner.native_formula_expression import compile_native
    expression=Expression('decimal_parse([value])',({'name':'value','dtype':'string','nullable':True},))
    assert compile_native(expression,expression.tree,pl.DataFrame({'c0':[value]})) is None


@pytest.mark.parametrize('pattern',['(?=a)a','a$','(a)\\1','a*'])
def test_unqualified_regex_keeps_supervised_reference(pattern):
    import json
    import polars as pl
    from runner.dataflow_formula import FormulaExpression as Expression
    from runner.native_formula_expression import compile_native
    expression=Expression('regex_match([value], '+json.dumps(pattern)+', "search", "")',({'name':'value','dtype':'string','nullable':True},))
    assert compile_native(expression,expression.tree,pl.DataFrame({'c0':['a']})) is None
