"""Numeric predicates describe represented values, never arbitrary decimal truth."""
import math
from decimal import Decimal
import pytest
from runner import dataflow as df, dataflow_formula as formula, dataflow_columns as columns


def operation(expression, name, dtype='bool', policy='preserve'):
    return {'id':name,'expression':expression,'outputs':[{'name':name,'dtype':dtype,'write':'new'}],'nullPolicy':policy}


@pytest.mark.parametrize('dtype,values,expected',[
    ('int64',[0,-1,2**63-1,-2**63],[True,True,True,True]),
    ('float64',[0.0,-0.0,-3.0,1.25,1e100],[True,True,True,False,True]),
])
def test_integrality_uses_no_int64_conversion(dtype,values,expected):
    table=df.derive(df.Table(({'name':'value','dtype':dtype,'nullable':True},),tuple((v,) for v in values+[None]),tuple(str(i) for i in range(len(values)+1)),'a'*64))
    out=formula.apply(table,{'operations':[operation('is_integer([value])','integral'),operation('is_finite([value])','finite')]},lambda:None)
    assert [r[-2] for r in out.rows]==expected+[None]
    assert [r[-1] for r in out.rows]==[True]*len(values)+[None]
    assert out.columns[-1]['dtype']=='bool'
    assert list(out.row_ids)==list(table.row_ids)


@pytest.mark.parametrize('name',['is_finite','is_integer'])
def test_predicates_refuse_text_boolean_wrong_arity_and_do_not_mask_expression_errors(name):
    for expression,cols in [(f'{name}("3")',[]),(f'{name}(true)',[]),(f'{name}()',[]),(f'{name}(1,2)',[])]:
        with pytest.raises(ValueError):formula.FormulaExpression(expression,cols)
    with pytest.raises(ValueError):formula.FormulaExpression(f'{name}(1 / 0)',[]).evaluate([])
    predicate=formula.FormulaExpression(f'{name}([value])',[{'name':'value','dtype':'float64'}])
    # Defensive scalar semantics; nonfinite numbers are not admitted table values.
    for value in [math.inf,-math.inf,math.nan]:assert predicate.evaluate([value]) is False
    table=df.derive(df.Table(({'name':'value','dtype':'float64','nullable':True},),((None,),),('n',),'a'*64))
    with pytest.raises(ValueError):formula.apply(table,{'operations':[operation(f'{name}([value])','result',policy='error')]},lambda:None)


def test_columns_failed_numeric_parse_becomes_null_without_losing_originals():
    raw=['12.00','1.5','NaN','Infinity','1e999','invalid','-0','9007199254740993']
    table=df.derive(df.Table(({'name':'original','dtype':'string','nullable':False},),tuple((v,) for v in raw),tuple(map(str,range(len(raw)))),'a'*64))
    copied=formula.apply(table,{'operations':[operation('[original]','parsed','string')]},lambda:None)
    config={'rows':[{'source':'parsed','included':True,'outputName':'parsed','targetType':'float64'}],'newColumns':'keep','castFailure':'null','trimText':False,'missingPolicy':'preserve'}
    converted=columns.apply(copied,config,lambda:None)
    result=formula.apply(converted,{'operations':[operation('is_finite([parsed])','finite'),operation('is_integer([parsed])','integral')]},lambda:None)
    rows=list(result.rows)
    assert [r[1] for r in rows]==raw
    assert [r[0] for r in rows][:7]==[12.,1.5,None,None,None,None,-0.]
    assert [r[-2:] for r in rows][:7]==[(True,True),(True,False),(None,None),(None,None),(None,None),(None,None),(True,True)]
    assert rows[-1][0]==9007199254740992.0  # Float parsing is not exact decimal/int parsing.
    assert rows[-1][-1] is True             # Predicate reports represented float, not source precision.
    config['castFailure']='error'
    with pytest.raises(ValueError):list(columns.apply(copied,config,lambda:None).rows)


def test_float_price_example_is_bounded_tolerance_not_financial_precision():
    amount=3*float('12.34')
    assert abs(Decimal(str(amount))-Decimal('37.02'))<=Decimal('0.0000001')
    rounded=float('9007199254740992.5')
    assert rounded.is_integer() and Decimal('9007199254740992.5')%1 != 0
