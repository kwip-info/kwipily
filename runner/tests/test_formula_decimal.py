"""Original text precision is never inferred from an already-rounded binary float."""
import pytest
from runner import dataflow as df, dataflow_formula as formula
from runner.dataflow_aggregate import aggregate_table


def evaluate(expression, values=(), columns=()):
    return formula.FormulaExpression(expression, list(columns)).evaluate(list(values))


def test_decimal_predicates_preserve_source_fraction_and_finiteness():
    column = [{'name':'raw','dtype':'string','nullable':True}]
    cases = [('9007199254740992.5',False),('1.00000000000000000001',False),
             ('1e-400',False),('-1e-400',False),('1e400',True),('12.000',True)]
    for text, integral in cases:
        assert evaluate('decimal_is_integer([raw])',[text],column) is integral
        assert evaluate('decimal_is_finite([raw])',[text],column) is True
    for text in ['NaN','Infinity','-Infinity','bad','']:
        assert evaluate('decimal_is_finite([raw])',[text],column) is False
        assert evaluate('decimal_is_integer([raw])',[text],column) is False
        assert evaluate('decimal_parse([raw])',[text],column) is None
    assert evaluate('decimal_parse([raw])',[None],column) is None


def test_decimal_arithmetic_retains_subcent_and_large_small_values():
    assert evaluate('decimal_multiply("3", "0.005")') == '0.015'
    assert evaluate('decimal_add("1e50", "0.015")') == '1'+'0'*50+'.015'
    assert evaluate('decimal_subtract("1.00000000000000000001", "1")') == '0.'+'0'*19+'1'
    assert evaluate('decimal_abs("-0012.3400")') == '12.34'
    assert evaluate('decimal_compare("9007199254740992.5", "9007199254740992")') == 1
    assert evaluate('decimal_compare("-0", "0")') == 0
    assert evaluate('decimal_parse(" 1.230e2 ")') == '123'
    with pytest.raises(ValueError):evaluate('decimal_multiply("not a number", "3")')


@pytest.mark.parametrize('expression',[
    'decimal_parse(1.2)','decimal_is_integer(true)','decimal_add("1", 2)',
    'decimal_compare("1")','decimal_multiply("1", "2", "3")'])
def test_decimal_calls_require_explicit_text_and_exact_arity(expression):
    with pytest.raises(ValueError):evaluate(expression)


def test_decimal_budgets_refuse_instead_of_rounding_or_misclassifying():
    for expression in ['decimal_is_finite("1e9999999999999999")',
                       'decimal_parse("1e20000")',
                       'decimal_multiply("1e16000", "1e16000")']:
        with pytest.raises(ValueError, match='budget|allowance|bound'):evaluate(expression)


def test_sum_decimal_is_exact_text_stable_and_honors_missing_policy():
    source=df.derive(df.Table(({'name':'value','dtype':'string','nullable':True},),
        (('1e50',),('0.015',),('-1e50',),(None,)),('a','b','c','d'),'a'*64))
    config={'groupBy':[],'nullKeys':'group','maxRows':100000,
            'aggregations':[{'output':'total','op':'sumDecimal','column':'value','nullPolicy':'skip'}]}
    result=aggregate_table(source,config)
    assert result.rows == (('0.015',),)
    assert result.columns[0]['dtype']=='string'
    reversed_source=df.derive(source,rows=source.rows[::-1],ids=source.row_ids[::-1])
    assert aggregate_table(reversed_source,config).rows==result.rows
    config['aggregations'][0]['nullPolicy']='propagate'
    assert aggregate_table(source,config).rows==((None,),)
    config['aggregations'][0]['nullPolicy']='error'
    with pytest.raises(ValueError,match='missing'):aggregate_table(source,config)


def test_sum_decimal_refuses_invalid_and_numeric_columns():
    config={'groupBy':[],'nullKeys':'group','maxRows':100000,
            'aggregations':[{'output':'total','op':'sumDecimal','column':'value','nullPolicy':'skip'}]}
    for dtype,value in [('string','NaN'),('string','bad'),('float64',1.2),('bool',True)]:
        source=df.derive(df.Table(({'name':'value','dtype':dtype,'nullable':False},),
            ((value,),),('a',),'a'*64))
        with pytest.raises(ValueError):aggregate_table(source,config)


def test_exact_operations_match_independent_rational_oracle_despite_ambient_context():
    from decimal import Decimal, localcontext
    from fractions import Fraction
    import random
    rng=random.Random(1909)
    with localcontext() as context:
        context.prec=2
        for _ in range(100):
            left=str(rng.randrange(-10**20,10**20))+'e'+str(rng.randrange(-40,41))
            right=str(rng.randrange(-10**20,10**20))+'e'+str(rng.randrange(-40,41))
            a,b=Fraction(Decimal(left)),Fraction(Decimal(right))
            for function,want in [('decimal_add',a+b),('decimal_subtract',a-b),('decimal_multiply',a*b)]:
                actual=evaluate(f'{function}("{left}","{right}")')
                assert Fraction(Decimal(actual))==want
