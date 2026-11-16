"""Registry discovery examples must execute under the real Formula contract."""
import json
from pathlib import Path
import pytest
from runner import dataflow as df, dataflow_formula as formula


def schema():
    registry=json.loads((Path(__file__).parents[1]/'dataflow-operators.json').read_text())
    return next(x for x in registry['operators'] if x['id']=='kwip.data.formula')['configSchema']


def test_discovery_lists_real_function_signatures_and_numeric_only_plus():
    expression=schema()['properties']['operations']['items']['properties']['expression']
    signatures=expression.get('x-functionSignatures',{})
    assert set(signatures)==set(formula.FUNCTIONS)
    assert 'numeric-only' in expression.get('description','')
    assert 'column-to-column' in expression['description']
    assert 'not' in schema().get('description','')


def test_advertised_suffix_example_executes_preserving_ids_and_nulls():
    config=schema().get('examples',[])[0]
    columns=({'name':'date_text','dtype':'string','nullable':True},)
    table=df.derive(df.Table(columns,(('2026-10-23',),('2026-10-24',),(None,)),('a','b','c'),'a'*64))
    result=formula.apply(table,config,lambda:None)
    assert [row[-1] for row in result.rows]==['2026-10-23T00:00:00Z','2026-10-24T00:00:00Z',None]
    assert list(result.row_ids)==['a','b','c']
    assert result.columns[-1]['dtype']=='string'


@pytest.mark.parametrize('expression',['[date_text] + "T00:00:00Z"','concat([date_text], "suffix")'])
def test_unsupported_text_arithmetic_and_concat_stay_refused(expression):
    config={'operations':[{'id':'bad','expression':expression,'outputs':[{'name':'out','dtype':'string','write':'new'}],'nullPolicy':'preserve'}]}
    with pytest.raises(ValueError):
        formula.compile_plan(({'name':'date_text','dtype':'string','nullable':False},),config)
