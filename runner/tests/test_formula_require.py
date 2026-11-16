"""A reusable report must refuse unmet input assumptions, including missing conditions."""
import pytest
from runner.dataflow_formula import FormulaExpression


def value(expression,columns=(),row=()):
 return FormulaExpression(expression,list(columns)).evaluate(list(row))


def test_require_passes_original_typed_value_only_for_true():
 for expression,want in [('require(true, "2011-01")','2011-01'),('require(2 = 2, 8)',8),('require(true, false)',False),('require(true, null)',None)]:
  assert value(expression)==want
 for condition in ['false','null']:
  with pytest.raises(ValueError,match='require condition must be true'):value(f'require({condition}, "kept")')


@pytest.mark.parametrize('expression',['require(1, "x")','require("true", 1)','require(true)','require(true, 1, 2)'])
def test_require_rejects_ambiguous_condition_or_arity(expression):
 with pytest.raises(ValueError):value(expression)


def test_require_checks_every_row_and_preserves_value_type():
 columns=[{'name':'months','dtype':'int64','nullable':False},{'name':'label','dtype':'string','nullable':False}]
 formula=FormulaExpression('require([months] = 2, [label])',columns)
 assert formula.dtype=='string'
 assert formula.evaluate([2,'2011-03'])=='2011-03'
 with pytest.raises(ValueError,match='require condition must be true'):formula.evaluate([3,'2011-03'])
