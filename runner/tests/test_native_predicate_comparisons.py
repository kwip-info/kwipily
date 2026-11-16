"""Typed predicate qualification matches the scalar language without numeric coercion."""
import itertools
import pytest
import polars as pl
from runner.dataflow_expression import Expression
from runner.dataflow_shared import _boolean_kernel

COLUMNS=tuple({"name":name,"dtype":dtype,"nullable":True} for name,dtype in
              (("i","int64"),("j","int64"),("x","float64"),("y","float64"),("s","string"),("t","string"),("b","bool")))
ROWS=[(i,j,x,y,s,t,b) for i,j,x,y,s,t,b in zip(
    [None,-2**63,2**63-1,2**53+1,0,1,-1],
    [0,2**63-1,2**53,2**53,0,None,-2],
    [None,-0.0,1.0,1e300,-1e-300,2.0,0.0],
    [0.0,0.0,2.0,1e299,None,2.0,-0.0],
    [None,"","a","é","😀","z","a"],
    ["",None,"b","z","é","z","a"],
    [None,True,False,True,None,False,True])]

@pytest.mark.parametrize("text",[
    "[i] = [j]", "[i] != [j]", "[i] < [j]", "[i] >= 0",
    "[x] <= [y]", "[x] > 0.0", "[x] = 0", "[i] = null",
    "[s] IS NULL", "[s] IS NOT NULL", "[s] = [t]", "[s] < [t]",
    "[s] >= \"é\"", "[b] = false", "[i] IS NOT NULL AND [i] > 0",
    "NOT ([s] IS NULL OR [s] = \"\")", "[b] OR ([i] = [j])",
])
def test_native_predicate_matches_reference_exactly(text):
    reference=Expression(text,COLUMNS)
    kernel=_boolean_kernel(text,COLUMNS)
    assert kernel is not None, text
    frame=pl.DataFrame(ROWS,schema=[("c"+str(i), {"int64":pl.Int64,"float64":pl.Float64,"string":pl.String,"bool":pl.Boolean}[c["dtype"]]) for i,c in enumerate(COLUMNS)],orient="row",strict=True)
    assert frame.select(kernel.alias("match")).get_column("match").to_list()==[reference.evaluate(row) for row in ROWS]

@pytest.mark.parametrize("text",["[i] = [x]", "[i] < 9007199254740992.0", "[x] = 9007199254740993", "[i] + 1 > 0", "[s]", "[i]"])
def test_unqualified_or_nonboolean_predicate_keeps_reference(text):
    assert _boolean_kernel(text,COLUMNS) is None
