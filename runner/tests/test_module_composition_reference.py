"""Hand-computed primary checks; finite differences are supplemental only."""
from dataclasses import FrozenInstanceError, replace
import inspect
import pytest
from runner import module_composition_reference as r

P = r.AffineParameters(2, 1, 3, -2)

@pytest.mark.parametrize('kind,output,dx,dp', [
    ('sequenceAB', 13, 18, (18, 9, 15, 3)),
    ('sequenceBA', 9, 18, (12, 3, 12, 6)),
    ('branch', 18, 24, (24, 12, 15, 3)),
])
def test_literal_outputs_weighted_input_and_every_weight_and_bias(kind, output, dx, dp):
    value = r.evaluate(kind, P, (2,), (3,))
    assert value.output == (output,)
    assert value.inputInfluence == (dx,)
    assert value.parameterInfluence == tuple(zip(('a', 'b', 'c', 'd'), dp))

def test_literal_outer_private_owner_gradients_keep_bias_roles_distinct():
    v = r.evaluate_outer(P, P, (2,), (3,))
    assert v.output == (32,)
    assert v.inputInfluence == (48,)
    assert v.parameterInfluence == tuple(zip(('left.a','left.b','left.c','left.d','right.a','right.b','right.c','right.d'), (24,12,15,3,24,-12,9,-3)))
    # Separate instances may start equal; their gradients must never be merged as aliases.
    assert dict(v.parameterInfluence)['left.b'] != dict(v.parameterInfluence)['right.b']

def test_literal_left_only_mean_mse_sgd_leaves_private_right_untouched():
    step = r.left_sgd_step(P, P, (2,), (0,))
    assert step.loss == 324
    assert step.gradients == tuple(zip(('left.a','left.b','left.c','left.d'), (288,144,180,36)))
    assert (step.left.a,step.left.b,step.left.c,step.left.d) == pytest.approx((-12.4,-6.2,-6,-3.8))
    assert step.right == P
    assert P == r.AffineParameters(2,1,3,-2)

def test_four_immutable_fixtures_and_wrong_order_or_dropped_bias_are_observable():
    assert len(r.FIXTURES) == 4 and len({f.id for f in r.FIXTURES}) == 4
    expected = ((-14,6,26),(7,1.75,-3.5),(3.75,1.75,1),(-1.4375,3.25,7.9375))
    for f, branch in zip(r.FIXTURES, expected):
        assert r.evaluate('branch',f.parameters,f.rows,f.upstream).output == branch
        ab = r.evaluate('sequenceAB',f.parameters,f.rows,f.upstream)
        assert ab.output != r.evaluate('sequenceBA',f.parameters,f.rows,f.upstream).output
        assert ab.output != r.evaluate('sequenceAB',replace(f.parameters,b=0),f.rows,f.upstream).output
    with pytest.raises(FrozenInstanceError):
        r.FIXTURES[0].parameters.a = 7
    assert all(name not in inspect.getsource(r) for name in ('import torch','import numpy','from runner','autograd'))

def test_supplemental_central_differences_nonuniform_rows_and_upstream():
    h = 1e-5
    for f in r.FIXTURES:
        for kind in ('sequenceAB','sequenceBA','branch'):
            value = r.evaluate(kind,f.parameters,f.rows,f.upstream)
            def weighted(p,rows):
                return sum(y*u for y,u in zip(r.evaluate(kind,p,rows,f.upstream).output,f.upstream))
            for name, influence in value.parameterInfluence:
                plus=replace(f.parameters,**{name:getattr(f.parameters,name)+h})
                minus=replace(f.parameters,**{name:getattr(f.parameters,name)-h})
                assert influence == pytest.approx((weighted(plus,f.rows)-weighted(minus,f.rows))/(2*h),abs=1e-7)
            for index, influence in enumerate(value.inputInfluence):
                plus=list(f.rows);minus=list(f.rows);plus[index]+=h;minus[index]-=h
                assert influence == pytest.approx((weighted(f.parameters,plus)-weighted(f.parameters,minus))/(2*h),abs=1e-7)

@pytest.mark.parametrize('rows,upstream', [((),()),((1,),(1,2)),((True,),(1,)),((float('nan'),),(1,)),((1,),(float('inf'),)),((1,)*33,(1,)*33)])
def test_invalid_or_unbounded_inputs_refuse(rows,upstream):
    with pytest.raises((ValueError,TypeError)):
        r.evaluate('branch',P,rows,upstream)

def test_invalid_kind_parameters_target_counts_and_learning_rate_refuse():
    with pytest.raises(ValueError):r.evaluate('wrong',P,(1,),(1,))
    with pytest.raises((ValueError,TypeError)):r.evaluate('branch',r.AffineParameters(True,1,2,3),(1,),(1,))
    with pytest.raises(ValueError):r.left_sgd_step(P,P,(1,2),(0,))
    for rate in (True,0,-1,float('nan'),2):
        with pytest.raises((ValueError,TypeError)):r.left_sgd_step(P,P,(1,),(0,),learning_rate=rate)

def test_literal_uneven_weighted_mean_update_normalizes_observations_once():
    step=r.left_sgd_step(P,P,(2,3),(0,2))
    assert step.loss == 450
    assert step.gradients == tuple(zip(('left.a','left.b','left.c','left.d'),(432,168,258,42)))
    assert (step.left.a,step.left.b,step.left.c,step.left.d) == pytest.approx((-19.6,-7.4,-9.9,-4.1))
    assert step.right == P

def test_private_instances_with_distinct_coefficients_use_their_own_values():
    right=r.AffineParameters(-1,.5,2,1)
    value=r.evaluate_outer(P,right,(2,),(3,))
    assert value.output == (9.5,)
    assert value.inputInfluence == (15,)
    assert dict(value.parameterInfluence)['left.c'] == 15
    assert dict(value.parameterInfluence)['right.c'] == -7.5
