import pytest
from runner import learning_gradients as lesson

def test_hand_calculated_two_layer_mean_influence():
 result=lesson.mean_derivatives((2,2,1),[1,2,-1,.5,.25,-.5,2,-3,.75],[[1,-2],[.5,1]])
 assert result['output']==[[2.75],[7.75]]
 assert result['input']==[[2.5,1.25],[2.5,1.25]]
 assert result['firstWeight']==[[1.5,-1],[-2.25,1.5]]

def test_hand_calculated_sgd_fresh_memory_and_unequal_means():
 a=lesson.adjustment((1,1,1),[1,0,2,0],[[1],[3]],[[0],[0]],rate=.1,momentum=.5)
 assert a['objective']==20
 assert a['gradients']==[40,16,20,8]
 assert a['parameters']==[-3,-1.6,0,-.8]
 assert a['memory']==[40,16,20,8]
 uneven=lesson.adjustment((1,1,1),[1,0,2,0],[[1],[1],[3]],[[0],[0],[0]])
 assert uneven['objective']==pytest.approx(44/3)
 assert uneven['objective']!=20

def test_scalar_oracle_finite_difference_and_freeze_is_not_no_grad():
 p=[1,2,-1,.5,.25,-.5,2,-3,.75];rows=[[1,-2],[.5,1]]
 expected=lesson.mean_derivatives((2,2,1),p,rows)
 for j in range(4):
  hi=p.copy();lo=p.copy();hi[j]+=1e-5;lo[j]-=1e-5
  a=sum(r[0]for r in lesson.mean_derivatives((2,2,1),hi,rows)['output'])/2
  b=sum(r[0]for r in lesson.mean_derivatives((2,2,1),lo,rows)['output'])/2
  assert (a-b)/2e-5==pytest.approx(sum(expected['firstWeight'],[])[j],abs=1e-9)
 frozen=lesson.adjustment((1,1,1),[1,0,2,0],[[1],[3]],[[0],[0]],freeze_first=True)
 assert frozen['parameters'][:2]==[1,0]
 assert lesson.mean_derivatives((1,1,1),frozen['parameters'],[[1]])['input'][0][0]!=0
