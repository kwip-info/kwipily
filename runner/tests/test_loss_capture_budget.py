"""Bounded objective previews refuse before full tensor conversion."""
import json
import pytest
import torch
from runner import engine


def header():
 return dict(schemaVersion=2,runID='r',selection='firstRowsOfValidationPartition',predictionInput={'dtype':'float32','sampleShape':[2]},predictionRows=[],evaluationCount=2,captureStatus='captured')


def capture(limit=100000,raw=True,decode=None):
 return engine._objective_prediction_rows(torch.tensor([[1.,2.],[3.,4.]]),torch.tensor([[.25,-.5],[.75,1.]]),{'target':torch.tensor([0,1])},torch.tensor([.125,.375]),include_raw=raw,decode_axis=decode,header=header(),byte_limit=limit)


def test_geometry_refusal_precedes_any_tensor_conversion(monkeypatch):
 def forbidden(*args,**kwargs):raise AssertionError('Tensor conversion occurred before geometry refusal')
 monkeypatch.setattr(torch.Tensor,'tolist',forbidden)
 with pytest.raises(ValueError,match='prediction-row capture'):
  capture(128)


@pytest.mark.parametrize('raw',[True,False])
@pytest.mark.parametrize('decode',[None,-1,0])
def test_exact_json_fit_and_one_byte_over(raw,decode):
 expected=[{'input':[1.,2.],'output':[.25,-.5],'targets':{'target':0}}, {'input':[3.,4.],'output':[.75,1.],'targets':{'target':1}}]
 if raw:
  for row,value in zip(expected,[.125,.375]):row['objectiveOutput']=value
 if decode is not None:
  for row,value in zip(expected,[0,1]):row['decoded']=value
 artifact={**header(),'predictionRows':expected}
 size=len(json.dumps(artifact,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
 assert capture(size,raw,decode)==expected
 with pytest.raises(ValueError,match='prediction-row capture'):capture(size-1,raw,decode)


def test_nested_raw_and_integer_targets_preserve_geometry():
 x=torch.tensor([[1.,2.]])
 y=torch.tensor([[[.25,.5],[.75,1.]]])
 result=engine._objective_prediction_rows(x,y,{'target':torch.tensor([[1,2]])},y,include_raw=True,decode_axis=-1,header=header(),byte_limit=2000)
 assert result==[{'input':[1.,2.],'output':[[.25,.5],[.75,1.]],'targets':{'target':[1,2]},'objectiveOutput':[[.25,.5],[.75,1.]],'decoded':[1,1]}]


def test_exact_number_scan_is_chunk_bounded_and_refuses_before_row_materialization(monkeypatch):
 original=torch.Tensor.tolist;seen=[]
 def bounded(value):
  seen.append(tuple(value.shape))
  assert value.numel()<=4096
  return original(value)
 monkeypatch.setattr(torch.Tensor,'tolist',bounded)
 with pytest.raises(ValueError,match='prediction-row capture'):
  engine._objective_prediction_rows(torch.full((1,5000),1e30),torch.zeros((1,2)),{'target':torch.tensor([0])},torch.zeros(1),include_raw=False,decode_axis=None,header=header(),byte_limit=12000)
 assert seen and all(len(shape)==1 for shape in seen)


def test_output_geometry_not_just_header_refuses_before_conversion(monkeypatch):
 def forbidden(*args,**kwargs):raise AssertionError('Tensor conversion preceded shape admission')
 monkeypatch.setattr(torch.Tensor,'tolist',forbidden)
 with pytest.raises(ValueError,match='prediction-row capture'):
  engine._objective_prediction_rows(torch.zeros((2,2)),torch.zeros((2,20,20)),{'target':torch.zeros((2,20),dtype=torch.int64)},torch.zeros((2,20)),include_raw=True,decode_axis=-1,header=header(),byte_limit=512)


def test_sequence_preview_honors_admitted_prefix_instead_of_all_validation_rows():
 rows=engine._objective_prediction_rows(torch.ones((200,32),dtype=torch.int64),torch.zeros((200,32,89)),{'target':torch.ones((200,32),dtype=torch.int64)},torch.zeros((200,32)),include_raw=False,decode_axis=-1,header=header(),row_limit=8)
 assert len(rows)==8
 assert len(rows[0]['output'])==32 and len(rows[0]['output'][0])==89
 assert len(json.dumps({**header(),'predictionRows':rows},separators=(',',':')).encode())<1_000_000
