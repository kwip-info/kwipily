"""Actual paired teaching examples; independent expected values, no completion grader."""
import csv
import hashlib
import json
from pathlib import Path
import pytest
from runner import dataflow
from runner.dataflow_cli import run_dataflow
ROOT=Path(__file__).resolve().parents[1]/'dataflow_examples'

@pytest.mark.parametrize('lesson',['expression','calculate','regex-match','regex-extract','regex-replace','output'])
def test_paired_example_executes_real_rows(lesson,tmp_path):
 graph=json.loads((ROOT/f'data02-{lesson}.json').read_text())
 destination=tmp_path/'destination';destination.mkdir()
 outputs=dataflow.execute(graph,{'source':str(ROOT/'data02-observations.csv')},destination_bindings={'tool':str(destination)} if lesson=='output' else {})
 result=outputs['tool'];source=outputs['source']['table']
 assert [column['name'] for column in outputs['cast']['table'].columns]==[column['name'] for column in source.columns]
 if lesson=='expression':assert result['matching'].values('order_id')==['A01','A04','A07'];assert result['nonmatching'].values('order_id')==['A02','A03','A05','A06','A08']
 elif lesson=='calculate':assert result['table'].values('order_total')==[25.,80.,20.,60.,10.,0.,60.,6.]
 elif lesson=='regex-match':assert result['matching'].values('order_id')==['A01','A02','A04','A06','A07','A08'];assert result['nonmatching'].values('order_id')==['A03','A05']
 elif lesson=='regex-extract':assert result['table'].values('email_domain')==['example.com','example.org',None,'north.example',None,'example.com','example.org','example.com']
 elif lesson=='regex-replace':assert result['table'].values('clean_note')==['paid today','waiting review','paid','paid yesterday','missing email','refunded','paid',' paid ']
 else:
  with (destination/'prepared-orders.csv').open(newline='') as stream:rows=list(csv.reader(stream,delimiter=';'))
  assert rows==[['order_id','quantity','unit_price'],['A01','2','12.5'],['A02','1','80.0'],['A03','4','5.0'],['A04','3','20.0'],['A05','1','10.0'],['A06','0','9.0'],['A07','2','30.0'],['A08','1','6.0']]
 for value in result.values():assert value.source_digest==source.source_digest;assert set(value.row_ids)<=set(source.row_ids)
 if 'matching' in result:assert set(result['matching'].row_ids).isdisjoint(result['nonmatching'].row_ids);assert len(result['matching'].rows)+len(result['nonmatching'].rows)==8
 else:assert result['table'].row_ids==source.row_ids


def test_teaching_descriptor_is_complete_and_honest():
 catalog=json.loads((ROOT/'data02-lessons.json').read_text());assert catalog['schemaVersion']==1
 lessons=catalog['lessons'];assert len(lessons)==6;assert len({l['id'] for l in lessons})==6
 reviewed_versions={'data02-expression':4,'data02-calculate':3,'data02-regex-match':4,
                    'data02-regex-extract':3,'data02-regex-replace':3,'data02-output':3}
 assert {lesson['id'] for lesson in lessons}==set(reviewed_versions)
 assert {l['operatorID'] for l in lessons}=={'kwip.data.'+op for op in ['filterRows','formula','outputDelimited']}
 for lesson in lessons:
  assert lesson['mode']=='guidedPractice' and lesson['isAssessed'] is False
  for field in ['title','objective','terms','inspect','disclosure']:assert lesson[field]
  assert len(lesson['steps'])>=4
  assert (ROOT/lesson['graph']).is_file() and (ROOT/lesson['source']).is_file() and (ROOT/lesson['reading']).is_file()
  assert lesson['sourceNode']=='source' and lesson['contentVersion']==reviewed_versions[lesson['id']]
  graph=json.loads((ROOT/lesson['graph']).read_text())
  assert lesson['operatorID'] in [node['operator'] for node in graph['nodes']]
  assert all(node['operator'] in {'kwip.data.sourceCSV','kwip.data.columns','kwip.data.filterRows','kwip.data.formula','kwip.data.outputDelimited'} for node in graph['nodes'])


def test_integrated_teaching_sink_is_replayable_and_complete(tmp_path):
 destination=tmp_path/'destination';destination.mkdir();events=[]
 manifest=run_dataflow(ROOT/'data02-output.json',{'source':str(ROOT/'data02-observations.csv')},tmp_path/'archive',events.append,destination_bindings={'tool':str(destination)})
 receipt=next(e for e in events if e['event']=='dataflowOutputSaved')
 assert receipt['rowCount']==8 and receipt['completeSelectedTable'] is True
 assert receipt['sha256']==hashlib.sha256((destination/'prepared-orders.csv').read_bytes()).hexdigest()
 assert manifest['outputs'][0]['sha256']==receipt['sha256']
