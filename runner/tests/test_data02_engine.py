"""DATA02 numerical/security oracles authored with current tools.

Coverage retained: expression complements/IDs/nulls, typed Formula arithmetic,
regex captures/flags/group-zero/replacement limits, supervised timeout and
cancellation, exact portable replay, output privacy/grants/atomic rollback,
cache side-effect refusal and empty-branch reuse. No tests were removed.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import pytest
from runner import dataflow as d
from runner.dataflow_cli import run_dataflow
from runner.tests.test_columns import config as columns_config,item
from runner.tests.test_filter_rows import config as filter_config


def fixture(tmp_path,op,config,rows='text,value\nAB12,2\nCD34,5\nno,9\n,1\n'):
 path=tmp_path/'source.csv';path.write_text(rows)
 graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{}},{'id':'cast','operator':'kwip.data.columns','config':columns_config([item('text'),item('value',dtype='int64')])},{'id':'tool','operator':'kwip.data.'+({'filterExpression':'filterRows','regexMatch':'filterRows','calculate':'formula','regexExtract':'formula','regexReplace':'formula'}.get(op,op)),'config':config}], 'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'cast','port':'table'}},{'from':{'node':'cast','port':'table'},'to':{'node':'tool','port':'table'}}]}
 return graph,{'s':str(path)}

def config(op,**changes):
 """Test case labels select explicit current configs, never retired descriptors."""
 if op=='outputDelimited':return {**d.DEFINITIONS[op]['defaultConfig'],**changes}
 if op=='filterExpression':return filter_config(mode='expression',expression=changes.get('expression','[value] > 0'),nullPolicy=changes.get('nullPolicy','error'))
 if op=='regexMatch':return filter_config(**dict(dict(mode='pattern',column='text',pattern='[A-Za-z]+'),**changes))
 if op=='calculate':
  c=dict(expression='[value] * 2',outputColumn='derived',dtype='float64',replaceExisting=False,nullPolicy='error');c.update(changes)
  expression=c['expression'];outputs=[dict(name=c['outputColumn'],dtype=c['dtype'],write='replace' if c['replaceExisting'] else 'new')]
 else:
  c=dict(column='text',pattern='([A-Za-z]+)' if op=='regexExtract' else ' +',flags=[],nullPolicy='preserve',replaceExisting=False)
  c.update(dict(groups=[dict(group='1',output='extracted')],noMatchPolicy='null') if op=='regexExtract' else dict(replacement=' ',replaceMode='all',outputColumn='cleaned'));c.update(changes)
  quote=lambda value:json.dumps(value,ensure_ascii=False)
  if op=='regexExtract':
   expression='regex_extract('+', '.join(['['+c['column']+']',*map(quote,[c['pattern'],'search',c['noMatchPolicy'],'|'.join(c['flags']),*[group['group'] for group in c['groups']]])])+')'
   outputs=[dict(name=group['output'],dtype='string',write='replace' if c['replaceExisting'] else 'new') for group in c['groups']]
  else:
   expression='regex_replace('+', '.join(['['+c['column']+']',*map(quote,[c['pattern'],c['replacement'],c['replaceMode'],'|'.join(c['flags'])])])+')'
   outputs=[dict(name=c['outputColumn'],dtype='string',write='replace' if c['replaceExisting'] else 'new')]
 return dict(operations=[dict(id='checked-operation',expression=expression,outputs=outputs,nullPolicy=c['nullPolicy'])])

def test_expression_partitions_and_identity(tmp_path):
 g,b=fixture(tmp_path,'filterExpression',config('filterExpression',expression='[value] >= 5 AND [text] IS NOT NULL'))
 out=d.execute(g,b);assert list(out['tool']['matching'].rows)==[('CD34',5),('no',9)]
 assert list(out['tool']['nonmatching'].rows)==[('AB12',2),(None,1)]
 assert set(out['tool']['matching'].row_ids)|set(out['tool']['nonmatching'].row_ids)==set(out['s']['table'].row_ids)


def test_calculated_column(tmp_path):
 g,b=fixture(tmp_path,'calculate',config('calculate',expression='([value] + 1) * 3',dtype='int64'))
 out=d.execute(g,b);assert out['tool']['table'].values('derived')==[9,18,30,6]
 assert list(out['tool']['table'].row_ids)==list(out['s']['table'].row_ids)

@pytest.mark.parametrize('expr', ['__import__("os")','[missing] + 1','[text] + 1','[value] / 0','[value] ** 2','[value] > true'])
def test_invalid_expressions(tmp_path,expr):
 g,b=fixture(tmp_path,'calculate',config('calculate',expression=expr))
 with pytest.raises(ValueError):d.execute(g,b)


def test_regex_match_extract_replace(tmp_path):
 for op,changes,expected in [('regexMatch',{'pattern':'^[A-Z]+[0-9]+$','nullPolicy':'nonmatching'},None),('regexExtract',{'pattern':'(?P<letters>[A-Z]+)([0-9]+)','groups':[{'group':'letters','output':'prefix'},{'group':'2','output':'number'}]},['AB','CD',None,None]),('regexReplace',{'pattern':'[0-9]+','replacement':'#'},['AB#','CD#','no',None])]:
  g,b=fixture(tmp_path,op,config(op,**changes));out=d.execute(g,b)['tool']
  if op=='regexMatch':assert out['matching'].values('value')==[2,5];assert out['nonmatching'].values('value')==[9,1]
  else:assert out['table'].values('prefix' if op=='regexExtract' else 'cleaned')==expected


def test_output_deferred_and_private(tmp_path):
 dest=tmp_path/'destination';dest.mkdir();g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited',columns=['value','text']))
 out=d.execute(g,b,destination_bindings={'tool':str(dest)})
 assert list(csv.reader((dest/'output.csv').open()))==[['value','text'],['2','AB12'],['5','CD34'],['9','no'],['1','']]
 with pytest.raises(ValueError,match='exist'):d.execute(g,b,destination_bindings={'tool':str(dest)})
 assert list(out['tool']['table'].row_ids)==list(out['cast']['table'].row_ids)


def test_later_failure_leaves_no_output(tmp_path):
 dest=tmp_path/'destination';dest.mkdir();g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited'))
 g['nodes'].append({'id':'fail','operator':'kwip.data.formula','config':config('calculate',expression='1 / 0')});g['edges'].append({'from':{'node':'tool','port':'table'},'to':{'node':'fail','port':'table'}})
 with pytest.raises(ValueError):d.execute(g,b,destination_bindings={'tool':str(dest)})
 assert list(dest.iterdir())==[]


def test_portable_replay_outputs_no_private_bindings(tmp_path):
 dest=tmp_path/'destination';dest.mkdir();g,b=fixture(tmp_path,'regexReplace',config('regexReplace',pattern='[0-9]+',replacement='#'))
 g['nodes'].append({'id':'sink','operator':'kwip.data.outputDelimited','config':config('outputDelimited')});g['edges'].append({'from':{'node':'tool','port':'table'},'to':{'node':'sink','port':'table'}})
 graph=tmp_path/'graph.json';graph.write_text(json.dumps(g));archive=tmp_path/'archive';events=[]
 run_dataflow(graph,b,archive,events.append,destination_bindings={'sink':str(dest)})
 assert [e for e in events if e['event']=='dataflowOutputSaved'][0]['rowCount']==4
 assert str(dest) not in (archive/'dataflow.json').read_text()
 assert 'dataflow_expression.py' in [a['path'] for a in json.loads((archive/'manifest.json').read_text())['artifacts']]
 other=tmp_path/'other';other.mkdir();bindings=tmp_path/'sources.json';bindings.write_text(json.dumps(b));destinations=tmp_path/'destinations.json';destinations.write_text(json.dumps({'sink':str(other)}))
 replay=subprocess.run([sys.executable,'-I',str(archive/'prep.py'),'--source-bindings',str(bindings),'--destination-bindings',str(destinations),'--output',str(tmp_path/'replay')],capture_output=True,text=True)
 assert replay.returncode==0,replay.stderr
 assert (other/'output.csv').read_bytes()==(dest/'output.csv').read_bytes()


def test_regex_catastrophic_pattern_is_supervised():
 from runner.dataflow_regex import supervised
 import time
 started=time.monotonic()
 with pytest.raises(ValueError,match='budget'):supervised('regexMatch',config('regexMatch',pattern='(a+)+$'),['a'*5000+'!'],lambda:None,timeout=.15)
 assert time.monotonic()-started<2


def test_regex_cancellation_reaps_worker():
 from runner.dataflow_regex import supervised
 import time
 started=time.monotonic()
 def cancelled():
  if time.monotonic()-started>.1:raise KeyboardInterrupt('cancelled')
 with pytest.raises(KeyboardInterrupt):supervised('regexMatch',config('regexMatch',pattern='(a+)+$'),['a'*5000+'!'],cancelled)
 assert time.monotonic()-started<2

@pytest.mark.parametrize('op,changes', [('regexExtract',{'groups':[{'group':'missing','output':'x'}]}),('regexReplace',{'replacement':r'\g<missing>'}),('regexMatch',{'pattern':'('}),('regexMatch',{'nullPolicy':'error'})])
def test_regex_refuses_invalid_configuration_or_missing(tmp_path,op,changes):
 g,b=fixture(tmp_path,op,config(op,**changes))
 with pytest.raises(ValueError):d.execute(g,b)


def test_cache_never_reuses_output_side_effect(tmp_path):
 dest=tmp_path/'dest';dest.mkdir();g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited'));graph=tmp_path/'g.json';graph.write_text(json.dumps(g));first=tmp_path/'first'
 run_dataflow(graph,b,first,lambda _:None,cache_policy='materialize',destination_bindings={'tool':str(dest)})
 (dest/'output.csv').unlink()
 second=run_dataflow(graph,b,tmp_path/'second',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest(),destination_bindings={'tool':str(dest)})
 assert 'tool' in second['cache']['executedNodeIDs'];assert 'tool' not in second['cache']['reusedNodeIDs'];assert (dest/'output.csv').exists()


def test_empty_branch_can_be_archived_and_cached(tmp_path):
 g,b=fixture(tmp_path,'filterExpression',config('filterExpression',expression='[value] > 100'));graph=tmp_path/'g.json';graph.write_text(json.dumps(g));first=tmp_path/'first'
 run_dataflow(graph,b,first,lambda _:None,cache_policy='materialize')
 second=run_dataflow(graph,b,tmp_path/'second',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest())
 assert 'tool' in second['cache']['reusedNodeIDs']

@pytest.mark.parametrize('updates', [{'filename':'../escape.csv'},{'filename':'/tmp/escape.csv'},{'columns':['missing']}])
def test_output_refuses_unsafe_targets(tmp_path,updates):
 dest=tmp_path/'dest';dest.mkdir();g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited',**updates))
 with pytest.raises(ValueError):d.execute(g,b,destination_bindings={'tool':str(dest)})
 assert list(dest.iterdir())==[]


def test_output_refuses_symlinks(tmp_path):
 dest=tmp_path/'dest';dest.mkdir();other=tmp_path/'other.csv';other.write_text('keep');(dest/'output.csv').symlink_to(other)
 g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited',overwrite=True))
 with pytest.raises(ValueError,match='symlink'):d.execute(g,b,destination_bindings={'tool':str(dest)})
 assert other.read_text()=='keep'


def test_output_rolls_back_overwrite_when_delivery_cancelled(tmp_path):
 dest=tmp_path/'dest';dest.mkdir();path=dest/'output.csv';path.write_text('previous')
 g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited',overwrite=True))
 def cancel(receipt):raise KeyboardInterrupt('cancel')
 with pytest.raises(KeyboardInterrupt):d.execute(g,b,destination_bindings={'tool':str(dest)},on_output_saved=cancel)
 assert path.read_text()=='previous';assert list(dest.iterdir())==[path]


def test_filter_null_policy_and_calculation_preserve(tmp_path):
 g,b=fixture(tmp_path,'filterExpression',config('filterExpression',expression='[text] = "AB12"',nullPolicy='nonmatching'))
 assert d.execute(g,b)['tool']['nonmatching'].values('value')==[5,9,1]
 g,b=fixture(tmp_path,'calculate',config('calculate',expression='[text]',dtype='string',nullPolicy='preserve'))
 assert d.execute(g,b)['tool']['table'].values('derived')==['AB12','CD34','no',None]


def test_typed_math_overflow_and_replace_policy(tmp_path):
 for changes in [{'expression':'9223372036854775807 + 1','dtype':'int64'},{'expression':'1e308 * 1e308'},{'expression':'1','dtype':'int64','outputColumn':'value'}]:
  g,b=fixture(tmp_path,'calculate',config('calculate',**changes))
  with pytest.raises(ValueError):d.execute(g,b)
 g,b=fixture(tmp_path,'calculate',config('calculate',expression='[value] + 1',dtype='int64',outputColumn='value',replaceExisting=True))
 assert d.execute(g,b)['tool']['table'].values('value')==[3,6,10,2]


def test_regex_group_zero_flags_first_and_all(tmp_path):
 g,b=fixture(tmp_path,'regexExtract',config('regexExtract',pattern='(ab)',flags=['ignoreCase'],groups=[{'group':'0','output':'whole'}]))
 assert d.execute(g,b)['tool']['table'].values('whole')==['AB',None,None,None]
 for mode,expected in [('first','Xb ab'),('all','Xb Xb')]:
  g,b=fixture(tmp_path,'regexReplace',config('regexReplace',pattern='a',replacement='X',replaceMode=mode),rows='text,value\nab ab,1\n')
  assert d.execute(g,b)['tool']['table'].values('cleaned')==[expected]


def test_output_missing_binding_before_source_read(tmp_path):
 g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited'));Path(b['s']).unlink()
 with pytest.raises(ValueError,match='Destination bindings'):d.execute(g,b)


def test_output_archive_failure_rolls_back(tmp_path):
 dest=tmp_path/'dest';dest.mkdir();g,b=fixture(tmp_path,'outputDelimited',config('outputDelimited'));graph=tmp_path/'g.json';graph.write_text(json.dumps(g))
 def fail(event):
  if event['event']=='dataflowCompleted':raise KeyboardInterrupt('cancel delivery')
 with pytest.raises(KeyboardInterrupt):run_dataflow(graph,b,tmp_path/'archive',fail,destination_bindings={'tool':str(dest)})
 assert list(dest.iterdir())==[];assert not (tmp_path/'archive'/'manifest.json').exists()
