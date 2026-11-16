from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from runner.tests.test_dataflow_reshape import table,config
from runner.dataflow_reshape import reshape_table,verify_output

@pytest.mark.parametrize('op,expected',[('first',1),('last',3),('countRows',3),('countNonNull',2),('sum',4),('mean',2.0),('min',1),('max',3)])
def test_reducers_and_missing_semantics(op,expected):
    t=table(['id','key','v'],[(1,'x',1),(1,'x',None),(1,'x',3)],['int64','string','int64'])
    c=config(mode='wide',rowKeys=['id'],columnKey='key',duplicates='aggregate',values=[dict(column='v',aggregation=op,prefix='',nullPolicy='skip')])
    assert list(reshape_table(t,c).rows)==[(1,expected)]
    verify_output(t,c,reshape_table(t,c))
    if not op.startswith('count'):
        c['values'][0]['nullPolicy']='propagate';assert list(reshape_table(t,c).rows)==[(1,None)]
        c['values'][0]['nullPolicy']='error'
        with pytest.raises(ValueError,match='missing'):reshape_table(t,c)

@pytest.mark.parametrize('engine',['disk','polars'])
def test_wide_sorted_keys_missing_group_collision_and_replay(engine):
    from runner.polars_preparation import preparation_engine
    t=table(['id','key','v'],[(None,2,10),(None,1,20),(1,2,30)],['int64']*3)
    c=config(mode='wide',rowKeys=['id'],columnKey='key',columnOrder='sorted',values=[dict(column='v',aggregation='first',prefix='x',nullPolicy='skip')])
    with preparation_engine(engine):out=reshape_table(t,c)
    assert [col['name'] for col in out.columns]==['id','x1','x2']
    assert list(out.rows)==[(None,20,10),(1,None,30)]
    verify_output(t,c,out)
    with pytest.raises(ValueError,match='Cached'):verify_output(t,c,replace(out,rows=((None,999,10),(1,None,30))))
    c['values'].append(dict(column='v',aggregation='max',prefix='x',nullPolicy='skip'))
    with pytest.raises(ValueError,match='names'):reshape_table(t,c)

def test_exact_sum_finite_mean_overflow_and_fill_type():
    t=table(['key','v'],[('a',1e308),('a',1e308)],['string','float64'])
    c=config(mode='wide',columnKey='key',duplicates='aggregate',values=[dict(column='v',aggregation='mean',prefix='',nullPolicy='skip')])
    assert list(reshape_table(t,c).rows)==[(1e308,)]
    c['values'][0]['aggregation']='sum'
    with pytest.raises(ValueError,match='overflow'):reshape_table(t,c)
    t=table(['key','v'],[('a',2**63-1),('a',1)],['string','int64'])
    with pytest.raises(ValueError,match='overflow'):reshape_table(t,c)
    c['values'][0]['aggregation']='first';c.update(missingCombinations='fill',fillValue='0')
    with pytest.raises(ValueError,match='fill'):reshape_table(t,c)

def test_cancel_spill_cleanup_and_scratch_budget(monkeypatch,tmp_path):
    import runner.dataflow_reshape as mod
    monkeypatch.setattr(mod.tempfile,'tempdir',str(tmp_path))
    t=table(['key','v'],[('a',1)],['string','int64'])
    c=config(mode='wide',columnKey='key',values=[dict(column='v',aggregation='first',prefix='',nullPolicy='skip')])
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls>=3:raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError):reshape_table(t,c,cancel)
    assert list(tmp_path.iterdir())==[]
    monkeypatch.setattr(mod,'MAX_SCRATCH_BYTES',4096)
    with pytest.raises(ValueError,match='scratch'):reshape_table(t,c)
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('wide',[False,True])
def test_standalone_module_replay_outside_repository(tmp_path,wide):
    # This is isolated portable worker-module execution. Root's Code integration has its own replay.
    t=table(['id','jan','feb'],[(1,10,12),(2,20,None)],['int64']*3)
    c=config(idColumns=['id'],valueColumns=['jan','feb'])
    if wide:
        t=table(['id','key','v'],[(1,'feb',12),(1,'jan',10),(2,'jan',30)],['int64','string','int64'])
        c=config(mode='wide',rowKeys=['id'],columnKey='key',values=[dict(column='v',aggregation='first',prefix='',nullPolicy='skip')])
    payload=dict(columns=t.columns,rows=list(t.rows),ids=list(t.row_ids),source=t.source_digest,config=c)
    path=tmp_path/'fixture.json';path.write_text(json.dumps(payload))
    code="""import json,sys
from runner.dataflow import Table,derive
from runner.dataflow_reshape import reshape_table,verify_output
x=json.load(open(sys.argv[1]));t=derive(Table(tuple(x['columns']),tuple(map(tuple,x['rows'])),tuple(x['ids']),x['source']))
y=reshape_table(t,x['config']);verify_output(t,x['config'],y)
print(json.dumps({'rows':list(y.rows),'ids':list(y.row_ids),'source':y.source_digest}))
"""
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2])}
    replay=json.loads(subprocess.check_output([sys.executable,'-c',code,str(path)],cwd=tmp_path,env=env,text=True))
    output=reshape_table(t,c)
    assert replay==dict(rows=[list(r) for r in output.rows],ids=list(output.row_ids),source=output.source_digest)


def test_temporal_metadata_and_canonical_names():
    from runner.dataflow_temporal import metadata
    t=table(['id','jan','feb'],[(1,0,1)],['int64','date','date'])
    t=replace(t,columns=(t.columns[0],{**t.columns[1],'temporal':metadata('date')},{**t.columns[2],'temporal':metadata('date')}))
    c=config(idColumns=['id'],valueColumns=['jan','feb'])
    out=reshape_table(t,c)
    assert out.columns[-1]['temporal']==metadata('date')
    assert list(out.rows)==[(1,'jan',0),(1,'feb',1)]
    wide=config(mode='wide',rowKeys=['id'],columnKey='value',values=[dict(column='value',aggregation='min',prefix='day_',nullPolicy='skip')])
    output=reshape_table(out,wide)
    assert [col['name'] for col in output.columns]==['id','day_1970-01-01','day_1970-01-02']
    assert output.columns[-1]['temporal']==metadata('date')
    verify_output(out,wide,output)


def test_long_images_are_values_without_scalar_conversion():
    from runner.tests.test_image_tables import table as images
    t,_,_=images()
    image_name=next(col['name'] for col in t.columns if col['dtype']=='image')
    c=config(valueColumns=[image_name])
    output=reshape_table(t,c)
    assert output.columns[-1]['dtype']=='image'
    assert [row[-1] for row in output.rows]==[row[t.index(image_name)] for row in t.rows]
    verify_output(t,c,output)

@pytest.mark.parametrize('wide',[False,True])
def test_output_byte_budget_refuses_before_materializing(monkeypatch,wide):
    import runner.dataflow_reshape as mod
    t=table(['key','v'],[('a',1)],['string','int64'])
    c=config(valueColumns=['v'])
    if wide:c=config(mode='wide',columnKey='key',values=[dict(column='v',aggregation='first',prefix='',nullPolicy='skip')])
    monkeypatch.setattr(mod,'MAX_ARTIFACT_BYTES',100)
    with pytest.raises(ValueError,match='serialized byte budget'):reshape_table(t,c)


def test_wide_and_output_share_aggregate_scratch_budget():
    from runner.dataflow_disk import disk_scope,ScratchBudget
    t=table(['key','v'],[('a',1)],['string','int64'])
    c=config(mode='wide',columnKey='key',values=[dict(column='v',aggregation='first',prefix='',nullPolicy='skip')])
    budget=ScratchBudget(32768)
    with disk_scope(budget=budget):
        with pytest.raises(ValueError,match='scratch'):reshape_table(t,c)
    assert budget.used==0
