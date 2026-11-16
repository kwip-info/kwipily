"""Actual main.py training transport, separate from direct engine proofs."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner.tests.test_runner import graph
from runner.tests.test_scheduler_operation_contract import recipe,IDS
from runner.tests.test_optimizer_operation_contract import recipe as optimizer
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('kind',IDS)
def test_actual_cli_retains_scheduler_and_tick_events(tmp_path,kind):
    selected=recipe(kind)
    if kind=='scheduler_one_cycle_lr':selected['config']['planned_total_batches']=100
    model=tmp_path/'model.json';model.write_text(json.dumps(graph()))
    out=tmp_path/'artifacts'
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(model),'--output',str(out),'--registry',str(ROOT/'Sources/KwipMLCore/Resources/components.json'),'--epochs','2','--batch-size','128','--device','cpu','--learning-rate','0','--optimizer-recipe-json',json.dumps(optimizer()),'--scheduler-recipe-json',json.dumps(selected)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines()]
    manifest=json.loads((out/'manifest.json').read_text());trace=json.loads((out/'rate-trace.json').read_text())
    assert manifest['settings']['schedulerRecipe']==selected
    assert trace['recipe']==selected and trace['initialization']['tick']==0
    ticks=[e for e in events if e.get('event')=='schedulerTick']
    assert len(ticks)==(14 if kind=='scheduler_one_cycle_lr' else 2)
    assert [e['record'] for e in ticks]==trace['records']
    assert all(e['scheduler']==selected for e in ticks)
    assert len((out/'training_recipe.py').read_bytes())<=8192
