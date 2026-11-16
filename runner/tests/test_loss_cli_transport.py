"""Actual CLI delivery preserves typed targets and objective identity."""
import hashlib,json,subprocess,sys
from pathlib import Path
import pytest
from runner.tests.test_loss_engine import prepared
from runner.tests.test_loss_prepared_data import fixture,CASES
ROOT=Path(__file__).resolve().parents[2]

def cli(args):
 result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),*args],capture_output=True,text=True,timeout=60)
 events=[json.loads(line)for line in result.stdout.splitlines()if line.strip()]
 return result,events

@pytest.mark.parametrize('op',CASES)
def test_eight_real_prepared_cli_runs_and_delivers_configured_objective_export(tmp_path,op):
 graph,r,m,data=prepared(tmp_path/'source',op)
 path=tmp_path/'graph.json';path.write_text(json.dumps(graph));p=data[1]['preparation'];manifest=tmp_path/'source/preparation/manifest.json';out=tmp_path/'run'
 args=['--graph',str(path),'--output',str(out),'--device','cpu','--epochs','2','--batch-size','5',
       '--prepared-dataflow',str(manifest),'--prepared-manifest-sha256',p['manifestSHA256'],'--prepared-graph-digest',p['graphDigest'],
       '--prepared-training-node','p','--prepared-training-port','training','--prepared-validation-node','p','--prepared-validation-port','validation',
       '--objective-recipe-json',json.dumps(r),'--objective-target-mapping-json',json.dumps(m),'--objective-scalarization','mean']
 result,events=cli(args);assert result.returncode==0,(events,result.stderr)
 assert events[-1]['event']=='completed'
 saved=json.loads((out/'manifest.json').read_text());assert saved['settings']['objectiveRecipe']==r and saved['settings']['targetMapping']==m
 assert (out/'objective_recipe.py').is_file()
 assert saved['provenance']['objectiveMapping']==m and 'targetColumn'not in saved['provenance']
 assert ('accuracy'in json.loads((out/'metrics.json').read_text())[0])==(op in ('loss_cross_entropy','loss_nll'))

@pytest.mark.parametrize('flags',[
 ['--capabilities'],['--learning-runtime'],['--export-code'],['--preview'],['--dataflow-registry'],['--inspect-data','missing.csv']])
def test_objective_transport_refuses_other_operation_modes_before_side_effects(tmp_path,flags):
 _,m,r,_=fixture('loss_mse');out=tmp_path/'run'
 result,events=cli([*flags,'--output',str(out),'--objective-recipe-json',json.dumps(r),'--objective-target-mapping-json',json.dumps(m)])
 assert result.returncode==1 and events[-1]['event']=='error'and 'training only'in events[-1]['message']
 assert not out.exists()

@pytest.mark.parametrize('parts',[
 ['--objective-scalarization','sum'],['--objective-target-mapping-json','{}'],
 ['--objective-recipe-json','{}'],['--objective-recipe-json','{}','--objective-recipe-json','{}']])
def test_orphan_malformed_or_repeated_objective_controls_refuse(tmp_path,parts):
 result,events=cli(['--output',str(tmp_path/'run'),*parts]);assert result.returncode==1 and events[-1]['event']=='error'
 assert not(tmp_path/'run').exists()
