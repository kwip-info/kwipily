import copy
import pytest
from runner.game_rules import Definition, RuleError, load_definition
from runner.game_rule_examples import examples

@pytest.mark.parametrize('name',['take-away','placement','gravity'])
def test_authored_games_and_replay(name):
    game=Definition(examples()[name]);result=game.run_tests()
    assert result['tests'] and all(row['status']=='passed' for row in result['tests']),result
    assert game.identity==Definition(copy.deepcopy(game.source)).identity

def test_invalid_move_is_atomic():
    g=Definition(examples()['take-away']);s=g.initial()
    with pytest.raises(RuleError,match='illegal'):g.step(s,{'take':4})
    assert s==g.initial()

def test_terminal_no_more_moves():
    g=Definition(examples()['take-away']);s={'remaining':1,'to_move':0}
    out=g.step(s,{'take':1});assert out['terminal'] and out['winner']==0 and out['rewards']==[1,-1]
    assert g.legal_actions(out['state'])==[]
    with pytest.raises(RuleError,match='terminal'):g.step(out['state'],{'take':1})

@pytest.mark.parametrize('expression',['__import__("os")','state.__class__','open("x")','[0]*999999999','2 ** 100000000','(lambda: 1)()'])
def test_forbidden_language(expression):
    x=examples()['take-away'];x['legal']=expression
    with pytest.raises(RuleError):Definition(x)

def test_duplicate_json_keys_rejected():
    with pytest.raises(RuleError):load_definition('{"version":1,"version":1}')

def test_static_wrong_type_and_next_reference():
    for value in ('"oops"','next.unknown'):
        x=examples()['take-away'];x['transitions'][0]['updates']['remaining']=value
        with pytest.raises(RuleError):Definition(x)

def test_budget_and_cancellation():
    g=Definition(examples()['take-away'],maximum_operations=10)
    with pytest.raises(RuleError,match='budget'):g.run_tests()
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):Definition(examples()['take-away'],check_cancelled=cancel)

def test_failed_test_is_not_success():
    x=examples()['take-away'];x['tests'][0]['expect']['state']['remaining']=999
    r=Definition(x).run_tests();assert r['tests'][0]['status']=='failed'

def test_invariant_and_ambiguous_transition():
    x=examples()['take-away'];x['invariants']=['state.remaining > 0']
    g=Definition(x)
    with pytest.raises(RuleError,match='invariant'):g.step({'remaining':1,'to_move':0},{'take':1})
    x=examples()['take-away'];x['transitions']*=2
    with pytest.raises(RuleError,match='ambiguous'):Definition(x).step(Definition(x).initial(),{'take':1})

def test_source_roundtrip_and_schema_version():
    import json
    x=examples()['take-away'];assert load_definition(json.dumps(x))==x
    x['version']=2
    with pytest.raises(RuleError):Definition(x)

def test_standard_tables_and_definition_bytes():
    import hashlib,json
    from runner.game_rules_tables import apply
    outputs=apply({'definition':examples()['gravity']},lambda:None)
    assert set(outputs)=={'definition','summary','tests','trace'}
    record=dict(zip([c['name'] for c in outputs['definition'].columns],outputs['definition'].rows[0]))
    assert hashlib.sha256(record['json'].encode()).hexdigest()==record['sha256']
    assert Definition(json.loads(record['json'])).identity==record['sha256']
    assert outputs['summary'].rows[0][2]==0

def test_key_order_independent_and_next_topological():
    import json
    x=examples()['take-away'];x['transitions'][0]['updates']={'remaining':'state.remaining - action.take','to_move':'1 - state.to_move if next.remaining > 0 else state.to_move'}
    a=Definition(x);b=Definition(json.loads(json.dumps(x,sort_keys=True)))
    assert a.identity==b.identity and a.step(a.initial(),{'take':2})==b.step(b.initial(),{'take':2})

def test_huge_action_domain_refused_before_allocation():
    x=examples()['take-away'];x['actions']['take']['maximum']=1_000_000_000
    with pytest.raises(RuleError,match='domain'):Definition(x)

def test_real_preparation_roundtrip_and_legacy_view(tmp_path):
    import json
    from runner.dataflow_cli import run_dataflow,inspect_dataflow
    from runner.dataflow import digest
    import hashlib
    graph={'schemaVersion':1,'nodes':[{'id':'rules','operator':'kwip.data.gameRules','config':{'definition':examples()['placement']}},{'id':'inspect','operator':'kwip.data.view','config':{}}],'edges':[{'from':{'node':'rules','port':'tests'},'to':{'node':'inspect','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));events=[]
    run_dataflow(path,{},tmp_path/'out',events.append)
    assert events[-1]['event']=='dataflowCompleted'
    manifest=tmp_path/'out/manifest.json';inspect_dataflow(manifest,hashlib.sha256(manifest.read_bytes()).hexdigest(),events.append)
    assert events[-1]['event']=='dataflowInspectionCompleted'

def test_repeat_allocation_budget():
    x=examples()['take-away'];x['transitions'][0]['updates']['remaining']='len(repeat(repeat(repeat(0, 256), 256), 256))'
    with pytest.raises(RuleError,match='allocation'):Definition(x).step(Definition(x).initial(),{'take':1})

def test_resource_failure_cannot_pass_expected_error():
    x=examples()['gravity'];x['tests'][0]['expectError']='budget'
    with pytest.raises(RuleError,match='budget'):Definition(x,maximum_operations=1000).run_tests()

def test_missing_expectation_is_refused():
    x=examples()['take-away'];x['tests'][0].pop('expect')
    with pytest.raises(RuleError,match='expectation'):Definition(x).run_tests()

def test_grid_draw_and_gravity_full_column():
    g=Definition(examples()['placement']);s=g.initial();out=None
    for i in [0,1,2,4,3,5,7,6,8]:out=g.step(s,{'cell':i});s=out['state']
    assert out['terminal'] and out['winner'] is None and out['rewards']==[0,0]
    g=Definition(examples()['gravity']);s=g.initial()
    for _ in range(6):s=g.step(s,{'column':0})['state']
    assert {'column':0} not in g.legal_actions(s)
    with pytest.raises(RuleError,match='illegal'):g.step(s,{'column':0})

@pytest.mark.parametrize('mirror',[False,True])
def test_independent_gravity_diagonal_positions(mirror):
    g=Definition(examples()['gravity']);s=g.initial();s['to_move']=1
    p0=[(5,0),(4,1),(3,2),(2,3),(4,3)];p1=[(5,1),(5,2),(4,2),(5,3),(3,3)]
    for value,points in ((1,p0),(2,p1)):
        for r,c in points:s['board'][r*7+(6-c if mirror else c)]=value
    assert g.outcome(s)=={'terminal':True,'winner':0}

def test_bad_reward_nonfinite_observation_and_illegal_identity():
    x=examples()['take-away'];x['reward']='1'
    with pytest.raises(RuleError,match='zero-sum'):Definition(x).step(Definition(x).initial(),{'take':1})
    x=examples()['take-away'];x['observations']=['1 / 0'];g=Definition(x)
    with pytest.raises(RuleError):g.step(g.initial(),{'take':1})
    x=examples()['take-away'];x['legal']='state.unknown == 1'
    with pytest.raises(RuleError,match='Unknown'):Definition(x)

def test_fixture_sources_match_native_resources():
    import json
    from pathlib import Path
    for name,value in examples().items():assert json.loads((Path(__file__).parents[2]/'Sources/KwipMLCore/Resources/GameRules'/f'{name}.json').read_text())==value

def test_portable_preparation_code_replays_same_engine(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    import numpy
    from runner.preparation_plan import build_plan,derive_preparation_code
    graph={'schemaVersion':1,'nodes':[{'id':'rules','operator':'kwip.data.gameRules','config':{'definition':examples()['take-away']}}],'edges':[]}
    code=derive_preparation_code(build_plan(graph,engine='disk',engine_version='1'))
    target=tmp_path/'preparation.py';target.write_text(code)
    bindings=tmp_path/'bindings.json';bindings.write_text('{}')
    import os
    completed=subprocess.run([sys.executable,str(target),'--source-bindings',str(bindings)],env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[2])},cwd=Path(__file__).parents[2],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    events=[json.loads(line) for line in completed.stdout.splitlines()];assert events[-1]['event']=='preparationPlanCompleted'
    assert events[0]['outputs']['summary']['preview']['rows'][0][1:3]==[3,0]

def test_multifield_action_encoding_is_canonical():
    import json
    x=examples()['take-away'];x['actions']={'take':dict(type='integer',minimum=1,maximum=3),'flag':dict(type='boolean')}
    a=Definition(x);b=Definition(json.loads(json.dumps(x,sort_keys=True)))
    assert a.identity==b.identity and a.legal_actions(a.initial())==b.legal_actions(b.initial())

def test_unknown_assertion_fields_and_boolean_coercion_do_not_pass():
    for expectation in ({'state':{'unknown':None}},{'terminal':0}):
        x=examples()['take-away'];x['tests'][0]['expect']=expectation
        assert Definition(x).run_tests()['tests'][0]['status']=='failed'

@pytest.mark.parametrize('field,expression',[('currentPlayer','action.take'),('legal','player == 0'),('reward','action.take'),('observations','action.take')])
def test_expression_contexts_do_not_leak_action_or_player(field,expression):
    x=examples()['take-away'];x[field]=[expression] if field=='observations' else expression
    with pytest.raises(RuleError,match='Unknown'):Definition(x)

def test_nested_comprehension_allocation_counts_cells_before_state_commit():
    x=examples()['take-away'];x['transitions'][0]['updates']['remaining']='len([repeat(0,256) for i in range(256)])'
    g=Definition(x);s=g.initial()
    with pytest.raises(RuleError,match='allocation'):g.step(s,{'take':1})
    assert s==g.initial()
