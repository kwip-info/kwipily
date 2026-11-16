"""Standard table materialization for isolated rule testing and definition export."""
import json
if __package__:
    from .game_rules import Definition
    from . import dataflow as df
else:
    from game_rules import Definition
    import dataflow as df

def validate(config):
    if type(config) is not dict or set(config)!={'definition'}:raise ValueError('Game definition requires one canonical definition object.')
    return Definition(config['definition'])

def apply(config,check_cancelled):
    g=Definition(config['definition'],check_cancelled=check_cancelled);result=g.run_tests()
    def table(port,schema,values):
        columns=tuple(dict(name=n,dtype=t,nullable=False) for n,t in schema)
        rows=tuple(tuple((v.get(n,'') if type(v.get(n,'')) is str else json.dumps(v.get(n),sort_keys=True,separators=(',',':'))) if t=='string' else v[n] for n,t in schema) for v in values)
        ids=tuple(df.digest([g.identity,port,i]) for i in range(len(rows)))
        return df.derive(df.Table(columns,rows,ids,g.identity))
    failed=sum(x['status']=='failed' for x in result['tests'])
    return {'summary':table('summary',[('definition','string'),('passed','int64'),('failed','int64'),('moves','int64')],[dict(definition=g.identity,passed=len(result['tests'])-failed,failed=failed,moves=len(result['trace']))]),
            'tests':table('tests',[(n,t) for n,t in [('test','string'),('status','string'),('detail','string'),('moves','int64'),('definition','string')]],result['tests']),
            'trace':table('trace',[(n,t) for n,t in [('test','string'),('move','int64'),('before','string'),('action','string'),('after','string'),('status','string'),('rule','string'),('terminal','bool'),('winner','string'),('rewards','string')]],result['trace']),
            'definition':table('definition',[('schema','string'),('sha256','string'),('json','string')],[dict(schema='kwip.game-definition.v1',sha256=g.identity,json=g.canonical)])}
