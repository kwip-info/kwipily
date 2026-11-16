"""Pure, bounded authored-graph upgrades. Never reads sources or historical values.

Only equivalent current-tool configurations are admitted. A refusal leaves the
caller's graph untouched; saved bytes are the owner's responsibility.
"""
import copy
import json
from pathlib import Path

PREFIX='kwip.data.'
RETIRED=frozenset('missing standardScalerFit imputerFit oneHotEncoderFit transform calculate regexExtract regexReplace split groupSplit temporalSplit selectRename cast filter filterExpression regexMatch'.split())
FIT=frozenset(('standardScalerFit','imputerFit','oneHotEncoderFit'))


def repair(node,reason):
    raise ValueError(f'Retired data tool {node}: {reason} Original source is unchanged; repair this tool with current Columns, Filter, Formula, Partition or Preprocess before saving or running.')


def _rule(name):
    if __package__:from .preprocess_parameters import rule_defaults,CLEANUP_DEFAULTS
    else:from preprocess_parameters import rule_defaults,CLEANUP_DEFAULTS
    return {k:v for k,v in rule_defaults(name).items() if k not in CLEANUP_DEFAULTS}


def _filter():
    return dict(mode='simple',conditions=[],match='all',expression='',column='value',pattern='',matchMode='search',flags=[],nullPolicy='error')


def _column_ref(name,node):
    if type(name) is not str or ']' in name or '\n' in name:repair(node,'the source column cannot be represented exactly in Formula syntax.')
    return '['+name+']'


def _write(c,name,schema,node):
    if not c['replaceExisting']:return 'new'
    if schema is None:repair(node,'Replace existing needs a known complete input schema to distinguish replacement from a new destination.')
    return 'replace' if name in {x['name'] for x in schema} else 'new'


def _configuration(op,c,node,schema):
    if op=='split':repair(node,'legacy NumPy two-way shuffled membership differs from current three-way Partition. Choose and review a new split explicitly.')
    if op=='filter':repair(node,'the legacy single-output filter stops on an empty result; current Filter emits complementary branches. Review the intended empty-result behavior explicitly.')
    if op in ('selectRename','cast'):
        if op=='selectRename':
            if set(c['rename'])-set(c['columns']):repair(node,'Rename controls do not match selected sources.')
            rows=[dict(source=n,included=True,outputName=c['rename'].get(n,n),targetType='keep') for n in c['columns']]
        else:
            if not schema:repair(node,'Cast needs the known complete input schema to preserve original column order.')
            if set(c['columns'])-{x['name'] for x in schema}:repair(node,'a cast source column is absent from the supplied input schema.')
            rows=[dict(source=x['name'],included=True,outputName=x['name'],targetType=c['columns'].get(x['name'],'keep')) for x in schema]
        return 'columns',dict(rows=rows,castFailure='error',trimText=False,newColumns='drop',missingPolicy=c.get('missingPolicy','error'))
    if op=='missing':
        if c['policy']=='drop':repair(node,'legacy Drop stops when all observations disappear; review the empty-table policy in current Preprocess explicitly.')
        if __package__:from .preprocess_parameters import rule_defaults,typed
        else:from preprocess_parameters import rule_defaults,typed
        rows=[];types={x['name']:x['dtype'] for x in schema or []}
        for name in c['columns']:
            row=_rule(name);row['missing']='error' if c['policy']=='error' else 'constant'
            if c['policy']=='fill':
                if name not in types or not typed(c['value'],types[name]):repair(node,'the fixed replacement needs a known matching type; legacy implicit literal casts cannot be guessed.')
                row['value']=c['value']
            rows.append(row)
        return 'preprocess',dict(mode='table',columns=rows)
    if op in ('groupSplit','temporalSplit'):
        result={k:v for k,v in c.items() if k not in ('nullKeys','allocationUnit')}
        result['strategy']='entity' if op=='groupSplit' else 'time'
        return 'partition',result
    if op in ('filterExpression','regexMatch'):
        result=_filter()
        if op=='filterExpression':result.update(mode='expression',expression=c['expression'],nullPolicy=c['nullPolicy'])
        else:result.update(mode='pattern',**c)
        return 'filterRows',result
    if op=='calculate':
        outputs=[dict(name=c['outputColumn'],dtype=c['dtype'],write=_write(c,c['outputColumn'],schema,node))]
        expression=c['expression']
    elif op in ('regexExtract','regexReplace'):
        source=_column_ref(c['column'],node);quote=lambda x:json.dumps(x,ensure_ascii=False)
        if op=='regexExtract':
            expression='regex_extract('+', '.join([source,*map(quote,[c['pattern'],'search',c['noMatchPolicy'],'|'.join(c['flags']),*[g['group'] for g in c['groups']]])])+')'
            outputs=[dict(name=g['output'],dtype='string',write=_write(c,g['output'],schema,node)) for g in c['groups']]
        else:
            expression='regex_replace('+', '.join([source,*map(quote,[c['pattern'],c['replacement'],c['replaceMode'],'|'.join(c['flags'])])])+')'
            outputs=[dict(name=c['outputColumn'],dtype='string',write=_write(c,c['outputColumn'],schema,node))]
    else:repair(node,'a fitted transform must be fused with its exact verified partition consumers.')
    return 'formula',dict(operations=[dict(id='upgraded-operation',expression=expression,outputs=outputs,nullPolicy=c['nullPolicy'])])


def upgrade_graph(graph,known_input_schemas=None):
    """Return a current graph, or raise before mutating the supplied object.

    Schemas are caller-owned exact declarations/evidence keyed by original node ID;
    this function never inspects a user's source file to invent a column order.
    """
    if type(graph) is not dict or set(graph)!={'schemaVersion','nodes','edges'} or graph['schemaVersion']!=1 or type(graph['nodes']) is not list or not 1<=len(graph['nodes'])<=128 or type(graph['edges']) is not list or len(graph['edges'])>256:raise ValueError('Invalid bounded migration graph')
    result=copy.deepcopy(graph);schemas=known_input_schemas or {};nodes={}
    for node in result['nodes']:
        if type(node) is not dict or set(node)!={'id','operator','config'} or type(node['id']) is not str or node['id'] in nodes:raise ValueError('Invalid migration node')
        nodes[node['id']]=node
    retired=[n for n in result['nodes'] if n['operator'].removeprefix(PREFIX) in RETIRED]
    if not retired:return result
    # Validate the frozen legacy controls with their bounded decode-only descriptors.
    if __package__:from . import dataflow as df
    else:import dataflow as df
    for n in retired:
        op=n['operator'].removeprefix(PREFIX)
        df.schema_validate(n['config'],df.RETIRED_DEFINITIONS[op]['configSchema'])
    # Dependency traversal controls fusion only; preserve the authored node list.
    dependencies={ident:set() for ident in nodes}
    for edge in result['edges']:
        if type(edge) is not dict or set(edge)!={'from','to'} or any(type(endpoint) is not dict or set(endpoint)!={'node','port'} or type(endpoint['node']) is not str or type(endpoint['port']) is not str or endpoint['node'] not in nodes for endpoint in edge.values()):
            repair('graph','the dependency endpoints cannot be proved.')
        dependencies[edge['to']['node']].add(edge['from']['node'])
    pending=set(nodes);completed=set();ordered_fits=[]
    while pending:
        ready=[ident for ident in nodes if ident in pending and dependencies[ident]<=completed]
        if not ready:repair('graph','a dependency cycle prevents exact fitted-chain fusion.')
        ordered_fits.extend(nodes[ident] for ident in ready if nodes[ident]['operator'].removeprefix(PREFIX) in FIT)
        completed.update(ready);pending.difference_update(ready)
    removed=set();rewrites={}
    for fit in ordered_fits:
        ident=fit['id'];incoming=[e for e in result['edges'] if e['to']==dict(node=ident,port='training')]
        state_edges=[e for e in result['edges'] if e['from']==dict(node=ident,port='state')]
        if len(incoming)!=1 or not state_edges:repair(ident,'fit/apply fusion requires one Training input and explicit transform consumers.')
        source=incoming[0]['from'];family=source['node']
        if source['port']!='training':repair(ident,'fit/apply fusion cannot prove the Training role.')
        family_node=nodes.get(family,{})
        if family_node.get('operator') not in (PREFIX+'partition',PREFIX+'groupSplit',PREFIX+'temporalSplit',PREFIX+'preprocess'):repair(ident,'fit/apply fusion requires direct role outputs of one Partition or prior Preprocess.')
        roles={};training=False
        for edge in state_edges:
            transform=nodes.get(edge['to']['node']);target=edge['to']
            if not transform or transform['operator']!=PREFIX+'transform' or target['port']!='state':repair(ident,'the learned state has a consumer other than its paired transform.')
            inputs=[e for e in result['edges'] if e['to']['node']==transform['id'] and e['to']['port']=='table']
            if len(inputs)!=1:repair(ident,'each paired transform needs one exact role input.')
            endpoint=inputs[0]['from'];role=endpoint['port']
            if endpoint['node']!=family or role not in ('training','validation','test'):repair(ident,'paired transforms must consume direct matching roles from the same partition family.')
            if role=='training':training=True
            roles[role]=endpoint;removed.add(transform['id']);rewrites[(transform['id'],'table')]=dict(node=ident,port=role)
        if not training:repair(ident,'a Training transform is required to preserve the learned application population.')
        if __package__:from .preprocess_parameters import rule_defaults
        else:from preprocess_parameters import rule_defaults
        rules=[];op=fit['operator'].removeprefix(PREFIX);c=fit['config']
        for name in c['columns']:
            row=_rule(name)
            if op=='standardScalerFit':row['scaling']='standard'
            elif op=='imputerFit':row.update(missing=c['strategy'],emptyPolicy=c['emptyPolicy'],emptyValue=c.get('emptyValue',0),tiePolicy=c.get('tiePolicy','error'))
            else:row.update(encoding='oneHot',vocabularyOrder=c['vocabularyOrder'],unknownPolicy=c['unknownPolicy'],maxCategories=c['maxCategories'])
            rules.append(row)
        fit.update(operator=PREFIX+'preprocess',config=dict(mode='partitions',columns=rules))
        result['edges']=[e for e in result['edges'] if e['to']['node']!=ident and e['to']['node'] not in removed]
        result['edges'] += [{'from':endpoint,'to':dict(node=ident,port=role)} for role,endpoint in roles.items()]
        for edge in result['edges']:
            key=(edge['from']['node'],edge['from']['port'])
            if key in rewrites:edge['from']=rewrites[key]
    for n in retired:
        ident=n['id'];op=n['operator'].removeprefix(PREFIX)
        if ident in removed or op=='preprocess':continue
        if op=='transform':repair(ident,'transform has no exactly fusible fitted-state owner.')
        new,c=_configuration(op,n['config'],ident,schemas.get(ident));n.update(operator=PREFIX+new,config=c)
    result['nodes']=[n for n in result['nodes'] if n['id'] not in removed]
    for edge in result['edges']:
        key=(edge['from']['node'],edge['from']['port'])
        if key in rewrites:edge['from']=rewrites[key]
    return result
