"""Versioned preparation demand and deterministic, explicitly incomplete samples.

Sampling means a source-order prefix, not a statistically representative/random
sample. A complete consumer promotes its entire dependency branch to complete.
"""
from dataclasses import dataclass
import json
import math

if __package__:
    from . import dataflow
else:
    import dataflow

if __package__:
    from .preparation_demand_configuration import ALL,MAX_SAMPLE_ROWS,normalize_sampling,is_complete_sampling,sampling_row_limit
else:
    from preparation_demand_configuration import ALL,MAX_SAMPLE_ROWS,normalize_sampling,is_complete_sampling,sampling_row_limit


@dataclass(frozen=True)
class DemandPlan:
    canonical_json:str
    def to_document(self):return json.loads(self.canonical_json)
    @property
    def complete_node_ids(self):return tuple(self.to_document()['completeNodeIDs'])
    @property
    def sampled_node_ids(self):return tuple(self.to_document()['sampledNodeIDs'])
    @property
    def sampling(self):return self.to_document()['sampling']
    @property
    def has_samples(self):return bool(self.sampled_node_ids)
    def is_complete(self,node_id):
        if node_id not in self.complete_node_ids+self.sampled_node_ids:raise ValueError('Unknown demand node')
        return node_id in self.complete_node_ids
    def policy_for(self,node_id):return dict(ALL) if self.is_complete(node_id) else self.sampling


def build_demand_plan(graph,sampling=None,complete_nodes=None):
    nodes,inputs,order=dataflow.validate(graph)
    policy=normalize_sampling(sampling)
    requested=[] if complete_nodes is None else complete_nodes
    if not isinstance(requested,(list,tuple)) or any(not isinstance(x,str) for x in requested) or len(set(requested))!=len(requested) or set(requested)-set(nodes):
        raise ValueError('Complete demand must identify distinct known dataflow nodes')
    complete=set(requested)
    complete.update(key for key,(op,_) in nodes.items() if op in ('view','outputDelimited'))
    if is_complete_sampling(policy):complete.update(nodes)
    pending=list(complete)
    while pending:
        for predecessor,_ in inputs[pending.pop()].values():
            if predecessor not in complete:complete.add(predecessor);pending.append(predecessor)
    document={'schemaVersion':1,'graphDigest':dataflow.digest(graph),'sampling':policy,
              'selection':'sourceOrderPrefix','completeNodeIDs':[key for key in order if key in complete],
              'sampledNodeIDs':[key for key in order if key not in complete],
              'percentageRequiresCardinalityScan':policy['mode']=='percentage' and policy['percentage']<100 and len(complete)<len(nodes)}
    return DemandPlan(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False))


def validate_demand_plan(plan,graph):
    if not isinstance(plan,DemandPlan):raise ValueError('Expected an immutable preparation demand plan')
    document=plan.to_document()
    expected=build_demand_plan(graph,document.get('sampling'),document.get('completeNodeIDs'))
    if expected!=plan:raise ValueError('Preparation demand does not match the captured graph')
    return plan


def sample_table(table,policy,check_cancelled=lambda:None):
    check_cancelled()
    limit=sampling_row_limit(policy,len(table.rows))
    if limit is None or limit>=len(table.rows):return table
    # Stream into the engine's owned row store rather than taking an unbounded
    # sequence slice or allocating all selected rows/IDs in a Python list.
    def pairs():
        for index,(row,identity) in enumerate(zip(table.rows,table.row_ids)):
            if index>=limit:break
            if index%256==0:check_cancelled()
            yield row,identity
    return dataflow.streaming_rows(table,pairs(),check_cancelled=check_cancelled)


def coverage_report(plan,node_row_counts):
    if not isinstance(plan,DemandPlan):raise ValueError('Expected preparation demand')
    expected=set(plan.complete_node_ids+plan.sampled_node_ids)
    if not isinstance(node_row_counts,dict) or set(node_row_counts)!=expected:
        raise ValueError('Successful coverage must describe every executed node exactly once')
    nodes={}
    for node in plan.complete_node_ids+plan.sampled_node_ids:
        counts=node_row_counts[node]
        if not isinstance(counts,dict) or any(not isinstance(port,str) or type(count) is not int or count<0 for port,count in counts.items()):
            raise ValueError('Coverage row counts must be nonnegative integers by port')
        nodes[node]={'completeness':'complete' if plan.is_complete(node) else 'sampled','rowCounts':dict(counts)}
    return {**plan.to_document(),'status':'sampled' if plan.has_samples else 'complete','nodes':nodes}


def validate_coverage(report,graph,*,demand_plan=None):
    """Validate coverage against the pinned plan when supplied, never row samples.

    The plan/manifest SHA pins remain the caller's authority. Structural validation
    alone cannot authenticate a caller-provided claim about original source rows.
    """
    if not isinstance(report,dict):raise ValueError('Execution coverage must be an object')
    nodes,inputs,_=dataflow.validate(graph)
    inferred=build_demand_plan(graph,report.get('sampling'),report.get('completeNodeIDs'))
    plan=inferred if demand_plan is None else validate_demand_plan(demand_plan,graph)
    raw_nodes=report.get('nodes')
    if not isinstance(raw_nodes,dict):raise ValueError('Execution coverage requires per-node evidence')
    counts={}
    for key,(operator,_) in nodes.items():
        entry=raw_nodes.get(key)
        if not isinstance(entry,dict) or set(entry)!={'completeness','rowCounts'}:
            raise ValueError('Invalid per-node execution coverage')
        counts[key]=entry['rowCounts']
        required=dataflow.configured_ports(operator,nodes[key][1],inputs[key])[2]-{'state'}
        if not isinstance(counts[key],dict) or set(counts[key])!=required:
            raise ValueError('Execution coverage table ports differ from graph')
    expected=coverage_report(plan,counts)
    if report!=expected:raise ValueError('Execution coverage differs from its immutable demand plan')
    return expected
