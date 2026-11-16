"""Versioned, path-free preparation snapshots shared by execution and derived Code.

This captures the existing preparation graph, not an arbitrary Python program or a
model/training workflow. Source grants and destination grants remain caller-owned.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import re

if __package__:
    from . import dataflow
else:
    import dataflow

MAX_PLAN_BYTES = 2_000_000
DISK_ENGINE_VERSION = '1'
ENGINE_IDS = ('disk', 'polars')


def _canonical(value):
    try:
        content = json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False)
        encoded = content.encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('Preparation plan requires finite, UTF-8 JSON values') from error
    if len(encoded) > MAX_PLAN_BYTES:
        raise ValueError('Preparation plan exceeds two million bytes')
    return content


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Preparation plan contains a duplicate JSON key')
        result[key] = value
    return result


@dataclass(frozen=True)
class PreparationPlan:
    """Canonical immutable snapshot; accessors return detached mutable JSON copies."""
    canonical_json: str

    @classmethod
    def from_json(cls, content):
        if not isinstance(content, str):
            raise ValueError('Preparation plan must be JSON text')
        try:
            size = len(content.encode('utf-8'))
        except UnicodeError as error:
            raise ValueError('Preparation plan must be UTF-8 JSON text') from error
        if size > MAX_PLAN_BYTES:
            raise ValueError('Preparation plan exceeds two million bytes')
        try:
            document = json.loads(content, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError) as error:
            raise ValueError('Invalid preparation plan JSON') from error
        return validate_plan(document)

    def to_document(self):
        return json.loads(self.canonical_json)

    @property
    def plan_digest(self):
        return hashlib.sha256(self.canonical_json.encode('utf-8')).hexdigest()

    @property
    def graph_digest(self):
        return self.to_document()['graphDigest']


def build_plan(graph, *, engine, engine_version, node_labels=None, sampling=None, complete_nodes=None, workload_policy=None):
    """Validate and capture a local preparation graph without resolving any files.

    The source adapter produces typed tables before the selected preparation
    engine. Unsupported connector/media adapters are intentionally not claimed.
    Labels affect the presentation snapshot, while graphDigest retains the
    existing scientific graph identity separately from planDigest.
    """
    if engine not in ENGINE_IDS:
        raise ValueError('Unsupported preparation engine')
    if not isinstance(engine_version, str) or not re.fullmatch(r'[A-Za-z0-9.+_-]{1,64}', engine_version):
        raise ValueError('Preparation engine version must be an explicit bounded identity')
    # Copy before validating so later edits cannot mutate the captured request.
    captured = json.loads(_canonical(graph))
    try:
        nodes, _, _ = dataflow.validate(captured)
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError('Malformed preparation graph') from error
    labels = {} if node_labels is None else json.loads(_canonical(node_labels))
    dataflow.validate_node_labels(labels, nodes)
    if __package__:from .preparation_demand import build_demand_plan
    else:from preparation_demand import build_demand_plan
    demand=build_demand_plan(captured,sampling,complete_nodes)
    versions = {}
    for op, _ in nodes.values():
        version = dataflow.DEFINITIONS[op]['version']
        versions[dataflow.PREFIX + op] = '.'.join(str(version[key]) for key in ('major', 'minor', 'patch'))
    document = {
        'schemaVersion': 2,
        'scope': 'dataPreparation',
        'engine': {'id': engine, 'version': engine_version},
        'graph': captured,
        'graphDigest': dataflow.digest(captured),
        'registryDigest': dataflow.digest(dataflow.REGISTRY),
        'operatorVersions': versions,
        'nodeLabels': labels,
        'sources': [{'nodeID': key, 'adapter': 'captured.predictions' if op=='capturedTable' else 'local.fileCollection' if op=='sourceFiles' else ('local.csv.polars' if nodes[key][1].get('parserProfile')=='polars' else 'local.csv') if nodes[key][1].get('format','csv')=='csv' else 'local.'+nodes[key][1]['format'], 'outputKind': 'typed.table'}
                    for key, (op, _) in nodes.items() if op in ('sourceCSV','sourceFiles','capturedTable')],
        'destinations': [key for key, (op, _) in nodes.items() if op == 'outputDelimited'],
        'materialization': 'allDeclaredPorts',
        'demand':demand.to_document(),
    }
    if workload_policy is not None:
        if __package__:from .workload_policy import WorkloadPolicy
        else:from workload_policy import WorkloadPolicy
        policy=workload_policy if isinstance(workload_policy,WorkloadPolicy) else WorkloadPolicy.from_document(workload_policy)
        document['schemaVersion']=3
        document['workloadPolicy']=policy.to_document()
        if __package__:from .preparation_kernels import build_kernel_plan
        else:from preparation_kernels import build_kernel_plan
        document['kernelPlan']=build_kernel_plan(nodes)
    return PreparationPlan(_canonical(document))


def validate_plan(document):
    """Reject altered identities, unrecognized scopes and runtime registry drift."""
    if isinstance(document, PreparationPlan):
        document = document.to_document()
    if not isinstance(document, dict):
        raise ValueError('Preparation plan must be an object')
    if document.get('registryDigest') != dataflow.digest(dataflow.REGISTRY):
        raise ValueError('Preparation plan registry does not match this runtime')
    try:
        engine = document['engine']
        demand=document.get('demand')
        if document.get('schemaVersion') in (2,3) and not isinstance(demand,dict):
            raise ValueError('Preparation plan requires captured demand')
        expected = build_plan(document['graph'], engine=engine['id'],
                              engine_version=engine['version'], node_labels=document['nodeLabels'],
                              sampling=demand.get('sampling') if demand else None,
                              complete_nodes=demand.get('completeNodeIDs') if demand else None,
                              workload_policy=document['workloadPolicy'] if document.get('schemaVersion')==3 else None)
        if document.get('schemaVersion')==1:
            legacy=expected.to_document();legacy.pop('demand');legacy['schemaVersion']=1
            expected=PreparationPlan(_canonical(legacy))
        elif document.get('schemaVersion')==3:
            if __package__:from .preparation_kernels import validate_kernel_plan
            else:from preparation_kernels import validate_kernel_plan
            nodes,_,_=dataflow.validate(document['graph'])
            captured=expected.to_document()
            captured['kernelPlan']=validate_kernel_plan(document.get('kernelPlan'),nodes)
            expected=PreparationPlan(_canonical(captured))
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError('Invalid preparation plan structure') from error
    if _canonical(document) != expected.canonical_json:
        raise ValueError('Preparation plan metadata or identity does not match its validated graph')
    return expected


def _installed_engine_version(engine):
    if engine == 'disk':
        return DISK_ENGINE_VERSION
    try:
        return importlib.metadata.version('polars')
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError('The recorded Polars preparation engine is not installed') from error


def execute_plan(plan, source_bindings, *, destination_bindings=None,
                 check_cancelled=lambda: None, **execution_options):
    """Execute the same validated snapshot used by Code with caller-owned grants.

    Scientific algorithms, output transactions, callbacks and cancellation remain
    in the shared dataflow engine. No graph/code evaluation occurs in this layer.
    """
    check_cancelled()
    captured = validate_plan(plan)
    document = captured.to_document()
    engine = document['engine']
    if _installed_engine_version(engine['id']) != engine['version']:
        raise ValueError('Preparation engine version does not match the captured plan')
    if set(execution_options) & {'engine', 'node_labels'}:
        raise ValueError('Execution cannot override the captured engine or node labels')
    if set(execution_options)&{'demand_plan','kernel_plan'}:raise ValueError('Execution cannot override captured preparation demand')
    if __package__:from .preparation_demand import DemandPlan,build_demand_plan
    else:from preparation_demand import DemandPlan,build_demand_plan
    demand=DemandPlan(json.dumps(document['demand'],sort_keys=True,separators=(',',':'),allow_nan=False)) if 'demand' in document else build_demand_plan(document['graph'])
    if __package__:from .workload_policy import WorkloadPolicy,workload_scope
    else:from workload_policy import WorkloadPolicy,workload_scope
    policy=WorkloadPolicy.from_document(document['workloadPolicy']) if 'workloadPolicy' in document else WorkloadPolicy()
    with workload_scope(policy):
        return dataflow.execute(document['graph'], source_bindings,
                            engine=engine['id'], node_labels=document['nodeLabels'],
                            demand_plan=demand,kernel_plan=document.get('kernelPlan'),
                            destination_bindings=destination_bindings,
                            check_cancelled=check_cancelled, **execution_options)


def derive_preparation_code(plan):
    """Readable executable replay using the shipped, version-pinned runtime APIs.

    JSON literals and escaped single-line comments cannot become arbitrary code.
    This script needs the exported runtime modules and the recorded dependencies;
    it does not claim to be an independent implementation of each algorithm.
    """
    captured = validate_plan(plan)
    document = captured.to_document()
    _, _, order = dataflow.validate(document['graph'])
    nodes = {node['id']: node for node in document['graph']['nodes']}
    lines = [
        '# Kwip ML — Data preparation only',
        '# Table source adapters → typed.table → Polars preparation (exact fallback where required).'
        if document['engine']['id'] == 'polars' else '# Table source adapters → typed.table → disk preparation.',
        '# Uses the accompanying Kwip ML runtime modules; this is not a model or training export.',
        '# Source/destination bindings are separate local JSON files, never embedded grants.',
        '# Node outputs keep the shared runtime row identities, diagnostics and provenance.',
        f'# Plan SHA-256: {captured.plan_digest}',
        f'# Graph SHA-256: {captured.graph_digest}',
        f'# Engine: {document["engine"]["id"]} {document["engine"]["version"]}',
        'import argparse',
        'import json',
        '',
        'try:',
        '    from runner.preparation_plan import execute_plan',
        '    from runner.dataflow_storage import bounded_regular_read',
        '    from runner.dataflow_cli import output_preview, bound_preview',
        'except ModuleNotFoundError as error:',
        "    if error.name != 'runner':",
        '        raise',
        '    from preparation_plan import execute_plan',
        '    from dataflow_storage import bounded_regular_read',
        '    from dataflow_cli import output_preview, bound_preview',
        '',
        '# The graph below is the validated canvas snapshot. Configuration is JSON data.',
        'nodes = []',
    ]
    # Keep the original graph order for its existing graph digest. Describe its
    # topological execution order independently instead of rewriting identities.
    lines.append('# Execution order: ' + json.dumps(order, ensure_ascii=True))
    for node in document['graph']['nodes']:
        op = node['operator'][len(dataflow.PREFIX):]
        label = document['nodeLabels'].get(node['id'], dataflow.DEFINITIONS[op]['title'])
        lines.append('# ' + json.dumps({'node': node['id'], 'tool': dataflow.DEFINITIONS[op]['title'],
                                       'label': label}, ensure_ascii=True))
        lines.append('nodes.append(json.loads(' + repr(_canonical(node)) + '))')
    lines += ['edges = json.loads(' + repr(_canonical(document['graph']['edges'])) + ')']
    metadata = {key: value for key, value in document.items() if key != 'graph'}
    lines += [
        'PLAN = json.loads(' + repr(_canonical(metadata)) + ')',
        "PLAN['graph'] = {'schemaVersion': 1, 'nodes': nodes, 'edges': edges}",
        '',
        'def main():',
        "    parser = argparse.ArgumentParser(description='Replay the captured Kwip ML preparation plan.')",
        "    parser.add_argument('--source-bindings', required=True, help='Local JSON map: source node ID to source path')",
        "    parser.add_argument('--destination-bindings', help='Local JSON output grants required by output nodes')",
        '    args = parser.parse_args()',
        '    def read_bindings(path):',
        '        return json.loads(bounded_regular_read(path, 2_000_000))',
        '    def completed(node, ports):',
        "        event = {'event': 'preparationNodeCompleted', 'nodeID': node,",
        "                 'outputs': {port: output_preview(value) for port, value in ports.items()}}",
        '        bound_preview(event)',
        '        print(json.dumps(event, allow_nan=False))',
        '    execute_plan(PLAN, read_bindings(args.source_bindings),',
        '                 destination_bindings=read_bindings(args.destination_bindings) if args.destination_bindings else {},',
        '                 on_node=completed, retain_results=False)',
        "    print(json.dumps({'event': 'preparationPlanCompleted',",
        f"                      'planDigest': {captured.plan_digest!r}, 'graphDigest': {captured.graph_digest!r}}}))",
        '',
        "if __name__ == '__main__':",
        '    main()',
        '',
    ]
    if any(node['operator']==dataflow.PREFIX+'sourceFiles' for node in document['graph']['nodes']):
        # Image references require the same verified artifact publisher as a Run.
        # Delegate instead of printing references whose pixel closure is absent.
        start=lines.index('def main():')
        lines[start:]=[
            'def main():',
            "    parser = argparse.ArgumentParser(description='Replay the captured File Input preparation plan.')",
            "    parser.add_argument('--source-bindings', required=True, help='External source path/collection bindings JSON')",
            "    parser.add_argument('--destination-bindings', help='External destination bindings JSON')",
            "    parser.add_argument('--output', required=True, help='New run directory including verified table and pixel artifacts')",
            '    args = parser.parse_args()',
            '    from pathlib import Path',
            '    import tempfile',
            '    try:',
            '        from runner.dataflow_cli import run_dataflow',
            '        from runner.workload_policy import WorkloadPolicy',
            '    except ModuleNotFoundError as error:',
            "        if error.name != 'runner': raise",
            '        from dataflow_cli import run_dataflow',
            '        from workload_policy import WorkloadPolicy',
            "    with tempfile.TemporaryDirectory(prefix='kwip-file-replay-') as temporary:",
            "        graph = Path(temporary) / 'graph.json'",
            "        graph.write_text(json.dumps(PLAN['graph'], allow_nan=False))",
            "        run_dataflow(graph, json.loads(bounded_regular_read(args.source_bindings, 2_000_000)), args.output,",
            '                     lambda event: print(json.dumps(event, allow_nan=False)),',
            "                     engine=PLAN['engine']['id'], node_labels=PLAN['nodeLabels'],",
            "                     workload_policy=PLAN.get('workloadPolicy',WorkloadPolicy().to_document()),",
            "                     sampling=PLAN['demand']['sampling'], complete_nodes=PLAN['demand']['completeNodeIDs'],",
            "                     destination_bindings=json.loads(bounded_regular_read(args.destination_bindings, 2_000_000)) if args.destination_bindings else {})",
            '', "if __name__ == '__main__':", '    main()', '',
        ]
    return '\n'.join(lines)
