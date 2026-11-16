"""Independently verify the four prepared routing classifiers from saved weights.

Expected values use only Python scalar arithmetic. Torch is used solely to read
checkpoints and measure the saved portable model, never to compute expectations.
This intentionally accepts the exact four bundled examples, not arbitrary graphs.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def affine(weight, bias, values, width, count):
    require(len(values) == width and len(weight) == len(bias) == count
            and all(len(row) == width for row in weight), 'Saved affine shape differs from the prepared example.')
    require(all(math.isfinite(v) for row in weight for v in row)
            and all(math.isfinite(v) for v in bias + values), 'Expected finite saved weights and observations.')
    return [math.fsum([offset, *[w*x for w,x in zip(row, values)]]) for row,offset in zip(weight,bias)]


def scalar_reference(operation, state, features):
    """One observation, with row-major flattened traces for every named output."""
    require(operation in ('split','chunk','gather','einsum'), 'Unknown prepared routing example.')
    require(len(features) == 2 and all(math.isfinite(v) for v in features), 'Expected two finite measurements.')
    result = {'input': features}
    def project(name, count):
        values = affine(state[f'layers.{name}-learn.weight'], state[f'layers.{name}-learn.bias'], features, 2, count)
        result[name+'-learn'] = result[name] = values
        return values
    if operation == 'einsum':
        left, right = project('a',6), project('b',6)
        routed = [math.fsum(left[row*3+k]*right[k*2+col] for k in range(3))
                  for row in range(2) for col in range(2)]
        result['routing'] = routed
    elif operation == 'gather':
        values = project('array',8)
        selectors, routed = [], []
        for row in range(2):
            for start in (0,2):
                # MaxPool's first index wins equal maxima, within each sample row.
                index = start if values[row*4+start] >= values[row*4+start+1] else start+1
                selectors.append(index); routed.append(values[row*4+index])
        result['positions'] = {'out':routed, 'indices':selectors}
        result['routing'] = routed
    else:
        extent = 6 if operation == 'split' else 13
        values = project('array',2*extent)
        sections = [(0,2),(2,4),(4,6)] if operation == 'split' else [(0,3),(3,6),(6,9),(9,12),(12,13)]
        parts = {f'part{i}':[v for row in range(2) for v in values[row*extent+start:row*extent+end]]
                 for i,(start,end) in enumerate(sections)}
        result['routing'] = parts
        if operation == 'split':
            result['join-first'] = [v for row in range(2) for part in ('part2','part0') for v in parts[part][row*2:row*2+2]]
            routed = [v for row in range(2) for part in ('part2','part0','part1') for v in parts[part][row*2:row*2+2]]
            result['join-all'] = routed
        else:
            routed = parts['part4']
    result['flat'] = routed
    result['head'] = result['output'] = affine(state['layers.head.weight'], state['layers.head.bias'], routed, len(routed), 2)
    return result


def verify_graph(graph, operation):
    require(graph['schemaVersion'] == 2, 'Expected schema-2 prepared model.')
    expected = {'input':('input',{'features':2}), 'flat':('flatten',{'start_dim':1,'end_dim':-1}),
                'output':('output',{})}
    wires = [('flat','out','head','in'), ('head','out','output','in')]
    def branch(name, shape):
        expected[name+'-learn'] = ('linear',{'in_features':2,'out_features':math.prod(shape),'bias':True})
        expected[name] = ('reshape',{'shape':shape})
        wires.extend([('input','out',name+'-learn','in'),(name+'-learn','out',name,'in')])
    if operation == 'split':
        branch('array',[2,6]); width=12
        expected.update({'routing':('split',{'mode':'size','dim':2,'split_size':2,'sections':[1,1]}),
                         'join-first':('concat',{'dim':2}), 'join-all':('concat',{'dim':2})})
        wires.extend([('array','out','routing','in'), ('routing','part2','join-first','a'),
                      ('routing','part0','join-first','b'), ('routing','part1','join-all','b'),
                      ('join-first','out','join-all','a'), ('join-all','out','flat','in')])
    elif operation == 'chunk':
        branch('array',[2,13]); width=2
        expected['routing'] = ('chunk',{'chunks':6,'dim':2})
        wires.extend([('array','out','routing','in'),('routing','part4','flat','in')])
    elif operation == 'gather':
        branch('array',[2,4]); width=4
        expected['positions'] = ('max_pool1d',{'kernel_size':2,'stride':2,'padding':0,'dilation':1,'ceil_mode':False,'return_indices':True})
        expected['routing'] = ('gather',{'dim':2})
        wires.extend([('array','out','positions','in'),('array','out','routing','source'),
                      ('positions','indices','routing','indices'),('routing','out','flat','in')])
    elif operation == 'einsum':
        branch('a',[2,3]); branch('b',[3,2]); width=4
        expected['routing'] = ('einsum',{'pattern':'multiply','equation':'nab,nbc->nac'})
        wires.extend([('a','out','routing','a'),('b','out','routing','b'),('routing','out','flat','in')])
    else:
        raise ValueError('Unknown prepared routing example.')
    expected['head'] = ('linear',{'in_features':width,'out_features':2,'bias':True})
    actual = {node['id']:(node['type'],node['config']) for node in graph['nodes']}
    require(len(actual) == len(graph['nodes']) and actual == expected, 'Saved architecture/configuration differs from the prepared routing example.')
    edges = [(e['source'],e['sourcePort'],e['target'],e['targetPort']) for e in graph['edges']]
    require(len(edges) == len(wires) and set(edges) == set(wires), 'Saved named-port connectivity differs from the independent reference.')
    return [node for node,(kind,_) in expected.items() if kind == 'linear']


def sample_shapes(operation):
    shapes = {'input':[2], 'head':[2], 'output':[2]}
    if operation == 'einsum':
        shapes.update({'a-learn':[6], 'a':[2,3], 'b-learn':[6], 'b':[3,2], 'routing':[2,2], 'flat':[4]})
    elif operation == 'gather':
        shapes.update({'array-learn':[8], 'array':[2,4], 'positions':{'out':[2,2], 'indices':[2,2]}, 'routing':[2,2], 'flat':[4]})
    elif operation == 'split':
        shapes.update({'array-learn':[12], 'array':[2,6], 'routing':{f'part{i}':[2,2] for i in range(3)},
                       'join-first':[2,4], 'join-all':[2,6], 'flat':[12]})
    else:
        shapes.update({'array-learn':[26], 'array':[2,13], 'routing':{f'part{i}':[2,3 if i<4 else 1] for i in range(5)}, 'flat':[2]})
    return shapes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def compare(actual, expected, atol, rtol, name):
    require(len(actual) == len(expected) and len(actual)>0, f'{name}: missing or mismatched measured values.')
    require(all(math.isfinite(v) for v in actual+expected), f'{name}: nonfinite values.')
    errors = [abs(a-b) for a,b in zip(actual,expected)]
    require(all(error <= atol+rtol*abs(wanted) for error,wanted in zip(errors,expected)), f'{name}: scalar reference mismatch (maximum {max(errors)}).')
    return max(errors)


def verify(args):
    import torch
    package=args.package.resolve()
    artifacts=(args.artifacts or (args.archive/'training/artifacts' if args.archive else package)).resolve()
    manifest, predictions, activations = [read(artifacts/name) for name in ('manifest.json','predictions.json','activations.json')]
    require(manifest['runID'] == predictions['runID'] == activations['runID'], 'Worker artifact run identities differ.')
    require(manifest['runtime']['device'] == args.training_device, 'Training device differs from the requested receipt.')
    require(not manifest['runtime'].get('operationDevices'), 'Unexpected operation fallback.')
    affine_nodes=verify_graph(manifest['graph'],args.component)
    identity={}
    if args.archive:
        capture=read(args.archive/'capture.json'); export=read(package/'export-manifest.json')
        require(export['runID'] == capture['runID'] and export['requestIdentity'] == capture['requestIdentity']
                and export['sourceRevision'] == capture['revision'], 'Export/capture ownership differs.')
        for option,key in [('run_id','runID'),('request','requestIdentity'),('project_id','projectID'),('revision','revision')]:
            value=getattr(args,option)
            if value is not None: require(capture[key] == value, f'Unexpected capture {key}.')
        require(export['modelManifestSHA256'] == sha(artifacts/'manifest.json'), 'Export manifest source hash differs.')
        for entry in export['files']:
            path=(package/entry['path']).resolve()
            require(path.is_relative_to(package) and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Export file integrity differs.')
        identity={key:capture[key] for key in ('runID','requestIdentity','projectID','revision','directoryID')}
    for name in ('model.py','checkpoint.pt'):
        require(sha(package/name) == sha(artifacts/name), f'Exported {name} differs from the saved run.')
    checkpoint=torch.load(package/'checkpoint.pt',map_location='cpu',weights_only=True)
    tensors=checkpoint['modelState']
    require(set(tensors) == {f'layers.{node}.{key}' for node in affine_nodes for key in ('weight','bias')}, 'Unexpected learned state.')
    require(checkpoint['epoch'] == manifest['checkpoint']['completedEpoch'], 'Checkpoint epoch differs.')
    state={key:value.tolist() for key,value in tensors.items()}
    rows=predictions['predictionRows']
    require(0<len(rows)<=manifest['evaluation']['count'], 'Captured validation observations are unavailable.')
    features=[row['input'] for row in rows]
    refs=[scalar_reference(args.component,state,row) for row in features]
    expected=[v for record in refs for v in record['output']]
    contract_path=package/'input-contract.json'
    contract=read(contract_path) if contract_path.exists() else None
    if contract:
        require(contract['input']['dtype'] == 'float32' and contract['input']['shape'] == ['N',2], 'Unexpected portable input contract.')
    validation_path=args.validation_table or (args.archive/'preparation/artifacts/node-006.json' if args.archive else None)
    if validation_path:
        require(contract is not None, 'Prepared-table verification requires the exported input contract.')
        validation=read(validation_path); columns=[column['name'] for column in validation['columns']]
        indices=[columns.index(name) for name in contract['input']['featureColumns']]
        prepared=[[row[i] for i in indices] for row in validation['rows'][:len(rows)]]
        require(torch.equal(torch.tensor(prepared,dtype=torch.float32),torch.tensor(features,dtype=torch.float32)) and len(prepared) == len(rows), 'Predictions differ from prepared validation rows.')
        require([int(row[columns.index('label')]) for row in validation['rows'][:len(rows)]] == [row['label'] for row in rows], 'Prepared labels differ.')
    spec=importlib.util.spec_from_file_location('saved_tensor_routing_model',package/'model.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    devices=('cpu','mps') if args.inference_device == 'both' else (args.inference_device,)
    device_results={}
    for device in devices:
        model=module.Model().to(device); model.load_state_dict(tensors,strict=True); model.eval()
        x=torch.tensor(features,dtype=torch.float32,device=device)
        with torch.inference_mode():
            actual=model(x).cpu(); singles=torch.cat([model(row[None,:]).cpu() for row in x])
        error=compare(actual.flatten().tolist(),expected,args.absolute_tolerance,args.relative_tolerance,device+' whole model')
        batch_error=compare(singles.flatten().tolist(),actual.flatten().tolist(),args.absolute_tolerance,args.relative_tolerance,device+' batching')
        classes=[max(range(2),key=lambda i:record['output'][i]) for record in refs]
        require(actual.argmax(1).tolist() == classes == [row['predicted'] for row in rows], 'Independent class decisions differ.')
        device_results[device]={'scalarWholeModelMaximumError':error,'batchOneMaximumError':batch_error,'classIndicesAllMatch':True}
    captured={}
    shapes=sample_shapes(args.component)
    require(set(activations['previews']) == set(refs[0]), 'Captured stages differ from the reference graph.')
    for node,preview in activations['previews'].items():
        require(preview.get('captureStatus','captured') == 'captured', 'Activation capture is unavailable.')
        pieces=preview['outputs'] if 'outputs' in preview else {'out':preview}
        expected_ports=refs[0][node] if isinstance(refs[0][node],dict) else {'out':refs[0][node]}
        require(set(pieces) == set(expected_ports), 'Captured named ports differ.')
        for port,piece in pieces.items():
            shape=shapes[node][port] if isinstance(shapes[node],dict) else shapes[node]
            require(piece['shape'] == [min(8,len(rows)),*shape], 'Captured shape differs from the independently resolved stage.')
            require(piece['dtype'] == ('int64' if port=='indices' else 'float32'), 'Captured dtype differs from the source/selector contract.')
            flat=[v for record in refs for v in (record[node][port] if isinstance(record[node],dict) else record[node])]
            observed=piece['values']
            error=compare(observed,flat[:len(observed)],0 if port=='indices' else args.absolute_tolerance,0 if port=='indices' else args.relative_tolerance,node+'.'+port)
            captured[node+'.'+port]={'capturedValuesCompared':len(observed),'fullShape':piece['shape'],'scalarMaximumError':error}
    helper_result=None
    if (package/'inference.py').exists():
        scope={'__file__':str(package/'inference.py'),'__name__':'saved_routing_inference'}
        exec(compile((package/'inference.py').read_text(),str(package/'inference.py'),'exec'),scope)
        predictor=scope['load_model'](); x=torch.tensor(features,dtype=torch.float32)
        helper=predictor.predict(x,batch_size=len(rows))
        helper_result=compare(helper.flatten().tolist(),expected,args.absolute_tolerance,args.relative_tolerance,'inference helper')
        cli=subprocess.run([sys.executable,str(package/'inference.py'),'--example','--batch-size','1'],capture_output=True,text=True)
        require(cli.returncode == 0, 'Saved inference CLI failed: '+cli.stderr)
    return {'status':'passed','component':args.component,'identity':identity,'workerRunID':manifest['runID'],
            'trainingDevice':manifest['runtime']['device'],'python':sys.version.split()[0],'torch':torch.__version__,
            'checkpointEpoch':checkpoint['epoch'],'heldOutRowsCompared':len(rows),'inferenceDevices':device_results,
            'helperScalarMaximumError':helper_result,'capturedPreviewComparisons':captured,
            'absoluteTolerance':args.absolute_tolerance,'relativeTolerance':args.relative_tolerance,
            'sha256':{name:sha(package/name) for name in ('model.py','checkpoint.pt')},'manifestSHA256':sha(artifacts/'manifest.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',required=True,choices=('split','chunk','gather','einsum'))
    parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--artifacts',type=Path)
    parser.add_argument('--validation-table',type=Path)
    parser.add_argument('--training-device',required=True,choices=('cpu','mps'))
    parser.add_argument('--inference-device',default='cpu',choices=('cpu','mps','both'))
    parser.add_argument('--run-id'); parser.add_argument('--request'); parser.add_argument('--project-id')
    parser.add_argument('--revision',type=int)
    parser.add_argument('--absolute-tolerance',type=float,default=2e-6)
    parser.add_argument('--relative-tolerance',type=float,default=2e-6)
    args=parser.parse_args()
    require(args.absolute_tolerance>=0 and args.relative_tolerance>=0 and math.isfinite(args.absolute_tolerance+args.relative_tolerance), 'Expected finite nonnegative tolerances.')
    print(json.dumps(verify(args),indent=2))


if __name__ == '__main__': main()
