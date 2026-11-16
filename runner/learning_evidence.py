"""Application-owned execution of small learning fixtures. No submitted result receipts or code run here."""
import csv
from contextlib import contextmanager, ExitStack
import io
import os
import uuid
import hashlib
import json
import math
from pathlib import Path
import stat
import tempfile
import time
import numpy as np
import torch
if __package__:
    from .artifact_storage import ArtifactStorage
    from .worker_authorization import directory
    from .product_execution import ProductExecutionAdapter
    from .engine import GraphModel,validate
    from . import dataflow
    from . import learning_tensor_transforms, learning_layer_norm, learning_embedding, learning_embedding_bag, learning_matrix, learning_bilinear, learning_lazy_linear, learning_convolution, learning_pooling, learning_configurable_activation, learning_adaptive_pooling, learning_tensor_composition, learning_remaining_activation, learning_tensor_routing, learning_spatial_patches, learning_stateful_normalization, learning_optimizers, learning_schedulers, learning_recurrent, learning_attention, learning_losses, learning_checkpoints, learning_initialization, learning_gradients, module_composition_learning, learning_data_loading, learning_program_export
else:
    from artifact_storage import ArtifactStorage
    from worker_authorization import directory
    from product_execution import ProductExecutionAdapter
    from engine import GraphModel,validate
    import dataflow
    import learning_tensor_transforms, learning_layer_norm, learning_embedding, learning_embedding_bag, learning_matrix, learning_bilinear, learning_lazy_linear, learning_convolution, learning_pooling, learning_configurable_activation, learning_adaptive_pooling, learning_tensor_composition, learning_remaining_activation, learning_tensor_routing, learning_spatial_patches, learning_stateful_normalization, learning_optimizers, learning_schedulers, learning_recurrent, learning_attention, learning_losses, learning_checkpoints, learning_initialization, learning_gradients, module_composition_learning, learning_data_loading, learning_program_export

MAX_FIXTURE_BYTES=65_536

VERSION={'major':1,'minor':0,'patch':0}
AFFINE_OPTIONAL_BIAS_VERSION={'major':1,'minor':1,'patch':0}
AFFINE_TYPES=frozenset({'input','linear','add','concat','output'})
COMPOSITION_TYPES=AFFINE_TYPES|{'relu','tanh'}
ACTIVATION_LABELS={'relu':'ReLU','tanh':'Tanh','sigmoid':'Sigmoid'}
ASSESSMENT_POLICIES={
    ('assessment.frozen-standardization',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES},
    ('assessment.affine-composition',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':VERSION},'submissionSchema':1,'allowedTypes':COMPOSITION_TYPES},
    ('assessment.affine-composition',(1,1,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION},'submissionSchema':2,'allowedTypes':COMPOSITION_TYPES},
    ('assessment.affine-from-probes',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES},
    ('assessment.affine-backward-chain',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES},
    ('assessment.activation-relu',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION,'relu':AFFINE_OPTIONAL_BIAS_VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES|{'relu'},'requiredExecutedType':'relu'},
    ('assessment.activation-tanh',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION,'tanh':AFFINE_OPTIONAL_BIAS_VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES|{'tanh'},'requiredExecutedType':'tanh'},
    ('assessment.activation-sigmoid',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION,'sigmoid':VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES|{'sigmoid'},'requiredExecutedType':'sigmoid'},
    ('assessment.activation-gelu-exact',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION,'gelu':VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES|{'gelu'},'requiredGeluMode':False},
    ('assessment.activation-gelu-tanh',(1,0,0)):{'backend':'pytorch','capabilities':{'linear':AFFINE_OPTIONAL_BIAS_VERSION,'gelu':VERSION},'submissionSchema':2,'allowedTypes':AFFINE_TYPES|{'gelu'},'requiredGeluMode':True},
    ('assessment.training-standardization',(1,0,0)):{'backend':'numpy','capabilities':{'kwip.data.standardScalerFit':VERSION,'kwip.data.transform':VERSION},'submissionSchema':1,'allowedTypes':None},
    ('assessment.training-standardization',(1,1,0)):{'backend':'numpy','capabilities':{'kwip.data.standardScalerFit':VERSION,'kwip.data.transform':VERSION},'submissionSchema':1,'allowedTypes':frozenset('kwip.data.'+name for name in ('sourceCSV','cast','split','standardScalerFit','transform'))},
    ('assessment.training-standardization',(1,2,0)):{'backend':'numpy','capabilities':{'kwip.data.standardScalerFit':VERSION,'kwip.data.transform':VERSION},'submissionSchema':1,'measureTraining':True,'allowedTypes':frozenset('kwip.data.'+name for name in ('sourceCSV','cast','split','standardScalerFit','transform'))},
}

ASSESSMENT_POLICIES[('assessment.lazy-linear-inferred-width',(1,0,0))]={'backend':'pytorch','capabilities':{'lazy_linear':VERSION},'submissionSchema':1,'lazyLinear':True}

for dimension in (1,2,3):
    ASSESSMENT_POLICIES[(f'assessment.conv{dimension}d-shared-kernel',(1,0,0))]={'backend':'pytorch','capabilities':{f'conv{dimension}d':VERSION},'submissionSchema':1,'convolution':dimension}

for dimension in (1,2,3):
    ASSESSMENT_POLICIES[(f'assessment.conv-transpose{dimension}d-shared-kernel',(1,0,0))]={'backend':'pytorch','capabilities':{f'conv_transpose{dimension}d':VERSION},'submissionSchema':1,'transposedConvolution':dimension}

for dimension in (1,2,3):
    ASSESSMENT_POLICIES[(f'assessment.max-pool{dimension}d-windows',(1,0,0))]={'backend':'pytorch','capabilities':{f'max_pool{dimension}d':VERSION},'submissionSchema':1,'pooling':dimension,'average':False}
ASSESSMENT_POLICIES[('assessment.avg-pool1d-windows',(1,0,0))]={'backend':'pytorch','capabilities':{'avg_pool1d':VERSION},'submissionSchema':1,'pooling':1,'average':True}

for task in learning_data_loading.TASKS:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'dataset_source':VERSION,'data_loader':VERSION},'submissionSchema':1,'dataLoading':task}
for task in ('program-export-computation','program-export-size-range'):
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'program_export':VERSION},'submissionSchema':1,'programExport':task}

for task,component in learning_losses.TASKS.items():
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'lossObjective':task}

for task,component in learning_attention.TASKS.items():
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION,'attention_mask':VERSION},'submissionSchema':1,'attention':task}
for task,component in learning_recurrent.TASKS.items():
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'recurrent':task}

for task,component in [('avg-pool2d-windows','avg_pool2d'),('adaptive-avg-pool1d-bins','adaptive_avg_pool1d'),('adaptive-avg-pool2d-bins','adaptive_avg_pool2d'),('adaptive-max-pool2d-bins','adaptive_max_pool2d')]:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'adaptivePooling':component}

for task,component in [('silu-values','silu'),('leaky-relu-slope','leaky_relu'),('elu-alpha','elu')]:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'configurableActivation':component}

for task,component in [('concat-sample-axis','concat'),('add-shared-pattern','add'),('stack-new-direction','stack'),('permute-sample-order','permute')]:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'tensorComposition':component}

for task,component in [('split-size-order','split'),('split-sections-order','split'),('chunk-uneven-parts','chunk'),('gather-repeated-positions','gather'),('einsum-combine-tables','einsum'),('einsum-diagonal-view','einsum')]:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'tensorRouting':task}

for task,(component,_,_,_) in learning_stateful_normalization.TASKS.items():
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'statefulNormalization':task}

for task,component in [('unfold-overlapping-windows','unfold'),('unfold-padded-spaced-windows','unfold'),('fold-overlap-addition','fold'),('fold-uncovered-positions','fold')]:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'spatialPatch':task}

for operation in ('reshape','transpose','flatten'):
    ASSESSMENT_POLICIES[(f'assessment.tensor-{operation}',(1,0,0))]={'backend':'pytorch','capabilities':{operation:VERSION},'submissionSchema':1,'tensorTransforms':True}

for axis_name in ('last-axis','first-sample-axis'):
    ASSESSMENT_POLICIES[(f'assessment.softmax-{axis_name}',(1,0,0))]={'backend':'pytorch','capabilities':{'softmax':VERSION},'submissionSchema':1,'tensorTransforms':True,'allowedTensorOperations':frozenset({'softmax','reshape','transpose','flatten'})}

for task in ('rows','sample','affine'):
    ASSESSMENT_POLICIES[(f'assessment.layer-norm-{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'layer_norm':VERSION},'submissionSchema':1,'layerNorm':True}

for task in ('bag-fixed-sum','bag-packed-mean','bag-packed-max','bag-fixed-weighted-sum','bag-packed-weighted-sum','bag-padding-norm'):
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'bag_input':VERSION,'embedding_bag':VERSION},'submissionSchema':1,'embeddingBag':True}

for task in ('gather','padding-frequency','renormalize'):
    ASSESSMENT_POLICIES[(f'assessment.embedding-{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'embedding':VERSION},'submissionSchema':1,'embedding':True}

for task in ('contraction','broadcast-gradients'):
    ASSESSMENT_POLICIES[(f'assessment.matrix-{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'matmul':VERSION},'submissionSchema':1,'matrix':True}

for task,bias in (('with-bias',True),('without-bias',False)):
    ASSESSMENT_POLICIES[(f'assessment.bilinear-{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'bilinear':VERSION},'submissionSchema':1,'bilinear':True,'biasEnabled':bias}

for task,component in learning_optimizers.TASKS.items():
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION},'submissionSchema':1,'optimizer':task}

for task,component in learning_schedulers.TASKS.items():
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:VERSION,'optimizer_sgd':VERSION},'submissionSchema':1,'scheduler':task}

for task in learning_checkpoints.TASKS:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'checkpoint_lifecycle':VERSION},'submissionSchema':1,'checkpointLifecycle':task}

for task in learning_gradients.TASKS:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'gradient_policy':VERSION,'trainability':VERSION,'optimizer_sgd':VERSION},'submissionSchema':1,'gradientLearning':task}

for task in learning_initialization.TASKS:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'initialization':VERSION,'optimizer_sgd':VERSION},'submissionSchema':1,'initialization':task}

for task in module_composition_learning.TASKS:
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{'moduleComposition':VERSION},'submissionSchema':1,'moduleComposition':task}


def assessment_policy(request):
    version=request.get('assessmentVersion');ident=request.get('assessmentID')
    if not isinstance(ident,str) or not isinstance(version,dict) or set(version)!={'major','minor','patch'} or any(type(v) is not int for v in version.values()):raise ValueError('Unsupported learning assessment request')
    policy=ASSESSMENT_POLICIES.get((ident,tuple(version[k] for k in ('major','minor','patch'))))
    if policy is None:raise ValueError('Unsupported learning assessment request')
    return policy


def runtime_versions(registry=None):
    if registry is not None:
        descriptor=strict_json(bounded_read(registry))
        if not isinstance(descriptor,dict) or type(descriptor.get('schemaVersion')) is not int or descriptor['schemaVersion'] not in (1,2) or not isinstance(descriptor.get('components'),list):raise ValueError('Unsupported learning component registry')
    return {'pytorch':str(torch.__version__),'numpy':np.__version__}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def strict_json(raw):
    def invalid(_):raise ValueError('Learning input requires finite JSON numbers')
    return json.loads(raw,parse_constant=invalid)
def bounded_read(path):
    path=Path(path);info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>2_000_000:raise ValueError('Learning input must be a bounded regular file')
    with path.open('rb') as stream:raw=stream.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('Learning input exceeds 2 MB')
    return raw

for task,component in learning_remaining_activation.TASKS.items():
    ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]={'backend':'pytorch','capabilities':{component:{'major':1,'minor':2,'patch':0}},'submissionSchema':1,'remainingActivation':task}


def tensor(value):
    if not isinstance(value,dict) or set(value)!={'shape','values'} or not isinstance(value['shape'],list) or not 1<=len(value['shape'])<=8 or any(type(v) is not int or not 1<=v<=4096 for v in value['shape']):raise ValueError('Malformed fixture/parameter tensor shape')
    count=math.prod(value['shape'])
    if count>4096 or not isinstance(value['values'],list) or len(value['values'])!=count or not all(finite(v) for v in value['values']):raise ValueError('Malformed/nonfinite or oversized fixture/parameter tensor')
    return value['shape'],value['values']

def validate_request(request):
    required={'assessmentID','assessmentVersion','fixtureDigest','runtime','budget','fixtures'}
    if not isinstance(request,dict) or set(request)!=required:raise ValueError('Unsupported learning assessment request')
    policy=assessment_policy(request)
    attention_keys=None
    if policy.get('attention'):
        operation=learning_attention.TASKS[policy['attention']]
        attention_keys=set(learning_attention.roles(operation)) | {'parameter.'+p['name'] for p in learning_attention.parameter_shapes(operation,learning_attention.target(operation))} | {'upstream.'+r for r in learning_attention.outputs(operation)}
    if not isinstance(request['fixtureDigest'],str) or len(request['fixtureDigest'])!=64 or any(c not in '0123456789abcdef' for c in request['fixtureDigest']):raise ValueError('Fixture identity requires a SHA-256 digest')
    budget=request['budget']
    if not isinstance(budget,dict) or set(budget)!={'milliseconds','maximumTensorElements'} or not finite(budget['milliseconds']) or not 0<budget['milliseconds']<=60000 or type(budget['maximumTensorElements']) is not int or not 1<=budget['maximumTensorElements']<=1_000_000:raise ValueError('Invalid assessment compute budget')
    fixtures=request['fixtures']
    if not isinstance(fixtures,list) or not 1<=len(fixtures)<=32:raise ValueError('Assessment fixture bound exceeded')
    if policy.get('moduleComposition'):
        expected_elements=sum(math.prod(shape) for shape in module_composition_learning.SHAPES[policy['moduleComposition']].values())
        if len(fixtures)!=4 or not expected_elements<=budget['maximumTensorElements']<=4096:
            raise ValueError('Composition lessons require four complete fixtures and room for their measured tensors.')
    ids=[]
    for fixture in fixtures:
        if not isinstance(fixture,dict) or set(fixture)!={'id','inputs'} or not isinstance(fixture['id'],str) or not 1<=len(fixture['id'])<=128 or not isinstance(fixture['inputs'],dict) or not 1<=len(fixture['inputs'])<=(len(attention_keys) if attention_keys is not None else 16):raise ValueError('Invalid learning fixture')
        if attention_keys is not None and set(fixture['inputs'])!=attention_keys:raise ValueError('Attention fixture sequences, learned tables and result influence require the exact trusted key set')
        if policy.get('moduleComposition'):
            shapes={'coefficients':[4],'rows':[3,1],'upstream':[3,1],'targets':[3,1]}
            if set(fixture['inputs'])!=set(shapes) or any(not isinstance(fixture['inputs'][key],dict) or fixture['inputs'][key].get('shape')!=shape for key,shape in shapes.items()):
                raise ValueError('Composition fixtures require exact coefficient, observation, influence and target geometry.')
        ids.append(fixture['id'])
        for value in fixture['inputs'].values():tensor(value)
        if attention_keys is not None:
            inputs=fixture['inputs'];observations=inputs[learning_attention.roles(operation)[0]]['shape'][0]
            if observations not in (1,2,3):raise ValueError('Attention fixtures support one to three independent observations')
            shapes={**{role:[observations]+shape for role,shape in learning_attention.samples(operation).items()},**{'parameter.'+p['name']:list(p['shape']) for p in learning_attention.parameter_shapes(operation,learning_attention.target(operation))},**{'upstream.'+role:[observations]+shape for role,shape in learning_attention.outputs(operation).items()}}
            if any(inputs[key]['shape']!=shape for key,shape in shapes.items()):raise ValueError('Attention fixture geometry differs from the trusted task')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate fixture identities')

@contextmanager
def fixture_csv(rows, root=None):
    """One bounded fixture; captured requests provide their managed input owner."""
    text=io.StringIO(newline='')
    writer=csv.writer(text);writer.writerow(['x','y']);writer.writerows(rows)
    content=text.getvalue().encode('utf-8')
    if len(content)>MAX_FIXTURE_BYTES:raise ValueError('Learning fixture exceeds its storage allowance')
    with ExitStack() as stack:
        # Direct library callers have no captured operation. Worker requests always
        # supply their input directory, including standalone execute_request calls.
        if root is None:
            root=os.path.realpath(stack.enter_context(tempfile.TemporaryDirectory(prefix='kwip-learning-')))
        root=Path(os.path.abspath(os.fspath(root)))
        descriptor=directory(str(root));identity=os.fstat(descriptor)
        name='.learning-fixture-'+uuid.uuid4().hex+'.csv';published=None
        try:
            with ArtifactStorage(descriptor,maximum_bytes=MAX_FIXTURE_BYTES) as storage:
                published=storage.atomic_bytes(name,content,MAX_FIXTURE_BYTES)
                visible=directory(str(root))
                try:
                    current=os.fstat(visible)
                    if (current.st_dev,current.st_ino)!=(identity.st_dev,identity.st_ino):raise ValueError('Learning fixture owner changed')
                finally:os.close(visible)
                yield root/name
        finally:
            try:
                if published is not None:
                    try:
                        current=os.stat(name,dir_fd=descriptor,follow_symlinks=False)
                        if (current.st_dev,current.st_ino)==(published.st_dev,published.st_ino):os.unlink(name,dir_fd=descriptor)
                    except FileNotFoundError:pass
            finally:os.close(descriptor)


def generate_evidence(request,submission,*,submission_digest,cancel=lambda:False,registry=None,fixture_root=None):
    validate_request(request)
    policy=assessment_policy(request)
    backend=policy['backend'];capabilities=policy['capabilities']
    optional_bias=backend=='pytorch' and policy['submissionSchema']==2
    if backend=='numpy':
        capabilities={o['id']:o['version'] for o in dataflow.REGISTRY['operators'] if o['id'] in capabilities}
    runtime={'backend':backend,'version':runtime_versions(registry)[backend],'capabilities':capabilities}
    started=time.monotonic()
    evidence={'assessmentID':request['assessmentID'],'assessmentVersion':dict(request['assessmentVersion']),'fixtureDigest':request['fixtureDigest'],'runtime':runtime,'submissionDigest':submission_digest,'fixtures':[],'elapsedMilliseconds':0.0,'maximumTensorElements':0,'cancelled':False}
    def stopped():
        evidence['elapsedMilliseconds']=(time.monotonic()-started)*1000
        if cancel():evidence['cancelled']=True;return True
        return evidence['elapsedMilliseconds']>request['budget']['milliseconds'] or evidence['maximumTensorElements']>request['budget']['maximumTensorElements']
    try:
        if request['runtime']!=runtime or stopped():return evidence
        if not isinstance(submission,dict) or type(submission.get('schemaVersion')) is not int or submission['schemaVersion']!=policy['submissionSchema']:raise ValueError('Unsupported graph submission')
        if policy.get('dataLoading') or policy.get('programExport'):
            task=policy.get('dataLoading') or policy['programExport']
            if submission.get('taskID')!=task:raise ValueError('This construction belongs to a different lesson task.')
            executor=learning_data_loading if policy.get('dataLoading') else learning_program_export
            executor.execute(submission,request['fixtures'],evidence,stopped,tensor)
            return evidence
        if policy.get('moduleComposition'):
            module_composition_learning.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['moduleComposition'],registry=registry)
            return evidence
        if policy.get('gradientLearning'):
            learning_gradients.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['gradientLearning'],registry=registry)
            return evidence
        if policy.get('initialization'):
            learning_initialization.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['initialization'],registry=registry)
            return evidence
        if policy.get('checkpointLifecycle'):
            learning_checkpoints.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['checkpointLifecycle'],registry=registry)
            return evidence
        if policy.get('lossObjective'):
            learning_losses.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['lossObjective'])
            return evidence
        if policy.get('attention'):
            learning_attention.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['attention'])
            return evidence
        if policy.get('recurrent'):
            learning_recurrent.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['recurrent'])
            return evidence
        if policy.get('scheduler'):
            learning_schedulers.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['scheduler'])
            return evidence
        elif policy.get('optimizer'):
            learning_optimizers.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['optimizer'])
            return evidence
        if policy.get('statefulNormalization'):
            learning_stateful_normalization.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['statefulNormalization'])
            return evidence
        if policy.get('spatialPatch'):
            learning_spatial_patches.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['spatialPatch'])
            return evidence
        if policy.get('tensorRouting'):
            learning_tensor_routing.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['tensorRouting'])
            return evidence
        if policy.get('remainingActivation'):
            learning_remaining_activation.execute(submission,request['fixtures'],evidence,stopped,tensor,task=policy['remainingActivation'])
            return evidence
        if policy.get('tensorComposition'):
            learning_tensor_composition.execute(submission,request['fixtures'],evidence,stopped,tensor,operation=policy['tensorComposition'])
            return evidence
        if policy.get('adaptivePooling'):
            learning_adaptive_pooling.execute(submission,request['fixtures'],evidence,stopped,tensor,operation=policy['adaptivePooling'])
            return evidence
        if policy.get('configurableActivation'):
            learning_configurable_activation.execute(submission,request['fixtures'],evidence,stopped,tensor,component=policy['configurableActivation'])
            return evidence
        if policy.get('pooling'):
            learning_pooling.execute(submission,request['fixtures'],evidence,stopped,tensor,dimensions=policy['pooling'],average=policy['average'])
            return evidence
        if policy.get('transposedConvolution'):
            learning_convolution.execute_transposed(submission,request['fixtures'],evidence,stopped,tensor,dimensions=policy['transposedConvolution'])
            return evidence
        if policy.get('convolution'):
            learning_convolution.execute(submission,request['fixtures'],evidence,stopped,tensor,dimensions=policy['convolution'])
            return evidence
        if policy.get('lazyLinear'):
            learning_lazy_linear.execute(submission,request['fixtures'],evidence,stopped,tensor)
            return evidence
        if policy.get('bilinear'):
            learning_bilinear.execute(submission,request['fixtures'],evidence,stopped,tensor,bias_enabled=policy['biasEnabled'])
            return evidence
        if policy.get('matrix'):
            learning_matrix.execute(submission,request['fixtures'],evidence,stopped,tensor)
            return evidence
        if policy.get('embeddingBag'):
            learning_embedding_bag.execute(submission,request['fixtures'],evidence,stopped,tensor)
            return evidence
        if policy.get('embedding'):
            learning_embedding.execute(submission,request['fixtures'],evidence,stopped,tensor)
            return evidence
        if policy.get('layerNorm'):
            learning_layer_norm.execute(submission,request['fixtures'],evidence,stopped,tensor)
            return evidence
        if policy.get('tensorTransforms'):
            learning_tensor_transforms.execute(submission,request['fixtures'],evidence,stopped,tensor,allowed_operations=policy.get('allowedTensorOperations',frozenset({'reshape','transpose','flatten'})))
            return evidence
        expected_kind='affineGraph' if backend=='pytorch' else 'normalizationGraph'
        keys={'schemaVersion','kind','graph','parameters' if backend=='pytorch' else 'outputs'}
        if set(submission)!=keys or submission['kind']!=expected_kind:raise ValueError('Submit a graph and its declared inputs, never authored assessment receipts')
        graph=submission['graph']
        if backend=='pytorch':
            _,widths=validate(graph,registry)
            if any(n['type'] not in policy['allowedTypes'] for n in graph['nodes']):
                if policy.get('requiredExecutedType'):
                    label=ACTIVATION_LABELS[policy['requiredExecutedType']]
                    raise ValueError(f'This {label} lesson permits only Input, Linear, {label}, Add, Concatenate and Output.')
                if 'requiredGeluMode' in policy:raise ValueError('This GELU lesson permits only Input, Linear, GELU, Add, Concatenate and Output.')
                raise ValueError('This recovery assessment requires affine-preserving components: Input, Linear, Add, Concatenate and Output.')
            if policy.get('requiredExecutedType'):
                # A required stage must contribute to the measured Output, not
                # merely exist as a palette/node declaration in the submission.
                output=next(n['id'] for n in graph['nodes'] if n['type']=='output')
                predecessors={node['id']:[] for node in graph['nodes']}
                for edge in graph['edges']:predecessors[edge['target']].append(edge['source'])
                ancestors=set();pending=[output]
                while pending:
                    ident=pending.pop()
                    if ident in ancestors:continue
                    ancestors.add(ident);pending.extend(predecessors[ident])
                if not any(n['type']==policy['requiredExecutedType'] and n['id'] in ancestors for n in graph['nodes']):
                    label=ACTIVATION_LABELS[policy['requiredExecutedType']]
                    raise ValueError(f'This {label} lesson requires a {label} component on the executed path to Output.')
            if 'requiredGeluMode' in policy:
                activations=[n for n in graph['nodes'] if n['type']=='gelu']
                if not activations:raise ValueError('This GELU lesson requires at least one GELU component.')
                if any(n['config']['approximate_tanh'] is not policy['requiredGeluMode'] for n in activations):raise ValueError('Every GELU approximation mode must match the selected lesson: '+('tanh approximation.' if policy['requiredGeluMode'] else 'exact (tanh approximation disabled).'))
            parameters=submission['parameters']
            linear={n['id']:n for n in graph['nodes'] if n['type']=='linear'}
            if not optional_bias and any(n['config'].get('bias',True) is False for n in linear.values()):raise ValueError('This affine assessment requires enabled bias. Enable bias on each Linear component before submitting.')
            if not isinstance(parameters,dict) or set(parameters)!=set(linear):raise ValueError('Provide explicit parameter data for every Linear component')
            parameter_count=sum((n['config']['in_features']+int(n['config'].get('bias',True)))*n['config']['out_features'] for n in linear.values())
            evidence['maximumTensorElements']=parameter_count
            if stopped():return evidence
            torch.set_num_threads(1)
            model=GraphModel(graph,registry).cpu().eval()
            with torch.no_grad():
                for ident,n in linear.items():
                    values=parameters[ident]
                    active=('weight','bias') if n['config'].get('bias',True) else ('weight',)
                    if not isinstance(values,dict) or set(values)!=set(active):raise ValueError('Linear parameters must contain exactly the active tensors: '+', '.join(active))
                    layer=model.layers[model.layer_keys[ident]]
                    for name in active:
                        shape,flat=tensor(values[name]);target=getattr(layer,name)
                        if shape!=list(target.shape):raise ValueError('Parameter dimensions differ from the compiled graph')
                        measured=torch.tensor(flat,dtype=target.dtype).reshape(shape)
                        if not torch.isfinite(measured).all():raise ValueError('Parameter overflows backend dtype')
                        target.copy_(measured)
                adapter=ProductExecutionAdapter(model)
                for fixture in request['fixtures']:
                    if stopped():break
                    if set(fixture['inputs'])!={'x'}:raise ValueError('Affine fixture requires its input tensor')
                    shape,values=tensor(fixture['inputs']['x'])
                    if shape!=[2,2]:raise ValueError('Affine fixture requires two rows and two features')
                    projected=parameter_count+sum(widths.values())*shape[0]
                    evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],projected)
                    if stopped():break
                    output,captured=adapter.forward(torch.tensor(values,dtype=torch.float32).reshape(shape),inspect=True)
                    if not torch.isfinite(output).all():raise ValueError('Model produced nonfinite learning outputs')
                    evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':{'output':{'shape':list(output.shape),'values':output.flatten().tolist()}},'partitions':{}})
        else:
            if policy['allowedTypes'] is not None and isinstance(graph,dict) and isinstance(graph.get('nodes'),list) and any(not isinstance(n,dict) or n.get('operator') not in policy['allowedTypes'] for n in graph['nodes']):raise ValueError('This normalization lesson permits only Source CSV, Cast, Split, Standard Scaler Fit and Apply Fitted Transform.')
            dataflow.validate(graph)
            sources=[n['id'] for n in graph['nodes'] if n['operator']=='kwip.data.sourceCSV']
            if len(sources)!=1:raise ValueError('Standardization assessment uses one adapter-owned fixture source')
            outputs=submission['outputs']
            if not isinstance(outputs,dict) or set(outputs)!={'state','validation','training'} or any(not isinstance(p,dict) or set(p)!={'node','port'} or any(not isinstance(s,str) for s in p.values()) for p in outputs.values()):raise ValueError('Select explicit measured state/training/validation outputs')
            for fixture in request['fixtures']:
                if stopped():break
                if set(fixture['inputs'])!={'training','validation'}:raise ValueError('Standardization fixture requires explicit partitions')
                training_shape,training=tensor(fixture['inputs']['training']);validation_shape,validation=tensor(fixture['inputs']['validation'])
                if training_shape!=[3,2] or validation_shape!=[1,2]:raise ValueError('Standardization assessment requires three training and one validation row')
                order=np.random.default_rng(42).permutation(4).tolist();rows=[None]*4;rows[order[0]]=validation
                for offset,index in enumerate(order[1:]):rows[index]=training[offset*2:offset*2+2]
                # A seed-42, 25% split of this adapter-owned CSV reproduces the declared fixture partitions.
                retained=0
                def observed(_node,ports):
                    nonlocal retained
                    retained+=sum(len(v.rows)*len(v.columns) for v in ports.values() if isinstance(v,dataflow.Table))
                    evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],retained)
                    if stopped():raise InterruptedError('Assessment interrupted at a dataflow node boundary')
                with fixture_csv(rows,fixture_root) as path:
                    result=dataflow.execute(graph,{sources[0]:str(path)},on_node=observed)
                selected={key:result[port['node']][port['port']] for key,port in outputs.items()}
                state=selected['state'];train=selected['training'];heldout=selected['validation']
                if not isinstance(state,dict) or state.get('operator')!='kwip.data.standardScalerFit' or not isinstance(train,dataflow.Table) or not isinstance(heldout,dataflow.Table) or train.role!='training' or heldout.role!='validation':raise ValueError('Select a measured scaler state and explicit training/validation tables')
                dataflow.load_state(json.dumps(state))
                values=[row[heldout.index(c)] for row in heldout.rows for c in ['x','y']]
                evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':{'mean':{'shape':[len(state['means'])],'values':state['means']},'scale':{'shape':[len(state['scales'])],'values':state['scales']},'validation':{'shape':[len(heldout.rows),2],'values':values}},'partitions':{'training':list(train.row_ids),'validation':list(heldout.row_ids),'fit':state['fitEvidence']['observationIDs']}})
                if policy.get('measureTraining',False):
                    # Split shuffles output rows again. Align measured values to fixture input
                    # identity, never by values, while retaining observed partition IDs below.
                    source_ids=result[sources[0]]['table'].row_ids
                    expected_ids=[source_ids[index] for index in order[1:]]
                    if len(train.row_ids)!=len(expected_ids) or set(train.row_ids)!=set(expected_ids):raise ValueError('Selected training rows differ from the fixture training partition')
                    by_id=dict(zip(train.row_ids,train.rows))
                    evidence['fixtures'][-1]['tensors']['training']={'shape':[len(train.rows),2],'values':[by_id[rid][train.index(c)] for rid in expected_ids for c in ['x','y']]}
    except InterruptedError:
        pass
    except Exception as error:
        evidence['backendError']=str(error)[:500]
        evidence['fixtures']=[]
    finally:
        evidence['elapsedMilliseconds']=(time.monotonic()-started)*1000
    return evidence

def execute_request(request_path,submission_path,expected_sha256,registry=None,*,check_cancelled=lambda:None):
    check_cancelled()
    request=strict_json(bounded_read(request_path));raw=bounded_read(submission_path)
    if not isinstance(expected_sha256,str) or sha(raw)!=expected_sha256:raise ValueError('Learning submission digest changed before execution')
    def checked():
        check_cancelled()
        return False
    check_cancelled()
    result=generate_evidence(request,strict_json(raw),submission_digest=sha(raw),registry=registry,cancel=checked,
                             fixture_root=Path(os.path.abspath(os.fspath(request_path))).parent)
    # generate_evidence converts backend exceptions into learning diagnostics.
    # Recheck retained authority before allowing even a diagnostic receipt out.
    check_cancelled()
    return result
