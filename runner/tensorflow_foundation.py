"""Bounded TensorFlow CPU graph adapter; no imports of the PyTorch worker.

Arrays supplied by a caller must come from its verified source/partition boundary.
This module validates numerical boundaries; it does not grant source permissions.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np

SUPPORTED=('input','linear','relu','tanh','add','concat','output')

def _tf():
    import tensorflow as tf
    return tf

def capabilities():
    tf=_tf()
    return {'schemaVersion':1,'backendID':'tensorflow','runtimeVersion':tf.__version__,
            'devices':['cpu'],'supportedOperations':['construct','fit','predict','evaluate','weightsSave','weightsLoad','export'],
            'unsupportedOperations':['gpu','resumeOptimizer','customCode','arbitraryLayers'],
            'components':list(SUPPORTED),'stateFormat':'keras-weights-h5','task':'binaryClassification'}

def _integer(value,low,high,name):
    if type(value) is not int or not low<=value<=high:raise ValueError(f'{name} must be integer {low}..{high}.')
    return value

def validate_graph(graph):
    if not isinstance(graph,dict) or graph.get('schemaVersion')!=2:raise ValueError('Expected graph schema 2.')
    nodes=graph.get('nodes'); edges=graph.get('edges')
    if not isinstance(nodes,list) or not 3<=len(nodes)<=128 or not isinstance(edges,list) or len(edges)>256:raise ValueError('Invalid graph bounds.')
    by_id={}; incoming={}; outgoing={}
    for n in nodes:
        i=n.get('id'); kind=n.get('type'); c=n.get('config')
        if not isinstance(i,str) or not i or len(i)>128 or i in by_id:raise ValueError('Invalid node identity.')
        if kind not in SUPPORTED or not isinstance(c,dict):raise ValueError('Unsupported TensorFlow component.')
        expected={'features'} if kind=='input' else {'in_features','out_features','bias'} if kind=='linear' else set()
        if set(c)!=expected:raise ValueError('Unknown or missing configuration fields.')
        for key in expected-{'bias'}:_integer(c[key],1,512,key)
        if kind=='linear' and type(c['bias']) is not bool:raise ValueError('Bias must be boolean.')
        by_id[i]=n;incoming[i]={};outgoing[i]=[]
    if sum(n['type']=='input' for n in nodes)!=1 or sum(n['type']=='output' for n in nodes)!=1:raise ValueError('One input and output required.')
    edge_ids=set()
    for e in edges:
        a=e.get('source');b=e.get('target');p=e.get('targetPort');eid=e.get('id')
        if not isinstance(eid,str) or not eid or eid in edge_ids:raise ValueError('Invalid edge identity.')
        edge_ids.add(eid)
        if a not in by_id or b not in by_id or e.get('sourcePort')!='out' or by_id[a]['type']=='output':raise ValueError('Invalid source connection.')
        required=('a','b') if by_id[b]['type'] in ('concat','add') else () if by_id[b]['type']=='input' else ('in',)
        if p not in required or p in incoming[b]:raise ValueError('Invalid target connection.')
        incoming[b][p]=a;outgoing[a].append(b)
    for i,n in by_id.items():
        required={'a','b'} if n['type'] in ('concat','add') else set() if n['type']=='input' else {'in'}
        if set(incoming[i])!=required:raise ValueError('Disconnected input port.')
        if n['type']!='output' and not outgoing[i]:raise ValueError('Disconnected branch.')
    order=[];widths={};pending=set(by_id);parameters=0
    while pending:
        ready=[i for i in by_id if i in pending and all(a in widths for a in incoming[i].values())]
        if not ready:raise ValueError('Graph contains a cycle.')
        for i in ready:
            n=by_id[i];c=n['config'];kind=n['type'];w=[widths[incoming[i][p]] for p in sorted(incoming[i])]
            if kind=='input':width=c['features']
            elif kind=='linear':
                if w[0]!=c['in_features']:raise ValueError('Linear input width mismatch.')
                width=c['out_features'];parameters+=width*(w[0]+int(c['bias']))
            elif kind=='concat':width=sum(w)
            elif kind=='add':
                if w[0]!=w[1]:raise ValueError('Add width mismatch.')
                width=w[0]
            else:width=w[0]
            if width>4096 or parameters>2_000_000:raise ValueError('Graph exceeds resource limits.')
            widths[i]=width;order.append(i);pending.remove(i)
    input_id=next(i for i in order if by_id[i]['type']=='input');output_id=next(i for i in order if by_id[i]['type']=='output')
    if widths[input_id]!=2 or widths[output_id]!=2 or parameters==0:raise ValueError('Foundation requires two features and two logits with trainable Dense weights.')
    return by_id,incoming,order

def portable_graph(graph):
    validate_graph(graph)
    result={key:graph[key] for key in ('schemaVersion','name','nodes','edges') if key in graph}
    return json.loads(json.dumps(result,allow_nan=False))

def build(graph,seed=42):
    by_id,incoming,order=validate_graph(graph);_integer(seed,0,2**32-1,'seed');tf=_tf()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    values={};inputs=None
    with tf.device('/CPU:0'):
        for index,i in enumerate(order):
            n=by_id[i];kind=n['type'];c=n['config'];name=f'node_{index}'
            xs=[values[incoming[i][p]] for p in sorted(incoming[i])]
            if kind=='input':value=tf.keras.Input(shape=(c['features'],),dtype='float32',name=name);inputs=value
            elif kind=='linear':value=tf.keras.layers.Dense(c['out_features'],use_bias=c['bias'],name=name)(xs[0])
            elif kind in ('relu','tanh'):value=tf.keras.layers.Activation(kind,name=name)(xs[0])
            elif kind=='add':value=tf.keras.layers.Add(name=name)(xs)
            elif kind=='concat':value=tf.keras.layers.Concatenate(axis=-1,name=name)(xs)
            else:value=xs[0]
            values[i]=value
        model=tf.keras.Model(inputs,values[next(i for i in order if by_id[i]['type']=='output')])
    model._kwipml_graph_digest=hashlib.sha256(json.dumps(portable_graph(graph),sort_keys=True,allow_nan=False).encode()).hexdigest()
    return model

def _x(value):
    x=np.asarray(value)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=2 or not 1<=len(x)<=100000 or not np.isfinite(x).all():raise ValueError('Inputs require finite float32 [rows,2], 1..100000 rows.')
    return x

def _y(value,count):
    y=np.asarray(value)
    if y.dtype not in (np.dtype('int32'),np.dtype('int64')) or y.shape!=(count,) or not np.isin(y,[0,1]).all():raise ValueError('Labels require integer [rows] values 0 or 1.')
    return y

def validate_configuration(settings):
    if not isinstance(settings,dict) or set(settings)!={'seed','epochs','batchSize','learningRate','device'}:raise ValueError('Expected seed, epochs, batchSize, learningRate and device.')
    _integer(settings['seed'],0,2**32-1,'seed');_integer(settings['epochs'],1,1000,'epochs');_integer(settings['batchSize'],1,100000,'batchSize')
    rate=settings['learningRate']
    if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<rate<=1:raise ValueError('learningRate must be in (0,1].')
    if settings['device']!='cpu':raise ValueError('TensorFlow foundation supports explicit CPU only; GPU is not silently substituted.')
    return dict(settings)

def predict(model,x):
    x=_x(x);tf=_tf()
    with tf.device('/CPU:0'):out=np.asarray(model(x,training=False))
    if out.shape!=(len(x),2) or not np.isfinite(out).all():raise ValueError('Model produced invalid/nonfinite logits.')
    return out

def evaluate(model,x,y):
    x=_x(x);y=_y(y,len(x));logits=predict(model,x).astype('float64')
    shifted=logits-logits.max(axis=1,keepdims=True)
    loss=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)),y]
    return {'loss':float(loss.mean()),'accuracy':float((logits.argmax(axis=1)==y).mean()),'samples':len(y)}

def fit(graph,train_x,train_y,validation_x,validation_y,settings,*,check_cancelled=lambda:None,event=lambda value:None):
    check_cancelled();settings=validate_configuration(settings);validate_graph(graph)
    x=_x(train_x);y=_y(train_y,len(x));v=_x(validation_x);w=_y(validation_y,len(v))
    if set(map(tuple,x))&set(map(tuple,v)):raise ValueError('Training and validation feature rows overlap.')
    if len(np.unique(y))!=2:raise ValueError('Training must contain both classes.')
    model=build(graph,settings['seed']);tf=_tf();optimizer=tf.keras.optimizers.Adam(settings['learningRate']);rng=np.random.default_rng(settings['seed']);history=[]
    with tf.device('/CPU:0'):
        for epoch in range(settings['epochs']):
            order=rng.permutation(len(x));total=0.0
            for start in range(0,len(x),settings['batchSize']):
                check_cancelled();idx=order[start:start+settings['batchSize']]
                with tf.GradientTape() as tape:
                    logits=model(x[idx],training=True)
                    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[idx],logits=logits))
                gradients=tape.gradient(loss,model.trainable_variables)
                if not np.isfinite(float(loss)) or any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):raise ValueError('Training loss or gradient became nonfinite.')
                optimizer.apply_gradients(zip(gradients,model.trainable_variables));total+=float(loss)*len(idx)
                if any(not np.isfinite(np.asarray(p)).all() for p in model.weights):raise ValueError('Training weights became nonfinite.')
            check_cancelled();metrics=evaluate(model,v,w);record={'epoch':epoch+1,'loss':total/len(x),'validationLoss':metrics['loss'],'validationAccuracy':metrics['accuracy']};history.append(record);event(record)
    check_cancelled();return model,history

def _digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def export(model,graph,directory):
    """New-directory transactional weights/source export; contains no source rows."""
    validate_graph(graph);directory=Path(directory)
    digest=hashlib.sha256(json.dumps(portable_graph(graph),sort_keys=True,allow_nan=False).encode()).hexdigest()
    if getattr(model,'_kwipml_graph_digest',None)!=digest:raise ValueError('Export graph differs from constructed model graph.')
    if any(not np.isfinite(np.asarray(p)).all() for p in model.weights):raise ValueError('Export requires finite weights.')
    if directory.exists() or directory.is_symlink():raise FileExistsError(directory)
    directory.parent.mkdir(parents=True,exist_ok=True)
    staged=Path(tempfile.mkdtemp(prefix='.tensorflow-',dir=directory.parent))
    try:
        model.save_weights(staged/'model.weights.h5')
        (staged/'graph.json').write_text(json.dumps(portable_graph(graph),sort_keys=True,allow_nan=False))
        shutil.copyfile(__file__,staged/'tensorflow_foundation.py')
        (staged/'inference.py').write_text("from pathlib import Path\nimport json, sys, numpy as np\nfrom tensorflow_foundation import load, predict\nroot=Path(__file__).resolve().parent\nprint(json.dumps({'logits':predict(load(root),np.load(sys.argv[1],allow_pickle=False)).tolist()}))\n")
        (staged/'requirements.txt').write_text(f"tensorflow=={_tf().__version__}\nkeras=={_tf().keras.__version__}\nnumpy=={np.__version__}\n")
        inventory={p.name:{'sha256':_digest(p),'bytes':p.stat().st_size} for p in staged.iterdir()}
        manifest={'schemaVersion':1,'backendID':'tensorflow','runtimeVersion':_tf().__version__,'stateFormat':'keras-weights-h5','device':'cpu','task':'binaryClassification','files':inventory}
        (staged/'manifest.json').write_text(json.dumps(manifest,sort_keys=True))
        staged.rename(directory)
    except BaseException:
        shutil.rmtree(staged,ignore_errors=True);raise
    return manifest

save=export

def load(directory):
    directory=Path(directory);manifest_path=directory/'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size>100000:raise ValueError('Invalid manifest.')
    manifest=json.loads(manifest_path.read_text())
    expected={'graph.json','model.weights.h5','tensorflow_foundation.py','inference.py','requirements.txt'}
    if manifest.get('schemaVersion')!=1 or manifest.get('backendID')!='tensorflow' or set(manifest.get('files',{}))!=expected:raise ValueError('Invalid TensorFlow manifest.')
    for name in expected:
        p=directory/name;entry=manifest['files'][name]
        if p.is_symlink() or not p.is_file() or p.stat().st_size>50_000_000 or p.stat().st_size!=entry['bytes'] or _digest(p)!=entry['sha256']:raise ValueError('Artifact integrity failure.')
    graph=json.loads((directory/'graph.json').read_text());model=build(graph)
    model.load_weights(directory/'model.weights.h5')
    if any(not np.isfinite(np.asarray(p)).all() for p in model.weights):raise ValueError('Nonfinite saved weights.')
    return model
