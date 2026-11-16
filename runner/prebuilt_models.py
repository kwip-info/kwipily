"""Bounded scikit-learn adapters with inert, independently executable inference state.

No pickle/joblib, arbitrary classes, optimizer continuation or GPU execution.
The shared capability catalog is the application-owned Core resource, copied to this worker.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import warnings
import numpy as np

FORMAT='kwipml.prebuilt-model.v1'
MAX_STATE=32_000_000
MAX_ROWS=100_000
MAX_CELLS=2_000_000
MAX_NODES=100_000
VERSION='1.7.2'

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def require(ok,message):
    if not ok:raise ValueError(message)
def catalog():return json.loads(Path(__file__).with_name('prebuilt-models.json').read_text())
def defaults(algorithm='logistic'):
    entry=catalog()[algorithm]
    return dict(task=entry['task'],algorithm=algorithm,options={f['key']:f['default'] for f in entry['fields']})
def configuration(c):
    require(type(c) is dict and set(c)=={'task','algorithm','options'},'Choose an explicit task, algorithm and options.')
    entry=catalog().get(c['algorithm']);require(entry is not None and c['task']==entry['task'],'Algorithm does not support the selected task.')
    options=c['options'];require(type(options) is dict and set(options)=={f['key'] for f in entry['fields']},'Algorithm options differ from its reviewed capabilities.')
    for f in entry['fields']:
        v=options[f['key']]
        if f['kind']=='bool':require(type(v) is bool,'Use a Boolean intercept option.')
        else:require(type(v) in ((int,) if f['kind']=='integer' else (int,float)) and math.isfinite(v) and f['minimum']<=v<=f['maximum'],'Option '+f['key']+' exceeds its bound.')
    return c

def features(x,width=None):
    require(type(x) is np.ndarray and x.dtype==np.float64 and x.ndim==2 and 1<=len(x)<=MAX_ROWS and 1<=x.shape[1]<=128 and x.size<=MAX_CELLS,'Use a bounded dense float64 matrix: 100000 rows, 128 features, 2000000 cells.')
    require((width is None or x.shape[1]==width) and np.isfinite(x).all() and np.max(np.abs(x))<=1e10,'Features are nonfinite, oversized or differ from saved feature order.')
    return x

def targets(y,n,task):
    require(y is not None and y.shape==(n,),'Select one target per observation.')
    if task=='classification':
        require(y.dtype.kind in 'iuUS' and all(type(v) in (str,int) for v in y.tolist()),'Classification labels must be uniformly integer or string categories.')
        classes=np.unique(y);require(2<=len(classes)<=128 and all(1<=len(str(v))<=200 and (not isinstance(v,str) or bool(v.strip())) for v in classes),'Fit requires 2–128 bounded categories.')
    else:require(y.dtype.kind in 'fiu' and np.isfinite(y).all() and np.max(np.abs(y))<=1e10,'Regression targets must be finite bounded numbers.')
    return y

def fit(c,x,y=None,check=lambda:None):
    configuration(c);features(x);check()
    import sklearn
    require(sklearn.__version__==VERSION,'Prebuilt models require the reviewed scikit-learn '+VERSION+'.')
    task=c['task'];a=c['algorithm'];o=c['options']
    if task!='clustering':targets(y,len(x),task)
    from sklearn.linear_model import LogisticRegression,Ridge,ElasticNet
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
    from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
    from sklearn.mixture import GaussianMixture
    common=dict(max_iter=o.get('maxIterations',200),tol=o.get('tolerance',.0001))
    if a=='logistic':est=LogisticRegression(C=o['C'],fit_intercept=o['fitIntercept'],**common)
    elif a=='ridge':est=Ridge(alpha=o['alpha'],fit_intercept=o['fitIntercept'],solver='svd')
    elif a=='elasticNet':est=ElasticNet(alpha=o['alpha'],l1_ratio=o['l1Ratio'],fit_intercept=o['fitIntercept'],selection='cyclic',**common)
    elif a in ('randomForestClassifier','randomForestRegressor'):
        node_bound=o['trees']*min(2**(o['maxDepth']+1)-1,2*len(x)-1)
        require(node_bound*(len(np.unique(y)) if task=='classification' else 1)<=MAX_CELLS,'Forest class-value arrays may exceed 2000000 cells; reduce trees/depth/classes.')
        require(node_bound<=MAX_NODES,'Requested forest may exceed the 100000-node state limit; reduce trees/depth or rows.')
        cls=RandomForestClassifier if task=='classification' else RandomForestRegressor
        est=cls(n_estimators=o['trees'],max_depth=o['maxDepth'],min_samples_leaf=o['minSamplesLeaf'],random_state=o['seed'],n_jobs=1)
    elif a in ('kmeans','miniBatchKmeans'):
        require(o['clusters']<=len(x),'More clusters than fit observations.')
        args=dict(n_clusters=o['clusters'],n_init=o['initializations'],random_state=o['seed'],init='k-means++',**common)
        est=KMeans(**args) if a=='kmeans' else MiniBatchKMeans(batch_size=o['batchSize'],**args)
    elif a=='gaussianMixture':
        require(o['clusters']<=len(x),'More mixture components than fit observations.')
        est=GaussianMixture(n_components=o['clusters'],covariance_type='diag',n_init=o['initializations'],random_state=o['seed'],reg_covar=o['regCovar'],**common)
    else:
        # sklearn DBSCAN can materialize quadratic neighborhoods. Admit only a
        # bounded population; 2000² indices are bounded before sklearn allocation.
        require(len(x)<=2000,'DBSCAN fit is limited to 2000 rows to bound worst-case neighborhood memory. Choose K-means for larger populations.')
        est=DBSCAN(eps=o['eps'],min_samples=o['minSamples'],n_jobs=1,algorithm='brute')
    from threadpoolctl import threadpool_limits
    try:
        if __package__:from .worker_resources import cpu_threads
        else:from worker_resources import cpu_threads
        threads=cpu_threads()
    except ModuleNotFoundError:threads=1
    from sklearn.exceptions import ConvergenceWarning
    with threadpool_limits(limits=threads),warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always',ConvergenceWarning);est.fit(x,y)
    check()
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in observed) and bool(getattr(est,'converged_',True))
    require(converged,'Estimator did not converge; increase iterations, review scaling or change configuration. No model was published.')
    if a=='dbscan':
        core=set(est.core_sample_indices_.tolist())
        return None,dict(values=est.labels_.tolist(),status=['Noise' if v==-1 else 'Core' if i in core else 'Border' for i,v in enumerate(est.labels_)])
    params={}
    if a=='logistic':params=dict(coefficients=est.coef_.tolist(),intercept=est.intercept_.tolist())
    elif task=='regression' and not a.startswith('randomForest'):params=dict(coefficients=est.coef_.tolist(),intercept=float(est.intercept_))
    elif a.startswith('randomForest'):
        params={'trees':[]}
        for tree in est.estimators_:
            t=tree.tree_;params['trees'].append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,:].tolist()))
    elif a in ('kmeans','miniBatchKmeans'):params=dict(centers=est.cluster_centers_.tolist())
    elif a=='gaussianMixture':params=dict(means=est.means_.tolist(),variances=est.covariances_.tolist(),weights=est.weights_.tolist())
    model=dict(schemaVersion=1,stateFormat=FORMAT,runtimeVersion=VERSION,configuration=c,featureCount=x.shape[1],featureColumns=[],classes=est.classes_.tolist() if task=='classification' else [],parameters=params,inputContract='prepared numeric features; no implicit preprocessing')
    validate(model);prediction=predict(model,x)
    if a in ('kmeans','miniBatchKmeans'):prediction['fitInertia']=float(est.inertia_)
    return model,prediction

def array(value,shape,positive=False):
    # JSON size is bounded by load; dimensions are checked before NumPy allocation.
    require(type(value) is list and len(value)==shape[0],'Invalid state array shape.')
    if len(shape)>1:
        for row in value:array(row,shape[1:],positive)
    else:require(all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1e100 and (not positive or v>0) for v in value),'Invalid state numeric value.')
    return np.asarray(value,dtype=np.float64)

def validate(m):
    keys={'schemaVersion','stateFormat','runtimeVersion','configuration','featureCount','featureColumns','classes','parameters','inputContract'}
    require(type(m) is dict and set(m)==keys and m['schemaVersion']==1 and type(m['schemaVersion']) is int and m['stateFormat']==FORMAT and m['runtimeVersion']==VERSION,'Unsupported inert model state.')
    c=configuration(m['configuration']);a=c['algorithm'];task=c['task'];w=m['featureCount'];p=m['parameters']
    require(type(w) is int and 1<=w<=128 and type(p) is dict and a!='dbscan','Invalid model width or fit-only model.')
    names=m['featureColumns'];require(type(names) is list and (not names or len(names)==w) and len(set(names))==len(names) and all(type(v) is str and 1<=len(v)<=200 for v in names),'Invalid saved feature order.')
    require(m['inputContract']=='prepared numeric features; no implicit preprocessing','Unsupported input boundary.')
    classes=m['classes'];require(type(classes) is list,'Invalid classes.')
    if task=='classification':require(2<=len(classes)<=128 and len(set(classes))==len(classes) and all(type(v) in (str,int) for v in classes) and len({type(v) for v in classes})==1 and all(1<=len(str(v))<=200 and (not isinstance(v,str) or bool(v.strip())) for v in classes),'Invalid class order.')
    else:require(classes==[],'Unexpected classification labels.')
    if a=='logistic':
        require(set(p)=={'coefficients','intercept'},'Invalid linear state.')
        h=1 if len(classes)==2 else len(classes);array(p['coefficients'],(h,w));array(p['intercept'],(h,))
    elif a in ('ridge','elasticNet'):
        require(set(p)=={'coefficients','intercept'},'Invalid regression state.');array(p['coefficients'],(w,));array([p['intercept']],(1,))
    elif a.startswith('randomForest'):
        require(set(p)=={'trees'} and type(p['trees']) is list and len(p['trees'])==c['options']['trees'],'Invalid forest tree count.')
        total=0
        for tree in p['trees']:
            require(type(tree) is dict and set(tree)=={'left','right','feature','threshold','value'},'Invalid tree fields.')
            n=len(tree['left']);total+=n;require(1<=n and total<=MAX_NODES,'Forest node budget exceeded.')
            require(all(type(tree[k]) is list and len(tree[k])==n for k in ('right','feature','threshold','value')),'Invalid tree dimensions.')
            array(tree['threshold'],(n,));values=array(tree['value'],(n,len(classes) if task=='classification' else 1))
            if task=='classification':require((values>=0).all() and np.allclose(values.sum(axis=1),1),'Invalid tree probabilities.')
            visited=set();stack=[(0,0)]
            while stack:
                i,depth=stack.pop();require(i not in visited and 0<=i<n and depth<=c['options']['maxDepth'],'Tree cycle, shared child or excessive depth.');visited.add(i)
                l,r,f=tree['left'][i],tree['right'][i],tree['feature'][i]
                require(all(type(v) is int for v in (l,r,f)),'Tree indices must be integers.')
                if l==-1:require(r==-1 and f==-2,'Invalid leaf.')
                else:require(0<=l<n and 0<=r<n and l!=r and 0<=f<w,'Invalid child or feature.');stack.extend([(l,depth+1),(r,depth+1)])
            require(len(visited)==n,'Unreachable tree state.')
    elif a in ('kmeans','miniBatchKmeans'):require(set(p)=={'centers'},'Invalid centroid state.');array(p['centers'],(c['options']['clusters'],w))
    else:
        k=c['options']['clusters'];require(set(p)=={'means','variances','weights'},'Invalid mixture state.');array(p['means'],(k,w));array(p['variances'],(k,w),True);weights=array(p['weights'],(k,),True);require(abs(weights.sum()-1)<1e-9,'Invalid mixture weights.')
    require(len(canonical(m))<=MAX_STATE,'Model state exceeds 32MB.');return m

def load(raw):
    require(type(raw) is bytes and len(raw)<=MAX_STATE,'Model exceeds 32MB.')
    def unique(pairs):
        d={}
        for k,v in pairs:require(k not in d,'Duplicate model field.');d[k]=v
        return d
    return validate(json.loads(raw,object_pairs_hook=unique))

def predict(m,x):
    validate(m);features(x,m['featureCount']);c=m['configuration'];a=c['algorithm'];p=m['parameters'];task=c['task']
    if task=='classification' or task=='clustering':
        require(len(x)*(len(m['classes']) if task=='classification' else c['options']['clusters'])<=MAX_CELLS,'Complete score output exceeds 2000000 cells; reduce workload or class/component count.')
    if a=='logistic':
        z=x@np.asarray(p['coefficients']).T+np.asarray(p['intercept'])
        if len(m['classes'])==2:
            z=z[:,0];prob=np.exp(-np.logaddexp(0,-z));prob=np.column_stack((1-prob,prob))
        else:
            z-=z.max(axis=1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True)
        return dict(values=[m['classes'][i] for i in prob.argmax(axis=1)],probabilities=prob.tolist())
    if a in ('ridge','elasticNet'):return dict(values=(x@np.asarray(p['coefficients'])+p['intercept']).tolist())
    if a.startswith('randomForest'):
        v=np.zeros((len(x),len(m['classes']) if task=='classification' else 1))
        # sklearn trees compare float32 input, even though the model boundary is float64.
        rounded=x.astype(np.float32)
        for tree in p['trees']:
            at=np.zeros(len(x),dtype=np.int64)
            for _ in range(c['options']['maxDepth']+1):
                left=np.asarray(tree['left']);active=left[at]!=-1
                if not active.any():break
                rows=np.where(active)[0];node=at[rows];f=np.asarray(tree['feature'])[node]
                at[rows]=np.where(rounded[rows,f]<=np.asarray(tree['threshold'])[node],left[node],np.asarray(tree['right'])[node])
            v+=np.asarray(tree['value'])[at]
        v/=len(p['trees'])
        return dict(values=[m['classes'][i] for i in v.argmax(axis=1)],probabilities=v.tolist()) if task=='classification' else dict(values=v[:,0].tolist())
    # Bound row×cluster×feature scratch by processing chunks.
    values=[];scores=[]
    for start in range(0,len(x),256):
        block=x[start:start+256]
        if a in ('kmeans','miniBatchKmeans'):
            distances=np.sqrt(np.maximum(0,((block[:,None,:]-np.asarray(p['centers'])[None,:,:])**2).sum(axis=2)))
            values.extend(distances.argmin(axis=1).tolist());scores.extend(distances.tolist())
        else:
            means=np.asarray(p['means']);var=np.asarray(p['variances'])
            z=np.log(np.asarray(p['weights']))-.5*(np.log(2*np.pi*var).sum(axis=1)[None,:]+((block[:,None,:]-means[None,:,:])**2/var).sum(axis=2))
            z-=z.max(axis=1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True)
            values.extend(prob.argmax(axis=1).tolist());scores.extend(prob.tolist())
    return dict(values=values,**({'distances':scores} if a in ('kmeans','miniBatchKmeans') else {'memberships':scores}))

def evaluate(m,x,y):
    prediction=predict(m,x);task=m['configuration']['task'];require(task!='clustering','Clustering has no labeled accuracy.');require(y.shape==(len(x),),'Target count differs.')
    if task=='classification':
        from sklearn.metrics import accuracy_score,log_loss,precision_recall_fscore_support,confusion_matrix
        require(all(v in m['classes'] for v in y.tolist()),'Evaluation contains categories absent from Training.')
        precision,recall,f1,_=precision_recall_fscore_support(y,prediction['values'],labels=m['classes'],average='macro',zero_division=0)
        return dict(count=len(x),accuracy=float(accuracy_score(y,prediction['values'])),logLoss=float(log_loss(y,prediction['probabilities'],labels=m['classes'])),macroPrecision=float(precision),macroRecall=float(recall),macroF1=float(f1),classOrder=m['classes'],confusionMatrix=confusion_matrix(y,prediction['values'],labels=m['classes']).tolist())
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    return dict(count=len(x),mae=float(mean_absolute_error(y,prediction['values'])),rmse=float(np.sqrt(mean_squared_error(y,prediction['values']))),rSquared=float(r2_score(y,prediction['values'],force_finite=False)) if len(x)>1 and np.var(y)>0 else None)

def export(m,destination):
    validate(m);d=Path(destination);d.mkdir(mode=0o700,exist_ok=False)
    (d/'state.json').write_bytes(canonical(m))
    shutil.copyfile(__file__,d/'prebuilt_models.py');shutil.copyfile(Path(__file__).with_name('prebuilt-models.json'),d/'prebuilt-models.json')
    (d/'inference.py').write_text("import json,sys\nfrom pathlib import Path\nimport numpy as np\nfrom prebuilt_models import load,predict\ndef bounded(path,limit,label):\n    with path.open('rb') as stream: raw=stream.read(limit+1)\n    if len(raw)>limit: raise ValueError(label+' exceeds its byte budget')\n    return raw\nif __name__=='__main__':\n    model=load(bounded(Path(__file__).with_name('state.json'),32000000,'Model'))\n    values=json.loads(bounded(Path(sys.argv[1]),64000000,'Input'))\n    if not isinstance(values,list) or not 1<=len(values)<=100000 or any(not isinstance(row,list) or len(row)!=model['featureCount'] for row in values) or len(values)*model['featureCount']>2000000: raise ValueError('Input exceeds its row, feature or cell budget')\n    print(json.dumps(predict(model,np.asarray(values,dtype=np.float64)),allow_nan=False))\n")
    (d/'README.txt').write_text('Prebuilt model inference: Python + NumPy. Run python inference.py INPUT.json. Inputs are prepared numeric features in saved featureColumns order. No implicit preprocessing, training data, arbitrary code loading or optimizer resume. Distances/memberships are not churn probabilities.\n')
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(d.iterdir())]
    (d/'manifest.json').write_bytes(canonical(dict(schemaVersion=1,stateFormat=FORMAT,artifacts=files)))
