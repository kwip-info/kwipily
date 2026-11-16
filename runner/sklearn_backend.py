"""Reviewed binary logistic-regression lifecycle; no pickle or executable state loading.

Estimator state is portable inference state, not optimizer/checkpoint continuation.
The caller owns verified source partitions and process cancellation during sklearn.fit.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import warnings

import numpy as np

BACKEND_ID = 'sklearn'
STATE_FORMAT = 'kwipml.logistic-regression.v1'
MAX_ROWS = 100_000
MAX_FEATURES = 128
MAX_STATE_BYTES = 100_000
SUPPORTED_RUNTIME_VERSION = '1.7.2'


def default_configuration():
    return {'C': 1.0, 'maxIterations': 200, 'tolerance': 0.0001, 'fitIntercept': True}


def capabilities():
    import sklearn
    return {'schemaVersion': 1, 'backendID': BACKEND_ID, 'runtimeVersion': sklearn.__version__,
            'prebuiltCatalogSHA256': hashlib.sha256(Path(__file__).with_name('prebuilt-models.json').read_bytes()).hexdigest(),
            'devices': ['cpu'], 'supportedOperations': ['fit','predict','evaluate','save','load','export'],
            'unsupportedOperations': ['partialFit','resume','finetune','autograd','gpu']}


def validate_configuration(config):
    if not isinstance(config,dict) or set(config) != set(default_configuration()):
        raise ValueError('Logistic regression requires exactly C, maxIterations, tolerance and fitIntercept.')
    for key,low,high in [('C',1e-8,1e8),('tolerance',1e-12,1.)]:
        value=config[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'{key} must be finite in [{low}, {high}].')
    if type(config['maxIterations']) is not int or not 1<=config['maxIterations']<=10_000:
        raise ValueError('maxIterations must be an integer from 1 to 10000.')
    if type(config['fitIntercept']) is not bool:raise ValueError('fitIntercept must be a boolean.')
    return dict(config)


def _features(x, width=None):
    if not isinstance(x,np.ndarray) or x.dtype != np.float64 or x.ndim != 2:
        raise ValueError('Features must be an explicit dense float64 NumPy matrix; conversion belongs at the data boundary.')
    if not 1<=x.shape[0]<=MAX_ROWS or not 1<=x.shape[1]<=MAX_FEATURES or x.size>2_000_000:
        raise ValueError('Features exceed the 100000 row, 128 column or 2000000 cell bound.')
    if width is not None and x.shape[1]!=width:raise ValueError('Feature width differs from the fitted estimator.')
    if not np.isfinite(x).all() or np.any(np.abs(x)>1e10):raise ValueError('Features must be finite within ±1e10.')
    return x


def _labels(y,count,require_both=False):
    if not isinstance(y,np.ndarray) or y.dtype != np.int64 or y.shape != (count,) or not np.isin(y,[0,1]).all():
        raise ValueError('Targets must be an int64 vector of 0/1 labels matching the row count.')
    if require_both and set(y.tolist()) != {0,1}:raise ValueError('Fitting requires both binary classes.')
    return y


def _state(model):
    keys={'schemaVersion','stateFormat','backendID','runtimeVersion','device','task','configuration',
          'featureCount','classes','coefficients','intercept','iterations','converged'}
    if not isinstance(model,dict) or set(model)!=keys:raise ValueError('Invalid logistic state fields.')
    if type(model['schemaVersion']) is not int or model['schemaVersion']!=1 or model['stateFormat']!=STATE_FORMAT or model['backendID']!=BACKEND_ID or model['device']!='cpu' or model['task']!='binaryClassification':
        raise ValueError('Unsupported logistic state format/backend/task.')
    if not isinstance(model['runtimeVersion'],str) or not 1<=len(model['runtimeVersion'])<=100:
        raise ValueError('Missing runtime identity.')
    validate_configuration(model['configuration'])
    width=model['featureCount']
    if type(width) is not int or not 1<=width<=MAX_FEATURES:raise ValueError('Invalid feature count.')
    if model['classes']!=[0,1] or any(type(v) is not int for v in model['classes']):raise ValueError('Unsupported class order.')
    if not isinstance(model['coefficients'],list) or len(model['coefficients'])!=width:raise ValueError('Invalid coefficient shape.')
    if any(type(v) not in (float,int) or not math.isfinite(v) or abs(v)>1e100 for v in [*model['coefficients'],model['intercept']]):
        raise ValueError('Fitted values must be finite within the inference arithmetic bound.')
    if not model['configuration']['fitIntercept'] and model['intercept']!=0:raise ValueError('Disabled intercept owns no value.')
    if type(model['iterations']) is not int or not 1<=model['iterations']<=model['configuration']['maxIterations'] or type(model['converged']) is not bool:
        raise ValueError('Invalid solver outcome.')
    return model


def fit(configuration, train_x, train_y, *, check_cancelled=lambda:None):
    config=validate_configuration(configuration);_features(train_x);_labels(train_y,len(train_x),True)
    check_cancelled()
    import sklearn
    if sklearn.__version__ != SUPPORTED_RUNTIME_VERSION:
        raise ValueError(f'This reviewed adapter requires scikit-learn {SUPPORTED_RUNTIME_VERSION}; detected {sklearn.__version__}.')
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    # lbfgs is full-batch and has no random initialization; exposing an unused seed
    # or minibatch setting would misrepresent the estimator's actual behavior.
    estimator=LogisticRegression(C=config['C'],tol=config['tolerance'],max_iter=config['maxIterations'],
                                 fit_intercept=config['fitIntercept'],solver='lbfgs',penalty='l2')
    try:
        if __package__:from .worker_resources import cpu_threads
        else:from worker_resources import cpu_threads
        threads=cpu_threads()
    except ModuleNotFoundError:threads=1  # Standalone transferable module default.
    with threadpool_limits(limits=threads),warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always',ConvergenceWarning)
        estimator.fit(train_x,train_y)
    check_cancelled()
    result={'schemaVersion':1,'stateFormat':STATE_FORMAT,'backendID':BACKEND_ID,'runtimeVersion':sklearn.__version__,
            'device':'cpu','task':'binaryClassification','configuration':config,'featureCount':train_x.shape[1],
            'classes':estimator.classes_.tolist(),'coefficients':estimator.coef_[0].tolist(),
            'intercept':float(estimator.intercept_[0]),'iterations':int(estimator.n_iter_[0]),
            'converged':not any(issubclass(item.category,ConvergenceWarning) for item in observed)}
    return _state(result)


def predict(model,x):
    model=_state(model);x=_features(x,model['featureCount'])
    scores=x @ np.asarray(model['coefficients'],dtype=np.float64)+model['intercept']
    if not np.isfinite(scores).all():raise ValueError('Nonfinite logistic decision scores.')
    # Stable sigmoid retains extreme finite inputs without overflow warnings.
    positive=np.empty_like(scores);mask=scores>=0
    positive[mask]=1/(1+np.exp(-scores[mask]));exp=np.exp(scores[~mask]);positive[~mask]=exp/(1+exp)
    return {'classes':(scores>0).astype(np.int64).tolist(),
            'probabilities':np.column_stack([1-positive,positive]).tolist(),'decisionScores':scores.tolist()}


def evaluate(model,x,y):
    result=predict(model,x);_labels(y,len(x))
    scores=np.asarray(result['decisionScores'])
    # softplus(z)-y*z avoids clipping away large confident errors.
    loss=np.maximum(scores,0)-y*scores+np.log1p(np.exp(-np.abs(scores)))
    return {'count':len(x),'accuracy':float(np.mean(np.asarray(result['classes'])==y)),
            'logLoss':float(np.mean(loss))}


def _encode(model):
    return json.dumps(_state(model),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def save(model,path):
    raw=_encode(model)
    # Exclusive creation never overwrites another run or follows a terminal symlink.
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'wb') as stream:stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load(path):
    try:descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError as error:raise ValueError('Estimator state must be a readable regular file.') from error
    with os.fdopen(descriptor,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_STATE_BYTES:raise ValueError('Estimator state exceeds file bounds.')
        raw=stream.read(MAX_STATE_BYTES+1)
    if len(raw)>MAX_STATE_BYTES:raise ValueError('Estimator state exceeds file bounds.')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate state field.')
            result[key]=value
        return result
    return _state(json.loads(raw,object_pairs_hook=unique))


def export(model,directory,*,check_cancelled=lambda:None):
    _state(model);destination=Path(directory)
    if destination.exists() or destination.is_symlink():raise FileExistsError('Export destination already exists.')
    check_cancelled();stage=Path(tempfile.mkdtemp(prefix='.sklearn-export-',dir=destination.parent));owned=None
    try:
        save(model,stage/'state.json')
        # This reviewed module imports sklearn only for discovery/training. Portable
        # inference imports only NumPy and the standard library; no dynamic user code.
        shutil.copyfile(__file__,stage/'sklearn_backend.py')
        (stage/'inference.py').write_text("import json, sys\nfrom pathlib import Path\nimport numpy as np\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom sklearn_backend import load, predict\nif __name__ == '__main__':\n    x=np.asarray(json.loads(Path(sys.argv[1]).read_text()),dtype=np.float64)\n    print(json.dumps(predict(load(Path(__file__).with_name('state.json')),x),allow_nan=False))\n")
        (stage/'README.txt').write_text('Binary logistic-regression inference. Install NumPy, then run python inference.py INPUT.json. Input is a row-major numeric matrix in the original feature order. No preprocessing is implicit. state.json contains coefficients/intercept, not an optimizer checkpoint. No pickle loading. Training uses scikit-learn; inference does not.\n')
        inventory={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in stage.iterdir()}
        (stage/'manifest.json').write_text(json.dumps({'schemaVersion':1,'backendID':BACKEND_ID,'stateFormat':STATE_FORMAT,'artifacts':inventory},sort_keys=True))
        check_cancelled()
        # Reserve exclusively, then publish the manifest last. Never replace an
        # empty directory created by another operation during export.
        destination.mkdir(mode=0o700);info=destination.stat();owned=(info.st_dev,info.st_ino)
        for child in stage.iterdir():
            if child.name!='manifest.json':
                check_cancelled();os.rename(child,destination/child.name)
        check_cancelled();os.rename(stage/'manifest.json',destination/'manifest.json')
    except BaseException:
        if owned is not None:
            try:
                info=destination.lstat()
                if (info.st_dev,info.st_ino)==owned:shutil.rmtree(destination)
            except FileNotFoundError:pass
        raise
    finally:
        if stage.exists():shutil.rmtree(stage)
    return destination
