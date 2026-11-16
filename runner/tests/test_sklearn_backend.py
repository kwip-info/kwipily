import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("sklearn", reason="BACK-01 optional isolated framework environment")
from runner import sklearn_backend as backend


def samples():
    x = np.array([[-3., 0.], [-2., 1.], [-1., -1.], [1., 1.], [2., -1.], [3., 0.]])
    return x, np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)


def test_real_fit_save_reload_export_matches_sklearn_and_independent_equation(tmp_path):
    from sklearn.linear_model import LogisticRegression
    x, y = samples()
    configuration = backend.default_configuration()
    result = backend.fit(configuration, x, y)
    reference = LogisticRegression(C=1., tol=1e-4, max_iter=200, fit_intercept=True, solver='lbfgs').fit(x,y)
    heldout = np.array([[-1.5, .25], [1.25, -.5], [0., 0.]])
    expected = reference.predict_proba(heldout)
    assert np.allclose(backend.predict(result, heldout)['probabilities'], expected, atol=1e-12)
    state = tmp_path/'state.json'
    backend.save(result, state)
    loaded = backend.load(state)
    assert backend.predict(result, heldout) == backend.predict(loaded, heldout)
    assert backend.evaluate(loaded, x, y)['accuracy'] == 1
    output = tmp_path/'export'
    backend.export(result, output)
    input_path=tmp_path/'input.json'; input_path.write_text(json.dumps(heldout.tolist()))
    process=subprocess.run([sys.executable, '-I', str(output/'inference.py'),str(input_path)],capture_output=True,text=True,cwd=tmp_path)
    assert process.returncode == 0, process.stderr
    assert np.allclose(json.loads(process.stdout)['probabilities'],expected,atol=1e-12)
    with pytest.raises(FileExistsError): backend.export(result,output)


@pytest.mark.parametrize('change', [{'C':0}, {'maxIterations':True}, {'tolerance':float('nan')}, {'fitIntercept':1}, {'extra':1}])
def test_settings_reject_invalid_before_fit(change):
    config=backend.default_configuration();config.update(change)
    with pytest.raises(ValueError): backend.fit(config,*samples())


def test_data_contract_and_lifecycle_rejections(tmp_path):
    x,y=samples()
    for bad in (x.astype(np.float32),np.array([[float('inf'),0.]]),np.empty((0,2)),x[:,0]):
        with pytest.raises(ValueError):backend.fit(backend.default_configuration(),bad,y)
    with pytest.raises(ValueError):backend.fit(backend.default_configuration(),x,np.zeros(6,dtype=np.int64))
    with pytest.raises(ValueError):backend.fit(backend.default_configuration(),x,y.astype(float))
    with pytest.raises(RuntimeError,match='cancel'):
        backend.fit(backend.default_configuration(),x,y,check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError('cancel')))
    model=backend.fit(backend.default_configuration(),x,y)
    backend.save(model,tmp_path/'state.json')
    state=json.loads((tmp_path/'state.json').read_text());state['coefficients'][0]=float('nan')
    (tmp_path/'bad.json').write_text(json.dumps(state))
    with pytest.raises(ValueError):backend.load(tmp_path/'bad.json')
    (tmp_path/'link.json').symlink_to(tmp_path/'state.json')
    with pytest.raises(ValueError):backend.load(tmp_path/'link.json')


def test_zero_logit_tie_is_class_zero_and_stable_extreme_probabilities():
    x,y=samples();model=backend.fit(backend.default_configuration(),x,y)
    result=backend.predict(model,np.array([[0.,0.],[-1e10,0.],[1e10,0.]]))
    assert result['classes']==[0,0,1]
    assert np.isfinite(result['probabilities']).all()
    assert backend.capabilities()['unsupportedOperations'] == ['partialFit','resume','finetune','autograd','gpu']


def test_nonconvergence_is_reported_and_disabled_intercept_has_no_owned_value():
    x,y=samples();config=backend.default_configuration();config['maxIterations']=1;config['fitIntercept']=False
    model=backend.fit(config,x,y)
    assert model['intercept']==0
    assert model['iterations']==1
    assert model['converged'] is False


def test_export_cancellation_cleans_staging_without_publishing(tmp_path):
    calls=[]
    def cancel_late():
        calls.append(True)
        if len(calls)>1:raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError):backend.export(backend.fit(backend.default_configuration(),*samples()),tmp_path/'export',check_cancelled=cancel_late)
    assert list(tmp_path.iterdir())==[]
