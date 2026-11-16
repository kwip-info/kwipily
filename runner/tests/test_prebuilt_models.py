import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from runner import prebuilt_models as p

class PrebuiltModelsTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(42).normal(size=(60,3))
    def test_supervised_and_cluster_roundtrips(self):
        from sklearn.metrics import adjusted_rand_score
        for name in p.catalog():
            with self.subTest(name=name):
                c=p.defaults(name);task=p.catalog()[name]['task']
                y=(self.x[:,0]>0).astype(int) if task=='classification' else self.x[:,0]*2-self.x[:,1] if task=='regression' else None
                model,fit=p.fit(c,self.x,y)
                self.assertEqual(len(fit['values']),60)
                if name=='dbscan':
                    self.assertIsNone(model);continue
                self.assertEqual(p.predict(model,self.x),p.predict(p.load(p.canonical(model)),self.x))
                with tempfile.TemporaryDirectory() as d:
                    p.export(model,Path(d)/'portable')
                    self.assertEqual(p.load((Path(d)/'portable/state.json').read_bytes()),model)
                if task=='regression':self.assertTrue(np.isfinite(p.evaluate(model,self.x,y)['rmse']))
                if task=='classification':self.assertGreater(p.evaluate(model,self.x,y)['accuracy'],.85)
                if task=='clustering':self.assertGreater(adjusted_rand_score(fit['values'],p.predict(model,self.x)['values']),.99)
    def test_tamper_and_admission(self):
        m,_=p.fit(p.defaults('kmeans'),self.x)
        m['parameters']['centers'][0][0]=float('nan')
        with self.assertRaises(ValueError):p.load(json.dumps(m).encode())
        with self.assertRaises(ValueError):p.fit(p.defaults('dbscan'),np.ones((20001,2)))
        with self.assertRaises(ValueError):p.fit(p.defaults('ridge'),self.x,np.full(60,np.nan))
        with self.assertRaises(ValueError):p.fit({**p.defaults('ridge'),'extra':1},self.x,np.ones(60))
    def test_no_synthetic_progress_and_cancel(self):
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):p.fit(p.defaults('kmeans'),self.x,check=cancel)
    def test_multiclass_and_independent_parity(self):
        from sklearn.linear_model import LogisticRegression
        y=np.argmax(self.x,axis=1)
        model,_=p.fit(p.defaults('logistic'),self.x,y)
        expected=LogisticRegression(C=1,max_iter=200,tol=.0001).fit(self.x,y)
        actual=p.predict(model,self.x)
        np.testing.assert_allclose(actual['probabilities'],expected.predict_proba(self.x),rtol=1e-9,atol=1e-12)
        self.assertEqual(actual['values'],expected.predict(self.x).tolist())

class IndependentAdapterParityTests(unittest.TestCase):
    def test_regression_forests_and_mixtures_against_upstream(self):
        from sklearn.linear_model import Ridge,ElasticNet
        from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
        from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
        from sklearn.mixture import GaussianMixture
        x=np.random.default_rng(1).normal(size=(50,3));y=x[:,0]*2-x[:,1];labels=(y>0).astype(int);new=x[:13]+.017
        refs={
            'ridge':Ridge(alpha=1,fit_intercept=True,solver='svd'),
            'elasticNet':ElasticNet(alpha=1,l1_ratio=.5,fit_intercept=True,max_iter=200,tol=.0001,selection='cyclic'),
            'randomForestClassifier':RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=1,random_state=42,n_jobs=1),
            'randomForestRegressor':RandomForestRegressor(n_estimators=100,max_depth=8,min_samples_leaf=1,random_state=42,n_jobs=1),
            'kmeans':KMeans(n_clusters=4,n_init=10,max_iter=200,tol=.0001,random_state=42),
            'miniBatchKmeans':MiniBatchKMeans(n_clusters=4,n_init=10,max_iter=200,tol=.0001,random_state=42,batch_size=1024),
            'gaussianMixture':GaussianMixture(n_components=4,covariance_type='diag',n_init=10,max_iter=200,tol=.0001,reg_covar=1e-6,random_state=42),
            'dbscan':DBSCAN(eps=.5,min_samples=5,n_jobs=1,algorithm='brute')}
        for a,ref in refs.items():
            with self.subTest(a=a):
                task=p.catalog()[a]['task'];target=labels if task=='classification' else y if task=='regression' else None
                m,out=p.fit(p.defaults(a),x,target);ref.fit(x,target)
                if a=='dbscan':self.assertEqual(out['values'],ref.labels_.tolist());continue
                actual=p.predict(m,new)
                np.testing.assert_allclose(actual['values'],ref.predict(new),atol=1e-10)
                if 'probabilities' in actual:np.testing.assert_allclose(actual['probabilities'],ref.predict_proba(new),atol=1e-10)
                if 'memberships' in actual:np.testing.assert_allclose(actual['memberships'],ref.predict_proba(new),atol=1e-10)
                if 'distances' in actual:np.testing.assert_allclose(actual['distances'],ref.transform(new),atol=1e-10)

if __name__=='__main__':unittest.main()
