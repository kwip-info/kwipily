#!/usr/bin/env python3
"""Run with the repaired runtime interpreter; exercises shipped backend dependencies."""
import json, importlib.util, tempfile, sqlite3
from pathlib import Path
import polars as pl
import numpy as np
import scipy
from scipy import linalg, sparse, optimize, special, fft
import sklearn
from sklearn.linear_model import LogisticRegression
import torch
import tensorflow as tf
from threadpoolctl import threadpool_info, threadpool_limits
with threadpool_limits(limits=2):
    pools = [p for p in threadpool_info() if p.get("prefix") == "libscipy_openblas"]
    assert pools and all(p["num_threads"] == 2 for p in pools), pools
assert importlib.util.find_spec('tkinter') is None
x=np.array([[-3.,0.],[-2.,1.],[-1.,-1.],[1.,1.],[2.,-1.],[3.,0.]])
y=np.array([0,0,0,1,1,1])
m=LogisticRegression(random_state=42).fit(x,y)
assert np.array_equal(m.predict(x),y)
a=np.array([[4.,1.],[1.,3.]])
assert np.allclose(a@linalg.solve(a,np.ones(2)),np.ones(2))
assert np.allclose(linalg.expm(np.zeros((2,2))),np.eye(2))
assert np.isfinite(special.gamma(3))
assert np.allclose(fft.ifft(fft.fft(x[:,0])).real,x[:,0])
assert np.allclose(sparse.csr_matrix(a).toarray(),a)
assert optimize.minimize(lambda z: np.sum(z*z),np.ones(2)).success
layer=torch.nn.Linear(2,2);opt=torch.optim.SGD(layer.parameters(),lr=.01)
loss=torch.nn.functional.cross_entropy(layer(torch.tensor(x,dtype=torch.float32)),torch.tensor(y));loss.backward();opt.step()
tf_model=tf.keras.Sequential([tf.keras.layers.Input(shape=(2,)),tf.keras.layers.Dense(2)])
tf_model.compile(optimizer='sgd',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
assert np.isfinite(tf_model.train_on_batch(x,y))
with tempfile.TemporaryDirectory() as d:
 csv_path=Path(d)/'polars.csv';csv_path.write_text('x,y\n1,a\n2,b\n')
 batches=pl.scan_csv(csv_path,infer_schema=False,glob=False).collect_batches(chunk_size=1,maintain_order=True,engine='streaming')
 assert sum(batch.height for batch in batches)==2
 frame=pl.DataFrame({'x':[1,2],'y':['a','b']});ipc=Path(d)/'polars.arrow';frame.write_ipc(ipc)
 assert pl.read_ipc(ipc).equals(frame)
 with sqlite3.connect(Path(d)/'rows.sqlite3') as db:
  db.execute('PRAGMA cache_size=-2048');db.execute('CREATE TABLE rows (n INTEGER PRIMARY KEY, value TEXT)')
  db.execute('INSERT INTO rows VALUES (1, ?)', ('table-spill',));db.commit()
  assert db.execute('SELECT value FROM rows WHERE n=1').fetchone()==('table-spill',)
 p=Path(d)/'model.keras';tf_model.save(p);reloaded=tf.keras.models.load_model(p)
 assert np.allclose(tf_model(x).numpy(),reloaded(x).numpy())
print(json.dumps({'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,'tensorflow':tf.__version__,'sqlite':sqlite3.sqlite_version,'polars':pl.__version__,'tkAbsent':True,'numericalAndTrainingSmoke':'passed'}))
