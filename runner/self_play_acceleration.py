"""Bounded generic memoization and a single-owner batched neural evaluator.

Rules remain authoritative CPU operations. No board layout or game identity dispatch.
"""
import copy,time,threading,concurrent.futures
from collections import OrderedDict,deque
if __package__:
 from .game_episodes import encoded,digest,require
else:
 from game_episodes import encoded,digest,require

class BoundedCache:
 def __init__(self,maximum_bytes=16*1024*1024):
  require(type(maximum_bytes) is int and 0<=maximum_bytes<=268435456,'Invalid cache byte allowance.')
  self.maximum=maximum_bytes;self.bytes=0;self.values=OrderedDict();self.hits=0;self.misses=0
 def get(self,key):
  if key not in self.values:self.misses+=1;return None
  self.hits+=1;value,size=self.values.pop(key);self.values[key]=(value,size);return copy.deepcopy(value)
 def put(self,key,value):
  size=len(encoded(value))+len(key)
  if size>self.maximum:return
  if key in self.values:self.bytes-=self.values.pop(key)[1]
  while self.values and self.bytes+size>self.maximum:self.bytes-=self.values.popitem(last=False)[1][1]
  self.values[key]=(copy.deepcopy(value),size);self.bytes+=size

class CachedRules:
 """Complete state/action/actor keys, never observation-only legal-state keys."""
 def __init__(self,definition,maximum_bytes=16*1024*1024):
  object.__setattr__(self,'definition',definition);object.__setattr__(self,'cache',BoundedCache(maximum_bytes))
 def __getattr__(self,name):return getattr(self.definition,name)
 def __setattr__(self,name,value):setattr(self.definition,name,value)
 def call(self,name,*args):
  self.definition.check_cancelled();key=name+':'+digest(args);value=self.cache.get(key)
  if value is not None:return value
  value=getattr(self.definition,name)(*args);self.cache.put(key,value);return copy.deepcopy(value)
 def outcome(self,state):return self.call('outcome',state)
 def legal_actions(self,state):return self.call('legal_actions',state)
 def observation(self,state,player):return self.call('observation',state,player)
 def step(self,state,action):return self.call('step',state,action)

class InferenceQueue:
 """At most one outstanding leaf per caller; one model/device owner, finite queue wait."""
 def __init__(self,model,*,batch_size=16,wait_seconds=.002,cache_bytes=16*1024*1024,check=lambda:None):
  require(1<=batch_size<=256 and 0<=wait_seconds<=.05,'Invalid inference batching.')
  self.model=model;self.batch=batch_size;self.wait=wait_seconds;self.check=check;self.cache=BoundedCache(cache_bytes)
  self.cv=threading.Condition();self.pending=deque();self.closed=False;self.calls=0;self.rows=0;self.seconds=0.
  self.thread=threading.Thread(target=self._loop,name='kwip-policy-inference',daemon=True);self.thread.start()
 def predict(self,x):
  self.check();key=digest(x)
  with self.cv:
   require(not self.closed,'Inference queue closed.');cached=self.cache.get(key)
   if cached is not None:return cached
   require(len(self.pending)<256,'Inference queue capacity reached.')
   future=concurrent.futures.Future();self.pending.append((key,x,future));self.cv.notify()
  deadline=time.monotonic()+65
  while True:
   self.check()
   try:return future.result(timeout=.1)
   except concurrent.futures.TimeoutError:
    if time.monotonic()>deadline:raise TimeoutError('Batched inference response timed out.')
 def _loop(self):
  try:
   while True:
    with self.cv:
     self.cv.wait_for(lambda:self.pending or self.closed)
     if self.closed and not self.pending:return
     end=time.monotonic()+self.wait
     while len(self.pending)<self.batch and not self.closed and time.monotonic()<end:self.cv.wait(end-time.monotonic())
     jobs=[self.pending.popleft() for _ in range(min(self.batch,len(self.pending)))]
    try:
     self.check();start=time.monotonic();values=self.model.predict_batch([x for _,x,_ in jobs]);self.seconds+=time.monotonic()-start;self.calls+=1;self.rows+=len(jobs)
     require(len(values)==len(jobs),'Inference batch size differs.')
     with self.cv:
      for (key,x,future),value in zip(jobs,values):self.cache.put(key,value);future.set_result(value)
    except BaseException as error:
     for _,_,future in jobs:
      if not future.done():future.set_exception(error)
  finally:
   with self.cv:
    for _,_,f in self.pending:
     if not f.done():f.set_exception(RuntimeError('Inference queue stopped.'))
    self.pending.clear()
 def close(self):
  with self.cv:self.closed=True;self.cv.notify_all()
  self.thread.join(timeout=65)
  require(not self.thread.is_alive(),'Inference owner did not stop.')

class CachedPolicy:
 def __init__(self,model,maximum_bytes=16777216):self.model=model;self.cache=BoundedCache(maximum_bytes)
 def __getattr__(self,key):return getattr(self.model,key)
 def predict(self,x):
  key=digest(x);out=self.cache.get(key)
  if out is None:out=self.model.predict(x);self.cache.put(key,out)
  return out
