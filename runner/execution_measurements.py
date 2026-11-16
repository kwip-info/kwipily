"""Opt-in, finite per-plan phase aggregates; never retain inputs, outputs or exceptions.

No telemetry sink or dataset serialization is installed here. The caller owns the
captured observation policy. One aggregate per admitted node, no per-row trace.
"""
from contextlib import contextmanager
from copy import deepcopy
import time
import threading

PHASES=('dispatch','cacheKey','cacheLookup','execution','observation','release')


class ExecutionMeasurements:
    def __init__(self,node_ids,*,clock=time.perf_counter_ns,on_node=None):
        nodes=tuple(node_ids)
        if any(type(n) is not str or not n for n in nodes) or len(set(nodes))!=len(nodes):
            raise ValueError('Measurements require distinct admitted node identities')
        self._order=nodes;self._allowed=set(nodes);self._entries={};self._clock=clock;self._on_node=on_node
        self._lock=threading.RLock();self._emission_lock=threading.RLock()
        self._active=set();self._phase_owner={};self._failure_phase={}

    @contextmanager
    def node(self,node_id):
        with self._lock:
            if node_id not in self._allowed or node_id in self._entries:
                raise ValueError('Measurement node is foreign, repeated or concurrently owned')
            entry={'nodeID':node_id,'status':'running','cacheHit':False,'milliseconds':{p:0.0 for p in PHASES}}
            self._entries[node_id]=entry;self._active.add(node_id)
        start=self._clock()
        try:
            yield
            with self._lock:entry['status']='completed'
        except BaseException as error:
            with self._lock:
                entry['status']='failed' if isinstance(error,Exception) else 'interrupted'
                entry['failurePhase']=self._failure_phase.get(node_id) or 'boundary'
            raise
        finally:
            total=max(0,self._clock()-start)/1_000_000
            with self._lock:
                entry['wallMilliseconds']=total
                entry['unattributedMilliseconds']=max(0,total-sum(entry['milliseconds'].values()))
                self._active.remove(node_id);self._failure_phase.pop(node_id,None)
                snapshot=deepcopy(entry)
            if self._on_node is not None:
                try:
                    with self._emission_lock:self._on_node(snapshot)
                except BaseException:
                    if entry["status"]=="completed":raise

    @contextmanager
    def phase(self,node_id,phase):
        with self._lock:
            if node_id not in self._active or phase not in PHASES or node_id in self._phase_owner:
                raise ValueError('Measurement requires its active node and one owned fixed phase')
            self._phase_owner[node_id]=threading.get_ident()
        start=self._clock()
        try:yield
        except BaseException:
            with self._lock:self._failure_phase[node_id]=phase
            raise
        finally:
            # A monotonic host clock, not a caller timestamp or backend prose.
            elapsed=max(0,self._clock()-start)
            with self._lock:
                self._entries[node_id]['milliseconds'][phase]+=elapsed/1_000_000
                self._phase_owner.pop(node_id,None)

    def cache_hit(self,node_id,reused):
        with self._lock:
            if node_id not in self._active or type(reused) is not bool:
                raise ValueError('Cache measurement requires its active node')
            self._entries[node_id]['cacheHit']=reused

    def report(self):
        with self._lock:
            return {'schemaVersion':1,'scope':'host phase wall time; not CPU, GPU, frame or whole-process-memory measurement',
                    'nodes':deepcopy([self._entries[n] for n in self._order if n in self._entries])}
