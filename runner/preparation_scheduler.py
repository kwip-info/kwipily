"""Bounded ready groups for qualified in-process native preparation kernels.

The numerical runtime keeps its one captured process-wide pool. This does not
spawn arbitrary Python processes or multiply that pool for each canvas branch.
Unqualified operators and external effects are exclusive barriers.
"""
from graphlib import TopologicalSorter,CycleError
from concurrent.futures import ThreadPoolExecutor,TimeoutError
from contextvars import copy_context
import sqlite3
import threading


class StableReadyQueue:
    def __init__(self,order,dependencies):
        self.order=tuple(order);self.rank={node:i for i,node in enumerate(self.order)}
        if len(self.rank)!=len(self.order) or set(dependencies)!=set(self.order) or any(set(parents)-set(self.order) for parents in dependencies.values()):
            raise ValueError('Ready scheduling requires the exact admitted graph.')
        self.sorter=TopologicalSorter(dependencies)
        try:self.sorter.prepare()
        except CycleError as error:raise ValueError('Ready graph contains a cycle.') from error
        self.pending=set();self.inflight=set();self.completed=set()

    @property
    def finished(self):return len(self.completed)==len(self.order)

    def take(self,capable,maximum):
        if type(maximum) is not int or maximum<1:raise ValueError('Invalid ready-task reservation.')
        # This first scheduler uses stable bounded ready groups: all effects and
        # observations commit in reference order before the next group is admitted.
        if self.inflight:return ()
        self.pending.update(self.sorter.get_ready())
        ready=sorted(self.pending,key=self.rank.__getitem__)
        chosen=[]
        for node in ready:
            if chosen and not capable(node):break
            chosen.append(node)
            if len(chosen)>=maximum or not capable(node):break
        self.inflight.update(chosen);self.pending.difference_update(chosen)
        return tuple(chosen)

    def complete(self,node):
        if node not in self.inflight:raise ValueError('Only an admitted running task can complete.')
        self.sorter.done(node);self.inflight.remove(node);self.completed.add(node)


def maximum_native_tasks():
    if __package__:
        from .worker_resources import cpu_threads,parallel_tasks
        from .workload_policy import current_policy
    else:
        from worker_resources import cpu_threads,parallel_tasks
        from workload_policy import current_policy
    # SQLite FULLMUTEX support is required for concurrent immutable ID reads.
    admitted=parallel_tasks()
    return min(cpu_threads(),current_policy().limit('parallelTasks'),admitted if admitted is not None else cpu_threads()) if sqlite3.threadsafety==3 else 1


def can_overlap(operator,configuration,arguments,kernel):
    if __package__:
        from .dataflow_columnar import ColumnarRows,ColumnarIDs
        from .dataflow_shared import _boolean_kernel,supports_shared_table
    else:
        from dataflow_columnar import ColumnarRows,ColumnarIDs
        from dataflow_shared import _boolean_kernel,supports_shared_table
    table=arguments.get('table')
    if table is None or not supports_shared_table(table):return False
    if operator=='columns' and kernel=='columns.sharedProjection.v1':return True
    return (operator=='filterRows' and kernel=='filter.booleanSelectionOrReference.v1'
            and _boolean_kernel(configuration['expression'],table.columns) is not None)


def run_ready_group(jobs,check):
    """Compute qualified siblings; return in order and drain before any release.

    Callers alone commit observations, cache state and external effects. Contexts
    capture the same operation policy, scratch budget and grants. A failure never
    triggers another execution or write attempt.
    """
    if not jobs:return []
    if len(jobs)>maximum_native_tasks():raise ValueError('Ready group exceeds its captured task/CPU reservation.')
    stopped=threading.Event();check_lock=threading.RLock()
    def guarded_check():
        if stopped.is_set():raise InterruptedError('Preparation ready group stopped.')
        with check_lock:check()
    def work(function):
        if __package__:from .dataflow_columnar import columnar_reader_scope
        else:from dataflow_columnar import columnar_reader_scope
        with columnar_reader_scope():
            guarded_check();return function(guarded_check)
    pool=ThreadPoolExecutor(max_workers=len(jobs),thread_name_prefix='kwip-native')
    futures=[]
    try:
        for key,function in jobs:futures.append(pool.submit(copy_context().run,work,function))
        results=[]
        for future in futures:
            while True:
                guarded_check()
                try:results.append(future.result(timeout=0.05));break
                except TimeoutError:
                    # A kernel's own TimeoutError must not be mistaken for the
                    # wait timeout and retried forever.
                    if future.done():raise
        return results
    except BaseException:
        stopped.set()
        for future in futures:future.cancel()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
