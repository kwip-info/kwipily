"""Bounded final-forward host wall intervals and owned-process RSS samples.

Never kernel timing, per-operation allocated memory, process peak or GPU memory.
Instrumentation failure is unavailable evidence, not an ML computation failure.
"""
import os
import time

MAX_BYTES = 131072
MAX_DURATION_NS = 86400 * 10**9
MAX_RESIDENT_BYTES = 2**63 - 1


def resident_bytes():
    if __package__:
        from .worker_memory import ResidentMemoryReader
    else:
        from worker_memory import ResidentMemoryReader
    return ResidentMemoryReader().read(os.getpid())


class PerformanceCapture:
    def __init__(self, nodes, device, *, clock=None, memory_reader=None, synchronize=None, dependencies=None):
        self.nodes = tuple(nodes)
        if not 1 <= len(self.nodes) <= 128 or len(set(self.nodes)) != len(self.nodes) or any(type(n) is not str or not n or len(n)>128 for n in self.nodes):
            raise ValueError('Performance capture requires one to128 unique bounded node identities.')
        if device not in ('cpu', 'mps'):
            raise ValueError('Performance capture supports CPU or MPS.')
        self.dependencies = {node: set() for node in self.nodes} if dependencies is None else {node: set(values) for node, values in dependencies.items()}
        if set(self.dependencies) != set(self.nodes) or any(not values <= set(self.nodes) or node in values for node, values in self.dependencies.items()):
            raise ValueError('Performance dependencies must belong to the captured graph.')
        self.completed = set()
        self.device = device
        self.clock = clock if clock is not None else time.perf_counter_ns
        self.memory_reader = memory_reader if memory_reader is not None else resident_bytes
        if synchronize is None and device == 'mps':
            import torch
            synchronize = torch.mps.synchronize
        self.synchronize = synchronize
        self.records = []
        self.pending = None

    def _memory(self):
        try:
            value = self.memory_reader()
            return value if type(value) is int and 0 < value <= MAX_RESIDENT_BYTES else None
        except Exception:
            return None

    def _clock(self):
        try:
            value = self.clock()
            return value if type(value) is int and 0 <= value <= 2**63-1 else None
        except Exception:
            return None

    def _sync(self):
        if self.device == 'cpu': return True
        try:
            self.synchronize()
            return True
        except Exception:
            return False

    def observe(self, state, node):
        if state == 'started':
            if self.pending is not None or node not in self.dependencies or node in self.completed or not self.dependencies[node] <= self.completed:
                raise ValueError('Performance observation does not follow captured graph dependencies.')
            before = self._memory()
            synchronized = self._sync()
            self.pending = (node, self._clock(), before, synchronized)
        elif state == 'completed':
            if self.pending is None or node != self.pending[0]:
                raise ValueError('Performance completion has no matching operation start.')
            _, start, before, initial_sync = self.pending
            final_sync = self._sync()
            end = self._clock()
            after = self._memory()
            duration = None
            reason = None
            if not initial_sync or not final_sync:
                reason = 'deviceSynchronizationUnavailable'
            elif start is None or end is None or not 0 <= end-start <= MAX_DURATION_NS:
                reason = 'clockUnavailable'
            else:
                duration = end-start
            self.records.append(dict(nodeID=node, durationNanoseconds=duration, timingUnavailableReason=reason,
                residentBytesBefore=before, residentBytesAfter=after,
                memoryUnavailableReason=None if before is not None and after is not None else 'residentMemoryUnavailable'))
            self.completed.add(node)
            self.pending = None
        else:
            raise ValueError('Unknown performance observation state.')

    def artifact(self, run_id, graph_hash, policy):
        if self.pending is not None or len(self.records) != len(self.nodes):
            raise ValueError('Incomplete operation capture cannot become a completed performance profile.')
        return dict(schemaVersion=1,runID=run_id,graphHash=graph_hash,device=self.device,stage='validationCapture',
            selection='firstValidationRows',maximumRows=8,policy=dict(policy),
            timingMethod='hostWall' if self.device=='cpu' else 'synchronizedHostWall',
            memoryMethod='processResidentBytes',records=[dict(record) for record in self.records])
