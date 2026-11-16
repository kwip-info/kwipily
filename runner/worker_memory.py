"""macOS per-process resident bytes via libproc, independent of the worker's GIL.

ABI: macOS SDK sys/resource.h rusage_info_v0 and libproc.h proc_pid_rusage.
RSS is sampled: it is neither an allocation reservation nor a GPU/whole-app cap.
"""
import ctypes
import os
import sys


class _UsageV0(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            'user_time', 'system_time', 'pkg_idle_wakeups', 'interrupt_wakeups',
            'pageins', 'wired_size', 'resident_size', 'physical_footprint',
            'process_start', 'process_exit')]


class ResidentMemoryReader:
    def __init__(self):
        if sys.platform != 'darwin':
            raise ValueError('Resident memory budgets require the supported macOS libproc monitor.')
        self._library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self._read = self._library.proc_pid_rusage
        self._read.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_UsageV0)]
        self._read.restype = ctypes.c_int

    def read(self, pid):
        if type(pid) is not int or not 0 < pid <= 2147483647:
            raise ValueError('A live owned worker PID is required for memory observation.')
        value = _UsageV0()
        if self._read(pid, 0, ctypes.byref(value)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if not value.resident_size:
            raise ValueError('Resident memory observation was unavailable.')
        return int(value.resident_size)
