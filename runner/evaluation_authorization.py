"""Immutable single-evaluation authority; no experiment ledger or phase semantics."""
import fcntl
import os
from pathlib import Path
import stat

if __package__:
    from .execution_budget import validate_versioned_request, validate_remaining_budget
    from .worker_authorization import (directory, read_regular, object_json, require, hash_value,
        canonical_uuid, file_identity, signature, validate_identity, sha)
else:
    from execution_budget import validate_versioned_request, validate_remaining_budget
    from worker_authorization import (directory, read_regular, object_json, require, hash_value,
        canonical_uuid, file_identity, signature, validate_identity, sha)
import json

OWNERSHIP_FIELDS = {'schemaVersion','kind','evaluationID','projectID','requestIdentity','requestSHA256',
                    'directoryIdentity','ownerLockIdentity','activityLockIdentity'}
REQUEST_FIELDS = {'schemaVersion','kind','evaluationID','projectID','requestIdentity','snapshotReceiptSHA256','settings'}
CLAIM_FIELDS = {'schemaVersion','evaluationID','requestSHA256','ownershipSHA256','state'}


class EvaluationAuthority:
    identity_key = 'evaluationID'
    kind = 'frozenEvaluation'
    namespace = 'workflow-evaluations'
    ownership_fields = OWNERSHIP_FIELDS
    claim_fields = CLAIM_FIELDS
    supports_launch_budget = True

    def __init__(self, path, request_pin, ownership_pin, *, inherited_fd=None, launch_budget_sha256=None):
        self._closed = False
        self._directory_fd = self._owner_fd = self.activity_fd = None
        self._path = os.fspath(path)
        self._launch_budget_pin = launch_budget_sha256
        self._launch_budget_bytes = None
        self.remaining_seconds = None
        self._request_pin = request_pin; self._ownership_pin = ownership_pin
        try:
            hash_value(request_pin); hash_value(ownership_pin)
            if inherited_fd is not None:
                require(type(inherited_fd) is int and inherited_fd > 2, 'Invalid inherited activity descriptor')
                self.activity_fd = os.dup(inherited_fd)
                os.set_inheritable(self.activity_fd, False)
            self._directory_fd = directory(self._path)
            info = os.fstat(self._directory_fd)
            require(info.st_uid == os.geteuid() and info.st_mode & 0o022 == 0, 'Unsafe evaluation directory')
            self._ownership_bytes = read_regular(self._directory_fd, 'ownership.json', 16384)
            require(sha(self._ownership_bytes) == ownership_pin, 'Evaluation ownership digest mismatch')
            self._ownership = object_json(self._ownership_bytes)
            metadata = self._ownership
            require(set(metadata) == self.ownership_fields and type(metadata['schemaVersion']) is int and metadata['schemaVersion'] == 1
                    and metadata['kind'] == self.kind, 'Invalid evaluation ownership')
            for key in (self.identity_key,'projectID'): canonical_uuid(metadata[key])
            for key in ('requestIdentity','requestSHA256'): hash_value(metadata[key])
            for key in ('directoryIdentity','ownerLockIdentity','activityLockIdentity'): validate_identity(metadata[key])
            require(metadata['requestSHA256'] == request_pin and file_identity(info) == metadata['directoryIdentity'])
            location = Path(self._path)
            require(location.name == metadata[self.identity_key] and location.parent.name == metadata['projectID']
                    and location.parent.parent.name == self.namespace, 'Wrong evaluation directory identity')
            self._owner_fd = self._lock('owner.lock', metadata['ownerLockIdentity'])
            if inherited_fd is None:
                self.activity_fd = self._lock('activity.lock', metadata['activityLockIdentity'])
                fcntl.flock(self.activity_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            else:
                self._validate_lock('activity.lock', self.activity_fd, metadata['activityLockIdentity'])
            self._request_bytes = read_regular(self._directory_fd, 'request.json', 256000)
            require(sha(self._request_bytes) == request_pin, 'Evaluation request digest mismatch')
            self.request = object_json(self._request_bytes)
            # Keep an independent decoded pin so frequent cancellation checks
            # detect caller mutation without repeatedly parsing identical bytes.
            self._request_snapshot = object_json(self._request_bytes)
            self._validate_request()
            self._load_launch_budget()
            self._check(False)
            if inherited_fd is None:
                claim = json.dumps(self._claim(), sort_keys=True, separators=(',', ':')).encode()
                descriptor = os.open('claim.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=self._directory_fd)
                try:
                    remaining = memoryview(claim)
                    while remaining:
                        count = os.write(descriptor, remaining)
                        require(count > 0, 'Incomplete evaluation claim write'); remaining = remaining[count:]
                    os.fsync(descriptor)
                finally: os.close(descriptor)
                os.fsync(self._directory_fd)
            self.check()
        except BaseException:
            self.close(); raise

    def _claim(self):
        claim = dict(schemaVersion=1, **{self.identity_key:self._ownership[self.identity_key]}, requestSHA256=self._request_pin,
                    ownershipSHA256=self._ownership_pin, state='claimed')
        if self._launch_budget_pin is not None:claim['launchBudgetSHA256']=self._launch_budget_pin
        return claim

    def _validate_request(self):
        value = self.request
        validate_versioned_request(value, REQUEST_FIELDS | ({'retainComponentOutputs'} if 'retainComponentOutputs' in value else set()))
        require('retainComponentOutputs' not in value or value['retainComponentOutputs'] is False,'Invalid evaluation retention override')
        require(value['kind'] == 'frozenEvaluation', 'Invalid evaluation request kind')
        for key in ('evaluationID','projectID'): canonical_uuid(value[key])
        for key in ('requestIdentity','snapshotReceiptSHA256'): hash_value(value[key])
        for key in ('evaluationID','projectID','requestIdentity'):
            require(value[key] == self._ownership[key], 'Evaluation request identity differs from ownership')
        settings=value['settings'];base={'device','batchSize','previewLimit'}
        require(isinstance(settings,dict) and (set(settings)==base or set(settings)==base|{'settingsVersion','partitionRole'} and type(settings['settingsVersion']) is int and settings['settingsVersion']==2 and settings['partitionRole']=='validation'), 'Invalid evaluation settings')
        require(isinstance(settings['device'],str) and settings['device'] in ('auto','cpu','mps'))
        require(type(settings['batchSize']) is int and 1<=settings['batchSize']<=100000)
        require(type(settings['previewLimit']) is int and 0<=settings['previewLimit']<=100)

    def _load_launch_budget(self):
        bounded = self.supports_launch_budget and self.request['schemaVersion'] == 2
        if not bounded:
            require(self._launch_budget_pin is None, 'Unexpected launch budget for legacy or inline-budget request')
            return
        hash_value(self._launch_budget_pin)
        raw = read_regular(self._directory_fd, 'launch-budget.json', 4096)
        require(sha(raw) == self._launch_budget_pin, 'Launch budget digest mismatch')
        value = object_json(raw)
        require(set(value) == {'schemaVersion','requestSHA256','remainingSeconds'}
                and type(value['schemaVersion']) is int and value['schemaVersion'] == 1,
                'Invalid launch budget fields')
        require(value['requestSHA256'] == self._request_pin, 'Launch budget belongs to another request')
        validate_remaining_budget(self.request['executionBudget'], value['remainingSeconds'])
        self._launch_budget_bytes = raw
        self.remaining_seconds = value['remainingSeconds']

    def _validate_lock(self, name, descriptor, pin):
        info = os.fstat(descriptor); named = os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1 and info.st_mode & 0o022 == 0)
        require(signature(info) == signature(named) and file_identity(info) == pin, 'Evaluation lock identity changed')

    def _lock(self, name, pin):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=self._directory_fd)
        try:
            self._validate_lock(name, descriptor, pin)
            return descriptor
        except BaseException: os.close(descriptor); raise

    def _owner_held(self):
        try:
            fcntl.flock(self._owner_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        # This is our independent probe description, never an inherited lock.
        fcntl.flock(self._owner_fd, fcntl.LOCK_UN)
        raise ValueError('Evaluation owner is no longer held exclusively')

    def _read_control(self,name,maximum):
        return read_regular(self._directory_fd,name,maximum)

    def _check(self, claimed):
        require(not self._closed, 'Evaluation authority is closed')
        current = directory(self._path)
        try:
            info = os.fstat(current)
            require(file_identity(info) == self._ownership['directoryIdentity'] and info.st_uid == os.geteuid()
                    and info.st_mode & 0o022 == 0, 'Evaluation directory changed')
        finally: os.close(current)
        require(self._read_control('ownership.json', 16384) == self._ownership_bytes, 'Evaluation ownership changed')
        for name, descriptor, key in [('owner.lock',self._owner_fd,'ownerLockIdentity'), ('activity.lock',self.activity_fd,'activityLockIdentity')]:
            self._validate_lock(name, descriptor, self._ownership[key])
        self._owner_held()
        require(self._read_control('request.json', 256000) == self._request_bytes
                and self.request == self._request_snapshot, 'Evaluation request changed')
        if self._launch_budget_bytes is not None:
            require(self._read_control('launch-budget.json', 4096) == self._launch_budget_bytes, 'Launch budget changed')
        if claimed:
            claim = object_json(self._read_control('claim.json', 4096))
            require(set(claim) == (self.claim_fields | ({'launchBudgetSHA256'} if self._launch_budget_pin is not None else set())) and type(claim['schemaVersion']) is int and claim == self._claim(), 'Evaluation claim changed')

    def check(self): self._check(True)
    def close(self):
        if self._closed: return
        self._closed = True
        for name in ('activity_fd','_owner_fd','_directory_fd'):
            descriptor = getattr(self, name, None)
            if descriptor is not None:
                os.close(descriptor); setattr(self, name, None)
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def __del__(self): self.close()


def claim_evaluation(directory, request_sha256, ownership_sha256, *, launch_budget_sha256=None):
    return EvaluationAuthority(directory, request_sha256, ownership_sha256, launch_budget_sha256=launch_budget_sha256)


def inherit_evaluation(directory, request_sha256, ownership_sha256, activity_fd, *, launch_budget_sha256=None):
    return EvaluationAuthority(directory, request_sha256, ownership_sha256, inherited_fd=activity_fd, launch_budget_sha256=launch_budget_sha256)
