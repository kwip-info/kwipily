"""Dedicated frozen evaluation artifacts; native selection supplies archive authority."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import errno
import stat
import uuid

import torch

if __package__:
    from .artifact_storage import ArtifactStorage, ArtifactStorageBudget
    from .execution_budget import validate_versioned_request
    from .worker_authorization import directory, read_regular, object_json, require, hash_value
    from .frozen_checkpoint import load_frozen
    from .prepared_data import load_evaluation_partition
    from .final_evaluation import evaluate_classifier,evaluate_frozen_objective
else:
    from artifact_storage import ArtifactStorage, ArtifactStorageBudget
    from execution_budget import validate_versioned_request
    from worker_authorization import directory, read_regular, object_json, require, hash_value
    from frozen_checkpoint import load_frozen
    from prepared_data import load_evaluation_partition
    from final_evaluation import evaluate_classifier,evaluate_frozen_objective

EVALUATION_STORAGE_BYTES = 443_000_000

REQUEST_FIELDS = {'schemaVersion','kind','evaluationID','projectID','requestIdentity','snapshotReceiptSHA256','settings'}
REFERENCE_FIELDS = {'projectID','directoryID','runID','requestIdentity','outcomeSHA256','modelManifestSHA256','preparationManifestSHA256','checkpointSHA256'}
LIMITS = {'outcome.json':256000,'capture.json':8000000,'workflow.json':2000000,
          'model-registry.json':2000000,'data-registry.json':2000000,'runtime-evidence.json':256000,
          'training/artifacts/manifest.json':20000000,'training/artifacts/checkpoint.pt':100000000,
          'preparation/artifacts/manifest.json':2000000}


def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def identity(value):
    require(isinstance(value, str), 'Expected UUID identity')
    try: uuid.UUID(value)
    except (ValueError, AttributeError): raise ValueError('Expected UUID identity') from None


def path_parts(value):
    require(isinstance(value, str) and '\0' not in value and '\\' not in value, 'Invalid snapshot path')
    parts = value.split('/')
    require(all(part and part not in ('.', '..') for part in parts), 'Invalid snapshot path')
    return parts


def limit(path):
    if path in LIMITS: return LIMITS[path]
    parts = path_parts(path)
    require(len(parts) == 3 and parts[:2] == ['preparation', 'artifacts'], 'Unsupported snapshot artifact')
    return 50000000


def read_nested(parent, path, maximum):
    parts = path_parts(path); current = os.dup(parent)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current); current = child
        return read_regular(current, parts[-1], maximum)
    finally: os.close(current)


def settings(value):
    base={'device','batchSize','previewLimit'}
    require(isinstance(value,dict) and (set(value)==base or set(value)==base|{'settingsVersion','partitionRole'} and type(value['settingsVersion']) is int and value['settingsVersion']==2 and value['partitionRole']=='validation'), 'Invalid evaluation settings fields')
    require(value['device'] in ('auto','cpu','mps') and isinstance(value['device'], str), 'Invalid evaluation device')
    require(type(value['batchSize']) is int and 1 <= value['batchSize'] <= 100000, 'Invalid evaluation batch size')
    require(type(value['previewLimit']) is int and 0 <= value['previewLimit'] <= 100, 'Invalid evaluation preview limit')


def _file_identity(info): return info.st_dev, info.st_ino


class _VerifiedScratch:
    """Private scratch anchored to its operation owner, including cleanup after rename."""
    def __init__(self, parent):
        self.parent_path = Path(os.path.abspath(os.fspath(parent)))
        self.parent = directory(str(self.parent_path))
        self.parent_identity = os.fstat(self.parent)
        self.name = '.verified-' + uuid.uuid4().hex
        self.path = self.parent_path / self.name
        self.fd = None
        self.owned = {}
        try:
            os.mkdir(self.name, mode=0o700, dir_fd=self.parent)
            self.fd = os.open(self.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.parent)
            self.identity = os.fstat(self.fd)
        except BaseException:
            if self.fd is not None: os.close(self.fd)
            os.close(self.parent)
            raise

    def prepare(self, name):
        parts = path_parts(name)
        current = os.dup(self.fd)
        try:
            for index, part in enumerate(parts[:-1]):
                relative = '/'.join(parts[:index + 1])
                if relative not in self.owned:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
                os.close(current); current = child
                metadata = os.fstat(current)
                previous = self.owned.setdefault(relative, metadata)
                require(_file_identity(previous) == _file_identity(metadata), 'Verified scratch ancestor changed')
        finally: os.close(current)

    def record(self, name, info):
        self.owned[name] = info

    def verify(self):
        named = directory(str(self.parent_path))
        try: require(_file_identity(os.fstat(named)) == _file_identity(self.parent_identity), 'Verified scratch parent changed')
        finally: os.close(named)
        require(_file_identity(os.stat(self.name, dir_fd=self.parent, follow_symlinks=False)) == _file_identity(self.identity),
                'Verified scratch directory changed')

    def _remove_owned(self):
        # Only entries actually published by this capture may be removed. Walking
        # held descriptors also cleans owned bytes when the visible root is replaced.
        for name, expected in sorted(self.owned.items(), key=lambda item: item[0].count('/'), reverse=True):
            current = os.dup(self.fd)
            try:
                parts = path_parts(name)
                for index, part in enumerate(parts[:-1]):
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
                    os.close(current); current = child
                    ancestor = self.owned.get('/'.join(parts[:index + 1]))
                    if ancestor is None or _file_identity(os.fstat(current)) != _file_identity(ancestor): break
                else:
                    visible = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
                    if _file_identity(visible) == _file_identity(expected):
                        if stat.S_ISDIR(expected.st_mode): os.rmdir(parts[-1], dir_fd=current)
                        else: os.unlink(parts[-1], dir_fd=current)
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.ENOTEMPTY, errno.EEXIST): raise
            finally: os.close(current)
        try:
            visible = os.stat(self.name, dir_fd=self.parent, follow_symlinks=False)
            if _file_identity(visible) == _file_identity(self.identity): os.rmdir(self.name, dir_fd=self.parent)
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST): raise

    def close(self):
        try: self._remove_owned()
        finally:
            os.close(self.fd); os.close(self.parent)


@contextmanager
def _verified_scratch(parent):
    scratch = _VerifiedScratch(parent)
    try: yield scratch
    finally: scratch.close()


@contextmanager
def verified_request(request_path, expected_sha256, *, check_cancelled=lambda: None, storage_budget=None, scratch_parent=None):
    check_cancelled(); hash_value(expected_sha256)
    path = Path(os.path.abspath(os.fspath(request_path)))
    require(path.name == 'request.json', 'Evaluation request must be named request.json')
    root = directory(str(path.parent))
    try:
        root_identity = os.fstat(root)
        raw = read_regular(root, 'request.json', 256000)
        require(sha(raw) == expected_sha256, 'Evaluation request changed')
        request = object_json(raw)
        validate_versioned_request(request, REQUEST_FIELDS | ({'retainComponentOutputs'} if 'retainComponentOutputs' in request else set()))
        require('retainComponentOutputs' not in request or request['retainComponentOutputs'] is False,'Invalid evaluation retention override')
        require(request['kind'] == 'frozenEvaluation', 'Unsupported evaluation request')
        for key in ('evaluationID','projectID'): identity(request[key])
        for key in ('requestIdentity','snapshotReceiptSHA256'): hash_value(request[key])
        settings(request['settings'])
        receipt_raw = read_nested(root, 'input/snapshot.json', 256000)
        require(sha(receipt_raw) == request['snapshotReceiptSHA256'], 'Frozen snapshot receipt changed')
        receipt = object_json(receipt_raw)
        role=request['settings'].get('partitionRole','test')
        require(type(receipt.get('schemaVersion')) is int and
                (role=='test' and set(receipt)=={'schemaVersion','reference','testEndpoint','files'} and receipt['schemaVersion']==1 or
                 role=='validation' and set(receipt)=={'schemaVersion','partitionRole','reference','testEndpoint','files'} and receipt['schemaVersion']==2 and receipt['partitionRole']=='validation'), 'Unsupported frozen snapshot receipt')
        parent = receipt['reference']; endpoint = receipt['testEndpoint']; entries = receipt['files']
        require(isinstance(parent, dict) and set(parent) == REFERENCE_FIELDS, 'Invalid frozen parent fields')
        for key in ('projectID','directoryID','runID'): identity(parent[key])
        for key in REFERENCE_FIELDS - {'projectID','directoryID','runID'}: hash_value(parent[key])
        require(parent['projectID'] == request['projectID'], 'Frozen parent belongs to another project')
        require(isinstance(endpoint, dict) and set(endpoint) == {'node','port'} and
                all(isinstance(v, str) and 1 <= len(v) <= 200 and '\0' not in v for v in endpoint.values()), 'Invalid protected test endpoint')
        require(isinstance(entries, list) and 1 <= len(entries) <= 320, 'Invalid frozen inventory count')
        inventory = {}; total = 0
        for entry in entries:
            check_cancelled()
            require(isinstance(entry, dict) and set(entry) == {'path','bytes','sha256'}, 'Invalid inventory entry')
            name = entry['path']; maximum = limit(name); hash_value(entry['sha256'])
            require(name not in inventory and type(entry['bytes']) is int and 0 <= entry['bytes'] <= maximum, 'Invalid inventory bounds or duplicate')
            total += entry['bytes']; require(total <= 340000000, 'Frozen snapshot exceeds total budget')
            inventory[name] = entry
        require(set(LIMITS).issubset(inventory), 'Frozen snapshot is missing required files')
        with _verified_scratch(path.parent if scratch_parent is None else scratch_parent) as scratch:
            captured = scratch.path
            budget = storage_budget if storage_budget is not None else ArtifactStorageBudget(340_000_000, check_cancelled)
            with ArtifactStorage(scratch.fd, budget=budget, check_cancelled=check_cancelled) as storage:
                for name, entry in inventory.items():
                    check_cancelled()
                    data = read_nested(root, 'input/' + name, limit(name))
                    require(len(data) == entry['bytes'] and sha(data) == entry['sha256'], 'Frozen artifact changed: ' + name)
                    scratch.prepare(name)
                    scratch.record(name, storage.atomic_bytes(name, data, limit(name)))
            scratch.verify()
            pins = {'outcome.json':'outcomeSHA256','training/artifacts/manifest.json':'modelManifestSHA256',
                    'preparation/artifacts/manifest.json':'preparationManifestSHA256','training/artifacts/checkpoint.pt':'checkpointSHA256'}
            for name, key in pins.items(): require(inventory[name]['sha256'] == parent[key], 'Frozen parent pin differs from snapshot')
            outcome = object_json((captured / 'outcome.json').read_bytes())
            capture = object_json((captured / 'capture.json').read_bytes())
            require(outcome.get('status') == 'completed', 'Frozen parent outcome is not completed')
            for key in ('projectID','directoryID','runID','requestIdentity'):
                require(outcome.get(key) == parent[key] and capture.get(key) == parent[key], 'Frozen parent identity differs from captured archive')
            for key in ('modelManifestSHA256','preparationManifestSHA256'):
                require(outcome.get(key) == parent[key], 'Frozen outcome pin differs from selection')
            preparation = object_json((captured / 'preparation/artifacts/manifest.json').read_bytes())
            prep_files = preparation.get('artifacts')
            require(isinstance(prep_files, list), 'Preparation inventory missing')
            declared = {'preparation/artifacts/' + value['path'] for value in prep_files}
            require(set(inventory) == set(LIMITS) | declared, 'Snapshot inventory differs from preparation artifacts')
            frozen = load_frozen(captured / 'training/artifacts/checkpoint.pt',
                expected_manifest_sha256=parent['modelManifestSHA256'], expected_checkpoint_sha256=parent['checkpointSHA256'],
                registry_path=captured / 'model-registry.json', expected_registry_sha256=inventory['model-registry.json']['sha256'])
            (x, y), provenance, _ = load_evaluation_partition(captured / 'preparation/artifacts/manifest.json',
                endpoint['node'], endpoint['port'], frozen.training_provenance, partition_role=role,
                expected_manifest_sha256=parent['preparationManifestSHA256'],
                input_dtype=frozen.input_contract.get('dtype','float32'), sample_shape=frozen.input_contract.get('sampleShape'),
                named_plan=frozen.input_contract.get('namedInputPlan'),model_graph=frozen.archived_graph)
            scratch.verify()
            yield dict(root=root, root_identity=root_identity, path=path, request=request, receipt=receipt, captured=captured, inventory=inventory, frozen=frozen, x=x, y=y, provenance=provenance)
    finally: os.close(root)


def execute_request(request_path, expected_sha256, emit, *, check_cancelled=lambda: None):
    output = None; storage = None; manifest_identity = None
    budget = ArtifactStorageBudget(EVALUATION_STORAGE_BYTES, check_cancelled)
    try:
        with verified_request(request_path, expected_sha256, check_cancelled=check_cancelled, storage_budget=budget) as verified:
            root=verified["root"]; root_identity=verified["root_identity"]; path=verified["path"]
            request=verified["request"]; receipt=verified["receipt"]; frozen=verified["frozen"]
            x=verified["x"]; y=verified["y"]; provenance=verified["provenance"]
            parent=receipt["reference"]; endpoint=receipt["testEndpoint"]
            check_cancelled()
            os.mkdir('artifacts', mode=0o700, dir_fd=root)
            output = os.open('artifacts', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
            storage = ArtifactStorage(output, budget=budget, check_cancelled=check_cancelled)
            event_identity = {key:request[key] for key in ('evaluationID','requestIdentity')}
            count=provenance[request['settings'].get('partitionRole','test')+'Samples']
            emit(dict(event='evaluationStarted', **event_identity, totalCount=count))
            last_progress = -1
            progress_step = max(1, (count + 999) // 1000)
            def progress(completed, count):
                nonlocal last_progress
                check_cancelled()
                if completed == 0 or completed == count or completed - last_progress >= progress_step:
                    emit(dict(event='evaluationProgress', **event_identity, completedCount=completed, totalCount=count))
                    last_progress = completed
                check_cancelled()
            context=frozen.training_provenance.get('objective')
            evaluator=evaluate_frozen_objective if context else evaluate_classifier
            result = evaluator(frozen.model, x, y, provenance[request['settings'].get('partitionRole','test')+'RowIDs'],
                batch_size=request['settings']['batchSize'], device=request['settings']['device'],
                preview_limit=0 if request.get('retainComponentOutputs') is False else request['settings']['previewLimit'], on_batch=progress,
                **(dict(recipe=context['recipe'],scalarization=context['scalarization']) if context else {}))
            def publish(name, chunks, maximum):
                nonlocal manifest_identity
                digest = hashlib.sha256(); count = 0
                def write(stream):
                    nonlocal count
                    for chunk in chunks:
                        check_cancelled()
                        stream.write(chunk); count += len(chunk); digest.update(chunk)
                info = storage.atomic_write(name, maximum, write)
                if name == 'manifest.json': manifest_identity = (info.st_dev, info.st_ino)
                return dict(path=name, bytes=count, sha256=digest.hexdigest())
            predictions = publish('predictions.jsonl', (encoded(row) + b'\n' for row in result['predictions']), 100000000)
            summary = publish('summary.json', [encoded({key:value for key,value in result.items() if key != 'predictions'})], 1000000)
            if __package__:from .prediction_table import prediction_table
            else:from prediction_table import prediction_table
            if context:
                values=torch.tensor([r['output'] for r in result['predictions']],dtype=torch.float32)
                targets=y
            else:
                values=torch.tensor([r['logits'] for r in result['predictions']],dtype=torch.float64)
                targets=torch.tensor([r['target'] for r in result['predictions']])
            table=prediction_table(values,targets,[r['observationID'] for r in result['predictions']],
                request['settings'].get('partitionRole','test'),expected_sha256,classification=context is None,check_cancelled=check_cancelled)
            prediction_table_file=publish('prediction-table.json',[encoded(table)],50000000)
            manifest = dict(schemaVersion=4 if context else 3, kind='frozenEvaluation', status='completed',
                **{key:request[key] for key in ('evaluationID','projectID','requestIdentity','snapshotReceiptSHA256','settings')},
                requestSHA256=expected_sha256, parent=parent, testEndpoint=endpoint,
                testProvenance={key:value for key,value in provenance.items() if key != request['settings'].get('partitionRole','test')+'RowIDs'},
                frozenLineage=frozen.lineage, runtime=dict(python=platform.python_version(), torch=str(torch.__version__), device=result['device']),
                artifacts=[summary, predictions, prediction_table_file])
            check_cancelled()
            # Recheck the named directory before committing the final receipt.
            named = directory(str(path.parent))
            try:
                info = os.fstat(named)
                require((info.st_dev, info.st_ino) == (root_identity.st_dev, root_identity.st_ino), 'Evaluation directory changed')
            finally: os.close(named)
            visible_output = os.stat('artifacts', dir_fd=root, follow_symlinks=False)
            held_output = os.fstat(output)
            require((visible_output.st_dev, visible_output.st_ino) == (held_output.st_dev, held_output.st_ino), 'Evaluation output directory changed')
            committed = publish('manifest.json', [encoded(manifest)], 2000000)
            emit(dict(event='evaluationCompleted', **event_identity, manifestSHA256=committed['sha256'], evaluatedCount=result['evaluatedCount']))
            return manifest
    except BaseException:
        if output is not None and manifest_identity is not None:
            try:
                info = os.stat('manifest.json', dir_fd=output, follow_symlinks=False)
                if (info.st_dev, info.st_ino) == manifest_identity: os.unlink('manifest.json', dir_fd=output)
            except FileNotFoundError: pass
        raise
    finally:
        if storage is not None: storage.close()
        if output is not None: os.close(output)
