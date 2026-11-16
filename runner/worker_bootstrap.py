"""Trusted entry routing. Establish worker authority before importing ML modules."""
from dataclasses import dataclass
from pathlib import Path
import ctypes
import hashlib
import json
import os
import stat
import sys
if __package__:
    from .worker_resources import apply_verified_resources,resident_limit
else:
    from worker_resources import apply_verified_resources,resident_limit


@dataclass(frozen=True)
class RuntimeContext:
    entry: Path
    native_library: Path
    registry: Path | None
    isolated: bool = True
    native_manifest: Path | None = None
    native_source: Path | None = None


@dataclass
class PreparedWorker:
    authority: object
    arguments: list[str]
    native_library: object


def _regular(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError('Native lifetime guard attestation requires bounded regular files.')
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError('Native lifetime guard attestation exceeds size limit.')
        return raw
    finally:
        if fd is not None: os.close(fd)


def _verify_native(runtime):
    if (runtime.native_manifest is None) != (runtime.native_source is None):
        raise ValueError('Native lifetime guard attestation requires both manifest and source.')
    try:
        library = _regular(runtime.native_library, 10_000_000)
        if runtime.native_manifest is not None:
            manifest = json.loads(_regular(runtime.native_manifest, 16384))
            if (not isinstance(manifest, dict) or set(manifest) != {'schemaVersion', 'sourceSHA256', 'librarySHA256'}
                    or type(manifest['schemaVersion']) is not int or manifest['schemaVersion'] != 1):
                raise ValueError('Invalid native lifetime guard attestation manifest.')
            source = _regular(runtime.native_source, 10_000_000)
            if (manifest['sourceSHA256'] != hashlib.sha256(source).hexdigest()
                    or manifest['librarySHA256'] != hashlib.sha256(library).hexdigest()):
                raise ValueError('Native lifetime guard attestation digest mismatch; rebuild the development app.')
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f'Native lifetime guard attestation failed: {error}') from error


def _arguments(intent, runtime):
    arguments = list(intent['arguments'])
    if intent.get('schemaVersion')==2 and intent.get('toolMode')=='modelTraining' and '--bag-input-mapping-json' in arguments:
        if runtime.registry is None:raise ValueError('Bag training requires the selected verified runtime registry.')
        if __package__:from .worker_authorization import object_json
        else:from worker_authorization import object_json
        if __package__:from .worker_input_contract import same_json
        else:from worker_input_contract import same_json
        graph=Path(arguments[arguments.index('--graph')+1])
        captured=graph.parent/'model-registry.json'
        # OperationAuthority has already checked this exact captured inventory.
        # Runtime files can be root-owned in a store install; their attested bytes,
        # rather than capture ownership rules, identify the selected runtime here.
        if not same_json(object_json(_regular(captured,2_000_000)),object_json(_regular(runtime.registry,2_000_000))):
            raise ValueError('Captured bag registry differs from the selected runtime registry.')
    if intent.get('schemaVersion')==2 and intent.get('toolMode')=='learningAssessment' and '--registry' in arguments:
        if __package__:from .operation_authorization import validate_tool_request
        else:from operation_authorization import validate_tool_request
        inventory=validate_tool_request(intent)
        expected=Path(arguments[1]).parent/'model-registry.json'
        if 'input/model-registry.json' not in inventory or arguments[7]!=str(expected):
            raise ValueError('Assessment registry must be its exact captured input.')
        # The retained OperationAuthority already verifies every input byte/hash and
        # anchors these paths before this routing call and throughout execution.
        return arguments
    # These pure routes consume only their already-admitted captured inventory.
    # Injecting a model registry changes their exact argument contract. Keep
    # generic model/code and all other registry-consuming routes unchanged.
    if intent.get('schemaVersion') == 2 and (
        (intent.get('toolMode') == 'mixtureValidation' and len(arguments) == 2 and arguments[0] == '--validate-mixture')
        or (intent.get('toolMode') == 'modelAsset' and len(arguments) == 2 and arguments[0] == '--model-asset')
        or (intent.get('toolMode') == 'selfPlay' and len(arguments) == 2 and arguments[0] == '--self-play')
        or (intent.get('toolMode') == 'gameSession' and len(arguments) == 2 and arguments[0] == '--game-session')
        or (intent.get('toolMode') == 'textGeneration' and len(arguments) == 2 and arguments[0] == '--generate-text')
        or (intent.get('toolMode') == 'tableViewQuery' and len(arguments) == 2 and arguments[0] == '--query-table-view')
        or (intent.get('toolMode') == 'sourceInspection' and len(arguments) == 4
            and arguments[0] == '--inspect-data' and arguments[2] == '--file-options')
        or (intent.get('toolMode') == 'codeExport' and len(arguments) == 3
            and arguments[:2] == ['--export-workflow-code','--workflow-code-plan'])
    ):
        return arguments
    if runtime.registry is not None:
        values = [arguments[index + 1] for index, argument in enumerate(arguments) if argument == '--registry']
        if len(values) > 1 or any(Path(value).resolve() != runtime.registry.resolve() for value in values):
            raise ValueError('Worker registry must be the verified runtime registry.')
        if not values: arguments += ['--registry', str(runtime.registry)]
    return arguments


def _arm_worker(authority, arguments, lifetime_fd, runtime, ready):
    """Shared guard boundary for both independent evaluation and group phases."""
    try:
        _verify_native(runtime)
        library = ctypes.PyDLL(str(runtime.native_library))
        arm = library.kwip_lifetime_arm
        arm.argtypes = [ctypes.c_int]
        arm.restype = ctypes.c_int
        result = arm(lifetime_fd)
        if result != 0: raise ValueError(f'Native lifetime guard could not arm (error {result}).')
        os.close(lifetime_fd)
        authority.check()
        envelope=authority.request if hasattr(authority,'request') else authority.intent
        apply_verified_resources(envelope)
        print(json.dumps(ready), flush=True)
        return PreparedWorker(authority, arguments, library)
    except BaseException:
        authority.close()
        raise


def _parent_revoked():
    import fcntl
    import selectors
    flags = fcntl.fcntl(0, fcntl.F_GETFL)
    if flags & os.O_ACCMODE != os.O_RDONLY or flags & os.O_NONBLOCK or not stat.S_ISFIFO(os.fstat(0).st_mode):
        raise ValueError('Evaluation parent lifetime requires a blocking read-only FIFO.')
    with selectors.DefaultSelector() as selector:
        selector.register(0, selectors.EVENT_READ)
        return bool(selector.select(timeout=0))


def prepare(argv, runtime):
    """Return None for ordinary CLI, supervisor exit code, or retained child state."""
    if not any(isinstance(arg, str) and arg.startswith('--kwip-') for arg in argv):
        return None
    modes = ('--kwip-supervise', '--kwip-worker', '--kwip-evaluate-supervise', '--kwip-evaluate-worker', '--kwip-export-supervise', '--kwip-export-worker', '--kwip-operation-supervise', '--kwip-operation-worker')
    if not argv or argv[0] not in modes:
        raise ValueError('Invalid internal worker control arguments.')
    evaluation = argv[0] in modes[2:4]
    exporting = argv[0] in modes[4:6]
    operation = argv[0] in modes[6:8]
    worker = argv[0] in ('--kwip-worker', '--kwip-evaluate-worker', '--kwip-export-worker', '--kwip-operation-worker')
    prefix = 4 if evaluation or exporting or operation else 5
    launch_budget_pin = None
    budget_options = []
    if (evaluation or exporting) and len(argv) > prefix and argv[prefix] == '--kwip-launch-budget-sha256':
        if len(argv) <= prefix + 1:raise ValueError('Missing launch budget digest.')
        launch_budget_pin = argv[prefix + 1]
        budget_options = argv[prefix:prefix + 2]
        prefix += 2
    if len(argv) != prefix + (4 if worker else 0):
        raise ValueError('Invalid internal worker control arity.')
    if worker:
        if argv[prefix] != '--kwip-lifetime-fd' or argv[prefix + 2] != '--kwip-activity-fd':
            raise ValueError('Invalid internal worker descriptor options.')
        if any(not value.isascii() or not value.isdecimal() for value in (argv[prefix + 1], argv[prefix + 3])):
            raise ValueError('Invalid internal worker descriptor values.')
        lifetime_fd, activity_fd = int(argv[prefix + 1]), int(argv[prefix + 3])
        if lifetime_fd <= 2 or activity_fd <= 2 or lifetime_fd == activity_fd:
            raise ValueError('Invalid internal worker descriptors.')
    # All authority and guard work precedes engine imports.
    if __package__:
        from .worker_supervisor import supervise
    else:
        from worker_supervisor import supervise
    if operation:
        if __package__:
            from .operation_authorization import claim_operation, inherit_operation
        else:
            from operation_authorization import claim_operation, inherit_operation
        folder, request_pin, ownership_pin = argv[1:4]
        if not worker:
            if _parent_revoked(): return 130
            with claim_operation(folder, request_pin, ownership_pin) as authority:
                _arguments(authority.request, runtime)
                command = [sys.executable] + (['-I'] if runtime.isolated else [])
                command += ['-B', str(runtime.entry), '--kwip-operation-worker', folder, request_pin, ownership_pin]
                return supervise(command, parent_fd=0, activity_fd=authority.activity_fd,
                                 max_wall_seconds=authority.request.get('remainingSeconds'),
                                 max_resident_bytes=resident_limit(authority.request))
        authority = inherit_operation(folder, request_pin, ownership_pin, activity_fd)
        try:
            arguments = _arguments(authority.request, runtime)
        except BaseException:
            authority.close(); raise
        ready = {'event':'operationReady', 'operationID':authority.request['operationID'],
                 'requestSHA256':request_pin, 'ownershipSHA256':ownership_pin}
        field = 'toolMode' if authority.request['schemaVersion']==2 else 'phase'
        ready[field] = authority.request[field]
        return _arm_worker(authority, arguments, lifetime_fd, runtime, ready)
    if exporting:
        if __package__:
            from .portable_export_authorization import claim_export, inherit_export
        else:
            from portable_export_authorization import claim_export, inherit_export
        folder, request_pin, ownership_pin = argv[1:4]
        arguments = ['--export-evaluation', str(Path(folder) / 'request.json'), '--export-request-sha256', request_pin]
        if not worker:
            if _parent_revoked(): return 130
            with claim_export(folder, request_pin, ownership_pin, launch_budget_sha256=launch_budget_pin) as authority:
                command = [sys.executable] + (['-I'] if runtime.isolated else [])
                command += ['-B', str(runtime.entry), '--kwip-export-worker', folder, request_pin, ownership_pin] + budget_options
                return supervise(command, parent_fd=0, activity_fd=authority.activity_fd, max_wall_seconds=authority.remaining_seconds,
                                 max_resident_bytes=resident_limit(authority.request))
        authority = inherit_export(folder, request_pin, ownership_pin, activity_fd, launch_budget_sha256=launch_budget_pin)
        return _arm_worker(authority, arguments, lifetime_fd, runtime,
            {'event':'exportReady','exportID':authority.request['exportID'],
             'requestSHA256':request_pin,'ownershipSHA256':ownership_pin})
    if evaluation:
        if __package__:
            from .evaluation_authorization import claim_evaluation, inherit_evaluation
        else:
            from evaluation_authorization import claim_evaluation, inherit_evaluation
        folder, request_pin, ownership_pin = argv[1:4]
        arguments = ['--evaluate-frozen', str(Path(folder) / 'request.json'), '--evaluation-request-sha256', request_pin]
        if runtime.registry is not None: arguments += ['--registry', str(runtime.registry)]
        if not worker:
            if _parent_revoked(): return 130
            with claim_evaluation(folder, request_pin, ownership_pin, launch_budget_sha256=launch_budget_pin) as authority:
                command = [sys.executable] + (['-I'] if runtime.isolated else [])
                command += ['-B', str(runtime.entry), '--kwip-evaluate-worker', folder, request_pin, ownership_pin] + budget_options
                return supervise(command, parent_fd=0, activity_fd=authority.activity_fd, max_wall_seconds=authority.remaining_seconds,
                                 max_resident_bytes=resident_limit(authority.request))
        authority = inherit_evaluation(folder, request_pin, ownership_pin, activity_fd, launch_budget_sha256=launch_budget_pin)
        ready = {'event':'evaluationReady', 'evaluationID':authority.request['evaluationID'],
                 'requestSHA256':request_pin, 'ownershipSHA256':ownership_pin}
        return _arm_worker(authority, arguments, lifetime_fd, runtime, ready)
    if __package__:
        from .worker_authorization import claim_launch, inherit_launch
    else:
        from worker_authorization import claim_launch, inherit_launch
    group, nonce, intent_pin, ownership_pin = argv[1:5]
    if not worker:
        with claim_launch(group, nonce, intent_pin, ownership_pin) as authority:
            _arguments(authority.intent, runtime)
            command = [sys.executable] + (['-I'] if runtime.isolated else [])
            command += ['-B', str(runtime.entry), '--kwip-worker', group, nonce, intent_pin, ownership_pin]
            return supervise(command, parent_fd=0, activity_fd=authority.activity_fd,
                             max_wall_seconds=authority.intent.get('remainingSeconds'),
                             max_resident_bytes=resident_limit(authority.intent))
    authority = inherit_launch(group, nonce, intent_pin, ownership_pin, activity_fd)
    try:
        arguments = _arguments(authority.intent, runtime)
    except BaseException:
        authority.close(); raise
    return _arm_worker(authority, arguments, lifetime_fd, runtime,
                       {'event':'launchReady', 'nonce':nonce, 'phase':authority.intent['phase']})
