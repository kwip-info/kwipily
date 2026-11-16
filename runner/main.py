#!/usr/bin/env python3
"""One JSON event per stdout line. Diagnostics never masquerade as success."""
import argparse
import json
from pathlib import Path
import signal
import sys

class Cancelled(Exception): pass
class JSONArgumentParser(argparse.ArgumentParser):
    def error(self,message): raise ValueError(message)

def emit(event): print(json.dumps(event,allow_nan=False),flush=True)
def cancel(signum,frame): raise Cancelled('Run cancelled.')

def main(check_cancelled=lambda:None):
    flags={value.split('=',1)[0] for value in sys.argv[1:] if value.startswith('--')}
    if '--validate-mixture' in flags:
        return _mixture_validation_main(check_cancelled)
    if '--self-play' in flags:
        return _self_play_main(check_cancelled)
    if '--game-session' in flags:
        return _game_session_main(check_cancelled)
    if '--model-asset' in flags:
        return _model_asset_main(check_cancelled)
    if '--generate-text' in flags:
        return _generation_main(check_cancelled)
    if '--file-options' in flags:
        return _file_inspection_main(check_cancelled)
    if '--query-table-view' in flags:
        return _table_view_main(check_cancelled)
    if flags & {'--export-workflow-code','--workflow-code-plan'}:
        return _workflow_code_main(check_cancelled)
    # Spill files stay with the retained managed operation and its disk admission.
    from dataflow_disk import disk_scope
    authority=getattr(check_cancelled,'__self__',None)
    from evaluation_authorization import EvaluationAuthority
    directory=authority._path if isinstance(authority,EvaluationAuthority) else None
    with disk_scope(directory=directory,check_cancelled=check_cancelled):
        return _main(check_cancelled)

def _file_inspection_main(check_cancelled):
    """File collection inspection needs no optional model imports or live folders."""
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        arguments=sys.argv[1:]
        if len(arguments)!=4 or arguments[0]!='--inspect-data' or arguments[2]!='--file-options':
            raise ValueError('File Input inspection requires its exact source/options arguments.')
        if __package__:
            from .source_inspection_operation_contract import parse_file_options_json
            from .file_input_source import preview_files
        else:
            from source_inspection_operation_contract import parse_file_options_json
            from file_input_source import preview_files
        result=preview_files(arguments[1],parse_file_options_json(arguments[3]),check_cancelled)
        check_cancelled();emit(result);return 0
    except (Exception,KeyboardInterrupt) as error:
        emit({'event':'error','message':str(error) or 'File Input inspection interrupted.'})
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1

def _table_view_main(check_cancelled):
    """Pure managed query entry: no ML imports and no caller-owned source paths."""
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        if __package__:
            from .operation_authorization import OperationAuthority
            from .table_view import run_query
        else:
            from operation_authorization import OperationAuthority
            from table_view import run_query
        authority=getattr(check_cancelled,'__self__',None)
        arguments=sys.argv[1:]
        if (not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='tableViewQuery'
            or len(arguments)!=2 or arguments[0]!='--query-table-view' or authority.request['arguments']!=arguments):
            raise ValueError('Table View requires its exact managed query authority.')
        check_cancelled()
        run_query(arguments[1],emit,check_cancelled=check_cancelled,scratch_directory=authority._path)
        check_cancelled()
        return 0
    except (Exception,KeyboardInterrupt) as error:
        emit({'event':'error','message':str(error) or 'Table View query interrupted.'})
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1


def _workflow_code_main(check_cancelled):
    """Generate source from one captured plan without granting data-file access."""
    signal.signal(signal.SIGTERM,cancel); signal.signal(signal.SIGINT,cancel)
    try:
        if __package__:
            from .operation_authorization import OperationAuthority
            from .dataflow_storage import bounded_regular_read
        else:
            from operation_authorization import OperationAuthority
            from dataflow_storage import bounded_regular_read
        import hashlib
        authority=getattr(check_cancelled,'__self__',None)
        arguments=sys.argv[1:]
        if (not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='codeExport'
            or len(arguments)!=3 or arguments[:2]!=['--export-workflow-code','--workflow-code-plan']
            or authority.request['arguments']!=arguments):
            raise ValueError('Workflow code export requires its exact managed snapshot authority.')
        check_cancelled()
        raw=bounded_regular_read(arguments[2],2_000_000)
        entry=authority.request['inputFiles'][0]
        if (entry['path']!='input/workflow-code.json' or len(raw)!=entry['bytes']
            or hashlib.sha256(raw).hexdigest()!=entry['sha256']):
            raise ValueError('Workflow code snapshot changed before generation.')
        def unique(pairs):
            result={}
            for key,value in pairs:
                if key in result:raise ValueError('Duplicate workflow code snapshot field.')
                result[key]=value
            return result
        def constant(value):raise ValueError('Nonfinite workflow code snapshot value.')
        snapshot=json.loads(raw,object_pairs_hook=unique,parse_constant=constant)
        check_cancelled()
        if __package__:from .workflow_code import generate_package
        else:from workflow_code import generate_package
        package=generate_package(snapshot,check_cancelled=check_cancelled)
        check_cancelled()
        if (not isinstance(package,dict) or not isinstance(package.get('code'),str)
            or not isinstance(package.get('files'),dict)
            or any(not isinstance(name,str) or not isinstance(text,str) for name,text in package['files'].items())):
            raise ValueError('Workflow code generator returned an invalid source package.')
        event=dict(package,event='codeExport')
        if len(json.dumps(event,allow_nan=False).encode('utf-8'))>1_000_000:
            raise ValueError('Generated workflow source package exceeds its display byte budget.')
        check_cancelled();emit(event)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        emit({'event':'error','message':str(error) or 'Workflow code generation interrupted.'})
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1


def _main(check_cancelled=lambda:None):
    from engine import run, inspect_data, capabilities, GraphModel, export_source, validate
    from preview import preview
    from worker_resources import configure_torch
    import torch
    configure_torch(torch)
    signal.signal(signal.SIGTERM,cancel); signal.signal(signal.SIGINT,cancel)
    parser=JSONArgumentParser(allow_abbrev=False)
    parser.add_argument('--export-program'); parser.add_argument('--program-request-sha256')
    parser.add_argument('--export-evaluation'); parser.add_argument('--export-request-sha256')
    parser.add_argument('--evaluate-frozen'); parser.add_argument('--evaluation-request-sha256')
    parser.add_argument('--fit-backend'); parser.add_argument('--backend-request-sha256')
    parser.add_argument('--export-workflow-code',action='store_true'); parser.add_argument('--workflow-code-plan')
    parser.add_argument('--graph'); parser.add_argument('--output')
    parser.add_argument('--sample-capture-json')
    parser.add_argument('--training-progress-json')
    parser.add_argument('--training-control-json')
    parser.add_argument('--optimizer-recipe-json')
    parser.add_argument('--scheduler-recipe-json')
    parser.add_argument('--dataset-recipe-json')
    parser.add_argument('--loader-recipe-json')
    parser.add_argument('--initialization-recipe-json')
    parser.add_argument('--gradient-policy-json')
    parser.add_argument('--trainability-json')
    parser.add_argument('--derivative-capture-json')
    parser.add_argument('--objective-recipe-json')
    parser.add_argument('--named-input-plan-json')
    parser.add_argument('--objective-target-mapping-json')
    parser.add_argument('--objective-scalarization')
    parser.add_argument('--bag-input-mapping-json')
    parser.add_argument('--capture-gradients',action='store_true')
    parser.add_argument('--gradient-every-epochs',type=int); parser.add_argument('--gradient-max-batches-per-epoch',type=int); parser.add_argument('--gradient-max-captures',type=int)
    parser.add_argument('--epochs',type=int,default=100); parser.add_argument('--seed',type=int,default=42); parser.add_argument('--registry')
    parser.add_argument('--batch-size',type=int,default=1024); parser.add_argument('--sample-limit',type=int,default=0); parser.add_argument('--split-seed',type=int,default=42); parser.add_argument('--validation-percent',type=int,default=20); parser.add_argument('--learning-rate',type=float,default=.03); parser.add_argument('--export-code',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--dataset-csv'); parser.add_argument('--target-column')
    columns=parser.add_mutually_exclusive_group(); columns.add_argument('--feature-columns'); columns.add_argument('--feature-columns-json')
    parser.add_argument('--preview',action='store_true'); parser.add_argument('--stop-at'); parser.add_argument('--resume-from'); parser.add_argument('--cache'); parser.add_argument('--checkpoint')
    continuation=parser.add_mutually_exclusive_group(); continuation.add_argument('--resume-checkpoint'); continuation.add_argument('--finetune-checkpoint')
    parser.add_argument('--prepared-manifest-sha256'); parser.add_argument('--prepared-graph-digest'); parser.add_argument('--prepared-dataflow'); parser.add_argument('--prepared-training-node'); parser.add_argument('--prepared-training-port'); parser.add_argument('--prepared-validation-node'); parser.add_argument('--prepared-validation-port')
    parser.add_argument('--inspect-dataflow'); parser.add_argument('--dataflow-manifest-sha256')
    parser.add_argument('--dataflow-live-previews',choices=('true','false')); parser.add_argument('--dataflow-measurements',choices=('true','false')); parser.add_argument('--dataflow-required-nodes'); parser.add_argument('--dataflow-model-input-nodes'); parser.add_argument('--dataflow-capture-outputs',choices=('true','false')); parser.add_argument('--dataflow-node-labels'); parser.add_argument('--execute-dataflow'); parser.add_argument('--source-bindings'); parser.add_argument('--destination-bindings'); parser.add_argument('--defer-database-writes',choices=('true',)); parser.add_argument('--dataflow-registry',action='store_true')
    parser.add_argument('--retain-partial-dataflow',action='store_true')
    parser.add_argument('--dataflow-sampling');parser.add_argument('--dataflow-complete-nodes')
    parser.add_argument('--composite-source'); parser.add_argument('--composite-source-sha256')
    parser.add_argument('--learning-evidence-request'); parser.add_argument('--learning-submission'); parser.add_argument('--learning-submission-sha256'); parser.add_argument('--learning-runtime',action='store_true')
    parser.add_argument('--table-options'); parser.add_argument('--csv-options'); parser.add_argument('--csv-separator'); parser.add_argument('--inspect-data'); parser.add_argument('--capabilities',action='store_true')
    from dataflow_cli import add_cache_arguments,cache_arguments
    add_cache_arguments(parser)
    try:
        args=parser.parse_args()
        if args.dataflow_sampling is not None or args.dataflow_complete_nodes is not None:
            incompatible = ('--capabilities','--inspect-dataflow','--inspect-data','--dataflow-registry',
                '--preview','--export-code','--export-program','--fit-backend','--evaluate-frozen',
                '--export-evaluation','--learning-runtime','--learning-evidence-request','--resume-from')
            modes = {item.split('=',1)[0] for item in sys.argv[1:] if item.startswith('--')}
            if not args.execute_dataflow or modes.intersection(incompatible):
                raise ValueError('Preparation sampling and complete-node demand require only --execute-dataflow mode.')
        if args.export_program or args.program_request_sha256:
            authority=getattr(check_cancelled,'__self__',None)
            from operation_authorization import OperationAuthority
            if not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='programExport':
                raise ValueError('Program export requires its managed operation authority.')
            allowed={'--export-program','--program-request-sha256'}
            if args.registry is not None:allowed.add('--registry')
            flags=[arg.split('=',1)[0] for arg in sys.argv[1:] if arg.startswith('--')]
            if set(flags)!=allowed or len(flags)!=len(set(flags)) or not all((args.export_program,args.program_request_sha256)):
                raise ValueError('Program export requires only its retained request and SHA256 pin.')
            from program_export_cli import execute_export
            def program_event(value):
                value=dict(value);name=value.pop('type')
                emit(dict(value,event=name))
            execute_export(args.export_program,args.program_request_sha256,program_event,check_cancelled=check_cancelled)
            return 0
        if args.training_progress_json is not None:
            flags=[part.split('=',1)[0] for part in sys.argv[1:] if part.startswith('--')]
            if flags.count('--training-progress-json')!=1 or set(flags)&{'--preview','--capabilities','--learning-runtime','--learning-evidence-request','--export-code','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint'}:
                raise ValueError('Training reporting applies only to model training.')
        objective_recipe=None; target_mapping=None; objective_scalarization='mean'; objective_context=None
        objective_flags={'--objective-recipe-json','--objective-target-mapping-json','--objective-scalarization'}
        explicit_flags=[part.split('=',1)[0] for part in sys.argv[1:] if part.startswith('--')]
        if args.named_input_plan_json is not None:
            if explicit_flags.count('--named-input-plan-json')!=1 or len(args.named_input_plan_json.encode())>65536:raise ValueError('Provide one bounded named input mapping.')
            if not args.prepared_dataflow or set(explicit_flags)&{'--preview','--export-code','--capabilities','--learning-runtime','--learning-evidence-request','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint','--bag-input-mapping-json','--dataset-csv'}:raise ValueError('Named input mapping configures verified prepared training only.')
            from named_inputs import validate_plan
            validate_plan(json.loads(args.named_input_plan_json))
        if args.bag_input_mapping_json is not None:
            if explicit_flags.count('--bag-input-mapping-json') != 1:raise ValueError('Bag input mapping cannot be repeated.')
            if set(explicit_flags)&{'--preview','--export-code','--capabilities','--learning-runtime','--learning-evidence-request','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint'}:raise ValueError('Bag input mapping configures prepared training only.')
            if not args.prepared_dataflow:raise ValueError('Bag input mapping requires prepared training partitions.')
            if set(explicit_flags)&{'--feature-columns','--feature-columns-json'}:raise ValueError('Bag input mapping replaces ordinary feature column controls.')
            if set(explicit_flags)&{'--dataset-csv','--csv-separator'}:raise ValueError('Bag input mapping cannot combine CSV controls.')
            if len(args.bag_input_mapping_json.encode('utf-8'))>32768:raise ValueError('Bag mapping exceeds32768 UTF-8 bytes.')
        if objective_flags.intersection(explicit_flags):
            if any(explicit_flags.count(flag)>1 for flag in objective_flags):raise ValueError('Objective controls cannot be repeated.')
            if not args.objective_recipe_json or not args.objective_target_mapping_json:raise ValueError('Explicit objectives require both their recipe and typed target mapping.')
            if set(explicit_flags) & {'--preview','--export-code','--capabilities','--learning-runtime','--learning-evidence-request','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint'}:raise ValueError('Objective recipes configure training only.')
            if len(args.objective_recipe_json.encode('utf-8'))>32768:raise ValueError('Objective recipe exceeds32768bytes.')
            if len(args.objective_target_mapping_json.encode('utf-8'))>524288:raise ValueError('Objective target mapping exceeds524288bytes.')
            from loss_configuration import validate_recipe as validate_objective_recipe
            from loss_prepared_data import validate_target_mapping
            objective_recipe=validate_objective_recipe(json.loads(args.objective_recipe_json))
            target_mapping=validate_target_mapping(json.loads(args.objective_target_mapping_json),objective_recipe)
            objective_scalarization=args.objective_scalarization if args.objective_scalarization is not None else 'mean'
            if objective_scalarization not in ('mean','sum'):raise ValueError('Objective scalarization must be mean or sum.')
            if not args.prepared_dataflow:raise ValueError('Explicit objectives require prepared typed target partitions.')
            if set(explicit_flags)&{'--feature-columns','--feature-columns-json','--target-column','--dataset-csv'}:raise ValueError('Explicit objectives use their typed target mapping without duplicate legacy column controls.')
            objective_context={'recipe':objective_recipe,'targetMapping':target_mapping}
        gradient_policy=None;trainability=None;derivative_capture=None
        policy_flags=[part.split('=',1)[0] for part in sys.argv[1:] if part.startswith('--')]
        selected_options={}
        for flag,attribute,limit in [('--gradient-policy-json','gradient_policy_json',32768),('--trainability-json','trainability_json',32768),('--derivative-capture-json','derivative_capture_json',256)]:
            text=getattr(args,attribute)
            if text is None:continue
            if policy_flags.count(flag)!=1:raise ValueError('Gradient/trainability controls cannot be repeated.')
            if set(policy_flags)&{'--preview','--export-code','--capabilities','--learning-runtime','--learning-evidence-request','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint'}:raise ValueError('Gradient/trainability controls configure training only.')
            if len(text.encode('utf8'))>limit:raise ValueError('Gradient/trainability control exceeds portable byte limit.')
            selected_options[flag]=text
        if selected_options:
            from worker_authorization import validate_gradient_arguments
            validate_gradient_arguments({**selected_options,**({'--objective-recipe-json':args.objective_recipe_json,'--objective-scalarization':objective_scalarization} if objective_context is not None else {})})
            from gradient_policy import validate_recipe as validate_gradient_policy,validate_capture as validate_derivative_capture
            from trainability import validate_recipe as validate_trainability
            if args.gradient_policy_json is not None:gradient_policy=validate_gradient_policy(json.loads(args.gradient_policy_json))
            if args.trainability_json is not None:trainability=validate_trainability(json.loads(args.trainability_json))
            if args.derivative_capture_json is not None:derivative_capture=validate_derivative_capture(json.loads(args.derivative_capture_json))
        dataset_recipe=None;loader_recipe=None
        loading_options={}
        for flag,attribute in (('--dataset-recipe-json','dataset_recipe_json'),('--loader-recipe-json','loader_recipe_json')):
            text=getattr(args,attribute)
            if text is None:continue
            if explicit_flags.count(flag)!=1:raise ValueError('Dataset/DataLoader recipes cannot be repeated.')
            if set(explicit_flags)&{'--preview','--export-code','--capabilities','--learning-runtime','--learning-evidence-request','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint'}:raise ValueError('Dataset/DataLoader recipes configure training only.')
            loading_options[flag]=text
        if loading_options:
            from worker_authorization import validate_loading_arguments,object_json
            validate_loading_arguments({**loading_options,'--batch-size':str(args.batch_size)})
            from dataset_loader_configuration import validate_recipe as loading_recipe
            if args.dataset_recipe_json is not None:dataset_recipe=loading_recipe(object_json(args.dataset_recipe_json))
            if args.loader_recipe_json is not None:loader_recipe=loading_recipe(object_json(args.loader_recipe_json))
        initialization_recipe=None
        if args.initialization_recipe_json is not None:
            initialization_flags=[part.split('=',1)[0] for part in sys.argv[1:] if part.startswith('--')]
            if initialization_flags.count('--initialization-recipe-json')!=1:raise ValueError('Initialization recipes cannot be repeated.')
            if set(initialization_flags)&{'--preview','--export-code','--capabilities','--learning-runtime','--learning-evidence-request','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint'}:raise ValueError('Initialization recipes configure training only.')
            if len(args.initialization_recipe_json.encode('utf-8'))>32768:raise ValueError('Initialization recipe exceeds32768bytes.')
            from initialization import validate_recipe as validate_initialization_recipe
            initialization_recipe=validate_initialization_recipe(json.loads(args.initialization_recipe_json))
        optimizer_recipe=None
        if args.optimizer_recipe_json is not None:
            if len(args.optimizer_recipe_json.encode('utf-8'))>8192:raise ValueError('Optimizer recipe exceeds8192bytes.')
            from optimizer_configuration import validate_recipe
            optimizer_recipe=validate_recipe(json.loads(args.optimizer_recipe_json))
            explicit_flags={part.split('=')[0] for part in sys.argv[1:] if part.startswith('--')}
            if explicit_flags & {'--preview','--export-code','--capabilities','--learning-runtime','--learning-evidence-request','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation'}:raise ValueError('Optimizer recipes configure training only.')
        scheduler_recipe=None
        if args.scheduler_recipe_json is not None:
            if optimizer_recipe is None:raise ValueError('Schedulers require an explicit optimizer recipe.')
            if len(args.scheduler_recipe_json.encode('utf-8'))>8192:raise ValueError('Scheduler recipe exceeds8192bytes.')
            from scheduler_configuration import validate_recipe as validate_scheduler_recipe
            scheduler_recipe=validate_scheduler_recipe(json.loads(args.scheduler_recipe_json))
        if args.fit_backend or args.backend_request_sha256:
            allowed={'--fit-backend','--backend-request-sha256','--output'}
            # The verified packaged bootstrap (including supervised re-entry)
            # appends its pinned registry to every worker command. It is runtime
            # context, not an extra authored backend setting.
            context=globals().get('_KWIP_RUNTIME_CONTEXT')
            if args.registry is not None and isinstance(context,dict) and context.get('registry') is not None:
                if Path(args.registry).resolve()==Path(context['registry']).resolve():allowed.add('--registry')
            flags=[arg.split('=',1)[0] for arg in sys.argv[1:] if arg.startswith('--')]
            if set(flags)!=allowed or len(flags)!=len(set(flags)) or not all((args.fit_backend,args.backend_request_sha256,args.output)):
                raise ValueError('Backend fitting requires only its request, SHA256 pin and new output directory.')
            from backend_fit_cli import execute_request
            execute_request(args.fit_backend,args.backend_request_sha256,args.output,emit,check_cancelled=check_cancelled)
            return 0
        if (args.dataflow_live_previews is not None or args.dataflow_capture_outputs is not None or args.dataflow_required_nodes is not None or args.dataflow_model_input_nodes is not None or args.dataflow_measurements is not None) and not args.execute_dataflow:raise ValueError('Output retention requires preparation execution.')
        if args.retain_partial_dataflow and (not args.execute_dataflow or any((
                args.graph,args.preview,args.inspect_dataflow,args.dataflow_registry,args.inspect_data,args.capabilities,
                args.learning_runtime,args.learning_evidence_request,args.learning_submission,args.learning_submission_sha256,
                args.evaluate_frozen,args.evaluation_request_sha256,args.export_code,args.resume_from,args.resume_checkpoint,args.finetune_checkpoint))):
            raise ValueError('--retain-partial-dataflow requires --execute-dataflow without another command.')
        if args.export_evaluation or args.export_request_sha256:
            allowed={'--export-evaluation','--export-request-sha256'}
            flags=[arg.split('=',1)[0] for arg in sys.argv[1:] if arg.startswith('--')]
            if any(flag not in allowed for flag in flags) or len(flags)!=len(set(flags)):
                raise ValueError('Portable export cannot be combined with other commands or repeated options.')
            if not args.export_evaluation or not args.export_request_sha256:
                raise ValueError('Portable export requires its request and SHA-256 pin.')
            from portable_export_cli import execute_export
            execute_export(args.export_evaluation,args.export_request_sha256,emit,check_cancelled=check_cancelled)
            return 0
        if args.evaluate_frozen or args.evaluation_request_sha256:
            allowed={'--evaluate-frozen','--evaluation-request-sha256','--registry'}
            flags=[arg.split('=',1)[0] for arg in sys.argv[1:] if arg.startswith('--')]
            if any(flag not in allowed for flag in flags) or len(flags)!=len(set(flags)):
                raise ValueError('Frozen evaluation cannot be combined with training, inspection or repeated options.')
            if not args.evaluate_frozen or not args.evaluation_request_sha256:
                raise ValueError('Frozen evaluation requires its request and SHA-256 pin.')
            from frozen_evaluation_cli import execute_request
            execute_request(args.evaluate_frozen,args.evaluation_request_sha256,emit)
            return 0
        training_control=None
        if args.training_control_json is not None:
            explicit_flags=[part.split('=',1)[0] for part in sys.argv[1:] if part.startswith('--')]
            if explicit_flags.count('--training-control-json')!=1 or set(explicit_flags)&{'--preview','--capabilities','--learning-runtime','--learning-evidence-request','--export-code','--execute-dataflow','--dataflow-registry','--resume-from','--inspect-dataflow','--inspect-data','--fit-backend','--evaluate-frozen','--export-evaluation','--stop-at','--cache','--checkpoint'}:raise ValueError('Training control policy applies only to training.')
            if len(args.training_control_json.encode())>1024:raise ValueError('Training control policy is oversized.')
            from training_control import validate_policy as validate_training_control
            training_control=validate_training_control(json.loads(args.training_control_json))
        training_progress=None
        if args.training_progress_json is not None:
            if args.graph is None or args.output is None or len(args.training_progress_json.encode())>256:
                raise ValueError('Training reporting requires a training graph and bounded policy.')
            from training_progress import validate_policy
            training_progress=validate_policy(json.loads(args.training_progress_json))
        sample_capture=None
        if args.sample_capture_json is not None:
            if any((args.preview,args.resume_from,args.export_code,args.execute_dataflow,args.dataflow_registry,args.inspect_data,args.inspect_dataflow,args.capabilities,args.learning_runtime,args.learning_evidence_request,args.learning_submission,args.learning_submission_sha256)):
                raise ValueError('Sample capture applies only to training runs.')
            if len(args.sample_capture_json.encode())>1024:raise ValueError('Sample capture policy exceeds its size bound.')
            from capture_policy import validate_training_sample_capture_policy
            sample_capture=validate_training_sample_capture_policy(json.loads(args.sample_capture_json))
        gradient_options=[args.gradient_every_epochs,args.gradient_max_batches_per_epoch,args.gradient_max_captures]
        gradient_capture=None
        if args.capture_gradients or any(value is not None for value in gradient_options):
            if not args.capture_gradients:raise ValueError('Gradient capture options require --capture-gradients.')
            if any((args.preview,args.resume_from,args.export_code,args.execute_dataflow,args.dataflow_registry,args.inspect_data,args.inspect_dataflow,args.capabilities,args.learning_runtime,args.learning_evidence_request,args.learning_submission,args.learning_submission_sha256)):raise ValueError('Gradient capture applies only to training runs.')
            from gradient_capture import DEFAULT,validate_config
            gradient_capture=validate_config({**DEFAULT,**{key:value for key,value in zip(('everyEpochs','maxBatchesPerEpoch','maxCaptures'),gradient_options) if value is not None}})
        if args.inspect_dataflow:
            allowed={'--inspect-dataflow','--dataflow-manifest-sha256','--registry'}
            if any(arg.split('=',1)[0] not in allowed for arg in sys.argv[1:] if arg.startswith('--')):raise ValueError('Dataflow inspection cannot be combined with other commands or execution options')
            from dataflow_cli import inspect_dataflow
            inspect_dataflow(args.inspect_dataflow,args.dataflow_manifest_sha256,emit,check_cancelled=check_cancelled);return 0
        if args.dataflow_manifest_sha256 is not None:raise ValueError('Dataflow manifest pin requires --inspect-dataflow')
        if args.destination_bindings and not args.execute_dataflow:raise ValueError('Destination bindings require --execute-dataflow.')
        if args.dataflow_node_labels and not args.execute_dataflow:raise ValueError('Dataflow node labels require --execute-dataflow.')
        if args.table_options is not None and (not args.inspect_data or args.csv_options is not None or args.csv_separator is not None):raise ValueError('--table-options requires source inspection and cannot combine CSV flags')
        if args.csv_options is not None and (not args.inspect_data or args.csv_separator is not None):raise ValueError('--csv-options requires source inspection and cannot combine --csv-separator')
        if args.csv_separator is not None and not args.inspect_data:raise ValueError('--csv-separator applies only to --inspect-data; use a preparation source for noncomma training data.')
        cache_options=cache_arguments(args,executing=bool(args.execute_dataflow))
        learning_flags=[args.learning_evidence_request,args.learning_submission,args.learning_submission_sha256]
        if any(learning_flags) or args.learning_runtime:
            allowed={'--learning-evidence-request','--learning-submission','--learning-submission-sha256','--learning-runtime','--registry'}
            if any(arg.startswith('--') and arg.split('=',1)[0] not in allowed for arg in sys.argv[1:]):raise ValueError('Learning evidence cannot be combined with other execution or inspection flags.')
            from learning_evidence import execute_request,runtime_versions
            if args.learning_runtime:
                if any(learning_flags):raise ValueError('Learning runtime discovery cannot execute a submission.')
                check_cancelled();versions=runtime_versions(args.registry);check_cancelled()
                emit({'event':'learningRuntime','versions':versions});return 0
            if not all(learning_flags):raise ValueError('Learning execution requires request, submission and its SHA-256 digest.')
            emit({'event':'learningEvidence','evidence':execute_request(*learning_flags,registry=args.registry,check_cancelled=check_cancelled)});return 0
        if bool(args.composite_source)!=bool(args.composite_source_sha256):raise ValueError('Composite source requires both snapshot path and SHA-256 pin.')
        if args.composite_source and (args.preview or args.resume_from or args.export_code or args.execute_dataflow or args.dataflow_registry or args.inspect_data or args.capabilities):raise ValueError('Composite source provenance applies only to training runs.')
        feature_columns=[v.strip() for v in args.feature_columns.split(',')] if args.feature_columns else None
        if args.feature_columns_json is not None:
            if len(args.feature_columns_json.encode('utf-8'))>64_000:raise ValueError('Feature column JSON exceeds size limit.')
            feature_columns=json.loads(args.feature_columns_json)
            if not isinstance(feature_columns,list) or not 1<=len(feature_columns)<=128 or any(not isinstance(v,str) or not v.strip() or len(v)>200 for v in feature_columns) or len(set(feature_columns))!=len(feature_columns):raise ValueError('Feature column JSON requires 1–128 distinct nonempty column names.')
            if not args.prepared_dataflow and len(feature_columns)!=2:raise ValueError('Feature column JSON requires two distinct nonempty column names for legacy CSV training.')
        if not args.prepared_dataflow and (args.prepared_manifest_sha256 or args.prepared_graph_digest):raise ValueError('Prepared integrity pins require --prepared-dataflow.')
        if (args.resume_checkpoint or args.finetune_checkpoint) and (args.preview or args.resume_from or args.checkpoint or args.export_code or args.execute_dataflow or args.dataflow_registry or args.inspect_data or args.capabilities):raise ValueError('Training continuation cannot be combined with inference, dataflow or inspection commands.')
        if args.dataflow_registry:
            from dataflow import REGISTRY
            emit({'event':'dataflowRegistry','registry':REGISTRY}); return 0
        if args.execute_dataflow:
            check_cancelled()
            from dataflow_cli import read_json,run_dataflow,read_node_labels
            from preparation_demand_configuration import parse_sampling_json,parse_complete_nodes_json
            if not args.source_bindings or not args.output:raise ValueError('Dataflow requires --source-bindings and --output.')
            if args.graph or args.preview or args.resume_from or args.dataset_csv or args.export_code:raise ValueError('Dataflow execution cannot be combined with model execution flags.')
            run_dataflow(args.execute_dataflow,read_json(args.source_bindings,1_000_000),args.output,emit,retain_outputs=args.dataflow_capture_outputs!='false',live_previews=None if args.dataflow_live_previews is None else args.dataflow_live_previews=='true',collect_measurements=args.dataflow_measurements=='true',model_input_nodes=parse_complete_nodes_json(args.dataflow_model_input_nodes) if args.dataflow_model_input_nodes is not None else None,required_nodes=parse_complete_nodes_json(args.dataflow_required_nodes) if args.dataflow_required_nodes is not None else None,sampling=parse_sampling_json(args.dataflow_sampling) if args.dataflow_sampling is not None else None,complete_nodes=parse_complete_nodes_json(args.dataflow_complete_nodes) if args.dataflow_complete_nodes is not None else None,node_labels=read_node_labels(args.dataflow_node_labels) if args.dataflow_node_labels else None,retain_partial=args.retain_partial_dataflow,check_cancelled=check_cancelled,destination_bindings=read_json(args.destination_bindings,1_000_000) if args.destination_bindings else None,defer_database_writes=args.defer_database_writes=='true',**cache_options); return 0
        if args.source_bindings:raise ValueError('Source bindings require --execute-dataflow.')
        if args.capabilities:
            check_cancelled();discovery=capabilities();check_cancelled()
            import importlib.util
            optional={};errors={}
            for backend,module in [('sklearn','sklearn_backend'),('tensorflow','tensorflow_foundation')]:
                if importlib.util.find_spec(backend) is not None:
                    try:
                        import importlib
                        optional[backend]=importlib.import_module(module).capabilities()
                    except Exception as error:errors[backend]=str(error)[:1000]
                check_cancelled()
            discovery['optionalBackends']=optional
            discovery['optionalBackendErrors']=errors
            emit(discovery);return 0
        if args.inspect_data:
            from csv_parser import parse_options_json
            from source_inspection_operation_contract import parse_table_options_json
            inspection=inspect_data(args.inspect_data,args.csv_separator if args.csv_separator is not None else ',',options=parse_options_json(args.csv_options) if args.csv_options is not None else None,table_options=parse_table_options_json(args.table_options) if args.table_options is not None else None,check_cancelled=check_cancelled)
            check_cancelled();emit(inspection);return 0
        if args.export_code:
            check_cancelled()
            if not args.graph: raise ValueError('Code export requires --graph.')
            path=Path(args.graph)
            if path.stat().st_size>2_000_000: raise ValueError('Graph exceeds 2 MB limit.')
            graph=json.loads(path.read_text())
            source=export_source(GraphModel(graph,args.registry,bundled_dataset=False));check_cancelled()
            emit({'event':'codeExport','code':source}); return 0
        if not args.graph or not args.output: raise ValueError('Training requires --graph and --output.')
        prepared=None
        prepared_fields=[args.prepared_training_node,args.prepared_training_port,args.prepared_validation_node,args.prepared_validation_port]
        if args.prepared_dataflow:
            if not all(prepared_fields) or (objective_context is None and ((not feature_columns and args.bag_input_mapping_json is None) or not args.target_column)):raise ValueError('Prepared data requires explicit training/validation ports and an objective mapping or feature/bag mapping with target columns.')
            if not args.prepared_manifest_sha256 or not args.prepared_graph_digest:raise ValueError('Prepared data requires manifest SHA-256 and graph digest pins.')
            explicit_flags={part.split('=')[0] for part in sys.argv[1:] if part.startswith('--')}
            if explicit_flags & {'--split-seed','--validation-percent','--sample-limit','--dataset-csv'}:raise ValueError('Prepared data is already partitioned; split/sample and CSV options cannot be combined.')
        elif any(prepared_fields):raise ValueError('Prepared node/port options require --prepared-dataflow.')
        if args.dataset_csv and (not feature_columns or not args.target_column): raise ValueError('CSV training requires --feature-columns and --target-column.')
        if not args.dataset_csv and not args.prepared_dataflow and (feature_columns or args.target_column): raise ValueError('Column mapping requires --dataset-csv.')
        check_cancelled()
        p=Path(args.graph)
        if p.stat().st_size>2_000_000: raise ValueError('Graph file exceeds 2 MB POC limit.')
        graph=json.loads(p.read_text())
        composite_source=None
        if args.composite_source:
            validate(graph,args.registry,bundled_dataset=not bool(args.prepared_dataflow),**({'objective_context':objective_context} if objective_context is not None else {}))
            from composite_source import load_source
            composite_source=load_source(args.composite_source,args.composite_source_sha256,graph,
                                         registry=args.registry,objective_context=objective_context)
        if args.named_input_plan_json is not None and not args.prepared_dataflow:raise ValueError('Named input mapping requires verified prepared tables.')
        if args.prepared_dataflow:
            _,_,layouts,dtypes=validate(graph,args.registry,include_types=True,bundled_dataset=False,**({'objective_context':objective_context} if objective_context is not None else {}))
            source_node=next(node for node in graph['nodes'] if node['type'] in ('input','index_input','bag_input'))
            source_id=source_node['id']
            from prepared_data import load_prepared
            check_cancelled()
            if args.named_input_plan_json is not None:
                if len(args.named_input_plan_json.encode())>65536 or args.bag_input_mapping_json is not None:raise ValueError('Named input plan exceeds its bound or conflicts with bag mapping.')
                from named_prepared_data import load_named_prepared
                prepared=load_named_prepared(args.prepared_dataflow,*prepared_fields,json.loads(args.named_input_plan_json),graph,expected_manifest_sha256=args.prepared_manifest_sha256,expected_graph_digest=args.prepared_graph_digest,objective_recipe=objective_recipe,target_mapping=target_mapping,registry=args.registry,check=check_cancelled)
            elif source_node['type']=='bag_input':
                if args.bag_input_mapping_json is None:raise ValueError('Prepared bag input requires an explicit bag input mapping.')
                from bag_input_mapping import decode_mapping_json,graph_admission
                from prepared_bag_loader import load_prepared_bags
                admission=graph_admission(graph,layouts)
                mapping=decode_mapping_json(args.bag_input_mapping_json,admission['source_config'],target_column=args.target_column,objective_mapping=target_mapping)
                output_id=next(node['id'] for node in graph['nodes'] if node['type']=='output')
                prepared=load_prepared_bags(args.prepared_dataflow,*prepared_fields,source_config=admission['source_config'],mapping=mapping,num_embeddings=admission['num_embeddings'],embedding_dim=admission['embedding_dim'],mode=admission['mode'],expected_manifest_sha256=args.prepared_manifest_sha256,expected_graph_digest=args.prepared_graph_digest,target_column=args.target_column,**({'objective_mapping':target_mapping,'objective_recipe':objective_recipe,'prediction_shape':layouts[output_id].sample} if objective_context is not None else {}))
            elif args.bag_input_mapping_json is not None:raise ValueError('Bag input mapping requires an actual bag source, not ordinary tensor input.')
            elif objective_context is not None:
                from loss_prepared_data import load_prepared_objective
                output_id=next(node['id'] for node in graph['nodes'] if node['type']=='output')
                prepared=load_prepared_objective(args.prepared_dataflow,*prepared_fields,target_mapping,objective_recipe,expected_manifest_sha256=args.prepared_manifest_sha256,expected_graph_digest=args.prepared_graph_digest,input_dtype=dtypes[source_id],sample_shape=layouts[source_id].sample,prediction_shape=layouts[output_id].sample)
            else:
                prepared=load_prepared(args.prepared_dataflow,*prepared_fields,feature_columns,args.target_column,expected_manifest_sha256=args.prepared_manifest_sha256,expected_graph_digest=args.prepared_graph_digest,input_dtype=dtypes[source_id],sample_shape=layouts[source_id].sample)
        if args.preview or args.resume_from:
            preview(graph,args.output,seed=args.seed,device=args.device,registry=args.registry,dataset_csv=args.dataset_csv,feature_columns=feature_columns,target_column=args.target_column,stop_at=args.stop_at,resume_from=args.resume_from,cache=args.cache,checkpoint=args.checkpoint,emit=emit,sample_limit=args.sample_limit,split_seed=args.split_seed,validation_percent=args.validation_percent,prepared=prepared,check_cancelled=check_cancelled)
            return 0
        if args.stop_at or args.cache or args.checkpoint: raise ValueError('Stop, cache and checkpoint flags apply to inference --preview or --resume-from only.')
        run(graph,args.output,args.epochs,args.seed,emit,args.registry,device=args.device,dataset_csv=args.dataset_csv,feature_columns=feature_columns,target_column=args.target_column,batch_size=args.batch_size,sample_limit=args.sample_limit,split_seed=args.split_seed,validation_percent=args.validation_percent,learning_rate=args.learning_rate,prepared=prepared,resume_checkpoint=args.resume_checkpoint,finetune_checkpoint=args.finetune_checkpoint,composite_source=composite_source,gradient_capture=gradient_capture,check_cancelled=check_cancelled,sample_capture=sample_capture,optimizer_recipe=optimizer_recipe,scheduler_recipe=scheduler_recipe,initialization_recipe=initialization_recipe,gradient_policy=gradient_policy,trainability=trainability,derivative_capture=derivative_capture,dataset_recipe=dataset_recipe,loader_recipe=loader_recipe,training_progress=training_progress,training_control=training_control,**({'objective_recipe':objective_recipe,'target_mapping':target_mapping,'objective_scalarization':objective_scalarization} if objective_context is not None else {}))
        return 0
    except (Exception,KeyboardInterrupt) as error:
        event={'event':'error','message':str(error) or 'Run interrupted.'}
        from csv_parser import CSVParseError
        if isinstance(error,CSVParseError):event['csvDiagnostic']=error.diagnostic()
        from dataflow import DataFlowExecutionError
        if isinstance(error,DataFlowExecutionError):event['dataflowError']=error.detail
        emit(event)
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1

def entry():
    # Isolated development re-entry cannot rely on the current directory/PYTHONPATH.
    runner = Path(__file__).resolve().parent
    sys.path.insert(0, str(runner))
    prepared = None
    if any(argument.startswith('--kwip-') for argument in sys.argv[1:]):
        try:
            from worker_bootstrap import RuntimeContext, PreparedWorker, prepare
            supplied = globals().get('_KWIP_RUNTIME_CONTEXT')
            if supplied is None:
                native = runner.parent / '.local/native-watchdog/lifetime_watchdog.dylib'
                runtime = RuntimeContext(Path(__file__).resolve(), native, None,
                    native_manifest=native.with_suffix('.json'), native_source=runner/'native/lifetime_watchdog.c')
            else:
                runtime = RuntimeContext(**supplied)
            result = prepare(sys.argv[1:], runtime)
            if isinstance(result, int): return result
            if not isinstance(result, PreparedWorker): raise ValueError('Missing worker authorization.')
            prepared = result
            sys.argv = [sys.argv[0], *prepared.arguments]
        except (Exception, KeyboardInterrupt) as error:
            emit({'event':'error', 'message':str(error) or 'Worker startup interrupted.'})
            return 1
    try:
        return main(check_cancelled=prepared.authority.check if prepared is not None else lambda:None)
    finally:
        if prepared is not None: prepared.authority.close()





def _model_asset_main(check_cancelled):
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        if __package__:
            from .operation_authorization import OperationAuthority
            from .model_asset_worker import execute
        else:
            from operation_authorization import OperationAuthority
            from model_asset_worker import execute
        authority=getattr(check_cancelled,'__self__',None)
        if not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='modelAsset' or authority.request['arguments']!=sys.argv[1:]:raise ValueError('Model import/export requires its retained managed authority.')
        execute(authority,emit,check_cancelled)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        emit(dict(event='error',message=str(error) or 'Model operation interrupted.'))
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1

def _generation_main(check_cancelled):
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        if __package__:
            from .operation_authorization import OperationAuthority
            from .text_generation_worker import execute
        else:
            from operation_authorization import OperationAuthority
            from text_generation_worker import execute
        authority=getattr(check_cancelled,'__self__',None)
        if not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='textGeneration' or authority.request['arguments']!=sys.argv[1:]:raise ValueError('Generation requires its retained managed authority.')
        if __package__:from .dataflow_disk import disk_scope
        else:from dataflow_disk import disk_scope
        with disk_scope(directory=authority._path,check_cancelled=check_cancelled):execute(authority,emit,check_cancelled)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        emit(dict(event='error',message=str(error) or 'Generation interrupted.'))
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1

def _game_session_main(check_cancelled):
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        if __package__:
            from .operation_authorization import OperationAuthority
            from .game_session_worker import execute
        else:
            from operation_authorization import OperationAuthority
            from game_session_worker import execute
        authority=getattr(check_cancelled,'__self__',None)
        if not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='gameSession' or authority.request['arguments']!=sys.argv[1:]:raise ValueError('Game session requires its retained managed authority.')
        execute(authority,emit,check_cancelled)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        emit(dict(event='error',message=str(error) or 'Game session interrupted.'))
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1

def _self_play_main(check_cancelled):
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        if __package__:
            from .operation_authorization import OperationAuthority
            from .self_play_worker import execute
        else:
            from operation_authorization import OperationAuthority
            from self_play_worker import execute
        authority=getattr(check_cancelled,'__self__',None)
        if not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='selfPlay' or authority.request['arguments']!=sys.argv[1:]:raise ValueError('Self-play requires its retained managed authority.')
        execute(authority,emit,check_cancelled)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        emit(dict(event='error',message=str(error) or 'Self-play interrupted.'))
        return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1




def _mixture_validation_main(check_cancelled):
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        if __package__:
            from .operation_authorization import OperationAuthority
            from .moe import preflight
            from .worker_resources import configure_torch
        else:
            from operation_authorization import OperationAuthority
            from moe import preflight
            from worker_resources import configure_torch
        import torch
        configure_torch(torch)
        authority=getattr(check_cancelled,'__self__',None)
        if not isinstance(authority,OperationAuthority) or authority.request.get('toolMode')!='mixtureValidation' or authority.request['arguments']!=sys.argv[1:]:raise ValueError('MoE validation requires its exact managed authority.')
        source=Path(authority._path)/'input/mixture.json'
        if sys.argv[1:]!=['--validate-mixture',str(source)]:raise ValueError('MoE validation input must be its owned captured file.')
        result=preflight(json.loads(source.read_bytes()),check_cancelled)
        check_cancelled();emit(dict(event='mixtureValidated',**result));return 0
    except (Exception,KeyboardInterrupt) as error:
        emit(dict(event='error',message=str(error) or 'MoE validation cancelled.'));return 130 if isinstance(error,(Cancelled,KeyboardInterrupt)) else 1

if __name__=='__main__': sys.exit(entry())
