"""Pinned ordinary workflow phases using the shared retained one-shot authority."""
import os
from pathlib import Path

if __package__:
    from .table_view import parse_contract as parse_table_view, verify_inputs as verify_table_view
    from .program_export_operation_contract import parse_contract as parse_program, verify_inputs as verify_program
    from .backend_fit_operation_contract import parse_contract as parse_backend, verify_inputs as verify_backend
    from .model_training_operation_contract import parse_contract as parse_training, verify_inputs as verify_training
    from .preparation_inspection_operation_contract import parse_contract as parse_saved_inspection, verify_inputs as verify_saved_inspection
    from .source_inspection_operation_contract import parse_contract as parse_inspection, verify_inputs as verify_inspection
    from .preparation_operation_contract import parse_contract as parse_preparation, verify_inputs as verify_preparation
    from .preview_operation_contract import parse_contract, verify_inputs
    from .execution_budget import validate_remaining_budget
    from .evaluation_authorization import EvaluationAuthority, OWNERSHIP_FIELDS, CLAIM_FIELDS
    from .worker_authorization import require, canonical_uuid, hash_value, validate_phase_arguments, directory, read_regular, sha
else:
    from table_view import parse_contract as parse_table_view, verify_inputs as verify_table_view
    from program_export_operation_contract import parse_contract as parse_program, verify_inputs as verify_program
    from backend_fit_operation_contract import parse_contract as parse_backend, verify_inputs as verify_backend
    from model_training_operation_contract import parse_contract as parse_training, verify_inputs as verify_training
    from preparation_inspection_operation_contract import parse_contract as parse_saved_inspection, verify_inputs as verify_saved_inspection
    from source_inspection_operation_contract import parse_contract as parse_inspection, verify_inputs as verify_inspection
    from preparation_operation_contract import parse_contract as parse_preparation, verify_inputs as verify_preparation
    from preview_operation_contract import parse_contract, verify_inputs
    from execution_budget import validate_remaining_budget
    from evaluation_authorization import EvaluationAuthority, OWNERSHIP_FIELDS, CLAIM_FIELDS
    from worker_authorization import require, canonical_uuid, hash_value, validate_phase_arguments, directory, read_regular, sha

REQUEST_FIELDS = {'schemaVersion','kind','operationID','projectID','runID','directoryID',
                  'requestIdentity','phase','arguments'}
TOOL_FIELDS = {'schemaVersion','kind','operationID','projectID','requestIdentity','toolMode',
               'arguments','inputFiles','executionBudget','remainingSeconds'}


def validate_tool_request(value,*,preparation_root=None,preparation_cache=None):
    require(set(value)==TOOL_FIELDS and value['kind']=='workflowOperation', 'Invalid tool request fields')
    for key in ('operationID','projectID'):canonical_uuid(value[key])
    hash_value(value['requestIdentity'])
    validate_remaining_budget(value['executionBudget'],value['remainingSeconds'])
    mode=value['toolMode']
    if mode=='runtimeCapabilities':
        require(value['arguments']==['--capabilities'] and value['inputFiles']==[], 'Invalid runtime capability discovery inputs')
        return {}
    if mode=='modelAsset':
        if __package__:from .model_asset_operation_contract import parse_contract as parse_model
        else:from model_asset_operation_contract import parse_contract as parse_model
        return parse_model(value)[1]
    if mode=='textGeneration':
        if __package__:from .text_generation_operation_contract import parse_contract as parse_generation
        else:from text_generation_operation_contract import parse_contract as parse_generation
        return parse_generation(value)[1]
    if mode=='selfPlay':
        if __package__:from .self_play_operation_contract import parse_contract as parse_game
        else:from self_play_operation_contract import parse_contract as parse_game
        return parse_game(value)[1]
    if mode=='gameSession':
        if __package__:from .game_session_operation_contract import parse_contract as parse_game
        else:from game_session_operation_contract import parse_contract as parse_game
        return parse_game(value)[1]
    if mode=='programExport':return parse_program(value)[1]
    if mode=='modelPreview':return parse_contract(value)[1]
    if mode=='backendFit':return parse_backend(value)[1]
    if mode=='modelTraining':return parse_training(value)[1]
    if mode=='dataPreparation':
        if preparation_root is not None:
            require(isinstance(preparation_cache,dict),'Missing preparation verification state')
            return verify_preparation(value,preparation_root,preparation_cache)
        return parse_preparation(value)[1]
    if mode=='tableViewQuery':
        require(value['remainingSeconds']<=30,'Table View query exceeds its time allowance')
        return parse_table_view(value)
    if mode=='sourceInspection':return parse_inspection(value)
    if mode=='preparationInspection':return parse_saved_inspection(value)
    require(mode in ('learningRuntime','learningAssessment','codeExport','mixtureValidation'), 'Unsupported operation tool')
    ceilings={'mixtureValidation':30,'learningRuntime':10,'learningAssessment':65,'codeExport':30}
    require(value['remainingSeconds'] <= ceilings[mode], 'Tool worker allowance exceeds capability ceiling')
    arguments=value['arguments']
    workflow_code=mode=='codeExport' and isinstance(arguments,list) and arguments[:2]==['--export-workflow-code','--workflow-code-plan']
    required_inputs={'mixtureValidation':{'input/mixture.json'},'learningRuntime':set(),'learningAssessment':{'input/assessment.json','input/submission.json'},
                     'codeExport':{'input/workflow-code.json'} if workflow_code else {'input/graph.json'}}[mode]
    files=value['inputFiles']
    if mode=='learningAssessment' and isinstance(files,list) and len(files)==3:
        required_inputs=required_inputs|{'input/model-registry.json'}
    require(isinstance(files,list) and len(files)==len(required_inputs), 'Invalid tool input inventory')
    inventory={}
    for entry in files:
        require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'}, 'Invalid tool input entry')
        name=entry['path']
        require(isinstance(name,str) and name in required_inputs and name not in inventory,
                'Invalid or repeated tool input path')
        require(type(entry['bytes']) is int and 0<=entry['bytes']<=2000000, 'Tool input exceeds byte limit')
        hash_value(entry['sha256']);inventory[name]=entry
    arguments=value['arguments']
    require(isinstance(arguments,list) and all(isinstance(arg,str) and arg and '\0' not in arg for arg in arguments),
            'Invalid tool arguments')
    if mode=='learningRuntime':require(arguments==['--learning-runtime'], 'Unexpected runtime discovery arguments')
    elif mode=='mixtureValidation':
        require(len(arguments)==2 and arguments[0]=='--validate-mixture','Invalid MoE validation arguments')
    elif mode=='codeExport':
        require(len(arguments)==3 and arguments[:2]==(['--export-workflow-code','--workflow-code-plan'] if workflow_code else ['--export-code','--graph']), 'Invalid code export arguments')
    else:
        captured_registry='input/model-registry.json' in inventory
        require(set(inventory)==required_inputs, 'Missing assessment inputs')
        require(len(arguments)==(8 if captured_registry else 6) and arguments[:6:2]==['--learning-evidence-request','--learning-submission','--learning-submission-sha256']
                and arguments[5]==inventory['input/submission.json']['sha256'], 'Invalid assessment arguments')
        if captured_registry:require(arguments[6]=='--registry', 'Invalid captured assessment registry flag')
    return inventory


def validate_request(value,*,preparation_root=None,preparation_cache=None):
    require(isinstance(value,dict) and type(value.get('schemaVersion')) is int, 'Invalid operation schema')
    if value['schemaVersion']==2:return validate_tool_request(value,preparation_root=preparation_root,preparation_cache=preparation_cache)
    require(isinstance(value,dict) and set(value) in (REQUEST_FIELDS,REQUEST_FIELDS|{'executionBudget','remainingSeconds'}),
            'Invalid operation request fields')
    require(type(value['schemaVersion']) is int and value['schemaVersion']==1
            and value['kind']=='workflowOperation', 'Invalid operation request kind')
    for key in ('operationID','projectID','runID','directoryID'):canonical_uuid(value[key])
    hash_value(value['requestIdentity'])
    if 'executionBudget' in value:
        validate_remaining_budget(value['executionBudget'],value['remainingSeconds'])
    return validate_phase_arguments(value['phase'],value['arguments'])


class OperationAuthority(EvaluationAuthority):
    supports_launch_budget=False
    identity_key='operationID'
    kind='workflowOperation'
    namespace='workflow-operations'
    ownership_fields=(OWNERSHIP_FIELDS-{'evaluationID'})|{'operationID'}
    claim_fields=(CLAIM_FIELDS-{'evaluationID'})|{'operationID'}

    def _read_control(self,name,maximum):
        if self.request.get('schemaVersion')!=2 or self.request.get('toolMode') not in ('dataPreparation','backendFit'):
            return super()._read_control(name,maximum)
        # Exactly three launch controls; backend inputs have a separate bounded proof.
        require(name in ('ownership.json','request.json','claim.json'),'Unknown operation control')
        if not hasattr(self,'_control_proofs'):self._control_proofs={}
        if name not in self._control_proofs:
            if __package__:from .worker_authorization import ImmutableControlProof
            else:from worker_authorization import ImmutableControlProof
            self._control_proofs[name]=ImmutableControlProof(self._directory_fd,name,maximum)
        return self._control_proofs[name].read()

    def close(self):
        if hasattr(self,'_backend_signatures'):
            if __package__:from .backend_fit_operation_contract import close_inputs
            else:from backend_fit_operation_contract import close_inputs
            close_inputs(self._backend_signatures)
        for proof in getattr(self,'_control_proofs',{}).values():proof.close()
        super().close()

    def _validate_request(self):
        validate_request(self.request)
        for key in ('operationID','projectID','requestIdentity'):
            require(self.request[key]==self._ownership[key], 'Operation request differs from retained owner')

    def _check(self,claimed):
        super()._check(claimed)
        value=self.request
        if value.get('schemaVersion')==2 and value.get('toolMode')=='backendFit':
            # The backend proof binds the complete, exact request representation.
            # The live owner/lock/control checks above are never memoized.
            self._check_tool_inputs(None)
            return
        if value['schemaVersion']==2 and value.get('toolMode')=='dataPreparation':
            if not hasattr(self,'_preparation_signatures'):self._preparation_signatures={}
            # Validate the current request and retained-root contract once per
            # check. All file, control-byte, ownership and lock checks still run.
            validate_request(value,preparation_root=self._path,preparation_cache=self._preparation_signatures)
            return
        options=validate_request(value)
        if value['schemaVersion']==2:
            self._check_tool_inputs(options)
            return
        phase=Path(self._path).parent.parent.parent/'workflow-runs'/value['projectID']/value['directoryID']/value['phase']
        require(options.get('--output')==str(phase/'artifacts'), 'Operation output differs from captured phase')
        required={'--execute-dataflow':'request.json','--source-bindings':'bindings.json'} if value['phase']=='preparation' else {'--graph':'request.json'}
        if value['phase']=='preparation' and '--destination-bindings' in options:required['--destination-bindings']='destinations.json'
        phase_fd=directory(str(phase))
        try:
            for option,name in required.items():
                require(options.get(option)==str(phase/name), 'Operation input differs from captured phase')
                control=read_regular(phase_fd,name,1000000 if name in ('bindings.json','destinations.json') else 2000000)
                if name=='destinations.json':
                    pin=sha(control)
                    if hasattr(self,'_destination_control_pin'):require(pin==self._destination_control_pin,'Output destination control changed after authority capture')
                    else:self._destination_control_pin=pin
            try: output=os.open('artifacts',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=phase_fd)
            except FileNotFoundError:pass
            else:os.close(output)
        finally:os.close(phase_fd)
        if value['phase']=='training':
            if __package__:from .worker_input_contract import verify_phase_bag_inputs,requires_bag_admission
            else:from worker_input_contract import verify_phase_bag_inputs,requires_bag_admission
            if requires_bag_admission(options,phase/'request.json'):
                if not hasattr(self,'_bag_input_signatures'):self._bag_input_signatures={}
                verify_phase_bag_inputs(phase,options,value,self._bag_input_signatures)


    def _check_tool_inputs(self,inventory):
        if self.request['toolMode']=='modelAsset':
            if __package__:from .model_asset_operation_contract import verify_inputs as verify_model
            else:from model_asset_operation_contract import verify_inputs as verify_model
            if not hasattr(self,'_model_asset_signatures'):self._model_asset_signatures={}
            verify_model(self.request,self._path,self._model_asset_signatures)
            return
        if self.request['toolMode']=='textGeneration':
            if __package__:from .text_generation_operation_contract import verify_inputs as verify_generation
            else:from text_generation_operation_contract import verify_inputs as verify_generation
            if not hasattr(self,"_generation_signatures"):self._generation_signatures={}
            verify_generation(self.request,self._path,self._generation_signatures)
            return
        if self.request['toolMode']=='selfPlay':
            if __package__:from .self_play_operation_contract import verify_inputs as verify_game
            else:from self_play_operation_contract import verify_inputs as verify_game
            if not hasattr(self,'_selfplay_signatures'):self._selfplay_signatures={}
            verify_game(self.request,self._path,self._selfplay_signatures)
            return
        if self.request['toolMode']=='gameSession':
            if __package__:from .game_session_operation_contract import verify_inputs as verify_game
            else:from game_session_operation_contract import verify_inputs as verify_game
            if not hasattr(self,'_game_signatures'):self._game_signatures={}
            verify_game(self.request,self._path,self._game_signatures)
            return
        if self.request['toolMode']=='programExport':
            if not hasattr(self,'_program_signatures'):self._program_signatures={}
            verify_program(self.request,self._path,self._program_signatures)
            return
        if self.request['toolMode']=='backendFit':
            if not hasattr(self,'_backend_signatures'):self._backend_signatures={}
            verify_backend(self.request,self._path,self._backend_signatures)
            return
        if self.request['toolMode']=='modelTraining':
            if not hasattr(self,'_training_signatures'):self._training_signatures={}
            verify_training(self.request,self._path,self._training_signatures)
            return
        if self.request['toolMode']=='preparationInspection':
            if not hasattr(self,'_saved_inspection_signatures'):self._saved_inspection_signatures={}
            verify_saved_inspection(self.request,self._path,self._saved_inspection_signatures)
            return
        if self.request['toolMode']=='tableViewQuery':
            if not hasattr(self,'_table_view_signatures'):self._table_view_signatures={}
            verify_table_view(self.request,self._path,self._table_view_signatures)
            return
        if self.request['toolMode']=='sourceInspection':
            if not hasattr(self,'_inspection_signatures'):self._inspection_signatures={}
            verify_inspection(self.request,self._path,self._inspection_signatures)
            return
        if self.request['toolMode']=='dataPreparation':
            if not hasattr(self,'_preparation_signatures'):self._preparation_signatures={}
            verify_preparation(self.request,self._path,self._preparation_signatures)
            return
        if self.request['toolMode']=='modelPreview':
            if not hasattr(self,'_preview_signatures'):self._preview_signatures={}
            verify_inputs(self.request,self._path,self._preview_signatures)
            return
        if self.request['toolMode'] in ('learningRuntime','runtimeCapabilities'):return
        arguments=self.request['arguments']
        base=Path(self._path)
        if self.request['toolMode']=='mixtureValidation':
            require(arguments==['--validate-mixture',str(base/'input/mixture.json')],'MoE snapshot differs from retained operation')
        elif self.request['toolMode']=='codeExport':
            name='input/workflow-code.json' if arguments[0]=='--export-workflow-code' else 'input/graph.json'
            require(arguments[2]==str(base/name), 'Code snapshot path differs from retained operation')
        else:
            require(arguments[1]==str(base/'input/assessment.json') and arguments[3]==str(base/'input/submission.json'),
                    'Assessment paths differ from retained operation')
            if 'input/model-registry.json' in inventory:
                require(arguments[7]==str(base/'input/model-registry.json'), 'Assessment registry differs from retained operation')
        fd=directory(str(base/'input'))
        try:
            for name,entry in inventory.items():
                raw=read_regular(fd,Path(name).name,2000000)
                require(len(raw)==entry['bytes'] and sha(raw)==entry['sha256'], 'Captured tool input changed')
        finally:os.close(fd)


def claim_operation(folder,request_sha256,ownership_sha256):
    return OperationAuthority(folder,request_sha256,ownership_sha256)


def inherit_operation(folder,request_sha256,ownership_sha256,activity_fd):
    return OperationAuthority(folder,request_sha256,ownership_sha256,inherited_fd=activity_fd)
