"""Standard-library-only consumption of app-published, pinned launch authority.

APFS admission belongs to the Swift ownership publisher. This reader preserves
its pinned identities; it does not admit filesystems or create replacement locks.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import uuid

if __package__:
    from .execution_budget import validate_remaining_budget
else:
    from execution_budget import validate_remaining_budget

INTENT_FIELDS = {'schemaVersion','projectID','groupID','generation','runID','directoryID','nonce',
                 'ordinal','requestIdentity','ownershipSHA256','phase','arguments'}
METADATA_FIELDS = {'schemaVersion','groupID','generation','directoryIdentity','ownerLockIdentity',
                   'activityLockIdentity','createdAtUnixSeconds'}
PREPARATION = {'--dataflow-live-previews','--dataflow-measurements','--defer-database-writes','--dataflow-required-nodes','--dataflow-model-input-nodes','--dataflow-capture-outputs','--dataflow-sampling','--dataflow-complete-nodes','--destination-bindings','--retain-partial-dataflow','--execute-dataflow','--source-bindings','--dataflow-cache-policy','--output',
               '--dataflow-reuse-manifest','--dataflow-reuse-manifest-sha256','--dataflow-cache-limit-bytes'}
TRAINING = {'--graph','--output','--registry','--epochs','--seed','--device','--batch-size',
            '--learning-rate','--prepared-dataflow','--prepared-manifest-sha256','--prepared-graph-digest',
            '--prepared-training-node','--prepared-training-port','--prepared-validation-node',
            '--prepared-validation-port','--feature-columns-json','--target-column','--bag-input-mapping-json','--named-input-plan-json',
            '--gradient-every-epochs','--gradient-max-batches-per-epoch','--gradient-max-captures',
            '--resume-checkpoint','--finetune-checkpoint','--composite-source','--composite-source-sha256',
            '--dataset-recipe-json','--loader-recipe-json','--capture-gradients','--training-control-json','--training-progress-json','--sample-capture-json','--gradient-policy-json','--trainability-json','--derivative-capture-json','--initialization-recipe-json','--optimizer-recipe-json','--scheduler-recipe-json','--objective-recipe-json','--objective-target-mapping-json','--objective-scalarization'}


def require(condition, message='Invalid launch authority'):
    if not condition: raise ValueError(message)


def canonical_uuid(value):
    require(isinstance(value,str))
    try: require(str(uuid.UUID(value)).upper()==value)
    except (AttributeError, ValueError): raise ValueError('Expected canonical UUID') from None


def hash_value(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def file_identity(info): return {'device':info.st_dev & 0xffffffff,'inode':info.st_ino}
def signature(info): return (info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_uid,info.st_nlink)


def object_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON field')
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    require(isinstance(value,dict))
    return value


def directory(path):
    path=os.fspath(path)
    require(isinstance(path,str) and path.startswith('/') and '\0' not in path and str(Path(path))==path,'Expected normalized absolute path')
    pieces=path.split('/')[1:]
    require(all(part not in ('','.', '..') for part in pieces))
    if sys.platform == 'darwin':
        # The kernel rejects symlinks in every component without opening each
        # ancestor for read-data, which a sandbox may deny independently of the leaf.
        # Darwin rejects combining O_NOFOLLOW_ANY with the older O_NOFOLLOW flag.
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW_ANY)
    descriptor=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    try:
        for part in pieces:
            next_fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=descriptor)
            os.close(descriptor);descriptor=next_fd
        return descriptor
    except BaseException:
        os.close(descriptor);raise


def read_regular(parent,name,maximum):
    descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=parent)
    try:
        before=os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size<=maximum and before.st_uid==os.geteuid() and before.st_nlink==1 and before.st_mode & 0o022==0,'Expected private bounded regular file')
        chunks=[];length=0
        while True:
            chunk=os.read(descriptor,min(65536,maximum+1-length))
            if not chunk:break
            chunks.append(chunk);length+=len(chunk)
            require(length<=maximum,'File exceeds bound')
        after=os.fstat(descriptor);named=os.stat(name,dir_fd=parent,follow_symlinks=False)
        require(signature(before)==signature(after)==signature(named),'File changed during read')
        return b''.join(chunks)
    finally:os.close(descriptor)


class ImmutableControlProof:
    """Small immutable control bytes with a live descriptor/named-file proof.

    The retained file is checked on every read. Changed metadata, replacement,
    links, permission or content changes invalidate it; no timed authority cache.
    Callers must still verify the containing directory and owner locks.
    """
    def __init__(self,parent,name,maximum):
        self.descriptor=None
        descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=parent)
        try:
            before=os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_size<=maximum and before.st_uid==os.geteuid()
                and before.st_nlink==1 and before.st_mode & 0o022==0,'Expected private bounded control')
            chunks=[];length=0
            while True:
                chunk=os.read(descriptor,min(65536,maximum+1-length))
                if not chunk:break
                chunks.append(chunk);length+=len(chunk);require(length<=maximum,'Control exceeds bound')
            after=os.fstat(descriptor);named=os.stat(name,dir_fd=parent,follow_symlinks=False)
            require(signature(before)==signature(after)==signature(named),'Control changed during admission')
            self.parent=parent;self.name=name;self.pin=signature(after);self.content=b''.join(chunks)
            self.descriptor=descriptor
        except BaseException:os.close(descriptor);raise
    def read(self):
        require(self.descriptor is not None,'Control proof is closed')
        require(signature(os.fstat(self.descriptor))==self.pin
            ==signature(os.stat(self.name,dir_fd=self.parent,follow_symlinks=False)),'Immutable control changed')
        return self.content
    def close(self):
        if self.descriptor is not None:os.close(self.descriptor);self.descriptor=None
    def __del__(self):self.close()


def validate_identity(value):
    require(isinstance(value,dict) and set(value)=={'device','inode'})
    require(all(type(v) is int and 0<=v<=2**64-1 for v in value.values()))
    require(value['device']<=2**32-1)


def validate_intent(intent):
    require(isinstance(intent,dict) and type(intent.get('schemaVersion')) is int)
    version=intent['schemaVersion']
    require(version in (1,2) and set(intent)==(INTENT_FIELDS if version==1 else INTENT_FIELDS|{'executionBudget','remainingSeconds'}))
    if version==2:validate_remaining_budget(intent['executionBudget'],intent['remainingSeconds'])
    for field in ('projectID','groupID','generation','runID','directoryID','nonce'):canonical_uuid(intent[field])
    for field in ('requestIdentity','ownershipSHA256'):hash_value(intent[field])
    require(type(intent['ordinal']) is int and 1<=intent['ordinal']<=32)
    return validate_phase_arguments(intent['phase'], intent['arguments'])


def validate_phase_arguments(phase, arguments):
    require(phase in ('preparation','training'))
    objective_flags={'--objective-recipe-json','--objective-target-mapping-json','--objective-scalarization'}
    has_objective=phase=='training' and isinstance(arguments,list) and any(isinstance(arg,str) and arg.split('=',1)[0] in objective_flags for arg in arguments)
    options=parse_bounded_options(arguments, PREPARATION if phase=='preparation' else TRAINING,
                                  {'--capture-gradients','--retain-partial-dataflow'},
                                  **({'maximum_argument_bytes':524288,'maximum_total_bytes':786432} if has_objective else {}))
    if has_objective:
        for key,text in options.items():
            if key not in objective_flags and isinstance(text,str):require(len(text.encode())<=65536,'Launch argument exceeds its size bound')
        require({'--objective-recipe-json','--objective-target-mapping-json'}<=set(options),'Objective requires recipe and typed mapping')
        require(len(options['--objective-recipe-json'].encode())<=32768 and len(options['--objective-target-mapping-json'].encode())<=524288,'Objective controls exceed their size bounds')
        require(options.get('--objective-scalarization','mean') in ('mean','sum'),'Invalid objective scalarization')
        if __package__:
            from .loss_configuration import validate_recipe as validate_objective_recipe
            from .loss_target_mapping import validate_target_mapping
        else:
            from loss_configuration import validate_recipe as validate_objective_recipe
            from loss_target_mapping import validate_target_mapping
        validate_target_mapping(json.loads(options['--objective-target-mapping-json']),validate_objective_recipe(json.loads(options['--objective-recipe-json'])))
        require('--prepared-dataflow' in options and not {'--target-column','--feature-columns-json'}&set(options),'Objective requires prepared typed mapping without legacy columns')
    if '--initialization-recipe-json' in options:
        require(len(options['--initialization-recipe-json'].encode('utf-8'))<=32768,'Initialization recipe exceeds its size bound')
        if __package__:from .initialization import validate_recipe as validate_initialization_recipe
        else:from initialization import validate_recipe as validate_initialization_recipe
        validate_initialization_recipe(json.loads(options['--initialization-recipe-json']))
    validate_gradient_arguments(options)
    validate_loading_arguments(options)
    if '--named-input-plan-json' in options:
        require('--prepared-dataflow' in options and '--bag-input-mapping-json' not in options,'Named inputs require prepared partitions without bag mapping')
        require(len(options['--named-input-plan-json'].encode())<=65536,'Named input mapping exceeds its byte allowance')
        if __package__:from .named_inputs import validate_plan
        else:from named_inputs import validate_plan
        validate_plan(json.loads(options['--named-input-plan-json']))
    if '--bag-input-mapping-json' in options:
        if __package__:from .worker_input_contract import validate_bag_arguments
        else:from worker_input_contract import validate_bag_arguments
        validate_bag_arguments(options)
    if phase=='preparation':
        validate_preparation_cache_allowance(options)
        validate_model_input_nodes(options)
        if '--dataflow-required-nodes' in options:
            if __package__:from .preparation_demand_configuration import parse_complete_nodes_json
            else:from preparation_demand_configuration import parse_complete_nodes_json
            parse_complete_nodes_json(options['--dataflow-required-nodes'])
        require(options.get('--defer-database-writes','true')=='true','Database writes must use the native broker')
        require(options.get('--dataflow-capture-outputs','true') in ('true','false'),'Invalid preparation output retention')
        require(options.get('--dataflow-live-previews','false') in ('true','false'),'Invalid preparation live-preview choice')
        require(options.get('--dataflow-measurements','false') in ('true','false'),'Invalid preparation measurements')
    if '--training-control-json' in options:
        if __package__:from .training_control import validate_policy as validate_training_control
        else:from training_control import validate_policy as validate_training_control
        require(len(options['--training-control-json'].encode())<=1024,'Training control policy is oversized')
        validate_training_control(json.loads(options['--training-control-json']))
    if '--training-progress-json' in options:
        if __package__:from .training_progress import validate_policy
        else:from training_progress import validate_policy
        require(len(options['--training-progress-json'].encode())<=256,'Training reporting policy is oversized')
        validate_policy(json.loads(options['--training-progress-json']))
    if '--sample-capture-json' in options:
        if __package__:
            from .capture_policy import validate_training_sample_capture_policy
        else:
            from capture_policy import validate_training_sample_capture_policy
        require(len(options['--sample-capture-json'].encode())<=1024,'Sample capture policy exceeds its size bound')
        validate_training_sample_capture_policy(json.loads(options['--sample-capture-json']))
    require(not {'--resume-checkpoint','--finetune-checkpoint'}<=set(options))
    return options


def validate_preparation_cache_allowance(options):
    value=options.get('--dataflow-cache-limit-bytes')
    if value is None:return
    require(options.get('--dataflow-cache-policy') in ('materialize','reuse'), 'Cache allowance requires materialize or reuse policy')
    require(isinstance(value,str) and re.fullmatch('[1-9][0-9]{0,8}',value) is not None and int(value)<=200000000,
            'Cache allowance must be canonical integer bytes from 1 to 200000000')


def parse_bounded_options(arguments,allowed,flags=frozenset(),*,maximum_count=128,maximum_argument_bytes=65536,maximum_total_bytes=131072):
    require(isinstance(arguments,list) and 1<=len(arguments)<=maximum_count)
    require(all(isinstance(arg,str) and arg and '\0' not in arg and len(arg.encode())<=maximum_argument_bytes for arg in arguments))
    require(sum(len(arg.encode()) for arg in arguments)<=maximum_total_bytes)
    options={};index=0
    while index<len(arguments):
        key=arguments[index]
        require(key in allowed and key not in options,'Unsupported or repeated operation option')
        if key in flags:options[key]=True;index+=1;continue
        require(index+1<len(arguments) and not arguments[index+1].startswith('--'),'Missing operation option value')
        options[key]=arguments[index+1];index+=2
    return options


def validate_claim(raw,nonce,pin):
    value=object_json(raw)
    require(set(value)=={'schemaVersion','nonce','intentSHA256','state'} and type(value['schemaVersion']) is int and value['schemaVersion']==1)
    require(value['nonce']==nonce and value['intentSHA256']==pin and value['state']=='claimed')


class LaunchAuthority:
    def __init__(self,group_directory,nonce,intent_sha256,ownership_sha256,*,_inherited_fd=None):
        self.activity_fd=None;self._owner_fd=None;self._directory_fd=None;self._closed=False
        canonical_uuid(nonce);hash_value(intent_sha256);hash_value(ownership_sha256)
        self._path=os.fspath(group_directory);self._nonce=nonce;self._intent_pin=intent_sha256;self._ownership_pin=ownership_sha256
        try:
            # Validate/duplicate before any other open: a closed descriptor number
            # must not be reused by our directory walk and mistaken for authority.
            if _inherited_fd is not None:
                require(type(_inherited_fd) is int and _inherited_fd>2,'Invalid inherited activity descriptor')
                original=os.fstat(_inherited_fd)
                require(stat.S_ISREG(original.st_mode) and original.st_uid==os.geteuid() and original.st_nlink==1 and original.st_mode & 0o022==0)
                self.activity_fd=fcntl.fcntl(_inherited_fd,fcntl.F_DUPFD_CLOEXEC,3)
                require(signature(os.fstat(self.activity_fd))==signature(original),'Inherited activity changed while duplicating')
            self._directory_fd=directory(self._path)
            info=os.fstat(self._directory_fd)
            require(info.st_uid==os.geteuid() and info.st_mode & 0o022==0,'Unsafe group directory permissions')
            self._metadata_bytes=read_regular(self._directory_fd,'ownership.json',16384)
            require(sha(self._metadata_bytes)==ownership_sha256,'Ownership digest mismatch')
            self._metadata=object_json(self._metadata_bytes)
            metadata=self._metadata
            require(set(metadata)==METADATA_FIELDS and type(metadata['schemaVersion']) is int and metadata['schemaVersion']==1)
            canonical_uuid(metadata['groupID']);canonical_uuid(metadata['generation'])
            timestamp=metadata['createdAtUnixSeconds']
            require(type(timestamp) in (int,float) and math.isfinite(timestamp) and 0<=timestamp<=253402300799)
            for key in ('directoryIdentity','ownerLockIdentity','activityLockIdentity'):validate_identity(metadata[key])
            require(file_identity(info)==metadata['directoryIdentity'])
            self._owner_fd=self._lock('owner.lock',metadata['ownerLockIdentity'])
            if _inherited_fd is None:
                self.activity_fd=self._lock('activity.lock',metadata['activityLockIdentity'])
                fcntl.flock(self.activity_fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
            else:
                inherited=os.fstat(self.activity_fd)
                named=os.stat('activity.lock',dir_fd=self._directory_fd,follow_symlinks=False)
                require(signature(inherited)==signature(named) and file_identity(inherited)==metadata['activityLockIdentity'],'Inherited activity identity mismatch')
                # The trusted supervisor already holds SH on this very description.
                # Never reopen/relock it: doing so could change inherited authority.
            self._intent_bytes=read_regular(self._directory_fd,f'launch-{nonce}.json',256000)
            require(sha(self._intent_bytes)==intent_sha256,'Intent digest mismatch')
            self.intent=object_json(self._intent_bytes)
            if _inherited_fd is None:
                self._validate(False)
                claim=json.dumps(dict(schemaVersion=1,nonce=nonce,intentSHA256=intent_sha256,state='claimed'),sort_keys=True,separators=(',',':')).encode()
                descriptor=os.open(f'claim-{nonce}.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self._directory_fd)
                try:
                    view=memoryview(claim)
                    while view:
                        written=os.write(descriptor,view)
                        require(written>0,'Incomplete claim write');view=view[written:]
                    os.fsync(descriptor)
                finally:os.close(descriptor)
                os.fsync(self._directory_fd)
            self.check()
        except BaseException:
            self.close();raise

    def _lock(self,name,pin):
        descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=self._directory_fd)
        try:
            info=os.fstat(descriptor);named=os.stat(name,dir_fd=self._directory_fd,follow_symlinks=False)
            require(stat.S_ISREG(info.st_mode) and info.st_uid==os.geteuid() and info.st_nlink==1 and info.st_mode & 0o022==0)
            require(file_identity(info)==pin and signature(info)==signature(named),'Lock identity mismatch')
            return descriptor
        except BaseException:os.close(descriptor);raise

    def _validate(self,claimed):
        require(not self._closed,'Authority is closed')
        current=directory(self._path)
        try:
            info=os.fstat(current)
            require(file_identity(info)==self._metadata['directoryIdentity'] and info.st_uid==os.geteuid() and info.st_mode & 0o022==0)
        finally:os.close(current)
        require(read_regular(self._directory_fd,'ownership.json',16384)==self._metadata_bytes)
        for name,descriptor,key in [('owner.lock',self._owner_fd,'ownerLockIdentity'),('activity.lock',self.activity_fd,'activityLockIdentity')]:
            info=os.fstat(descriptor);named=os.stat(name,dir_fd=self._directory_fd,follow_symlinks=False)
            require(stat.S_ISREG(named.st_mode) and file_identity(info)==file_identity(named)==self._metadata[key])
            require(named.st_uid==os.geteuid() and named.st_nlink==1 and info.st_nlink==1 and named.st_mode & 0o022==0)
        intent=self.intent
        require(intent==object_json(self._intent_bytes),'In-memory intent changed')
        options=validate_intent(intent)
        require(intent['nonce']==self._nonce and intent['ownershipSHA256']==self._ownership_pin and intent['generation']==self._metadata['generation'])
        require(intent['groupID']==self._metadata['groupID'])
        path=Path(self._path)
        require(path.name==intent['groupID'] and path.parent.name==intent['projectID'] and path.parent.parent.name=='workflow-experiments')
        require(read_regular(self._directory_fd,f'launch-{self._nonce}.json',256000)==self._intent_bytes)
        slot=f"phase-{intent['runID']}-{intent['phase']}.json"
        require(read_regular(self._directory_fd,slot,256000)==self._intent_bytes,'Phase slot differs from intent')
        if intent['phase']=='training':
            prior_raw=read_regular(self._directory_fd,f"phase-{intent['runID']}-preparation.json",256000)
            prior=object_json(prior_raw);validate_intent(prior)
            require(prior['phase']=='preparation')
            require(prior['schemaVersion']==intent['schemaVersion'], 'Training changed captured budget schema')
            if intent['schemaVersion']==2:
                require(prior['executionBudget']==intent['executionBudget'], 'Training changed captured execution budget')
                require(intent['remainingSeconds']<=prior['remainingSeconds'], 'Training reset remaining execution budget')
            for key in ('projectID','groupID','generation','runID','directoryID','ordinal','requestIdentity','ownershipSHA256'):require(prior[key]==intent[key])
            require(read_regular(self._directory_fd,f"launch-{prior['nonce']}.json",256000)==prior_raw)
            validate_claim(read_regular(self._directory_fd,f"claim-{prior['nonce']}.json",4096),prior['nonce'],sha(prior_raw))
        ledger=object_json(read_regular(self._directory_fd,'group.json',256000))
        require(type(ledger.get('schemaVersion')) is int and ledger['schemaVersion']==1)
        require(ledger.get('projectID')==intent['projectID'] and ledger.get('groupID')==intent['groupID'] and ledger.get('stopRequested') is False)
        variants=ledger.get('variants');require(isinstance(variants,list) and 1<=len(variants)<=32)
        ordinals=set();selected=None;running=0
        for index,variant in enumerate(variants,1):
            require(isinstance(variant,dict))
            ordinal=variant.get('ordinal');require(type(ordinal) is int and 1<=ordinal<=32 and ordinal not in ordinals)
            require(ordinal==index and variant.get('status') in ('pending','running','completed','failed','cancelled','interrupted'))
            ordinals.add(ordinal)
            if variant.get('status')=='running':running+=1
            if ordinal==intent['ordinal']:selected=variant
        require(running==1 and selected is not None and selected.get('status')=='running')
        for key in ('runID','directoryID','requestIdentity'):require(selected.get(key)==intent[key])
        phase=path.parent.parent.parent/'workflow-runs'/intent['projectID']/intent['directoryID']/intent['phase']
        require(options.get('--output')==str(phase/'artifacts'))
        required={'--execute-dataflow':'request.json','--source-bindings':'bindings.json'} if intent['phase']=='preparation' else {'--graph':'request.json'}
        if intent['phase']=='preparation' and '--destination-bindings' in options:required['--destination-bindings']='destinations.json'
        phase_fd=directory(str(phase))
        try:
            for option,name in required.items():
                require(options.get(option)==str(phase/name))
                control=read_regular(phase_fd,name,1000000 if name in ('bindings.json','destinations.json') else 2000000)
                if name=='destinations.json':
                    pin=sha(control)
                    if hasattr(self,'_destination_control_pin'):require(pin==self._destination_control_pin,'Output destination control changed after authority capture')
                    else:self._destination_control_pin=pin
            try:
                output=os.open('artifacts',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=phase_fd)
            except FileNotFoundError:pass
            else:os.close(output)
        finally:os.close(phase_fd)
        if intent['phase']=='training':
            if __package__:from .worker_input_contract import verify_phase_bag_inputs,requires_bag_admission
            else:from worker_input_contract import verify_phase_bag_inputs,requires_bag_admission
            if requires_bag_admission(options,phase/'request.json'):
                if not hasattr(self,'_bag_input_signatures'):self._bag_input_signatures={}
                verify_phase_bag_inputs(phase,options,intent,self._bag_input_signatures)
        if claimed:validate_claim(read_regular(self._directory_fd,f'claim-{self._nonce}.json',4096),self._nonce,self._intent_pin)

    def check(self):self._validate(True)
    def close(self):
        if self._closed:return
        self._closed=True
        for name in ('activity_fd','_owner_fd','_directory_fd'):
            descriptor=getattr(self,name,None)
            if descriptor is not None:
                os.close(descriptor);setattr(self,name,None)
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def claim_launch(group_directory,nonce,intent_sha256,ownership_sha256):
    return LaunchAuthority(group_directory,nonce,intent_sha256,ownership_sha256)


def inherit_launch(group_directory,nonce,intent_sha256,ownership_sha256,activity_fd):
    """Validate a trusted supervisor-inherited lease without consuming another claim.

    Duplicates the caller descriptor; caller ownership is unchanged. Does not
    authenticate an arbitrary same-user descriptor or acquire replacement locks.
    """
    require(type(activity_fd) is int and activity_fd>2,'Invalid inherited activity descriptor')
    return LaunchAuthority(group_directory,nonce,intent_sha256,ownership_sha256,
                           _inherited_fd=activity_fd)


def validate_gradient_arguments(options):
    if not {'--gradient-policy-json','--trainability-json','--derivative-capture-json'}&set(options):return
    if __package__:
        from .gradient_policy import validate_recipe,validate_capture,validate_compatibility
        from .trainability import validate_recipe as validate_trainability
    else:
        from gradient_policy import validate_recipe,validate_capture,validate_compatibility
        from trainability import validate_recipe as validate_trainability
    policy=None
    if '--gradient-policy-json' in options:
        require(len(options['--gradient-policy-json'].encode('utf8'))<=32768,'Gradient policy exceeds32768 bytes')
        policy=validate_recipe(json.loads(options['--gradient-policy-json']))
    if '--trainability-json' in options:
        require(len(options['--trainability-json'].encode('utf8'))<=32768,'Trainability exceeds32768 bytes')
        validate_trainability(json.loads(options['--trainability-json']))
    capture=None
    if '--derivative-capture-json' in options:
        require(len(options['--derivative-capture-json'].encode('utf8'))<=256,'Derivative permission exceeds256 bytes')
        capture=validate_capture(json.loads(options['--derivative-capture-json']))
    if policy is not None:
        capture=validate_capture(capture)
        require(not policy['config']['requests'] or capture['values'],'Enable derivative value capture before running requested analysis')
        objective=json.loads(options['--objective-recipe-json']) if '--objective-recipe-json' in options else None
        validate_compatibility(policy,objective,options.get('--objective-scalarization','mean'))
    elif capture is not None:require(not capture['values'],'Derivative capture requires selected gradient policy')


def validate_loading_arguments(options):
    """Exact inline policies; source/content admission stays in the real engine."""
    if __package__:from .dataset_loader_configuration import validate_recipe
    else:from dataset_loader_configuration import validate_recipe
    for flag,component in (('--dataset-recipe-json','dataset_source'),('--loader-recipe-json','data_loader')):
        if flag not in options:continue
        text=options[flag]
        require(type(text) is str and len(text.encode('utf8'))<=32768,'Dataset/DataLoader recipe exceeds32768 UTF-8 bytes')
        selected=validate_recipe(object_json(text))
        require(selected['componentID']==component,'Dataset/DataLoader recipe input kind differs')
        if component=='data_loader' and '--batch-size' in options:
            require(str(selected['config']['batch_size'])==options['--batch-size'],'Captured batch size must mirror selected DataLoader batch_size')


def validate_model_input_nodes(options):
    if '--dataflow-model-input-nodes' not in options:return
    if __package__:from .preparation_demand_configuration import parse_complete_nodes_json
    else:from preparation_demand_configuration import parse_complete_nodes_json
    selected=parse_complete_nodes_json(options['--dataflow-model-input-nodes'])
    required=parse_complete_nodes_json(options.get('--dataflow-required-nodes','[]'))
    require(set(selected)<=set(required),'Model input nodes must be explicitly required preparation nodes')
