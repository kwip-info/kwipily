"""Data-only, offline model operation authority; never admits source Python."""
import os
from pathlib import Path
import re
if __package__:
    from .worker_authorization import require,hash_value,directory,read_regular,object_json
    from .worker_input_contract import verify_pinned_file,same_json
else:
    from worker_authorization import require,hash_value,directory,read_regular,object_json
    from worker_input_contract import verify_pinned_file,same_json

if __package__:from .table_transfer_policy import MAX_ARTIFACT_BYTES
else:from table_transfer_policy import MAX_ARTIFACT_BYTES

TABLE_INPUTS={'input/training.table.json','input/validation.table.json'}
ALL_TABLE_INPUTS=TABLE_INPUTS|{'input/evaluation.table.json','input/evaluationPopulation.table.json'}

def segment_path(name):
    return type(name) is str and re.fullmatch(r'input/table-pack-[0-9a-f]{32}\.kwipdata',name) is not None

def resume_path(name):
    if type(name) is not str:return False
    p=name.split('/')
    leaf=lambda n:allowed(n) or bool(re.fullmatch(r'training-state(?:-\d{5}-of-\d{5})?\.safetensors',n)) or n in {'adapter_config.json','adapter_model.safetensors','kwip-adapter.json','kwip-continuation.json','training.json','result-manifest.json'}
    return len(p)==3 and p[:2]==['input','resume'] and p[2] in ('training.json','result-manifest.json') or len(p)==4 and p[:2]==['input','resume'] and p[2] in ('latest','model') and leaf(p[3])

NAMES={'rl_agent_config.json','encoder/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json','tokenizer/added_tokens.json','kwip-decisions.json','preprocessor_config.json','normalizer.json','kwip-training.json','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','generation_config.json','added_tokens.json','vocab.json','merges.txt','tokenizer.model','model.safetensors.index.json','README.md','LICENSE','LICENSE.txt','NOTICE','NOTICE.txt','kwip-manifest.json'}

def allowed(name):
    return name in NAMES or bool(re.fullmatch(r'model(?:-\d{5}-of-\d{5})?\.safetensors',name))

def parse_contract(value,base=None):
    args=value['arguments'];require(type(args) is list and len(args)==2 and args[0]=='--model-asset','Model asset needs exact managed arguments')
    root=Path(base) if base is not None else Path(args[1]).parent.parent
    require(args[1]==str(root/'input/model-request.json') and root.name==value['operationID'] and root.parent.name==value['projectID'] and root.parent.parent.name=='workflow-operations','Model operation path escapes its owner')
    files=value['inputFiles'];require(type(files) is list and 4<=len(files)<=512,'Invalid model file inventory')
    inventory={};total=0
    for f in files:
        require(type(f) is dict and set(f)=={'path','bytes','sha256'},'Invalid model file entry')
        name=f['path'];is_source=isinstance(name,str) and name.startswith('source/') and allowed(name[7:])
        require(is_source or segment_path(name) or (isinstance(name,str) and name.startswith('input/adapter/') and name[14:] in {'kwip-training.json','adapter_config.json','adapter_model.safetensors','kwip-adapter.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','tokenizer.model','README.md','LICENSE','LICENSE.txt','NOTICE','NOTICE.txt'}) or resume_path(name) or name in ('input/evaluationPopulation.table.json','input/evaluation.table.json','input/model-request.json','input/prompts.table.json','input/training.table.json','input/validation.table.json'),'Unreviewed model path')
        require(name not in inventory and type(f['bytes']) is int and 0<f['bytes']<=(2_000_000_000 if name.endswith('.safetensors') else MAX_ARTIFACT_BYTES if name in ALL_TABLE_INPUTS or segment_path(name) else 8_000_000),'Invalid model file size/duplicate')
        hash_value(f['sha256']);inventory[name]=f;total+=f['bytes']
    require(total<=4_120_000_000 and {'input/model-request.json'}<=set(inventory) and (bool({'source/config.json'} & set(inventory)) and bool({'source/tokenizer.json','source/preprocessor_config.json'} & set(inventory)) or {'source/rl_agent_config.json','source/encoder/config.json','source/tokenizer/tokenizer.json','source/tokenizer/tokenizer_config.json'}<=set(inventory)) and any(n.endswith('.safetensors') for n in inventory),'Missing safe model authority or exceeded byte budget')
    return {'--model-asset':args[1]},inventory

def verify_inputs(value,root,cache):
    _,inventory=parse_contract(value,root)
    for name,f in inventory.items():verify_pinned_file(root,name,f,cache)
    fd=directory(str(Path(root)/'input'))
    try:inner=object_json(read_regular(fd,'model-request.json',2_000_000))
    finally:os.close(fd)
    require(set(inner)=={'schemaVersion','projectID','operationID','requestIdentity','action','settings','sourceIdentity'} and type(inner['schemaVersion']) is int and inner['schemaVersion'] in (1,2,3,4) and inner['projectID']==value['projectID'] and inner['operationID']==value['operationID'] and inner['requestIdentity']==value['requestIdentity'],'Model operation identity differs')
    require(inner['action'] in ('inspect','tokenize','generate','export','train','attachAdapter','prepare','evaluate') and isinstance(inner['settings'],dict),'Unsupported model action')
    segments={n for n in inventory if segment_path(n)}
    require(not segments or inner['schemaVersion']==4,'Segmented model tables require capture version4')
    require(inner['schemaVersion']!=4 or bool(segments),'Segment capture requires pinned segments')
    if inner['schemaVersion']==1:
        require(sum(f['bytes'] for f in inventory.values())<=4_020_000_000 and all(n.endswith('.safetensors') or f['bytes']<=8_000_000 for n,f in inventory.items()),'Legacy model capture exceeds its original byte budget')
    elif inner['schemaVersion'] in (3,4):
        require(inner['action'] in ('train','prepare','evaluate') and 'input/prompts.table.json' not in inventory,'Invalid expanded model action')
        require((inner['action']=='evaluate' and {'input/training.table.json','input/evaluation.table.json','input/evaluationPopulation.table.json'}<=set(inventory)) or (inner['action'] in ('train','prepare') and TABLE_INPUTS<=set(inventory)),'Missing expanded table inputs')
    else:
        require(inner['action']=='train' and TABLE_INPUTS<=set(inventory) and 'input/prompts.table.json' not in inventory,'Owned table capture requires Training and Validation only')
    source_fd=directory(str(Path(root)/'source'))
    try:source_config=object_json(read_regular(source_fd,'rl_agent_config.json' if 'source/rl_agent_config.json' in inventory else 'config.json',8000000))
    finally:os.close(source_fd)
    ast=source_config.get('model_type')=='audio-spectrogram-transformer' and source_config.get('architectures')==['ASTForAudioClassification']
    require(({'source/rl_agent_config.json','source/encoder/config.json','source/tokenizer/tokenizer.json','source/tokenizer/tokenizer_config.json'}<=set(inventory) if 'source/rl_agent_config.json' in inventory else ('source/preprocessor_config.json' if ast else 'source/tokenizer.json') in inventory),'Model source lacks its task-matched processor/tokenizer authority')
    action=inner['action'];names=set(inventory)
    expected_tables={'input/training.table.json','input/evaluation.table.json','input/evaluationPopulation.table.json'} if action=='evaluate' else TABLE_INPUTS if action in ('train','prepare') else {'input/prompts.table.json'} if action=='generate' else set()
    require(not (names & (ALL_TABLE_INPUTS|{'input/prompts.table.json'}))-expected_tables,'Unexpected table authority for model action')
    resumed={n for n in names if n.startswith('input/resume/')}
    mode=inner['settings'].get('resumeMode')
    require((not resumed and mode is None) or (inner['schemaVersion'] in (3,4) and action=='train' and mode=='exact' and {'input/resume/training.json','input/resume/result-manifest.json','input/resume/latest/kwip-continuation.json'}<=resumed and all('/model/' not in n for n in resumed)),'Invalid or unused resume authority')
    require(action=='attachAdapter' or not any(n.startswith('input/adapter/') for n in names),'Unexpected adapter authority')
    hash_value(inner['sourceIdentity'])
    return inner
