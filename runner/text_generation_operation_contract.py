"""Framework-free pinned source admission for the managed generation worker."""
import os
from pathlib import Path
if __package__:
    from .worker_authorization import require,hash_value,directory,read_regular,object_json
    from .worker_input_contract import verify_pinned_file,same_json
else:
    from worker_authorization import require,hash_value,directory,read_regular,object_json
    from worker_input_contract import verify_pinned_file,same_json
SOURCE_NAMES={'model.py','checkpoint.pt','inference.py','input-contract.json','README.md','requirements.txt','training_recipe.py','rate-trace.json','objective_recipe.py','initialization_recipe.py','gradient_policy.py','trainability_recipe.py','data_loading_recipe.py','named_inputs.py','tokenizers.json','portable_text_encoding.py'}


def file_limit(name):
    if name=='input/generation-kernel.py':return 200_000
    if name=='input/generation.json':return 2_000_000
    if name=='input/prompts.table.json':return 8_000_000
    if name=='source/export-manifest.json':return 2_000_000
    require(type(name) is str and name.startswith('source/') and name[7:] in SOURCE_NAMES,'Unreviewed generation source path')
    return 100_000_000 if name=='source/checkpoint.pt' else 4_000_000 if name=='source/rate-trace.json' else 2_000_000


def parse_contract(value,base=None):
    a=value['arguments'];require(type(a) is list and len(a)==2 and a[0]=='--generate-text','Generation needs exact managed arguments')
    root=Path(base) if base is not None else Path(a[1]).parent.parent
    require(a[1]==str(root/'input/generation.json') and root.name==value['operationID'] and root.parent.name==value['projectID'] and root.parent.parent.name=='workflow-operations','Generation path escapes owned operation')
    files=value['inputFiles'];require(type(files) is list and 9<=len(files)<=20,'Invalid generation inventory')
    inventory={};total=0
    for f in files:
        require(type(f) is dict and set(f)=={'path','bytes','sha256'},'Invalid generation artifact entry')
        name=f['path'];limit=file_limit(name)
        require(name not in inventory and type(f['bytes']) is int and 0<f['bytes']<=limit,'Invalid generation artifact size or duplicate');hash_value(f['sha256']);inventory[name]=f;total+=f['bytes']
    required={'input/generation.json','input/generation-kernel.py','source/export-manifest.json'}|{'source/'+n for n in ('model.py','checkpoint.pt','inference.py','input-contract.json','named_inputs.py','tokenizers.json','portable_text_encoding.py')}
    require(required<=set(inventory) and total<=140_000_000,'Missing captured generation authority or excess bytes')
    return {'--generate-text':a[1]},inventory


def verify_inputs(value,root,cache):
    _,inventory=parse_contract(value,root)
    for name,f in inventory.items():verify_pinned_file(root,name,f,cache)
    def read(folder,name,limit):
        fd=directory(str(Path(root)/folder))
        try:return read_regular(fd,name,limit)
        finally:os.close(fd)
    inner=object_json(read('input','generation.json',2_000_000))
    require(set(inner)=={'schemaVersion','projectID','operationID','requestIdentity','configuration','sourceManifestSHA256'} and inner['schemaVersion']==1 and inner['projectID']==value['projectID'] and inner['operationID']==value['operationID'] and inner['requestIdentity']==value['requestIdentity'],'Generation identity mismatch')
    require(inner['sourceManifestSHA256']==inventory['source/export-manifest.json']['sha256'],'Generation source pin differs')
    m=object_json(read('source','export-manifest.json',2_000_000));require(m.get('schemaVersion')==1 and m.get('kind')=='kwipmlTrainedModel' and type(m.get('files')) is list,'Invalid generation model manifest')
    expected={'input/generation.json','input/generation-kernel.py','source/export-manifest.json'}|({'input/prompts.table.json'} if 'input/prompts.table.json' in inventory else set())
    for f in m['files']:
        require(type(f) is dict and set(f)=={'path','bytes','sha256'} and f['path'] in SOURCE_NAMES,'Unreviewed model source')
        name='source/'+f['path'];require(name not in expected and name in inventory and same_json(dict(f,path=name),inventory[name]),'Model inventory differs');expected.add(name)
    require(set(inventory)==expected,'Extra generation input')
    contract=object_json(read('source','input-contract.json',2_000_000));require(contract.get('schemaVersion')==4,'Generation requires named sequence model')
    objective=contract.get('objective',{});recipe=objective.get('recipe',{})
    require(recipe.get('componentID')=='loss_cross_entropy' and recipe.get('configurationVersion')==1 and recipe.get('schemaVersion')==1 and recipe.get('config')==dict(class_axis=-1,class_weights=[],ignore_index=-100,label_smoothing=0,reduction='mean',target_mode='class_indices') and objective.get('scalarization')=='mean','Generation requires canonical next-token objective')
    return inner
