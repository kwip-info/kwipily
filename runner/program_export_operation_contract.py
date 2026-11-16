"""Framework-free admission under the existing retained one-shot operation lease."""
import os
from pathlib import Path
if __package__:
    from .worker_authorization import require,hash_value,canonical_uuid,directory,read_regular,object_json,sha
    from .worker_input_contract import verify_pinned_file,same_json
    from .execution_budget import validate_versioned_request
    from .program_export_configuration import normalize_program_export
else:
    from worker_authorization import require,hash_value,canonical_uuid,directory,read_regular,object_json,sha
    from worker_input_contract import verify_pinned_file,same_json
    from execution_budget import validate_versioned_request
    from program_export_configuration import normalize_program_export

SOURCE_NAMES={'model.py','checkpoint.pt','inference.py','input-contract.json','README.md','requirements.txt','training_recipe.py','rate-trace.json','objective_recipe.py','initialization_recipe.py','gradient_policy.py','trainability_recipe.py','data_loading_recipe.py'}
REQUIRED={'model.py','checkpoint.pt','inference.py','input-contract.json'}
INNER_FIELDS={'schemaVersion','kind','projectID','exportID','requestIdentity','sourceDirectory','sourceManifestSHA256','configuration'}


def file_limit(name):
    if name=='input/program-export.json':return 262144
    if name=='source/export-manifest.json':return 2_000_000
    if type(name) is str and name.startswith('source/') and name[7:] in SOURCE_NAMES:
        return 100_000_000 if name=='source/checkpoint.pt' else 4_000_000 if name=='source/rate-trace.json' else 131_072 if name=='source/data_loading_recipe.py' else 2_000_000
    raise ValueError('Unreviewed program source path')


def parse_contract(value,base=None):
    arguments=value['arguments']
    require(type(arguments) is list and len(arguments)==4 and arguments[0]=='--export-program' and arguments[2]=='--program-request-sha256','Program operation requires its exact request and pin arguments')
    path=arguments[1];require(type(path) is str and path.startswith('/') and len(path.encode())<=4096 and '\0' not in path and all(p and p not in ('.','..') for p in path.split('/')[1:]),'Invalid program request path')
    root=Path(base) if base is not None else Path(path).parent.parent
    require(path==str(root/'input/program-export.json') and root.name==value['operationID'] and root.parent.name==value['projectID'] and root.parent.parent.name=='workflow-operations','Program request escapes owned operation')
    hash_value(arguments[3])
    files=value['inputFiles'];require(type(files) is list and 6<=len(files)<=15,'Invalid program source inventory count')
    inventory={};total=0
    for row in files:
        require(type(row) is dict and set(row)=={'path','bytes','sha256'},'Invalid program inventory row')
        name=row['path'];limit=file_limit(name)
        require(name not in inventory and type(row['bytes']) is int and 0<row['bytes']<=limit,'Invalid program source size or duplicate')
        hash_value(row['sha256']);inventory[name]=row;total+=row['bytes']
    require(total<=130_000_000 and {'input/program-export.json','source/export-manifest.json'}|{'source/'+n for n in REQUIRED}<=set(inventory),'Program captured source inventory exceeds bounds or is incomplete')
    require(inventory['input/program-export.json']['sha256']==arguments[3],'Program inner request pin differs')
    return {'--export-program':path,'--program-request-sha256':arguments[3]},inventory


def verify_inputs(value,root,cache):
    _,inventory=parse_contract(value,root)
    for name,row in inventory.items():verify_pinned_file(root,name,row,cache)
    def read(folder,name,limit):
        fd=directory(str(Path(root)/folder))
        try:return read_regular(fd,name,limit)
        finally:os.close(fd)
    inner=object_json(read('input','program-export.json',262144));validate_versioned_request(inner,INNER_FIELDS)
    require(inner['kind']=='programExport' and inner['projectID']==value['projectID'] and inner['exportID']==value['operationID'] and inner['requestIdentity']==value['requestIdentity'],'Program request differs from retained operation identity')
    require(inner['sourceDirectory']==str(Path(root)/'source'),'Program source escapes retained operation')
    normalize_program_export(inner['configuration']);hash_value(inner['sourceManifestSHA256'])
    if inner['schemaVersion']==2:require(same_json(inner['executionBudget'],value['executionBudget']),'Inner program budget differs from managed operation')
    require(inner['sourceManifestSHA256']==inventory['source/export-manifest.json']['sha256'],'Program source manifest pin differs')
    manifest=object_json(read('source','export-manifest.json',2_000_000))
    require(type(manifest.get('schemaVersion')) is int and manifest['schemaVersion']==1 and manifest.get('kind')=='kwipmlTrainedModel','Unsupported program trained source manifest')
    canonical_uuid(manifest.get('runID'))
    for key in ('requestIdentity','modelManifestSHA256'):hash_value(manifest.get(key))
    rows=manifest.get('files');require(type(rows) is list and 4<=len(rows)<=13,'Invalid trained program source inventory')
    expected={'input/program-export.json','source/export-manifest.json'}
    for row in rows:
        require(type(row) is dict and set(row)=={'path','bytes','sha256'} and type(row['path']) is str and row['path'] in SOURCE_NAMES,'Invalid trained source artifact')
        name='source/'+row['path'];require(name not in expected and name in inventory,'Missing or duplicate trained source artifact')
        require(same_json(dict(row,path=name),inventory[name]),'Trained source inventory differs from operation')
        expected.add(name)
    require(set(inventory)==expected,'Extra input outside trained source manifest')
    contract=object_json(read('source','input-contract.json',2_000_000))
    require(type(contract.get('schemaVersion')) is int and contract['schemaVersion'] in (1,2),'Bag packet programs require existing packet inference export')
    spec=contract.get('input');require(type(spec) is dict and spec.get('dtype') in ('float32','int64'),'Unsupported program input type')
    shape=spec.get('shape');require(type(shape) is list and 2<=len(shape)<=8 and shape[0]=='N' and all(type(n) is int and 1<=n<=4096 for n in shape[1:]),'Invalid ordinary program input axes')
