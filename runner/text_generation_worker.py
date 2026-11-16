"""Managed inference entry; no callers' source paths or unverified Python imports."""
import hashlib
import types
import json
from pathlib import Path
if __package__:
    from .text_generation import generate
    from .text_generation_operation_contract import verify_inputs
    from .dataflow_storage import bounded_regular_read
    from . import dataflow as df
else:
    from text_generation import generate
    from text_generation_operation_contract import verify_inputs
    from dataflow_storage import bounded_regular_read
    import dataflow as df


def execute(authority,emit,check):
    root=Path(authority._path);request=authority.request
    inner=verify_inputs(request,root,{})
    c=inner['configuration'];c=dict(c);mode=c.pop('promptMode');prompt=c.pop('prompt');column=c.pop('promptColumn')
    names=[c.pop(k) for k in ('textColumn','idsColumn','countColumn','stopColumn')]
    if len(set(names))!=4 or any(type(n) is not str or not n.strip() or len(n.encode())>200 for n in names):raise ValueError('Choose distinct output column names.')
    if mode=='one':
        columns=[dict(name='prompt',dtype='string',nullable=False)];rows=[[prompt]];ids=['prompt-0']
    elif mode=='table':
        if __package__:from .dataflow_table_reader import read_table_artifact
        else:from dataflow_table_reader import read_table_artifact
        pin=next(f['sha256'] for f in request['inputFiles'] if f['path']=='input/prompts.table.json')
        table=read_table_artifact(root/'input/prompts.table.json',pin,check)
        if not 1<=len(table.rows)<=128:raise ValueError('Prompt table must contain 1–128 complete stable rows; sample upstream.')
        columns=list(table.columns);rows=[list(row) for row in table.rows];ids=list(table.row_ids)
    else:raise ValueError('Unknown prompt mode.')
    if any(col.get('dtype')=='image' for col in columns):raise ValueError('Keep the prompt and scalar/token columns with Columns before generation. Image-asset transport is not admitted by this text path.')
    if set(names)&{col['name'] for col in columns}:raise ValueError('Output column names collide with source columns; rename the outputs.')
    if mode=='table':
        choices=[i for i,col in enumerate(columns) if col['name']==column and col['dtype']=='string']
        if len(choices)!=1:raise ValueError('Choose one String prompt column.')
        prompts=[row[choices[0]] for row in rows]
    else:prompts=[prompt]
    # Only app-generated, inventory-verified source is importable in this worker.
    inference_bytes=bounded_regular_read(root/'source/inference.py',2_000_000)
    expected=next(f for f in request['inputFiles'] if f['path']=='source/inference.py')
    if len(inference_bytes)!=expected['bytes'] or hashlib.sha256(inference_bytes).hexdigest()!=expected['sha256']:raise ValueError('Inference source changed before loading.')
    module=types.ModuleType('kwipml_generation_inference');module.__file__=str(root/'source/inference.py')
    exec(compile(inference_bytes,module.__file__,'exec'),module.__dict__)
    predictor=module.load_model();check()
    def cancelled():check();return False
    result=generate(predictor,prompts,c,cancelled=cancelled,progress=lambda done,total:emit(dict(event='generationProgress',operationID=request['operationID'],completed=done,total=total)))
    columns=columns+[dict(name=n,dtype=t,nullable=False) for n,t in zip(names,['string','int64Sequence','int64','string'])]
    appended=[row+[r['generated_text'],r['generated_ids'],r['new_tokens'],r['stop_reason']] for row,r in zip(rows,result['rows'])]
    pin=inner['sourceManifestSHA256']
    if __package__:from .image_artifacts import table_schema_version
    else:from image_artifacts import table_schema_version
    table=dict(schemaVersion=table_schema_version(columns),kind='table',columns=columns,rows=appended,lineage=dict(role='unassigned',sourceDigest=pin,membershipDigest=df.membership_digest(ids),observationIDs=ids,dataVersion=df.digest(dict(columns=columns,rows=appended,ids=ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[]))
    output=root/'output';output.mkdir(mode=0o700)
    inventory=[]
    for name,value in [('table.json',table),('generation.json',result)]:
        raw=json.dumps(value,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>8_000_000:raise ValueError('Generation artifact exceeds its byte allowance.')
        check()
        with (output/name).open('xb') as stream:stream.write(raw)
        inventory.append(dict(path='output/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    check();verify_inputs(request,root,{})
    emit(dict(event='generationCompleted',operationID=request['operationID'],requestIdentity=request['requestIdentity'],files=inventory))
