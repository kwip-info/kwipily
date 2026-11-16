"""Minimized portable packages from native-authorized saved evaluation evidence.

The caller retains inactive saved-record read authority and disclosure permission.
This builder independently validates the supplied manifest pin and captured bytes.
It never copies an entire private snapshot or reads a live source.
"""
import errno
import io
import json
import os
from pathlib import Path

import torch

if __package__:
    from .artifact_storage import ArtifactStorage, ArtifactStorageBudget
    from .frozen_evaluation_cli import verified_request, read_nested, object_json, REFERENCE_FIELDS
    from .worker_authorization import directory, read_regular, require, hash_value
    from .engine import export_source
    from .dataflow_fitted import portable_parameters
    from .dataflow_storage import completed_inventory,prepared_table_document
    from . import portable_evaluation as portable
else:
    from artifact_storage import ArtifactStorage, ArtifactStorageBudget
    from frozen_evaluation_cli import verified_request, read_nested, object_json, REFERENCE_FIELDS
    from worker_authorization import directory, read_regular, require, hash_value
    from engine import export_source
    from dataflow_fitted import portable_parameters
    from dataflow_storage import completed_inventory,prepared_table_document
    import portable_evaluation as portable

EXPORT_STORAGE_BYTES=622_000_000
EXPORT_MANIFEST_BYTES=2_000_000

FIELDS={'schemaVersion','kind','status','evaluationID','projectID','requestIdentity','snapshotReceiptSHA256','settings','requestSHA256','parent','testEndpoint','testProvenance','frozenLineage','runtime','artifacts'}


def build_package(evaluation_root,destination,expected_manifest_sha256,include_test_rows=False,check_cancelled=lambda:None):
    require(type(include_test_rows) is bool,'Protected-row option must be boolean')
    check_cancelled();hash_value(expected_manifest_sha256)
    root=directory(os.path.abspath(os.fspath(evaluation_root)))
    try:
        manifest_raw=read_nested(root,'artifacts/manifest.json',2_000_000)
        require(portable.sha(manifest_raw)==expected_manifest_sha256,'Evaluation manifest changed')
        manifest=object_json(manifest_raw)
        require(set(manifest)==FIELDS and type(manifest['schemaVersion']) is int and manifest['schemaVersion'] in (1,2,3) and manifest['kind']=='frozenEvaluation' and manifest['status']=='completed','Unsupported completed evaluation manifest')
        request_pin=manifest['requestSHA256'];hash_value(request_pin)
    finally:os.close(root)
    budget=ArtifactStorageBudget(EXPORT_STORAGE_BYTES,check_cancelled)
    with verified_request(Path(evaluation_root)/'request.json',request_pin,check_cancelled=check_cancelled,storage_budget=budget,
                          scratch_parent=Path(os.path.abspath(os.fspath(destination))).parent) as verified:
        request=verified['request'];receipt=verified['receipt'];frozen=verified['frozen'];captured=verified['captured'];provenance=verified['provenance']
        require('namedInputPlan' not in frozen.input_contract,'Named-input evaluation replay packages are not supported yet; export the trained model package, which retains its named mapping and tokenizer.')
        role=request['settings'].get('partitionRole','test');row_key=role+'RowIDs'
        require(portable.sha(read_nested(verified['root'],'artifacts/manifest.json',2_000_000))==expected_manifest_sha256,'Evaluation manifest changed during capture')
        for key in ('evaluationID','projectID','requestIdentity','snapshotReceiptSHA256','settings'):
            require(manifest[key]==request[key],'Evaluation identity/settings differ from captured request')
        require(manifest['parent']==receipt['reference'] and manifest['testEndpoint']==receipt['testEndpoint'],'Evaluation parent or selected endpoint differs')
        require(manifest['testProvenance']=={k:v for k,v in provenance.items() if k!=row_key} and manifest['frozenLineage']==frozen.lineage,'Evaluation provenance differs from frozen computation')
        runtime=manifest['runtime'];require(isinstance(runtime,dict) and set(runtime)=={'python','torch','device'} and all(isinstance(v,str) and 1<=len(v)<=200 for v in runtime.values()) and runtime['device'] in ('cpu','mps'),'Invalid saved runtime evidence')
        portable.require_supported_evaluation(manifest['schemaVersion'],runtime['device'])
        entries=manifest['artifacts'];require(isinstance(entries,list) and len(entries)==(3 if manifest['schemaVersion']>=3 else 2),'Invalid evaluation output inventory')
        output={};limits={'summary.json':1_000_000,'predictions.jsonl':100_000_000,**({'prediction-table.json':50_000_000} if manifest['schemaVersion']>=3 else {})}
        for entry in entries:
            check_cancelled();require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid evaluation output entry')
            name=entry['path'];require(isinstance(name,str) and name in limits and name not in output,'Invalid output path')
            require(type(entry['bytes']) is int and 0<=entry['bytes']<=limits[name],'Invalid output bound');hash_value(entry['sha256'])
            raw=read_nested(verified['root'],'artifacts/'+name,limits[name]);require(len(raw)==entry['bytes'] and portable.sha(raw)==entry['sha256'],'Evaluation output changed')
            output[name]=raw
        original_summary=object_json(output['summary.json'])
        require(set(original_summary)==portable.METRICS|{'preview'},'Invalid saved summary fields')
        summary={k:v for k,v in original_summary.items() if k!='preview'}
        predictions=portable.predictions_valid(output['predictions.jsonl'],summary,provenance[row_key])
        require(summary['batchSize']==request['settings']['batchSize'] and summary['device']==runtime['device'],'Saved execution settings differ')
        require(original_summary['preview']==predictions[:request['settings']['previewLimit']],'Saved preview differs')
        require([p['target'] for p in predictions]==verified['y'].tolist(),'Saved targets differ from frozen prepared test')
        preparation,prepared_files,_=completed_inventory(captured/'preparation/artifacts/manifest.json',
            receipt['reference']['preparationManifestSHA256'],lazy=True,check_cancelled=check_cancelled)
        endpoint=receipt['testEndpoint'];selected=None;states={}
        for entry in preparation['artifacts']:
            check_cancelled()
            if entry.get('kind')=='state':
                value=object_json(prepared_files[entry['path']]);states[value['contentDigest']]=value
            elif entry.get('kind')=='table' and entry.get('nodeID')==endpoint['node'] and entry.get('port')==endpoint['port']:
                value=object_json(prepared_files[entry['path']])
                require(all(column.get('dtype')!='image' for column in value['columns']),
                    'Image-containing tables cannot be exported as scalar portable evaluation packages. Select scalar metadata columns before fitting and training; an image export adapter is not available yet.')
                selected=prepared_table_document(prepared_files,entry) if value.get('kind')=='segmentedTable' else value
        require(selected is not None,'Selected test table missing')
        require(all(column.get('dtype')!='image' for column in selected['columns']),
                'Image-containing tables cannot be exported as scalar portable evaluation packages. Select scalar metadata columns before fitting and training; an image export adapter is not available yet.')
        chain=provenance['preparation']['fittedStateDigests'];prefix=selected['lineage']['fittedStateDigests']
        params=[portable_parameters(states[state]) for state in chain]
        contract=dict(schemaVersion=1,kind='portableEvaluationInput',columns=selected['columns'],featureColumns=provenance['featureColumns'],targetColumn=provenance['targetColumn'],
            **frozen.input_contract,fittedStateDigests=chain,appliedPrefix=prefix,testEndpoint=endpoint,testCount=len(provenance[row_key]),membershipDigest=provenance['preparation']['membershipDigest'],selectedDataVersion=provenance['preparation']['selectedDataVersion'])
        if role=='validation':contract.update(schemaVersion=2,partitionRole='validation')
        portable.contract_valid(contract,params)
        weights=io.BytesIO();torch.save({k:v.detach().cpu().clone() for k,v in frozen.model.state_dict().items()},weights)
        files={'model.py':export_source(frozen.model).encode(),'weights.pt':weights.getvalue(),'input-contract.json':portable.encoded(contract),'fitted-parameters.json':portable.encoded(params),'saved-summary.json':portable.encoded(summary)}
        for name in ('portable_evaluation.py','portable_fitted.py','preprocess_parameters.py','dataflow_temporal.py','model_output_configuration.py','final_evaluation.py','execution_device.py'):
            files[name]=Path(__file__).with_name(name).read_bytes()
        files['requirements.txt']=('torch=='+str(torch.__version__).split('+')[0]+'\n').encode()
        files['README.md']=('''# Frozen evaluation package

This package contains executable model source and learned parameters. Use packages only from a trusted source. Inventory hashes detect inconsistency, not authorship.

Install requirements.txt in a Python environment, then run:

    python portable_evaluation.py --replay --batch-size 1024 --device cpu

Replay is available only when test rows were explicitly included. Otherwise provide a JSON table with exactly schemaVersion:1, columns, rows, observationIDs:

    python portable_evaluation.py --input observations.json --batch-size 1024 --device cpu

Use the exact ordered columns in input-contract.json. Values must obey each dtype/nullability; identities must be unique. The table represents the selected endpoint, including its already-applied transform prefix. Remaining learned transforms are applied without fitting. External results are labeled externalInput and are never claimed as original protected test replay.

Default exports contain aggregate saved metrics, model weights, schema and learned statistics/vocabularies. Learned parameters may reveal information about source data: this is minimization, not anonymization. Optional rows include only the selected protected population and its predictions. No optimizer state or training/validation rows are included.

Replay requires complete original membership, order and values. Counts, labels, confusion and class decisions must match exactly; logits and losses use relative tolerance 1e-5 and absolute tolerance 1e-6 across devices/batching. Numerical differences beyond this fail verification. Two-logit categorical cross-entropy is summed over every observation; mean loss divides by total observations. No refitting or automatic test-based model selection occurs.

Original checkpoint/state/evaluation hashes are preserved separately from derived portable file hashes. Local prior-inspection history is not transferred and this package does not prove an untouched holdout.
''').encode()
        if role=='validation':
            files['README.md']=files['README.md'].replace(b'test rows',b'Validation rows').replace(b'protected test replay',b'frozen Validation replay').replace(b'selected protected population',b'selected frozen Validation population').replace(b'No optimizer state or training/validation rows are included.',b'No optimizer state, training rows or protected Test rows are included.').replace(b'untouched holdout',b'untouched holdout; Validation was already used during training')
        if include_test_rows:
            table=dict(schemaVersion=1,columns=selected['columns'],rows=selected['rows'],observationIDs=selected['lineage']['observationIDs'])
            portable.table_valid(table,contract,True)
            files['test-input.json']=portable.encoded(table);files['saved-predictions.jsonl']=output['predictions.jsonl']
        identities=dict(evaluationID=request['evaluationID'],evaluationManifestSHA256=expected_manifest_sha256,checkpointSHA256=receipt['reference']['checkpointSHA256'],stateDigest=frozen.lineage['stateDigest'],snapshotReceiptSHA256=request['snapshotReceiptSHA256'])
        result=dict(schemaVersion=2,kind='portableFrozenEvaluation',includesTestRows=include_test_rows,identities=identities,files=[])
        if role=='validation':result.update(schemaVersion=3,partitionRole='validation')
        limits={**portable.LIMITS,'preprocess_parameters.py':200_000,'dataflow_temporal.py':200_000,'model_output_configuration.py':200_000,**(portable.OPTIONAL if include_test_rows else {})};total=0
        for name,raw in sorted(files.items()):
            require(len(raw)<=limits[name],'Portable artifact exceeds bound');total+=len(raw)
            result['files'].append(dict(path=name,bytes=len(raw),sha256=portable.sha(raw)))
        require(total<=280_000_000,'Portable package exceeds total budget');check_cancelled()
        # Exclusive directory creation; receipt last is the completion marker.
        destination=Path(os.path.abspath(os.fspath(destination)));parent=directory(str(destination.parent));fd=None;owned=None;storage=None;published={}
        try:
            check_cancelled();os.mkdir(destination.name,0o700,dir_fd=parent)
            fd=os.open(destination.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);owned=os.fstat(fd)
            storage=ArtifactStorage(fd,budget=budget,check_cancelled=check_cancelled)
            for name,raw in [*sorted(files.items()),('manifest.json',portable.encoded(result))]:
                info=storage.atomic_bytes(name,raw,EXPORT_MANIFEST_BYTES if name=='manifest.json' else limits[name])
                published[name]=(info.st_dev,info.st_ino)
            os.fsync(fd);check_cancelled()
            named=os.stat(destination.name,dir_fd=parent,follow_symlinks=False)
            require((owned.st_dev,owned.st_ino)==(named.st_dev,named.st_ino),'Portable destination changed')
            os.fsync(parent);return result
        except BaseException:
            if fd is not None:
                for name,identity in published.items():
                    try:
                        info=os.stat(name,dir_fd=fd,follow_symlinks=False)
                        if (info.st_dev,info.st_ino)==identity:os.unlink(name,dir_fd=fd)
                    except FileNotFoundError:pass
                try:
                    named=os.stat(destination.name,dir_fd=parent,follow_symlinks=False)
                    if (owned.st_dev,owned.st_ino)==(named.st_dev,named.st_ino):os.rmdir(destination.name,dir_fd=parent)
                except OSError as error:
                    if error.errno not in (errno.ENOENT,errno.ENOTEMPTY,errno.EEXIST):raise
            raise
        finally:
            if storage is not None:storage.close()
            if fd is not None:os.close(fd)
            os.close(parent)
