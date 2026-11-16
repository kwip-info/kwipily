"""Offline model tasks run under the same retained, budgeted operation worker."""
import json
from pathlib import Path
if __package__:
    from . import pretrained_models as models
    from .model_asset_operation_contract import verify_inputs
else:
    import pretrained_models as models
    from model_asset_operation_contract import verify_inputs


def execute(authority,emit,check_cancelled):
    from worker_resources import configure_torch
    import torch
    configure_torch(torch)
    root=Path(authority._path);inner=verify_inputs(authority.request,root,{})
    source=root/'source';action=inner['action'];settings=inner['settings']
    info=models.inspect_model(source,check_cancelled=check_cancelled)
    if info['identity']!=inner['sourceIdentity']:raise ValueError('Model data differs from the captured source identity.')
    output=root/'output';output.mkdir(mode=0o700,exist_ok=True)
    files=[]
    def write(name,value):
        import hashlib
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>8_000_000:raise ValueError('Model result exceeds its byte budget.')
        path=output/name
        with path.open('xb') as stream:stream.write(raw)
        files.append(dict(path='output/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    def exact(required,optional=()):
        if not set(required)<=set(settings) or not set(settings)<=set(required)|set(optional):raise ValueError('Unknown or missing model operation setting.')
    if action=='inspect':
        exact([])
        if info['task']=='causalLanguageModel':
            if __package__:from .pretrained_training import training_inventory
            else:from pretrained_training import training_inventory
            info['training']=training_inventory(source,check_cancelled=check_cancelled,info=info)
            info['capabilities']+=['fineTuneFull','fineTuneSelected']+(['fineTuneLoRA'] if 'lora' in info['training']['supportedModes'] else [])
        write('model-metadata.json',info)
    elif action in ('train','prepare','evaluate'):
        if info['task']=='typedDecisions':
            if __package__:from .laya_training_worker import execute as execute_typed
            else:from laya_training_worker import execute as execute_typed
            execute_typed(authority,inner,info,output,files,write,emit,check_cancelled)
        else:
            if __package__:
                from .pretrained_training import fine_tune
                from .dataflow_table_reader import read_table_artifact
            else:
                from pretrained_training import fine_tune
                from dataflow_table_reader import read_table_artifact
            texts={}
            roles=('training','evaluation','evaluationPopulation') if action=='evaluate' else ('training','validation')
            for role in roles:
                path='input/'+role+'.table.json'
                pin=next(f['sha256'] for f in authority.request['inputFiles'] if f['path']==path)
                table=read_table_artifact(root/path,pin,check_cancelled)
                try:
                    if role=='evaluationPopulation':
                        if table.role!=settings.get('evaluationRole') or not table.partition_digest:raise ValueError('Evaluation needs its exact originating protected Partition population.')
                        texts['populationIDs']=set(table.row_ids);texts['populationSource']=table.source_digest
                        continue
                    names=[settings.get('promptColumn','prompt'),settings.get('responseColumn','response')] if settings.get('exampleFormat','text')=='promptResponse' else [settings['textColumn']]
                    choices=[]
                    for name in names:
                        found=[i for i,c in enumerate(table.columns) if c['name']==name and c['dtype']=='string']
                        if len(found)!=1:raise ValueError('Choose String columns present in the captured partition.')
                        choices.append(found[0])
                    # Partition role and membership must remain disjoint. Merely using
                    # two wires from the same table is not held-out validation.
                    expected=settings.get('evaluationRole','validation') if role.startswith('evaluation') else role
                    if table.role != expected and (action=='evaluate' or table.role!='unassigned'):raise ValueError('Connect the declared Training/Validation partition, never the protected Test table.')
                    texts[role]=[dict(prompt=row[choices[0]],response=row[choices[1]]) if len(choices)==2 else row[choices[0]] for row in table.rows]
                    texts[role+'IDs']=set(table.row_ids)
                    if role=='evaluation':
                        texts['evaluationSource']=table.source_digest
                        if not 1<=len(texts['evaluationIDs'])<=100000:raise ValueError('Evaluation exposure tracking accepts 1–100000 observations.')
                finally:
                    table.rows.store.close()
                    table.row_ids.store.close()
            if texts['trainingIDs'] & texts['evaluationIDs' if action=='evaluate' else 'validationIDs']:raise ValueError('Training and Validation observations overlap. Connect disjoint Partition outputs.')
            if action=='evaluate' and (texts['populationIDs']!=texts['evaluationIDs'] or texts['populationSource']!=texts['evaluationSource']):raise ValueError('Evaluation rows differ from the complete declared protected population; filtered or sampled holdouts cannot be scored as complete.')
            allowance=(authority.request['executionBudget'].get('resources') or {}).get('maximumResidentBytes') or 2_000_000_000
            settings=dict(settings,maximumResidentBytes=min(settings['maximumResidentBytes'],allowance))
            settings=dict(settings);resume_mode=settings.pop('resumeMode',None);evaluation_role=settings.pop('evaluationRole',None)
            if resume_mode is not None and resume_mode!='exact':raise ValueError('Unsupported continuation operation.')
            resume_root=root/'input/resume' if resume_mode else None
            if action=='prepare':
                if __package__:from .pretrained_training import prepared_preview
                else:from pretrained_training import prepared_preview
                write('tokens.json',prepared_preview(source,texts['training'],texts['validation'],settings,check_cancelled))
            elif action=='evaluate':
                if evaluation_role not in ('validation','test'):raise ValueError('Choose explicit Validation or Test scoring.')
                if __package__:from .pretrained_training import evaluate
                else:from pretrained_training import evaluate
                summary=evaluate(source,texts['evaluation'],settings,check_cancelled);rows=summary.pop('rows')
                summary.update(kind='causalEvaluation',role=evaluation_role,observationIDs=sorted(texts['evaluationIDs']),sourceDigest=texts['evaluationSource'],modelIdentity=info['identity'])
                write('tokens.json',summary)
                columns=[dict(name=name,dtype=dtype,nullable=name=='perplexity') for name,dtype in [('window','int64'),('input_ids','int64Sequence'),('target_ids','int64Sequence'),('predicted_ids','int64Sequence'),('tokens','int64'),('loss','float64'),('perplexity','float64')]]
                if __package__:from . import dataflow as df
                else:import dataflow as df
                ids=[info['identity']+':'+evaluation_role+':'+str(i) for i in range(len(rows))]
                if __package__:from .image_artifacts import table_schema_version
                else:from image_artifacts import table_schema_version
                write('table.json',dict(schemaVersion=table_schema_version(columns),kind='table',columns=columns,rows=rows,lineage=dict(role=evaluation_role,sourceDigest=info['identity'],membershipDigest=df.membership_digest(ids),observationIDs=ids,dataVersion=df.digest(dict(columns=columns,rows=rows,ids=ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[])))
            else:
                def pause_requested():
                    marker=root/'pause-after-epoch'
                    if not marker.exists():return False
                    if __package__:from .worker_authorization import directory,read_regular
                    else:from worker_authorization import directory,read_regular
                    import os
                    fd=directory(str(root))
                    try:return read_regular(fd,'pause-after-epoch',64)==b'pause'
                    finally:os.close(fd)
                report=fine_tune(source,output/'training',texts['training'],texts['validation'],settings,emit=emit,check_cancelled=check_cancelled,run_id=inner['operationID'],resume_from=resume_root if resume_mode=='exact' else None,pause_requested=pause_requested)
            if action=='train':
                from hashlib import sha256
                for path in sorted((output/'training').rglob('*')):
                    if not path.is_file():continue
                    check_cancelled();h=sha256()
                    with path.open('rb') as f:
                        while chunk:=f.read(1_048_576):check_cancelled();h.update(chunk)
                    files.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h.hexdigest()))
                metadata=models.inspect_model(output/'training/model',check_cancelled=check_cancelled)
                write('model-metadata.json',metadata)
                if settings.get('capturePredictions'):
                    if __package__:
                        from .pretrained_training import prediction_rows
                        from . import dataflow as df
                    else:
                        from pretrained_training import prediction_rows
                        import dataflow as df
                    rows=prediction_rows(output/'training/model',texts['validation'],settings,check_cancelled)
                    columns=[dict(name=name,dtype=dtype,nullable=name=='perplexity') for name,dtype in [('window','int64'),('input_ids','int64Sequence'),('target_ids','int64Sequence'),('predicted_ids','int64Sequence'),('tokens','int64'),('loss','float64'),('perplexity','float64')]]
                    ids=[metadata['identity']+':validation:'+str(i) for i in range(len(rows))]
                    if __package__:from .image_artifacts import table_schema_version
                    else:from image_artifacts import table_schema_version
                    write('table.json',dict(schemaVersion=table_schema_version(columns),kind='table',columns=columns,rows=rows,lineage=dict(role='validation',sourceDigest=metadata['identity'],membershipDigest=df.membership_digest(ids),observationIDs=ids,dataVersion=df.digest(dict(columns=columns,rows=rows,ids=ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[])))
    elif action=='attachAdapter':
        exact(['maximumResidentBytes'])
        if __package__:
            from .pretrained_training import load_adapter,_copy_metadata
        else:
            from pretrained_training import load_adapter,_copy_metadata
        allowance=(authority.request['executionBudget'].get('resources') or {}).get('maximumResidentBytes') or 2_000_000_000
        allowance=min(allowance,settings['maximumResidentBytes'])
        adapter=root/'input/adapter'
        adapter_bytes=sum(f['bytes'] for f in authority.request['inputFiles'] if f['path'].startswith('input/adapter/'))
        if info['parameterCount']*16+adapter_bytes*4+64_000_000>allowance:raise ValueError('Adapter attachment/merge exceeds the captured memory allowance.')
        model=load_adapter(adapter,source,check_cancelled=check_cancelled,maximum_resident_bytes=allowance).merge_and_unload(safe_merge=True)
        target=output/'model';target.mkdir(mode=0o700);model.save_pretrained(target,safe_serialization=True,max_shard_size='2GB');_copy_metadata(source,target)
        from hashlib import sha256
        for path in sorted(target.iterdir()):
            check_cancelled();h=sha256()
            with path.open('rb') as f:
                while chunk:=f.read(1_048_576):check_cancelled();h.update(chunk)
            files.append(dict(path='output/model/'+path.name,bytes=path.stat().st_size,sha256=h.hexdigest()))
        write('model-metadata.json',models.inspect_model(target,check_cancelled=check_cancelled))
    elif action=='tokenize':
        exact(['texts','maximumTokens','truncation','padding','paddingSide'])
        value=models.tokenize(source,settings['texts'],max_tokens=settings['maximumTokens'],truncation=settings['truncation'],padding=settings['padding'],padding_side=settings['paddingSide'],check_cancelled=check_cancelled)
        write('tokens.json',value)
    elif action=='export':
        exact(['shardBytes'])
        value=models.export_model(source,output/'model',shard_bytes=settings['shardBytes'],check_cancelled=check_cancelled)
        from hashlib import sha256
        for path in sorted((output/'model').rglob('*')):
            if not path.is_file():continue
            check_cancelled();h=sha256()
            with path.open('rb') as f:
                while chunk:=f.read(1_048_576):check_cancelled();h.update(chunk)
            files.append(dict(path='output/model/'+str(path.relative_to(output/'model')),bytes=path.stat().st_size,sha256=h.hexdigest()))
        write('model-metadata.json',value)
    else:
        exact(['configuration','maximumTokens','device'])
        if __package__:
            from .text_generation import settings as validate_settings,choose
            from . import dataflow as df
            from .image_artifacts import table_schema_version
        else:
            from text_generation import settings as validate_settings,choose
            import dataflow as df
            if __package__:from .image_artifacts import table_schema_version
            else:from image_artifacts import table_schema_version
        full=dict(settings['configuration'])
        required={'promptMode','prompt','promptColumn','textColumn','idsColumn','countColumn','stopColumn','settings'}
        if set(full)!=required:raise ValueError('Invalid captured generation configuration.')
        config=validate_settings(full['settings'])
        names=[full[key] for key in ('textColumn','idsColumn','countColumn','stopColumn')]
        if len(set(names))!=4 or any(type(name) is not str or not name.strip() or len(name.encode())>200 for name in names):raise ValueError('Choose distinct bounded output columns.')
        if full['promptMode']=='table':
            if __package__:from .dataflow_table_reader import read_table_artifact
            else:from dataflow_table_reader import read_table_artifact
            pin=next(f['sha256'] for f in authority.request['inputFiles'] if f['path']=='input/prompts.table.json')
            table=read_table_artifact(root/'input/prompts.table.json',pin,check_cancelled)
            columns=list(table.columns);original=[list(row) for row in table.rows];row_ids=list(table.row_ids)
            choices=[i for i,c in enumerate(columns) if c['name']==full['promptColumn'] and c['dtype']=='string']
            if len(choices)!=1:raise ValueError('Choose a captured text prompt column.')
            texts=[row[choices[0]] for row in original]
        elif full['promptMode']=='one':
            texts=[full['prompt']];columns=[dict(name='prompt',dtype='string',nullable=False)];original=[[full['prompt']]];row_ids=['prompt-0']
        else:raise ValueError('Unsupported prompt source.')
        if set(names)&{c['name'] for c in columns} or any(c['dtype']=='image' for c in columns):raise ValueError('Keep scalar/token source columns and give outputs distinct names.')
        if not 1<=len(texts)<=config['maxPrompts'] or len(texts)*config['maxNewTokens']>config['maxTotalTokens'] or any(type(x) is not str or len(x.encode())>262144 for x in texts) or sum(len(x.encode()) for x in texts)>1_048_576:raise ValueError('Prompts exceed the captured row/token/byte budget; sample upstream.')
        maximum=settings['maximumTokens']
        if type(maximum) is not int or not 1<=maximum<=info['maximumTokens']:raise ValueError('Generation context exceeds model context.')
        native_config=models._json(source/'config.json')
        heads=native_config.get('n_head',native_config.get('num_attention_heads',1))
        estimate=info['weightBytes']*4+64_000_000+maximum*info['vocabularySize']*4+maximum*maximum*heads*8
        allowance=(authority.request['executionBudget'].get('resources') or {}).get('maximumResidentBytes') or 2_000_000_000
        if estimate>allowance:raise ValueError('Estimated model/context memory exceeds the captured allowance. Reduce Maximum tokens or increase the project memory limit.')
        model=models.load_model(source,device=settings['device'],maximum_resident_bytes=allowance,check_cancelled=check_cancelled)
        tokenizer,special=models._tokenizer(source);tokenizer.no_padding();tokenizer.no_truncation()
        rows=[];records=[]
        for index,text in enumerate(texts):
            ids=tokenizer.encode(text,add_special_tokens=True).ids
            if not ids:raise ValueError('Prompt must encode at least one token.')
            if len(ids)>maximum and config['longPrompt']=='refuse':raise ValueError('Prompt exceeds context. Choose Keep latest context explicitly.')
            ids=ids[-maximum:]
            generator=torch.Generator(device='cpu').manual_seed((config['seed']+index)%(2**63))
            generated=[];reason='tokenLimit'
            for step in range(config['maxNewTokens']):
                check_cancelled();context=ids[-maximum:]
                with torch.no_grad():scores=model(input_ids=torch.tensor([context],device=settings['device']),attention_mask=torch.ones((1,len(context)),dtype=torch.long,device=settings['device']),use_cache=False).logits[0,-1].float().cpu()
                excluded={v for v in (special['pad_token'],special['bos_token']) if v is not None}
                if not config['allowUnknown'] and special.get('unk_token') is not None:excluded.add(special['unk_token'])
                chosen=choose(scores,excluded,config,generator)
                generated.append(chosen);ids.append(chosen)
                if config['stopAtEOS'] and special['eos_token']==chosen:reason='eos';break
            decoded=tokenizer.decode(ids,skip_special_tokens=True)
            rows.append(original[index]+[decoded,generated,len(generated),reason])
            records.append(dict(generated_text=decoded,generated_ids=generated,new_tokens=len(generated),stop_reason=reason))
            emit(dict(event='modelAssetProgress',operationID=inner['operationID'],completed=index+1,total=len(texts)))
        columns=columns+[dict(name=n,dtype=t,nullable=False) for n,t in zip(names,['string','int64Sequence','int64','string'])]
        table=dict(schemaVersion=table_schema_version(columns),kind='table',columns=columns,rows=rows,lineage=dict(role='unassigned',sourceDigest=info['identity'],membershipDigest=df.membership_digest(row_ids),observationIDs=row_ids,dataVersion=df.digest(dict(columns=columns,rows=rows,ids=row_ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[]))
        write('table.json',table)
        write('generation.json',dict(schemaVersion=1,modelIdentity=info['identity'],settings=config,rows=records))
    check_cancelled()
    emit(dict(event='modelAssetCompleted',operationID=inner['operationID'],requestIdentity=inner['requestIdentity'],files=files))
