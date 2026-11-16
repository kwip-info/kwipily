"""Prepared-data bridge for frozen text-feature classification; existing worker custody."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
if __package__:
    from . import dataflow as df, document_classification as dc
    from .dataflow_storage import completed_inventory,admit_model_ports
    from .dataflow_table_reader import read_table_artifact
else:
    import dataflow as df
    import document_classification as dc
    from dataflow_storage import completed_inventory,admit_model_ports
    from dataflow_table_reader import read_table_artifact


def write_prediction_table(stage,table,model,prediction,valid,config,check_cancelled,target_column=None):
    """Complete typed working table; presentation belongs to the shared table viewer."""
    dc.require(all(c['dtype'] in ('string','float64','int64','bool') for c in table.columns),'Prediction outputs preserve scalar columns; select text/identity/labels before classification.')
    results=[None]*len(table.rows)
    for j,i in enumerate(valid):
        probabilities=prediction['probabilities'][j]
        results[i]=(prediction['classes'][j],max(probabilities),prediction['noFeatures'][j],probabilities)
    source_columns=list(table.columns);names=[c['name'] for c in source_columns]
    identity=[i for i,c in enumerate(source_columns) if c['name'] in (target_column,'name','relative_path','document','id')]
    remainder=[i for i in range(len(source_columns)) if i not in identity]
    cols=[source_columns[i] for i in identity]
    additions=[('predicted_category','string'),('top_probability','float64'),('classification_status','string')]+[('probability_'+str(i)+'_'+model['classes'][i][:64],'float64') for i in range(len(model['classes']))]
    dc.require(not set(names)&{n for n,_ in additions} and len(source_columns)+len(additions)<=df.MAX_COLUMNS,'Prediction output names/width collide with the source schema; rename columns or reduce classes explicitly.')
    cols += [dict(name=n,dtype=t,nullable=True,**({'displayLabel':'Probability · '+model['classes'][i-3][:200]} if i>=3 else {})) for i,(n,t) in enumerate(additions)]
    cols += [source_columns[i] for i in remainder]
    def pairs():
        for i,(row,rid) in enumerate(df.row_pairs(table)):
            if i%32==0:check_cancelled()
            result=results[i]
            if result is None:values=[None,None,'No extractable text']+[None]*len(model['classes'])
            else:values=[result[0],result[1],'No vocabulary matches' if result[2] else 'Below review threshold' if config.get('reviewThreshold') is not None and result[1]<config['reviewThreshold'] else 'Classified']+result[3]
            yield tuple(row[i] for i in identity)+tuple(values)+tuple(row[i] for i in remainder),rid
    predicted=df.streaming_rows(table,pairs(),tuple(cols),check_cancelled)
    try:
        artifact=dict(schemaVersion=1,kind='table',columns=list(predicted.columns),rows=[list(r) for r in predicted.rows],lineage={**predicted.evidence(),'observationIDs':list(predicted.row_ids),'dataVersion':predicted.data_version,'partitionDigest':predicted.partition_digest,'fittedStateDigests':list(predicted.fitted_states),'diagnostics':list(predicted.diagnostics)})
        raw=dc.canonical(artifact);dc.require(len(raw)<=50_000_000,'Prediction table exceeds retained working-data allowance; reduce workload.')
        (stage/'predictions.json').write_bytes(raw)
    finally:predicted.rows.store.close();predicted.row_ids.store.close()
    return cols


def run_prepared(request,output,emit,check_cancelled=lambda:None):
    config=request['configuration']
    operation=config.get('operation','fit')
    dc.require(operation in ('fit','predict','evaluate'),'Unsupported document classifier operation.')
    if operation!='fit':return run_prediction(request,output,emit,check_cancelled)
    dc.require({'task','settings'}<=set(config)<= {'task','settings','operation','reviewThreshold'} and config['task']=='documentClassification','Invalid document estimator task settings.')
    dc.estimator_options(config['settings']);p=request['prepared'];dc.require(len(p['featureColumns'])==1 and p['targetColumn'] not in p['featureColumns'],'Select one prepared text column and a separate category column.')
    destination=Path(output);dc.require(not destination.exists() and not destination.is_symlink(),'Run output already exists.')
    manifest,files,pin=completed_inventory(p['manifest'],p['manifestSHA256'],lazy=True,check_cancelled=check_cancelled,selected_nodes={p['trainingNode'],p['validationNode']})
    dc.require(manifest['graphDigest']==p['graphDigest'],'Prepared graph differs from the selected estimator.')
    admit_model_ports(files,manifest,{(p['trainingNode'],p['trainingPort']),(p['validationNode'],p['validationPort'])})
    tables={};states={};base=Path(p['manifest']).parent
    try:
        for entry in manifest['artifacts']:
            check_cancelled()
            if entry.get('kind')=='state':
                state=df.load_state(files[entry['path']].decode());states[state['contentDigest']]=state
            if entry.get('kind')=='table':
                for role in ['training','validation']:
                    if (entry.get('nodeID'),entry.get('port'))==(p[role+'Node'],p[role+'Port']):
                        dc.require(role not in tables,'Duplicate prepared document port.')
                        tables[role]=read_table_artifact(base/entry['path'],entry['sha256'],check_cancelled=check_cancelled)
        dc.require(set(tables)=={'training','validation'},'Prepared document partitions are missing.')
        train=tables['training'];validation=tables['validation']
        dc.require(train.role=='training' and validation.role=='validation' and train.partition_digest and train.partition_digest==validation.partition_digest and train.source_digest==validation.source_digest and train.columns==validation.columns and train.fitted_states==validation.fitted_states,'Document populations have incompatible preparation lineage.')
        train_ids=list(train.row_ids);val_ids=list(validation.row_ids)
        dc.require(set(train_ids).isdisjoint(val_ids),'Document populations overlap; held-out leakage.')
        fitted=[states.get(x) for x in train.fitted_states]
        dc.require(fitted and all(x is not None and x['graphDigest']==manifest['graphDigest'] and x['partitionDigest']==train.partition_digest and set(df.fit_observation_ids(x))<=set(train_ids) for x in fitted),'Prepared feature state is absent or includes nontraining observations.')
        state=fitted[-1];dc.require(state.get('schemaVersion')==2 and state.get('operator')==df.PREFIX+'textEncode' and state.get('processor',{}).get('kind')=='documentTFIDF','Connect Training/Validation from Text encoding → Document features.')
        text=p['featureColumns'][0];target=p['targetColumn'];dc.require(state['columns']==[text],'Mapped text differs from the fitted processor.')
        index=train.index(text);label_index=train.index(target)
        dc.require(train.columns[index]['dtype']=='string' and train.columns[label_index]['dtype']=='string','Document text and category must be explicit string columns.')
        train_text=[];train_labels=[];val_text=[];val_labels=[]
        for table,texts,labels in [(train,train_text,train_labels),(validation,val_text,val_labels)]:
            for i,(row,_) in enumerate(df.row_pairs(table)):
                if i%32==0:check_cancelled()
                texts.append(row[index]);labels.append(row[label_index])
            dc.documents(texts,check_cancelled);dc.labels_valid(labels,len(texts))
        dc.require(not set(dc.digest(x) for x in train_text)&set(dc.digest(x) for x in val_text),'Identical document text crosses Training/Validation; group duplicates before Partition.')
        emit(dict(event='backendStarted',backendID='sklearn',trainingSamples=len(train_text),validationSamples=len(val_text)))
        model=dc.fit(train_text,train_labels,state['processor']['configuration'],config['settings'],processor=state['processor'],check=check_cancelled)
        evaluation=dc.evaluate(model,val_text,val_labels,check_cancelled)
        provenance=dict(dataset='prepared-document-features-v1',sourceHash=train.source_digest,samples=len(train_text)+len(val_text),sourceSamples=None,trainingSamples=len(train_text),validationSamples=len(val_text),trainingRowIDs=train_ids,validationRowIDs=val_ids,splitHash=df.digest({'training':df.membership_digest(train_ids),'validation':df.membership_digest(val_ids),'partition':train.partition_digest}),split='prepared partitions; no resplitting',featureColumns=[text],targetColumn=target,
            preparation=dict(manifestSHA256=pin,graphDigest=manifest['graphDigest'],registryDigest=manifest.get('registryDigest'),trainingNode=p['trainingNode'],trainingPort=p['trainingPort'],validationNode=p['validationNode'],validationPort=p['validationPort'],trainingDataVersion=train.data_version,validationDataVersion=validation.data_version,fittedStateDigests=list(train.fitted_states),artifactDirectory='preparation'))
        stage=Path(tempfile.mkdtemp(prefix='.document-fit-',dir=destination.parent));owned=None
        try:
            (stage/'model-state.json').write_bytes(dc.canonical(model));dc.export(model,stage/'portable')
            prediction=dc.predict(model,val_text,check_cancelled)
            prediction_columns=write_prediction_table(stage,validation,model,prediction,range(len(val_text)),config,check_cancelled,target)
            # History keeps preparation evidence under its original capture policy. The
            # portable model package contains only required state, never training rows.
            for name,raw in files.items():
                check_cancelled();dc.require(Path(name).name==name and name not in ('.','..'),'Unsafe preparation inventory filename.')
                (stage/'preparation').mkdir(exist_ok=True);(stage/'preparation'/name).write_bytes(raw)
            inventory=[];total=0
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    raw=path.read_bytes();total+=len(raw);dc.require(len(raw)<=50_000_000 and total<=200_000_000,'Classifier run artifacts exceed their retained budget.')
                    inventory.append(dict(path=path.relative_to(stage).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            result=dict(schemaVersion=1,backendID='sklearn',runtimeVersion='1.7.2',device='cpu',task='documentClassification',stateFormat=dc.FORMAT,configuration=config,provenance=provenance,evaluation=evaluation,solver=dict(iterations=model['iterations'],converged=model['converged']),predictionColumns=prediction_columns,predictionPartition='validation',artifacts=inventory)
            if request.get('capturePredictions',False):result['predictions']={**dc.predict(model,val_text[:8],check_cancelled),'rowIDs':val_ids[:8]}
            dc.require(len(dc.canonical(result))<=2_000_000,'Classifier receipt exceeds its metadata allowance; reduce workload or identity lengths.')
            check_cancelled();destination.mkdir(mode=0o700);info=destination.stat();owned=(info.st_dev,info.st_ino)
            for child in stage.iterdir():check_cancelled();child.rename(destination/child.name)
            (destination/'manifest.json').write_bytes(dc.canonical(result));emit(dict(event='backendCompleted',**result));check_cancelled()
            return result
        except BaseException:
            if owned is not None:
                try:
                    actual=destination.lstat()
                    if (actual.st_dev,actual.st_ino)==owned:shutil.rmtree(destination)
                except FileNotFoundError:pass
            raise
        finally:shutil.rmtree(stage)
    finally:
        for table in tables.values():table.rows.store.close();table.row_ids.store.close()


def run_prediction(request,output,emit,check_cancelled=lambda:None):
    """Apply captured inert package state to a captured table, never refit."""
    from contextlib import ExitStack
    p=request['prepared'];config=request['configuration'];operation=config['operation']
    destination=Path(output);dc.require(not destination.exists() and not destination.is_symlink(),'Run output already exists.')
    model_ref=request.get('model');dc.require(type(model_ref) is dict and set(model_ref)=={'path','sha256'},'Capture a document-classifier package before prediction.')
    if __package__:from .dataflow_storage import bounded_regular_read
    else:from dataflow_storage import bounded_regular_read
    raw=bounded_regular_read(model_ref['path'],dc.MAX_STATE);dc.require(hashlib.sha256(raw).hexdigest()==model_ref['sha256'],'Classifier state differs from captured model pin.')
    model=dc.load(raw)
    manifest,files,pin=completed_inventory(p['manifest'],p['manifestSHA256'],lazy=True,check_cancelled=check_cancelled,selected_nodes={p['trainingNode']})
    dc.require(manifest['graphDigest']==p['graphDigest'],'Prediction preparation graph differs.')
    selected=[e for e in manifest['artifacts'] if e.get('kind')=='table' and (e.get('nodeID'),e.get('port'))==(p['trainingNode'],p['trainingPort'])]
    dc.require(len(selected)==1 and len(p['featureColumns'])==1,'Choose one captured text table and text column.')
    entry=selected[0];table=read_table_artifact(Path(p['manifest']).parent/entry['path'],entry['sha256'],check_cancelled=check_cancelled)
    try:
        dc.require(all(c['dtype'] in ('string','float64','int64','bool') for c in table.columns),'Document predictions currently preserve scalar source columns only; select text/identity/labels before prediction.')
        text_index=table.index(p['featureColumns'][0]);dc.require(table.columns[text_index]['dtype']=='string','Prediction text must be a string column.')
        label_index=table.index(p['targetColumn']) if operation=='evaluate' else None
        if label_index is not None:dc.require(table.columns[label_index]['dtype']=='string','Evaluation category must be an explicit string column.')
        texts=[];labels=[];ids=[];valid=[];errors={}
        for i,(row,rid) in enumerate(df.row_pairs(table)):
            if i%32==0:check_cancelled()
            ids.append(rid)
            try:dc.documents([row[text_index]],check_cancelled)
            except ValueError as error:
                if operation=='evaluate':raise ValueError('Evaluation document has no usable text; resolve extraction before computing metrics.') from error
                errors[i]=str(error);continue
            valid.append(i);texts.append(row[text_index])
            if label_index is not None:labels.append(row[label_index])
        dc.require(ids,'Prediction source has no documents.')
        emit(dict(event='backendStarted',backendID='sklearn',operation=operation,documents=len(ids)))
        prediction=dc.predict(model,texts,check_cancelled) if texts else None
        metrics=dc.evaluate(model,texts,labels,check_cancelled) if operation=='evaluate' else dict(count=len(ids),classified=len(texts),failed=len(errors),operation='predict')
        stage=Path(tempfile.mkdtemp(prefix='.document-predict-',dir=destination.parent));owned=None
        try:
            # Complete working rows are distinct from optional captured samples.
            cols=write_prediction_table(stage,table,model,prediction,valid,config,check_cancelled,p['targetColumn'])
            (stage/'model-state.json').write_bytes(raw);dc.export(model,stage/'portable')
            inventory=[]
            for path in sorted(stage.rglob('*')):
                if path.is_file():content=path.read_bytes();inventory.append(dict(path=path.relative_to(stage).as_posix(),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
            dc.require(sum(e['bytes'] for e in inventory)<=200_000_000,'Prediction artifacts exceed retained operation allowance.')
            result=dict(schemaVersion=1,backendID='sklearn',runtimeVersion='1.7.2',device='cpu',task='documentClassification',operation=operation,stateFormat=dc.FORMAT,configuration=config,
                        provenance=dict(modelSHA256=model_ref['sha256'],preparation=dict(manifestSHA256=pin,graphDigest=manifest['graphDigest'],trainingNode=p['trainingNode'],trainingPort=p['trainingPort'],validationNode=p['validationNode'],validationPort=p['validationPort']),featureColumns=p['featureColumns'],targetColumn=p['targetColumn'],inputRowIDs=ids),evaluation=metrics,classOrder=model['classes'],predictionColumns=cols,predictionPartition=table.role,artifacts=inventory)
            if request.get('capturePredictions',False) and prediction is not None:result['predictions']={**{k:v[:8] if k!='classOrder' else v for k,v in prediction.items()},'rowIDs':[ids[i] for i in valid[:8]]}
            dc.require(len(dc.canonical(result))<=2_000_000,'Classifier receipt exceeds its metadata allowance; reduce workload or identity lengths.')
            check_cancelled();destination.mkdir(mode=0o700);info=destination.stat();owned=(info.st_dev,info.st_ino)
            for child in stage.iterdir():check_cancelled();child.rename(destination/child.name)
            (destination/'manifest.json').write_bytes(dc.canonical(result));emit(dict(event='backendCompleted',**result));check_cancelled();return result
        except BaseException:
            if owned is not None:
                try:
                    actual=destination.lstat()
                    if (actual.st_dev,actual.st_ino)==owned:shutil.rmtree(destination)
                except FileNotFoundError:pass
            raise
        finally:shutil.rmtree(stage)
    finally:table.rows.store.close();table.row_ids.store.close()
