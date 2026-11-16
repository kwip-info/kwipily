"""Pinned prepared tables -> reviewed Prebuilt models -> ordinary working tables."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
if __package__:
    from . import prebuilt_models as pb,dataflow as df
    from .dataflow_storage import completed_inventory,admit_model_ports,bounded_regular_read
    from .dataflow_table_reader import read_table_artifact
else:
    import prebuilt_models as pb,dataflow as df
    from dataflow_storage import completed_inventory,admit_model_ports,bounded_regular_read
    from dataflow_table_reader import read_table_artifact

def configuration(c):
    pb.require(type(c) is dict and set(c)=={'task','algorithm','operation','settings'} and c['task']=='prebuilt','Unsupported Prebuilt models request.')
    pb.require(c['operation'] in ('fit','predict','evaluate'),'Choose Fit, Predict or Evaluate.')
    entry=pb.catalog().get(c['algorithm']);pb.require(entry is not None,'Unsupported algorithm.')
    result=dict(task=entry['task'],algorithm=c['algorithm'],options=c['settings']);pb.configuration(result)
    pb.require(c['algorithm']!='dbscan' or c['operation']=='fit','DBSCAN is fit-only analysis; it cannot predict new rows.')
    return result

def validate_request(request):
    config=configuration(request['configuration']);p=request['prepared'];names=p['featureColumns']
    pb.require(type(names) is list and 1<=len(names)<=128 and len(set(names))==len(names) and all(type(n) is str and 1<=len(n)<=200 for n in names),'Select 1–128 unique ordered features.')
    if config['task']=='clustering':pb.require(p['targetColumn']=='','Clustering does not use a target.')
    elif request['configuration']['operation']!='predict':pb.require(p['targetColumn'] and p['targetColumn'] not in names,'Select a separate target.')
    return config

def table_bytes(table,columns,rows,ids):
    derived=df.derive(table,rows=tuple(tuple(row) for row in rows),ids=tuple(ids),columns=tuple(columns))
    return pb.canonical(dict(schemaVersion=1,kind='table',columns=columns,rows=rows,lineage={**derived.evidence(),'observationIDs':ids,'dataVersion':derived.data_version,'partitionDigest':derived.partition_digest,'fittedStateDigests':list(derived.fitted_states),'diagnostics':list(derived.diagnostics)}))

def complete_preprocess_path(manifest,files,node,port,state_nodes):
    """Do not silently omit same-schema Formula/Columns operations after a fit.

    Fixed preparation before the first learned step remains an explicit input
    prerequisite. Only identity Views may intervene within the saved chain.
    """
    graph=json.loads(files['dataflow.json'])
    pb.require(df.digest(graph)==manifest['graphDigest'],'Preparation graph identity differs.')
    nodes={n['id']:n for n in graph['nodes']}
    inputs={(e['to']['node'],e['to']['port']):e['from'] for e in graph['edges']}
    pending=list(reversed(state_nodes));seen=set()
    while pending:
        if node in seen or node not in nodes:return False
        seen.add(node);op=nodes[node]['operator']
        if op==df.PREFIX+'view' and port=='table':input_port='table'
        elif op==df.PREFIX+'preprocess' and node==pending[0] and port=='training':
            pending.pop(0)
            if not pending:return True
            input_port='training'
        else:return False
        previous=inputs.get((node,input_port))
        if previous is None:return False
        node,port=previous['node'],previous['port']
    return False

def run_prepared(request,output,emit,check_cancelled=lambda:None):
    c=validate_request(request);p=request['prepared'];op=request['configuration']['operation'];cluster=c['task']=='clustering';fit=op=='fit'
    destination=Path(output);pb.require(not destination.exists() and not destination.is_symlink(),'Run destination exists.')
    roles=['training'] if cluster or not fit else ['training','validation']
    manifest,files,pin=completed_inventory(p['manifest'],p['manifestSHA256'],lazy=True,check_cancelled=check_cancelled,selected_nodes={p[r+'Node'] for r in roles})
    pb.require(manifest['graphDigest']==p['graphDigest'],'Preparation graph differs from captured input.')
    admit_model_ports(files,manifest,{(p[r+'Node'],p[r+'Port']) for r in roles})
    tables={};stage=None;owned=None
    try:
        for role in roles:
            selected=[e for e in manifest['artifacts'] if e.get('kind')=='table' and (e.get('nodeID'),e.get('port'))==(p[role+'Node'],p[role+'Port'])]
            pb.require(len(selected)==1,'Prepared input table is missing or ambiguous.');entry=selected[0]
            tables[role]=read_table_artifact(Path(p['manifest']).parent/entry['path'],entry['sha256'],check_cancelled=check_cancelled)
        train=tables['training'];learned_parameters=None
        if fit and train.fitted_states:
            states={};state_nodes={}
            for e in manifest['artifacts']:
                if e.get('kind')=='state':
                    state=df.load_state(files[e['path']].decode());states[state['contentDigest']]=state;state_nodes[state['contentDigest']]=e['nodeID']
            ids=set(train.row_ids)
            for pin_state in train.fitted_states:
                state=states.get(pin_state)
                pb.require(state is not None and state['graphDigest']==manifest['graphDigest'] and state['fitEvidence']['sourceDigest']==train.source_digest and set(df.fit_observation_ids(state))<=ids,'Fitted preparation is missing or includes observations outside the fit population.')
            # Only a complete, unmodified Preprocess chain can be applied to raw
            # new inputs. Other valid numeric fits keep their prepared-only export.
            chain=[states[key] for key in train.fitted_states]
            if 1<=len(chain)<=16 and all(s['operator']==df.PREFIX+'preprocess' and
                    s['inputFittedStateDigests']==list(train.fitted_states[:i]) and
                    (i==0 or s['inputSchema']==chain[i-1]['outputSchema'])
                    for i,s in enumerate(chain)) and chain[-1]['outputSchema']==list(train.columns) and complete_preprocess_path(
                        manifest,files,p['trainingNode'],p['trainingPort'],[state_nodes[key] for key in train.fitted_states]):
                if __package__:from .dataflow_preprocess import parameters
                else:from dataflow_preprocess import parameters
                learned_parameters=[parameters(s) for s in chain]
        def extract(table,want_target):
            pb.require(len(table.rows)<=pb.MAX_ROWS and len(table.rows)*len(p['featureColumns'])<=pb.MAX_CELLS,'Estimator input exceeds its admitted rows/cells.')
            indices=[table.index(n) for n in p['featureColumns']]
            pb.require(all(table.columns[i]['dtype'] in ('float64','int64') for i in indices),'Select numeric features; encode categorical values with Training-fitted preparation.')
            ti=table.index(p['targetColumn']) if want_target else None
            if ti is not None and c['task']=='regression':
                pb.require(table.columns[ti]['dtype'] in ('float64','int64'),'Select an explicit numeric target; convert text or Boolean columns in Columns before regression.')
            rows=[];targets=[]
            for i,(row,_) in enumerate(df.row_pairs(table)):
                if i%128==0:check_cancelled()
                pb.require(all(type(row[j]) in (int,float) for j in indices),'Features contain missing or nonnumeric values; prepare them explicitly.')
                rows.append([row[j] for j in indices])
                if ti is not None:
                    if c['task']=='regression':pb.require(type(row[ti]) in (int,float),'Regression requires a nonmissing numeric target; prepare invalid values explicitly.')
                    targets.append(row[ti])
            x=np.asarray(rows,dtype=np.float64);pb.features(x)
            if ti is not None:
                values=targets
                pb.require(values and all(type(v)==type(values[0]) for v in values),'Targets must have one nonmissing scalar type.')
                y=np.asarray(values,dtype=np.float64 if c['task']=='regression' else None)
            else:y=None
            return x,y
        train_ids=list(train.row_ids)
        if fit and not cluster:
            val=tables['validation'];val_ids=list(val.row_ids)
            pb.require(train.role=='training' and val.role=='validation' and train.partition_digest and train.partition_digest==val.partition_digest and train.source_digest==val.source_digest and train.fitted_states==val.fitted_states and not set(train_ids)&set(val_ids),'Supervised fit needs matching disjoint prepared Training/Validation populations.')
        else:val=train;val_ids=train_ids
        x,y=extract(train,not cluster and (fit or op=='evaluate'))
        emit(dict(event='backendStarted',backendID='sklearn',phase='fit' if fit else 'frozen inference',**(dict(trainingSamples=len(x),validationSamples=len(val_ids)) if fit else dict(inputSamples=len(x)))))
        if fit:
            model,prediction=pb.fit(c,x,y,check=check_cancelled)
            if model is not None:model['featureColumns']=p['featureColumns'];pb.validate(model)
        else:
            ref=request.get('model');base={'path','sha256'};with_rules=base|{'preparationPath','preparationSHA256'}
            pb.require(type(ref) is dict and set(ref) in (base,with_rules),'Capture a Prebuilt model package.')
            raw=bounded_regular_read(ref['path'],pb.MAX_STATE);pb.require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Model differs from captured pin.');model=pb.load(raw)
            pb.require(model['configuration']==c and model['featureColumns']==p['featureColumns'],'Algorithm/options or feature order differs from the frozen package; use its saved configuration.')
            if set(ref)==with_rules:
                if __package__:from . import prebuilt_preparation as frozen
                else:import prebuilt_preparation as frozen
                prepared_rules=bounded_regular_read(ref['preparationPath'],frozen.MAX_BYTES)
                pb.require(hashlib.sha256(prepared_rules).hexdigest()==ref['preparationSHA256'],'Learned preparation differs from captured pin.')
                envelope=frozen.validate(json.loads(prepared_rules),raw)
                required=[s['sourceContentDigest'] for s in envelope['parameters']]+[envelope['contentDigest']]
                pb.require(list(train.fitted_states[-len(required):])==required,'The model-bound learned preparation must be applied to these inputs before prediction.')
            prediction=pb.predict(model,x)
        if fit and not cluster:
            vx,vy=extract(val,True);prediction=pb.predict(model,vx);metrics=pb.evaluate(model,vx,vy)
        elif not cluster and op=='evaluate':metrics=pb.evaluate(model,x,y)
        else:metrics=dict(count=len(val_ids))
        if cluster:
            labels=np.asarray(prediction['values']);unique=sorted(set(labels.tolist())-{-1})
            metrics.update(clusters=len(unique),noiseRows=int(np.sum(labels==-1)),population='fit' if fit else 'inference')
            if 'fitInertia' in prediction:metrics['inertia']=prediction['fitInertia']
            # Diagnostics are sampled, fitting never is. O(512²) pairwise work.
            rng=np.random.default_rng(42);sample=np.sort(rng.choice(len(x),min(512,len(x)),replace=False));sample=sample[labels[sample]!=-1]
            metrics.update(silhouette=None,silhouetteSampleRows=len(sample),silhouettePopulationRows=len(x),silhouetteSeed=42,silhouetteNoisePolicy='exclude noise')
            if 2<=len(set(labels[sample]))<len(sample):
                from sklearn.metrics import silhouette_score
                metrics['silhouette']=float(silhouette_score(x[sample],labels[sample]))
        stage=Path(tempfile.mkdtemp(prefix='.prebuilt-',dir=destination.parent))
        attachment=None
        if model is not None:
            model_raw=pb.canonical(model);(stage/'model-state.json').write_bytes(model_raw);pb.export(model,stage/'portable')
            if learned_parameters is not None:
                if __package__:from . import prebuilt_preparation as frozen
                else:import prebuilt_preparation as frozen
                envelope=frozen.create(model_raw,learned_parameters);raw=pb.canonical(envelope)
                package=stage/'portable-prepared';shutil.copytree(stage/'portable',package)
                (package/'preparation.json').write_bytes(raw)
                (package/'README.txt').write_text('Model + learned Preprocess rules. In Kwipily: Load model -> Preprocess -> Apply saved rules -> Prebuilt models Predict. Fixed source typing/cleanup remains explicit. inference.py accepts prepared numeric features in saved featureColumns order; it does not automatically apply preparation.json. No training rows, refitting or optimizer resume.\n')
                package_files=[dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(package.iterdir()) if f.name!='manifest.json']
                (package/'manifest.json').write_bytes(pb.canonical(dict(schemaVersion=1,stateFormat=pb.FORMAT,artifacts=package_files)))
                attachment=dict(path='portable-prepared/preparation.json',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),contentDigest=envelope['contentDigest'])
        names=[col['name'] for col in val.columns];output_name='cluster' if cluster else 'prediction'
        extra=[dict(name=output_name,dtype='string' if c['task']=='classification' and type(prediction['values'][0]) is str else 'float64' if c['task']=='regression' else 'int64',nullable=False)]
        if 'status' in prediction:extra.append(dict(name='cluster_status',dtype='string',nullable=False))
        score_key=next((k for k in ('probabilities','memberships','distances') if k in prediction),None)
        if score_key:
            for j in range(len(prediction[score_key][0])):extra.append(dict(name=score_key+'_'+str(j),displayLabel=('Probability · '+str(model['classes'][j]) if score_key=='probabilities' else ('Membership · component ' if score_key=='memberships' else 'Distance · centroid ')+str(j)),dtype='float64',nullable=False))
        pb.require(not set(names)&{v['name'] for v in extra} and len(names)+len(extra)<=df.MAX_COLUMNS,'Prediction column names collide or table is too wide; rename/select source columns.')
        columns=list(val.columns)+extra;rows=[]
        for i,(row,_) in enumerate(df.row_pairs(val)):
            if i%128==0:check_cancelled()
            out=list(row)+[prediction['values'][i]]
            if 'status' in prediction:out.append(prediction['status'][i])
            if score_key:out+=prediction[score_key][i]
            rows.append(out)
        raw=table_bytes(val,columns,rows,val_ids);pb.require(len(raw)<=50_000_000,'Complete output table exceeds its 50MB admitted limit.');(stage/'predictions.json').write_bytes(raw)
        # Cluster profiles are genuine standard tables over the fit/inference population.
        if cluster:
            cols=[dict(name='cluster',dtype='int64',nullable=False),dict(name='rows',dtype='int64',nullable=False)]+[dict(name='mean_'+str(i),displayLabel='Mean · '+n,dtype='float64',nullable=False) for i,n in enumerate(p['featureColumns'])]
            profiles=[[k,int(np.sum(labels==k))]+x[labels==k].mean(axis=0).tolist() for k in sorted(set(labels.tolist()))]
            (stage/'profiles.json').write_bytes(table_bytes(val,cols,profiles,['profile:'+str(r[0]) for r in profiles]))
        if 'confusionMatrix' in metrics:
            confusion_columns=[dict(name='actual',dtype='string',nullable=False),dict(name='predicted',dtype='string',nullable=False),dict(name='count',dtype='int64',nullable=False)]
            confusion_rows=[[str(model['classes'][i]),str(model['classes'][j]),count] for i,row in enumerate(metrics['confusionMatrix']) for j,count in enumerate(row)]
            (stage/'confusion.json').write_bytes(table_bytes(val,confusion_columns,confusion_rows,['confusion:'+str(i) for i in range(len(confusion_rows))]))
        population=pb.canonical(dict(trainingRowIDs=train_ids,validationRowIDs=val_ids,inputRowIDs=val_ids) if fit else dict(inputRowIDs=val_ids))
        pb.require(len(population)<=50_000_000,'Population identities exceed the retained 50MB artifact allowance.')
        (stage/'population.json').write_bytes(population)
        for name,raw in files.items():
            check_cancelled();pb.require(Path(name).name==name and name not in ('.','..'),'Unsafe preparation inventory filename.');(stage/'preparation').mkdir(exist_ok=True);(stage/'preparation'/name).write_bytes(raw)
        inventory=[]
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                raw=path.read_bytes();pb.require(len(raw)<=50_000_000,'Artifact exceeds 50MB.');inventory.append(dict(path=path.relative_to(stage).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        pb.require(sum(f['bytes'] for f in inventory)<=200_000_000,'Operation exceeds retained artifact budget.')
        provenance=dict(featureColumns=p['featureColumns'],targetColumn=p['targetColumn'],trainingSamples=len(train_ids),validationSamples=len(val_ids),populationFile='population.json',populationSHA256=hashlib.sha256(population).hexdigest(),preparation={k:p[k] for k in ('manifestSHA256','graphDigest','trainingNode','trainingPort','validationNode','validationPort')})
        if not fit:
            # Request slots retain their legacy names for project compatibility;
            # a frozen operation has one input population, never a new training split.
            provenance=dict(featureColumns=p['featureColumns'],targetColumn=p['targetColumn'],inputSamples=len(val_ids),
                            populationFile='population.json',populationSHA256=hashlib.sha256(population).hexdigest(),
                            preparation=dict(manifestSHA256=p['manifestSHA256'],graphDigest=p['graphDigest'],inputNode=p['trainingNode'],inputPort=p['trainingPort']),
                            modelSHA256=request['model']['sha256'])
            if 'preparationPath' in request['model']:
                provenance.update(modelPreparationSHA256=request['model']['preparationSHA256'],modelPreparationDigest=envelope['contentDigest'])
        result=dict(schemaVersion=1 if fit else 2,backendID='sklearn',runtimeVersion=pb.VERSION,device='cpu',task='prebuilt',stateFormat=pb.FORMAT,configuration=request['configuration'],provenance=provenance,evaluation=metrics,predictionColumns=columns,predictionPartition=val.role,artifacts=inventory)
        if attachment is not None:result['learnedPreparation']=attachment
        if 'confusionMatrix' in metrics:result['confusionMetadata']=dict(columns=confusion_columns,rowCount=len(confusion_rows),partition=val.role)
        if cluster:result['profilesMetadata']=dict(columns=cols,rowCount=len(profiles),partition=val.role)
        pb.require(len(pb.canonical(result))<=1_000_000,'Receipt exceeds metadata limit.')
        check_cancelled();destination.mkdir(mode=0o700);info=destination.stat();owned=(info.st_dev,info.st_ino)
        for path in stage.iterdir():check_cancelled();path.rename(destination/path.name)
        (destination/'manifest.json').write_bytes(pb.canonical(result));emit(dict(event='backendCompleted',**result));return result
    except BaseException:
        if owned is not None and destination.exists() and (destination.stat().st_dev,destination.stat().st_ino)==owned:shutil.rmtree(destination)
        raise
    finally:
        if stage is not None:shutil.rmtree(stage)
        for table in tables.values():table.rows.store.close();table.row_ids.store.close()
