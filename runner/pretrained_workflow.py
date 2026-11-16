"""Portable causal task bridge. Uses the same verified data reader and task engine.

No model, credentials or source path is embedded in derived Code. Grants and
resource allowance are explicit user invocation bindings outside the app sandbox.
"""
from pathlib import Path
if __package__:
    from . import pretrained_training as ft
    from .dataflow_storage import completed_inventory
    from .dataflow_table_reader import read_table_artifact
    from .training_control import validate_policy
    from .training_progress import validate_policy as reporting_policy
else:
    import pretrained_training as ft
    from dataflow_storage import completed_inventory
    from dataflow_table_reader import read_table_artifact
    from training_control import validate_policy
    from training_progress import validate_policy as reporting_policy


def canvas_settings(config,maximum_resident_bytes):
    if not isinstance(config,dict) or not isinstance(config.get('fineTuning'),dict) or not isinstance(config.get('runSetup'),dict):raise ValueError('Use captured imported Train settings.')
    fine=dict(config['fineTuning']);run=config['runSetup']
    if any(k in config for k in ('gradientPolicy','initialization','trainability','objective','datasetAccess','batching')) or 'experiment' in run or run.get('preparationCachePolicy','off')!='off' or config.get('trainingMode','fresh')!='fresh':raise ValueError('Causal Code cannot silently drop incompatible canvas recipes, experiments or cache policies.')
    for key in ('startMode','resumeRecord','resumeEpoch'):fine.pop(key,None)
    # This bridge resolves the same legacy omissions as native Train. The task
    # adapter remains the single validator and executor of all resolved settings.
    defaults=dict(mode='full',groups=[],targets=[],rank=8,alpha=16,dropout=0,bias='none',additionalModules=[],textColumn='text',maximumTokens=128,stride=127,appendEOS=True,accumulation=1)
    c={**defaults,**fine,'device':run.get('device','auto'),'epochs':run.get('epochs',1),'batchSize':run.get('batchSize',8),'seed':run.get('seed',0),'maximumResidentBytes':maximum_resident_bytes,'capturePredictions':False,
       'optimizer':config.get('optimizer',dict(schemaVersion=1,componentID='optimizer_adamw',configurationVersion=1,config={'lr':run.get('learningRate',.001)})),
       'scheduler':config.get('scheduler'),'trainingControl':run.get('trainingControl',validate_policy(dict(schemaVersion=1,modelSelection='finalEpoch',monitor='validationLoss',earlyStopping=dict(enabled=False,patience=5,minimumImprovement=.001)))), 'reporting':reporting_policy(dict(everyBatches=10,everySeconds=1.))}
    return ft.training_settings(c)


def run_prepared(training,prepared,bindings,destination,emit,check_cancelled=lambda:None):
    config=training['configuration'];model=(bindings.get('models') or {}).get(config['model'])
    if not isinstance(model,str) or not model:raise ValueError('Bind the original safe model folder by Load model ID in models.')
    resources=bindings.get('resources')
    if not isinstance(resources,dict) or set(resources)!={'maximumResidentBytes'} or type(resources['maximumResidentBytes']) is not int or resources['maximumResidentBytes']<1:raise ValueError('Bind an explicit positive maximumResidentBytes allowance.')
    c=canvas_settings(config['train'],resources['maximumResidentBytes'])
    start=config['train']['fineTuning'].get('startMode','base')
    if start=='weights':raise ValueError('For weights reuse, bind the inference model explicitly and use New from loaded model in the captured Train.')
    resume=(bindings.get('checkpoints') or {}).get(training['id']) if start=='resume' else None
    if start=='resume' and (not isinstance(resume,str) or not resume):raise ValueError('CPU resume needs an explicit verified checkpoint package binding keyed by Train ID.')
    mapping=training['prepared'];manifest,files,_=completed_inventory(prepared['manifest'],prepared['manifestSHA256'],lazy=True,check_cancelled=check_cancelled,selected_nodes={mapping['trainingNode'],mapping['validationNode']})
    if manifest['graphDigest']!=prepared['graphDigest']:raise ValueError('Prepared graph differs from the selected task.')
    tables={};texts={};members={};base=Path(prepared['manifest']).parent
    try:
        for entry in manifest['artifacts']:
            if entry.get('kind')!='table':continue
            for role in ('training','validation'):
                if (entry.get('nodeID'),entry.get('port'))==(mapping[role+'Node'],mapping[role+'Port']):
                    if role in tables:raise ValueError('Duplicate prepared task port.')
                    tables[role]=read_table_artifact(base/entry['path'],entry['sha256'],check_cancelled)
        if set(tables)!={'training','validation'}:raise ValueError('Complete Training and Validation tables are required.')
        for role,table in tables.items():
            if table.role!=role:raise ValueError('Connect declared Training and Validation populations.')
            names=[c['promptColumn'],c['responseColumn']] if c['exampleFormat']=='promptResponse' else [c['textColumn']]
            indices=[]
            for name in names:
                found=[i for i,column in enumerate(table.columns) if column['name']==name and column['dtype']=='string']
                if len(found)!=1:raise ValueError('Choose captured String columns.')
                indices+=found
            texts[role]=[dict(prompt=row[indices[0]],response=row[indices[1]]) if len(indices)==2 else row[indices[0]] for row in table.rows]
            members[role]=set(table.row_ids)
        if members['training'] & members['validation']:raise ValueError('Training and Validation observations overlap.')
        check_cancelled()
        return ft.fine_tune(model,destination,texts['training'],texts['validation'],c,emit=emit,check_cancelled=check_cancelled,resume_from=resume)
    finally:
        for table in tables.values():table.rows.store.close();table.row_ids.store.close()
