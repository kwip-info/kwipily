"""Fixed-declaration standalone prepared-source/loading export; no data copies."""
import hashlib
import inspect
if __package__:
    from . import dataset_loader_configuration as configuration,dataset_sources as sources,batch_loading as loading,model_input,embedding_bag_packet as packet
    from .bag_export import _selected
else:
    import dataset_loader_configuration as configuration,dataset_sources as sources,batch_loading as loading,model_input,embedding_bag_packet as packet
    from bag_export import _selected


def data_loading_recipe_code(dataset_recipe,loader_recipe,legacy_batch_size):
    for recipe,component in ((dataset_recipe,'dataset_source'),(loader_recipe,'data_loader')):
        if recipe is not None and configuration.validate_recipe(recipe)['componentID']!=component:
            raise ValueError('Portable loading selects the correct source/loader recipe kinds.')
    if type(legacy_batch_size) is not int or not 1<=legacy_batch_size<=100000:
        raise ValueError('Portable loading requires the actual captured batch size.')
    if loader_recipe is not None and loader_recipe['config']['batch_size']!=legacy_batch_size:
        raise ValueError('Portable batch size differs from selected loader.')
    header='import copy, json, math, hashlib\nfrom contextlib import contextmanager\nfrom dataclasses import dataclass\nimport torch\nfrom torch.utils.data import Dataset, IterableDataset, TensorDataset, Subset, get_worker_info, DataLoader\n'
    body=_selected(configuration,{'MAX_RECIPE_BYTES','MAX_ROWS','DATASET_DEFAULTS','LOADER_DEFAULTS','_int','validate_recipe'})
    body+=_selected(packet,{'MAXIMUM_BAGS','MAXIMUM_ITEMS','MAXIMUM_OUTPUT_ELEMENTS','MAXIMUM_TABLE_SHAPE_ELEMENTS','EmbeddingBagPacket','validate_packet','collate_bags','_positive_integer','_tensor'},{'_tensor':'_packet_tensor'}).replace('def _tensor(', 'def _packet_tensor(')
    body+=_selected(model_input,{'_GEOMETRY','_tensor','_packet','observation_count','select_observations','canonical_input_tensors'})
    body+=_selected(sources,{'SourceRecord','IndexedTrainingDataset','TrainingTensorDataset','RestartableTrainingIterable','TrainingSource','make_source'})
    body+=_selected(loading,{'ORDER_VERSION','MAX_WORKING_BYTES','BatchPlan','TrainingBatch','epoch_seed','_configuration','plan_epoch','_identity','_bags','_Collator','EpochBatches','open_epoch'})
    body+='\nDATASET_RECIPE = '+repr(dataset_recipe)+'\nLOADER_RECIPE = '+repr(loader_recipe)+'\nCAPTURED_BATCH_SIZE = '+repr(legacy_batch_size)+'\n'
    body+='''
def create_training_source(inputs,roles,row_ids):
    recipe=DATASET_RECIPE or dict(schemaVersion=1,configurationVersion=1,componentID='dataset_source',config=dict(DATASET_DEFAULTS))
    return make_source(inputs,roles,row_ids,recipe)
def planned_training_epoch(source,epoch):
    return plan_epoch(source,LOADER_RECIPE,epoch,legacy_batch_size=CAPTURED_BATCH_SIZE)
def open_training_epoch(source,epoch,*,check_cancelled=None):
    return open_epoch(source,LOADER_RECIPE,epoch,legacy_batch_size=CAPTURED_BATCH_SIZE,check_cancelled=check_cancelled)
def forward_batch(model,batch):
    if type(batch.input) is EmbeddingBagPacket:
        return model(batch.input.indices,offsets=batch.input.offsets,per_sample_weights=batch.input.per_sample_weights)
    return model(batch.input)
'''
    result=header+'DATA_LOADING_HELPER_SHA256 = '+repr(hashlib.sha256(body.encode()).hexdigest())+'\n'+body
    if len(result.encode())>131072:raise ValueError('Portable data-loading helper exceeds131072 bytes.')
    return result
