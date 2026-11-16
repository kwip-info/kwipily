"""Bounded inference activation cache; never a training/autograd cache."""
import hashlib
import io
import json
import os
from pathlib import Path
import torch
if __package__:
    from .output_routing import map_stage, summarize_stage
    from .artifact_storage import ArtifactStorage
    from .worker_resources import configure_torch
    from .product_execution import ProductExecutionAdapter
    from .prepared_data import bounded_read
    from .engine import RECIPE_VERSION, GraphModel, validate, dataset, csv_dataset, load_data, resolve_device, atomic_json
else:
    from output_routing import map_stage, summarize_stage
    from artifact_storage import ArtifactStorage
    from worker_resources import configure_torch
    from product_execution import ProductExecutionAdapter
    from prepared_data import bounded_read
    from engine import RECIPE_VERSION, GraphModel, validate, dataset, csv_dataset, load_data, resolve_device, atomic_json

PREVIEW_STORAGE_BYTES=33100000
PREVIEW_ARTIFACT_LIMITS={'activation-cache.pt':32000000,'activation-cache.json':100000,'preview.json':1000000}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


if __package__:
    from .graph_semantics import semantic_graph
else:
    from graph_semantics import semantic_graph


def file_hash(path,max_bytes):
    path=Path(path)
    if path.stat().st_size>max_bytes: raise ValueError('Artifact exceeds the bounded preview loading limit.')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(65536),b''): h.update(chunk)
    return h.hexdigest()


def weight_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preview(graph,output,seed=42,device='auto',registry=None,dataset_csv=None,feature_columns=None,target_column=None,stop_at=None,resume_from=None,cache=None,checkpoint=None,emit=lambda event:None,on_execute=lambda ident:None,sample_limit=0,split_seed=42,validation_percent=20,prepared=None,check_cancelled=lambda:None):
    check_cancelled()
    order,widths=validate(graph,registry,bundled_dataset=prepared is None)
    if type(seed) is not int or not 0<=seed<=2**32-1: raise ValueError('Seed must be an integer from 0 to 4294967295.')
    if stop_at is not None and stop_at not in order: raise ValueError('Preview stop component is absent from graph.')
    if resume_from is not None and resume_from not in order: raise ValueError('Resume component is absent from graph.')
    if bool(resume_from)!=bool(cache): raise ValueError('Run from requires both a selected component and activation cache.')
    device=resolve_device(device); configure_torch(torch); torch.manual_seed(seed)
    if prepared is not None:
        if dataset_csv is not None or sample_limit!=0 or split_seed!=42 or validation_percent!=20:raise ValueError('Prepared partitions cannot be resplit or sampled by preview.')
        data,source,_=prepared
    else:
        data,source=load_data(dataset_csv,feature_columns,target_column,split_seed,sample_limit,validation_percent)
    check_cancelled()
    x=data[2][:8]; model=GraphModel(graph,registry,bundled_dataset=prepared is None).to(device); weights='initialized'
    model.materialize_lazy_parameters()
    if checkpoint:
        checkpoint=Path(checkpoint); checkpoint_bytes=bounded_read(checkpoint,100_000_000)
        source_graph=checkpoint.with_name('graph.json')
        if semantic_graph(json.loads(bounded_read(source_graph,2_000_000)))!=semantic_graph(graph): raise ValueError('Checkpoint graph does not match the current semantic architecture.')
        state=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not isinstance(state.get('modelState'),dict): raise ValueError('Checkpoint has no modelState.')
        expected=model.state_dict(); saved=state['modelState']
        if set(saved)!=set(expected) or any(not isinstance(saved[key],torch.Tensor) or saved[key].shape!=value.shape or saved[key].dtype!=value.dtype for key,value in expected.items()):
            raise ValueError('Checkpoint parameter names, shape or dtype are incompatible.')
        model.load_state_dict(saved,strict=True); weights='checkpoint'
    input_provenance={'partition':'validation','count':len(x),'maximumRows':8,'rowIDs':source.get('validationRowIDs',[])[:len(x)],'featureColumns':source.get('featureColumns'),'targetColumn':source.get('targetColumn'),'sourceHash':source.get('sourceHash'),'splitHash':source.get('splitHash'),'preparation':source.get('preparation'),'inferenceOnly':True}
    initial_weights_hash=weight_hash(model.state_dict())
    mutating=[ident for ident in model.layer_keys if isinstance(model.layers[model.layer_keys[ident]],torch.nn.Embedding) and model.layers[model.layer_keys[ident]].max_norm is not None] if model.layer_keys else []
    cache_schema=3 if any(isinstance(layout,dict) for layout in model.layouts.values()) else (2 if mutating else 1)
    adapter=ProductExecutionAdapter(model)
    signature=digest({'schemaVersion':1,'recipeVersion':RECIPE_VERSION,'graph':semantic_graph(graph),'source':source,'seed':seed,'device':device,'weights':weight_hash(model.state_dict()),'input':x.tolist()})
    start=order.index(resume_from) if resume_from else 0
    end=order.index(stop_at)+1 if stop_at else len(order)
    if start>=end: raise ValueError('Resume component must precede or equal the stop component.')
    values={}; reused=[]
    if cache:
        cache=Path(cache); sidecar=cache.with_suffix('.json')
        meta=json.loads(bounded_read(sidecar,100_000))
        if meta.get('schemaVersion')!=cache_schema or meta.get('recipeVersion')!=RECIPE_VERSION or meta.get('signature')!=signature: raise ValueError('Activation cache is stale: architecture, source, seed, device or weights do not match. Preview again.')
        cache_bytes=bounded_read(cache,32_000_000)
        if hashlib.sha256(cache_bytes).hexdigest()!=meta.get('fileHash'): raise ValueError('Activation cache integrity check failed. Preview again.')
        cached=torch.load(io.BytesIO(cache_bytes),map_location='cpu',weights_only=True)
        if not isinstance(cached,dict): raise ValueError('Malformed activation cache.')
        if cache_schema in (2,3):
            if set(cached)!={'activations','mutatedWeights'} or not isinstance(cached['activations'],dict) or not isinstance(cached['mutatedWeights'],dict):raise ValueError('Malformed forward-state activation cache.')
            mutations=cached['mutatedWeights'];cached=cached['activations']
            for ident in mutating:
                if ident not in order[:start]:continue
                weight=mutations.get(ident);layer=model.layers[model.layer_keys[ident]]
                if not isinstance(weight,torch.Tensor) or weight.layout!=torch.strided or weight.dtype!=layer.weight.dtype or weight.shape!=layer.weight.shape or not torch.isfinite(weight).all():
                    raise ValueError('Activation cache is missing valid forward-mutated embedding weights; preview again.')
                with torch.no_grad():layer.weight.copy_(weight)

        for ident in order[:start]:
            value=cached.get(ident)
            try:adapter.validate_stage(value,ident,rows=len(x),device='cpu')
            except ValueError as error:raise ValueError(f'Activation cache is incomplete or invalid before {resume_from}; preview through that component first. {error}') from error
            values[ident]=map_stage(value,lambda tensor:tensor.detach().to(device)); reused.append(ident)
    check_cancelled()
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise ValueError('Output directory must be empty for preview.')
    model.to(device).eval(); x=x.to(device); executed=[]
    adapter.validate_input(x)
    emit({'event':'previewStarted','device':device,'weights':weights,'resumeFrom':resume_from,'stopAt':stop_at})
    with torch.inference_mode():
        for ident in order[start:end]:
            check_cancelled()
            values[ident]=model.execute_node(ident,values,x)
            adapter.validate_stage(values[ident],ident,rows=len(x))
            on_execute(ident); executed.append(ident)
        cpu_values={i:map_stage(v,lambda tensor:tensor.detach().cpu().clone()) for i,v in values.items()}
    previews={i:summarize_stage(v) for i,v in cpu_values.items()}
    for node in graph['nodes']:
        if node['type'] in ('input','index_input') and node['id'] in previews:previews[node['id']].update(input_provenance)
    cache_path=output/'activation-cache.pt'
    final_weights_hash=weight_hash(model.state_dict())
    payload=cpu_values
    if cache_schema in (2,3):
        payload={'activations':cpu_values,'mutatedWeights':{ident:model.layers[model.layer_keys[ident]].weight.detach().cpu().clone() for ident in mutating if ident in values}}
    check_cancelled()
    with ArtifactStorage(output,PREVIEW_STORAGE_BYTES,check_cancelled=check_cancelled) as storage:
        storage.atomic_write('activation-cache.pt',PREVIEW_ARTIFACT_LIMITS['activation-cache.pt'],lambda stream:torch.save(payload,stream));check_cancelled()
        meta={'schemaVersion':cache_schema,'recipeVersion':RECIPE_VERSION,'signature':signature,'fileHash':file_hash(cache_path,32_000_000),'weights':weights,'device':device,'executedNodeIDs':executed,'reusedNodeIDs':reused,'stopAt':stop_at,'inferenceOnly':True,'inputProvenance':input_provenance}
        storage.atomic_json('activation-cache.json',meta,PREVIEW_ARTIFACT_LIMITS['activation-cache.json'])
        result={'event':'previewCompleted','initialWeightsHash':initial_weights_hash,'finalWeightsHash':final_weights_hash,'recipeVersion':RECIPE_VERSION,'previews':previews,'executedNodeIDs':executed,'reusedNodeIDs':reused,'cachePath':str(cache_path),'outputDir':str(output),'weights':weights,'device':device,'sourceHash':source.get('sourceHash'),'stopAt':stop_at,'inputProvenance':input_provenance}
        check_cancelled()
        storage.atomic_json('preview.json',result,PREVIEW_ARTIFACT_LIMITS['preview.json']);check_cancelled()
        emit(result); return result
