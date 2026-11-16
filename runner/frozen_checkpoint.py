"""Pinned checkpoint loading for inference; no optimizer or RNG restoration."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile
import torch
if __package__:
    from .engine import GraphModel
    from .graph_semantics import semantic_graph
    from .prepared_data import bounded_read
    from .checkpoint_lifecycle import MAX_CHECKPOINT_BYTES, digest, tensor_digest, validate_objective_checkpoint
else:
    from engine import GraphModel
    from graph_semantics import semantic_graph
    from prepared_data import bounded_read
    from checkpoint_lifecycle import MAX_CHECKPOINT_BYTES, digest, tensor_digest, validate_objective_checkpoint


@dataclass(frozen=True)
class FrozenModel:
    model: torch.nn.Module
    archived_graph: dict
    input_contract: dict
    training_provenance: dict
    lineage: dict


def _pinned(path, limit, expected):
    if not isinstance(expected,str) or re.fullmatch('[0-9a-f]{64}',expected) is None:
        raise ValueError('Frozen model requires explicit SHA-256 identity pins')
    raw=bounded_read(path,limit)
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Frozen model artifact identity mismatch')
    return raw


def _check_storage_archive(raw):
    # KwipML's checkpoint writer uses the current uncompressed torch.save ZIP
    # format. Bound expanded entries before tensor deserialization as well as
    # bounding the bytes read from disk; weights_only is not an allocation limit.
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries=archive.infolist()
            if (not 1<=len(entries)<=10000 or len({e.filename for e in entries})!=len(entries) or
                sum(e.file_size for e in entries)>MAX_CHECKPOINT_BYTES):
                raise ValueError('Frozen checkpoint expanded storage exceeds the archive budget')
            if any(e.flag_bits & 1 or e.compress_type!=zipfile.ZIP_STORED for e in entries):
                raise ValueError('Frozen checkpoint requires the uncompressed torch storage archive format')
    except zipfile.BadZipFile as error:
        raise ValueError('Frozen checkpoint storage archive is invalid') from error


def load_frozen(checkpoint_path, *, expected_manifest_sha256, expected_checkpoint_sha256,
                registry_path, expected_registry_sha256):
    """Caller obtains pins from independently verified archive selection.

    The registry is copied from verified bytes to a private temporary file because
    GraphModel's current validator takes a path. It never rereads caller-controlled
    registry bytes during construction. Temporary initialization preserves CPU RNG.
    """
    path=Path(checkpoint_path)
    manifest_raw=_pinned(path.with_name('manifest.json'),20_000_000,expected_manifest_sha256)
    checkpoint_raw=_pinned(path,MAX_CHECKPOINT_BYTES,expected_checkpoint_sha256)
    registry_raw=_pinned(registry_path,2_000_000,expected_registry_sha256)
    try:
        manifest=json.loads(manifest_raw)
        metadata=manifest['checkpoint'];stored=metadata['contract'];graph=manifest['graph']
        provenance=manifest['provenance']
        if (type(manifest.get('schemaVersion')) is not int or manifest['schemaVersion']!=1 or
            manifest.get('status')!='completed' or not isinstance(manifest.get('runID'),str) or
            not 1<=len(manifest['runID'])<=200 or type(metadata.get('schemaVersion')) is not int or
            metadata['schemaVersion']!=1 or type(stored.get('schemaVersion')) is not int or stored['schemaVersion']!=1):
            raise ValueError('Frozen model requires a completed versioned parent archive')
        if (metadata['sha256']!=expected_checkpoint_sha256 or type(metadata['bytes']) is not int or
            metadata['bytes']!=len(checkpoint_raw)):
            raise ValueError('Frozen checkpoint differs from its parent manifest')
        if digest(graph)!=manifest['graphHash'] or digest(semantic_graph(graph))!=stored['semanticGraphDigest']:
            raise ValueError('Frozen architecture differs from its checkpoint contract')
        for key in ('sourceHash','splitHash'):
            if not isinstance(provenance.get(key),str) or re.fullmatch('[0-9a-f]{64}',provenance[key]) is None:
                raise ValueError('Frozen training provenance is incomplete')
        if (stored['data']['sourceDigest']!=provenance['sourceHash'] or
            stored['data']['splitDigest']!=provenance['splitHash'] or
            stored['data'].get('preparation')!=provenance.get('preparation')):
            raise ValueError('Frozen training provenance differs from checkpoint contract')
        # Frozen evaluation still maps dense test columns via dtype/sampleShape.
        # Refuse packet sources before tensor deserialization or native allocation;
        # the trained portable packet inference export owns that supported route.
        if any(isinstance(node,dict) and node.get('type') in ('bag_input','embedding_bag')
               for node in graph.get('nodes',[])):
            raise ValueError('Frozen model evaluation does not support Bag Input packets; use the trained packet inference export for EmbeddingBag predictions.')
        _check_storage_archive(checkpoint_raw)
        checkpoint=torch.load(io.BytesIO(checkpoint_raw),map_location='cpu',weights_only=True)
        if (not isinstance(checkpoint,dict) or type(checkpoint.get('checkpointSchemaVersion')) is not int or
            checkpoint['checkpointSchemaVersion']!=1 or checkpoint.get('contract')!=stored or
            type(checkpoint.get('epoch')) is not int or not 1<=checkpoint['epoch']<=10000 or
            checkpoint['epoch']!=metadata['completedEpoch']):
            raise ValueError('Frozen checkpoint payload disagrees with its manifest')
        objective=validate_objective_checkpoint(stored,manifest,checkpoint)
        context={k:objective[k] for k in ('recipe','targetMapping')} if objective else None
        if objective and (provenance.get('objectiveRecipe')!=objective['recipe'] or provenance.get('objectiveMapping')!=objective['targetMapping']):raise ValueError('Frozen preparation objective differs from checkpoint.')
        with tempfile.TemporaryDirectory(prefix='kwip-frozen-registry-') as folder:
            registry=Path(folder)/'components.json';registry.write_bytes(registry_raw)
            with torch.random.fork_rng(devices=[]):
                model=GraphModel(graph,registry=registry,bundled_dataset=provenance.get('dataset') not in ('prepared-dataflow-v1','named-prepared-v1','prepared-objective-dataflow-v1'),objective_context=context)
                model.materialize_lazy_parameters()
        saved=checkpoint.get('modelState');expected=model.state_dict()
        if not isinstance(saved,dict) or set(saved)!=set(expected):
            raise ValueError('Frozen checkpoint parameter names are incompatible')
        for name,value in saved.items():
            reference=expected[name]
            if (not isinstance(value,torch.Tensor) or value.is_nested or value.layout!=torch.strided or
                value.is_quantized or value.shape!=reference.shape or value.dtype!=reference.dtype or
                not torch.isfinite(value).all()):
                raise ValueError('Frozen checkpoint parameter shape, dtype or values are incompatible')
        model.load_state_dict(saved,strict=True)
        model.eval();model.requires_grad_(False)
        input_ids=[ident for ident in model.order if model.nodes[ident]['type'] in ('input','index_input')]
        if (len(input_ids)!=1 and not graph.get('namedInputs')) or (objective is None and tuple(model.layouts[model.output_id].sample)!=(2,)) or model.dtypes[model.output_id]!='float32':
            raise ValueError('Frozen evaluation currently requires one input and two classifier outputs')
        input_id=input_ids[0]
        lineage={'parentRunID':manifest['runID'],'manifestSHA256':expected_manifest_sha256,
                 'checkpointSHA256':expected_checkpoint_sha256,'registrySHA256':expected_registry_sha256,
                 'completedEpoch':checkpoint['epoch'],'trainingRuntime':stored['runtime'],
                 'stateDigest':tensor_digest([saved[name] for name in sorted(saved)]),'mode':'frozenInference'}
        input_contract={'dtype':model.dtypes[input_id],'sampleShape':list(model.layouts[input_id].sample)}
        if provenance.get('namedInputPlan') is not None:
            if stored['data'].get('namedInputPlan')!=provenance['namedInputPlan']:raise ValueError('Frozen named mapping differs from its checkpoint contract')
            input_contract={'namedInputPlan':provenance['namedInputPlan']}
        return FrozenModel(model,graph,input_contract,provenance,lineage)
    except (KeyError,TypeError,RuntimeError,json.JSONDecodeError) as error:
        raise ValueError('Frozen checkpoint archive is malformed or incompatible') from error
