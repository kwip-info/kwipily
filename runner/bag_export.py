"""Bounded fixed-definition source generation for standalone bag exports."""
import ast
import hashlib
import inspect


def _selected(module,names,rename=None):
    tree=ast.parse(inspect.getsource(module))
    selected=[];found=set()
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if rename and node.id in rename:node.id=rename[node.id]
            return node
    for node in tree.body:
        name=node.name if isinstance(node,(ast.FunctionDef,ast.ClassDef)) else (
            node.targets[0].id if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) else None)
        if name in names:
            found.add(name);selected.append(Rename().visit(node))
    if found!=set(names):raise ValueError('Standalone bag helper definitions differ from their reviewed source inventory.')
    return ast.unparse(ast.Module(body=selected,type_ignores=[]))+'\n'


def _coerce_bag_packet(value,offsets,per_sample_weights,config):
    if type(value) is EmbeddingBagPacket:
        if offsets is not None or per_sample_weights is not None:
            raise ValueError('Supply either a typed bag packet or separate indices/offsets/weights, never both.')
        return value
    if not isinstance(value,torch.Tensor):
        raise ValueError('Bag inference requires actual index tensors or its exported typed packet.')
    return EmbeddingBagPacket(config['layout'],value,offsets,per_sample_weights,config['include_last_offset'])


def standalone_bag_helpers():
    """Copy only reviewed declarations; no filesystem/runtime imports in output."""
    if __package__:
        from . import embedding_bag_configuration as configuration,embedding_bag_packet as packet,bag_source as source,embedding_bag as native
    else:
        import embedding_bag_configuration as configuration,embedding_bag_packet as packet,bag_source as source,embedding_bag as native
    body='import torch\nimport math\nimport struct\nfrom dataclasses import dataclass\n'
    body+=_selected(configuration,{'MAX_ELEMENTS','FLOAT32_MAX','DEFAULTS','FIELDS','_integer','_positive','normalize_embedding_bag'}, {'FIELDS':'EMBEDDING_BAG_FIELDS'})
    body+=_selected(packet,{'MAXIMUM_BAGS','MAXIMUM_ITEMS','MAXIMUM_OUTPUT_ELEMENTS','MAXIMUM_TABLE_SHAPE_ELEMENTS','EmbeddingBagPacket','validate_packet','_positive_integer','_tensor'})
    body+=_selected(source,{'FIELDS','normalize_bag_source','execute_bag_source'}, {'FIELDS':'BAG_SOURCE_FIELDS'})
    body+=_selected(native,{'MAXIMUM_PARAMETERS','CheckedEmbeddingBag','_device_controls'})
    body+=inspect.getsource(_coerce_bag_packet)+'\n'
    result='BAG_HELPER_SHA256 = '+repr(hashlib.sha256(body.encode('utf8')).hexdigest())+'\n'+body
    if len(result.encode('utf8'))>131072:raise ValueError('Standalone bag helper source exceeds its131072-byte bound.')
    return result
