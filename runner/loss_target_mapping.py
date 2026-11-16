"""Torch-free exact typed target mapping admission, shared by worker authority."""
import copy
import math
if __package__:
    from .loss_configuration import validate_recipe
else:
    from loss_configuration import validate_recipe

def _columns(value):
    return (type(value) is list and 1 <= len(value) <= 128
            and all(type(v) is str and v.strip() and len(v) <= 200 for v in value)
            and len(set(value)) == len(value))


def validate_target_mapping(mapping, recipe):
    """Validate the Core mapping's exact versioned wire contract before allocation."""
    recipe = validate_recipe(recipe)
    keys = {'schemaVersion', 'componentID', 'inputColumns', 'roles', 'tripletSourceGroups'}
    if (type(mapping) is not dict or set(mapping) != keys
            or type(mapping['schemaVersion']) is not int or mapping['schemaVersion'] not in (1,2)
            or mapping['componentID'] != recipe['componentID']
            or not _columns(mapping['inputColumns'])):
        raise ValueError('Choose a version 1 target mapping for this exact objective and distinct input columns.')
    roles = mapping['roles']; groups = mapping['tripletSourceGroups']; op = recipe['componentID']
    if mapping['schemaVersion']==2:
        c=recipe['config']
        if op!='loss_cross_entropy' or c!=canonical_sequence_configuration():raise ValueError('Sequence targets require unweighted, unsmoothed next-token mean cross entropy with ignore index -100.')
        if type(roles) is not list or len(roles)!=1 or groups!=[]:raise ValueError('Sequence target mapping requires only the target sequence.')
        r=roles[0]
        if type(r) is not dict or set(r)!={'role','columns','sampleShape','dtype'} or r['role']!='target' or r['dtype']!='int64' or not _columns(r['columns']) or len(r['columns'])!=1 or set(r['columns'])&set(mapping['inputColumns']) or type(r['sampleShape']) is not list or len(r['sampleShape'])!=1 or type(r['sampleShape'][0]) is not int or not 1<=r['sampleShape'][0]<=4096:raise ValueError('Sequence targets need one typed sequence column and exact context length.')
        return copy.deepcopy(mapping)
    if type(roles) is not list or len(roles) > 3 or type(groups) is not list:
        raise ValueError('Objective roles and triplet source groups must be bounded lists.')
    if op == 'loss_triplet_margin':
        if (roles or len(groups) != 3 or any(not _columns(g) for g in groups)
                or len({len(g) for g in groups}) != 1
                or [v for g in groups for v in g] != mapping['inputColumns']):
            raise ValueError('Triplet requires three equally sized ordered source groups and no dummy target.')
        return copy.deepcopy(mapping)
    if groups:
        raise ValueError('Only Triplet accepts source groups.')
    expected = {'target'}
    if op == 'loss_ctc': expected |= {'input_lengths', 'target_lengths'}
    if op == 'loss_bce_with_logits':
        expected |= {k for k in ('weight', 'pos_weight') if recipe['config'][k + '_mode'] == 'mapped'}
    used = set(mapping['inputColumns']); names = set()
    for role in roles:
        if (type(role) is not dict or set(role) != {'role', 'columns', 'sampleShape', 'dtype'}
                or type(role['role']) is not str or role['role'] not in expected
                or role['role'] in names or not _columns(role['columns'])
                or used.intersection(role['columns'])):
            raise ValueError('Each objective role needs its own distinct ordered target columns.')
        shape = role['sampleShape']
        if (type(shape) is not list or len(shape) > 8
                or any(type(v) is not int or not 1 <= v <= 128 for v in shape)
                or math.prod(shape) != len(role['columns'])):
            raise ValueError('Target columns must exactly fill a bounded row-major sample shape.')
        integer_target = op in ('loss_nll', 'loss_ctc') or op == 'loss_cross_entropy' and recipe['config']['target_mode'] == 'class_indices'
        dtype = 'int64' if integer_target else 'float32'
        if role['dtype'] != dtype:
            raise ValueError('Target dtype differs from this configured objective.')
        if op == 'loss_ctc' and (len(shape) != 1 if role['role'] == 'target' else bool(shape)):
            raise ValueError('CTC targets are integer sequences and lengths are integer scalars.')
        names.add(role['role']); used.update(role['columns'])
    if names != expected:
        raise ValueError('Connect exactly the target roles used by this objective recipe.')
    return copy.deepcopy(mapping)




def canonical_sequence_configuration():
    return dict(reduction='mean',target_mode='class_indices',class_axis=-1,class_weights=[],ignore_index=-100,label_smoothing=0.)
