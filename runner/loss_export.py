"""Generate portable Torch-only objective code from the same reviewed primitives."""
import ast
import json
from pathlib import Path
if __package__:
    from .loss_configuration import validate_recipe
else:
    from loss_configuration import validate_recipe

MAX_SOURCE_BYTES = 131072


def _definitions(filename):
    """Include owned definitions, without transporting package imports or app paths."""
    source = Path(__file__).with_name(filename).read_text()
    lines = source.splitlines(keepends=True)
    return '\n'.join(''.join(lines[n.lineno - 1:n.end_lineno])
                     for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)))


def objective_recipe_code(recipe, scalarization='mean'):
    recipe = validate_recipe(recipe)
    if type(scalarization) is not str or scalarization not in ('mean', 'sum'):
        raise ValueError('Choose mean or sum for unreduced objective scalarization.')
    configuration = Path(__file__).with_name('loss_configuration.py').read_text()
    header = '''# KwipML portable configured objective. Requires PyTorch and the Python standard library.
# Predictions and prepared target roles preserve their observation axis.
# This contains the configured objective, not model weights or an optimizer.
import copy,json,math,struct
import torch
from torch import nn
from torch.nn import functional as F
class BatchedLayout:
    def __init__(self,sample):
        if not isinstance(sample,(tuple,list)) or not 1<=len(sample)<=7:
            raise ValueError('Prediction requires 1..7 sample directions.')
        if any(type(v) is not int or v<=0 for v in sample):
            raise ValueError('Prediction sample dimensions must be positive integers.')
        self.sample=tuple(sample)
'''
    footer = '\nOBJECTIVE_RECIPE = json.loads(' + repr(json.dumps(recipe, allow_nan=False, separators=(',', ':'))) + ')\n'
    footer += 'OBJECTIVE_SCALARIZATION = ' + repr(scalarization) + '\n'
    footer += '''
def create_training_objective(device='cpu'):
    """Create fixed configured weights on the selected CPU/MPS device."""
    return create_objective(copy.deepcopy(OBJECTIVE_RECIPE),device=device,dtype=torch.float32)

def evaluate_training_objective(objective,prediction,target_roles):
    """Return raw values, the backpropagation scalar and bounded measured metadata."""
    if type(objective) is not LossObjective or objective.recipe!=OBJECTIVE_RECIPE:
        raise ValueError('Objective differs from this exported configured recipe.')
    return evaluate_objective(objective,prediction,target_roles,OBJECTIVE_SCALARIZATION)
'''
    result = header + '\n' + configuration + '\n' + _definitions('loss_objectives.py') + '\n' + footer
    if len(result.encode('utf-8')) > MAX_SOURCE_BYTES:
        raise ValueError('Portable objective source exceeds its 131072-byte artifact bound.')
    compile(result, 'objective_recipe.py', 'exec')
    return result
