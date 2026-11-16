"""Batch-preserving concrete shape rules and tensor transforms.

Mirrors native BatchedTensorLayout: symbolic positive N, 1..7 sample axes,
positive dimensions, <=100M elements per sample. Only matmul broadcasts sample axes.
These helpers do not register new graph operators by themselves.
"""
from dataclasses import dataclass
import math

MAX_ELEMENTS = 100_000_000
MAX_RANK = 8


def _product(shape):
    total = 1
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0 or dimension > MAX_ELEMENTS // total:
            raise ValueError('Sample dimensions must be positive integers with at most 100M elements.')
        total *= dimension
    return total


@dataclass(frozen=True)
class BatchedLayout:
    sample: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.sample, (tuple, list)) or not 1 <= len(self.sample) < MAX_RANK:
            raise ValueError('A batched tensor requires 1..7 concrete sample axes.')
        object.__setattr__(self, 'sample', tuple(self.sample))
        _product(self.sample)

    def _axis(self, axis):
        rank = len(self.sample) + 1
        if type(axis) is not int or not -rank <= axis < rank:
            raise ValueError('Axis is outside the tensor rank.')
        normalized = axis % rank
        if normalized == 0:
            raise ValueError('Axis 0 is the batch; choose a sample axis.')
        return normalized - 1

    def softmax(self, axis):
        self._axis(axis)
        return self

    def layer_norm(self, normalized_shape):
        if not isinstance(normalized_shape,(list,tuple)) or not 1<=len(normalized_shape)<=len(self.sample):
            raise ValueError('Normalize only trailing sample dimensions; the batch must stay separate.')
        _product(normalized_shape)
        if tuple(normalized_shape)!=self.sample[-len(normalized_shape):]:
            raise ValueError('Normalized shape must match the trailing sample dimensions exactly.')
        return self

    def reshape(self, shape):
        if not isinstance(shape, (list, tuple)) or not 1 <= len(shape) < MAX_RANK:
            raise ValueError('Reshape requires 1..7 sample axes.')
        if any(type(d) is not int for d in shape) or shape.count(-1) > 1:
            raise ValueError('Reshape allows integer axes and at most one inferred -1.')
        known = _product([d for d in shape if d != -1])
        total = _product(self.sample)
        result = list(shape)
        if -1 in result:
            if total % known:
                raise ValueError('Reshape cannot infer an integer dimension.')
            result[result.index(-1)] = total // known
        if _product(result) != total:
            raise ValueError('Reshape must preserve the number of elements per observation.')
        return BatchedLayout(tuple(result))

    def transpose(self, first, second):
        a, b = self._axis(first), self._axis(second)
        result = list(self.sample)
        result[a], result[b] = result[b], result[a]
        return BatchedLayout(tuple(result))

    def flatten(self, start, end):
        first, last = self._axis(start), self._axis(end)
        if first > last:
            raise ValueError('Flatten start must precede or equal its end.')
        return BatchedLayout(self.sample[:first] + (_product(self.sample[first:last+1]),) + self.sample[last+1:])

    def linear(self, input_features, output_features):
        if type(input_features) is not int or self.sample[-1] != input_features:
            raise ValueError('Linear input width differs from the last sample dimension.')
        return BatchedLayout(self.sample[:-1] + (output_features,))

    def bilinear(self, other, first_features, second_features, output_features):
        if any(type(v) is not int or not 1 <= v <= 512
               for v in (first_features, second_features, output_features)):
            raise ValueError('Bilinear widths must be integers from 1 to 512.')
        if self.sample[:-1] != other.sample[:-1]:
            raise ValueError('Bilinear leading sample dimensions must match exactly; broadcasting is not enabled.')
        if self.sample[-1] != first_features or other.sample[-1] != second_features:
            raise ValueError('Bilinear input widths must match the respective final sample dimensions.')
        return BatchedLayout(self.sample[:-1] + (output_features,))

    def concat(self, other, axis=-1):
        selected = self._axis(axis)
        if len(self.sample) != len(other.sample) or any(a != b for i, (a, b) in enumerate(zip(self.sample, other.sample)) if i != selected):
            raise ValueError('Concatenation requires matching ranks and nonselected dimensions.')
        result = list(self.sample)
        result[selected] += other.sample[selected]
        return BatchedLayout(tuple(result))

    def add(self, other, broadcast=False):
        if type(broadcast) is not bool:
            raise ValueError('Sample broadcasting must be Boolean.')
        if not broadcast:
            if self != other:
                raise ValueError('Add requires identical sample shapes; implicit broadcasting is not enabled.')
            return self
        count=max(len(self.sample),len(other.sample))
        left=(1,)*(count-len(self.sample))+self.sample
        right=(1,)*(count-len(other.sample))+other.sample
        result=[]
        for a,b in zip(left,right):
            if a!=b and a!=1 and b!=1:
                raise ValueError('Add sample directions must match or have size 1; observations never broadcast.')
            result.append(max(a,b))
        return BatchedLayout(tuple(result))

    def stack(self, other, axis=-1):
        if self != other:
            raise ValueError('Stack requires identical sample shapes, without broadcasting.')
        rank=len(self.sample)+2
        if rank>MAX_RANK or type(axis) is not int or not -rank<=axis<rank or axis%rank==0:
            raise ValueError('Stack needs input rank 2–7 and a new sample direction; observations cannot move.')
        selected=axis%rank-1
        return BatchedLayout(self.sample[:selected]+(2,)+self.sample[selected:])

    def permute(self, dims):
        rank=len(self.sample)+1
        if type(dims) is not list or len(dims)!=rank or any(type(v) is not int or not -rank<=v<rank for v in dims):
            raise ValueError('Permute requires the complete input-rank direction order.')
        order=[v%rank for v in dims]
        if order[0]!=0 or set(order)!=set(range(rank)):
            raise ValueError('Permute uses each direction once, with observations first.')
        return BatchedLayout(tuple(self.sample[v-1] for v in order[1:]))

    def matmul(self, other):
        a, b = self.sample, other.sample
        if len(a) < 2 or len(b) < 2:
            raise ValueError('Matrix multiplication requires two matrix axes per sample; reshape vectors explicitly.')
        if a[-1] != b[-2]:
            raise ValueError('Matrix multiplication contraction widths must match.')
        count = max(len(a), len(b)) - 2
        left = (1,) * (count - len(a) + 2) + a[:-2]
        right = (1,) * (count - len(b) + 2) + b[:-2]
        result = []
        for x, y in zip(left, right):
            if x != y and x != 1 and y != 1:
                raise ValueError('Matrix multiplication sample axes must match or be 1; the batch never broadcasts.')
            result.append(max(x, y))
        return BatchedLayout(tuple(result) + (a[-2], b[-1]))


def normalize_configuration(operation, config):
    """Canonical JSON settings, matching native TensorTransformConfiguration.

    Integral JSON floats are accepted at this boundary; concrete layout algebra
    remains strictly integer typed. Shape-dependent checks follow at execution.
    """
    expected = {'reshape': {'shape'}, 'transpose': {'dim0', 'dim1'}, 'flatten': {'start_dim', 'end_dim'}}
    if operation not in expected or type(config) is not dict or set(config) != expected[operation]:
        raise ValueError('Unsupported layout operation or configuration fields.')

    def integer(value):
        if type(value) is int:
            return value
        if type(value) is float and math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError('Transform settings require integer numbers, not booleans or fractions.')

    if operation == 'reshape':
        raw = config['shape']
        if type(raw) is not list or not 1 <= len(raw) < MAX_RANK:
            raise ValueError('Reshape requires a JSON array of 1..7 sample dimensions.')
        shape = [integer(v) for v in raw]
        if shape.count(-1) > 1:
            raise ValueError('Only one reshape dimension can be inferred with -1.')
        _product([1 if v == -1 else v for v in shape])
        return {'shape': shape}
    result = {key: integer(value) for key, value in config.items()}
    if any(value == 0 or not -MAX_RANK <= value < MAX_RANK for value in result.values()):
        raise ValueError('Axis must be a bounded sample axis; zero is reserved for the batch.')
    return result


def apply_layout(tensor, operation, config):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[0] <= 0:
        raise ValueError('Layout operations require a nonempty batch and concrete sample axes.')
    shape = BatchedLayout(tuple(tensor.shape[1:]))
    config = normalize_configuration(operation, config)
    if operation == 'reshape':
        result = shape.reshape(config['shape'])
        return tensor.reshape((tensor.shape[0],) + result.sample)
    if operation == 'transpose':
        shape.transpose(config['dim0'], config['dim1'])
        return tensor.transpose(config['dim0'], config['dim1'])
    shape.flatten(config['start_dim'], config['end_dim'])
    return tensor.flatten(config['start_dim'], config['end_dim'])
