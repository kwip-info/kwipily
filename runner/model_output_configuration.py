"""Model Output is an identity boundary with optional assertions, never a transform."""
import inspect


def validate_model_output_config(config):
    if type(config) is dict and 'task' in config:
        if config!={'task':'gamePolicyValue','taskVersion':1} or type(config.get('taskVersion')) is not int:
            raise ValueError('Model Output: unsupported versioned task.')
        return dict(config)
    defaults={'checkExpectedOutput':False,'expectedDType':'any','expectedSampleRank':'any','expectedSampleDimensions':[]}
    if type(config) is not dict or set(config)-set(defaults):
        raise ValueError('Model Output: unsupported expected-output configuration.')
    value=dict(defaults,**config)
    if (type(value['checkExpectedOutput']) is not bool
            or value['expectedDType'] not in ('any','float32','float64','int32','int64','bool')
            or value['expectedSampleRank'] not in ('any','exact')
            or type(value['expectedSampleDimensions']) is not list
            or len(value['expectedSampleDimensions'])>7
            or any(v is not None and (type(v) is not int or not 1<=v<=100_000_000) for v in value['expectedSampleDimensions'])):
        raise ValueError('Model Output: choose supported dtype, Any/Exact rank and positive exact sample dimensions (maximum seven).')
    return value


def check_model_output(value, config, node_id='output'):
    settings=validate_model_output_config(config)
    if settings.get('task')=='gamePolicyValue':
        if type(value) is not dict or set(value)!={'policy','value'}:
            raise ValueError('Game policy/value requires explicit policy and value tensors.')
        policy, scalar=value['policy'],value['value']
        if (str(policy.dtype)!='torch.float32' or str(scalar.dtype)!='torch.float32'
                or policy.ndim!=2 or not 1<=policy.shape[1]<=256 or policy.shape[0]<1
                or tuple(scalar.shape)!=(policy.shape[0],1)
                or not policy.isfinite().all().item() or not scalar.isfinite().all().item()
                or not (scalar.abs()<=1).all().item()):
            raise ValueError('Game policy/value requires finite float32 [batch, actions] logits and [batch, 1] value in −1…1.')
        return value
    if not settings['checkExpectedOutput']:return value
    prefix=f'{node_id}: Model Output'
    dtype=str(value.dtype).removeprefix('torch.')
    expected=settings['expectedDType']
    if expected!='any' and dtype!=expected:
        raise ValueError(f'{prefix}: expected dtype {expected}; captured {dtype}. Change the expected type or the upstream operation.')
    if settings['expectedSampleRank']=='exact':
        dimensions=settings['expectedSampleDimensions'];shape=tuple(value.shape)
        if len(shape)!=len(dimensions)+1:
            raise ValueError(f'{prefix}: expected sample rank {len(dimensions)} after the batch axis; captured shape {shape}.')
        for axis,(actual,wanted) in enumerate(zip(shape[1:],dimensions),start=1):
            if wanted is not None and actual!=wanted:
                raise ValueError(f'{prefix}: sample dimension {axis} expected {wanted}; captured {actual}. Change the expectation or upstream shape.')
    return value


# Embed the actual reviewed implementation in derived Code; replay needs no app-private import.
MODEL_OUTPUT_CODE_SOURCE=inspect.getsource(validate_model_output_config)+'\n'+inspect.getsource(check_model_output)
