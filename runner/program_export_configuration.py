"""Exact version-one inference-program controls; defaults are never merged."""
DEFAULTS = {'strict': True, 'batch_mode': 'static', 'example_batch_size': 2,
            'minimum_batch_size': 2, 'maximum_batch_size': 2, 'device': 'cpu',
            'prefer_deferred_runtime_asserts': False}


def normalize_program_export(value):
    if type(value) is not dict or set(value) != set(DEFAULTS):
        raise ValueError('Program export requires exactly seven configuration controls')
    if any(type(value[k]) is not bool for k in ('strict', 'prefer_deferred_runtime_asserts')):
        raise ValueError('Program export boolean controls must be booleans')
    if value['batch_mode'] not in ('static', 'bounded') or value['device'] not in ('cpu', 'mps'):
        raise ValueError('Program export requires static/bounded batch and cpu/mps device')
    for key, ceiling in [('example_batch_size', 128), ('minimum_batch_size', 4096), ('maximum_batch_size', 4096)]:
        if type(value[key]) is not int or not 1 <= value[key] <= ceiling:
            raise ValueError('Program export batch sizes exceed reviewed bounds')
    lo, hi, example = (value[k] for k in ('minimum_batch_size', 'maximum_batch_size', 'example_batch_size'))
    if value['batch_mode'] == 'static' and not lo == hi == example:
        raise ValueError('Static batch sizes must be identical')
    if value['batch_mode'] == 'bounded' and not (lo < hi and lo <= example <= hi and example >= 2):
        raise ValueError('Bounded batch requires ordered bounds and example batch at least two')
    return dict(value)


def capability():
    """Independent literal runtime producer; never reads an authored bundle."""
    return {'schemaVersion':1,'version':'1.0.0','adapterID':'kwip.runner.program_export',
            'recipeSchemaVersion':1,'configurationVersion':1,
            'modes':['program-export','shape-constraints','program-reload-execution'],
            'devices':['cpu','mps'],
            'execution':{'evaluationOnly':True,'singleTensorInputOutput':True,'leadingBatch':['static','bounded'],
                         'fixedSampleAxes':True,'devicePolicy':'explicit-no-fallback','bagPacket':'refused'},
            'controls':{'strict':{'type':'boolean','default':True},
                        'batch_mode':{'type':'enum','values':['static','bounded'],'default':'static'},
                        'example_batch_size':{'type':'integer','minimum':1,'maximum':128,'default':2},
                        'minimum_batch_size':{'type':'integer','minimum':1,'maximum':4096,'default':2},
                        'maximum_batch_size':{'type':'integer','minimum':1,'maximum':4096,'default':2},
                        'device':{'type':'enum','values':['cpu','mps'],'default':'cpu'},
                        'prefer_deferred_runtime_asserts':{'type':'boolean','default':False}},
            'budgets':{'parameterElements':2000000,'bufferElements':2000000,'constantElements':2000000,
                       'inputElements':2000000,'outputElements':2000000,'tensorRank':8,'sampleAxis':4096,
                       'graphNodes':4096,'programBytes':100000000,'manifestBytes':2000000,
                       'archiveEntries':512,'archiveUncompressedBytes':100000000},
            'nativeAPIs':{'export':'torch.export.export(mod, args, kwargs=None, *, dynamic_shapes=None, strict=False, preserve_module_call_signature=(), prefer_deferred_runtime_asserts_over_guards=False)',
                          'save':'torch.export.save(ep, f, *, extra_files=None, opset_version=None, pickle_protocol=2)',
                          'load':'torch.export.load(f, *, extra_files=None, expected_opset_version=None)'},
            'trust':'trusted-app-generated-only'}
