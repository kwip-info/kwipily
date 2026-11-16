#!/usr/bin/env python3
"""Validate the first migrated component family and reproduce shipped declarations.

No dynamic code loading: adapter IDs are references to reviewed bundled code.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate {label}')


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError('Unknown or missing manifest keys')


REMAINING_ACTIVATION_DEFINITIONS = {'prelu': {'id': 'prelu',
           'title': 'PReLU',
           'category': 'Activations',
           'inputs': ['in'],
           'outputs': ['out'],
           'fields': [{'key': 'num_parameters',
                       'title': 'Slope count',
                       'kind': 'integer',
                       'defaultValue': 1,
                       'min': 1,
                       'max': 4096},
                      {'key': 'init',
                       'title': 'Initial slope',
                       'kind': 'number',
                       'defaultValue': 0.25,
                       'min': -3.4028234663852886e+38,
                       'max': 3.4028234663852886e+38}]},
 'log_softmax': {'id': 'log_softmax',
                 'title': 'LogSoftmax',
                 'category': 'Activations',
                 'inputs': ['in'],
                 'outputs': ['out'],
                 'fields': [{'key': 'dim',
                             'title': 'Normalization direction',
                             'kind': 'integer',
                             'defaultValue': -1,
                             'min': -8,
                             'max': 7}]},
 'dropout': {'id': 'dropout',
             'title': 'Dropout',
             'category': 'Activations',
             'inputs': ['in'],
             'outputs': ['out'],
             'fields': [{'key': 'p',
                         'title': 'Drop probability',
                         'kind': 'number',
                         'defaultValue': 0.5,
                         'min': 0,
                         'max': 1}]}}

REMAINING_ACTIVATION_CONTRACTS = {'prelu': {'version': {'major': 1, 'minor': 2, 'patch': 0},
           'adapterID': 'kwip.runner.graph-v2.prelu',
           'graphSchemaVersions': [2],
           'maximumRank': 8,
           'preservesBatch': True,
           'graphDTypes': ['float32'],
           'maximumTensorElements': 100000000,
           'parameterCountMinimum': 1,
           'parameterCountMaximum': 4096,
           'initMinimum': -3.4028234663852886e+38,
           'initMaximum': 3.4028234663852886e+38,
           'channelAxis': 1,
           'learnedParameters': True,
           'zeroInputGradient': 'slope'},
 'log_softmax': {'version': {'major': 1, 'minor': 2, 'patch': 0},
                 'adapterID': 'kwip.runner.graph-v2.log_softmax',
                 'graphSchemaVersions': [2],
                 'maximumRank': 8,
                 'preservesBatch': True,
                 'graphDTypes': ['float32'],
                 'maximumTensorElements': 100000000,
                 'axisMinimum': -8,
                 'axisMaximum': 7,
                 'excludesBatch': True,
                 'outputKind': 'logarithmicProbabilities'},
 'dropout': {'version': {'major': 1, 'minor': 2, 'patch': 0},
             'adapterID': 'kwip.runner.graph-v2.dropout',
             'graphSchemaVersions': [2],
             'maximumRank': 8,
             'preservesBatch': True,
             'graphDTypes': ['float32'],
             'maximumTensorElements': 100000000,
             'probabilityMinimum': 0,
             'probabilityMaximum': 1,
             'modeSource': 'parentModule',
             'stochasticTraining': True,
             'evaluationIdentity': True,
             'checkpointGuarantees': {'cpu': 'epochBoundaryExactCPU', 'mps': 'weightsOnly'}}}


def validate_remaining_activation(row, root):
    ident = row['definition']['id']
    if json.dumps(row['definition'],sort_keys=True) != json.dumps(REMAINING_ACTIVATION_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed remaining activation definition/settings')
    if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid remaining activation publication/version')
    if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Invalid remaining activation adapter/ports')
    if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(REMAINING_ACTIVATION_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed remaining activation runtime contract')
    exact(row['presentation'],'theme icon shortTitle purpose')
    if row['presentation']['theme'] != 'amber' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid remaining activation presentation')
    exact(row['sources'],'native worker tests evidence')
    references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
    learning = row['learning']
    if row['published'] and learning is None: raise ValueError('Published remaining activation requires learning')
    if learning is not None:
        exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
        lessons = {'prelu':['assessment.prelu-shared-slope','assessment.prelu-channel-slopes'],'log_softmax':['assessment.logsoftmax-sample-scores'],'dropout':['assessment.dropout-training-pattern','assessment.dropout-evaluation-pattern']}
        if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['lessonIDs'] != lessons[ident] or learning['exampleID'] != 'prepared-'+ident.replace('_','-')+'-classifier' or learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid remaining activation learning identity')
        if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing remaining activation prose')
        if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing remaining activation learning metadata')
        from datetime import date
        try: date.fromisoformat(learning['updatedAt'])
        except ValueError: raise ValueError('Invalid remaining activation date')
        references += [learning['exampleSource'],'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv']
    for reference in references:
        if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe remaining activation source reference')
        path=(root/reference).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe remaining activation source reference: '+reference)

def validate_activation(value, root):
    """Reviewed activation metadata; no arbitrary adapter or executable loading."""
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.activations' or type(value['version']) is not int or value['version'] < 1:
        raise ValueError('Unsupported activation bundle identity/version')
    if value['backend'] != 'pytorch': raise ValueError('Unsupported backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids = []
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        d = row['definition']
        exact(d, 'id title category inputs outputs fields')
        ids.append(d['id'])
        if d['id'] in REMAINING_ACTIVATION_DEFINITIONS:
            validate_remaining_activation(row,root)
            continue
        if d['id'] not in ('sigmoid', 'tanh', 'relu', 'gelu', 'softmax', 'silu', 'leaky_relu', 'elu'): raise ValueError('Unsupported activation definition')
        fields = ([{'key':'approximate_tanh','title':'Tanh approximation','kind':'boolean','defaultValue':False}] if d['id'] == 'gelu' else ([{'key':'dim','title':'Normalization axis','kind':'integer','defaultValue':-1,'min':-8,'max':7}] if d['id'] == 'softmax' else []))
        new_definitions = {'silu': {'id': 'silu', 'title': 'SiLU', 'category': 'Activations', 'inputs': ['in'], 'outputs': ['out'], 'fields': []}, 'leaky_relu': {'id': 'leaky_relu', 'title': 'LeakyReLU', 'category': 'Activations', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'negative_slope', 'title': 'Negative slope', 'kind': 'number', 'defaultValue': 0.01, 'min': -3.4028234663852886e+38, 'max': 3.4028234663852886e+38}]}, 'elu': {'id': 'elu', 'title': 'ELU', 'category': 'Activations', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'alpha', 'title': 'Alpha', 'kind': 'number', 'defaultValue': 1, 'min': -3.4028234663852886e+38, 'max': 3.4028234663852886e+38}]}}
        if d['id'] in new_definitions: fields = new_definitions[d['id']]['fields']
        definition = {'id':d['id'],'title':{'sigmoid':'Sigmoid','tanh':'Tanh','relu':'ReLU','gelu':'GELU','softmax':'Softmax','silu':'SiLU','leaky_relu':'LeakyReLU','elu':'ELU'}[d['id']],'category':'Activations','inputs':['in'],'outputs':['out'],'fields':fields}
        if json.dumps(d, sort_keys=True) != json.dumps(definition, sort_keys=True):
            raise ValueError('Unsupported activation definition/settings')
        if row['adapterID'] != 'kwip.runner.graph-v2.' + d['id'] or row['shapeRuleID'] != 'kwip.shape.' + d['id'] + '.v1':
            raise ValueError('Unknown activation adapter reference')
        if type(row['published']) is not bool or row['configurationVersion'] != 2:
            raise ValueError('Invalid publication/configuration version')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'amber' or row['presentation']['icon'] != d['id'] + '-v2':
            raise ValueError('Unknown activation theme/icon reference')
        if any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Empty presentation')
        if not (root / ('Sources/KwipMLApp/Resources/ComponentIcons/' + d['id'] + '-v2.png')).is_file():
            raise ValueError('Missing activation artwork reference')
        if row['outputConditions'] != {}: raise ValueError('Unsupported activation output conditions')
        expected = {'version':{'major':1,'minor':(0 if d['id'] == 'sigmoid' else 1),'patch':0},'adapterID':row['adapterID'],
                    'graphSchemaVersions':([2] if d['id'] == 'sigmoid' else [1,2]),'graphDTypes':['float32'],'preservesBatch':True,
                    'preservesShape':True,'configuration':{},'outputDTypes':{'out':'float32'},
                    'maximumRank':8,'parameterless':True,'deterministic':True}
        if d['id'] in new_definitions:
            expected = {'silu': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.silu', 'graphSchemaVersions': [2], 'graphDTypes': ['float32'], 'preservesBatch': True, 'preservesShape': True, 'configuration': {}, 'outputDTypes': {'out': 'float32'}, 'maximumRank': 8, 'parameterless': True, 'deterministic': True, 'inplace': False}, 'leaky_relu': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.leaky_relu', 'graphSchemaVersions': [2], 'graphDTypes': ['float32'], 'preservesBatch': True, 'preservesShape': True, 'configuration': {'negative_slope': {'default': 0.01, 'minimum': -3.4028234663852886e+38, 'maximum': 3.4028234663852886e+38}}, 'outputDTypes': {'out': 'float32'}, 'maximumRank': 8, 'parameterless': True, 'deterministic': True, 'inplace': False}, 'elu': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.elu', 'graphSchemaVersions': [2], 'graphDTypes': ['float32'], 'preservesBatch': True, 'preservesShape': True, 'configuration': {'alpha': {'default': 1, 'minimum': -3.4028234663852886e+38, 'maximum': 3.4028234663852886e+38}}, 'outputDTypes': {'out': 'float32'}, 'maximumRank': 8, 'parameterless': True, 'deterministic': True, 'inplace': False}}[d['id']]
        if d['id'] == 'relu': expected['inplace'] = False
        if d['id'] == 'gelu': expected = {'version':{'major':1,'minor':0,'patch':0},'adapterID':row['adapterID'],'graphSchemaVersions':[2],'approximateTanhModes':[False,True]}
        if d['id'] == 'softmax': expected = {'version':{'major':1,'minor':0,'patch':0},'adapterID':row['adapterID'],'graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True}
        if json.dumps(row['runtimeContract'], sort_keys=True) != json.dumps(expected, sort_keys=True): raise ValueError('Invalid activation runtime contract')
        exact(row['sources'], 'native worker tests evidence')
        for reference in row['sources'].values():
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path = (root / reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource' + (' lessonIDs' if d['id'] in new_definitions else ''))
            if d['id'] in new_definitions and learning['lessonIDs'] != [{'silu': 'assessment.silu-values', 'leaky_relu': 'assessment.leaky-relu-slope', 'elu': 'assessment.elu-alpha'}[d['id']]]: raise ValueError('Unknown assessment identity')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1: raise ValueError('Invalid learning content version')
            if learning['exampleID'] != 'prepared-' + d['id'] + '-classifier': raise ValueError('Unknown example ID')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(p,str) or not p.strip() for p in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            for key in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle'):
                if not isinstance(learning[key],str) or not learning[key].strip(): raise ValueError('Missing learning metadata')
            for reference in (learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv'):
                path = (root / reference).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe learning reference')
    unique(ids, 'component ID')


STATEFUL_NORMALIZATION_DEFINITIONS = {'batch_norm1d': {'id': 'batch_norm1d',
                  'title': 'BatchNorm1d',
                  'category': 'Normalization',
                  'inputs': ['in'],
                  'outputs': ['out'],
                  'fields': [{'key': 'num_features',
                              'title': 'Channels',
                              'kind': 'integer',
                              'defaultValue': 2,
                              'min': 1,
                              'max': 512},
                             {'key': 'eps',
                              'title': 'Safety value',
                              'kind': 'number',
                              'defaultValue': 1e-05,
                              'min': 1.1754943508222875e-38,
                              'max': 3.4028234663852886e+38},
                             {'key': 'averaging',
                              'title': 'Averaging method',
                              'kind': 'choice',
                              'defaultValue': 'moving',
                              'choices': [{'value': 'moving', 'title': 'Moving summary'},
                                          {'value': 'cumulative', 'title': 'Cumulative summary'}]},
                             {'key': 'momentum',
                              'title': 'Running-stat update weight',
                              'kind': 'number',
                              'defaultValue': 0.1,
                              'min': 0,
                              'max': 1},
                             {'key': 'affine',
                              'title': 'Learn scale and shift',
                              'kind': 'boolean',
                              'defaultValue': True},
                             {'key': 'bias',
                              'title': 'Learn shift',
                              'kind': 'boolean',
                              'defaultValue': True},
                             {'key': 'track_running_stats',
                              'title': 'Track running statistics',
                              'kind': 'boolean',
                              'defaultValue': True}]},
 'batch_norm2d': {'id': 'batch_norm2d',
                  'title': 'BatchNorm2d',
                  'category': 'Normalization',
                  'inputs': ['in'],
                  'outputs': ['out'],
                  'fields': [{'key': 'num_features',
                              'title': 'Channels',
                              'kind': 'integer',
                              'defaultValue': 2,
                              'min': 1,
                              'max': 512},
                             {'key': 'eps',
                              'title': 'Safety value',
                              'kind': 'number',
                              'defaultValue': 1e-05,
                              'min': 1.1754943508222875e-38,
                              'max': 3.4028234663852886e+38},
                             {'key': 'averaging',
                              'title': 'Averaging method',
                              'kind': 'choice',
                              'defaultValue': 'moving',
                              'choices': [{'value': 'moving', 'title': 'Moving summary'},
                                          {'value': 'cumulative', 'title': 'Cumulative summary'}]},
                             {'key': 'momentum',
                              'title': 'Running-stat update weight',
                              'kind': 'number',
                              'defaultValue': 0.1,
                              'min': 0,
                              'max': 1},
                             {'key': 'affine',
                              'title': 'Learn scale and shift',
                              'kind': 'boolean',
                              'defaultValue': True},
                             {'key': 'bias',
                              'title': 'Learn shift',
                              'kind': 'boolean',
                              'defaultValue': True},
                             {'key': 'track_running_stats',
                              'title': 'Track running statistics',
                              'kind': 'boolean',
                              'defaultValue': True}]},
 'instance_norm1d': {'id': 'instance_norm1d',
                     'title': 'InstanceNorm1d',
                     'category': 'Normalization',
                     'inputs': ['in'],
                     'outputs': ['out'],
                     'fields': [{'key': 'num_features',
                                 'title': 'Channels',
                                 'kind': 'integer',
                                 'defaultValue': 2,
                                 'min': 1,
                                 'max': 512},
                                {'key': 'eps',
                                 'title': 'Safety value',
                                 'kind': 'number',
                                 'defaultValue': 1e-05,
                                 'min': 1.1754943508222875e-38,
                                 'max': 3.4028234663852886e+38},
                                {'key': 'momentum',
                                 'title': 'Running-stat update weight',
                                 'kind': 'number',
                                 'defaultValue': 0.1,
                                 'min': 0,
                                 'max': 1},
                                {'key': 'affine',
                                 'title': 'Learn scale and shift',
                                 'kind': 'boolean',
                                 'defaultValue': False},
                                {'key': 'bias',
                                 'title': 'Learn shift',
                                 'kind': 'boolean',
                                 'defaultValue': True},
                                {'key': 'track_running_stats',
                                 'title': 'Track running statistics',
                                 'kind': 'boolean',
                                 'defaultValue': False}]},
 'instance_norm2d': {'id': 'instance_norm2d',
                     'title': 'InstanceNorm2d',
                     'category': 'Normalization',
                     'inputs': ['in'],
                     'outputs': ['out'],
                     'fields': [{'key': 'num_features',
                                 'title': 'Channels',
                                 'kind': 'integer',
                                 'defaultValue': 2,
                                 'min': 1,
                                 'max': 512},
                                {'key': 'eps',
                                 'title': 'Safety value',
                                 'kind': 'number',
                                 'defaultValue': 1e-05,
                                 'min': 1.1754943508222875e-38,
                                 'max': 3.4028234663852886e+38},
                                {'key': 'momentum',
                                 'title': 'Running-stat update weight',
                                 'kind': 'number',
                                 'defaultValue': 0.1,
                                 'min': 0,
                                 'max': 1},
                                {'key': 'affine',
                                 'title': 'Learn scale and shift',
                                 'kind': 'boolean',
                                 'defaultValue': False},
                                {'key': 'bias',
                                 'title': 'Learn shift',
                                 'kind': 'boolean',
                                 'defaultValue': True},
                                {'key': 'track_running_stats',
                                 'title': 'Track running statistics',
                                 'kind': 'boolean',
                                 'defaultValue': False}]},
 'group_norm': {'id': 'group_norm',
                'title': 'GroupNorm',
                'category': 'Normalization',
                'inputs': ['in'],
                'outputs': ['out'],
                'fields': [{'key': 'num_channels',
                            'title': 'Channels',
                            'kind': 'integer',
                            'defaultValue': 2,
                            'min': 1,
                            'max': 512},
                           {'key': 'num_groups',
                            'title': 'Channel groups',
                            'kind': 'integer',
                            'defaultValue': 1,
                            'min': 1,
                            'max': 512},
                           {'key': 'eps',
                            'title': 'Safety value',
                            'kind': 'number',
                            'defaultValue': 1e-05,
                            'min': 1.1754943508222875e-38,
                            'max': 3.4028234663852886e+38},
                           {'key': 'affine',
                            'title': 'Learn scale and shift',
                            'kind': 'boolean',
                            'defaultValue': True},
                           {'key': 'bias',
                            'title': 'Learn shift',
                            'kind': 'boolean',
                            'defaultValue': True}]}}

STATEFUL_NORMALIZATION_CONTRACTS = {'batch_norm1d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                  'adapterID': 'kwip.runner.graph-v2.batch_norm1d',
                  'graphSchemaVersions': [2],
                  'inputSampleRanks': [1, 2],
                  'channelAxis': 1,
                  'channelMinimum': 1,
                  'channelMaximum': 512,
                  'spatialMinimum': 1,
                  'spatialMaximum': 4096,
                  'preservesBatch': True,
                  'graphDTypes': ['float32'],
                  'epsilonMinimum': 1.1754943508222875e-38,
                  'epsilonMaximum': 3.4028234663852886e+38,
                  'affineModes': [False, True],
                  'biasModes': [False, True],
                  'forwardVarianceCorrection': 0,
                  'maximumTensorElements': 100000000,
                  'momentumMinimum': 0.0,
                  'momentumMaximum': 1.0,
                  'trackingModes': [False, True],
                  'runningVarianceCorrection': 1,
                  'runningMeanDType': 'float32',
                  'runningVarianceDType': 'float32',
                  'counterDType': 'int64',
                  'counterMaximum': 1000000000,
                  'statisticsScope': 'batchSpatial',
                  'evaluationUsesRunningStatisticsWhenTracked': True,
                  'averagingModes': ['moving', 'cumulative'],
                  'counterBehavior': 'trainingCalls',
                  'minimumCurrentStatisticsMembers': 2},
 'batch_norm2d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                  'adapterID': 'kwip.runner.graph-v2.batch_norm2d',
                  'graphSchemaVersions': [2],
                  'inputSampleRanks': [3],
                  'channelAxis': 1,
                  'channelMinimum': 1,
                  'channelMaximum': 512,
                  'spatialMinimum': 1,
                  'spatialMaximum': 4096,
                  'preservesBatch': True,
                  'graphDTypes': ['float32'],
                  'epsilonMinimum': 1.1754943508222875e-38,
                  'epsilonMaximum': 3.4028234663852886e+38,
                  'affineModes': [False, True],
                  'biasModes': [False, True],
                  'forwardVarianceCorrection': 0,
                  'maximumTensorElements': 100000000,
                  'momentumMinimum': 0.0,
                  'momentumMaximum': 1.0,
                  'trackingModes': [False, True],
                  'runningVarianceCorrection': 1,
                  'runningMeanDType': 'float32',
                  'runningVarianceDType': 'float32',
                  'counterDType': 'int64',
                  'counterMaximum': 1000000000,
                  'statisticsScope': 'batchSpatial',
                  'evaluationUsesRunningStatisticsWhenTracked': True,
                  'averagingModes': ['moving', 'cumulative'],
                  'counterBehavior': 'trainingCalls',
                  'minimumCurrentStatisticsMembers': 2},
 'instance_norm1d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                     'adapterID': 'kwip.runner.graph-v2.instance_norm1d',
                     'graphSchemaVersions': [2],
                     'inputSampleRanks': [2],
                     'channelAxis': 1,
                     'channelMinimum': 1,
                     'channelMaximum': 512,
                     'spatialMinimum': 1,
                     'spatialMaximum': 4096,
                     'preservesBatch': True,
                     'graphDTypes': ['float32'],
                     'epsilonMinimum': 1.1754943508222875e-38,
                     'epsilonMaximum': 3.4028234663852886e+38,
                     'affineModes': [False, True],
                     'biasModes': [False, True],
                     'forwardVarianceCorrection': 0,
                     'maximumTensorElements': 100000000,
                     'momentumMinimum': 0.0,
                     'momentumMaximum': 1.0,
                     'trackingModes': [False, True],
                     'runningVarianceCorrection': 1,
                     'runningMeanDType': 'float32',
                     'runningVarianceDType': 'float32',
                     'counterDType': 'int64',
                     'counterMaximum': 1000000000,
                     'statisticsScope': 'observationChannelSpatial',
                     'evaluationUsesRunningStatisticsWhenTracked': True,
                     'counterBehavior': 'constantZero',
                     'minimumCurrentStatisticsSpatialElements': 2},
 'instance_norm2d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                     'adapterID': 'kwip.runner.graph-v2.instance_norm2d',
                     'graphSchemaVersions': [2],
                     'inputSampleRanks': [3],
                     'channelAxis': 1,
                     'channelMinimum': 1,
                     'channelMaximum': 512,
                     'spatialMinimum': 1,
                     'spatialMaximum': 4096,
                     'preservesBatch': True,
                     'graphDTypes': ['float32'],
                     'epsilonMinimum': 1.1754943508222875e-38,
                     'epsilonMaximum': 3.4028234663852886e+38,
                     'affineModes': [False, True],
                     'biasModes': [False, True],
                     'forwardVarianceCorrection': 0,
                     'maximumTensorElements': 100000000,
                     'momentumMinimum': 0.0,
                     'momentumMaximum': 1.0,
                     'trackingModes': [False, True],
                     'runningVarianceCorrection': 1,
                     'runningMeanDType': 'float32',
                     'runningVarianceDType': 'float32',
                     'counterDType': 'int64',
                     'counterMaximum': 1000000000,
                     'statisticsScope': 'observationChannelSpatial',
                     'evaluationUsesRunningStatisticsWhenTracked': True,
                     'counterBehavior': 'constantZero',
                     'minimumCurrentStatisticsSpatialElements': 2},
 'group_norm': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                'adapterID': 'kwip.runner.graph-v2.group_norm',
                'graphSchemaVersions': [2],
                'inputSampleRanks': [1, 2, 3, 4, 5, 6, 7],
                'channelAxis': 1,
                'channelMinimum': 1,
                'channelMaximum': 512,
                'spatialMinimum': 1,
                'spatialMaximum': 4096,
                'preservesBatch': True,
                'graphDTypes': ['float32'],
                'epsilonMinimum': 1.1754943508222875e-38,
                'epsilonMaximum': 3.4028234663852886e+38,
                'affineModes': [False, True],
                'biasModes': [False, True],
                'forwardVarianceCorrection': 0,
                'maximumTensorElements': 100000000,
                'groupMinimum': 1,
                'groupMaximum': 512,
                'requiresChannelDivisibility': True,
                'minimumGroupMembers': 2,
                'statisticsScope': 'observationGroupSpatial',
                'usesCurrentStatistics': True}}

STATEFUL_NORMALIZATION_LESSONS = {'batch_norm1d': ['assessment.batch-norm1d-current-group', 'assessment.batch-norm1d-remembered-summary'], 'batch_norm2d': ['assessment.batch-norm2d-current-group', 'assessment.batch-norm2d-remembered-summary'], 'instance_norm1d': ['assessment.instance-norm1d-individual-patterns', 'assessment.instance-norm1d-remembered-summary'], 'instance_norm2d': ['assessment.instance-norm2d-individual-grids', 'assessment.instance-norm2d-remembered-summary'], 'group_norm': ['assessment.group-norm-channel-teams', 'assessment.group-norm-channel-adjustments']}

def validate_stateful_normalization_row(row, root):
    exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
    ident=row['definition'].get('id')
    if ident not in STATEFUL_NORMALIZATION_DEFINITIONS or json.dumps(row['definition'],sort_keys=True) != json.dumps(STATEFUL_NORMALIZATION_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed normalization controls')
    if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid normalization publication')
    if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unknown normalization adapter/ports')
    if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(STATEFUL_NORMALIZATION_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed normalization runtime/state policy')
    exact(row['presentation'],'theme icon shortTitle purpose')
    if row['presentation']['theme'] != 'mauve' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid normalization artwork')
    exact(row['sources'],'native worker tests evidence')
    if row['published'] and row['learning'] is None: raise ValueError('Published normalization requires lessons')
    refs=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png','Sources/KwipMLCore/StatefulNormalizationCapabilityBridge.swift']
    if row['learning'] is not None:
        l=row['learning'];exact(l,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
        if type(l['contentVersion']) is not int or l['contentVersion']<1 or l['lessonIDs'] != STATEFUL_NORMALIZATION_LESSONS[ident] or l['exampleID'] != 'prepared-'+ident.replace('_','-')+'-classifier': raise ValueError('Unknown normalization learning identity')
        if not isinstance(l['paragraphs'],list) or not l['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in l['paragraphs']): raise ValueError('Missing normalization learning')
        if any(not isinstance(l[k],str) or not l[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')): raise ValueError('Missing normalization content')
        refs += [l['exampleSource'],'Sources/KwipMLApp/Resources/'+l['sourceResource']+'.csv']+['Sources/KwipMLCore/StatefulNormalization'+n+'.swift'for n in ('LearningTask','LessonReference','LearningSubmission')]
    for ref in refs:
        if not isinstance(ref,str) or Path(ref).is_absolute(): raise ValueError('Unsafe normalization source')
        path=(root/ref).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe normalization source')

RMS_NORM_DEFINITION = {'id': 'rms_norm', 'title': 'RMS normalization', 'category': 'Normalization', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'normalized_shape', 'title': 'Trailing sample dimensions', 'kind': 'integerList', 'defaultValue': [2], 'min': 1, 'max': 100000000, 'minItems': 1, 'maxItems': 7}, {'key': 'eps', 'title': 'Epsilon', 'kind': 'number', 'defaultValue': 1e-05, 'min': 1.1754943508222875e-38, 'max': 3.4028234663852886e+38}, {'key': 'elementwise_affine', 'title': 'Learn elementwise scale', 'kind': 'boolean', 'defaultValue': True}]}
RMS_NORM_CONTRACT = {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.rms_norm', 'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True, 'epsilonMinimum': 1.1754943508222875e-38, 'epsilonMaximum': 3.4028234663852886e+38, 'elementwiseAffineModes': [False, True], 'usesCurrentInputStatistics': True, 'subtractsMean': False, 'hasBias': False}

def validate_rms_normalization(row, root):
    if json.dumps(row['definition'],sort_keys=True) != json.dumps(RMS_NORM_DEFINITION,sort_keys=True) or json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(RMS_NORM_CONTRACT,sort_keys=True):
        raise ValueError('RMS normalization differs from its reviewed shape, epsilon or scale-only contract')
    if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2 or row['adapterID'] != 'kwip.runner.graph-v2.rms_norm' or row['shapeRuleID'] != 'kwip.shape.rms_norm.v1' or row['outputConditions'] != {}:
        raise ValueError('Invalid RMS normalization identity/ports/version')
    exact(row['presentation'], 'theme icon shortTitle purpose')
    if row['presentation']['theme'] != 'mauve' or row['presentation']['icon'] != 'normalization-v3' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()):
        raise ValueError('Invalid RMS normalization family artwork')
    exact(row['sources'], 'native worker tests evidence')
    learning=row['learning']
    exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
    if type(learning['contentVersion']) is not int or learning['contentVersion']<1 or learning['exampleID'] != 'prepared-rms-norm-classifier' or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']):
        raise ValueError('Invalid RMS normalization learning')
    if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')):
        raise ValueError('Missing RMS learning metadata')
    for reference in list(row['sources'].values())+[learning['exampleSource'],'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv','Sources/KwipMLApp/Resources/ComponentIcons/normalization-v3.png']:
        if not isinstance(reference,str) or Path(reference).is_absolute():raise ValueError('Unsafe RMS source')
        path=(root/reference).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing RMS source')

def validate_normalization(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.normalization' or type(value['version']) is not int or value['version'] < 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported normalization bundle identity/version/backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids=[]
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ids.append(row['definition'].get('id'))
        if row['definition'].get('id') == 'rms_norm':
            validate_rms_normalization(row,root)
            continue
        if row['definition'].get('id') in STATEFUL_NORMALIZATION_DEFINITIONS:
            validate_stateful_normalization_row(row,root)
            continue
        if json.dumps(row['definition'],sort_keys=True) != json.dumps({'id': 'layer_norm', 'title': 'Layer normalization', 'category': 'Normalization', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'normalized_shape', 'title': 'Trailing sample dimensions', 'kind': 'integerList', 'defaultValue': [2], 'min': 1, 'max': 100000000, 'minItems': 1, 'maxItems': 7}, {'key': 'eps', 'title': 'Epsilon', 'kind': 'number', 'defaultValue': 1e-05, 'min': 1.1754943508222875e-38, 'max': 3.4028234663852886e+38}, {'key': 'elementwise_affine', 'title': 'Learn elementwise scale', 'kind': 'boolean', 'defaultValue': True}, {'key': 'bias', 'title': 'Learn elementwise bias', 'kind': 'boolean', 'defaultValue': True}]},sort_keys=True): raise ValueError('Unsupported normalization definition/settings')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid publication/configuration version')
        if row['adapterID'] != 'kwip.runner.graph-v2.layer_norm' or row['shapeRuleID'] != 'kwip.shape.layer_norm.v1' or row['outputConditions'] != {}: raise ValueError('Unknown normalization adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps({'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.layer_norm', 'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True, 'epsilonMinimum': 1.1754943508222875e-38, 'epsilonMaximum': 3.4028234663852886e+38, 'elementwiseAffineModes': [False, True], 'biasModes': [False, True], 'varianceCorrection': 0, 'usesCurrentInputStatistics': True},sort_keys=True): raise ValueError('Invalid normalization runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'mauve' or row['presentation']['icon'] != 'layer_norm-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid normalization presentation')
        exact(row['sources'], 'native worker tests evidence')
        references=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/layer_norm-v2.png']
        learning=row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-layer-norm-classifier': raise ValueError('Invalid normalization learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')): raise ValueError('Missing learning metadata')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path=(root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
    unique(ids, 'component ID')


def validate_layer(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.layers' or type(value['version']) is not int or value['version'] < 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported layer bundle identity/version/backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids=[]
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ids.append(row['definition'].get('id'))
        ident = row['definition'].get('id')
        definitions = {'bilinear': {'id': 'bilinear', 'title': 'Bilinear', 'category': 'Layers', 'inputs': ['a', 'b'], 'outputs': ['out'], 'fields': [{'key': 'in1_features', 'title': 'First input features', 'defaultValue': 2, 'min': 1, 'max': 512, 'kind': 'integer'}, {'key': 'in2_features', 'title': 'Second input features', 'defaultValue': 2, 'min': 1, 'max': 512, 'kind': 'integer'}, {'key': 'out_features', 'title': 'Output features', 'defaultValue': 2, 'min': 1, 'max': 512, 'kind': 'integer'}, {'key': 'bias', 'title': 'Learn bias', 'defaultValue': True, 'kind': 'boolean'}]}, 'linear': {'id': 'linear', 'title': 'Linear', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'in_features', 'title': 'Input features', 'defaultValue': 2, 'min': 1, 'max': 512, 'kind': 'integer'}, {'key': 'out_features', 'title': 'Output features', 'defaultValue': 8, 'min': 1, 'max': 512, 'kind': 'integer'}, {'key': 'bias', 'title': 'Bias', 'kind': 'boolean', 'defaultValue': True}]}, 'lazy_linear': {'id': 'lazy_linear', 'title': 'Lazy Linear', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'out_features', 'title': 'Output features', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'bias', 'title': 'Learn bias', 'kind': 'boolean', 'defaultValue': True}]}}
        runtime_contracts = {'linear': {'version': {'major': 1, 'minor': 2, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.linear', 'graphSchemaVersions': [1, 2], 'maximumRank': 8, 'leadingAxesMinimum': 1, 'leadingAxesMaximum': 7, 'biasModes': [False, True], 'legacySchema1Bias': True, 'widthMinimum': 1, 'widthMaximum': 512, 'preservesBatch': True}, 'bilinear': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.bilinear', 'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True, 'minimumSampleRank': 1, 'sampleBroadcasting': False, 'maximumSampleElements': 100000000, 'maximumParameters': 2000000, 'featureMinimum': 1, 'featureMaximum': 512, 'biasModes': [False, True], 'graphDTypes': ['float32']}, 'lazy_linear': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.lazy-linear', 'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True, 'minimumSampleRank': 1, 'maximumSampleElements': 100000000, 'maximumParameters': 2000000, 'inputFeatureMinimum': 1, 'inputFeatureMaximum': 4096, 'outputFeatureMinimum': 1, 'outputFeatureMaximum': 512, 'biasModes': [False, True], 'graphDTypes': ['float32'], 'inputWidthInferred': True, 'materializationPolicy': 'validatedShapeBeforeOptimizer'}}
        if ident not in definitions: raise ValueError('Unknown layer adapter')
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(definitions[ident],sort_keys=True): raise ValueError('Unsupported layer definition/settings')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid publication/configuration version')
        if row['adapterID'] != runtime_contracts[ident]['adapterID'] or row['shapeRuleID'] != 'kwip.shape.'+ident.replace('_','-')+'.v1' or row['outputConditions'] != {}: raise ValueError('Unknown layer adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(runtime_contracts[ident],sort_keys=True): raise ValueError('Invalid layer runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'rose' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid layer presentation')
        exact(row['sources'], 'native worker tests evidence')
        references=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning=row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != {'linear':'prepared-parallel-classifier','bilinear':'prepared-bilinear-classifier','lazy_linear':'prepared-lazy-linear-classifier'}[ident]: raise ValueError('Invalid layer learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')): raise ValueError('Missing learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid layer content date')
            if learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid layer example resource')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path=(root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
    unique(ids, 'component ID')


def validate_tensor_layout(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.tensor-layout' or type(value['version']) is not int or value['version'] < 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported layer bundle identity/version/backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids=[]
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ids.append(row['definition'].get('id'))
        ident = row['definition'].get('id')
        definitions = {'reshape': {'id': 'reshape', 'title': 'Reshape', 'category': 'Tensor layout', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'shape', 'title': 'Sample dimensions', 'kind': 'integerList', 'defaultValue': [-1], 'min': -1, 'max': 100000000, 'minItems': 1, 'maxItems': 7}]}, 'transpose': {'id': 'transpose', 'title': 'Transpose', 'category': 'Tensor layout', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'dim0', 'title': 'First axis', 'kind': 'integer', 'defaultValue': 1, 'min': -8, 'max': 7}, {'key': 'dim1', 'title': 'Second axis', 'kind': 'integer', 'defaultValue': 2, 'min': -8, 'max': 7}]}, 'flatten': {'id': 'flatten', 'title': 'Flatten', 'category': 'Tensor layout', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'start_dim', 'title': 'First axis', 'kind': 'integer', 'defaultValue': 1, 'min': -8, 'max': 7}, {'key': 'end_dim', 'title': 'Last axis', 'kind': 'integer', 'defaultValue': -1, 'min': -8, 'max': 7}]}}
        if ident not in definitions: raise ValueError('Unknown tensor-layout operation')
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(definitions[ident],sort_keys=True): raise ValueError('Unsupported tensor-layout definition/settings')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid publication/configuration version')
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unknown tensor-layout adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps({'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.tensor-layout', 'graphSchemaVersions': [2], 'operations': ['reshape', 'transpose', 'flatten'], 'maximumRank': 8, 'preservesBatch': True},sort_keys=True): raise ValueError('Invalid tensor-layout runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'violet' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid layer presentation')
        exact(row['sources'], 'native worker tests evidence')
        references=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning=row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-tensor-layout-classifier': raise ValueError('Invalid layer learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')): raise ValueError('Missing learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid layer content date')
            if learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid layer example resource')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path=(root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
    unique(ids, 'component ID')
    if set(ids) != {'reshape','transpose','flatten'}: raise ValueError('Tensor-layout bundle requires exactly three reviewed operations')


def validate_tensor_math(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.tensor-math' or type(value['version']) is not int or value['version'] < 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported tensor-math bundle identity/version/backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids=[]
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ids.append(row['definition'].get('id'))
        ident = row['definition'].get('id')
        definitions = {'matmul': {'id':'matmul','title':'Matrix multiply','category':'Tensor math','inputs':['a','b'],'outputs':['out'],'fields':[]}}
        if ident not in definitions: raise ValueError('Unknown tensor-math operation')
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(definitions[ident],sort_keys=True): raise ValueError('Unsupported tensor-math definition/settings')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid publication/configuration version')
        if row['adapterID'] != 'kwip.runner.graph-v2.matrixMultiply' or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unknown tensor-math adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps({'version': {'major':1,'minor':0,'patch':0}, 'adapterID':'kwip.runner.graph-v2.matrixMultiply','graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True,'minimumSampleRank':2,'sampleBroadcasting':True,'maximumOutputElements':100000000,'graphDTypes':['float32']},sort_keys=True): raise ValueError('Invalid tensor-math runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'blue' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid tensor-math presentation')
        exact(row['sources'], 'native worker tests evidence')
        references=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning=row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-matrix-classifier': raise ValueError('Invalid tensor-math learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')): raise ValueError('Missing learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid tensor-math content date')
            if learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid tensor-math example resource')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path=(root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
    unique(ids, 'component ID')
    if set(ids) != {'matmul'}: raise ValueError('Tensor-math bundle requires exactly the reviewed Matrix multiply operation')


def validate_convolution(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.convolution' or type(value['version']) is not int or value['version'] < 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported convolution bundle identity/version/backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids=[]
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ids.append(row['definition'].get('id'))
        ident = row['definition'].get('id')
        definitions = {'conv1d': {'id': 'conv1d', 'title': 'Conv1d', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'in_channels', 'title': 'Input channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'out_channels', 'title': 'Output channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'kernel_size', 'title': 'Kernel size', 'kind': 'integer', 'defaultValue': 3, 'min': 1, 'max': 4096}, {'key': 'stride', 'title': 'Stride', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 4096}, {'key': 'dilation', 'title': 'Dilation', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 4096}, {'key': 'groups', 'title': 'Groups', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 512}, {'key': 'bias', 'title': 'Learn bias', 'kind': 'boolean', 'defaultValue': True}, {'key': 'padding_policy', 'title': 'Padding policy', 'kind': 'choice', 'defaultValue': 'explicit', 'choices': [{'value': 'explicit', 'title': 'Explicit amount'}, {'value': 'valid', 'title': 'Valid · no padding'}, {'value': 'same', 'title': 'Same length'}]}, {'key': 'padding', 'title': 'Padding per side', 'kind': 'integer', 'defaultValue': 0, 'min': 0, 'max': 4096}, {'key': 'padding_mode', 'title': 'Padding mode', 'kind': 'choice', 'defaultValue': 'zeros', 'choices': [{'value': 'zeros', 'title': 'Zeros'}, {'value': 'reflect', 'title': 'Reflect'}, {'value': 'replicate', 'title': 'Replicate'}, {'value': 'circular', 'title': 'Circular'}]}]}, 'conv2d': {'id': 'conv2d', 'title': 'Conv2d', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'in_channels', 'title': 'Input channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'out_channels', 'title': 'Output channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'kernel_size', 'title': 'Kernel [height, width]', 'kind': 'integerList', 'defaultValue': [3, 3], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'stride', 'title': 'Stride [height, width]', 'kind': 'integerList', 'defaultValue': [1, 1], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'dilation', 'title': 'Dilation [height, width]', 'kind': 'integerList', 'defaultValue': [1, 1], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'groups', 'title': 'Groups', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 512}, {'key': 'bias', 'title': 'Learn bias', 'kind': 'boolean', 'defaultValue': True}, {'key': 'padding_policy', 'title': 'Padding policy', 'kind': 'choice', 'defaultValue': 'explicit', 'choices': [{'value': 'explicit', 'title': 'Explicit amount'}, {'value': 'valid', 'title': 'Valid · no padding'}, {'value': 'same', 'title': 'Same spatial size'}]}, {'key': 'padding', 'title': 'Padding [height, width]', 'kind': 'integerList', 'defaultValue': [0, 0], 'min': 0, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'padding_mode', 'title': 'Padding mode', 'kind': 'choice', 'defaultValue': 'zeros', 'choices': [{'value': 'zeros', 'title': 'Zeros'}, {'value': 'reflect', 'title': 'Reflect'}, {'value': 'replicate', 'title': 'Replicate'}, {'value': 'circular', 'title': 'Circular'}]}]}, 'conv3d': {'id': 'conv3d', 'title': 'Conv3d', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'in_channels', 'title': 'Input channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'out_channels', 'title': 'Output channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'kernel_size', 'title': 'Kernel [depth, height, width]', 'kind': 'integerList', 'defaultValue': [3, 3, 3], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'stride', 'title': 'Stride [depth, height, width]', 'kind': 'integerList', 'defaultValue': [1, 1, 1], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'dilation', 'title': 'Dilation [depth, height, width]', 'kind': 'integerList', 'defaultValue': [1, 1, 1], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'groups', 'title': 'Groups', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 512}, {'key': 'bias', 'title': 'Learn bias', 'kind': 'boolean', 'defaultValue': True}, {'key': 'padding_policy', 'title': 'Padding policy', 'kind': 'choice', 'defaultValue': 'explicit', 'choices': [{'value': 'explicit', 'title': 'Explicit amount'}, {'value': 'valid', 'title': 'Valid · no padding'}, {'value': 'same', 'title': 'Same spatial size'}]}, {'key': 'padding', 'title': 'Padding [depth, height, width]', 'kind': 'integerList', 'defaultValue': [0, 0, 0], 'min': 0, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'padding_mode', 'title': 'Padding mode', 'kind': 'choice', 'defaultValue': 'zeros', 'choices': [{'value': 'zeros', 'title': 'Zeros'}, {'value': 'reflect', 'title': 'Reflect'}, {'value': 'replicate', 'title': 'Replicate'}, {'value': 'circular', 'title': 'Circular'}]}]}, 'conv_transpose1d': {'id': 'conv_transpose1d', 'title': 'ConvTranspose1d', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'in_channels', 'title': 'Input channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'out_channels', 'title': 'Output channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'kernel_size', 'title': 'Kernel size', 'kind': 'integer', 'defaultValue': 3, 'min': 1, 'max': 4096}, {'key': 'stride', 'title': 'Stride', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 4096}, {'key': 'padding', 'title': 'Padding · reduce output extent', 'kind': 'integer', 'defaultValue': 0, 'min': 0, 'max': 4096}, {'key': 'output_padding', 'title': 'Output padding · extent adjustment', 'kind': 'integer', 'defaultValue': 0, 'min': 0, 'max': 4096}, {'key': 'dilation', 'title': 'Dilation', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 4096}, {'key': 'groups', 'title': 'Groups', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 512}, {'key': 'bias', 'title': 'Learn bias', 'kind': 'boolean', 'defaultValue': True}]}, 'conv_transpose2d': {'id': 'conv_transpose2d', 'title': 'ConvTranspose2d', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'in_channels', 'title': 'Input channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'out_channels', 'title': 'Output channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'kernel_size', 'title': 'Kernel [height, width]', 'kind': 'integerList', 'defaultValue': [3, 3], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'stride', 'title': 'Stride [height, width]', 'kind': 'integerList', 'defaultValue': [1, 1], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'padding', 'title': 'Padding [height, width] · reduce extent', 'kind': 'integerList', 'defaultValue': [0, 0], 'min': 0, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'output_padding', 'title': 'Output padding [height, width] · extent adjustment', 'kind': 'integerList', 'defaultValue': [0, 0], 'min': 0, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'dilation', 'title': 'Dilation [height, width]', 'kind': 'integerList', 'defaultValue': [1, 1], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'groups', 'title': 'Groups', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 512}, {'key': 'bias', 'title': 'Learn bias', 'kind': 'boolean', 'defaultValue': True}]}, 'conv_transpose3d': {'id': 'conv_transpose3d', 'title': 'ConvTranspose3d', 'category': 'Layers', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'in_channels', 'title': 'Input channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'out_channels', 'title': 'Output channels', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 512}, {'key': 'kernel_size', 'title': 'Kernel [depth, height, width]', 'kind': 'integerList', 'defaultValue': [3, 3, 3], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'stride', 'title': 'Stride [depth, height, width]', 'kind': 'integerList', 'defaultValue': [1, 1, 1], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'padding', 'title': 'Padding [depth, height, width] · reduce extent', 'kind': 'integerList', 'defaultValue': [0, 0, 0], 'min': 0, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'output_padding', 'title': 'Output padding [depth, height, width] · extent adjustment', 'kind': 'integerList', 'defaultValue': [0, 0, 0], 'min': 0, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'dilation', 'title': 'Dilation [depth, height, width]', 'kind': 'integerList', 'defaultValue': [1, 1, 1], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'groups', 'title': 'Groups', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 512}, {'key': 'bias', 'title': 'Learn bias', 'kind': 'boolean', 'defaultValue': True}]}}
        if ident not in definitions: raise ValueError('Unknown convolution operation')
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(definitions[ident],sort_keys=True): raise ValueError('Unsupported convolution definition/settings')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid publication/configuration version')
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unknown convolution adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps({'conv1d': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.conv1d', 'graphSchemaVersions': [2], 'sampleRank': 2, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'groupMinimum': 1, 'groupMaximum': 512, 'paddingMinimum': 0, 'paddingMaximum': 4096, 'maximumParameters': 2000000, 'maximumMultiplyAdds': 128000000, 'paddingPolicies': ['explicit', 'valid', 'same'], 'paddingModes': ['zeros', 'reflect', 'replicate', 'circular'], 'biasModes': [False, True]}, 'conv2d': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.conv2d', 'graphSchemaVersions': [2], 'sampleRank': 3, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'groupMinimum': 1, 'groupMaximum': 512, 'paddingMinimum': 0, 'paddingMaximum': 4096, 'maximumParameters': 2000000, 'maximumMultiplyAdds': 128000000, 'paddingPolicies': ['explicit', 'valid', 'same'], 'paddingModes': ['zeros', 'reflect', 'replicate', 'circular'], 'biasModes': [False, True], 'spatialParameterLength': 2, 'maximumSampleElements': 100000000, 'maximumPaddedSampleElements': 100000000}, 'conv3d': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.conv3d', 'graphSchemaVersions': [2], 'sampleRank': 4, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'groupMinimum': 1, 'groupMaximum': 512, 'paddingMinimum': 0, 'paddingMaximum': 4096, 'maximumParameters': 2000000, 'maximumMultiplyAdds': 128000000, 'paddingPolicies': ['explicit', 'valid', 'same'], 'paddingModes': ['zeros', 'reflect', 'replicate', 'circular'], 'biasModes': [False, True], 'spatialParameterLength': 3, 'maximumSampleElements': 100000000, 'maximumPaddedSampleElements': 100000000}, 'conv_transpose1d': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.conv_transpose1d', 'graphSchemaVersions': [2], 'sampleRank': 2, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'groupMinimum': 1, 'groupMaximum': 512, 'paddingMinimum': 0, 'paddingMaximum': 4096, 'maximumParameters': 2000000, 'maximumMultiplyAdds': 128000000, 'biasModes': [False, True], 'spatialParameterKind': 'scalar', 'maximumSampleElements': 100000000, 'maximumUncroppedSampleElements': 100000000, 'outputPaddingMinimum': 0, 'outputPaddingMaximum': 4096, 'configurationKeys': ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'dilation', 'groups', 'bias'], 'paddingMode': 'zeros', 'outputPaddingRule': 'lessThanStrideOrDilation', 'supportsDynamicOutputSize': False}, 'conv_transpose2d': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.conv_transpose2d', 'graphSchemaVersions': [2], 'sampleRank': 3, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'groupMinimum': 1, 'groupMaximum': 512, 'paddingMinimum': 0, 'paddingMaximum': 4096, 'maximumParameters': 2000000, 'maximumMultiplyAdds': 128000000, 'biasModes': [False, True], 'spatialParameterLength': 2, 'maximumSampleElements': 100000000, 'spatialParameterKind': 'pair', 'maximumUncroppedSampleElements': 100000000, 'outputPaddingMinimum': 0, 'outputPaddingMaximum': 4096, 'configurationKeys': ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'dilation', 'groups', 'bias'], 'paddingMode': 'zeros', 'outputPaddingRule': 'lessThanStrideOrDilation', 'supportsDynamicOutputSize': False}, 'conv_transpose3d': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.conv_transpose3d', 'graphSchemaVersions': [2], 'sampleRank': 4, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'groupMinimum': 1, 'groupMaximum': 512, 'paddingMinimum': 0, 'paddingMaximum': 4096, 'maximumParameters': 2000000, 'maximumMultiplyAdds': 128000000, 'biasModes': [False, True], 'spatialParameterLength': 3, 'maximumSampleElements': 100000000, 'spatialParameterKind': 'triple', 'maximumUncroppedSampleElements': 100000000, 'outputPaddingMinimum': 0, 'outputPaddingMaximum': 4096, 'configurationKeys': ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'dilation', 'groups', 'bias'], 'paddingMode': 'zeros', 'outputPaddingRule': 'lessThanStrideOrDilation', 'supportsDynamicOutputSize': False}}[ident],sort_keys=True): raise ValueError('Invalid convolution runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'rose' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid convolution presentation')
        exact(row['sources'], 'native worker tests evidence')
        references=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning=row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident.replace('conv_transpose','conv-transpose')+'-classifier': raise ValueError('Invalid convolution learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')): raise ValueError('Missing learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid convolution content date')
            if learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid convolution example resource')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path=(root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
    unique(ids, 'component ID')
    if set(ids) not in ({'conv1d','conv2d','conv3d'}, {'conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d'}): raise ValueError('Convolution bundle requires the three reviewed dimensions and either all or no transposed counterparts')


# Reviewed bag pair uses configuration1 and its own coordinate contract.
# These authored expectations are independent of runtime capability producers.
BAG_EMBEDDING_DEFINITIONS = {'bag_input': {'id': 'bag_input',
               'title': 'Bag input',
               'category': 'Boundaries',
               'inputs': [],
               'outputs': ['indices', 'offsets', 'per_sample_weights'],
               'fields': [{'key': 'layout',
                           'title': 'Group layout',
                           'kind': 'choice',
                           'defaultValue': 'fixed',
                           'choices': [{'value': 'fixed', 'title': 'Fixed rows'},
                                       {'value': 'packed', 'title': 'Packed groups'}]},
                          {'key': 'maximum_items_per_bag',
                           'title': 'Maximum items per group',
                           'kind': 'integer',
                           'defaultValue': 4,
                           'min': 1,
                           'max': 128},
                          {'key': 'per_sample_weights',
                           'title': 'Use item weights',
                           'kind': 'boolean',
                           'defaultValue': False},
                          {'key': 'include_last_offset',
                           'title': 'Include final boundary',
                           'kind': 'boolean',
                           'defaultValue': False}]},
 'embedding_bag': {'id': 'embedding_bag',
                   'title': 'Embedding bag',
                   'category': 'Embeddings',
                   'inputs': ['indices', 'offsets', 'per_sample_weights'],
                   'outputs': ['out'],
                   'fields': [{'key': 'num_embeddings',
                               'title': 'Table rows',
                               'kind': 'integer',
                               'defaultValue': 16,
                               'min': 1,
                               'max': 100000000},
                              {'key': 'embedding_dim',
                               'title': 'Vector dimensions',
                               'kind': 'integer',
                               'defaultValue': 4,
                               'min': 1,
                               'max': 4096},
                              {'key': 'mode',
                               'title': 'Combine selected vectors',
                               'kind': 'choice',
                               'defaultValue': 'mean',
                               'choices': [{'value': 'mean', 'title': 'Mean'},
                                           {'value': 'sum', 'title': 'Sum'},
                                           {'value': 'max', 'title': 'Maximum'}]},
                              {'key': 'padding_idx',
                               'title': 'Padding row',
                               'kind': 'integer',
                               'defaultValue': None,
                               'min': -100000000,
                               'max': 99999999,
                               'allowsNull': True},
                              {'key': 'max_norm',
                               'title': 'Maximum norm',
                               'kind': 'number',
                               'defaultValue': None,
                               'min': 1.1754943508222875e-38,
                               'max': 3.4028234663852886e+38,
                               'allowsNull': True},
                              {'key': 'norm_type',
                               'title': 'Norm order',
                               'kind': 'number',
                               'defaultValue': 2,
                               'min': 1.1754943508222875e-38,
                               'max': 3.4028234663852886e+38},
                              {'key': 'scale_grad_by_freq',
                               'title': 'Scale gradients by frequency',
                               'kind': 'boolean',
                               'defaultValue': False},
                              {'key': 'sparse',
                               'title': 'Sparse gradients',
                               'kind': 'boolean',
                               'defaultValue': False},
                              {'key': 'include_last_offset',
                               'title': 'Include final boundary',
                               'kind': 'boolean',
                               'defaultValue': False},
                              {'key': 'freeze',
                               'title': 'Freeze learned weights',
                               'kind': 'boolean',
                               'defaultValue': False},
                              {'key': 'layout',
                               'title': 'Group layout',
                               'kind': 'choice',
                               'defaultValue': 'fixed',
                               'choices': [{'value': 'fixed', 'title': 'Fixed rows'},
                                           {'value': 'packed', 'title': 'Packed groups'}]},
                              {'key': 'per_sample_weights',
                               'title': 'Use item weights',
                               'kind': 'boolean',
                               'defaultValue': False}]}}
BAG_EMBEDDING_RUNTIME_CONTRACTS = {'bag_input': {'version': {'major': 1, 'minor': 0, 'patch': 0},
               'adapterID': 'kwip.runner.graph-v2.bag_input',
               'graphSchemaVersions': [2],
               'packetSchemaVersion': 1,
               'layouts': ['fixed', 'packed'],
               'coordinateDTypes': {'indices': 'int64', 'offsets': 'int64', 'per_sample_weights': 'float32'},
               'maximumItemsPerBag': 128,
               'maximumBags': 4096,
               'maximumItems': 1000000,
               'maximumOutputElements': 2000000,
               'integerDerivatives': 'unavailable',
               'sourceWeights': 'realPairedInput',
               'batching': 'wholeBags',
               'emptyPackedBags': 'preserve'},
 'embedding_bag': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                   'adapterID': 'kwip.runner.graph-v2.embedding_bag',
                   'graphSchemaVersions': [2],
                   'packetSchemaVersion': 1,
                   'layouts': ['fixed', 'packed'],
                   'coordinateDTypes': {'indices': 'int64',
                                        'offsets': 'int64',
                                        'per_sample_weights': 'float32'},
                   'maximumItemsPerBag': 128,
                   'maximumBags': 4096,
                   'maximumItems': 1000000,
                   'maximumOutputElements': 2000000,
                   'integerDerivatives': 'unavailable',
                   'sourceWeights': 'realPairedInput',
                   'batching': 'wholeBags',
                   'emptyPackedBags': 'preserve',
                   'modes': ['mean', 'sum', 'max'],
                   'weightedModes': ['sum'],
                   'outputDType': 'float32',
                   'maximumParameters': 2000000,
                   'maximumStaticTableElements': 100000000,
                   'paddingGradient': 'suppressed',
                   'frozenWeightsMayRenormalize': True,
                   'trainableSparseAdam': False,
                   'mpsMaxNormExecution': 'reject',
                   'mpsTrainableFrequencyScaling': 'reject',
                   'sourceWeightsDerivatives': 'sumOnly'}}
BAG_EMBEDDING_SOURCES = {'bag_input': {'native': 'Sources/KwipMLCore/BagSourceConfiguration.swift',
               'worker': 'runner/bag_source.py',
               'tests': 'runner/tests/test_bag_source_semantics.py',
               'evidence': 'docs/production/TORCH_EMBEDDING_BAG_LEARNING_CANVAS_20261009.md'},
 'embedding_bag': {'native': 'Sources/KwipMLCore/EmbeddingBagConfiguration.swift',
                   'worker': 'runner/embedding_bag.py',
                   'tests': 'runner/tests/test_embedding_bag.py',
                   'evidence': 'docs/production/TORCH_EMBEDDING_BAG_LEARNING_CANVAS_20261009.md'}}
BAG_EMBEDDING_LESSONS = ['assessment.bag-fixed-sum','assessment.bag-packed-mean',
 'assessment.bag-packed-max','assessment.bag-fixed-weighted-sum',
 'assessment.bag-packed-weighted-sum','assessment.bag-padding-norm']

def validate_bag_embedding(row,root):
    ident=row['definition'].get('id')
    if json.dumps(row['definition'],sort_keys=True)!=json.dumps(BAG_EMBEDDING_DEFINITIONS[ident],sort_keys=True):raise ValueError('Unsupported bag definition/settings')
    if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion']!=1:raise ValueError('Invalid bag publication/configuration version')
    if row['adapterID']!='kwip.runner.graph-v2.'+ident or row['shapeRuleID']!='kwip.shape.'+ident+'.v1' or row['outputConditions']!={}:raise ValueError('Unknown bag adapter/ports; typed configured roles are authoritative')
    if json.dumps(row['runtimeContract'],sort_keys=True)!=json.dumps(BAG_EMBEDDING_RUNTIME_CONTRACTS[ident],sort_keys=True):raise ValueError('Invalid bag runtime contract')
    exact(row['presentation'],'theme icon shortTitle purpose')
    icon='bag-input-v1' if ident=='bag_input' else 'embedding-bag-v1'
    if row['presentation']['theme']!=('taupe' if ident=='bag_input' else 'purple') or row['presentation']['icon']!=icon or any(type(v) is not str or not v.strip() for v in row['presentation'].values()):raise ValueError('Invalid bag presentation')
    exact(row['sources'],'native worker tests evidence')
    if row['sources']!=BAG_EMBEDDING_SOURCES[ident]:raise ValueError('Bag source authorities differ')
    references=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+icon+'.png']
    learning=row['learning']
    if row['published'] and learning is None:raise ValueError('Published bag component requires learning/example')
    if learning is not None:
        exact(learning,'contentVersion updatedAt lessonIDs exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
        if type(learning['contentVersion']) is not int or learning['contentVersion']<1 or learning['lessonIDs']!=BAG_EMBEDDING_LESSONS or learning['exampleID']!='prepared-embedding-bag-classifier':raise ValueError('Invalid bag learning identity')
        if type(learning['paragraphs']) is not list or not learning['paragraphs'] or any(type(v) is not str or not v.strip() for v in learning['paragraphs']):raise ValueError('Missing bag learning paragraphs')
        if any(type(learning[k]) is not str or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')):raise ValueError('Missing bag learning metadata')
        from datetime import date
        try:date.fromisoformat(learning['updatedAt'])
        except ValueError:raise ValueError('Invalid bag content date')
        if learning['sourceResource']!='bag-observations' or learning['exampleSource']!='Sources/KwipMLCore/EmbeddingBagWorkflowExample.swift':raise ValueError('Invalid bag example source/resource')
        references += [learning['exampleSource'],'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv']
    for reference in references:
        if type(reference) is not str or Path(reference).is_absolute():raise ValueError('Unsafe bag source reference')
        path=(root/reference).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing/unsafe bag source reference')

def validate_embedding(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.embeddings' or type(value['version']) is not int or value['version'] < 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported embedding bundle identity/version/backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids=[]
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ids.append(row['definition'].get('id'))
        ident = row['definition'].get('id')
        if ident in BAG_EMBEDDING_DEFINITIONS:
            validate_bag_embedding(row,root)
            continue
        definitions = {'embedding': {'id': 'embedding', 'title': 'Embedding', 'category': 'Embeddings', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'num_embeddings', 'title': 'Table rows', 'kind': 'integer', 'defaultValue': 16, 'min': 1, 'max': 100000000}, {'key': 'embedding_dim', 'title': 'Vector dimensions', 'kind': 'integer', 'defaultValue': 8, 'min': 1, 'max': 4096}, {'key': 'padding_idx', 'title': 'Padding row', 'kind': 'integer', 'defaultValue': None, 'min': -100000000, 'max': 99999999, 'allowsNull': True}, {'key': 'max_norm', 'title': 'Maximum norm', 'kind': 'number', 'defaultValue': None, 'min': 1.1754943508222875e-38, 'max': 3.4028234663852886e+38, 'allowsNull': True}, {'key': 'norm_type', 'title': 'Norm order', 'kind': 'number', 'defaultValue': 2, 'min': 1.1754943508222875e-38, 'max': 3.4028234663852886e+38}, {'key': 'scale_grad_by_freq', 'title': 'Scale gradients by frequency', 'kind': 'boolean', 'defaultValue': False}, {'key': 'sparse', 'title': 'Sparse gradients', 'kind': 'boolean', 'defaultValue': False}, {'key': 'freeze', 'title': 'Freeze learned weights', 'kind': 'boolean', 'defaultValue': False}]}, 'index_input': {'id': 'index_input', 'title': 'Integer input', 'category': 'Boundaries', 'inputs': [], 'outputs': ['out'], 'fields': [{'key': 'shape', 'title': 'Sample dimensions', 'kind': 'integerList', 'defaultValue': [2], 'min': 1, 'max': 100000000, 'minItems': 1, 'maxItems': 7}]}}
        if ident not in definitions: raise ValueError('Unknown embedding operation')
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(definitions[ident],sort_keys=True): raise ValueError('Unsupported embedding definition/settings')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid publication/configuration version')
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unknown embedding adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps({'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.embedding', 'graphSchemaVersions': [2], 'inputDType': 'int64', 'weightDType': 'float32', 'maximumInputSampleRank': 6, 'maximumOutputRank': 8, 'maximumPreparedColumns': 128, 'maximumParameters': 2000000, 'preservesBatch': True, 'outOfRangePolicy': 'reject', 'paddingGradient': 'suppressed', 'frequencyScaling': True, 'frozenWeightsMayRenormalize': True, 'trainingOptimizer': 'Adam', 'trainableSparseGradients': False, 'previewCacheSchemaForMaxNorm': 2, 'mpsMaxNormExecution': 'cpuFallback'},sort_keys=True): raise ValueError('Invalid embedding runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != ('purple' if ident == 'embedding' else 'taupe') or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid embedding presentation')
        exact(row['sources'], 'native worker tests evidence')
        references=list(row['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning=row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-embedding-classifier': raise ValueError('Invalid embedding learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle')): raise ValueError('Missing learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid embedding content date')
            if learning['sourceResource'] != 'embedding-observations': raise ValueError('Invalid embedding example resource')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path=(root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
    unique(ids, 'component ID')
    if set(ids) not in ({'embedding','index_input'},{'embedding','index_input','bag_input','embedding_bag'}):raise ValueError('Embedding bundle requires its legacy pair and a complete bag source/operator pair')
    if 'bag_input' in ids:
        bag_rows=[row for row in value['components'] if row['definition']['id'] in BAG_EMBEDDING_DEFINITIONS]
        if value['version']<2 or bag_rows[0]['published']!=bag_rows[1]['published']:raise ValueError('Bag pair requires bundle2 and matching publication flags')



NEW_POOLING_DEFINITIONS = {'avg_pool2d': {'id': 'avg_pool2d',
                'title': 'AvgPool2d',
                'category': 'Pooling',
                'inputs': ['in'],
                'outputs': ['out'],
                'fields': [{'key': 'kernel_size',
                            'title': 'Window [height, width]',
                            'kind': 'integerList',
                            'defaultValue': [2, 2],
                            'min': 1,
                            'max': 4096,
                            'minItems': 2,
                            'maxItems': 2},
                           {'key': 'stride',
                            'title': 'Stride [height, width] · empty follows window',
                            'kind': 'integerList',
                            'defaultValue': None,
                            'min': 1,
                            'max': 4096,
                            'allowsNull': True,
                            'minItems': 2,
                            'maxItems': 2},
                           {'key': 'padding',
                            'title': 'Padding [height, width]',
                            'kind': 'integerList',
                            'defaultValue': [0, 0],
                            'min': 0,
                            'max': 2048,
                            'minItems': 2,
                            'maxItems': 2},
                           {'key': 'ceil_mode',
                            'title': 'Include partial final window',
                            'kind': 'boolean',
                            'defaultValue': False},
                           {'key': 'count_include_pad',
                            'title': 'Include padding in average divisor',
                            'kind': 'boolean',
                            'defaultValue': True},
                           {'key': 'divisor_override',
                            'title': 'Custom divisor · empty uses window count',
                            'kind': 'integer',
                            'defaultValue': None,
                            'min': -2147483647,
                            'max': 2147483647,
                            'allowsNull': True,
                            'enabledValue': 1}]},
 'adaptive_avg_pool1d': {'id': 'adaptive_avg_pool1d',
                         'title': 'AdaptiveAvgPool1d',
                         'category': 'Pooling',
                         'inputs': ['in'],
                         'outputs': ['out'],
                         'fields': [{'key': 'output_size',
                                     'title': 'Output length',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 4096}]},
 'adaptive_avg_pool2d': {'id': 'adaptive_avg_pool2d',
                         'title': 'AdaptiveAvgPool2d',
                         'category': 'Pooling',
                         'inputs': ['in'],
                         'outputs': ['out'],
                         'fields': [{'key': 'output_height',
                                     'title': 'Output height · empty keeps input',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 4096,
                                     'allowsNull': True},
                                    {'key': 'output_width',
                                     'title': 'Output width · empty keeps input',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 4096,
                                     'allowsNull': True}]},
 'adaptive_max_pool2d': {'id': 'adaptive_max_pool2d',
                         'title': 'AdaptiveMaxPool2d',
                         'category': 'Pooling',
                         'inputs': ['in'],
                         'outputs': ['out', 'indices'],
                         'fields': [{'key': 'output_height',
                                     'title': 'Output height · empty keeps input',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 4096,
                                     'allowsNull': True},
                                    {'key': 'output_width',
                                     'title': 'Output width · empty keeps input',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 4096,
                                     'allowsNull': True},
                                    {'key': 'return_indices',
                                     'title': 'Expose winning indices',
                                     'kind': 'boolean',
                                     'defaultValue': False}]}}
NEW_POOLING_CONTRACTS = {'avg_pool2d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                'adapterID': 'kwip.runner.graph-v2.avg_pool2d',
                'graphSchemaVersions': [2],
                'sampleRank': 3,
                'preservesBatch': True,
                'graphDTypes': ['float32'],
                'channelMinimum': 1,
                'channelMaximum': 512,
                'spatialMinimum': 1,
                'spatialMaximum': 4096,
                'spatialAxes': ['height', 'width'],
                'kernelMinimum': 1,
                'kernelMaximum': 4096,
                'strideMinimum': 1,
                'strideMaximum': 4096,
                'paddingMinimum': 0,
                'paddingMaximum': 2048,
                'paddingKernelDivisor': 2,
                'nullStrideFollowsKernel': True,
                'ceilModes': [False, True],
                'outputDTypes': {'out': 'float32'},
                'countIncludePadModes': [False, True],
                'divisorOverrideMinimum': -2147483647,
                'divisorOverrideMaximum': 2147483647,
                'divisorOverrideExcludesZero': True,
                'nullDivisorFollowsWindow': True},
 'adaptive_avg_pool1d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                         'adapterID': 'kwip.runner.graph-v2.adaptive_avg_pool1d',
                         'graphSchemaVersions': [2],
                         'sampleRank': 2,
                         'preservesBatch': True,
                         'graphDTypes': ['float32'],
                         'channelMinimum': 1,
                         'channelMaximum': 512,
                         'spatialMinimum': 1,
                         'spatialMaximum': 4096,
                         'spatialAxes': ['length'],
                         'outputSizeMinimum': 1,
                         'outputSizeMaximum': 4096,
                         'nullOutputKeepsInputAxis': False,
                         'outputDTypes': {'out': 'float32'},
                         'outputElementMaximum': 100000000},
 'adaptive_avg_pool2d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                         'adapterID': 'kwip.runner.graph-v2.adaptive_avg_pool2d',
                         'graphSchemaVersions': [2],
                         'sampleRank': 3,
                         'preservesBatch': True,
                         'graphDTypes': ['float32'],
                         'channelMinimum': 1,
                         'channelMaximum': 512,
                         'spatialMinimum': 1,
                         'spatialMaximum': 4096,
                         'spatialAxes': ['height', 'width'],
                         'outputSizeMinimum': 1,
                         'outputSizeMaximum': 4096,
                         'nullOutputKeepsInputAxis': True,
                         'outputDTypes': {'out': 'float32'},
                         'outputElementMaximum': 100000000},
 'adaptive_max_pool2d': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                         'adapterID': 'kwip.runner.graph-v2.adaptive_max_pool2d',
                         'graphSchemaVersions': [2],
                         'sampleRank': 3,
                         'preservesBatch': True,
                         'graphDTypes': ['float32'],
                         'channelMinimum': 1,
                         'channelMaximum': 512,
                         'spatialMinimum': 1,
                         'spatialMaximum': 4096,
                         'spatialAxes': ['height', 'width'],
                         'outputSizeMinimum': 1,
                         'outputSizeMaximum': 4096,
                         'nullOutputKeepsInputAxis': True,
                         'outputDTypes': {'out': 'float32', 'indices': 'int64'},
                         'outputElementMaximum': 100000000,
                         'returnIndicesModes': [False, True],
                         'indicesMeaning': 'flattenedSpatialPosition'}}


def validate_additional_pooling(row, root):
    ident = row['definition']['id']
    if json.dumps(row['definition'], sort_keys=True) != json.dumps(NEW_POOLING_DEFINITIONS[ident], sort_keys=True):
        raise ValueError('Unsupported additional pooling definition/default settings')
    if row['adapterID'] != 'kwip.runner.graph-v2.' + ident or row['shapeRuleID'] != 'kwip.shape.' + ident + '.v1':
        raise ValueError('Unknown additional pooling adapter reference')
    if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2:
        raise ValueError('Invalid additional pooling publication/configuration version')
    if json.dumps(row['runtimeContract'], sort_keys=True) != json.dumps(NEW_POOLING_CONTRACTS[ident], sort_keys=True):
        raise ValueError('Unsupported additional pooling runtime contract')
    exact(row['presentation'], 'theme icon shortTitle purpose')
    if row['presentation']['theme'] != 'green' or row['presentation']['icon'] != ident + '-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()):
        raise ValueError('Invalid additional pooling theme/icon reference')
    if row['outputConditions'] != ({'indices':'return_indices'} if ident == 'adaptive_max_pool2d' else {}):
        raise ValueError('Unsupported additional pooling output condition')
    exact(row['sources'], 'native worker tests evidence')
    references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/' + ident + '-v2.png']
    learning = row['learning']
    if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
    if learning is not None:
        exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
        if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1: raise ValueError('Invalid additional pooling content version')
        expected_lesson = 'assessment.' + ident.replace('_','-') + ('-windows' if ident == 'avg_pool2d' else '-bins')
        if learning['lessonIDs'] != [expected_lesson]: raise ValueError('Unknown additional pooling assessment identity')
        if learning['exampleID'] != 'prepared-' + ident.replace('_','') + '-classifier': raise ValueError('Unknown additional pooling example ID')
        if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
        if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing learning metadata')
        from datetime import date
        try: date.fromisoformat(learning['updatedAt'])
        except ValueError: raise ValueError('Invalid additional pooling content date')
        references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv']
    for reference in references:
        if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
        path = (root / reference).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')


TENSOR_COMPOSITION_DEFINITIONS = {'concat': {'id': 'concat',
            'title': 'Concatenate',
            'category': 'Merge',
            'inputs': ['a', 'b'],
            'outputs': ['out'],
            'fields': [{'key': 'dim',
                        'title': 'Join direction',
                        'kind': 'integer',
                        'defaultValue': -1,
                        'min': -8,
                        'max': 7}]},
 'add': {'id': 'add',
         'title': 'Add',
         'category': 'Merge',
         'inputs': ['a', 'b'],
         'outputs': ['out'],
         'fields': [{'key': 'alpha',
                     'title': 'B multiplier',
                     'kind': 'number',
                     'defaultValue': 1,
                     'min': -3.4028234663852886e+38,
                     'max': 3.4028234663852886e+38},
                    {'key': 'sample_broadcast',
                     'title': 'Broadcast sample dimensions',
                     'kind': 'boolean',
                     'defaultValue': False}]},
 'stack': {'id': 'stack',
           'title': 'Stack',
           'category': 'Merge',
           'inputs': ['a', 'b'],
           'outputs': ['out'],
           'fields': [{'key': 'dim',
                       'title': 'New axis position',
                       'kind': 'integer',
                       'defaultValue': -1,
                       'min': -8,
                       'max': 7}]},
 'permute': {'id': 'permute',
             'title': 'Permute',
             'category': 'Tensor layout',
             'inputs': ['in'],
             'outputs': ['out'],
             'fields': [{'key': 'dims',
                         'title': 'Direction order · observations first',
                         'kind': 'integerList',
                         'defaultValue': [0, 2, 1],
                         'min': -8,
                         'max': 7,
                         'minItems': 2,
                         'maxItems': 8}]}}

TENSOR_COMPOSITION_CONTRACTS = {'concat': {'version': {'major': 1, 'minor': 0, 'patch': 0},
            'adapterID': 'kwip.runner.graph-v2.concat',
            'graphSchemaVersions': [1, 2],
            'maximumRank': 8,
            'preservesBatch': True,
            'graphDTypes': ['float32', 'int64'],
            'maximumTensorElements': 100000000,
            'inputCount': 2,
            'axisMinimum': -8,
            'axisMaximum': 7,
            'excludesBatch': True,
            'inputOrder': ['a', 'b']},
 'add': {'version': {'major': 1, 'minor': 0, 'patch': 0},
         'adapterID': 'kwip.runner.graph-v2.add',
         'graphSchemaVersions': [1, 2],
         'maximumRank': 8,
         'preservesBatch': True,
         'graphDTypes': ['float32', 'int64'],
         'maximumTensorElements': 100000000,
         'inputCount': 2,
         'alphaMinimum': -3.4028234663852886e+38,
         'alphaMaximum': 3.4028234663852886e+38,
         'sampleBroadcastModes': [False, True],
         'broadcastBatch': False,
         'integerAlpha': 1,
         'integerBroadcast': False},
 'stack': {'version': {'major': 1, 'minor': 0, 'patch': 0},
           'adapterID': 'kwip.runner.graph-v2.stack',
           'graphSchemaVersions': [2],
           'maximumRank': 8,
           'preservesBatch': True,
           'graphDTypes': ['float32', 'int64'],
           'maximumTensorElements': 100000000,
           'inputCount': 2,
           'axisMinimum': -8,
           'axisMaximum': 7,
           'excludesBatch': True,
           'inputOrder': ['a', 'b'],
           'maximumInputRank': 7,
           'newAxisSize': 2},
 'permute': {'version': {'major': 1, 'minor': 0, 'patch': 0},
             'adapterID': 'kwip.runner.graph-v2.permute',
             'graphSchemaVersions': [2],
             'maximumRank': 8,
             'preservesBatch': True,
             'graphDTypes': ['float32', 'int64'],
             'maximumTensorElements': 100000000,
             'inputCount': 1,
             'axisOrderMinimumItems': 2,
             'axisOrderMaximumItems': 8,
             'batchAxisFirst': True}}


def validate_tensor_composition(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.tensor-composition' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported tensor-composition identity/version/backend')
    if not isinstance(value['components'], list): raise ValueError('Missing composition components')
    ids = [r['definition']['id'] for r in value['components']]
    if ids != ['concat','add','stack','permute']: raise ValueError('Requires four ordered composition definitions')
    tasks = dict(concat='concat-sample-axis', add='add-shared-pattern', stack='stack-new-direction', permute='permute-sample-order')
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(TENSOR_COMPOSITION_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed composition definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid composition publication/version')
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unreviewed composition adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(TENSOR_COMPOSITION_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed composition runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != ('violet' if ident == 'permute' else 'blue') or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid composition presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident+'-classifier' or learning['lessonIDs'] != ['assessment.'+tasks[ident]] or learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid composition learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing composition lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing composition learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid composition date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe composition source reference')
            path = (root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe composition source reference: '+reference)

TENSOR_ROUTING_DEFINITIONS = {'split': {'id': 'split',
           'title': 'Split',
           'category': 'Tensor layout',
           'inputs': ['in'],
           'outputs': ['part0',
                       'part1',
                       'part2',
                       'part3',
                       'part4',
                       'part5',
                       'part6',
                       'part7'],
           'fields': [{'key': 'mode',
                       'title': 'Split mode',
                       'kind': 'choice',
                       'defaultValue': 'size',
                       'choices': [{'value': 'size',
                                    'title': 'Equal-sized parts'},
                                   {'value': 'sections',
                                    'title': 'Chosen section sizes'}]},
                      {'key': 'dim',
                       'title': 'Sample direction',
                       'kind': 'integer',
                       'defaultValue': -1,
                       'min': -8,
                       'max': 7},
                      {'key': 'split_size',
                       'title': 'Split size',
                       'kind': 'integer',
                       'defaultValue': 2,
                       'min': 1,
                       'max': 100000000},
                      {'key': 'sections',
                       'title': 'Section sizes',
                       'kind': 'integerList',
                       'defaultValue': [1, 1],
                       'min': 1,
                       'max': 100000000,
                       'minItems': 1,
                       'maxItems': 8}]},
 'chunk': {'id': 'chunk',
           'title': 'Chunk',
           'category': 'Tensor layout',
           'inputs': ['in'],
           'outputs': ['part0',
                       'part1',
                       'part2',
                       'part3',
                       'part4',
                       'part5',
                       'part6',
                       'part7'],
           'fields': [{'key': 'chunks',
                       'title': 'Requested parts',
                       'kind': 'integer',
                       'defaultValue': 2,
                       'min': 1,
                       'max': 8},
                      {'key': 'dim',
                       'title': 'Sample direction',
                       'kind': 'integer',
                       'defaultValue': -1,
                       'min': -8,
                       'max': 7}]},
 'gather': {'id': 'gather',
            'title': 'Gather',
            'category': 'Tensor layout',
            'inputs': ['source', 'indices'],
            'outputs': ['out'],
            'fields': [{'key': 'dim',
                        'title': 'Sample direction',
                        'kind': 'integer',
                        'defaultValue': -1,
                        'min': -8,
                        'max': 7}]},
 'einsum': {'id': 'einsum',
            'title': 'Einsum',
            'category': 'Tensor layout',
            'inputs': ['a', 'b'],
            'outputs': ['out'],
            'fields': [{'key': 'pattern',
                        'title': 'Arrangement preset',
                        'kind': 'choice',
                        'defaultValue': 'multiply',
                        'choices': [{'value': 'multiply',
                                     'title': 'Combine tables'},
                                    {'value': 'outer',
                                     'title': 'Pair each value'},
                                    {'value': 'diagonal',
                                     'title': 'Read a diagonal'},
                                    {'value': 'reorder',
                                     'title': 'Swap sample directions'},
                                    {'value': 'custom',
                                     'title': 'Custom notation'}]},
                       {'key': 'equation',
                        'title': 'Custom notation',
                        'kind': 'text',
                        'defaultValue': 'nab,nbc->nac',
                        'minItems': 1,
                        'maxItems': 128}]}}

TENSOR_ROUTING_CONTRACTS = {'split': {'version': {'major': 1, 'minor': 0, 'patch': 0},
           'adapterID': 'kwip.runner.graph-v2.split',
           'graphSchemaVersions': [2],
           'maximumRank': 8,
           'preservesBatch': True,
           'graphDTypes': ['float32'],
           'maximumTensorElements': 100000000,
           'parameterless': True,
           'maximumOutputs': 8,
           'outputPorts': ['part0',
                           'part1',
                           'part2',
                           'part3',
                           'part4',
                           'part5',
                           'part6',
                           'part7'],
           'axisMinimum': -8,
           'axisMaximum': 7,
           'excludesBatch': True,
           'views': True,
           'splitSizeMinimum': 1,
           'splitSizeMaximum': 100000000,
           'sectionCountMinimum': 1,
           'sectionCountMaximum': 8},
 'chunk': {'version': {'major': 1, 'minor': 0, 'patch': 0},
           'adapterID': 'kwip.runner.graph-v2.chunk',
           'graphSchemaVersions': [2],
           'maximumRank': 8,
           'preservesBatch': True,
           'graphDTypes': ['float32'],
           'maximumTensorElements': 100000000,
           'parameterless': True,
           'maximumOutputs': 8,
           'outputPorts': ['part0',
                           'part1',
                           'part2',
                           'part3',
                           'part4',
                           'part5',
                           'part6',
                           'part7'],
           'axisMinimum': -8,
           'axisMaximum': 7,
           'excludesBatch': True,
           'views': True,
           'chunksMinimum': 1,
           'chunksMaximum': 8,
           'mayReturnFewer': True},
 'gather': {'version': {'major': 1, 'minor': 0, 'patch': 0},
            'adapterID': 'kwip.runner.graph-v2.gather',
            'graphSchemaVersions': [2],
            'maximumRank': 8,
            'preservesBatch': True,
            'graphDTypes': ['float32'],
            'maximumTensorElements': 100000000,
            'parameterless': True,
            'axisMinimum': -8,
            'axisMaximum': 7,
            'excludesBatch': True,
            'indicesDType': 'int64',
            'broadcast': False,
            'negativeIndices': False,
            'sparseGradient': False},
 'einsum': {'version': {'major': 1, 'minor': 0, 'patch': 0},
            'adapterID': 'kwip.runner.graph-v2.einsum',
            'graphSchemaVersions': [2],
            'maximumRank': 8,
            'preservesBatch': True,
            'graphDTypes': ['float32'],
            'maximumTensorElements': 100000000,
            'parameterless': True,
            'maximumOperands': 2,
            'equationMinimumCharacters': 1,
            'equationMaximumCharacters': 128,
            'asciiOnly': True,
            'explicitOutput': True,
            'reservedBatchLabel': 'n',
            'maximumContractionTerms': 100000000}}

def validate_tensor_routing(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.tensor-routing' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported tensor-routing identity/version/backend')
    if not isinstance(value['components'], list): raise ValueError('Missing routing components')
    ids = [r['definition']['id'] for r in value['components']]
    if ids != ['split','chunk','gather','einsum']: raise ValueError('Requires four ordered routing definitions')
    tasks = {'split':['split-size-order','split-sections-order'],'chunk':['chunk-uneven-parts'],'gather':['gather-repeated-positions'],'einsum':['einsum-combine-tables','einsum-diagonal-view']}
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(TENSOR_ROUTING_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed routing definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid routing publication/version')
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unreviewed routing adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(TENSOR_ROUTING_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed routing runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'violet' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid routing presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident+'-classifier' or learning['lessonIDs'] != ['assessment.'+task for task in tasks[ident]] or learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid routing learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing routing lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing routing learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid routing date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv', 'Sources/KwipMLCore/TensorRoutingLearningTask.swift', 'Sources/KwipMLCore/TensorRoutingLessonReference.swift', 'Sources/KwipMLCore/TensorRoutingLearningSubmission.swift', 'Sources/KwipMLCore/TensorRoutingCapabilityBridge.swift']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe routing source reference')
            path = (root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe routing source reference: '+reference)

SPATIAL_PATCH_DEFINITIONS = {'unfold': {'id': 'unfold',
            'title': 'Unfold',
            'category': 'Tensor layout',
            'inputs': ['in'],
            'outputs': ['out'],
            'fields': [{'key': 'kernel_size',
                        'title': 'Patch size',
                        'kind': 'integerList',
                        'defaultValue': [2, 2],
                        'min': 1,
                        'max': 4096,
                        'minItems': 2,
                        'maxItems': 2},
                       {'key': 'stride',
                        'title': 'Spacing between patches',
                        'kind': 'integerList',
                        'defaultValue': [1, 1],
                        'min': 1,
                        'max': 4096,
                        'minItems': 2,
                        'maxItems': 2},
                       {'key': 'dilation',
                        'title': 'Spacing within a patch',
                        'kind': 'integerList',
                        'defaultValue': [1, 1],
                        'min': 1,
                        'max': 4096,
                        'minItems': 2,
                        'maxItems': 2},
                       {'key': 'padding',
                        'title': 'Zero padding',
                        'kind': 'integerList',
                        'defaultValue': [0, 0],
                        'min': 0,
                        'max': 4096,
                        'minItems': 2,
                        'maxItems': 2}]},
 'fold': {'id': 'fold',
          'title': 'Fold',
          'category': 'Tensor layout',
          'inputs': ['in'],
          'outputs': ['out'],
          'fields': [{'key': 'kernel_size',
                      'title': 'Patch size',
                      'kind': 'integerList',
                      'defaultValue': [2, 2],
                      'min': 1,
                      'max': 4096,
                      'minItems': 2,
                      'maxItems': 2},
                     {'key': 'stride',
                      'title': 'Spacing between patches',
                      'kind': 'integerList',
                      'defaultValue': [1, 1],
                      'min': 1,
                      'max': 4096,
                      'minItems': 2,
                      'maxItems': 2},
                     {'key': 'dilation',
                      'title': 'Spacing within a patch',
                      'kind': 'integerList',
                      'defaultValue': [1, 1],
                      'min': 1,
                      'max': 4096,
                      'minItems': 2,
                      'maxItems': 2},
                     {'key': 'padding',
                      'title': 'Zero padding',
                      'kind': 'integerList',
                      'defaultValue': [0, 0],
                      'min': 0,
                      'max': 4096,
                      'minItems': 2,
                      'maxItems': 2},
                     {'key': 'output_size',
                      'title': 'Output grid size',
                      'kind': 'integerList',
                      'defaultValue': [2, 2],
                      'min': 1,
                      'max': 4096,
                      'minItems': 2,
                      'maxItems': 2}]}}

SPATIAL_PATCH_CONTRACTS = {'unfold': {'version': {'major': 1, 'minor': 0, 'patch': 0},
            'adapterID': 'kwip.runner.graph-v2.unfold',
            'graphSchemaVersions': [2],
            'preservesBatch': True,
            'graphDTypes': ['float32'],
            'parameterless': True,
            'inputSampleRank': 3,
            'outputSampleRank': 2,
            'channelMinimum': 1,
            'channelMaximum': 512,
            'spatialMinimum': 1,
            'spatialMaximum': 4096,
            'kernelMinimum': 1,
            'kernelMaximum': 4096,
            'strideMinimum': 1,
            'strideMaximum': 4096,
            'dilationMinimum': 1,
            'dilationMaximum': 4096,
            'paddingMinimum': 0,
            'paddingMaximum': 4096,
            'paddingMode': 'zeros',
            'maximumTensorElements': 100000000,
            'maximumWorkElements': 100000000,
            'packedOrder': 'channelKernelHeightKernelWidth',
            'windowOrder': 'heightWidth'},
 'fold': {'version': {'major': 1, 'minor': 0, 'patch': 0},
          'adapterID': 'kwip.runner.graph-v2.fold',
          'graphSchemaVersions': [2],
          'preservesBatch': True,
          'graphDTypes': ['float32'],
          'parameterless': True,
          'inputSampleRank': 2,
          'outputSampleRank': 3,
          'channelMinimum': 1,
          'channelMaximum': 512,
          'spatialMinimum': 1,
          'spatialMaximum': 4096,
          'kernelMinimum': 1,
          'kernelMaximum': 4096,
          'strideMinimum': 1,
          'strideMaximum': 4096,
          'dilationMinimum': 1,
          'dilationMaximum': 4096,
          'paddingMinimum': 0,
          'paddingMaximum': 4096,
          'paddingMode': 'zeros',
          'maximumTensorElements': 100000000,
          'maximumWorkElements': 100000000,
          'overlapMode': 'add',
          'uncoveredValue': 0}}

def validate_spatial_patch(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.spatial-patches' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported tensor-spatial-patch identity/version/backend')
    if not isinstance(value['components'], list): raise ValueError('Missing spatial-patch components')
    ids = [r['definition']['id'] for r in value['components']]
    if ids != ['unfold','fold']: raise ValueError('Requires two ordered spatial patch definitions')
    tasks = {'unfold':['unfold-overlapping-windows','unfold-padded-spaced-windows'],'fold':['fold-overlap-addition','fold-uncovered-positions']}
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(SPATIAL_PATCH_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed spatial-patch definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 2: raise ValueError('Invalid spatial-patch publication/version')
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unreviewed spatial-patch adapter/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(SPATIAL_PATCH_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed spatial-patch runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'violet' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid spatial-patch presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident+'-classifier' or learning['lessonIDs'] != ['assessment.'+task for task in tasks[ident]] or learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid spatial-patch learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing spatial-patch lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing spatial-patch learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid spatial-patch date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv', 'Sources/KwipMLCore/SpatialPatchLearningTask.swift', 'Sources/KwipMLCore/SpatialPatchLessonReference.swift', 'Sources/KwipMLCore/SpatialPatchLearningSubmission.swift', 'Sources/KwipMLCore/SpatialPatchCapabilityBridge.swift']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe spatial-patch source reference')
            path = (root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe spatial-patch source reference: '+reference)

OPTIMIZER_DEFINITIONS = {'optimizer_sgd': {'id': 'optimizer_sgd',
                   'title': 'SGD',
                   'category': 'Training',
                   'inputs': [],
                   'outputs': ['out'],
                   'fields': [{'key': 'lr',
                               'title': 'Learning rate',
                               'kind': 'number',
                               'defaultValue': 0.001,
                               'min': 0,
                               'max': 3.4028234663852886e+38},
                              {'key': 'momentum',
                               'title': 'Momentum',
                               'kind': 'number',
                               'defaultValue': 0.0,
                               'min': 0,
                               'max': 1.0},
                              {'key': 'dampening',
                               'title': 'Momentum dampening',
                               'kind': 'number',
                               'defaultValue': 0.0,
                               'min': 0,
                               'max': 1.0},
                              {'key': 'weight_decay',
                               'title': 'Weight decay',
                               'kind': 'number',
                               'defaultValue': 0.0,
                               'min': 0,
                               'max': 3.4028234663852886e+38},
                              {'key': 'nesterov',
                               'title': 'Look ahead',
                               'kind': 'boolean',
                               'defaultValue': False},
                              {'key': 'maximize',
                               'title': 'Maximize objective',
                               'kind': 'boolean',
                               'defaultValue': False}]},
 'optimizer_adam': {'id': 'optimizer_adam',
                    'title': 'Adam',
                    'category': 'Training',
                    'inputs': [],
                    'outputs': ['out'],
                    'fields': [{'key': 'lr',
                                'title': 'Learning rate',
                                'kind': 'number',
                                'defaultValue': 0.001,
                                'min': 0,
                                'max': 3.4028234663852886e+38},
                               {'key': 'beta1',
                                'title': 'First running average',
                                'kind': 'number',
                                'defaultValue': 0.9,
                                'min': 0,
                                'max': 0.9999999403953552},
                               {'key': 'beta2',
                                'title': 'Second running average',
                                'kind': 'number',
                                'defaultValue': 0.999,
                                'min': 0,
                                'max': 0.9999999403953552},
                               {'key': 'eps',
                                'title': 'Stability amount',
                                'kind': 'number',
                                'defaultValue': 1e-08,
                                'min': 1.1754943508222875e-38,
                                'max': 3.4028234663852886e+38},
                               {'key': 'weight_decay',
                                'title': 'Weight decay',
                                'kind': 'number',
                                'defaultValue': 0.0,
                                'min': 0,
                                'max': 3.4028234663852886e+38},
                               {'key': 'amsgrad',
                                'title': 'Keep largest squared average',
                                'kind': 'boolean',
                                'defaultValue': False},
                               {'key': 'decoupled_weight_decay',
                                'title': 'Separate weight decay',
                                'kind': 'boolean',
                                'defaultValue': False},
                               {'key': 'maximize',
                                'title': 'Maximize objective',
                                'kind': 'boolean',
                                'defaultValue': False}]},
 'optimizer_adamw': {'id': 'optimizer_adamw',
                     'title': 'AdamW',
                     'category': 'Training',
                     'inputs': [],
                     'outputs': ['out'],
                     'fields': [{'key': 'lr',
                                 'title': 'Learning rate',
                                 'kind': 'number',
                                 'defaultValue': 0.001,
                                 'min': 0,
                                 'max': 3.4028234663852886e+38},
                                {'key': 'beta1',
                                 'title': 'First running average',
                                 'kind': 'number',
                                 'defaultValue': 0.9,
                                 'min': 0,
                                 'max': 0.9999999403953552},
                                {'key': 'beta2',
                                 'title': 'Second running average',
                                 'kind': 'number',
                                 'defaultValue': 0.999,
                                 'min': 0,
                                 'max': 0.9999999403953552},
                                {'key': 'eps',
                                 'title': 'Stability amount',
                                 'kind': 'number',
                                 'defaultValue': 1e-08,
                                 'min': 1.1754943508222875e-38,
                                 'max': 3.4028234663852886e+38},
                                {'key': 'weight_decay',
                                 'title': 'Weight decay',
                                 'kind': 'number',
                                 'defaultValue': 0.01,
                                 'min': 0,
                                 'max': 3.4028234663852886e+38},
                                {'key': 'amsgrad',
                                 'title': 'Keep largest squared average',
                                 'kind': 'boolean',
                                 'defaultValue': False},
                                {'key': 'maximize',
                                 'title': 'Maximize objective',
                                 'kind': 'boolean',
                                 'defaultValue': False}]},
 'optimizer_rmsprop': {'id': 'optimizer_rmsprop',
                       'title': 'RMSprop',
                       'category': 'Training',
                       'inputs': [],
                       'outputs': ['out'],
                       'fields': [{'key': 'lr',
                                   'title': 'Learning rate',
                                   'kind': 'number',
                                   'defaultValue': 0.01,
                                   'min': 0,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'alpha',
                                   'title': 'Squared-average memory',
                                   'kind': 'number',
                                   'defaultValue': 0.99,
                                   'min': 0,
                                   'max': 1.0},
                                  {'key': 'eps',
                                   'title': 'Stability amount',
                                   'kind': 'number',
                                   'defaultValue': 1e-08,
                                   'min': 1.1754943508222875e-38,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'weight_decay',
                                   'title': 'Weight decay',
                                   'kind': 'number',
                                   'defaultValue': 0.0,
                                   'min': 0,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'momentum',
                                   'title': 'Momentum',
                                   'kind': 'number',
                                   'defaultValue': 0.0,
                                   'min': 0,
                                   'max': 1.0},
                                  {'key': 'centered',
                                   'title': 'Center the squared average',
                                   'kind': 'boolean',
                                   'defaultValue': False},
                                  {'key': 'maximize',
                                   'title': 'Maximize objective',
                                   'kind': 'boolean',
                                   'defaultValue': False}]},
 'optimizer_adagrad': {'id': 'optimizer_adagrad',
                       'title': 'Adagrad',
                       'category': 'Training',
                       'inputs': [],
                       'outputs': ['out'],
                       'fields': [{'key': 'lr',
                                   'title': 'Learning rate',
                                   'kind': 'number',
                                   'defaultValue': 0.01,
                                   'min': 0,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'lr_decay',
                                   'title': 'Learning-rate decay',
                                   'kind': 'number',
                                   'defaultValue': 0.0,
                                   'min': 0,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'weight_decay',
                                   'title': 'Weight decay',
                                   'kind': 'number',
                                   'defaultValue': 0.0,
                                   'min': 0,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'initial_accumulator_value',
                                   'title': 'Starting accumulated squares',
                                   'kind': 'number',
                                   'defaultValue': 0.0,
                                   'min': 0,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'eps',
                                   'title': 'Stability amount',
                                   'kind': 'number',
                                   'defaultValue': 1e-10,
                                   'min': 1.1754943508222875e-38,
                                   'max': 3.4028234663852886e+38},
                                  {'key': 'maximize',
                                   'title': 'Maximize objective',
                                   'kind': 'boolean',
                                   'defaultValue': False}]}}

OPTIMIZER_CONTRACTS = {'optimizer_sgd': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                   'adapterID': 'kwip.runner.optimizer.optimizer_sgd',
                   'recipeSchemaVersions': [1],
                   'configurationVersions': [1],
                   'graphDTypes': ['float32'],
                   'devices': ['cpu', 'mps'],
                   'oneParameterGroup': True,
                   'orderedTrainableParameters': True,
                   'excludesBuffers': True,
                   'foreach': False,
                   'differentiable': False,
                   'cpuResume': 'epochBoundaryExactCPU',
                   'mpsResume': 'weightsOnly',
                   'numericMaximum': 3.4028234663852886e+38,
                   'epsilonMinimum': 1.1754943508222875e-38,
                   'epsilonMaximum': 3.4028234663852886e+38,
                   'defaults': {'lr': 0.001,
                                'momentum': 0.0,
                                'dampening': 0.0,
                                'weight_decay': 0.0,
                                'nesterov': False,
                                'maximize': False},
                   'fractionMaximum': 1.0,
                   'statePolicy': 'momentumOnlyNoCounter',
                   'counterDType': 'float32',
                   'counterDevice': 'cpu',
                   'counterMaximum': 1000000000},
 'optimizer_adam': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                    'adapterID': 'kwip.runner.optimizer.optimizer_adam',
                    'recipeSchemaVersions': [1],
                    'configurationVersions': [1],
                    'graphDTypes': ['float32'],
                    'devices': ['cpu', 'mps'],
                    'oneParameterGroup': True,
                    'orderedTrainableParameters': True,
                    'excludesBuffers': True,
                    'foreach': False,
                    'differentiable': False,
                    'cpuResume': 'epochBoundaryExactCPU',
                    'mpsResume': 'weightsOnly',
                    'numericMaximum': 3.4028234663852886e+38,
                    'epsilonMinimum': 1.1754943508222875e-38,
                    'epsilonMaximum': 3.4028234663852886e+38,
                    'defaults': {'lr': 0.001,
                                 'beta1': 0.9,
                                 'beta2': 0.999,
                                 'eps': 1e-08,
                                 'weight_decay': 0.0,
                                 'amsgrad': False,
                                 'decoupled_weight_decay': False,
                                 'maximize': False},
                    'fractionMaximum': 0.9999999403953552,
                    'statePolicy': 'adamMomentsOptionalMaximum',
                    'counterDType': 'float32',
                    'counterDevice': 'cpu',
                    'counterMaximum': 1000000000},
 'optimizer_adamw': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                     'adapterID': 'kwip.runner.optimizer.optimizer_adamw',
                     'recipeSchemaVersions': [1],
                     'configurationVersions': [1],
                     'graphDTypes': ['float32'],
                     'devices': ['cpu', 'mps'],
                     'oneParameterGroup': True,
                     'orderedTrainableParameters': True,
                     'excludesBuffers': True,
                     'foreach': False,
                     'differentiable': False,
                     'cpuResume': 'epochBoundaryExactCPU',
                     'mpsResume': 'weightsOnly',
                     'numericMaximum': 3.4028234663852886e+38,
                     'epsilonMinimum': 1.1754943508222875e-38,
                     'epsilonMaximum': 3.4028234663852886e+38,
                     'defaults': {'lr': 0.001,
                                  'beta1': 0.9,
                                  'beta2': 0.999,
                                  'eps': 1e-08,
                                  'weight_decay': 0.01,
                                  'amsgrad': False,
                                  'maximize': False},
                     'fractionMaximum': 0.9999999403953552,
                     'statePolicy': 'adamMomentsOptionalMaximum',
                     'counterDType': 'float32',
                     'counterDevice': 'cpu',
                     'counterMaximum': 1000000000},
 'optimizer_rmsprop': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                       'adapterID': 'kwip.runner.optimizer.optimizer_rmsprop',
                       'recipeSchemaVersions': [1],
                       'configurationVersions': [1],
                       'graphDTypes': ['float32'],
                       'devices': ['cpu', 'mps'],
                       'oneParameterGroup': True,
                       'orderedTrainableParameters': True,
                       'excludesBuffers': True,
                       'foreach': False,
                       'differentiable': False,
                       'cpuResume': 'epochBoundaryExactCPU',
                       'mpsResume': 'weightsOnly',
                       'numericMaximum': 3.4028234663852886e+38,
                       'epsilonMinimum': 1.1754943508222875e-38,
                       'epsilonMaximum': 3.4028234663852886e+38,
                       'defaults': {'lr': 0.01,
                                    'alpha': 0.99,
                                    'eps': 1e-08,
                                    'weight_decay': 0.0,
                                    'momentum': 0.0,
                                    'centered': False,
                                    'maximize': False},
                       'fractionMaximum': 1.0,
                       'statePolicy': 'squaredOptionalMomentumAndCentered',
                       'counterDType': 'float32',
                       'counterDevice': 'cpu',
                       'counterMaximum': 1000000000},
 'optimizer_adagrad': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                       'adapterID': 'kwip.runner.optimizer.optimizer_adagrad',
                       'recipeSchemaVersions': [1],
                       'configurationVersions': [1],
                       'graphDTypes': ['float32'],
                       'devices': ['cpu', 'mps'],
                       'oneParameterGroup': True,
                       'orderedTrainableParameters': True,
                       'excludesBuffers': True,
                       'foreach': False,
                       'differentiable': False,
                       'cpuResume': 'epochBoundaryExactCPU',
                       'mpsResume': 'weightsOnly',
                       'numericMaximum': 3.4028234663852886e+38,
                       'epsilonMinimum': 1.1754943508222875e-38,
                       'epsilonMaximum': 3.4028234663852886e+38,
                       'defaults': {'lr': 0.01,
                                    'lr_decay': 0.0,
                                    'weight_decay': 0.0,
                                    'initial_accumulator_value': 0.0,
                                    'eps': 1e-10,
                                    'maximize': False},
                       'fractionMaximum': 1.0,
                       'statePolicy': 'initializedAccumulator',
                       'counterDType': 'float32',
                       'counterDevice': 'cpu',
                       'counterMaximum': 1000000000}}

def validate_optimizer(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.optimizers' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported optimizer identity/version/backend')
    if not isinstance(value['components'], list): raise ValueError('Missing optimizer components')
    if [r['definition']['id'] for r in value['components']] != list(OPTIMIZER_DEFINITIONS): raise ValueError('Requires five ordered optimizer definitions')
    tasks = dict(zip(OPTIMIZER_DEFINITIONS, ['sgd-momentum-steps','adam-running-averages','adamw-separate-decay','rmsprop-squared-memory','adagrad-accumulated-steps']))
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(OPTIMIZER_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed optimizer definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1: raise ValueError('Invalid optimizer publication/version')
        if row['adapterID'] != 'kwip.runner.optimizer.'+ident or row['shapeRuleID'] != 'kwip.recipe.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unreviewed optimizer recipe/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(OPTIMIZER_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed optimizer runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'green' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid optimizer presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published optimizer requires learning')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident.removeprefix('optimizer_')+'-classifier' or learning['lessonIDs'] != ['assessment.'+tasks[ident]] or learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid optimizer learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing optimizer lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing optimizer learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid optimizer date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv', 'Sources/KwipMLCore/OptimizerLearningTask.swift', 'Sources/KwipMLCore/OptimizerLessonReference.swift', 'Sources/KwipMLCore/OptimizerLearningSubmission.swift', 'Sources/KwipMLCore/OptimizerCapabilityBridge.swift']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe optimizer source reference')
            path = (root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe optimizer source reference: '+reference)


SCHEDULER_DEFINITIONS = {'scheduler_step_lr': {'id': 'scheduler_step_lr',
                       'title': 'StepLR',
                       'category': 'Training',
                       'inputs': [],
                       'outputs': ['out'],
                       'fields': [{'key': 'step_size',
                                   'title': 'Passes between reductions',
                                   'kind': 'integer',
                                   'defaultValue': 2,
                                   'min': 1,
                                   'max': 10000},
                                  {'key': 'gamma',
                                   'title': 'Rate multiplier',
                                   'kind': 'number',
                                   'defaultValue': 0.1,
                                   'min': 0,
                                   'max': 3.4028234663852886e+38}]},
 'scheduler_cosine_annealing_lr': {'id': 'scheduler_cosine_annealing_lr',
                                   'title': 'CosineAnnealingLR',
                                   'category': 'Training',
                                   'inputs': [],
                                   'outputs': ['out'],
                                   'fields': [{'key': 'T_max',
                                               'title': 'Passes to minimum',
                                               'kind': 'integer',
                                               'defaultValue': 10,
                                               'min': 1,
                                               'max': 10000},
                                              {'key': 'eta_min',
                                               'title': 'Minimum learning rate',
                                               'kind': 'number',
                                               'defaultValue': 0.0,
                                               'min': 0,
                                               'max': 3.4028234663852886e+38}]},
 'scheduler_reduce_on_plateau': {'id': 'scheduler_reduce_on_plateau',
                                 'title': 'ReduceLROnPlateau',
                                 'category': 'Training',
                                 'inputs': [],
                                 'outputs': ['out'],
                                 'fields': [{'key': 'metric',
                                             'title': 'Measure to watch',
                                             'kind': 'choice',
                                             'defaultValue': 'validation_loss',
                                             'choices': [{'value': 'validation_loss',
                                                          'title': 'Validation '
                                                                   'loss'},
                                                         {'value': 'validation_accuracy',
                                                          'title': 'Validation '
                                                                   'accuracy'}]},
                                            {'key': 'mode',
                                             'title': 'Improvement direction',
                                             'kind': 'choice',
                                             'defaultValue': 'min',
                                             'choices': [{'value': 'min',
                                                          'title': 'Smaller is '
                                                                   'better'},
                                                         {'value': 'max',
                                                          'title': 'Larger is '
                                                                   'better'}]},
                                            {'key': 'factor',
                                             'title': 'Reduction multiplier',
                                             'kind': 'number',
                                             'defaultValue': 0.1,
                                             'min': 0,
                                             'max': 1},
                                            {'key': 'patience',
                                             'title': 'Passes to wait',
                                             'kind': 'integer',
                                             'defaultValue': 10,
                                             'min': 0,
                                             'max': 10000},
                                            {'key': 'threshold',
                                             'title': 'Required improvement',
                                             'kind': 'number',
                                             'defaultValue': 0.0001,
                                             'min': 0,
                                             'max': 3.4028234663852886e+38},
                                            {'key': 'threshold_mode',
                                             'title': 'Improvement comparison',
                                             'kind': 'choice',
                                             'defaultValue': 'rel',
                                             'choices': [{'value': 'rel',
                                                          'title': 'Relative '
                                                                   'change'},
                                                         {'value': 'abs',
                                                          'title': 'Absolute '
                                                                   'change'}]},
                                            {'key': 'cooldown',
                                             'title': 'Recovery passes',
                                             'kind': 'integer',
                                             'defaultValue': 0,
                                             'min': 0,
                                             'max': 10000},
                                            {'key': 'min_lr',
                                             'title': 'Minimum learning rate',
                                             'kind': 'number',
                                             'defaultValue': 0.0,
                                             'min': 0,
                                             'max': 3.4028234663852886e+38},
                                            {'key': 'eps',
                                             'title': 'Smallest rate change',
                                             'kind': 'number',
                                             'defaultValue': 1e-08,
                                             'min': 0,
                                             'max': 3.4028234663852886e+38}]},
 'scheduler_one_cycle_lr': {'id': 'scheduler_one_cycle_lr',
                            'title': 'OneCycleLR',
                            'category': 'Training',
                            'inputs': [],
                            'outputs': ['out'],
                            'fields': [{'key': 'max_lr',
                                        'title': 'Peak learning rate',
                                        'kind': 'number',
                                        'defaultValue': 0.1,
                                        'min': 0,
                                        'max': 3.4028234663852886e+38},
                                       {'key': 'planned_total_batches',
                                        'title': 'Planned total batches',
                                        'kind': 'integer',
                                        'defaultValue': 10,
                                        'min': 3,
                                        'max': 10000},
                                       {'key': 'pct_start',
                                        'title': 'Rising portion',
                                        'kind': 'number',
                                        'defaultValue': 0.3,
                                        'min': 1.1754943508222875e-38,
                                        'max': 1},
                                       {'key': 'anneal_strategy',
                                        'title': 'Curve shape',
                                        'kind': 'choice',
                                        'defaultValue': 'cos',
                                        'choices': [{'value': 'cos',
                                                     'title': 'Smooth cosine'},
                                                    {'value': 'linear',
                                                     'title': 'Straight '
                                                              'lines'}]},
                                       {'key': 'cycle_momentum',
                                        'title': 'Vary remembered movement',
                                        'kind': 'boolean',
                                        'defaultValue': True},
                                       {'key': 'base_momentum',
                                        'title': 'Minimum remembered movement',
                                        'kind': 'number',
                                        'defaultValue': 0.85,
                                        'min': 0,
                                        'max': 1},
                                       {'key': 'max_momentum',
                                        'title': 'Maximum remembered movement',
                                        'kind': 'number',
                                        'defaultValue': 0.95,
                                        'min': 0,
                                        'max': 1},
                                       {'key': 'div_factor',
                                        'title': 'Starting rate divisor',
                                        'kind': 'number',
                                        'defaultValue': 25.0,
                                        'min': 1.1754943508222875e-38,
                                        'max': 3.4028234663852886e+38},
                                       {'key': 'final_div_factor',
                                        'title': 'Final rate divisor',
                                        'kind': 'number',
                                        'defaultValue': 10000.0,
                                        'min': 1.1754943508222875e-38,
                                        'max': 3.4028234663852886e+38},
                                       {'key': 'three_phase',
                                        'title': 'Use three phases',
                                        'kind': 'boolean',
                                        'defaultValue': False}]},
 'scheduler_linear_lr': {'id': 'scheduler_linear_lr',
                         'title': 'LinearLR',
                         'category': 'Training',
                         'inputs': [],
                         'outputs': ['out'],
                         'fields': [{'key': 'start_factor',
                                     'title': 'Starting rate fraction',
                                     'kind': 'number',
                                     'defaultValue': 0.3333333333333333,
                                     'min': 1.1754943508222875e-38,
                                     'max': 1},
                                    {'key': 'end_factor',
                                     'title': 'Ending rate fraction',
                                     'kind': 'number',
                                     'defaultValue': 1.0,
                                     'min': 1.1754943508222875e-38,
                                     'max': 1},
                                    {'key': 'total_iters',
                                     'title': 'Passes to ending fraction',
                                     'kind': 'integer',
                                     'defaultValue': 5,
                                     'min': 1,
                                     'max': 10000}]}}

SCHEDULER_CONTRACTS = {'scheduler_step_lr': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                       'adapterID': 'kwip.runner.scheduler.scheduler_step_lr',
                       'recipeSchemaVersions': [1],
                       'configurationVersions': [1],
                       'workflowSchemaVersions': [7],
                       'graphDTypes': ['float32'],
                       'devices': ['cpu', 'mps'],
                       'requiresExplicitOptimizer': True,
                       'oneParameterGroup': True,
                       'optimizerComponentIDs': ['optimizer_sgd',
                                                 'optimizer_adam',
                                                 'optimizer_adamw',
                                                 'optimizer_rmsprop',
                                                 'optimizer_adagrad'],
                       'momentumCyclingOptimizerIDs': [],
                       'timing': 'afterCompletedEpochValidation',
                       'constructorInitializationSeparate': True,
                       'tickMaximum': 10000,
                       'traceByteMaximum': 4000000,
                       'numericMaximum': 3.4028234663852886e+38,
                       'positiveMinimum': 1.1754943508222875e-38,
                       'scheduledBetaMaximum': 0.9999999403953552,
                       'defaults': {'step_size': 2, 'gamma': 0.1},
                       'cpuResume': 'epochBoundaryExactCPU',
                       'mpsResume': 'weightsOnly',
                       'fineTune': 'freshOptimizerAndScheduler',
                       'statePolicy': 'strictRecipeAndRecomputedTrace',
                       'terminalNextRatePolicy': 'nonnegativeFinite',
                       'horizonPolicy': 'boundedEpochTicks',
                       'export': 'freshTrainingSequence'},
 'scheduler_cosine_annealing_lr': {'version': {'major': 1,
                                               'minor': 0,
                                               'patch': 0},
                                   'adapterID': 'kwip.runner.scheduler.scheduler_cosine_annealing_lr',
                                   'recipeSchemaVersions': [1],
                                   'configurationVersions': [1],
                                   'workflowSchemaVersions': [7],
                                   'graphDTypes': ['float32'],
                                   'devices': ['cpu', 'mps'],
                                   'requiresExplicitOptimizer': True,
                                   'oneParameterGroup': True,
                                   'optimizerComponentIDs': ['optimizer_sgd',
                                                             'optimizer_adam',
                                                             'optimizer_adamw',
                                                             'optimizer_rmsprop',
                                                             'optimizer_adagrad'],
                                   'momentumCyclingOptimizerIDs': [],
                                   'timing': 'afterCompletedEpochValidation',
                                   'constructorInitializationSeparate': True,
                                   'tickMaximum': 10000,
                                   'traceByteMaximum': 4000000,
                                   'numericMaximum': 3.4028234663852886e+38,
                                   'positiveMinimum': 1.1754943508222875e-38,
                                   'scheduledBetaMaximum': 0.9999999403953552,
                                   'defaults': {'T_max': 10, 'eta_min': 0.0},
                                   'cpuResume': 'epochBoundaryExactCPU',
                                   'mpsResume': 'weightsOnly',
                                   'fineTune': 'freshOptimizerAndScheduler',
                                   'statePolicy': 'strictRecipeAndRecomputedTrace',
                                   'terminalNextRatePolicy': 'nonnegativeFinite',
                                   'horizonPolicy': 'decreasingHalfCycle',
                                   'export': 'freshTrainingSequence'},
 'scheduler_reduce_on_plateau': {'version': {'major': 1,
                                             'minor': 0,
                                             'patch': 0},
                                 'adapterID': 'kwip.runner.scheduler.scheduler_reduce_on_plateau',
                                 'recipeSchemaVersions': [1],
                                 'configurationVersions': [1],
                                 'workflowSchemaVersions': [7],
                                 'graphDTypes': ['float32'],
                                 'devices': ['cpu', 'mps'],
                                 'requiresExplicitOptimizer': True,
                                 'oneParameterGroup': True,
                                 'optimizerComponentIDs': ['optimizer_sgd',
                                                           'optimizer_adam',
                                                           'optimizer_adamw',
                                                           'optimizer_rmsprop',
                                                           'optimizer_adagrad'],
                                 'momentumCyclingOptimizerIDs': [],
                                 'timing': 'afterCompletedEpochValidation',
                                 'constructorInitializationSeparate': True,
                                 'tickMaximum': 10000,
                                 'traceByteMaximum': 4000000,
                                 'numericMaximum': 3.4028234663852886e+38,
                                 'positiveMinimum': 1.1754943508222875e-38,
                                 'scheduledBetaMaximum': 0.9999999403953552,
                                 'defaults': {'mode': 'min',
                                              'factor': 0.1,
                                              'patience': 10,
                                              'threshold': 0.0001,
                                              'threshold_mode': 'rel',
                                              'cooldown': 0,
                                              'min_lr': 0.0,
                                              'eps': 1e-08,
                                              'metric': 'validation_loss'},
                                 'cpuResume': 'epochBoundaryExactCPU',
                                 'mpsResume': 'weightsOnly',
                                 'fineTune': 'freshOptimizerAndScheduler',
                                 'statePolicy': 'strictRecipeAndRecomputedTrace',
                                 'terminalNextRatePolicy': 'nonnegativeFinite',
                                 'horizonPolicy': 'boundedEpochTicks',
                                 'export': 'freshTrainingSequence'},
 'scheduler_one_cycle_lr': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                            'adapterID': 'kwip.runner.scheduler.scheduler_one_cycle_lr',
                            'recipeSchemaVersions': [1],
                            'configurationVersions': [1],
                            'workflowSchemaVersions': [7],
                            'graphDTypes': ['float32'],
                            'devices': ['cpu', 'mps'],
                            'requiresExplicitOptimizer': True,
                            'oneParameterGroup': True,
                            'optimizerComponentIDs': ['optimizer_sgd',
                                                      'optimizer_adam',
                                                      'optimizer_adamw',
                                                      'optimizer_rmsprop',
                                                      'optimizer_adagrad'],
                            'momentumCyclingOptimizerIDs': ['optimizer_sgd',
                                                            'optimizer_adam',
                                                            'optimizer_adamw',
                                                            'optimizer_rmsprop'],
                            'timing': 'afterOptimizerBatch',
                            'constructorInitializationSeparate': True,
                            'tickMaximum': 10000,
                            'traceByteMaximum': 4000000,
                            'numericMaximum': 3.4028234663852886e+38,
                            'positiveMinimum': 1.1754943508222875e-38,
                            'scheduledBetaMaximum': 0.9999999403953552,
                            'defaults': {'max_lr': 0.1,
                                         'planned_total_batches': 10,
                                         'pct_start': 0.3,
                                         'anneal_strategy': 'cos',
                                         'cycle_momentum': True,
                                         'base_momentum': 0.85,
                                         'max_momentum': 0.95,
                                         'div_factor': 25.0,
                                         'final_div_factor': 10000.0,
                                         'three_phase': False},
                            'cpuResume': 'epochBoundaryExactCPU',
                            'mpsResume': 'weightsOnly',
                            'fineTune': 'freshOptimizerAndScheduler',
                            'statePolicy': 'strictRecipeAndRecomputedTrace',
                            'terminalNextRatePolicy': 'finiteUnusedAtExhaustedHorizon',
                            'horizonPolicy': 'plannedTotalBatches',
                            'export': 'freshTrainingSequence'},
 'scheduler_linear_lr': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                         'adapterID': 'kwip.runner.scheduler.scheduler_linear_lr',
                         'recipeSchemaVersions': [1],
                         'configurationVersions': [1],
                         'workflowSchemaVersions': [7],
                         'graphDTypes': ['float32'],
                         'devices': ['cpu', 'mps'],
                         'requiresExplicitOptimizer': True,
                         'oneParameterGroup': True,
                         'optimizerComponentIDs': ['optimizer_sgd',
                                                   'optimizer_adam',
                                                   'optimizer_adamw',
                                                   'optimizer_rmsprop',
                                                   'optimizer_adagrad'],
                         'momentumCyclingOptimizerIDs': [],
                         'timing': 'afterCompletedEpochValidation',
                         'constructorInitializationSeparate': True,
                         'tickMaximum': 10000,
                         'traceByteMaximum': 4000000,
                         'numericMaximum': 3.4028234663852886e+38,
                         'positiveMinimum': 1.1754943508222875e-38,
                         'scheduledBetaMaximum': 0.9999999403953552,
                         'defaults': {'start_factor': 0.3333333333333333,
                                      'end_factor': 1.0,
                                      'total_iters': 5},
                         'cpuResume': 'epochBoundaryExactCPU',
                         'mpsResume': 'weightsOnly',
                         'fineTune': 'freshOptimizerAndScheduler',
                         'statePolicy': 'strictRecipeAndRecomputedTrace',
                         'terminalNextRatePolicy': 'nonnegativeFinite',
                         'horizonPolicy': 'boundedEpochTicks',
                         'export': 'freshTrainingSequence'}}

def validate_scheduler(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.schedulers' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported scheduler identity/version/backend')
    if not isinstance(value['components'], list): raise ValueError('Missing scheduler components')
    if [r['definition']['id'] for r in value['components']] != list(SCHEDULER_DEFINITIONS): raise ValueError('Requires five ordered scheduler definitions')
    tasks = dict(zip(SCHEDULER_DEFINITIONS, ['step-lr-intervals','cosine-lr-curve','plateau-lr-improvement','one-cycle-lr-batches','linear-lr-warmup']))
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(SCHEDULER_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed scheduler definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1: raise ValueError('Invalid scheduler publication/version')
        if row['adapterID'] != 'kwip.runner.scheduler.'+ident or row['shapeRuleID'] != 'kwip.recipe.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unreviewed scheduler recipe/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(SCHEDULER_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed scheduler runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'green' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid scheduler presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published scheduler requires learning')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident.removeprefix('scheduler_').replace('_','-')+'-classifier' or learning['lessonIDs'] != ['assessment.'+tasks[ident]] or learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid scheduler learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing scheduler lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing scheduler learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid scheduler date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv', 'Sources/KwipMLCore/SchedulerLearningTask.swift', 'Sources/KwipMLCore/SchedulerLessonReference.swift', 'Sources/KwipMLCore/SchedulerLearningSubmission.swift', 'Sources/KwipMLCore/SchedulerCapabilityBridge.swift']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe scheduler source reference')
            path = (root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe scheduler source reference: '+reference)


RECURRENT_DEFINITIONS = {'rnn': {'id': 'rnn',
         'title': 'RNN',
         'category': 'Layers',
         'inputs': ['in', 'hidden'],
         'outputs': ['out', 'hidden'],
         'fields': [{'key': 'input_size',
                     'title': 'Input width',
                     'kind': 'integer',
                     'defaultValue': 2,
                     'min': 1,
                     'max': 4096},
                    {'key': 'hidden_size',
                     'title': 'Hidden width',
                     'kind': 'integer',
                     'defaultValue': 2,
                     'min': 1,
                     'max': 512},
                    {'key': 'bias', 'title': 'Use bias', 'kind': 'boolean', 'defaultValue': True},
                    {'key': 'state_mode',
                     'title': 'Starting state',
                     'kind': 'choice',
                     'defaultValue': 'zero',
                     'choices': [{'value': 'zero', 'title': 'Start from zeros'},
                                 {'value': 'supplied', 'title': 'Use state connections'}]},
                    {'key': 'num_layers',
                     'title': 'Layer count',
                     'kind': 'integer',
                     'defaultValue': 1,
                     'min': 1,
                     'max': 8},
                    {'key': 'bidirectional',
                     'title': 'Reverse direction too',
                     'kind': 'boolean',
                     'defaultValue': False},
                    {'key': 'dropout',
                     'title': 'Between-layer dropout',
                     'kind': 'number',
                     'defaultValue': 0,
                     'min': 0,
                     'max': 1},
                    {'key': 'nonlinearity',
                     'title': 'Behavior',
                     'kind': 'choice',
                     'defaultValue': 'tanh',
                     'choices': [{'value': 'tanh', 'title': 'Smooth bounded response'},
                                 {'value': 'relu', 'title': 'Keep positive values'}]}]},
 'lstm': {'id': 'lstm',
          'title': 'LSTM',
          'category': 'Layers',
          'inputs': ['in', 'hidden', 'cell'],
          'outputs': ['out', 'hidden', 'cell'],
          'fields': [{'key': 'input_size',
                      'title': 'Input width',
                      'kind': 'integer',
                      'defaultValue': 2,
                      'min': 1,
                      'max': 4096},
                     {'key': 'hidden_size',
                      'title': 'Hidden width',
                      'kind': 'integer',
                      'defaultValue': 2,
                      'min': 1,
                      'max': 512},
                     {'key': 'bias', 'title': 'Use bias', 'kind': 'boolean', 'defaultValue': True},
                     {'key': 'state_mode',
                      'title': 'Starting state',
                      'kind': 'choice',
                      'defaultValue': 'zero',
                      'choices': [{'value': 'zero', 'title': 'Start from zeros'},
                                  {'value': 'supplied', 'title': 'Use state connections'}]},
                     {'key': 'num_layers',
                      'title': 'Layer count',
                      'kind': 'integer',
                      'defaultValue': 1,
                      'min': 1,
                      'max': 8},
                     {'key': 'bidirectional',
                      'title': 'Reverse direction too',
                      'kind': 'boolean',
                      'defaultValue': False},
                     {'key': 'dropout',
                      'title': 'Between-layer dropout',
                      'kind': 'number',
                      'defaultValue': 0,
                      'min': 0,
                      'max': 1},
                     {'key': 'proj_size',
                      'title': 'Projected hidden width',
                      'kind': 'integer',
                      'defaultValue': 0,
                      'min': 0,
                      'max': 511}]},
 'gru': {'id': 'gru',
         'title': 'GRU',
         'category': 'Layers',
         'inputs': ['in', 'hidden'],
         'outputs': ['out', 'hidden'],
         'fields': [{'key': 'input_size',
                     'title': 'Input width',
                     'kind': 'integer',
                     'defaultValue': 2,
                     'min': 1,
                     'max': 4096},
                    {'key': 'hidden_size',
                     'title': 'Hidden width',
                     'kind': 'integer',
                     'defaultValue': 2,
                     'min': 1,
                     'max': 512},
                    {'key': 'bias', 'title': 'Use bias', 'kind': 'boolean', 'defaultValue': True},
                    {'key': 'state_mode',
                     'title': 'Starting state',
                     'kind': 'choice',
                     'defaultValue': 'zero',
                     'choices': [{'value': 'zero', 'title': 'Start from zeros'},
                                 {'value': 'supplied', 'title': 'Use state connections'}]},
                    {'key': 'num_layers',
                     'title': 'Layer count',
                     'kind': 'integer',
                     'defaultValue': 1,
                     'min': 1,
                     'max': 8},
                    {'key': 'bidirectional',
                     'title': 'Reverse direction too',
                     'kind': 'boolean',
                     'defaultValue': False},
                    {'key': 'dropout',
                     'title': 'Between-layer dropout',
                     'kind': 'number',
                     'defaultValue': 0,
                     'min': 0,
                     'max': 1}]},
 'rnn_cell': {'id': 'rnn_cell',
              'title': 'RNN Cell',
              'category': 'Layers',
              'inputs': ['in', 'hidden'],
              'outputs': ['out', 'hidden'],
              'fields': [{'key': 'input_size',
                          'title': 'Input width',
                          'kind': 'integer',
                          'defaultValue': 2,
                          'min': 1,
                          'max': 4096},
                         {'key': 'hidden_size',
                          'title': 'Hidden width',
                          'kind': 'integer',
                          'defaultValue': 2,
                          'min': 1,
                          'max': 512},
                         {'key': 'bias', 'title': 'Use bias', 'kind': 'boolean', 'defaultValue': True},
                         {'key': 'state_mode',
                          'title': 'Starting state',
                          'kind': 'choice',
                          'defaultValue': 'zero',
                          'choices': [{'value': 'zero', 'title': 'Start from zeros'},
                                      {'value': 'supplied', 'title': 'Use state connections'}]},
                         {'key': 'nonlinearity',
                          'title': 'Behavior',
                          'kind': 'choice',
                          'defaultValue': 'tanh',
                          'choices': [{'value': 'tanh', 'title': 'Smooth bounded response'},
                                      {'value': 'relu', 'title': 'Keep positive values'}]}]},
 'lstm_cell': {'id': 'lstm_cell',
               'title': 'LSTM Cell',
               'category': 'Layers',
               'inputs': ['in', 'hidden', 'cell'],
               'outputs': ['out', 'hidden', 'cell'],
               'fields': [{'key': 'input_size',
                           'title': 'Input width',
                           'kind': 'integer',
                           'defaultValue': 2,
                           'min': 1,
                           'max': 4096},
                          {'key': 'hidden_size',
                           'title': 'Hidden width',
                           'kind': 'integer',
                           'defaultValue': 2,
                           'min': 1,
                           'max': 512},
                          {'key': 'bias', 'title': 'Use bias', 'kind': 'boolean', 'defaultValue': True},
                          {'key': 'state_mode',
                           'title': 'Starting state',
                           'kind': 'choice',
                           'defaultValue': 'zero',
                           'choices': [{'value': 'zero', 'title': 'Start from zeros'},
                                       {'value': 'supplied', 'title': 'Use state connections'}]}]},
 'gru_cell': {'id': 'gru_cell',
              'title': 'GRU Cell',
              'category': 'Layers',
              'inputs': ['in', 'hidden'],
              'outputs': ['out', 'hidden'],
              'fields': [{'key': 'input_size',
                          'title': 'Input width',
                          'kind': 'integer',
                          'defaultValue': 2,
                          'min': 1,
                          'max': 4096},
                         {'key': 'hidden_size',
                          'title': 'Hidden width',
                          'kind': 'integer',
                          'defaultValue': 2,
                          'min': 1,
                          'max': 512},
                         {'key': 'bias', 'title': 'Use bias', 'kind': 'boolean', 'defaultValue': True},
                         {'key': 'state_mode',
                          'title': 'Starting state',
                          'kind': 'choice',
                          'defaultValue': 'zero',
                          'choices': [{'value': 'zero', 'title': 'Start from zeros'},
                                      {'value': 'supplied', 'title': 'Use state connections'}]}]}}

RECURRENT_CONTRACTS = {'rnn': {'version': {'major': 1, 'minor': 0, 'patch': 0},
         'adapterID': 'kwip.runner.graph-v2.rnn',
         'graphSchemaVersions': [2],
         'preservesBatch': True,
         'graphDTypes': ['float32'],
         'inputSampleRank': 2,
         'stateSampleRank': 2,
         'inputMinimum': 1,
         'inputMaximum': 4096,
         'hiddenMinimum': 1,
         'hiddenMaximum': 512,
         'layerMinimum': 1,
         'layerMaximum': 8,
         'timeMinimum': 1,
         'timeMaximum': 4096,
         'maximumParameters': 2000000,
         'maximumTensorElements': 100000000,
         'maximumGateElements': 100000000,
         'maximumWorkElements': 100000000,
         'stateModes': ['zero', 'supplied'],
         'inputPorts': ['in', 'hidden'],
         'outputPorts': ['out', 'hidden'],
         'stateLayout': 'observationFirst',
         'nativeStateLayout': 'layerDirectionObservationWidth',
         'statePersistence': 'explicitConnectionsOnly',
         'cellHiddenAlias': False,
         'dropoutPolicy': 'betweenLayersTrainingOnly',
         'parameterOrder': 'nativeLayerDirectionGateOrder',
         'projectionPolicy': 'notApplicable',
         'projectionMaximum': 0},
 'lstm': {'version': {'major': 1, 'minor': 0, 'patch': 0},
          'adapterID': 'kwip.runner.graph-v2.lstm',
          'graphSchemaVersions': [2],
          'preservesBatch': True,
          'graphDTypes': ['float32'],
          'inputSampleRank': 2,
          'stateSampleRank': 2,
          'inputMinimum': 1,
          'inputMaximum': 4096,
          'hiddenMinimum': 1,
          'hiddenMaximum': 512,
          'layerMinimum': 1,
          'layerMaximum': 8,
          'timeMinimum': 1,
          'timeMaximum': 4096,
          'maximumParameters': 2000000,
          'maximumTensorElements': 100000000,
          'maximumGateElements': 100000000,
          'maximumWorkElements': 100000000,
          'stateModes': ['zero', 'supplied'],
          'inputPorts': ['in', 'hidden', 'cell'],
          'outputPorts': ['out', 'hidden', 'cell'],
          'stateLayout': 'observationFirst',
          'nativeStateLayout': 'layerDirectionObservationWidth',
          'statePersistence': 'explicitConnectionsOnly',
          'cellHiddenAlias': False,
          'dropoutPolicy': 'betweenLayersTrainingOnly',
          'parameterOrder': 'nativeLayerDirectionGateOrder',
          'projectionPolicy': 'cpuOnly',
          'projectionMaximum': 511},
 'gru': {'version': {'major': 1, 'minor': 0, 'patch': 0},
         'adapterID': 'kwip.runner.graph-v2.gru',
         'graphSchemaVersions': [2],
         'preservesBatch': True,
         'graphDTypes': ['float32'],
         'inputSampleRank': 2,
         'stateSampleRank': 2,
         'inputMinimum': 1,
         'inputMaximum': 4096,
         'hiddenMinimum': 1,
         'hiddenMaximum': 512,
         'layerMinimum': 1,
         'layerMaximum': 8,
         'timeMinimum': 1,
         'timeMaximum': 4096,
         'maximumParameters': 2000000,
         'maximumTensorElements': 100000000,
         'maximumGateElements': 100000000,
         'maximumWorkElements': 100000000,
         'stateModes': ['zero', 'supplied'],
         'inputPorts': ['in', 'hidden'],
         'outputPorts': ['out', 'hidden'],
         'stateLayout': 'observationFirst',
         'nativeStateLayout': 'layerDirectionObservationWidth',
         'statePersistence': 'explicitConnectionsOnly',
         'cellHiddenAlias': False,
         'dropoutPolicy': 'betweenLayersTrainingOnly',
         'parameterOrder': 'nativeLayerDirectionGateOrder',
         'projectionPolicy': 'notApplicable',
         'projectionMaximum': 0},
 'rnn_cell': {'version': {'major': 1, 'minor': 0, 'patch': 0},
              'adapterID': 'kwip.runner.graph-v2.rnn_cell',
              'graphSchemaVersions': [2],
              'preservesBatch': True,
              'graphDTypes': ['float32'],
              'inputSampleRank': 1,
              'stateSampleRank': 1,
              'inputMinimum': 1,
              'inputMaximum': 4096,
              'hiddenMinimum': 1,
              'hiddenMaximum': 512,
              'layerMinimum': 1,
              'layerMaximum': 1,
              'timeMinimum': 1,
              'timeMaximum': 1,
              'maximumParameters': 2000000,
              'maximumTensorElements': 100000000,
              'maximumGateElements': 100000000,
              'maximumWorkElements': 100000000,
              'stateModes': ['zero', 'supplied'],
              'inputPorts': ['in', 'hidden'],
              'outputPorts': ['out', 'hidden'],
              'stateLayout': 'observationFirst',
              'nativeStateLayout': 'observationFirst',
              'statePersistence': 'explicitConnectionsOnly',
              'cellHiddenAlias': True,
              'dropoutPolicy': 'none',
              'parameterOrder': 'nativeLayerDirectionGateOrder',
              'projectionPolicy': 'notApplicable',
              'projectionMaximum': 0},
 'lstm_cell': {'version': {'major': 1, 'minor': 0, 'patch': 0},
               'adapterID': 'kwip.runner.graph-v2.lstm_cell',
               'graphSchemaVersions': [2],
               'preservesBatch': True,
               'graphDTypes': ['float32'],
               'inputSampleRank': 1,
               'stateSampleRank': 1,
               'inputMinimum': 1,
               'inputMaximum': 4096,
               'hiddenMinimum': 1,
               'hiddenMaximum': 512,
               'layerMinimum': 1,
               'layerMaximum': 1,
               'timeMinimum': 1,
               'timeMaximum': 1,
               'maximumParameters': 2000000,
               'maximumTensorElements': 100000000,
               'maximumGateElements': 100000000,
               'maximumWorkElements': 100000000,
               'stateModes': ['zero', 'supplied'],
               'inputPorts': ['in', 'hidden', 'cell'],
               'outputPorts': ['out', 'hidden', 'cell'],
               'stateLayout': 'observationFirst',
               'nativeStateLayout': 'observationFirst',
               'statePersistence': 'explicitConnectionsOnly',
               'cellHiddenAlias': True,
               'dropoutPolicy': 'none',
               'parameterOrder': 'nativeLayerDirectionGateOrder',
               'projectionPolicy': 'notApplicable',
               'projectionMaximum': 0},
 'gru_cell': {'version': {'major': 1, 'minor': 0, 'patch': 0},
              'adapterID': 'kwip.runner.graph-v2.gru_cell',
              'graphSchemaVersions': [2],
              'preservesBatch': True,
              'graphDTypes': ['float32'],
              'inputSampleRank': 1,
              'stateSampleRank': 1,
              'inputMinimum': 1,
              'inputMaximum': 4096,
              'hiddenMinimum': 1,
              'hiddenMaximum': 512,
              'layerMinimum': 1,
              'layerMaximum': 1,
              'timeMinimum': 1,
              'timeMaximum': 1,
              'maximumParameters': 2000000,
              'maximumTensorElements': 100000000,
              'maximumGateElements': 100000000,
              'maximumWorkElements': 100000000,
              'stateModes': ['zero', 'supplied'],
              'inputPorts': ['in', 'hidden'],
              'outputPorts': ['out', 'hidden'],
              'stateLayout': 'observationFirst',
              'nativeStateLayout': 'observationFirst',
              'statePersistence': 'explicitConnectionsOnly',
              'cellHiddenAlias': True,
              'dropoutPolicy': 'none',
              'parameterOrder': 'nativeLayerDirectionGateOrder',
              'projectionPolicy': 'notApplicable',
              'projectionMaximum': 0}}

def validate_recurrent(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.recurrent' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported recurrent identity/version/backend')
    if not isinstance(value['components'], list) or [r['definition']['id'] for r in value['components']] != list(RECURRENT_DEFINITIONS):
        raise ValueError('Requires six ordered recurrent definitions')
    tasks = dict(zip(RECURRENT_DEFINITIONS, ['rnn-sequence-memory','lstm-separate-memory','gru-reset-memory','rnn-cell-step','lstm-cell-step','gru-cell-step']))
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation inputConditions outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        if json.dumps(row['definition'], sort_keys=True) != json.dumps(RECURRENT_DEFINITIONS[ident], sort_keys=True): raise ValueError('Unreviewed recurrent definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1: raise ValueError('Invalid recurrent publication/version')
        expected_inputs = {key: 'state_mode=supplied' for key in RECURRENT_DEFINITIONS[ident]['inputs'] if key != 'in'}
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['inputConditions'] != expected_inputs or row['outputConditions'] != {}: raise ValueError('Unreviewed recurrent adapter/conditional state ports')
        if json.dumps(row['runtimeContract'], sort_keys=True) != json.dumps(RECURRENT_CONTRACTS[ident], sort_keys=True): raise ValueError('Unreviewed recurrent runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'rose' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid recurrent presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published recurrent component requires learning')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident.replace('_','-')+'-classifier' or learning['lessonIDs'] != ['assessment.'+tasks[ident]] or learning['sourceResource'] != 'workflow-observations': raise ValueError('Invalid recurrent learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing recurrent lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing recurrent learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid recurrent date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv', 'Sources/KwipMLCore/RecurrentLearningTask.swift', 'Sources/KwipMLCore/RecurrentLessonReference.swift', 'Sources/KwipMLCore/RecurrentLearningSubmission.swift', 'Sources/KwipMLCore/RecurrentCapabilityBridge.swift']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe recurrent source reference')
            path = (root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe recurrent source reference: '+reference)


ATTENTION_DEFINITIONS = {'multihead_attention': {'id': 'multihead_attention',
                         'title': 'Multihead attention',
                         'category': 'Layers',
                         'inputs': ['query', 'key', 'value', 'attn_mask', 'key_padding_mask'],
                         'outputs': ['out', 'weights'],
                         'fields': [{'key': 'embed_dim',
                                     'title': 'Model width',
                                     'kind': 'integer',
                                     'defaultValue': 4,
                                     'min': 1,
                                     'max': 512},
                                    {'key': 'num_heads',
                                     'title': 'Heads',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 32},
                                    {'key': 'kdim',
                                     'title': 'Key width',
                                     'kind': 'integer',
                                     'defaultValue': None,
                                     'min': 1,
                                     'max': 4096,
                                     'allowsNull': True,
                                     'enabledValue': 4},
                                    {'key': 'vdim',
                                     'title': 'Value width',
                                     'kind': 'integer',
                                     'defaultValue': None,
                                     'min': 1,
                                     'max': 4096,
                                     'allowsNull': True,
                                     'enabledValue': 4},
                                    {'key': 'dropout',
                                     'title': 'Drop probability',
                                     'kind': 'number',
                                     'defaultValue': 0,
                                     'min': 0,
                                     'max': 1},
                                    {'key': 'bias',
                                     'title': 'Use bias',
                                     'kind': 'boolean',
                                     'defaultValue': True},
                                    {'key': 'add_bias_kv',
                                     'title': 'Add learned key/value position',
                                     'kind': 'boolean',
                                     'defaultValue': False},
                                    {'key': 'add_zero_attn',
                                     'title': 'Add zero position',
                                     'kind': 'boolean',
                                     'defaultValue': False},
                                    {'key': 'return_weights',
                                     'title': 'Output attention weights',
                                     'kind': 'boolean',
                                     'defaultValue': True},
                                    {'key': 'average_weights',
                                     'title': 'Average across heads',
                                     'kind': 'boolean',
                                     'defaultValue': True},
                                    {'key': 'attention_mask_mode',
                                     'title': 'Attention mask',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'causal',
                                                  'title': 'Earlier positions only'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]},
                                    {'key': 'padding_mask_mode',
                                     'title': 'Padding mask',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]}]},
 'transformer': {'id': 'transformer',
                 'title': 'Transformer',
                 'category': 'Layers',
                 'inputs': ['source',
                            'target',
                            'src_mask',
                            'tgt_mask',
                            'memory_mask',
                            'src_key_padding_mask',
                            'tgt_key_padding_mask',
                            'memory_key_padding_mask'],
                 'outputs': ['out'],
                 'fields': [{'key': 'd_model',
                             'title': 'Model width',
                             'kind': 'integer',
                             'defaultValue': 4,
                             'min': 1,
                             'max': 512},
                            {'key': 'nhead',
                             'title': 'Heads',
                             'kind': 'integer',
                             'defaultValue': 2,
                             'min': 1,
                             'max': 32},
                            {'key': 'dim_feedforward',
                             'title': 'Feedforward width',
                             'kind': 'integer',
                             'defaultValue': 8,
                             'min': 1,
                             'max': 4096},
                            {'key': 'dropout',
                             'title': 'Drop probability',
                             'kind': 'number',
                             'defaultValue': 0,
                             'min': 0,
                             'max': 1},
                            {'key': 'bias',
                             'title': 'Use bias',
                             'kind': 'boolean',
                             'defaultValue': True},
                            {'key': 'activation',
                             'title': 'Activation',
                             'kind': 'choice',
                             'defaultValue': 'relu',
                             'choices': [{'value': 'relu', 'title': 'ReLU'},
                                         {'value': 'gelu', 'title': 'GELU'}]},
                            {'key': 'layer_norm_eps',
                             'title': 'Layer norm epsilon',
                             'kind': 'number',
                             'defaultValue': 1e-05,
                             'min': 1.401298464324817e-45,
                             'max': 3.4028234663852886e+38},
                            {'key': 'norm_first',
                             'title': 'Normalize before attention',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'num_encoder_layers',
                             'title': 'Encoder layers',
                             'kind': 'integer',
                             'defaultValue': 1,
                             'min': 1,
                             'max': 8},
                            {'key': 'num_decoder_layers',
                             'title': 'Decoder layers',
                             'kind': 'integer',
                             'defaultValue': 1,
                             'min': 1,
                             'max': 8},
                            {'key': 'source_mask_mode',
                             'title': 'Source attention',
                             'kind': 'choice',
                             'defaultValue': 'none',
                             'choices': [{'value': 'none', 'title': 'Allow all'},
                                         {'value': 'causal', 'title': 'Earlier positions only'},
                                         {'value': 'supplied', 'title': 'Use mask connection'}]},
                            {'key': 'target_mask_mode',
                             'title': 'Target attention',
                             'kind': 'choice',
                             'defaultValue': 'none',
                             'choices': [{'value': 'none', 'title': 'Allow all'},
                                         {'value': 'causal', 'title': 'Earlier positions only'},
                                         {'value': 'supplied', 'title': 'Use mask connection'}]},
                            {'key': 'memory_mask_mode',
                             'title': 'Memory attention',
                             'kind': 'choice',
                             'defaultValue': 'none',
                             'choices': [{'value': 'none', 'title': 'Allow all'},
                                         {'value': 'supplied', 'title': 'Use mask connection'}]},
                            {'key': 'source_padding_mode',
                             'title': 'Source padding',
                             'kind': 'choice',
                             'defaultValue': 'none',
                             'choices': [{'value': 'none', 'title': 'Allow all'},
                                         {'value': 'supplied', 'title': 'Use mask connection'}]},
                            {'key': 'target_padding_mode',
                             'title': 'Target padding',
                             'kind': 'choice',
                             'defaultValue': 'none',
                             'choices': [{'value': 'none', 'title': 'Allow all'},
                                         {'value': 'supplied', 'title': 'Use mask connection'}]},
                            {'key': 'memory_padding_mode',
                             'title': 'Memory padding',
                             'kind': 'choice',
                             'defaultValue': 'none',
                             'choices': [{'value': 'none', 'title': 'Allow all'},
                                         {'value': 'supplied', 'title': 'Use mask connection'}]}]},
 'transformer_encoder': {'id': 'transformer_encoder',
                         'title': 'Transformer encoder',
                         'category': 'Layers',
                         'inputs': ['in', 'mask', 'src_key_padding_mask'],
                         'outputs': ['out'],
                         'fields': [{'key': 'd_model',
                                     'title': 'Model width',
                                     'kind': 'integer',
                                     'defaultValue': 4,
                                     'min': 1,
                                     'max': 512},
                                    {'key': 'nhead',
                                     'title': 'Heads',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 32},
                                    {'key': 'dim_feedforward',
                                     'title': 'Feedforward width',
                                     'kind': 'integer',
                                     'defaultValue': 8,
                                     'min': 1,
                                     'max': 4096},
                                    {'key': 'dropout',
                                     'title': 'Drop probability',
                                     'kind': 'number',
                                     'defaultValue': 0,
                                     'min': 0,
                                     'max': 1},
                                    {'key': 'bias',
                                     'title': 'Use bias',
                                     'kind': 'boolean',
                                     'defaultValue': True},
                                    {'key': 'activation',
                                     'title': 'Activation',
                                     'kind': 'choice',
                                     'defaultValue': 'relu',
                                     'choices': [{'value': 'relu', 'title': 'ReLU'},
                                                 {'value': 'gelu', 'title': 'GELU'}]},
                                    {'key': 'layer_norm_eps',
                                     'title': 'Layer norm epsilon',
                                     'kind': 'number',
                                     'defaultValue': 1e-05,
                                     'min': 1.401298464324817e-45,
                                     'max': 3.4028234663852886e+38},
                                    {'key': 'norm_first',
                                     'title': 'Normalize before attention',
                                     'kind': 'boolean',
                                     'defaultValue': False},
                                    {'key': 'num_layers',
                                     'title': 'Layers',
                                     'kind': 'integer',
                                     'defaultValue': 1,
                                     'min': 1,
                                     'max': 8},
                                    {'key': 'final_norm',
                                     'title': 'Final normalization',
                                     'kind': 'boolean',
                                     'defaultValue': False},
                                    {'key': 'mask_mode',
                                     'title': 'Attention mask',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'causal',
                                                  'title': 'Earlier positions only'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]},
                                    {'key': 'padding_mask_mode',
                                     'title': 'Padding mask',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]}]},
 'transformer_decoder': {'id': 'transformer_decoder',
                         'title': 'Transformer decoder',
                         'category': 'Layers',
                         'inputs': ['target',
                                    'memory',
                                    'tgt_mask',
                                    'memory_mask',
                                    'tgt_key_padding_mask',
                                    'memory_key_padding_mask'],
                         'outputs': ['out'],
                         'fields': [{'key': 'd_model',
                                     'title': 'Model width',
                                     'kind': 'integer',
                                     'defaultValue': 4,
                                     'min': 1,
                                     'max': 512},
                                    {'key': 'nhead',
                                     'title': 'Heads',
                                     'kind': 'integer',
                                     'defaultValue': 2,
                                     'min': 1,
                                     'max': 32},
                                    {'key': 'dim_feedforward',
                                     'title': 'Feedforward width',
                                     'kind': 'integer',
                                     'defaultValue': 8,
                                     'min': 1,
                                     'max': 4096},
                                    {'key': 'dropout',
                                     'title': 'Drop probability',
                                     'kind': 'number',
                                     'defaultValue': 0,
                                     'min': 0,
                                     'max': 1},
                                    {'key': 'bias',
                                     'title': 'Use bias',
                                     'kind': 'boolean',
                                     'defaultValue': True},
                                    {'key': 'activation',
                                     'title': 'Activation',
                                     'kind': 'choice',
                                     'defaultValue': 'relu',
                                     'choices': [{'value': 'relu', 'title': 'ReLU'},
                                                 {'value': 'gelu', 'title': 'GELU'}]},
                                    {'key': 'layer_norm_eps',
                                     'title': 'Layer norm epsilon',
                                     'kind': 'number',
                                     'defaultValue': 1e-05,
                                     'min': 1.401298464324817e-45,
                                     'max': 3.4028234663852886e+38},
                                    {'key': 'norm_first',
                                     'title': 'Normalize before attention',
                                     'kind': 'boolean',
                                     'defaultValue': False},
                                    {'key': 'num_layers',
                                     'title': 'Layers',
                                     'kind': 'integer',
                                     'defaultValue': 1,
                                     'min': 1,
                                     'max': 8},
                                    {'key': 'final_norm',
                                     'title': 'Final normalization',
                                     'kind': 'boolean',
                                     'defaultValue': False},
                                    {'key': 'target_mask_mode',
                                     'title': 'Target attention',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'causal',
                                                  'title': 'Earlier positions only'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]},
                                    {'key': 'memory_mask_mode',
                                     'title': 'Memory attention',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]},
                                    {'key': 'target_padding_mode',
                                     'title': 'Target padding',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]},
                                    {'key': 'memory_padding_mode',
                                     'title': 'Memory padding',
                                     'kind': 'choice',
                                     'defaultValue': 'none',
                                     'choices': [{'value': 'none', 'title': 'Allow all'},
                                                 {'value': 'supplied',
                                                  'title': 'Use mask connection'}]}]},
 'scaled_dot_product_attention': {'id': 'scaled_dot_product_attention',
                                  'title': 'Scaled dot-product attention',
                                  'category': 'Layers',
                                  'inputs': ['query', 'key', 'value', 'mask'],
                                  'outputs': ['out'],
                                  'fields': [{'key': 'dropout',
                                              'title': 'Drop probability',
                                              'kind': 'number',
                                              'defaultValue': 0,
                                              'min': 0,
                                              'max': 1},
                                             {'key': 'scale',
                                              'title': 'Custom scale',
                                              'kind': 'number',
                                              'defaultValue': None,
                                              'min': 1.401298464324817e-45,
                                              'max': 3.4028234663852886e+38,
                                              'allowsNull': True,
                                              'enabledValue': 1},
                                             {'key': 'mask_mode',
                                              'title': 'Attention mask',
                                              'kind': 'choice',
                                              'defaultValue': 'none',
                                              'choices': [{'value': 'none', 'title': 'Allow all'},
                                                          {'value': 'causal',
                                                           'title': 'Earlier positions only'},
                                                          {'value': 'supplied',
                                                           'title': 'Use mask connection'}]}]},
 'attention_mask': {'id': 'attention_mask',
                    'title': 'Attention mask',
                    'category': 'Layers',
                    'inputs': ['query', 'key'],
                    'outputs': ['mask', 'padding'],
                    'fields': [{'key': 'mode',
                                'title': 'Mask',
                                'kind': 'choice',
                                'defaultValue': 'none',
                                'choices': [{'value': 'none', 'title': 'Allow all'},
                                            {'value': 'causal', 'title': 'Earlier positions only'},
                                            {'value': 'blocked', 'title': 'Blocked pairs'}]},
                               {'key': 'blocked_queries',
                                'title': 'Blocked query positions',
                                'kind': 'integerList',
                                'defaultValue': [],
                                'min': 0,
                                'max': 4095,
                                'minItems': 0,
                                'maxItems': 4096},
                               {'key': 'blocked_keys',
                                'title': 'Blocked key positions',
                                'kind': 'integerList',
                                'defaultValue': [],
                                'min': 0,
                                'max': 4095,
                                'minItems': 0,
                                'maxItems': 4096},
                               {'key': 'padding_keys',
                                'title': 'Padding positions',
                                'kind': 'integerList',
                                'defaultValue': [],
                                'min': 0,
                                'max': 4095,
                                'minItems': 0,
                                'maxItems': 4096}]}}

ATTENTION_CONTRACTS = {'multihead_attention': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                         'adapterID': 'kwip.runner.graph-v2.multihead_attention',
                         'graphSchemaVersions': [2],
                         'preservesBatch': True,
                         'graphDTypes': ['float32'],
                         'floatInputSampleRanks': [2],
                         'maximumParameters': 2000000,
                         'maximumTensorElements': 100000000,
                         'maximumScoreElements': 100000000,
                         'maximumMaskElements': 100000000,
                         'maximumWorkElements': 100000000,
                         'sequenceMinimum': 1,
                         'sequenceMaximum': 4096,
                         'modelWidthMaximum': 512,
                         'inputWidthMaximum': 4096,
                         'headMaximum': 32,
                         'layerMaximum': 8,
                         'maskDType': 'int64',
                         'maskConvention': 'oneBlockedZeroAllowed',
                         'maskBroadcast': 'perObservationAcrossHeadsOnly',
                         'statePersistence': 'none',
                         'parameterOrder': 'nativeModuleOrder',
                         'dropoutPolicy': 'trainingOnly',
                         'utilityMaximumItems': 4096,
                         'utilityCoordinateMaximum': 4095,
                         'multiheadWeights': 'nativeOptional',
                         'sdpaGroupedQuery': False,
                         'sdpaMPSDropoutExecution': 'notApplicable'},
 'transformer': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                 'adapterID': 'kwip.runner.graph-v2.transformer',
                 'graphSchemaVersions': [2],
                 'preservesBatch': True,
                 'graphDTypes': ['float32'],
                 'floatInputSampleRanks': [2],
                 'maximumParameters': 2000000,
                 'maximumTensorElements': 100000000,
                 'maximumScoreElements': 100000000,
                 'maximumMaskElements': 100000000,
                 'maximumWorkElements': 100000000,
                 'sequenceMinimum': 1,
                 'sequenceMaximum': 4096,
                 'modelWidthMaximum': 512,
                 'inputWidthMaximum': 4096,
                 'headMaximum': 32,
                 'layerMaximum': 8,
                 'maskDType': 'int64',
                 'maskConvention': 'oneBlockedZeroAllowed',
                 'maskBroadcast': 'perObservationAcrossHeadsOnly',
                 'statePersistence': 'none',
                 'parameterOrder': 'nativeModuleOrder',
                 'dropoutPolicy': 'trainingOnly',
                 'utilityMaximumItems': 4096,
                 'utilityCoordinateMaximum': 4095,
                 'multiheadWeights': 'notApplicable',
                 'sdpaGroupedQuery': False,
                 'sdpaMPSDropoutExecution': 'notApplicable'},
 'transformer_encoder': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                         'adapterID': 'kwip.runner.graph-v2.transformer_encoder',
                         'graphSchemaVersions': [2],
                         'preservesBatch': True,
                         'graphDTypes': ['float32'],
                         'floatInputSampleRanks': [2],
                         'maximumParameters': 2000000,
                         'maximumTensorElements': 100000000,
                         'maximumScoreElements': 100000000,
                         'maximumMaskElements': 100000000,
                         'maximumWorkElements': 100000000,
                         'sequenceMinimum': 1,
                         'sequenceMaximum': 4096,
                         'modelWidthMaximum': 512,
                         'inputWidthMaximum': 4096,
                         'headMaximum': 32,
                         'layerMaximum': 8,
                         'maskDType': 'int64',
                         'maskConvention': 'oneBlockedZeroAllowed',
                         'maskBroadcast': 'perObservationAcrossHeadsOnly',
                         'statePersistence': 'none',
                         'parameterOrder': 'nativeModuleOrder',
                         'dropoutPolicy': 'trainingOnly',
                         'utilityMaximumItems': 4096,
                         'utilityCoordinateMaximum': 4095,
                         'multiheadWeights': 'notApplicable',
                         'sdpaGroupedQuery': False,
                         'sdpaMPSDropoutExecution': 'notApplicable'},
 'transformer_decoder': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                         'adapterID': 'kwip.runner.graph-v2.transformer_decoder',
                         'graphSchemaVersions': [2],
                         'preservesBatch': True,
                         'graphDTypes': ['float32'],
                         'floatInputSampleRanks': [2],
                         'maximumParameters': 2000000,
                         'maximumTensorElements': 100000000,
                         'maximumScoreElements': 100000000,
                         'maximumMaskElements': 100000000,
                         'maximumWorkElements': 100000000,
                         'sequenceMinimum': 1,
                         'sequenceMaximum': 4096,
                         'modelWidthMaximum': 512,
                         'inputWidthMaximum': 4096,
                         'headMaximum': 32,
                         'layerMaximum': 8,
                         'maskDType': 'int64',
                         'maskConvention': 'oneBlockedZeroAllowed',
                         'maskBroadcast': 'perObservationAcrossHeadsOnly',
                         'statePersistence': 'none',
                         'parameterOrder': 'nativeModuleOrder',
                         'dropoutPolicy': 'trainingOnly',
                         'utilityMaximumItems': 4096,
                         'utilityCoordinateMaximum': 4095,
                         'multiheadWeights': 'notApplicable',
                         'sdpaGroupedQuery': False,
                         'sdpaMPSDropoutExecution': 'notApplicable'},
 'scaled_dot_product_attention': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                                  'adapterID': 'kwip.runner.graph-v2.scaled_dot_product_attention',
                                  'graphSchemaVersions': [2],
                                  'preservesBatch': True,
                                  'graphDTypes': ['float32'],
                                  'floatInputSampleRanks': [3],
                                  'maximumParameters': 2000000,
                                  'maximumTensorElements': 100000000,
                                  'maximumScoreElements': 100000000,
                                  'maximumMaskElements': 100000000,
                                  'maximumWorkElements': 100000000,
                                  'sequenceMinimum': 1,
                                  'sequenceMaximum': 4096,
                                  'modelWidthMaximum': 512,
                                  'inputWidthMaximum': 4096,
                                  'headMaximum': 32,
                                  'layerMaximum': 8,
                                  'maskDType': 'int64',
                                  'maskConvention': 'oneBlockedZeroAllowed',
                                  'maskBroadcast': 'perObservationAcrossHeadsOnly',
                                  'statePersistence': 'none',
                                  'parameterOrder': 'nativeModuleOrder',
                                  'dropoutPolicy': 'trainingOnly',
                                  'utilityMaximumItems': 4096,
                                  'utilityCoordinateMaximum': 4095,
                                  'multiheadWeights': 'notApplicable',
                                  'sdpaGroupedQuery': False,
                                  'sdpaMPSDropoutExecution': 'deviceLocalPublicOps'},
 'attention_mask': {'version': {'major': 1, 'minor': 0, 'patch': 0},
                    'adapterID': 'kwip.runner.graph-v2.attention_mask',
                    'graphSchemaVersions': [2],
                    'preservesBatch': True,
                    'graphDTypes': ['float32'],
                    'floatInputSampleRanks': [2, 3],
                    'maximumParameters': 2000000,
                    'maximumTensorElements': 100000000,
                    'maximumScoreElements': 100000000,
                    'maximumMaskElements': 100000000,
                    'maximumWorkElements': 100000000,
                    'sequenceMinimum': 1,
                    'sequenceMaximum': 4096,
                    'modelWidthMaximum': 512,
                    'inputWidthMaximum': 4096,
                    'headMaximum': 32,
                    'layerMaximum': 8,
                    'maskDType': 'int64',
                    'maskConvention': 'oneBlockedZeroAllowed',
                    'maskBroadcast': 'perObservationAcrossHeadsOnly',
                    'statePersistence': 'none',
                    'parameterOrder': 'nativeModuleOrder',
                    'dropoutPolicy': 'notApplicable',
                    'utilityMaximumItems': 4096,
                    'utilityCoordinateMaximum': 4095,
                    'multiheadWeights': 'notApplicable',
                    'sdpaGroupedQuery': False,
                    'sdpaMPSDropoutExecution': 'notApplicable'}}

ATTENTION_INPUT_CONDITIONS = {'multihead_attention': {'attn_mask': 'attention_mask_mode=supplied',
                         'key_padding_mask': 'padding_mask_mode=supplied'},
 'transformer': {'src_mask': 'source_mask_mode=supplied',
                 'tgt_mask': 'target_mask_mode=supplied',
                 'memory_mask': 'memory_mask_mode=supplied',
                 'src_key_padding_mask': 'source_padding_mode=supplied',
                 'tgt_key_padding_mask': 'target_padding_mode=supplied',
                 'memory_key_padding_mask': 'memory_padding_mode=supplied'},
 'transformer_encoder': {'mask': 'mask_mode=supplied', 'src_key_padding_mask': 'padding_mask_mode=supplied'},
 'transformer_decoder': {'tgt_mask': 'target_mask_mode=supplied',
                         'memory_mask': 'memory_mask_mode=supplied',
                         'tgt_key_padding_mask': 'target_padding_mode=supplied',
                         'memory_key_padding_mask': 'memory_padding_mode=supplied'},
 'scaled_dot_product_attention': {'mask': 'mask_mode=supplied'},
 'attention_mask': {}}

ATTENTION_DEFINITIONS["rotary_position"] = {'id': 'rotary_position', 'title': 'Rotary position', 'category': 'Layers', 'inputs': ['query', 'key'], 'outputs': ['query', 'key'], 'fields': [{'key': 'sequence_axis', 'title': 'Sequence axis', 'kind': 'integer', 'defaultValue': -2, 'min': -3, 'max': 2}, {'key': 'rotation', 'title': 'Rotation', 'kind': 'choice', 'defaultValue': 'full', 'choices': [{'value': 'full', 'title': 'Full feature width'}, {'value': 'custom', 'title': 'Custom width'}]}, {'key': 'rotary_dim', 'title': 'Rotated width', 'kind': 'integer', 'defaultValue': 2, 'min': 2, 'max': 4096}, {'key': 'pairing', 'title': 'Pairing', 'kind': 'choice', 'defaultValue': 'adjacent', 'choices': [{'value': 'adjacent', 'title': 'Adjacent pairs · Llama'}, {'value': 'halves', 'title': 'Split halves'}]}, {'key': 'base', 'title': 'Frequency base', 'kind': 'number', 'defaultValue': 10000, 'min': 1, 'max': 1000000000.0}, {'key': 'start_position', 'title': 'Starting position', 'kind': 'integer', 'defaultValue': 0, 'min': 0, 'max': 1000000}, {'key': 'position_scale', 'title': 'Position scale', 'kind': 'number', 'defaultValue': 1, 'min': 0.0001, 'max': 10000}]}
ATTENTION_CONTRACTS["rotary_position"] = {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.rotary_position', 'graphSchemaVersions': [2], 'preservesBatch': True, 'floatInputSampleRanks': [2, 3], 'graphDTypes': ['float32'], 'positionSource': 'inlineSequentialOnly', 'sequenceMaximum': 4096, 'featureMaximum': 4096, 'headMaximum': 32, 'positionMaximum': 1000000, 'effectivePositionMaximum': 1000000, 'frequencyBaseMinimumExclusive': 1.0, 'frequencyBaseMaximum': 1000000000.0, 'positionScaleMinimum': 0.0001, 'positionScaleMaximum': 10000.0, 'pairings': ['adjacent', 'halves'], 'partialRotation': 'evenPrefixPreserveSuffix', 'maximumTensorElements': 100000000, 'maximumWorkElements': 100000000, 'statePersistence': 'none', 'rotationArithmetic': 'realFloat32SinCos', 'sequentialPositions': 'configuredOffsetPerCompleteForward'}
ATTENTION_INPUT_CONDITIONS["rotary_position"] = {}

def validate_attention(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.attention' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported attention identity/version/backend')
    if not isinstance(value['components'], list) or [r['definition']['id'] for r in value['components']] != list(ATTENTION_DEFINITIONS):
        raise ValueError('Requires the ordered reviewed attention and rotary definitions')
    tasks = dict(zip(ATTENTION_DEFINITIONS, ['multihead-attention-connections','transformer-source-target','transformer-encoder-context','transformer-decoder-memory','scaled-attention-similarity','multihead-attention-connections']))
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation inputConditions outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        if json.dumps(row['definition'], sort_keys=True) != json.dumps(ATTENTION_DEFINITIONS[ident], sort_keys=True): raise ValueError('Unreviewed attention definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1: raise ValueError('Invalid attention publication/version')
        expected_outputs = {'weights':'return_weights=true'} if ident == 'multihead_attention' else {}
        if row['adapterID'] != 'kwip.runner.graph-v2.'+ident or row['shapeRuleID'] != 'kwip.shape.'+ident+'.v1' or row['inputConditions'] != ATTENTION_INPUT_CONDITIONS[ident] or row['outputConditions'] != expected_outputs: raise ValueError('Unreviewed attention adapter/conditional mask or weights ports')
        if json.dumps(row['runtimeContract'], sort_keys=True) != json.dumps(ATTENTION_CONTRACTS[ident], sort_keys=True): raise ValueError('Unreviewed attention runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'rose' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid attention presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published attention component requires teaching and example content')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != 'prepared-'+ident.replace('_','-')+'-classifier' or learning['lessonIDs'] != (['rotary_position'] if ident=='rotary_position' else ['assessment.'+tasks[ident]]) or learning['sourceResource'] != ('text-training-observations' if ident=='rotary_position' else 'workflow-observations'): raise ValueError('Invalid attention learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing attention lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing attention learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid attention date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv', 'Sources/KwipMLCore/AttentionLearningTask.swift', 'Sources/KwipMLCore/AttentionLessonReference.swift', 'Sources/KwipMLCore/AttentionLearningSubmission.swift', 'Sources/KwipMLCore/AttentionCapabilityBridge.swift']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe attention source reference')
            target = (root/reference).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file(): raise ValueError('Missing/unsafe attention source reference: '+reference)


LOSS_DEFINITIONS = {'loss_cross_entropy': {'id': 'loss_cross_entropy',
                        'title': 'CrossEntropyLoss',
                        'category': 'Training',
                        'inputs': [],
                        'outputs': ['out'],
                        'fields': [{'key': 'reduction',
                                    'title': 'Combine scores',
                                    'kind': 'choice',
                                    'defaultValue': 'mean',
                                    'choices': [{'value': 'none',
                                                 'title': 'Keep individual '
                                                          'scores'},
                                                {'value': 'mean',
                                                 'title': 'Mean'},
                                                {'value': 'sum',
                                                 'title': 'Sum'}]},
                                   {'key': 'target_mode',
                                    'title': 'Desired class format',
                                    'kind': 'choice',
                                    'defaultValue': 'class_indices',
                                    'choices': [{'value': 'class_indices',
                                                 'title': 'Chosen class '
                                                          'positions'},
                                                {'value': 'probabilities',
                                                 'title': 'Class '
                                                          'distributions'}]},
                                   {'key': 'class_axis',
                                    'title': 'Class direction',
                                    'kind': 'integer',
                                    'defaultValue': -1,
                                    'min': -7,
                                    'max': 7},
                                   {'key': 'class_weights',
                                    'title': 'Class importance',
                                    'kind': 'numberList',
                                    'defaultValue': [],
                                    'min': 0,
                                    'max': 3.4028234663852886e+38,
                                    'minItems': 0,
                                    'maxItems': 512},
                                   {'key': 'ignore_index',
                                    'title': 'Ignore class',
                                    'kind': 'integer',
                                    'defaultValue': -100,
                                    'min': -2147483648,
                                    'max': 2147483647},
                                   {'key': 'label_smoothing',
                                    'title': 'Smoothing share',
                                    'kind': 'number',
                                    'defaultValue': 0.0,
                                    'min': 0,
                                    'max': 1}]},
 'loss_bce_with_logits': {'id': 'loss_bce_with_logits',
                          'title': 'BCEWithLogitsLoss',
                          'category': 'Training',
                          'inputs': [],
                          'outputs': ['out'],
                          'fields': [{'key': 'reduction',
                                      'title': 'Combine scores',
                                      'kind': 'choice',
                                      'defaultValue': 'mean',
                                      'choices': [{'value': 'none',
                                                   'title': 'Keep individual '
                                                            'scores'},
                                                  {'value': 'mean',
                                                   'title': 'Mean'},
                                                  {'value': 'sum',
                                                   'title': 'Sum'}]},
                                     {'key': 'weight_mode',
                                      'title': 'Element importance source',
                                      'kind': 'choice',
                                      'defaultValue': 'none',
                                      'choices': [{'value': 'none',
                                                   'title': 'No extra '
                                                            'importance'},
                                                  {'value': 'configured',
                                                   'title': 'Configured '
                                                            'values'},
                                                  {'value': 'mapped',
                                                   'title': 'Data '
                                                            'connection'}]},
                                     {'key': 'weight_shape',
                                      'title': 'Element importance shape',
                                      'kind': 'integerList',
                                      'defaultValue': [],
                                      'min': 1,
                                      'max': 4096,
                                      'minItems': 0,
                                      'maxItems': 7},
                                     {'key': 'weight_values',
                                      'title': 'Element importance values',
                                      'kind': 'numberList',
                                      'defaultValue': [],
                                      'min': 0,
                                      'max': 3.4028234663852886e+38,
                                      'minItems': 0,
                                      'maxItems': 4096},
                                     {'key': 'pos_weight_mode',
                                      'title': 'Positive-label importance '
                                               'source',
                                      'kind': 'choice',
                                      'defaultValue': 'none',
                                      'choices': [{'value': 'none',
                                                   'title': 'No extra '
                                                            'importance'},
                                                  {'value': 'configured',
                                                   'title': 'Configured '
                                                            'values'},
                                                  {'value': 'mapped',
                                                   'title': 'Data '
                                                            'connection'}]},
                                     {'key': 'pos_weight_shape',
                                      'title': 'Positive-label importance '
                                               'shape',
                                      'kind': 'integerList',
                                      'defaultValue': [],
                                      'min': 1,
                                      'max': 4096,
                                      'minItems': 0,
                                      'maxItems': 7},
                                     {'key': 'pos_weight_values',
                                      'title': 'Positive-label importance '
                                               'values',
                                      'kind': 'numberList',
                                      'defaultValue': [],
                                      'min': 0,
                                      'max': 3.4028234663852886e+38,
                                      'minItems': 0,
                                      'maxItems': 4096}]},
 'loss_mse': {'id': 'loss_mse',
              'title': 'MSELoss',
              'category': 'Training',
              'inputs': [],
              'outputs': ['out'],
              'fields': [{'key': 'reduction',
                          'title': 'Combine scores',
                          'kind': 'choice',
                          'defaultValue': 'mean',
                          'choices': [{'value': 'none',
                                       'title': 'Keep individual scores'},
                                      {'value': 'mean', 'title': 'Mean'},
                                      {'value': 'sum', 'title': 'Sum'}]},
                         {'key': 'target_broadcast',
                          'title': 'Reuse target directions of size 1',
                          'kind': 'boolean',
                          'defaultValue': False}]},
 'loss_l1': {'id': 'loss_l1',
             'title': 'L1Loss',
             'category': 'Training',
             'inputs': [],
             'outputs': ['out'],
             'fields': [{'key': 'reduction',
                         'title': 'Combine scores',
                         'kind': 'choice',
                         'defaultValue': 'mean',
                         'choices': [{'value': 'none',
                                      'title': 'Keep individual scores'},
                                     {'value': 'mean', 'title': 'Mean'},
                                     {'value': 'sum', 'title': 'Sum'}]},
                        {'key': 'target_broadcast',
                         'title': 'Reuse target directions of size 1',
                         'kind': 'boolean',
                         'defaultValue': False}]},
 'loss_nll': {'id': 'loss_nll',
              'title': 'NLLLoss',
              'category': 'Training',
              'inputs': [],
              'outputs': ['out'],
              'fields': [{'key': 'reduction',
                          'title': 'Combine scores',
                          'kind': 'choice',
                          'defaultValue': 'mean',
                          'choices': [{'value': 'none',
                                       'title': 'Keep individual scores'},
                                      {'value': 'mean', 'title': 'Mean'},
                                      {'value': 'sum', 'title': 'Sum'}]},
                         {'key': 'class_axis',
                          'title': 'Class direction',
                          'kind': 'integer',
                          'defaultValue': -1,
                          'min': -7,
                          'max': 7},
                         {'key': 'class_weights',
                          'title': 'Class importance',
                          'kind': 'numberList',
                          'defaultValue': [],
                          'min': 0,
                          'max': 3.4028234663852886e+38,
                          'minItems': 0,
                          'maxItems': 512},
                         {'key': 'ignore_index',
                          'title': 'Ignore class',
                          'kind': 'integer',
                          'defaultValue': -100,
                          'min': -2147483648,
                          'max': 2147483647}]},
 'loss_kl_div': {'id': 'loss_kl_div',
                 'title': 'KLDivLoss',
                 'category': 'Training',
                 'inputs': [],
                 'outputs': ['out'],
                 'fields': [{'key': 'reduction',
                             'title': 'Combine scores',
                             'kind': 'choice',
                             'defaultValue': 'mean',
                             'choices': [{'value': 'none',
                                          'title': 'Keep individual scores'},
                                         {'value': 'mean', 'title': 'Mean'},
                                         {'value': 'sum', 'title': 'Sum'},
                                         {'value': 'batchmean',
                                          'title': 'Total per observation'}]},
                            {'key': 'log_target',
                             'title': 'Targets are log-probabilities',
                             'kind': 'boolean',
                             'defaultValue': False}]},
 'loss_ctc': {'id': 'loss_ctc',
              'title': 'CTCLoss',
              'category': 'Training',
              'inputs': [],
              'outputs': ['out'],
              'fields': [{'key': 'reduction',
                          'title': 'Combine scores',
                          'kind': 'choice',
                          'defaultValue': 'mean',
                          'choices': [{'value': 'none',
                                       'title': 'Keep individual scores'},
                                      {'value': 'mean', 'title': 'Mean'},
                                      {'value': 'sum', 'title': 'Sum'}]},
                         {'key': 'blank',
                          'title': 'Blank class',
                          'kind': 'integer',
                          'defaultValue': 0,
                          'min': 0,
                          'max': 511},
                         {'key': 'zero_infinity',
                          'title': 'Mask impossible alignments',
                          'kind': 'boolean',
                          'defaultValue': False},
                         {'key': 'target_layout',
                          'title': 'Target sequence layout',
                          'kind': 'choice',
                          'defaultValue': 'padded',
                          'choices': [{'value': 'padded',
                                       'title': 'Padded rows'},
                                      {'value': 'packed',
                                       'title': 'Packed active labels'}]}]},
 'loss_triplet_margin': {'id': 'loss_triplet_margin',
                         'title': 'TripletMarginLoss',
                         'category': 'Training',
                         'inputs': [],
                         'outputs': ['out'],
                         'fields': [{'key': 'reduction',
                                     'title': 'Combine scores',
                                     'kind': 'choice',
                                     'defaultValue': 'mean',
                                     'choices': [{'value': 'none',
                                                  'title': 'Keep individual '
                                                           'scores'},
                                                 {'value': 'mean',
                                                  'title': 'Mean'},
                                                 {'value': 'sum',
                                                  'title': 'Sum'}]},
                                    {'key': 'margin',
                                     'title': 'Required distance gap',
                                     'kind': 'number',
                                     'defaultValue': 1.0,
                                     'min': 1.401298464324817e-45,
                                     'max': 3.4028234663852886e+38},
                                    {'key': 'p',
                                     'title': 'Distance power',
                                     'kind': 'number',
                                     'defaultValue': 2.0,
                                     'min': 1.401298464324817e-45,
                                     'max': 16},
                                    {'key': 'eps',
                                     'title': 'Distance cushion',
                                     'kind': 'number',
                                     'defaultValue': 1e-06,
                                     'min': 0,
                                     'max': 3.4028234663852886e+38},
                                    {'key': 'swap',
                                     'title': 'Compare matching example with '
                                              'nonmatch too',
                                     'kind': 'boolean',
                                     'defaultValue': False}]}}

LOSS_CONTRACTS = {'loss_cross_entropy': {'version': '1.0.0',
                        'adapterID': 'kwip.runner.objective.loss_cross_entropy',
                        'recipeSchemaVersions': [1],
                        'configurationVersion': 1,
                        'predictionDTypes': ['float32'],
                        'devices': ['cpu', 'mps'],
                        'predictionRanks': [2, 8],
                        'sampleDimensionMaximum': 4096,
                        'classMinimum': 2,
                        'classMaximum': 512,
                        'classAxisMinimum': -7,
                        'classAxisMaximum': 7,
                        'excludesBatchClassAxis': True,
                        'reductions': ['none', 'mean', 'sum'],
                        'scalarizations': ['mean', 'sum'],
                        'targetRoles': ['target'],
                        'maximumTensorElements': 100000000,
                        'maximumWorkElements': 100000000,
                        'recipeMaximumBytes': 32768,
                        'fixedWeights': 'buffers',
                        'mappedWeights': 'preparedData',
                        'batchBroadcast': False},
 'loss_bce_with_logits': {'version': '1.0.0',
                          'adapterID': 'kwip.runner.objective.loss_bce_with_logits',
                          'recipeSchemaVersions': [1],
                          'configurationVersion': 1,
                          'predictionDTypes': ['float32'],
                          'devices': ['cpu', 'mps'],
                          'predictionRanks': [2, 8],
                          'sampleDimensionMaximum': 4096,
                          'classMinimum': 2,
                          'classMaximum': 512,
                          'classAxisMinimum': -7,
                          'classAxisMaximum': 7,
                          'excludesBatchClassAxis': True,
                          'reductions': ['none', 'mean', 'sum'],
                          'scalarizations': ['mean', 'sum'],
                          'targetRoles': ['target', 'weight', 'pos_weight'],
                          'maximumTensorElements': 100000000,
                          'maximumWorkElements': 100000000,
                          'recipeMaximumBytes': 32768,
                          'fixedWeights': 'buffers',
                          'mappedWeights': 'preparedData',
                          'batchBroadcast': False},
 'loss_mse': {'version': '1.0.0',
              'adapterID': 'kwip.runner.objective.loss_mse',
              'recipeSchemaVersions': [1],
              'configurationVersion': 1,
              'predictionDTypes': ['float32'],
              'devices': ['cpu', 'mps'],
              'predictionRanks': [2, 8],
              'sampleDimensionMaximum': 4096,
              'classMinimum': 2,
              'classMaximum': 512,
              'classAxisMinimum': -7,
              'classAxisMaximum': 7,
              'excludesBatchClassAxis': True,
              'reductions': ['none', 'mean', 'sum'],
              'scalarizations': ['mean', 'sum'],
              'targetRoles': ['target'],
              'maximumTensorElements': 100000000,
              'maximumWorkElements': 100000000,
              'recipeMaximumBytes': 32768,
              'fixedWeights': 'buffers',
              'mappedWeights': 'preparedData',
              'batchBroadcast': False},
 'loss_l1': {'version': '1.0.0',
             'adapterID': 'kwip.runner.objective.loss_l1',
             'recipeSchemaVersions': [1],
             'configurationVersion': 1,
             'predictionDTypes': ['float32'],
             'devices': ['cpu', 'mps'],
             'predictionRanks': [2, 8],
             'sampleDimensionMaximum': 4096,
             'classMinimum': 2,
             'classMaximum': 512,
             'classAxisMinimum': -7,
             'classAxisMaximum': 7,
             'excludesBatchClassAxis': True,
             'reductions': ['none', 'mean', 'sum'],
             'scalarizations': ['mean', 'sum'],
             'targetRoles': ['target'],
             'maximumTensorElements': 100000000,
             'maximumWorkElements': 100000000,
             'recipeMaximumBytes': 32768,
             'fixedWeights': 'buffers',
             'mappedWeights': 'preparedData',
             'batchBroadcast': False},
 'loss_nll': {'version': '1.0.0',
              'adapterID': 'kwip.runner.objective.loss_nll',
              'recipeSchemaVersions': [1],
              'configurationVersion': 1,
              'predictionDTypes': ['float32'],
              'devices': ['cpu', 'mps'],
              'predictionRanks': [2, 8],
              'sampleDimensionMaximum': 4096,
              'classMinimum': 2,
              'classMaximum': 512,
              'classAxisMinimum': -7,
              'classAxisMaximum': 7,
              'excludesBatchClassAxis': True,
              'reductions': ['none', 'mean', 'sum'],
              'scalarizations': ['mean', 'sum'],
              'targetRoles': ['target'],
              'maximumTensorElements': 100000000,
              'maximumWorkElements': 100000000,
              'recipeMaximumBytes': 32768,
              'fixedWeights': 'buffers',
              'mappedWeights': 'preparedData',
              'batchBroadcast': False},
 'loss_kl_div': {'version': '1.0.0',
                 'adapterID': 'kwip.runner.objective.loss_kl_div',
                 'recipeSchemaVersions': [1],
                 'configurationVersion': 1,
                 'predictionDTypes': ['float32'],
                 'devices': ['cpu', 'mps'],
                 'predictionRanks': [2, 8],
                 'sampleDimensionMaximum': 4096,
                 'classMinimum': 2,
                 'classMaximum': 512,
                 'classAxisMinimum': -7,
                 'classAxisMaximum': 7,
                 'excludesBatchClassAxis': True,
                 'reductions': ['none', 'mean', 'sum', 'batchmean'],
                 'scalarizations': ['mean', 'sum'],
                 'targetRoles': ['target'],
                 'maximumTensorElements': 100000000,
                 'maximumWorkElements': 100000000,
                 'recipeMaximumBytes': 32768,
                 'fixedWeights': 'buffers',
                 'mappedWeights': 'preparedData',
                 'batchBroadcast': False},
 'loss_ctc': {'version': '1.0.0',
              'adapterID': 'kwip.runner.objective.loss_ctc',
              'recipeSchemaVersions': [1],
              'configurationVersion': 1,
              'predictionDTypes': ['float32'],
              'devices': ['cpu', 'mps'],
              'predictionRanks': [2, 8],
              'sampleDimensionMaximum': 4096,
              'classMinimum': 2,
              'classMaximum': 512,
              'classAxisMinimum': -7,
              'classAxisMaximum': 7,
              'excludesBatchClassAxis': True,
              'reductions': ['none', 'mean', 'sum'],
              'scalarizations': ['mean', 'sum'],
              'targetRoles': ['target', 'input_lengths', 'target_lengths'],
              'maximumTensorElements': 100000000,
              'maximumWorkElements': 100000000,
              'recipeMaximumBytes': 32768,
              'fixedWeights': 'buffers',
              'mappedWeights': 'preparedData',
              'batchBroadcast': False},
 'loss_triplet_margin': {'version': '1.0.0',
                         'adapterID': 'kwip.runner.objective.loss_triplet_margin',
                         'recipeSchemaVersions': [1],
                         'configurationVersion': 1,
                         'predictionDTypes': ['float32'],
                         'devices': ['cpu', 'mps'],
                         'predictionRanks': [2, 8],
                         'sampleDimensionMaximum': 4096,
                         'classMinimum': 2,
                         'classMaximum': 512,
                         'classAxisMinimum': -7,
                         'classAxisMaximum': 7,
                         'excludesBatchClassAxis': True,
                         'reductions': ['none', 'mean', 'sum'],
                         'scalarizations': ['mean', 'sum'],
                         'targetRoles': [],
                         'maximumTensorElements': 100000000,
                         'maximumWorkElements': 100000000,
                         'recipeMaximumBytes': 32768,
                         'fixedWeights': 'buffers',
                         'mappedWeights': 'preparedData',
                         'batchBroadcast': False}}

LOSS_TASKS = dict(zip(LOSS_DEFINITIONS, ['cross-entropy-class-choice','bce-independent-labels','mse-error-size','l1-absolute-error','nll-log-probability-choice','kl-distribution-comparison','ctc-sequence-alignment','triplet-relative-distance']))
LOSS_EXAMPLE_SUFFIXES = dict(zip(LOSS_DEFINITIONS, ['CrossEntropyClassifier','BCEMultilabel','MSERegression','L1Regression','NLLClassifier','KLDistribution','CTCSequence','TripletDistance']))
LOSS_EXAMPLE_IDS = dict(zip(LOSS_DEFINITIONS, ['prepared-cross-entropy-classifier','prepared-bce-multilabel','prepared-mse-regression','prepared-l1-regression','prepared-nll-classifier','prepared-kl-distribution','prepared-ctc-sequence','prepared-triplet-distance']))

def validate_loss(value, root):
    exact(value, 'schemaVersion id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['id'] != 'kwip.losses' or type(value['version']) is not int or value['version'] != 1 or value['backend'] != 'pytorch':
        raise ValueError('Unsupported loss identity/version/backend')
    if not isinstance(value['components'], list) or [r['definition']['id'] for r in value['components']] != list(LOSS_DEFINITIONS): raise ValueError('Requires eight ordered objective definitions')
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        ident = row['definition']['id']
        # Strict JSON distinguishes Boolean controls from numeric defaults, including number-list values.
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(LOSS_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed objective definition')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1: raise ValueError('Invalid objective publication/version')
        if row['adapterID'] != 'kwip.runner.objective.'+ident or row['shapeRuleID'] != 'kwip.recipe.'+ident+'.v1' or row['outputConditions'] != {}: raise ValueError('Unreviewed objective recipe/ports')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(LOSS_CONTRACTS[ident],sort_keys=True): raise ValueError('Unreviewed objective runtime contract')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'green' or row['presentation']['icon'] != ident+'-v2' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Invalid objective presentation')
        exact(row['sources'], 'native worker tests evidence')
        references = list(row['sources'].values()) + ['Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png']
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published objective requires learning')
        if learning is not None:
            exact(learning,'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['exampleID'] != LOSS_EXAMPLE_IDS[ident] or learning['lessonIDs'] != ['assessment.'+LOSS_TASKS[ident]]: raise ValueError('Invalid objective learning identity')
            if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing objective lesson prose')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','sourceResource','title','subtitle','disclosureTitle','formula','buttonTitle','exampleSource')): raise ValueError('Missing objective learning metadata')
            from datetime import date
            try: date.fromisoformat(learning['updatedAt'])
            except ValueError: raise ValueError('Invalid objective date')
            references += [learning['exampleSource'], 'Sources/KwipMLApp/Resources/'+learning['sourceResource']+'.csv', 'Sources/KwipMLCore/LossLearningTask.swift', 'Sources/KwipMLCore/LossLessonReference.swift', 'Sources/KwipMLCore/LossLearningSubmission.swift', 'Sources/KwipMLCore/LossCapabilityBridge.swift']
        for reference in references:
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe objective source reference')
            path = (root/reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe objective source reference: '+reference)


INITIALIZATION_CONTRACT = {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.initialization', 'recipeSchemaVersions': [1], 'configurationVersions': [1], 'algorithms': ['xavier_uniform', 'kaiming_uniform', 'normal', 'zeros'], 'seedMinimum': 0, 'seedMaximum': 9223372036854775807, 'seedDerivation': 'kwip.initialization.owner-seed.v1', 'generatorPlacement': 'cpu-float32', 'devices': ['cpu', 'mps'], 'parameterDTypes': ['float32'], 'maximumRecipeBytes': 32768, 'maximumRules': 128, 'maximumModelParameters': 200000, 'ownerPolicy': 'expandedNodeRelativeNativeParameter', 'ownerOrder': 'nodeThenParameter', 'aliasPolicy': 'rejectDuplicateParameterIdentity', 'fanPolicy': 'nativeStoredDimensions', 'applicationTiming': 'freshAfterLazyBeforeOptimizer', 'restoredWeights': 'preserve', 'candidateCommit': 'validateAllBeforeCopy', 'preservesParameterIdentity': True, 'initializesBuffers': False, 'standaloneHelperMaximumBytes': 131072}
INITIALIZATION_DEFINITION = {'id':'initialization','title':'Starting settings','category':'Training','inputs':[],'outputs':['out'],'fields':[]}
INITIALIZATION_MODES = ['xavier-uniform','seeded-repeat','kaiming-uniform','normal','zeros']
INITIALIZATION_LESSONS = ['assessment.initialization-starting-values','assessment.initialization-starting-spread']
INITIALIZATION_EXAMPLES = ['prepared-initialization-starting-values','prepared-initialization-starting-spread']

def validate_initialization(value, root):
    exact(value,'schemaVersion kind id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or type(value['version']) is not int or value['version'] != 1 or value['kind'] != 'trainingControl' or value['id'] != 'kwip.initialization' or value['backend'] != 'pytorch':
        raise ValueError('Unsupported initialization control identity/version')
    if type(value['components']) is not list or len(value['components']) != 1: raise ValueError('Requires one initialization control')
    row=value['components'][0]
    exact(row,'definition published configurationVersion adapterID coverageID editorExtension modes presentation sources runtimeContract learning')
    if json.dumps(row['definition'],sort_keys=True) != json.dumps(INITIALIZATION_DEFINITION,sort_keys=True): raise ValueError('Initialization requires its control-only definition and typed owner editor')
    if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1 or row['adapterID'] != 'kwip.runner.initialization' or row['coverageID'] != 'pytorch.torch.nn.init.xavier_uniform_' or row['editorExtension'] != 'initializationOwnersV1' or row['modes'] != INITIALIZATION_MODES:
        raise ValueError('Unreviewed initialization recipe identity/editor/modes')
    if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(INITIALIZATION_CONTRACT,sort_keys=True): raise ValueError('Unreviewed initialization descriptor')
    exact(row['presentation'],'theme icon shortTitle purpose')
    if row['presentation']['theme'] != 'green' or row['presentation']['icon'] != 'initialization-v1' or any(type(x) is not str or not x.strip() for x in row['presentation'].values()): raise ValueError('Unreviewed initialization presentation')
    exact(row['sources'],'native worker tests evidence')
    refs=list(row['sources'].values())+['docs/design/initialization-icons-v1/initialization-v1.png','Sources/KwipMLApp/Resources/ComponentIcons/initialization-v1.png']
    learning=row['learning']
    if row['published'] and learning is None: raise ValueError('Published initialization requires actual lesson/example references')
    if learning is not None:
        exact(learning,'contentVersion updatedAt lessonIDs exampleIDs sources paragraphs')
        if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['lessonIDs'] != INITIALIZATION_LESSONS or learning['exampleIDs'] != INITIALIZATION_EXAMPLES: raise ValueError('Unreviewed initialization learning identity')
        from datetime import date
        try: date.fromisoformat(learning['updatedAt'])
        except (ValueError,TypeError): raise ValueError('Invalid initialization content date')
        if type(learning['sources']) is not list or not learning['sources'] or type(learning['paragraphs']) is not list or not learning['paragraphs'] or any(type(x) is not str or not x.strip() for x in learning['paragraphs']): raise ValueError('Missing initialization learning sources/copy')
        required={'Sources/KwipMLCore/InitializationLearningTask.swift','Sources/KwipMLCore/InitializationLessonReference.swift','Sources/KwipMLCore/InitializationLearningSubmission.swift','runner/learning_initialization.py'}
        if not required.issubset(learning['sources']): raise ValueError('Initialization teaching requires actual assessed task/reference/submission/worker')
        refs += learning['sources']
    for ref in refs:
        if type(ref) is not str or Path(ref).is_absolute(): raise ValueError('Unsafe initialization source reference')
        path=(root/ref).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe initialization source reference: '+ref)
    import hashlib
    source=(root/'docs/design/initialization-icons-v1/initialization-v1.png').read_bytes()
    if source != (root/'Sources/KwipMLApp/Resources/ComponentIcons/initialization-v1.png').read_bytes(): raise ValueError('Initialization registered artwork differs from original')

def render_initialization_swift(value):
    encoded=json.dumps(value['components'],ensure_ascii=False,indent=2)
    declarations='\n'.join('        '+json.dumps(row['definition']['id'])+': InitializationCapabilityBridge.declaration,' for row in value['components'] if row['published'])
    delimiter='#'
    while '"'+delimiter in encoded: delimiter += '#'
    template = '''// Generated by scripts/component_bundles.py. Edit components/initialization/bundle.json.
import Foundation

/// Typed training controls, not tensor-forward layers.
public enum BundledInitializationComponents {
    public struct Presentation: Decodable {
        public let theme: String
        public let icon: String
        public let shortTitle: String
        public let purpose: String
    }
    public struct Learning: Decodable {
        public let contentVersion: Int
        public let updatedAt: String
        public let lessonIDs: [String]
        public let exampleIDs: [String]
        public let sources: [String]
        public let paragraphs: [String]
    }
    public struct Entry: Decodable {
        public let definition: ComponentDefinition
        public let published: Bool
        public let configurationVersion: Int
        public let adapterID: String
        public let coverageID: String
        public let editorExtension: String
        public let modes: [String]
        public let runtimeContract: JSONValue
        public let presentation: Presentation
        public let learning: Learning?
    }
    public static let declarations: [String: (ComponentDefinition) throws -> CapabilityDescriptor] = DECLARATIONS
    public static let manifestDigest = "DIGEST"
    public static let entries: [String: Entry] = {
        let json = DELIMITER"""
ENCODED
"""DELIMITER
        do {
            let entries = try JSONDecoder().decode([Entry].self, from: Data(json.utf8))
            return Dictionary(uniqueKeysWithValues: entries.map { ($0.definition.id, $0) })
        } catch { preconditionFailure("Invalid generated initialization bundle: \\(error)") }
    }()
    public static func metadata(_ id: String) -> Entry? { entries[id] }
    public static func definition(_ id: String) -> ComponentDefinition {
        guard let entry = entries[id] else { preconditionFailure("Unknown starting-settings control") }
        return entry.definition
    }
}
'''
    return template.replace('DIGEST',manifest_digest(value)).replace('DELIMITER',delimiter).replace('ENCODED',encoded).replace('DECLARATIONS','[\n'+declarations+'\n    ]' if declarations else '[:]')

def render_initialization_capabilities(value):
    data={row['definition']['id']:row['runtimeContract'] for row in value['components']}
    return '# Generated by scripts/component_bundles.py. Edit components/initialization/bundle.json.\nimport json\nCAPABILITIES = json.loads('+repr(json.dumps(data,sort_keys=True,separators=(',',':')))+')\n'


# Artifact-backed metadata is intentionally separate from tensor component definitions.
LIFECYCLE_CONTRACT = {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.checkpoint-lifecycle', 'checkpointSchemaVersions': [1], 'modes': ['weights-save', 'continuation-save', 'weights-restore', 'continuation-restore', 'incompatible-artifact-refusal'], 'sourceTrust': 'ownedCompletedHistory', 'loadPolicy': 'weightsOnlyCPUMap', 'artifactLimits': {'manifest.json': 20000000, 'checkpoint.pt': 100000000, 'metrics.json': 4000000, 'rate-trace.json': 4000000}, 'requiredArtifacts': ['manifest.json', 'checkpoint.pt', 'metrics.json'], 'schedulerTracePolicy': 'requiredExactlyWhenParentHasScheduler', 'saveTiming': 'successfulRunCompletionAtEpochBoundary', 'exactResumeDevices': ['cpu'], 'fineTuneDevices': ['cpu', 'mps'], 'cpuResume': 'epochBoundaryExactCPU', 'mpsResume': 'weightsOnly', 'fineTune': 'restoreModelStateResetApplicableTrainingState', 'parentImmutable': True, 'externalImport': False, 'periodicSave': False}
LIFECYCLE_BINDINGS = [{'id': 'weights-save', 'surface': 'history', 'action': 'exportTrainedPackage'}, {'id': 'continuation-save', 'surface': 'history', 'action': 'retainCompletedCheckpoint'}, {'id': 'weights-restore', 'surface': 'train', 'action': 'selectParent', 'trainingMode': 'fineTune'}, {'id': 'continuation-restore', 'surface': 'train', 'action': 'selectParent', 'trainingMode': 'exactResume'}]
LIFECYCLE_LESSON_IDS = ['assessment.checkpoint-restore-weights', 'assessment.checkpoint-continue-training']

def validate_lifecycle(value, root):
    exact(value, 'schemaVersion kind id version backend capabilities')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or type(value['version']) is not int or value['version'] != 1 or value['kind'] != 'artifactLifecycle' or value['id'] != 'kwip.lifecycle' or value['backend'] != 'pytorch':
        raise ValueError('Unsupported artifact lifecycle identity/version')
    if not isinstance(value['capabilities'], list) or len(value['capabilities']) != 1:
        raise ValueError('Requires exactly one checkpoint lifecycle capability')
    row = value['capabilities'][0]
    exact(row, 'id published configurationVersion adapterID coverageID modes bindings presentation sources runtimeContract learning')
    if row['id'] != 'checkpoint_lifecycle' or row['adapterID'] != 'kwip.runner.checkpoint-lifecycle' or row['coverageID'] != 'pytorch.torch.save' or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1 or type(row['published']) is not bool:
        raise ValueError('Unreviewed lifecycle identity/publication')
    if json.dumps(row['modes']) != json.dumps(LIFECYCLE_CONTRACT['modes']) or json.dumps(row['bindings'], sort_keys=True) != json.dumps(LIFECYCLE_BINDINGS, sort_keys=True):
        raise ValueError('Unreviewed lifecycle modes/action bindings')
    # Canonical JSON distinguishes booleans from integers unlike Python dict equality.
    if json.dumps(row['runtimeContract'], sort_keys=True) != json.dumps(LIFECYCLE_CONTRACT, sort_keys=True):
        raise ValueError('Unreviewed lifecycle runtime descriptor')
    exact(row['presentation'], 'theme icon shortTitle purpose')
    if row['presentation']['theme'] != 'green' or row['presentation']['icon'] != 'docs/design/lifecycle-icons-v1/checkpoint-save-restore-v1.png' or any(not isinstance(x,str) or not x.strip() for x in row['presentation'].values()):
        raise ValueError('Unreviewed lifecycle presentation')
    exact(row['sources'], 'native worker tests evidence')
    refs = list(row['sources'].values()) + [row['presentation']['icon']]
    learning = row['learning']
    if row['published'] and learning is None:
        raise ValueError('Published lifecycle requires actual learning references')
    if learning is not None:
        exact(learning, 'contentVersion updatedAt lessonIDs exampleIDs sources paragraphs')
        if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1 or learning['lessonIDs'] != LIFECYCLE_LESSON_IDS or learning['exampleIDs'] != ['prepared-checkpoint-restore-weights', 'prepared-checkpoint-continue-training']:
            raise ValueError('Unreviewed lifecycle learning identity')
        from datetime import date
        try: date.fromisoformat(learning['updatedAt'])
        except (TypeError, ValueError): raise ValueError('Invalid lifecycle content date')
        if not isinstance(learning['paragraphs'],list) or not learning['paragraphs'] or any(not isinstance(x,str) or not x.strip() for x in learning['paragraphs']):
            raise ValueError('Missing lifecycle learning copy')
        if not isinstance(learning['sources'],list) or not learning['sources']:
            raise ValueError('Missing lifecycle learning references')
        refs += learning['sources']
    for reference in refs:
        if not isinstance(reference,str) or Path(reference).is_absolute():
            raise ValueError('Unsafe lifecycle source reference')
        path=(root/reference).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing/unsafe lifecycle source reference: '+reference)


def render_lifecycle_swift(value):
    encoded=json.dumps(value['capabilities'][0],ensure_ascii=False,indent=2)
    delimiter='#'
    while '"'+delimiter in encoded: delimiter += '#'
    return '// Generated by scripts/component_bundles.py. Edit components/lifecycle/bundle.json.\nimport Foundation\n\n/// Application-owned artifact actions. These are not canvas components or executable plugins.\npublic enum BundledLifecycleCapabilities {\n    public enum Surface: String, Decodable { case history, train }\n    public enum Action: String, Decodable { case exportTrainedPackage, retainCompletedCheckpoint, selectParent }\n    public struct Binding: Decodable {\n        public let id: String\n        public let surface: Surface\n        public let action: Action\n        public let trainingMode: WorkflowTrainingMode?\n    }\n    public struct Presentation: Decodable {\n        public let theme: String\n        public let icon: String\n        public let shortTitle: String\n        public let purpose: String\n    }\n    public struct Learning: Decodable {\n        public let contentVersion: Int\n        public let updatedAt: String\n        public let lessonIDs: [String]\n        public let exampleIDs: [String]\n        public let sources: [String]\n        public let paragraphs: [String]\n    }\n    public struct Entry: Decodable {\n        public let id: String\n        public let published: Bool\n        public let configurationVersion: Int\n        public let adapterID: String\n        public let coverageID: String\n        public let modes: [String]\n        public let bindings: [Binding]\n        public let presentation: Presentation\n        public let sources: [String: String]\n        public let runtimeContract: JSONValue\n        public let learning: Learning?\n    }\n    public static let manifestDigest = "DIGEST"\n    public static let entry: Entry = {\n        let json = DELIMITER"""\nENCODED\n"""DELIMITER\n        do { return try JSONDecoder().decode(Entry.self, from: Data(json.utf8)) }\n        catch { preconditionFailure("Invalid generated lifecycle metadata: \\(error)") }\n    }()\n}\n'.replace('DIGEST',manifest_digest(value)).replace('DELIMITER',delimiter).replace('ENCODED',encoded)


def render_lifecycle_capabilities(value):
    # Discovery is actual runtime support, independent of teaching publication status.
    data={row['id']:row['runtimeContract'] for row in value['capabilities']}
    return '# Generated by scripts/component_bundles.py. Edit components/lifecycle/bundle.json.\nimport json\nCAPABILITIES = json.loads('+repr(json.dumps(data,sort_keys=True,separators=(',',':')))+')\n'



# Two source-backed recipe controls; no tensor-forward adapter registration.
GRADIENT_CONTROL_DEFINITIONS = {'gradient_policy': {'id': 'gradient_policy',
                     'title': 'Gradient policy',
                     'category': 'Training',
                     'inputs': [],
                     'outputs': ['out'],
                     'fields': [{'key': 'accumulation_steps',
                                 'title': 'Small groups per update',
                                 'kind': 'integer',
                                 'defaultValue': 1,
                                 'min': 1,
                                 'max': 32},
                                {'key': 'gradient_scale',
                                 'title': 'Combine changes',
                                 'kind': 'choice',
                                 'defaultValue': 'mean',
                                 'choices': [{'title': 'Mean', 'value': 'mean'},
                                             {'title': 'Sum',
                                              'value': 'sum'}]}]},
 'trainability': {'id': 'trainability',
                  'title': 'Trainability',
                  'category': 'Training',
                  'inputs': [],
                  'outputs': ['out'],
                  'fields': []}}
GRADIENT_CONTROL_CONTRACTS = {'gradient_policy': {'version': '1.0.0',
                     'adapterID': 'kwip.runner.gradient_policy',
                     'recipeSchemaVersion': 1,
                     'configurationVersion': 1,
                     'devices': ['cpu', 'mps'],
                     'dtype': 'float32',
                     'derivativeOrder': 1,
                     'derivativeLayout': 'dense',
                     'accumulationMinimum': 1,
                     'accumulationMaximum': 32,
                     'requestMaximum': 32,
                     'uniqueTargetMaximum': 32,
                     'derivativeElementMaximum': 65536,
                     'upstreamElementMaximum': 65536,
                     'rowCountMaximum': 32,
                     'recipeByteMaximum': 32768,
                     'derivativeArtifactByteMaximum': 8388608,
                     'analysisPhase': 'finalSnapshotEvaluation',
                     'capturePermission': 'derivativeCapture.values',
                     'unavailableCauses': ['integerInput',
                                           'frozenParameter',
                                           'disconnected',
                                           'recordingDisabled',
                                           'sparseGradient'],
                     'nativeBackwardFailure': 'refuseWithoutFallback',
                     'reset': 'windowStart',
                     'partialWindow': 'flush',
                     'objectiveScaling': 'exactWindowNumeratorDenominator',
                     'oneCycleCadence': 'optimizerUpdate',
                     'continuation': 'cpuEpochBoundaryExactMPSWeightsOnly'},
 'trainability': {'version': '1.0.0',
                  'adapterID': 'kwip.runner.trainability',
                  'recipeSchemaVersion': 1,
                  'configurationVersion': 1,
                  'devices': ['cpu', 'mps'],
                  'dtype': 'float32',
                  'ownerMaximum': 128,
                  'recipeByteMaximum': 32768,
                  'ownerModes': ['inherit', 'trainable', 'frozen'],
                  'ownerResolution': 'materializedExpandedNodeNativeParameter',
                  'application': 'beforeOptimizerOwnership',
                  'moduleMode': 'existingTrainEvalTransitions',
                  'frozenInference': 'evaluationNoGrad',
                  'inputDerivativesThroughFrozenParameters': True,
                  'emptyTrainingOwnership': 'refuse',
                  'transition': 'freshOrFineTuneFreshOptimizer',
                  'exactResume': 'identicalPolicyAndOrderedOwnership'}}
GRADIENT_CONTROL_MODES = {'gradient_policy':['requested-gradients','reset-gradients','accumulated-gradients'],'trainability':['frozen-inference','explicit-trainability-transition','unavailable-gradient-capture']}
GRADIENT_CONTROL_LESSONS = {'gradient_policy':['assessment.gradient-selected-influence','assessment.gradient-combine-small-groups'],'trainability':['assessment.trainability-keep-settings','assessment.trainability-next-run']}

def validate_gradient_controls(value, root):
    exact(value,'schemaVersion kind id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or type(value['version']) is not int or value['version'] != 1 or value['kind'] != 'trainingControl' or value['id'] != 'kwip.gradient-controls' or value['backend'] != 'pytorch':
        raise ValueError('Unsupported gradient control identity/version/backend')
    if type(value['components']) is not list or len(value['components']) != 2 or [r.get('definition',{}).get('id') for r in value['components']] != list(GRADIENT_CONTROL_DEFINITIONS): raise ValueError('Requires two ordered gradient controls')
    provenance=json.loads((root/'docs/design/gradient-control-icons-v1/provenance-20261009.json').read_text())
    for row in value['components']:
        exact(row,'definition published configurationVersion adapterID coverageID editorExtension modes presentation sources runtimeContract learning')
        ident=row['definition']['id'];gradient=ident=='gradient_policy';icon='gradient-policy-v1' if gradient else 'trainability-v1'
        if json.dumps(row['definition'],sort_keys=True) != json.dumps(GRADIENT_CONTROL_DEFINITIONS[ident],sort_keys=True): raise ValueError('Unreviewed gradient control ports/fields')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion'] != 1 or row['adapterID'] != 'kwip.runner.'+ident or row['coverageID'] != ('pytorch.torch.autograd.grad' if gradient else 'pytorch.torch.no_grad') or row['editorExtension'] != ('gradientRequestsV1' if gradient else 'trainabilityOwnersV1') or row['modes'] != GRADIENT_CONTROL_MODES[ident]: raise ValueError('Unreviewed gradient control identity/editor/modes')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(GRADIENT_CONTROL_CONTRACTS[ident],sort_keys=True) or len(json.dumps(row['runtimeContract'],separators=(',',':')).encode())>4096: raise ValueError('Unreviewed gradient control runtime descriptor')
        exact(row['presentation'],'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'green' or row['presentation']['icon'] != icon or any(type(x) is not str or not x.strip() for x in row['presentation'].values()): raise ValueError('Unreviewed gradient control presentation')
        exact(row['sources'],'native worker tests evidence')
        refs=list(row['sources'].values())
        learning=row['learning']
        if row['published'] and learning is None: raise ValueError('Published gradient control requires actual teaching and examples')
        if learning is not None:
            exact(learning,'contentVersion updatedAt lessonIDs exampleIDs sources paragraphs')
            lessons=GRADIENT_CONTROL_LESSONS[ident]
            if type(learning['contentVersion']) is not int or learning['contentVersion']<1 or learning['lessonIDs'] != lessons or learning['exampleIDs'] != ['prepared-'+x.removeprefix('assessment.') for x in lessons]: raise ValueError('Unreviewed gradient teaching identities')
            from datetime import date
            try:date.fromisoformat(learning['updatedAt'])
            except (TypeError,ValueError):raise ValueError('Invalid gradient lesson date')
            if type(learning['sources']) is not list or not learning['sources'] or type(learning['paragraphs']) is not list or not learning['paragraphs'] or any(type(x) is not str or not x.strip() for x in learning['paragraphs']):raise ValueError('Missing gradient teaching references/prose')
            required={'Sources/KwipMLCore/GradientLearningTask.swift','Sources/KwipMLCore/GradientLearningSubmission.swift','Sources/KwipMLCore/GradientLessonReference.swift','Sources/KwipMLCore/GradientLearningExamples.swift','runner/learning_gradients.py'}
            if not required.issubset(learning['sources']):raise ValueError('Gradient teaching requires genuine assessed task/reference/submission/worker/examples')
            refs+=learning['sources']
        original='docs/design/gradient-control-icons-v1/'+icon+'.png';registered='Sources/KwipMLApp/Resources/ComponentIcons/'+icon+'.png'
        refs += [original,registered]
        for ref in refs:
            if type(ref) is not str or Path(ref).is_absolute():raise ValueError('Unsafe gradient source reference')
            path=(root/ref).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing/unsafe gradient source: '+ref)
        asset=next((x for x in provenance['assets'] if x['assetID']==icon),None)
        data=(root/original).read_bytes()
        if asset is None or asset['destination'] != original or hashlib.sha256(data).hexdigest()!=asset['sha256'] or not data.startswith(b'\x89PNG\r\n\x1a\n') or data!=(root/registered).read_bytes():raise ValueError('Gradient artwork differs from generated original/provenance')

def render_gradient_control_swift(value):
    return (render_initialization_swift(value)
        .replace('components/initialization/bundle.json','components/gradient_controls/bundle.json')
        .replace('BundledInitializationComponents','BundledGradientControlComponents')
        .replace('InitializationCapabilityBridge','GradientControlCapabilityBridge')
        .replace('initialization bundle','gradient control bundle')
        .replace('Unknown starting-settings control','Unknown gradient/training control'))

def render_gradient_control_capabilities(value):
    data={row['definition']['id']:row['runtimeContract'] for row in value['components']}
    return '# Generated by scripts/component_bundles.py. Edit components/gradient_controls/bundle.json.\nimport json\nCAPABILITIES = json.loads('+repr(json.dumps(data,sort_keys=True,separators=(',',':')))+')\n'



# Reviewed source recipes and inference artifact action; never tensor-forward adapters.
DATA_LOADING_DEFINITIONS = {'dataset_source': {'id': 'dataset_source',
                    'title': 'Dataset',
                    'category': 'Training',
                    'inputs': [],
                    'outputs': ['out'],
                    'fields': [{'key': 'kind',
                                'title': 'Observation access',
                                'kind': 'choice',
                                'defaultValue': 'indexed',
                                'choices': [{'value': 'indexed',
                                             'title': 'Indexed rows'},
                                            {'value': 'boundedIterable',
                                             'title': 'Finite stream'},
                                            {'value': 'tensor',
                                             'title': 'Tensor tuple'},
                                            {'value': 'subset',
                                             'title': 'Selected indices'}]},
                               {'key': 'indices',
                                'title': 'Selected original row positions',
                                'kind': 'integerList',
                                'defaultValue': [],
                                'min': 0,
                                'max': 99999,
                                'minItems': 0,
                                'maxItems': 4096},
                               {'key': 'max_rows',
                                'title': 'Maximum rows',
                                'kind': 'integer',
                                'defaultValue': 100000,
                                'min': 1,
                                'max': 100000}]},
 'data_loader': {'id': 'data_loader',
                 'title': 'DataLoader',
                 'category': 'Training',
                 'inputs': [],
                 'outputs': ['out'],
                 'fields': [{'key': 'batch_size',
                             'title': 'Batch size',
                             'kind': 'integer',
                             'defaultValue': 32,
                             'min': 1,
                             'max': 1024},
                            {'key': 'shuffle',
                             'title': 'Mix rows',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'drop_last',
                             'title': 'Drop final short group',
                             'kind': 'boolean',
                             'defaultValue': False},
                            {'key': 'seed',
                             'title': 'Mixing seed',
                             'kind': 'integer',
                             'defaultValue': 42,
                             'min': 0,
                             'max': 9007199254740991},
                            {'key': 'num_workers',
                             'title': 'Reader processes',
                             'kind': 'integer',
                             'defaultValue': 0,
                             'min': 0,
                             'max': 2},
                            {'key': 'prefetch_factor',
                             'title': 'Prefetch groups per reader',
                             'kind': 'integer',
                             'defaultValue': None,
                             'min': 1,
                             'max': 2,
                             'allowsNull': True},
                            {'key': 'timeout',
                             'title': 'Reader timeout (seconds)',
                             'kind': 'number',
                             'defaultValue': 0.0,
                             'min': 0,
                             'max': 30},
                            {'key': 'collation',
                             'title': 'Combine rows',
                             'kind': 'choice',
                             'defaultValue': 'auto',
                             'choices': [{'value': 'auto',
                                          'title': 'Automatic'},
                                         {'value': 'fixed_stack',
                                          'title': 'Fixed stack'},
                                         {'value': 'padded_sequence',
                                          'title': 'Padded sequence'},
                                         {'value': 'whole_bags',
                                          'title': 'Whole bags'}]}]}}
DATA_LOADING_CONTRACTS = {'dataset_source': {'version': '1.0.0',
                    'recipeSchemaVersion': 1,
                    'configurationVersion': 1,
                    'workflowSchemaVersion': 13,
                    'compilerContractVersion': 12,
                    'recipeByteMaximum': 32768,
                    'devices': ['cpu', 'mps'],
                    'adapterID': 'kwip.runner.dataset_source',
                    'modes': ['indexed-source',
                              'bounded-iterable-source',
                              'tensor-source',
                              'selected-index-view'],
                    'sourceKinds': ['indexed',
                                    'boundedIterable',
                                    'tensor',
                                    'subset'],
                    'nativeTypes': ['Dataset',
                                    'IterableDataset',
                                    'TensorDataset',
                                    'Subset'],
                    'partition': 'trainingOnly',
                    'rowMaximum': 100000,
                    'subset': 'uniqueOrderedOriginalTrainingIndices',
                    'iterable': 'restartableFinitePrefix',
                    'tensorSourceDomains': ['tensor'],
                    'inputDomains': ['tensor', 'bagPacket']},
 'data_loader': {'version': '1.0.0',
                 'recipeSchemaVersion': 1,
                 'configurationVersion': 1,
                 'workflowSchemaVersion': 13,
                 'compilerContractVersion': 12,
                 'recipeByteMaximum': 32768,
                 'devices': ['cpu', 'mps'],
                 'adapterID': 'kwip.runner.data_loader',
                 'modes': ['batch-order',
                           'shuffle-seed',
                           'short-final-batch',
                           'drop-last',
                           'collation',
                           'bounded-workers',
                           'continuation-order'],
                 'batchMinimum': 1,
                 'batchMaximum': 1024,
                 'workerMinimum': 0,
                 'workerMaximum': 2,
                 'prefetchMinimum': 1,
                 'prefetchMaximum': 2,
                 'timeoutMinimum': 0.0,
                 'timeoutMaximum': 30.0,
                 'workerConfiguration': 'zeroNullPrefetchZeroTimeoutOrPositivePrefetchPositiveTimeout',
                 'workingByteMaximum': 268435456,
                 'collations': ['auto',
                                'fixed_stack',
                                'padded_sequence',
                                'whole_bags'],
                 'multiprocessingContext': 'spawn',
                 'persistentWorkers': False,
                 'inOrder': True,
                 'pinMemory': False,
                 'orderVersion': 'kwip.data-loader.epoch-order.v1',
                 'gradientTransport': 'zeroWorkerOnly',
                 'partialBatch': 'retainOrDropGlobally',
                 'validation': 'orderedComplete',
                 'continuation': 'cpuEpochBoundaryExactMPSWeightsOnly',
                 'portableHelper': 'data_loading_recipe.py',
                 'portableHelperByteMaximum': 131072}}
DATA_LOADING_LESSONS = {'dataset_source': ['assessment.dataset-indexed-observations', 'assessment.dataset-aligned-tensors', 'assessment.dataset-selected-view', 'assessment.dataset-finite-stream'], 'data_loader': ['assessment.loader-complete-and-short-groups', 'assessment.loader-repeatable-mixing', 'assessment.loader-aligned-sequences', 'assessment.loader-continue-grouping']}
PROGRAM_EXPORT_CONTRACT = {'adapterID': 'kwip.runner.program_export',
 'budgets': {'archiveEntries': 512,
             'archiveUncompressedBytes': 100000000,
             'bufferElements': 2000000,
             'constantElements': 2000000,
             'graphNodes': 4096,
             'inputElements': 2000000,
             'manifestBytes': 2000000,
             'outputElements': 2000000,
             'parameterElements': 2000000,
             'programBytes': 100000000,
             'sampleAxis': 4096,
             'tensorRank': 8},
 'configurationVersion': 1,
 'controls': {'batch_mode': {'default': 'static',
                             'type': 'enum',
                             'values': ['static', 'bounded']},
              'device': {'default': 'cpu',
                         'type': 'enum',
                         'values': ['cpu', 'mps']},
              'example_batch_size': {'default': 2,
                                     'maximum': 128,
                                     'minimum': 1,
                                     'type': 'integer'},
              'maximum_batch_size': {'default': 2,
                                     'maximum': 4096,
                                     'minimum': 1,
                                     'type': 'integer'},
              'minimum_batch_size': {'default': 2,
                                     'maximum': 4096,
                                     'minimum': 1,
                                     'type': 'integer'},
              'prefer_deferred_runtime_asserts': {'default': False,
                                                  'type': 'boolean'},
              'strict': {'default': True, 'type': 'boolean'}},
 'devices': ['cpu', 'mps'],
 'execution': {'bagPacket': 'refused',
               'devicePolicy': 'explicit-no-fallback',
               'evaluationOnly': True,
               'fixedSampleAxes': True,
               'leadingBatch': ['static', 'bounded'],
               'singleTensorInputOutput': True},
 'modes': ['program-export', 'shape-constraints', 'program-reload-execution'],
 'nativeAPIs': {'export': 'torch.export.export(mod, args, kwargs=None, *, '
                          'dynamic_shapes=None, strict=False, '
                          'preserve_module_call_signature=(), '
                          'prefer_deferred_runtime_asserts_over_guards=False)',
                'load': 'torch.export.load(f, *, extra_files=None, '
                        'expected_opset_version=None)',
                'save': 'torch.export.save(ep, f, *, extra_files=None, '
                        'opset_version=None, pickle_protocol=2)'},
 'recipeSchemaVersion': 1,
 'schemaVersion': 1,
 'trust': 'trusted-app-generated-only',
 'version': '1.0.0'}
PROGRAM_EXPORT_FIELDS = [{'key': 'strict',
  'title': 'Strict capture',
  'kind': 'boolean',
  'defaultValue': True},
 {'key': 'batch_mode',
  'title': 'Batch sizes',
  'kind': 'choice',
  'defaultValue': 'static',
  'choices': [{'value': 'static', 'title': 'Static batch'},
              {'value': 'bounded', 'title': 'Bounded batch'}]},
 {'key': 'example_batch_size',
  'title': 'Example batch size',
  'kind': 'integer',
  'defaultValue': 2,
  'min': 1,
  'max': 128},
 {'key': 'minimum_batch_size',
  'title': 'Minimum batch size',
  'kind': 'integer',
  'defaultValue': 2,
  'min': 1,
  'max': 4096},
 {'key': 'maximum_batch_size',
  'title': 'Maximum batch size',
  'kind': 'integer',
  'defaultValue': 2,
  'min': 1,
  'max': 4096},
 {'key': 'device',
  'title': 'Capture device',
  'kind': 'choice',
  'defaultValue': 'cpu',
  'choices': [{'value': 'cpu', 'title': 'CPU'},
              {'value': 'mps', 'title': 'Apple GPU (MPS)'}]},
 {'key': 'prefer_deferred_runtime_asserts',
  'title': 'Deferred runtime checks',
  'kind': 'boolean',
  'defaultValue': False}]
PROGRAM_EXPORT_BINDINGS = [{'id': 'program-export', 'surface': 'history', 'action': 'exportProgramPackage'}]
PROGRAM_EXPORT_LESSONS = ['assessment.program-export-computation', 'assessment.program-export-size-range']


def _same_json(first, second):
    return json.dumps(first,sort_keys=True,separators=(',',':'),ensure_ascii=False)==json.dumps(second,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def _validate_data_program_metadata(row, root, lesson_ids, asset_id, required_sources):
    exact(row['presentation'],'theme icon shortTitle purpose')
    original='docs/design/data-loading-export-icons-v1/'+asset_id+'.png'
    if row['presentation']['theme']!='green' or row['presentation']['icon']!=original or any(type(v) is not str or not v.strip() for v in row['presentation'].values()):
        raise ValueError('Unreviewed data/program presentation')
    exact(row['sources'],'native worker tests evidence')
    refs=list(row['sources'].values())+[original]
    learning=row['learning']
    if row['published'] and learning is None:raise ValueError('Published source/action requires assessed teaching')
    if learning is not None:
        exact(learning,'contentVersion updatedAt lessonIDs exampleIDs sources paragraphs')
        if type(learning['contentVersion']) is not int or learning['contentVersion']<1 or learning['lessonIDs']!=lesson_ids or learning['exampleIDs']!=['prepared-'+i.removeprefix('assessment.') for i in lesson_ids]:
            raise ValueError('Unreviewed assessed teaching identities')
        from datetime import date
        try:date.fromisoformat(learning['updatedAt'])
        except (TypeError,ValueError):raise ValueError('Invalid content date')
        if type(learning['sources']) is not list or not learning['sources'] or any(type(v) is not str for v in learning['sources']) or not required_sources.issubset(learning['sources']):
            raise ValueError('Missing independent task/reference/submission/worker sources')
        if type(learning['paragraphs']) is not list or not learning['paragraphs'] or any(type(v) is not str or not v.strip() for v in learning['paragraphs']):
            raise ValueError('Missing approachable source/action teaching copy')
        refs+=learning['sources']
    for ref in refs:
        if type(ref) is not str or Path(ref).is_absolute():raise ValueError('Unsafe source/action reference')
        path=(root/ref).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing/unsafe source/action reference: '+ref)
    provenance=json.loads((root/'docs/design/data-loading-export-icons-v1/provenance-v1.json').read_text())
    matching=[a for a in provenance['assets'] if a.get('path')==original]
    raw=(root/original).read_bytes()
    if len(matching)!=1 or matching[0].get('generator')!='built-in image_gen' or matching[0].get('inspected') is not True or hashlib.sha256(raw).hexdigest()!=matching[0].get('sha256') or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Generated source/action artwork provenance differs')
    if row['published']:
        registered=root/'Sources/KwipMLApp/Resources/ComponentIcons'/(asset_id+'.png')
        if not registered.is_file() or registered.read_bytes()!=raw:
            raise ValueError('Published source/action registered artwork differs from generated original')


def validate_data_loading(value,root):
    exact(value,'schemaVersion kind id version backend components')
    if type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['version']) is not int or value['version']!=1 or value['id']!='kwip.training-data' or value['kind']!='trainingControl' or value['backend']!='pytorch':
        raise ValueError('Unreviewed training data recipe family')
    if type(value['components']) is not list or [r.get('definition',{}).get('id') for r in value['components']]!=list(DATA_LOADING_DEFINITIONS):raise ValueError('Requires ordered Dataset and DataLoader recipes')
    for row in value['components']:
        exact(row,'definition published configurationVersion adapterID coverageID editorExtension modes presentation sources runtimeContract learning')
        ident=row['definition']['id'];dataset=ident=='dataset_source'
        if not _same_json(row['definition'],DATA_LOADING_DEFINITIONS[ident]):raise ValueError('Unreviewed data recipe ports/controls')
        if type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion']!=1 or row['adapterID']!='kwip.runner.'+ident or row['coverageID']!='pytorch.torch.utils.data.'+('Dataset' if dataset else 'DataLoader') or row['editorExtension']!=('datasetRecipeV1' if dataset else 'dataLoaderRecipeV1') or row['modes']!=DATA_LOADING_CONTRACTS[ident]['modes']:
            raise ValueError('Unreviewed data recipe identity/modes')
        if not _same_json(row['runtimeContract'],DATA_LOADING_CONTRACTS[ident]):raise ValueError('Unreviewed data recipe runtime descriptor')
        _validate_data_program_metadata(row,root,DATA_LOADING_LESSONS[ident],'dataset-v1' if dataset else 'data-loader-v1',{'Sources/KwipMLCore/DataLoadingLearningTask.swift','Sources/KwipMLCore/DataLoadingLearningSubmission.swift','Sources/KwipMLCore/DataLoadingLessonReference.swift','runner/learning_data_loading.py'})


def validate_program_export(value,root):
    exact(value,'schemaVersion kind id version backend capabilities')
    if type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['version']) is not int or value['version']!=1 or value['id']!='kwip.program-export' or value['kind']!='artifactLifecycle' or value['backend']!='pytorch' or type(value['capabilities']) is not list or len(value['capabilities'])!=1:
        raise ValueError('Unreviewed executable program action family')
    row=value['capabilities'][0]
    exact(row,'id published configurationVersion adapterID coverageID modes bindings configurationFields presentation sources runtimeContract learning')
    if row['id']!='program_export' or type(row['published']) is not bool or type(row['configurationVersion']) is not int or row['configurationVersion']!=1 or row['adapterID']!='kwip.runner.program_export' or row['coverageID']!='pytorch.torch.export.export' or row['modes']!=PROGRAM_EXPORT_CONTRACT['modes']:
        raise ValueError('Unreviewed executable program identity/modes')
    if not _same_json(row['runtimeContract'],PROGRAM_EXPORT_CONTRACT) or not _same_json(row['configurationFields'],PROGRAM_EXPORT_FIELDS) or not _same_json(row['bindings'],PROGRAM_EXPORT_BINDINGS):
        raise ValueError('Unreviewed executable program descriptor/fields/action')
    _validate_data_program_metadata(row,root,PROGRAM_EXPORT_LESSONS,'program-export-v1',{'Sources/KwipMLCore/ProgramExportLearningTask.swift','Sources/KwipMLCore/ProgramExportLearningSubmission.swift','Sources/KwipMLCore/ProgramExportLessonReference.swift','runner/learning_program_export.py'})


def render_data_loading_swift(value):
    return (render_initialization_swift(value)
        .replace('components/initialization/bundle.json','components/training-data/bundle.json')
        .replace('BundledInitializationComponents','BundledDataLoadingComponents')
        .replace('InitializationCapabilityBridge','DataLoadingCapabilityBridge')
        .replace('initialization bundle','training data recipe bundle')
        .replace('Unknown starting-settings control','Unknown training data recipe'))


def render_program_export_swift(value):
    return (render_lifecycle_swift(value)
        .replace('components/lifecycle/bundle.json','components/program-export/bundle.json')
        .replace('BundledLifecycleCapabilities','BundledProgramExportCapabilities')
        .replace('case history, train','case history')
        .replace('case exportTrainedPackage, retainCompletedCheckpoint, selectParent','case exportProgramPackage')
        .replace('public let bindings: [Binding]','public let bindings: [Binding]\n        public let configurationFields: [ComponentField]')
        .replace('generated lifecycle metadata','generated executable program metadata'))


def render_data_program_capabilities(value,path):
    rows=value.get('components',value.get('capabilities',[]))
    data={(row['definition']['id'] if 'definition' in row else row['id']):row['runtimeContract'] for row in rows}
    return '# Generated by scripts/component_bundles.py. Edit '+path+'.\nimport json\nCAPABILITIES = json.loads('+repr(json.dumps(data,sort_keys=True,separators=(',',':')))+')\n'


def validate(value, root):
    if isinstance(value,dict) and value.get("id")=="kwip.training-data":
        return validate_data_loading(value,root)
    if isinstance(value,dict) and value.get("id")=="kwip.program-export":
        return validate_program_export(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.gradient-controls':
        return validate_gradient_controls(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.initialization':
        return validate_initialization(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.lifecycle':
        return validate_lifecycle(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.attention':
        return validate_attention(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.recurrent':
        return validate_recurrent(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.losses':
        return validate_loss(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.schedulers':
        return validate_scheduler(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.optimizers':
        return validate_optimizer(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.spatial-patches':
        return validate_spatial_patch(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.tensor-routing':
        return validate_tensor_routing(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.tensor-composition':
        return validate_tensor_composition(value,root)
    if isinstance(value,dict) and value.get("id") == "kwip.convolution":
        return validate_convolution(value,root)
    if isinstance(value,dict) and value.get("id") == "kwip.embeddings":
        return validate_embedding(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.tensor-math':
        return validate_tensor_math(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.tensor-layout':
        return validate_tensor_layout(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.layers':
        return validate_layer(value,root)
    if isinstance(value,dict) and value.get('id') == 'kwip.normalization':
        return validate_normalization(value,root)
    if isinstance(value, dict) and value.get('id') == 'kwip.activations':
        return validate_activation(value, root)
    exact(value, 'schemaVersion id version backend components')
    if value['schemaVersion'] != 1 or value['id'] != 'kwip.pooling' or type(value['version']) is not int or value['version'] < 1:
        raise ValueError('Unsupported bundle identity/version')
    if value['backend'] != 'pytorch': raise ValueError('Unsupported backend')
    if not isinstance(value['components'], list) or not value['components']: raise ValueError('Missing components')
    ids = []
    for row in value['components']:
        exact(row, 'definition published configurationVersion adapterID shapeRuleID presentation outputConditions sources runtimeContract learning')
        d = row['definition']
        exact(d, 'id title category inputs outputs fields')
        ids.append(d['id'])
        if d['id'] in NEW_POOLING_DEFINITIONS:
            validate_additional_pooling(row, root)
            continue
        if d['id'] not in ('max_pool1d', 'max_pool2d', 'max_pool3d', 'avg_pool1d'): raise ValueError('Unsupported pooling adapter')
        if row['adapterID'] != 'kwip.runner.graph-v2.' + d['id'] or row['shapeRuleID'] != 'kwip.shape.' + d['id'] + '.v1':
            raise ValueError('Unknown adapter reference')
        if type(row['published']) is not bool or row['configurationVersion'] != 2: raise ValueError('Invalid publication/configuration version')
        average = d['id'] == 'avg_pool1d'
        if d['inputs'] != ['in'] or d['outputs'] != (['out'] if average else ['out', 'indices']): raise ValueError('Unsupported ports')
        if not d['title'] or d['category'] != 'Pooling': raise ValueError('Invalid presentation identity')
        exact(row['presentation'], 'theme icon shortTitle purpose')
        if row['presentation']['theme'] != 'green': raise ValueError('Unknown theme')
        if row['presentation']['icon'] != d['id']+'-v2': raise ValueError('Unknown icon reference')
        if any(not isinstance(x, str) or not x.strip() for x in row['presentation'].values()): raise ValueError('Empty presentation')
        fields = d['fields']
        unique([f['key'] for f in fields], 'field')
        by_key = {f['key']: f for f in fields}
        if set(by_key) != ({'kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad'} if average else {'kernel_size', 'stride', 'padding', 'dilation', 'ceil_mode', 'return_indices'}): raise ValueError('Unsupported fields')
        for field in fields:
            if set(field) - {'key','title','kind','defaultValue','min','max','minItems','maxItems','allowsNull'}: raise ValueError('Unknown field key')
            default = field['defaultValue']; kind = field.get('kind', 'integer')
            valid = False
            if default is None:
                valid = field.get('allowsNull') is True
            elif kind == 'boolean': valid = type(default) is bool
            elif kind in ('integer', 'integerList'):
                def integer(x): return type(x) is int and field['min'] <= x <= field['max']
                if field['min'] > field['max']: raise ValueError('Invalid field bounds')
                valid = integer(default) if kind == 'integer' else isinstance(default,list) and field['minItems'] <= len(default) <= field['maxItems'] and all(integer(x) for x in default)
            if not valid: raise ValueError('Invalid field default')
        for port, condition in row['outputConditions'].items():
            if port not in d['outputs'] or condition not in by_key or by_key[condition].get('kind') != 'boolean': raise ValueError('Invalid output condition')
        if row['outputConditions'] != ({} if average else {'indices':'return_indices'}): raise ValueError('Unsupported output condition')
        learning = row['learning']
        if row['published'] and learning is None: raise ValueError('Published component requires learning/example')
        if learning is not None:
            exact(learning, 'contentVersion updatedAt exampleID sourceResource title subtitle disclosureTitle formula paragraphs buttonTitle exampleSource lessonIDs')
            if learning['lessonIDs'] != ['assessment.'+d['id'].replace('_','-')+'-windows']: raise ValueError('Unknown pooling assessment identity')
            if type(learning['contentVersion']) is not int or learning['contentVersion'] < 1: raise ValueError('Invalid learning content version')
            if not isinstance(learning['paragraphs'], list) or not learning['paragraphs'] or any(not isinstance(p,str) or not p.strip() for p in learning['paragraphs']): raise ValueError('Missing learning paragraphs')
            if any(not isinstance(learning[k],str) or not learning[k].strip() for k in ('updatedAt','exampleID','title','subtitle','formula','disclosureTitle','buttonTitle')): raise ValueError('Missing learning metadata')
            expected_id = 'prepared-' + d['id'].replace('_', '') + '-classifier'
            if learning['exampleID'] != expected_id: raise ValueError('Unknown example ID')
            for reference in (learning['exampleSource'], 'Sources/KwipMLApp/Resources/' + learning['sourceResource'] + '.csv'):
                path = (root / reference).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe learning reference')
        contract = row['runtimeContract']
        dimension = int(d['id'][-2])
        if contract.get('adapterID') != row['adapterID'] or contract.get('sampleRank') != dimension + 1 or contract.get('graphSchemaVersions') != [2] or contract.get('outputDTypes') != ({'out':'float32'} if average else {'out':'float32','indices':'int64'}):
            raise ValueError('Invalid runtime contract')
        reviewed = {'max_pool1d': {'definition': {'id': 'max_pool1d', 'title': 'MaxPool1d', 'category': 'Pooling', 'inputs': ['in'], 'outputs': ['out', 'indices'], 'fields': [{'key': 'kernel_size', 'title': 'Window length', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 4096}, {'key': 'stride', 'title': 'Stride · empty follows window', 'kind': 'integer', 'defaultValue': None, 'min': 1, 'max': 4096, 'allowsNull': True}, {'key': 'padding', 'title': 'Padding per side', 'kind': 'integer', 'defaultValue': 0, 'min': 0, 'max': 2048}, {'key': 'dilation', 'title': 'Window dilation', 'kind': 'integer', 'defaultValue': 1, 'min': 1, 'max': 4096}, {'key': 'ceil_mode', 'title': 'Include partial final window', 'kind': 'boolean', 'defaultValue': False}, {'key': 'return_indices', 'title': 'Expose winning indices', 'kind': 'boolean', 'defaultValue': False}]}, 'runtimeContract': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.max_pool1d', 'graphSchemaVersions': [2], 'sampleRank': 2, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'paddingMinimum': 0, 'paddingMaximum': 2048, 'paddingKernelDivisor': 2, 'nullStrideFollowsKernel': True, 'ceilModes': [False, True], 'returnIndicesModes': [False, True], 'outputDTypes': {'out': 'float32', 'indices': 'int64'}}}, 'max_pool2d': {'definition': {'id': 'max_pool2d', 'title': 'MaxPool2d', 'category': 'Pooling', 'inputs': ['in'], 'outputs': ['out', 'indices'], 'fields': [{'key': 'kernel_size', 'title': 'Window [height, width]', 'kind': 'integerList', 'defaultValue': [2, 2], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'stride', 'title': 'Stride [height, width] · empty follows window', 'kind': 'integerList', 'defaultValue': None, 'min': 1, 'max': 4096, 'allowsNull': True, 'minItems': 2, 'maxItems': 2}, {'key': 'padding', 'title': 'Padding [height, width]', 'kind': 'integerList', 'defaultValue': [0, 0], 'min': 0, 'max': 2048, 'minItems': 2, 'maxItems': 2}, {'key': 'dilation', 'title': 'Dilation [height, width]', 'kind': 'integerList', 'defaultValue': [1, 1], 'min': 1, 'max': 4096, 'minItems': 2, 'maxItems': 2}, {'key': 'ceil_mode', 'title': 'Include partial final window', 'kind': 'boolean', 'defaultValue': False}, {'key': 'return_indices', 'title': 'Expose winning indices', 'kind': 'boolean', 'defaultValue': False}]}, 'runtimeContract': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.max_pool2d', 'graphSchemaVersions': [2], 'sampleRank': 3, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'spatialMinimum': 1, 'spatialMaximum': 4096, 'spatialAxes': ['height', 'width'], 'indicesMeaning': 'flattenedSpatialPosition', 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'paddingMinimum': 0, 'paddingMaximum': 2048, 'paddingKernelDivisor': 2, 'nullStrideFollowsKernel': True, 'ceilModes': [False, True], 'returnIndicesModes': [False, True], 'outputDTypes': {'out': 'float32', 'indices': 'int64'}}}, 'max_pool3d': {'definition': {'id': 'max_pool3d', 'title': 'MaxPool3d', 'category': 'Pooling', 'inputs': ['in'], 'outputs': ['out', 'indices'], 'fields': [{'key': 'kernel_size', 'title': 'Window [depth, height, width]', 'kind': 'integerList', 'defaultValue': [2, 2, 2], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'stride', 'title': 'Stride [depth, height, width] · empty follows window', 'kind': 'integerList', 'defaultValue': None, 'min': 1, 'max': 4096, 'allowsNull': True, 'minItems': 3, 'maxItems': 3}, {'key': 'padding', 'title': 'Padding [depth, height, width]', 'kind': 'integerList', 'defaultValue': [0, 0, 0], 'min': 0, 'max': 2048, 'minItems': 3, 'maxItems': 3}, {'key': 'dilation', 'title': 'Dilation [depth, height, width]', 'kind': 'integerList', 'defaultValue': [1, 1, 1], 'min': 1, 'max': 4096, 'minItems': 3, 'maxItems': 3}, {'key': 'ceil_mode', 'title': 'Include partial final window', 'kind': 'boolean', 'defaultValue': False}, {'key': 'return_indices', 'title': 'Expose winning indices', 'kind': 'boolean', 'defaultValue': False}]}, 'runtimeContract': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.max_pool3d', 'graphSchemaVersions': [2], 'sampleRank': 4, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'spatialMinimum': 1, 'spatialMaximum': 4096, 'spatialAxes': ['depth', 'height', 'width'], 'indicesMeaning': 'flattenedSpatialPosition', 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'dilationMinimum': 1, 'dilationMaximum': 4096, 'paddingMinimum': 0, 'paddingMaximum': 2048, 'paddingKernelDivisor': 2, 'nullStrideFollowsKernel': True, 'ceilModes': [False, True], 'returnIndicesModes': [False, True], 'outputDTypes': {'out': 'float32', 'indices': 'int64'}}}, 'avg_pool1d': {'definition': {'id': 'avg_pool1d', 'title': 'AvgPool1d', 'category': 'Pooling', 'inputs': ['in'], 'outputs': ['out'], 'fields': [{'key': 'kernel_size', 'title': 'Window length', 'kind': 'integer', 'defaultValue': 2, 'min': 1, 'max': 4096}, {'key': 'stride', 'title': 'Stride · empty follows window', 'kind': 'integer', 'defaultValue': None, 'min': 1, 'max': 4096, 'allowsNull': True}, {'key': 'padding', 'title': 'Padding per side', 'kind': 'integer', 'defaultValue': 0, 'min': 0, 'max': 2048}, {'key': 'ceil_mode', 'title': 'Include partial final window', 'kind': 'boolean', 'defaultValue': False}, {'key': 'count_include_pad', 'title': 'Include padding in average divisor', 'kind': 'boolean', 'defaultValue': True}]}, 'runtimeContract': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.avg_pool1d', 'graphSchemaVersions': [2], 'sampleRank': 2, 'preservesBatch': True, 'graphDTypes': ['float32'], 'channelMinimum': 1, 'channelMaximum': 512, 'lengthMinimum': 1, 'lengthMaximum': 4096, 'kernelMinimum': 1, 'kernelMaximum': 4096, 'strideMinimum': 1, 'strideMaximum': 4096, 'paddingMinimum': 0, 'paddingMaximum': 2048, 'paddingKernelDivisor': 2, 'nullStrideFollowsKernel': True, 'ceilModes': [False, True], 'countIncludePadModes': [False, True], 'outputDTypes': {'out': 'float32'}}}}[d['id']]
        if json.dumps(d,sort_keys=True) != json.dumps(reviewed['definition'],sort_keys=True): raise ValueError('Unsupported pooling definition/default settings')
        if json.dumps(row['runtimeContract'],sort_keys=True) != json.dumps(reviewed['runtimeContract'],sort_keys=True): raise ValueError('Unsupported pooling runtime contract')
        icon = root / ('Sources/KwipMLApp/Resources/ComponentIcons/'+d['id']+'-v2.png')
        if not icon.is_file(): raise ValueError('Missing pooling artwork reference')
        exact(row['sources'], 'native worker tests evidence')
        for reference in row['sources'].values():
            if not isinstance(reference,str) or Path(reference).is_absolute(): raise ValueError('Unsafe source reference')
            path = (root / reference).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('Missing/unsafe source reference')
    unique(ids, 'component ID')
    if set(ids) != {'max_pool1d','max_pool2d','max_pool3d','avg_pool1d'} | set(NEW_POOLING_DEFINITIONS): raise ValueError('Pooling bundle requires eight reviewed operations')


def render_swift(value, family):
    if family == 'pooling':
        example_suffixes = {r['definition']['id']: ('AvgPool' if r['definition']['id'].startswith('avg_') else 'MaxPool') + r['definition']['id'][-2:] for r in value['components']}
        declaration_names = {r['definition']['id']: ('avgPool' if r['definition']['id'].startswith('avg_') else 'maxPool') + r['definition']['id'][-2:] + 'Declaration' for r in value['components']}
        example_suffixes.update({'adaptive_avg_pool1d':'AdaptiveAvgPool1d','adaptive_avg_pool2d':'AdaptiveAvgPool2d','adaptive_max_pool2d':'AdaptiveMaxPool2d'})
        declaration_names.update({'adaptive_avg_pool1d':'adaptiveAvgPool1dDeclaration','adaptive_avg_pool2d':'adaptiveAvgPool2dDeclaration','adaptive_max_pool2d':'adaptiveMaxPool2dDeclaration'})
    elif family == 'activations':
        example_suffixes = {'sigmoid': 'Sigmoid', 'tanh': 'Tanh', 'relu': 'ReLU', 'gelu': 'GELU', 'softmax': 'Softmax', 'silu': 'SiLU', 'leaky_relu': 'LeakyReLU', 'elu': 'ELU'}
        example_suffixes.update({'prelu':'PReLU','log_softmax':'LogSoftmax','dropout':'Dropout'})
        declaration_names = {'sigmoid': 'sigmoidDeclaration', 'tanh': 'tanhDeclaration', 'relu': 'reluDeclaration', 'gelu': 'geluDeclaration', 'softmax': 'softmaxDeclaration', 'silu': 'siluDeclaration', 'leaky_relu': 'leakyReLUDeclaration', 'elu': 'eluDeclaration'}
        declaration_names.update({'prelu':'preluDeclaration','log_softmax':'logSoftmaxDeclaration','dropout':'dropoutDeclaration'})
    elif family == 'embeddings':
        example_suffixes = {'embedding': 'Embedding', 'index_input': 'Embedding','bag_input':'EmbeddingBag','embedding_bag':'EmbeddingBag'}
        declaration_names = {ident: 'declaration' if ident in BAG_EMBEDDING_DEFINITIONS else 'embeddingDeclaration' for ident in example_suffixes}
    elif family == 'convolution':
        example_suffixes = {f'conv{d}d': f'Conv{d}d' for d in [1,2,3]}
        example_suffixes.update({f'conv_transpose{d}d': f'ConvTranspose{d}d' for d in [1,2,3]})
        declaration_names = {ident: (ident.replace('conv_transpose','convTranspose'))+'Declaration' for ident in example_suffixes}
    elif family == 'attention':
        example_suffixes = dict(zip(ATTENTION_DEFINITIONS, ['MultiheadAttention','Transformer','TransformerEncoder','TransformerDecoder','ScaledDotProductAttention','AttentionMask','RotaryPosition']))
        declaration_names = dict(zip(ATTENTION_DEFINITIONS, ['multiheadAttentionDeclaration','transformerDeclaration','transformerEncoderDeclaration','transformerDecoderDeclaration','scaledDotProductAttentionDeclaration','attentionMaskDeclaration','rotaryPositionDeclaration']))
    elif family == 'recurrent':
        example_suffixes = dict(zip(RECURRENT_DEFINITIONS, ['RNN','LSTM','GRU','RNNCell','LSTMCell','GRUCell']))
        declaration_names = dict(zip(RECURRENT_DEFINITIONS, ['rnnDeclaration','lstmDeclaration','gruDeclaration','rnnCellDeclaration','lstmCellDeclaration','gruCellDeclaration']))
    elif family == 'losses':
        example_suffixes = LOSS_EXAMPLE_SUFFIXES
        declaration_names = dict(zip(LOSS_DEFINITIONS, ['crossEntropyDeclaration','bceWithLogitsDeclaration','mseDeclaration','l1Declaration','nllDeclaration','klDivDeclaration','ctcDeclaration','tripletMarginDeclaration']))
    elif family == 'schedulers':
        example_suffixes = dict(zip(SCHEDULER_DEFINITIONS, ['StepLR','CosineAnnealingLR','ReduceLROnPlateau','OneCycleLR','LinearLR']))
        declaration_names = dict(zip(SCHEDULER_DEFINITIONS, ['stepLRDeclaration','cosineAnnealingLRDeclaration','reduceOnPlateauDeclaration','oneCycleLRDeclaration','linearLRDeclaration']))
    elif family == 'optimizers':
        example_suffixes = dict(zip(OPTIMIZER_DEFINITIONS, ['SGD','Adam','AdamW','RMSprop','Adagrad']))
        declaration_names = {ident: ident.removeprefix('optimizer_')+'Declaration' for ident in example_suffixes}
    elif family == 'spatial-patches':
        example_suffixes = {ident: ident.capitalize() for ident in ['unfold','fold']}
        declaration_names = {ident: ident+'Declaration' for ident in example_suffixes}
    elif family == 'tensor-routing':
        example_suffixes = {ident: ident.capitalize() for ident in ['split','chunk','gather','einsum']}
        declaration_names = {ident: ident+'Declaration' for ident in example_suffixes}
    elif family == 'tensor-composition':
        example_suffixes = {ident: ident.capitalize() for ident in ['concat','add','stack','permute']}
        declaration_names = {ident: ident+'Declaration' for ident in example_suffixes}
    elif family == 'tensor-math':
        example_suffixes = {'matmul': 'Matrix'}
        declaration_names = {'matmul': 'matrixDeclaration'}
    elif family == 'tensor-layout':
        example_suffixes = {ident: 'TensorLayout' for ident in ['reshape','transpose','flatten']}
        declaration_names = {ident: ident+'Declaration' for ident in example_suffixes}
    elif family == 'layers':
        example_suffixes = {'linear': 'Parallel', 'bilinear': 'Bilinear', 'lazy_linear': 'LazyLinear'}
        declaration_names = {'linear': 'linearDeclaration', 'bilinear': 'bilinearDeclaration', 'lazy_linear': 'lazyLinearDeclaration'}
    elif family == 'normalization':
        example_suffixes = {'layer_norm': 'LayerNorm', 'rms_norm': 'RMSNorm', **{i:STATEFUL_NORMALIZATION_DEFINITIONS[i]['title'] for i in STATEFUL_NORMALIZATION_DEFINITIONS}}
        declaration_names = {'layer_norm': 'layerNormDeclaration', 'rms_norm': 'rmsNormDeclaration', **{i:STATEFUL_NORMALIZATION_DEFINITIONS[i]['title'][0].lower()+STATEFUL_NORMALIZATION_DEFINITIONS[i]['title'][1:]+'Declaration' for i in STATEFUL_NORMALIZATION_DEFINITIONS}}
    else:
        raise ValueError('Unsupported bundle renderer')
    encoded = json.dumps(value['components'], ensure_ascii=False, indent=2)
    # Escape Swift raw-string terminators even in owner-authored documentation.
    delimiter = '#'
    while '"' + delimiter in encoded: delimiter += '#'
    swift = '''// Generated by scripts/component_bundles.py. Edit components/pooling/bundle.json.
import Foundation

public enum BundledPoolingComponents {
    public struct Presentation: Decodable {
        public let theme: String
        public let icon: String
        public let shortTitle: String
        public let purpose: String
    }
    public struct Learning: Decodable, BundledComponentLearningContent {
        public let contentVersion: Int
        public let updatedAt: String
        public let exampleID: String
        public let sourceResource: String
        public let title: String
        public let subtitle: String
        public let disclosureTitle: String
        public let formula: String
        public let paragraphs: [String]
        public let buttonTitle: String
    }
    public struct Entry: Decodable {
        public let learning: Learning?
        public let definition: ComponentDefinition
        public let runtimeContract: JSONValue
        public let published: Bool
        public let presentation: Presentation
        public let outputConditions: [String: String]
    }
    static let examples: [String: (String) throws -> WorkflowDocument] = [
EXAMPLES
    ]
    public static func example(_ id: String, device: String) throws -> WorkflowDocument {
        guard entries[id]?.published == true, let factory = examples[id] else {
            throw GraphConfigurationError("This component has no available bundled example.")
        }
        return try factory(device)
    }
    static let declarations: [String: (ComponentDefinition) throws -> CapabilityDescriptor] = [
DECLARATIONS
    ]
    public static let manifestDigest = "DIGEST"
    public static let entries: [String: Entry] = {
        let json = DELIMITER"""
ENCODED
"""DELIMITER
        do {
            let rows = try JSONDecoder().decode([Entry].self, from: Data(json.utf8))
            return Dictionary(uniqueKeysWithValues: rows.map { ($0.definition.id, $0) })
        } catch { preconditionFailure("Invalid generated pooling bundle: \\(error)") }
    }()
    public static func definition(_ id: String) -> ComponentDefinition {
        guard let entry = entries[id] else { preconditionFailure("Unknown bundled pooling component: \\(id)") }
        return entry.definition
    }
}
'''.replace('EXAMPLES', ',\n'.join('        \"'+r['definition']['id']+'\": { try WorkflowExamples.prepared'+example_suffixes[r['definition']['id']]+('' if family == 'losses' else 'Classifier')+'(device: $0) }' for r in value['components'] if r['learning'] is not None)).replace('DECLARATIONS', ',\n'.join('        \"'+r['definition']['id']+'\": '+('EmbeddingBagCapabilityBridge.' if family == 'embeddings' and r['definition']['id'] in BAG_EMBEDDING_DEFINITIONS else 'LossCapabilityBridge.' if family == 'losses' else 'AttentionCapabilityBridge.' if family == 'attention' else 'RecurrentCapabilityBridge.' if family == 'recurrent' else 'SchedulerCapabilityBridge.' if family == 'schedulers' else 'OptimizerCapabilityBridge.' if family == 'optimizers' else 'StatefulNormalizationCapabilityBridge.' if family == 'normalization' and r['definition']['id'] in STATEFUL_NORMALIZATION_DEFINITIONS else 'SpatialPatchCapabilityBridge.' if family == 'spatial-patches' else 'TensorRoutingCapabilityBridge.' if family == 'tensor-routing' else 'RemainingActivationCapabilityBridge.' if family == 'activations' and r['definition']['id'] in REMAINING_ACTIVATION_DEFINITIONS else 'BuiltInCapabilityBridge.')+declaration_names[r['definition']['id']] for r in value['components'] if family == 'pooling' or r['published'])).replace('DIGEST', manifest_digest(value)).replace('DELIMITER',delimiter).replace('ENCODED',encoded)
    if family == 'attention':
        swift = swift.replace('BundledPoolingComponents','BundledAttentionComponents').replace('components/pooling/bundle.json','components/attention/bundle.json').replace('pooling bundle','attention bundle').replace('pooling component','attention component')
        swift = swift.replace('        public let outputConditions:', '        public let inputConditions: [String: String]\n        public let outputConditions:')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
    if family == 'recurrent':
        swift = swift.replace('BundledPoolingComponents','BundledRecurrentComponents').replace('components/pooling/bundle.json','components/recurrent/bundle.json').replace('pooling bundle','recurrent bundle').replace('pooling component','recurrent component')
        swift = swift.replace('        public let outputConditions:', '        public let inputConditions: [String: String]\n        public let outputConditions:')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
    if family == 'pooling':
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
    if family == 'activations':
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]?\n        public let exampleID: String')
        # Unpublished metadata can be generated for review without admitting it.
        swift = swift.replace('BundledPoolingComponents', 'BundledActivationComponents').replace('components/pooling/bundle.json', 'components/activations/bundle.json').replace('pooling bundle', 'activation bundle').replace('pooling component', 'activation component')
    if family == 'normalization':
        swift = swift.replace('BundledPoolingComponents','BundledNormalizationComponents').replace('components/pooling/bundle.json','components/normalization/bundle.json').replace('pooling bundle','normalization bundle').replace('pooling component','normalization component')
    if family == 'layers':
        swift = swift.replace('BundledPoolingComponents','BundledLayerComponents').replace('components/pooling/bundle.json','components/layers/bundle.json').replace('pooling bundle','layer bundle').replace('pooling component','layer component')
    if family == 'tensor-layout':
        swift = swift.replace('BundledPoolingComponents','BundledTensorLayoutComponents').replace('components/pooling/bundle.json','components/tensor-layout/bundle.json').replace('pooling bundle','tensor-layout bundle').replace('pooling component','tensor-layout component')
    if family == 'tensor-math':
        swift = swift.replace('BundledPoolingComponents','BundledTensorMathComponents').replace('components/pooling/bundle.json','components/tensor-math/bundle.json').replace('pooling bundle','tensor-math bundle').replace('pooling component','tensor-math component')
    if family == 'embeddings':
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]?\n        public let exampleID: String')
        swift = swift.replace('BundledPoolingComponents','BundledEmbeddingComponents').replace('components/pooling/bundle.json','components/embeddings/bundle.json').replace('pooling bundle','embedding bundle').replace('pooling component','embedding component')
    if family == 'convolution':
        swift = swift.replace('BundledPoolingComponents','BundledConvolutionComponents').replace('components/pooling/bundle.json','components/convolution/bundle.json').replace('pooling bundle','convolution bundle').replace('pooling component','convolution component')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
    if family == 'spatial-patches':
        swift = swift.replace('BundledPoolingComponents','BundledSpatialPatchComponents').replace('components/pooling/bundle.json','components/spatial-patches/bundle.json').replace('pooling bundle','spatial-patches bundle').replace('pooling component','spatial-patches component')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
    if family == 'tensor-routing':
        swift = swift.replace('BundledPoolingComponents','BundledTensorRoutingComponents').replace('components/pooling/bundle.json','components/tensor-routing/bundle.json').replace('pooling bundle','tensor-routing bundle').replace('pooling component','tensor-routing component')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
    if family == 'tensor-composition':
        swift = swift.replace('BundledPoolingComponents','BundledTensorCompositionComponents').replace('components/pooling/bundle.json','components/tensor-composition/bundle.json').replace('pooling bundle','tensor-composition bundle').replace('pooling component','tensor-composition component')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
    if family == 'optimizers':
        swift = swift.replace('BundledPoolingComponents','BundledOptimizerComponents').replace('components/pooling/bundle.json','components/optimizers/bundle.json').replace('pooling bundle','optimizer bundle').replace('pooling component','optimizer component')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
    if family == 'schedulers':
        swift = swift.replace('BundledPoolingComponents','BundledSchedulerComponents').replace('components/pooling/bundle.json','components/schedulers/bundle.json').replace('pooling bundle','scheduler bundle').replace('pooling component','scheduler component')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
    if family == 'losses':
        swift = swift.replace('BundledPoolingComponents','BundledLossComponents').replace('components/pooling/bundle.json','components/losses/bundle.json').replace('pooling bundle','objective bundle').replace('pooling component','objective component')
        swift = swift.replace('        public let exampleID: String', '        public let lessonIDs: [String]\n        public let exampleID: String')
        swift = swift.replace('    public static func definition(', '    public static func metadata(_ id: String) -> Entry? { entries[id] }\n\n    public static func definition(')
        # Empty unpublished dictionaries need an explicit dictionary literal in Swift.
        swift = swift.replace(' = [\n\n    ]', ' = [:]')
    return swift


def render_loss_adapters(value):
    published = [r for r in value['components'] if r['published']]
    ids = [r['definition']['id'] for r in published]
    return ('# Generated by scripts/component_bundles.py. Edit components/losses/bundle.json.\n'
            'from functools import partial\nfrom loss_configuration import canonical_configuration\n'
            'IDS = '+repr(tuple(ids))+'\nCAPABILITIES = '+repr({r['definition']['id']:r['runtimeContract'] for r in published})+'\n'
            'NORMALIZERS = {ident: partial(canonical_configuration, ident) for ident in IDS}\n')


def render_optimizer_adapters(value):
    published = [r for r in value['components'] if r['published']]
    ids = [r['definition']['id'] for r in published]
    return ('# Generated by scripts/component_bundles.py. Edit components/optimizers/bundle.json.\n'
            'from functools import partial\nfrom optimizer_configuration import canonical_configuration\n'
            'IDS = '+repr(tuple(ids))+'\nCAPABILITIES = '+repr({r['definition']['id']:r['runtimeContract'] for r in published})+'\n'
            'NORMALIZERS = {ident: partial(canonical_configuration, ident) for ident in IDS}\n')


def render_scheduler_adapters(value):
    published = [r for r in value['components'] if r['published']]
    ids = [r['definition']['id'] for r in published]
    return ('# Generated by scripts/component_bundles.py. Edit components/schedulers/bundle.json.\n'
            'from functools import partial\nfrom scheduler_configuration import canonical_configuration\n'
            'IDS = '+repr(tuple(ids))+'\nCAPABILITIES = '+repr({r['definition']['id']:r['runtimeContract'] for r in published})+'\n'
            'NORMALIZERS = {ident: partial(canonical_configuration, ident) for ident in IDS}\n')


def render_tensor_routing_adapters(value):
    ids = tuple(row['definition']['id'] for row in value['components'] if row['published'])
    return ('# Generated by scripts/component_bundles.py. Edit components/tensor-routing/bundle.json.\n'
            'from functools import partial\n'
            'from tensor_routing_configuration import capability, normalize_tensor_routing, resolve_tensor_routing, configured_inputs\n\n'
            f'IDS = {ids!r}\n'
            'CAPABILITIES = {ident: partial(capability, ident) for ident in IDS}\n'
            'NORMALIZERS = {ident: partial(normalize_tensor_routing, ident) for ident in IDS}\n'
            'RESOLVERS = {ident: partial(resolve_tensor_routing, ident) for ident in IDS}\n'
            'CONFIGURED_INPUTS = {ident: partial(configured_inputs, ident) for ident in IDS}\n')

def render_spatial_patch_adapters(value):
    ids = tuple(row['definition']['id'] for row in value['components'] if row['published'])
    return ('# Generated by scripts/component_bundles.py. Edit components/spatial-patches/bundle.json.\n'
            'from functools import partial\n'
            'from spatial_patch_configuration import capability, normalize_spatial_patch, resolve_spatial_patch\n\n'
            f'IDS = {ids!r}\n'
            'CAPABILITIES = {ident: partial(capability, ident) for ident in IDS}\n'
            'NORMALIZERS = {ident: partial(normalize_spatial_patch, ident) for ident in IDS}\n'
            'RESOLVERS = {ident: partial(resolve_spatial_patch, ident) for ident in IDS}\n'
)



def render_attention_adapters(value):
    ids = tuple(row['definition']['id'] for row in value['components'] if row['published'])
    return ('# Generated by scripts/component_bundles.py. Edit components/attention/bundle.json.\n'
            'from functools import partial\n'
            'from attention_configuration import capability, normalize_attention, resolve_attention, configured_input_ports\n\n'
            'IDS = '+repr(ids)+'\n'
            'CAPABILITIES = {ident: capability(ident) for ident in IDS}\n'
            'NORMALIZERS = {ident: partial(normalize_attention, ident) for ident in IDS}\n'
            'RESOLVERS = {ident: partial(resolve_attention, ident) for ident in IDS}\n'
            'CONFIGURED_INPUTS = {ident: partial(configured_input_ports, ident) for ident in IDS}\n')


def render_recurrent_adapters(value):
    ids = tuple(row['definition']['id'] for row in value['components'] if row['published'])
    return ('# Generated by scripts/component_bundles.py. Edit components/recurrent/bundle.json.\n'
            'from functools import partial\n'
            'from recurrent_configuration import capability, normalize_recurrent, resolve_recurrent, configured_input_ports\n\n'
            f'IDS = {ids!r}\n'
            'CAPABILITIES = {ident: partial(capability, ident) for ident in IDS}\n'
            'NORMALIZERS = {ident: partial(normalize_recurrent, ident) for ident in IDS}\n'
            'RESOLVERS = {ident: partial(resolve_recurrent, ident) for ident in IDS}\n'
            'CONFIGURED_INPUTS = {ident: partial(configured_input_ports, ident) for ident in IDS}\n')


def render(root):
    value = json.loads((root / 'components/pooling/bundle.json').read_text())
    validate(value, root)
    registry = json.loads((root / 'components/registry-base.json').read_text())
    unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in value['components']], 'component ID')
    registry['components'].extend(row['definition'] for row in value['components'] if row['published'])
    swift = render_swift(value, "pooling")
    imports = []
    for prefix in ('.', ''):
        lines = []
        for row in value['components']:
            ident = row['definition']['id']; name = ident.replace('_pool', 'pool')
            if ident.startswith('adaptive_'):
                if not any('adaptive_pooling_configuration' in line for line in lines):
                    lines.append(f'    from {prefix}adaptive_pooling_configuration import resolve_adaptive_pooling, capability as adaptive_pooling_capability')
                continue
            imports_line = f"    from {prefix}{name}_configuration import resolve_{name}"
            if row['published']: imports_line += f", capability as {name}_capability"
            lines.append(imports_line)
        imports.append('\n'.join(lines))
    worker = '# Generated by scripts/component_bundles.py. Do not edit.\nif __package__:\n' + imports[0] + '\nelse:\n' + imports[1] + '\n\n'
    for row in value['components']:
        ident = row['definition']['id']
        if ident.startswith('adaptive_'):
            name = ident.replace('_pool','pool')
            worker += f'def resolve_{name}(config, layout):\n    return resolve_adaptive_pooling({ident!r}, config, layout)\n\ndef {name}_capability():\n    return adaptive_pooling_capability({ident!r})\n\n'
    worker += 'RESOLVERS = {\n' + ''.join(f"    {r['definition']['id']!r}: resolve_{r['definition']['id'].replace('_pool','pool')},\n" for r in value['components']) + '}\n'
    worker += 'CAPABILITIES = {\n' + ''.join(f"    {r['definition']['id']!r}: {r['definition']['id'].replace('_pool','pool')}_capability,\n" for r in value['components'] if r['published']) + '}\n'
    result = {'Sources/KwipMLCore/Resources/components.json': json.dumps(registry, indent=2, ensure_ascii=False)+'\n',
            'Sources/KwipMLCore/BundledPoolingComponents.generated.swift': swift, 'runner/pooling_adapters_generated.py': worker}
    activation_path = root / 'components/activations/bundle.json'
    if activation_path.exists():
        activations = json.loads(activation_path.read_text())
        validate(activations, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in activations['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in activations['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledActivationComponents.generated.swift'] = render_swift(activations, 'activations')
        result['runner/remaining_activation_adapters_generated.py'] = '# Generated by scripts/component_bundles.py. Edit components/activations/bundle.json.\nfrom functools import partial\nfrom remaining_activation_configuration import capability, normalize_remaining_activation, resolve_remaining_activation\n\nIDS = ("prelu", "log_softmax", "dropout")\nCAPABILITIES = {ident: partial(capability, ident) for ident in IDS}\nNORMALIZERS = {ident: partial(normalize_remaining_activation, ident) for ident in IDS}\nRESOLVERS = {ident: partial(resolve_remaining_activation, ident) for ident in IDS}\n'.replace('IDS = ("prelu", "log_softmax", "dropout")', 'IDS = '+repr(tuple(row['definition']['id'] for row in activations['components'] if row['published'] and row['definition']['id'] in REMAINING_ACTIVATION_DEFINITIONS)))
    normalization_path = root / 'components/normalization/bundle.json'
    if normalization_path.exists():
        normalization = json.loads(normalization_path.read_text())
        validate(normalization, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in normalization['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in normalization['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledNormalizationComponents.generated.swift'] = render_swift(normalization, 'normalization')
        result['runner/stateful_normalization_adapters_generated.py'] = '# Generated by scripts/component_bundles.py. Edit components/normalization/bundle.json.\nfrom functools import partial\nfrom stateful_normalization_configuration import capability, normalize_stateful_normalization, resolve_stateful_normalization\n\nIDS = '+repr(tuple(r['definition']['id'] for r in normalization['components'] if r['published'] and r['definition']['id'] in STATEFUL_NORMALIZATION_DEFINITIONS))+'\nCAPABILITIES = {ident: partial(capability, ident) for ident in IDS}\nNORMALIZERS = {ident: partial(normalize_stateful_normalization, ident) for ident in IDS}\nRESOLVERS = {ident: partial(resolve_stateful_normalization, ident) for ident in IDS}\n'
    layer_path = root / 'components/layers/bundle.json'
    if layer_path.exists():
        layers = json.loads(layer_path.read_text())
        validate(layers, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in layers['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in layers['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledLayerComponents.generated.swift'] = render_swift(layers, 'layers')
    tensor_path = root / 'components/tensor-layout/bundle.json'
    if tensor_path.exists():
        tensor_layout = json.loads(tensor_path.read_text())
        validate(tensor_layout, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in tensor_layout['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in tensor_layout['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledTensorLayoutComponents.generated.swift'] = render_swift(tensor_layout, 'tensor-layout')
    matrix_path = root / 'components/tensor-math/bundle.json'
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text())
        validate(matrix, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in matrix['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in matrix['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledTensorMathComponents.generated.swift'] = render_swift(matrix, 'tensor-math')
    embedding_path = root / 'components/embeddings/bundle.json'
    if embedding_path.exists():
        embedding = json.loads(embedding_path.read_text())
        validate(embedding, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in embedding['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in embedding['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledEmbeddingComponents.generated.swift'] = render_swift(embedding, 'embeddings')
    convolution_path = root / 'components/convolution/bundle.json'
    if convolution_path.exists():
        convolution = json.loads(convolution_path.read_text())
        validate(convolution, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in convolution['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in convolution['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledConvolutionComponents.generated.swift'] = render_swift(convolution, 'convolution')
    composition_path = root / 'components/tensor-composition/bundle.json'
    if composition_path.exists():
        composition = json.loads(composition_path.read_text())
        validate(composition, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in composition['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in composition['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledTensorCompositionComponents.generated.swift'] = render_swift(composition, 'tensor-composition')
        result['runner/tensor_composition_adapters_generated.py'] = '# Generated by scripts/component_bundles.py. Edit components/tensor-composition/bundle.json.\nfrom functools import partial\nfrom tensor_composition_configuration import capability, normalize_tensor_composition, resolve_tensor_composition\n\nIDS = ("concat", "add", "stack", "permute")\nCAPABILITIES = {ident: partial(capability, ident) for ident in IDS}\nNORMALIZERS = {ident: partial(normalize_tensor_composition, ident) for ident in IDS}\nRESOLVERS = {ident: partial(resolve_tensor_composition, ident) for ident in IDS}\n'
    routing_path = root / 'components/tensor-routing/bundle.json'
    if routing_path.exists():
        routing = json.loads(routing_path.read_text())
        validate(routing, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in routing['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in routing['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledTensorRoutingComponents.generated.swift'] = render_swift(routing, 'tensor-routing')
        result['runner/tensor_routing_adapters_generated.py'] = render_tensor_routing_adapters(routing)
    patch_path = root / 'components/spatial-patches/bundle.json'
    if patch_path.exists():
        patches = json.loads(patch_path.read_text())
        validate(patches, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in patches['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in patches['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledSpatialPatchComponents.generated.swift'] = render_swift(patches, 'spatial-patches')
        result['runner/spatial_patch_adapters_generated.py'] = render_spatial_patch_adapters(patches)
    optimizer_path = root / 'components/optimizers/bundle.json'
    if optimizer_path.exists():
        optimizers = json.loads(optimizer_path.read_text())
        validate(optimizers, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in optimizers['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in optimizers['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledOptimizerComponents.generated.swift'] = render_swift(optimizers, 'optimizers')
        result['runner/optimizer_adapters_generated.py'] = render_optimizer_adapters(optimizers)
    scheduler_path = root / 'components/schedulers/bundle.json'
    if scheduler_path.exists():
        schedulers = json.loads(scheduler_path.read_text())
        validate(schedulers, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in schedulers['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in schedulers['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledSchedulerComponents.generated.swift'] = render_swift(schedulers, 'schedulers')
        result['runner/scheduler_adapters_generated.py'] = render_scheduler_adapters(schedulers)
    recurrent_path = root / 'components/recurrent/bundle.json'
    if recurrent_path.exists():
        recurrent = json.loads(recurrent_path.read_text())
        validate(recurrent, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in recurrent['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in recurrent['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledRecurrentComponents.generated.swift'] = render_swift(recurrent, 'recurrent')
        result['runner/recurrent_adapters_generated.py'] = render_recurrent_adapters(recurrent)
    attention_path = root / 'components/attention/bundle.json'
    if attention_path.exists():
        attention = json.loads(attention_path.read_text())
        validate(attention, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in attention['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in attention['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledAttentionComponents.generated.swift'] = render_swift(attention, 'attention')
        result['runner/attention_adapters_generated.py'] = render_attention_adapters(attention)
    loss_path = root / 'components/losses/bundle.json'
    if loss_path.exists():
        losses = json.loads(loss_path.read_text())
        validate(losses, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in losses['components']], 'component ID')
        registry['components'].extend(row['definition'] for row in losses['components'] if row['published'])
        result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledLossComponents.generated.swift'] = render_swift(losses, 'losses')
        result['runner/loss_adapters_generated.py'] = render_loss_adapters(losses)
    lifecycle_path = root / 'components/lifecycle/bundle.json'
    if lifecycle_path.exists():
        lifecycle = json.loads(lifecycle_path.read_text())
        validate(lifecycle, root)
        result['Sources/KwipMLCore/BundledLifecycleCapabilities.generated.swift'] = render_lifecycle_swift(lifecycle)
        result['runner/lifecycle_capabilities_generated.py'] = render_lifecycle_capabilities(lifecycle)
    initialization_path = root / 'components/initialization/bundle.json'
    if initialization_path.exists():
        initialization = json.loads(initialization_path.read_text())
        validate(initialization, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in initialization['components']], 'component ID')
        if any(row['published'] for row in initialization['components']):
            registry['components'].extend(row['definition'] for row in initialization['components'] if row['published'])
            result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledInitializationComponents.generated.swift'] = render_initialization_swift(initialization)
        result['runner/initialization_adapters_generated.py'] = render_initialization_capabilities(initialization)
    gradient_path = root / 'components/gradient_controls/bundle.json'
    if gradient_path.exists():
        gradient_controls = json.loads(gradient_path.read_text())
        validate(gradient_controls, root)
        unique([x['id'] for x in registry['components']] + [x['definition']['id'] for x in gradient_controls['components']], 'component ID')
        if any(row['published'] for row in gradient_controls['components']):
            registry['components'].extend(row['definition'] for row in gradient_controls['components'] if row['published'])
            result['Sources/KwipMLCore/Resources/components.json'] = json.dumps(registry, indent=2, ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledGradientControlComponents.generated.swift'] = render_gradient_control_swift(gradient_controls)
        result['runner/gradient_controls_generated.py'] = render_gradient_control_capabilities(gradient_controls)
    data_loading_path=root/'components/training-data/bundle.json'
    if data_loading_path.exists():
        data_loading=json.loads(data_loading_path.read_text());validate(data_loading,root)
        unique([x['id'] for x in registry['components']]+[r['definition']['id'] for r in data_loading['components']],'component ID')
        registry['components'].extend(r['definition'] for r in data_loading['components'] if r['published'])
        result['Sources/KwipMLCore/Resources/components.json']=json.dumps(registry,indent=2,ensure_ascii=False)+'\n'
        result['Sources/KwipMLCore/BundledDataLoadingComponents.generated.swift']=render_data_loading_swift(data_loading)
        result['runner/data_loading_capabilities_generated.py']=render_data_program_capabilities(data_loading,'components/training-data/bundle.json')
    program_path=root/'components/program-export/bundle.json'
    if program_path.exists():
        program=json.loads(program_path.read_text());validate(program,root)
        result['Sources/KwipMLCore/BundledProgramExportCapabilities.generated.swift']=render_program_export_swift(program)
        result['runner/program_export_capabilities_generated.py']=render_data_program_capabilities(program,'components/program-export/bundle.json')
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    outputs = render(ROOT)
    drift = []
    for path, data in outputs.items():
        target = ROOT / path
        if args.check:
            if not target.exists() or target.read_text() != data: drift.append(path)
        else: target.write_text(data)
    if drift: parser.exit(1, 'Component bundle output drift: '+', '.join(drift)+'\n')


if __name__ == '__main__': main()
