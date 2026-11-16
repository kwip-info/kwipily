"""Versioned physical-kernel choices for the single captured preparation plan.

Only qualified implementations are selected. A reference choice is explicit and
has no implied parallel/device capability. This metadata contains IDs and fixed
reasons, never configuration text, table cells or private source/destination paths.
"""
PURE=frozenset(('columns','sampleRows','join','union','aggregate','filterRows','view','partition',
                'formula','sort','unique','reshapeTable','splitText','sequenceWindows','gameRules'))
SNAPSHOTS=frozenset(('sourceCSV','sourceFiles','capturedTable'))
FITTED=frozenset(('preprocess','textEncode'))
MODELS=frozenset(('prepareMedia','extractText','transcribeAudio','classifyAudio','decide'))
WRITES=frozenset(('outputDelimited',))
KNOWN=PURE|SNAPSHOTS|FITTED|MODELS|WRITES


def build_kernel_plan(nodes):
    records={}
    for ident,(operator,configuration) in nodes.items():
        if operator not in KNOWN:raise ValueError('Operator execution capability is not classified')
        effect=('snapshot' if operator in SNAPSHOTS else 'fitted' if operator in FITTED else
                'modelOrMedia' if operator in MODELS else 'externalWrite' if operator in WRITES else 'pure')
        kernel='reference.'+operator+'.v1';reason='referenceSemantics';parallel=False
        if operator=='aggregate' and all(f['op']!='joinText' for f in configuration['aggregations']):
            kernel='aggregate.exactStreaming.v1';reason='qualifiedSequentialGroupScan'
        elif operator=='aggregate':reason='orderedTextRequiresReference'
        elif operator=='columns' and all(row['targetType'] in ('keep','image') for row in configuration['rows'] if row['included']):
            kernel='columns.sharedProjection.v1';reason='qualifiedImmutableProjection';parallel=True
        elif operator=='filterRows' and configuration['mode']=='expression':
            kernel='filter.booleanSelectionOrReference.v1';reason='qualifiedBooleanSubsetOtherwiseReference';parallel=True
        elif operator=='join':
            kernel='join.nativePairsOrReference.v1';reason='qualifiedNativeParentPairsOtherwiseReference'
        elif operator=='sort':
            kernel='sort.nativeOrderOrReference.v1';reason='qualifiedScalarExactOrderingOtherwiseReference'
        elif operator=='union':
            kernel='union.nativeConcatOrReference.v1';reason='qualifiedNativeParentBuffersOtherwiseReference'
        elif operator=='formula':
            kernel='formula.nativeExpressionsOrReference.v2';reason='qualifiedExactNativeExpressionsOtherwiseReference'
        elif effect!='pure':reason='ownedBoundaryRequiresReference'
        records[ident]={'kernelID':kernel,'effect':effect,'parallel':parallel,'reason':reason}
    return {'schemaVersion':4,'nodes':records,'fusion':{'schemaVersion':1,'maximumNodesPerRegion':32,'mode':'pureLinearStatic'},'workBoundaries':{'schemaVersion':1,'sqliteInstructions':32768}}


def validate_kernel_plan(value,nodes):
    expected=build_kernel_plan(nodes)
    # Canonical encoding keeps booleans distinct from integers in untrusted JSON.
    import json,copy
    def canonical(record):return json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)
    if (type(value) is not dict or set(value)!=({'schemaVersion','nodes','fusion','workBoundaries'} if value.get('schemaVersion')==4 else {'schemaVersion','nodes'})
        or type(value['schemaVersion']) is not int or value['schemaVersion'] not in (1,2,3,4)
        or type(value['nodes']) is not dict or set(value['nodes'])!=set(nodes)):
        raise ValueError('Physical kernel choices differ from qualified captured operators')
    if value['schemaVersion']==4 and (canonical(value['fusion'])!=canonical(expected['fusion']) or canonical(value['workBoundaries'])!=canonical(expected['workBoundaries'])):
        raise ValueError('Captured fusion policy differs from the qualified compiler')
    for ident,(operator,configuration) in nodes.items():
        current=expected['nodes'][ident]
        reference={'kernelID':'reference.'+operator+'.v1','effect':current['effect'],'parallel':False,
                   'reason':'ownedBoundaryRequiresReference' if current['effect']!='pure' else
                            'orderedTextRequiresReference' if operator=='aggregate' and any(f['op']=='joinText' for f in configuration['aggregations']) else 'referenceSemantics'}
        choices=[current,reference]
        if operator=='formula':choices.append({'kernelID':'formula.sharedColumns.v1','effect':'pure','parallel':False,'reason':'qualifiedChangedColumnsPythonEvaluation'})
        if current['parallel']:choices.append({**current,'parallel':False})
        if canonical(value['nodes'][ident]) not in {canonical(choice) for choice in choices}:
            raise ValueError('Physical kernel choices differ from qualified captured operators')
    # Preserve the pinned choice and version instead of rewriting old plans to
    # today's preferred optimizer selection.
    return copy.deepcopy(value)
