"""Strict capture choices for optional training samples, not required ML evidence."""

_FIELDS = frozenset({'schemaVersion', 'activationValues', 'predictionRows'})


def validate_training_sample_capture_policy(value):
    """Validate an explicitly present wire policy and return an independent copy.

    Envelope readers must call this for present fields, including JSON null.
    Only the legacy resolver below treats absence represented by None as default.
    """
    if not isinstance(value, dict) or type(value.get('schemaVersion')) is not int or value['schemaVersion'] not in (1, 2, 3, 4):
        raise ValueError('Unsupported training sample capture schema.')
    fields = _FIELDS if value['schemaVersion'] == 1 else _FIELDS | {'performanceMeasurements'}
    if value['schemaVersion']>=3:fields |= {'retainComponentOutputs','retainDiagnosticDetails'}
    if value['schemaVersion']==4:fields |= {'tensorHeatmaps','heatmapFrameLimit'}
    if set(value) != fields:
        raise ValueError('Unknown or missing training sample capture fields.')
    for field in fields - {'schemaVersion','heatmapFrameLimit'}:
        if type(value[field]) is not bool:
            raise ValueError(f'Training sample capture {field} must be a boolean.')
    if value['schemaVersion']==4:
        limit=value['heatmapFrameLimit']
        if type(limit) is not int or not 0<=limit<=24 or (limit>0 and not value['tensorHeatmaps']):raise ValueError('Heatmap playback requires heatmaps and from 0 to 24 retained frames.')
    if value['schemaVersion']>=3:
        if not value['retainComponentOutputs'] and value['activationValues']:
            raise ValueError('Disabled history outputs cannot collect activation values.')
        if not value['retainComponentOutputs'] and value.get('tensorHeatmaps',False):
            raise ValueError('Disabled history outputs cannot collect tensor heatmaps.')
        if not value['retainDiagnosticDetails'] and value['performanceMeasurements']:
            raise ValueError('Disabled history diagnostics cannot collect timing/memory samples.')
    return dict(value)


def resolve_training_sample_capture_policy(value=None):
    """Resolve legacy absence without rewriting historical requests or receipts."""
    if value is None:
        return {'schemaVersion': 1, 'activationValues': True, 'predictionRows': True}
    return validate_training_sample_capture_policy(value)
