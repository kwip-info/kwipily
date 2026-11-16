#!/usr/bin/env python3
"""Offline, idempotent 99-to-88 consumer-scope migration; accepts no evidence."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'docs/production/TORCH_V1_SCOPE_REVIEW_20261008.md'
GROUPS = {
    'torch.nn.Module': {
        'torch.nn.Module':['branched-structure','nested-structure','parameter-ownership','train-eval','architecture-export'],
        'torch.nn.Sequential':['sequential-structure'],
        'torch.nn.ModuleList':['ordered-structure'],
        'torch.nn.ModuleDict':['named-structure']},
    'torch.utils.data.Dataset': {
        'torch.utils.data.Dataset':['indexed-source'],
        'torch.utils.data.IterableDataset':['bounded-iterable-source'],
        'torch.utils.data.TensorDataset':['tensor-source'],
        'torch.utils.data.Subset':['selected-index-view']},
    'torch.save': {'torch.save':['weights-save','continuation-save'],
                   'torch.load':['weights-restore','continuation-restore','incompatible-artifact-refusal']},
    'torch.autograd.grad': {'torch.autograd.grad':['requested-gradients'],
                            'torch.autograd.backward':['reset-gradients','accumulated-gradients']},
    'torch.nn.init.xavier_uniform_': {
        'torch.nn.init.xavier_uniform_':['xavier-uniform','seeded-repeat'],
        'torch.nn.init.kaiming_uniform_':['kaiming-uniform','seeded-repeat'],
        'torch.nn.init.normal_':['normal','seeded-repeat'],
        'torch.nn.init.zeros_':['zeros','seeded-repeat']}}

SEMANTICS = {
    'torch.nn.Module':'Editable named/ordered interfaces, sequential/branched/nested structure, instance parameter ownership, train/eval propagation and portable export. Storage classes are distinct required semantics, not aliases.',
    'torch.utils.data.Dataset':'Indexed/tensor/subset modes plus bounded cancellable iterable consumption with repeat and seed policy. All four source/view modes are compulsory.',
    'torch.save':'Safe application-owned save and restore package lifecycle; weights-only versus exact continuation, compatibility/corruption refusal, privacy and portable identity. Both directions compulsory.',
    'torch.autograd.grad':'Visible selected input/parameter targets, requested derivatives versus reset/accumulated parameter gradients, differentiability and bounded capture. Both methods compulsory.',
    'torch.nn.init.xavier_uniform_':'User-selectable Xavier uniform, Kaiming uniform, normal and zeros, with algorithm-specific settings, reproducible seeds and persisted initialization policy.'}

def modes(unit):
    symbol, family = unit['canonical_symbol'], unit['family']
    if symbol == 'torch.utils.data.DataLoader': return ['batch-order','shuffle-seed','short-final-batch','drop-last','collation','bounded-workers','continuation-order']
    if symbol == 'torch.export.export': return ['program-export','shape-constraints','program-reload-execution']
    if symbol == 'torch.no_grad': return ['frozen-inference','explicit-trainability-transition','unavailable-gradient-capture']
    if family in {'linear','convolution','normalization','embedding','attention','recurrent'}: return ['inference','training','input-gradients','state-roundtrip']
    if family in {'pooling','activation','tensor_ops'}:
        extra = ['training','inference','seeded-repeat'] if symbol == 'torch.nn.Dropout' else []
        if symbol == 'torch.nn.PReLU': extra = ['training','state-roundtrip']
        return ['forward','input-gradients','shape-contract'] + extra
    if family == 'loss': return ['training-objective','evaluation-objective','target-contract','reduction','input-gradients']
    if family == 'optimizer': return ['configured-step','parameter-ownership','state-roundtrip','resume'] + (['closure-repeated-evaluation'] if symbol == 'torch.optim.LBFGS' else [])
    if family == 'scheduler': return ['configured-schedule','step-timing','state-roundtrip','resume','history-rate-trace'] + (['required-metric'] if symbol.endswith('ReduceLROnPlateau') else [])
    raise ValueError(f'No reviewed source mode proposal for {symbol}')

def migrate(catalog, inventory, selection):
    """Return new objects; never mutate accepted evidence or original inputs."""
    catalog, inventory = copy.deepcopy(catalog), copy.deepcopy(inventory)
    if catalog.get('catalog_version') == '0.2.0':
        if len(catalog['capabilities']) != 88: raise ValueError('Corrupt migrated consumer count')
        return catalog, inventory
    if catalog.get('catalog_version') != '0.1.0' or catalog['denominator_status'] != 'proposed':
        raise ValueError('Migration accepts only the original proposed 0.1.0 catalog')
    originals = {symbol for group in selection.values() for symbol in group}
    old = {unit['canonical_symbol']:unit for unit in catalog['capabilities']}
    if len(old) != 99 or set(old) != originals: raise ValueError('Original 99-unit selection mismatch')
    removed = {symbol for primary, members in GROUPS.items() for symbol in members if symbol != primary}
    units, redirects = [], {}
    for previous in catalog['capabilities']:
        symbol = previous['canonical_symbol']
        if symbol in removed: continue
        unit = copy.deepcopy(previous)
        if symbol in GROUPS:
            unit['source_members'] = [
                {'canonical_symbol':member, 'required_modes':required,
                 'original_capability_id':old[member]['id']} for member, required in GROUPS[symbol].items()]
            unit['mode_scope'] = list(dict.fromkeys(mode for required in GROUPS[symbol].values() for mode in required))
            unit['required_semantics'] = SEMANTICS[symbol]
            unit['consolidation'] = {'reference':REFERENCE, 'reason':'One consumer feature with all named source semantics mandatory; not API aliases.',
                                     'retired_catalog_ids':[old[member]['id'] for member in GROUPS[symbol] if member != symbol]}
            for member in GROUPS[symbol]: redirects[old[member]['id']] = unit['id']
        else:
            unit['mode_scope'] = modes(unit)
        unit['mode_scope_status'] = 'pending-operation-boundary-review'
        unit['scope_reference'] = REFERENCE
        # Existing present/POC evidence remains unchanged and earns no acceptance.
        units.append(unit)
    if len(units) != 88: raise ValueError('Reviewed denominator must be exactly 88')
    for row in inventory:
        row['capability_ids'] = list(dict.fromkeys(redirects.get(key,key) for key in row.get('capability_ids', [])))
        if row['symbol'] in originals:
            row['reason'] = 'Selected in owner-approved 88-unit consumer scope; grouped source semantics remain compulsory. Mapping does not mean implemented.'
    catalog.update(catalog_version='0.2.0', capabilities=units,
        denominator_status='approved-consumer-scope', approval={'reference':REFERENCE,'approved_on':'2026-10-08','authority':'owner','scope':'88 consumer units; not broad public API coverage'},
        scope_decision={'status':'owner-approved-consumer-scope','approved_unit_count':88,
            'reference':REFERENCE,'original_selection':'capabilities/selection-v1.json',
            'original_unit_count':99,'reason':'Consolidate storage/lifecycle/policy into consumer features without removing any selected canonical source. Public APIs outside this list remain out of selected v1 scope, not covered.'},
        boundary_review={'status':'pending','required':'Per-unit operation semantics, supported dtype/rank/device/refusal matrix and runtime alignment must be reviewed before complete credit and release.'},
        deferrals={'status':'pending-owner-review','units':[], 'maximum_units':8,
                   'required':'Owner-approved named final incomplete units and reasons; every required family must still have an accepted unit.'},
        migration={'from_catalog_version':'0.1.0','date':'2026-10-08','reference':REFERENCE,
                   'original_selection_sha256':hashlib.sha256((ROOT/'capabilities/selection-v1.json').read_bytes()).hexdigest(),
                   'canonical_source_count':99,'consumer_unit_count':88,'evidence_acceptance':'none'})
    return catalog, inventory

def main():
    path = ROOT/'capabilities/catalogs/v1/pytorch.json'
    inventory_path = ROOT/'capabilities/inventories/v1/pytorch.json'
    catalog, inventory = migrate(json.loads(path.read_text()), json.loads(inventory_path.read_text()),
                                 json.loads((ROOT/'capabilities/selection-v1.json').read_text())['pytorch'])
    inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
    catalog['inventory']['sha256'] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(catalog,indent=2)+'\n')
    print('PyTorch: 88 owner-approved consumer units, 99 selected canonical sources, no evidence accepted.')

if __name__ == '__main__': main()
