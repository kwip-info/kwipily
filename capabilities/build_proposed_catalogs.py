#!/usr/bin/env python3
"""Materialize reviewed selections as proposed (never automatically frozen) catalogs.

Run capture_inventory.py first to restore the extracted source inventories. This
script maps selected operations; it does not turn their metadata into implementation.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CAP=ROOT/'capabilities'
MODES={
    'optimizer':['step','state-roundtrip','resume'], 'scheduler':['step','state-roundtrip'],
    'loss':['forward','backward','reduction'], 'metrics':['evaluate','edge-cases'],
    'data':['construct','iterate'], 'serialization':['roundtrip','incompatible-artifact'],
    'autodiff':['differentiate','gradient-control'], 'initialization':['initialize','seeded-repeat'],
    'tensor_ops':['forward','shape-contract','gradient-where-supported'],
    'model_selection':['partition','fit-selection','reproducibility'],
    'preprocessing':['fit','transform','fit-transform'], 'imputation':['fit','transform'],
    'feature_selection':['fit','transform'], 'feature_extraction':['fit','transform'],
    'inspection':['evaluate','provenance'], 'calibration':['fit','evaluate'],
    'clustering':['fit','assign-or-transform'], 'decomposition':['fit','transform'],
}
VERSIONS={'pytorch':('2.14.1','2.14'),'sklearn':('1.9.1','1.9'),'tensorflow':('2.16.1','2.16.1')}
POC_BINDINGS={'input':None,'linear':'torch.nn.Linear','relu':'torch.nn.ReLU','tanh':'torch.nn.Tanh','concat':'torch.cat','add':'torch.add','output':None}
EVIDENCE_PATHS={'configuration':'Sources/KwipMLCore/Resources/components.json','execution':'runner/engine.py',
               'export':'runner/engine.py','validation':'Sources/KwipMLCore/GraphValidator.swift',
               'diagnostics':'docs/DIAGNOSTICS.md','lesson':'Sources/KwipMLApp/WorkspaceView.swift',
               'example':'Sources/KwipMLCore/Graph.swift','ui':'docs/uat/GATE_C_D_E.md',
               'persistence':'docs/CORE_EVIDENCE.md','failure_handling':'runner/tests/test_runner.py'}

def main():
    selection=json.loads((CAP/'selection-v1.json').read_text())
    # A proposal rebuild must never erase an approved denominator or evidence.
    for framework in selection:
        current = CAP/'catalogs/v1'/f'{framework}.json'
        if current.exists():
            existing = json.loads(current.read_text())
            if existing.get('denominator_status') != 'proposed':
                raise ValueError('Refusing to overwrite approved catalog; use an explicit versioned migration')
    sources={s['framework']:s for s in json.loads((CAP/'sources.json').read_text())}
    (CAP/'catalogs/v1').mkdir(parents=True,exist_ok=True)
    for framework,families in selection.items():
        inventory_path=CAP/'inventories/v1'/f'{framework}.json'
        inventory=json.loads(inventory_path.read_text());by_name={r['symbol']:r for r in inventory}
        # Rebuilding must not leave prior selected mappings behind.
        for row in inventory:
            if row['disposition']=='mapped':
                row.update(disposition='unsupported',reason='Documented API not selected in this proposed catalog.',capability_ids=[])
        units=[]
        for family,symbols in families.items():
            for symbol in symbols:
                if symbol not in by_name:raise ValueError(f'Not in captured official inventory: {symbol}')
                row=by_name[symbol]
                if row['canonical_symbol']!=symbol:raise ValueError(f'Select canonical API instead of alias: {symbol}')
                key=f'{framework}.{symbol}'
                modes=MODES.get(family,['fit','predict'] if framework=='sklearn' else ['inference','training'])
                unit={'id':key,'family':family,'canonical_symbol':symbol,'mode_scope':modes,
                      'mode_scope_status':'proposed; refine operation-specific modes before denominator freeze',
                      'configuration_contract':'pending','configuration_plan':{
                          'parameters':'Enumerate documented parameters and interactions; no default-only coverage credit.',
                          'required_cases':['defaults','non-default settings','valid boundaries','invalid types/shapes','state lifecycle where applicable'],
                          'limits':'Framework/operator-specific dtype, sparse/ragged, axis/layout, rank and device limits must be explicit.'},
                      'backend_matrix':{'status':'pending','cases':['cpu-float32']+(['mps-float32-where-supported'] if framework=='pytorch' else [])},
                      'evidence':{},'implementation_status':'planned',
                      'lesson_id':f'lesson.{key}','example_id':f'example.{key}',
                      'notes':'Lesson/example IDs reserve required deliverables; they do not identify shipped content. This unit is incomplete.'}
                if framework=='pytorch' and symbol in POC_BINDINGS.values():
                    unit['implementation_status']='bounded-poc-implementation'
                    unit['notes']='Current rank-2 dense float32 POC supports a bounded subset. Production modes/parameter coverage, dedicated lesson/example and evidence review remain incomplete.'
                    for dimension,path in EVIDENCE_PATHS.items():
                        file=ROOT/path
                        if not file.is_file():raise ValueError(f'Missing POC reference: {path}')
                        unit['evidence'][dimension]=[{'path':path,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                                                     'status':'present','scope':'poc-only','note':'Reference exists; not attested as full production capability evidence.'}]
                units.append(unit)
                row.update(disposition='mapped',reason=f'Selected proposed {family} capability. Mapping does not mean implemented.',capability_ids=[key])
        # Aliases share their canonical operation mapping without additional units.
        for row in inventory:
            if row['disposition']=='alias':row['capability_ids']=by_name[row['canonical_symbol']]['capability_ids'][:]
        inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
        version,docs=VERSIONS[framework]
        catalog={'schema_version':1,'catalog_version':'0.1.0','framework':framework,'framework_version':version,
                 'documentation_version':docs,'snapshot_date':'2026-10-05',
                 'runtime_status':'POC runtime only; production matrix pending' if framework=='pytorch' else 'Not installed or implemented in KwipML',
                 'version_alignment':'Sphinx docs expose major/minor; exact patch/API alignment still requires review.' if framework!='tensorflow' else 'Official page labels TensorFlow2.16.1; snapshot content hash pins this capture; runtime unverified.',
                 'denominator_status':'proposed','approval':None,'thresholds':{'minimum_percent':90,'target_percent':95},
                 'required_families':list(families),'inventory':{
                     'path':str(inventory_path.relative_to(ROOT)),'sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                     'scope':'All captured Sphinx Python objects under the framework prefix.' if framework!='tensorflow' else 'All unique tf API page links in the captured official navigation; class members and runtime exports may be absent.',
                     'framework_completeness':'not_established','canonicalization_status':'Exact documentation-target aliases resolved; other aliases require review.',
                     'source':sources[framework]},
                 'scope_decision':{'status':'review-needed','reason':'Broad API inventory exceeds proposed user-operation subset. No framework-wide90% claim; mode boundaries, exclusions and patch alignment must be reviewed before freeze.'},
                 'capabilities':units,'poc_components':[]}
        if framework=='pytorch':
            catalog['poc_components']=[{'id':key,'framework_binding':symbol,'status':'working-bounded-poc',
                                        'scope':'Two-feature binary-classification graph; see component contracts for exact per-node constraints.',
                                        'evidence':'docs/COMPONENT_CONTRACTS.md',
                                        'denominator_note':'Input/Output are product port roles, not extra public framework APIs.'} for key,symbol in POC_BINDINGS.items()]
        (CAP/'catalogs/v1'/f'{framework}.json').write_text(json.dumps(catalog,indent=2)+'\n')
        print(framework,len(units),'proposed units',len(families),'required families')
if __name__=='__main__':main()
