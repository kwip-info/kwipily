from scripts.tests.registry_history import historical_consolidation_copy
from scripts.tests.registry_history import historical_output_presentation,historical_pooling_output_prose,POOLING_HISTORY_PATH
"""Starting-value control authoring is explicit, source-backed and never a forward op."""
import ast,copy,hashlib,importlib.util,json,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('initialization_bundle',ROOT/'scripts/component_bundles.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
SWIFT='Sources/KwipMLCore/BundledInitializationComponents.generated.swift'
WORKER='runner/initialization_adapters_generated.py'
def source():return json.loads((ROOT/'components/initialization/bundle.json').read_text())
def test_actual_control_metadata_and_frozen_runtime_descriptor():
    value=source();b.validate(value,ROOT)
    row=value['components'][0]
    assert row['definition']=={'id':'initialization','title':'Starting settings','category':'Training','inputs':[],'outputs':['out'],'fields':[]}
    assert row['editorExtension']=='initializationOwnersV1'
    tree=ast.parse((ROOT/'runner/engine.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='initialization_capability')
    assert row['runtimeContract']==ast.literal_eval(fn.body[0].value)
    assert row['modes']==['xavier-uniform','seeded-repeat','kaiming-uniform','normal','zeros']
def test_generated_outputs_are_current_and_prior83_definitions_remain_exact():
    value=source();b.validate(value,ROOT);outputs=b.render(ROOT)
    assert SWIFT in outputs and WORKER in outputs
    for path,text in outputs.items():assert (ROOT/path).read_text()==text,path
    registry_path='Sources/KwipMLCore/Resources/components.json'
    registry=json.loads(outputs[registry_path])
    registry['components']=[c for c in registry['components'] if c['id'] not in ('rms_norm','rotary_position','mixture_experts','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')]
    assert registry['components'][-1] == source()['components'][0]['definition']
    assert len(registry['components']) == 84
    prior_registry=copy.deepcopy(registry)
    prior_registry['components']=[c for c in registry['components'] if c['id'] != 'initialization']
    assert len(prior_registry['components']) == 83
    # Project the original pair and pre-Bag optional lessonIDs field for the frozen receipt.
    old_embedding=json.loads((ROOT / 'Tests/historical/d13a57e/components/embeddings/bundle.json').read_bytes())
    outputs['Sources/KwipMLCore/BundledEmbeddingComponents.generated.swift']=b.render_swift(old_embedding,'embeddings').replace('        public let lessonIDs: [String]?\n', '')
    prior_registry['components']=historical_output_presentation(prior_registry['components'])
    outputs=historical_consolidation_copy(outputs)
    outputs[POOLING_HISTORY_PATH]=historical_pooling_output_prose(outputs[POOLING_HISTORY_PATH])
    prior={path:hashlib.sha256((json.dumps(prior_registry,indent=2,ensure_ascii=False)+'\n' if path == registry_path else text).encode()).hexdigest()for path,text in outputs.items()if path not in (SWIFT,WORKER,'Sources/KwipMLCore/BundledGradientControlComponents.generated.swift','runner/gradient_controls_generated.py','Sources/KwipMLCore/BundledDataLoadingComponents.generated.swift','runner/data_loading_capabilities_generated.py','Sources/KwipMLCore/BundledProgramExportCapabilities.generated.swift','runner/program_export_capabilities_generated.py')}
    assert hashlib.sha256(json.dumps(prior,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='46a5342cd020df5d06071120cfed8dabc2fdb0bd4c9110275052450b06d70a41'
def test_worker_metadata_has_no_forward_adapter_and_original_icon_is_pinned():
    value=source();b.validate(value,ROOT);outputs=b.render(ROOT);ns={};exec(outputs[WORKER],ns)
    assert ns['CAPABILITIES']=={'initialization':value['components'][0]['runtimeContract']}
    assert not {'SUPPORTED','NORMALIZERS','RESOLVERS'}&ns.keys()
    assert 'initializationOwnersV1'in outputs[SWIFT]
    art=json.loads((ROOT/'docs/design/initialization-icons-v1/provenance-20261009.json').read_text())
    original=ROOT/'docs/design/initialization-icons-v1/initialization-v1.png';resource=ROOT/'Sources/KwipMLApp/Resources/ComponentIcons/initialization-v1.png'
    assert original.read_bytes()==resource.read_bytes()
    assert hashlib.sha256(original.read_bytes()).hexdigest()==art['sha256']
@pytest.mark.parametrize('mutation',['identity','kind','unknown','tensor_input','tensor_output','fake_rule_text','editor','descriptor','bool_integer','seed','modes','icon','source','missing_source','publication','unknown_learning'])
def test_invalid_or_premature_metadata_is_rejected(mutation):
    value=source();b.validate(value,ROOT);row=value['components'][0]
    if mutation=='identity':row['definition']['id']='linear'
    elif mutation=='kind':value['kind']='tensor'
    elif mutation=='unknown':row['unsafeCallback']='code'
    elif mutation=='tensor_input':row['definition']['inputs']=['in']
    elif mutation=='tensor_output':row['definition']['outputs']=['tensor']
    elif mutation=='fake_rule_text':row['definition']['fields']=[{'key':'rules','kind':'text','defaultValue':'{}'}]
    elif mutation=='editor':row['editorExtension']='rawJSON'
    elif mutation=='descriptor':row['runtimeContract']['initializesBuffers']=True
    elif mutation=='bool_integer':row['runtimeContract']['maximumRules']=True
    elif mutation=='seed':row['runtimeContract']['seedMaximum']=9223372036854775808
    elif mutation=='modes':row['modes'].pop()
    elif mutation=='icon':row['presentation']['icon']='unreviewed'
    elif mutation=='source':row['sources']['native']='../escape.swift'
    elif mutation=='missing_source':row['sources']['native']='Sources/no-file.swift'
    elif mutation=='publication':row['published']=True;row['learning']=None
    else:row['learning']={'lessonIDs':['fake']}
    with pytest.raises(ValueError):b.validate(value,ROOT)

def test_swift_control_declarations_are_publication_gated_without_forward_dispatch():
    value=source();value['components'][0]['published']=False
    text=b.render_initialization_swift(value)
    assert 'public static let declarations' in text
    assert 'InitializationCapabilityBridge.declaration' not in text
    published=copy.deepcopy(value)
    published['components'][0]['published']=True
    text=b.render_initialization_swift(published)
    assert '"initialization": InitializationCapabilityBridge.declaration' in text
    assert 'SUPPORTED' not in text
