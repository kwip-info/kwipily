"""Independent exact descriptor pin; primitive discovery remains independent."""
from runner.tests.cli_discovery import capabilities_cli
import json,subprocess,sys
from pathlib import Path
from runner.module_composition_capability import capability

EXPECTED={
 'version':'1.0.0','adapterID':'kwip.runner.module-composition','graphSchemaVersions':[2],'planSchemaVersion':1,'modelAuthoringSchemaVersion':2,'compositeSourceSchemaVersion':3,'nativeExpansionContractVersion':2,'workflowSchemaVersion':11,'compilerContractVersion':10,'captureFormatVersion':6,
 'devices':['cpu','mps'],'modes':['branched-structure','nested-structure','parameter-ownership','train-eval','architecture-export','sequential-structure','ordered-structure','named-structure'],'sourceMembers':['torch.nn.Module','torch.nn.Sequential','torch.nn.ModuleList','torch.nn.ModuleDict'],'containerKinds':['graph','sequential','orderedList','namedMap'],
 'maximumDefinitions':64,'maximumExpandedNodes':128,'maximumEdges':256,'maximumContainers':129,'maximumChildReferences':256,'maximumBoundaryEndpoints':4096,'maximumDepth':8,'maximumPlanBytes':131072,'maximumSourceBytes':2000000,'maximumModelSourceBytes':2000000,'maximumParameters':200000,
 'parameterOwnership':'expandedNodeRelativeNativeParameter','stateOwnership':'bijectiveNativePersistentState','sharedParameters':False,'sequentialExecution':'nativeForwardValidatedChain','collectionExecution':'explicitDAG','sourceVerification':'independentCapturedRegistryReExpansion','exactResumeDevices':['cpu'],'fineTuneDevices':['cpu','mps'],'mpsResume':'weightsOnly','portableExport':'selfContainedNativeHierarchy','legacyAbsence':'unchanged'}

def test_exact_eight_modes_four_members_versions_limits_and_native_contract():
    assert capability()==EXPECTED

def test_bounded_torch_free_descriptor_import():
    path=Path(__file__).parents[1]/'module_composition_capability.py'
    script="import importlib.util,sys,json;s=importlib.util.spec_from_file_location('capability',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(json.dumps(m.capability()));assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-I','-c',script,str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==EXPECTED
    assert len(result.stdout.encode())<=4096

def test_each_call_returns_owned_arrays_and_dictionary():
    a=capability();b=capability();assert a==b==EXPECTED
    a['modes'].clear();a['sourceMembers'].append('foreign');a['version']='changed'
    assert b==capability()==EXPECTED

def test_published_sibling_does_not_widen_or_create_tensor_primitives():
    from runner.engine import capabilities
    discovered=capabilities()
    assert 'moduleComposition' not in discovered['componentCapabilities']
    assert discovered.get('compositionCapabilities')=={'moduleComposition':EXPECTED}


def test_actual_worker_cli_reports_exact_separate_sibling():
    root=Path(__file__).resolve().parents[2]
    result=capabilities_cli()
    assert result.returncode==0,result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(events)==1
    assert events[0].get('compositionCapabilities')=={'moduleComposition':EXPECTED}
    assert 'moduleComposition' not in events[0]['componentCapabilities']
