"""Packaged runtimes supply a registry even to registry-free captured tools."""
import json
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority, publish
from runner.tests.test_preparation_operation_authorization import preparation_operation
from runner.tests.test_code_operation_authorization import code_operation, workflow_code_operation
from runner.tests.test_table_view_operation import view_operation
from runner.tests.test_file_collection_admission import inspection
from runner.tests.test_workflow_code import snapshot
from runner.tests.test_evaluation_authorization import encoded, digest
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry, run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


def run_actual(f,tmp_path,libraries):
    guard,_=libraries
    registry=tmp_path/'registry.json';registry.write_text('{}')
    path=entry(tmp_path,guard,registry=registry)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    replacement="    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    status=main(check_cancelled=result.authority.check)\n    assert 'torch' not in sys.modules\n    sys.exit(status)"
    source=path.read_text();assert marker in source;path.write_text(source.replace(marker,replacement))
    return run_supervisor(path,control(f))


@pytest.mark.skipif(sys.platform!='darwin',reason='Real native lifetime guard')
def test_packaged_file_preview_preserves_captured_manifest(preparation_operation,tmp_path,libraries):
    f=preparation_operation;inspection(f)
    code,events,error=run_actual(f,tmp_path,libraries)
    assert code==0,(events,error)
    assert events[0]['event']=='operationReady'
    page=events[-1];assert page['event']=='dataPreview' and page['rowCount']==1
    assert page['collection']['selectedFiles']==1
    assert 'readme.txt' in json.dumps(page['rows'])


@pytest.mark.skipif(sys.platform!='darwin',reason='Real native lifetime guard')
def test_packaged_workflow_code_generates_real_source_package(workflow_code_operation,tmp_path,libraries):
    f=workflow_code_operation;raw=encoded(snapshot())
    (f[0]/'input/workflow-code.json').write_bytes(raw)
    f[1]['inputFiles'][0].update(bytes=len(raw),sha256=digest(raw));publish(f)
    code,events,error=run_actual(f,tmp_path,libraries)
    assert code==0,(events,error)
    package=events[-1];assert package['event']=='codeExport'
    assert {'workflow.py','workflow-plan.json','preparation.py'}<=set(package['files'])
    for name,source in package['files'].items():
        if name.endswith('.py'):compile(source,name,'exec')


@pytest.mark.skipif(sys.platform!='darwin',reason='Real native lifetime guard')
@pytest.mark.parametrize('route',['view','files','workflow'])
@pytest.mark.parametrize('mutation',['unknown','duplicate','registry','wrong-mode','wrong-path'])
def test_registry_free_routes_still_refuse_changed_authority(request,tmp_path,libraries,route,mutation):
    name={'view':'view_operation','files':'preparation_operation','workflow':'workflow_code_operation'}[route]
    f=request.getfixturevalue(name)
    if route=='files':inspection(f)
    args=f[1]['arguments']
    if mutation=='unknown':args+=['--unknown','x']
    elif mutation=='duplicate':args+=args[:2]
    elif mutation=='registry':args+=['--registry',str(tmp_path/'registry.json')]
    elif mutation=='wrong-mode':f[1]['toolMode']='runtimeCapabilities'
    else:args[1 if route!='workflow' else 2]=str(tmp_path/'unowned')
    publish(f)
    code,events,error=run_actual(f,tmp_path,libraries)
    assert code!=0 and all(e['event']!='operationReady' for e in events),(events,error)
    assert not (f[0]/'claim.json').exists()


def test_generic_model_code_still_receives_verified_registry(tmp_path):
    from runner.worker_bootstrap import _arguments,RuntimeContext
    registry=tmp_path/'registry.json';registry.write_text('{}')
    context=RuntimeContext(tmp_path/'main.py',tmp_path/'guard',registry)
    intent={'schemaVersion':2,'toolMode':'codeExport','arguments':['--export-code','--graph','/owned/graph.json']}
    assert _arguments(intent,context)==intent['arguments']+['--registry',str(registry)]
