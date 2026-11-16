import hashlib
import json
from pathlib import Path
import pytest
from runner.backend_fit_operation_contract import parse_contract, verify_inputs
from runner.backend_fit_cli import execute_request


def digest(raw):return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path):
    root=tmp_path/'operation';folder=root/'input/preparation';folder.mkdir(parents=True)
    graph=b'{}';(folder/'dataflow.json').write_bytes(graph)
    manifest=json.dumps({'schemaVersion':1,'kind':'dataflowExecution','status':'completed','graphDigest':digest(graph),'artifacts':[{'path':'dataflow.json','bytes':2,'sha256':digest(graph)}]}).encode()
    (folder/'manifest.json').write_bytes(manifest)
    request=json.dumps({'backendID':'sklearn','prepared':{'manifest':str(folder/'manifest.json'),'manifestSHA256':digest(manifest),'graphDigest':digest(graph)}}).encode()
    (root/'input/backend-request.json').write_bytes(request)
    files=[{'path':p.relative_to(root).as_posix(),'bytes':len(p.read_bytes()),'sha256':digest(p.read_bytes())} for p in (root/'input').rglob('*') if p.is_file()]
    value={'arguments':['--fit-backend',str(root/'input/backend-request.json'),'--backend-request-sha256',digest(request),'--output',str(root/'artifacts')],'inputFiles':files}
    return root,value


def test_retained_inventory_verified_and_content_tamper_rejected(tmp_path):
    root,value=fixture(tmp_path)
    parse_contract(value,root);verify_inputs(value,root,{})
    (root/'input/preparation/dataflow.json').write_text('[]')
    with pytest.raises(ValueError):verify_inputs(value,root,{})


def test_escape_or_extra_captured_file_or_wrong_pin_rejected(tmp_path):
    root,value=fixture(tmp_path)
    value['arguments'][-1]=str(tmp_path/'elsewhere')
    with pytest.raises(ValueError):parse_contract(value,root)
    value['arguments'][-1]=str(root/'artifacts')
    value['inputFiles'].append({'path':'input/preparation/extra.json','bytes':0,'sha256':digest(b'')})
    (root/'input/preparation/extra.json').write_bytes(b'')
    with pytest.raises(ValueError):verify_inputs(value,root,{})


def test_dispatch_refuses_unknown_backend_and_changed_request_without_importing_framework(tmp_path):
    path=tmp_path/'request.json';raw=b'{"backendID":"untrusted.module"}';path.write_bytes(raw)
    with pytest.raises(ValueError,match='Unsupported'):execute_request(path,digest(raw),tmp_path/'out',lambda _:None)
    with pytest.raises(ValueError,match='SHA256'):execute_request(path,'0'*64,tmp_path/'out',lambda _:None)


def model_inputs(tmp_path):
    root,value=fixture(tmp_path)
    for name,raw in [('model-state.json',b'{"inert":"model"}'),('model-preparation.json',b'{"inert":"rules"}')]:
        (root/'input'/name).write_bytes(raw)
    path=root/'input/backend-request.json';request=json.loads(path.read_bytes())
    request['configuration']=dict(task='prebuilt',operation='predict')
    request['model']=dict(path=str(root/'input/model-state.json'),sha256=digest((root/'input/model-state.json').read_bytes()),
        preparationPath=str(root/'input/model-preparation.json'),preparationSHA256=digest((root/'input/model-preparation.json').read_bytes()))
    def publish():
        raw=json.dumps(request).encode();path.write_bytes(raw);value['arguments'][3]=digest(raw)
        value['inputFiles']=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in (root/'input').rglob('*') if p.is_file()]
    publish();return root,value,request,publish


def test_backend_admits_exact_pinned_model_and_preparation_resources(tmp_path):
    # Admission pins inert bytes; the model worker separately validates contents.
    root,value,_,_=model_inputs(tmp_path);cache={}
    try:
        verify_inputs(value,root,cache)
        (root/'input/model-preparation.json').write_bytes(b'changed')
        with pytest.raises(ValueError):verify_inputs(value,root,cache)
    finally:
        from runner.backend_fit_operation_contract import close_inputs
        close_inputs(cache)


@pytest.mark.parametrize('change',['partial','wrong-pin','escape','unselected','fit','other-task'])
def test_backend_refuses_unbound_or_inapplicable_preparation(tmp_path,change):
    root,value,request,publish=model_inputs(tmp_path)
    if change=='partial':del request['model']['preparationSHA256']
    elif change=='wrong-pin':request['model']['preparationSHA256']='0'*64
    elif change=='escape':request['model']['preparationPath']=str(tmp_path/'outside.json')
    elif change=='unselected':del request['model']
    elif change=='fit':request['configuration']['operation']='fit'
    else:request['configuration']['task']='documentClassification'
    publish()
    with pytest.raises(ValueError):verify_inputs(value,root,{})
