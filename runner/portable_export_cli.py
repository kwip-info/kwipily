"""Pinned worker request to a fixed, independently verified private package stage."""
import os
from pathlib import Path

if __package__:
    from .worker_authorization import directory,read_regular,object_json,require,hash_value,sha
    from .frozen_evaluation_cli import read_nested
    from .portable_export_authorization import validate_request
    from .portable_evaluation_export import build_package
    from .portable_evaluation import validate_package
else:
    from worker_authorization import directory,read_regular,object_json,require,hash_value,sha
    from frozen_evaluation_cli import read_nested
    from portable_export_authorization import validate_request
    from portable_evaluation_export import build_package
    from portable_evaluation import validate_package


def execute_export(request_path,expected_sha256,emit,*,check_cancelled=lambda:None):
    check_cancelled();hash_value(expected_sha256)
    path=Path(os.path.abspath(os.fspath(request_path)));require(path.name=='request.json','Export request must be named request.json')
    fd=directory(str(path.parent))
    try:
        info=os.fstat(fd);raw=read_regular(fd,'request.json',256000)
        require(sha(raw)==expected_sha256,'Export request changed')
        request=validate_request(object_json(raw))
        require(path.parent.name==request['exportID'] and path.parent.parent.name==request['projectID'] and path.parent.parent.parent.name=='workflow-evaluation-exports','Export staging path differs from request')
        source=directory(request['evaluationDirectory'])
        try:
            manifest_raw=read_nested(source,'artifacts/manifest.json',2000000)
            require(sha(manifest_raw)==request['evaluationManifestSHA256'],'Saved evaluation manifest changed')
            source_manifest=object_json(manifest_raw)
            require(source_manifest.get('evaluationID')==request['evaluationID'] and source_manifest.get('projectID')==request['projectID'],'Saved evaluation identity differs from export request')
        finally:os.close(source)
        event_identity={key:request[key] for key in ('exportID','requestIdentity')}
        check_cancelled();emit(dict(event='exportStarted',**event_identity))
        destination=path.parent/'package'
        built=build_package(request['evaluationDirectory'],destination,request['evaluationManifestSHA256'],include_test_rows=request['includeTestRows'],check_cancelled=check_cancelled)
        check_cancelled();verified,_,_,_=validate_package(destination)
        require(verified==built,'Staged package changed during verification')
        package_manifest=read_nested(fd,'package/manifest.json',2000000)
        require(object_json(package_manifest)==built,'Package receipt changed')
        require(read_regular(fd,'request.json',256000)==raw,'Export request changed during staging')
        current=directory(str(path.parent))
        try:
            after=os.fstat(current);require((info.st_dev,info.st_ino)==(after.st_dev,after.st_ino),'Export staging directory changed')
        finally:os.close(current)
        check_cancelled()
        emit(dict(event='exportCompleted',**event_identity,manifestSHA256=sha(package_manifest),fileCount=len(built['files'])+1,totalBytes=sum(e['bytes'] for e in built['files'])+len(package_manifest),includesTestRows=request['includeTestRows']))
        return built
    finally:os.close(fd)
