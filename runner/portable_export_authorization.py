"""One-shot export staging authority, separate from saved evaluation claims."""
import os
from pathlib import Path

if __package__:
    from .execution_budget import validate_versioned_request
    from .evaluation_authorization import EvaluationAuthority,OWNERSHIP_FIELDS,CLAIM_FIELDS
    from .worker_authorization import require,canonical_uuid,hash_value,directory
else:
    from execution_budget import validate_versioned_request
    from evaluation_authorization import EvaluationAuthority,OWNERSHIP_FIELDS,CLAIM_FIELDS
    from worker_authorization import require,canonical_uuid,hash_value,directory

REQUEST_FIELDS={'schemaVersion','kind','exportID','projectID','evaluationID','requestIdentity','evaluationDirectory','evaluationManifestSHA256','includeTestRows'}


def validate_request(value):
    validate_versioned_request(value,REQUEST_FIELDS)
    require(value['kind']=='portableEvaluationExport','Invalid portable export request')
    for key in ('exportID','projectID','evaluationID'):canonical_uuid(value[key])
    for key in ('requestIdentity','evaluationManifestSHA256'):hash_value(value[key])
    require(type(value['includeTestRows']) is bool,'Invalid protected-row option')
    source=value['evaluationDirectory']
    require(isinstance(source,str) and source.startswith('/') and '\0' not in source and str(Path(source))==source,'Expected canonical saved evaluation directory')
    parts=source.split('/')[1:];require(all(p and p not in ('.','..') for p in parts),'Invalid saved evaluation path')
    location=Path(source)
    require(location.name==value['evaluationID'] and location.parent.name==value['projectID'] and location.parent.parent.name=='workflow-evaluations','Saved evaluation path differs from request identity')
    fd=directory(source)
    os.close(fd)
    return value


class PortableExportAuthority(EvaluationAuthority):
    identity_key='exportID'
    kind='portableEvaluationExport'
    namespace='workflow-evaluation-exports'
    ownership_fields=(OWNERSHIP_FIELDS-{'evaluationID'})|{'exportID'}
    claim_fields=(CLAIM_FIELDS-{'evaluationID'})|{'exportID'}

    def _validate_request(self):
        validate_request(self.request)
        for key in ('exportID','projectID','requestIdentity'):
            require(self.request[key]==self._ownership[key],'Export request differs from retained owner')


def claim_export(folder,request_sha256,ownership_sha256,*,launch_budget_sha256=None):return PortableExportAuthority(folder,request_sha256,ownership_sha256,launch_budget_sha256=launch_budget_sha256)
def inherit_export(folder,request_sha256,ownership_sha256,activity_fd,*,launch_budget_sha256=None):return PortableExportAuthority(folder,request_sha256,ownership_sha256,inherited_fd=activity_fd,launch_budget_sha256=launch_budget_sha256)
