"""Captured workload capacity versus fixed bounded metadata/decoder frames."""
if __package__:from .workload_policy import WorkloadPolicy,LEGACY,current_policy
else:from workload_policy import WorkloadPolicy,LEGACY,current_policy

MANIFEST_BYTES=2_000_000


def inventory_limit(policy):
    # A flat inventory still has a 2 MB framing boundary. Larger inventories
    # require versioned paging rather than raising a JSON decoder allowance.
    return min(MANIFEST_BYTES//32,policy.limit('planNodes')*32+policy.limit('collectionMembers')+128)


class ArtifactLimits:
    def __init__(self,manifest,*,admission_policy=None):
        self.segmented='tableStorageVersion' in manifest
        self.policy=LEGACY
        if self.segmented:
            if type(manifest['tableStorageVersion']) is not int or manifest['tableStorageVersion']!=1:raise ValueError('Unsupported retained table storage version.')
            self.policy=WorkloadPolicy.from_document(manifest.get('workloadPolicy'))
        self.total=min(self.policy.limit('retainedBytes'),(admission_policy or current_policy()).limit('retainedBytes')) if self.segmented else 200_000_000
        self.count=inventory_limit(self.policy) if self.segmented else 300
    def file(self,entry):
        if entry.get('kind')=='tableSegment':
            if not self.segmented:raise ValueError('Historical preparations cannot introduce segmented table data.')
            return self.total
        if self.segmented and entry.get('kind')=='tableViewIndex' and index_name(entry.get('path')):return self.total
        return 50_000_000


def request_policy(request):
    """Read only the explicit versioned launch policy; never infer from files."""
    budget=request.get('executionBudget')
    if budget is None:return None
    if type(budget) is not dict:raise ValueError('Invalid execution budget.')
    resources=budget.get('resources')
    if resources is None:return None
    if type(resources) is not dict:raise ValueError('Invalid execution resources.')
    document=resources.get('workloadPolicy')
    return None if document is None else WorkloadPolicy.from_document(document)


def segment_name(name):
    import re
    return type(name) is str and re.fullmatch(r'table-pack-[0-9a-f]{32}\.kwipdata',name) is not None


def index_name(name):
    import re
    return type(name) is str and re.fullmatch(r'view-[0-9]{3,5}\.sqlite3',name) is not None


def prepared_file_limit(name,policy=None):
    if name=='manifest.json':return MANIFEST_BYTES
    return policy.limit('retainedBytes') if policy is not None and (segment_name(name) or index_name(name)) else 50_000_000
