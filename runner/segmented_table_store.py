"""Live immutable segments with bounded range reads and owner-relative links.

A segment is published only after successful sealing. References inside it may
name their own segment relatively, avoiding self-referential content hashes.
Opening a selected range is not a complete retained inventory audit.
"""
import collections,hashlib,sqlite3,tempfile,uuid,weakref
if __package__:
    from .table_artifact_pack import TableArtifactPack,TableArtifactPackReader,encode_object,EncodingWorkBudget
    from .dataflow_disk import ScratchBudget,current_scope
else:
    from table_artifact_pack import TableArtifactPack,TableArtifactPackReader,encode_object,EncodingWorkBudget
    from dataflow_disk import ScratchBudget,current_scope


class StoredReference:
    def __init__(self,owner,reference):self.owner=owner;self.object=dict(reference)
    def reference(self,current_owner=None):
        if current_owner is self.owner:return {'pack':None,'object':dict(self.object)}
        if self.owner.descriptor is None:raise ValueError('An unsealed segment cannot be referenced externally.')
        return {'pack':dict(self.owner.descriptor),'object':dict(self.object)}


class SegmentWriter:
    def __init__(self,storage,*,maximum_bytes):
        self.storage=storage;self.maximum=maximum_bytes;self.active=None;self.descriptors=[];self.closed=False;self.failed=False
        self.encoding_work=EncodingWorkBudget(storage.check_cancelled)
        self.directory=None;self.index=None;self.reserved=0;self.owners={};self.captured_tables=weakref.WeakKeyDictionary()
    def __enter__(self):
        scope=current_scope();self.budget=scope[1] if scope else ScratchBudget()
        self.directory=tempfile.TemporaryDirectory(prefix='.segments-index-',dir=scope[0] if scope else None)
        try:
            self.index=sqlite3.connect(self.directory.name+'/objects.sqlite3')
            self.index.execute('PRAGMA journal_mode=OFF');self.index.execute('PRAGMA cache_size=-512')
            self.page_size=self.index.execute('PRAGMA page_size').fetchone()[0]
            self._admit();self.index.execute('CREATE TABLE objects (pin TEXT PRIMARY KEY, segment TEXT, offset INTEGER, bytes INTEGER)');self._account()
            return self
        except BaseException:self._cleanup();raise
    def _admit(self):
        pages=(self.budget.limit-self.budget.used+self.reserved)//self.page_size
        if pages<3:raise ValueError('Segment index exceeds its shared scratch allowance.')
        self.index.execute('PRAGMA max_page_count='+str(pages))
    def _account(self):
        size=self.index.execute('PRAGMA page_count').fetchone()[0]*self.page_size
        self.budget.resize(self.reserved,size);self.reserved=size
    def relative(self,reference):return reference.reference(self.active)
    def add_json(self,value):
        if self.failed:raise ValueError('A failed segment transaction cannot continue.')
        try:return self._add_json(value)
        except BaseException:self.failed=True;raise
    def _add_json(self,value):
        if self.closed or self.index is None:raise ValueError('Segment writer is closed.')
        # Check the bounded canonical pin before opening/writing another segment.
        # Duplicates therefore never spend the remaining retention allowance.
        content=encode_object(value,self.storage.check_cancelled,work_budget=self.encoding_work)
        pin=hashlib.sha256(content).hexdigest()
        previous=self.index.execute('SELECT segment,offset,bytes FROM objects WHERE pin=?',(pin,)).fetchone()
        if previous is not None:
            segment,offset,size=previous
            return StoredReference(self.owners[segment],{'sha256':pin,'offset':offset,'bytes':size})
        if self.active is None:
            if __package__:from .table_identity import enabled as deferred_identity
            else:from table_identity import enabled as deferred_identity
            self.active=TableArtifactPack(self.storage,self.maximum,name='table-pack-'+uuid.uuid4().hex+'.kwipdata',buffer_bytes=262144 if deferred_identity() else 0).__enter__()
            self.active_count=0
        raw=self.active._append_encoded(content)
        self._admit();self.index.execute('INSERT INTO objects VALUES (?,?,?,?)',(raw['sha256'],self.active.name,raw['offset'],raw['bytes']));self._account()
        self.owners[self.active.name]=self.active;self.active_count+=1
        return StoredReference(self.active,raw)
    def seal(self):
        if self.closed:raise ValueError('Segment writer is closed.')
        if self.failed:raise ValueError('A failed segment transaction cannot be published.')
        try:self.encoding_work.boundary()
        except BaseException:self.failed=True;raise
        if self.active is None:return
        active=self.active;self.active=None
        if self.active_count:
            active.__exit__(None,None,None);self.descriptors.append(dict(active.descriptor))
        else:active.__exit__(ValueError,ValueError('Empty segment discarded.'),None)
    def _cleanup(self):
        if self.index is not None:self.index.close();self.index=None
        if self.reserved:self.budget.resize(self.reserved,0);self.reserved=0
        if self.directory is not None:self.directory.cleanup();self.directory=None
    def __exit__(self,kind,error,traceback):
        poisoned=kind is None and self.failed
        if poisoned:kind=ValueError;error=ValueError('A failed segment transaction cannot be published.')
        try:
            if kind is None:
                try:self.seal()
                except BaseException as failure:
                    if self.active is not None:
                        active=self.active;self.active=None
                        active.__exit__(type(failure),failure,failure.__traceback__)
                    raise
            elif self.active is not None:
                active=self.active;self.active=None;active.__exit__(kind,error,traceback)
        finally:self.closed=True;self._cleanup()
        if poisoned:raise error
        return False


class _BoundResolver:
    def __init__(self,resolver,owner):self.resolver=resolver;self.owner=owner
    def read_node(self,reference):return self.resolver._read(reference,self.owner)
    def read_json(self,reference):return self.read_node(reference)[0]


class SegmentResolver:
    def __init__(self,root,*,maximum_bytes,check_cancelled=lambda:None):
        if type(maximum_bytes) is not int or maximum_bytes<0:raise ValueError('Invalid retained segment allowance.')
        self.root=root;self.maximum=maximum_bytes;self.check=check_cancelled
        self.descriptors={};self.readers=collections.OrderedDict();self.total=0;self.closed=False
    def _reader(self,descriptor):
        if self.closed:raise ValueError('Segment resolver is closed.')
        TableArtifactPackReader.validate_descriptor(descriptor,self.maximum)
        name=descriptor['path'];known=self.descriptors.get(name)
        if known is not None and known!=descriptor:raise ValueError('Conflicting retained segment descriptor.')
        cached=self.readers.pop(name,None)
        if cached is not None:
            cached._guard();self.readers[name]=cached
            return cached
        reader=TableArtifactPackReader(self.root,descriptor,maximum_bytes=self.maximum,check_cancelled=self.check)
        name=reader.descriptor['path'];known=self.descriptors.get(name)
        if known is not None and known!=reader.descriptor:
            reader.close();raise ValueError('Conflicting retained segment descriptor.')
        if known is None:
            if self.total+reader.descriptor['bytes']>self.maximum:
                reader.close();raise ValueError('Retained segments exceed their aggregate allowance.')
            self.descriptors[name]=dict(reader.descriptor);self.total+=reader.descriptor['bytes']
        previous=self.readers.pop(name,None)
        if previous is not None:previous.close()
        self.readers[name]=reader
        while len(self.readers)>4:self.readers.popitem(last=False)[1].close()
        return reader
    def _read(self,reference,owner=None):
        self.check()
        if type(reference) is not dict or set(reference)!={'pack','object'}:raise ValueError('Invalid segment object reference.')
        descriptor=reference['pack']
        if descriptor is None:
            if owner is None:raise ValueError('A relative reference requires its containing segment.')
            descriptor=owner
        reader=self._reader(descriptor)
        value=reader.read_json(reference['object'])
        return value,_BoundResolver(self,reader.descriptor)
    def read_node(self,reference):return self._read(reference)
    def read_json(self,reference):return self.read_node(reference)[0]
    def audit(self):
        # Snapshot the discovered inventory; callers traverse metadata first.
        for descriptor in tuple(self.descriptors.values()):self._reader(descriptor).audit()
    def close(self):
        for reader in self.readers.values():reader.close()
        self.readers.clear();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
