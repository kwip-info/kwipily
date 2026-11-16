"""One sealed retained pack with deduplicated, independently pinned JSON ranges.

The retention allowance bounds total bytes. Object framing remains bounded even
when that allowance is larger. This module never publishes dataset/event logs.
Table schema, selection and provenance readers are separate clients of this
primitive; a successful range read is not a full inventory audit.
"""
import hashlib,json,os,re,sqlite3,stat,tempfile
if __package__:from .dataflow_disk import ScratchBudget,current_scope
else:from dataflow_disk import ScratchBudget,current_scope

MAX_OBJECT_BYTES=2_097_152
MAGIC=b'KWIP-TABLE-PACK\x00\x01\n'
NAME='tables.kwipdata'


def _name(value):return type(value) is str and (value==NAME or re.fullmatch(r'table-pack-[0-9a-f]{32}\.kwipdata',value) is not None)


def _shape(value,check,depth=0,work=None):
    # One interval covers the complete structural walk. Restarting it for each
    # small nested row/container turns bounded work checks into per-cell full
    # launch-authority walks. Encoding and publication retain their own guards.
    if work is None:work=[0]
    # A bounded flat column needs no recursive Python visit per scalar. Admit
    # the same exact JSON types, escaping bound and work count in one loop;
    # structured values keep the general depth-checked path below.
    if depth==0 and type(value) in (list,tuple) and len(value)<=256:
        maximum=2+2*len(value);flat=True
        for child in value:
            kind=type(child)
            if kind is str:
                if len(child)>MAX_OBJECT_BYTES:raise ValueError('Table-pack text exceeds its object boundary.')
                maximum+=12*len(child)+2
            elif kind is int:maximum+=len(str(child))
            elif kind in (float,bool,type(None)):maximum+=32
            else:flat=False;break
        if flat:
            before=work[0];work[0]+=len(value)+1
            for _ in range(work[0]//256-before//256):check()
            return maximum
    work[0]+=1
    if work[0]%256==0:check()
    if depth>32:raise ValueError('Table-pack object nesting exceeds its frame boundary.')
    if type(value) is str:
        if len(value)>MAX_OBJECT_BYTES:raise ValueError('Table-pack text exceeds its object boundary.')
        return 12*len(value)+2
    if type(value) in (list,tuple,dict):
        if len(value)>MAX_OBJECT_BYTES//2:raise ValueError('Table-pack container exceeds its object boundary.')
        maximum=2+2*len(value)
        for item in value.items() if type(value) is dict else value:
            if type(value) is dict:
                key,child=item
                if type(key) is not str:raise ValueError('Table-pack keys must be text.')
                maximum+=_shape(key,check,depth+1,work)+_shape(child,check,depth+1,work)
            else:maximum+=_shape(item,check,depth+1,work)
        return maximum
    if type(value) is int:return len(str(value))
    if type(value) in (float,bool,type(None)):return 32
    raise ValueError('Unsupported table-pack JSON type.')


def _tokens(value,encoder):
    if type(value) is str:
        yield '"'
        # ASCII escaping can expand a code point to twelve bytes. Never ask the
        # encoder for a complete large string before enforcing the object bound.
        for offset in range(0,len(value),8192):yield encoder.encode(value[offset:offset+8192])[1:-1]
        yield '"'
    elif type(value) in (list,tuple):
        yield '['
        for index,child in enumerate(value):
            if index:yield ','
            yield from _tokens(child,encoder)
        yield ']'
    elif type(value) is dict:
        yield '{'
        for index,key in enumerate(sorted(value)):
            if index:yield ','
            yield from _tokens(key,encoder);yield ':';yield from _tokens(value[key],encoder)
        yield '}'
    else:yield encoder.encode(value)


class EncodingWorkBudget:
    """One owner's bounded encoding work; all I/O still uses the live guard.

    This does not cache a permission decision. It bounds pure structural/JSON
    work between checks across many tiny deduplicated objects. Large fragments,
    storage writes, sealing and publication retain their independent guards.
    """
    def __init__(self,check):
        self.check=check;self.started=False;self.objects=0;self.nodes=0;self.bytes=0
    def boundary(self):
        self.check();self.started=True;self.objects=self.nodes=self.bytes=0
    def begin(self):
        if not self.started or self.objects>=128:self.boundary()
        self.objects+=1
    def structural(self):
        self.nodes+=256
        if self.nodes>=4096:self.boundary()
    def admit(self,maximum,remainder):
        self.nodes+=remainder
        if self.nodes>=4096 or self.bytes+maximum>65_536:self.boundary()
        self.bytes+=maximum


def encode_object(value,check=lambda:None,*,work_budget=None):
    if work_budget is None:
        check();maximum=_shape(value,check)
    else:
        if type(work_budget) is not EncodingWorkBudget or work_budget.check!=check:raise ValueError('Encoding work belongs to a different authority.')
        work_budget.begin();work=[0]
        maximum=_shape(value,work_budget.structural,work=work)
        work_budget.admit(maximum,work[0]%256)
    encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=True)
    if maximum<=MAX_OBJECT_BYTES:
        # The structural walk proves this C-encoded allocation cannot exceed
        # one decoder frame, even under worst-case Unicode escaping. Larger
        # uncertain strings use the bounded fragment encoder below.
        content=bytearray(encoder.encode(value),'ascii')
        if work_budget is None or maximum>65_536:check()
        return content
    content=bytearray()
    checked_bytes=0
    for ordinal,token in enumerate(_tokens(value,encoder)):
        # String fragments are at most98 KiB after escaping. Admit every256
        # tokens or64 KiB encoded work, rather than invoking the full authority
        # walk for each punctuation/scalar token. Publication rechecks ownership.
        if ordinal%256==0 or len(content)-checked_bytes>=65_536:
            check();checked_bytes=len(content)
        raw=token.encode('ascii')
        if len(content)+len(raw)>MAX_OBJECT_BYTES:raise ValueError('Table-pack object exceeds its frame boundary.')
        content.extend(raw)
    check()
    return content


class TableArtifactPack:
    def __init__(self,storage,maximum_bytes,*,name=NAME,buffer_bytes=0):
        if type(maximum_bytes) is not int or maximum_bytes<len(MAGIC) or not _name(name):raise ValueError('Invalid table-pack retention allowance or segment name.')
        if type(buffer_bytes) is not int or buffer_bytes not in (0,262144):raise ValueError('Unsupported physical pack write grain.')
        self.storage=storage;self.maximum=maximum_bytes;self.check=storage.check_cancelled
        self.buffer_bytes=buffer_bytes;self.pending=bytearray();self.position=0
        self.name=name
        self.entered=False;self.closed=False;self.failed=False;self.descriptor=None;self.index=None;self.directory=None;self.reserved=0
    def __enter__(self):
        if self.entered:raise ValueError('Table pack cannot be reopened.')
        self.entered=True;scope=current_scope();self.budget=scope[1] if scope else ScratchBudget()
        self.directory=tempfile.TemporaryDirectory(prefix='.table-pack-index-',dir=scope[0] if scope else None)
        try:
            self.index=sqlite3.connect(self.directory.name+'/objects.sqlite3')
            self.index.execute('PRAGMA journal_mode=OFF');self.index.execute('PRAGMA cache_size=-512')
            self.page_size=self.index.execute('PRAGMA page_size').fetchone()[0]
            self._admit_index();self.index.execute('CREATE TABLE objects (pin TEXT PRIMARY KEY, offset INTEGER, bytes INTEGER)');self._account_index()
            self.context=self.storage.atomic_stream(self.name,self.maximum);self.sink=self.context.__enter__()
            self.hasher=hashlib.sha256();self.sink.write(MAGIC);self.hasher.update(MAGIC);self.position=len(MAGIC)
            return self
        except BaseException as error:
            self.failed=True;self.closed=True
            try:
                if hasattr(self,'sink'):self.context.__exit__(type(error),error,error.__traceback__)
            finally:self._cleanup()
            raise
    def _admit_index(self):
        pages=(self.budget.limit-self.budget.used+self.reserved)//self.page_size
        if pages<3:raise ValueError('Table-pack index exceeds its shared scratch allowance.')
        self.index.execute('PRAGMA max_page_count='+str(pages))
    def _account_index(self):
        current=self.index.execute('PRAGMA page_count').fetchone()[0]*self.page_size
        self.budget.resize(self.reserved,current);self.reserved=current
    def _flush(self):
        if self.pending:
            content=self.pending;self.pending=bytearray()
            # A failing sink's traceback may retain its memoryview. Detach the
            # bounded buffer rather than resizing that exported bytearray.
            self.sink.write(content)
    def _write(self,content):
        # Pending bytes are bounded working memory, not published artifacts.
        # The guarded sink still reserves the shared retained budget BEFORE I/O.
        if self.position+len(content)>self.maximum:raise ValueError('Table pack exceeds its retained file allowance.')
        if self.pending and len(self.pending)+len(content)>self.buffer_bytes:self._flush()
        if len(content)>=self.buffer_bytes:self.sink.write(content)
        else:self.pending.extend(content)
        self.position+=len(content);self.hasher.update(content)
    def add_json(self,value):
        if not self.entered or self.closed or self.failed:raise ValueError('Table pack is not writable.')
        try:
            content=encode_object(value,self.check)
            return self._append_encoded(content)
        except BaseException:self.failed=True;raise
    def _append_encoded(self,content):
        """Private append of bytes from the shared bounded canonical encoder.

        SegmentWriter already encoded the object for its deduplication lookup.
        No raw-byte append is exposed through a workflow or tool interface.
        """
        if not self.entered or self.closed or self.failed:raise ValueError('Table pack is not writable.')
        try:
            if type(content) is not bytearray or not 1<=len(content)<=MAX_OBJECT_BYTES:raise ValueError('Invalid encoded table-pack frame.')
            pin=hashlib.sha256(content).hexdigest()
            existing=self.index.execute('SELECT offset,bytes FROM objects WHERE pin=?',(pin,)).fetchone()
            if existing is None:
                offset=self.position;size=len(content)
                self._admit_index();self.index.execute('INSERT INTO objects VALUES (?,?,?)',(pin,offset,size));self._account_index()
                self._write(content)
            else:offset,size=existing
            return {'sha256':pin,'offset':offset,'bytes':size}
        except BaseException:self.failed=True;raise
    def _cleanup(self):
        if self.index is not None:self.index.close();self.index=None
        if self.reserved:self.budget.resize(self.reserved,0);self.reserved=0
        if self.directory is not None:self.directory.cleanup();self.directory=None
        self.pending.clear()
    def __exit__(self,kind,error,traceback):
        poisoned=kind is None and self.failed
        if poisoned:kind=ValueError;error=ValueError('Failed table-pack transaction cannot be published.')
        try:
            if kind is None:
                try:self._flush()
                except BaseException as failure:
                    self.context.__exit__(type(failure),failure,failure.__traceback__);raise
            size=self.sink.tell()
            self.context.__exit__(kind,error,traceback)
            if kind is None:
                self.descriptor={'schemaVersion':1,'kind':'tableArtifactPack','path':self.name,'bytes':size,'sha256':self.hasher.hexdigest()}
        finally:self.closed=True;self._cleanup()
        if poisoned:raise error
        return False


class TableArtifactPackReader:
    def __init__(self,root,descriptor,*,maximum_bytes,check_cancelled=lambda:None):
        self.root=os.fspath(root);self.check=check_cancelled;self.root_fd=None;self.fd=None
        self.validate_descriptor(descriptor,maximum_bytes)
        self.descriptor=dict(descriptor)
        self.name=descriptor['path']
        try:
            self.root_fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
            self.root_identity=self._identity(os.fstat(self.root_fd))
            self.fd=os.open(self.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=self.root_fd)
            info=os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size!=descriptor['bytes']:raise ValueError('Table pack must match its pinned regular file.')
            self.fingerprint=self._fingerprint(info)
            if os.pread(self.fd,len(MAGIC),0)!=MAGIC:raise ValueError('Unsupported table-pack framing.')
            self._guard()
        except (OSError,ValueError):
            self.close();raise ValueError('Table pack is not an unchanged pinned regular file.') from None
    @staticmethod
    def validate_descriptor(descriptor,maximum_bytes):
        if (type(maximum_bytes) is not int or type(descriptor) is not dict
            or set(descriptor)!={'schemaVersion','kind','path','bytes','sha256'}
            or type(descriptor['schemaVersion']) is not int or descriptor['schemaVersion']!=1
            or descriptor['kind']!='tableArtifactPack' or not _name(descriptor['path'])
            or type(descriptor['bytes']) is not int or not len(MAGIC)<=descriptor['bytes']<=maximum_bytes
            or type(descriptor['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',descriptor['sha256']) is None):
            raise ValueError('Invalid or oversized pinned table pack.')
    @staticmethod
    def _identity(info):return info.st_dev,info.st_ino
    @staticmethod
    def _fingerprint(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
    def _guard(self):
        self.check()
        if self.fd is None:raise ValueError('Table-pack reader is closed.')
        try:
            root=os.stat(self.root,follow_symlinks=False);named=os.stat(self.name,dir_fd=self.root_fd,follow_symlinks=False)
            if (not stat.S_ISDIR(root.st_mode) or self._identity(root)!=self.root_identity
                or not stat.S_ISREG(named.st_mode) or self._fingerprint(named)!=self.fingerprint
                or self._fingerprint(os.fstat(self.fd))!=self.fingerprint):raise ValueError('Table-pack identity changed.')
        except OSError:raise ValueError('Table-pack identity changed.') from None
    def audit(self):
        """Full retained-file verification without a whole-pack allocation."""
        self._guard();hasher=hashlib.sha256();offset=0;total=self.descriptor['bytes']
        while offset<total:
            self.check();block=os.pread(self.fd,min(262144,total-offset),offset)
            if not block:raise ValueError('Table-pack bytes changed.')
            hasher.update(block);offset+=len(block)
        self._guard()
        if hasher.hexdigest()!=self.descriptor['sha256']:raise ValueError('Table-pack pin differs.')
    def read_json(self,reference):
        self._guard()
        if (type(reference) is not dict or set(reference)!={'offset','bytes','sha256'}
            or type(reference['offset']) is not int or type(reference['bytes']) is not int
            or not len(MAGIC)<=reference['offset'] or not 1<=reference['bytes']<=MAX_OBJECT_BYTES
            or reference['offset']+reference['bytes']>self.descriptor['bytes']
            or type(reference['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',reference['sha256']) is None):
            raise ValueError('Invalid bounded table-pack range.')
        content=bytearray();offset=reference['offset'];remaining=reference['bytes']
        while remaining:
            self.check();block=os.pread(self.fd,min(262144,remaining),offset)
            if not block:raise ValueError('Table-pack range changed.')
            content.extend(block);remaining-=len(block);offset+=len(block)
        self._guard()
        if hashlib.sha256(content).hexdigest()!=reference['sha256']:raise ValueError('Table-pack range pin differs.')
        def unique(pairs):
            result={}
            for key,value in pairs:
                if key in result:raise ValueError('Duplicate table-pack JSON field.')
                result[key]=value
            return result
        def constant(_):raise ValueError('Nonfinite table-pack value.')
        try:value=json.loads(content,object_pairs_hook=unique,parse_constant=constant)
        except (ValueError,RecursionError,UnicodeError):raise ValueError('Invalid table-pack JSON.') from None
        _shape(value,self.check);return value
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
        if self.root_fd is not None:os.close(self.root_fd);self.root_fd=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def __del__(self):self.close()
