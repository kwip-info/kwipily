"""Immutable owned replay chunks and atomic iteration checkpoints, never shared History deletion."""
import contextlib,fcntl,hashlib,json,os,shutil,stat,uuid
from pathlib import Path
if __package__:from .game_episodes import require,encoded,digest
else:from game_episodes import require,encoded,digest

class CapacityError(ValueError):pass

def read_owned(root,name,limit):
    if __package__:from .worker_authorization import directory,read_regular
    else:from worker_authorization import directory,read_regular
    fd=directory(str(root))
    try:return read_regular(fd,name,limit)
    finally:os.close(fd)

def read_pinned(root,entry,limit):
    name=entry['path'];require(type(name) is str and '/' not in name and name not in ('.','..'),'Unsafe owned file name.')
    path=Path(root)/name;s=path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==entry['bytes'] and s.st_size<=limit,'Owned artifact size/type differs.')
    raw=read_owned(root,name,limit)
    require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],'Owned artifact hash differs.')
    return raw

def write_file(root,name,raw):
    with (Path(root)/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class ReplayStore:
    def __init__(self,root,*,maximum_positions=25000,maximum_bytes=128*1024*1024,retention='stop',checkpoints=3,maximum_chunks=500):
        require(type(maximum_positions) is int and 1<=maximum_positions<=100000 and type(maximum_bytes) is int and 65536<=maximum_bytes<=2*1024*1024*1024,'Invalid replay bounds.')
        require(retention in ('stop','rolling') and type(checkpoints) is int and 1<=checkpoints<=10,'Invalid replay retention.')
        require(type(maximum_chunks) is int and 1<=maximum_chunks<=1200,'Invalid replay chunk bound.')
        self.maximum_chunks=maximum_chunks
        self.verified={}
        self.root=Path(root);self.maximum_positions=maximum_positions;self.maximum_bytes=maximum_bytes;self.retention=retention;self.keep=checkpoints
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        require(not self.root.is_symlink() and self.root.is_dir(),'Replay requires its owned directory.')
        for n in ('chunks','checkpoints'):
            (self.root/n).mkdir(mode=0o700,exist_ok=True);require(not (self.root/n).is_symlink(),'Replay directory must not be a link.')
    @contextlib.contextmanager
    def exclusive(self):
        fd=os.open(self.root/'writer.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
        finally:os.close(fd)
    def checkpoint(self,name=None):
        if name is None:
            pointer=json.loads(read_owned(self.root,'latest.json',128));name=pointer['checkpoint']
        require(type(name) is str and 12<=len(name)<=18 and name.startswith('iteration-') and name[10:].isdigit(),'Invalid checkpoint name.')
        root=self.root/'checkpoints'/name;require(not root.is_symlink(),'Checkpoint is not a regular directory.')
        raw=read_owned(root,'manifest.json',2_000_000);require(len(raw)<=2_000_000,'Checkpoint manifest exceeds limit.');m=json.loads(raw)
        require(m['kind']=='kwip.self-play.checkpoint.v1' and m['iteration']==int(name[10:]),'Checkpoint identity differs.')
        require(set(m)=={'kind','iteration','pins','replay','state','files'} and type(m['files']) is list and {f['path'] for f in m['files']}=={'model.pt','optimizer.pt','rng.pt'} and len(m['files'])==3,'Invalid resume checkpoint inventory.')
        require(type(m['replay']) is list and len(m['replay'])<=400 and len({c['path'] for c in m['replay']})==len(m['replay']),'Invalid replay inventory.')
        for c in m['replay']:
            require(type(c['count']) is int and 1<=c['count']<=256 and len(c['offsets'])==c['count'] and c['offsets'][0]==0 and all(type(v) is int and 0<=v<c['bytes'] for v in c['offsets']) and c['offsets']==sorted(set(c['offsets'])),'Invalid replay row offsets.')
        for entry in m['files']:read_pinned(root,entry,100_000_000)
        for chunk in m['replay']:read_pinned(self.root/'chunks',chunk,4_000_000)
        return root,m
    def admit(self,positions,previous=None):
        old=[] if previous is None else list(previous['replay']);new=[]
        for start in range(0,len(positions),256):
            rows=positions[start:start+256];raw=b''.join(encoded(p)+b'\n' for p in rows)
            require(len(raw)<=4_000_000,'Replay chunk exceeds bounded row storage.')
            offsets=[];offset=0
            for row in rows:offsets.append(offset);offset+=len(encoded(row))+1
            new.append((raw,dict(path=hashlib.sha256(raw).hexdigest()+'.jsonl',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),count=len(rows),offsets=offsets)))
        chunks=old+[e for _,e in new]
        if self.retention=='rolling':
            while sum(c['count'] for c in chunks)>self.maximum_positions and old:
                removed=old.pop(0);chunks.remove(removed)
        candidates=sorted((p for p in (self.root/'checkpoints').iterdir() if p.name.startswith('iteration-')),key=lambda p:int(p.name[10:]))
        kept=candidates[-(self.keep-1):] if self.keep>1 else []
        kept+= [p for p in candidates if (p/'PINNED').exists() and p not in kept]
        if len(kept)>=18:raise CapacityError('At most 18 retained/pinned iteration boundaries are admitted; unpin an older owned boundary or begin a new run.')
        physical={c['path']:c for c in chunks}
        for path in kept:
            _,m=self.checkpoint(path.name)
            physical.update({c['path']:c for c in m['replay']})
        for path in (self.root/'chunks').iterdir():
            require(not path.is_symlink() and path.is_file() and path.name.endswith('.jsonl'),'Unexpected replay store entry.')
            if path.name not in physical:physical[path.name]=dict(bytes=path.stat().st_size)
        if len(physical)>self.maximum_chunks:raise CapacityError('Replay exceeds its '+str(self.maximum_chunks)+'-chunk owned inventory limit; no unrelated data was deleted.')
        if sum(c['count'] for c in chunks)>self.maximum_positions:
            raise CapacityError('Replay active-position capacity reached; reduce games or use rolling retention. No History was deleted.')
        physical_bytes=sum(c['bytes'] for c in physical.values())
        if physical_bytes>self.maximum_bytes:
            active_bytes=sum(c['bytes'] for c in chunks)
            raise CapacityError(f'Replay capacity reached: the next atomic commit needs {physical_bytes} bytes within a {self.maximum_bytes}-byte allowance ({active_bytes} bytes in next active replay). Rolling needs temporary headroom for the previous recoverable boundary and retained/pinned chunks; these are released only after commit when unreferenced. Reduce per-iteration collection or use an explicitly larger replay allowance for a new run. No History was deleted.')
        return old,new,chunks
    def commit(self,iteration,pins,positions,artifacts,state,previous=None):
        require(type(iteration) is int and 1<=iteration<=1000000 and type(positions) is list,'Invalid checkpoint boundary.')
        require(not (self.root/'checkpoints'/f'iteration-{iteration:02}').exists(),'Iteration is immutable; resume from its committed successor.')
        old,new,chunks=self.admit(positions,previous)
        temp=self.root/'checkpoints'/('.pending-'+uuid.uuid4().hex);temp.mkdir(mode=0o700)
        try:
            for raw,e in new:
                if not (self.root/'chunks'/e['path']).exists():write_file(self.root/'chunks',e['path'],raw)
                read_pinned(self.root/'chunks',e,4_000_000)
            files=[write_file(temp,n,b) for n,b in artifacts.items()]
            m=dict(kind='kwip.self-play.checkpoint.v1',iteration=iteration,pins=pins,replay=chunks,state=state,files=files)
            require(len(encoded(m))<=2_000_000,'Resume manifest exceeds the 2 MB bounded metadata allowance.')
            write_file(temp,'manifest.json',encoded(m));name=f'iteration-{iteration:02}'
            os.rename(temp,self.root/'checkpoints'/name)
            pointer=self.root/('.latest-'+uuid.uuid4().hex);write_file(self.root,pointer.name,encoded(dict(checkpoint=name)))
            os.replace(pointer,self.root/'latest.json')
            # Only this store's unpinned recent checkpoints; explicit pins protect a directory.
            candidates=sorted((p for p in (self.root/'checkpoints').iterdir() if p.name.startswith('iteration-')),key=lambda p:int(p.name[10:]))
            for path in candidates[:-self.keep]:
                if not (path/'PINNED').exists():shutil.rmtree(path)
            referenced=set()
            for path in (self.root/'checkpoints').iterdir():
                if path.name.startswith('iteration-'):
                    _,cm=self.checkpoint(path.name);referenced.update(c['path'] for c in cm['replay'])
            for path in (self.root/'chunks').iterdir():
                if path.name not in referenced and path.name.endswith('.jsonl') and path.is_file() and not path.is_symlink():path.unlink()
            return m
        finally:
            if temp.exists():shutil.rmtree(temp)
    def sample(self,manifest,count,rng):
        chunks=manifest['replay'];total=sum(c['count'] for c in chunks);require(total>0,'No terminal-target replay positions are available.')
        indices=[rng.randrange(total) for _ in range(count)];result=[];verified=set()
        for index in indices:
            for c in chunks:
                if index<c['count']:break
                index-=c['count']
            identity=(self.root/'chunks'/c['path']).stat();pin=(c['sha256'],identity.st_dev,identity.st_ino,identity.st_size,identity.st_mtime_ns,identity.st_ctime_ns)
            if self.verified.get(c['path'])!=pin:read_pinned(self.root/'chunks',c,4_000_000);self.verified[c['path']]=pin
            with (self.root/'chunks'/c['path']).open('rb') as f:f.seek(c['offsets'][index]);raw=f.readline(16385)
            require(len(raw)<=16384 and raw.endswith(b'\n'),'Replay row exceeds bounded query size.');result.append(json.loads(raw))
        return result
