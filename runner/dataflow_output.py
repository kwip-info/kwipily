"""Private grants, bounded local codecs, and atomic Table Output transactions."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from datetime import datetime, timezone
if __package__:
 from .table_output_writers import write_table,options,MAX_BYTES
else:
 from table_output_writers import write_table,options,MAX_BYTES
MAX_OUTPUT_BYTES=MAX_BYTES

class OutputCommitError(ValueError):
 def __init__(self,node,cause):super().__init__(cause);self.node=node

def atomic_rename(fd,source,target,exchange=False):
 """Exclusive publish or atomic exchange, never unchecked replacement."""
 libc=ctypes.CDLL(None,use_errno=True)
 if sys.platform=='darwin':fn=libc.renameatx_np;flags=2 if exchange else 4
 elif sys.platform.startswith('linux'):fn=libc.renameat2;flags=2 if exchange else 1
 else:raise ValueError('Atomic Table Output publication is unsupported on this platform')
 fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 if fn(fd,os.fsencode(source),fd,os.fsencode(target),flags):
  error=ctypes.get_errno();raise OSError(error,os.strerror(error))

def identity(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)

def safe_name(name):
 if type(name) is not str or not name or len(name)>200 or Path(name).name!=name or name in ('.','..') or '/' in name or '\\' in name or '\x00' in name:raise ValueError('Output filename must be a single safe filename, without a directory')
 return name

class OutputTransaction:
 def __init__(self,bindings,check_cancelled=lambda:None,image_reader=None):
  self.bindings=bindings;self.check=check_cancelled;self.image_reader=image_reader;self.staged=[];self.committed=False
 def stage(self,node,table,c):
  if c.get('destination') in ('database','api'):
   if not getattr(self,'defer_database_writes',False):raise ValueError('Database output requires the application broker and explicit private write authority. This preparation export cannot perform database writes.')
   return
  self.check();fmt,_=options(c)
  if node not in self.bindings:raise ValueError('Output requires an explicitly granted destination directory')
  directory=self.bindings[node]
  if not isinstance(directory,str) or len(directory)>4096 or not Path(directory).is_absolute():raise ValueError('Output destination binding must be an absolute granted directory')
  path=Path(directory)
  if path.resolve()!=path or not path.is_dir():raise ValueError('Output destination must be an existing directory without symlinks')
  name=safe_name(c['filename'])
  if any(item['path']==path/name for item in self.staged):raise ValueError('Output nodes cannot write the same destination file')
  chosen=c['columns'] or [column['name'] for column in table.columns]
  if not chosen or len(set(chosen))!=len(chosen):raise ValueError('Output requires unique included columns')
  indexes=[table.index(column) for column in chosen];columns=[dict(table.columns[j]) for j in indexes]
  if any(column['dtype'] in ('audio','tensor') for column in columns):raise ValueError('Audio/prepared tensors require an owned media export adapter; select scalar columns for Table Output.')
  images=any(column['dtype']=='image' for column in columns);policy=c.get('imagePolicy','stop')
  if policy not in ('stop','copyAssets'):raise ValueError('Unsupported image export policy')
  if images and policy=='stop':raise ValueError('Image columns require explicit copied-assets export bundle policy')
  if images and self.image_reader is None:raise ValueError('Copied image assets require a captured image byte reader; private paths are not authority')
  if images and c['overwrite']:raise ValueError('Image export bundles require a new destination name; existing bundles cannot be replaced')
  descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);temporary='.kwip-output-'+secrets.token_hex(16)
  item={'node':node,'fd':descriptor,'temporary':temporary,'name':name,'path':path/name,'overwrite':c['overwrite'],'backup':None,'installed':False,'bundle':images,'children':[]}
  self.staged.append(item);item['targetIdentity']=self._check_target(item)
  work_fd=descriptor;asset_manifest=[];bundle_fd=None
  try:
   if images:
    os.mkdir(temporary,mode=0o700,dir_fd=descriptor);bundle_fd=os.open(temporary,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor);work_fd=bundle_fd
    os.mkdir('assets',mode=0o700,dir_fd=work_fd);item['children'].append('assets')
   file_name='table.'+fmt if images else temporary
   stream_fd=os.open(file_name,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600,dir_fd=work_fd)
   if images:item['children'].append(file_name)
   copied={};asset_bytes=0
   def rows():
    nonlocal asset_bytes
    for row in table.rows:
     values=[]
     for j in indexes:
      value=row[j]
      if table.columns[j]['dtype']=='image' and value is not None:
       if __package__:from .image_artifacts import validate_reference,decode_pixels
       else:from image_artifacts import validate_reference,decode_pixels
       reference=validate_reference(value);digest=reference['sha256'];relative='assets/'+digest+'.kwipimg'
       if digest not in copied:
        self.check();raw=self.image_reader(reference);header,_=decode_pixels(raw,expected_sha256=digest)
        if any(header[k]!=reference[k] for k in ('width','height','channels','dtype','layout')):raise ValueError('Captured image geometry differs from its reference')
        asset_bytes+=len(raw)
        if asset_bytes>MAX_BYTES:raise ValueError('Image output bundle exceeds 50 MB budget')
        af=os.open(relative,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600,dir_fd=work_fd)
        item['children'].append(relative)
        with os.fdopen(af,'wb') as asset:asset.write(raw);asset.flush();os.fsync(asset.fileno())
        copied[digest]=relative;asset_manifest.append({'path':relative,'sha256':digest,'bytes':len(raw),'reference':reference})
       value=relative
      values.append(value)
     yield values
   encoded_columns=[{**col,'dtype':'string','exportedImageReferences':True} if col['dtype']=='image' else col for col in columns] if images else columns
   with os.fdopen(stream_fd,'w+b') as stream:
    write_table(stream,table,indexes,c,self.check,rows=rows(),columns=encoded_columns);stream.flush();os.fsync(stream.fileno());size=stream.tell();stream.seek(0);hasher=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1048576),b''):self.check();hasher.update(chunk)
   receipt={'nodeID':node,'path':str(path/name),'destinationName':name,'format':fmt,'rowCount':len(table.rows),'columns':columns,'sourceDigest':table.source_digest,'dataVersion':table.data_version,'partition':table.role,'sha256':hasher.hexdigest(),'bytes':size+asset_bytes,'status':'saved','completeSelectedTable':True,'savedAt':datetime.now(timezone.utc).isoformat()}
   if images:
    manifest={'schemaVersion':1,'table':{'path':file_name,'sha256':hasher.hexdigest(),'bytes':size},'assets':asset_manifest,'columns':columns}
    raw=json.dumps(manifest,sort_keys=True,allow_nan=False).encode();mf=os.open('manifest.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600,dir_fd=work_fd)
    item['children'].append('manifest.json')
    with os.fdopen(mf,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    receipt['assetBundleManifest']=manifest;receipt['bytes']+=len(raw)
    assets_fd=os.open('assets',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=work_fd)
    try:os.fsync(assets_fd)
    finally:os.close(assets_fd)
    os.fsync(work_fd)
   if receipt['bytes']>MAX_BYTES:raise ValueError('Table Output bundle exceeds 50 MB output budget')
   item['receipt']=receipt;self.check()
  finally:
   if bundle_fd is not None:os.close(bundle_fd)
 def _check_target(self,item):
  try:info=os.stat(item['name'],dir_fd=item['fd'],follow_symlinks=False)
  except FileNotFoundError:return None
  if not stat.S_ISREG(info.st_mode):raise ValueError('Output target must be a regular file, never a symlink or directory')
  if not item['overwrite']:raise ValueError('Output file already exists; choose another filename or explicitly allow overwrite')
  return identity(info)
 def commit(self):
  item=None
  try:
   for item in self.staged:
    self.check()
    if self._check_target(item)!=item['targetIdentity']:raise ValueError('Output destination changed after preparation; retry with a new filename')
   for item in self.staged:
    self.check();fd=item['fd'];name=item['name']
    if item['targetIdentity'] is not None:
     atomic_rename(fd,item['temporary'],name,exchange=True)
     old=os.stat(item['temporary'],dir_fd=fd,follow_symlinks=False)
     if identity(old)!=item['targetIdentity']:
      atomic_rename(fd,item['temporary'],name,exchange=True);raise ValueError('Output overwrite race detected; original destination preserved')
     item['backup']=item['temporary'];item['temporary']=None
    else:atomic_rename(fd,item['temporary'],name)
    item['installed']=True;item['installedIdentity']=os.stat(name,dir_fd=fd,follow_symlinks=False);os.fsync(fd)
   self.check();self.committed=True
   for item in self.staged:item['receipt']['savedAt']=datetime.now(timezone.utc).isoformat()
   return [item['receipt'] for item in self.staged]
  except (ValueError,OSError) as error:
   self.rollback();raise OutputCommitError(item['node'] if item else '',str(error)) from error
  except BaseException:self.rollback();raise
 def _remove(self,item,name):
  if item['bundle']:
   directory=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=item['fd'])
   try:
    for child in reversed(item['children']):
     try:
      if child=='assets':os.rmdir(child,dir_fd=directory)
      else:os.unlink(child,dir_fd=directory)
     except FileNotFoundError:pass
   finally:os.close(directory)
   os.rmdir(name,dir_fd=item['fd'])
  else:os.unlink(name,dir_fd=item['fd'])
 def rollback(self):
  for item in reversed(self.staged):
   if not item['installed']:continue
   try:current=os.stat(item['name'],dir_fd=item['fd'],follow_symlinks=False)
   except FileNotFoundError:current=None
   owned=item.get('installedIdentity')
   if current is None or owned is None or (current.st_dev,current.st_ino)!=(owned.st_dev,owned.st_ino):item['installed']=False;continue
   if item['backup']:
    atomic_rename(item['fd'],item['backup'],item['name'],exchange=True);os.unlink(item['backup'],dir_fd=item['fd']);item['backup']=None
   else:self._remove(item,item['name'])
   item['installed']=False
 def close(self):
  if not self.committed:self.rollback()
  for item in self.staged:
   for name in (item['temporary'],item['backup']):
    if name:
     try:self._remove(item,name)
     except FileNotFoundError:pass
   os.close(item['fd'])
  self.staged=[]
