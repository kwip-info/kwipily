"""Captured native grant to one canonical, same-project, owned campaign directory."""
import os,stat,uuid,hashlib,json,re
from pathlib import Path
if __package__:
 from .worker_authorization import directory,require
else:from worker_authorization import directory,require

def campaign_directory(operation,grant):
 require(type(grant) is dict and set(grant)=={'campaignID','device','inode','action','expectedIndexSHA256','parent'},'Invalid campaign grant.')
 require(type(grant['campaignID']) is str and str(uuid.UUID(grant['campaignID'])).upper()==grant['campaignID'],'Invalid campaign identity.')
 require(grant['parent'] is None or type(grant['parent']) is dict and grant['parent'].get('parent') is None,'Only one captured fork parent is allowed.')
 require(grant['action'] in ('start','resume','extend','fork'),'Invalid campaign action.')
 require(type(grant['device']) is int and type(grant['inode']) is int and grant['inode']>0,'Invalid campaign directory pin.')
 require(grant['expectedIndexSHA256'] is None or type(grant['expectedIndexSHA256']) is str and re.fullmatch(r'[a-f0-9]{64}',grant['expectedIndexSHA256']) is not None,'Invalid captured campaign journal hash.')
 require((grant['action']=='fork')==(grant['parent'] is not None),'Fork requires exactly one captured parent; other actions do not accept a parent.')
 path=Path(operation).parent/grant['campaignID']/'campaign';fd=directory(str(path))
 try:
  s=os.fstat(fd);require(s.st_dev==grant['device'] and s.st_ino==grant['inode'] and s.st_uid==os.geteuid() and not s.st_mode&0o077,'Campaign directory identity/ownership differs.')
 finally:os.close(fd)
 require(path.resolve()==path,'Campaign grant is not canonical.')
 return path
