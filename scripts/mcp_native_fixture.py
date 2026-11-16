#!/usr/bin/env python3
"""Create a new, isolated DEBUG native MCP acceptance bundle. Never owner grants.
The normal POC app and production app are untouched. Tokens remain private files.
"""
import argparse,json,os,plistlib,secrets,shutil,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def assemble():
    source=ROOT/'.local/KwipML POC.app'
    parent=ROOT/'.local'/('mcp-native-fixture-'+str(uuid.uuid4()))
    parent.mkdir(mode=0o700);app=parent/'Kwip ML MCP Test.app';shutil.copytree(source,app)
    info=app/'Contents/Info.plist';value=plistlib.loads(info.read_bytes())
    if value['CFBundleIdentifier']!='info.kwip.KwipML.POC':raise ValueError('Only a development POC bundle is allowed')
    value['KwipMLMCPFixtureDirectory']=str(parent);info.write_bytes(plistlib.dumps(value))
    client=str(uuid.uuid4());credential=parent/'fixture-credentials.json'
    credential.write_text(json.dumps({client:secrets.token_urlsafe(32)}));credential.chmod(0o600)
    directory=parent/'mcp';directory.mkdir(mode=0o700)
    actions=['catalog_list','catalog_get','projects_list','project_create','project_open','project_save','workflow_inspect','workflow_validate','patch_preview','patch_apply','patch_undo','run_start','run_cancel','jobs_list','job_get','results_read','lesson_read']
    policy={'schemaVersion':1,'enabled':True,'disabledActions':[],'disabledComponents':['kwip.data.outputDelimited','workflow.saveModel'],'projects':{},'clients':[{'id':client,'name':'Disposable local acceptance client','actions':actions,'disabledComponents':[]}],'auditRetention':50}
    file=directory/'settings.json';file.write_text(json.dumps(policy));file.chmod(0o600)
    subprocess.run(['/usr/bin/codesign','--force','--deep','--preserve-metadata=entitlements','--sign','-',str(app)],capture_output=True,check=True)
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(app)],capture_output=True,check=True)
    receipt=parent/'fixture.json';receipt.write_text(json.dumps({'app':str(app),'clientID':client,'storage':str(parent),'synthetic':True},indent=2)+'\n');receipt.chmod(0o600)
    print(json.dumps({'app':str(app),'receipt':str(receipt),'ownerDataUsed':False}))
if __name__=='__main__':assemble()
