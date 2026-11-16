#!/usr/bin/env python3
"""Independent synthetic HTTP client for the signed app/XPC/SDK transport gate.
No token, request authorization header or private project content is logged.
"""
import argparse,hashlib,json,os,plistlib,secrets,selectors,subprocess,time,urllib.request,urllib.error
from pathlib import Path

def verify(executable,receipt):
    app=Path(executable).resolve().parents[2]
    helper=app/'Contents/XPCServices/MCPService.xpc'
    def observed(target):
        result=subprocess.run(['/usr/bin/codesign','--display','--entitlements',':-',str(target)],check=True,capture_output=True,timeout=15)
        return plistlib.loads(result.stdout)
    app_authority=observed(app);helper_authority=observed(helper)
    assert app_authority=={'com.apple.security.app-sandbox':True,'com.apple.security.files.user-selected.read-write':True,'com.apple.security.files.bookmarks.app-scope':True}
    assert helper_authority=={'com.apple.security.app-sandbox':True,'com.apple.security.network.server':True}
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(app)],check=True,capture_output=True,timeout=15)
    token=secrets.token_urlsafe(32)
    setup={'fixedPort':None,'clients':[{'id':'probe','tokenSHA256':hashlib.sha256(token.encode()).hexdigest()}]}
    process=subprocess.Popen([str(executable)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    assertions=[]
    try:
        process.stdin.write(json.dumps(setup));process.stdin.close()
        selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
        if not selector.select(15):raise RuntimeError('Signed sandbox listener did not start within 15 seconds')
        line=process.stdout.readline();value=json.loads(line);url=value['url']
        def rpc(body,extra=None,expected=200):
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'}
            headers.update(extra or {})
            request=urllib.request.Request(url,json.dumps(body).encode(),headers,method='POST')
            try:
                with urllib.request.urlopen(request,timeout=10) as response:
                    status=response.status;raw=response.read(1048577)
            except urllib.error.HTTPError as error:status=error.code;raw=error.read(1048577)
            if status!=expected:raise AssertionError('Unexpected HTTP status '+str(status))
            if expected==200:return json.loads(raw)
        init=rpc({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'Synthetic independent HTTP client','version':'1'}}})
        assert init['result']['serverInfo']['name']=='Kwip ML';assertions.append('SDK initialize')
        for _ in range(3):
            reconnect=rpc({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'Reconnected independent client','version':'1'}}})
            assert reconnect['result']['serverInfo']['name']=='Kwip ML'
        assertions.append('repeated independent initialize with the same credential')
        rpc({'jsonrpc':'2.0','method':'notifications/initialized'},expected=202)
        tools=rpc({'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}})
        assert tools['result']['tools'][0]['name']=='probe_inspect';assertions.append('signed XPC tools/list roundtrip')
        result=rpc({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'probe_inspect','arguments':{}}})
        assert result['result']['structuredContent']['sandboxAppRoundtrip'] is True;assertions.append('signed XPC tools/call roundtrip')
        for headers in [{'Authorization':'Bearer '+secrets.token_urlsafe(32)},{'Origin':'https://evil.example'},{'Host':'evil.example:'+str(value['port'])}]:
            rpc({'jsonrpc':'2.0','id':4,'method':'tools/list','params':{}},headers,expected=401 if 'Authorization' in headers else 403)
        assertions.append('wrong token, foreign Origin and Host refused')
        receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.write_text(json.dumps({'passed':True,'assertions':assertions,'port':value['port'],'appSandbox':app_authority['com.apple.security.app-sandbox'],'observedAppEntitlements':app_authority,'observedHelperEntitlements':helper_authority,'helperLocalListenerOnly':True,'scope':'Synthetic ad-hoc signed development probe; not production identity or real canvas UAT'},indent=2)+'\n')
        print(json.dumps({'passed':True,'assertions':assertions}))
    finally:
        selector.close()
        if process.poll() is None:process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        process.stdout.close();process.stderr.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();verify(a.executable,a.receipt)
