#!/usr/bin/env python3
"""Actual signed downloader XPC redirect probe against a synthetic loopback origin."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import threading
from sparkle_build import ROOT,DEVELOPER,output_path
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--identity',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=output_path(ROOT,args.output);out.mkdir(parents=True,mode=0o700)
    app=out/'Downloader Probe.app';contents=app/'Contents';(contents/'MacOS').mkdir(parents=True);(contents/'XPCServices').mkdir()
    plist={'CFBundleIdentifier':'info.kwip.KwipML','CFBundleExecutable':'Probe','CFBundleName':'Kwip ML downloader probe','CFBundlePackageType':'APPL','CFBundleVersion':'1','LSMinimumSystemVersion':'14.0'}
    (contents/'Info.plist').write_bytes(plistlib.dumps(plist))
    service=contents/'XPCServices/Downloader.xpc'
    shutil.copytree(args.framework/'Versions/B/XPCServices/Downloader.xpc',service)
    def run(command):
        return subprocess.run(command,check=True,capture_output=True,text=True,timeout=60,env={**os.environ,'DEVELOPER_DIR':DEVELOPER})
    source=args.source.resolve()
    run(['xcrun','clang','-fobjc-arc','-DBUILDING_SPARKLE_SOURCES_EXTERNALLY=1','-framework','Foundation',
      '-I'+str(source/'Sparkle'),'-I'+str(source/'Downloader'),str(ROOT/'packaging/sparkle/downloader-probe.m'),
      str(source/'Sparkle/SPUDownloadData.m'),'-o',str(contents/'MacOS/Probe')])
    for target,ent in [(service,source/'Downloader/Downloader.entitlements'),(app,ROOT/'packaging/App.entitlements')]:
        run(['/usr/bin/codesign','--force','--sign',args.identity,'--options','runtime','--timestamp','--entitlements',str(ent),str(target)])
    run(['/usr/bin/codesign','--verify','--deep','--strict',str(app)])
    counts={'ok':0,'redirect':0,'trap':0,'authorized':0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            name=self.path[1:]
            if name not in counts:self.send_error(404);return
            counts[name]+=1
            if self.headers.get('Authorization')=='Bearer synthetic-probe-not-a-real-credential':counts['authorized']+=1
            if name=='redirect':
                self.send_response(302);self.send_header('Location',f'http://127.0.0.1:{self.server.server_port}/trap');self.send_header('Content-Length','0');self.end_headers()
            else:
                data=b'local-probe-bytes';self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        results=[]
        for endpoint in ['ok','redirect']:
            result=run([str(contents/'MacOS/Probe'),f'http://127.0.0.1:{server.server_port}/{endpoint}'])
            results.append(json.loads(result.stdout))
        passed=results[0]=={'completed':True,'success':True,'length':17} and results[1]=={'completed':True,'success':False,'length':0} and counts=={'ok':1,'redirect':1,'trap':0,'authorized':2}
        receipt={'passed':passed,'direct':results[0],'redirect':results[1],'requests':counts,'synthetic':True,'publicLaunch':False}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return 0 if passed else 1
    finally:server.shutdown();server.server_close();thread.join(timeout=2)

if __name__=='__main__':raise SystemExit(main())
