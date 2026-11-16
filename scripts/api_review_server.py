#!/usr/bin/env python3
"""Disposable literal-loopback fixture for native API table review. Synthetic data only.

Run: python3 scripts/api_review_server.py --port 8791 --log .local/api-uat-requests.jsonl
No credentials, external access, uploads or background installation. Stop with Ctrl-C.
"""
import argparse,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8791);parser.add_argument('--log',type=Path,required=True);args=parser.parse_args()
    args.log.parent.mkdir(parents=True,exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def record(self,value):
            with args.log.open('a') as f:f.write(json.dumps(value)+'\n')
        def reply(self,status,body):
            raw=json.dumps(body).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            p=urlsplit(self.path);q=parse_qs(p.query);page=int(q.get('cursor',['0'])[0]);self.record({'method':'GET','path':self.path})
            if p.path!='/orders':self.reply(404,{'error':'not_found'});return
            rows=[dict(order_id=i,customer=['Ada','Grace','Linus'][i%3],total=round(i*1.25,2),created_at='2026-10-19T12:00:00Z') for i in range(page*4+1,min(13,page*4+5))]
            self.reply(200,{'data':{'orders':rows},'paging':{'next_cursor':str(page+1) if page<2 else None}})
        def do_POST(self):
            count=int(self.headers.get('Content-Length','0'))
            if count>1_000_000:self.reply(413,{'error':'too_large'});return
            body=json.loads(self.rfile.read(count));self.record({'method':'POST','path':self.path,'body':body,'idempotencyKey':self.headers.get('Idempotency-Key')})
            if self.path=='/lost':self.connection.shutdown(2);self.connection.close();return
            if self.path!='/predictions':self.reply(404,{'error':'not_found'});return
            self.reply(202,{'result':{'accepted':True,'records':len(body.get('predictions',[])) if isinstance(body,dict) and 'predictions' in body else 1}})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print('Synthetic API review fixture on literal loopback port '+str(server.server_port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
