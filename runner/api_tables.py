"""Bounded JSON REST adapter. Only fixed authenticated broker entries dispatch network.

No environment credentials, cookies, proxy discovery, redirects or arbitrary code.
Portable options never carry credentials. DNS answers are checked then connected by
literal address, preserving TLS hostname verification against the original host.
"""
import base64,datetime,hashlib,http.client,ipaddress,json,math,socket,ssl,time,os,sys
from urllib.parse import urlsplit,urlunsplit,urlencode,parse_qsl,urljoin

def verified_tls_context():
    context=ssl.create_default_context()
    # Framework/bundled Python may point at an uninstalled OpenSSL CA directory.
    # Use macOS's shipped roots only when there are no defaults and no explicit
    # CA override. Hostname and certificate verification remain mandatory.
    if sys.platform=='darwin' and not context.get_ca_certs() and not any(k in os.environ for k in ('SSL_CERT_FILE','SSL_CERT_DIR')):
        context.load_verify_locations(cafile='/etc/ssl/cert.pem')
    return context

class APIError(ValueError):pass
DEFAULTS=dict(schemaVersion=1,kind='api',direction='input',url='',method='GET',auth='none',credentialHeader='X-API-Key',query={},headers={},allowLoopback=False,recordsPath='',mapping=[],missing='error',pagination='none',cursorPath='/next_cursor',cursorParameter='cursor',pageParameter='page',pageStart=1,pageSize=100,nextPath='/next',maximumPages=100,maximumBytes=10_000_000,maximumRows=100_000,timeoutSeconds=120,requestTimeout=30,retries=2,requestsPerSecond=5,delivery='row',batchRows=100,arrayPath='',maximumRequests=256,idempotencyHeader='',responsePath='',constants=[])
TYPES=('string','int64','float64','bool','datetime','json')
SENSITIVE=('authorization','cookie','set-cookie','proxy-authorization','token','password','secret','api-key','apikey','api_key','access_key','signature')
def sensitive(name):return any(v in name.lower() for v in SENSITIVE)
def url_check(value,loopback=False):
    try:
        if type(value) is not str:raise APIError("Enter a valid API URL.")
        p=urlsplit(value);port=p.port
        if not p.hostname or p.username is not None or p.password is not None or p.fragment or len(value)>2048:raise APIError('Use a bounded URL without credentials or fragment.')
        try:address=ipaddress.ip_address(p.hostname)
        except ValueError:address=None
        if address is not None and not address.is_global and not (loopback and address.is_loopback):raise APIError('Private network destinations require a supported explicit grant; metadata addresses are not permitted.')
        if p.scheme!='https' and not (p.scheme=='http' and loopback and address is not None and address.is_loopback):raise APIError('Use HTTPS, or explicitly enable literal loopback HTTP for local development.')
        if any(sensitive(k) for k,v in parse_qsl(p.query)):raise APIError('Store secrets in the private credential, not in the URL.')
        if p.query:raise APIError('Use the query grid instead of embedding query parameters in the URL.')
        return p
    except (ValueError,TypeError) as e:raise APIError('Enter a valid API URL.') from e

def options(raw):
    if type(raw) is not dict or set(raw)-set(DEFAULTS):raise APIError('Unsupported API setting.')
    c=DEFAULTS|raw
    for k in ('allowLoopback',):
        if type(c[k]) is not bool:raise APIError('Invalid API permission.')
    for k,low,high in [('schemaVersion',1,1),('maximumPages',1,1000),('maximumBytes',1024,50_000_000),('maximumRows',1,500_000),('timeoutSeconds',1,300),('requestTimeout',1,120),('retries',0,3),('requestsPerSecond',1,50),('batchRows',1,8192),('maximumRequests',1,256),('pageSize',1,8192),('pageStart',0,1_000_000)]:
        if type(c[k]) is not int or not low<=c[k]<=high:raise APIError('API limit outside supported range: '+k)
    for k,choices in dict(kind=['api'],direction=['input','output'],auth=['none','bearer','basic','apiKey'],missing=['error','null'],pagination=['none','cursor','page','offset','nextLink'],delivery=['row','batch']).items():
        if c[k] not in choices:raise APIError('Unsupported API choice: '+k)
    if c['method'] not in (['GET'] if c['direction']=='input' else ['POST','PUT','PATCH','DELETE']):raise APIError('Choose a supported request method.')
    url_check(c['url'],c['allowLoopback'])
    for k in ('headers','query'):
        if type(c[k]) is not dict or len(c[k])>32:raise APIError('Use at most32 request fields.')
        for name,value in c[k].items():
            if type(name) is not str or not name or type(value) is not str or len(name)>128 or len(value)>1024 or any(ch in name+value for ch in '\r\n\0') or sensitive(name):raise APIError('Use bounded nonsecret request fields; credentials belong in private storage.')
            if k=='headers' and (not all(ch.isalnum() or ch=='-' for ch in name) or name.lower() in ('host','content-length','content-type','connection','transfer-encoding','expect','upgrade','accept-encoding')):raise APIError('Reserved HTTP header.')
    for k in ('recordsPath','cursorPath','nextPath','arrayPath','responsePath'):
        if type(c[k]) is not str or len(c[k])>1024 or c[k] and not c[k].startswith('/') or '\0' in c[k]:raise APIError('Use JSON Pointer paths, for example /data/items.')
    for k in ('cursorParameter','pageParameter','credentialHeader','idempotencyHeader'):
        if type(c[k]) is not str or len(c[k])>128 or any(not(ch.isascii() and (ch.isalnum() or ch in '-_')) for ch in c[k]):raise APIError('Use a simple request field name.')
    if not c['cursorParameter'] or not c['pageParameter']:raise APIError('Name the paging query parameter.')
    if c['auth']=='apiKey' and (not c['credentialHeader'] or c['credentialHeader'].lower() in ('host','content-length','content-type','connection','transfer-encoding','cookie','authorization')):raise APIError('Choose a custom API credential header.')
    if c['idempotencyHeader'] and (sensitive(c['idempotencyHeader']) or c['idempotencyHeader'].lower() in ('host','content-length','content-type','connection','transfer-encoding')):raise APIError('Choose an endpoint-supported idempotency header.')
    if type(c['mapping']) is not list or len(c['mapping'])>128:raise APIError('Map at most128 columns.')
    targets=set();sources=set()
    for row in c['mapping']:
        if type(row) is not dict or set(row)!={'source','target','type'} or row['type'] not in TYPES:raise APIError('Map source, destination and supported type.')
        for k in ('source','target'):
            if type(row[k]) is not str or not row[k] or len(row[k])>256 or '\0' in row[k]:raise APIError('Map bounded source and destination names.')
        if row['target'] in targets or row['source'] in sources:raise APIError('Use distinct mapping sources and targets.')
        pointer=row['source'] if c['direction']=='input' else row['target']
        if not pointer.startswith('/'):raise APIError('Nested fields use JSON Pointer, starting with /.')
        if c['direction']=='output' and any(pointer.startswith(t+'/') or t.startswith(pointer+'/') for t in targets):raise APIError('JSON fields cannot shadow their parent.')
        targets.add(row['target']);sources.add(row['source'])
    if type(c['constants']) is not list or len(c['constants'])>32 or c['direction']=='input' and c['constants']:raise APIError('Constants belong to output JSON payloads, at most32.')
    for row in c['constants']:
        if type(row) is not dict or set(row)!={'target','json'} or type(row['target']) is not str or not row['target'].startswith('/') or len(row['target'])>256 or type(row['json']) is not str or len(row['json'].encode())>1024:raise APIError('Use a bounded JSON constant and destination path.')
        pointer=row['target']
        if pointer in targets or any(pointer.startswith(t+'/') or t.startswith(pointer+'/') for t in targets):raise APIError('Constant destinations must not overlap mapped fields.')
        decode_json(row['json'].encode());targets.add(pointer)
    return c

_MISSING=object()
def path(value,pointer,default=_MISSING):
    try:
        for part in pointer.split('/')[1:] if pointer else []:
            key=part.replace('~1','/').replace('~0','~');value=value[int(key)] if isinstance(value,list) else value[key]
        return value
    except (KeyError,IndexError,TypeError,ValueError):
        if default is _MISSING:raise APIError('Configured JSON field is missing.')
        return default

def put(value,pointer,item):
    keys=[p.replace('~1','/').replace('~0','~') for p in pointer.split('/')[1:]]
    if not keys:raise APIError('Choose a nonroot JSON destination.')
    for key in keys[:-1]:
        value=value.setdefault(key,{})
        if type(value) is not dict:raise APIError('Conflicting JSON destinations.')
    value[keys[-1]]=item

def scalar(v,dtype):
    if v is None:return None
    if dtype=='json':return json.dumps(v,allow_nan=False,separators=(',',':'))
    if dtype=='string' and type(v) is str:return v
    if dtype=='bool' and type(v) is bool:return v
    if dtype=='int64' and type(v) is int and -(2**63)<=v<2**63:return v
    if dtype=='float64' and type(v) in (int,float) and math.isfinite(v) and (type(v) is not int or abs(v)<=2**53):return float(v)
    if dtype=='datetime' and type(v) is str:
        try:
            d=datetime.datetime.fromisoformat(v.replace('Z','+00:00'))
            if d.tzinfo is None:raise ValueError('Timestamp requires offset')
            return d.astimezone(datetime.timezone.utc).isoformat()
        except ValueError:pass
    raise APIError('JSON value does not match the declared column type; no lossy cast was applied.')

class Transport:
    def __init__(self,c,secret,check=lambda:None):
        self.c=c;self.secret=secret;self.check=check;self.deadline=time.monotonic()+c['timeoutSeconds'];self.total=0;self.last=0
    def remaining(self):
        self.check();seconds=self.deadline-time.monotonic()
        if seconds<=0:raise APIError('API operation timed out.')
        return seconds
    def sleep(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:self.remaining();time.sleep(min(.05,end-time.monotonic()))
    def request(self,url,query,body=None,request_id=''):
        c=self.c;p=url_check(url,c['allowLoopback']);self.sleep(max(0,1/c['requestsPerSecond']-(time.monotonic()-self.last)))
        # Resolve once and connect to that checked literal; never re-resolve in http.client.
        addresses=socket.getaddrinfo(p.hostname,p.port or (443 if p.scheme=='https' else 80),type=socket.SOCK_STREAM)
        safe=[]
        for family,kind,proto,canon,sockaddr in addresses:
            ip=ipaddress.ip_address(sockaddr[0])
            if not ip.is_global and not (c['allowLoopback'] and ip.is_loopback and p.hostname in ('127.0.0.1','::1')):raise APIError('API address is outside its permitted network scope.')
            safe.append(sockaddr[0])
        if not safe:raise APIError('API hostname did not resolve.')
        connection=http.client.HTTPConnection(p.hostname,p.port or (443 if p.scheme=='https' else 80),timeout=min(c['requestTimeout'],self.remaining()))
        sock=socket.create_connection((safe[0],connection.port),timeout=connection.timeout)
        if p.scheme=='https':sock=verified_tls_context().wrap_socket(sock,server_hostname=p.hostname)
        connection.sock=sock
        headers={'Accept':'application/json','Accept-Encoding':'identity',**c['headers']}
        token=self.secret.get('password','');username=self.secret.get('username','')
        if c['auth']!='none' and (not token or any(ch in token+username for ch in '\r\n\0') or len(token)>4096):raise APIError('Choose a valid private API credential.')
        if c['auth']=='bearer':headers['Authorization']='Bearer '+token
        if c['auth']=='basic':headers['Authorization']='Basic '+base64.b64encode((username+':'+token).encode()).decode()
        if c['auth']=='apiKey':headers[c['credentialHeader']]=token
        if request_id and c['idempotencyHeader']:headers[c['idempotencyHeader']]=request_id
        encoded=None if body is None else json.dumps(body,allow_nan=False,separators=(',',':')).encode()
        if encoded is not None:
            if len(encoded)>c['maximumBytes']:raise APIError('API request body exceeds its configured byte limit.')
            headers['Content-Type']='application/json'
        try:
            self.last=time.monotonic();connection.request(c['method'],(p.path or '/')+('?' + urlencode(query) if query else ''),body=encoded,headers=headers)
            response=connection.getresponse();blocks=[]
            while True:
                self.remaining()
                if response.fp is None:break
                response.fp.raw._sock.settimeout(min(c['requestTimeout'],self.remaining()));block=response.read(min(65536,c['maximumBytes']+1-self.total))
                if not block:break
                self.total+=len(block)
                if self.total>c['maximumBytes']:raise APIError('API response byte limit exceeded.')
                blocks.append(block)
            return response.status,dict((k.lower(),v) for k,v in response.getheaders()),b''.join(blocks)
        finally:connection.close()

def decode_json(raw):
    try:return json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(APIError('Nonfinite JSON value.')))
    except (ValueError,UnicodeError,RecursionError):raise APIError('The response is not a bounded JSON document.')

def fetch(c,secret,check=lambda:None):
    c=options(c)
    if c['direction']!='input' or not c['mapping']:raise APIError('Map at least one API input column.')
    t=Transport(c,secret,check);url=c['url'];query=dict(c['query']);rows=[];seen=set();pages=0
    for page in range(c['maximumPages']):
        if c['pagination'] in ('page','offset'):query[c['pageParameter']]=str(c['pageStart']+page*(c['pageSize'] if c['pagination']=='offset' else 1))
        key=(url,tuple(sorted(query.items())))
        if key in seen:raise APIError('Pagination repeated a request; no complete snapshot was published.')
        seen.add(key)
        for attempt in range(c['retries']+1):
            status,headers,raw=t.request(url,query)
            if status not in (429,502,503,504) or attempt==c['retries']:break
            value=headers.get('retry-after','')
            try:delay=float(value) if value else 2**attempt
            except ValueError:
                from email.utils import parsedate_to_datetime
                try:delay=max(0,parsedate_to_datetime(value).timestamp()-time.time())
                except (ValueError,TypeError):delay=2**attempt
            if not 0<=delay<=t.remaining():raise APIError('Rate-limit wait exceeds the operation timeout.')
            t.sleep(delay)
        if not 200<=status<300:raise APIError('API read returned HTTP '+str(status)+'. No snapshot published.')
        value=decode_json(raw);items=path(value,c['recordsPath'])
        if type(items) is not list:raise APIError('Records path must select a JSON array.')
        for record in items:
            check();row=[]
            for field in c['mapping']:
                v=path(record,field['source'],None) if c['missing']=='null' else path(record,field['source'])
                row.append(scalar(v,field['type']))
            rows.append(row)
            if len(rows)>c['maximumRows'] or len(rows)*len(c['mapping'])>5_000_000:raise APIError('API table exceeds its row/cell budget; no truncated snapshot published.')
        pages+=1;more=False
        if c['pagination']=='cursor':
            cursor=path(value,c['cursorPath'],None);more=cursor is not None and cursor!=''
            if more:
                if type(cursor) not in (str,int):raise APIError('Cursor must be text or an integer.')
                query[c['cursorParameter']]=str(cursor)
        elif c['pagination'] in ('page','offset'):more=len(items)>=c['pageSize']
        elif c['pagination']=='nextLink':
            link=path(value,c['nextPath'],None);more=bool(link)
            if more:
                if type(link) is not str:raise APIError('Next link must be a URL.')
                n=urlsplit(urljoin(url,link));origin=urlsplit(c['url'])
                if (n.scheme,n.hostname,n.port or (443 if n.scheme=='https' else 80))!=(origin.scheme,origin.hostname,origin.port or (443 if origin.scheme=='https' else 80)) or n.username is not None or n.password is not None or n.fragment:raise APIError('Cross-origin pagination is refused.')
                query=dict(parse_qsl(n.query))
                if len(query)>32 or any(sensitive(k) or len(v)>4096 for k,v in query.items()):raise APIError('Next-link query violates the bounded nonsecret request policy.')
                url=urlunsplit((n.scheme,n.netloc,n.path,'',''))
        if not more:break
    else:raise APIError('Maximum pages reached before the source ended; no complete snapshot published.')
    return [dict(name=r['target'],dtype='string' if r['type']=='json' else r['type'],nullable=True) for r in c['mapping']],rows,dict(complete=True,pages=pages,rows=len(rows),responseBytes=t.total,capturedAt=datetime.datetime.now(datetime.timezone.utc).isoformat())

def deliver(c,secret,columns,rows,operation_id,check=lambda:None):
    c=options(c)
    if c['direction']!='output' or not c['mapping']:raise APIError('Map at least one API output column.')
    names=[v['name'] for v in columns];indexes=[]
    for field in c['mapping']:
        if field['source'] not in names:raise APIError('Mapped source column is absent.')
        index=names.index(field['source'])
        if columns[index].get('dtype') not in ('string','int64','int32','float32','float64','bool','date','datetime','time'):
            raise APIError('API output requires an explicit scalar representation for image, sequence or other artifact columns.')
        indexes.append(index)
    # Validate all values before first side effect. Work and request counts are bounded.
    step=c['batchRows'] if c['delivery']=='batch' else 1
    if math.ceil(len(rows)/step)>c['maximumRequests']:raise APIError('Request count exceeds the configured limit.')
    bodies=[]
    for start in range(0,len(rows),step):
        items=[]
        for row in rows[start:start+step]:
            check();item={}
            for constant in c['constants']:put(item,constant['target'],decode_json(constant['json'].encode()))
            if type(row) is not list or len(row)!=len(columns):raise APIError('Captured input row width is invalid.')
            for field,index in zip(c['mapping'],indexes):
                value=row[index];column=columns[index]
                if value is not None and column['dtype'] in ('date','datetime','time'):
                    if __package__:from .dataflow_temporal import format_value
                    else:from dataflow_temporal import format_value
                    value=format_value(value,column['dtype'],timezone=column.get('temporal',{}).get('timezone','UTC'))
                    # Temporal JSON values are ISO strings, never guessed epoch units.
                    val=value if field['type']=='json' else scalar(value,field['type'])
                else:
                    val=(decode_json(value.encode()) if type(value) is str else value) if field['type']=='json' else scalar(value,field['type'])
                put(item,field['target'],val)
            items.append(item)
        body=items[0] if c['delivery']=='row' else items
        if c['delivery']=='batch' and c['arrayPath']:root={};put(root,c['arrayPath'],body);body=root
        if len(json.dumps(body,allow_nan=False,separators=(',',':')).encode())>c['maximumBytes']:raise APIError('Request body exceeds byte budget; no requests sent.')
        bodies.append((start,len(items),body))
    response_budget=16_000
    t=Transport(c,secret,check);receipt=dict(schemaVersion=1,kind='apiWrite',operationID=operation_id,status='committed',inputRows=len(rows),acceptedRows=0,requests=[],message='HTTP acceptance does not confirm downstream completion.')
    for number,(start,count,body) in enumerate(bodies):
        entry=dict(request=number+1,firstRow=start+1,inputRows=count,outcome='unknown');receipt['requests'].append(entry)
        try:
            # Write retries deliberately disabled even with an idempotency header.
            # Stable operation/request identity is useful for endpoint reconciliation.
            key=hashlib.sha256((operation_id+':'+str(number)).encode()).hexdigest()
            status,headers,raw=t.request(c['url'],c['query'],body,key)
            entry['httpStatus']=status
            if 200<=status<300:
                entry['outcome']='accepted';receipt['acceptedRows']+=count
                if c['responsePath'] and raw:
                    response=path(decode_json(raw),c['responsePath']);encoded=json.dumps(response,allow_nan=False)
                    size=len(json.dumps(encoded).encode())
                    if size<=512 and size<=response_budget:entry['responseJSON']=encoded;response_budget-=size
                    else:entry['responseOmitted']=True
            else:
                entry['outcome']='rejected';receipt['status']='unknown' if receipt['acceptedRows'] else 'notStarted';receipt['message']='HTTP rejection. Earlier accepted requests, if any, were not rolled back.';break
        except BaseException:
            receipt['status']='unknown';receipt['message']='Delivery has an uncertain outcome. Inspect the endpoint before another Run Flow.';break
    return receipt
