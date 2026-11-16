#!/usr/bin/env python3
"""Independent HTTP acceptance client, restricted to our synthetic DEBUG fixture.
Tokens are read privately and never printed. Does not configure an AI harness.
"""
import argparse,json,time,urllib.request,urllib.error,uuid
from pathlib import Path

class Client:
    def __init__(self,fixture,url):
        f=json.loads(fixture.read_text());self.root=Path(f['storage'])
        if f.get('synthetic') is not True or not self.root.name.startswith('mcp-native-fixture-'):raise ValueError('Only a synthetic native fixture is accepted')
        parsed=urllib.parse.urlsplit(url)
        if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.path!='/mcp' or parsed.query or parsed.fragment:raise ValueError('Only localhost MCP is accepted')
        self.url=url;self.client=f['clientID'];self.token=json.loads((self.root/'fixture-credentials.json').read_text())[self.client];self.counter=100
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):raise ValueError('Redirect refused')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def rpc(self,method,params):
        self.counter+=1
        request=urllib.request.Request(self.url,json.dumps({'jsonrpc':'2.0','id':self.counter,'method':method,'params':params}).encode(),{'Authorization':'Bearer '+self.token,'Content-Type':'application/json','Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'})
        with self.opener.open(request,timeout=18) as response:
            value=json.loads(response.read(1048577))
        if 'error' in value:raise AssertionError('MCP protocol refusal: '+str(value['error'].get('code')))
        return value['result']
    def tool(self,name,args):
        result=self.rpc('tools/call',{'name':name,'arguments':args})
        value=result['structuredContent']
        if result.get('isError') or value.get('isError'):raise AssertionError(name+': '+value.get('code','refused'))
        return value
    def state(self):return json.loads((self.root/'client-state.json').read_text())
    def save(self,state):
        path=self.root/'client-state.json';path.write_text(json.dumps(state,indent=2)+'\n');path.chmod(0o600)
    def patch(self,project,operations):
        before=self.tool('workflow_inspect',{'projectID':project})
        preview=self.tool('patch_preview',{'projectID':project,'revision':before['revision'],'operations':operations})
        args={'projectID':project,'proposalID':preview['proposalID'],'requestID':str(uuid.uuid4())}
        first=self.tool('patch_apply',args);second=self.tool('patch_apply',args)
        assert first==second
        return first
    def stage(self,stage):
        if stage=='build-table':
            self.rpc('initialize',{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'Synthetic native canvas acceptance','version':'1'}})
            tools=self.rpc('tools/list',{})['tools'];assert len(tools)==17
            entries=self.tool('catalog_list',{})['components'];by_family={e['familyID']:e for e in entries}
            assert 'workflow.saveModel' not in by_family and 'kwip.data.outputDelimited' not in by_family
            args={'name':'MCP · Columns acceptance','requestID':str(uuid.uuid4())}
            project=self.tool('project_create',args)['projectID'];assert self.tool('project_create',args)['projectID']==project
            ops=[{'operation':'insert','componentID':by_family[id]['componentID'],'nodeID':node,'name':name,'x':x,'y':100} for id,node,name,x in [('kwip.data.sourceCSV','source','Synthetic table',0),('kwip.data.columns','columns','Columns',300),('kwip.data.view','view','View',600)]]
            ops += [{'operation':'configure','address':{'domain':'data','localID':'columns'},'configuration':{'rows':[{'source':'amount','included':True,'outputName':'value','targetType':'float64'},{'source':'id','included':True,'outputName':'id','targetType':'int64'}],'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}}]
            ops += [{'operation':'connect','from':{'address':{'domain':'data','localID':a},'port':'table'},'to':{'address':{'domain':'data','localID':b},'port':'table'}} for a,b in [('source','columns'),('columns','view')]]
            self.patch(project,ops)
            captured=self.tool('workflow_inspect',{'projectID':project});assert len(captured['document']['nodes'])==3 and len(captured['document']['edges'])==2
            self.save({'projectID':project,'assertions':['MCP initialize and all 17 action schemas','default file-write consumers hidden','idempotent native project creation','registry-derived placement and Columns configuration','atomic typed connections and identical Apply retry']})
            csv=self.root/'synthetic.csv';csv.write_text('id,amount\n'+''.join(f'{i},{i/4}\n' for i in range(50)))
            return {'stage':stage,'passed':True,'projectID':project,'syntheticSource':str(csv),'next':'Bind source in the native app; allow captured values for this synthetic project.'}
        if stage=='build-model':
            demo=json.loads((Path(__file__).resolve().parent.parent/'.local/mcp-demo-fixture.json').read_text())
            entries=self.tool('catalog_list',{})['components']
            def entry_for(payload):
                for entry in entries:
                    default=entry['defaultPayload']
                    if payload['kind']!=default['kind']:continue
                    if payload['kind']=='dataOperator' and payload['operatorID']!=default.get('operatorID'):continue
                    if payload['kind']=='modelPrimitive' and payload['type']!=default.get('type') and payload['type'] not in [v['id'] for v in entry.get('variants',[])]:continue
                    return entry
                raise AssertionError('Current authoring catalog has no eligible '+payload['kind'])
            project=self.tool('project_create',{'name':'MCP · Named text model acceptance','requestID':str(uuid.uuid4())})['projectID']
            ops=[]
            for node in demo['nodes']:
                payload=node['payload']
                if payload.get('type')=='index_input':continue
                entry=entry_for(payload)
                ops.append({'operation':'insert','componentID':entry['componentID'],'nodeID':node['address']['localID'],'name':node['name'],'x':node['x'],'y':node['y']})
                if payload.get('config') and payload['kind']!='train':
                    config=payload['config']
                    if payload['kind']=='dataOperator':config={**entry['defaultPayload']['config'],**config}
                    ops.append({'operation':'configure','address':node['address'],'configuration':config})
            for edge in demo['edges']:
                if edge['from']['address']['localID']=='input':continue
                ops.append({'operation':'connect','from':edge['from'],'to':edge['to']})
            ops.append({'operation':'modelInput','definition':{'domain':'model','localID':'output'},'type':'index_input','configuration':{'shape':[3]},'destinations':[{'address':{'domain':'model','localID':'lookup'},'port':'in'}],'inputName':'tokens'})
            self.patch(project,ops)
            inspected=self.tool('workflow_inspect',{'projectID':project})['document']
            actual=inspected['namedModelInputs']['output']['tokens']
            train=next(n for n in demo['nodes'] if n['address']['localID']=='train')['payload']['config']
            train['dataMapping']['namedInputPlan']['inputs'][0]['nodeID']=actual
            self.patch(project,[{'operation':'configure','address':{'domain':'control','localID':'train'},'configuration':train},{'operation':'move','address':{'domain':'model','localID':actual},'dx':40,'dy':250},{'operation':'rename','address':{'domain':'model','localID':actual},'name':'Tokens'}])
            valid=self.tool('workflow_validate',{'projectID':project,'terminal':{'domain':'control','localID':'train'}})
            assert valid['valid']
            public=Path(__file__).resolve().parent.parent/'Sources/KwipMLApp/Resources/text-training-observations.csv'
            source=self.root/'synthetic-text.csv';source.write_bytes(public.read_bytes())
            state=self.state();state['modelProjectID']=project;state['modelInputID']=actual;self.save(state)
            return {'stage':stage,'passed':True,'projectID':project,'syntheticSource':str(source),'nodeCount':len(inspected['nodes']),'next':'Bind the synthetic text source and allow captured values for this project.'}
        state=self.state();project=state['projectID']
        if stage=='negative':
            assertions=[]
            for headers,expected in [({'Authorization':'Bearer '+('x'*43)},401),({'Origin':'https://invalid.example'},403),({'Host':'invalid.example'},403)]:
                request=urllib.request.Request(self.url,json.dumps({'jsonrpc':'2.0','id':999,'method':'tools/list','params':{}}).encode(),{'Authorization':'Bearer '+self.token,'Content-Type':'application/json','Accept':'application/json, text/event-stream',**headers})
                try:self.opener.open(request,timeout=5);raise AssertionError('Invalid admission accepted')
                except urllib.error.HTTPError as error:assert error.code==expected
            assertions.append('wrong token, foreign Origin and Host refused by actual app helper')
            checks=[('workflow_inspect',{'projectID':str(uuid.uuid4())},'not_permitted'),('patch_preview',{'projectID':project,'revision':-1,'operations':[{'operation':'rename','address':{'domain':'data','localID':'source'},'name':'Should not apply'}]},'stale_or_invalid_patch'),('catalog_get',{'componentID':'kwip.data.cast'},'component_unavailable')]
            for name,args,code in checks:
                value=self.rpc('tools/call',{'name':name,'arguments':args})['structuredContent'];assert value.get('isError') and value['code']==code
            assertions += ['unshared project refused','stale revision refused without applying','retired raw component placement identity unavailable']
            (self.root/'NEGATIVE_CLIENT_PASSED.json').write_text(json.dumps({'passed':True,'assertions':assertions,'ownerDataUsed':False},indent=2)+'\n')
            return {'stage':stage,'passed':True,'assertions':assertions}
        if stage=='cancel-model':
            project=state['modelProjectID']
            doc=self.tool('workflow_inspect',{'projectID':project})['document']
            config=next(n for n in doc['nodes'] if n['address']['localID']=='train')['payload']['config']
            state['normalTrainConfiguration']=config
            long=json.loads(json.dumps(config));long['runSetup']['epochs']=10000
            self.patch(project,[{'operation':'configure','address':{'domain':'control','localID':'train'},'configuration':long}])
            job=self.tool('run_start',{'projectID':project,'terminal':{'domain':'control','localID':'train'},'requestID':str(uuid.uuid4())})['jobID']
            self.tool('run_cancel',{'projectID':project,'jobID':job})
            for _ in range(100):
                value=self.tool('job_get',{'projectID':project,'jobID':job})['job']
                if value['state']!='running':break
                time.sleep(.2)
            assert value['state']=='cancelled',value['state']
            self.patch(project,[{'operation':'configure','address':{'domain':'control','localID':'train'},'configuration':config}])
            state['modelJobID']=self.tool('run_start',{'projectID':project,'terminal':{'domain':'control','localID':'train'},'requestID':str(uuid.uuid4())})['jobID'];self.save(state)
            (self.root/'CANCELLATION_PASSED.json').write_text(json.dumps({'passed':True,'assertions':['real MCP-owned long run cancelled and acknowledged after cleanup','normal configuration restored and short run restarted'],'ownerDataUsed':False},indent=2)+'\n')
            return {'stage':stage,'passed':True,'cancelledJob':job,'restoredShortJob':state['modelJobID']}
        if stage=='run-model':
            project=state['modelProjectID'];job=self.tool('run_start',{'projectID':project,'terminal':{'domain':'control','localID':'train'},'requestID':str(uuid.uuid4())})['jobID']
            state['modelJobID']=job;self.save(state);return {'stage':stage,'started':True,'jobID':job}
        if stage=='inspect-model':
            project=state['modelProjectID'];job=self.tool('job_get',{'projectID':project,'jobID':state['modelJobID']})['job']
            if job['state']!='completed':return {'stage':stage,'jobState':job['state'],'phase':job.get('phase')}
            generate=self.tool('results_read',{'projectID':project,'address':{'domain':'control','localID':'generate'},'includeValues':True})
            tensor=self.tool('results_read',{'projectID':project,'address':{'domain':'model','localID':'lookup'},'includeValues':True})
            assert generate['available'] and tensor['available']
            assert tensor['capturedSample']['out']['values'] and tensor['capturedSample']['out']['dtype']=='float32'
            args={'projectID':project,'address':{'domain':'control','localID':'generate'},'includeValues':True}
            query=self.tool('results_read',{**args,'page':True})['queryID']
            for _ in range(60):
                page=self.tool('results_read',{**args,'queryID':query})
                if page['state']=='completed':break
                time.sleep(.2)
            assert page['state']=='completed' and page['rows'][0][1]=='babababababababa' and page['rows'][0][3]==16
            self.save(state);(self.root/'MODEL_CLIENT_PASSED.json').write_text(json.dumps({'passed':True,'assertions':['current-catalog model placement and named-input boundary','real training-plan validation','real CPU named text training','automatic bounded generation','captured tensor and exact generated text through independent MCP client'],'ownerDataUsed':False,'scope':'Actual DEBUG app and synthetic public text fixture, not production release acceptance'},indent=2)+'\n')
            return {'stage':stage,'passed':True,'generatedText':'babababababababa','tensorCaptured':True}
        if stage=='run-table':
            args={'projectID':project,'requestID':str(uuid.uuid4())};job=self.tool('run_start',args)['jobID'];assert self.tool('run_start',args)['jobID']==job
            state['jobID']=job;self.save(state);return {'stage':stage,'started':True,'jobID':job}
        if stage=='inspect-table':
            job=self.tool('job_get',{'projectID':project,'jobID':state['jobID']})['job']
            if job['state']!='completed':return {'stage':stage,'jobState':job['state']}
            address={'domain':'data','localID':'view'}
            metadata=self.tool('results_read',{'projectID':project,'address':address,'includeValues':True})
            assert metadata['available'] and metadata['currentConstruction'] and metadata['currentBindings']
            query=self.tool('results_read',{'projectID':project,'address':address,'includeValues':True,'page':True,'sortColumn':0,'descending':True})['queryID']
            for _ in range(60):
                page=self.tool('results_read',{'projectID':project,'address':address,'includeValues':True,'queryID':query})
                if page['state']=='completed':break
                time.sleep(.2)
            assert page['state']=='completed' and page['totalRows']==50 and page['rows'][0]==[12.25,49] and page['rows'][-1]==[0.,0]
            state['assertions'] += ['real isolated preparation run and identical start retry','stable job completion polling','captured View provenance','verified table paging and descending sort, exact 50 synthetic rows']
            self.save(state);(self.root/'TABLE_CLIENT_PASSED.json').write_text(json.dumps({'passed':True,'assertions':state['assertions'],'scope':'Independent HTTP client to actual DEBUG app; synthetic data only'},indent=2)+'\n')
            return {'stage':stage,'passed':True,'rowCount':50,'assertions':state['assertions']}
        raise ValueError('Unknown stage')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--url',required=True);parser.add_argument('--initialize',action='store_true');parser.add_argument('--stage',required=True,choices=['build-table','run-table','inspect-table','build-model','run-model','inspect-model','negative','cancel-model']);args=parser.parse_args()
    try:
        client=Client(args.fixture,args.url)
        if args.initialize:client.rpc('initialize',{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'Synthetic native canvas acceptance','version':'1'}})
        print(json.dumps(client.stage(args.stage)))
    except Exception as error:print(json.dumps({'passed':False,'failureType':type(error).__name__,'message':str(error) if isinstance(error,AssertionError) else 'Native fixture request failed'}));raise SystemExit(1)
