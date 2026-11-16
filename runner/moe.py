"""Versioned dropless pointwise expert routing. No network, scripts or pickle inputs."""
import json
import math
import re
import torch
from torch import nn


def default_graph(width, output, hidden=None):
    nodes=[dict(id='input',type='input',config={'features':width})]
    if hidden:
        nodes += [dict(id='hidden',type='linear',config=dict(in_features=width,out_features=hidden,bias=True)),
                  dict(id='activation',type='gelu',config={'approximate_tanh':False})]
    nodes += [dict(id='gate',type='linear',config=dict(in_features=hidden or width,out_features=output,bias=True)),dict(id='output',type='output',config={})]
    edges=[dict(id='edge'+str(i),source=a['id'],sourcePort='out',target=b['id'],targetPort='in') for i,(a,b) in enumerate(zip(nodes,nodes[1:]))]
    return dict(schemaVersion=2,nodes=nodes,edges=edges)


def default_config(width=2,output=2,experts=4):
    c=dict(version=1,input_features=width,output_features=output,expert_count=experts,top_k=2,
           temperature=1.,weighting='normalized',balance_coefficient=.01,z_coefficient=.001,
           capture='aggregate',sample_limit=128,mask=False,router=json.dumps(default_graph(width,experts),separators=(',',':')))
    for i in range(4):
        c['expert_'+str(i+1)]=json.dumps(dict(id='expert_'+chr(97+i),name='Expert '+chr(65+i),trainable=True,
            graph=default_graph(width,output,8)),separators=(',',':'))
    return c


def normalize(config):
    if type(config) is not dict or set(config)!=set(default_config()):raise ValueError('MoE requires exactly its versioned declared controls.')
    c=dict(config)
    for key,lo,hi in [('version',1,1),('input_features',1,512),('output_features',1,512),('expert_count',2,4),('top_k',1,4),('sample_limit',0,1024)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(f'MoE {key} requires an integer from {lo} to {hi}.')
    if c['top_k']>c['expert_count']:raise ValueError('Experts per unit cannot exceed the expert count.')
    for key,lo,hi in [('temperature',.0001,100.),('balance_coefficient',0.,10.),('z_coefficient',0.,10.)]:
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or not lo<=c[key]<=hi:raise ValueError(f'MoE {key} is outside its finite bounds.')
        c[key]=float(c[key])
    if c['weighting'] not in ('normalized','probability') or c['capture'] not in ('off','aggregate','sample') or type(c['mask']) is not bool:raise ValueError('Choose declared MoE weights, capture and mask policies.')
    identities=set()
    for key in ['router']+['expert_'+str(i+1) for i in range(4)]:
        if type(c[key]) is not str or not 1<=len(c[key])<=4096:raise ValueError('Each MoE graph must be bounded JSON of at most4096 characters.')
        try:record=json.loads(c[key])
        except (ValueError,TypeError):raise ValueError('MoE graphs must be inert valid JSON.') from None
        if key=='router':graph=record
        else:
            if type(record) is not dict or set(record)!={'id','name','trainable','graph'}:raise ValueError('Expert requires stable identity, label, trainability and graph.')
            if type(record['id']) is not str or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',record['id']) is None or record['id'] in identities:raise ValueError('Expert identities must be unique bounded names without dots.')
            identities.add(record['id'])
            if type(record['name']) is not str or not 1<=len(record['name'])<=80 or type(record['trainable']) is not bool:raise ValueError('Expert name/trainability is invalid.')
            graph=record['graph']
        if type(graph) is not dict or set(graph)!={'schemaVersion','nodes','edges'} or graph['schemaVersion']!=2:raise ValueError('Expert/router graph requires explicit graph schema2.')
        if type(graph['nodes']) is not list or not 2<=len(graph['nodes'])<=12 or type(graph['edges']) is not list or len(graph['edges'])>24:raise ValueError('Pointwise graph allows2–12 nodes and at most24 edges.')
        allowed={'input','output','linear','relu','tanh','sigmoid','gelu','silu','leaky_relu','elu','layer_norm','rms_norm','dropout','add'}
        if any(type(n) is not dict or n.get('type') not in allowed or n.get('parameterSettings') for n in graph['nodes']):raise ValueError('Use only independent pointwise graph operations. Nested routing, sharing and side effects are unavailable.')
    return c


def scientific_config(config):
    result=dict(config)
    def graph(source):
        return dict(schemaVersion=source['schemaVersion'],nodes=[{k:v for k,v in n.items() if k not in ('name','x','y')} for n in source['nodes']],edges=[{k:v for k,v in e.items() if k!='id'} for e in source['edges']])
    for key in ['router']+['expert_'+str(i+1) for i in range(4)]:
        row=json.loads(config[key])
        if key=='router':result[key]=graph(row)
        else:result[key]=dict(id=row['id'],trainable=row['trainable'],graph=graph(row['graph']))
    return result


def validate_graphs(config):
    if __package__:from .engine import validate
    else:from engine import validate
    c=normalize(config);count=0
    for graph,output in [(json.loads(c['router']),c['expert_count'])]+[(json.loads(c['expert_'+str(i+1)])['graph'],c['output_features']) for i in range(c['expert_count'])]:
        if sum(n['type']=='input' for n in graph['nodes'])!=1:raise ValueError('Pointwise graph requires one input.')
        source=next(n for n in graph['nodes'] if n['type']=='input')
        if source['config']!={'features':c['input_features']}:raise ValueError('Expert/router input width must match MoE input features.')
        _,_,layouts,_=validate(graph,include_types=True,bundled_dataset=False,training_required=False,architecture_only=True)
        final=next(n for n in graph['nodes'] if n['type']=='output')
        if layouts[final['id']].sample!=(output,):raise ValueError('All expert outputs must match the declared MoE output width; router outputs match expert count.')
        for n in graph['nodes']:
            cfg=n['config']
            if n['type']=='linear':count+=cfg['in_features']*cfg['out_features']+(cfg['out_features'] if cfg['bias'] else 0)
            elif n['type'] in ('layer_norm','rms_norm'):
                dims=cfg['normalized_shape'];count+=math.prod(dims)*(2 if n['type']=='layer_norm' and cfg.get('bias',True) else 1) if cfg.get('elementwise_affine',True) else 0
    if count>2_000_000:raise ValueError('All resident MoE parameters exceed the2million graph limit.')
    return c,count


class RoutedExperts(nn.Module):
    def __init__(self,config,factory=None):
        super().__init__()
        self.config,cnt=validate_graphs(config)
        if factory is None:
            if __package__:from .engine import GraphModel
            else:from engine import GraphModel
            factory=lambda g:GraphModel(g,bundled_dataset=False,architecture_only=True)
        c=self.config;self.router=factory(json.loads(c['router']));self.experts=nn.ModuleDict()
        self.labels={}
        for i in range(c['expert_count']):
            item=json.loads(c['expert_'+str(i+1)]);module=factory(item['graph'])
            module.requires_grad_(item['trainable']);self.experts[item['id']]=module;self.labels[item['id']]=item['name']
        # Conservative concentrated dispatch: every active subgraph's widest
        # intermediate retained across all layers, four float32 working copies.
        sources=[json.loads(c['router'])]+[json.loads(c['expert_'+str(i+1)])['graph'] for i in range(c['expert_count'])]
        self.working_elements_per_unit=4*sum(len(g['nodes'])*max([c['input_features'],c['output_features']]+[n['config'].get('out_features',1) for n in g['nodes']]) for g in sources)+c['top_k']*(c['input_features']+c['output_features'])
        self.last_terms=None;self.capture_enabled=True;self.reset_capture()

    def reset_capture(self):
        self._phases={};self._samples=[]

    def forward(self,x,mask=None):
        c=self.config
        if not isinstance(x,torch.Tensor) or x.dtype!=torch.float32 or x.ndim not in (2,3) or x.shape[-1]!=c['input_features'] or x.shape[0]<=0 or x.numel()>100_000_000 or not torch.isfinite(x).all().item():raise ValueError('MoE requires finite float32 [N,D] or [N,T,D] with the declared feature width and nonempty observations.')
        flat=x.reshape(-1,x.shape[-1])
        if flat.shape[0]*self.working_elements_per_unit>20_000_000:raise ValueError('MoE concentrated dispatch exceeds its bounded working allocation; reduce batch/context.')
        if c['mask'] != (mask is not None):raise ValueError('MoE valid-unit mask must match the saved mask policy.')
        if mask is not None:
            if mask.device!=x.device or mask.dtype not in (torch.bool,torch.int64) or tuple(mask.shape) not in (tuple(x.shape[:-1]),(x.shape[0],1) if x.ndim==2 else ()) or not ((mask==0)|(mask==1)).all().item():raise ValueError('MoE mask must match unit axes and contain only Boolean or int64 zero/one values on the same device.')
            valid=mask.reshape(-1).bool()
        else:valid=torch.ones(flat.shape[0],device=x.device,dtype=torch.bool)
        indexes=valid.nonzero().flatten();units=flat.index_select(0,indexes)
        output=x.new_zeros((flat.shape[0],c['output_features']))
        if not len(indexes):
            # A router may be parameter-free. Keep backward legal without giving
            # unused parameters zero gradients (AdamW would then decay them).
            zero=x.sum()*0
            if torch.is_grad_enabled() and not zero.requires_grad:zero.requires_grad_()
            self.last_terms=(zero,zero);return output.reshape(*x.shape[:-1],c['output_features'])+zero
        logits=self.router(units)
        if not torch.isfinite(logits).all().item():raise ValueError('MoE router produced nonfinite logits.')
        p=torch.softmax(logits/c['temperature'],dim=-1)
        ids=torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:c['top_k']]
        weights=p.gather(1,ids)
        if c['weighting']=='normalized':weights=torch.softmax((logits/c['temperature']).gather(1,ids),dim=-1)
        counts=torch.bincount(ids.flatten(),minlength=c['expert_count'])
        balance=c['expert_count']*((counts.to(p.dtype)/(len(indexes)*c['top_k'])).detach()*p.mean(0)).sum()
        z=torch.logsumexp(logits,dim=-1).square().mean();self.last_terms=(balance,z)
        combined=x.new_zeros((len(indexes),c['output_features']))
        for i,expert in enumerate(self.experts.values()):
            rows,slots=(ids==i).nonzero(as_tuple=True)
            if rows.numel():
                value=expert(units.index_select(0,rows))
                combined=combined.index_add(0,rows,value*weights[rows,slots,None])
        output=output.index_copy(0,indexes,combined)
        if not torch.isfinite(output).all().item():raise ValueError('MoE expert output is nonfinite.')
        if self.capture_enabled and c['capture']!='off':
            phase='training' if self.training else 'inference'
            stats=self._phases.setdefault(phase,dict(counts=[0]*c['expert_count'],mass=[0.]*c['expert_count'],visits=0))
            cc=counts.detach().cpu().tolist();mass=torch.zeros(c['expert_count'],device=x.device).scatter_add(0,ids.flatten(),weights.detach().flatten()).cpu().tolist()
            stats['visits']+=len(indexes)
            for i in range(len(cc)):stats['counts'][i]+=int(cc[i]);stats['mass'][i]+=mass[i]
            if c['capture']=='sample':
                limit=min(c['sample_limit']-len(self._samples),len(indexes))
                if limit>0:
                    source=indexes[:limit].cpu().tolist();selected=ids[:limit].cpu().tolist();ww=weights[:limit].detach().cpu().tolist()
                    self._samples += [dict(phase=phase,visit=stats['visits']-len(indexes)+offset,unit=int(row),experts=[list(self.experts)[i] for i in chosen],weights=w) for offset,(row,chosen,w) in enumerate(zip(source,selected,ww))]
        return output.reshape(*x.shape[:-1],c['output_features'])

    def auxiliary_loss(self):
        if self.last_terms is None:return next(self.parameters()).sum()*0
        return self.last_terms[0]*self.config['balance_coefficient']+self.last_terms[1]*self.config['z_coefficient']

    def routing_summary(self):
        result=[]
        if not self.capture_enabled or self.config['capture']=='off':return result
        for phase,s in self._phases.items():
            total=sum(s['counts'])
            result += [dict(phase=phase,expert=key,label=self.labels[key],assignments=s['counts'][i],assignmentShare=s['counts'][i]/total if total else None,meanGateWeight=s['mass'][i]/s['counts'][i] if s['counts'][i] else None,routedUnits=s['visits']) for i,key in enumerate(self.experts)]
        return result


def auxiliary_loss(model):
    terms=[m.auxiliary_loss() for m in model.modules() if isinstance(m,RoutedExperts)]
    return sum(terms) if terms else 0.


def routing_summary(model):
    return [dict(block=name,**row) for name,m in model.named_modules() if isinstance(m,RoutedExperts) for row in m.routing_summary()]


def objective_statistics(model):
    terms=[torch.stack((m.last_terms[0]*m.config['balance_coefficient'],m.last_terms[1]*m.config['z_coefficient'])) for m in model.modules() if isinstance(m,RoutedExperts) and m.last_terms is not None]
    if not terms:return None
    balance,z=torch.stack(terms).sum(0).detach().cpu().tolist()
    return dict(balancePenalty=float(balance),routerZPenalty=float(z))


def captured_tables(model,run_id,graph_hash):
    summary=routing_summary(model);samples=[]
    for name,m in model.named_modules():
        if isinstance(m,RoutedExperts) and m.capture_enabled:
            samples += [[name,row['phase'],row['visit'],row['unit'],json.dumps(row['experts']),json.dumps(row['weights'])] for row in m._samples]
    def table(names,types,rows):
        if __package__:from . import dataflow as df
        else:import dataflow as df
        columns=[dict(name=n,dtype=t,nullable=True) for n,t in zip(names,types)]
        ids=[run_id+':'+names[-1]+':'+str(i) for i in range(len(rows))]
        return dict(schemaVersion=1,kind='table',columns=columns,rows=rows,lineage=dict(runID=run_id,graphHash=graph_hash,scope='capturedUnitVisits',role='unassigned',sourceDigest=df.digest([run_id,graph_hash]),membershipDigest=df.membership_digest(ids),observationIDs=ids,dataVersion=df.digest(dict(columns=columns,rows=rows,ids=ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[]))
    result={}
    if summary:result['routing']=table(['block','phase','expert_id','expert','assignments','assignment_share','mean_gate_weight','routed_unit_visits'],['string']*4+['int64','float64','float64','int64'],[[r[k] for k in ('block','phase','expert','label','assignments','assignmentShare','meanGateWeight','routedUnits')] for r in summary])
    if samples:result['routingSample']=table(['block','phase','visit','unit_position','expert_ids','gate_weights'],['string','string','int64','int64','string','string'],samples)
    return result


def preflight(config,check=lambda:None):
    """Finite synthetic CPU structure/gradient check, no optimizer, files or user data."""
    check();c,count=validate_graphs(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4100);m=RoutedExperts(c);m.capture_enabled=False
        x=torch.linspace(-1,1,4*c['input_features']).reshape(4,c['input_features']).requires_grad_()
        mask=torch.ones(4,dtype=torch.int64) if c['mask'] else None
        check();y=m(x,mask);loss=y.square().mean()+m.auxiliary_loss()
        # Qualify each resident expert even if the synthetic top-k did not choose it.
        for expert in m.experts.values():check();loss=loss+expert(x).square().mean()
        loss=loss+m.router(x).square().mean();loss.backward();check()
        if not torch.isfinite(loss).item() or any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in m.parameters()):raise ValueError('MoE preflight produced nonfinite gradients.')
        return dict(parameters=count,device='cpu',optimizerUpdates=0,experts=len(m.experts))
