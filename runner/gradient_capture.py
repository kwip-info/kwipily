"""Read-only bounded post-backward gradient observations; never optimization policy."""
import math
import json
import torch

SCHEMA_VERSION=1
IDENTIFIER='kwip.gradient-summary.v1'
PHASE='afterBackwardBeforeStep'
LIMITS={'epochs':10000,'batchesPerEpoch':8,'captures':1024,'artifactBytes':8_000_000,'chunkElements':65536}
DEFAULT={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':256}

def capability():return {'schemaVersion':1,'id':IDENTIFIER,'supportedDevices':['cpu','mps'],'phase':PHASE,'limits':dict(LIMITS)}

def failed_capability():
    return {'schemaVersion':1,'id':'kwip.failed-gradient-observations.v1',
            'supportedDevices':['cpu','mps'],'phase':PHASE,
            'limits':{'eventBytes':100_000,'artifactBytes':8_000_000,'captures':1024}}

def validate_config(value):
    if value is None:return None
    if not isinstance(value,dict) or value.keys()!=DEFAULT.keys():raise ValueError('Gradient capture requires exactly schemaVersion, everyEpochs, maxBatchesPerEpoch and maxCaptures.')
    for key,maximum in [('schemaVersion',1),('everyEpochs',10000),('maxBatchesPerEpoch',8),('maxCaptures',1024)]:
        if type(value[key]) is not int or not 1<=value[key]<=maximum:raise ValueError(f'Gradient capture {key} must be an integer from 1 to {maximum}.')
    return dict(value)

class GradientCapture:
    def __init__(self,model,config,*,run_id,initial_epoch,final_epoch,device):
        self.config=validate_config(config)
        self.parameters=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        self.inventory={'parameterCount':len(self.parameters),'elementCount':sum(p.numel() for _,p in self.parameters),'parameters':[{'name':name,'nodeID':name.split('.')[1] if name.startswith('layers.') else name,'shape':list(p.shape),'elementCount':p.numel()} for name,p in self.parameters]}
        if len(self.parameters)>256 or self.inventory['elementCount']>2_000_000:raise ValueError('Gradient capture parameter inventory exceeds product bounds.')
        self.remaining=self.config['maxCaptures'];self.summaries=[]
        self.run_id=run_id;self.initial_epoch=initial_epoch;self.final_epoch=final_epoch;self.device=device
    def started(self):return {'schemaVersion':1,'config':self.config,'initialEpoch':self.initial_epoch,'inventory':self.inventory}
    def begin_epoch(self,epoch,total_batches,total_rows):
        count=min(self.config['maxBatchesPerEpoch'],total_batches) if (epoch-1)%self.config['everyEpochs']==0 else 0
        self.selected=set([0] if count==1 else [i*(total_batches-1)//(count-1) for i in range(count)])
        self.current={'event':'gradientSummary','schemaVersion':1,'runID':self.run_id,'epoch':epoch,'totalBatches':total_batches,'scheduledBatchCount':count,'observedBatchCount':0,'totalTrainingRows':total_rows,'observedTrainingRows':0,'captureBudgetRemaining':self.remaining,'status':'notScheduled' if count==0 else 'budgetExhausted','batches':[]}
    def observe(self,batch_index,training_rows):
        if batch_index not in self.selected or self.remaining==0:return
        row={'batchIndex':batch_index,'trainingRows':training_rows,'presentParameterCount':0,'presentElementCount':0,'missingParameterCount':0,'missingElementCount':0,'finiteElementCount':0,'nonfiniteElementCount':0,'zeroElementCount':0,'l2Norm':None,'maxAbs':None}
        norm=0.;maximum=0.
        with torch.no_grad():
            for _,parameter in self.parameters:
                gradient=parameter.grad
                if gradient is None:
                    row['missingParameterCount']+=1;row['missingElementCount']+=parameter.numel();continue
                if gradient.layout!=torch.strided or gradient.is_nested or gradient.dtype!=torch.float32 or gradient.shape!=parameter.shape:raise ValueError('Gradient capture requires matching dense float32 parameter gradients.')
                row['presentParameterCount']+=1;row['presentElementCount']+=gradient.numel()
                flattened=gradient.detach().reshape(-1)
                for start in range(0,flattened.numel(),LIMITS['chunkElements']):
                    # Transfer in the source dtype before CPU-only widening. A combined
                    # MPS transfer/cast can silently return zeros in the bundled runtime.
                    chunk=flattened[start:start+LIMITS['chunkElements']].cpu().to(dtype=torch.float64)
                    finite=torch.isfinite(chunk);count=int(finite.sum().item())
                    row['finiteElementCount']+=count;row['nonfiniteElementCount']+=chunk.numel()-count
                    row['zeroElementCount']+=int((chunk==0).sum().item())
                    values=chunk[finite]
                    if values.numel():
                        norm=math.hypot(norm,float(torch.linalg.vector_norm(values).item()))
                        maximum=max(maximum,float(values.abs().max().item()))
        if row['presentElementCount'] and row['nonfiniteElementCount']==0:row['l2Norm']=norm;row['maxAbs']=maximum
        self.remaining-=1;self.current['batches'].append(row)
    def end_epoch(self):
        event=self.current;event['observedBatchCount']=len(event['batches']);event['observedTrainingRows']=sum(b['trainingRows'] for b in event['batches']);event['captureBudgetRemaining']=self.remaining
        if event['batches']:event['status']='captured'
        self.summaries.append(event)
        return event
    def interrupted(self, stage, last_backward_batch_index):
        """Snapshot only completed observations; do not finalize a partial epoch."""
        current = getattr(self, 'current', None)
        if stage not in {'trainingForward','trainingBackward','optimizerStep','validationForward','validationLoss'} or current is None:
            return None
        if self.summaries and current['epoch'] <= self.summaries[-1]['epoch']:
            return None
        if not current['batches'] and not any(row['observedBatchCount'] for row in self.summaries):
            return None
        batches = [dict(batch) for batch in current['batches']]
        event = {'event':'gradientInterrupted','schemaVersion':1,
                 'id':'kwip.failed-gradient-observations.v1','runID':self.run_id,
                 'epoch':current['epoch'],'stage':stage,'lastBackwardBatchIndex':last_backward_batch_index,
                 'totalBatches':current['totalBatches'],'totalTrainingRows':current['totalTrainingRows'],
                 'scheduledBatchCount':current['scheduledBatchCount'],'observedBatchCount':len(batches),
                 'observedTrainingRows':sum(batch['trainingRows'] for batch in batches),
                 'captureBudgetRemaining':self.remaining,'batches':batches}
        if len(json.dumps(event,allow_nan=False).encode('utf-8')) > 100_000:
            raise ValueError('Interrupted gradient event exceeds its transport allowance.')
        return event

    def artifact(self):
        return {'schemaVersion':1,'id':IDENTIFIER,'runID':self.run_id,'device':self.device,'phase':PHASE,'config':self.config,'initialEpoch':self.initial_epoch,'finalEpoch':self.final_epoch,'inventory':self.inventory,'summaries':self.summaries}
