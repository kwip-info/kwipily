"""Observation-only, completed-work reporting; never advances a model or scheduler.

Sufficient statistics are borrowed from the existing loss calculation. Reporting
adds no forward pass, tensor copy or device synchronization. Time is monotonic
wall time at completed work boundaries, not an estimate of accelerator time.
"""
import math
import time

DEFAULT_POLICY = dict(everyBatches=10, everySeconds=1.)


def validate_policy(value):
    if (not isinstance(value,dict) or set(value)!={'everyBatches','everySeconds'}
        or type(value['everyBatches']) is not int or not 1<=value['everyBatches']<=10000
        or type(value['everySeconds']) not in (int,float)
        or not math.isfinite(value['everySeconds']) or not .1<=value['everySeconds']<=60):
        raise ValueError('Training reporting requires 1–10000 batches and 0.1–60 seconds.')
    return dict(value)


class TrainingProgress:
    def __init__(self,run_id,emit,policy=None,*,clock=time.monotonic,reduction='mean'):
        self.policy=validate_policy(DEFAULT_POLICY if policy is None else policy)
        self.run_id=run_id;self.emit=emit;self.clock=clock;self.start=clock();self.last=self.start
        self.batches=0;self.updates=0;self.sequence=0;self.pending=0;self.numerator=0.;self.denominator=0.;self.moe_sums=[0.,0.];self.moe_denominator=0.;self.reduction=reduction;self.moe=None;self.moe_sums=[0.,0.];self.moe_denominator=0.

    def _emit(self,epoch,epoch_batch,epoch_batches,kind,loss,**extra):
        self.sequence+=1;now=self.clock()
        self.emit(dict(event='trainingProgress',schemaVersion=2 if 'moe' in extra else 1,runID=self.run_id,sequence=self.sequence,
                       epoch=epoch,epochBatch=epoch_batch,epochBatches=epoch_batches,batches=self.batches,
                       updates=self.updates,elapsedSeconds=max(0.,now-self.start),kind=kind,trainingLoss=loss,
                       **extra))
        self.last=now

    def batch(self,*,epoch,epoch_batch,epoch_batches,numerator,denominator,updated):
        self.epoch_batches=epoch_batches
        self.batches+=1;self.updates+=int(updated);self.pending+=1
        self.numerator+=numerator;self.denominator+=denominator
        if self.moe is not None:
            weight=self.moe.get('aggregationWeight',denominator)
            self.moe_sums[0]+=self.moe['balancePenalty']*weight;self.moe_sums[1]+=self.moe['routerZPenalty']*weight;self.moe_denominator+=weight
        if not (epoch_batch==1 or self.pending>=self.policy['everyBatches'] or self.clock()-self.last>=self.policy['everySeconds'] or epoch_batch==epoch_batches):return
        divisor=1. if self.reduction=='sum' else self.denominator
        # A completely ignored interval has no measured loss, rather than zero.
        loss=self.numerator/divisor if divisor>0 else None
        if loss is not None and not math.isfinite(loss):raise ValueError('Training progress aggregate is nonfinite.')
        extra={}
        if self.moe_denominator>0 and loss is not None:
            balance,z=[value/self.moe_denominator for value in self.moe_sums]
            extra['moe']=dict(balancePenalty=balance,routerZPenalty=z,trainingObjective=loss+balance+z)
        self._emit(epoch,epoch_batch,epoch_batches,'training',loss,numerator=self.numerator,denominator=divisor,**extra)
        self.pending=0;self.numerator=0.;self.denominator=0.;self.moe_sums=[0.,0.];self.moe_denominator=0.

    def validation(self,epoch,training_loss,validation_loss,accuracy):
        self._emit(epoch,self.epoch_batches,self.epoch_batches,'validation',training_loss,validationLoss=validation_loss,accuracy=accuracy)

    @property
    def epoch_batches(self):return self._epoch_batches

    @epoch_batches.setter
    def epoch_batches(self,value):self._epoch_batches=value
