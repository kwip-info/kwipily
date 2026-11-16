"""Independent selected-model and epoch-boundary stopping policy.

Selected inference weights never replace terminal optimizer/RNG/continuation
state. Improvement for model selection is strict; stopping has its own delta.
"""
import math


def validate_policy(value):
    if type(value) is not dict or set(value)!={'schemaVersion','modelSelection','monitor','earlyStopping'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=1:
        raise ValueError('Unsupported training control policy.')
    if value['modelSelection'] not in ('finalEpoch','bestValidation') or value['monitor']!='validationLoss':
        raise ValueError('Select Final epoch or Best validation with Validation loss.')
    stop=value['earlyStopping']
    if type(stop) is not dict or set(stop)!={'enabled','patience','minimumImprovement'} or type(stop['enabled']) is not bool or type(stop['patience']) is not int or not 1<=stop['patience']<=10000:
        raise ValueError('Early stopping requires a boolean and patience from 1 to 10000 epochs.')
    delta=stop['minimumImprovement']
    if type(delta) not in (int,float) or not math.isfinite(delta) or not 0<=delta<=1_000_000:
        raise ValueError('Minimum improvement must be finite and nonnegative, at most 1000000.')
    return dict(value,earlyStopping=dict(stop))


class TrainingControl:
    def __init__(self,policy,*,capture=lambda:None,state=None,best_model=None):
        self.policy=validate_policy(policy);self.capture=capture
        self.last_epoch=0;self.selected_epoch=0;self.best_value=None;self.selected_value=None
        self.stopping_reference=None;self.unimproved_epochs=0;self.stop_epoch=None;self.best_model=best_model
        if state is not None:
            keys={'schemaVersion','lastEpoch','selectedEpoch','bestValue','selectedValue','stoppingReference','unimprovedEpochs','stopEpoch'}
            if type(state) is not dict or set(state)!=keys or type(state['schemaVersion']) is not int or state['schemaVersion']!=1:raise ValueError('Invalid training control continuation state.')
            for key in ('lastEpoch','selectedEpoch'):
                if type(state[key]) is not int or not 1<=state[key]<=10000:raise ValueError('Invalid selected/completed training epoch.')
            if state['selectedEpoch']>state['lastEpoch'] or type(state['unimprovedEpochs']) is not int or not 0<=state['unimprovedEpochs']<=10000:raise ValueError('Invalid training stopping counter.')
            for key in ('bestValue','selectedValue','stoppingReference'):
                if type(state[key]) not in (int,float) or not math.isfinite(state[key]):raise ValueError('Invalid retained validation measurement.')
            if state['stopEpoch'] is not None and (type(state['stopEpoch']) is not int or state['stopEpoch']!=state['lastEpoch']):raise ValueError('Invalid early-stop epoch.')
            self.last_epoch=state['lastEpoch'];self.selected_epoch=state['selectedEpoch'];self.best_value=state['bestValue'];self.selected_value=state['selectedValue'];self.stopping_reference=state['stoppingReference'];self.unimproved_epochs=state['unimprovedEpochs'];self.stop_epoch=state['stopEpoch']
            if self.policy['modelSelection']=='bestValidation' and best_model is None:raise ValueError('Selected model weights are required for exact continuation.')

    def observe(self,epoch,validation_loss):
        if type(epoch) is not int or not self.last_epoch<epoch<=10000 or type(validation_loss) not in (int,float) or not math.isfinite(validation_loss):raise ValueError('Training policy requires ordered finite validation observations.')
        self.last_epoch=epoch
        if self.best_value is None or validation_loss<self.best_value:
            self.best_value=validation_loss
            if self.policy['modelSelection']=='bestValidation':
                self.selected_epoch=epoch;self.selected_value=validation_loss;self.best_model=self.capture()
        if self.policy['modelSelection']=='finalEpoch':self.selected_epoch=epoch;self.selected_value=validation_loss
        stop=self.policy['earlyStopping']
        if self.stopping_reference is None or validation_loss<self.stopping_reference-stop['minimumImprovement']:
            self.stopping_reference=validation_loss;self.unimproved_epochs=0
        else:self.unimproved_epochs+=1
        self.stop_epoch=epoch if stop['enabled'] and self.unimproved_epochs>=stop['patience'] else None
        return self.stop_epoch is not None

    def state(self):
        return dict(schemaVersion=1,lastEpoch=self.last_epoch,selectedEpoch=self.selected_epoch,bestValue=self.best_value,selectedValue=self.selected_value,stoppingReference=self.stopping_reference,unimprovedEpochs=self.unimproved_epochs,stopEpoch=self.stop_epoch)

    def report(self):
        return dict(schemaVersion=1,policy=self.policy,completedEpoch=self.last_epoch,selectedEpoch=self.selected_epoch,validationLoss=self.selected_value,stopReason='earlyStopping' if self.stop_epoch is not None else 'epochLimit')
