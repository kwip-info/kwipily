"""Post-run frames sampled from existing first validation-batch forwards.

Only small JSON summaries survive each validation iteration. Shared scan/byte
budgets and a fixed maximum frame count bound memory and artifact growth.
"""
import json
import math
if __package__:
    from .tensor_heatmap import HeatmapCaptureBudget
    from .output_routing import map_stage
else:
    from tensor_heatmap import HeatmapCaptureBudget
    from output_routing import map_stage


class HeatmapPlayback:
    def __init__(self,*,epoch_limit,max_frames,maximum_bytes=600_000,maximum_source_elements=33_554_432):
        if type(max_frames) is not int or not 0<=max_frames<=24:raise ValueError('Heatmap playback retains from 0 to 24 frames.')
        self.max_frames=max_frames;self.epoch_limit=epoch_limit
        stride=math.ceil(max(1,epoch_limit-1)/max(1,max_frames-1))
        self.epochs=([epoch_limit] if max_frames==1 else sorted(set([1,*range(1,epoch_limit+1,stride),epoch_limit])))[:max_frames] if max_frames else []
        self.remaining_bytes=maximum_bytes;self.remaining_elements=maximum_source_elements;self.frames=[]

    def should_capture(self,epoch):return epoch in self.epochs and len(self.frames)<self.max_frames

    def capture(self,epoch,values,*,source_rows,check_cancelled=lambda:None):
        if not self.should_capture(epoch):return
        remaining=max(1,sum(e>=epoch for e in self.epochs))
        byte_budget=self.remaining_bytes//remaining
        outputs=sum(len(value) if isinstance(value,dict) else 1 for value in values.values())
        budget=HeatmapCaptureBudget(maximum_bytes=byte_budget,maximum_source_elements=self.remaining_elements//remaining,maximum_tiles_per_output=max(1,min(256,byte_budget//max(1,outputs*120))))
        summaries={node:map_stage(value,lambda tensor:budget.capture(tensor,check_cancelled)) for node,value in values.items()}
        self.remaining_bytes-=byte_budget-budget.remaining_bytes
        self.remaining_elements-=self.remaining_elements//remaining-budget.remaining_elements
        self.frames.append((epoch,source_rows,summaries))

    def decorate(self,previews):
        used=len(json.dumps(previews,allow_nan=False,separators=(',',':')).encode())
        for node,preview in previews.items():
            ports=preview.get('outputs',{'out':preview})
            for port,value in ports.items():
                frames=[]
                for epoch,rows,summaries in self.frames:
                    item=summaries[node]
                    summary=item[port] if 'schemaVersion' not in item else item
                    if summary['coverage'] in ('complete','partialPlanes'):
                        frames.append(dict(epoch=epoch,sourceRows=rows,summary=summary))
                playback=dict(schemaVersion=1,source='validationFirstBatch',frames=frames)
                extra=len(json.dumps(playback,allow_nan=False,separators=(',',':')).encode())+32
                if used+extra>1_800_000:
                    playback=dict(schemaVersion=1,source='validationFirstBatch',frames=[],reason='Heatmap playback artifact budget exhausted.')
                    extra=len(json.dumps(playback).encode())+32
                value['heatmapPlayback']=playback;used+=extra


def bound_completion_heatmaps(event,*,maximum_bytes=900_000):
    """Budget optional evidence against the actual serialized completion envelope.

    Keep scalar samples/predictions untouched. The artifact and event share the
    same preview objects, so pruning occurs before either is authenticated.
    Required legacy samples that already exceed the bound remain a separate
    admission constraint; optional heatmaps cannot make that base any larger
    than their explicit unavailable headers.
    """
    encoded=lambda value:len(json.dumps(value,allow_nan=False).encode())
    if encoded(event)<=maximum_bytes:return
    retained=[]
    for preview in event.get('previews',{}).values():
        for value in preview.get('outputs',{'out':preview}).values():
            if 'heatmap' not in value:continue
            summary=value['heatmap'];playback=value.get('heatmapPlayback')
            retained.append((value,summary,playback))
            value['heatmap']=dict(schemaVersion=1,sourceShape=summary['sourceShape'],rowAxis=summary.get('rowAxis'),columnAxis=summary.get('columnAxis'),planes=[],coverage='unavailable',reason='Heatmap completion transport budget exhausted.')
            if playback is not None:value['heatmapPlayback']=dict(schemaVersion=1,source='validationFirstBatch',frames=[],reason='Heatmap completion transport budget exhausted.')
    used=encoded(event)
    for value,summary,playback in retained:
        size=encoded(value)
        candidate=dict(value,heatmap=summary);delta=encoded(candidate)-size
        if used+delta<=maximum_bytes:
            value['heatmap']=summary;used+=delta;size+=delta
        if playback is not None:
            candidate_playback=dict(playback);frames=list(playback['frames'])
            while frames:
                candidate_playback['frames']=frames
                delta=encoded(dict(value,heatmapPlayback=candidate_playback))-size
                if used+delta<=maximum_bytes:
                    value['heatmapPlayback']=candidate_playback;used+=delta;break
                frames.pop()
