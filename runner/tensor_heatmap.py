"""Bounded derived summaries of actual captured tensors; never scalar samples.

The final two axes form each plane. Retained leading planes are explicit and
all elements of each retained plane contribute. Transfers are chunked, detached,
and performed only at an existing capture boundary, never in a training batch.
"""
import math
import torch


def summarize_tensor(tensor, *, max_tiles=8192, max_planes=32,
                     max_source_elements=8_388_608, check_cancelled=lambda:None):
    check_cancelled()
    shape=list(tensor.shape)
    base=dict(schemaVersion=1,sourceShape=shape,rowAxis=len(shape)-2 if len(shape)>=2 else None,
              columnAxis=len(shape)-1 if shape else None,aggregation='finiteMean',planes=[])
    if tensor.layout!=torch.strided or tensor.is_complex() or tensor.is_quantized:
        return dict(base,coverage='unavailable',reason='This tensor type does not support a heatmap.')
    height,width=shape[-2:] if len(shape)>=2 else (1,shape[0] if shape else 1)
    total_planes=math.prod(shape[:-2]) if len(shape)>2 else 1
    base.update(planeShape=[height,width],totalPlanes=total_planes)
    if tensor.numel()==0:return dict(base,coverage='empty',gridShape=[0,0])
    if any(type(n) is not int or n<1 for n in (max_tiles,max_planes,max_source_elements)):
        raise ValueError('Heatmap budgets must be positive integers.')
    planes=min(total_planes,max_planes,max_tiles)
    if planes*height*width>max_source_elements:
        return dict(base,coverage='unavailable',reason='Heatmap source scan exceeds its capture budget.')
    rows,cols=min(height,64),min(width,64)
    while rows*cols>max_tiles//planes:
        if rows>=cols:rows=max(1,(rows+1)//2)
        else:cols=max(1,(cols+1)//2)
    cells=rows*cols
    base['gridShape']=[rows,cols]
    leading=shape[:-2] if len(shape)>2 else []
    for plane_index in range(planes):
        check_cancelled()
        remainder=plane_index;fixed=[0]*len(leading)
        for axis in range(len(leading)-1,-1,-1):fixed[axis]=remainder%leading[axis];remainder//=leading[axis]
        plane=tensor.detach()
        for index in fixed:plane=plane.select(0,index)
        if len(shape)<2:plane=plane.reshape(height,width)
        sums=torch.zeros(cells,dtype=torch.float64)
        minima=torch.full((cells,),float('inf'),dtype=torch.float64)
        maxima=torch.full((cells,),float('-inf'),dtype=torch.float64)
        counts=[torch.zeros(cells,dtype=torch.int64) for _ in range(4)]
        # Both axes are chunked, including a single exceptionally wide row.
        chunk_width=min(width,262144)
        chunk_height=max(1,262144//chunk_width)
        for top in range(0,height,chunk_height):
            for left in range(0,width,chunk_width):
                check_cancelled()
                value=plane[top:top+chunk_height,left:left+chunk_width].to(device='cpu',dtype=torch.float64)
                y=(torch.arange(top,top+value.shape[0])*rows//height)[:,None]
                x=(torch.arange(left,left+value.shape[1])*cols//width)[None,:]
                indices=(y*cols+x).expand_as(value).reshape(-1)
                flat=value.reshape(-1);finite=torch.isfinite(flat)
                finite_values=torch.where(finite,flat,0.)
                sums.scatter_add_(0,indices,finite_values)
                for target,mask in zip(counts,(finite,torch.isnan(flat),torch.isposinf(flat),torch.isneginf(flat))):
                    target.scatter_add_(0,indices,mask.to(torch.int64))
                minima.scatter_reduce_(0,indices,torch.where(finite,flat,float('inf')),reduce='amin',include_self=True)
                maxima.scatter_reduce_(0,indices,torch.where(finite,flat,float('-inf')),reduce='amax',include_self=True)
        count=counts[0].tolist()
        means=(sums/counts[0].clamp_min(1)).tolist()
        # Extreme finite float64 inputs may overflow a sum. Do not serialize NaN/Inf
        # as an ordinary measurement or silently drop its source population.
        plane_result=dict(index=plane_index,fixedIndices=fixed,
                          mean=[v if n and math.isfinite(v) else None for v,n in zip(means,count)],
                          minimum=[v if n else None for v,n in zip(minima.tolist(),count)],
                          maximum=[v if n else None for v,n in zip(maxima.tolist(),count)],
                          finiteCount=count,nanCount=counts[1].tolist(),
                          positiveInfinityCount=counts[2].tolist(),negativeInfinityCount=counts[3].tolist())
        base['planes'].append(plane_result)
    return dict(base,coverage='complete' if planes==total_planes else 'partialPlanes')


class HeatmapCaptureBudget:
    """One operation's shared optional observation budget, separate from execution."""
    def __init__(self, *, maximum_bytes=800_000, maximum_source_elements=16_777_216, maximum_tiles_per_output=2048):
        self.remaining_bytes=maximum_bytes
        self.remaining_elements=maximum_source_elements
        self.maximum_tiles=maximum_tiles_per_output

    def capture(self,tensor,check_cancelled=lambda:None):
        import json
        unavailable=dict(schemaVersion=1,sourceShape=list(tensor.shape),coverage='unavailable',planes=[],reason='Heatmap capture budget exhausted.')
        if self.remaining_bytes<4096 or self.remaining_elements<1:return unavailable
        value=summarize_tensor(tensor,max_tiles=self.maximum_tiles,max_source_elements=self.remaining_elements,check_cancelled=check_cancelled)
        raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>self.remaining_bytes:return unavailable
        self.remaining_bytes-=len(raw)
        if value['coverage'] in ('complete','partialPlanes'):
            self.remaining_elements-=math.prod(value['planeShape'])*len(value['planes'])
        return value
