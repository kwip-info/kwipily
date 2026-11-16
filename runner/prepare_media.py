"""Deterministic bounded preparation; no fitting, model downloads or file grants."""
import copy
import json
from pathlib import Path
import math
import numpy as np

MAX_BYTES=49_990_000
MAX_RESAMPLE_WORK=20_000_000
MAX_WINDOWS=10000


CATALOG=json.loads(Path(__file__).with_name('prepare-media-options.json').read_text())


def defaults(kind='image'):
    result=copy.deepcopy(CATALOG['defaults']);result['kind']=kind
    if kind=='audio':result['inputColumn']='audio_content'
    return result


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_config(c):
    optional=set(['representation', 'fftSize', 'spectralHop', 'window', 'center', 'melBands', 'minimumHz', 'maximumHz', 'logFloorDB'])|{'outputMode','cropEnabled','cropX','cropY','cropWidth','cropHeight','rotation'}
    require(type(c) is dict and not set(c)-set(defaults()) and set(defaults())-optional<=set(c),'Prepare media requires its complete versioned settings.')
    c=dict(defaults(),**c)
    require(c['representation']=='samples' or c['kind']=='audio' and c['sampleFormat']=='float32','Spectral output requires Audio with explicit float32 conversion.')
    require(type(c['cropEnabled']) is bool and c['rotation'] in (0,90,180,270),'Choose an explicit crop and quarter-turn rotation.')
    for key,values in CATALOG['choices'].items():
        require(type(c[key]) is str and c[key] in values,'Unsupported Prepare media '+key+'.')
    for key,(low,high) in CATALOG['integerBounds'].items():
        require(type(c[key]) is int and low<=c[key]<=high,'Prepare media '+key+' exceeds its bound.')
    require(type(c['center']) is bool and c['spectralHop']<=c['fftSize'],'Choose valid spectral centering and hop.')
    require(c['width']*c['height']<=16_000_000,'Image preparation exceeds its pixel bound.')
    require(type(c['matte']) is list and len(c['matte'])==3 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in c['matte']),'Composite matte requires three finite unit values.')
    for key in ('inputColumn','outputColumn'):
        require(type(c[key]) is str and c[key].strip() and len(c[key])<=200 and '\0' not in c[key],'Prepare media column names must be bounded and nonblank.')
    require(c['inputColumn']!=c['outputColumn'],'Prepared data must use a new column; source media is retained.')
    if c['kind']=='audio':
        require(c['scaling']=='none' and c['sizeMode']=='preserve' and c['color']=='preserve' and c['alpha']=='preserve', 'Image settings cannot transform audio.')
        require(c['windowSamples']>0 or c['hopSamples']==0,'A hop requires a fixed window size.')
        require(c['windowSamples']==0 or c['hopSamples']==0 or c['hopSamples']<=c['windowSamples'],'Window hop cannot leave silent gaps.')
    return copy.deepcopy(c)


def checked_array(samples,shape,dtype):
    require(type(shape) is list and len(shape)==2 and all(type(n) is int and n>0 for n in shape),'Audio needs sample × channel shape.')
    require(dtype in ('uint8','int16','int32'),'Only qualified integer PCM audio can be prepared.')
    dt=np.dtype(dtype).newbyteorder('<');count=shape[0]*shape[1]
    require(count<=MAX_BYTES//dt.itemsize and type(samples) is bytes and len(samples)==count*dt.itemsize,'PCM audio exceeds its decoded byte allowance or shape.')
    return np.frombuffer(samples,dtype=dt).reshape(shape)


def _resample(values,rate,target,radius,check):
    """Rational-phase windowed-sinc low-pass with zero boundary extension.

    Evaluate a bounded phase at each output position, in cancellable blocks.
    There is no whole-population matrix, hidden rate choice or optional backend.
    """
    cutoff=min(1.,target/rate);half=math.ceil(radius/cutoff)
    count=(len(values)*target+rate-1)//rate;taps=2*half+1
    require(max(rate/target,target/rate)<=16 and count*taps*values.shape[1]<=MAX_RESAMPLE_WORK,
            'Resampling exceeds its filter-work allowance; shorten the source or use a closer rate.')
    require(count*values.shape[1]<=MAX_BYTES//4,'Resampled audio exceeds its byte allowance.')
    result=np.empty((count,values.shape[1]),dtype='<f4')
    offsets=np.arange(-half,half+1)
    for begin in range(0,count,256):
        check();end=min(count,begin+256)
        locations=np.arange(begin,end,dtype=np.float64)*rate/target
        indices=np.floor(locations).astype(np.int64)[:,None]+offsets
        distance=locations[:,None]-indices
        window=np.where(np.abs(distance)<=half,.5+.5*np.cos(np.pi*distance/half),0.)
        weights=cutoff*np.sinc(cutoff*distance)*window
        weights/=weights.sum(axis=1,keepdims=True)
        valid=(indices>=0)&(indices<len(values));weights*=valid
        for channel in range(values.shape[1]):
            result[begin:end,channel]=np.sum(values[np.clip(indices,0,len(values)-1),channel]*weights,axis=1)
    check();return result


def audio_windows(samples,*,shape,dtype,metadata,config,check_cancelled=lambda:None,axes=None):
    c=validate_config(config);check=check_cancelled;check()
    require(c['kind']=='audio','Choose audio preparation for an audio column.')
    axes=axes or (['channels','samples'] if dtype=='float32' else ['samples','channels'])
    require(axes in (['samples','channels'],['channels','samples']),'Audio preparation requires waveform axes.')
    prepared=axes==['channels','samples']
    if dtype=='float32':
        require(len(shape)==2 and shape[0]*shape[1]<=MAX_BYTES//4 and len(samples)==shape[0]*shape[1]*4,'Prepared audio exceeds its byte allowance.')
        values=np.frombuffer(samples,dtype='<f4').reshape(shape)
        require(np.isfinite(values).all(),'Prepared audio must be finite.')
    else:values=checked_array(samples,shape,dtype)
    if prepared:values=values.T
    shape=list(values.shape)
    if prepared:
        valid=metadata.get('validSamples',len(values));require(type(valid) is int and 0<valid<=len(values),'Prepared waveform valid length is invalid.')
        values=values[:valid]
    rate=metadata.get('sampleRate');bits=metadata.get('sourceBits')
    require(dtype=='float32' or type(bits) is int and {8:'uint8',16:'int16',24:'int32',32:'int32'}.get(bits)==dtype,'Audio precision is missing or incompatible; normalize PCM in the original Prepare media step and rerun it.')
    require(type(rate) is int and 1<=rate<=768000,'Audio sample rate is invalid.')
    if c['sampleFormat']=='float32' and dtype!='float32':
        require(values.size<=MAX_BYTES//4,'Float audio exceeds its decoded byte allowance.')
        values=values.astype('<f4')
        if bits==8:values=(values-128)/128
        else:values/=2**(bits-1)
    if c['channels']=='select':
        require(c['channelIndex']<shape[1],'Selected audio channel is missing.')
        values=values[:,c['channelIndex']:c['channelIndex']+1]
    elif c['channels']=='mono' and shape[1]>1:
        require(c['sampleFormat']=='float32','Mono averaging requires explicit float32 amplitude conversion.')
        values=values.mean(axis=1,keepdims=True,dtype=np.float32)
    target=c['targetRate'] or rate
    if target!=rate:
        require(c['sampleFormat']=='float32','Resampling requires explicit float32 amplitude conversion.')
        values=_resample(values,rate,target,c['filterRadius'],check)
    width=c['windowSamples'] or len(values);hop=c['hopSamples'] or width
    count=(len(values)+hop-1)//hop if c['tail']=='pad' else max(0,(len(values)-width)//hop+1)
    require(count<=MAX_WINDOWS,'Audio window expansion exceeds its window-count allowance.')
    require(width*values.shape[1]*values.dtype.itemsize<=MAX_BYTES and width<=MAX_BYTES//8,'An audio window or its int64 mask exceeds its artifact allowance.')
    # Charge expansion before creating any windows (the caller owns aggregate disk).
    require(count*(width*values.shape[1]*values.dtype.itemsize+width*8)<=190_000_000,'Audio window expansion exceeds the run artifact allowance.')
    for index in range(count):
        check();start=index*hop;end=min(start+width,len(values));valid=end-start
        window=np.zeros((values.shape[1],width),dtype=values.dtype)
        window[:,:valid]=values[start:end].T
        mask=np.zeros(width,dtype='<i8');mask[:valid]=1
        if c['maskConvention']=='blocked':mask=1-mask
        original_rate=metadata.get('sourceRate',rate);origin=metadata.get('sourceStart',0)
        yield dict(array=window,axes=['channels','samples'],mask=mask,
                   metadata=dict(sampleRate=target,sourceRate=original_rate,sourceStart=origin+start*original_rate//target,
                                 sourceEnd=min(metadata.get('sourceEnd',shape[0]),origin+(end*original_rate+target-1)//target),validSamples=valid,
                                 **({'sourceBits':bits} if window.dtype.name!='float32' else {})))


def image_tensor(pixels,config,check_cancelled=lambda:None):
    c=validate_config(config);check=check_cancelled;check()
    require(c['kind']=='image' and isinstance(pixels,np.ndarray) and pixels.ndim==3 and pixels.shape[2] in (1,2,3,4) and pixels.dtype.name in ('uint8','uint16','int32','float32'),'Prepare image requires qualified height × width × channel pixels.')
    require(pixels.nbytes<=MAX_BYTES and pixels.shape[0]*pixels.shape[1]<=16_000_000,'Image exceeds its decoded byte allowance.')
    if __package__:from .ocr import prepare_pixels
    else:from ocr import prepare_pixels
    crop=[c['cropX'],c['cropY'],c['cropX']+c['cropWidth'],c['cropY']+c['cropHeight']] if c['cropEnabled'] else None
    pixels,source_geometry=prepare_pixels(pixels,crop=crop,rotation=c['rotation'])
    source_height,source_width,channels=pixels.shape
    require(pixels.dtype.name!='float32' or np.isfinite(pixels).all(),'Image samples must be finite.')
    has_alpha=channels in (2,4)
    if has_alpha and c['alpha']!='preserve':
        require(pixels.dtype==np.uint8,'Alpha conversion currently requires uint8 source pixels.')
        color=pixels[:,:,:-1]
        if c['alpha']=='composite':
            a=pixels[:,:,-1:].astype(np.float32)/255
            matte=np.array(c['matte'] if channels==4 else [.299*c['matte'][0]+.587*c['matte'][1]+.114*c['matte'][2]],dtype=np.float32)*255
            color=np.rint(color*a+matte*(1-a)).astype(np.uint8)
        pixels=color;channels-=1;has_alpha=False
    if c['color']!='preserve':
        require(not has_alpha,'Choose an explicit alpha drop/composite policy before color conversion.')
        require(pixels.dtype==np.uint8,'Color conversion currently requires uint8 source pixels.')
        from PIL import Image
        pixels=np.asarray(Image.fromarray(pixels[:,:,0] if channels==1 else pixels).convert(c['color']))
        if pixels.ndim==2:pixels=pixels[:,:,None]
        channels=pixels.shape[2]
    geometry=[1,0,0,0,1,0,0,0,1]
    if c['sizeMode']!='preserve':
        from PIL import Image
        width,height=c['width'],c['height'];require(width*height*channels*max(4,pixels.dtype.itemsize)<=MAX_BYTES,'Prepared image exceeds its byte allowance.')
        mode=c['sizeMode'];scale=max(width/source_width,height/source_height) if mode=='crop' else min(width/source_width,height/source_height)
        w,h=(width,height) if mode=='resize' else (max(1,round(source_width*scale)),max(1,round(source_height*scale)))
        require(w*h*channels*4<=MAX_BYTES,'Intermediate resized image exceeds its byte allowance.')
        method={'nearest':Image.Resampling.NEAREST,'bilinear':Image.Resampling.BILINEAR,'bicubic':Image.Resampling.BICUBIC}[c['interpolation']]
        # Preserve high-precision integer values instead of silently passing via float32.
        require(pixels.dtype!=np.int32 or c['interpolation']=='nearest','Interpolating int32 images requires explicit float32 conversion before this adapter.')
        resized=[]
        for i in range(channels):
            check();plane=Image.fromarray(pixels[:,:,i]);resized.append(np.asarray(plane.resize((w,h),method)))
        pixels=np.stack(resized,axis=2)
        x=(width-w)//2 if mode!='crop' else -((w-width)//2)
        y=(height-h)//2 if mode!='crop' else -((h-height)//2)
        if mode=='fitPad':
            canvas=np.zeros((height,width,channels),dtype=pixels.dtype);canvas[y:y+h,x:x+w]=pixels;pixels=canvas
        elif mode=='crop':pixels=pixels[-y:-y+height,-x:-x+width]
        sx=source_width/w;sy=source_height/h
        geometry=[sx,0,(.5-x)*sx-.5,0,sy,(.5-y)*sy-.5,0,0,1]
    if c['sampleFormat']=='float32':
        original=pixels.dtype;pixels=pixels.astype('<f4')
        if c['scaling']=='unit':
            require(original in (np.dtype('uint8'),np.dtype('uint16')),'Unit scaling requires known unsigned integer pixel bounds.')
            pixels/=255 if original==np.uint8 else 65535
    else:require(c['scaling']=='none','Pixel scaling requires explicit float32 conversion.')
    require(pixels.dtype.name in ('uint8','uint16','int32','float32'),'Unsupported prepared image precision.')
    axes=['height','width','channels']
    if c['layout']=='CHW':pixels=pixels.transpose(2,0,1);axes=['channels','height','width']
    geometry=(source_geometry@np.asarray(geometry).reshape(3,3)).reshape(-1).tolist()
    check();return dict(array=np.ascontiguousarray(pixels),axes=axes,geometry=geometry)


def apply(table,config,check,image_reader=None):
    if __package__:
        from . import dataflow as df,media_artifacts as media
        from .image_artifacts import decode_pixels,validate_reference as image_reference,publish_pixels
    else:
        import dataflow as df,media_artifacts as media
        from image_artifacts import decode_pixels,validate_reference as image_reference,publish_pixels
    c=validate_config(config);column=table.index(c['inputColumn'])
    require(table.columns[column]['dtype'] in (('image',) if c['kind']=='image' else ('audio','tensor')),'Select a captured '+c['kind']+' column.')
    spectral=c['kind']=='audio' and c['representation']!='samples'
    count_name='_valid_frames' if spectral else '_valid_samples'
    names=[c['outputColumn']]+([c['outputColumn']+'_mask',c['outputColumn']+count_name,c['outputColumn']+'_parent'] if c['kind']=='audio' else [])
    require(not set(names)&{col['name'] for col in table.columns},'Prepared output columns already exist; choose a new name.')
    image_output=c['kind']=='image' and c['outputMode']=='image'
    if image_output:
        require(c['sampleFormat']=='preserve' and c['scaling']=='none','Image output preserves pixels; choose tensor output for float conversion/scaling.')
        names += [c['outputColumn']+'_source',c['outputColumn']+'_to_source']
        require(not set(names)&{col['name'] for col in table.columns},'Prepared image lineage columns already exist.')
    schema=list(table.columns)+[dict(name=names[0],dtype='image' if image_output else 'tensor',nullable=False)]
    if image_output:schema += [dict(name=names[1],dtype='image',nullable=False),dict(name=names[2],dtype='string',nullable=False)]
    if c['kind']=='audio':schema += [dict(name=names[1],dtype='tensor',nullable=False),dict(name=names[2],dtype='int64',nullable=False),dict(name=names[3],dtype='string',nullable=False)]
    recipe=df.digest(c)
    def publish(array,axes,metadata):
        check();array=np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'),copy=False))
        raw,ref=media.encode_asset(array.tobytes(),kind='tensor',shape=list(array.shape),axes=axes,dtype=array.dtype.name,metadata=metadata)
        return media.publish_encoded(raw,ref,check)
    def pairs():
        emitted=0
        for row,identity in df.row_pairs(table):
            check();source=row[column]
            require(source is not None,'Prepare media requires a nonmissing captured item; filter missing/error rows explicitly.')
            if c['kind']=='audio':
                require(source['axes'] in (['samples','channels'],['channels','samples']),'Audio preparation requires waveform axes, not spectral features.')
                _,samples=media.read_asset(source,check)
                for index,window in enumerate(audio_windows(samples,shape=source['shape'],dtype=source['dtype'],metadata=source['metadata'],config=c,check_cancelled=check,axes=source['axes'])):
                    emitted+=1;require(emitted<=MAX_WINDOWS,'Prepared media exceeds its expanded observation allowance.')
                    metadata=dict(window['metadata'],parentSHA256=source['sha256'],recipeSHA256=recipe)
                    if spectral:
                        if __package__:from .audio_spectral import spectral_features
                        else:from audio_spectral import spectral_features
                        features=spectral_features(window['array'],metadata['sampleRate'],c,check)
                        frames=features['array'].shape[1];centers=np.arange(frames)*c['spectralHop']+(0 if c['center'] else c['fftSize']//2)
                        valid=(centers<metadata.pop('validSamples')).astype('<i8')
                        metadata.update(validFrames=int(valid.sum()),spectral=features['metadata'])
                        if c['maskConvention']=='blocked':valid=1-valid
                        tensor=publish(features['array'],features['axes'],metadata)
                        mask=publish(valid,['frames'],dict(metadata,maskConvention=c['maskConvention']))
                        count=metadata['validFrames']
                    else:
                        tensor=publish(window['array'],window['axes'],metadata)
                        mask=publish(window['mask'],['samples'],dict(metadata,maskConvention=c['maskConvention']))
                        count=metadata['validSamples']
                    yield [*row,tensor,mask,count,identity],identity+':media:'+str(index)
            else:
                ref=image_reference(source);require(callable(image_reader),'Image preparation requires the owned captured pixel reader.')
                header,samples=decode_pixels(image_reader(ref),expected_sha256=ref['sha256'])
                dimensions=(ref['height'],ref['width'],ref['channels']) if ref['layout']=='HWC' else (ref['channels'],ref['height'],ref['width'])
                pixels=np.frombuffer(samples,dtype=np.dtype(ref['dtype']).newbyteorder('<')).reshape(dimensions)
                if ref['layout']=='CHW':pixels=pixels.transpose(1,2,0)
                # CMYK is not RGB even though both may have multiple channels.
                require(header['policy']['actualMode']!='CMYK','CMYK media requires an explicitly qualified color adapter.')
                result=image_tensor(pixels,c,check)
                if image_output:
                    pixels=result['array'];h,w,ch=(pixels.shape[1],pixels.shape[2],pixels.shape[0]) if c['layout']=='CHW' else pixels.shape
                    modes={(1,'uint8'):'L',(2,'uint8'):'LA',(3,'uint8'):'RGB',(4,'uint8'):'RGBA',(1,'uint16'):'I;16',(1,'int32'):'I',(1,'float32'):'F'}
                    require((ch,pixels.dtype.name) in modes,'Prepared image output mode is unsupported.')
                    output=publish_pixels(pixels.tobytes(),width=w,height=h,channels=ch,dtype=pixels.dtype.name,layout=c['layout'],policy=dict(orientation='stored',colorMode='preserve',palette='expandRGBA',actualMode=modes[(ch,pixels.dtype.name)]),check_cancelled=check)
                    yield [*row,output,ref,json.dumps(result['geometry'],separators=(',',':'))],identity
                else:
                    tensor=publish(result['array'],result['axes'],dict(parentSHA256=ref['sha256'],recipeSHA256=recipe,geometry=result['geometry']))
                    yield [*row,tensor],identity
    return df.streaming_rows(table,pairs(),schema,check)


def verify_output(table,config,output,check):
    """Verify cached provenance/observation correspondence without refitting data."""
    if __package__:from . import dataflow as df,media_artifacts as media
    else:import dataflow as df,media_artifacts as media
    c=validate_config(config);index=table.index(c['inputColumn']);recipe=df.digest(c)
    require(output.source_digest==table.source_digest and output.role==table.role and output.partition_digest==table.partition_digest and output.fitted_states==table.fitted_states,'Cached media partition/source lineage differs.')
    count=len(table.columns);require(list(output.columns[:count])==list(table.columns),'Cached media source schema differs.')
    expected_names=[c['outputColumn']]+([c['outputColumn']+'_mask',c['outputColumn']+'_valid_samples',c['outputColumn']+'_parent'] if c['kind']=='audio' else [])
    require([col['name'] for col in output.columns[count:]]==expected_names,'Cached media output columns differ.')
    cursor=iter(df.row_pairs(output))
    for row,rid in df.row_pairs(table):
        check();source=row[index];require(source is not None,'Cached media source is missing.')
        if c['kind']=='audio':
            rate=source['metadata']['sampleRate'];target=c['targetRate'] or rate;n=(source['shape'][0]*target+rate-1)//rate
            width=c['windowSamples'] or n;hop=c['hopSamples'] or width
            windows=(n+hop-1)//hop if c['tail']=='pad' else max(0,(n-width)//hop+1)
        else:windows=1
        for i in range(windows):
            check();pair=next(cursor,None);require(pair is not None,'Cached media is missing observations.')
            actual,identity=pair;require(list(actual[:count])==list(row) and identity==(rid+':media:'+str(i) if c['kind']=='audio' else rid),'Cached media observations differ from their source.')
            ref=media.validate_reference(actual[count]);metadata=ref['metadata']
            require(metadata.get('parentSHA256')==source['sha256'] and metadata.get('recipeSHA256')==recipe,'Cached media preparation recipe/parent differs.')
            if c['kind']=='audio':
                valid=min(width,n-i*hop);channels=source['shape'][1] if c['channels']=='preserve' else 1
                require(ref['shape']==[channels,width] and ref['axes']==['channels','samples'] and ref['dtype']==('float32' if c['sampleFormat']=='float32' else source['dtype']),'Cached audio tensor geometry differs.')
                require(metadata['sourceStart']==i*hop*rate//target and metadata['sourceEnd']==min(source['shape'][0],((i*hop+valid)*rate+target-1)//target) and metadata['validSamples']==valid and metadata['sampleRate']==target and metadata['sourceRate']==rate,'Cached audio interval differs.')
                mask=media.validate_reference(actual[count+1]);require(mask['shape']==[width] and mask['axes']==['samples'] and mask['dtype']=='int64' and mask['metadata']==dict(metadata,maskConvention=c['maskConvention']) and actual[count+2]==valid and actual[count+3]==rid,'Cached audio mask/parent/length differs.')
    require(next(cursor,None) is None,'Cached media has extra observations.')
