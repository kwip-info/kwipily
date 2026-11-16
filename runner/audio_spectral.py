"""Bounded deterministic spectral features; display domains are not data transforms."""
import math
import numpy as np
MAX_BYTES=49_990_000


def require(ok,message):
    if not ok:raise ValueError(message)


def spectral_features(samples,rate,c,check_cancelled=lambda:None):
    require(isinstance(samples,np.ndarray) and samples.ndim==2 and samples.dtype==np.float32 and samples.size>0 and samples.nbytes<=MAX_BYTES and np.isfinite(samples).all(),'Spectral features require bounded finite float32 channels × samples.')
    n=c['fftSize'];hop=c['spectralHop'];bands=c['melBands'];mode=c['representation'];center=c['center'];low=c['minimumHz'];high=c['maximumHz'] or rate/2
    require(type(rate) is int and rate>0 and type(n) is int and 16<=n<=4096 and type(hop) is int and 1<=hop<=n,'Choose a bounded FFT and positive hop no larger than the FFT.')
    require(type(center) is bool and c['window'] in ('hann','hamming','rectangular') and mode in ('magnitude','power','logMel'),'Unsupported spectral representation/window.')
    require(type(bands) is int and 8<=bands<=256 and 0<=low<high<=rate/2 and -120<=c['logFloorDB']<=-1,'Spectral frequency/mel/log bounds are invalid.')
    channels,length=samples.shape;pad=n//2 if center else 0;frames=1+(length+2*pad-n)//hop
    require(frames>0,'Recording is shorter than one FFT window; enable centered zero padding or choose a smaller FFT.')
    bins=bands if mode=='logMel' else n//2+1
    require(channels*frames*bins<=MAX_BYTES//4 and channels*frames*n<=25_000_000,'Spectral expansion exceeds its byte/work allowance; use shorter audio or a larger hop.')
    check_cancelled();values=np.pad(samples,((0,0),(pad,pad))) if pad else samples
    window=np.hanning(n+1)[:-1] if c['window']=='hann' else np.hamming(n+1)[:-1] if c['window']=='hamming' else np.ones(n)
    filters=None
    if mode=='logMel':
        # Explicit HTK mel triangles, unit peak. No hidden normalization or learned state.
        mel=lambda hz:2595*np.log10(1+np.asarray(hz)/700)
        hz=lambda m:700*(10**(np.asarray(m)/2595)-1)
        points=hz(np.linspace(mel(low),mel(high),bands+2));frequencies=np.arange(n//2+1)*rate/n
        filters=np.maximum(0,np.minimum((frequencies[None,:]-points[:-2,None])/(points[1:-1,None]-points[:-2,None]),(points[2:,None]-frequencies[None,:])/(points[2:,None]-points[1:-1,None])))
        require(np.all(filters.sum(axis=1)>0),'Mel bands have empty frequency support; increase the FFT size or reduce bands.')
    result=np.empty((channels,frames,bins),dtype='<f4')
    for start in range(0,frames,32):
        check_cancelled();end=min(frames,start+32)
        indices=np.arange(start,end)[:,None]*hop+np.arange(n)
        spectrum=np.fft.rfft(values[:,indices]*window,n=n,axis=-1)
        power=spectrum.real**2+spectrum.imag**2
        if mode=='magnitude':out=np.sqrt(power)
        elif mode=='power':out=power
        else:out=np.maximum(c['logFloorDB'],10*np.log10(np.maximum(power@filters.T,10**(c['logFloorDB']/10))))
        require(np.isfinite(out).all() and (np.abs(out)<=np.finfo(np.float32).max).all(),'Spectral output exceeds finite float32.');result[:,start:end]=out
    metadata=dict(representation=mode,fftSize=n,frameHopSamples=hop,frameOffsetSamples=-pad,window=c['window'],center=center,melBands=bands,minimumHz=int(low),maximumHz=high,logFloorDB=c['logFloorDB'],melScale='htk',melNormalization='none',logReference='unitPower')
    return dict(array=result,axes=['channels','frames','frequency'],metadata=metadata)


def validate_metadata(v):
    keys={'representation','fftSize','frameHopSamples','frameOffsetSamples','window','center','melBands','minimumHz','maximumHz','logFloorDB','melScale','melNormalization','logReference'}
    require(type(v) is dict and set(v)==keys,'Spectral metadata requires its complete versioned recipe.')
    require(v['representation'] in ('magnitude','power','logMel') and v['window'] in ('hann','hamming','rectangular') and type(v['center']) is bool,'Invalid spectral representation.')
    require(all(type(v[k]) is int for k in keys-{'representation','window','center','melScale','melNormalization','logReference','minimumHz','maximumHz'}),'Spectral geometry requires exact integers.')
    require(all(type(v[k]) in (int,float) and math.isfinite(v[k]) for k in ('minimumHz','maximumHz')),'Invalid spectral frequency values.')
    require(16<=v['fftSize']<=4096 and 1<=v['frameHopSamples']<=v['fftSize'] and v['frameOffsetSamples']==(-(v['fftSize']//2) if v['center'] else 0) and 8<=v['melBands']<=256 and 0<=v['minimumHz']<v['maximumHz']<=384000 and -120<=v['logFloorDB']<=-1 and v['melScale']=='htk' and v['melNormalization']=='none' and v['logReference']=='unitPower','Invalid spectral descriptor bounds.')
