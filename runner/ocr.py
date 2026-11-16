"""Reviewed English printed OCR. No acquisition, remote code, arbitrary paths or pickle fallback.

The application supplies exact owned model bytes. We construct only reviewed built-in
architectures; EasyOCR's Reader initializer/get_detector/get_recognizer are never called.
"""
from io import BytesIO
import hashlib
import math
import numpy as np

PACK='english-printed-v1'
ASSETS={
 'craft_mlt_25k.pth':(83152330,'4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17'),
 'english_g2.pth':(15143997,'e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8')}

def require(value,message):
    if not value:raise ValueError(message)


def defaults():
    import json
    from pathlib import Path
    return json.loads(Path(__file__).with_name('ocr-options.json').read_text())


def validate(config):
    require(type(config) is dict and set(config)==set(defaults()),'OCR requires its complete versioned settings.')
    c=dict(config)
    require(c['pack']==PACK and c['device']=='cpu','Select the verified English printed-text CPU pack.')
    require(c['decoder'] in ('greedy','beamsearch'),'Choose Greedy or Beam search.')
    for key,low,high in [('beamWidth',1,20),('minimumSize',1,1000),('canvasSize',64,2560),('maximumRegions',1,1000),('maximumCharacters',1,1000000)]:
        require(type(c[key]) is int and low<=c[key]<=high,'OCR '+key+' exceeds its bound.')
    for key,low,high in [('reviewThreshold',0,1),('textThreshold',0,1),('lowText',0,1),('linkThreshold',0,1),('magnification',.1,2),('widthThreshold',0,3),('slopeThreshold',0,1),('ycenterThreshold',0,3),('heightThreshold',0,3),('margin',0,.5)]:
        require(type(c[key]) in (int,float) and math.isfinite(c[key]) and low<=c[key]<=high,'OCR '+key+' exceeds its bound.')
    require(type(c['rotationCandidates']) is list and len(c['rotationCandidates'])<=3 and len(set(c['rotationCandidates']))==len(c['rotationCandidates']) and all(type(x) is int and x in (90,180,270) for x in c['rotationCandidates']),'Choose distinct quarter-turn recognition candidates.')
    for key in ('inputColumn','outputColumn','allowlist','blocklist','groupBy','pageColumn','separator'):
        require(type(c[key]) is str and len(c[key])<=(1000 if key in ('allowlist','blocklist') else 200) and '\0' not in c[key],'OCR '+key+' must be bounded text.')
    require(c['inputColumn'].strip() and c['outputColumn'].strip() and c['inputColumn']!=c['outputColumn'],'OCR needs distinct nonblank source/output columns.')
    require(not(c['allowlist'] and c['blocklist']),'Use a character allowlist or blocklist, not both.')
    return c


def prepare_pixels(pixels,*,crop=None,rotation=0):
    """Output-center → original-center matrix; integer half-open crop then clockwise turn."""
    require(isinstance(pixels,np.ndarray) and pixels.ndim==3 and 1<=pixels.shape[2]<=4 and pixels.size<=50_000_000,'Expected bounded HWC pixels.')
    h,w=pixels.shape[:2];matrix=np.eye(3)
    if crop is not None:
        require(type(crop) is list and len(crop)==4 and all(type(x) is int for x in crop),'Crop requires four integer bounds.')
        x0,y0,x1,y1=crop;require(0<=x0<x1<=w and 0<=y0<y1<=h,'Crop must lie inside the source image.')
        pixels=pixels[y0:y1,x0:x1];matrix[:2,2]=[x0,y0];h,w=pixels.shape[:2]
    require(type(rotation) is int and rotation in (0,90,180,270),'Image rotation requires quarter turns.')
    inverse={0:np.eye(3),90:np.array([[0,1,0],[-1,0,h-1],[0,0,1]]),180:np.array([[-1,0,w-1],[0,-1,h-1],[0,0,1]]),270:np.array([[0,-1,w-1],[1,0,0],[0,0,1]])}[rotation]
    return np.ascontiguousarray(np.rot90(pixels,k=-(rotation//90))),matrix@inverse


def map_quad(points,matrix):
    p=np.asarray(points,dtype=np.float64);m=np.asarray(matrix,dtype=np.float64)
    require(p.shape==(4,2) and m.shape==(3,3) and np.isfinite(p).all() and np.isfinite(m).all(),'OCR geometry must be a finite quadrilateral and transform.')
    out=np.c_[p,np.ones(4)]@m.T
    require(np.all(out[:,2]!=0),'OCR transform has invalid homogeneous coordinates.')
    return out[:,:2]/out[:,2,None]


def normalize_regions(results,*,config,width,height,matrix=None):
    c=validate(config);require(len(results)<=c['maximumRegions'],'OCR exceeds its region allowance; reduce the source or processing size.')
    normalized=[];characters=0
    for box,text,score in results:
        require(isinstance(text,str) and math.isfinite(float(score)) and 0<=float(score)<=1,'OCR returned invalid text/confidence.')
        points=map_quad(box,np.eye(3) if matrix is None else matrix)
        require(np.all(points[:,0]>=-1) and np.all(points[:,1]>=-1) and np.all(points[:,0]<=width+1) and np.all(points[:,1]<=height+1),'OCR geometry exceeds its source image.')
        characters+=len(text);require(characters<=c['maximumCharacters'],'OCR exceeds its recognized character allowance.')
        row=dict(text=text,confidence=float(score),review='lowConfidence' if score<c['reviewThreshold'] else 'none')
        for i,(x,y) in enumerate(points):row['x'+str(i)]=float(x);row['y'+str(i)]=float(y)
        normalized.append(row)
    # Stable top-to-bottom then left-to-right order, without inventing layout semantics.
    return sorted(normalized,key=lambda r:(round(min(r['y0'],r['y1'],r['y2'],r['y3'])/10),r['x0']))


class EnglishOCR:
    def __init__(self,read_asset,check=lambda:None):
        import torch
        import easyocr
        from easyocr.craft import CRAFT
        from easyocr.model.vgg_model import Model
        from easyocr.config import recognition_models
        from easyocr.utils import CTCLabelConverter
        from easyocr.detection import get_textbox
        import importlib.metadata
        require(importlib.metadata.version('easyocr')=='1.7.2','This OCR adapter requires reviewed EasyOCR1.7.2.')
        def state(name):
            check();raw=read_asset(name);size,pin=ASSETS[name]
            require(type(raw) is bytes and len(raw)==size and hashlib.sha256(raw).hexdigest()==pin,'OCR model asset is missing, changed or unverified; review the English pack in Models & assets.')
            values=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
            require(isinstance(values,dict) and 0<len(values)<=200 and all(type(k) is str and type(v) is torch.Tensor and v.dtype in (torch.float32,torch.int64) and torch.isfinite(v).all() for k,v in values.items()),'OCR state must contain only finite reviewed tensors.')
            return {k.removeprefix('module.'):v for k,v in values.items()}
        # Reader methods are reused, but no initializer, network directory or executable model config.
        self.reader=easyocr.Reader.__new__(easyocr.Reader)
        r=self.reader;r.device='cpu';r.model_lang='english'
        r.character=recognition_models['gen2']['english_g2']['characters'];r.lang_char=r.character
        r.converter=CTCLabelConverter(r.character,{}, {})
        r.detector=CRAFT(pretrained=False);r.detector.load_state_dict(state('craft_mlt_25k.pth'),strict=True);r.detector.eval()
        r.recognizer=Model(input_channel=1,output_channel=256,hidden_size=256,num_class=len(r.converter.character))
        r.recognizer.load_state_dict(state('english_g2.pth'),strict=True);r.recognizer.eval();r.get_textbox=get_textbox
        self.check=check

    def recognize(self,pixels,config):
        import torch
        import cv2
        c=validate(config);self.check()
        require(isinstance(pixels,np.ndarray) and pixels.dtype==np.uint8 and pixels.ndim==3 and pixels.shape[2] in (1,3) and pixels.shape[0]*pixels.shape[1]<=16_000_000,'OCR needs explicit uint8 RGB/grayscale image preparation.')
        gray=pixels[:,:,0] if pixels.shape[2]==1 else cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
        rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB) if pixels.shape[2]==1 else pixels
        with torch.inference_mode():
            horizontal,free=self.reader.detect(rgb,min_size=c['minimumSize'],text_threshold=c['textThreshold'],low_text=c['lowText'],link_threshold=c['linkThreshold'],canvas_size=c['canvasSize'],mag_ratio=c['magnification'],width_ths=c['widthThreshold'],slope_ths=c['slopeThreshold'],ycenter_ths=c['ycenterThreshold'],height_ths=c['heightThreshold'],add_margin=c['margin'],reformat=False)
            self.check();require(len(horizontal[0])+len(free[0])<=c['maximumRegions'],'OCR exceeds its region allowance before recognition.')
            results=[]
            for h,f in [(x,None) for x in horizontal[0]]+[(None,x) for x in free[0]]:
                self.check()
                results+=self.reader.recognize(gray,[] if h is None else [h],[] if f is None else [f],decoder=c['decoder'],beamWidth=c['beamWidth'],batch_size=1,workers=0,allowlist=c['allowlist'] or None,blocklist=c['blocklist'] or None,rotation_info=c['rotationCandidates'] or None,paragraph=False,reformat=False)
                require(sum(len(x[1]) for x in results)<=c['maximumCharacters'],'OCR exceeds its character allowance.')
        self.check();return normalize_regions(results,config=c,width=pixels.shape[1],height=pixels.shape[0])


def render_pdf(raw,*,pages,dpi,maximum_pages,check=lambda:None):
    """Explicit bounded rasterization from already-captured bytes; close all native buffers."""
    import pypdfium2 as pdfium
    require(type(raw) is bytes and len(raw)<=64*1024*1024,'PDF source exceeds its encoded byte allowance.')
    require(type(dpi) is int and 36<=dpi<=300 and type(maximum_pages) is int and 1<=maximum_pages<=100,'PDF rasterization limits are invalid.')
    require(type(pages) is list and len(pages)<=maximum_pages and all(type(x) is int and x>=1 for x in pages) and len(set(pages))==len(pages),'Choose bounded distinct positive PDF pages.')
    total=0
    try:
        with pdfium.PdfDocument(raw) as document:
            from pypdfium2 import raw as pdfraw
            require(pdfraw.FPDF_GetSecurityHandlerRevision(document.raw)==-1,'Encrypted PDFs are unsupported; use an unencrypted source copy.')
            selected=pages or list(range(1,len(document)+1))
            require(len(selected)<=maximum_pages and all(x<=len(document) for x in selected),'PDF page selection exceeds its allowance or page count.')
            for number in selected:
                check()
                page=document[number-1]
                try:
                    w,h=page.get_size();scale=dpi/72
                    width,height=math.ceil(w*scale),math.ceil(h*scale)
                    require(0<width<=16384 and 0<height<=16384 and width*height<=16_000_000 and width*height*4<50_000_000 and total+width*height*3<=200_000_000,'PDF rasterization exceeds its decoded pixel allowance.')
                    bitmap=page.render(scale=scale,rev_byteorder=True)
                    try:rgb=np.asarray(bitmap.to_pil().convert('RGB')).copy()
                    finally:bitmap.close()
                    total+=rgb.nbytes;require(total<=200_000_000,'PDF rendered pages exceed the aggregate allowance.')
                    # PDFium supplies the exact rotation/crop-aware device→page mapping.
                    from ctypes import c_double,byref
                    from pypdfium2 import raw as pdfraw
                    corners=[]
                    for dx,dy in [(0,0),(width,0),(0,height)]:
                        px,py=c_double(),c_double()
                        require(pdfraw.FPDF_DeviceToPage(page.raw,0,0,width,height,0,dx,dy,byref(px),byref(py)),'PDF page coordinate mapping failed.')
                        corners.append([px.value,py.value])
                    a,b,c=np.asarray(corners);mapping=[(b[0]-a[0])/width,(c[0]-a[0])/height,a[0],(b[1]-a[1])/width,(c[1]-a[1])/height,a[1],0.,0.,1.]
                    check();yield dict(page=number,pixels=rgb,pagePoints=[w,h],pixelToPointScale=72/dpi,pixelToPage=mapping)
                finally:page.close()
    except pdfium.PdfiumError as error:raise ValueError('PDF cannot be safely rendered; encrypted or malformed documents are unsupported.') from error
