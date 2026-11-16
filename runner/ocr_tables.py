"""OCR table operator: complete documents and source-linked regions, no hidden row samples."""
import json
from contextvars import ContextVar
from contextlib import contextmanager
from io import BytesIO
import hashlib
import zipfile
import numpy as np
if __package__:
    from . import ocr,dataflow as df
    from .image_artifacts import decode_pixels
    from . import media_artifacts as media
else:
    import ocr,dataflow as df,media_artifacts as media
    from image_artifacts import decode_pixels

ARCHIVES={
 'craft_mlt_25k.zip':('craft_mlt_25k.pth',77251756,'8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65'),
 'english_g2.zip':('english_g2.pth',14040947,'1b5eaebf1c062de6205560c97ffcfa8dc0e6f413c340e8adc5cfc57e159f61ff')}
_context=ContextVar('verified_ocr_job',default=None)


def unpack_asset(name,read_archive):
    archive,(_,size,pin)=next((k,v) for k,v in ARCHIVES.items() if v[0]==name)
    raw=read_archive(archive)
    ocr.require(type(raw) is bytes and len(raw)==size and hashlib.sha256(raw).hexdigest()==pin,'OCR pack archive differs from its reviewed integrity pin.')
    with zipfile.ZipFile(BytesIO(raw)) as z:
        ocr.require(z.namelist()==[name] and z.getinfo(name).file_size==ocr.ASSETS[name][0] and not z.getinfo(name).flag_bits&1,'OCR archive has unreviewed members or size.')
        return z.read(name)


@contextmanager
def job_scope(read_archive,check):
    # Lazy per-job model ownership; all graph OCR nodes share one verified CPU adapter.
    token=_context.set(dict(read=read_archive,check=check,adapter=None))
    try:yield
    finally:_context.reset(token)


def adapter():
    context=_context.get();ocr.require(context is not None,'OCR requires the owned model-pack capture; install English OCR in Models & assets.')
    if context['adapter'] is None:context['adapter']=ocr.EnglishOCR(lambda n:unpack_asset(n,context['read']),context['check'])
    return context['adapter']


def apply(table,config,check,image_reader,recognizer=None):
    c=ocr.validate(config);column=table.index(c['inputColumn']);kind=table.columns[column]['dtype']
    ocr.require(kind in ('image','tensor'),'OCR requires a captured image or image-shaped prepared tensor column.')
    ocr.require(callable(image_reader),'OCR requires the owned source image inventory.')
    names={x['name'] for x in table.columns}
    geometry_fields={x['name']+'_to_source' for x in table.columns if x['dtype']=='image' and x['name']+'_source' in names}
    ocr.require(c['outputColumn'] not in names,'OCR text output column already exists; choose an unused name.')
    group=table.index(c['groupBy']) if c['groupBy'] else None
    page=table.index(c['pageColumn']) if c['pageColumn'] else None
    ocr.require(group is None or table.columns[group]['dtype']=='string','OCR document grouping requires a string column.')
    ocr.require(page is None or table.columns[page]['dtype']=='int64','OCR page ordering requires an int64 column.')
    engine=recognizer or adapter()
    documents={};regions=[];character_total=0
    coordinate_names=[axis+str(i) for i in range(4) for axis in ('x','y')]
    region_schema=[dict(name=n,dtype=t,nullable=False) for n,t in [('document_id','string'),('page','int64'),('region','int64'),('text','string'),('confidence','float64'),('review','string'),('source_image','image'),*[(n,'float64') for n in coordinate_names],('coordinate_space','string'),('ocr_pack','string')]]
    region_schema += [dict(name='pdf_'+n,dtype='float64',nullable=True) for n in coordinate_names]
    for row,identity in df.row_pairs(table):
        check();source=row[column];ocr.require(source is not None,'OCR source is missing; filter failed/missing source rows explicitly.')
        original=source;matrix=np.eye(3)
        if kind=='image':
            header,samples=decode_pixels(image_reader(source),expected_sha256=source['sha256'])
            ocr.require(header['policy']['actualMode'] in ('RGB','L'),'OCR requires explicit RGB/grayscale conversion and alpha handling.')
            shape=(source['height'],source['width'],source['channels'])
            pixels=np.frombuffer(samples,dtype=np.dtype(source['dtype']).newbyteorder('<')).reshape(shape if source['layout']=='HWC' else (shape[2],shape[0],shape[1]))
            if source['layout']=='CHW':pixels=pixels.transpose(1,2,0)
            # Prepared-image companion columns bind the retained original and mapping.
            parent=c['inputColumn']+'_source';transform=c['inputColumn']+'_to_source'
            if parent in names and transform in names:
                original=row[table.index(parent)];matrix=np.asarray(json.loads(row[table.index(transform)]),dtype=np.float64).reshape(3,3)
        else:
            _,samples=media.read_asset(source,check)
            ocr.require(source['axes'] in (['height','width','channels'],['channels','height','width']),'OCR requires image axes; audio/general tensors cannot be decoded as images.')
            pixels=np.frombuffer(samples,dtype=np.dtype(source['dtype']).newbyteorder('<')).reshape(source['shape'])
            if source['axes'][0]=='channels':pixels=pixels.transpose(1,2,0)
            # Original image remains an explicit input column; do not invent missing source lineage.
            parent=source['metadata'].get('parentSHA256')
            candidates=[v for v in row if isinstance(v,dict) and v.get('kind')=='imagePixelsRef' and v.get('sha256')==parent]
            ocr.require(len(candidates)==1,'OCR prepared tensor requires its retained original image column.')
            original=candidates[0];matrix=np.asarray(source['metadata']['geometry'],dtype=np.float64).reshape(3,3)
        raw=engine.recognize(pixels,c)
        mapped=[]
        for item in raw:
            quad=[[item['x'+str(i)],item['y'+str(i)]] for i in range(4)]
            mapped.append((quad,item['text'],item['confidence']))
        normalized=ocr.normalize_regions(mapped,config=c,width=original['width'],height=original['height'],matrix=matrix)
        key=row[group] if group is not None else identity
        ocr.require(type(key) is str and 0<len(key)<=4096,'OCR document grouping requires nonblank string identities.')
        number=row[page] if page is not None else len(documents.get(key,{}).get('pages',[]))
        ocr.require(type(number) is int and number>=0,'OCR page/frame ordering requires nonnegative integer values.')
        page_fields={'unit_index','page_number','frame_index'}
        scalar_indices=[i for i,x in enumerate(table.columns) if x['dtype'] not in ('image','audio','tensor') and x['name'] not in page_fields and not x['name'].startswith('pdf_map_') and x['name'] not in geometry_fields and i!=page]
        scalar=[row[i] for i in scalar_indices]
        # Metadata such as unit_index may differ by page. Preserve first row; source identity is explicit.
        if key not in documents:documents[key]=dict(first=list(row),scalar=scalar,pages=[],reviews=0,regions=0,identity=identity)
        ocr.require(documents[key]['scalar']==scalar,'Grouped OCR pages disagree on scalar source metadata/labels; separate incompatible document groups.')
        doc=documents[key];ocr.require(not any(p[0]==number for p in doc['pages']),'OCR document pages have duplicate ordering identities.')
        doc['pages'].append((number,c['separator'].join(x['text'] for x in normalized)))
        for index,item in enumerate(normalized):
            doc['reviews']+=item['review']!='none';doc['regions']+=1
            pdf_values=[row[table.index('pdf_map_'+str(i))] for i in range(9)] if all('pdf_map_'+str(i) in names for i in range(9)) else []
            pdf_points=ocr.map_quad([[item['x'+str(i)],item['y'+str(i)]] for i in range(4)],np.asarray(pdf_values).reshape(3,3)).reshape(-1).tolist() if pdf_values and all(v is not None for v in pdf_values) else [None]*8
            regions.append(([key,number,index,item['text'],item['confidence'],item['review'],original,*[item[n] for n in coordinate_names],'sourceImagePixelsTopLeft',ocr.PACK,*pdf_points],df.digest([identity,index,config])))
        character_total+=sum(len(x['text']) for x in normalized)
        ocr.require(character_total<=c['maximumCharacters'] and len(regions)<=c['maximumRegions'],'OCR complete output exceeds its total character/region allowance.')
        ocr.require(len(documents)<=10000,'OCR document count exceeds its allowance.')
    # Documents retain scalar metadata, not a misleading single image for a multi-page document.
    kept=[i for i,x in enumerate(table.columns) if x['dtype'] not in ('image','audio','tensor') and x['name'] not in ('unit_index','page_number','frame_index') and not x['name'].startswith('pdf_map_') and x['name'] not in geometry_fields and i!=page]
    added=[(c['outputColumn'],'string'),('ocr_status','string'),('ocr_page_count','int64'),('ocr_region_count','int64'),('ocr_review_count','int64'),('ocr_pack','string')]
    ocr.require(not ({n for n,_ in added}&names),'OCR status/output names collide with source columns.')
    doc_schema=[table.columns[i] for i in kept]+[dict(name=n,dtype=t,nullable=False) for n,t in added]
    def pairs():
        for key,value in documents.items():
            check();text=c['separator'].join(t for _,t in sorted(value['pages']))
            yield ([value['first'][i] for i in kept]+[text,'recognized' if text else 'noText',len(value['pages']),value['regions'],value['reviews'],ocr.PACK],'ocr-document:'+df.digest(key))
    return {'documents':df.streaming_rows(table,pairs(),doc_schema,check,identity_prefix=''),'regions':df.streaming_rows(table,iter(regions),region_schema,check,identity_prefix='')}
