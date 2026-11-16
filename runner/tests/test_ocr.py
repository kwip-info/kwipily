"""OCR contracts: no network/path authority, exact output and geometry evidence."""
from pathlib import Path
import os
import numpy as np
import pytest
from runner import ocr


def test_config_refuses_unknown_pack_device_or_conflicting_character_filters():
    assert ocr.validate(ocr.defaults())['pack']=='english-printed-v1'
    for changes in ({'pack':'arbitrary'},{'device':'mps'},{'allowlist':'A','blocklist':'B'},{'maximumRegions':0},{'rotationCandidates':[45]},{'canvasSize':100000}):
        with pytest.raises(ValueError):ocr.validate(dict(ocr.defaults(),**changes))


def test_independent_crop_rotate_mapping_keeps_original_pixel_centers():
    image=np.arange(3*5,dtype=np.uint8).reshape(3,5,1)
    out,matrix=ocr.prepare_pixels(image,crop=[1,0,5,3],rotation=90)
    np.testing.assert_array_equal(out[:,:,0],np.rot90(image[:,1:5,0],k=-1))
    # Clockwise rotated output(0,0) refers to source(1,2).
    np.testing.assert_array_equal(ocr.map_quad([[0,0],[1,0],[1,2],[0,2]],matrix)[0],[1,2])
    with pytest.raises(ValueError):ocr.prepare_pixels(image,crop=[-1,0,2,2])


def test_regions_keep_low_confidence_text_and_blank_pages():
    c=ocr.defaults();c['reviewThreshold']=.6
    page=ocr.normalize_regions([([[2,3],[12,3],[12,8],[2,8]],'Invoice 42',.54)],config=c,width=20,height=10)
    assert page[0]['text']=='Invoice 42' and page[0]['review']=='lowConfidence'
    assert page[0]['confidence']==.54 and page[0]['x0']==2 and page[0]['y3']==8
    assert ocr.normalize_regions([],config=c,width=20,height=10)==[]
    for item in [([[0,0]]*4,'bad',float('nan')),([[0,0]]*4,'bad',1.1)]:
        with pytest.raises(ValueError):ocr.normalize_regions([item],config=c,width=20,height=10)


def test_pdf_render_is_explicit_bounded_and_preserves_rotated_page_geometry():
    from PIL import Image
    from io import BytesIO
    image=Image.new('RGB',(120,80),'white');data=BytesIO();image.save(data,format='PDF',resolution=72)
    pages=list(ocr.render_pdf(data.getvalue(),pages=[1],dpi=72,maximum_pages=2))
    assert len(pages)==1 and pages[0]['page']==1 and pages[0]['pixels'].shape==(80,120,3)
    assert pages[0]['pagePoints']==[120.,80.]
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(data.getvalue()) as document:
        page=document[0];page.set_rotation(90);page.close()
        rotated=BytesIO();document.save(rotated)
    result=list(ocr.render_pdf(rotated.getvalue(),pages=[1],dpi=72,maximum_pages=2))[0]
    assert result['pixels'].shape==(120,80,3)
    points=ocr.map_quad([[0,0],[80,0],[80,120],[0,120]],np.asarray(result['pixelToPage']).reshape(3,3))
    np.testing.assert_allclose(points.min(axis=0),[0,0],atol=1e-8)
    np.testing.assert_allclose(points.max(axis=0),[120,80],atol=1e-8)
    with pytest.raises(ValueError):list(ocr.render_pdf(data.getvalue(),pages=[2],dpi=72,maximum_pages=2))
    with pytest.raises(ValueError):list(ocr.render_pdf(data.getvalue(),pages=[1],dpi=10000,maximum_pages=2))


@pytest.mark.skipif(not os.environ.get('KWIPML_TEST_OCR_PACK'),reason='Explicit reviewed OCR weights required')
def test_actual_safe_offline_english_inference_without_reader_download_or_pickle(monkeypatch):
    import easyocr,torch
    from PIL import Image,ImageDraw,ImageFont
    monkeypatch.setattr(easyocr.Reader,'__init__',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unsafe Reader initializer')))
    original=torch.load
    def safe(*a,**k):
        assert k.get('weights_only') is True
        return original(*a,**k)
    monkeypatch.setattr(torch,'load',safe)
    root=Path(os.environ['KWIPML_TEST_OCR_PACK'])
    adapter=ocr.EnglishOCR(lambda name:(root/name).read_bytes())
    image=Image.new('RGB',(600,180),'white')
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',40)
    draw.text((24,30),'INVOICE 42',font=font,fill='black')
    result=adapter.recognize(np.asarray(image),ocr.defaults())
    assert any('INVOICE 42' in r['text'].upper() for r in result)
    assert all(0<=r['confidence']<=1 for r in result)


def test_preparation_adds_owned_image_with_exact_parent_and_transform():
    from runner import dataflow as df,prepare_media as prep,image_artifacts as images
    assets={}
    def publish(raw,ref):assets[ref['sha256']]=raw
    with images.artifact_scope(publish):
        source=np.arange(15,dtype=np.uint8).reshape(3,5,1)
        ref=images.publish_pixels(source.tobytes(),width=5,height=3,channels=1,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='preserve',palette='expandRGBA',actualMode='L'))
        table=df.Table((dict(name='image_content',dtype='image',nullable=False),),((ref,),),('row1',),'f'*64)
        config=prep.defaults();config.update(outputMode='image',outputColumn='ocr_image',cropEnabled=True,cropX=1,cropY=0,cropWidth=4,cropHeight=3,rotation=90)
        out=prep.apply(table,config,lambda:None,image_reader=lambda r:assets[r['sha256']])
        assert out.columns[1]['dtype']=='image'
        first=out.rows[0];header,raw=images.decode_pixels(assets[first[1]['sha256']])
        np.testing.assert_array_equal(np.frombuffer(raw,dtype=np.uint8).reshape(4,3),np.rot90(source[:,1:5,0],-1))
        assert first[2]==ref


@pytest.mark.parametrize('engine',['disk','polars'])
def test_ocr_complete_tables_group_pages_and_preserve_regions_with_independent_fixture(tmp_path,engine):
    from runner import dataflow as df,image_artifacts as images,ocr_tables
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_disk import disk_scope
    saved={}
    with preparation_engine(engine),disk_scope(directory=tmp_path),images.artifact_scope(lambda raw,ref:saved.setdefault(ref['sha256'],raw)):
        ref=images.publish_pixels(bytes(60*40*3),width=60,height=40,channels=3,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='RGB',palette='expandRGBA',actualMode='RGB'))
        columns=tuple(dict(name=n,dtype=t,nullable=False) for n,t in [('file_id','string'),('frame_index','int64'),('category','string'),('image_content','image')])
        table=df.Table(columns,(('document-a',1,'Invoice',ref),('document-a',0,'Invoice',ref)),('page-1','page-0'),'a'*64)
        class Fixture:
            def recognize(self,pixels,c):return ocr.normalize_regions([([[1,2],[20,2],[20,10],[1,10]],'Invoice 42',.54)],config=c,width=60,height=40)
        out=ocr_tables.apply(table,ocr.defaults(),lambda:None,lambda r:saved[r['sha256']],recognizer=Fixture())
        docs=out['documents'];regions=out['regions']
        assert len(docs.rows)==1 and len(regions.rows)==2
        assert docs.rows[0][docs.index('document_text')]=='Invoice 42\n\nInvoice 42'
        assert docs.rows[0][docs.index('category')]=='Invoice'
        assert docs.rows[0][docs.index('ocr_review_count')]==2
        assert all(r[regions.index('text')]=='Invoice 42' and r[regions.index('review')]=='lowConfidence' for r in regions.rows)
        assert len(set(regions.row_ids))==2


def test_explicit_pdf_file_reader_publishes_page_images_and_exact_point_map(tmp_path):
    from runner import file_input_source as source,file_input_options as options,image_artifacts as images
    from PIL import Image
    path=tmp_path/'scan.pdf';Image.new('RGB',(120,80),'white').save(path,format='PDF',resolution=72)
    config=options.defaults();config['extensions']=['pdf'];config['readers']['pdf'].update(renderMode='pageImages',renderDPI=72)
    assets={}
    with images.artifact_scope(lambda raw,ref:assets.setdefault(ref['sha256'],raw)):
        table=source.source_files(path,config)
    assert len(table.rows)==1
    row=dict(zip([c['name'] for c in table.columns],table.rows[0]))
    assert row['text_content'] is None and row['image_content']['width']==120
    assert row['page_number']==1 and row['frame_index']==0
    np.testing.assert_allclose([row['pdf_map_'+str(i)] for i in range(9)],[1,0,0,0,-1,80,0,0,1],atol=1e-10)


def test_ocr_archive_tamper_refused_before_deserialization():
    from runner import ocr_tables
    with pytest.raises(ValueError,match='integrity'):
        ocr_tables.unpack_asset('english_g2.pth',lambda name:b'not an archive')


def test_grouped_document_identity_cannot_leak_pages_across_partitions(tmp_path):
    from runner import dataflow as df,image_artifacts as images,ocr_tables
    assets={}
    class Blank:
        def recognize(self,pixels,c):return []
    with images.artifact_scope(lambda raw,ref:assets.setdefault(ref['sha256'],raw)):
        ref=images.publish_pixels(bytes(30),width=10,height=1,channels=3,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='RGB',palette='expandRGBA',actualMode='RGB'))
        schema=tuple(dict(name=n,dtype=t,nullable=False) for n,t in [('file_id','string'),('frame_index','int64'),('image_content','image')])
        first=df.Table(schema,(('same-original',0,ref),),('source-page0',),'b'*64,role='training')
        second=df.Table(schema,(('same-original',1,ref),),('source-page1',),'b'*64,role='validation')
        outputs=[ocr_tables.apply(t,ocr.defaults(),lambda:None,lambda r:assets[r['sha256']],recognizer=Blank())['documents'] for t in (first,second)]
        assert set(outputs[0].row_ids)&set(outputs[1].row_ids), 'Original document identity must survive grouping so split leakage is detectable.'
        assert all(t.rows[0][t.index('ocr_status')]=='noText' for t in outputs)
