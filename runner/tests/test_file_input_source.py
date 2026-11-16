import hashlib
import io
import json
from pathlib import Path
import pytest
from runner import file_input_source as source
from runner.file_input_options import defaults


def rows(table):
    try:return [dict(zip([c['name'] for c in table.columns],row)) for row in table.rows],list(table.row_ids)
    finally:table.rows.store.close()


def config(**values):return {**defaults(),**values}


def test_text_source_keeps_metadata_typed_content_and_stable_unit_ids(tmp_path):
    path=tmp_path/'sample.txt';path.write_bytes(b'one\r\ntwo\n')
    options=config();options['readers']['text']['units']='line';options['readers']['text']['keep_empty_lines']=False
    result,identities=rows(source.source_files(path,options))
    assert [r['text_content'] for r in result]==['one','two']
    assert [r['unit_index'] for r in result]==[0,1]
    assert {r['relative_path'] for r in result}=={'sample.txt'}
    assert len({r['file_id'] for r in result})==1 and len(set(identities))==2
    assert all(r['byte_size']==9 and r['reader_status']=='content' for r in result)

@pytest.mark.parametrize("engine",["disk","polars"])
def test_large_line_source_is_disk_backed_and_resource_budgeted(tmp_path,engine):
    from runner.dataflow_disk import disk_scope, ScratchBudget
    from runner.polars_preparation import preparation_engine
    path=tmp_path/'large.txt';path.write_bytes(b'x\n'*100001)
    options=config(metadataFields=['file_id']);options['readers']['text']['units']='line'
    with preparation_engine(engine),disk_scope(directory=tmp_path):
        table=source.source_files(path,options)
        try:
            assert len(table.rows)==100001
            assert table.rows[0][1]=='x' and table.rows[-1][2]==100000
        finally:table.rows.store.close()
    budget=ScratchBudget(1_048_576)
    with preparation_engine(engine),disk_scope(directory=tmp_path,budget=budget),pytest.raises(ValueError,match='scratch storage budget'):
        source.source_files(path,options)
    assert budget.used==0
    assert not list(tmp_path.glob('.table-rows-*')) and not list(tmp_path.glob('.table-columns-*'))


def test_directory_filters_hard_extensions_hidden_recursion_and_sampling_before_decode(tmp_path,monkeypatch):
    (tmp_path/'a.txt').write_text('alpha');(tmp_path/'b.txt').write_bytes(b'\xff')
    (tmp_path/'c.bin').write_bytes(b'unknown');(tmp_path/'.hidden.txt').write_text('hidden')
    (tmp_path/'nested').mkdir();(tmp_path/'nested'/'d.txt').write_text('nested')
    selected,_=rows(source.source_files(tmp_path,config(sourceMode='directory'),sampling={'mode':'rowCount','rowCount':1}))
    assert [r['name'] for r in selected]==['a.txt']
    options=config(sourceMode='directory',readMode='metadata',includeSubfolders=True,includeHidden=True,extensions=['txt'],excludePatterns=['b*'])
    selected,_=rows(source.source_files(tmp_path,options))
    assert [r['relative_path'] for r in selected]==['.hidden.txt','a.txt','nested/d.txt']
    all_files,_=rows(source.source_files(tmp_path,config(sourceMode='directory',readMode='metadata',extensions=[])))
    assert [r['name'] for r in all_files]==['a.txt','b.txt','c.bin']


def test_unknown_single_file_bypasses_directory_filter_and_obeys_error_policies(tmp_path):
    path=tmp_path/'sample.bin';path.write_bytes(b'opaque')
    result,_=rows(source.source_files(path,config(unknownFiles='metadata')))
    assert result[0]['detected_type']=='unknown' and result[0]['reader_status']=='metadata'
    result,_=rows(source.source_files(path,config(unknownFiles='skip')));assert result==[]
    with pytest.raises(ValueError):source.source_files(path,config(unknownFiles='error'))
    result,_=rows(source.source_files(path,config(unknownFiles='error',errorPolicy='record')))
    assert result[0]['reader_status']=='error' and result[0]['text_content'] is None


def test_read_error_record_and_skip_never_keep_partial_document_units(tmp_path,monkeypatch):
    from runner import file_input_documents
    path=tmp_path/'sample.txt';path.write_text('valid bytes')
    def broken(*args,**kwargs):
        yield {'text_content':'partial must disappear','unit_index':0}
        raise ValueError('intentional bounded reader failure')
    monkeypatch.setattr(file_input_documents,'extract_document',broken)
    result,_=rows(source.source_files(path,config(errorPolicy='record')))
    assert len(result)==1 and result[0]['reader_status']=='error' and result[0]['text_content'] is None
    result,_=rows(source.source_files(path,config(errorPolicy='skip')));assert result==[]


def test_metadata_does_not_read_file_body_and_symlinks_never_follow(tmp_path,monkeypatch):
    path=tmp_path/'a.txt';path.write_text('body');(tmp_path/'link.txt').symlink_to(path)
    def forbidden(*args):raise AssertionError('metadata mode must not read bodies')
    monkeypatch.setattr(source,'_read_member',forbidden)
    result,_=rows(source.source_files(tmp_path,config(sourceMode='directory',readMode='metadata')))
    assert [r['name'] for r in result]==['a.txt']
    with pytest.raises(ValueError,match='symbolic'):source.source_files(tmp_path,config(sourceMode='directory',readMode='metadata',symlinks='error'))


def test_image_frames_publish_real_pixels_mixed_nullable_table_and_no_partial_leak(tmp_path,monkeypatch):
    from PIL import Image
    from runner.image_artifacts import artifact_scope,decode_pixels
    path=tmp_path/'a.png';Image.new('RGB',(2,1),(255,0,0)).save(path)
    (tmp_path/'b.txt').write_text('text')
    published={}
    with artifact_scope(lambda content,ref:published.setdefault(ref['sha256'],content)):
        result,_=rows(source.source_files(tmp_path,config(sourceMode='directory')))
    assert result[0]['text_content'] is None and result[1]['image_content'] is None
    reference=result[0]['image_content'];header,pixels=decode_pixels(published[reference['sha256']],reference['sha256'])
    assert (header['width'],header['height'],header['channels'])==(2,1,3) and pixels==b'\xff\x00\x00'*2
    from runner import file_input_images
    original=file_input_images.decode_images
    def broken(*args,**kwargs):
        yield next(original(*args,**kwargs))
        raise ValueError('later frame broken')
    monkeypatch.setattr(file_input_images,'decode_images',broken);published.clear()
    with artifact_scope(lambda content,ref:published.setdefault(ref['sha256'],content)):
        result,_=rows(source.source_files(path,config(errorPolicy='record')))
    assert published=={} and result[0]['reader_status']=='error' and result[0]['image_content'] is None


def test_inspection_is_eight_scalar_rows_and_images_metadata_only(tmp_path):
    from PIL import Image
    for index in range(12):(tmp_path/(str(index).zfill(2)+'.txt')).write_text('text')
    Image.new('RGB',(1,1)).save(tmp_path/'00.png')
    preview=source.preview_files(tmp_path,config(sourceMode='directory'))
    assert preview['rowCount']==8 and preview['collection']['selectedFiles']==8
    assert preview['collection']['imagePixelsIncluded'] is False and not preview['collection']['complete']
    assert 'image_content' not in preview['columns'] and all(type(value) is str for row in preview['rows'] for value in row)


def test_managed_manifest_member_hash_tamper_and_incomplete_full_refusal(tmp_path):
    parent=tmp_path/'capture';(parent/'members').mkdir(parents=True)
    raw=b'hello';(parent/'members/000000.txt').write_bytes(raw)
    document={'schemaVersion':1,'kind':'kwipFileCollection','sourceMode':'file','readMode':'content','complete':True,'selection':source.selection_configuration(defaults()),'members':[{'relativePath':'a.txt','capturePath':'members/000000.txt','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'extension':'txt','status':'content'}]}
    manifest=parent/'manifest.json';manifest.write_text(json.dumps(document))
    assert rows(source.source_files({'manifest':manifest}))[0][0]['text_content']=='hello'
    (parent/'members/000000.txt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='SHA-256'):source.source_files({'manifest':manifest})
    document['complete']=False;manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='complete captured'):source.source_files({'manifest':manifest})


def test_manifest_selection_must_match_while_decoder_options_can_change(tmp_path):
    from runner.dataflow_disk import disk_scope
    path=tmp_path/'a.txt';path.write_text('first\nsecond')
    options=config()
    with disk_scope(),source.collection(path,options,lambda:None) as (manifest,raw,document):
        changed=config();changed['readers']['text']['units']='line'
        assert [r['text_content'] for r in rows(source.source_files({'manifest':manifest},changed))[0]]==['first','second']
        for values in ({'includeHidden':True},{'extensions':[]},{'maxFiles':999},{'readMode':'metadata'},{'errorPolicy':'skip'}):
            with pytest.raises(ValueError,match='selection differs'):
                source.source_files({'manifest':manifest},config(**values))


def test_source_graph_validates_strict_configuration_before_binding():
    from runner import dataflow
    graph={'schemaVersion':1,'nodes':[{'id':'files','operator':'kwip.data.sourceFiles','config':defaults()}],'edges':[]}
    dataflow.validate(graph)
    graph['nodes'][0]['config']['readers']['text']['unrecognized']=True
    with pytest.raises(ValueError):dataflow.validate(graph)


def test_file_preparation_derived_code_replays_text_and_image_closure(tmp_path):
    import os,subprocess,sys
    from PIL import Image
    from runner.preparation_plan import build_plan,derive_preparation_code
    sources=tmp_path/'sources';sources.mkdir();(sources/'a.txt').write_text('replay text')
    Image.new('RGB',(2,1),(3,4,5)).save(sources/'b.png')
    graph={'schemaVersion':1,'nodes':[{'id':'files','operator':'kwip.data.sourceFiles','config':config(sourceMode='directory')}],'edges':[]}
    plan=build_plan(graph,engine='polars',engine_version='2.0.0')
    code=tmp_path/'preparation.py';code.write_text(derive_preparation_code(plan))
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'files':str(sources)}))
    output=tmp_path/'run'
    result=subprocess.run([sys.executable,str(code),'--source-bindings',str(bindings),'--output',str(output)],cwd=Path(__file__).resolve().parents[2],env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2])},text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    manifest=json.loads((output/'manifest.json').read_text())
    assert any(item['kind']=='imagePixels' for item in manifest['artifacts'])
    assert (output/'previews.json').exists()


def test_file_inspection_cli_is_data_only_and_exact_argument_shape(tmp_path):
    import os,subprocess,sys
    root=Path(__file__).resolve().parents[2]
    path=tmp_path/'notes.txt';path.write_text('bounded preview')
    invocation=[sys.executable,str(root/'runner/main.py'),'--inspect-data',str(path),'--file-options',json.dumps(defaults())]
    result=subprocess.run(invocation,cwd=root,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(events)==1 and events[0]['event']=='dataPreview'
    assert events[0]['collection']['imagePixelsIncluded'] is False
    bad=subprocess.run(invocation+['--graph','untrusted.json'],cwd=root,text=True,capture_output=True)
    assert bad.returncode!=0 and json.loads(bad.stdout)['event']=='error'


@pytest.mark.parametrize('policy',['record','skip'])
def test_capture_integrity_failure_never_becomes_reader_diagnostic(tmp_path,policy):
    from runner.dataflow_disk import disk_scope
    path=tmp_path/'original.txt';path.write_text('trusted')
    options=config(errorPolicy=policy)
    with disk_scope(),source.collection(path,options,lambda:None) as (manifest,raw,document):
        body=manifest.parent/document['members'][0]['capturePath'];body.write_bytes(b'tampered')
        with pytest.raises(ValueError,match='byte authority|SHA-256'):
            source.source_files({'manifest':manifest},options)


def test_cancel_and_pixel_publication_failure_are_fatal_even_when_recording(tmp_path,monkeypatch):
    from PIL import Image
    from runner.image_artifacts import artifact_scope
    from runner import file_input_documents
    path=tmp_path/'source.txt';path.write_text('text')
    def cancelled(*args,**kwargs):raise InterruptedError('cancelled')
    monkeypatch.setattr(file_input_documents,'extract_document',cancelled)
    with pytest.raises(InterruptedError):source.source_files(path,config(errorPolicy='record'))
    image=tmp_path/'source.png';Image.new('RGB',(1,1)).save(image)
    def failed_publisher(*args):raise OSError('synthetic artifact storage failure')
    with artifact_scope(failed_publisher),pytest.raises(OSError,match='artifact storage'):
        source.source_files(image,config(errorPolicy='record'))


@pytest.mark.parametrize('policy',['record','skip'])
def test_portable_main_signal_cancellation_is_not_a_reader_error(tmp_path,monkeypatch,policy):
    from runner.main import Cancelled
    from runner import file_input_documents
    path=tmp_path/'notes.txt';path.write_text('content')
    def cancelled(*args,**kwargs):raise Cancelled('synthetic signal cancellation')
    monkeypatch.setattr(file_input_documents,'extract_document',cancelled)
    with pytest.raises(Cancelled):source.source_files(path,config(errorPolicy=policy))
