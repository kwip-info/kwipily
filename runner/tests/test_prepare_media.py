"""Independent numeric expectations for deterministic media preparation."""
import numpy as np
import pytest
from runner import prepare_media as prep


def test_audio_exact_pcm_normalization_window_masks_and_parent_offsets():
    config=prep.defaults('audio');config.update(windowSamples=4,hopSamples=3,tail='pad',sampleFormat='float32',channels='mono')
    source=np.array([[-32768,32767],[0,0],[16384,16384],[0,0],[32767,-32768]],dtype='<i2')
    windows=list(prep.audio_windows(source.tobytes(),shape=[5,2],dtype='int16',metadata={'sampleRate':48000,'sourceBits':16},config=config))
    assert len(windows)==2
    first=windows[0];second=windows[1]
    np.testing.assert_allclose(first['array'],[[-1/65536,0,.5,0]],rtol=0,atol=1e-8)
    assert first['metadata']=={'sampleRate':48000,'sourceRate':48000,'sourceStart':0,'sourceEnd':4,'validSamples':4}
    np.testing.assert_array_equal(second['mask'],[1,1,0,0])
    assert second['metadata']['sourceStart']==3 and second['metadata']['sourceEnd']==5
    assert second['array'].shape==(1,4) and second['array'][0,2:].tolist()==[0,0]


def test_preserved_audio_does_not_implicitly_resample_or_mix_channels():
    config=prep.defaults('audio');source=np.array([[1,2],[3,4]],dtype='<i4')
    result=list(prep.audio_windows(source.tobytes(),shape=[2,2],dtype='int32',metadata={'sampleRate':44100,'sourceBits':24},config=config))
    assert len(result)==1
    np.testing.assert_array_equal(result[0]['array'],source.T)
    assert result[0]['array'].dtype==np.dtype('<i4')
    assert result[0]['metadata']['sampleRate']==44100


def test_resampling_constant_signal_length_filter_and_original_sample_intervals():
    config=prep.defaults('audio');config.update(sampleFormat='float32',targetRate=16000,windowSamples=16,hopSamples=16,tail='pad')
    source=np.full((120,1),16384,dtype='<i2')
    windows=list(prep.audio_windows(source.tobytes(),shape=[120,1],dtype='int16',metadata={'sampleRate':48000,'sourceBits':16},config=config))
    assert len(windows)==3
    center=np.concatenate([x['array'][0,:x['metadata']['validSamples']] for x in windows])
    assert len(center)==40
    np.testing.assert_allclose(center[10:30],.5,atol=.002)
    assert windows[-1]['metadata']['sourceStart']==96 and windows[-1]['metadata']['sourceEnd']==120


def test_image_preserves_or_explicitly_normalizes_layout_and_fit_geometry():
    config=prep.defaults('image');source=np.array([[[255,0,0],[0,255,0]]],dtype=np.uint8)
    result=prep.image_tensor(source,config)
    np.testing.assert_array_equal(result['array'],source)
    assert result['axes']==['height','width','channels']
    config.update(sizeMode='fitPad',width=4,height=4,interpolation='nearest',sampleFormat='float32',scaling='unit',layout='CHW')
    result=prep.image_tensor(source,config)
    assert result['array'].shape==(3,4,4)
    np.testing.assert_array_equal(result['array'][:,0,:],0)
    np.testing.assert_array_equal(result['array'][:,1,0],[1,0,0])
    assert result['geometry']==[.5,0,-.25,0,.5,-.75,0,0,1]


def test_rejects_expansion_implicit_integer_resampling_alpha_loss_and_cancel():
    config=prep.defaults('audio');config.update(windowSamples=100000000)
    with pytest.raises(ValueError): prep.validate_config(config)
    config=prep.defaults('audio');config.update(targetRate=16000)
    with pytest.raises(ValueError): list(prep.audio_windows(bytes(100),shape=[50,1],dtype='int16',metadata={'sampleRate':48000,'sourceBits':16},config=config))
    config=prep.defaults('image');config['color']='RGB'
    with pytest.raises(ValueError,match='alpha'):prep.image_tensor(np.zeros((2,2,4),dtype=np.uint8),config)
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):prep.image_tensor(np.zeros((2,2,3),dtype=np.uint8),prep.defaults('image'),cancel)


def test_uint16_image_precision_and_resize_pixel_center_transform():
    c=prep.defaults('image')
    pixels=np.array([[[0],[65535]]],dtype='<u2')
    np.testing.assert_array_equal(prep.image_tensor(pixels,c)['array'],pixels)
    c.update(sizeMode='resize',width=4,height=2,interpolation='nearest')
    out=prep.image_tensor(pixels,c)
    assert out['geometry']==[.5,0,-.25,0,.5,-.25,0,0,1]


@pytest.mark.parametrize('engine',['disk','polars'])
def test_file_audio_prepare_graph_captures_exact_windows_masks_and_replay(tmp_path,engine):
    import json,struct
    from runner.tests.test_file_input_audio import wav
    from runner.file_input_options import defaults
    from runner.dataflow_cli import run_dataflow
    from runner.dataflow_storage import completed_inventory
    from runner.dataflow_table_reader import read_table_artifact
    source=tmp_path/'sample.wav';source.write_bytes(wav(struct.pack('<6h',-32768,0,16384,32767,0,-1),channels=1))
    read=defaults();read['readers']['audio']={'format':'pcmWav'}
    config=prep.defaults('audio');config.update(sampleFormat='float32',windowSamples=4,hopSamples=4,tail='pad')
    graph={'schemaVersion':1,'nodes':[dict(id='source',operator='kwip.data.sourceFiles',config=read),dict(id='prepare',operator='kwip.data.prepareMedia',config=config)],'edges':[dict(**{'from':dict(node='source',port='table'),'to':dict(node='prepare',port='table')})]}
    graph['nodes'].append(dict(id='view',operator='kwip.data.view',config={}))
    graph['edges'].append({'from':dict(node='prepare',port='table'),'to':dict(node='view',port='table')})
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));out=tmp_path/'out'
    run_dataflow(path,{'source':str(source)},out,lambda _:None,engine=engine)
    source.unlink();manifest,files,_=completed_inventory(out/'manifest.json')
    from runner.table_view import query_index
    view=next(e for e in manifest['artifacts'] if e['kind']=='tableViewIndex')
    page=query_index(out/view['path'],view['sha256'],offset=0,limit=8,search='',source_sha256=view['sourceSHA256'])
    assert len(page['rows'])>=1 and any(c['dtype']=='tensor' for c in page['columns'])
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='prepare' and e['kind']=='table')
    table=read_table_artifact(out/entry['path'],entry['sha256'])
    try:
        assert len(table.rows)==2
        first=table.rows[0][table.index('prepared')];second=table.rows[1][table.index('prepared')]
        assert first['shape']==[1,4] and first['axes']==['channels','samples']
        assert second['metadata']['validSamples']==2 and second['metadata']['sourceStart']==4
        assert table.rows[0][table.index('prepared_parent')]==table.rows[1][table.index('prepared_parent')]
    finally:table.rows.store.close()


def test_cached_prepared_media_republishes_owned_closure(tmp_path):
    import hashlib,json,struct
    from runner.tests.test_file_input_audio import wav
    from runner.file_input_options import defaults
    from runner.dataflow_cli import run_dataflow
    from runner.dataflow_storage import completed_inventory
    source=tmp_path/'sample.wav';source.write_bytes(wav(struct.pack('<4h',0,100,200,300),channels=1))
    config=defaults();config['readers']['audio']={'format':'pcmWav'}
    prepare=prep.defaults('audio');prepare['sampleFormat']='float32'
    graph={'schemaVersion':1,'nodes':[dict(id='source',operator='kwip.data.sourceFiles',config=config),dict(id='prepare',operator='kwip.data.prepareMedia',config=prepare)],'edges':[{'from':dict(node='source',port='table'),'to':dict(node='prepare',port='table')}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));first=tmp_path/'first';second=tmp_path/'second'
    run_dataflow(path,{'source':str(source)},first,lambda _:None,cache_policy='materialize')
    prior=first/'manifest.json';pin=hashlib.sha256(prior.read_bytes()).hexdigest()
    run_dataflow(path,{'source':str(source)},second,lambda _:None,cache_policy='reuse',reuse_manifest=prior,reuse_manifest_sha256=pin)
    manifest,_,_=completed_inventory(second/'manifest.json')
    assert 'prepare' in manifest['cache']['reusedNodeIDs']
    assert len([e for e in manifest['artifacts'] if e['kind']=='preparedTensor'])==2


def test_resampling_low_pass_rejects_alias_energy_and_honors_cancellation():
    n=np.arange(2400);signal=(np.sin(2*np.pi*18000*n/48000)*30000).astype('<i2')[:,None]
    c=prep.defaults('audio');c.update(sampleFormat='float32',targetRate=16000)
    result=next(prep.audio_windows(signal.tobytes(),shape=list(signal.shape),dtype='int16',metadata={'sampleRate':48000,'sourceBits':16},config=c))
    assert np.max(np.abs(result['array'][0,40:-40]))<.005
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):next(prep.audio_windows(signal.tobytes(),shape=list(signal.shape),dtype='int16',metadata={'sampleRate':48000,'sourceBits':16},config=c,check_cancelled=cancel))


@pytest.mark.parametrize('engine',['disk','polars'])
def test_captured_image_prepare_preserves_parent_pixels_and_float_tensor(tmp_path,engine):
    import json
    from PIL import Image
    from runner.file_input_options import defaults
    from runner.dataflow_cli import run_dataflow
    from runner.dataflow_storage import completed_inventory
    from runner.media_artifacts import decode_asset
    source=tmp_path/'sample.png';pixels=np.array([[[255,0,0],[0,255,0]]],dtype=np.uint8);Image.fromarray(pixels).save(source)
    c=prep.defaults('image');c.update(sampleFormat='float32',scaling='unit',layout='CHW')
    graph={'schemaVersion':1,'nodes':[dict(id='source',operator='kwip.data.sourceFiles',config=defaults()),dict(id='prepare',operator='kwip.data.prepareMedia',config=c)],'edges':[{'from':dict(node='source',port='table'),'to':dict(node='prepare',port='table')}]}
    graph['nodes'].append(dict(id='view',operator='kwip.data.view',config={}))
    graph['edges'].append({'from':dict(node='prepare',port='table'),'to':dict(node='view',port='table')})
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));out=tmp_path/'out'
    run_dataflow(path,{'source':str(source)},out,lambda _:None,engine=engine)
    source.unlink();manifest,files,_=completed_inventory(out/'manifest.json')
    from runner.table_view import query_index
    view=next(e for e in manifest['artifacts'] if e['kind']=='tableViewIndex')
    page=query_index(out/view['path'],view['sha256'],offset=0,limit=8,search='',source_sha256=view['sourceSHA256'])
    assert len(page['rows'])>=1 and any(c['dtype']=='tensor' for c in page['columns'])
    media=next(e for e in manifest['artifacts'] if e['kind']=='preparedTensor');_,raw=decode_asset(files[media['path']],expected=media['media'])
    np.testing.assert_array_equal(np.frombuffer(raw,dtype='<f4').reshape(3,1,2),pixels.transpose(2,0,1)/255)
    assert next(e for e in manifest['artifacts'] if e['kind']=='imagePixels')['sha256']==media['media']['metadata']['parentSHA256']


def test_blocked_mask_polarity_is_explicit_and_does_not_change_samples():
    c=prep.defaults('audio');c.update(windowSamples=4,tail='pad',maskConvention='blocked')
    window=next(prep.audio_windows(bytes(4),shape=[2,1],dtype='int16',metadata={'sampleRate':8000,'sourceBits':16},config=c))
    assert window['mask'].tolist()==[0,0,1,1]
    assert window['metadata']['validSamples']==2 and window['array'].tolist()==[[0,0,0,0]]


def test_crop_and_explicit_alpha_composite_keep_exact_geometry():
    c=prep.defaults('image');c.update(sizeMode='crop',width=2,height=1,interpolation='nearest')
    pixels=np.arange(3,dtype=np.uint8).reshape(1,3,1)
    result=prep.image_tensor(pixels,c)
    assert result['array'].reshape(-1).tolist()==[0,1]
    assert result['geometry']==[1.,0,0.,0,1.,0.,0,0,1]
    c=prep.defaults('image');c.update(alpha='composite',matte=[1,0,0])
    rgba=np.array([[[0,0,255,0],[0,0,255,255]]],dtype=np.uint8)
    assert prep.image_tensor(rgba,c)['array'].tolist()==[[[255,0,0],[0,0,255]]]


def test_prepared_waveform_repreparation_does_not_turn_padding_into_valid_audio():
    c=prep.defaults('audio');c.update(sampleFormat='float32')
    result=list(prep.audio_windows(np.array([[.5,.2,0,0]],dtype='<f4').tobytes(),shape=[1,4],dtype='float32',metadata=dict(sampleRate=16000,sourceRate=48000,sourceStart=480,sourceEnd=486,validSamples=2),config=c))
    assert result[0]['array'].shape==(1,2) and result[0]['metadata']['validSamples']==2
    assert result[0]['metadata']['sourceStart']==480 and result[0]['metadata']['sourceEnd']==486

@pytest.mark.parametrize('bits,dtype,values',[(8,'uint8',[0,255]),(16,'int16',[-32768,32767]),(24,'int32',[-8388608,8388607]),(32,'int32',[-2147483648,2147483647])])
def test_integer_waveform_composes_and_normalizes_original_precision(bits,dtype,values):
 c=prep.defaults('audio');c.update(windowSamples=4,tail='pad')
 source=np.array(values,dtype=dtype).reshape(2,1)
 first=next(prep.audio_windows(source.tobytes(),shape=[2,1],dtype=dtype,metadata=dict(sampleRate=16000,sourceBits=bits),config=c))
 assert first['metadata']['sourceBits']==bits
 for fmt in ['preserve','float32']:
  config=prep.defaults('audio');config['sampleFormat']=fmt
  second=next(prep.audio_windows(first['array'].tobytes(),shape=list(first['array'].shape),dtype=dtype,metadata=first['metadata'],config=config,axes=first['axes']))
  assert second['array'].shape==(1,2) and second['metadata']['validSamples']==2
  assert second['metadata']['sourceStart']==0 and second['metadata']['sourceEnd']==2
  expected=source.T if fmt=='preserve' else ((source.T.astype('float32')-128)/128 if bits==8 else source.T.astype('float32')/(2**(bits-1)))
  np.testing.assert_array_equal(second['array'],expected)

def test_legacy_integer_prepared_waveform_missing_precision_refuses_without_guessing():
 with pytest.raises(ValueError,match='precision'):
  next(prep.audio_windows(bytes(16),shape=[1,4],dtype='int32',metadata=dict(sampleRate=16000,validSamples=2,sourceStart=0,sourceEnd=2,sourceRate=16000),config=prep.defaults('audio'),axes=['channels','samples']))

@pytest.mark.parametrize('engine',['disk','polars'])
def test_integer_preparation_composes_through_owned_media_tables(engine):
 from runner import media_artifacts as media,dataflow as df
 from runner.polars_preparation import preparation_engine
 assets={}
 raw,ref=media.encode_asset(np.array([[1,-8388608],[2,8388607]],dtype='<i4').tobytes(),kind='audio',shape=[2,2],axes=['samples','channels'],dtype='int32',metadata=dict(sampleRate=16000,sourceBits=24))
 assets[ref['sha256']]=raw
 source=df.Table([dict(name='audio_content',dtype='audio',nullable=False)],[[ref]],['audio'],'source')
 first=prep.defaults('audio');first.update(channels='select',channelIndex=1,windowSamples=4,tail='pad')
 second=prep.defaults('audio');second.update(inputColumn='prepared',outputColumn='normalized',sampleFormat='float32')
 with preparation_engine(engine),media.artifact_scope(lambda raw,ref:assets.setdefault(ref['sha256'],raw),lambda ref:assets[ref['sha256']]):
  prepared=prep.apply(source,first,lambda:None)
  result=prep.apply(prepared,second,lambda:None)
  output=result.rows[0][result.index('normalized')];_,samples=media.read_asset(output)
 assert output['shape']==[1,2] and output['metadata']['validSamples']==2
 np.testing.assert_array_equal(np.frombuffer(samples,dtype='<f4'),[-1,8388607/8388608])
