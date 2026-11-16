import numpy as np
import pytest
from runner.audio_spectral import spectral_features


def test_tone_keeps_channels_time_and_independent_bin_peak():
    rate=16000;t=np.arange(1600)/rate
    samples=np.stack([np.sin(2*np.pi*1000*t),np.sin(2*np.pi*2000*t)]).astype('float32')
    result=spectral_features(samples,rate,dict(representation='power',fftSize=400,spectralHop=160,window='hann',center=False,melBands=80,minimumHz=0,maximumHz=8000,logFloorDB=-80))
    assert result['array'].shape==(2,8,201)
    assert np.argmax(result['array'][0,0])==25
    assert np.argmax(result['array'][1,0])==50
    assert result['metadata']['frameHopSamples']==160
    assert result['axes']==['channels','frames','frequency']


def test_log_mel_silence_is_finite_and_bounds_admitted_before_work():
    config=dict(representation='logMel',fftSize=400,spectralHop=160,window='hann',center=True,melBands=80,minimumHz=0,maximumHz=8000,logFloorDB=-80)
    result=spectral_features(np.zeros((1,800),dtype='float32'),16000,config)
    assert result['array'].shape==(1,6,80) and np.isfinite(result['array']).all()
    assert np.all(result['array']==-80)
    with pytest.raises(ValueError):spectral_features(np.zeros((1,800),dtype='float32'),16000,dict(config,maximumHz=9000))
    with pytest.raises(ValueError):spectral_features(np.zeros((1,800),dtype='float32'),16000,dict(config,spectralHop=0))


def test_spectral_graph_complete_artifacts_and_mask_round_trip(tmp_path):
    import json,struct
    from runner.file_input_options import defaults
    from runner.tests.test_file_input_audio import wav
    from runner import prepare_media as prep
    from runner.dataflow_cli import run_dataflow
    from runner.dataflow_storage import completed_inventory
    from runner.dataflow_table_reader import read_table_artifact
    source=tmp_path/'tone.wav';values=(np.sin(np.arange(1600)*2*np.pi*1000/16000)*16000).astype('<i2')
    source.write_bytes(wav(values.tobytes(),channels=1,rate=16000))
    read=defaults();read['readers']['audio']={'format':'pcmWav'}
    c=prep.defaults('audio');c.update(sampleFormat='float32',representation='power',fftSize=400,spectralHop=160,center=False)
    graph=dict(schemaVersion=1,nodes=[dict(id='source',operator='kwip.data.sourceFiles',config=read),dict(id='spectrum',operator='kwip.data.prepareMedia',config=c)],edges=[{'from':dict(node='source',port='table'),'to':dict(node='spectrum',port='table')}])
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));out=tmp_path/'out';run_dataflow(path,{'source':str(source)},out,lambda _:None)
    manifest,_,_=completed_inventory(out/'manifest.json');entry=next(x for x in manifest['artifacts'] if x.get('nodeID')=='spectrum' and x['kind']=='table')
    table=read_table_artifact(out/entry['path'],entry['sha256'])
    try:
        ref=table.rows[0][table.index('prepared')]
        assert ref['shape']==[1,8,201] and ref['axes']==['channels','frames','frequency'] and ref['metadata']['spectral']['frameOffsetSamples']==0
        assert table.rows[0][table.index('prepared_valid_frames')]==8
    finally:
        if hasattr(table.rows,'store'):table.rows.store.close()
