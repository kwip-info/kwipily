import json,os,shutil,hashlib
from pathlib import Path
import pytest

@pytest.mark.parametrize("capture",[True,False])
def test_real_offline_speech_complete_graph_and_source_removed_history(tmp_path,capture):
 model=os.environ.get('KWIPML_WHISPER_FIXTURE');audio=os.environ.get('KWIPML_SPEECH_WAV')
 if not model or not audio:pytest.skip('Select the pinned offline Whisper model and spoken PCM WAV.')
 from runner.pretrained_models import inspect_model
 from runner.speech_tables import DEFAULTS
 from runner.prepare_media import defaults as prepare_defaults
 from runner.file_input_options import defaults as file_defaults
 from runner.dataflow_cli import run_dataflow
 from runner.dataflow_storage import completed_inventory
 from runner.dataflow_table_reader import read_table_artifact
 source=tmp_path/'recording.wav';shutil.copyfile(audio,source);audio=str(source)
 info=inspect_model(model);assert info['task']=='automaticSpeechRecognition'
 folder=tmp_path/'speech-models'/info['identity'];folder.mkdir(parents=True)
 for entry in info['files']:shutil.copyfile(Path(model)/entry['path'],folder/entry['path'])
 (tmp_path/'speech-models.json').write_text(json.dumps({'speech':dict(identity=info['identity'],files=info['files'])}))
 c=prepare_defaults('audio');c.update(sampleFormat='float32',channels='mono',targetRate=16000)
 read=file_defaults();read['readers']['audio']={'format':'pcmWav'}
 export=tmp_path/'export';export.mkdir()
 nodes=[('source','sourceFiles',read),('prepare','prepareMedia',c),('speech','transcribeAudio',dict(DEFAULTS,language='en')),('export','outputDelimited',dict(filename='transcript.csv',separator=',',columns=['transcript','speech_status'],overwrite=False))]
 edges=[('source','table','prepare'),('prepare','table','speech'),('speech','transcripts','export')]
 graph=dict(schemaVersion=1,nodes=[dict(id=i,operator='kwip.data.'+op,config=c) for i,op,c in nodes],edges=[{'from':dict(node=a,port=p),'to':dict(node=b,port='table')} for a,p,b in edges])
 path=tmp_path/'graph.json';path.write_text(json.dumps(graph));out=tmp_path/'out';run_dataflow(path,{'source':audio},out,lambda _:None,destination_bindings={'export':str(export)},retain_outputs=capture,required_nodes=['speech','export'])
 assert 'country' in (export/'transcript.csv').read_text().lower()
 manifest,_,_=completed_inventory(out/'manifest.json')
 for port in ('transcripts','segments'):
  entry=next(x for x in manifest['artifacts'] if x.get('nodeID')=='speech' and x.get('port')==port and x['kind']=='table')
  table=read_table_artifact(out/entry['path'],entry['sha256'])
  try:
   assert len(table.rows)>0
   text=table.rows[0][table.index('transcript' if port=='transcripts' else 'text')]
   assert 'country' in text.lower() and 'americans' in text.lower()
   if port=='transcripts':assert table.rows[0][table.index('speech_status')]=='transcribed'
   if port=='segments':assert 0<=table.rows[0][table.index('start_seconds')]<table.rows[0][table.index('end_seconds')]<=11
  finally:
   if hasattr(table.rows,'store'):table.rows.store.close()
 # No model/source read is necessary to verify the completed output inventory.
 source.unlink();shutil.rmtree(folder);completed_inventory(out/'manifest.json')


def test_real_whisper_chunk_offsets_and_token_allowance_not_terminal_token_guess():
 model=os.environ.get('KWIPML_WHISPER_FIXTURE');audio=os.environ.get('KWIPML_SPEECH_WAV')
 if not model or not audio:pytest.skip('Select the pinned offline speech fixture.')
 import wave,numpy as np
 from runner.speech_models import Whisper
 from runner.speech_tables import DEFAULTS
 with wave.open(audio) as w:
  assert w.getframerate()==16000 and w.getsampwidth()==2
  values=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype('float32')/32768
 engine=Whisper(model)
 chunks=engine.transcribe(values,16000,dict(DEFAULTS,language='en',chunkSeconds=5))
 assert [x['start'] for x in chunks]==[0,5,10] and [x['end'] for x in chunks]==[5,10,11]
 assert all(not x['truncated'] for x in chunks)
 assert all(chunk['start']<=a<b<=chunk['end'] for chunk in chunks for a,b,_ in chunk['segments'])
 short=engine.transcribe(values,16000,dict(DEFAULTS,language='en',maximumTokens=16))
 assert any(x['truncated'] for x in short)
 calls=0
 def cancel():
  nonlocal calls
  calls+=1
  if calls>=4:raise InterruptedError('speech cancelled')
 engine.check=cancel
 with pytest.raises(InterruptedError,match='speech cancelled'):engine.transcribe(values,16000,dict(DEFAULTS,language='en'))
