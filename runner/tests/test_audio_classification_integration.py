"""Opt-in real pinned AST: complete graph, export and retained offline evidence."""
import json,os,shutil
from pathlib import Path
import pytest

@pytest.mark.parametrize('capture',[True,False])
def test_real_ast_complete_tables_export_and_source_removed_history(tmp_path,capture):
 model=os.environ.get('KWIPML_AST_FIXTURE');audio=os.environ.get('KWIPML_AUDIO_CLASSIFICATION_WAV')
 if not model or not audio:pytest.skip('Select the pinned offline AST and public spoken PCM WAV fixture.')
 from runner.pretrained_models import inspect_model
 from runner.audio_classification_tables import DEFAULTS
 from runner.prepare_media import defaults as prepare_defaults
 from runner.file_input_options import defaults as file_defaults
 from runner.dataflow_cli import run_dataflow
 from runner.dataflow_storage import completed_inventory
 from runner.dataflow_table_reader import read_table_artifact
 source=tmp_path/'recording.wav';shutil.copyfile(audio,source)
 info=inspect_model(model);assert info['task']=='audioClassification' and len(info['labels'])==527
 folder=tmp_path/'audio-models'/info['identity'];folder.mkdir(parents=True)
 for entry in info['files']:shutil.copyfile(Path(model)/entry['path'],folder/entry['path'])
 (tmp_path/'audio-models.json').write_text(json.dumps({'classify':dict(identity=info['identity'],files=info['files'])}))
 prepare=prepare_defaults('audio');prepare.update(sampleFormat='float32',channels='mono',targetRate=16000)
 read=file_defaults();read['readers']['audio']={'format':'pcmWav'}
 export=tmp_path/'export';export.mkdir()
 nodes=[('source','sourceFiles',read),('prepare','prepareMedia',prepare),('classify','classifyAudio',dict(DEFAULTS)),('export','outputDelimited',dict(filename='scores.csv',separator=',',columns=['label','score','rank','reported','start_seconds','end_seconds'],overwrite=False))]
 edges=[('source','table','prepare'),('prepare','table','classify'),('classify','scores','export')]
 graph=dict(schemaVersion=1,nodes=[dict(id=i,operator='kwip.data.'+op,config=c) for i,op,c in nodes],edges=[{'from':dict(node=a,port=p),'to':dict(node=b,port='table')} for a,p,b in edges])
 path=tmp_path/'graph.json';path.write_text(json.dumps(graph));out=tmp_path/'out'
 run_dataflow(path,{'source':str(source)},out,lambda _:None,destination_bindings={'export':str(export)},retain_outputs=capture,required_nodes=['classify','export'])
 assert len((export/'scores.csv').read_text().splitlines())==1055
 manifest,_,_=completed_inventory(out/'manifest.json')
 for port,count in [('predictions',2),('scores',1054)]:
  entry=next(x for x in manifest['artifacts'] if x.get('nodeID')=='classify' and x.get('port')==port and x['kind']=='table')
  table=read_table_artifact(out/entry['path'],entry['sha256'])
  try:
   assert len(table.rows)==count
   if port=='predictions':
    assert table.rows[0][table.index('audio_classification_label')]=='Speech'
    assert table.rows[1][table.index('audio_classification_start_seconds')]==8
    assert table.rows[1][table.index('audio_classification_end_seconds')]==11
   else:
    assert len({r[table.index('label')] for r in table.rows})==527
    assert sum(r[table.index('reported')] for r in table.rows)<=10
  finally:
   if hasattr(table.rows,'store'):table.rows.store.close()
 source.unlink();shutil.rmtree(folder);completed_inventory(out/'manifest.json')
