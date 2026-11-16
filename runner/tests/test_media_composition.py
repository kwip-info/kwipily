"""Stored/captured identity and source-order regressions across actual media transforms."""
import pytest
from runner import dataflow as df,media_artifacts as media,image_artifacts as images,ocr,ocr_tables,speech_tables,audio_classification_tables
from runner.polars_preparation import preparation_engine
from runner.tests.test_speech_tables import table,Recognizer
from runner.tests.test_audio_classification_tables import Classifier

@pytest.mark.parametrize('engine',['disk','polars'])
def test_expanded_speech_and_audio_ids_do_not_inherit_stored_source_prefix(engine):
 raw,ref,t=table()
 with preparation_engine(engine):
  store=df.row_store(t.columns,lambda:None,identity_prefix='source:');store.append([ref],'source:one');rows,ids=store.finish()
  source=df.Table(t.columns,rows,ids,'source')
  try:
   with media.artifact_scope(lambda *_:None,lambda *_:raw):
    speech=speech_tables.apply(source,dict(speech_tables.DEFAULTS),lambda:None,'speech',Recognizer())
    classified=audio_classification_tables.apply(source,dict(audio_classification_tables.DEFAULTS),lambda:None,'audio',Classifier())
   assert list(speech['transcripts'].row_ids)==['source:one']
   assert len(speech['segments'].rows)==1 and speech['segments'].rows[0][0]=='source:one'
   assert len(classified['predictions'].rows)==1 and len(classified['scores'].rows)==3
   assert len(set(classified['scores'].row_ids))==3
   assert all(row[0]=='source:one' for row in classified['scores'].rows)
  finally:store.close()

class Blank:
 def recognize(self,*args):return []

def image_fixture():
 saved={}
 with images.artifact_scope(lambda raw,ref:saved.setdefault(ref['sha256'],raw)):
  ref=images.publish_pixels(bytes(30),width=10,height=1,channels=3,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='RGB',palette='expandRGBA',actualMode='RGB'))
 columns=tuple(dict(name=n,dtype=t,nullable=False) for n,t in [('file_id','string'),('frame_index','int64'),('image_content','image')])
 return saved,ref,columns

@pytest.mark.parametrize('engine',['disk','polars'])
def test_ocr_derived_document_ids_and_grouped_source_order(engine):
 saved,ref,columns=image_fixture()
 with preparation_engine(engine):
  store=df.row_store(columns,lambda:None,identity_prefix='source:')
  for group,number,identity in [('one',5,'a'),('two',0,'b'),('one',2,'c')]:store.append([group,number,ref],'source:'+identity)
  rows,ids=store.finish();source=df.Table(columns,rows,ids,'source')
  try:
   out=ocr_tables.apply(source,dict(ocr.defaults(),pageColumn=''),lambda:None,lambda r:saved[r['sha256']],recognizer=Blank())
   docs=out['documents'];assert len(docs.rows)==2 and len(set(docs.row_ids))==2
   assert [row[docs.index('ocr_page_count')] for row in docs.rows]==[2,1]
  finally:store.close()

@pytest.mark.parametrize('config,changes,message',[
 ({'groupBy':'frame_index'}, {},'group'),({'pageColumn':'file_id'}, {},'order'),
 ({}, {'rows':[['one',0],['one',0]]},'duplicate')])
def test_ocr_refuses_incompatible_group_order_types_and_explicit_duplicates(config,changes,message):
 saved,ref,columns=image_fixture();rows=changes.get('rows',[['one',0],['one',1]])
 source=df.Table(columns,[[*r,ref] for r in rows],['p0','p1'],'source')
 with pytest.raises(ValueError,match=message):ocr_tables.apply(source,dict(ocr.defaults(),**config),lambda:None,lambda r:saved[r['sha256']],recognizer=Blank())

@pytest.mark.parametrize('engine',['disk','polars'])
def test_real_large_preparation_cache_rerun_keeps_complete_speech_outputs(tmp_path,monkeypatch,engine):
 import json,csv,hashlib,copy
 from runner.tests.test_file_input_audio import wav
 from runner.file_input_options import defaults as file_defaults
 from runner.prepare_media import defaults as prepare_defaults
 from runner.dataflow_cli import run_dataflow
 from runner.dataflow_table_reader import read_table_artifact
 folder=tmp_path/'audio';folder.mkdir();names=[str(i)+'.wav' for i in range(12)]
 for name in names:(folder/name).write_bytes(wav(bytes(320),channels=1,rate=16000))
 labels=tmp_path/'labels.csv'
 with labels.open('w') as stream:
  writer=csv.writer(stream);writer.writerow(['file','note']);writer.writerows((name,'x'*90000) for name in names)
 source=file_defaults();source.update(sourceMode='directory',extensions=['wav']);source['readers']['audio']={'format':'pcmWav'}
 prep=prepare_defaults('audio');prep.update(sampleFormat='float32',channels='mono',targetRate=16000)
 join=copy.deepcopy(df.DEFINITIONS['join']['defaultConfig']);join.update(leftKeys=['name'],rightKeys=['file'],leftPrefix='left_',rightPrefix='note_',cardinality='oneToOne')
 nodes=[('source','sourceFiles',source),('labels','sourceCSV',{}),('joined','join',join),('prepared','prepareMedia',prep),('speech','transcribeAudio',dict(speech_tables.DEFAULTS))]
 edges=[('source','table','joined','left'),('labels','table','joined','right'),('joined','table','prepared','table'),('prepared','table','speech','table')]
 graph=dict(schemaVersion=1,nodes=[dict(id=n,operator='kwip.data.'+op,config=c) for n,op,c in nodes],edges=[{'from':dict(node=a,port=ap),'to':dict(node=b,port=bp)} for a,ap,b,bp in edges])
 path=tmp_path/'graph.json';path.write_text(json.dumps(graph));bindings={'source':str(folder),'labels':str(labels)}
 monkeypatch.setattr(speech_tables,'adapter',lambda _:Recognizer())
 first=run_dataflow(path,bindings,tmp_path/'first',lambda _:None,cache_policy='materialize',engine=engine)
 assert next(e['bytes'] for e in first['artifacts'] if e.get('nodeID')=='prepared' and e['kind']=='table')>1000000
 manifest=tmp_path/'first/manifest.json'
 second=run_dataflow(path,bindings,tmp_path/'second',lambda _:None,cache_policy='reuse',reuse_manifest=manifest,reuse_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),engine=engine)
 assert 'prepared' in second['cache']['reusedNodeIDs']
 for port in ['transcripts','segments']:
  captured=[]
  for parent,result in [('first',first),('second',second)]:
   entry=next(e for e in result['artifacts'] if e.get('nodeID')=='speech' and e.get('port')==port and e['kind']=='table')
   table=read_table_artifact(tmp_path/parent/entry['path'],entry['sha256'])
   try:captured.append((list(table.rows),list(table.row_ids)))
   finally:table.rows.store.close();table.row_ids.store.close()
  assert len(captured[0][0])==12 and captured[0]==captured[1]

@pytest.mark.parametrize('config,message',[({'groupBy':'frame_index'},'group'),({'pageColumn':'file_id'},'order')])
def test_ocr_checks_column_types_even_for_empty_input_before_model_admission(config,message):
 _,_,columns=image_fixture();source=df.Table(columns,[],[],'source')
 with pytest.raises(ValueError,match=message):ocr_tables.apply(source,dict(ocr.defaults(),**config),lambda:None,lambda _:b'')
