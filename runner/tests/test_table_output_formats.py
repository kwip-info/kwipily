"""Independent rereads of actual Table Output options and publication safety."""
import csv
import io
import json
from pathlib import Path
import pytest
from runner.dataflow import Table
from runner.dataflow_output import OutputTransaction
from runner.table_output_writers import options

def table():
 return Table(({'name':'text','dtype':'string'},{'name':'n','dtype':'int64'},{'name':'f','dtype':'float64'},{'name':'flag','dtype':'bool'}),((None,1,1.2345678901234567,True),('',-2,-0.0,False),('é,"line\nnext',9223372036854775807,1e-10,None),('=2+3',4,4.5,True)),('a','b','c','d'),'source')

def export(tmp_path,fmt='csv',opts=None,source=None,**changes):
 reader=changes.pop('image_reader',None)
 config={**dict(filename='out.'+fmt,separator='\t' if fmt=='tsv' else ',',columns=[],overwrite=False,format=fmt,options=opts or {}),**changes}
 tx=OutputTransaction({'out':str(tmp_path)},image_reader=reader)
 try:tx.stage('out',source or table(),config);receipts=tx.commit()
 finally:tx.close()
 return tmp_path/config['filename'],receipts[0]

def test_legacy_roundtrip_complete_and_receipt(tmp_path):
 config=dict(filename='legacy.csv',separator=';',columns=['n','text'],overwrite=False)
 tx=OutputTransaction({'out':str(tmp_path)})
 try:tx.stage('out',table(),config);receipt=tx.commit()[0]
 finally:tx.close()
 with (tmp_path/'legacy.csv').open(newline='') as stream:rows=list(csv.reader(stream,delimiter=';'))
 assert rows==[['n','text'],['1',''],['-2',''],['9223372036854775807','é,"line\nnext'],['4','=2+3']]
 assert receipt['rowCount']==4 and receipt['format']=='csv' and receipt['completeSelectedTable']
 assert receipt['bytes']==(tmp_path/'legacy.csv').stat().st_size

@pytest.mark.parametrize('encoding,bom',[('utf-8',True),('utf-8',False),('utf-16-le',True),('utf-16-be',True)])
def test_csv_encoding_bom_newline_header_numeric(tmp_path,encoding,bom):
 path,_=export(tmp_path,opts=dict(encoding=encoding,bom=bom,newline='CRLF',header=False,nullText='NULL',numericFormat='custom',numericPattern='.2f',quoting='all',spreadsheetText='apostrophe'))
 raw=path.read_bytes();prefix={'utf-8':b'\xef\xbb\xbf','utf-16-le':b'\xff\xfe','utf-16-be':b'\xfe\xff'}[encoding]
 assert raw.startswith(prefix)==bom
 text=raw[len(prefix):].decode(encoding) if bom else raw.decode(encoding)
 assert '\r\n' in text
 rows=list(csv.reader(io.StringIO(text,newline='')))
 assert rows[0]==['NULL','1.00','1.23','true'] and rows[-1][0]=="'=2+3"

@pytest.mark.parametrize('quoting,escaping',[('minimal','double'),('minimal','escape'),('all','none'),('none','escape')])
def test_custom_delimiter_quote_escape(tmp_path,quoting,escaping):
 path,_=export(tmp_path,'tsv',dict(quoting=quoting,escaping=escaping,quoteChar="'",escapeChar='\\'),separator='|')
 with path.open(newline='') as stream:rows=list(csv.reader(stream,delimiter='|',quotechar="'",quoting=csv.QUOTE_NONE if quoting=='none' else csv.QUOTE_MINIMAL,doublequote=escaping=='double',escapechar='\\' if escaping=='escape' else None))
 assert rows[3][0]=='é,"line\nnext'

@pytest.mark.parametrize('opts',[{'quoting':'none'},{'quoteChar':','},{'escaping':'escape','escapeChar':','},{'encoding':'latin1'},{'numericFormat':'custom','numericPattern':'100000000f'},{'spreadsheetText':'stop'}])
def test_csv_refusals_clean(tmp_path,opts):
 with pytest.raises((ValueError,csv.Error)):export(tmp_path,opts=opts)
 assert not list(tmp_path.iterdir())

@pytest.mark.parametrize('fmt,opts',[('json',{'pretty':True,'ascii':True}),('json',{'shape':'rowsSchema'}),('ndjson',{'ascii':False,'nullPolicy':'omit'})])
def test_json_shape_unicode_null_and_no_invalid_numbers(tmp_path,fmt,opts):
 path,receipt=export(tmp_path,fmt,opts)
 raw=path.read_text();result=[json.loads(line) for line in raw.splitlines()] if fmt=='ndjson' else json.loads(raw)
 if opts.get('shape')=='rowsSchema':assert result['columns'][1]['dtype']=='int64';assert result['rows'][2][1]==9223372036854775807
 else:
  assert result[2]['text']=='é,"line\nnext' and result[2]['n']==9223372036854775807
  assert ('text' not in result[0]) if opts.get('nullPolicy')=='omit' else result[0]['text'] is None
 if opts.get('ascii'):assert '\\u00e9' in raw
 assert receipt['rowCount']==4

@pytest.mark.parametrize('fmt',['json','ndjson'])
def test_nonfinite_policy(tmp_path,fmt):
 source=Table(({'name':'v','dtype':'float64'},),((float('nan'),),(float('inf'),)),('a','b'),'x')
 with pytest.raises(ValueError,match='Nonfinite'):export(tmp_path,fmt,source=source)
 path,_=export(tmp_path,fmt,{'nonfinite':'null'},source=source)
 raw=path.read_text();assert 'NaN' not in raw and 'Infinity' not in raw

@pytest.mark.parametrize('codec,level',[('none',None),('snappy',None),('gzip',5),('brotli',3),('zstd',3),('lz4',None)])
def test_parquet_typed_settings(tmp_path,codec,level):
 import pyarrow.parquet as pq
 path,_=export(tmp_path,'parquet',dict(compression=codec,compressionLevel=level,rowGroupSize=2,dataPageSize=128,statistics=False))
 result=pq.read_table(path);assert result.column('n').to_pylist()==[1,-2,9223372036854775807,4]
 assert result.column('text').to_pylist()==[None,'','é,"line\nnext','=2+3']
 file=pq.ParquetFile(path);assert file.metadata.num_row_groups==2 and file.metadata.row_group(0).column(0).statistics is None
 assert json.loads(result.schema.metadata[b'kwip.columns'])[1]['dtype']=='int64'

@pytest.mark.parametrize('opts',[{'compression':'snappy','compressionLevel':3},{'compression':'gzip','compressionLevel':10},{'rowGroupSize':0}])
def test_parquet_invalid_opts(tmp_path,opts):
 with pytest.raises(ValueError):export(tmp_path,'parquet',opts)
 assert not list(tmp_path.iterdir())

def test_excel_literal_precision_layout(tmp_path):
 from openpyxl import load_workbook
 with pytest.raises(ValueError,match='precision'):export(tmp_path,'xlsx')
 path,_=export(tmp_path,'xlsx',dict(sheet='Orders',startCell='C3',freeze=True,autofilter=True,numberFormat='0.00',precisionPolicy='text'))
 book=load_workbook(path,data_only=False);sheet=book['Orders']
 assert sheet['C3'].value=='text' and sheet['C6'].value=='é,"line\nnext'
 assert sheet['C7'].value=='=2+3' and sheet['C7'].data_type=='s'
 assert sheet['D6'].value=='9223372036854775807' and sheet['D6'].data_type=='s'
 assert sheet.freeze_panes=='C4' and sheet.auto_filter.ref=='C3:F7'
 assert sheet['E4'].number_format=='0.00';book.close()

@pytest.mark.parametrize('opts',[{'startCell':'XFD1048576'},{'startCell':'A0'},{'sheet':'bad/name'}])
def test_excel_capacity_invalid_sheet(tmp_path,opts):
 with pytest.raises(ValueError):export(tmp_path,'xlsx',opts)
 assert not list(tmp_path.iterdir())

def test_target_changed_and_conflict_safe(tmp_path):
 tx=OutputTransaction({'out':str(tmp_path),'second':str(tmp_path)})
 config=dict(filename='out.csv',separator=',',columns=[],overwrite=True)
 path=tmp_path/'out.csv';path.write_text('old')
 try:
  tx.stage('out',table(),config)
  with pytest.raises(ValueError,match='same destination'):tx.stage('second',table(),config)
  path.write_text('concurrent change')
  with pytest.raises(ValueError,match='changed'):tx.commit()
 finally:tx.close()
 assert path.read_text()=='concurrent change' and list(tmp_path.iterdir())==[path]

def test_cancel_during_stage_cleans(tmp_path):
 calls=0
 def cancel():
  nonlocal calls;calls+=1
  if calls>2:raise KeyboardInterrupt('cancel')
 tx=OutputTransaction({'out':str(tmp_path)},cancel)
 try:
  with pytest.raises(KeyboardInterrupt):tx.stage('out',table(),dict(filename='out.csv',separator=',',columns=[],overwrite=False))
 finally:tx.close()
 assert not list(tmp_path.iterdir())

def test_bundle_captured_images_no_dangling_paths(tmp_path):
 from runner.image_artifacts import encode_pixels,decode_pixels
 raw,reference=encode_pixels(b'\x01\x02\x03',width=1,height=1,channels=3,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='RGB',palette='reject',actualMode='RGB'))
 source=Table(({'name':'image','dtype':'image'},{'name':'name','dtype':'string'}),((reference,'one'),(reference,'two'),(None,'none')),('a','b','c'),'x')
 config=dict(filename='bundle',separator=',',columns=[],overwrite=False,format='json',imagePolicy='copyAssets')
 tx=OutputTransaction({'out':str(tmp_path)},image_reader=lambda ref:raw)
 try:tx.stage('out',source,config);receipt=tx.commit()[0]
 finally:tx.close()
 manifest=json.loads((tmp_path/'bundle/manifest.json').read_text());rows=json.loads((tmp_path/'bundle/table.json').read_text())
 assert len(manifest['assets'])==1 and rows[0]['image']==rows[1]['image'] and rows[2]['image'] is None
 assert (tmp_path/'bundle'/rows[0]['image']).read_bytes()==raw
 assert receipt['assetBundleManifest']==manifest
 tx=OutputTransaction({'out':str(tmp_path)},image_reader=lambda ref:b'invalid')
 try:
  config['filename']='broken'
  with pytest.raises(ValueError):tx.stage('out',source,config)
 finally:tx.close()
 assert not (tmp_path/'broken').exists()
 assert sorted(p.name for p in tmp_path.iterdir())==['bundle']

def test_temporal_all_formats(tmp_path):
 from runner.dataflow_temporal import metadata
 source=Table(tuple({'name':k,'dtype':k,'temporal':metadata(k)} for k in ('date','datetime','time')),((0,1234567,1234567),),('a',),'x')
 for fmt in ('csv','json','ndjson','parquet','xlsx'):
  opts={'precisionPolicy':'text'} if fmt=='xlsx' else {}
  path,_=export(tmp_path,fmt,opts,source=source)
  if fmt=='csv':assert '1970-01-01T00:00:01.234567+00:00' in path.read_text()
  if fmt=='json':assert json.loads(path.read_text())[0]['time']=='00:00:01.234567'
  if fmt=='parquet':
   import pyarrow.parquet as pq
   result=pq.read_table(path);assert result.schema.field('datetime').type.unit=='us';assert result['datetime'][0].as_py().microsecond==234567

@pytest.mark.parametrize('fmt',['csv','tsv','parquet','json','ndjson','xlsx'])
def test_cli_complete_population_and_standalone_export(tmp_path,fmt):
 import subprocess
 import sys
 from runner.dataflow_cli import run_dataflow
 source=tmp_path/'source.csv';source.write_text('id,value\n'+''.join(f'{i},record{i}\n' for i in range(12)))
 graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}},{'id':'save','operator':'kwip.data.outputDelimited','config':dict(filename='complete.'+fmt,separator='\t' if fmt=='tsv' else ',',columns=[],overwrite=False,format=fmt,options={})}],'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'save','port':'table'}}]}
 graph_path=tmp_path/'graph.json';graph_path.write_text(json.dumps(graph));destination=tmp_path/'destination';destination.mkdir();events=[]
 archive=tmp_path/'archive';run_dataflow(graph_path,{'source':str(source)},archive,events.append,destination_bindings={'save':str(destination)},sampling={'mode':'rowCount','rowCount':3})
 receipt=next(event for event in events if event['event']=='dataflowOutputSaved')
 assert receipt['rowCount']==12 and receipt['completeSelectedTable']
 preview=next(event for event in events if event.get('nodeID')=='save' and event['event']=='dataflowPreview')
 assert len(preview['outputs']['table']['preview']['rows'])<=8
 assert str(destination) not in (archive/'dataflow.json').read_text()
 other=tmp_path/'replay-destination';other.mkdir();bindings=tmp_path/'sources.json';bindings.write_text(json.dumps({'source':str(source)}));destinations=tmp_path/'destinations.json';destinations.write_text(json.dumps({'save':str(other)}))
 replay=subprocess.run([sys.executable,'-I',str(archive/'prep.py'),'--source-bindings',str(bindings),'--destination-bindings',str(destinations),'--output',str(tmp_path/'replay')],capture_output=True,text=True)
 assert replay.returncode==0,replay.stderr
 filename='complete.'+fmt
 if fmt!='xlsx':assert (destination/filename).read_bytes()==(other/filename).read_bytes()
 else:
  from openpyxl import load_workbook
  for path in (destination/filename,other/filename):
   book=load_workbook(path);assert list(book.active.values)[-1]==('11','record11');book.close()

@pytest.mark.parametrize('engine',['polars','disk'])
def test_cli_captured_image_bundle_and_replay(tmp_path,engine):
 import hashlib
 import subprocess
 import sys
 from PIL import Image
 from runner.file_input_options import defaults
 from runner.dataflow_cli import run_dataflow
 from runner.image_artifacts import decode_pixels
 source=tmp_path/'image.png';Image.new('RGB',(2,1),(10,20,30)).save(source)
 graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceFiles','config':defaults()},{'id':'save','operator':'kwip.data.outputDelimited','config':dict(filename='images',separator=',',columns=['name','image_content'],overwrite=False,format='json',imagePolicy='copyAssets',options={})}],'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'save','port':'table'}}]}
 graph_path=tmp_path/'graph.json';graph_path.write_text(json.dumps(graph));destination=tmp_path/'destination';destination.mkdir();archive=tmp_path/'archive'
 run_dataflow(graph_path,{'source':str(source)},archive,lambda _:None,destination_bindings={'save':str(destination)},engine=engine)
 bundle=destination/'images';manifest=json.loads((bundle/'manifest.json').read_text());rows=json.loads((bundle/'table.json').read_text());raw=(bundle/rows[0]['image_content']).read_bytes()
 assert manifest['assets'][0]['sha256']==hashlib.sha256(raw).hexdigest()
 assert decode_pixels(raw)[1]==bytes((10,20,30))*2
 assert str(source) not in (bundle/'table.json').read_text()
 other=tmp_path/'other';other.mkdir();bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}));destinations=tmp_path/'destinations.json';destinations.write_text(json.dumps({'save':str(other)}))
 replay=subprocess.run([sys.executable,'-I',str(archive/'prep.py'),'--source-bindings',str(bindings),'--destination-bindings',str(destinations),'--output',str(tmp_path/'replay')],capture_output=True,text=True)
 assert replay.returncode==0,replay.stderr
 assert (other/'images'/rows[0]['image_content']).read_bytes()==raw
 assert json.loads((other/'images/manifest.json').read_text())==manifest

@pytest.mark.parametrize('fmt',['csv','json','ndjson'])
def test_temporal_explicit_representations(tmp_path,fmt):
 from runner.dataflow_temporal import metadata
 source=Table(({'name':'when','dtype':'datetime','temporal':metadata('datetime','America/Chicago')},),((1_234_567,),),'a','x')
 opts={'temporalFormat':'pattern','temporalPattern':'%Y/%m/%d %H:%M:%S.%f%z'} if fmt=='csv' else {'temporal':'integer'}
 path,_=export(tmp_path,fmt,opts,source=source)
 if fmt=='csv':assert '1969/12/31 18:00:01.234567-0600' in path.read_text()
 else:
  result=json.loads(path.read_text()) if fmt=='json' else json.loads(path.read_text().strip())
  assert (result[0] if fmt=='json' else result)['when']==1_234_567

@pytest.mark.parametrize('fmt',['csv','json','parquet','xlsx'])
def test_header_and_image_defaults_refuse_without_authority(tmp_path,fmt):
 source=Table(({'name':'image','dtype':'image'},),((None,),),('a',),'x')
 with pytest.raises(ValueError,match='Image columns'):export(tmp_path,fmt,source=source)
 with pytest.raises(ValueError,match='captured image'):export(tmp_path,fmt,source=source,imagePolicy='copyAssets')
 assert list(tmp_path.iterdir())==[]

def test_excel_exact_temporals_and_no_header(tmp_path):
 from runner.dataflow_temporal import metadata
 from openpyxl import load_workbook
 from datetime import date,time
 source=Table(({'name':'day','dtype':'date','temporal':metadata('date')},{'name':'clock','dtype':'time','temporal':metadata('time')}),((0,1_234_000),),('a',),'x')
 path,_=export(tmp_path,'xlsx',dict(header=False,startCell='B2',dateFormat='yyyy-mm-dd hh:mm:ss.000'),source=source)
 book=load_workbook(path);sheet=book.active
 assert sheet['B2'].value.date()==date(1970,1,1) and sheet['C2'].value==time(0,0,1,234000)
 assert sheet['B2'].number_format=='yyyy-mm-dd hh:mm:ss.000';assert sheet.max_row==2;book.close()

def test_atomic_replace_and_cancel_bundle_commit(tmp_path):
 path=tmp_path/'out.csv';path.write_text('old')
 export(tmp_path,overwrite=True)
 assert path.read_text().startswith('text,n,f,flag')
 from runner.image_artifacts import encode_pixels
 raw,reference=encode_pixels(b'\x01',width=1,height=1,channels=1,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='L',palette='reject',actualMode='L'))
 source=Table(({'name':'image','dtype':'image'},),((reference,),),('a',),'x')
 tx=OutputTransaction({'out':str(tmp_path)},image_reader=lambda ref:raw)
 try:
  tx.stage('out',source,dict(filename='bundle',separator=',',columns=[],overwrite=False,format='json',imagePolicy='copyAssets'))
  calls=0
  def cancel_after_publish():
   nonlocal calls;calls+=1
   if calls==3:raise KeyboardInterrupt('late cancel')
  tx.check=cancel_after_publish
  with pytest.raises(KeyboardInterrupt):tx.commit()
 finally:tx.close()
 assert list(tmp_path.iterdir())==[path]

def test_no_overwrite_race_preserves_concurrent_file(tmp_path,monkeypatch):
 import runner.dataflow_output as module
 tx=OutputTransaction({'out':str(tmp_path)})
 config=dict(filename='out.csv',separator=',',columns=[],overwrite=False)
 original=module.atomic_rename
 def race(fd,source,target,exchange=False):
  (tmp_path/'out.csv').write_text('racing writer')
  return original(fd,source,target,exchange)
 try:
  tx.stage('out',table(),config);monkeypatch.setattr(module,'atomic_rename',race)
  with pytest.raises(ValueError):tx.commit()
 finally:tx.close()
 assert (tmp_path/'out.csv').read_text()=='racing writer'
 assert len(list(tmp_path.iterdir()))==1

@pytest.mark.parametrize('fmt',['csv','tsv'])
@pytest.mark.parametrize('policy',['stop','apostrophe'])
def test_spreadsheet_text_policy_covers_header(tmp_path,fmt,policy):
 source=Table(({'name':'=SUM(A1:A2)','dtype':'string'},),(('ordinary value',),),('a',),'x')
 if policy=='stop':
  with pytest.raises(ValueError,match='Spreadsheet-like'):export(tmp_path,fmt,{'spreadsheetText':policy},source=source)
  assert not list(tmp_path.iterdir())
 else:
  path,_=export(tmp_path,fmt,{'spreadsheetText':policy},source=source)
  with path.open(newline='') as stream:result=list(csv.reader(stream,delimiter='\t' if fmt=='tsv' else ','))
  assert result==[["'=SUM(A1:A2)"],['ordinary value']]

@pytest.mark.parametrize('failure',['write','fsync'])
def test_image_asset_io_failure_cleans_owned_bundle(tmp_path,monkeypatch,failure):
 import runner.dataflow_output as module
 from runner.image_artifacts import encode_pixels
 raw,reference=encode_pixels(b'\x01',width=1,height=1,channels=1,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='L',palette='reject',actualMode='L'))
 source=Table(({'name':'image','dtype':'image'},),((reference,),),('a',),'x')
 config=dict(filename='bundle',separator=',',columns=[],overwrite=False,format='json',imagePolicy='copyAssets')
 original_open=module.os.open;original_fdopen=module.os.fdopen;original_fsync=module.os.fsync;asset_fds=set()
 def tracked_open(path,*args,**kwargs):
  fd=original_open(path,*args,**kwargs)
  if isinstance(path,str) and path.startswith('assets/'):asset_fds.add(fd)
  return fd
 class BrokenWriter:
  def __init__(self,stream):self.stream=stream
  def __enter__(self):return self
  def __exit__(self,*args):return self.stream.__exit__(*args)
  def write(self,data):self.stream.write(data[:3]);raise OSError('injected asset write failure')
  def __getattr__(self,name):return getattr(self.stream,name)
 def fdopen(fd,*args,**kwargs):
  stream=original_fdopen(fd,*args,**kwargs)
  return BrokenWriter(stream) if failure=='write' and fd in asset_fds else stream
 def fsync(fd):
  if failure=='fsync' and fd in asset_fds:raise OSError('injected asset fsync failure')
  return original_fsync(fd)
 monkeypatch.setattr(module.os,'open',tracked_open);monkeypatch.setattr(module.os,'fdopen',fdopen);monkeypatch.setattr(module.os,'fsync',fsync)
 tx=OutputTransaction({'out':str(tmp_path)},image_reader=lambda ref:raw)
 try:
  with pytest.raises(OSError,match='injected asset'):tx.stage('out',source,config)
 finally:tx.close()
 assert not list(tmp_path.iterdir())
