"""Bounded local Table Output codecs. Destination authority belongs to OutputTransaction."""
import csv
import io
import json
import math
import re
from datetime import datetime, timedelta, timezone, date

MAX_BYTES = 50_000_000
TEMPORAL = {'date', 'datetime', 'time'}
DEFAULTS = {
 'csv':dict(header=True,encoding='utf-8',bom=False,newline='LF',quoting='minimal',quoteChar='"',escaping='double',escapeChar='\\',nullText='',numericFormat='roundTrip',numericPattern='.6g',temporalFormat='iso',temporalPattern='%Y-%m-%dT%H:%M:%S.%f%z',spreadsheetText='preserve'),
 'tsv':{},
 'json':dict(shape='records',pretty=False,ascii=False,nullPolicy='null',temporal='iso',nonfinite='stop'),
 'ndjson':dict(ascii=False,nullPolicy='null',temporal='iso',nonfinite='stop'),
 'parquet':dict(compression='snappy',compressionLevel=None,rowGroupSize=65536,dataPageSize=1048576,statistics=True),
 'xlsx':dict(sheet='Table',startCell='A1',header=True,freeze=False,autofilter=False,numberFormat='General',dateFormat='yyyy-mm-dd hh:mm:ss.000',precisionPolicy='stop')}
DEFAULTS['tsv'] = dict(DEFAULTS['csv'])

def options(config):
 fmt=config.get('format','csv')
 if type(fmt) is not str or fmt not in DEFAULTS:raise ValueError('Unsupported Table Output format')
 given=config.get('options',{})
 if type(given) is not dict or set(given)-set(DEFAULTS[fmt]):raise ValueError('Unsupported option for selected Table Output format')
 opts={**DEFAULTS[fmt],**given}
 for key,default in DEFAULTS[fmt].items():
  if type(default) is str and type(opts[key]) is not str:raise ValueError(key+' requires text')
 def choice(key,values):
  if opts[key] not in values:raise ValueError('Unsupported Table Output '+key)
 if fmt in ('csv','tsv'):
  choice('encoding',('utf-8','utf-16-le','utf-16-be'));choice('newline',('LF','CRLF'));choice('quoting',('minimal','all','none'));choice('escaping',('double','escape','none'));choice('numericFormat',('roundTrip','custom'));choice('temporalFormat',('iso','pattern'));choice('spreadsheetText',('preserve','apostrophe','stop'))
  sep=config.get('separator','\t' if fmt=='tsv' else ',')
  if len(sep)!=1 or sep in '\r\n\0':raise ValueError('Delimiter must be one non-newline character')
  for key in ('quoteChar','escapeChar'):
   if type(opts[key]) is not str or len(opts[key])!=1 or opts[key] in '\r\n\0':raise ValueError(key+' must be one non-newline character')
  if opts['quoteChar']==sep or (opts['escaping']=='escape' and opts['escapeChar'] in (sep,opts['quoteChar'])):raise ValueError('Delimiter, quote and escape characters must differ')
  if opts['quoting']=='none' and opts['escaping']!='escape':raise ValueError('No quoting requires explicit escape-character handling')
  if opts['numericFormat']=='custom':
   if not re.fullmatch(r'\.?\d{0,2}[eEfFgG]',opts['numericPattern']):raise ValueError('Use a bounded Python numeric format such as .6g or .2f')
 elif fmt in ('json','ndjson'):
  choice('nullPolicy',('null','omit'));choice('temporal',('iso','integer'));choice('nonfinite',('stop','null'))
  if fmt=='json':choice('shape',('records','rowsSchema'))
  if fmt=='json' and opts['shape']=='rowsSchema' and opts['nullPolicy']=='omit':raise ValueError('Rows plus schema requires positional nulls')
 elif fmt=='parquet':
  choice('compression',('none','snappy','gzip','brotli','zstd','lz4'))
  level=opts['compressionLevel']
  if level is not None:
   bounds={'gzip':(0,9),'brotli':(0,11),'zstd':(-7,22)}
   if opts['compression'] not in bounds or type(level) is not int or not bounds[opts['compression']][0]<=level<=bounds[opts['compression']][1]:raise ValueError('Compression level is unsupported for selected codec')
  for key in ('rowGroupSize','dataPageSize'):
   if type(opts[key]) is not int or not 1<=opts[key]<=16_777_216:raise ValueError(key+' exceeds supported bounds')
 elif fmt=='xlsx':
  choice('precisionPolicy',('stop','text'))
  if not opts['sheet'] or len(opts['sheet'])>31 or re.search(r'[\\/*?:\[\]]',opts['sheet']) or opts['sheet'].startswith("'") or opts['sheet'].endswith("'"):raise ValueError('Invalid Excel sheet name')
  start_cell(opts['startCell'])
 for key,value in opts.items():
  if type(DEFAULTS[fmt][key]) is bool and type(value) is not bool:raise ValueError(key+' requires a boolean')
 return fmt,opts

def start_cell(value):
 if type(value) is not str or re.fullmatch('[A-Z]{1,3}[1-9][0-9]{0,6}',value) is None:raise ValueError('Start cell must be an Excel coordinate such as A1')
 letters=re.match('[A-Z]+',value)[0];column=0
 for letter in letters:column=column*26+ord(letter)-64
 row=int(value[len(letters):])
 if row>1048576 or column>16384:raise ValueError('Start cell exceeds Excel capacity')
 return row,column

def temporal_value(value,column):
 if __package__:from .dataflow_temporal import decode
 else:from dataflow_temporal import decode
 return decode(value,column['dtype'],timezone=column.get('temporal',{}).get('timezone','UTC'))

def scalar(value,column,opts,fmt):
 if value is None:return None
 kind=column['dtype']
 if kind in TEMPORAL:
  if fmt in ('json','ndjson') and opts['temporal']=='integer':return value
  if __package__:from .dataflow_temporal import format_value
  else:from dataflow_temporal import format_value
  pattern=opts['temporalPattern'] if fmt in ('csv','tsv') and opts['temporalFormat']=='pattern' else 'iso'
  return format_value(value,kind,pattern=pattern,timezone=column.get('temporal',{}).get('timezone','UTC'))
 if type(value) is float and not math.isfinite(value):
  if fmt in ('json','ndjson'):
   if opts['nonfinite']=='stop':raise ValueError('Nonfinite numbers require explicit JSON null policy')
   return None
 if fmt in ('csv','tsv'):
  if type(value) is bool:return str(value).lower()
  if type(value) in (int,float) and opts['numericFormat']=='custom':return format(value,opts['numericPattern'])
  if type(value) is str and value.lstrip().startswith(('=','+','-','@','\t','\r')):
   if opts['spreadsheetText']=='stop':raise ValueError('Spreadsheet-like text requires an explicit preservation policy')
   if opts['spreadsheetText']=='apostrophe':return "'"+value
 return value

class BudgetStream:
 def __init__(self,stream):self.stream=stream;self.size=0
 def write(self,raw):
  self.size+=len(raw)
  if self.size>MAX_BYTES:raise ValueError('Table Output exceeds 50 MB output budget')
  return self.stream.write(raw)

def write_table(stream,table,indexes,config,check,rows=None,columns=None):
 fmt,opts=options(config);columns=columns or [table.columns[i] for i in indexes]
 rows=rows if rows is not None else ([row[i] for i in indexes] for row in table.rows)
 def values():
  for i,row in enumerate(rows):
   if i%128==0:check()
   yield list(row)
 if fmt in ('csv','tsv','json','ndjson'):
  out=BudgetStream(stream)
  if fmt in ('csv','tsv'):
   encoding=opts['encoding'];bom={'utf-8':b'\xef\xbb\xbf','utf-16-le':b'\xff\xfe','utf-16-be':b'\xfe\xff'}[encoding]
   if opts['bom']:out.write(bom)
   def emit(row):
    text=io.StringIO(newline='');csv.writer(text,delimiter=config.get('separator','\t' if fmt=='tsv' else ','),lineterminator='\n' if opts['newline']=='LF' else '\r\n',quoting={'minimal':csv.QUOTE_MINIMAL,'all':csv.QUOTE_ALL,'none':csv.QUOTE_NONE}[opts['quoting']],quotechar=opts['quoteChar'],doublequote=opts['escaping']=='double',escapechar=opts['escapeChar'] if opts['escaping']=='escape' else None).writerow(row)
    out.write(text.getvalue().encode(encoding))
   if opts['header']:emit([scalar(c['name'],{'dtype':'string'},opts,fmt) for c in columns])
   for row in values():emit([opts['nullText'] if v is None else scalar(v,c,opts,fmt) for v,c in zip(row,columns)])
  else:
   pretty=opts.get('pretty',False)
   def encode(value):return json.dumps(value,ensure_ascii=opts['ascii'],allow_nan=False,indent=2 if pretty else None,separators=None if pretty else (',',':'))
   schema=fmt=='json' and opts['shape']=='rowsSchema'
   if fmt=='json':out.write((('{"columns":'+encode(columns)+',"rows":[') if schema else '[').encode())
   for i,row in enumerate(values()):
    vals=[scalar(v,c,opts,fmt) for v,c in zip(row,columns)]
    record=vals if schema else {c['name']:v for c,v in zip(columns,vals) if opts['nullPolicy']!='omit' or v is not None}
    out.write(((('\n' if fmt=='ndjson' else '') if i else '') if fmt=='ndjson' else (',' if i else '')).encode())
    out.write(encode(record).encode('utf-8'))
   if fmt=='json':out.write((']}' if schema else ']').encode())
   elif len(table.rows):out.write(b'\n')
 elif fmt=='parquet':
  import pyarrow as pa
  import pyarrow.parquet as pq
  kinds={'string':pa.string(),'int64':pa.int64(),'float64':pa.float64(),'bool':pa.bool_(),'date':pa.date32(),'time':pa.time64('us')}
  fields=[pa.field(c['name'],pa.timestamp('us',tz=c.get('temporal',{}).get('timezone','UTC')) if c['dtype']=='datetime' else kinds[c['dtype']],nullable=True,metadata={b'kwip.column':json.dumps(c,sort_keys=True).encode()}) for c in columns]
  schema=pa.schema(fields,metadata={b'kwip.columns':json.dumps(columns,sort_keys=True).encode()})
  codec=opts['compression'];codec=None if codec=='none' else codec
  with pq.ParquetWriter(stream,schema,compression=codec,compression_level=opts['compressionLevel'],data_page_size=opts['dataPageSize'],write_statistics=opts['statistics']) as writer:
   batch=[];batch_bytes=0
   def flush():
    writer.write_table(pa.Table.from_arrays([pa.array([row[j] for row in batch],type=fields[j].type) for j in range(len(columns))],schema=schema),row_group_size=opts['rowGroupSize'])
    if stream.tell()>MAX_BYTES:raise ValueError('Table Output exceeds 50 MB output budget')
   for row in values():
    batch_bytes+=sum(len(v.encode('utf-8')) if type(v) is str else 8 for v in row)+8*len(row)
    if (len(batch)+1)*len(columns)>1_000_000 or batch_bytes>MAX_BYTES:raise ValueError('Parquet row-group memory budget exceeded; choose a smaller row group')
    batch.append(row)
    if len(batch)>=opts['rowGroupSize']:flush();batch=[];batch_bytes=0
   if batch:flush()
 else:
  from openpyxl import Workbook
  from openpyxl.cell.cell import Cell
  from openpyxl.utils import get_column_letter
  top,left=start_cell(opts['startCell']);count=len(table.rows)+int(opts['header'])
  if top+count-1>1048576 or left+len(columns)-1>16384:raise ValueError('Table exceeds Excel capacity including start cell and header')
  # Bounded cell budget avoids an eager workbook (write-only cannot place start cells).
  if count*len(columns)>1_000_000:raise ValueError('Excel output exceeds one million cell memory budget; use Parquet')
  workbook=Workbook();sheet=workbook.active;sheet.title=opts['sheet'];position=top
  def emit(row,header=False):
   nonlocal position
   for offset,(value,column) in enumerate(zip(row,columns)):
    cell=sheet.cell(position,left+offset)
    if not header and value is not None:
     kind=column['dtype']
     temporal=temporal_value(value,column) if kind in TEMPORAL else None
     # Excel has no timezone type and openpyxl rereads clock values at milliseconds.
     temporal_loss=kind=='datetime' or (kind=='time' and value%1000!=0) or (kind=='date' and temporal<date(1900,3,1))
     loss=(type(value) is int and kind not in TEMPORAL and abs(value)>999_999_999_999_999) or temporal_loss
     if type(value) is float:
      if not math.isfinite(value):raise ValueError('Excel cannot preserve nonfinite numbers')
      loss=loss or float(format(value,'.16g'))!=value
     if loss:
      if opts['precisionPolicy']=='stop':raise ValueError('Excel integer/temporal precision requires explicit Text policy')
      value=temporal.isoformat() if temporal is not None else repr(value) if type(value) is float else str(value)
     elif temporal is not None:value=temporal
    cell.value=value
    if type(value) is str:cell.data_type='s'
    if not header:cell.number_format=opts['dateFormat'] if column['dtype'] in TEMPORAL else opts['numberFormat']
   position+=1
  if opts['header']:emit([c['name'] for c in columns],True)
  for row in values():emit(row)
  if opts['freeze']:sheet.freeze_panes=sheet.cell(top+int(opts['header']),left).coordinate
  if opts['autofilter'] and opts['header'] and count:sheet.auto_filter.ref=f'{get_column_letter(left)}{top}:{get_column_letter(left+len(columns)-1)}{position-1}'
  check();workbook.save(stream);workbook.close()
 check();stream.flush()
 if stream.tell()>MAX_BYTES:raise ValueError('Table Output exceeds 50 MB output budget')
 return fmt,opts
