"""Regex table operators run in a separately supervised, disposable Python process.

Pattern <=512 chars, each input <=16384 chars, input <=16 MB, output <=32 MB,
5-second total wall time; cancellation polls every 20 ms. Worker is always reaped.
Python re syntax, ASCII-independent Unicode by default, no custom callbacks.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

TIMEOUT_SECONDS=5.0
MAX_INPUT_BYTES=16_000_000
MAX_OUTPUT_BYTES=32_000_000


def _worker(payload):
 import re
 op=payload['operator'];c=payload['config'];values=payload['values']
 flags=sum({'ignoreCase':re.I,'multiline':re.M,'dotAll':re.S}[f] for f in c['flags'])
 try:regex=re.compile(c['pattern'],flags)
 except (re.error,RecursionError) as e:raise ValueError(f'Regex pattern: {e}; correct the indicated pattern position') from e
 groups=[]
 if op=='regexExtract':
  for item in c['groups']:
   raw=item['group'];group=int(raw) if raw.isdigit() else raw
   if isinstance(group,int) and not 0<=group<=regex.groups or isinstance(group,str) and group not in regex.groupindex:raise ValueError(f'Regex capture group {raw!r} does not exist in the pattern')
   groups.append(group)
 if op=='regexReplace':
  # Compile the replacement even when there are no matching input rows.
  try:regex.sub(c['replacement'],'')
  except (re.error,IndexError) as e:raise ValueError(f'Regex replacement group reference: {e}') from e
 result=[];output_size=0
 for i,value in enumerate(values):
  if value is None:
   if c['nullPolicy']=='error':raise ValueError(f'Regex input row {i+1} is null; choose an explicit missing-value policy')
   transformed=False if op=='regexMatch' else [None]*len(groups) if op=='regexExtract' else None
  elif op=='regexMatch':transformed=(regex.fullmatch(value) if c['matchMode']=='full' else regex.search(value)) is not None
  elif op=='regexExtract':
   match=regex.search(value)
   if match is None and c['noMatchPolicy']=='error':raise ValueError(f'Regex row {i+1} has no match; revise the pattern or choose null outputs')
   transformed=[match.group(g) if match else None for g in groups]
  else:transformed=regex.sub(c['replacement'],value,count=1 if c['replaceMode']=='first' else 0)
  output_size+=len(json.dumps(transformed,ensure_ascii=True).encode())
  if output_size>MAX_OUTPUT_BYTES or isinstance(transformed,str) and len(transformed)>16384:raise ValueError('Regex result exceeds output budget; shorten the replacement or reduce input')
  result.append(transformed)
 return result


def supervised(op,c,values,check_cancelled,*,timeout=TIMEOUT_SECONDS):
 for i,v in enumerate(values):
  if i%128==0:check_cancelled()
  if v is not None and (not isinstance(v,str) or len(v)>16384):raise ValueError('Regex requires text cells of at most 16384 characters; add Cast or shorten input')
 payload=json.dumps({'operator':op,'config':c,'values':values},allow_nan=False).encode()
 if len(payload)>MAX_INPUT_BYTES:raise ValueError('Regex input exceeds 16 MB budget')
 check_cancelled()
 with tempfile.TemporaryDirectory(prefix='kwip-regex-') as folder:
  source=Path(folder)/'input.json';source.write_bytes(payload)
  output=Path(folder)/'output.json';process=None
  try:
   with output.open('wb') as stream:
    process=subprocess.Popen([sys.executable,'-I',str(Path(__file__).resolve()),str(source)],stdout=stream,stderr=subprocess.DEVNULL)
    deadline=time.monotonic()+timeout
    while process.poll() is None:
     check_cancelled()
     if time.monotonic()>deadline:raise ValueError('Regex exceeded five-second execution budget; simplify the pattern or reduce input. No result accepted.')
     if output.stat().st_size>MAX_OUTPUT_BYTES+1_000_000:raise ValueError('Regex worker exceeded bounded output')
     time.sleep(.02)
   check_cancelled()
   if process.returncode!=0:raise ValueError('Regex worker exited without a result; simplify the pattern or reduce input')
   if output.stat().st_size>MAX_OUTPUT_BYTES+1_000_000:raise ValueError('Regex worker exceeded bounded output')
   answer=json.loads(output.read_bytes())
   if 'error' in answer:raise ValueError(answer['error'])
   result=answer['values']
   if not isinstance(result,list) or len(result)!=len(values):raise ValueError('Regex worker result shape mismatch')
   return result
  finally:
   if process is not None:
    if process.poll() is None:process.kill()
    process.wait()


def apply(table,op,c,check_cancelled):
 raise ValueError('Retired regex table execution is unavailable; use current Filter or Formula.')


if __name__=='__main__':
 try:
  try:
   import resource
   resource.setrlimit(resource.RLIMIT_CPU,(6,6))
   resource.setrlimit(resource.RLIMIT_AS,(512_000_000,512_000_000))
  except (ImportError,ValueError,OSError):pass
  raw=Path(sys.argv[1]).read_bytes()
  if len(raw)>MAX_INPUT_BYTES:raise ValueError('Regex input exceeds budget')
  result={'values':_worker(json.loads(raw))}
 except Exception as error:result={'error':str(error)[:500]}
 print(json.dumps(result,allow_nan=False),flush=True)
