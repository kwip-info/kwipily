"""Shared bounded delimited-text parser. No source contents enter diagnostics."""
import csv
import codecs
import hashlib
import io
import json
from pathlib import Path

DEFAULTS=dict(separator=',',quoteChar='"',escapeMode='double',escapeChar='\\',encoding='utf-8-sig',recordEndings='auto',allowQuotedNewlines=True,headerMode='firstRecord',skipRecords=0,skipBlankRecords=False,skipInitialSpace=False,emptyField='null',nullMarkers=[])

class CSVParseError(ValueError):
    def __init__(self,code,message,record=0,line=0):
        self.code,self.record,self.line=code,record,line
        super().__init__((f'CSV record {record}, line {line}: ' if record else 'CSV: ')+message)
    def diagnostic(self):return dict(code=self.code,record=self.record,line=self.line,message=str(self))


def normalize_options(options=None,separator=','):
    if options is None:options={}
    if not isinstance(options,dict) or set(options)-set(DEFAULTS):raise ValueError('Unknown CSV parsing option')
    c={**DEFAULTS,'separator':separator,**options}
    if not isinstance(c['separator'],str) or len(c['separator'])!=1 or c['separator'] in '\r\n\0' or (isinstance(c['separator'],str) and any(0xD800<=ord(x)<=0xDFFF for x in c['separator'])):raise ValueError('CSV delimiter must be one character other than a newline or NUL')
    for key,values in dict(escapeMode=['double','backslash','none','custom'],encoding=['utf-8-sig','utf-8','utf-16','cp1252'],recordEndings=['auto','lf','crlf','cr'],headerMode=['firstRecord','generated'],emptyField=['text','null']).items():
        if c[key] not in values:raise ValueError('Invalid CSV '+key)
    if not isinstance(c['quoteChar'],str) or len(c['quoteChar'])>1 or c['quoteChar'] in ('\r','\n','\0') or (isinstance(c['quoteChar'],str) and any(0xD800<=ord(x)<=0xDFFF for x in c['quoteChar'])):raise ValueError('CSV quote must be one character or empty')
    if not isinstance(c['escapeChar'],str) or len(c['escapeChar'])!=1 or c['escapeChar'] in '\r\n\0' or any(0xD800<=ord(x)<=0xDFFF for x in c['escapeChar']):raise ValueError('CSV escape must be one character other than a newline or NUL')
    if c['escapeMode']=='custom' and c['escapeChar'] in (c['separator'],c['quoteChar']):raise ValueError('CSV escape must differ from delimiter and quote')
    if c['quoteChar']==c['separator'] or (c['escapeMode']=='backslash' and '\\' in (c['separator'],c['quoteChar'])):raise ValueError('CSV delimiter must differ from quote and escape characters')
    for key in ('allowQuotedNewlines','skipBlankRecords','skipInitialSpace'):
        if type(c[key]) is not bool:raise ValueError('Invalid CSV '+key)
    if type(c['skipRecords']) is not int or not 0<=c['skipRecords']<=100000:raise ValueError('CSV skipped records must be between 0 and 100000')
    if not isinstance(c['nullMarkers'],list) or len(c['nullMarkers'])>32 or any(not isinstance(x,str) or not x or len(x)>200 or '\0' in x for x in c['nullMarkers']):raise ValueError('CSV null markers require at most 32 nonempty strings of 200 characters')
    if len(set(c['nullMarkers']))!=len(c['nullMarkers']):raise ValueError('CSV null markers must be unique')
    c['nullMarkers']=list(c['nullMarkers'])
    return c


def parse_options_json(value):
    def pairs(items):
        result={}
        for key,item in items:
            if key in result:raise ValueError('Duplicate CSV parsing option')
            result[key]=item
        return result
    options=json.loads(value,object_pairs_hook=pairs)
    if not isinstance(options,dict):raise ValueError('CSV options must be an object')
    if options.get('parserProfile')=='polars':
        if __package__:from .polars_csv_configuration import normalize_options as native_options
        else:from polars_csv_configuration import normalize_options as native_options
        return native_options(options)
    return normalize_options(options)


def parse_csv(path,options=None,separator=',',*,check_cancelled=lambda:None,max_bytes=10_000_000,max_rows=100_000,max_columns=128,preview=False):
    c=normalize_options(options,separator);check_cancelled();raw=bytearray()
    if preview:max_bytes=1_048_576
    read_limit=max_bytes if preview else max_bytes+1
    with Path(path).open('rb') as stream:
        while len(raw)<read_limit:
            check_cancelled();block=stream.read(min(262144,read_limit-len(raw)))
            if not block:break
            raw.extend(block)
    if len(raw)>max_bytes:raise CSVParseError('byteLimit','exceeds 10 MB source limit')
    at_boundary=preview and len(raw)==max_bytes
    try:text=codecs.getincrementaldecoder(c['encoding'])().decode(raw,final=not at_boundary)
    except UnicodeError as e:raise CSVParseError('encoding','cannot decode with the selected encoding; choose the file’s encoding') from e
    # csv.reader consumes physical lines; preserve their endings for embedded newlines.
    lines=io.StringIO(text,newline='');expected={'lf':'\n','crlf':'\r\n','cr':'\r'}.get(c['recordEndings'])
    physical=0;last_ending=''
    def checked_lines():
        nonlocal physical,last_ending
        for line in lines:
            physical+=1
            if physical%256==0:check_cancelled()
            ending='\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else '\r' if line.endswith('\r') else ''
            last_ending=ending
            if at_boundary and not ending:
                raise CSVParseError('sampleBudget','preview reached its 1 MiB byte budget; the final record was not inspected. Reduce skipped records or field sizes')
            yield line
        if at_boundary:
            raise CSVParseError('sampleBudget','preview reached its 1 MiB byte budget; later records were not inspected')
    reader=csv.reader(checked_lines(),delimiter=c['separator'],quotechar=c['quoteChar'] or None,quoting=csv.QUOTE_MINIMAL if c['quoteChar'] else csv.QUOTE_NONE,doublequote=c['escapeMode']=='double',escapechar=('\\' if c['escapeMode']=='backslash' else c['escapeChar'] if c['escapeMode']=='custom' else None),skipinitialspace=c['skipInitialSpace'],strict=True)
    record=0;header=None;rows=[];nulls=set(c['nullMarkers'])
    try:
        while True:
            start=reader.line_num+1
            try:row=next(reader)
            except StopIteration:break
            record+=1
            if expected and last_ending and last_ending!=expected:raise CSVParseError('recordEnding','record ending differs from the selected setting; choose Auto or the matching ending',record,reader.line_num)
            if record%256==0:check_cancelled()
            if not c['allowQuotedNewlines'] and reader.line_num>start:raise CSVParseError('quotedNewline','multiline fields are disabled; enable quoted newlines or correct this record',record,start)
            if record<=c['skipRecords']:continue
            if not row and c['skipBlankRecords']:continue
            if header is None:
                if c['headerMode']=='firstRecord':
                    header=row
                    if not 1<=len(header)<=max_columns or any(not x.strip() or len(x)>200 for x in header) or len(set(header))!=len(header):raise CSVParseError('header','requires 1–128 unique bounded header names; correct the header or use generated names',record,start)
                    continue
                header=[f'column_{i+1}' for i in range(len(row))]
                if not 1<=len(header)<=max_columns:raise CSVParseError('columnLimit','requires 1–128 columns',record,start)
            if len(row)!=len(header):raise CSVParseError('rowWidth',f'row width mismatch: has {len(row)} fields; expected {len(header)}. Check delimiter, quoting or blank-record settings',record,start)
            rows.append(tuple(None if x in nulls or (x=='' and c['emptyField']=='null') else x for x in row))
            if preview and len(rows)==8:break
            if len(rows)>max_rows:raise CSVParseError('rowLimit','exceeds row bound of 100,000 observations',record,start)
    except csv.Error as e:raise CSVParseError('syntax','invalid quoted/escaped record or field exceeds parser limit; check quote and escape settings',record+1,max(1,reader.line_num)) from e
    check_cancelled()
    if at_boundary and len(rows)<8:raise CSVParseError('sampleBudget','preview reached its 1 MiB byte budget before collecting 8 rows; reduce skipped records or field sizes')
    if not rows:raise CSVParseError('empty','has no observations after header and skipped records')
    return header,rows,hashlib.sha256(raw).hexdigest(),c

class _CSVByteStream(io.RawIOBase):
    """Incremental hashing source; TextIOWrapper never receives complete file bytes."""
    def __init__(self, stream, maximum, check):
        self.stream, self.maximum, self.check = stream, maximum, check
        self.count = 0; self.hasher = hashlib.sha256()
    def readable(self): return True
    def readinto(self, buffer):
        self.check()
        block = self.stream.read(min(len(buffer), self.maximum + 1 - self.count))
        self.count += len(block)
        if self.count > self.maximum: raise CSVParseError('byteLimit', 'exceeds source byte budget')
        self.hasher.update(block); buffer[:len(block)] = block
        return len(block)

class CSVStream:
    """Shared CSV dialect, streamed records and full-source digest after exhaustion."""
    def __init__(self, path, options=None, separator=',', *, check_cancelled=lambda:None,
                 max_bytes=10_000_000, max_columns=128):
        self.options = normalize_options(options, separator)
        self.check = check_cancelled; self.max_columns = max_columns
        self.file = Path(path).open('rb')
        self.raw = _CSVByteStream(self.file, max_bytes, check_cancelled)
        self.text = io.TextIOWrapper(io.BufferedReader(self.raw), encoding=self.options['encoding'], newline='')
        self.records = self._records()
        try: self.header = next(self.records)
        except BaseException: self.close(); raise
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def close(self): self.text.close(); self.file.close()
    def __iter__(self): return self.records
    @property
    def source_sha256(self): return self.raw.hasher.hexdigest()
    def _records(self):
        c = self.options; physical = 0; last_ending = ''; header = None; count = 0; record = 0
        expected = {'lf':'\n','crlf':'\r\n','cr':'\r'}.get(c['recordEndings']); nulls = set(c['nullMarkers'])
        def lines():
            nonlocal physical, last_ending
            while True:
                if physical % 256 == 0:self.check()
                # Bounds a physical line as well as the csv module's field budget.
                line = self.text.readline(1_048_577)
                if not line: return
                if len(line) > 1_048_576: raise CSVParseError('recordBudget', 'physical line exceeds 1 MiB parsing budget', record+1, physical+1)
                physical += 1
                last_ending = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else '\r' if line.endswith('\r') else ''
                yield line
        reader = csv.reader(lines(), delimiter=c['separator'], quotechar=c['quoteChar'] or None,
            quoting=csv.QUOTE_MINIMAL if c['quoteChar'] else csv.QUOTE_NONE, doublequote=c['escapeMode']=='double',
            escapechar='\\' if c['escapeMode']=='backslash' else c['escapeChar'] if c['escapeMode']=='custom' else None,
            skipinitialspace=c['skipInitialSpace'], strict=True)
        try:
            while True:
                start = reader.line_num+1
                try: row = next(reader)
                except StopIteration: break
                record += 1
                if record % 256 == 0: self.check()
                if expected and last_ending and expected != last_ending: raise CSVParseError('recordEnding', 'record ending differs from the selected setting; choose Auto or the matching ending', record, reader.line_num)
                if not c['allowQuotedNewlines'] and reader.line_num > start: raise CSVParseError('quotedNewline', 'multiline fields are disabled; enable quoted newlines or correct this record', record, start)
                if record <= c['skipRecords'] or (not row and c['skipBlankRecords']): continue
                if header is None:
                    header = row if c['headerMode']=='firstRecord' else [f'column_{i+1}' for i in range(len(row))]
                    if not 1 <= len(header) <= self.max_columns or any(not x.strip() or len(x)>200 for x in header) or len(set(header)) != len(header): raise CSVParseError('header', 'requires 1–128 unique bounded header names; correct the header or use generated names', record, start)
                    yield header
                    if c['headerMode']=='firstRecord': continue
                if len(row) != len(header): raise CSVParseError('rowWidth', f'row width mismatch: has {len(row)} fields; expected {len(header)}. Check delimiter, quoting or blank-record settings', record, start)
                count += 1
                yield tuple(None if x in nulls or (x=='' and c['emptyField']=='null') else x for x in row)
        except UnicodeError as error: raise CSVParseError('encoding', 'cannot decode with the selected encoding; choose the file’s encoding') from error
        except csv.Error as error: raise CSVParseError('syntax', 'invalid quoted/escaped record or field exceeds parser limit; check quote and escape settings', record+1, max(1,reader.line_num)) from error
        self.check()
        if not count: raise CSVParseError('empty', 'has no observations after header and skipped records')
