"""Literal delimited and fixed-width extraction with bounded streaming output.

Quoting is a strict field grammar, independent of Python CSV's one-byte delimiter.
No regex, locale guessing, lossy byte decoding or implicit observation resplitting.
"""
from itertools import zip_longest
from io import StringIO

TYPES=('string','int64','float64','bool')
MAX_CELL_BYTES=10_000_000
MAX_OUTPUT_BYTES=250_000_000

def defaults():
    return dict(mode='delimited',source='',missingSource='preserve',retainSource=False,
        delimiter=',',quoted=True,quote='"',escaping='double',escape='\\',outputMode='columns',
        fields=[dict(name='part',type='string',start=1,width=1)],extraFields='error',
        shortFields='missing',emptyTokens='keep',ordinal=False,ordinalName='token_ordinal',
        measurement='codepoints',shortRecord='missing',trim=False,conversionFailure='error')

def _df():
    if __package__:from . import dataflow
    else:import dataflow
    return dataflow

def validate_config(c):
    df=_df()
    if not isinstance(c,dict) or set(c)!=set(defaults()):raise ValueError('Split Text requires explicit settings.')
    for key,allowed in [('mode',('delimited','fixedWidth')),('missingSource',('preserve','error')),('escaping',('double','escape','none')),('outputMode',('columns','rows')),('extraFields',('error','drop','remainder')),('shortFields',('missing','empty','error')),('emptyTokens',('keep','drop')),('measurement',('codepoints','utf8')),('shortRecord',('missing','error')),('conversionFailure',('error','missing'))]:
        if c[key] not in allowed:raise ValueError('Invalid Split Text '+key+' policy.')
    for key in ('retainSource','quoted','ordinal','trim'):
        if type(c[key]) is not bool:raise ValueError('Split Text checkbox settings must be Boolean.')
    if not isinstance(c['source'],str) or len(c['source'])>200:raise ValueError('Choose a bounded source column name.')
    if not df.names([c['ordinalName']]):raise ValueError('Token ordinal needs a nonblank bounded name.')
    if not isinstance(c['delimiter'],str) or not 1<=len(c['delimiter'])<=128:raise ValueError('Delimiter must contain 1–128 literal code points.')
    if not isinstance(c['quote'],str) or len(c['quote'])!=1 or not isinstance(c['escape'],str) or len(c['escape'])!=1:raise ValueError('Quote and escape must each be one Unicode code point.')
    if c['mode']=='delimited':
        if c['quoted'] and c['quote'] in c['delimiter']:raise ValueError('Delimiter cannot contain the active quote character.')
        if c['escaping']=='escape' and (c['escape'] in c['delimiter'] or (c['quoted'] and c['escape']==c['quote'])):raise ValueError('Escape must differ from quote and every delimiter character.')
    if not isinstance(c['fields'],list) or not 1<=len(c['fields'])<=128:raise ValueError('Choose 1–128 output fields.')
    for f in c['fields']:
        if not isinstance(f,dict) or set(f)!={'name','type','start','width'} or not df.names([f['name']]) or f['type'] not in TYPES or type(f['start']) is not int or type(f['width']) is not int or not 1<=f['start']<=MAX_CELL_BYTES or not 1<=f['width']<=MAX_CELL_BYTES:raise ValueError('Each field needs a distinct name, scalar type, positive 1-based start and width.')
    active=c['fields'][:1] if c['mode']=='delimited' and c['outputMode']=='rows' else c['fields']
    if len({f['name'] for f in active})!=len(active):raise ValueError('Split Text output names must be unique.')

def source_digest(table,c):
    return _df().digest(['kwip.splitText.rows.v1',table.source_digest,c]) if c['mode']=='delimited' and c['outputMode']=='rows' else table.source_digest

def tokens(text,c,check=lambda:None):
    """Yield parsed fields without building an unbounded token list."""
    delimiter=c['delimiter'];quote=c['quote'];escape=c['escape'];i=0;n=len(text)
    while True:
        check();parts=StringIO();quoted=c['quoted'] and i<n and text[i]==quote
        if quoted:i+=1
        closed=not quoted
        while i<n:
            if i%4096==0:check()
            if c['escaping']=='escape' and text[i]==escape:
                if i+1==n:raise ValueError('Trailing escape character in Split Text source.')
                parts.write(text[i+1]);i+=2;continue
            if quoted and text[i]==quote:
                if c['escaping']=='double' and i+1<n and text[i+1]==quote:parts.write(quote);i+=2;continue
                i+=1;closed=True;break
            if not quoted and text.startswith(delimiter,i):break
            parts.write(text[i]);i+=1
        if not closed:raise ValueError('Unclosed quoted field in Split Text source.')
        if quoted and i<n and not text.startswith(delimiter,i):raise ValueError('Unexpected text after closing quote in Split Text source.')
        yield parts.getvalue()
        if i==n:return
        i+=len(delimiter)

def _plan(table,c,check):
    df=_df();validate_config(c);check()
    index=table.index(c['source'])
    if table.columns[index]['dtype']!='string':raise ValueError('Split Text source must be a text column.')
    expands=c['mode']=='delimited' and c['outputMode']=='rows'
    if expands and (table.role!='unassigned' or table.partition_digest or table.fitted_states):raise ValueError('Split Text Rows belongs before Partition and fitted preprocessing; existing isolation cannot be discarded.')
    keep=[i for i in range(len(table.columns)) if c['retainSource'] or i!=index]
    fields=c['fields'][:1] if expands else c['fields']
    columns=[table.columns[i] for i in keep]+[dict(name=f['name'],dtype=f['type'],nullable=True) for f in fields]
    if expands and c['ordinal']:columns.append(dict(name=c['ordinalName'],dtype='int64',nullable=False))
    if not df.names([col['name'] for col in columns]):raise ValueError('Split Text output names collide with retained columns or exceed the 128-column schema limit.')
    def convert(value,f):
        if value is None:return None
        try:
            if isinstance(value,bytes):value=value.decode('utf-8',errors='strict')
            if c['trim']:value=value.strip()
            return df.cast_value(value,f['type'])
        except (ValueError,UnicodeError) as e:
            if c['conversionFailure']=='missing':return None
            raise ValueError('Split Text conversion for '+f['name']+': '+str(e)) from e
    def pairs():
        total=0
        for row,rid in df.row_pairs(table):
            check();text=row[index]
            if text is not None and not isinstance(text,str):raise ValueError('Split Text source violates its text schema.')
            if text is None and c['missingSource']=='error':raise ValueError('Split Text encountered missing source text.')
            if text is not None and len(text.encode('utf-8'))>MAX_CELL_BYTES:raise ValueError('Split Text source exceeds its 10 MB string budget.')
            prefix=tuple(row[i] for i in keep)
            if expands:
                values=iter([None]) if text is None else tokens(text,c,check)
                for ordinal,value in enumerate(values,1):
                    check()
                    if value=='' and c['emptyTokens']=='drop':continue
                    out=prefix+(convert(value,fields[0]),)+((ordinal,) if c['ordinal'] else ())
                    total+=sum(len(v.encode('utf-8')) for v in out if isinstance(v,str))
                    if total>MAX_OUTPUT_BYTES:raise ValueError('Split Text exceeds its 250 MB output text budget.')
                    yield out,'split:'+df.digest(['kwip.splitText.token.v1',table.source_digest,rid,ordinal])
                continue
            if text is None:values=[None]*len(fields)
            elif c['mode']=='fixedWidth':
                record=text.encode('utf-8') if c['measurement']=='utf8' else text;values=[]
                for f in fields:
                    start=f['start']-1;end=start+f['width']
                    if end>len(record):
                        if c['shortRecord']=='error':raise ValueError('Split Text encountered a short fixed-width record.')
                        values.append(None)
                    else:values.append(record[start:end])
            else:
                values=[];parts=tokens(text,c,check);remainder=None
                for value in parts:
                    if len(values)<len(fields):values.append(value)
                    elif c['extraFields']=='error':raise ValueError('Split Text encountered extra delimited fields.')
                    elif c['extraFields']=='remainder':
                        if remainder is None:remainder=StringIO();remainder.write(values[-1])
                        remainder.write(c['delimiter']);remainder.write(value)
                if remainder is not None:values[-1]=remainder.getvalue()
                if len(values)<len(fields):
                    if c['shortFields']=='error':raise ValueError('Split Text encountered short delimited fields.')
                    values+=['' if c['shortFields']=='empty' else None]*(len(fields)-len(values))
            out=prefix+tuple(convert(v,f) for v,f in zip(values,fields))
            total+=sum(len(v.encode('utf-8')) for v in out if isinstance(v,str))
            if total>MAX_OUTPUT_BYTES:raise ValueError('Split Text exceeds its 250 MB output text budget.')
            yield out,rid
    return tuple(columns),pairs(),expands

def apply(table,c,check=lambda:None):
    df=_df();columns,pairs,expands=_plan(table,c,check)
    base=df.Table(columns,(),(),source_digest(table,c),diagnostics=table.diagnostics) if expands else table
    return df.streaming_rows(base,pairs,columns,check)

def verify_output(table,c,output,check=lambda:None):
    columns,pairs,expands=_plan(table,c,check)
    expected_lineage=('unassigned','',()) if expands else (table.role,table.partition_digest,table.fitted_states)
    if output.columns!=columns or output.source_digest!=source_digest(table,c) or (output.role,output.partition_digest,output.fitted_states)!=expected_lineage or output.diagnostics!=table.diagnostics:raise ValueError('Cached Split Text schema or lineage mismatch.')
    for expected,actual in zip_longest(pairs,_df().row_pairs(output)):
        check()
        if expected!=actual:raise ValueError('Cached Split Text values or observation identities mismatch.')

# Registry authoring input. Runtime semantic validation remains authoritative.
def config_schema():
    d=defaults();properties={}
    enums=dict(mode=['delimited','fixedWidth'],missingSource=['preserve','error'],escaping=['double','escape','none'],outputMode=['columns','rows'],extraFields=['error','drop','remainder'],shortFields=['missing','empty','error'],emptyTokens=['keep','drop'],measurement=['codepoints','utf8'],shortRecord=['missing','error'],conversionFailure=['error','missing'])
    for key,value in d.items():
        if key in enums:properties[key]={'type':'string','enum':enums[key]}
        elif type(value) is bool:properties[key]={'type':'boolean'}
        elif isinstance(value,str):properties[key]={'type':'string','maxLength':128 if key=='delimiter' else 1 if key in ('quote','escape') else 200,'minLength':0 if key=='source' else 1}
    properties['fields']={'type':'array','minItems':1,'maxItems':128,'items':{'type':'object','additionalProperties':False,'required':['name','type','start','width'],'properties':{'name':{'type':'string','minLength':1,'maxLength':200},'type':{'type':'string','enum':list(TYPES)},'start':{'type':'integer','minimum':1,'maximum':MAX_CELL_BYTES},'width':{'type':'integer','minimum':1,'maximum':MAX_CELL_BYTES}}}}
    return {'type':'object','additionalProperties':False,'required':list(d),'properties':properties}
