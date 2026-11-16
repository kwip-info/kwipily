"""Pure scalar preprocessing rules and bounded portable inference; no fitting or source IO."""
import json
import math
import unicodedata
if __package__:
    from . import dataflow_temporal as temporal
else:
    import dataflow_temporal as temporal


CLEANUP_DEFAULTS = dict(unicodeNormalization="none", removeControls=False, removePunctuation=False, removeDigits=False, whitespace="none", case="preserve", emptyAfterCleanup="keep")

def text_cleanup_enabled(rule):
    return any(rule.get(k, v) != v for k, v in CLEANUP_DEFAULTS.items())

def clean_text(value, rule):
    if type(value) is not str or not text_cleanup_enabled(rule): return value
    normalization=rule.get("unicodeNormalization", "none")
    if normalization != "none": value=unicodedata.normalize(normalization, value)
    value="".join(c for c in value if not (rule.get("removeControls",False) and unicodedata.category(c)=="Cc" or rule.get("removePunctuation",False) and unicodedata.category(c).startswith("P") or rule.get("removeDigits",False) and unicodedata.category(c)=="Nd"))
    whitespace=rule.get("whitespace","none")
    if whitespace=="trim": value=value.strip()
    elif whitespace=="collapse": value=" ".join(value.split())
    case=rule.get("case","preserve")
    if case=="lower": value=value.lower()
    elif case=="upper": value=value.upper()
    return None if value=="" and rule.get("emptyAfterCleanup","keep")=="missing" else value

def rule_defaults(column):
    return dict(column=column,missing='keep',value=0,emptyPolicy='error',emptyValue=0,
                tiePolicy='error',encoding='none',vocabularyOrder='lexicographic',
                unknownPolicy='error',maxCategories=32,scaling='none',center=True,unitVariance=True,**CLEANUP_DEFAULTS)


def numeric(v):
    try:return type(v) in (int,float) and math.isfinite(v)
    except OverflowError:return False

def typed(v,t):
    return (type(v) is str if t=='string' else type(v) is bool if t=='bool' else type(v) is int and -2**63<=v<2**63 if t=='int64' else numeric(v) if t=='float64' else False)


def validate_config(c):
    if type(c) is not dict or set(c)!={'mode','columns'} or c['mode'] not in ('table','partitions') or type(c['columns']) is not list or len(c['columns'])>128:raise ValueError('Choose fixed Table rules or Training/Validation/Test preprocessing, with at most 128 column rules.')
    names=[]
    for r in c['columns']:
        if type(r) is not dict or not set(rule_defaults(''))-set(CLEANUP_DEFAULTS)<=set(r)<=set(rule_defaults('')):raise ValueError('Invalid preprocessing column controls.')
        name=r['column'];names.append(name)
        if r.get('unicodeNormalization','none') not in ('none','NFC','NFKC') or r.get('whitespace','none') not in ('none','trim','collapse') or r.get('case','preserve') not in ('preserve','lower','upper') or r.get('emptyAfterCleanup','keep') not in ('keep','missing') or any(type(r.get(k,False)) is not bool for k in ('removeControls','removePunctuation','removeDigits')):raise ValueError('Invalid Text cleanup controls.')
        if c['mode']!='table' and text_cleanup_enabled(r):raise ValueError('Text cleanup requires fixed Table mode before Partition.')
        if type(name) is not str or not name.strip() or len(name)>200 or '\0' in name:raise ValueError('Choose exact nonblank column names.')
        if r['missing'] not in ('keep','error','drop','constant','mean','median','mostFrequent') or r['emptyPolicy'] not in ('error','constant') or r['tiePolicy'] not in ('smallest','error') or r['encoding'] not in ('none','oneHot') or r['vocabularyOrder'] not in ('lexicographic','firstSeen') or r['unknownPolicy'] not in ('error','ignore') or r['scaling'] not in ('none','standard'):raise ValueError('Invalid preprocessing policy.')
        if type(r['maxCategories']) is not int or not 1<=r['maxCategories']<=128 or type(r['center']) is not bool or type(r['unitVariance']) is not bool:raise ValueError('Use 1–128 categories and explicit scaling switches.')
        if any(type(r[k]) not in (str,int,float,bool) or (type(r[k]) is float and not math.isfinite(r[k])) for k in ('value','emptyValue')):raise ValueError('Replacement constants must be finite scalar values.')
        if c['mode']=='table' and (r['missing'] in ('mean','median','mostFrequent') or r['encoding']!='none' or r['scaling']!='none'):raise ValueError('Learned imputation, encoding and scaling require Training/Validation/Test mode.')
        if r['encoding']!='none' and r['scaling']!='none':raise ValueError('Choose encoding or scaling for a source column, not both.')
    if len(set(names))!=len(names):raise ValueError('Configure each preprocessing column only once.')
    return c


def validate_schema(schema,c):
    validate_config(c)
    if not isinstance(schema,list) or not 1<=len(schema)<=128:raise ValueError('Invalid preprocessing input schema.')
    if any(type(x) is not dict or set(x)-{'name','dtype','nullable','displayLabel','temporal'} or type(x.get('name')) is not str or not x['name'].strip() or len(x['name'])>200 or x.get('dtype') not in ('float64','int64','string','bool','image','audio','tensor','int64Sequence','date','datetime','time') or type(x.get('nullable')) is not bool or ('displayLabel' in x and (type(x['displayLabel']) is not str or len(x['displayLabel'])>400)) for x in schema):raise ValueError('Invalid preprocessing typed column schema.')
    for col in schema:temporal.validate_column(col)
    names=[x['name'] for x in schema]
    if len(set(names))!=len(names):raise ValueError('Duplicate preprocessing schema names.')
    for r in c['columns']:
        if r['column'] not in names:raise ValueError('Preprocessing column is absent: '+r['column'])
        col=schema[names.index(r['column'])];t=col['dtype']
        if t!='string' and text_cleanup_enabled(r):raise ValueError('Text cleanup requires a Text column: '+r['column'])
        if t in ('image','audio','tensor','int64Sequence','date','datetime','time') and (r['missing'] not in ('keep','error','drop') or r['encoding']!='none' or r['scaling']!='none'):raise ValueError('Image and temporal columns support Preserve, Drop row or Stop run only.')
        if r['missing'] in ('mean','median') or r['scaling']=='standard':
            if t not in ('float64','int64'):raise ValueError('Statistical numeric preprocessing requires a numeric column: '+r['column'])
        if r['encoding']=='oneHot' and t!='string':raise ValueError('One-hot encoding requires a Text column.')
        if r['missing']=='constant' and not typed(r['value'],t):raise ValueError('Replacement constant differs from column type: '+r['column'])
        if r['emptyPolicy']=='constant' and r['missing'] in ('mean','median','mostFrequent'):
            if not typed(r['emptyValue'], 'float64' if r['missing'] in ('mean','median') else t):raise ValueError('All-missing constant differs from column type.')
    return schema


def output_schema(schema,config,rules):
    by_name={r['column']:r for r in rules};out=[];original={c['name'] for c in schema}
    for col in schema:
        r=by_name.get(col['name'])
        if r is None:out.append(dict(col));continue
        if r['encoding']=='oneHot':
            for value in r['vocabulary']:
                name='onehot:'+json.dumps([col['name'],value],ensure_ascii=False,separators=(',',':'))
                if len(name)>200 or name in original:raise ValueError('Encoded name exceeds bounds or collides with an input column.')
                out.append(dict(name=name,displayLabel=f"{col['name']} = {value}",dtype='float64',nullable=False))
        else:
            out.append({**col,'dtype':'float64' if r['missing'] in ('mean','median') or r['scaling']=='standard' else col['dtype'],
                        'nullable':(col['nullable'] or r.get('emptyAfterCleanup','keep')=='missing') if r['missing']=='keep' else False})
    if len(out)>128 or len({c['name'] for c in out})!=len(out):raise ValueError('Preprocessing output exceeds 128 unique columns.')
    return out


def validate(params):
    fields={'schemaVersion','kind','operator','columns','inputSchema','outputSchema','sourceContentDigest','configuration','rules'}
    if type(params) is not dict or set(params)!=fields or type(params['schemaVersion']) is not int or params['schemaVersion']!=1 or params['kind']!='portableFittedParameters' or params['operator']!='kwip.data.preprocess':raise ValueError('Invalid portable Preprocess parameters.')
    if type(params['sourceContentDigest']) is not str or len(params['sourceContentDigest'])!=64 or any(c not in '0123456789abcdef' for c in params['sourceContentDigest']):raise ValueError('Invalid preprocessing parameter digest.')
    c=validate_config(params['configuration']);schema=validate_schema(params['inputSchema'],c)
    if params['columns']!=[r['column'] for r in c['columns']] or type(params['rules']) is not list or len(params['rules'])!=len(c['columns']):raise ValueError('Preprocessing parameter dimensions differ.')
    for source,r in zip(c['columns'],params['rules']):
        if type(r) is not dict or set(r)!=set(source)|{'fill','vocabulary','mean','scale'} or any(r[k]!=v for k,v in source.items()):raise ValueError('Learned preprocessing differs from configured rules.')
        t=next(col['dtype'] for col in schema if col['name']==r['column'])
        if r['missing'] not in ('keep','error','drop') and not typed(r['fill'],'float64' if r['missing'] in ('mean','median') else t):raise ValueError('Invalid learned replacement type.')
        vocab=r['vocabulary']
        if type(vocab) is not list or len(vocab)>r['maxCategories'] or any(type(x) is not str for x in vocab) or len(set(vocab))!=len(vocab) or (r['encoding']=='oneHot' and not vocab) or (r['encoding']=='none' and vocab):raise ValueError('Invalid learned vocabulary.')
        if r['vocabularyOrder']=='lexicographic' and vocab!=sorted(vocab):raise ValueError('Vocabulary order differs.')
        if not numeric(r['mean']) or not numeric(r['scale']) or r['scale']<=0:raise ValueError('Invalid learned scaling parameters.')
    if params['outputSchema']!=output_schema(schema,c,params['rules']):raise ValueError('Preprocessing output schema differs from learned parameters.')
    return params


def transform_row(row,schema,rules,counts=None):
    if len(row)!=len(schema):raise ValueError('Preprocessing row width differs from schema.')
    by_name={r['column']:r for r in rules}
    # Fixed cleanup precedes missing policy. All exclusions precede other transformations.
    row=tuple(clean_text(value,by_name.get(col['name'],{})) if col['dtype']=='string' else value for value,col in zip(row,schema))
    if any(value is None and by_name.get(col['name'],{}).get('missing')=='drop' for value,col in zip(row,schema)):return None
    out=[]
    for value,col in zip(row,schema):
        r=by_name.get(col['name'])
        if r is None:out.append(value);continue
        if value is None:
            if r['missing']=='error':raise ValueError('Missing value in column '+col['name'])
            if r['missing'] not in ('keep','drop'):
                value=r['fill']
                if counts is not None:counts[('missingValuesImputed',col['name'])]=counts.get(('missingValuesImputed',col['name']),0)+1
        if value is not None and col['dtype'] in temporal.DTYPES and not temporal.valid_value(value,col['dtype']):raise ValueError('Preprocessing value violates temporal type: '+col['name'])
        if value is not None and col['dtype'] not in ('image','audio','tensor','int64Sequence','date','datetime','time') and not typed(value,col['dtype'] if r['missing'] not in ('mean','median') else 'float64'):raise ValueError('Preprocessing value violates column type: '+col['name'])
        if r['encoding']=='oneHot':
            if type(value) is not str:raise ValueError('Resolve missing categories before encoding: '+col['name'])
            if value not in r['vocabulary']:
                if r['unknownPolicy']=='error':raise ValueError('Unknown category in column '+col['name'])
                if counts is not None:counts[('unknownCategoryIgnored',col['name'])]=counts.get(('unknownCategoryIgnored',col['name']),0)+1
            out.extend(float(value==x) for x in r['vocabulary'])
        elif r['scaling']=='standard':
            if not numeric(value):raise ValueError('Scaling requires finite nonmissing values: '+col['name'])
            v=(value-r['mean'])/r['scale']
            if not math.isfinite(v):raise ValueError('Scaling numerical overflow.')
            out.append(v)
        else:out.append(float(value) if value is not None and r['missing'] in ('mean','median') else float(value) if value is not None and col['dtype']=='float64' else value)
    return tuple(out)


def apply(rows,schema,params):
    validate(params)
    if schema!=params['inputSchema'] or not isinstance(rows,(list,tuple)) or len(rows)*len(schema)>5_000_000:raise ValueError('Portable preprocessing input exceeds bounded schema/rows.')
    counts={};output=[];output_bytes=0
    for row in rows:
        value=transform_row(row,schema,params['rules'],counts)
        if value is not None:
            output_bytes+=len(json.dumps(value,ensure_ascii=False,allow_nan=False).encode())
            if output_bytes>100_000_000:raise ValueError('Portable preprocessing output exceeds 100 MB byte budget.')
            output.append(list(value))
    diagnostics=[dict(code=code,column=column,count=count,stateDigest=params['sourceContentDigest']) for (code,column),count in counts.items()]
    return output,params['outputSchema'],diagnostics
