"""Pure, local tokenizer kernel shared by preparation and exported inference."""
import re
import unicodedata
MAX_SEQUENCE=65536


def special_tokens(c):
    return (['<unk>'] if c['unknownPolicy']=='unk' else [])+(['<pad>'] if c['pad'] else [])+(['<bos>'] if c['bos'] else [])+(['<eos>'] if c['eos'] else [])


def tokens(value,c):
    if value is None:
        if c['missingText']=='error':raise ValueError('Text contains a missing value; choose Empty explicitly or repair the source.')
        value=''
    if type(value) is not str:raise ValueError('Text encoding requires a String column; use Columns for explicit conversion.')
    if len(value.encode('utf-8'))>1048576:raise ValueError('One text observation exceeds 1 MiB; read smaller text records before encoding.')
    if c['unicodeNormalization']!='none':value=unicodedata.normalize(c['unicodeNormalization'],value)
    if c['letterCase']=='lower':value=value.lower()
    elif c['letterCase']=='casefold':value=value.casefold()
    if c['whitespace']=='trim':value=value.strip()
    elif c['whitespace']=='collapse':value=' '.join(value.split())
    values=list(value) if c['tokenizer']=='character' else re.findall(r'\s+|\S+',value)
    if len(values)+int(c['bos'])+int(c['eos'])>MAX_SEQUENCE:raise ValueError('Text observation exceeds 65536 tokens; split into smaller records. No content was silently truncated.')
    return values


def encode(text,metadata):
    if type(metadata) is not dict or set(metadata)!={'schemaVersion','configuration','vocabulary','unicodeVersion','sourceStateSHA256'} or type(metadata['schemaVersion']) is not int or metadata['schemaVersion']!=1 or metadata['unicodeVersion']!=unicodedata.unidata_version:raise ValueError('Tokenizer metadata/version is invalid.')
    c=metadata['configuration'];v=metadata['vocabulary'];special=special_tokens(c)
    if type(v) is not list or not v or len(v)>65536 or any(type(t) is not str or not t for t in v) or len(set(v))!=len(v) or v[:len(special)]!=special:raise ValueError('Saved vocabulary order or special roles are invalid.')
    lookup={t:i for i,t in enumerate(v)};unknown=lookup.get('<unk>') if c['unknownPolicy']=='unk' else None
    result=[lookup['<bos>']] if c['bos'] else []
    for token in tokens(text,c):
        if token not in lookup and unknown is None:raise ValueError('Text contains a token outside the captured vocabulary.')
        result.append(lookup.get(token,unknown))
    if c['eos']:result.append(lookup['<eos>'])
    return result
