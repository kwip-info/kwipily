"""Versioned, path-free named model inputs. Conversion owns bounded batch buffers.

Representations are independent of source format and device. Image/audio are
reserved representations, not implied decoders or implicit preprocessing.
"""
import math
import re
import struct

MAX_INPUTS=8
MAX_BATCH_ROWS=4096
MAX_BATCH_BYTES=268435456
MAX_SEQUENCE=65536
NAME=re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')


def validate_plan(plan):
    if type(plan) is not dict or type(plan.get('schemaVersion')) is not int or plan['schemaVersion'] not in (1,2,3) or set(plan)!=({'schemaVersion','inputs','roles','maximumBatchBytes'}|({'sequenceObjective'} if plan['schemaVersion']==2 else set())):
        raise ValueError('Unsupported named input plan; expected exact supported versioned fields.')
    if type(plan['maximumBatchBytes']) is not int or not 1<=plan['maximumBatchBytes']<=MAX_BATCH_BYTES:
        raise ValueError('Named input batch byte allowance must be 1 byte through 256 MiB.')
    for group in ('inputs','roles'):
        entries=plan[group]
        if type(entries) is not list or not (1 if group=='inputs' else 0)<=len(entries)<=MAX_INPUTS:
            raise ValueError('Provide 1–8 named inputs and at most eight objective roles.')
        seen=set();nodes=set()
        for entry in entries:
            required={'name','representation','dtype','sampleShape','columns'}|({'nodeID'} if group=='inputs' else set())
            optional={'lengthPolicy','paddingValue','vocabularySize'}
            if plan['schemaVersion']==3:optional=optional|{'sampleAxes'}
            if type(entry) is not dict or not required<=set(entry)<=required|optional:
                raise ValueError('Unknown or missing named tensor mapping field.')
            name=entry['name']
            if type(name) is not str or NAME.fullmatch(name) is None or name in seen:raise ValueError('Input/role names must be unique identifiers.')
            seen.add(name)
            if group=='inputs':
                node=entry['nodeID']
                if type(node) is not str or not node or len(node)>128 or '.' in node or node in nodes:raise ValueError('Each named input must own one distinct graph boundary.')
                nodes.add(node)
            if entry['representation'] not in ('numeric','tokens','prepared'):
                raise ValueError('Choose numeric columns, token sequences or explicitly prepared tensor artifacts.')
            if entry['dtype'] not in ('float32','int64'):raise ValueError('Choose float32 or exact int64 tensor values.')
            shape=entry['sampleShape']
            if type(shape) is not list or not (1 if group=='inputs' else 0)<=len(shape)<=7 or any(type(d) is not int or not 1<=d<=(1048576 if entry['representation']=='prepared' else 4096) for d in shape) or math.prod(shape)>1048576:
                raise ValueError('Tensor sample shape is outside the bounded contract.')
            cols=entry['columns']
            if type(cols) is not list or not cols or len(cols)>128 or any(type(c) is not str or not c.strip() or len(c)>200 or '\0' in c for c in cols) or len(set(cols))!=len(cols):raise ValueError('Choose distinct ordered source columns.')
            if entry['representation']=='prepared':
                axes=entry.get('sampleAxes')
                if plan['schemaVersion']!=3 or len(cols)!=1 or type(axes) is not list or len(axes)!=len(shape) or len(set(axes))!=len(axes) or any(type(axis) is not str or axis not in ('channels','samples','height','width','features','tokens','frequency','time') for axis in axes) or set(entry)&{'lengthPolicy','paddingValue','vocabularySize'}:raise ValueError('Prepared tensors require exact named axes and one captured tensor column in mapping v3.')
            elif entry['representation']=='numeric':
                if len(cols)!=math.prod(shape) or optional&set(entry):raise ValueError('Numeric columns must fill the exact sample shape without sequence policies.')
            else:
                if entry['dtype']!='int64' or len(shape)!=1 or len(cols)!=1:raise ValueError('Token input requires one typed integer sequence column and one sequence direction.')
                if set(entry)&{'lengthPolicy','paddingValue','vocabularySize'} != {'lengthPolicy','paddingValue','vocabularySize'} or 'sampleAxes' in entry:raise ValueError('Token mappings require explicit length, padding and vocabulary policies.')
                if entry['lengthPolicy'] not in ('error','pad','truncate','padTruncate'):raise ValueError('Choose an explicit token length policy.')
                vocab=entry['vocabularySize'];pad=entry['paddingValue']
                if type(vocab) is not int or not 1<=vocab<=65536 or type(pad) is not int or not 0<=pad<vocab:raise ValueError('Token padding ID must belong to the declared vocabulary.')
    if {c for e in plan['inputs'] for c in e['columns']}&{c for e in plan['roles'] for c in e['columns']}:
        raise ValueError('Outcome columns cannot also be model input features.')
    if plan['schemaVersion']==2:
        seq=plan['sequenceObjective']
        if type(seq) is not dict or set(seq)!={'schemaVersion','inputName','targetName','lengthColumn','maskColumn'} or type(seq['schemaVersion']) is not int or seq['schemaVersion']!=1:raise ValueError('Sequence objective requires exact versioned fields.')
        for key in ('inputName','targetName','lengthColumn','maskColumn'):
            if type(seq[key]) is not str or len(seq[key])>200 or '\0' in seq[key] or key!='maskColumn' and not seq[key].strip():raise ValueError('Choose explicit sequence input, target and valid-length columns.')
        x=next((e for e in plan['inputs'] if e['name']==seq['inputName']),None)
        if x is None or len(plan['roles'])!=1 or plan['roles'][0]['name']!='target' or seq['targetName']!='target':raise ValueError('Sequence objective binds one existing token input and the target role.')
        y=plan['roles'][0]
        for e in (x,y):
            if e['representation']!='tokens' or e['lengthPolicy']!='error':raise ValueError('Window input and targets require exact token sequences; resizing is not allowed.')
        if any(x[k]!=y[k] for k in ('sampleShape','vocabularySize','paddingValue')):raise ValueError('Sequence input and target shape/vocabulary/padding must match.')
        if not 2<=x['vocabularySize']<=512:raise ValueError('Sequence cross entropy currently supports 2–512 vocabulary entries.')
        columns={c for e in plan['inputs']+plan['roles'] for c in e['columns']}
        if seq['lengthColumn'] in columns or seq['maskColumn'] in columns or seq['lengthColumn']==seq['maskColumn']:raise ValueError('Length and mask metadata must be separate from model inputs and targets.')
    return plan


def valid_sequence(value):
    return type(value) in (list,tuple) and len(value)<=MAX_SEQUENCE and all(type(v) is int and 0<=v<2**63 for v in value)


def convert_rows(plan,columns,rows,*,media_reader=None):
    import torch
    validate_plan(plan)
    if type(rows) not in (list,tuple) or not 1<=len(rows)<=MAX_BATCH_ROWS:raise ValueError('Convert only a bounded nonempty batch of at most 4096 observations.')
    names=[c['name'] for c in columns]
    if len(set(names))!=len(names):raise ValueError('Input schema has duplicate columns.')
    size=sum(math.prod(e['sampleShape'])*(8 if e['dtype']=='int64' else 4) for e in plan['inputs']+plan['roles'])*len(rows)
    if size>plan['maximumBatchBytes']:raise ValueError('Named input batch exceeds its byte allowance; reduce the batch or sample shape.')
    groups=[]
    for group in ('inputs','roles'):
        result={}
        for entry in plan[group]:
            try:indices=[names.index(c) for c in entry['columns']]
            except ValueError:raise ValueError('Mapped source column is absent: '+entry['name']) from None
            token=entry['representation']=='tokens'
            prepared=entry['representation']=='prepared'
            allowed=('tensor',) if prepared else ('int64Sequence',) if token else ('int64',) if entry['dtype']=='int64' else ('int64','float64')
            if any(columns[i]['dtype'] not in allowed for i in indices):raise ValueError('Mapped source type does not satisfy '+entry['name'])
            output=[]
            for row in rows:
                if type(row) not in (list,tuple) or len(row)!=len(names):raise ValueError('Input row width differs from its captured schema.')
                values=[row[i] for i in indices]
                if prepared:
                    if not callable(media_reader):raise ValueError('Prepared inputs require an owned captured media reader.')
                    if __package__:from .media_artifacts import validate_reference,decode_asset
                    else:from media_artifacts import validate_reference,decode_asset
                    import numpy as np
                    reference=validate_reference(values[0])
                    if reference['kind']!='preparedTensorRef' or reference['shape']!=entry['sampleShape'] or reference['axes']!=entry['sampleAxes']:raise ValueError('Prepared tensor shape/axes differ from '+entry['name']+'.')
                    if reference['dtype']!=entry['dtype']:raise ValueError('Prepared tensor dtype differs from '+entry['name']+'; no coercion is enabled.')
                    _,content=decode_asset(media_reader(reference),expected=reference)
                    values=torch.from_numpy(np.frombuffer(content,dtype=np.dtype(entry['dtype']).newbyteorder('<')).copy().reshape(entry['sampleShape']))
                elif token:
                    values=values[0]
                    if not valid_sequence(values) or any(v>=entry['vocabularySize'] for v in values):raise ValueError('Token IDs must be exact integers within the saved vocabulary.')
                    length=entry['sampleShape'][0];policy=entry['lengthPolicy']
                    if len(values)<length and policy not in ('pad','padTruncate') or len(values)>length and policy not in ('truncate','padTruncate'):raise ValueError('Token sequence length differs from '+entry['name']+'; choose an explicit padding/truncation policy.')
                    values=list(values[:length])+[entry['paddingValue']]*max(0,length-len(values))
                elif entry['dtype']=='int64':
                    if any(type(v) is not int or not -(2**63)<=v<2**63 for v in values):raise ValueError('Integer input/target values require exact int64; no coercion is enabled.')
                else:
                    converted=[]
                    for v in values:
                        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('Numeric model inputs must be finite and nonmissing.')
                        try:f=struct.unpack('f',struct.pack('f',v))[0]
                        except (OverflowError,struct.error):raise ValueError('Numeric input exceeds float32 range.') from None
                        if not math.isfinite(f) or v!=0 and f==0:raise ValueError('Numeric input underflows or overflows float32.')
                        converted.append(f)
                    values=converted
                output.append(values)
            result[entry['name']]=torch.stack(output) if prepared else torch.tensor(output,dtype=getattr(torch,entry['dtype'])).reshape(len(rows),*entry['sampleShape'])
        groups.append(result)
    if plan['schemaVersion']==2:
        seq=plan['sequenceObjective'];entry=next(e for e in plan['inputs'] if e['name']==seq['inputName'])
        n=entry['sampleShape'][0];padding=entry['paddingValue']
        try:length_index=names.index(seq['lengthColumn']);mask_index=names.index(seq['maskColumn']) if seq['maskColumn'] else None
        except ValueError:raise ValueError('Sequence valid-length or loss-mask column is absent.') from None
        if columns[length_index]['dtype']!='int64' or mask_index is not None and columns[mask_index]['dtype']!='int64Sequence':raise ValueError('Sequence length/mask columns have incompatible types.')
        x=groups[0][seq['inputName']];y=groups[1]['target']
        for i,row in enumerate(rows):
            valid=row[length_index]
            if type(valid) is not int or not 1<=valid<=n:raise ValueError('Every sequence example needs 1–context valid targets.')
            expected=[1]*valid+[0]*(n-valid)
            if mask_index is None:
                if valid!=n:raise ValueError('Incomplete sequences require an explicit loss mask.')
            else:
                mask=row[mask_index]
                if type(mask) not in (list,tuple) or len(mask)!=n or any(type(v) is not int for v in mask) or list(mask)!=expected:raise ValueError('Loss mask must match the nonempty prefix of valid targets exactly.')
            if not torch.equal(x[i,1:valid],y[i,:valid-1]):raise ValueError('Sequence target is not the exact next-token shift.')
            if not torch.all(x[i,valid:]==padding).item() or not torch.all(y[i,valid:]==padding).item():raise ValueError('Padded sequence positions differ from the declared padding ID.')
            y[i,valid:]=-100
    return tuple(groups)


def sequence_prediction_row_limit(plan,output_shape,count):
    """Conservative JSON allowance, shared with Core; exact encoding stays checked."""
    if plan.get('schemaVersion')!=2:return min(200,count)
    elements=sum(math.prod(e['sampleShape']) for e in plan['inputs']+plan['roles'])+math.prod(output_shape)+plan['roles'][0]['sampleShape'][0]
    allowance=900_000//(4096+32*elements)
    if allowance<1:raise ValueError('Sequence prediction geometry exceeds the bounded preview allowance; disable optional prediction-row capture before training.')
    return min(8,count,allowance)
