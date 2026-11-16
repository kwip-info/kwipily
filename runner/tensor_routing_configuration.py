"""Allocation-free strict named-port routing and explicit batched Einsum notation."""
import math,re
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout
OPERATIONS=frozenset({'split','chunk','gather','einsum'})
MAXIMUM=100_000_000
PRESETS={'multiply':'nab,nbc->nac','outer':'na,nb->nab','diagonal':'naa->na','reorder':'nab->nba'}
PART_PORTS=tuple(f'part{i}' for i in range(8))

def _integer(value,lo,hi):
    if type(value) not in (int,float) or not lo<=value<=hi or value!=int(value):raise ValueError('Routing count/direction requires an exact bounded integer, not Boolean.')
    return int(value)

def _text(value):
    if type(value) is not str or not 1<=len(value)<=128 or any(ord(c)<32 or ord(c)>126 for c in value):raise ValueError('Einsum notation requires 1–128 printable ASCII characters.')
    return value.replace(' ','')

def parse_equation(equation):
    equation=_text(equation)
    if equation.count('->')!=1:raise ValueError('Einsum requires exactly one explicit output arrow.')
    left,output=equation.split('->');operands=tuple(left.split(','))
    if not 1<=len(operands)<=2:raise ValueError('Einsum supports one or two operands.')
    for labels in (*operands,output):
        if not 2<=len(labels)<=8 or re.fullmatch('[A-Za-z]+',labels) is None or labels[0]!='n' or labels.count('n')!=1:raise ValueError('Keep observation label n exactly once, first, with 1–7 sample labels.')
    if len(set(output))!=len(output) or not set(output)<=set(''.join(operands)):raise ValueError('Output labels must be unique and present in an input.')
    return operands,output

def normalize_tensor_routing(operation,config):
    defaults={'split':{'mode':'size','dim':-1,'split_size':2,'sections':[1,1]},'chunk':{'chunks':2,'dim':-1},'gather':{'dim':-1},'einsum':{'pattern':'multiply','equation':PRESETS['multiply']}}
    if type(operation) is not str or operation not in defaults or type(config) is not dict or set(config)-set(defaults[operation]):raise ValueError('Routing requires only its declared configuration fields.')
    c={**defaults[operation],**config}
    if operation=='einsum':
        if type(c['pattern']) is not str or c['pattern'] not in (*PRESETS,'custom'):raise ValueError('Choose a reviewed arrangement or custom notation.')
        equation=_text(c['equation']);equation=equation if c['pattern']=='custom' else PRESETS[c['pattern']];parse_equation(equation)
        return {'pattern':c['pattern'],'equation':equation}
    dim=_integer(c['dim'],-8,7)
    if dim==0:raise ValueError('Routing cannot select the observation direction.')
    if operation=='gather':return {'dim':dim}
    if operation=='chunk':return {'chunks':_integer(c['chunks'],1,8),'dim':dim}
    if type(c['mode']) is not str or c['mode'] not in ('size','sections'):raise ValueError('Split arrangement must be size or sections.')
    size=_integer(c['split_size'],1,MAXIMUM);raw=c['sections']
    if type(raw) is not list or not 1<=len(raw)<=8:raise ValueError('Split requires 1–8 positive sections.')
    sections=[_integer(v,1,MAXIMUM) for v in raw]
    return {'mode':c['mode'],'dim':dim,'split_size':size if c['mode']=='size' else 2,'sections':sections if c['mode']=='sections' else [1,1]}

def configured_inputs(operation,config):
    c=normalize_tensor_routing(operation,config)
    if operation=='gather':return ('source','indices')
    if operation=='einsum':return ('a','b') if len(parse_equation(c['equation'])[0])==2 else ('a',)
    return ('in',)

def resolve_tensor_routing(operation,config,layouts):
    c=normalize_tensor_routing(operation,config);ports=configured_inputs(operation,c)
    if type(layouts) not in (tuple,list) or len(layouts)!=len(ports) or any(not isinstance(v,BatchedLayout) for v in layouts):raise ValueError('Routing requires its exact configured concrete input layouts.')
    left=layouts[0];work=0
    if operation in ('split','chunk'):
        axis=left._axis(c['dim']);extent=left.sample[axis]
        if operation=='split' and c['mode']=='sections':
            sections=c['sections']
            if sum(sections)!=extent:raise ValueError('Sections must add to the selected input extent.')
        else:
            step=c['split_size'] if operation=='split' else (extent+c['chunks']-1)//c['chunks'];count=(extent+step-1)//step
            if count>8:raise ValueError('Routing supports at most eight actual parts; increase part size.')
            sections=[min(step,extent-i*step) for i in range(count)]
        outputs={}
        for i,size in enumerate(sections):sample=list(left.sample);sample[axis]=size;outputs[f'part{i}']=BatchedLayout(tuple(sample))
    elif operation=='gather':
        axis=left._axis(c['dim']);right=layouts[1]
        if len(left.sample)!=len(right.sample) or any(b>a for i,(a,b) in enumerate(zip(left.sample,right.sample)) if i!=axis):raise ValueError('Gather indices need equal rank and bounded nonselected extents; no broadcasting.')
        outputs={'out':right}
    else:
        operands,output=parse_equation(c['equation']);sizes={}
        for labels,layout in zip(operands,layouts):
            if len(labels)!=len(layout.sample)+1:raise ValueError('Einsum input labels must match its full tensor rank.')
            local={}
            for label,extent in zip(labels[1:],layout.sample):
                if label in local and local[label]!=extent:raise ValueError('Repeated labels inside an operand require equal extents.')
                local[label]=extent
            for label,extent in local.items():
                prior=sizes.get(label,1)
                if prior!=extent and prior!=1 and extent!=1:raise ValueError('Shared sample labels must match or have extent one.')
                sizes[label]=max(prior,extent)
        work=1
        for extent in sizes.values():
            if work>MAXIMUM//extent:raise ValueError('Einsum contraction exceeds its named-coordinate work budget.')
            work*=extent
        outputs={'out':BatchedLayout(tuple(sizes[label] for label in output[1:]))}
    return {'config':c,'inputs':ports,'outputs':outputs,'workPerObservation':work}

def capability(operation):
    if type(operation) is not str or operation not in OPERATIONS:raise ValueError('Unknown routing operation.')
    d={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.'+operation,'graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True,'graphDTypes':['float32'],'maximumTensorElements':MAXIMUM,'parameterless':True}
    if operation in ('split','chunk','gather'):d.update(axisMinimum=-8,axisMaximum=7,excludesBatch=True)
    if operation in ('split','chunk'):d.update(maximumOutputs=8,outputPorts=list(PART_PORTS),views=True)
    if operation=='split':d.update(splitSizeMinimum=1,splitSizeMaximum=MAXIMUM,sectionCountMinimum=1,sectionCountMaximum=8)
    elif operation=='chunk':d.update(chunksMinimum=1,chunksMaximum=8,mayReturnFewer=True)
    elif operation=='gather':d.update(indicesDType='int64',broadcast=False,negativeIndices=False,sparseGradient=False)
    else:d.update(maximumOperands=2,equationMinimumCharacters=1,equationMaximumCharacters=128,asciiOnly=True,explicitOutput=True,reservedBatchLabel='n',maximumContractionTerms=MAXIMUM)
    return d
