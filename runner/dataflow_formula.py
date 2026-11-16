"""Ordered typed column formulas. Regex runs only in a bounded disposable worker.

Guided editors serialize the same allowlisted expression calls as the text editor.
No eval, Python AST, arbitrary imports, source rereads or whole-table materialization.
"""
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
if __package__:from .dataflow_expression import Expression, NumericExpressionRequired
else:
    if __name__=="__main__":sys.path.insert(0,str(Path(__file__).resolve().parent))
    from dataflow_expression import Expression, NumericExpressionRequired

TOKEN=re.compile(r'\s*(?:(\[[^\]\n]{1,200}\])|("(?:[^"\\]|\\.)*")|((?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)|([A-Za-z_][A-Za-z_0-9]*)|(<=|>=|!=|==|[(),+*/%<>=-]))')
if __package__:from . import formula_temporal, dataflow_temporal, formula_decimal
else:import formula_temporal, dataflow_temporal, formula_decimal

DTYPES=('string','int64','float64','bool')+dataflow_temporal.DTYPES
FUNCTIONS=('lower','upper','trim','length','abs','round','is_finite','is_integer','require','coalesce','regex_replace','regex_extract','regex_match')+formula_temporal.FUNCTIONS+formula_decimal.FUNCTIONS

class FormulaExpression(Expression):
    token_pattern=TOKEN
    def __init__(self,text,columns):
        self.regexes={};self.column_contracts={c['name']:dict(c) for c in columns};super().__init__(text,columns)
    def parse_call(self,name,depth):
        if name not in FUNCTIONS:raise ValueError(f'Unsupported Formula function {name!r}')
        self.take();args=[]
        if self.peek()!=')':
            while True:
                args.append(self.parse(0,depth+1))
                if self.peek()!=',':break
                self.take()
        if self.take()!=')':self.fail('expected closing function parenthesis')
        return ('call',name,tuple(args))
    @staticmethod
    def literal(node):
        if node[0]!='literal' or type(node[1]) is not str:raise ValueError('Function options must be explicit string literals')
        return node[1]
    def call_type(self,node):
        name,args=node[1:];types=[self.type_of(arg) for arg in args]
        if name in formula_decimal.FUNCTIONS:return formula_decimal.call_type(name,types)
        if name in formula_temporal.FUNCTIONS:
            start=2 if name=='temporal_difference' else 1
            return formula_temporal.call_contract(name,types,[self.literal(arg) for arg in args[start:]])[0]
        if name in ('lower','upper','trim','length'):
            if len(types)!=1 or types[0] not in ('string','null'):raise ValueError(f'{name} needs one text argument')
            return 'int64' if name=='length' else 'string'
        if name in ('abs','round'):
            if len(types)!=1 or types[0] not in ('int64','float64','null'):raise ValueError(f'{name} needs one numeric argument')
            return types[0] if name=='abs' else 'int64'
        if name in ('is_finite','is_integer'):
            if len(types)!=1 or types[0] not in ('int64','float64','null'):raise ValueError(f'{name} needs one numeric argument; parse text explicitly')
            return 'bool'
        if name=='require':
            if len(types)!=2 or types[0] not in ('bool','null') or types[1] not in DTYPES+('null',):raise ValueError('require needs a Boolean condition and one scalar value')
            return types[1]
        if name=='coalesce':
            nonnull=set(types)-{'null'}
            if not 2<=len(types)<=16 or len(nonnull)>1 and not nonnull<={'int64','float64'}:raise ValueError('coalesce needs 2–16 compatible scalar arguments')
            return 'float64' if 'float64' in nonnull else next(iter(nonnull),'null')
        minimum={'regex_replace':5,'regex_match':4,'regex_extract':6}[name]
        if len(args)<minimum or name!='regex_extract' and len(args)!=minimum or len(args)>37 or types[0] not in ('string','null'):raise ValueError(f'{name} arguments differ from the supported signature')
        options=[self.literal(arg) for arg in args[1:]];pattern=options[0]
        if not 1<=len(pattern)<=512:raise ValueError('Regex pattern needs 1–512 characters')
        flags=options[3] if name=='regex_replace' else options[2] if name=='regex_match' else options[3]
        selected=[] if not flags else flags.split('|')
        if len(set(selected))!=len(selected) or not set(selected)<={'ignoreCase','multiline','dotAll'}:raise ValueError('Unknown or duplicate regex flags')
        if name=='regex_replace':
            if options[2] not in ('first','all') or len(options[1])>4096:raise ValueError('Choose first/all replacement and a bounded replacement')
        else:
            if options[1] not in ('search','full'):raise ValueError('Choose search or full matching')
            if name=='regex_extract' and options[2] not in ('null','error'):raise ValueError('Choose null/error no-match policy')
        return 'captures' if name=='regex_extract' else 'bool' if name=='regex_match' else 'string'
    def prepare_regex(self,node=None):
        node=self.tree if node is None else node
        if node[0]=='call':
            if node[1].startswith('regex_'):
                name,args=node[1:];options=[self.literal(arg) for arg in args[1:]]
                flags=options[3] if name in ('regex_replace','regex_extract') else options[2]
                flag_value=sum({'ignoreCase':re.I,'multiline':re.M,'dotAll':re.S}[f] for f in flags.split('|') if f)
                try:
                    regex=re.compile(options[0],flag_value)
                    if name=='regex_replace':regex.sub(options[1],'')
                    if name=='regex_extract':
                        for raw in options[4:]:
                            group=int(raw) if raw.isdigit() else raw
                            if isinstance(group,int) and not 0<=group<=regex.groups or isinstance(group,str) and group not in regex.groupindex:raise ValueError(f'Capture group {raw!r} does not exist')
                    self.regexes[node]=regex
                except (re.error,IndexError,RecursionError) as error:raise ValueError(f'Regex pattern/replacement: {error}') from error
            for arg in node[2]:self.prepare_regex(arg)
        elif node[0]=='binary':self.prepare_regex(node[2]);self.prepare_regex(node[3])
        elif node[0] in ('unary','isNull'):self.prepare_regex(node[2])
    def call_value(self,node,values):
        name=node[1]
        if name in formula_decimal.FUNCTIONS:return formula_decimal.call_value(name,values)
        if name in formula_temporal.FUNCTIONS:return formula_temporal.call_value(name,values,[self.type_of(arg) for arg in node[2]])
        if name=='require':
            if values[0] is not True:raise ValueError('require condition must be true; the input does not meet this workflow assumption')
            return values[1]
        if name=='coalesce':return next((value for value in values if value is not None),None)
        value=values[0]
        if value is None:return [None]*(len(values)-5) if name=='regex_extract' else None
        if name=='lower':return value.lower()
        if name=='upper':return value.upper()
        if name=='trim':return value.strip()
        if name=='length':return len(value)
        if name=='abs':return abs(value)
        if name=='round':return round(value)
        if name=='is_finite':return type(value) is int or math.isfinite(value)
        if name=='is_integer':return type(value) is int or (math.isfinite(value) and value.is_integer())
        if len(value)>16384:raise ValueError('Regex text cells must not exceed 16384 characters')
        regex=self.regexes[node]
        if name=='regex_replace':
            result=regex.sub(values[2],value,count=1 if values[3]=='first' else 0)
            if len(result)>16384:raise ValueError('Regex result exceeds text budget')
            return result
        match=regex.fullmatch(value) if values[2]=='full' else regex.search(value)
        if name=='regex_match':return match is not None
        if match is None and values[3]=='error':raise ValueError('Regex has no match; choose missing outputs or revise pattern')
        return [match.group(int(raw) if raw.isdigit() else raw) if match else None for raw in values[5:]]
    @staticmethod
    def bounded(value):
        if isinstance(value,str):
            if len(value)>16384:raise ValueError("Formula text result exceeds 16384 characters")
            return value
        if isinstance(value,list):
            if len(value)>32:raise ValueError('At most 32 capture outputs')
            for item in value:FormulaExpression.bounded(item)
            return value
        return Expression.bounded(value)


def valid_name(name):return type(name) is str and bool(name.strip()) and len(name)<=200 and '\0' not in name


def validate_config(config):
    if type(config) is not dict or set(config)!={'operations'} or type(config['operations']) is not list or not 1<=len(config['operations'])<=64:raise ValueError('Formula needs 1–64 ordered operations')
    ids=set();written=set()
    for operation in config['operations']:
        if type(operation) is not dict or set(operation)!={'id','expression','outputs','nullPolicy'}:raise ValueError('Formula operation fields differ from the contract')
        if not valid_name(operation['id']) or operation['id'] in ids:raise ValueError('Formula operation identities must be unique')
        ids.add(operation['id'])
        if type(operation['expression']) is not str or not 1<=len(operation['expression'])<=4096 or operation['nullPolicy'] not in ('preserve','error'):raise ValueError('Formula expression or missing-value policy is invalid')
        outputs=operation['outputs']
        if type(outputs) is not list or not 1<=len(outputs)<=32:raise ValueError('Choose 1–32 output mappings')
        for output in outputs:
            if type(output) is not dict or set(output)!={'name','dtype','write'} or not valid_name(output['name']) or output['dtype'] not in DTYPES or output['write'] not in ('new','replace'):raise ValueError('Choose exact output names, scalar types and new/replace targets')
            if output['name'] in written:raise ValueError('Formula output names must be unique across operations')
            written.add(output['name'])


def compile_plan(columns,config):
    validate_config(config);columns=[dict(column) for column in columns];plan=[]
    for column in columns:dataflow_temporal.validate_column(column)
    for index,operation in enumerate(config['operations']):
        try:
            expression=FormulaExpression(operation['expression'],columns)
            captures=expression.dtype=='captures'
            expected=len(expression.tree[2])-5 if captures and expression.tree[0]=='call' else 1
            if len(operation['outputs'])!=expected:raise ValueError('Output mapping count differs from expression results')
            targets=[]
            for output in operation['outputs']:
                actual='string' if captures else expression.dtype
                if actual not in ('null',output['dtype']) and not(actual=='int64' and output['dtype']=='float64'):raise ValueError('Output type differs from expression; choose the exact type or cast upstream')
                existing=next((i for i,c in enumerate(columns) if c['name']==output['name']),None)
                if (existing is not None)!=(output['write']=='replace'):raise ValueError('New outputs must not exist; replacement outputs must already exist')
                if existing is not None and columns[existing]['dtype'] in ('image','audio','tensor'):raise ValueError('Formula cannot replace image content; use an explicit image adapter')
                target=len(columns) if existing is None else existing
                column={'name':output['name'],'dtype':output['dtype'],'nullable':operation['nullPolicy']=='preserve'}
                if output['dtype'] in dataflow_temporal.DTYPES:
                    column['temporal']=formula_temporal.expression_metadata(expression)
                    dataflow_temporal.validate_column(column)
                if existing is None:columns.append(column)
                else:columns[existing]=column
                targets.append((target,output['dtype']))
            if len(columns)>128:raise ValueError('Formula exceeds 128 output columns')
            plan.append((expression,targets,operation['nullPolicy']))
        except ValueError as error:
            wrapped=NumericExpressionRequired if isinstance(error,NumericExpressionRequired) else ValueError
            raise wrapped(f'Formula operation {index+1}: {error}') from error
    return columns,plan


def evaluate_row(source,plan,position):
    row=list(source)
    for index,(expression,targets,policy) in enumerate(plan):
        try:
            value=expression.evaluate(row);values=value if expression.dtype=='captures' else [value]
            for (target,dtype),value in zip(targets,values):
                if value is None and policy=='error':raise ValueError('Result is missing; choose Preserve or resolve missing values')
                if value is not None and dtype=='float64':value=float(value)
                FormulaExpression.bounded(value)
                if value is not None and dtype in dataflow_temporal.DTYPES and not dataflow_temporal.valid_value(value,dtype):raise ValueError('Result violates canonical temporal bounds')
                if target==len(row):row.append(value)
                else:row[target]=value
        except (ValueError,OverflowError,TypeError) as error:raise ValueError(f'Formula operation {index+1}, row {position+1}: {error}') from error
    return row


def _supervised(columns,config,rows,check,timeout,offset=0):
    payload=json.dumps({'columns':columns,'config':config,'rows':rows,'offset':offset},allow_nan=False).encode()
    if len(payload)>16_000_000:raise ValueError('Formula regex batch exceeds input budget')
    with tempfile.TemporaryDirectory(prefix='kwip-formula-') as folder:
        source=Path(folder)/'input.json';source.write_bytes(payload);output=Path(folder)/'output.json';process=None
        try:
            check()
            with output.open('wb') as stream:
                process=subprocess.Popen([sys.executable,'-I',str(Path(__file__).resolve()),str(source)],stdout=stream,stderr=subprocess.DEVNULL)
                deadline=time.monotonic()+timeout
                while process.poll() is None:
                    check()
                    if time.monotonic()>deadline:raise ValueError('Formula regex exceeded five-second total execution budget; no result accepted')
                    if output.stat().st_size>32_000_000:raise ValueError('Formula regex output exceeds budget')
                    time.sleep(.02)
            check()
            if process.returncode!=0 or output.stat().st_size>32_000_000:raise ValueError('Formula regex worker failed or exceeded output budget')
            answer=json.loads(output.read_bytes())
            if 'error' in answer:raise ValueError(answer['error'])
            if type(answer.get('rows')) is not list or len(answer['rows'])!=len(rows):raise ValueError('Formula worker output shape differs')
            return answer['rows']
        finally:
            if process is not None:
                if process.poll() is None:process.kill()
                process.wait()


def apply(table,config,check,*,timeout=5.0,share=False,native_extended=True):
    if __package__:from . import dataflow as df
    else:import dataflow as df
    check();columns,plan=compile_plan(table.columns,config)
    regex=any('regex_' in operation['expression'] for operation in config['operations'])
    changed_targets={target for _,targets,_ in plan for target,_ in targets}
    unchanged=[i for i in range(len(table.columns)) if i not in changed_targets]
    deadline=time.monotonic()+timeout
    def pairs():
        if not regex:
            for position,(row,rid) in enumerate(df.row_pairs(table)):
                if position%128==0:check()
                yield evaluate_row(row,plan,position),rid
        else:
            batch=[];size=0;offset=0
            def flush():
                remaining=deadline-time.monotonic()
                if remaining<=0:raise ValueError('Formula regex exceeded five-second total execution budget')
                rows=_supervised(list(table.columns),config,[list(row) for row,_ in batch],check,remaining,offset)
                for result,(original,_) in zip(rows,batch):
                    if type(result) not in (list,tuple) or len(result)!=len(columns):raise ValueError('Formula worker output shape differs')
                    for index in unchanged:
                        a,b=result[index],original[index];dtype=table.columns[index]['dtype']
                        if dtype in ('image','audio','tensor','int64Sequence'):
                            same=json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
                        else:
                            same=type(a) is type(b) and a==b
                            if same and type(a) is float:same=math.copysign(1,a)==math.copysign(1,b)
                        if not same:raise ValueError('Formula worker changed an unmapped input column.')
                return [(row,rid) for row,(_,rid) in zip(rows,batch)]
            for position,(row,rid) in enumerate(df.row_pairs(table)):
                if position%128==0:check()
                length=len(json.dumps(row,allow_nan=False).encode())
                if batch and (len(batch)>=8192 or size+length>1_000_000):
                    yield from flush();offset+=len(batch);batch=[];size=0
                batch.append((row,rid));size+=length
            yield from flush()
    if share:
        if __package__:
            from .dataflow_shared import supports_shared_table,computed_table
            from .dataflow_columnar import ColumnarStore
            from .dataflow_disk import MAX_ROW_BYTES
        else:
            from dataflow_shared import supports_shared_table,computed_table
            from dataflow_columnar import ColumnarStore
            from dataflow_disk import MAX_ROW_BYTES
        if supports_shared_table(table):
            if __package__:from .dataflow_shared import native_formula_table
            else:from dataflow_shared import native_formula_table
            native=native_formula_table(table,columns,plan,check,extended=native_extended,deadline=deadline if regex else None)
            if native is not None:return native
            changed=sorted({target for _,targets,_ in plan for target,_ in targets})
            store=ColumnarStore([columns[index] for index in changed],check,identity_prefix=table.row_ids.store.identity_prefix,budget=table.rows.store.budget)
            try:
                for position,(result,rid) in enumerate(pairs()):
                    if position>=df.MAX_TABLE_CELLS//len(columns):raise ValueError('Table exceeds the five million cell storage budget')
                    if len(json.dumps(result,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode('utf-8'))>MAX_ROW_BYTES:
                        raise ValueError('A columnar table row exceeds the 1 MiB record budget.')
                    store.append(tuple(result[index] for index in changed),rid)
                store.finish();return computed_table(table,columns,store,changed,check)
            except BaseException:store.close();raise
    return df.streaming_rows(table,pairs(),columns,check)


if __name__=='__main__':
    # Only bundled helpers are imported; -I excludes user/site/project import injection.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    try:
        payload_path=Path(sys.argv[1])
        if payload_path.stat().st_size>16_000_000:raise ValueError('Formula input exceeds budget')
        payload=json.loads(payload_path.read_bytes());_,plan=compile_plan(payload['columns'],payload['config'])
        for expression,_,_ in plan:expression.prepare_regex()
        rows=[evaluate_row(row,plan,i+payload.get('offset',0)) for i,row in enumerate(payload['rows'])]
        result=json.dumps({'rows':rows},allow_nan=False)
        if len(result.encode())>32_000_000:raise ValueError('Formula output exceeds budget')
        print(result)
    except BaseException as error:print(json.dumps({'error':str(error)}))
