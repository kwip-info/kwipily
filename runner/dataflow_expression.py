"""Small typed expression language. No Python parsing, evaluation or function calls.

Columns use [name]; strings use JSON double quotes. Literals: true,false,null.
Operators: + - * / %, = == != < <= > >=, AND OR NOT, IS [NOT] NULL.
Null arithmetic/comparison propagates null; AND/OR use three-valued logic.
"""
import json
import math
import re

TOKEN = re.compile(r'\s*(?:(\[[^\]\n]{1,200}\])|("(?:[^"\\]|\\.)*")|((?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)|([A-Za-z]+)|(<=|>=|!=|==|[()+*/%<>=-]))')
NUMERIC={'int64','float64'}

class NumericExpressionRequired(ValueError):
 code = "expression_numeric_required"

class Expression:
 token_pattern=TOKEN
 def __init__(self,text,columns):
  if not isinstance(text,str) or not 1<=len(text)<=4096:raise ValueError('Expression requires 1–4096 characters')
  self.tokens=[];pos=0;self.columns={c['name']:(i,c['dtype']) for i,c in enumerate(columns)}
  while pos<len(text):
   if text[pos:].isspace():break
   match=self.token_pattern.match(text,pos)
   if match is None:raise ValueError(f'Expression character {pos+1}: use [column], a literal, or a supported operator')
   value=next(v for v in match.groups() if v is not None);self.tokens.append((value,pos));pos=match.end()
   if len(self.tokens)>256:raise ValueError('Expression exceeds 256 tokens')
  self.i=0;self.tree=self.parse(0,0)
  if self.i!=len(self.tokens):self.fail('unexpected token; check parentheses and operators')
  self.dtype=self.type_of(self.tree)
 def fail(self,message):
  pos=self.tokens[self.i][1]+1 if self.i<len(self.tokens) else (self.tokens[-1][1]+2 if self.tokens else 1)
  raise ValueError(f'Expression character {pos}: {message}')
 def peek(self):return self.tokens[self.i][0].upper() if self.i<len(self.tokens) else ''
 def take(self):
  if self.i>=len(self.tokens):self.fail('expected a value')
  value=self.tokens[self.i][0];self.i+=1;return value
 def parse(self,minimum,depth):
  if depth>32:self.fail('nesting exceeds 32 levels')
  token=self.take();upper=token.upper()
  if token=='(':
   left=self.parse(0,depth+1)
   if self.take()!=')':self.fail('expected closing parenthesis')
  elif upper in ('NOT','-','+'):
   left=('unary',upper,self.parse(25 if upper=='NOT' else 60,depth+1))
  elif token.startswith('['):left=('column',token[1:-1])
  elif token.startswith('"'):
   try:value=json.loads(token)
   except ValueError:self.fail('invalid quoted string')
   left=('literal',value)
  elif upper in ('TRUE','FALSE','NULL'):left=('literal',{'TRUE':True,'FALSE':False,'NULL':None}[upper])
  elif token[0].isdigit() or token[0]=='.':
   try:value=float(token) if any(c in token for c in '.eE') else int(token)
   except ValueError:self.fail('invalid number')
   left=('literal',value)
  elif hasattr(self,'parse_call') and self.peek()=='(':left=self.parse_call(token,depth)
  else:self.fail('use a [column] or a supported literal')
  precedence={'OR':10,'AND':20,'IS':30,'=':30,'==':30,'!=':30,'<':30,'<=':30,'>':30,'>=':30,'+':40,'-':40,'*':50,'/':50,'%':50}
  while self.peek() in precedence and precedence[self.peek()]>=minimum:
   op=self.take().upper()
   if op=='IS':
    negate=self.peek()=='NOT'
    if negate:self.take()
    if self.peek()!='NULL':self.fail('IS supports NULL or NOT NULL')
    self.take();left=('isNull',negate,left)
   else:left=('binary',op,left,self.parse(precedence[op]+1,depth+1))
  return left
 def type_of(self,node):
  kind=node[0]
  if kind=='call':return self.call_type(node)
  if kind=='column':
   if node[1] not in self.columns:raise ValueError(f'Expression references unknown column [{node[1]}]; choose an input column')
   if self.columns[node[1]][1] in ('image','audio','tensor'):raise ValueError('Image columns cannot be scalar expression values; use an explicit image adapter.')
   return self.columns[node[1]][1]
  if kind=='literal':
   v=node[1]
   if v is None:return 'null'
   if type(v) is bool:return 'bool'
   if type(v) is str:return 'string'
   self.bounded(v);return 'int64' if type(v) is int else 'float64'
  if kind=='isNull':self.type_of(node[2]);return 'bool'
  if kind=='unary':
   t=self.type_of(node[2]);allowed={'bool','null'} if node[1]=='NOT' else NUMERIC|{'null'}
   if t not in allowed:raise ValueError(f'Expression {node[1]} requires '+('boolean' if node[1]=='NOT' else 'numeric')+' values; use Cast explicitly')
   return t
  op=node[1];a=self.type_of(node[2]);b=self.type_of(node[3]);types={a,b}-{'null'}
  if op in ('AND','OR'):
   if not types<={'bool'}:raise ValueError(f'Expression {op} requires boolean comparisons')
   return 'bool'
  if op in ('=','==','!=','<','<=','>','>='):
   if len(types)>1 and not types<=NUMERIC:raise ValueError('Expression compares different types; add an explicit Cast')
   if op not in ('=','==','!=') and types=={'bool'}:raise ValueError('Boolean values support only equality comparisons')
   return 'bool'
  if not types<=NUMERIC:raise NumericExpressionRequired('Expression arithmetic requires numeric columns; add Cast explicitly')
  return 'float64' if op=='/' or 'float64' in types else 'int64'
 @staticmethod
 def bounded(v):
  if type(v) is int and not -(2**63)<=v<2**63:raise ValueError('Expression integer overflow; result must fit int64')
  if type(v) is float and not math.isfinite(v):raise ValueError('Expression produced a nonfinite number')
  if isinstance(v,str) and len(v)>4096:raise ValueError('Expression string exceeds 4096 characters')
  return v
 def evaluate(self,row):
  def visit(n):
   kind=n[0]
   if kind=='call':return self.bounded(self.call_value(n,[visit(arg) for arg in n[2]]))
   if kind=='literal':return n[1]
   if kind=='column':return row[self.columns[n[1]][0]]
   if kind=='isNull':return (visit(n[2]) is not None) if n[1] else (visit(n[2]) is None)
   if kind=='unary':
    a=visit(n[2])
    if a is None:return None
    return self.bounded(not a if n[1]=='NOT' else -a if n[1]=='-' else a)
   a=visit(n[2]);b=visit(n[3]);op=n[1]
   if op=='AND':return False if a is False or b is False else None if a is None or b is None else True
   if op=='OR':return True if a is True or b is True else None if a is None or b is None else False
   if a is None or b is None:return None
   try:
    if op in ('=','=='):v=a==b
    elif op=='!=':v=a!=b
    elif op=='<':v=a<b
    elif op=='<=':v=a<=b
    elif op=='>':v=a>b
    elif op=='>=':v=a>=b
    elif op=='+':v=a+b
    elif op=='-':v=a-b
    elif op=='*':v=a*b
    elif op=='/':v=a/b
    else:v=a%b
   except (ZeroDivisionError,OverflowError,TypeError) as e:raise ValueError('Expression arithmetic failed; check zero divisors and input types') from e
   return self.bounded(v)
  return self.bounded(visit(self.tree))


def apply(table,op,config,check_cancelled):
    raise ValueError('Retired expression execution is unavailable; use current Filter or Formula.')
