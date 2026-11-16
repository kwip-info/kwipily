"""Native ComponentField declaration/value contract in retained Block provenance."""
import copy
import pytest
from runner.composite_source import validate_bodies, validate_field


def field(kind, default, **options):
    return dict(key='setting', title='Setting', kind=kind, defaultValue=default, **options)


def document(declaration, value):
    ref={'id':'block','version':{'major':1,'minor':0,'patch':0}}
    body={'nodes':[],'instances':[],'edges':[],'configurationVersion':2}
    return {'root':dict(body,instances=[{'id':'instance','definition':ref,'overrides':{'setting':value}}]),
            'definitions':[{'reference':ref,'body':body,'parameters':[{'field':declaration,'targets':[]}]}]}


FIELDS=[field('integer',2,min=1,max=512),field('number',0.25,min=0,max=1),
        field('boolean',True),field('integerList',[2,3],min=1,max=4096,minItems=2,maxItems=2),
        field('choice','same',choices=[{'value':'same','title':'Same'},{'value':'valid','title':'Valid'}])]


@pytest.mark.parametrize('declaration,value',[(FIELDS[0],3.0),(FIELDS[1],0.75),(FIELDS[2],False),
    (FIELDS[3],[3.0,5]),(FIELDS[4],'valid'),
    (field('integer',None,min=-1,max=8,allowsNull=True),None),
    (field('number',None,min=0,max=1,allowsNull=True),0.5)])
def test_every_native_field_kind_accepts_valid_defaults_and_overrides(declaration,value):
    validate_bodies(document(declaration,value))


@pytest.mark.parametrize('index,value',[(0,True),(0,2.5),(0,513),(0,None),(0,2**1000),
    (1,True),(1,float('inf')),(1,float('nan')),(1,-0.1),(1,None),
    (2,1),(2,None),(3,[True,2]),(3,[1,2.5]),(3,[1]),(3,[1,2,3]),(3,[0,2]),
    (3,'1,2'),(3,None),(4,'unknown'),(4,True),(4,None)])
def test_invalid_values_are_rejected_without_coercion(index,value):
    with pytest.raises(ValueError):validate_bodies(document(FIELDS[index],value))


@pytest.mark.parametrize('index,changes',[(0,{'min':True}),(0,{'max':2**63}),(0,{'minItems':1}),
    (0,{'choices':[]}),(0,{'allowsNull':1}),(1,{'max':float('inf')}),(1,{'min':2}),
    (2,{'min':0}),(2,{'allowsNull':True}),(3,{'minItems':0}),(3,{'maxItems':65}),
    (3,{'minItems':True}),(3,{'allowsNull':True}),(3,{'defaultValue':[0,2]}),
    (4,{'choices':[]}),(4,{'choices':[{'value':'same','title':' ' }]}),
    (4,{'choices':[{'value':'same','title':'A'},{'value':'same','title':'B'}]}),
    (4,{'min':0}),(4,{'defaultValue':'absent'}),(4,{'allowsNull':True})])
def test_invalid_declaration_cannot_be_hidden_by_valid_override(index,changes):
    declaration=copy.deepcopy(FIELDS[index]);declaration.update(changes)
    with pytest.raises(ValueError):validate_bodies(document(declaration,FIELDS[index]['defaultValue']))


def test_missing_default_is_not_equivalent_to_explicit_nullable_default():
    declaration=field('number',None,min=0,max=1,allowsNull=True)
    del declaration['defaultValue']
    with pytest.raises(ValueError):validate_field(declaration)


def test_duplicate_parameters_rejected_even_without_any_instance():
    doc=document(FIELDS[0],2);doc['root']['instances']=[]
    doc['definitions'][0]['parameters']*=2
    with pytest.raises(ValueError):validate_bodies(doc)
