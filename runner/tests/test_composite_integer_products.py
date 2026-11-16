"""Typed derived bindings are bounded provenance, never executable expressions."""
import copy
import pytest
from runner.composite_source import validate_bodies
from runner.tests.test_composite_fields import field, document


def fixture():
    doc=document(field('integer',2,min=1,max=512),2)
    definition=doc['definitions'][0]
    definition['parameters'][0]['targets']=[{'componentID':'first','field':'out_features'}]
    definition['parameters'].append({'field':dict(field('integer',3,min=1,max=512),key='multiplier'),
                                     'targets':[{'componentID':'second','field':'out_features'}]})
    definition['body']['nodes']=[{'id':name,'type':'linear','config':{'in_features':2,'out_features':2,'bias':True}}
                                for name in ('first','second')]
    definition['integerProducts']=[{'lhs':'setting','rhs':'multiplier','targets':[{'componentID':'second','field':'in_features'}]}]
    return doc


def test_valid_product_provenance_and_old_documents():
    doc=fixture();validate_bodies(doc)
    del doc['definitions'][0]['integerProducts'];validate_bodies(doc)


@pytest.mark.parametrize('change',[
    {'lhs':'missing'}, {'rhs':'missing'}, {'lhs':True}, {'targets':[]},
    {'targets':[{'componentID':'missing','field':'in_features'}]},
    {'targets':[{'componentID':'second','field':'absent'}]},
    {'targets':[{'componentID':'first','field':'out_features'}]},
    {'targets':[{'componentID':'second','field':'in_features'}]*2},
    {'operation':'eval'},
])
def test_malformed_or_conflicting_product_binding_refused(change):
    doc=fixture();doc['definitions'][0]['integerProducts'][0].update(change)
    with pytest.raises(ValueError):validate_bodies(doc)


@pytest.mark.parametrize('kind,default,options',[('number',2,{}),('boolean',True,{}),
    ('integer',2,{'allowsNull':True})])
def test_sources_must_be_nonnullable_integer_parameters(kind,default,options):
    doc=fixture();f=doc['definitions'][0]['parameters'][0]['field']
    f.update(kind=kind,defaultValue=default,**options)
    if kind=='boolean':f.pop('min');f.pop('max')
    doc['root']['instances'][0]['overrides']['setting']=default
    with pytest.raises(ValueError):validate_bodies(doc)


def test_overflowing_selected_operands_refused_without_mutation():
    doc=fixture()
    f=doc['definitions'][0]['parameters'][0]['field'];f.update(max=2**62,defaultValue=2**62)
    doc['root']['instances'][0]['overrides']['setting']=2**62
    before=copy.deepcopy(doc)
    with pytest.raises(ValueError):validate_bodies(doc)
    assert doc==before


@pytest.mark.parametrize('bindings',[True,{},'x',[{}]*129])
def test_bounded_binding_collection(bindings):
    doc=fixture();doc['definitions'][0]['integerProducts']=bindings
    with pytest.raises(ValueError):validate_bodies(doc)


def test_nested_integer_target_is_checked_against_declared_bounds():
    doc=fixture();definition=doc['definitions'][0]
    ref={'id':'nested','version':{'major':1,'minor':0,'patch':0}}
    nested={'reference':ref,'body':{'nodes':[],'instances':[],'edges':[]},
            'parameters':[{'field':dict(field('integer',2,min=1,max=8),key='width'),'targets':[]}]}
    definition['body']['instances']=[{'id':'child','definition':ref,'overrides':{}}]
    definition['integerProducts'][0]['targets']=[{'componentID':'child','field':'width'}]
    doc['definitions'].append(nested)
    validate_bodies(doc)
    doc['root']['instances'][0]['overrides']['setting']=4
    with pytest.raises(ValueError):validate_bodies(doc)


def test_product_cannot_target_boolean_primitive_setting():
    doc=fixture();doc['definitions'][0]['integerProducts'][0]['targets'][0]['field']='bias'
    with pytest.raises(ValueError):validate_bodies(doc)
