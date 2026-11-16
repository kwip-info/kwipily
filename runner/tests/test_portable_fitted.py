"""Portable fitted parameters reveal learned computation, not fit membership."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from runner import dataflow as df, dataflow_fitted as fitted
from runner import portable_fitted as portable
from runner.tests.legacy_fitted_fixtures import frozen_state


def fixture(operator, config, rows, dtype='float64'):
    schema=({'name':'x','dtype':dtype,'nullable':True},)
    table=df.derive(df.Table(schema,tuple((v,) for v in rows),tuple('PRIVATE-ID-'+str(i) for i in range(len(rows))),'PRIVATE-SOURCE','training',partition_digest='PRIVATE-PARTITION'))
    return table,frozen_state(table,operator,{'columns':['x'],**config},'PRIVATE-GRAPH')


@pytest.mark.parametrize('operator,config,training,held,dtype',[
    (fitted.SCALER,{},[1,3],[5,7],'int64'),
    (fitted.IMPUTER,{'strategy':'mean','emptyPolicy':'error'},[None,2,4],[None,8],'float64'),
    (fitted.IMPUTER,{'strategy':'median','emptyPolicy':'error'},[None,2,8],[None,9],'int64'),
    (fitted.IMPUTER,{'strategy':'mostFrequent','tiePolicy':'smallest','emptyPolicy':'error'},[2**63-1,None,2**63-1],[None,-2**63],'int64'),
    (fitted.IMPUTER,{'strategy':'mostFrequent','tiePolicy':'smallest','emptyPolicy':'error'},['z',None,'z'],[None,'new'],'string'),
    (fitted.ENCODER,{'vocabularyOrder':'firstSeen','unknownPolicy':'ignore','maxCategories':10},['z','a'],['new','a'],'string'),
])
def test_native_portable_parity_and_privacy(operator,config,training,held,dtype):
    table,state=fixture(operator,config,training,dtype)
    params=fitted.portable_parameters(state)
    raw=json.dumps(params)
    assert 'PRIVATE-' not in raw
    for key in ('fitEvidence','fitDataVersion','partitionDigest','graphDigest','observedCounts','emptyColumns','inputFittedStateDigests','configuration'):
        assert key not in params
    assert params['sourceContentDigest']==state['contentDigest']
    other=df.derive(table,rows=tuple((v,) for v in held),ids=tuple('held:'+str(i) for i in range(len(held))),role='test')
    native=fitted.transform(other,state,'PRIVATE-GRAPH')
    rows,columns,diagnostics=portable.apply(other.rows,list(other.columns),params)
    assert tuple(map(tuple,rows))==native.rows
    assert columns==list(native.columns)
    assert diagnostics==list(native.diagnostics)
    assert state==fitted.load(json.dumps(state))


def test_legacy_scaler_projection_and_independent_import():
    table,state=fixture(fitted.SCALER,{},[1.,3.])
    for key in ('outputSchema','inputFittedStateDigests','contentDigest'):state.pop(key)
    state['contentDigest']=df.digest(state)
    params=fitted.portable_parameters(state)
    spec=importlib.util.spec_from_file_location('standalone',Path(portable.__file__))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.apply(table.rows,list(table.columns),params)[0]==[[-1.],[1.]]


def test_projection_validates_native_evidence_before_redaction():
    _,state=fixture(fitted.SCALER,{},[1.,3.])
    state['fitEvidence']['role']='test'
    state['contentDigest']=df.digest({k:v for k,v in state.items() if k!='contentDigest'})
    with pytest.raises(ValueError):fitted.portable_parameters(state)


@pytest.mark.parametrize('change',[
    lambda p:p.update(extra='no'),lambda p:p.update(schemaVersion=True),
    lambda p:p.update(sourceContentDigest='bad'),lambda p:p.update(scales=[0]),
    lambda p:p.update(means=[float('nan')]),lambda p:p.update(means=[True]),
    lambda p:p.update(columns=['absent']),lambda p:p['outputSchema'][0].update(dtype='string'),
    lambda p:p.update(inputSchema=p['inputSchema']*129),
])
def test_reject_malformed_parameters(change):
    _,state=fixture(fitted.SCALER,{},[1.,3.]);params=fitted.portable_parameters(state)
    change(params)
    with pytest.raises(ValueError):portable.validate(params)


def test_missing_unknown_and_overflow_are_rejected():
    table,state=fixture(fitted.SCALER,{},[1.,3.]);params=fitted.portable_parameters(state)
    for value in (None,True,float('inf')):
        with pytest.raises(ValueError):portable.apply([[value]],list(table.columns),params)
    params['means']=[-1e308];params['scales']=[1e-300]
    with pytest.raises(ValueError):portable.apply([[1e308]],list(table.columns),params)
    table,state=fixture(fitted.ENCODER,{'vocabularyOrder':'lexicographic','unknownPolicy':'error','maxCategories':10},['a','b'],'string')
    params=fitted.portable_parameters(state)
    for value in (None,'unseen'):
        with pytest.raises(ValueError):portable.apply([[value]],list(table.columns),params)
    bad=copy.deepcopy(params);bad['vocabularies']=[['a','a']]
    with pytest.raises(ValueError):portable.validate(bad)


def test_row_width_bounds_and_imputer_dtype():
    table,state=fixture(fitted.IMPUTER,{'strategy':'mostFrequent','emptyPolicy':'error','tiePolicy':'smallest'},[1,None,1],'int64')
    params=fitted.portable_parameters(state)
    for rows in ([[2**63]],[[True]],[[1,2]],[[]],[[1]]*100001):
        with pytest.raises(ValueError):portable.apply(rows,list(table.columns),params)
    params['fillValues']=[2**63]
    with pytest.raises(ValueError):portable.validate(params)


def test_literal_promotion_and_integer_precision_contract():
    table,state=fixture(fitted.IMPUTER,{'strategy':'mean','emptyPolicy':'error'},[2,None,4],'int64')
    rows,columns,_=portable.apply([[None],[9]],list(table.columns),fitted.portable_parameters(state))
    assert rows==[[3.0],[9.0]] and all(type(row[0]) is float for row in rows)
    assert columns==[{'name':'x','dtype':'float64','nullable':False}]
    table,state=fixture(fitted.IMPUTER,{'strategy':'mostFrequent','tiePolicy':'smallest','emptyPolicy':'error'},[2**63-1,None,2**63-1],'int64')
    rows,columns,_=portable.apply([[None],[-2**63]],list(table.columns),fitted.portable_parameters(state))
    assert rows==[[9223372036854775807],[-9223372036854775808]]
    assert all(type(row[0]) is int for row in rows) and columns[0]['dtype']=='int64'


def test_multiple_column_order_unselected_values_and_diagnostics_append():
    schema=({'name':'left','dtype':'int64','nullable':True},
            {'name':'untouched','dtype':'string','nullable':True},
            {'name':'right','dtype':'int64','nullable':True})
    table=df.derive(df.Table(schema,((2,'a',8),(4,None,12)),('train:a','train:b'),'source','training',partition_digest='partition'))
    state=frozen_state(table,fitted.IMPUTER,{'columns':['right','left'],'strategy':'mean','emptyPolicy':'error'},'graph')
    previous={'code':'priorDiagnostic','count':4}
    held=df.derive(table,rows=((None,'held',None),(7,None,20)),ids=('held:a','held:b'),role='test',diagnostics=(previous,))
    params=fitted.portable_parameters(state)
    rows,columns,diagnostics=portable.apply(held.rows,list(held.columns),params)
    assert rows==[[3.0,'held',10.0],[7.0,None,20.0]]
    assert [col['name'] for col in columns]==['left','untouched','right']
    assert columns[1]==schema[1]
    assert [item['column'] for item in diagnostics]==['right','left']
    assert [item['count'] for item in diagnostics]==[1,1]
    native=fitted.transform(held,state,'graph')
    assert list(native.diagnostics)==[previous,*diagnostics]
    assert tuple(map(tuple,rows))==native.rows
