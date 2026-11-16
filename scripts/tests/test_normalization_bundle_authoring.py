import copy, importlib.util, json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('normalization_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source(): return json.loads((ROOT/'components/normalization/bundle.json').read_text())
def test_reviewed_normalization_authoring(): bundles.validate(source(),ROOT)
@pytest.mark.parametrize('mutation',['batch','eps','affine','bias','state','field','port','icon','source','version','typedVersion','lesson','duplicate'])
def test_normalization_refuses_semantic_drift(mutation):
    value=source();r=value['components'][0]
    if mutation=='batch':r['runtimeContract']['preservesBatch']=False
    elif mutation=='eps':r['definition']['fields'][1]['min']=0
    elif mutation=='affine':r['runtimeContract']['elementwiseAffineModes']=[True]
    elif mutation=='bias':r['definition']['fields'].pop()
    elif mutation=='state':r['runtimeContract']['usesCurrentInputStatistics']=False
    elif mutation=='field':r['definition']['fields'][0]['defaultValue']=[True]
    elif mutation=='port':r['definition']['outputs'].append('statistics')
    elif mutation=='icon':r['presentation']['icon']='../bad'
    elif mutation=='source':r['sources']['worker']='../outside.py'
    elif mutation=='version':r['configurationVersion']=1
    elif mutation=='typedVersion':r['runtimeContract']['version']['major']=1.0
    elif mutation=='lesson':r['learning']=None
    elif mutation=='duplicate':value['components'].append(copy.deepcopy(r))
    with pytest.raises(ValueError):bundles.validate(value,ROOT)

@pytest.mark.parametrize('mutation',['mean','bias','affine','epsilon','typed','field','port','lesson','source'])
def test_rms_refuses_contract_and_authoring_drift(mutation):
    value=source();r=next(row for row in value['components'] if row['definition']['id']=='rms_norm')
    if mutation=='mean':r['runtimeContract']['subtractsMean']=True
    elif mutation=='bias':r['runtimeContract']['hasBias']=True
    elif mutation=='affine':r['runtimeContract']['elementwiseAffineModes']=[True]
    elif mutation=='epsilon':r['definition']['fields'][1]['min']=0
    elif mutation=='typed':r['runtimeContract']['preservesBatch']=1
    elif mutation=='field':r['definition']['fields'].append({'key':'bias'})
    elif mutation=='port':r['definition']['outputs'].append('state')
    elif mutation=='lesson':r['learning']=None
    elif mutation=='source':r['sources']['worker']='../outside.py'
    with pytest.raises((ValueError,TypeError)):bundles.validate(value,ROOT)
