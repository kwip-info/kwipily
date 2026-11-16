"""Independent current teaching-population contracts; no retired data operators."""
from pathlib import Path
import pytest
from runner import dataflow as df
from runner.dataflow_preprocess import rule_defaults
from runner.polars_preparation import preparation_engine
from runner.polars_csv_configuration import DEFAULTS

ROOT=Path(__file__).resolve().parents[2]

def prepare(source,partition,rules,types,engine):
    columns={'rows':[{'source':name,'included':True,'outputName':name,'targetType':dtype} for name,dtype in types],
             'newColumns':'keep','castFailure':'error','trimText':False,'missingPolicy':'preserve'}
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{**DEFAULTS,'emptyField':'null'}},
           {'id':'columns','operator':'kwip.data.columns','config':columns},
           {'id':'partition','operator':'kwip.data.partition','config':partition},
           {'id':'prepare','operator':'kwip.data.preprocess','config':{'mode':'partitions','columns':rules}}],
           'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'columns','port':'table'}},
                    {'from':{'node':'columns','port':'table'},'to':{'node':'partition','port':'table'}}]+
                   [{'from':{'node':'partition','port':role},'to':{'node':'prepare','port':role}} for role in ('training','validation','test')]}
    with preparation_engine(engine):return df.execute(graph,{'source':str(source)})

@pytest.mark.parametrize('engine',['disk','polars'])
def test_classifier_practice_reserves_three_populations_and_fits_training_only(engine):
    result=prepare(ROOT/'Sources/KwipMLApp/Resources/workflow-observations.csv',
        {'strategy':'random','seed':42,'validationPercent':25,'testPercent':25,'stratifyColumn':'label'},
        [dict(rule_defaults(name),missing='error',scaling='standard') for name in ('signal','context')],
        [('signal','float64'),('context','float64'),('label','int64')],engine)
    assert [len(result['prepare'][role].rows) for role in ('training','validation','test')]==[32,16,16]
    fit=list(df.fit_observation_ids(result['prepare']['state']))
    assert set(fit)==set(result['partition']['training'].row_ids)
    assert set(fit).isdisjoint(result['partition']['validation'].row_ids)
    assert set(fit).isdisjoint(result['partition']['test'].row_ids)
    for role in ('training','validation','test'):
        assert result['prepare'][role].role==role
        assert result['prepare'][role].fitted_states==(result['prepare']['state']['contentDigest'],)

@pytest.mark.parametrize('engine',['disk','polars'])
def test_missing_category_practice_uses_independent_training_mean_and_known_categories(engine):
    result=prepare(ROOT/'Sources/KwipMLCore/Resources/preparation-practice-observations.csv',
        {'strategy':'time','timeColumn':'observed_at','timeEncoding':'iso8601Offset','trainingEnd':'2026-01-08T00:00:00Z',
         'validationEnd':'2026-01-12T00:00:00Z','embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'},
        [dict(rule_defaults('amount'),missing='mean'),dict(rule_defaults('region'),encoding='oneHot',vocabularyOrder='lexicographic',unknownPolicy='ignore',maxCategories=2)],
        [('amount','float64'),('label','int64')],engine)
    out=result['prepare'];assert [len(out[role].rows) for role in ('training','validation','test')]==[8,4,4]
    amount=out['training'].index('amount')
    assert [row[amount] for row in out['training'].rows]==[2.,4.,9.,8.,10.,9.,14.,16.]
    assert [row[amount] for row in out['validation'].rows]==[1000.,9.,2000.,3000.]
    assert [col['name'] for col in out['training'].columns]==['amount','label','observed_at','onehot:["region","north"]','onehot:["region","south"]']
    north=out['validation'].index('onehot:["region","north"]');south=out['validation'].index('onehot:["region","south"]')
    assert [(row[north],row[south]) for row in out['validation'].rows]==[(0.,0.),(0.,0.),(0.,1.),(0.,0.)]
