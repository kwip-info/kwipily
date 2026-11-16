"""Independent SQL and arithmetic checks for grouped aggregation."""
import sqlite3
from dataclasses import replace
import pytest
from runner.dataflow import derive
from runner.dataflow_aggregate import aggregate_table, validate_config, verify_aggregate_output
from runner.tests.test_dataflow_join import table


def config(**changes):
    value={'groupBy':['key'],'nullKeys':'group','aggregations':[
        {'output':'n','op':'countRows'},
        {'output':'present','op':'countNonNull','column':'value'},
        *({'output':r,'op':r,'column':'value','nullPolicy':'skip'} for r in ('sum','mean','min','max'))],
        'maxRows':100000}
    value.update(changes);return value


def test_grouped_values_match_sql_with_nulls_and_first_occurrence_order():
    source=table([(2,10),(1,3),(2,None),(None,8),(1,9),(None,4)])
    actual=aggregate_table(source,config())
    db=sqlite3.connect(':memory:')
    try:
        db.execute('create table t(k integer,v integer)')
        db.executemany('insert into t values (?,?)',source.rows)
        expected=db.execute('select k,count(*),count(v),sum(v),avg(v),min(v),max(v) from t group by k order by min(rowid)').fetchall()
    finally:db.close()
    assert actual.rows==tuple(expected)
    assert [c['dtype'] for c in actual.columns]==['int64','int64','int64','int64','float64','int64','int64']
    verify_aggregate_output(source,config(),actual)


def test_null_key_policies_and_global_summary():
    source=table([(1,4),(None,8)])
    assert aggregate_table(source,config(nullKeys='drop')).rows[0][1]==1
    with pytest.raises(ValueError,match='missing group'):aggregate_table(source,config(nullKeys='error'))
    assert aggregate_table(source,config(groupBy=[])).rows==((2,2,12,6.0,4,8),)
    with pytest.raises(ValueError,match='no rows'):aggregate_table(table([(None,3)]),config(nullKeys='drop'))


def test_missing_reducer_values_are_explicit():
    source=table([(1,None)])
    assert aggregate_table(source,config()).rows==((1,1,0,None,None,None,None),)
    source=table([(1,None),(1,4)])
    for reducer in ('sum','mean','min','max'):
        field={'output':'out','op':reducer,'column':'value','nullPolicy':'propagate'}
        assert aggregate_table(source,config(aggregations=[field])).rows==((1,None),)
        with pytest.raises(ValueError,match='missing reducer'):
            aggregate_table(source,config(aggregations=[{**field,'nullPolicy':'error'}]))


def test_identity_and_reductions_are_stable_under_reordering():
    source=table([(1,1e16),(1,1.0),(1,-1e16),(2,8.0)],dtypes=['int64','float64'])
    result=aggregate_table(source,config())
    reordered=derive(source,rows=source.rows[::-1],ids=source.row_ids[::-1])
    other=aggregate_table(reordered,config())
    assert dict(zip(result.row_ids,result.rows))==dict(zip(other.row_ids,other.rows))
    assert result.rows[0][3]==1.0
    changed=derive(source,rows=[(1,2e16),*source.rows[1:]])
    assert aggregate_table(changed,config()).row_ids==result.row_ids
    reduced=derive(source,rows=source.rows[:-1],ids=source.row_ids[:-1])
    assert aggregate_table(reduced,config()).row_ids[0]==result.row_ids[0]
    removed=derive(source,rows=source.rows[1:],ids=source.row_ids[1:])
    assert aggregate_table(removed,config()).row_ids[0]!=result.row_ids[0]


def test_numeric_key_normalization_and_stable_large_mean():
    source=table([(0.0,1.0),(-0.0,2.0),(1,3.0),(1.0,4.0)],dtypes=['float64','float64'])
    result=aggregate_table(source,config())
    assert len(result.rows)==2
    reverse=derive(source,rows=source.rows[::-1],ids=source.row_ids[::-1])
    assert set(aggregate_table(reverse,config()).row_ids)==set(result.row_ids)
    mean=config(groupBy=[],aggregations=[{'output':'m','op':'mean','column':'value','nullPolicy':'skip'}])
    assert aggregate_table(table([(1,1e308),(1,1e308)],dtypes=['int64','float64']),mean).rows==((1e308,),)


def test_overflow_types_and_partition_boundaries():
    with pytest.raises(ValueError,match='overflow'):aggregate_table(table([(1,2**63-1),(1,1)]),config())
    for changed in ({'role':'training'},{'partition_digest':'x'},{'fitted_states':('x',)}):
        with pytest.raises(ValueError,match='before splitting'):aggregate_table(replace(table([(1,2)]),**changed),config())
    with pytest.raises(ValueError,match='numeric'):aggregate_table(table([(1,True)],dtypes=['int64','bool']),config())
    field={'output':'lo','op':'min','column':'value','nullPolicy':'skip'}
    assert aggregate_table(table([(1,'b'),(1,'a')],dtypes=['int64','string']),config(aggregations=[field])).rows==((1,'a'),)


@pytest.mark.parametrize('change',[{'groupBy':['key','key']},{'nullKeys':'guess'},{'maxRows':True},
    {'aggregations':[]},{'aggregations':[{'output':'x','op':'countRows','column':'value'}]},
    {'aggregations':[{'output':'x','op':'sum','column':'value'}]},
    {'aggregations':[{'output':'key','op':'countRows'}]},{'unknown':1}])
def test_configuration_rejects_ambiguous_fields(change):
    with pytest.raises(ValueError):validate_config(config(**change))


def test_cached_output_verification_rejects_forged_values_or_membership():
    source=table([(1,2),(1,3)]);result=aggregate_table(source,config())
    for forged in (derive(result,rows=[(1,999,*result.rows[0][2:])]),replace(result,row_ids=('forged',)),replace(result,role='training')):
        with pytest.raises(ValueError,match='Cached aggregate'):verify_aggregate_output(source,config(),forged)


def test_float_schema_admitted_integers_do_not_merge_lossily():
    source=table([(2**53,1),(2**53+1,2)],dtypes=['float64','int64'])
    result=aggregate_table(source,config())
    assert len(result.rows)==2
    assert [row[0] for row in result.rows]==[2**53,2**53+1]


def test_diagnostic_null_counts_exclude_dropped_key_rows():
    result=aggregate_table(table([(1,None),(1,3),(None,None)]),config(nullKeys='drop'))
    info=result.diagnostics[-1]
    assert info['droppedKeyRows']==1
    assert info['reducers'][0]=={'output':'n','op':'countRows','nullInputRows':0}
    assert all(item['nullInputRows']==1 for item in info['reducers'][1:])


def text_field(**changes):
    field={'output':'items','op':'joinText','column':'value','nullPolicy':'skip','separator':'\n','distinct':False,'skipEmpty':False}
    field.update(changes);return field


def test_join_text_preserves_input_order_and_multiline_separator():
    source=table([(1,'second'),(2,'only'),(1,'first'),(1,'second'),(1,'last')],dtypes=['int64','string'])
    c=config(aggregations=[text_field()])
    result=aggregate_table(source,c)
    assert result.rows==((1,'second\nfirst\nsecond\nlast'),(2,'only'))
    assert result.columns[1]['dtype']=='string'
    reordered=derive(source,rows=source.rows[::-1],ids=source.row_ids[::-1])
    other=aggregate_table(reordered,c)
    assert other.rows[0]==(1,'last\nsecond\nfirst\nsecond')
    verify_aggregate_output(source,c,result)
    assert aggregate_table(source,config(aggregations=[text_field(distinct=True,separator=' | ')])).rows[0]==(1,'second | first | last')


def test_join_text_null_and_empty_policies():
    source=table([(1,None),(1,''),(1,'a'),(1,'a')],dtypes=['int64','string'])
    assert aggregate_table(source,config(aggregations=[text_field()])).rows==((1,'\na\na'),)
    assert aggregate_table(source,config(aggregations=[text_field(skipEmpty=True,distinct=True)])).rows==((1,'a'),)
    assert aggregate_table(source,config(aggregations=[text_field(nullPolicy='propagate')])).rows==((1,None),)
    with pytest.raises(ValueError,match='missing reducer'):aggregate_table(source,config(aggregations=[text_field(nullPolicy='error')]))
    assert aggregate_table(table([(1,None)],dtypes=['int64','string']),config(aggregations=[text_field()])).rows==((1,None),)
    assert aggregate_table(table([(1,'')],dtypes=['int64','string']),config(aggregations=[text_field(skipEmpty=True)])).rows==((1,''),)


def test_join_text_rejects_nontext_and_oversized_join_before_allocating(monkeypatch):
    with pytest.raises(ValueError,match='text'):aggregate_table(table([(1,2)]),config(aggregations=[text_field()]))
    import runner.dataflow_aggregate as module
    monkeypatch.setattr(module,'MAX_ARTIFACT_BYTES',2000)
    with pytest.raises(ValueError,match='byte'):aggregate_table(table([(1,'x'*1500),(1,'y'*1500)],dtypes=['int64','string']),config(aggregations=[text_field()]))


@pytest.mark.parametrize('change',[{'separator':4},{'separator':'x'*1025},{'distinct':1},{'skipEmpty':'yes'}])
def test_join_text_invalid_options_refused(change):
    with pytest.raises(ValueError):validate_config(config(aggregations=[text_field(**change)]))


def test_text_join_budget_is_exact_escaped_artifact_size():
    import runner.dataflow_aggregate as module
    source=table([(1,'é\n"'),(1,'🚀'),(2,None)],dtypes=['int64','string'])
    c=config(aggregations=[text_field(separator='\t|\n')])
    output=aggregate_table(source,c);plan=module.plan_aggregate(source,c)
    lineage={**output.evidence(),'dataVersion':output.data_version,'partitionDigest':output.partition_digest,
             'fittedStateDigests':list(output.fitted_states),'diagnostics':list(output.diagnostics)}
    artifact={'schemaVersion':1,'kind':'table','columns':output.columns,'rows':output.rows,'lineage':lineage}
    assert plan.artifact_bytes==len(module.encoded(artifact))
    verify_aggregate_output(source,c,output)
