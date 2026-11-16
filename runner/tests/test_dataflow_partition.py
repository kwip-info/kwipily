"""Three-way splitting invariants, independent assignments and clock boundaries."""
from dataclasses import replace
import pytest
from runner.dataflow import derive
from runner.dataflow_partition import validate_config
from runner import dataflow_partition_tool as current_partition

ROLES=("training","validation","test")

def current_config(operator,config):
    return {**{k:v for k,v in config.items() if k not in ("nullKeys","allocationUnit")},"strategy":"entity" if operator=="groupSplit" else "time"}

def partition_table(table,operator,config):
    return current_partition.apply(table,current_config(operator,config),lambda:None)

def verify_partition_output(table,operator,config,outputs):
    return current_partition.verify_output(table,current_config(operator,config),outputs,lambda:None)
from runner.tests.test_dataflow_join import table


def group(**changes):
    value={'groupBy':['key'],'nullKeys':'error','seed':42,'allocationUnit':'groups','validationPercent':20,'testPercent':20}
    value.update(changes);return value


def temporal(**changes):
    value={'timeColumn':'key','timeEncoding':'epochMilliseconds','trainingEnd':10,'validationEnd':20,
           'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'}
    value.update(changes);return value


def assert_partition(source,outputs,excluded=()):
    assert set(outputs)=={'training','validation','test','exclusions'}
    assert set(outputs['exclusions'].row_ids)==set(excluded)
    ids=[set(outputs[role].row_ids) for role in ('training','validation','test')]
    assert all(ids) and not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    assert set.union(*ids,set(excluded))==set(source.row_ids)
    assert all(outputs[role].role==role and outputs[role].partition_digest for role in ROLES)
    assert len({value.partition_digest for value in outputs.values()})==1


def test_group_split_preserves_entities_and_documents_group_not_row_allocation():
    source=table([(i,i*10+j) for i in range(10) for j in range(i+1)])
    outputs=partition_table(source,'groupSplit',group())
    assert_partition(source,outputs)
    keys={role:{row[0] for row in outputs[role].rows} for role in ROLES}
    assert [len(keys[r]) for r in ('training','validation','test')]==[6,2,2]
    assert not keys['training']&keys['validation'] and not keys['training']&keys['test'] and not keys['validation']&keys['test']
    for value in outputs.values():assert list(value.row_ids)==[i for i in source.row_ids if i in value.row_ids]
    verify_partition_output(source,'groupSplit',group(),outputs)


def test_group_membership_reorder_stable_seeded_and_feature_independent():
    source=table([(i,i) for i in range(30)])
    original=partition_table(source,'groupSplit',group())
    reverse=derive(source,rows=source.rows[::-1],ids=source.row_ids[::-1])
    reordered=partition_table(reverse,'groupSplit',group())
    changed=partition_table(derive(source,rows=[(r[0],r[1]+99) for r in source.rows]),'groupSplit',group())
    for role in original:
        assert set(original[role].row_ids)==set(reordered[role].row_ids)==set(changed[role].row_ids)
    different=partition_table(source,'groupSplit',group(seed=43))
    assert any(set(original[r].row_ids)!=set(different[r].row_ids) for r in original)


def test_largest_remainder_ties_and_too_few_groups():
    source=table([(i,i) for i in range(4)])
    outputs=partition_table(source,'groupSplit',group(validationPercent=25,testPercent=25))
    assert [len(outputs[r].rows) for r in ('training','validation','test')]==[2,1,1]
    with pytest.raises(ValueError,match='nonempty'):partition_table(table([(1,1),(2,2)]),'groupSplit',group())
    with pytest.raises(ValueError,match='missing'):partition_table(table([(None,1),(1,2),(2,3)]),'groupSplit',group())


def test_temporal_cutoffs_ties_and_gap_exclusions_are_exact():
    source=table([(21,1),(10,2),(11,3),(12,4),(20,5),(22,6),(23,7),(10,8),(19,9)])
    outputs=partition_table(source,'temporalSplit',temporal(embargoMilliseconds=2))
    assert [r[0] for r in outputs['training'].rows]==[10,10]
    assert [r[0] for r in outputs['validation'].rows]==[19,20]
    assert [r[0] for r in outputs['test'].rows]==[23]
    excluded=[source.row_ids[i] for i in (0,2,3,5)]
    assert_partition(source,outputs,excluded)
    info=outputs['training'].diagnostics[-1]
    assert info['excludedRows']==4
    assert info['inputRows']==len(source.rows)
    assert info['inputSourceDigest']==source.source_digest
    assert sum(item['rows'] for item in info['partitions'])+info['excludedRows']==info['inputRows']
    verify_partition_output(source,'temporalSplit',temporal(embargoMilliseconds=2),outputs)


def test_iso_equivalent_offsets_and_submillisecond_precision():
    source=table([('2026-01-01T01:00:00+01:00',1),('2026-01-01T00:00:00.000001Z',2),
                  ('2026-01-01T00:00:00.000002Z',3)],dtypes=['string','int64'])
    config=temporal(timeEncoding='iso8601Offset',trainingEnd='2026-01-01T00:00:00Z',validationEnd='2026-01-01T00:00:00.000001Z')
    outputs=partition_table(source,'temporalSplit',config)
    assert [len(outputs[r].rows) for r in ROLES]==[1,1,1]
    assert_partition(source,outputs)


def test_entity_isolation_is_independent_of_chronology():
    source=table([(1,7),(15,7),(30,8)])
    with pytest.raises(ValueError,match='entit'):partition_table(source,'temporalSplit',temporal(entityColumns=['value'],entityPolicy='requireDisjoint'))
    outputs=partition_table(source,'temporalSplit',temporal(entityColumns=['value']))
    assert_partition(source,outputs)
    source=table([(1,7),(15,None),(30,8)])
    with pytest.raises(ValueError,match='missing'):partition_table(source,'temporalSplit',temporal(entityColumns=['value']))


@pytest.mark.parametrize('timestamp',['2026-01-01T00:00:00','2026-01-01T00:00:60Z','2026-01-01T00:00:00.1234567Z','2026-01-01T00:00:00+24:00','0001-01-01T00:00:00+00:01'])
def test_invalid_iso_cutoffs_are_rejected(timestamp):
    with pytest.raises(ValueError):validate_config('temporalSplit',temporal(timeEncoding='iso8601Offset',trainingEnd=timestamp,validationEnd='2026-01-02T00:00:00Z'))


@pytest.mark.parametrize('op,config',[
    ('groupSplit',group(groupBy=[])),('groupSplit',group(seed=True)),('groupSplit',group(validationPercent=50,testPercent=50)),
    ('temporalSplit',temporal(trainingEnd=True)),('temporalSplit',temporal(trainingEnd=20)),
    ('temporalSplit',temporal(entityPolicy='requireDisjoint')),('temporalSplit',temporal(embargoMilliseconds=-1))])
def test_invalid_configs_rejected(op,config):
    with pytest.raises(ValueError):validate_config(op,config)


def test_partition_and_fitted_sources_rejected_and_cache_forgery_detected():
    source=table([(i,i) for i in range(10)])
    for update in ({'role':'test'},{'partition_digest':'x'},{'fitted_states':('x',)}):
        with pytest.raises(ValueError,match='before fitted'):partition_table(replace(source,**update),'groupSplit',group())
    outputs=partition_table(source,'groupSplit',group())
    wrong=dict(outputs);wrong['test']=replace(wrong['test'],row_ids=('forged',)*len(wrong['test'].rows))
    with pytest.raises(ValueError,match='Cached Partition'):verify_partition_output(source,'groupSplit',group(),wrong)


def test_fractional_count_tie_awards_validation_before_test():
    outputs=partition_table(table([(i,i) for i in range(7)]),'groupSplit',group())
    # 4.2/1.4/1.4 groups: final tied remainder goes to validation.
    assert [len(outputs[r].rows) for r in ('training','validation','test')]==[4,2,1]


def test_float_keys_normalize_equal_values_without_rounding_large_integers():
    source=table([(0.0,1),(-0.0,2),(2**53,3),(2**53+1,4)],dtypes=['float64','int64'])
    outputs=partition_table(source,'groupSplit',group(validationPercent=33,testPercent=33))
    assert_partition(source,outputs)
    assert sorted(len(outputs[r].rows) for r in ROLES)==[1,1,2]
    assert any([row[1] for row in value.rows]==[1,2] for value in outputs.values())


def test_missing_entity_on_excluded_row_is_not_silently_ignored():
    source=table([(1,1),(11,None),(15,2),(30,3)])
    with pytest.raises(ValueError,match='missing'):
        partition_table(source,'temporalSplit',temporal(embargoMilliseconds=2,entityColumns=['value']))


def test_temporal_order_ties_use_identity_and_leave_input_unchanged():
    source=table([(30,3),(1,1),(15,2),(15,9)])
    before=replace(source)
    outputs=partition_table(source,'temporalSplit',temporal())
    assert list(outputs['validation'].row_ids)==sorted((source.row_ids[2],source.row_ids[3]))
    assert source==before and source.role=='unassigned' and source.diagnostics==()


def test_partition_evidence_binds_parent_source_even_with_same_rows_and_ids():
    source=table([(i,i) for i in range(10)])
    original=partition_table(source,'groupSplit',group())
    other=partition_table(replace(source,source_digest='b'*64),'groupSplit',group())
    assert list(original['training'].row_ids)==list(other['training'].row_ids)
    assert original['training'].partition_digest!=other['training'].partition_digest
