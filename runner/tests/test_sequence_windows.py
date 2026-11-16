import copy
import pytest
from runner import dataflow as df
from runner.sequence_windows import DEFAULTS, apply, validate_config, verify_output
from runner.polars_preparation import preparation_engine


def table(rows,role='unassigned',ids=None):
    columns=({'name':'tokens','dtype':'int64Sequence','nullable':False},{'name':'doc','dtype':'string','nullable':False},{'name':'order','dtype':'int64','nullable':False})
    return df.derive(df.Table(columns,tuple(tuple(r) for r in rows),tuple(ids or [role+str(i) for i in range(len(rows))]),'a'*64,role,partition_digest='b'*64 if role!='unassigned' else None))

def config(**values):return {**DEFAULTS,'contextLength':3,'stride':3,**values}

@pytest.mark.parametrize('engine',['disk','polars'])
def test_exact_shift_stride_ids_and_owned_bounded_storage(engine):
    source=table([([2,3,4,5,6,7,8],'a',0)])
    with preparation_engine(engine):
        out=apply({'table':source},config(),'c'*64,lambda:None)['table']
        names=[c['name'] for c in out.columns];rows=list(out.rows)
        assert [r[names.index('input_ids')] for r in rows]==[[2,3,4],[5,6,7]]
        assert [r[names.index('target_ids')] for r in rows]==[[3,4,5],[6,7,8]]
        assert [r[names.index('start')] for r in rows]==[0,3]
        assert len(set(out.row_ids))==2 and out.source_digest==source.source_digest
        assert list(out.rows)==list(apply({'table':source},config(),'c'*64,lambda:None)['table'].rows)
        assert list(out.row_ids)==list(apply({'table':source},config(),'c'*64,lambda:None)['table'].row_ids)


def test_padding_counts_targets_not_inputs_and_never_empty_loss():
    out=apply({'table':table([([2,3,4],'a',0),([8],'b',0),([],'c',0)])},config(incomplete='pad',paddingID=1),'c'*64,lambda:None)['table']
    names=[c['name'] for c in out.columns];r=out.rows[0]
    assert len(out.rows)==1
    assert r[names.index('input_ids')]==[2,3,1]
    assert r[names.index('target_ids')]==[3,4,1]
    assert r[names.index('loss_mask')]==[1,1,0]
    assert r[names.index('valid_length')]==2

@pytest.mark.parametrize('stride,expected',[(1,[[2,3],[3,4],[4,5]]),(2,[[2,3],[4,5]]),(5,[[2,3]])])
def test_overlap_and_gaps_are_explicit(stride,expected):
    o=apply({'table':table([([2,3,4,5,6],'a',0)])},config(contextLength=2,stride=stride),'c'*64,lambda:None)['table']
    i=[c['name'] for c in o.columns].index('input_ids');assert [r[i] for r in o.rows]==expected


def test_grouping_uses_numeric_order_and_no_document_or_partition_crossing():
    c=config(mode='partitions',scope='document',documentColumn='doc',orderColumn='order')
    t=table([([5,6],'a',10),([2,3,4],'a',2),([8,9],'b',0)],'training')
    v=table([([7,8,9,10],'a',0)],'validation')
    o=apply({'training':t,'validation':v},c,'c'*64,lambda:None)
    names=[c['name'] for c in o['training'].columns];i=names.index('input_ids')
    assert [r[i] for r in o['training'].rows]==[[2,3,4]]
    assert [r[i] for r in o['validation'].rows]==[[7,8,9]]
    assert o['validation'].role=='validation' and o['validation'].partition_digest==v.partition_digest
    assert set(o['training'].row_ids).isdisjoint(o['validation'].row_ids)
    with pytest.raises(ValueError,match='order'):apply({'table':table([([1,2],'a',0),([3,4],'a',0)])},config(scope='document',documentColumn='doc',orderColumn='order'),'c'*64,lambda:None)
    reverse=apply({'table':t},config(scope='document',documentColumn='doc',orderColumn='order',direction='descending'),'c'*64,lambda:None)['table']
    assert reverse.rows[0][i]==[5,6,2]

@pytest.mark.parametrize('change',[{'contextLength':0},{'stride':False},{'paddingID':-1},{'inputColumn':'target_ids'},{'targetColumn':'document_id'},{'scope':'document'},{'direction':'shuffle'},{'extra':1}])
def test_configuration_refuses_invalid_bounds_and_collisions(change):
    with pytest.raises(ValueError):validate_config(config(**change))


def test_invalid_tokens_overlap_and_cancel_do_not_publish_outputs():
    for ids in ([True,2],[-1,2],[1.5,2],[65536,2]):
        with pytest.raises(ValueError):apply({'table':table([(ids,'a',0)])},config(),'c'*64,lambda:None)
    with pytest.raises(ValueError):apply({'training':table([([1,2],'a',0)],'training',['same']),'validation':table([([1,2],'a',0)],'validation',['same'])},config(mode='partitions'),'c'*64,lambda:None)
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):apply({'table':table([([1,2],'a',0)])},config(),'c'*64,cancel)


def test_each_configuration_output_name_and_cache_tamper():
    c=config(incomplete='pad',inputColumn='x',targetColumn='y',maskColumn='m',lengthColumn='n',paddingID=9)
    args={'table':table([([2,3,4],'a',0)])};o=apply(args,c,'c'*64,lambda:None)
    assert [x['name'] for x in o['table'].columns][-4:]==['x','y','n','m']
    verify_output(args,c,'c'*64,o,lambda:None)
    damaged=dict(o);t=o['table'];rows=[list(r) for r in t.rows];rows[0][-1]=[1,1,1]
    damaged['table']=df.derive(t,rows=rows)
    with pytest.raises(ValueError):verify_output(args,c,'c'*64,damaged,lambda:None)


def test_many_fragment_group_keeps_only_window_buffer_and_disk_ordering():
    source=table([([i%100],'a',i) for i in range(10000)])
    out=apply({'table':source},config(scope='document',documentColumn='doc',orderColumn='order',contextLength=128,stride=128),'c'*64,lambda:None)['table']
    assert len(out.rows)==78

@pytest.mark.parametrize('engine',['disk','polars'])
def test_window_population_owns_new_ids_from_prefixed_source_storage(engine):
    with preparation_engine(engine):
        source=table([([2,3,4,5],'a',0)])
        store=df.row_store(source.columns,lambda:None,identity_prefix='snapshot:')
        store.append(source.rows[0],'snapshot:0');rows,ids=store.finish()
        source=df.derive(source,rows=rows,ids=ids)
        out=apply({'table':source},config(),'c'*64,lambda:None)['table']
        assert len(out.rows)==1 and not out.row_ids[0].startswith('snapshot:')
        assert out.source_digest==source.source_digest
        assert out.rows[0][3]=='snapshot:0'
