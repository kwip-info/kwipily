"""Ready admission and effect barriers, independent of machine-speed assertions."""
import pytest
from runner.preparation_scheduler import StableReadyQueue


def test_stable_bounded_ready_groups_do_not_cross_a_dependency_or_effect_barrier():
    queue=StableReadyQueue(['source','left','right','write','after'],{
        'source':set(),'left':{'source'},'right':{'source'},'write':{'left','right'},'after':{'write'}})
    pure=lambda node:node in ('left','right','after')
    assert queue.take(pure,2)==('source',)
    with pytest.raises(ValueError):queue.complete('left')
    queue.complete('source')
    assert queue.take(pure,2)==('left','right')
    queue.complete('right')
    assert queue.take(pure,2)==()
    queue.complete('left')
    assert queue.take(pure,2)==('write',)
    queue.complete('write')
    assert queue.take(pure,2)==('after',)
    queue.complete('after')
    assert queue.finished
    with pytest.raises(ValueError):queue.complete('after')


def test_effect_barrier_waits_for_an_inflight_independent_pure_branch():
    queue=StableReadyQueue(['left','write','right'],{n:set() for n in ('left','write','right')})
    assert queue.take(lambda n:n!='write',2)==('left',)
    assert queue.take(lambda n:n!='write',2)==()
    queue.complete('left')
    assert queue.take(lambda n:n!='write',2)==('write',)
    assert queue.take(lambda n:n!='write',2)==()
    queue.complete('write')
    assert queue.take(lambda n:n!='write',2)==('right',)
    queue.complete('right')
    assert queue.finished


@pytest.mark.parametrize('bad',[0,-1,True,1.5])
def test_invalid_reservation_is_rejected(bad):
    queue=StableReadyQueue(['n'],{'n':set()})
    with pytest.raises(ValueError):queue.take(lambda n:True,bad)


@pytest.mark.parametrize('failure',[False,True])
@pytest.mark.parametrize('branches',[2,4])
def test_real_native_branches_overlap_and_drain_before_effects(tmp_path,failure,branches):
    """Fresh captured CPU pools; a barrier makes concurrency admission deterministic.

    This proves overlapping real node lifetimes and exact outputs, not a speedup
    or simultaneous Python-bytecode execution. No dataset is returned in stdout.
    """
    import json,subprocess,sys
    from pathlib import Path
    script=Path(__file__).with_name('fixtures')/'native_ready_group.py'
    result=subprocess.run([sys.executable,str(script),str(tmp_path),str(int(failure)),str(branches)],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-3000:]
    proof=json.loads(result.stdout)
    assert proof['maximumConcurrentNodeLifetimes']==branches and proof['activeAfterExit']==0
    assert proof['polarsPoolThreads']==(1 if branches==2 else 5) and proof['capturedParallelTasks']==4
    assert proof['failureInjected']==failure
    assert proof['externalFiles']==(0 if failure else branches)
