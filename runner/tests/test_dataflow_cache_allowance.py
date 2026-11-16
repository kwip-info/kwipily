import hashlib
import json
import pytest
from runner import dataflow_cli as cli
from runner.tests.test_dataflow_cache import BASE

def run(output,**kw):
    return cli.run_dataflow(BASE/'preparation.json',{'source':str(BASE/'observations.csv')},output,lambda _:None,cache_policy='materialize',**kw)

def test_exact_materialized_allowance_and_reuse_refusal_preserve_prior(tmp_path):
    first=tmp_path/'first';baseline=run(first)
    size=sum(a['bytes'] for a in baseline['artifacts'] if a['kind'] in ('table','state'))
    exact=run(tmp_path/'exact',cache_limit_bytes=size)
    assert exact['cache']['materializedBytes']==size
    assert exact['cache']['materializedBytesLimit']==size
    with pytest.raises(ValueError,match='cache.*allowance'):
        run(tmp_path/'short',cache_limit_bytes=size-1)
    raw=(first/'manifest.json').read_bytes();pin=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError,match='cache.*allowance'):
        cli.run_dataflow(BASE/'preparation.json',{'source':str(BASE/'observations.csv')},tmp_path/'reuse',lambda _:None,
            cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin,cache_limit_bytes=size-1)
    assert (first/'manifest.json').read_bytes()==raw
    assert not (tmp_path/'short/manifest.json').exists() and not (tmp_path/'reuse/manifest.json').exists()
    assert not list(tmp_path.glob('.dataflow-*'))

@pytest.mark.parametrize('limit',[0,-1,200_000_001,True,1.5])
def test_bad_cache_allowance_refused_before_staging(tmp_path,limit):
    with pytest.raises(ValueError,match='allowance'):run(tmp_path/'bad',cache_limit_bytes=limit)
    assert not (tmp_path/'bad').exists()

def test_off_rejects_explicit_allowance(tmp_path):
    with pytest.raises(ValueError,match='materialize|reuse'):
        cli.run_dataflow(BASE/'preparation.json',{'source':str(BASE/'observations.csv')},tmp_path/'bad',lambda _:None,cache_limit_bytes=1)

def test_additive_allowance_evidence_rejects_malformed_pairs_and_accepts_old_manifest(tmp_path):
    from runner.dataflow_cache import Materialization
    first=tmp_path/'first';baseline=run(first)
    path=first/'manifest.json'
    for change in [{'materializedBytesLimit':0},{'materializedBytes':True},{'materializedBytes':200_000_001},{'materializedBytes':0}]:
        value=json.loads(json.dumps(baseline));value['cache'].update(change);raw=json.dumps(value).encode();path.write_bytes(raw)
        with pytest.raises(ValueError,match='allowance'):
            Materialization('reuse',path,hashlib.sha256(raw).hexdigest())
    value=json.loads(json.dumps(baseline));del value['cache']['materializedBytes'];raw=json.dumps(value).encode();path.write_bytes(raw)
    with pytest.raises(ValueError,match='allowance'):Materialization('reuse',path,hashlib.sha256(raw).hexdigest())
    del value['cache']['materializedBytesLimit'];raw=json.dumps(value).encode();path.write_bytes(raw)
    Materialization('reuse',path,hashlib.sha256(raw).hexdigest())


def test_actual_cli_accepts_allowance_and_refuses_off_and_other_operations(tmp_path):
    from runner.tests.test_dataflow_cache_cli import call,command,ROOT
    import sys
    p,events=call(command(tmp_path,tmp_path/'ok','--dataflow-cache-policy','materialize','--dataflow-cache-limit-bytes',200000000))
    assert p.returncode==0,events
    assert events[-1]['cache']['materializedBytesLimit']==200000000
    assert events[-1]['cache']['materializedBytes']>0
    for args in [command(tmp_path,tmp_path/'off','--dataflow-cache-limit-bytes',1),
                 [sys.executable,str(ROOT/'runner/main.py'),'--capabilities','--dataflow-cache-limit-bytes','1']]:
        p,events=call(args)
        assert p.returncode==1 and events[-1]['event']=='error'
    assert not (tmp_path/'off').exists()


def test_explicit_workload_qualifies_cache_allowance_and_reuse(tmp_path):
    from runner.workload_policy import WorkloadPolicy, workload_scope
    from runner.dataflow_cache import Materialization
    policy = WorkloadPolicy('balanced')
    large = 300_000_000
    with workload_scope(policy):
        result = run(tmp_path/'large-budget', cache_limit_bytes=large)
        assert result['cache']['materializedBytesLimit'] == large
        manifest = tmp_path/'large-budget/manifest.json'
        retained = manifest.read_bytes()
        reused = cli.run_dataflow(BASE/'preparation.json', {'source':str(BASE/'observations.csv')},
            tmp_path/'reused', lambda _:None, cache_policy='reuse', reuse_manifest=manifest,
            reuse_manifest_sha256=hashlib.sha256(retained).hexdigest(), cache_limit_bytes=large)
        assert reused['cache']['reusedNodeIDs']
        assert manifest.read_bytes() == retained
        with pytest.raises(ValueError, match='allowance'):
            run(tmp_path/'over-budget', cache_limit_bytes=policy.limit('retainedBytes')+1)
        assert not (tmp_path/'over-budget').exists()
    # A historical caller cannot grant itself a larger cache from an archive alone.
    with pytest.raises(ValueError, match='allowance|bytes exceeded'):
        Materialization('reuse', manifest, hashlib.sha256(retained).hexdigest(), cache_limit_bytes=large)


def test_omitted_cache_allowance_respects_narrower_captured_retention(tmp_path):
    from runner.workload_policy import WorkloadPolicy, workload_scope
    policy=WorkloadPolicy('balanced').overriding({'retainedBytes':100_000_000})
    with workload_scope(policy):
        result=run(tmp_path/'narrow-default')
        assert result['cache']['materializedBytesLimit']==100_000_000
        manifest=tmp_path/'narrow-default/manifest.json'
        cli.run_dataflow(BASE/'preparation.json',{'source':str(BASE/'observations.csv')},tmp_path/'narrow-reuse',lambda _:None,
            cache_policy='reuse',reuse_manifest=manifest,reuse_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
