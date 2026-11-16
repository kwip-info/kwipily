"""Pinned launch claims and delayed/duplicate races using real local descriptors."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]


def uid(): return str(uuid.uuid4()).upper()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
def identity(path):
    info = path.stat()
    return {'device': info.st_dev & 0xffffffff, 'inode': info.st_ino}


@pytest.fixture
def launch_fixture(tmp_path):
    workspace = tmp_path.resolve()
    project, group, generation, run, directory, nonce = [uid() for _ in range(6)]
    folder = workspace / 'workflow-experiments' / project / group
    folder.mkdir(parents=True, mode=0o700)
    for name in ['owner.lock', 'activity.lock']:
        (folder / name).touch(mode=0o600)
    metadata = dict(schemaVersion=1, groupID=group, generation=generation,
                    directoryIdentity=identity(folder), ownerLockIdentity=identity(folder/'owner.lock'),
                    activityLockIdentity=identity(folder/'activity.lock'), createdAtUnixSeconds=1.0)
    raw = encoded(metadata); (folder/'ownership.json').write_bytes(raw)
    phase = workspace / 'workflow-runs' / project / directory / 'preparation'
    phase.mkdir(parents=True)
    (phase/'request.json').write_text('{}')
    (phase/'bindings.json').write_text('{}')
    intent = dict(schemaVersion=1, projectID=project, groupID=group, generation=generation,
                  runID=run, directoryID=directory, nonce=nonce, ordinal=1,
                  requestIdentity='a'*64, ownershipSHA256=digest(raw), phase='preparation',
                  arguments=['--execute-dataflow', str(phase/'request.json'), '--source-bindings',
                             str(phase/'bindings.json'), '--output', str(phase/'artifacts')])
    ledger = dict(schemaVersion=1, groupID=group, projectID=project, train={'domain':'control','localID':'train'},
                  sourceRequestIdentity='b'*64, inputsManifestSHA256='c'*64, authoringSHA256='d'*64,
                  revision=1, stopRequested=False, variants=[dict(ordinal=1, requestIdentity='a'*64,
                  status='running', runID=run, directoryID=directory)])
    (folder/'group.json').write_bytes(encoded(ledger))
    (folder/f'launch-{nonce}.json').write_bytes(encoded(intent))
    (folder/f'phase-{run}-preparation.json').write_bytes(encoded(intent))
    return folder, intent, metadata, ledger


def claim(f):
    from runner.worker_authorization import claim_launch
    folder, intent, _, _ = f
    raw = (folder/f"launch-{intent['nonce']}.json").read_bytes()
    return claim_launch(str(folder), intent['nonce'], digest(raw), digest((folder/'ownership.json').read_bytes()))


def test_claim_holds_activity_and_check_rejects_changed_ledger(launch_fixture):
    folder, intent, _, ledger = launch_fixture
    with claim(launch_fixture) as authority:
        assert authority.intent == intent
        authority.check()
        contender = os.open(folder/'activity.lock', os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError): fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally: os.close(contender)
        record = json.loads((folder/f"claim-{intent['nonce']}.json").read_bytes())
        assert record == dict(schemaVersion=1, nonce=intent['nonce'], intentSHA256=digest(encoded(intent)), state='claimed')
        ledger['stopRequested'] = True
        (folder/'group.json').write_bytes(encoded(ledger))
        with pytest.raises((ValueError, OSError)): authority.check()
    contender = os.open(folder/'activity.lock', os.O_RDWR)
    try: fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally: os.close(contender)
    with pytest.raises((ValueError, OSError)): claim(launch_fixture)


@pytest.mark.parametrize('mutation', ['metadata', 'lock', 'schema', 'extra', 'phase', 'output', 'equals',
    'reserved', 'stop', 'variant', 'duplicate', 'huge', 'booleanOrdinal', 'generation', 'symlinkPhase'])
def test_invalid_launch_never_creates_claim(launch_fixture, mutation):
    folder, intent, metadata, ledger = launch_fixture
    if mutation == 'metadata':
        metadata['extra'] = True; (folder/'ownership.json').write_bytes(encoded(metadata))
    elif mutation == 'lock':
        (folder/'activity.lock').rename(folder/'old-activity.lock'); (folder/'activity.lock').touch()
    elif mutation == 'schema': intent['schemaVersion'] = True
    elif mutation == 'extra': intent['unexpected'] = 1
    elif mutation == 'phase': intent['phase'] = 'training'
    elif mutation == 'output': intent['arguments'][-1] = str(folder/'escape')
    elif mutation == 'equals': intent['arguments'][-2:] = ['--output='+intent['arguments'][-1]]
    elif mutation == 'reserved': intent['arguments'] += ['--kwip-activity-fd', '99']
    elif mutation == 'stop': ledger['stopRequested'] = 0
    elif mutation == 'variant': ledger['variants'][0]['status'] = 'pending'
    elif mutation == 'duplicate': ledger['variants'].append(dict(ledger['variants'][0]))
    elif mutation == 'huge': intent['arguments'] += ['x'*65537]
    elif mutation == 'booleanOrdinal': intent['ordinal'] = True
    elif mutation == 'generation': intent['generation'] = uid()
    elif mutation == 'symlinkPhase':
        phase=Path(intent['arguments'][1]).parent
        target=phase.with_name('elsewhere'); phase.rename(target); phase.symlink_to(target, target_is_directory=True)
    (folder/f"launch-{intent['nonce']}.json").write_bytes(encoded(intent))
    (folder/f"phase-{intent['runID']}-{intent['phase']}.json").write_bytes(encoded(intent))
    (folder/'group.json').write_bytes(encoded(ledger))
    with pytest.raises((ValueError, OSError)): claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


def test_supplied_hashes_and_symlink_ancestor_rejected(launch_fixture, tmp_path):
    from runner.worker_authorization import claim_launch
    folder, intent, _, _ = launch_fixture
    pin=digest(encoded(intent)); ownership=digest((folder/'ownership.json').read_bytes())
    for wanted, owner in [('0'*64, ownership), (pin, '0'*64)]:
        with pytest.raises((ValueError, OSError)): claim_launch(str(folder), intent['nonce'], wanted, owner)
    alias=tmp_path/'alias';alias.symlink_to(folder.parent, target_is_directory=True)
    with pytest.raises((ValueError, OSError)): claim_launch(str(alias/folder.name),intent['nonce'],pin,ownership)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


def child_claim(f):
    folder,intent,_,_=f
    script='''
import sys
sys.path.insert(0,sys.argv[1])
from worker_authorization import claim_launch
try:
 with claim_launch(*sys.argv[2:]) as authority: authority.check(); print('CLAIMED',flush=True)
except (ValueError,OSError): print('REJECTED',flush=True);sys.exit(3)
'''
    return subprocess.Popen([sys.executable,'-I','-c',script,str(ROOT/'runner'),str(folder),intent['nonce'],
                             digest(encoded(intent)),digest((folder/'ownership.json').read_bytes())],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def test_two_real_processes_choose_one_claim_winner(launch_fixture):
    processes=[]
    try:
        processes=[child_claim(launch_fixture),child_claim(launch_fixture)]
        results=[p.communicate(timeout=8) for p in processes]
        assert sorted(p.returncode for p in processes)==[0,3],results
        assert sorted(out for out,_ in results)==[b'CLAIMED\n',b'REJECTED\n']
    finally:
        for p in processes:
            if p.poll() is None:p.kill()
            p.communicate(timeout=8)


def test_recovery_exclusive_revoke_blocks_delayed_bootstrap(launch_fixture):
    folder,intent,_,_=launch_fixture
    owner=os.open(folder/'owner.lock',os.O_RDWR);activity=os.open(folder/'activity.lock',os.O_RDWR)
    try:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(activity,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rejected=child_claim(launch_fixture)
        try:
            output,error=rejected.communicate(timeout=8)
            assert rejected.returncode==3,(output,error)
        finally:
            if rejected.poll() is None:rejected.kill();rejected.communicate(timeout=8)
        (folder/f"claim-{intent['nonce']}.json").write_bytes(encoded(dict(schemaVersion=1,nonce=intent['nonce'],intentSHA256=digest(encoded(intent)),state='revoked')))
    finally:os.close(activity);os.close(owner)
    delayed=child_claim(launch_fixture)
    try:
        output,error=delayed.communicate(timeout=8)
        assert delayed.returncode==3,(output,error)
        assert output==b'REJECTED\n'
    finally:
        if delayed.poll() is None:delayed.kill();delayed.communicate(timeout=8)


def test_failed_claims_and_explicit_close_do_not_leak_descriptors(launch_fixture):
    from runner.tests.test_worker_supervisor import descriptor_snapshot
    before=descriptor_snapshot()
    authority=claim(launch_fixture);authority.close();authority.close()
    for _ in range(20):
        with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert descriptor_snapshot()==before


@pytest.mark.parametrize('mutation',['freshNonce','partialSlot','mismatchedSlot','missingSlot'])
def test_phase_slot_prevents_new_nonce_and_incomplete_publication(launch_fixture,mutation):
    folder,intent,_,_=launch_fixture
    slot=folder/f"phase-{intent['runID']}-preparation.json"
    if mutation=='freshNonce':
        intent['nonce']=uid()
        (folder/f"launch-{intent['nonce']}.json").write_bytes(encoded(intent))
    elif mutation=='partialSlot':slot.write_bytes(b'{')
    elif mutation=='mismatchedSlot':slot.write_bytes(encoded({**intent,'nonce':uid()}))
    else:slot.unlink()
    with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


@pytest.mark.parametrize('arguments',[['--out','elsewhere'],['--out=elsewhere'],['--grap','elsewhere'],
    ['--capabilities'],['--graph','elsewhere'],['--output','elsewhere'],['--dataflow-cache-policy'],
    ['--dataflow-cache-policy','--output'],['positional']])
def test_noncanonical_or_contradictory_options_never_claim(launch_fixture,arguments):
    folder,intent,_,_=launch_fixture
    intent['arguments']+=arguments
    raw=encoded(intent)
    (folder/f"launch-{intent['nonce']}.json").write_bytes(raw)
    (folder/f"phase-{intent['runID']}-preparation.json").write_bytes(raw)
    with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


@pytest.mark.parametrize('location',['project','run','phase','request','bindings','output'])
def test_actual_run_path_components_cannot_redirect_to_symlinks(launch_fixture,location):
    folder,intent,_,_=launch_fixture
    phase=Path(intent['arguments'][1]).parent
    chosen={'project':phase.parent.parent,'run':phase.parent,'phase':phase,
            'request':phase/'request.json','bindings':phase/'bindings.json','output':phase/'artifacts'}[location]
    if location=='output':chosen.mkdir()
    destination=chosen.with_name(chosen.name+'-moved')
    chosen.rename(destination);chosen.symlink_to(destination,target_is_directory=destination.is_dir())
    with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


def test_training_requires_matching_claimed_preparation_slot(launch_fixture):
    folder,intent,_,_=launch_fixture
    preparation=dict(intent);prep_nonce=intent['nonce']
    phase=Path(intent['arguments'][1]).parent.parent/'training'
    phase.mkdir();(phase/'request.json').write_text('{}')
    intent.update(nonce=uid(),phase='training',arguments=['--graph',str(phase/'request.json'),'--output',str(phase/'artifacts')])
    raw=encoded(intent)
    (folder/f"launch-{intent['nonce']}.json").write_bytes(raw)
    (folder/f"phase-{intent['runID']}-training.json").write_bytes(raw)
    with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()
    (folder/f"claim-{prep_nonce}.json").write_bytes(encoded(dict(schemaVersion=1,nonce=prep_nonce,intentSHA256=digest(encoded(preparation)),state='claimed')))
    with claim(launch_fixture) as authority:
        assert authority.intent['phase']=='training'
        authority.check()


@pytest.mark.parametrize('name', ['owner.lock', 'activity.lock', 'ownership.json', 'intent', 'slot', 'request', 'bindings'])
@pytest.mark.parametrize('mutation', ['hardlink', 'writable'])
def test_owned_launch_files_reject_alias_links_and_unsafe_write_permissions(launch_fixture, name, mutation):
    folder,intent,_,_=launch_fixture
    phase=Path(intent['arguments'][1]).parent
    paths={'intent':folder/f"launch-{intent['nonce']}.json",
           'slot':folder/f"phase-{intent['runID']}-preparation.json",
           'request':phase/'request.json','bindings':phase/'bindings.json'}
    target=paths.get(name,folder/name)
    if mutation=='hardlink':os.link(target,target.with_name(target.name+'.alias'))
    else:target.chmod(0o666)
    with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


@pytest.mark.parametrize('name', ['owner.lock','activity.lock','claim'])
@pytest.mark.parametrize('mutation', ['hardlink','writable'])
def test_authority_check_rejects_new_alias_or_writable_protocol_file(launch_fixture,name,mutation):
    folder,intent,_,_=launch_fixture
    with claim(launch_fixture) as authority:
        target=folder/(f"claim-{intent['nonce']}.json" if name=='claim' else name)
        if mutation=='hardlink':os.link(target,target.with_name(target.name+'.alias'))
        else:target.chmod(0o666)
        with pytest.raises((ValueError,OSError)):authority.check()


def test_launch_ledger_is_bounded_to_256000_bytes(launch_fixture):
    folder,intent,_,_=launch_fixture
    ledger=folder/'group.json'
    content=ledger.read_bytes()
    ledger.write_bytes(content+b' '*(256001-len(content)))
    with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


@pytest.mark.parametrize('mutation',['reordered','gap','unknownStatus'])
def test_ledger_variants_require_contiguous_order_and_known_statuses(launch_fixture,mutation):
    folder,intent,_,ledger=launch_fixture
    other=dict(ordinal=2,requestIdentity='f'*64,status='pending')
    if mutation=='reordered':ledger['variants']=[other,ledger['variants'][0]]
    elif mutation=='gap':ledger['variants'].append({**other,'ordinal':3})
    else:ledger['variants'].append({**other,'status':'mystery'})
    (folder/'group.json').write_bytes(encoded(ledger))
    with pytest.raises((ValueError,OSError)):claim(launch_fixture)
    assert not (folder/f"claim-{intent['nonce']}.json").exists()


def test_preparation_retention_flag_is_authorized_and_sealed(launch_fixture):
    folder, intent, _, _ = launch_fixture
    intent['arguments'].append('--retain-partial-dataflow')
    (folder/f"launch-{intent['nonce']}.json").write_bytes(encoded(intent))
    (folder/f"phase-{intent['runID']}-preparation.json").write_bytes(encoded(intent))
    with claim(launch_fixture) as authority:
        assert authority.intent['arguments'][-1] == '--retain-partial-dataflow'
        authority.check()


@pytest.mark.parametrize('mode', ['duplicate', 'training'])
def test_retention_flag_cannot_repeat_or_enter_training(launch_fixture, mode):
    from runner.worker_authorization import validate_intent
    _, intent, _, _ = launch_fixture
    if mode == 'training':
        intent['phase'] = 'training'
        intent['arguments'] = ['--graph', 'request.json', '--output', 'artifacts', '--retain-partial-dataflow']
    else:
        intent['arguments'] += ['--retain-partial-dataflow', '--retain-partial-dataflow']
    with pytest.raises(ValueError): validate_intent(intent)
