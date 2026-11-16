"""Bounded owned CSV staging, independent of retired lesson execution."""
from pathlib import Path
import pytest
from runner import learning_evidence as worker


def owner(tmp_path):
    root=tmp_path/'operation/input';root.mkdir(parents=True)
    protected=root/'assessment.json';protected.write_text('{}')
    return root,protected


def test_owned_csv_fixtures_live_under_request_owner_one_at_a_time(tmp_path):
    root,protected=owner(tmp_path);seen=[]
    for rows in [[[0,10],[2,12],[4,14]], [[3,9],[4,9],[5,9]]]:
        with worker.fixture_csv(rows,root) as fixture:
            assert fixture.parent==root and fixture.stat().st_size<=65536
            assert all(not previous.exists() for previous in seen)
            assert fixture.read_text().splitlines()[0]=='x,y'
            seen.append(fixture)
        assert not fixture.exists()
    assert set(root.iterdir())=={protected}


def test_owned_csv_fixture_bound_is_checked_before_any_write(tmp_path,monkeypatch):
    root,protected=owner(tmp_path)
    monkeypatch.setattr(worker,'MAX_FIXTURE_BYTES',1)
    with pytest.raises(ValueError,match='storage allowance'):
        with worker.fixture_csv([[0,10]],root):pytest.fail('Oversized fixture published')
    assert set(root.iterdir())=={protected}


def test_backend_failure_cleans_only_owned_csv_fixture(tmp_path):
    root,protected=owner(tmp_path)
    with pytest.raises(RuntimeError,match='fixture execution failed'):
        with worker.fixture_csv([[0,10]],root) as fixture:
            assert fixture.parent==root
            raise RuntimeError('fixture execution failed')
    assert not fixture.exists() and set(root.iterdir())=={protected}


def test_cleanup_preserves_foreign_fixture_replacement(tmp_path):
    root,protected=owner(tmp_path)
    with pytest.raises(RuntimeError,match='fixture replaced'):
        with worker.fixture_csv([[0,10]],root) as fixture:
            fixture.rename(fixture.with_suffix('.moved'))
            fixture.write_bytes(b'foreign fixture')
            raise RuntimeError('fixture replaced')
    assert fixture.read_bytes()==b'foreign fixture'
    assert protected.read_text()=='{}'
