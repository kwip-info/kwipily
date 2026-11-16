import hashlib
import json
from pathlib import Path
import pytest
from scripts.required_model_qualification import verify_fixtures, verify_python_result, verify_swift_result


def fixture(tmp_path):
    model=tmp_path/'model';model.mkdir();weights=model/'weights';weights.write_bytes(b'qualified bytes')
    spec={'MODEL':{'kind':'directory','requiredFiles':['weights']}}
    manifest={'MODEL':{'path':str(model),'files':{'weights':hashlib.sha256(weights.read_bytes()).hexdigest()}}}
    return spec,manifest,weights


def test_fixture_must_be_present_pinned_and_unchanged(tmp_path):
    spec,manifest,weights=fixture(tmp_path)
    assert verify_fixtures(spec,manifest)['MODEL']['files']['weights']['bytes']==15
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='pin'):verify_fixtures(spec,manifest)
    with pytest.raises(ValueError,match='Missing'):verify_fixtures(spec,{})


def test_required_files_cannot_be_omitted_or_linked(tmp_path):
    spec,manifest,weights=fixture(tmp_path)
    manifest['MODEL']['files']={}
    with pytest.raises(ValueError,match='required'):verify_fixtures(spec,manifest)
    manifest['MODEL']['files']={'../outside':'a'*64}
    with pytest.raises(ValueError):verify_fixtures(spec,manifest)
    spec,manifest,weights=fixture_again(tmp_path)
    held=weights.with_name('held');weights.rename(held);weights.symlink_to(held)
    with pytest.raises(ValueError,match='regular'):verify_fixtures(spec,manifest)


def fixture_again(tmp_path):
    child=tmp_path/'second';child.mkdir();return fixture(child)


def test_success_exit_cannot_mask_skips_missing_or_duplicate_cases(tmp_path):
    report=tmp_path/'tests.xml'
    report.write_text('<testsuites><testsuite><testcase name="test_real"/></testsuite></testsuites>')
    assert verify_python_result(report,['test_real'])==1
    for body in ['<testcase name="test_real"><skipped/></testcase>','<testcase name="other"/>','<testcase name="test_real"/><testcase name="test_real"/>','<testcase name="test_real"><failure/></testcase>']:
        report.write_text('<testsuites><testsuite>'+body+'</testsuite></testsuites>')
        with pytest.raises(ValueError):verify_python_result(report,['test_real'])


def test_swift_requires_each_named_case_pass_not_summary():
    names=['RealTests/testWholeFlow']
    assert verify_swift_result("Test Case '-[KwipMLAppTests.RealTests testWholeFlow]' passed (1.0 seconds).",names)==1
    for text in ['Executed 1 test, with 0 failures',"Test Case 'RealTests.testWholeFlow' skipped (0.0 seconds).", "Test Case 'OtherTests.testWholeFlow' passed (1.0 seconds)."]:
        with pytest.raises(ValueError):verify_swift_result(text,names)


def test_reviewed_model_pin_cannot_be_replaced_by_tiny_fixture(tmp_path):
    spec, manifest, _ = fixture(tmp_path)
    spec['MODEL']['qualifiedPins']={'weights':'0'*64}
    with pytest.raises(ValueError,match='reviewed'):
        verify_fixtures(spec,manifest)


def test_source_identity_rejects_tracked_and_untracked_mutations(tmp_path):
    import subprocess
    from scripts.required_model_qualification import source_identity, require_unchanged_source
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'runner').mkdir()
    tracked=tmp_path/'runner/tracked.py';tracked.write_text('original')
    subprocess.run(['git','add','.'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=tmp_path,check=True)
    before=source_identity(tmp_path)
    tracked.write_text('changed')
    with pytest.raises(ValueError,match='Source changed'):require_unchanged_source(tmp_path,before)
    untracked=tmp_path/'runner/new.py';untracked.write_text('first')
    before=source_identity(tmp_path);untracked.write_text('second')
    with pytest.raises(ValueError,match='Source changed'):require_unchanged_source(tmp_path,before)
    require_unchanged_source(tmp_path,source_identity(tmp_path))
