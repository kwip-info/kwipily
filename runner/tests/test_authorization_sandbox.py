"""Kernel no-follow traversal must work with sandbox-authorized leaf directories."""
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='macOS sandbox and O_NOFOLLOW_ANY')


@pytest.mark.parametrize('symlink_component',[False,True])
def test_authorized_directory_does_not_require_read_data_access_to_ancestor(tmp_path,symlink_component):
    base=tmp_path.resolve()
    leaf=base/'real'/'leaf';leaf.mkdir(parents=True)
    target=leaf
    if symlink_component:
        alias=base/'alias';alias.symlink_to(base/'real',target_is_directory=True)
        target=alias/'leaf'
    script=base/'probe.py'
    script.write_text(f'''
import os,sys
sys.path.insert(0,{str(ROOT/'runner')!r})
from worker_authorization import directory
# Verify the policy exercises the original ancestor-open failure.
try:
    fd=os.open({str(base/'real')!r},os.O_RDONLY|os.O_DIRECTORY)
except PermissionError:
    pass
else:
    os.close(fd)
    raise AssertionError('Sandbox did not deny ancestor read-data access')
try:
    fd=directory({str(target)!r})
except OSError:
    if not {symlink_component!r}:raise
    print('SYMLINK_REJECTED')
else:
    try:
        assert not {symlink_component!r},'Symlink traversal was accepted'
        assert os.fstat(fd).st_ino==os.stat({str(leaf)!r}).st_ino
        print('AUTHORIZED_LEAF_OPENED')
    finally:os.close(fd)
''')
    policy='(version 1) (allow default) (deny file-read-data (literal '+json.dumps(str(base/'real'))+'))'
    result=subprocess.run(['/usr/bin/sandbox-exec','-p',policy,sys.executable,'-I','-B',str(script)],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.strip()==('SYMLINK_REJECTED' if symlink_component else 'AUTHORIZED_LEAF_OPENED')
