"""One real cold CLI capability probe per exact worker/runtime/environment identity.

Only successful, unmodified discovery is reusable. Adversarial flag routing,
packaged isolation and descriptor/negative probes retain their own subprocesses.
Callers still assert every component's complete capability contract.
"""
from functools import lru_cache
import hashlib
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def probe_identity():
    digest=hashlib.sha256()
    paths=[p for p in (ROOT/'runner').rglob('*') if p.is_file() and 'tests' not in p.parts and p.suffix in ('.py','.json')]
    paths.append(ROOT/'Sources/KwipMLCore/Resources/components.json')
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    environment=tuple(sorted((k,v) for k,v in os.environ.items() if k!='PYTEST_CURRENT_TEST'))
    distributions=tuple(sorted((d.metadata.get("Name",""),d.version) for d in metadata.distributions()))
    return sys.executable,digest.hexdigest(),environment,sys.version,distributions


@lru_cache(maxsize=8)
def _probe(identity):
    result=subprocess.run([identity[0],str(ROOT/'runner/main.py'),'--capabilities'],cwd=ROOT,
                          capture_output=True,text=True,check=True,timeout=30)
    return result.stdout,result.stderr


def capabilities_cli():
    stdout,stderr=_probe(probe_identity())
    # A fresh result object prevents tests from sharing mutable subprocess state.
    return subprocess.CompletedProcess([sys.executable,str(ROOT/'runner/main.py'),'--capabilities'],0,stdout,stderr)
