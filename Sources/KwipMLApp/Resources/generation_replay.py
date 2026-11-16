"""Replay captured generation: python generate.py. Requires the exported Torch runtime."""
import hashlib
import json
from pathlib import Path
import types
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'generation-manifest.json').read_bytes())
for entry in manifest['files']:
    name=entry['path']
    if '/' in name or name in ('.','..'):raise ValueError('Invalid replay inventory path')
    raw=(ROOT/name).read_bytes()
    if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Replay artifact changed: '+name)
namespace={'__name__':'kwipml_generation_kernel'}
exec(compile((ROOT/'text_generation.py').read_bytes(),str(ROOT/'text_generation.py'),'exec'),namespace)
inference=types.ModuleType('kwipml_generation_inference');inference.__file__=str(ROOT/'inference.py')
exec(compile((ROOT/'inference.py').read_bytes(),inference.__file__,'exec'),inference.__dict__)
request=json.loads((ROOT/'generation-input.json').read_bytes())
result=namespace['generate'](inference.load_model(),request['prompts'],request['settings'])
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
