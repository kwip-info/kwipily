"""Independent real torch.export/save/load and inference-only boundary tests."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest
import torch
from torch import nn

from runner.program_export_configuration import DEFAULTS, normalize_program_export
from runner.program_export import export_program, reload_program, build_package


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def identity(): return {'runID': str(uuid.uuid4()).upper(), 'requestIdentity': 'a'*64, 'modelManifestSHA256': 'b'*64, 'exportManifestSHA256': 'c'*64}
def config(**values): return dict(DEFAULTS, **values)
def contract(dtype='float32', shape=(2,)): return {'dtype': dtype, 'sampleShape': list(shape)}
def loaded(raw, manifest): return reload_program(raw, manifest, expected_manifest_sha256=digest(canonical(manifest)))


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[2., -1.], [.5, 3.]]))
        self.bias = nn.Parameter(torch.tensor([.25, -2.]))
    def forward(self, x): return x @ self.weight.T + self.bias


class Branch(nn.Module):
    def __init__(self):
        super().__init__(); self.affine = Affine()
        self.register_buffer('scale', torch.tensor([.5, -2.]))
    def forward(self, x): return self.affine(x) + torch.tanh(x) * self.scale


def affine_expected(x):
    return torch.stack([2*x[:, 0]-x[:, 1]+.25, .5*x[:, 0]+3*x[:, 1]-2], dim=1)


def test_configuration_exact_defaults_nondefaults_and_invalid_controls():
    assert normalize_program_export(DEFAULTS) == DEFAULTS
    assert normalize_program_export(config(strict=False, batch_mode='bounded', example_batch_size=3, minimum_batch_size=1, maximum_batch_size=9, device='mps', prefer_deferred_runtime_asserts=True))['maximum_batch_size'] == 9
    assert normalize_program_export(config(example_batch_size=1, minimum_batch_size=1, maximum_batch_size=1))['example_batch_size'] == 1
    for key in DEFAULTS:
        missing = config(); missing.pop(key)
        with pytest.raises(ValueError): normalize_program_export(missing)
    invalid = [dict(DEFAULTS, extra=1), config(strict=1), config(prefer_deferred_runtime_asserts=0), config(batch_mode='auto'), config(device='cuda'), config(example_batch_size=True), config(example_batch_size=0), config(example_batch_size=129), config(minimum_batch_size=0), config(maximum_batch_size=4097), config(maximum_batch_size=3), config(batch_mode='bounded'), config(batch_mode='bounded', example_batch_size=1, minimum_batch_size=1, maximum_batch_size=4), config(batch_mode='bounded', example_batch_size=3, minimum_batch_size=4, maximum_batch_size=8)]
    for value in invalid:
        with pytest.raises(ValueError): normalize_program_export(value)


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('branched', [False, True])
def test_actual_static_export_reload_matches_independent_affine_and_branch(strict, branched):
    model = Branch() if branched else Affine()
    x = torch.tensor([[1., -2.], [3., .5]])
    raw, manifest = export_program(model, x, contract(), config(strict=strict), source_identity=identity())
    assert isinstance(torch.export.load(io.BytesIO(raw)), torch.export.ExportedProgram)
    actual = loaded(raw, manifest)(x)
    expected = affine_expected(x) + (torch.tanh(x)*torch.tensor([.5, -2.]) if branched else 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert manifest['schemaVersion'] == 1 and manifest['kind'] == 'kwipmlExportedProgram'
    assert manifest['runtime'] == {'torch': torch.__version__, 'device': 'cpu'}
    assert manifest['input'] == contract() and manifest['output'] == contract()
    assert manifest['shapeConstraints'] == {'mode': 'static', 'leadingAxis': 'batch', 'minimum': 2, 'maximum': 2, 'example': 2}
    assert manifest['rangeConstraints'] == []
    with pytest.raises(ValueError, match='batch'): loaded(raw, manifest)(torch.zeros(3, 2))


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('minimum', [1, 2])
def test_bounded_leading_axis_actual_native_guards_and_unchanged_sample_axes(strict, minimum):
    settings = config(strict=strict, batch_mode='bounded', example_batch_size=3, minimum_batch_size=minimum, maximum_batch_size=7)
    raw, manifest = export_program(Affine(), torch.zeros(3, 2), contract(), settings, source_identity=identity())
    native = torch.export.load(io.BytesIO(raw)).module()
    assert manifest['rangeConstraints']
    for count in [minimum, 4, 7]:
        x = torch.arange(count*2, dtype=torch.float32).reshape(count, 2)/3
        torch.testing.assert_close(loaded(raw, manifest)(x), affine_expected(x), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(native(x), affine_expected(x), rtol=1e-6, atol=1e-6)
    for bad in [torch.zeros(8, 2), torch.zeros(3, 3), torch.zeros(3, 2, dtype=torch.float64), torch.zeros(0, 2), torch.full((3, 2), float('nan'))]:
        with pytest.raises(ValueError): loaded(raw, manifest)(bad)
    with pytest.raises(Exception): native(torch.zeros(8, 2))


def test_eval_capture_preserves_state_modes_rng_and_native_normalization_math():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(.8))
    with torch.no_grad():
        model[0].running_mean.copy_(torch.tensor([2., -1.])); model[0].running_var.copy_(torch.tensor([4., 9.]))
        model[0].weight.copy_(torch.tensor([1.5, .5])); model[0].bias.copy_(torch.tensor([.25, -2.]))
    model.train(); model[1].eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}; modes = [m.training for m in model.modules()]; rng = torch.random.get_rng_state().clone()
    x = torch.tensor([[1., 2.], [3., -2.]])
    raw, manifest = export_program(model, x, contract(), config(), source_identity=identity())
    expected = (x-torch.tensor([2., -1.]))/torch.sqrt(torch.tensor([4., 9.])+1e-5)*torch.tensor([1.5, .5])+torch.tensor([.25, -2.])
    torch.testing.assert_close(loaded(raw, manifest)(x), expected, rtol=1e-6, atol=1e-6)
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert any(row['dtype'] == 'int64' for row in manifest['state'])


def test_integer_tensor_native_program_not_a_fake_float_input():
    model = nn.Embedding(5, 2)
    with torch.no_grad(): model.weight.copy_(torch.arange(10).reshape(5, 2))
    x = torch.tensor([[0, 3], [4, 1]], dtype=torch.int64)
    raw, manifest = export_program(model, x, contract('int64'), config(), source_identity=identity())
    assert manifest['output']['sampleShape'] == [2, 2]
    torch.testing.assert_close(loaded(raw, manifest)(x), torch.tensor([[[0., 1.], [6., 7.]], [[8., 9.], [2., 3.]]]))


def test_mutating_untraceable_or_packet_input_refuses_without_mutating_caller():
    class Mutating(nn.Module):
        def __init__(self): super().__init__(); self.register_buffer('counter', torch.tensor(0))
        def forward(self, x): self.counter.add_(1); return x+1
    class Dependent(nn.Module):
        def forward(self, x): return x+1 if x.sum().item()>0 else x-1
    model = Mutating(); rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match='mutat'): export_program(model, torch.zeros(2, 2), contract(), config(), source_identity=identity())
    assert model.counter.item() == 0 and torch.equal(rng, torch.random.get_rng_state())
    for strict in [True, False]:
        with pytest.raises(ValueError, match='[Ee]xport|trace|control'): export_program(Dependent(), torch.zeros(2, 2), contract(), config(strict=strict), source_identity=identity())
    with pytest.raises(ValueError, match='[Bb]ag|packet'): export_program(Affine(), object(), {'kind': 'bagPacket'}, config(), source_identity=identity())


def test_bounds_nonfinite_state_bad_input_identity_and_cancellation_refuse():
    model = Affine()
    for x, spec, settings in [(torch.zeros(3, 2), contract(), config()), (torch.zeros(2, 3), contract(), config()), (torch.zeros(2, 2, dtype=torch.float64), contract(), config()), (torch.zeros(2, 2), contract(shape=(4096,4096)), config()), (torch.zeros(2, 2), dict(contract(), extra=1), config())]:
        with pytest.raises(ValueError): export_program(model, x, spec, settings, source_identity=identity())
    with torch.no_grad(): model.weight[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'): export_program(model, torch.zeros(2, 2), contract(), config(), source_identity=identity())
    source = identity(); source['exportManifestSHA256'] = 'stale'
    with pytest.raises(ValueError): export_program(Affine(), torch.zeros(2, 2), contract(), config(), source_identity=source)
    def cancelled(): raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError): export_program(Affine(), torch.zeros(2, 2), contract(), config(), source_identity=identity(), check_cancelled=cancelled)


def test_hash_schema_runtime_state_and_constraints_tamper_refuse_before_native_load(monkeypatch):
    raw, manifest = export_program(Affine(), torch.zeros(2, 2), contract(), config(), source_identity=identity())
    with pytest.raises(ValueError): reload_program(raw, manifest, expected_manifest_sha256='0'*64)
    corrupt = raw[:-1]+bytes([raw[-1]^1])
    with pytest.raises(ValueError): loaded(corrupt, manifest)
    native_load = torch.export.load
    called = []
    monkeypatch.setattr(torch.export, 'load', lambda *a, **k: called.append(True))
    for mutate in [lambda m: m.update(schemaVersion=99), lambda m: m['runtime'].update(torch='0.0.0'), lambda m: m['shapeConstraints'].update(maximum=9), lambda m: m['input'].update(dtype='float64'), lambda m: m.update(extra=1)]:
        altered = copy.deepcopy(manifest); mutate(altered)
        with pytest.raises(ValueError): loaded(raw, altered)
    assert called == []
    monkeypatch.setattr(torch.export, 'load', native_load)
    altered = copy.deepcopy(manifest); altered['state'][0]['shape'] = [7, 7]
    with pytest.raises(ValueError, match='state'): loaded(raw, altered)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_explicit_device_no_fallback_and_actual_reload_where_available(device):
    settings = config(device=device)
    available = device == 'cpu' or torch.backends.mps.is_available()
    if not available:
        with pytest.raises(ValueError, match='MPS|mps'): export_program(Affine(), torch.zeros(2, 2), contract(), settings, source_identity=identity())
        return
    x = torch.tensor([[1., -2.], [3., .5]], device=device)
    raw, manifest = export_program(Affine().to(device), x, contract(), settings, source_identity=identity())
    assert manifest['runtime']['device'] == device
    actual = loaded(raw, manifest)(x)
    assert actual.device.type == device
    torch.testing.assert_close(actual.cpu(), affine_expected(x.cpu()), rtol=1e-5, atol=1e-6)
    if device == 'mps':
        with pytest.raises(ValueError, match='device'): loaded(raw, manifest)(x.cpu())


def test_portable_saved_package_reloads_without_original_class_source(tmp_path):
    destination = tmp_path/'portable'
    manifest = build_package(Branch(), torch.zeros(2, 2), contract(), config(), destination, source_identity=identity())
    assert {p.name for p in destination.iterdir()} == {'program.pt2', 'manifest.json', 'inference.py', 'README.md', 'requirements.txt'}
    text = (destination/'README.md').read_text()
    assert 'untrusted' in text and '2.14' in text and 'inference' in text.lower()
    check = tmp_path/'check.py'
    check.write_text("import sys, torch\nsys.path.insert(0,sys.argv[1])\nfrom inference import load_model\nx=torch.tensor([[1.,-2.],[3.,.5]])\ny=load_model().predict(x)\ne=torch.stack([2*x[:,0]-x[:,1]+.25,.5*x[:,0]+3*x[:,1]-2],1)+torch.tanh(x)*torch.tensor([.5,-2.])\ntorch.testing.assert_close(y,e,rtol=0,atol=0)\nprint('independent portable passed')\n")
    environment = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME'}}
    done = subprocess.run([sys.executable, '-I', str(check), str(destination)], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=45)
    assert done.returncode == 0, done.stdout+done.stderr
    assert 'independent portable passed' in done.stdout
    assert all(row['path'] != 'model.py' and 'checkpoint' not in row['path'] for row in manifest['files'])


def test_native_signature_must_match_metadata_even_with_fresh_manifest_pin():
    raw,manifest=export_program(Affine(),torch.zeros(2,2),contract(),config(),source_identity=identity())
    for field in ('input','output'):
        forged=copy.deepcopy(manifest);forged[field]['sampleShape']=[3]
        with pytest.raises(ValueError,match='signature'):loaded(raw,forged)
    forged=copy.deepcopy(manifest);forged['input']['dtype']='int64'
    with pytest.raises(ValueError,match='signature'):loaded(raw,forged)


def test_archive_paths_duplicates_and_expansion_are_refused_before_native_load(monkeypatch):
    import zipfile
    raw,manifest=export_program(Affine(),torch.zeros(2,2),contract(),config(),source_identity=identity())
    for name in ('../private','/absolute','a\\\\b'):
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'x')
        data=archive.getvalue();forged=copy.deepcopy(manifest)
        for row in forged['files']:
            if row['path']=='program.pt2':row.update(bytes=len(data),sha256=digest(data))
        with pytest.raises(ValueError,match='archive'):loaded(data,forged)


def test_portable_helper_refuses_changed_file_and_redirect_without_opening_external(tmp_path):
    import importlib.util
    target=tmp_path/'package';build_package(Affine(),torch.zeros(2,2),contract(),config(),target,source_identity=identity())
    spec=importlib.util.spec_from_file_location('owned_program_helper',target/'inference.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    original=(target/'program.pt2').read_bytes();(target/'program.pt2').write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='hash'):helper.load_model()
    (target/'program.pt2').write_bytes(original)
    manifest=json.loads((target/'manifest.json').read_bytes());manifest['files'][0]['path']='../missing-external'
    (target/'manifest.json').write_bytes(canonical(manifest))
    with pytest.raises(ValueError,match='inventory'):helper.load_model()
