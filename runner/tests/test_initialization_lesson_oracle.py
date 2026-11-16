"""Independent bounded lesson oracle probe; no KwipML initializer imports."""
import hashlib
import json
import math

import pytest
import torch


def owner_seed(seed, node, parameter):
    raw = json.dumps(['kwip.initialization.owner-seed.v1', seed, node, parameter], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63)


def normal_reference(shape, seed, mean, std):
    # CPU's small-array normal kernel obtains double standard draws. Float32
    # randn would round too early; larger vector kernels have another stream.
    if math.prod(shape) >= 16:
        raise ValueError('This independent lesson oracle admits fewer than16 learned values per array.')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=torch.float64) * std + mean).float()


def uniform_reference(shape, seed, bound):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    low, high = float(torch.tensor(-bound, dtype=torch.float32)), float(torch.tensor(bound, dtype=torch.float32))
    unit = torch.rand(shape, generator=generator, dtype=torch.float32)
    return (unit.double() * (high-low) + low).float()


SEEDS = [0, 7, 42, 2**63-1, 7483421056799741684, 4227900481719295887]
SHAPES = [(3, 2), (2, 3), (4, 3), (2, 4), (3, 4), (5, 2), (3, 5), (2,), (3,), (4,), (5,)]


def test_frozen_independent_utf8_owner_seed_vectors():
    assert owner_seed(42, 'first', 'weight') == 7483421056799741684
    assert owner_seed(173, 'second', 'weight') == 4227900481719295887
    assert owner_seed(2**63-1, 'Block/é/模型', 'nested.weight') == 8908088362943642764
    assert owner_seed(42, 'first', 'weight') != owner_seed(42, 'second', 'weight')


def test_small_normal_native_matches_independent_double_standard_stream_exactly():
    for seed in SEEDS:
        for shape in SHAPES:
            for mean, std in [(0., 1.), (.15, .2), (-2., .7), (3., 0.), (1e6, 1e6)]:
                actual = torch.empty(shape, dtype=torch.float32)
                torch.nn.init.normal_(actual, mean, std, generator=torch.Generator().manual_seed(seed))
                assert torch.equal(actual, normal_reference(shape, seed, mean, std)), (seed, shape, mean, std)
    with pytest.raises(ValueError, match='fewer than16'):
        normal_reference((4, 4), 42, .15, .2)


def test_xavier_and_kaiming_nondefaults_match_independent_fan_and_uniform_math():
    for seed in SEEDS:
        for shape in [shape for shape in SHAPES if len(shape) == 2] + [(2, 3, 2), (3, 2, 3)]:
            fan_in, fan_out = shape[1]*math.prod(shape[2:]), shape[0]*math.prod(shape[2:])
            for gain in [0., .7, 2.]:
                actual = torch.empty(shape, dtype=torch.float32)
                torch.nn.init.xavier_uniform_(actual, gain=gain, generator=torch.Generator().manual_seed(seed))
                expected = uniform_reference(shape, seed, gain*math.sqrt(6./(fan_in+fan_out)))
                assert torch.equal(actual, expected), ('xavier', seed, shape, gain)
            for slope in [0., .2, 1.]:
                for mode, fan in [('fan_in', fan_in), ('fan_out', fan_out)]:
                    for behavior in ['relu', 'leaky_relu']:
                        actual = torch.empty(shape, dtype=torch.float32)
                        torch.nn.init.kaiming_uniform_(actual, a=slope, mode=mode, nonlinearity=behavior, generator=torch.Generator().manual_seed(seed))
                        gain = math.sqrt(2.) if behavior == 'relu' else math.sqrt(2./(1.+slope*slope))
                        assert torch.equal(actual, uniform_reference(shape, seed, math.sqrt(3.)*gain/math.sqrt(fan))), ('kaiming', seed, shape, slope, mode, behavior)


def test_canonical_lesson_wrong_seed_owner_and_controls_change_real_expected_values():
    for shape in [(3, 2), (4, 3), (3, 4), (5, 2)]:
        seed = owner_seed(42, 'first', 'weight')
        normal = normal_reference(shape, seed, .15, .2)
        assert not torch.equal(normal, normal_reference(shape, owner_seed(41, 'first', 'weight'), .15, .2))
        assert not torch.equal(normal, normal_reference(shape, owner_seed(42, 'second', 'weight'), .15, .2))
        assert not torch.equal(normal, normal_reference(shape, seed, .16, .2))
        assert not torch.equal(normal, normal_reference(shape, seed, .15, .21))
        assert not torch.equal(normal, uniform_reference(shape, seed, .7*math.sqrt(6./sum(shape))))
    state = torch.get_rng_state().clone()
    normal_reference((3, 2), owner_seed(173, 'first', 'weight'), .15, .2)
    uniform_reference((2, 3), owner_seed(173, 'second', 'weight'), .7)
    assert torch.equal(torch.get_rng_state(), state)
