# Conv1d / Conv2d / Conv3d independent numerical receipt

2026-10-08. Bounded family verification follows [delivery efficiency](DELIVERY_EFFICIENCY_20261008.md). This is one numerical batch for the existing standard convolutions, not acceptance of additional depthwise/separable/transpose convolution units. No worker, schema, configuration limits, padding policy or catalog mode changed. Newly added verification passed on its first actual execution; no product RED is invented.

## Independent oracle and finite matrix

`runner/tests/test_convolution_independent.py` computes grouped cross-correlation by Python coordinates/index offsets and `math.fsum`. Expected forward values and input/weight/optional bias gradients contain no PyTorch tensor operation or autograd reference. Padding maps original coordinates independently: zeros omit out-of-bounds contributions, reflection maps to mirrored input positions, replication clamps to the edge, and circular padding wraps. Gradients sum every repeated mapping back to the original input or shared parameter.

The 168 new cases cross spatial dimensions 1/2/3, CPU/actual MPS, bias disabled/enabled, batch 1/3 and seven coherent geometry cases:

| Case | Distinct behavior |
|---|---|
| Same + zeros | Even kernel, stride 1, asymmetric zero padding |
| Same + reflect | Mirrored coordinate routing and its repeated input contributions |
| Same + replicate | Clamped boundary routing |
| Same + circular | Wrapped boundary routing |
| Valid + nondefault dilation | First spatial dilation 2, no padding |
| Explicit + nondefault stride | First spatial stride 2, padding 1, groups 1 |
| Explicit + circular | Padding 1 on each spatial side, groups 2 |

Kernel sizes are 2 on every spatial axis. Input channels are 2 and output channels 4; groups 2 isolates channel groups while groups 1 verifies cross-channel mixing. Spatial sizes are 5, `[5,4]`, `[5,4,3]`. Every sample is a stride-two noncontiguous input view. Explicit nonsymmetric bounded binary-representable inputs, weights, bias and arbitrary upstream values keep reference arithmetic controlled. This is representative coverage of distinct interactions, not every policy/mode/geometry combination or a promise for arbitrary ill-conditioned weights.

Actual real-registry compiled GraphModels enforce channel/spatial shape contracts and execute the existing convolution node path. Both the worker-owned module and generated standalone module separately meet the scalar forward and all-gradient oracle after strict state loading. Full generated graph output and every public-input/parameter gradient also match the worker graph. That complete graph comparison is an export/routing check; the convolution primitive's scalar oracle does not claim independent derivatives for the surrounding Linear projection/head.

Output shape/dtype/finite values, parameter state/count, bias absence/presence, unchanged input/weights, fixed-weight train/eval equality and batch-row isolation are checked. Kernels retain cross-correlation ordering (no silent reversal). No extra freeze or initialization control is invented.

CPU fixture tolerances are `atol=1e-6, rtol=1e-5`; MPS `atol=1e-5, rtol=1e-4`. None was widened after observation. Product graph execution remains float32; existing float64 raw references remain their separate reference domain.

## Actual execution

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest runner/tests/test_convolution_independent.py runner/tests/test_conv1d_contract.py runner/tests/test_conv2d_contract.py runner/tests/test_conv3d_contract.py runner/tests/test_conv1d_graph.py runner/tests/test_conv2d_graph.py runner/tests/test_conv3d_graph.py runner/tests/test_conv1d_discovery.py runner/tests/test_conv2d_discovery.py runner/tests/test_conv3d_discovery.py runner/tests/test_conv1d_lifecycle.py runner/tests/test_conv2d_lifecycle.py runner/tests/test_conv3d_lifecycle.py -q -rA
```

**550 passed, no skips, in 8.01 seconds**, Python 3.13.6 / PyTorch 2.14.1 with CPU and actual MPS available.

| Suite group | Passing tests |
|---|---:|
| New independent convolution family matrix | 168 |
| Existing Conv1d / Conv2d / Conv3d contracts | 51 / 87 / 93 |
| Existing Conv1d / Conv2d / Conv3d graph/export/refusals | 43 / 38 / 38 |
| Existing per-operation exact discovery | 2 / 2 / 2 |
| Existing Conv1d / Conv2d / Conv3d lifecycle | 7 / 9 / 10 |

MPS tensors executed all applicable cases with explicit `PYTORCH_ENABLE_MPS_FALLBACK=1`. This receipt does not assert that every 3D or padding kernel ran natively on the GPU, nor infer a particular CPU fallback from successful output. The sole observed warning was PyTorch's even-kernel `padding='same'` possible zero-padded input copy; no CPU-fallback warning occurred in this run. These are pinned local device observations, not cross-version/hardware guarantees.

Existing strict contracts/refusals retain scalar/paired/triple type requirements, channels/group divisibility, schema/dtype/rank/layout checks, invalid padding policies/modes, same-stride restrictions, empty/nonmatching runtime shapes, padding feasibility, parameters/aggregate preallocation checks, work/sample/output ceilings and export guards. Existing actual lifecycle tests run once per family rather than per new numerical case: CPU exact trained model/optimizer/RNG/metrics continuation and immutable parent, gradient inventory, frozen trained reload, portable export, fresh-optimizer fine-tuning, initialized/trained cache reuse and configuration incompatibility. Conv2d/3d existing MPS training/preview checks also execute. CPU exact continuation is not generalized to MPS weights-only state transfer.

Ignored `.local/convolution-independent-worker.log` records exact command/runtime, per-module counts and source hashes. New test SHA-256: `da8f132d9715db6f56e7cab3f35f0daf4763d3a98f9c588a251970f3dd84c74f`; shared geometry `4318d9390e158183b73945105457460f72b65b373e58c4bc2f524494f9ce8037`; engine `71f77392cd5c91b7e8762f7e219518fb948f8aad31cb6249c291ee2f344433ef`; observed registry `d8a98e04a4808b23fa6c401aa7ec229cdbae18d2a20bf6b5f85ef55a94cdfcb2`. Shared-source changes require applicability review before evidence binding. Scoped `git diff --check` passed.

This is development-runtime numerical/lifecycle evidence. Per-unit authored bundles, assessed lessons, native changed interactions, artwork acceptance, packaged launch, catalog mode completion and formal release gates remain separately owned. A family receipt cannot confer credit on a member missing those deliverables.
