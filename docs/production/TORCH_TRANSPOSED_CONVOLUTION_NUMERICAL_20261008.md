# Transposed-convolution independent numerical evidence · 2026-10-08

Finite worker-only family slice for existing ConvTranspose1d/2d/3d. No engine, configuration, schema, registry, authored bundle, application, lesson, catalog or native change is made here. Root owns the component/assessed-lesson bundle closure and final acceptance bindings.

## Actual execution

Python3.13.6, PyTorch2.14.1, actual CPU and available MPS on the development Mac. Explicit `PYTORCH_ENABLE_MPS_FALLBACK=1` permits the current product fallback boundary. The successful log reports no warnings and no skips. Selected MPS execution does not establish that every operation used a native GPU kernel, exact GPU continuation or supported-machine acceptance.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest \
  runner/tests/test_transposed_convolution_independent.py \
  runner/tests/test_conv_transpose1d_contract.py \
  runner/tests/test_conv_transpose2d_contract.py \
  runner/tests/test_conv_transpose3d_contract.py \
  runner/tests/test_conv_transpose1d_graph.py \
  runner/tests/test_conv_transpose2d_graph.py \
  runner/tests/test_conv_transpose3d_graph.py \
  runner/tests/test_conv_transpose1d_discovery.py \
  runner/tests/test_conv_transpose2d_discovery.py \
  runner/tests/test_conv_transpose3d_discovery.py \
  runner/tests/test_conv_transpose1d_lifecycle.py \
  runner/tests/test_conv_transpose2d_lifecycle.py \
  runner/tests/test_conv_transpose3d_lifecycle.py -q -rA
```

Actual GREEN `.local/transposed-convolution-independent-worker.log`:498 passed in6.15s, no skips.145 new checks =144 independent real worker/export cases plus one manually calculated oracle sanity check;353 existing contract/graph/discovery/lifecycle checks replayed once.

The first supplemental execution `.local/transposed-convolution-independent-first.log` passed all144 actual worker/export cases. One new manually calculated sanity expectation had an arithmetic typo (`dX[0]`17 instead of29); correcting this test-only expected sum gave the final GREEN. This is a test-authoring failure, not a product RED or runtime fix. No numerical failure or engine defect was observed.

## Independent measured claims

The oracle uses Python integer coordinate indexing and `math.fsum`. An input position scatters its weighted contribution to each valid output position. Per-group input/output channel partitions use the actual transposed-kernel layout `[Cin,Cout/groups,*kernel]`. Overlapping contributions sum. Cropped positions contribute neither output nor derivatives. Bias is added once per retained output position.

Output extent is calculated independently for every spatial direction as `(input−1)×stride −2×padding + dilation×(kernel−1) + output_padding +1`. Output padding changes the retained scatter extent; it does not append prescribed zero-valued cells. A hand-computed stride2/padding1/output-padding1 example retains a nonzero final kernel contribution, and the same behavior is checked through the real worker matrix. Input, kernel and bias gradients are direct sums of each contribution times the explicitly supplied upstream values; expected derivatives do not use PyTorch autograd or a Torch reference operator.

The144 cases cover all three spatial dimensions, both actual devices, batch1/3, bias off/on, and six representative configurations:

| Case | Distinct geometry |
|---|---|
| overlap | kernel3, stride1, no crop, groups2 |
| crop | larger first-axis crop, remaining-axis crop, groups1 |
| stride extent | stride2, kernel3, padding1, output-padding1, groups1 |
| dilated extent | dilation2, stride1, output-padding1; legal through the dilation alternative, groups2 |
| stride plus dilation | stride2/dilation2/kernel2/padding1/output-padding1, groups2 |
| anisotropic | independently chosen per-axis kernel/stride/padding/dilation/output-padding for2D/3D, groups1 |

Each uses float32, channels-first shapes, explicitly noncontiguous input views with unused storage sentinels, independently supplied signed weights/bias and signed upstream values. Expected output, input gradient, kernel gradient and bias gradient are compared to the real `GraphModel.execute_node` layer and the actual generated standalone module loaded with its strict state dictionary. CPU tolerance is `atol1e-6, rtol1e-5`; MPS is `atol1e-5, rtol1e-4`. These finite representative tolerances are not a blanket numerical-accuracy claim for all bounded sizes or conditioning.

Additional checks require exact shape/dtype, finite outputs/derivatives, unchanged input/state, parameter counts and bias ownership, identical evaluation-mode behavior and isolation of unchanged batch rows when another row changes. Complete real graphs include the existing trainable expansion and head layers; generated export/state loading preserves complete output and every input/outer-layer/transposed-layer derivative. Export-to-worker parity complements the independent layer oracle; it is not mislabeled independent arithmetic for the surrounding Linear layers.

## Reused contract and lifecycle evidence

Existing three-family tests cover the strict nine controls, scalar versus explicit paired/triple spatial settings, exactly channels-first sample rank, shape inference, zero-only padding, static output extent, malformed values/layout/dtypes, group divisibility, invalid output-padding alternatives, geometry/resource budgets and discovery boundaries. No dynamic `output_size`, alternative padding modes or wider configuration promise is added.

Existing lifecycle tests execute exact CPU checkpoint continuation (model/optimizer/RNG/metrics), frozen generated exports/captured parameters, fresh fine-tune optimizer/epoch semantics, initialized and trained preview/cache reuse, stale-cache/configuration refusal, and prior optimizer schedules. MPS tests execute real training and initialized/trained preview from weights. Exact CPU continuation evidence is not extended to an exact MPS continuation promise. Existing float64 graph/reference checks remain supplemental to the new independent float32 CPU/MPS derivative matrix.

## Frozen source identities

| Path | SHA-256 |
|---|---|
| `runner/tests/test_transposed_convolution_independent.py` | `4f2a16e188dc871852df4932b2735cd0f4550949bce29d65aafa44c9d776d2c9` |
| `runner/engine.py` | `71f77392cd5c91b7e8762f7e219518fb948f8aad31cb6249c291ee2f344433ef` |
| `runner/graph_semantics.py` | `7bdaa382ba51e02d0740860c1219e4fa3854c788c4d78eb848ae9e5ce5f0f43b` |
| `runner/conv_transpose1d_configuration.py` | `896bdddd08a3a808d76280d7ae3aeafdb8f550441730a93964f9ada27b578d95` |
| `runner/conv_transpose2d_configuration.py` | `328bc749ff8f28a8b190250df650087cf27466e77abeceddbf6e1e9b65caf1b0` |
| `runner/conv_transpose3d_configuration.py` | `1fcba4a213fdf12c8caeb9d04b6c8bf1bb83fc8a33c003ac0ac0825767c757b1` |
| observed `Sources/KwipMLCore/Resources/components.json` | `d8a98e04a4808b23fa6c401aa7ec229cdbae18d2a20bf6b5f85ef55a94cdfcb2` |

The observed registry contains the three current production transposed-convolution IDs; the new numerical matrix uses it directly, not private injected definitions. Root may migrate metadata ownership without changing those definitions; reconcile final bytes separately. No Swift build, package, native lesson, icon or UAT result is claimed in this worker receipt.

## Actual native-run portable export parity addendum

Root supplied actual native-trained export `.local/KwipML-model-58D81F52`, run `58D81F52-8DE5-4306-B43E-0D0EA5F7B749`, captured epoch3. Its checkpoint contract records MPS training and `weightsOnly` guarantee. The actual portable CPU float32 package was checked with Python3.13.6/PyTorch2.14.1 using `.venv/bin/python .local/transposed-convolution-native-export-parity.py`; detailed synthetic inputs/results are in ignored `.local/transposed-convolution-native-export-parity.json`. `inference.py --example --batch-size1` also succeeded.

For four prepared `[N,2]` rows, independent Python `math.fsum` uses actual learned projection weights, the indexed scalar scatter-add reference for grouped ConvTranspose2d (kernel3, stride2, padding1, output-padding[1,0], groups2, dilation1), scalar ReLU and learned classifier weights. Reshaped input is `[N,2,3,4]`; convolution output `[N,4,6,7]` flattens to168 responses. No Torch expected convolution, activation or affine operator is used. Maximum absolute error versus batch-one loaded inference is `2.1975410691155162e-7`; versus whole-batch/direct strict-state-loaded Model is `6.291534315749914e-7`. Batch-one/whole-batch rounding difference is `7.152557373046875e-7`; direct Model and loaded whole-batch predictor agree exactly.

Export `model.py` SHA-256: `7d36a69b403d38469ac0a26b2be13f8ec7c9f9e6cad7b5fc138a6673b400dee8`; `checkpoint.pt`: `e5db5654e59a519994ba37a6d850b41ee912a7d1ab4c6cd0c17a82c81c05cf42`; `inference.py`: `ceec96cc996318d43ca950e39f0237a3f235e8ed5e59458cbada0120a3ef3b25`. Only prepared synthetic rows are exercised. This finite actual-package result does not include raw fitted preparation, certify another machine/runtime, or convert the MPS `weightsOnly` contract into exact training continuation. Supplemental numerical test source and its frozen hash remain unchanged.
