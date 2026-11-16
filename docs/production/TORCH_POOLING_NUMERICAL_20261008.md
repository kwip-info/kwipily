# Pooling independent numerical evidence · 2026-10-08

Finite four-component worker slice: MaxPool1d/2d/3d and AvgPool1d. Only the new supplemental test and this receipt are authored here. Engine, configuration, schemas, component metadata, application and lessons are unchanged by this slice. Root owns bundle/lesson/native/catalog acceptance. No adaptive pooling or unpool capability credit is proposed.

## Actual evidence

Python3.13.6/PyTorch2.14.1, actual CPU and available MPS. Explicit `PYTORCH_ENABLE_MPS_FALLBACK=1` permits current product fallback behavior; no native-kernel or exact MPS continuation guarantee is inferred. Combined actual log `.local/pooling-independent-worker.log`:269 passed in2.36s, no skips and no warnings.156 new checks plus113 existing checks; each existing module replayed once. The initial supplemental156 checks passed immediately (`.local/pooling-independent-first.log`); no invented product RED or runtime fix. A later explicit channel-isolation assertion also passed in the combined run.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest \
  runner/tests/test_pooling_independent.py \
  runner/tests/test_maxpool1d_contract.py \
  runner/tests/test_maxpool1d_discovery.py \
  runner/tests/test_maxpool1d_graph.py \
  runner/tests/test_maxpool2d_contract.py \
  runner/tests/test_maxpool2d_discovery.py \
  runner/tests/test_maxpool2d_graph.py \
  runner/tests/test_maxpool3d_contract.py \
  runner/tests/test_maxpool3d_graph.py \
  runner/tests/test_avgpool1d_contract.py \
  runner/tests/test_avgpool1d_graph.py -q -rA
```

## Independent scalar matrix

152 new numerical cases use Python coordinate indexing and `math.fsum`; expected values/indices/gradients use no Torch reference operator or expected autograd. MaxPool1d/2d/3d each own40 cases: five geometries × return-indices off/on × actual CPU/MPS × batch1/3. AvgPool1d owns32 cases: four geometries × include-padding off/on × actual CPU/MPS × batch1/3. Four additional cases exercise actual CPU fine-tuning, one for each component.

All input tensors are float32 channels-first noncontiguous views with unused storage sentinels, two channels, and explicit signed upstream values. Max inputs are entirely negative with repeated equal maxima, so an accidental zero-padding winner or a different tie route cannot hide behind plausible positive values. The oracle visits each window in spatial coordinate order and selects the first equal maximum, returning its flattened index within the input spatial plane (not a global batch/channel offset). Exact int64 indices and scalar accumulation of upstream values at original winners are checked. First-tie agreement is observed for the tested finite inputs on both devices; this is not a universal backend tie/NaN claim. Every representative window contains real finite input; all-padding negative-infinity behavior remains separately covered by existing refusal tests.

Max geometries cover overlapping windows, padded stride2 ceiling windows, nullable stride following the kernel with removal of trailing windows starting wholly beyond the input, dilated windows, and independently chosen paired/triple spatial controls for2D/3D. The independent shape formula includes the documented final trailing-window correction. Average geometries cover overlap, padded ceiling, nullable stride/trailing-window removal and an unpadded final partial ceiling window. Include-padding divides by positions inside the declared padded extent; exclude-padding divides only by real input count. The additional ceil extension beyond declared padding is not silently added to the divisor. Each original input derivative is the sum of upstream/divisor from overlapping windows.

Actual `GraphModel.execute_node` values and input gradients are compared with scalar expectations at `atol2e-6, rtol2e-5`. Max values and indices are effectively exact for the supplied integer-valued floats. Inputs/state must remain unchanged, batch and channel changes must leave other batch rows/channels identical, and return-indices must be int64. Each graph contains only learned expansion/head state, with no pooling module parameter/state entries. Complete generated standalone Models load strict actual state dictionaries, reproduce worker outputs and all input/surrounding-parameter derivatives, and agree in evaluation mode. Whole-graph worker/export parity is separately identified; it is not presented as independent arithmetic for surrounding Linear layers.

## Existing boundaries and lifecycle replay

Existing ten modules verify strict configuration keys/types, scalar versus fixed pair/triple lists, nullable stride, shape bounds, output geometry, ceiling/dilation/padding limits and typed indices ports. Average count_include_pad remains its specific control; groups, learned bias, nonzero padding modes, adaptive outputs and unpool are not controls for these components. Empty/mismatched layouts and all-padding nonfinite captures retain current rejection behavior. Discovery remains tied to each exact current descriptor.

Existing per-operation graph tests execute initialized cache reuse and exact CPU checkpoint continuation of model/optimizer/RNG state, captured samples where applicable, and actual MPS forward/backward. The four new fine-tune tests verify immutable parent checkpoint bytes, fresh seed/epoch, reset optimizer step, weightsOnlyFineTune lineage and exact expansion/head-only state keys. Pooling nodes remain parameterless; trainable surrounding layers make these actual training/fine-tune checks. MPS execution and copied weight inference do not imply exact MPS optimizer/RNG continuation. No native/UI, supported-machine, precision-all-sizes or final release gate is claimed.

## Frozen numerical and runtime identities

| Path | SHA-256 |
|---|---|
| `runner/tests/test_pooling_independent.py` | `4372a83354a8d2e940dcedb0db63110dc6e516e62e4ba70afd03fd78dbbcbab2` |
| `runner/engine.py` | `71f77392cd5c91b7e8762f7e219518fb948f8aad31cb6249c291ee2f344433ef` |
| `runner/graph_semantics.py` | `7bdaa382ba51e02d0740860c1219e4fa3854c788c4d78eb848ae9e5ce5f0f43b` |
| `runner/maxpool1d_configuration.py` | `67c0e98529a274d2a2a17343f5583415229f4bd624a63539b4fa4fff7972df7e` |
| `runner/maxpool2d_configuration.py` | `43f0571889bc763d8cccf38e634730a78b80cb05579e7db868ab8dff8040b953` |
| `runner/maxpool3d_configuration.py` | `0507293b191f73f6c43730c1c9b2f13b0ad8913ff13922195e43f8bf37846251` |
| `runner/spatial_maxpool_configuration.py` | `f537b9cd898123a3f9c86ea24b684fc8cf0055c765ff73fe0432f3cabe9c7309` |
| `runner/avgpool1d_configuration.py` | `cf353b8859173558d34c8df0020fbd598b33ca35b2cae6027073d2435cc23347` |
| `runner/pooling_adapters_generated.py` | `35a9b733018c6a956a8d8f48e70e8b634279cab30234a5bca584fc550ec72f4f` |
| `Sources/KwipMLCore/Resources/components.json` | `ffa7e7ee563faba2b941b80d44011452f645bafab31583dcc0351dad1342a07f` |

Registry identities in the new matrix are the actual production entries, not injected private metadata. Root may migrate authored ownership while preserving definitions; final acceptance must reconcile the final shared bytes.

## Actual native-trained MaxPool2d portable export addendum

Root supplied export `.local/KwipML-model-7A3F95AF`: run `7A3F95AF-2E00-42E4-81A1-1E7B9C1DB107`, project `696B09C0-4666-4FA7-AF57-3E0FEC83535C`, request `a96af4d0143fda961f83b3f7f2f31d251a6738d844583f645216abcbbae63672`, source revision0. Actual checkpoint records epoch3, MPS training and `weightsOnly` guarantee. Portable inference was executed on CPU float32 with Python3.13.6/PyTorch2.14.1 using `.venv/bin/python .local/pooling-export-parity.py`; ignored `.local/pooling-export-parity.json` contains exact prepared rows, learned-model results and spatial indices. Standalone `inference.py --example --batch-size1` succeeded.

Four synthetic prepared `[N,2]` rows were independently calculated with Python scalar learned projection40, reshape `[N,2,4,5]`, rectangular MaxPool2d kernel[2,3]/stride[1,2]/padding0/dilation1/ceilfalse, scalar ReLU, flatten12 and learned affine classifier. Expected maxima and first-winner indices use the independent indexed oracle. The48 spatial-plane indices exactly match an actual observation of the exported operator stage; expected arithmetic uses no Torch operator. Maximum absolute error is `7.233581866827876e-8` versus batch-one predictions and `1.340686930362267e-7` versus whole-batch/direct strict-state Model. Max-value error is `1.341104507446289e-7`; batch-one/whole rounding difference is `1.1920928955078125e-7`. Direct strict-loaded Model and whole-batch predictor agree exactly.

`model.py` SHA-256: `4e726b07d0cb6743a56be1c16d2e8da687d6c38566b0289ea83a139d9c49aa12`; `checkpoint.pt`: `17c2be76363a20fbac52ffb90aca953e6e64bb69f354ce9fe835b2bd62f1c9a8`; `inference.py`: `ceec96cc996318d43ca950e39f0237a3f235e8ed5e59458cbada0120a3ef3b25`. This checks actual exported learned weights on finite prepared rows; raw fitted preprocessing is not included. It adds no exact MPS continuation, cross-runtime/hardware or invented native observation claim. Frozen supplemental test source/hash is unchanged.
