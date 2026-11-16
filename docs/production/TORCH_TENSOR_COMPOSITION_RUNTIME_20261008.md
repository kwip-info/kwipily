# Tensor composition runtime receipt — 2026-10-08

Scope: configured Concatenate, Add, Stack and Permute under the frozen `TORCH_TENSOR_COMPOSITION_CLOSURE.md` contract. This is worker evidence, not a bundle acceptance or coverage claim. Root owns Core/App lessons, native checks, artwork, integration accounting and acceptance.

## Implementation

`runner/tensor_composition_configuration.py` provides allocation-free `normalize_tensor_composition(operation, config, dtype='float32')`, `resolve_tensor_composition(operation, config, left, right=None, dtype='float32')` and `capability(operation)`. Resolver returns `BatchedLayout`. Exact authored keys are `dim`, `alpha` + `sample_broadcast`, or `dims`. Integral JSON numeric axes normalize to integers; booleans, fractional/nonfinite axes and extra fields refuse. Complete signed permutation aliases retain their authored representation and resolve against rank.

`BatchedLayout` gained Stack, true Permute and explicit sample-only Add broadcasting. Engine validates actual source layout, dense float32/int64 dtype, equal nonempty observation batches and devices; checks logical source/output element budgets before finite scans or operation allocation; and refuses nonfinite floating inputs/results. Add reuses only sample directions, never the observation batch. Alpha zero still validates the second source. Integer Add retains historical signed int64 arithmetic and permits multiplier one without broadcasting. Composition introduces no parameters.

Portable `model.py` exports the same checked adapter without runner imports. Graph schema 1/2 permit configured float32 Concatenate/Add; Stack/Permute require schema 2. Existing intermediate-width, parameter, training and resource limits remain in force. Capability discovery, authored bundle and generated callable table match exactly for all four IDs; the existing tensor-layout descriptor stays unchanged.

Historical empty Concatenate/Add settings normalize to explicit defaults for execution, but semantic identity strips only those defaults. Existing graph hashes, cache provenance and strict checkpoint resume stay stable; nondefaults retain semantic identity even when a signed/positive axis describes equivalent geometry.

## TDD and finite verification

Host: macOS 26.6 arm64, Python 3.13.6, actual PyTorch 2.14.1, MPS available. Tests require actual CPU and MPS execution, without device skips or fake registry fixtures.

- Initial configuration RED: 49 failed / 2 passed against an explicitly refusing helper stub; initial runtime RED: 47 failed / 2 passed before integration. The two runtime passes were inapplicable unary mixed-device/batch cases, subsequently removed. One initial configuration parameter returning without an assertion was also removed. These counts are historical development observations, not the final suite size.
- Final owned configuration/runtime suite: **157 passed**, no skips/warnings, against the actual generated production registry. Command: `.venv/bin/python -m pytest runner/tests/test_tensor_composition_configuration.py runner/tests/test_tensor_composition_engine.py -q`. Local log: `.local/tensor-composition-runtime-full-green.log`.
- Applicable prior worker regression: **865 passed**, no skips, one existing MPS `embedding_renorm_` CPU-fallback warning. Files: `test_runner`, `test_tensor_layout`, `test_tensor_layout_axes_independent`, `test_multi_output_preview`, `test_checkpoint_lifecycle`, `test_preview`, `test_frozen_checkpoint`, `test_embedding_axes_independent`, `test_configurable_activations`, `test_pooling_independent`. Local log: `.local/tensor-composition-prior-regression.log`.

Final tests include independent Python coordinate loops for ordered outputs and mixed-sign weighted gradients into both operands; CPU/MPS batches one and three; noncontiguous source views and no input mutation; all four full graph/portable model value and gradient parity; other axes, signed aliases, rank limits, zero/noninteger/negative Add multipliers and unequal-rank broadcasting; complete integer graphs through Embedding; signed int64 overflow parity with historical Add; actual per-member CPU/MPS training and saved captures/checkpoints/portable inference; and integer adaptive-max named indices preserved through composition, portable code and cached preview continuation.

Resource tests separately refuse oversized sources and oversized combined outputs while each input remains individually within budget, before any finite scan or allocation. Configuration tests reject mixed/unsupported types, unknown settings, invalid dimensions, moved batch axes, incompatible extents and unauthorized integer Add behavior. Extreme finite alpha is accepted as configuration; actual overflowing output refuses.

Historical semantic SHA-256 goldens were captured from pre-change HEAD code and remain exact:

| Legacy graph | SHA-256 |
|---|---|
| Concatenate | `06fd861ab19628b13f29088354e58b3d95d8fe0bee9a4c1cdc5052be2c4a1376` |
| Add | `e249cc929059b1ed11eb24f86c049b40f2a66f57c828d0a26702bc088437f02f` |

Actual CPU checkpoint resume from historical empty settings to explicit defaults matches uninterrupted training model, optimizer and RNG state exactly for both operations. Nondefault resume/cache requests refuse as stale/incompatible. Actual preview cache continues from the operation while reusing its upstream operands. Named pooling indices remain int64 in cached nested output storage.

During development, tests were corrected to request explicit activation capture, use the real `inspect=True` API, remove schema-2 shape operators from schema-1 fixtures, and follow the authored bundle's nested definition identity. These are fixture corrections; no production contracts were relaxed to obtain GREEN. Stack/Permute runtime failures while production generation was pending were treated as a dependency, not bypassed with a fixture registry.

## Limits

Finite binary Concatenate/Add/Stack and single-source Permute only. No batch broadcasting, scalar/ragged/sparse tensors, mixed dtype/device, float64 production adapter, in-place edits, arbitrary custom code or many-source variadic composition. Int64 has value/type evidence, not an integer-gradient claim. This receipt does not replace native UAT, the root's previous accepted-unit replay, or lesson assessment evidence.
