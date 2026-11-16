# Scheduler final-source prior runner replay — 2026-10-08

Status: terminal GREEN for the bound prior runner replay. No acceptance or catalog mutation. Root owns Swift/native/package and scheduler closure.

## Selection and command

The current catalog has54 accepted units. This replay recursively collected their evidence `path` fields, deduplicated actual `runner/tests/*.py` references, and checked all bound paths exist. It selected71 test files. This preserves the accepted evidence selection rather than silently expanding into unrelated product/backends or using a historical49-unit list. The newly authored scheduler runtime/CLI/learning suites are separate scoped evidence and are not included in the prior numerator.

The actual ignored reproducible helper is `.local/replay-scheduler-prior-runner.py`; selection `.local/scheduler-prior-runner-selection.json` records the54 exact IDs, command inputs, 566 before-run bound source/test/art hashes, and62 before-run old document hashes. Mutable `docs/design/component-icons-v2/catalog.json` is deliberately not classified as a frozen receipt. No old source/test/receipt or catalog was edited.

Actual command (host `.venv`, no signed bundled Python):

```sh
.venv/bin/python -m pytest -q \
  runner/tests/test_adaptive_pooling_configuration.py \
  runner/tests/test_adaptive_pooling_engine.py \
  runner/tests/test_bilinear_axes_independent.py \
  runner/tests/test_bilinear_contract.py \
  runner/tests/test_bilinear_graph.py \
  runner/tests/test_bilinear_lifecycle.py \
  runner/tests/test_checkpoint_lifecycle.py \
  runner/tests/test_configurable_activations.py \
  runner/tests/test_conv1d_lifecycle.py \
  runner/tests/test_conv2d_lifecycle.py \
  runner/tests/test_conv3d_lifecycle.py \
  runner/tests/test_conv_transpose1d_lifecycle.py \
  runner/tests/test_conv_transpose2d_lifecycle.py \
  runner/tests/test_conv_transpose3d_lifecycle.py \
  runner/tests/test_convolution_independent.py \
  runner/tests/test_embedding_axes_independent.py \
  runner/tests/test_embedding_lifecycle.py \
  runner/tests/test_gelu.py \
  runner/tests/test_gelu_independent.py \
  runner/tests/test_layer_norm_axes_independent.py \
  runner/tests/test_layer_norm_contract.py \
  runner/tests/test_layer_norm_execution.py \
  runner/tests/test_layer_norm_lifecycle.py \
  runner/tests/test_lazy_linear_axes_independent.py \
  runner/tests/test_lazy_linear_lifecycle.py \
  runner/tests/test_learning_adaptive_pooling.py \
  runner/tests/test_learning_backward_chain.py \
  runner/tests/test_learning_bilinear.py \
  runner/tests/test_learning_configurable_activation.py \
  runner/tests/test_learning_convolution.py \
  runner/tests/test_learning_frozen_standardization.py \
  runner/tests/test_learning_gelu.py \
  runner/tests/test_learning_layer_norm.py \
  runner/tests/test_learning_lazy_linear.py \
  runner/tests/test_learning_optimizers.py \
  runner/tests/test_learning_pooling.py \
  runner/tests/test_learning_recovery.py \
  runner/tests/test_learning_remaining_activation.py \
  runner/tests/test_learning_softmax.py \
  runner/tests/test_learning_spatial_patches.py \
  runner/tests/test_learning_stateful_normalization.py \
  runner/tests/test_learning_tensor_composition.py \
  runner/tests/test_learning_tensor_routing.py \
  runner/tests/test_learning_transposed_convolution.py \
  runner/tests/test_linear_acceptance_matrix.py \
  runner/tests/test_linear_axes_independent.py \
  runner/tests/test_linear_bias.py \
  runner/tests/test_matrix_axes_independent.py \
  runner/tests/test_optimizer_cli_delivery.py \
  runner/tests/test_optimizer_operation_contract.py \
  runner/tests/test_optimizer_runtime.py \
  runner/tests/test_pooling_independent.py \
  runner/tests/test_product_execution.py \
  runner/tests/test_relu_graph.py \
  runner/tests/test_relu_lesson_assessment.py \
  runner/tests/test_remaining_activation_runtime.py \
  runner/tests/test_sigmoid_graph.py \
  runner/tests/test_sigmoid_learning_evidence.py \
  runner/tests/test_softmax.py \
  runner/tests/test_softmax_axes_independent.py \
  runner/tests/test_spatial_patch_runtime.py \
  runner/tests/test_stateful_normalization_runtime.py \
  runner/tests/test_tanh_graph.py \
  runner/tests/test_tanh_lesson_assessment.py \
  runner/tests/test_tensor_composition_configuration.py \
  runner/tests/test_tensor_composition_engine.py \
  runner/tests/test_tensor_layout.py \
  runner/tests/test_tensor_layout_axes_independent.py \
  runner/tests/test_tensor_routing_runtime.py \
  runner/tests/test_transform_graph.py \
  runner/tests/test_transposed_convolution_independent.py
```

Catalog SHA-256 at selection: `533ea15b3669250345f4e592c2b3823e7732b34c62906f781aa109598702f8f7`.

| Relevant final-source bound path | Before-run SHA-256 |
|---|---|
| `runner/engine.py` | `7cbf1b1578e6e94434887ed660f3a6c33f36f99f554500cdef5132b627b98b72` |
| `runner/checkpoint_lifecycle.py` | `1309f381f8b3b526ac31e732c8fb0dd403624f0593595bbc26cd712f3db365a9` |
| `runner/optimizer_configuration.py` | `bb91d8c63a5b44896378a8bd22635481bf49f9377ff5aa19c0492f2adf5aba30` |
| `Sources/KwipMLApp/Resources/inference.py` | `2a0ade05644aaeb62cc061e475f9bbdcea4a65584d0720d3d2aed92bd8e623f9` |

## Actual terminal result

Host pytest exited0: **5049passed, zero failures/errors/skips, three upstream warnings**, 155.68seconds (helper wall156.37seconds). No source repairs were needed. Actual terminal: `5049 passed, 3 warnings in 155.68s (0:02:35)`.

- Log `.local/scheduler-prior-runner.log`: SHA-256 `85870cc8efe24fee7f66d1cc0ffc01e4b499756a9feed6d4a225f5b5e504ee4c`.
- Selection `.local/scheduler-prior-runner-selection.json`: SHA-256 `683321ef6cb55908e046b7e7ea902d0cb51507222104b24939ca08dcb7da1a74`.
- Result `.local/scheduler-prior-runner-result.json`: SHA-256 `0bce72ce7437421e15f32c22f5d02b173c4c64e379a7efeb49752eaf3aa8fc25`.
- All566 bound source/test/art snapshots were byte-identical before/after. All62 old document snapshots were byte-identical. No missing bound paths.

Warnings were existing upstream convolution same-padding, MPS embedding renormalization CPU fallback, and nested-tensor prototype disclosures; inspect the retained log for exact originating test lines. This receipt establishes the54 accepted units' bound runner regression on current sources, not a blanket framework/backend completeness claim. Authoring/Swift/native/distribution and the new scheduler slice remain their separately reported gates. No old acceptance receipts, catalog bindings or numerator were edited.
