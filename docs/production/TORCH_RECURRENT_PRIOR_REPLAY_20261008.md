# Recurrent slice prior accepted runner replay — 2026-10-08

Status: **terminal host GREEN**, no acceptance/catalog mutation. Root owns Swift, native, packaging and accounting closure.

## Exact accepted selection

Baseline `9d6b199` and the current catalog agree on **59 accepted units** after recursive removal of only `sha256` fields. Every other nested semantic field is unchanged. This replay recursively collected their bound evidence paths and selected exactly **76 actual runner Python test files**; it did not substitute an older 54-unit selection or add the new recurrent runtime/learning suite to the prior numerator. All bound paths exist.

Selection `.local/recurrent-prior-runner-selection.json` records the 59 exact IDs, all bound paths/current snapshots, 167 frozen host source/input hashes, 72 frozen prior document hashes and 110 original baseline PNG hashes. Every frozen document/image was independently compared against its actual committed bytes at `9d6b199` before execution. The mutable icon catalog `docs/design/component-icons-v2/catalog.json` is excluded from frozen receipts. Core/App sources and tests were recorded separately as provisional snapshots because root still owns active Swift fixture repairs; they were not falsely included in this Python freeze gate.

## Actual command and result

Reproducible helper: `.local/replay-recurrent-prior-runner.py`. Host `.venv`, never signed bundled Python:

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
  runner/tests/test_learning_schedulers.py \
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
  runner/tests/test_scheduler_cli_transport.py \
  runner/tests/test_scheduler_operation_contract.py \
  runner/tests/test_scheduler_portable_trace.py \
  runner/tests/test_scheduler_runtime.py \
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

Terminal exit **0**: **5313 passed, zero failures/errors/skips, three existing upstream warnings**, pytest 166.71 seconds; helper wall 167.32 seconds. No runner repairs were needed. Warnings were the existing convolution same-padding copy disclosure, MPS embedding renormalization CPU fallback disclosure and rejected nested-tensor prototype disclosure.

| Evidence | SHA-256 |
| --- | --- |
| `.local/recurrent-prior-runner-selection.json` | `96fe97d9bbc746ab7a629ea3e2a92f93be652feafee0ce685480f22766e35a28` |
| `.local/recurrent-prior-runner-result.json` | `270ef4fb6d25c6f109f11c19b0dc71125ef6552253ef7de21ccad94d56a7604e` |
| `.local/recurrent-prior-runner.log` | `50b8a9f0bbec7ad3c7fd9a781cceb009aec40a84fd4324911eede592b785ac73` |

**No frozen host source drift, prior receipt drift or original icon drift.** All 167/72/110 respective hashes are unchanged. Root's six metadata test fixture files did change during this Python replay; these are explicitly reported as mutable Core snapshots rather than scientific host dependencies:

- `Tests/KwipMLCoreTests/BilinearBundleMetadataTests.swift`
- `Tests/KwipMLCoreTests/ConvolutionBundleMetadataTests.swift`
- `Tests/KwipMLCoreTests/EmbeddingBundleMetadataTests.swift`
- `Tests/KwipMLCoreTests/MatrixBundleMetadataTests.swift`
- `Tests/KwipMLCoreTests/TensorLayoutBundleMetadataTests.swift`
- `Tests/KwipMLCoreTests/TransposedConvolutionBundleMetadataTests.swift`

Their source-bound Swift repair evidence remains root-owned. This receipt does not refresh their catalog hashes or imply a Swift pass. The ignored accounting proposal preserves prior59 recursive non-SHA semantics, keeps every new recurrent binding unverified/planned, and leaves future acceptance/native/source-freeze gates open. The host replay establishes only the actual prior59 bound runner regression on its unchanged host source inputs.
