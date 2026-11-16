# Matrix multiply independent numerical receipt

2026-10-08. Supplemental verification of existing schema-2 Matrix multiply under the reviewed forward, input-gradients and shape-contract modes. No new trainable parameters, checkpoint state, modes, budgets or execution behavior were introduced. First execution passed; no product RED is invented.

## Independent checks

`runner/tests/test_matrix_axes_independent.py` uses Python coordinate enumeration and index offsets to align sample prefixes after the observation batch. Expected matrix products and gradients for both original operands use scalar multiplication and `math.fsum`. Broadcast gradient contributions reduce back to their original singleton/absent sample axes. Neither tensor matrix multiplication nor autograd computes the reference; actual autograd measures the implementation against it.

Sixty-four principal cases cross CPU/actual MPS float32, total ranks 3–8 and batch 1/3. Every operand is a stride-two noncontiguous view with explicitly bounded binary-representable values. Cases include equal leading sample dimensions, complementary singleton prefixes and a shorter right-hand operand sample prefix. Rank 3 has no prefix and is counted once. Every result retains separate batch rows; batch 1 is an actual observation, never a broadcast sample dimension. Forward values, shapes, float32/finite output, both input gradients, unchanged operands and independent row behavior are checked.

Eight safe dimension controls across both devices cover 1-by-1 products, contraction width 512, and a 512-row or 512-column result with other dimensions one. These are bounded wide fixtures, not new feature minima/maxima or a full dimension cross-product. Matrix shape limits remain the existing concrete sample/total output budgets.

Four actual current-registry graph cases cover rank-5 complementary singleton prefixes and rank-8 shorter-prefix broadcasting on CPU/MPS. The captured product and both intermediate-input gradients independently meet the scalar oracle. The actual generated standalone Model loads the same state and matches the worker graph's final output and every public-input/parameter gradient. Matrix multiply itself owns no parameters: only the Linear projection/head nodes have state. Complete graph gradient parity is an export/routing check; it is not a separately scalar-derived gradient oracle for the surrounding Linear layers.

One refusal test verifies that observation counts cannot broadcast, incompatible sample prefixes/contraction widths fail, and the total output-element budget is checked before `torch.matmul` allocation. Expanded singleton views keep the budget negative control small. Existing suites retain malformed schema/configuration/dtype/layout/device/batch controls, finite differences, actual MPS comparison and measured learner graph/budget/cancellation checks.

CPU tolerances are `atol=1e-6, rtol=1e-5`; MPS `atol=1e-5, rtol=1e-4`. No tolerance was widened after observation. The helper's existing raw float64 reference domain does not broaden the product graph's float32 contract. No CUDA, mixed precision, integer matrix products, vector promotion or batch broadcasting is claimed.

## Actual execution

```sh
.venv/bin/python -m pytest runner/tests/test_matrix_axes_independent.py runner/tests/test_matrix_multiply.py runner/tests/test_matrix_graph.py runner/tests/test_learning_matrix.py -q -rA
```

**115 passed, no skips, in 1.75 seconds**. Python 3.13.6 / PyTorch 2.14.1 with CPU and actual MPS available.

| Suite | Passing tests |
|---|---:|
| New independent axes/broadcast/gradient/export/refusal checks | 77 |
| Existing matrix helper contract/numerics/device tests | 23 |
| Existing graph/export/discovery tests | 6 |
| Existing measured matrix learner worker tests | 9 |

New graph cases use the real registry; existing graph fixtures use their intentionally compatible private registry. No worker source changed. Scoped `git diff --check` passed.

Ignored `.local/matrix-axes-independent-worker.log` records exact command, runtime, counts and source hashes. New test SHA-256: `170b2c49d545927a66960736e00566c9b85612b05334b92f1d4c9a1a45c69199`; matrix helper: `e9fed90d65f8186dbe1146e6cf23cf4046238e22f595bd3bac008f9f33282fdc`; engine: `71f77392cd5c91b7e8762f7e219518fb948f8aad31cb6249c291ee2f344433ef`; observed registry: `b5276a3a5d92e7ec1053498e18b7097808540e7ebed42cd8114edf0f8793444e`. Shared source changes require applicability review before binding these observations.

This is local development-runtime evidence. Native current-bundle/lesson acceptance, packaged launch, catalog accounting and formal release gates remain separate. The test set is finite representative sample-prefix coverage, not every axis-size combination or additional state lifecycle support.
