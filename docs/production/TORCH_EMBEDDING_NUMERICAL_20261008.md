# Embedding independent numerical receipt

2026-10-08. Supplemental verification under [the bounded closure plan](TORCH_EMBEDDING_CLOSURE.md), preserving the eight current controls and inference/training/parameter-gradients/state-roundtrip modes. No worker implementation, configuration version, resource ceiling or catalog scope changed. Newly authored verification passed on its first actual execution; no product RED is invented.

## Independent lookup, gradients and table state

`runner/tests/test_embedding_axes_independent.py` independently computes row gathers, repeated-row gradient scatter sums, optional per-row frequency divisors and selected-row norm limits using Python indices, scalar powers and `math.fsum`. No PyTorch row gather, embedding, tensor norm, autograd or worker output produces the expected reference. Actual autograd measures gradients for the table parameters. Integer row selectors have no differentiable input gradient; none is fabricated.

Seventy-two primary cases cross CPU/actual MPS, integer-input ranks 2–7, batches 1/3 and three policies: no padding, positive padding with frequency scaling, and equivalent normalized negative padding. Every source is an actual stride-two noncontiguous `int64` view. Output is dense float32 with one added vector axis (ranks 3–8). Explicit repeated rows, imported nonzero padding values and two entirely unselected rows make their distinct behaviors observable. Worker-owned modules come from the current real graph registry; generated standalone Models load exact table state and separately meet the scalar oracle.

Both outputs and dense table gradients compare independently. Imported padding values remain visible; its gradients are zero. Repeated non-padding rows accumulate all uses; frequency scaling divides those contributions by the row's occurrence count over the complete input. Unselected rows retain their original weights and receive zero gradient. Inputs remain unchanged and observations remain separate. Parameter shape/state and freeze flags are measured; integer indices do not acquire gradients.

Twelve norm cases cross devices, norm orders 1/2/3 and freeze enabled/disabled at rank 3/batch 3. Maximum norm is a meaningful nondefault 0.75. The scalar reference computes each selected row's norm and applies the existing renormalization convention `maximum / (norm + 1e-7)` only when needed. It includes selected padding rows. Forward values and resulting table state compare independently; unselected rows remain unchanged. Frozen weights can still renormalize, while producing no parameter gradient. Trainable max-norm gradients are scatter gradients of the resulting table; no derivative through the in-place renormalization operation is claimed.

Four sparse-frozen cases check CPU/MPS inference with and without maximum norm, preserving sparse configuration while requiring no sparse backward kernel. Generated full-graph outputs match. Two additional rank-7 graph cases on CPU/MPS compare generated final output and every trainable parameter gradient under padding/frequency scaling. Surrounding Linear layers own their own parameters; table-gradient primitive expectations above remain scalar-independent, while complete graph gradient parity is an export/state/routing check.

One refusal case checks sparse plus frequency, the graph's two-million parameter ceiling, floating/boolean/out-of-range indices, empty/rank-invalid inputs, the 100-million output-element allocation boundary with tiny expanded views, and trainable sparse Adam refusal before worker artifact directory creation. Existing strict graph/dtype/configuration suites and lifecycle receipts are reused rather than duplicating a large full-training matrix.

CPU fixture tolerances are `atol=1e-6, rtol=1e-5`; MPS `atol=1e-5, rtol=1e-4`. They were not widened after observation. Fixtures are bounded; no general conditioning, alternate dtype, sparse MPS backward, input-gradient or additional optimizer promise is implied.

## Actual execution and MPS applicability

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest runner/tests/test_embedding_axes_independent.py runner/tests/test_embedding_contract.py runner/tests/test_embedding_graph.py runner/tests/test_embedding_discovery.py runner/tests/test_embedding_lifecycle.py runner/tests/test_embedding_preview.py runner/tests/test_learning_embedding.py -q -rA
```

**145 passed, no skips, in 2.10 seconds**. Python 3.13.6 / PyTorch 2.14.1, CPU and actual MPS available. Explicit `PYTORCH_ENABLE_MPS_FALLBACK=1` matches the existing declared MPS maximum-norm policy: `aten::embedding_renorm_` falls back to CPU. The single warning identifies that fallback. These cases do not claim native MPS renormalization or its performance. Other tested tensor execution/gradients use their selected device. No payload or backend settings were mutated.

| Suite | Passing tests |
|---|---:|
| New scalar gather/scatter/norm/export/refusal checks | 91 |
| Existing primitive/configuration tests | 9 |
| Existing graph/export tests | 4 |
| Existing exact discovery descriptor | 1 |
| Existing training/checkpoint/capture/fine-tune tests | 19 |
| Existing activation-cache/renormalization preview tests | 12 |
| Existing measured embedding lessons | 9 |

The first run passed 145 in 2.82 seconds. Replacing an unrelated layout-rank negative control with direct embedding rank/empty controls yielded the final run above. No worker defect occurred. Existing regression replay covers actual CPU exact model/optimizer/RNG/metric resume, trained table/source reload, MPS weights-only fine-tuning, immutable parent, frequency-aware gradient captures, typed captured coordinates, norm-mutation cache restoration and corruption refusal. Exact CPU continuation is not generalized to MPS.

Ignored `.local/embedding-axes-independent-worker.log` records exact command, runtime, counts and source hashes. New test SHA-256: `860d37f404eb5125f68d59a15ef5e2077c2be17d9cffe0c97a215226f0945886`; checked embedding: `eb8990d130fae757a0f095327e9a5e09408de605b30c57699fcfa5fa51d3c5e0`; engine: `71f77392cd5c91b7e8762f7e219518fb948f8aad31cb6249c291ee2f344433ef`; observed registry: `4ea31cc7bd985b1ee268ca8be2e184d3c0a9dcc3022d98eb5d936d6d4dabd4e6`. Later source changes require applicability review. Scoped `git diff --check` passed.

This is local development-runtime numerical evidence. Current native lesson acceptance, packaged launch, generated-art rendering, authored-bundle accounting and formal release gates remain separately owned. Representative policies are deliberately separated from rank coverage; this does not claim every cross-product of all eight settings.
