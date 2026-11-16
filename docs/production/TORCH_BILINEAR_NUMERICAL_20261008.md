# Bilinear independent float32 numerical receipt

2026-10-08. Supplemental verification under [the bounded closure plan](TORCH_BILINEAR_CLOSURE.md), preserving [prior accepted delivery](TORCH_BILINEAR.md). No worker behavior, configuration version, dtype, budget or catalog scope changed. Newly added checks passed on first execution; no product defect or RED is invented.

## Finite verification

`runner/tests/test_bilinear_axes_independent.py` constructs actual worker-owned Bilinear modules through the current real registry and generated standalone Models with strict state loading. Its expected values and every derivative use Python scalar products and `math.fsum`; no PyTorch Bilinear, contraction, einsum, matrix multiplication or autograd produces the reference. Actual autograd measurements compare against that independent reference.

Fifty-six primary cases cover CPU and actual MPS float32, ranks 2–8, bias enabled/disabled and batch 1/3. Both operands are distinct stride-two noncontiguous views, with independently specified feature widths 3 and 2 and output width 2. Leading sample dimensions alternate 2/1; both operands must share these exact dimensions, retaining the no-broadcast contract.

Each worker/generated module compares output, left/right input gradients, shared weight gradients and optional bias gradients. Shared gradients independently sum all batch/sample uses. Output shape/dtype/finite values, state keys/count, unchanged inputs, unchanged fixed-weight train/eval state and separate-row behavior are checked. Evaluation does not alter parameters. Gradients remain attached to the proper input and parameter roles; disabled bias has no parameter or gradient entry.

Sixteen additional rank-3 cases exercise `(in1,in2,out)` = `(1,1,1)`, `(512,1,1)`, `(1,512,1)` and `(1,1,512)` on each device/bias mode. Leading sample dimensions are one to retain graph width budgets; these safe cases do not authorize a 512-cubed weight or change the two-million parameter limit.

Eight complete graph cases cover ranks 5/8 and both bias modes on CPU/MPS, including the two independent Linear/Reshape branches, Bilinear, Flatten and final Linear. Generated Model state and every input/parameter gradient match the worker graph. The primitive/module oracle above remains independently scalar; complete graph parameter-gradient parity is a state/export routing check, not a separately scalar-derived oracle for the Linear branches.

Original bounded numerical tolerances are retained: CPU `atol=1e-6, rtol=1e-5`; MPS `atol=1e-5, rtol=1e-4`. Parameters, operands and upstream gradients are explicit bounded binary-representable values. This is not a precision claim for arbitrary ill-conditioned values, other dtypes/devices or all shape combinations.

## Actual execution

```sh
.venv/bin/python -m pytest runner/tests/test_bilinear_axes_independent.py runner/tests/test_bilinear_contract.py runner/tests/test_bilinear_discovery.py runner/tests/test_bilinear_graph.py runner/tests/test_bilinear_lifecycle.py runner/tests/test_learning_bilinear.py -q -rA
```

**135 passed, no skips, in 3.94 seconds**, Python 3.13.6 / PyTorch 2.14.1 with actual CPU and MPS.

| Suite | Passing tests |
|---|---:|
| New scalar leading-axis/module/state/export checks | 80 |
| Existing configuration/layout/resource contract | 22 |
| Existing descriptor discovery | 2 |
| Existing graph/value/gradient/export | 10 |
| Existing actual trained lifecycle, both bias modes | 2 |
| Existing measured supplied-parameter lessons | 19 |

Existing graph/lifecycle fixtures intentionally construct a private compatible registry; new supplemental tests use the real current registry. The lifecycle replay preserves actual CPU training, exact model/optimizer/RNG/metric resume, parent immutability, gradient inventory, trained export, cached inference, incompatible bias refusal and weights-only fine-tuning. Learning replay independently checks original supplied operand/parameter gradients, authored weight transpose routing, higher-rank unequal widths, privacy/policy/shape/refusal and budget/cancellation controls. Those float64 learning and raw-model references do not expand product float32 support. No additional MPS exact-resume guarantee is claimed.

Ignored `.local/bilinear-axes-independent-worker.log` records exact command, runtime, counts and source hashes. New test SHA-256: `1d0243fa75b17b706034bb70473efa331ad9d705047559c80e90d4b0e1cd7a71`. Observed engine: `71f77392cd5c91b7e8762f7e219518fb948f8aad31cb6249c291ee2f344433ef`; registry: `2b0774cf2bf33e5b631abde9184c6239dd1aa38f7cca455f481d368ae3b4df97`. Shared source changes require an applicability review before evidence binding. Scoped `git diff --check` passed.

This is local development-runtime numerical evidence. Native current-bundle reconciliation, packaged-launch acceptance, catalog accounting and formal release gates remain independently owned; preserved historical native observations are not silently relabeled as current observations.
