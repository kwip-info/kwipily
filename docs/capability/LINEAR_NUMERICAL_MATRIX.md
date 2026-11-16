# Linear numerical acceptance matrix

2026-10-05. Additional verification of existing implementation, not a red-first implementation claim or catalog completion. Only `runner/tests/test_linear_acceptance_matrix.py` and this evidence document were added; no production source changed.

## Scope and observations

Executed 40 tests: 32 forward/backward cases and eight seed/reset/branch cases. All passed in 2.49 seconds, including actual CPU and MPS. Host: arm64 macOS 26.6, Python 3.13.6, PyTorch 2.14.1. These are local hardware observations, not a clean-machine or cross-version guarantee.

The numerical cases cross both devices, bias enabled/disabled, batch 1/7 and internal width pairs `(1,1)`, `(2,3)`, `(3,2)`, `(512,512)`. Each complete product graph projects its required two input features into the first width, applies the tested width pair, and projects to the required two outputs. This exercises the largest configured Linear width without claiming arbitrary application input/output dimensionality. Inputs are noncontiguous dense float32 views. Parameters are seeded and scaled by 0.3 to keep conditioning bounded; this is not a stress test of every possible weight magnitude.

The product adapter executes the graph. The independent algebraic oracle clones parameters and evaluates matrix multiplication, transpose and optional addition directly; it does not call Linear.forward. Arbitrary upstream gradients compare input and every active parameter gradient. CPU tolerances are `rtol=1e-5, atol=1e-6`; MPS tolerances are `rtol=1e-4, atol=1e-5`.

| Device | Numerical cases | Maximum output absolute error | Maximum gradient absolute error |
|---|---:|---:|---:|
| CPU | 16 | 1.0710209608078003e-08 | 6.146728992462158e-08 |
| MPS | 16 | 5.587935447692871e-09 | 2.9802322387695312e-08 |

Maximum relative errors were 0.0001160040992544964 (CPU) and 0.0009559685713611543 (MPS), using `abs(actual-reference)/max(abs(reference),float32 epsilon)`. Near-zero references can produce a larger relative number while satisfying the declared combined absolute-plus-relative criterion. Pass/fail uses `torch.testing.assert_close`, not the relative statistic alone. No tolerance was widened after observing results.

Eight additional tests verify same-seed initialization, explicitly requested parameter reset, preserved bias ownership, independent branch parameter storage and summed fan-out input gradients on both devices/bias modes. Reset checks operate on raw underlying modules; they do not claim a native reset UI exists. The branch oracle explicitly sums the two matrix derivatives and checks each weight/bias derivative.

## Reproduction and local evidence

```sh
.venv/bin/python -B -m pytest runner/tests/test_linear_acceptance_matrix.py -q
.venv/bin/python -B -m runner.tests.test_linear_acceptance_matrix
```

The module command prints per-case numerical measurements. A one-off environment-bound report is retained at ignored `.local/acceptance/linear-matrix.json`, schemaVersion 1. It includes exact dimensions/bias/batch/device, status, tolerances, output/gradient absolute maxima, relative maxima, available devices, runtime/environment, source revision and test-source SHA-256. The original checkout HEAD was `5912f7024280362ac3df9a95e2f2b44259457beb`, but production sources were dirty at observation: that base revision alone does not identify what executed. After the unchanged production slice was committed, exact bytes of engine.py, product_execution.py, graph_semantics.py, data.py and the component registry were compared with Git commit `2d05d89ff0fe81070624f567f471c072ac7cf472`; all matched. The report now records SHA-256 for each relevant production source, `productionSourcesDirtyAtObservation: true`, the original base and this resolved production-content commit under `observationProvenance`. This is a post-observation provenance resolution, not a numerical rerun; worker sources were held unchanged through resolution. The new test file was uncommitted at observation, so its separate hash identifies the test code. No report file is added to Git. The report covers 32 numerical cases; the eight additional tests are represented by the pytest evidence above, not fabricated numeric rows.

All recorded cases executed: no MPS skips on this host. A machine without MPS will explicitly skip its cases rather than count unavailable hardware as acceptance. Full run artifacts and environment receipts are observations to review, not self-proving attestations.

## Remaining acceptance boundaries

This matrix does not replace invalid-input, export, checkpoint/resume, cache, cancellation, learning or native UAT evidence. Existing dedicated suites cover those separately and must be reviewed for the declared Linear modes. It does not establish support for float64 in the product adapter, sparse/nested tensors, CUDA, arbitrary ranks, higher-width application inputs, every batch size, every initialization distribution, mixed precision or other optimizers. Raw GraphModel float64 reference tooling and standalone PyTorch exports retain their documented independent semantics. No denominator, exclusion, mode review or capability completion field changed.
