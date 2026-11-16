# LazyLinear independent float32 numerical receipt

2026-10-08. Supplemental verification of the existing [accepted LazyLinear adapter](TORCH_LAZY_LINEAR.md). No worker, materialization policy, configuration version, limits or catalog modes changed. New tests passed on their first actual execution; no product RED is invented.

## Finite scope

`runner/tests/test_lazy_linear_axes_independent.py` creates real current-registry GraphModels containing actual uninitialized `nn.LazyLinear`. It invokes the existing `materialize_lazy_parameters` safe path before assigning explicit bounded parameters. The original authored graph remains unchanged; input width is inferred and never introduced as editable `in_features` configuration.

Fifty-six primary cases cover float32 CPU and actual MPS, ranks 2–8, bias enabled/disabled, batches 1/3, inferred input width 3 and output width 2. Earlier sample axes alternate 2/1. Actual noncontiguous inputs use stride-two last-axis views. Expected outputs, input gradients and shared weight/optional bias gradients reuse the independent `math.fsum` scalar reference in `test_linear_axes_independent.py`; the reference does not call Linear/LazyLinear, tensor matrix multiplication or autograd. Actual measured gradients sum correctly across every leading dimension.

Checks retain output shape/dtype and batch rows, finite values/gradients, parameter counts/state keys, absent disabled-bias state, unchanged inputs and independent batch-row behavior. Fixed-weight train/eval output is equal and parameters remain unchanged. These primitive checks do not claim that training makes no state changes.

Repeated explicit materialization leaves all parameter bytes and CPU/MPS RNG state unchanged. After initialized state is loaded into the actual generated `nn.LazyLinear` architecture, forward execution makes no additional CPU/MPS RNG draw or random reinitialization. Exported source retains lazy authorship. Each generated module separately meets the scalar oracle, and the complete generated graph matches worker output and every input/parameter gradient using the same loaded state dictionary. This verifies both the lazy boundary and its shape/export routing without claiming all-eager and mixed-lazy seeded initialization equivalence.

Twelve additional rank-3 cases check `(inferred input, output)` = `(1,1)`, `(1,512)` and `(4096,1)` on both devices/bias modes. Width 4096 is constructed from a legal 512-feature projection and repeated named-port Concatenate operations rather than inventing a 4096-feature application data source or configured eager Linear width. Earlier axes are one for endpoint cases to retain existing budgets. The inferred width limit 4096, output limit 512 and two-million parameter limit remain intact; maximum limits are not claimed simultaneously.

Original bounded fixture tolerances remain CPU `atol=1e-6, rtol=1e-5`, MPS `atol=1e-5, rtol=1e-4`. No float64 product, mixed precision, other devices or general precision guarantee is introduced.

## Actual execution

```sh
.venv/bin/python -m pytest runner/tests/test_lazy_linear_axes_independent.py runner/tests/test_lazy_linear_contract.py runner/tests/test_lazy_linear_discovery.py runner/tests/test_lazy_linear_graph.py runner/tests/test_lazy_linear_lifecycle.py -q -rA
```

**109 passed, no skips, in 3.43 seconds**. Python 3.13.6 / PyTorch 2.14.1; actual CPU and MPS available.

| Suite | Passing tests |
|---|---:|
| New scalar rank/materialization/state/export checks | 68 |
| Existing configuration/inferred-resource contract | 10 |
| Existing discovery descriptor | 2 |
| Existing lazy graph and refusal tests | 21 |
| Existing actual lifecycle, including MPS | 8 |

Existing graph/lifecycle tests use compatible private-registry fixtures; new supplemental cases use the current real registry. Replay retains true deferred construction, materialization without running the authored graph, strict schema/dtype/budget rejection, explicit width guards before initialization, deterministic registration, CPU exact trained resume with optimizer/RNG/metrics, fine-tune/frozen load, immutable parent, initialization/cache/checkpoint width compatibility, trained standalone reload and actual MPS training/previews. MPS remains weights-only; CPU exact-resume promises are not generalized.

Ignored `.local/lazy-linear-axes-independent-worker.log` records exact command, runtime, module counts and source/reference hashes. New test SHA-256: `9e308f66b5b09120ef9511ada0b8f949a301a2d107f3bc5dd0668b76ed8fd01c`; reused scalar-reference file: `be2459e53ff59a6ba881403d9887768c4ee38134ec6376e8840d6b9bbb181131`. Observed engine: `71f77392cd5c91b7e8762f7e219518fb948f8aad31cb6249c291ee2f344433ef`; registry: `5a537fc49fcbfca6778e69cc8adb127152fc093a4a8f01ad2ed34598cd31235a`. Applicability review is required after shared source changes. Scoped `git diff --check` passed.

This is local development-runtime numerical evidence, not native lesson acceptance, packaged-launch verification, catalog completion or formal release readiness. Generated LazyLinear artwork already has distinct original provenance in [MODEL_RECEIPT.md](../design/component-icons-v2/MODEL_RECEIPT.md); this test slice neither regenerates nor attests native icon rendering.
