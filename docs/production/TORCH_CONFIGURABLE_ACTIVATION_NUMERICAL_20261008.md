# SiLU / LeakyReLU / ELU worker numerical evidence · 2026-10-08

Finite new schema-2-only activation family. Worker production changes are the new `runner/configurable_activation_configuration.py`, additive `runner/graph_semantics.py` admission and additive `runner/engine.py` shape/dtype/config/execution/export/discovery routes. Existing operator settings/versions remain unchanged. Root and other assigned agents own authored metadata, Swift controls, generated artwork, assessed lessons, native acceptance and coverage credit.

## Contract and reference

Reviewed primary documentation: [PyTorch2.14 SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html), [LeakyReLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LeakyReLU.html), [ELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ELU.html). Values apply independently and preserve shape. The worker uses the matching standard functional operations with explicit `inplace=False`; portable generated code uses the same declared configuration and explicit out-of-place execution. Each component is deterministic/parameterless in its declared dense float32 CPU/MPS scope; this does not promise bitwise equality across hardware.

SiLU accepts exactly empty configuration. LeakyReLU normalizes omitted `negative_slope` to0.01; ELU normalizes omitted `alpha` to1.0. Signed coefficients must be real finite numbers in `[-3.4028234663852886e38,3.4028234663852886e38]`, excluding Boolean values. Unknown keys, explicit inplace requests even false, null/text/NaN/infinity/overflow are refused. The default omission and explicit default have equal semantic identities. Schema1 is refused. Each exact descriptor declares version1.0.0, adapter `kwip.runner.graph-v2.<id>`, schema[2], float32, rank2–8, preserved batch/shape, zero learned parameters and immutable out-of-place behavior. No configuration, axis or state behavior is silently broadened.

Independent expectations use scalar stable sigmoid times x for SiLU, signed slope times x on LeakyReLU’s nonpositive branch, and signed alpha times `math.expm1(x)` on ELU’s nonpositive branch. Scalar analytic derivatives use no autograd/reference Torch operator. Actual zero derivatives on CPU/MPS are verified: SiLU0.5, LeakyReLU negative_slope, ELUalpha. Negative/zero coefficients are explicitly tested; positivity is not invented as a constraint.

## Actual test-first record

`.local/configurable-activation-worker-red.log`: actual missing-module RED collection error before implementation. The first helper/semantics-only integration attempt (`.local/configurable-activation-worker-first-green.log`) correctly failed unsupported worker execution/discovery:233 failed22 passed, while strict helper controls already worked. That intermediate log is not described as a GREEN. Completing the additive engine paths yielded255 passed in2.02s (`.local/configurable-activation-worker-green.log`). No assertion or numerical tolerance was weakened.

After adding finite endpoint/default identity, rank/resource preallocation guards and actual MPS trained-preview controls, combined actual `.local/configurable-activation-regression-worker.log`:397 passed in7.43s, no skips or warnings.263 new family checks plus134 existing Sigmoid/Tanh/ReLU/GELU/Softmax regression checks. Runtime is actual development Python3.13.6/PyTorch2.14.1 on available CPU/MPS, with explicit fallback permission. The initial typed command used nonexistent test_softmax_graph.py and collected no tests; the successful corrected command below uses the actual test_softmax.py module.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest \
  runner/tests/test_configurable_activations.py \
  runner/tests/test_sigmoid_graph.py \
  runner/tests/test_tanh_graph.py \
  runner/tests/test_relu_graph.py \
  runner/tests/test_gelu.py \
  runner/tests/test_softmax.py -q -rA
```

## Finite verified behavior

Seven variants (SiLU, default/negative/zero LeakyReLU, default/negative/zero ELU) cover rank2–8, actual CPU/MPS and batch1/3 with noncontiguous input storage and explicit scalar derivatives.196 cases preserve shapes and input bytes, verify train/eval equivalence, generated standalone values/gradients and strict learned-state loading, and require no activation state keys. Another14 cases independently check saturation from−1000 to1000, near-zero values, zero derivatives and out-of-place sibling-input preservation. Fourteen hand-computed affine chain-rule cases verify output, input derivative and all surrounding Linear weight/bias derivatives. These sparse coefficient/tensor examples do not claim blanket accuracy for every coefficient or ill-conditioned graph.

Three exact descriptor/default/layout/schema checks,22 strict bad-configuration cases and3 typed integer/parameterless-training refusal checks cover admission. Signed range endpoints/default semantic identity are verified, huge integer overflow is refused, and rank9/sample budgets fail before Linear allocation. A large valid LeakyReLU coefficient with sufficiently large negative input produces infinity; the existing ProductExecutionAdapter rejects that captured output. Finite configuration does not guarantee every resulting float32 product is finite; the engine does not silently clamp the mathematical result.

Three real CPU lifecycle cases train full versus resumed sequences and compare exact model/optimizer/RNG state and metrics; initialized parameter settings remain part of semantic checkpoint/cache identity. Trained cache reuse and stale changed-type/coefficient refusal, fresh fine-tune seed/epoch/reset optimizer and trained generated-state export parity are verified. Three actual MPS cases train and preview a captured checkpoint; those checkpoints retain the `weightsOnly` contract. Exact CPU continuation is not advertised as exact MPS continuation. No CPU/MPS native-kernel promise follows from the absence of warnings.

The initial worker RED/GREEN used temporary exact new definitions while metadata was authored independently. The final fixture now reads actual production REGISTRY directly, verifies one unique authored record per ID, its ports/category/configuration/defaults/signed bounds, and never injects or replaces records or invents titles. Final actual-registry replay `.local/configurable-activation-actual-registry-worker.log` passes397 checks in6.69s, no skips or warnings, using the same scoped command above. This is actual worker/generated-model evidence for the authored33-entry registry; native palette/lesson acceptance remains root-owned.

## Source identities

| Path | SHA-256 |
|---|---|
| `runner/configurable_activation_configuration.py` | `352b73b32345df3610994b588bb5e99a3cfeb50491dc09f30d365a2a080f5839` |
| `runner/engine.py` | `8e1a8f5014e53bcb2c14876b162b54d055438b69ed5159784edae977a6dcb753` |
| `runner/graph_semantics.py` | `44bf3f35954188bc17495aa04ef5c42afef5760c5cae71bb4b9062522b598fb0` |
| `runner/tests/test_configurable_activations.py` | `14eaf58dfb0a93eafe8dcf04c47e87e4bad37b50dcc578101957eb5f49b534ea` |
| actual `Sources/KwipMLCore/Resources/components.json` | `dbda80d878d24ec36654c4aca7360121ca3cbd185cf6161bf444b80277f6d283` |

No Swift/build/native/catalog evidence is manufactured here. New helper/package inclusion, authored registration and final source reconciliation are root-owned release seams. No further numerical or lifecycle matrix is needed unless final metadata or a scoped failure changes behavior.
