# LayerNorm v1 bundle closure · 2026-10-08

Owner: root agent; accepted existing implementation: [LayerNorm evidence](../LAYER_NORM_COMPONENT.md). Status: Accepted. Frozen four-mode receipt: [LayerNorm acceptance](TORCH_LAYER_NORM_ACCEPTANCE_20261008.md).

## Finite acceptance

Preserve layer_norm capability1.0.0, configuration/schema2, exact normalized_shape/eps/elementwise_affine/bias fields, float32 CPU/MPS rank2–8 and trailing sample suffix1–7. Retain inference, training, input-gradients and state-roundtrip in the reviewed inventory. Existing three measured lessons and both grouped/whole-sample prepared examples remain the delivery foundation.

Migrate one authoritative definition into the existing authored normalization bundle/generator; exact metadata and canonical example links, no duplicate configuration authority. Test selected compiled graph runtime admission before capture and again at queue dispatch, support-loss cleanup, verified controlled cancellation and disconnected-node negative controls. An isolated ReLU variant of the prepared LayerNorm flow avoids a GELU node masking the missing LayerNorm admission trigger.

Supplement independent numerical checks across supported ranks/suffixes, noncontiguous and ill-conditioned inputs, preserving existing state/checkpoint/export and lesson controls. Reuse the already generated themed LayerNorm artwork with verified provenance. Scoped native configuration/assessed lesson persistence, prepared execution/History/checkpoint/export and light/dark checks precede a frozen four-mode acceptance receipt. Reconcile affected evidence only after appropriate replay.

Later: other normalization capabilities, remaining framework coverage and production/UAT gates retain their own criteria. No blanket PyTorch or supported-machine claim.

## Actual automated evidence · 2026-10-08

Metadata RED:1test/1failure for missing example links (`.local/layer-norm-bundle-metadata-red.log`). Selected-flow admission RED:4tests/28assertion failures at missing/lost support; verified and disconnected controls passed (`.local/layer-norm-workflow-admission-red.log`). Shared admission guard now includes layer_norm; all4admission tests pass. Authored normalization validator RED:1 unsupported-family failure/13rejection controls passed;82normalization/activation/pooling authoring checks now pass (`.local/layer-norm-authoring-green.log`), deterministic output/drift check passes. Learn discovery RED:1duplicate generic-card failure; specialized two-scope card retained and generic duplication removed.

Final scoped Swift run:25Core+15App=40checks pass, no skips/failures; enables actual development runtime (`KWIPML_TEST_DEVELOPMENT_ROOT`), including both prepared examples through real execution and verified History, measured lessons/receipts/configuration/metadata/admission/discovery (`.local/layer-norm-final-scoped-swift.log`). Earlier affected run accidentally omitted that required environment and reports3skips; it is not the final actual-run evidence.

Supplemental numerical checks (`runner/tests/test_layer_norm_axes_independent.py`, SHA256 f1229bb1678079b5bd990aec60c25e620b8522de238131c1e022ad20fa89045a):746tests pass without skips, including672new cases and74existing execution/contract/discovery/lifecycle/lesson checks. Python3.13.6/PyTorch2.14.1, actualCPUandMPS. Strict ordinary/near-constant scalar-oracle matrix448cases retains its numerical thresholds across rank2–8/suffix1–7/allaffine-bias modes/noncontiguous input. Large-offset224cases are explicitly conditioning characterization, not a universal accuracy claim.

### Discovered backend precision limit

The first strict matrix reported92failures/580passes (`.local/layer-norm-conditioning-red.log`): CPUfloat32 inputs near1024 withquarter-step variation lost input/scale gradient precision. Measured maxima: output2.2962e-5, input-gradient0.01097847, scale-gradient0.01195264; MPS input1.18e-6/scale2.62e-6. Worker versus separately constructed pinned nn.LayerNorm forward/gradients are exactly equal: this is backend conditioning, not routing. Subtracting the shared1024offset restores strict scalar agreement; CPUfloat64 math control agrees to2.46e-12.

The explicit characterization requires exact direct-backend parity, finite outputs/gradients, strict centered controls andfloat64math control; conservative fixture-only absolute bounds areCPU output1e-4,input/scale0.02,bias3e-5 andMPS output3e-5,input/scale3e-4,bias3e-5. These are labeled separately from ordinary-input tolerances. No automatic dtype/operator change or promise of arbitrary ill-conditioned accuracy. Bundle Information explains training-fitted centering/scaling and applying that same fitted transform to held-out data. Measurements preserved locally under `.local/layer-norm-conditioning-measurements.json` and `.local/layer-norm-direct-backend-control.json`.

Native candidate verification and frozen four-mode receipt remain open; this checkpoint earns no coverage credit.

Packaged candidate3abdd00f77a0e6bee4ea58513746d740e15c3616e47ed9edcc632a01c6801b2c passes current-worker integrity and deep strict local signature verification (`.local/layer-norm-package.log`). Runtime build-owned `.local/layer-norm-runtime-20261008`. Native launch observation pending; startup AX timeout is not acceptance evidence.

## Closure

Frozen receipt accepted all4reviewed modes and bound10evidence dimensions per mode.74accounting checks pass; current numerator6/88(6.82%),80minimum/family/boundary/deferral gates open. Actual native private failure/repair/restart and corrected-checkbox guidance, MPS whole-sample training/capture, standalone package export and independent checkpoint parity, Light/Dark generated artwork and Information observed. Earlier pending checkpoints above are chronology, superseded by acceptance. Affected prior5activation reconciliation recorded in TORCH_ACTIVATION_LAYER_NORM_REPLAY_20261008.md.
