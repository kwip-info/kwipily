# Gradient/trainability standalone helpers · 2026-10-09

Scoped foundation evidence only against merged contract f4af1ac. No engine, checkpoint, CLI, authorization, generated registry, Core/App, native build, acceptance catalog or old receipt was changed. The two adopted units and six modes are not credited by this helper slice. Shared training/continuation/export/observer/packaged consumer integration remains a separate explicit lease.

## Actual RED and GREEN

Command (host .venv; installed Torch2.14.1; actual CPU and MPS):

```sh
.venv/bin/python -m pytest -q runner/tests/test_gradient_policy.py runner/tests/test_trainability.py runner/tests/test_requested_gradients.py
```

Initial importable minimal baselines ran real native sum autograd (12 per input instead of requested mean6), preserved inherited flags instead of freezing, and ignored nondefault/invalid controls. Terminal RED: **11 failed in0.60s**, no missing import or compiler/setup error. `.local/gradient-helpers-red.log` SHA256 `609ccfdb9250a4d046465c1ed452bf46815e39a078979f15b6559be344a3b5af`. The initial numeric assertion subsequently adopted the frozen result envelope's ordered targets array; its independently expected6 remained unchanged.

First narrow GREEN:11 tests. Expanded terminal found one test-only assertion bug: a substring search for rowIDs also matched the required rowIDsSHA256 key (141 passed/1 failed). Corrected it to check absence of the exact raw rowIDs key and output values, preserving the required hash. The expanded production implementation had no failure at that stage. Final scoped terminal: **147 passed, zero skipped, three upstream warnings in1.56s**. `.local/gradient-helpers-final-green.log` SHA256 `60cfeafee7d7aac8fc7dd15e40336a2a63ecd2e726805963fe3a32f14239af49`. Two warnings disclose the existing native KL mean convention; one concerns Torch sparse fixture construction. No numerical expectations were loosened.

## Implemented finite interfaces

- gradient_policy: canonical_configuration(values), validate_recipe(recipe), validate_request(request), validate_capture(value=None), validate_compatibility(policy,objective_recipe=None,scalarization='mean'), window_denominator(objective_recipe,roles,prediction_shape,scalarization='mean'), scaled_contribution(part,denominator,scale), capability(). Configuration fills only top-level defaults; recipe decode requires every canonical field. Exact integer controls, checked Int64 range sum, IDs/selector objects, unique request/target bounds, finite representable float32 upstream arrays, exact upstream row shape, canonical32768 UTF8 budget and strict independent permission are enforced. Frozen dedicated descriptor is literal, not a forward adapter.
- trainability: canonical_configuration(values), validate_recipe(recipe), native_owners(model), resolve_owners(model,recipe), apply_trainability(model,recipe), capability(). Actual materialized layer_keys/native named_parameters resolution, whole-model200000 parameters, finite dense float32 CPU/MPS, foreign/buffer/lazy/alias refusal. Validate complete selection before changing flags; preserve registered object/value/buffer identity. Complete returned inventory retains actual native parameter order; rule array canonical order is separate. All-frozen inference remains valid; actual Train empty-ownership refusal belongs to subsequent engine integration.
- requested_gradients: analyze(model,inputs,request,*,device,recording=True,check_cancelled=...), and analyze_requested_gradients(model,prepared_inputs,policy,*,partition_row_ids,source_identity,device,check_cancelled=...,derivative_capture=None). Qualified prepared_inputs/partition_row_ids dictionaries contain actual training/validation tensors and aligned row IDs. source_identity exact keys are runID, graphSHA256, checkpointSHA256, preparationManifestSHA256, preparationGraphDigest. The caller's future engine/archive layer must supply verified saved-source identities; this helper does not independently read archives or pretend caller-supplied hash strings authenticate files. Explicit recording=False is the separate genuine no-grad assay; normal final analysis always records.

Requested output is selected through actual inspect=True graph values and native named output routing. Input targets must be the genuine input endpoint, not an intermediate. Native autograd.grad computes first-order input/native parameter influence; target order preserved. Integer/frozen/disconnected/recordingDisabled/sparse remain structured unavailable causes without values; genuine zero is observed with actual zeros. Native unsupported backward is always a hard refusal, including reportUnavailable; no fallback. Target appearances budget is preflighted across all requests before any forward; full upstream config budget is separate. Conservative encoded artifact estimate precedes execution; exact8MiB encoding check follows. Derivative permission false with requests refuses before execution; empty requests returns no artifact.

Analysis snapshots/restores actual module modes, CPU/MPS RNG, learned parameter/buffer values and original .grad object references/contents in finally, including cancellation/failure and existing sparse .grad. Its artifact omits actual input/output/target rows and activations. No analysis is silently executed during inference load. Export helper generation is not implemented in this standalone slice.

## Actual independent checks

Actual CPU/MPS scalar chain rule, signed VJP, parameter targets and centered finite-difference comparison; actual production GraphModel's parallel affine/ReLU/tanh composition against independently calculated derivatives. Freeze permits earlier parameter/input derivatives; explicit unfreeze preserves values. Correct measured zero versus each actual unavailable category. Real sparse Embedding gradient reported without densification; integer input remains integer. Deliberate native backward NotImplemented refuses without fallback. Exact mode/RNG/parameter/.grad isolation is measured after success, missing output and cancellation.

Window helpers reuse production loss_partition read-only, not a second trainer.64 cases cover all eight objectives × mean/sum/none+mean/none+sum × CPU/MPS. Scaled numerator backward matches whole native selected objective gradients and existing independent plain-Python scalar/path enumeration expectations. Additional cases cover exact float32-rounded class-weight denominator, local all-ignored chunk within a valid complete window versus globally undefined pre-forward refusal, probability CE, sample-broadcast MSE/L1, zero-weight BCE, KL batchmean and CTC none-mean distinction. One hand-calculated unequal2+1-row affine MSE window verifies actual accumulated .grad, analytic SGD update and reset before the next short window. This is helper arithmetic evidence, not an engine scheduler/continuation claim. Frozen BatchNorm parameters still update train-mode running statistics; eval stops those changes and disables dropout without changing frozen flags.

No capability discovery installation, full window training hook, optimizer ownership integration, CPU full-run resume, MPS fine-tune, portable source generation, public artifact writer/reader or native UI test is claimed here. Those remain the exact merged contract's subsequent gates.

## Frozen source/test SHA256

| Path | SHA256 |
|---|---|
| runner/gradient_policy.py | 2f64a406639f62352df57f79078e72e9a0f7c3373b3420da3557ac4b2fb4a441 |
| runner/trainability.py | 42eeadbf186c57eb3005dcf19c0b218d351e9d17b2bc6a275dc183d0b1482b04 |
| runner/requested_gradients.py | 9aaef3dbd17166c9ba0b70a63eb54efdcdda4455930f2c2009a3920015b58d02 |
| runner/tests/test_gradient_policy.py | 0616ce0fd8c3fefb177e8b0ceeea6ce0b705f912a513e77709ca3990f86288e4 |
| runner/tests/test_trainability.py | bc1714e4982337fcf3ed2d0864be15ff7ba05c5d4f77b611061958318d889147 |
| runner/tests/test_requested_gradients.py | 5c7a76b42283486058c90ee801bb1c8d0a91d31fb04a92d3c267cafa276b6fc9 |
