# Explicit objective engine integration · 2026-10-08

This is scoped host-runtime evidence, not component acceptance, Core-factory evidence, packaged execution or native UAT. The eight frozen objective identities train actual learned models from genuine sealed CSV preparation with typed target roles. Omitted objective arguments preserve the prior binary path; no objective is granted a tensor-forward graph adapter.

## Interfaces and preserved boundaries

`engine.run` adds optional `objective_recipe`, `target_mapping`, and `objective_scalarization='mean'`. Recipe and mapping must be supplied together. `validate` and `GraphModel` accept optional `objective_context={recipe,targetMapping}`. Product execution checks the actual selected float32 output against its declared objective geometry. Root's `load_prepared_objective` supplies `(train_x,train_roles,val_x,val_roles)`, provenance and sealed files. Triplet uses three real ordered source groups and a shared learned model, with no dummy target.

Every actual training batch, including the final short batch, is checked before model materialization/state changes for undefined objective mean and malformed target/weight/length data. Validation combines objective-specific contributions rather than averaging chunk means; all-ignored local chunks contribute zero, while globally undefined means refuse. Only class-index CE/NLL expose measured accuracy; other objectives omit it, and Plateau accuracy requests refuse before model construction.

The explicit-only checkpoint contract contains `objective={schemaVersion:1,recipe,targetMapping,scalarization}`. Registered configured buffers are stored as `objectiveState`; they are not trainable parameters. Parent recipe/settings/provenance/state must match, including weights-only loading. CPU exact resume restores objective identity/buffers; weights-only initializes the selected new objective and discloses both resets. Exact resume refuses changed reduction, weights or smoothing. Old checkpoints omit these additions.

Root's standalone `objective_recipe.py` generator is delivered separately from inference-only `model.py` and optimizer `training_recipe.py`, bounded to131072UTF8bytes. Exported actual native objectives replay validation results on CPU and MPS. No source-qualified control address is invented by the worker; qualified source capture remains the parent application boundary.

## Output and privacy schema

Final `completed.objective` and `manifest.evaluation.objective` include `rawPreview={shape,dtype,values,captureStatus}`. Shape is the actual native-selected raw reduction shape, including scalar `[]`. Values are at most the first eight actual flattened elements. `activationValues=false` preserves shape/dtype but withholds values with `values=[]` and `captureStatus='omittedByPolicy'`.

For `reduction='none'`, captured prediction rows also contain `objectiveOutput`: the actual raw scalar or sample array for that observation. This is independently governed by `predictionRows`; disabling it leaves no rows or objective arrays. Mean/sum rows omit this key. Existing finite JSON and 1MB prediction artifact/manifest storage limits remain enforced. No new artifact is introduced, and epoch/checkpoint history does not contain this final display preview.

## Actual TDD and checks

- Initial API RED:24 failures (`.local/loss-engine-initial-red.log`), then24 passes.
- Actual lifecycle/export RED:15 failures/34 passes, including missing export helpers and accepted foreign objective state, then49 passes.
- Actual capture RED:32 missing-raw-preview failures across eight objectives and all four independent privacy choices. First replay retained four fixture failures (Triplet test incorrectly selected the bundled dataset context); corrected to the actual prepared context.81 passes followed.
-48 additional CPU/MPS cases exercise sum, none/mean and none/sum; saved models and standalone objective helpers replay actual partition raw shape and effective scalar.
- Exact-resume configuration refusal, weights-only new configured-buffer initialization, unsupported Plateau accuracy refusal before `GraphModel`, and three fixed-buffer exact-resume cases complete the scoped engine matrix.

The original objective scalar/gradient and partition suites independently test numerical correctness; these integration checks deliberately verify transport, delivery, state and saved-helper continuity, rather than claiming their native replay as a second independent oracle. Tests use host `.venv` only. Actual Core factories, lesson dispatcher, main authorization, App verifier, native UI, full prior bound replay and acceptance accounting remain parent-owned separate gates.

Source and historical RED/GREEN pins: `.local/loss-engine-source-pins.json`. Final combined terminal and its log hash are appended after completion below.

Final command: `.venv/bin/python -m pytest runner/tests/test_loss_engine.py runner/tests/test_loss_partition.py runner/tests/test_loss_objectives.py runner/tests/test_product_execution.py runner/tests/test_checkpoint_lifecycle.py runner/tests/test_training_sample_capture.py runner/tests/test_optimizer_runtime.py runner/tests/test_scheduler_runtime.py -q`. Actual terminal **911 passed, zero failed/skipped,24 upstream warnings,16.57seconds**. This includes138 new engine cases,116 partition cases,275 objective cases and382 adjacent prior checks. Log `.local/loss-engine-final-adjacent-green.log`, SHA256 `a6969df6fd016966defa5542b9c8e6e1373f3e0610d86ed58a350059d21de707`. No full bound prior catalog replay is claimed.

## Additive pre-allocation prediction capture repair · 2026-10-09

The original 911-check receipt above is historical for its pinned engine bytes. A subsequently reviewed loss-only capture seam now admits requested prediction rows before building nested Python arrays. Shapes alone establish a minimum serialized JSON size before decoding or `Tensor.tolist`; a bounded scan of at most 4,096 numbers at a time then measures actual JSON numeric spellings. Requests exceeding the existing 1 MB artifact budget refuse with an actionable instruction to turn off prediction-row capture or reduce output shape. Rows are never silently truncated. This leaves the legacy binary branch and the independent eight-value rawPreview/privacy behavior intact.

Meaningful bounded RED: six new tests failed because the helper did not exist (`.local/loss-capture-budget-red.log`). Final helper suite: ten passed, covering output-geometry refusal before any conversion, exact byte fit and one-byte overflow with optional unreduced objective output and decoded classes, nested tensor geometry, positive/negative class directions and chunk-bounded numeric scanning. Final actual host command:

```
.venv/bin/python -m pytest runner/tests/test_loss_capture_budget.py runner/tests/test_loss_engine.py runner/tests/test_model_training_operation.py -q
```

Actual terminal result: **169 passed, zero failures/skips, five upstream warnings in 11.63 seconds**. This includes the existing 138 actual loss-engine cases (all eight CPU/MPS paths and all four independent capture-permission combinations) and 21 legacy model-training-operation cases. It does not rerun or substitute for the root's sixteen genuine Core-factory canonical runs or native UAT.

Current source/log SHA256 bindings:

- `runner/engine.py`: `3e377577e4d81a27d5d7dbed4b280dc27f3450782c9159234f1d2844fa231179`
- `runner/tests/test_loss_capture_budget.py`: `9bb4592b3e0a2ecdba36aa9806e6930c9f0c9e93af524e100e742679732bec18`
- `.local/loss-capture-budget-red.log`: `8c7d2827ff7aa18261747a3a17bb2a672731b692e07f765b30a5fd419db66818`
- `.local/loss-capture-budget-final-green.log`: `2f4d61c97214bae6a0ce36067be8dcbb3a8fe5008ddc7e32aacbdf4a4b362540`
- `.local/loss-capture-budget-scoped-final-green.log`: `d8dd47cab44ba8587aa42ebbe642a9e318418fda46051e1434fb48910d772de0`
