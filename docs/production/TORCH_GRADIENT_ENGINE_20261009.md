# Gradient policy and trainability runtime integration — 2026-10-09

Scoped runtime evidence only; no component acceptance, App, native UI, packaging or full prior-bound replay claim.

## Actual tests

The importable legacy loop produced eight semantic failures (actual CPU/MPS accumulated-update and frozen-weight numerical mismatches, silently accepted incompatible reduction/permission/all-frozen training, and absent discovery). After integration these controls pass. Transport semantic RED was 11 failed/3 passed; a preceding run included six fixture-import setup errors and is retained separately. Portable objective arithmetic initially executed deliberately incorrect row-denominator/prediction-sum behavior: all16 objective/device cases failed numerically before repair. The earlier unsupported-keyword run was setup failure, not semantic RED.

Final integrated command selected test_gradient_policy, test_trainability, test_requested_gradients, test_gradient_engine, test_gradient_checkpoint, test_gradient_export and test_gradient_transport: **273 passed, zero skips, five upstream warnings, 19.86s**. After expanding only portable tests to mean/sum/none+mean/none+sum across eight objectives and CPU/MPS, the final portable file separately passed **72 tests, zero skips, two upstream warnings, 2.79s**. Its first expanded run failed from a test builder keyword typo; corrected to its existing scale argument without production change. Counts overlap and are not added as independent executions.

Adjacent actual prior replay: checkpoint_lifecycle, checkpoint_lifecycle_modes, initialization_engine, optimizer_runtime, scheduler_runtime, gradient_capture, failed_gradient_capture, capture_policy, loss_engine and one direct accepted-baseline comparison: **593 passed, zero skips, five upstream warnings,18.20s**. The baseline comparison imports engine.py from accepted ee26ff9 and runs the same genuine prepared MSE/SGD workflow: checkpoint recursive values, metrics, model/objective/optimizer helper bytes and selected manifest settings/runtime/provenance/evaluation/lineage remain equal when new policies are omitted. This is a finite adjacent regression, not the later full catalog replay.

## Runtime boundary

New run arguments are keyword-only: gradient_policy, trainability, derivative_capture, gradient_observer. The trusted observer defaults absent; detached bounded lists at windowStart/beforeStep/afterStep record actual sequential prepared-row indices, native ownership, flags, .grad presence and optimizer state inventory. It is not a user wire or alternate trainer.

With selected policy M=ceil(rows/batch), U=ceil(M/K); each window admits original typed targets before forward, resets all model gradients, accumulates true objective numerator divided by the full-window denominator, steps once including the short final window, and clears gradients. Sum uses denominator1. OneCycle ticks actual updates; epoch schedulers remain epoch-timed. Sparse or nonfinite gradients refuse. All eight objectives retain native reduction/scalarization semantics. Frozen parameters remain in the model; optimizer ownership contains only trainable parameters. Fine-tune validates original parent optimizer ownership before constructing the child subset; exact CPU resume binds complete policy/flags/schedule and saved optimizer/RNG/metrics/objective state. MPS remains weights-only. Checkpoints contain no pending accumulated gradients.

## Exact additive metadata

* checkpoint.contract.gradientPolicy = {schemaVersion:1, recipe, schedule:{microbatchesPerEpoch,optimizerUpdatesPerEpoch,accumulationSteps,gradientScale}, checkpointBoundary:'completedEpochNoPendingGradients'}.
* provenance.gradientPolicy = {schemaVersion:1,recipe,implementation:'kwip.runner.gradient_policy',torchVersion,schedule}; settings.gradientPolicyRecipe plus microbatchesPerEpoch/optimizerUpdatesPerEpoch.
* runtime.updateCadence = {schemaVersion:1,unit:'optimizerUpdate',microbatchesPerEpoch,optimizerUpdatesPerEpoch,accumulationSteps}.
* checkpoint payload updateSchedule = {schemaVersion:1,completedEpochs,completedMicrobatches,completedOptimizerUpdates,pendingMicrobatches:0}.
* checkpoint.contract.trainability = {schemaVersion:1,recipe,parameterOwnership,optimizerParameterNames}; provenance same plus implementation:'kwip.runner.trainability'/torchVersion; settings.trainabilityRecipe. Ownership is ordered actual native {node,parameter,name,shape,dtype,requiresGrad} metadata.
* manifest.derivativeCapture exact schema1/values Bool; nonempty requests without opt-in refuse. derivativeResults = {schemaVersion:1,path:'derivative-results.json',bytes,sha256,requests,elements,phase:'finalSnapshotEvaluation'}.
* derivative artifact = {schemaVersion:1,runID,phase:'finalSnapshotEvaluation',device,dtype:'float32',torchVersion,recipe,source:{graphSHA256,checkpointSHA256,preparationManifestSHA256,preparationGraphDigest},requests}. Each result retains normalized request identity, hashed row identities, output metadata and actual target statuses/values; no output sample arrays or preparation rows. Values permission does not broaden predictionRows/activationValues.

Artifact bounds: gradient_policy.py131072 UTF8 bytes, trainability_recipe.py131072, derivative-results.json8388608. Existing training_recipe.py8192 and total training budget remain unchanged. Selected portable model supports differentiable inspect and native flags. Helpers are Torch/stdlib-only; policy source carries a hash of the full policy but omits request/upstream arrays. It exposes objective-owned window_denominator and objective_contribution, begin/backward/finish functions and explicit native analysis. No helper autoexecutes training on inference load; no app imports. Objective source namespace isolates exact objective admission/functions from policy validators.

Discovery adds gradientPolicyCapabilities.gradient_policy and trainabilityCapabilities.trainability; existing componentCapabilities are not repurposed. CLI training-only flags --gradient-policy-json/--trainability-json/--derivative-capture-json are checked in main and both authority paths. Legacy omitted-policy path preserves generated source/checkpoint behavior.

## Source and log pins

```json
{
  "sources": {
    "runner/engine.py": "f33e822503bbe32a976ededceddfce4aac51c556801aa417c91e943d1a62a084",
    "runner/checkpoint_lifecycle.py": "092835fb7d9be5a2337c4cd32380ea5ed2424dd986a3022edf1ec7400fb0a793",
    "runner/main.py": "caf67822315b7ba07190449d18454d77cf7bda79ce3d0a1407138c8595171567",
    "runner/model_training_operation_contract.py": "9706238700b47f55b85834914fab6e7c4007dd48c53357834bfbefce0beca47a",
    "runner/worker_authorization.py": "ca5846ff69fec1b4a375c2fe01b108fe7c82d35211f82c1bda7498a34943ba02",
    "runner/gradient_portable.py": "929b6e6099cb4a646576394c10e111a3e22c9160fedce74f4f1dfc94b4ac6526",
    "runner/gradient_policy.py": "2f64a406639f62352df57f79078e72e9a0f7c3373b3420da3557ac4b2fb4a441",
    "runner/trainability.py": "42eeadbf186c57eb3005dcf19c0b218d351e9d17b2bc6a275dc183d0b1482b04",
    "runner/requested_gradients.py": "9aaef3dbd17166c9ba0b70a63eb54efdcdda4455930f2c2009a3920015b58d02",
    "runner/tests/test_gradient_engine.py": "3e78ec5b834298e46d96d696967edff976868663f6d6742afa9c6817ea607134",
    "runner/tests/test_gradient_checkpoint.py": "1e93bc2974c400f2be1c832e77823688dfe916820888a5b30c424e2f990c6a04",
    "runner/tests/test_gradient_export.py": "c2290d393ac749af0a83557a242b6d0a7ff2a71e6f99f8fb5ea72024209632ac",
    "runner/tests/test_gradient_transport.py": "24746865408cfaa8902270e5dbbd1332f83628ff0ae20a854fd78160dad7f2ab"
  },
  "logs": {
    "gradient-engine-red.log": "a28cf57ba737010f86cd7e2ce93ee6cd2d555fadc5479e1213bbd9bff7b8223e",
    "gradient-transport-semantic-red.log": "1c4fcbcf0a7cecd80691931800e47fbe837dffcbae5ee04bda4142b5380e0350",
    "gradient-portable-objective-semantic-red.log": "7dd1184e6882c38fda0b82d5876a4851901c8c7e1729b745d917ae40fbecef02",
    "gradient-runtime-final-green.log": "0c2885dd2ad4238dd9ecfddb510ed9063f0f19cd7bdb148e15922c401cffe079",
    "gradient-adjacent-prior.log": "dc1772fb20a9aa239ad73d7c768ad1cd1022290b1f4151544849392cb6c50a1e",
    "gradient-portable-objective-final-corrected-green.log": "6c7d7094ec2dd36aaa527a6e51b855a34c529a297eaf25c9d4513509d15405f4"
  }
}
```
