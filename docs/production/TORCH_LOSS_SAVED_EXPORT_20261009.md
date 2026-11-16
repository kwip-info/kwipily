# Explicit objective saved-model export — scoped evidence

This slice adds portable saved exports for the eight explicit objectives. It does not grant bundle coverage, native signoff, complete release readiness, or MPS inference support. The standalone inference helper deliberately runs CPU predictions; tests include both CPU-trained and MPS-trained checkpoints.

## Contract

Legacy implicit exports retain input-contract schema 1, the exact two-logit output, CLI logits/classIndices, and existing typed integer-input support. Explicit objective exports use schema 2: captured prepared float32 input shape/ordered feature names; actual float32 modelOutput shape; canonical objective recipe, qualified control source, typed target-role mapping and mean/sum scalarization. Shapes derive from the saved graph and saved registry under explicit-objective validation, never an open editor or a fabricated two-class output.

The independently hash-pinned objective_recipe.py is bounded UTF-8 (131072 bytes). Inference verifies but never imports or executes it. It is a separate fresh training-objective helper, with configured fixed buffers, native reduction-selected raw score and separately scalarized training score. Those scores do not replace the model prediction. No source rows, source bindings, absolute paths, preparation tables or row membership are copied. Exact checkpoint bytes remain intact, including existing model/optimizer/RNG/objective state and provenance. Fresh objective construction is not checkpoint restoration or training continuation.

Input/output ranks, dimensions and total elements are bounded; N times sample size cannot exceed 100M before model execution or tensor creation from JSON/synthetic examples. Package inventory, regular-file/no-symlink checks, hashes, staged-storage reservation and atomic exclusive publication remain enforced.

## Actual RED and setup failures

- `.local/loss-saved-export-red.log`: initial new Swift test setup compile failure (missing `try`); no behavioral RED claim.
- `.local/loss-saved-export-semantic-red.log`: two actual App tests executed, two unexpected thrown errors at the pre-existing explicit-objective export guard. Actual prepared/trained runs succeeded before rejection; this is the meaningful saved-export RED.
- `.local/loss-portable-inference-red.log`: 32 new Python cases failed. The 16 positive CPU/MPS model cases and the separate none/scalarization case reached real training then rejected the schema-2 objective package; the budget test exposed absent early admission. Fourteen other failures were the cache-directory resealer fixture problem described below, not additional production RED claims.
- `.local/loss-portable-inference-green-candidate.log`: 42 passed /14 failed. Fourteen failures were the new test resealer attempting to include Python's import-created cache directory. The fixture now reseals files only; no inventory requirement was relaxed.
- `.local/loss-saved-export-green-candidate.log`: seven methods executed, eight ordinary assertion failures only. Canonical JSON decoding represents exact whole numeric values as integer enum cases, so raw JSONValue enum equality misrepresented equivalent recipe controls. Tests now compare strict LossConfiguration.decodeRecipe equality; source, typed mapping, all settings and versions remain required.

## Actual final verification

- `.local/loss-saved-export-green.log`: **7 App methods passed, zero failures/unexpected/skips**, 55.512 seconds, terminal exit 0. Two new methods cover 17 genuine supervised runs (eight objectives on CPU and MPS plus one MSE tamper/storage run), 16 standalone isolated Python direct-output replays, cold History after host retirement, actual publication/checkpoint identity/privacy, original artifact tamper, and storage/atomic publication refusal. The unchanged legacy export three methods and typed-input two methods also pass.

Python final verified separately:

- `.local/loss-portable-inference-green-v2.log`: **56 passed, zero failures/skips**, 6.84 seconds. New objective cases 32 plus unchanged legacy cases 24. Four native KL mean-reduction warnings are retained in the log.
- All eight objectives train on CPU and MPS; saved model/checkpoint are copied into standalone packages and original source/run directories removed. CPU inference matches direct saved model output for batch size 1 and full heldout batch. Triplet uses six inputs and [N,3,2] output; CTC retains [N,3,3]. Fresh objective helper raw/scalar evaluation is tested separately; `none` plus sum scalarization preserves distinct arrays and scalar.
- Optional optimizer/objective helper files are replaced with deliberate import failures, resealed and verified without execution by inference. CLI returns generic modelOutput without invented classes. Negative controls cover typed contract changes, wrong source domain/foreign target mapping, unknown fields, geometry/budgets, optional helper hash tamper/symlink and unknown inventory paths. Old inference loading/batching/RNG and typed-input tests remain in the combined replay.

No engine or objective numeric implementation changed in this slice. Root owns native export checks and final acceptance.

## Final evidence identities

| Path | SHA-256 |
|---|---|
| `Sources/KwipMLApp/WorkflowHistoryReader.swift` | `c49e61a8a9ccc1ee181e02d427e85c136d9eae1459ae51e8d546b8537135e164` |
| `Sources/KwipMLApp/TrainedModelInputContract.swift` | `8de209aba4d799cf6a5953c744375c637cc5169324aabd44c2ed56cecfd7dc72` |
| `Sources/KwipMLApp/Resources/inference.py` | `a72a16709d12ff02257c554730a7ba44804b51c23ebc1e4f38f8a8c66b67ee42` |
| `Tests/KwipMLAppTests/ObjectiveSavedExportTests.swift` | `45e6b5bda8293767f85d39d7f33a1cbaec88fd466e8eb71073a91df415a9d05a` |
| `runner/tests/test_objective_portable_inference.py` | `24dba0dbbccf126d00f2db87fe0a3f6d89109ac113a3454885b047b20e72cb23` |
| `.local/loss-saved-export-semantic-red.log` | `a34fec6ae2200b3db08b4bc152a5657c75b773a7cb66e98ce54b69c74c3bf886` |
| `.local/loss-saved-export-green.log` | `fb25ac2bbaf8d46ba824e2007949ca772d5ca28bd46e425059d81663083275a0` |
| `.local/loss-portable-inference-red.log` | `987937213056383d3337ca6d6c760cb71ea29487960dfa1f521c52a80d1894f8` |
| `.local/loss-portable-inference-green-v2.log` | `ec8eef697217edb7e49406b1e88bb64c3c6258d2cc174ad36fc1af347eca3fed` |

These are final source/log pins, not a claim of a captured global before/after manifest. Setup failures and intermediate fixture corrections remain separate above.

## Additional independent fresh-step verification

`runner/tests/test_objective_portable_step.py`:16 cases passed, zero skips (1.93s), covering all8 objectives × CPU/MPS. Standalone packages retain no original private source/run. Direct exported models reproduce full/microbatch outputs on the actual device; an independent functional oracle checks raw/scalar scores and input/all-parameter gradients. Fresh SGD (momentum0.5, decay0.01) and StepLR (gamma0.5) match independent first-update/buffer/rate expectations. CTC uses the explicit differentiable CPU scoring transfer on MPS. Existing checkpoint bytes stay unchanged. This is fresh training, not continuation; the convenience inference helper remains CPU-only. Six native KL warnings remain visible. No production change or artificial RED was needed for this added verification.

Test SHA256: b27168908646f974d601dcc8636e06de0e1840e3206e40d47194ee9fbb1902ac. Log `.local/objective-portable-step-final.log` SHA256:3240e8aadee948531fce4c2cc9e43ac619e5ef85eda5574d8474e5868acb0898.
