# Five scheduler bundle acceptance · 2026-10-08

Frozen finite acceptance for StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR and LinearLR. Each retains configured-schedule, step-timing, state-roundtrip, resume and history-rate-trace; Plateau also retains required-metric. All reviewed modes meet the bounded dense finite float32 CPU/MPS contract. Atomic accounting remains a separate step. Recipes are configurable training controls connected to Train, rather than tensor-forward modules.

## Delivered contract

Workflow schema7/compiler6 resolves exact canonical configuration and qualified optimizer/scheduler sources, requires an explicit connected optimizer and ignores disconnected recipes. Shared inspector/palette/port templates preserve draft ownership, Apply/Discard, inactive-control retention, runtime discovery/revocation and old optional-field omission. Typed controls admit both OneCycle curves/two or three phases, momentum cycling compatibility, fixed user horizon, decreasing Cosine half-cycle, and explicit Plateau metric/mode/threshold/patience/cooldown/floor/epsilon.

Actual optimizer updates determine batch versus completed-epoch ticks. Constructor initialization is recorded separately; used and next rates remain distinct. OneCycle terminal unused next values are retained honestly and never used for another update. One bounded10000-record/4MB hash-pinned rate trace supplies real qualified Activity events, checkpoint restoration, verified History and portable export. Strict optimizer/scheduler state ownership and independently recomputed curves/metric transitions reject repinned but scientifically invalid evidence. CPU exact epoch-boundary resume restores real state without replaying constructor effects. MPS supports fresh weights-only fine-tuning and refuses exact resume. Portable inference validates optional trace bytes without executing the training recipe; fresh training helpers document actual tick/metric/horizon usage and remain under8192bytes.

Five original generated Training-family PNGs with exact prompts/source hashes accompany five real prepared examples and assessed no-code lessons. Lessons define groups, passes, adjustment size, remembered change and validation results plainly; independent scalar references grade actual updates/rates/timing/continuation. Private assessment and cold restoration withhold reference answers. No claimed blanket expertise follows a passed fixture.

## Actual verification

Authoring1211 checks pass; scoped runtime382, lesson-worker59, train-only authority21 and actual CLI5 pass. Portable optional-trace/adjacent inference29 pass. Counts overlap across suite scopes and are not added as unique tests. First broad regression exposed stale metadata fixtures and a real Swift/worker legacy Concat/Add semantic-identity mismatch. Existing pinned hashes were preserved; defaults-versus-behavior independent test first failed twice, then correct default-only normalization passed. Final full Core replay reports953 tests, five explicit skips and zero failures in30.680s. Final22archive/History/artifact/portable-export methods pass in12.231s. Final shared App replay reports30 methods, one opt-in benchmark skip and zero failures in120.227s, including actual optimizer/scheduler guided/practice/private/cold workers, typed inspectors/discovery and execution sessions. Prior runner replay executes71 bound test files for accepted54:5049pass, no failures/errors/skips in155.68s;566 source/test/art and62 prior document snapshots remain byte-exact. Existing Swift Sendable and upstream warnings remain recorded, without claiming Swift6/distribution readiness.

All five Core-emitted canonical examples execute on CPU and MPS:10cases. CPU exact resumes reproduce model/optimizer/scheduler/full trace/update/RNG/metrics; all five MPS exact requests refuse and actual fine-tunes reset scheduler/trace. Every model exports and infers on both devices, with independent scalar references, full/microbatch/CLI and changed-trace refusal. Actual fresh exported recipes run full schedules including Plateau's patience-triggered reduction and OneCycle's complete horizon. Maximum scalar inference error1.54e-7, update error1.33e-7, persisted rate error3.47e-18; helper sizes3805–4200bytes. Source-qualified fixtures and detailed pins are in TORCH_SCHEDULERS_CANONICAL_20261008.md.

## Native and actual saved package

Signed sandbox candidate verifies129 current worker files before/after signing. OneCycle lesson projectE273658D-867B-4548-9C9C-920A930975FD passes actual guided and private ten-criterion construction checks with both typed recipe wires, exact nondefault controls and framed rate/update/resume tables. Private tables are withheld; cold launch independently verifies saved assessment source/submission/evidence.

Genuine OneCycle canvas project2805A0A3-41A6-4B8C-8CAD-B35F30EE3004 completes three epochs on MPS and retains six actual source-qualified scheduler Activity events. RunBB7E23F3-C6AB-4DFB-89B2-3D85C766D4BB, archiveF89F626A-895F-43F3-BB22-2D0A851C28CA, revision0, request2e52398281bb84e08b9e05fa7f6f1eed5e5350aa7538d785e26031222939925f. History independently verifies33artifacts before and after cold launch. Actual Export→Choose destination→Save package produces .local/KwipML-model-BB7E23F3, including exact pinned rate trace and fresh training helper. No generated fixture substitutes for the native publication.

Independent anchored verification of that unchanged saved package validates all33archive pins and eight portable files, qualified control source, actual48/16train/validation rows, six learned parameter update counts, scheduler/optimizer state, independent six-tick curve/momentum and actual CPU/MPS inference/microbatch/CLI classes. Scalar error1.54e-7, microbatch1.79e-7 and full ten-step fresh helper update1.33e-7. Exact numerical bounds and identities are in TORCH_SCHEDULERS_NATIVE_EXPORT_20261008.md. A proof-script keyword mistake was preserved and corrected; no product behavior was changed by that correction.

## Limits and closure

This closes five declared bundles; no denominator/mode deletion or framework-wide/API completeness claim. Multiple optimizer groups, arbitrary scheduler chaining/callables, authorable last_epoch, sparse/differentiable/fused optimizer options, exact MPS continuation, schedules exceeding bounded trace/horizon and Cosine oscillation beyond its declared half-cycle are outside this reviewed contract. The rapid native run finished before observation, so retained real Activity events do not establish observed in-flight animation. Physical gestures, themes/accessibility/novice/hardware and distribution stay in their existing release tasks. No external publication or App Store submission occurred. Prior frozen receipts are retained rather than rewritten; SHA-only refreshed shared sources require the actual new regressions above.

Scoped evidence: TORCH_SCHEDULERS_CLOSURE, AUTHORING, RUNTIME, LEARNING, ARCHIVE, HISTORY_EXPORT, ROOT_INTEGRATION, PRIOR_REPLAY, CANONICAL, NATIVE and NATIVE_EXPORT_20261008.md; original icon provenance is docs/design/component-icons-v2/scheduler-provenance-20261008.json.

## Frozen exact local evidence

- `.local/scheduler-core-final-regression.log` SHA-256 `90324639e296f740ec349e312befbb11cb8485d4cc6a189ce3e38f4662b2d7c5`.
- `.local/scheduler-app-final-regression.log` SHA-256 `3952c75e3463e0c6955dfe2644af2b7874a5782b6fa479e1b84e966768dd9a7c`.
- `.local/scheduler-packaged-build.log` SHA-256 `237e1a94c042046b6957db1a459aed5177726e3db3500cabbf78eb0a4e63da85`.
- `.local/scheduler-prior-runner.log` SHA-256 `85870cc8efe24fee7f66d1cc0ffc01e4b499756a9feed6d4a225f5b5e504ee4c`.
- `.local/scheduler-native-export-20261008/verification.json` SHA-256 `8269109fe8337d3b0b721208bc6d2c6f3391c51b0d90ef73bcdbd0e8b46ac06b`.
- `.local/scheduler-canonical-host-20261008/receipt.json` SHA-256 `38595a0707be5b66431c2831063e2e7d175a4502071d43b88c733528c9da8174`.
