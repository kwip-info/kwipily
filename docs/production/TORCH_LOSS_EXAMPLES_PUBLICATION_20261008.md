# Loss bundled examples and registry publication · 2026-10-08

Owner: Trevor Ewert. Implementation: Codex. Status: example/runtime and generated publication prerequisites verified; complete App/native bundle acceptance remains open. No change to the accepted70/88 numerator.

## Actual examples

Eight public factory wrappers share one typed universal-canvas builder. Each has an actual64-observation bundled CSV, explicit supported float64/int64 table casts, held-out split, typed target-role mapping, prediction model, scoring recipe, run setup and Train connection. Model inputs/float target roles use the verified float32 conversion at the prepared-objective boundary. No dummy target column is introduced. Triplet reshapes six inputs to three two-value groups before one shared Linear layer. Sequence output has three time positions and three classes; visible LogSoftmax precedes NLL/KL/CTC scoring. The examples use the interesting configurations authored with their matching assessed lessons.

## Verification

The first actual scoped RED terminated1: three methods, nine failures (one unexpected thrown missing-example error), proving all eight factories were absent. After implementation, the first candidate terminated1 with eight semantic unsupported-cast-choice failures, zero unexpected: the data registry does not support float32 casts. Corrected to its supported float64 casts while retaining explicit float32 model/role conversion. Final three example methods passed, zero failures/skips. They validate all eight selected objective plans, schema8/compiler7, no fabricated target, exact lesson configuration, workflow round trips, shared-triplet geometry and invalid identity/device refusal.

A second genuine Swift run exported the compiled model, preparation, canonical recipe and mapping snapshots through an explicitly supplied local evidence directory. The actual Python preparation/engine consumed these compiled snapshots and their bundled CSVs: **16 runs, all eight objectives on CPU and MPS, three epochs, batches32, seed42, learning rate0.03, exact16 held-out observations**. All history losses were finite; accuracy existed only for index CrossEntropy/NLL. Canonical objective/mapping persisted and actual objective_recipe.py was produced. This is genuine host runtime evidence, not native App or UI-saved export acceptance.

Actual learning metadata references these real resources/factories and the eight assessed tasks, dated/versioned. Generator regeneration and --check passed. Python loss authoring suite34PASS; publication regression12Core+4App methods passed (examples3/compiler5/shared learning4/canvas4), zero failures/skips. The declaration test class is LossDeclarationTests; it is intentionally not counted in this selector and is replayed with runtime admission separately. One local metadata-script import-path setup error occurred before any metadata mutation; reran with the repository PYTHONPATH.

## Remaining acceptance

Runtime discovery/admission; production canvas recipe/mapping editing; native phase transport; archives/History/model export; actual native learning/private/cold and training/save/replay checks; reviewed catalog/art evidence accounting. Generated registry publication is a source availability prerequisite, not complete bundle acceptance. No distribution or broad framework claim.

## Source and evidence identities

- `Sources/KwipMLCore/WorkflowLossExamples.swift` SHA256 `3234f7ec48f2a883300a4df6839bfe77201bf5cacd3d54995e5c3c1860451e80`.
- `Tests/KwipMLCoreTests/LossWorkflowExampleTests.swift` SHA256 `d0df2d8ed2260fe3c14e4bffcb6d9479e5d05e3c8d12acc45ae14cd6bc82167b`.
- `components/losses/bundle.json` SHA256 `9da851b7fe29e94bd22f3f2e392bdeddf69b35328da3da64d46235fe500f3604`.
- `Sources/KwipMLCore/BundledLossComponents.generated.swift` SHA256 `3bd019dc5d898a19f3ef725a99fac13ef0d598b5dcfe6e0cb1fbfd822bab93a7`.
- `runner/loss_adapters_generated.py` SHA256 `cad0b2a0d19c0eeab9d94bc4063b5444c86908af7b3ce3ff29034b7498aad587`.
- `Sources/KwipMLApp/Resources/loss-bce-observations.csv` SHA256 `6049318d719ed5cbfe56fff25723714930f87147c3fda2649649c99885a442c9`.
- `Sources/KwipMLApp/Resources/loss-cross-entropy-observations.csv` SHA256 `3b77a4d2f0933c209c05f5225e41eb28603b7525d7e0a66070b8a6bb9ac578b8`.
- `Sources/KwipMLApp/Resources/loss-ctc-observations.csv` SHA256 `67fccf379bec4a9955686a09beba4b571ddb55d13b1be473d747ca0064931f71`.
- `Sources/KwipMLApp/Resources/loss-kl-observations.csv` SHA256 `335264e190dda1452cd223ecd38eac22eaf90f5c1f413198fe7df8fbc0507675`.
- `Sources/KwipMLApp/Resources/loss-l1-observations.csv` SHA256 `f97e9d8990016104af1a4c38b661679213d3856b8329faa6af76fe414816cf53`.
- `Sources/KwipMLApp/Resources/loss-mse-observations.csv` SHA256 `82968e3096b718ea691700749b85b4422494bbf9c6e88532f009a9484d642a47`.
- `Sources/KwipMLApp/Resources/loss-nll-observations.csv` SHA256 `3b77a4d2f0933c209c05f5225e41eb28603b7525d7e0a66070b8a6bb9ac578b8`.
- `Sources/KwipMLApp/Resources/loss-triplet-observations.csv` SHA256 `4039256d9e739b1887730a4df2a7f8add79d55594577bdf24b429354090cd34c`.
- `.local/loss-examples-red.log` SHA256 `b58f29648a7dce02929a4b25c5b84ebd19afc69eb87ffc4311264b20301e7981`.
- `.local/loss-examples-first-green.log` SHA256 `3e319de294f8bfad775d7575ce03a1008cf548072067dc714e9c6509ce6a2eb8`.
- `.local/loss-examples-green.log` SHA256 `46e19e670b94d2499973a70d987b072bd864f3d02f37e2bf37eea76dd7a53aa0`.
- `.local/loss-examples-snapshots-green.log` SHA256 `562b991a5f8a2003c59414e4e580b98117b5b1ee98fbeacdca10e8b24cb0e843`.
- `.local/loss-example-runs-green.log` SHA256 `9be7a35d6379dc8b034487c2122ee4ea60fa957653a3cfa4799f59750736afcf`.
- `.local/loss-example-runs-verified.json` SHA256 `568dfea8e0a0d35f4e02fbe2b60b11feffbf413d2e32df8a953249467ea7a718`.
- `.local/loss-publication-generator-green.log` SHA256 `3a2dc5b6ebd1855b7a929d27021d2670ce5796c886d77cd1cbaa0dc3b56396ea`.
- `.local/loss-publication-swift-green.log` SHA256 `e512daff014fc8c86f08f2fb8cef24903b7cf8dc793ccb235724d4a2c555baed`.

## Next boundary exposed after publication

The next serialized run selected LossRuntimeCapabilitiesTests, LossRuntimeAdmissionTests and LossDeclarationTests. It terminated1: declaration3PASS; decoder3methods16semantic failures for the intentionally missing exact-descriptor verifier; App7methods8failures (six unexpected prerequisite incompatibleRegistry errors from BuiltInCapabilityBridge excluding newly published objective definitions). This is not an App-admission GREEN or a clean seven-method gate RED. The App worker test also needs the development-root environment on replay, which was omitted in this initial run. Assigned narrow BuiltIn bridge integration separately; strict decoder can proceed from its actual semantic RED, while App admission must receive a fresh meaningful RED after prerequisites are repaired.

- `.local/loss-runtime-admission-red.log` SHA256 `d7d8014335cf9c0dc77ea24a59c94329c128a0bd05f5275e712348433b5e4cbb`.
