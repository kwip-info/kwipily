# Objective phase admission and cold History — scoped receipt

Date: 2026-10-09. Finite automated implementation complete; native, export integration and loss coverage acceptance remain separate root-owned gates.

## Exact boundaries

WorkflowPhaseArguments now admits the three explicit objective options only as a complete training selection: canonical recipe JSON (32768 bytes), exact typed target mapping JSON (524288 bytes), and scalarization Mean/Sum. All seven prepared-reference fields are required and validated. Recipe fields cannot rely on omitted defaults; role keys/configuration must match the selected objective. Legacy feature/target options cannot be mixed into this selection. Unknown/duplicate flags and noncanonical/incomplete selections refuse.

Only a complete objective selection receives the 786432-byte phase aggregate allowance. Generic argument values retain their old 65536-byte cap, and preparation/legacy phases retain their 131072-byte aggregate bound. The independent operation envelope decode limit remains 256000 bytes. A structurally valid larger phase mapping is not a promise that the stronger envelope admits its run; the test explicitly verifies that independent refusal.

Cold History forwards the model registry actually decoded and hash-verified from the captured archive into phase verification. It does not substitute the current built-in registry. WorkflowVerifiedModelContent retains four additive default-nil fields from the strictly verified trained result: objectiveContext, objectivePredictionPreview, objectiveEvaluation and objectiveModelCode. Existing legacy binary prediction fields, source/registry recompilation, artifact inventories, pins and anchored re-reads remain in place. No export implementation changed in this slice; the pre-existing explicit-objective export refusal remains for its own task.

## Actual evidence, including unsuccessful setup observations

All Swift commands used `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, `swift test --jobs 4`; real-worker scopes also set `KWIPML_TEST_DEVELOPMENT_ROOT` to the repository.

- `.local/loss-history-read-red.log` and `.local/loss-history-read-semantic-red.log`: actual host setup reached an ownership admission refusal before training, two methods/four assertions each. Despite the second filename, neither is the intended History behavioral RED. No fabricated runtime evidence or ownership bypass was introduced. Test temporary roots were corrected to the existing exact Darwin realpath pattern, and root repaired the independently demonstrated phase/Session prerequisites.
- Root's `.local/loss-session-live-integration.log`: one actual MSE Session method reached successful live/trained result assertions and failed only its cold History reopen assertion (1 failure, 0 unexpected). This is the genuine History RED used before the registry forwarding/content retention implementation.
- `.local/loss-phase-arguments-red.log`: 4 methods, 18 assertion failures, 0 unexpected, exit 1. Real factory-derived canonical structural selections were refused by the old CLI whitelist. Declared structural reference pins in this argument test are not runtime execution evidence.
- `.local/loss-phase-arguments-green.log`: 9 methods (4 new phase arguments, 5 existing ownership), 0 failure/unexpected/skip, exit 0.
- `.local/loss-history-phase-final-green.log`: 22 App methods, 0 failure/unexpected/skip, exit 0 at 01:16:47; elapsed 27.421 seconds. New History 2, phase arguments 4, root Session 2, existing History reader 9, existing ownership 5. The enabled actual legacy cold-host History method also passed.

New History checks use genuine current runtime discovery, source permissions/bindings and captured supervised preparation/training. They create all eight completed objective archives plus a separate MSE archive, retire their hosts and cold-read saved content. Checks compare exact qualified objective recipe/source, captured registry hash, worker identity, actual output shapes, typed prediction/evaluation/code and honest optional accuracy. Changed prediction bytes and a removed actual objective code artifact refuse on access; restoring the original prediction bytes restores verification. The two root Session methods additionally verify genuine direct and supervised MSE publication.

No synthetic runtime proof, fake completed receipt, installed-registry fallback or cached authorization was used for the new History execution evidence.

## Source and proof identities at frozen handoff

These hashes identify scoped files/logs at handoff, not a repository-wide before/after audit. WorkflowHistoryReader retains a root-authored pre-existing optional-target/export guard outside this slice; its changes here are only the new typed fields, captured registry forwarding and verified field retention.

| Path | SHA-256 |
| --- | --- |
| `Sources/KwipMLApp/WorkflowPhaseArguments.swift` | `1d38369b10a24f127f4bc25665040a7244b09e8665fe8189afbde8862a524f22` |
| `Tests/KwipMLAppTests/LossPhaseArgumentsTests.swift` | `6af11e1abdfdf7bd3408e98ec212b2058372abbbe64f9938d75f90aa9283827d` |
| `Sources/KwipMLApp/WorkflowHistoryReader.swift` | `62dd11eb566bb9b160a9c6f0a21174325f2e87e19c113e30e06d6e228fdc21c4` |
| `Tests/KwipMLAppTests/LossHistoryReadTests.swift` | `ccbea2591bd83e452d9a6c1826890775a7abda8b3ef43396512a8d490ac7ac21` |
| `.local/loss-phase-arguments-red.log` | `184ce9d074ea8045603f91904f7b13f6a1b559501421a59135d1968c3d8b2faf` |
| `.local/loss-phase-arguments-green.log` | `95c0843be36a5db353aa4db24a2f7f2c04d1732a85f1aa66f7b74006afa8b88f` |
| `.local/loss-session-live-integration.log` | `f95ab807b2e513664d3c0fdd0847976ba44ce79a21de59c50b0987f948fcf0f9` |
| `.local/loss-history-read-red.log` | `16cad71ec175fc3abeb690740f916d7f1dfb7bf3690a8c9429c8e8e5cca39944` |
| `.local/loss-history-read-semantic-red.log` | `87a9493412b556bb2cd5f6abaae7d97141700c8bc815c5d0eddc1f43cd47669b` |
| `.local/loss-history-phase-final-green.log` | `280cedb364a434eebf6bb523d9b1015202ee0d1cafa1ef1407455e438739f3b5` |

Sources/tests are frozen and the serialized Swift lane is released. No tracker/catalog percentage, old frozen receipt, native acceptance or export acceptance is claimed by this receipt.
