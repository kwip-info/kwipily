# Save/load application discovery and admission · 2026-10-09

Scope: the frozen TORCH-LIFE.1 contract, using existing Train and History controls. This is scoped automated application acceptance, not complete bundle acceptance or native UAT. The two assessed lessons, coherent packaged native/export checks and guarded accounting remain open. Coverage remains78/88.

## Behavior

The application installs save/load support only from the separate `lifecycleCapabilities.checkpoint_lifecycle` discovery payload and verifies the complete typed descriptor. Missing, contradictory, misplaced or failed discovery leaves restore unavailable; refresh revokes previously verified support. Lifecycle support does not add a canvas component or change the component coverage count.

Both exact continuation and fine-tuning refuse before source/checkpoint staging when lifecycle support is unavailable. Existing queued dispatch boundaries recheck the same training-readiness function. Fresh training remains available under its existing reviewed runtime contract. The parent chooser displays an actionable runtime reason and the original generated save/restore icon; selecting or inspecting owned History is not unnecessarily blocked.

The asset manifest keeps its repository-relative provenance path. Resource loading derives the bundled PNG basename rather than treating that path as a resource identifier. The exact original PNG and bundled copy share SHA256 `6fe639fbcae65fcd07ecec152855f731644725d281b93f1a01205abfeb2fbcc4`. Existing palette mappings remain unchanged.

## Actual verification

- Semantic RED: two admission methods, six assertion failures, zero unexpected failures; restore staging occurred before support was verified. The earlier fixture setup failure is retained separately and is not claimed as the semantic RED.
- First scoped run: three Core tests passed; fourteen App tests included two artwork assertions failing because the manifest path was used as the resource name. Both real History snapshot tests passed, including actual CPU parent training, continuation/full-run state parity, fresh fine-tuning, source identity refusal and immutable snapshots.
- Final scoped rerun after the resource fix: three Core and twelve App tests passed, zero failures/skips. Includes exact descriptor/refusal, discovery lifecycle, restore-before-staging/fresh compatibility, original decoded artwork and existing component artwork coverage. The two already-passing real History tests were not repeated for the resource-only correction.

No native display observation is claimed for this source candidate. Existing compiler concurrency warnings remain; this slice does not claim Swift6 migration.

## Frozen identities

- `.local/checkpoint-lifecycle-app-admission-semantic-red.log`: `f3af0e5baee123cbfeeda237b10761d37be8e78237ca4cb0a4c970d5518696aa`
- `.local/checkpoint-lifecycle-app-green.log`: `0950375e0d8408fc91179b52e051fa403106a98de261e3aba8fb83c662a24cb0`
- `.local/checkpoint-lifecycle-app-final-green.log`: `d8bcbe5d8714b284f73ea2cdb693a3d259e8c0d9c1ba80f9b9ca941de9722df8`
- `Sources/KwipMLApp/CapabilityActions.swift`: `f47944b81df85c0703ed63f1f941961a7b4530eaa978fef25bfd8aeb464528b4`
- `Sources/KwipMLApp/WorkspaceStore.swift`: `208b884b3c2703fe5563b4f005084ba2739fbec7bca6f8bfb2bfbd513a0fe557`
- `Sources/KwipMLApp/WorkflowWorkspaceSession.swift`: `2cd99b58439af04d2872067d8506637579d497757f5d6ad09c531141dd400c73`
- `Sources/KwipMLApp/ComponentArtworkCatalog.swift`: `0d8bf9501cd22a287f96a849f32761c37e4ca6ce5c728bbbfb33e559bddecfa0`
- `Sources/KwipMLApp/WorkflowCheckpointSelectionView.swift`: `9a5bf50eaae81630c9ec628fb1462b11df85ec2f425215c3c414b00ac0dce957`
- `Sources/KwipMLApp/WorkflowControlInspectorView.swift`: `85830e328bbc83c382c23118b0e1b9b4c329718767dde08bb801bd820fe09006`
- `Tests/KwipMLAppTests/CheckpointLifecycleAdmissionTests.swift`: `61e4d83f64b88c7fa412f3482d75987224079ad72007f6b155da31a13218c739`
- `Tests/KwipMLAppTests/CheckpointLifecycleArtworkTests.swift`: `b39dc87ee311bacf64172415f2d914109b813a9b1e7f212345891f476e0c99dd`
