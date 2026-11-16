# Prior-unit reconciliation after LazyLinear · 2026-10-08

Read-only current dependency audit of the prior eleven accepted units, plus an independent authoring replay. Original frozen receipts, catalogs and coverage reports remain untouched by this work. New LazyLinear native/package acceptance and accounting remain pending/root-owned. Base HEAD `d11a96de52893a2d52168090d8ebf34466c6b81f`; the uncommitted source slice is separately identified by dependency hashes below.

## Exact stale verified dependencies

All eleven prior units have at least one changed bound source at this observation. The table records only currently asserted verified dependencies whose stored SHA differs from the file. No other stale dependency was found. Hashes are proposed reconciliation values, not automatic renewed credit. Repeated bindings must be refreshed separately in their original dimensions and unit mode scopes.

| Unit | Changed dependency | Existing dimensions | Current SHA256 |
|---|---|---|---|
| torch.nn.Linear | `components/layers/bundle.json` | configuration | `cdc72f016af7ec71bc87e21d9e6e7a828161b26b94a0af30a8b6d2e611ea6bd9` |
| torch.nn.Linear | `scripts/component_bundles.py` | validation | `79071372e50afe7c326c396fc3ccccca3f5b4d6b7159cb6cfb45c86e3eb9a6dc` |
| torch.nn.Linear | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.Bilinear | `components/layers/bundle.json` | configuration, ui | `cdc72f016af7ec71bc87e21d9e6e7a828161b26b94a0af30a8b6d2e611ea6bd9` |
| torch.nn.Bilinear | `scripts/component_bundles.py` | validation | `79071372e50afe7c326c396fc3ccccca3f5b4d6b7159cb6cfb45c86e3eb9a6dc` |
| torch.nn.Bilinear | `Sources/KwipMLCore/BundledLayerComponents.generated.swift` | example | `5e82d27168a70ffb89833553f0d9032bc53e9b3349955732b94e9a7caf16b3e1` |
| torch.nn.Bilinear | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.Bilinear | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `05d86a17a34229a8670690bcc71952740fd3a6b79156605f555ecc7e8dc5fc61` |
| torch.nn.LayerNorm | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics, lesson | `0fb3d402e222633693df770dd531e90a92c794710193357cd4c52c4da860d3ae` |
| torch.nn.LayerNorm | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.LayerNorm | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `05d86a17a34229a8670690bcc71952740fd3a6b79156605f555ecc7e8dc5fc61` |
| torch.nn.ReLU | `runner/learning_evidence.py` | lesson | `7056c02131d6d210746dc8eea644c3522d9ef3761da47ab20a7e5d3d7d6fb165` |
| torch.nn.ReLU | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.Tanh | `runner/learning_evidence.py` | lesson | `7056c02131d6d210746dc8eea644c3522d9ef3761da47ab20a7e5d3d7d6fb165` |
| torch.nn.Tanh | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.Sigmoid | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.GELU | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | validation | `05d86a17a34229a8670690bcc71952740fd3a6b79156605f555ecc7e8dc5fc61` |
| torch.nn.GELU | `runner/learning_evidence.py` | lesson | `7056c02131d6d210746dc8eea644c3522d9ef3761da47ab20a7e5d3d7d6fb165` |
| torch.nn.GELU | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.Softmax | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics, lesson | `0fb3d402e222633693df770dd531e90a92c794710193357cd4c52c4da860d3ae` |
| torch.nn.Softmax | `Sources/KwipMLApp/TensorLearningStore.swift` | diagnostics, lesson | `a92c29b1b6211552d11d42384bd5322946226c820fb9f446b6584dce5e23f425` |
| torch.nn.Softmax | `runner/learning_evidence.py` | diagnostics, lesson | `7056c02131d6d210746dc8eea644c3522d9ef3761da47ab20a7e5d3d7d6fb165` |
| torch.nn.Softmax | `Sources/KwipMLApp/TensorLearningView.swift` | lesson, ui | `1773dcc13ccab3782805a5d531d84aeb03d3b96d027e745c6d1aacf3634d1e7d` |
| torch.nn.Softmax | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.nn.Softmax | `Sources/KwipMLApp/TensorLearningCanvasAdapter.swift` | ui | `cd72f28751861063387474c2ed2f4262b791b78e24458a8cc8d7be106037dd35` |
| torch.nn.Softmax | `Sources/KwipMLCore/TensorLearningDraft.swift` | persistence | `7333ab957da39aa8825935a51f32eb471c1287ac8e587ab3dafda2ae0dcdde50` |
| torch.nn.Softmax | `Sources/KwipMLCore/TensorLearningReceipt.swift` | persistence | `b83fd3d95e5613ad50fc9ef1a24b5f1a47a4c5ec0fa25e6839241f8da73398da` |
| torch.nn.Softmax | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `05d86a17a34229a8670690bcc71952740fd3a6b79156605f555ecc7e8dc5fc61` |
| torch.reshape | `scripts/component_bundles.py` | configuration | `79071372e50afe7c326c396fc3ccccca3f5b4d6b7159cb6cfb45c86e3eb9a6dc` |
| torch.reshape | `Sources/KwipMLApp/TensorLearningStore.swift` | diagnostics | `a92c29b1b6211552d11d42384bd5322946226c820fb9f446b6584dce5e23f425` |
| torch.reshape | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics | `0fb3d402e222633693df770dd531e90a92c794710193357cd4c52c4da860d3ae` |
| torch.reshape | `Sources/KwipMLCore/TensorLearningCanvas.swift` | lesson | `973a950780e4d758d3977a3af3ea66eaf589d442cd368d9284847b7afb4b18b9` |
| torch.reshape | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.reshape | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `05d86a17a34229a8670690bcc71952740fd3a6b79156605f555ecc7e8dc5fc61` |
| torch.flatten | `scripts/component_bundles.py` | configuration | `79071372e50afe7c326c396fc3ccccca3f5b4d6b7159cb6cfb45c86e3eb9a6dc` |
| torch.flatten | `Sources/KwipMLApp/TensorLearningStore.swift` | diagnostics | `a92c29b1b6211552d11d42384bd5322946226c820fb9f446b6584dce5e23f425` |
| torch.flatten | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics | `0fb3d402e222633693df770dd531e90a92c794710193357cd4c52c4da860d3ae` |
| torch.flatten | `Sources/KwipMLCore/TensorLearningCanvas.swift` | lesson | `973a950780e4d758d3977a3af3ea66eaf589d442cd368d9284847b7afb4b18b9` |
| torch.flatten | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.flatten | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `05d86a17a34229a8670690bcc71952740fd3a6b79156605f555ecc7e8dc5fc61` |
| torch.transpose | `scripts/component_bundles.py` | configuration | `79071372e50afe7c326c396fc3ccccca3f5b4d6b7159cb6cfb45c86e3eb9a6dc` |
| torch.transpose | `Sources/KwipMLApp/TensorLearningStore.swift` | diagnostics | `a92c29b1b6211552d11d42384bd5322946226c820fb9f446b6584dce5e23f425` |
| torch.transpose | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics | `0fb3d402e222633693df770dd531e90a92c794710193357cd4c52c4da860d3ae` |
| torch.transpose | `Sources/KwipMLCore/TensorLearningCanvas.swift` | lesson | `973a950780e4d758d3977a3af3ea66eaf589d442cd368d9284847b7afb4b18b9` |
| torch.transpose | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `c6eabbd9f68571a532367c663af79a0fc57aee00ce701cf4ba38e651e6350c63` |
| torch.transpose | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `05d86a17a34229a8670690bcc71952740fd3a6b79156605f555ecc7e8dc5fc61` |

## Semantic preservation and executed receipts

The prior Linear and Bilinear manifest records compare exactly equal as parsed JSON to the current records. Layers generator changes admit one new exact LazyLinear record and its actual factory; the existing two definitions/runtime contracts remain unchanged. No ordinary Linear/Bilinear materialization or RNG code was edited in this authoring slice. Existing tensor-layout bundle definitions and its shared runtime descriptor are unchanged.

Root's new learning path is explicitly dispatched only for assessment.lazy-linear-inferred-width1.0.0. Existing tensor/Bilinear/activation worker branches remain distinct. Core draft/receipt/task/canvas additions allow the new task without replacing older task IDs. Shared tensor view/store changes render measured Lazy state rows only for the new task; old output/gradient formative paths remain. Audience guide adds plain-language Lazy instructions, while prior task branches remain. Admission adds lazy_linear to selected compiled graph/queued checks, preserving older operation guards.

Actual evidence inspected:

- Agent-run `.local/prior-lazy-linear-authoring-replay.log`: **169 passed, no skips, 0.14 seconds**. Shared authoring plus activation/normalization/layers/tensor-layout/Bilinear/Lazy negative controls executed. `scripts/component_bundles.py --check` completed successfully; registry count remains30.
- Root-run `.local/lazy-linear-final-scoped-swift-green.log`: **45 Core +34 App passed, zero failures/skips**. Includes authored metadata/new task/reference/submission/receipt state and old tensor/Bilinear/LayerNorm projections/lessons; actual TensorLearningStore measured submission, staleness, Softmax measurement/reopening and Embedding routes are visible in the log. This audit did not execute Swift or UI.
- Root-run `.local/lazy-linear-learning-dispatch-regression.log`: **185 passed,6 skipped,13.52seconds**. These six explicit guided-mode-not-applicable skips are not counted as acceptance. Existing dispatch branches execute beside the new Lazy policy; this is automated replay, not native acceptance.
- Parent-supplied native observations: both bias settings received wrong-output→repair→private pass, stale evidence after edits and reopening. These observations belong to new Lazy task acceptance and do not attest native replay of every prior lesson. Corrected pinned-SDK packaging/current native confirmation remains pending at this snapshot.

## Bounded prior-unit reconciliation plan

1. Linear/Bilinear manifest and generator dependencies: use exact unchanged-record comparison,169authoring replay and root's authored metadata regression. Bilinear generated launcher remains its existing factory; no prior unit's trained-state numerical contract changed.
2. Shared worker lesson policy dependencies (ReLU/Tanh/GELU/Softmax): use185actual dispatch passes, preserving unavailable guided-mode cases as skips. Do not imply these tests are fresh native assessments.
3. Tensor draft/receipt/canvas/store/view and audience-guide dependencies (Softmax and layout units; LayerNorm guide): use the45Core/34App scoped test receipt, exact operation-gated source review and existing older frozen measured/privacy/reopening receipts. No broad native claim for untouched lesson IDs.
4. Admission dependencies (Bilinear/LayerNorm/GELU/Softmax/layout): the guard adds Lazy only. Reconcile with existing admission regressions plus the current scoped Lazy admission cases; repeat any old native representative flow only if a failure or root's targeted inspection requires it.
5. Shared Learn catalog dependency (all eleven): root should confirm new Lazy lesson/example discoverability and uniqueness plus one retained prior example/lesson navigation in the packaged candidate. This current native catalog confirmation is not established by this audit. No eleven-unit retraining is required solely for additive catalog entries.

After the bounded checks are accepted, root may append this supplemental receipt in affected dimensions/modes and update only listed source hashes. Preserve each original frozen receipt and each unit's original accepted mode list; never refresh all hashes without reviewing changed semantics. Re-run binding regressions/coverage after updating. New Lazy credit waits for its own frozen acceptance and current native checks; global release gates remain open.

## New LazyLinear binding dependencies

The staged `capabilities/tests/test_lazy_linear_acceptance_binding.py` preserves inference,training,input-gradients,state-roundtrip and requires future `docs/production/TORCH_LAZY_LINEAR_ACCEPTANCE_20261008.md` separately in every mode and all ten evidence dimensions. Actual primary lesson `assessment.lazy-linear-inferred-width`; example `prepared-lazy-linear-classifier`. Root owns the actual incomplete-catalog RED and later binding.

Required dependency map: configuration uses Layers manifest/native/strict configuration; execution and export use engine,direct lazy configuration,independent axes tests and existing materialization/lifecycle tests; validation uses exact runtime decoder/bridge/generator/admission tests; diagnostics and lesson use new Lazy reference/assessment,explicit state criteria,shared tensor controllers and `runner/learning_lazy_linear.py` plus actual policy/refusal tests; example uses generated Layers factory and existing prepared example source; UI uses catalog/generated icon/provenance and root current native receipt; persistence uses new draft/receipt/reload measurement plus real checkpoint/resume/fine-tune/cache suites; failure_handling uses optional admission/configuration/cancel/budget/version controls. Independent runtime gradients support input-gradients; a state-focused lesson must not be relabeled as an independently assessed gradient exercise.

Direct worker, independent numerical test, Layers manifest, new Core lesson reference, learning executor and capability bridge hashes are explicit mutation controls: an unchanged narrative receipt cannot hide stale source. Consumer denominator remains88/minimum80; this document does not grant new coverage or establish broad PyTorch API parity.


Root closure addendum: current native catalog resolves distinct new lesson/example; package integrity/deep signature passed with pinned-Xcode watchdog. Actual lesson refusal/repair/private assessment/restart and prepared example training/History/export/Light/Dark supplement automated replays, recorded in TORCH_LAZY_LINEAR_ACCEPTANCE_20261008.md. Original11receipts unchanged; reviewed affected source hashes are rebound, not rewritten historical observations.
