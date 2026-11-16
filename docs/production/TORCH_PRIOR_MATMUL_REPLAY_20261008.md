# Prior-unit reconciliation after Matmul · 2026-10-08

Read-only audit of the twelve previously accepted consumer units. Original frozen acceptance receipts, catalogs, reports and tracker are unchanged by this audit. New Matmul native/package acceptance remains root-owned and pending at this snapshot. Base HEAD `a13502abffa2c3dc5b38c171a6cb44349dc70521`; uncommitted source hashes below identify this candidate. This is a bounded applicability review, not renewed credit or a whole-framework coverage claim.

## Exact stale asserted dependencies

All twelve prior units have stale asserted source hashes. The table lists every verified dependency whose recorded hash differs from its current file, grouped by unit/path/dimension. No additional stale asserted path was found. Each occurrence retains its original per-mode scope; proposed hash refresh must preserve those bindings rather than replace them with aggregate evidence.

| Unit | Changed dependency | Bound dimensions | Current SHA256 |
|---|---|---|---|
| torch.nn.Linear | `scripts/component_bundles.py` | validation | `e38a13795947abeffa32868349715b888f1f6502ae9b9959fb2bbaea3b523ae1` |
| torch.nn.Linear | `Sources/KwipMLCore/PreparedWorkflowExample.swift` | example | `702b5f39588a762b11cddeba65a646baac91a643c669eda1cce50ee4dcdd36cf` |
| torch.nn.Linear | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.Bilinear | `scripts/component_bundles.py` | validation | `e38a13795947abeffa32868349715b888f1f6502ae9b9959fb2bbaea3b523ae1` |
| torch.nn.Bilinear | `Sources/KwipMLCore/PreparedWorkflowExample.swift` | example | `702b5f39588a762b11cddeba65a646baac91a643c669eda1cce50ee4dcdd36cf` |
| torch.nn.Bilinear | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.LazyLinear | `scripts/component_bundles.py` | validation | `e38a13795947abeffa32868349715b888f1f6502ae9b9959fb2bbaea3b523ae1` |
| torch.nn.LazyLinear | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics | `8b9505f79194c86e744dfd38583eaf9be541eec5cb89f9e793f3b2ff1d3b7212` |
| torch.nn.LazyLinear | `Sources/KwipMLApp/LearnCatalogContent.swift` | example, ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.LayerNorm | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics, lesson | `8b9505f79194c86e744dfd38583eaf9be541eec5cb89f9e793f3b2ff1d3b7212` |
| torch.nn.LayerNorm | `Sources/KwipMLCore/PreparedWorkflowExample.swift` | example | `702b5f39588a762b11cddeba65a646baac91a643c669eda1cce50ee4dcdd36cf` |
| torch.nn.LayerNorm | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.ReLU | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.Tanh | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.Sigmoid | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.GELU | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.nn.Softmax | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics, lesson | `8b9505f79194c86e744dfd38583eaf9be541eec5cb89f9e793f3b2ff1d3b7212` |
| torch.nn.Softmax | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.reshape | `scripts/component_bundles.py` | configuration | `e38a13795947abeffa32868349715b888f1f6502ae9b9959fb2bbaea3b523ae1` |
| torch.reshape | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics | `8b9505f79194c86e744dfd38583eaf9be541eec5cb89f9e793f3b2ff1d3b7212` |
| torch.reshape | `Sources/KwipMLCore/PreparedWorkflowExample.swift` | example | `702b5f39588a762b11cddeba65a646baac91a643c669eda1cce50ee4dcdd36cf` |
| torch.reshape | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.flatten | `scripts/component_bundles.py` | configuration | `e38a13795947abeffa32868349715b888f1f6502ae9b9959fb2bbaea3b523ae1` |
| torch.flatten | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics | `8b9505f79194c86e744dfd38583eaf9be541eec5cb89f9e793f3b2ff1d3b7212` |
| torch.flatten | `Sources/KwipMLCore/PreparedWorkflowExample.swift` | example | `702b5f39588a762b11cddeba65a646baac91a643c669eda1cce50ee4dcdd36cf` |
| torch.flatten | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |
| torch.transpose | `scripts/component_bundles.py` | configuration | `e38a13795947abeffa32868349715b888f1f6502ae9b9959fb2bbaea3b523ae1` |
| torch.transpose | `Sources/KwipMLApp/LessonAudienceGuide.swift` | diagnostics | `8b9505f79194c86e744dfd38583eaf9be541eec5cb89f9e793f3b2ff1d3b7212` |
| torch.transpose | `Sources/KwipMLCore/PreparedWorkflowExample.swift` | example | `702b5f39588a762b11cddeba65a646baac91a643c669eda1cce50ee4dcdd36cf` |
| torch.transpose | `Sources/KwipMLApp/LearnCatalogContent.swift` | ui | `14f69be9014b3464bfad3443ba038a0a9ee3d78f1921e563bbd19482f872493b` |

## Semantic applicability

- `scripts/component_bundles.py`: adds a separate exact `kwip.tensor-math` validator/renderer and one Matmul factory/bridge mapping. Existing activation, normalization, layers and tensor-layout manifest records are byte-identical to HEAD; their contract validators and generated declarations remain distinct. Registry count stays30, and registry-base removes only Matmul. The prior matrices/materialization/state code is unchanged. All188 authoring checks passed without skips, including prior-family controls.
- `Sources/KwipMLCore/PreparedWorkflowExample.swift`: adds Tensor math metadata/resource and launcher lookup before older namespaces. Stable Matmul identity is unique; prior factories/resources remain unchanged. Old Linear/Bilinear/LayerNorm/layout example bindings become hash-stale because they cite this shared dispatcher, not because their prepared graphs changed.
- `Sources/KwipMLApp/LearnCatalogContent.swift`: adds Tensor math authored entries, their existing Tensors & linear algebra grouping, and suppression of a duplicate generic Matmul card when the specialized matrix-example card exists. Existing specialized Lazy/layout/normalization entries remain; shared catalog tests should verify old identities/categories and launch mappings.
- `Sources/KwipMLApp/LessonAudienceGuide.swift`: adds defined Matrix/Array/Directions/Contraction/Broadcasting/Change sensitivity terms only when the operation is a MatrixLearningTask. Existing task branches and their steps remain. Old LayerNorm/Softmax/layout/Lazy diagnostics and teaching bindings cite this shared file; their assessment worker/reference contracts are unchanged.

No prior unit's engine, operation worker, strict runtime descriptor, numerical reference, export logic, parameter state, checkpoint, or saved assessment task identity is changed by this slice. Current prior receipts stay immutable. The shared generated production registry's ordering moves Matmul into its authored family; it does not alter any other definition. Specialized lesson registries retain their independent semantics.

## Actual receipts inspected

- Agent `.local/matmul-authoring-red.log`: actual unsupported-family RED (1 failed/18 refusal checks passed). `.local/matmul-authoring-regression.log`:188 passed, no skips,0.18seconds. Generator `--check` clean. Parsed prior manifests equal HEAD; base removes only Matmul; generatedregistry30.
- Numerical agent `docs/production/TORCH_MATMUL_NUMERICAL_20261008.md` and `.local/matrix-axes-independent-worker.log`:115 CPU/MPS checks passed, no skips,1.75seconds, independent values/input gradients/sample broadcasting/export/refusals. These concern new Matmul acceptance, not prior-unit numerical replay.
- Root `.local/matmul-scoped-swift-green.log`:24 Core and23 App executed, no failures;2 App opt-in tests skipped in that initial run. Includes older BilinearBundleMetadataTests, LazyLinearBundleMetadataTests, LazyLinearBundleDiscoveryTests and LessonAudienceGuideTests, plus current Matmul authored metadata/discovery, both lesson tasks, saved-receipt compatibility, input/gradient submission and6 selected/queued admission checks.
- Root `.local/matmul-actual-runtime-swift-green.log`: both previously skipped opt-ins subsequently executed successfully (2 passed,zero skips): real Swift-reference/worker seeded matrix grading, and actual prepared example code/training/verifiedHistory. This closes the automated opt-in gap, not native acceptance.
- Root `.local/matmul-prior-unit-replay.log`: fresh **3 Core +4 App passed, no skips**. Core runs PreparedWorkflowExampleTests; App runs GELUCatalogTests, LayerNormBundleDiscoveryTests, LazyLinearBundleDiscoveryTests and TensorLayoutLessonRegistrationTests. The selected nonexistent Linear/Bilinear discovery class names produced no tests and supply no evidence. PreparedWorkflowExampleTests exercises `.parallel`, layout, Softmax, both LayerNorm/GELU modes and Guided (plus Embedding/Conv1d separate tests); it does not exercise `.bilinear` or either authored `.component("linear")`/`.component("bilinear")` dispatcher route.
- Root `.local/matmul-prior-factory-replay.log`: **4 App passed, no skips**, comprising BilinearLessonRegistrationTests (1 actual authored metadata/example discovery), LinearLessonRegistrationTests (1 actual authored prepared-flow launch), and LazyLinearLessonRegistrationTests (2 current lesson/runtime/state/discovery checks). Lazy matched the substring filter as well. Core executed0 tests in this invocation; these are4 App checks, not4 Core/factory cases. The two actual Linear/Bilinear tests close the previously identified dispatcher replay gap.
- Parent supplied root observation: compatibility behavioral RED was repaired by preserving the original failed-receipt nextStep; task mode changes intentionally clear the current result, and lifecycle assertions were aligned to this existing contract. Current scoped GREEN includes both task private completion/reopening. No fabricated admission regression: the existing prepared graph already reaches matrix admission through its Tensor layout stages; six checks verify actual paths.

Older similarly named `.local/matmul-discovery-final.log` and `.local/matmul-registration-final.log` are dated2026-10-06 and are historical receipts; they are not presented as this candidate's fresh replay. This audit executed no Swift or native UI.

## Finite hash reconciliation plan

1. Generator bindings (Linear/Bilinear/Lazy validation; reshape/flatten/transpose configuration): use188 actual authoring checks and unchanged original manifest/generated contract comparison. Refresh only listed path occurrences after root acceptance; preserve existing mode scopes and frozen receipt hashes.
2. Shared PreparedWorkflowExample bindings: the fresh3 Core replay covers the older direct numeric cases including Linear `.parallel`, LayerNorm and layout. The previously missing authored dispatcher replay is now closed by the existing App `LinearLessonRegistrationTests` and `BilinearLessonRegistrationTests` in the4 App factory replay. Both inspect catalog entries and execute `PreparedWorkflowExample.component("linear"/"bilinear").document`, checking the observed model and resource. Existing authoring asserts the unchanged generated factory strings and metadata asserts identities, but does not replace execution of this changed dispatcher. Actual new Matmul launch already passed. No need to retrain unchanged prior examples solely because the dispatcher gained a unique new case.
3. Catalog bindings across all twelve: fresh4 App replay now verifies GELU both-mode card, LayerNorm scopes, Lazy and all3 layout identities/dedup/categories/factories. Linear/Bilinear authored entry/factory checks now passed in the subsequent4 App factory replay, alongside2 Lazy lesson checks. No remaining prior-unit dispatcher/catalog replay gap is identified for this bounded additive source change.
4. Audience guide bindings: root's current LessonAudienceGuideTests passed, including loops over every TensorLearningTask operation, explicit Softmax/LayerNorm/layout rules and readable criterion feedback. This directly covers the older tensor task branches; supplement only any identity omitted from that operation list; no new private numerical corpus or derivative assessment is needed for an operation-gated prose addition.
5. Remaining native/package work concerns Matmul's real lesson wrong→repair/private/reopen, artwork/light+dark/drag/Undo, actual trainedHistory/export/checkpoint and current package. Parent reports these pending at this snapshot. Older frozen native receipts remain original observations and do not imply fresh prior-unit native replay.
6. After these bounded applicable checks and root acceptance, refresh exactly the stale dependency rows above and add this reconciliation document as supporting evidence if desired. Do not alter original frozen receipts, expand mode scopes, create parameter-state requirements for parameterless Matmul, or claim any additional unit before its own ten-dimension bindings pass.
