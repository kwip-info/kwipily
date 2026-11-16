# Dataset, DataLoader and executable-program authoring slice

Status: scoped automated authoring and lesson-worker acceptance recorded; representative native verification and complete-bundle admission are recorded in the final batch acceptance linked below.

## Frozen contracts

Dataset and DataLoader are distinct optional recipes connected to Train.dataset and Train.loader. They do not replace Train.data or authorize another source. The exact schema-1/configuration-1 envelope contains componentID and config. Dataset supports indexed, tensor, unique ordered subset and finite restartable iterable views of the original training partition. Loader supports bounded batch size, shuffle, final-group policy, nonnegative Int64 seed, zero to two readers, worker-dependent prefetch/timeout and auto/fixed_stack/padded_sequence/whole_bags collation. Spawn, ordered delivery, no persistent workers and no pinning are fixed policies. Recipe UTF-8 admission is 32 KiB.

Program export is an inference-only artifact action. Its seven controls preserve strict/deferred capture policy, static or bounded leading batch, example/minimum/maximum sizes and explicit CPU/MPS. Sample axes remain fixed. Packet inputs are explicitly outside this executable-program export; their existing trained inference export remains distinct.

## Core evidence

Root owns every Swift invocation. Initial CLT toolchain and later test-expression complexity failures were prerequisites, not semantic RED. Root's correct-toolchain `.local/dataset-loader-config-red.log` executed 13 methods with 14 failures (13 unexpected), including positive rejecting-scaffold controls and workflow codec tests. `.local/dataset-loader-config-green.log` subsequently passed all 13 methods.

Root's `.local/dataset-loader-lesson-compiler-red-2.log` executed 16 methods with 10 failures (9 unexpected): negative seed admission, positive lesson scaffolds and shared compiler new-contract cases. The scoped implementations now reject negative seeds and provide eight data-loading and two program-export task/reference/submission/example contracts. Their independent criteria use row-selection arithmetic, hand affine arithmetic and explicit measured-check expectations. Root replay `.local/data-loading-native-first-green-capture-red.log` subsequently passed all **25 Core methods in 0.072 s**, including all nine new lesson methods, eleven strict configuration methods, three compiler methods and two codec methods. Native artifact/capture App work in the same log is separate and not reported as an App-wide GREEN.

## Python lesson evidence

`.local/program-export-lessons-red.log`: 4 failed / 7 passed in 0.46 s against a rejecting executor scaffold. Actual failure cases included static/bounded isolated reload against hand affine results, portable-process refusal and cancellation.

`.local/program-export-lessons-green.log`: 11 passed in 2.90 s. Executor builds a genuine five-file torch.export package, measures reloaded output, calls actual minimum/maximum and rejected outside batch sizes, rejects changed program bytes and stale manifest pins, and invokes the generated inference helper in a separate Python -I process without the original model class source. It persists the supplied selected affine snapshot and compares its parent bytes after export. This is an immutable selected snapshot measurement, not a claim of trained-app checkpoint provenance; the independent export CLI receipt covers actual trained-source provenance.

`.local/data-loading-lessons-red.log`: 8 failed / 8 passed in 0.45 s against a rejecting executor scaffold. Meaningful positives require actual native source types/order, short and dropped final groups, real reproducible mixing/alignment, padded sequences/true lengths and cancellation. Seven measured source/grouping cases are implemented through production make_source/open_epoch; their first replay exposed Python sequence iteration attempting an out-of-range IndexedTrainingDataset index. The lesson now uses explicit native indexed positions, while iterable sources use their real iterator. It does not change production source semantics.

The eighth continuation task first failed its real regression on that explicit refusal (`.local/data-loading-continuation-red.log`). The implementation then passed `.local/data-loading-continuation-green.log`: 1 passed / 16 deselected in 3.26 s. It writes the actual private rows/lengths/weights/observation IDs to CSV, runs completed source→cast→split preparation and its independent pinned paired bag loader, then executes real CPU full-two-epoch, one-epoch parent and resumed-two-epoch engine runs with two spawned readers. It compares actual model/optimizer/RNG/loading state, observed second-epoch order and unchanged checkpoint bytes, and checks that no owned reader process remains. The per-original-observation rank difference includes held-out rank −1 in both executions, a measured zero difference rather than a fabricated training membership. No production callback imports test fixtures.

A further valid-but-wrong program capture policy test exposed static-task acceptance of a bounded configuration (`.local/program-export-lesson-policy-red.log`: 1 failed / 11 deselected in 2.12 s). Its minimum-boundary check now additionally verifies the actual captured task mode/example/minimum/maximum against the public requested controls.

Final isolated callback union `.local/data-loading-program-lessons-final-green.log`: **29 passed in 6.71 s**, including all eight data-loading tasks, both executable-program tasks, actual continuation, independent hand arrays and negative construction/privacy/admission controls. This remains direct worker evidence; shared managed dispatch/receipt/private assessment UI integration is separate.

## Privacy and audience

Submission envelopes admit only typed construction or export recipe; injected expected answers, arbitrary Train controls, ordinary tensor Input replacements and foreign recipe wiring refuse. Fixtures validate all mapped cells and identities before tensor allocation. Outputs are real measurements; no private expected values are read by workers. Learning copy defines observations, datasets, groups, collation, reader processes, seeds, checkpoints and executable programs in practical no-code language. Assessment privacy and persistence remain shared lifecycle responsibilities for root integration.

## Remaining finite work

Authored manifest publication; shared no-code canvas, managed dispatcher and cold-receipt integration; native checks and complete-bundle accounting. Generated dataset/data-loader/program-export assets already exist under docs/design/data-loading-export-icons-v1 and must be reused.

## Authored source state

`components/training-data/bundle.json` now contains the two strict field templates, exact worker-produced capability literals, four/seven original mode lists, eight assessed lesson identities and generated icon originals. Entries intentionally remain unpublished until shared registration acceptance. Program export descriptor production was subsequently frozen independently in program_export_configuration.capability(); the new authored artifact action pins that literal and its exact seven configuration fields. Generator/registry/discovery integration remains root-owned.

## Narrow bundle generation

Root granted the existing `scripts/component_bundles.py` seam plus two new authoring test files. Dataset family initial semantic RED `.local/data-loading-authoring-red.log`: 2 failed / 8 passed in 0.13 s; program family initial semantic RED `.local/program-export-authoring-red.log`: 2 failed / 7 passed in 0.13 s. New families were rejected and their generated typed metadata was absent.

The generator now extends the existing trainingControl renderer for Dataset/DataLoader and artifactLifecycle renderer for executable-program export. The only shared addition is strict source/learning/artwork validation used by these two families. No fake tensor-forward resolver, dynamic plugin loading or source-authority bypass was added. Frozen literal descriptors are compared against independently implemented worker producers by the new tests. Program action metadata adds its seven standard configurationFields and the History exportProgramPackage binding. Authored entries remain unpublished.

Initial scoped generation GREEN: 19 passed in 0.22 s. A further publication-art negative test demonstrated missing registered artwork could be admitted: `.local/data-program-published-art-red.log`, 2 failed / 19 deselected in 0.03 s. Published entries now require the registered PNG bytes to equal the provenance-pinned generated original. Final scoped authoring GREEN `.local/data-loading-program-authoring-final-green.log`: **21 passed in 0.20 s**. The approved generator and `--check` both succeeded; only four new generated files were added and the existing forward registry had no diff.

The bounded older initialization/lifecycle/gradient authoring union `.local/data-loading-program-authoring-adjacent-green.log` executed 89 tests: 86 passed / 3 failed in 0.91 s. All three stop on stale pre-bag registry expectations (prior-prefix position37, initialization84 versus86, lifecycle84 versus88), before reaching any later hash assertions. This is reported without silently updating old snapshot hashes or claiming the adjacent union passed. Root owns or must authorize their precise frozen-snapshot reconstruction repair. The new unpublished recipes contribute no registry definitions.

New generated metadata: `BundledDataLoadingComponents.generated.swift`, `BundledProgramExportCapabilities.generated.swift`, `data_loading_capabilities_generated.py`, `program_export_capabilities_generated.py`. Palette publication, runtime grants, copied registered artwork, managed lessons and native action integration remain independent root-owned acceptance work.

## Final automated verification — native seal pending

Root's `.local/data-loading-lifecycle-green-managed-lessons.log` passed the selected **four Core methods** (two capability bridge and two formative projection tests) with zero failures in **0.050 s**. The adjacent atomic lesson controls passed both methods in **0.029 s**; the data-loading lifecycle verifier passed both methods in **1.149 s**. Earlier shared task/draft/receipt and Learn integration acceptance remains recorded in the chronology above.

The same root replay passed both actual managed lesson lifecycle methods in **82.141 s**: eight no-code Dataset/DataLoader lessons in 56.586 s and two program-export action lessons in 25.555 s. These exercise **all ten tasks through twenty guided/private submissions**, genuine runtime discovery and source-pinned workers, independent grading, cold saved-receipt replay, stale construction handling and practice/private display guards. This is actual managed execution rather than injected capability or synthetic receipt evidence.

Standalone source-dispatch verification in `.local/data-program-learning-final-green.log` passed **53 Python tests in 21.12 s**, including continuation-policy refusal checks and both real lesson executors. The broader App selection in the managed log had one separate `DataLoadingFailedGradientTests` failure; it is therefore **not** reported as an App-wide GREEN. The scoped lesson suites above passed. Native visual/interaction verification and the final bundle seal remain pending; these automated results do not establish those gates.

## Final root closure

The pending checkpoints above record their historical state. The final sealed candidate passed the representative native journeys and automated checks in [batch acceptance](TORCH_DATA_LOADING_EXPORT_ACCEPTANCE_20261009.md) and [native evidence](TORCH_DATA_LOADING_EXPORT_NATIVE_20261009.md). The separate coverage accounting records the admitted scope and preservation checks; these scoped receipts do not claim final release UAT.
