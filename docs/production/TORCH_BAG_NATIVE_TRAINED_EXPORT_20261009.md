# EmbeddingBag native trained export — 2026-10-09

Scope: native saved-run export staging in `WorkflowHistoryReader`, the packet variant of `TrainedModelInputContract`, and `BagHistoryExportTests`. Python inference admission belongs to the companion worker change. This receipt does not assert native UI acceptance, framework admission, or release completion.

## Contract

Bag exports use input-contract schema 3. Binary packages contain exactly `schemaVersion`, `input`, `output`; selected-objective packages additionally contain the existing canonical `objective` metadata. Ordinary schema 1/2 export branches remain separate.

The packet input contains `kind: bagPacket`, inner schema 1, actual captured `sourceID`, all four source controls, named coordinate roles and dtypes, the exact table-to-packet mapping, conservative minimum reachable consumer vocabulary (`numEmbeddings`), and `preprocessing: alreadyPrepared`. It has no fictitious dense input dtype, shape, or feature width. Binary output retains `[N, 2]` logits and unnamed class indices; selected objectives retain the actual numeric output axes and canonical recipe, qualified source, target mapping, and scalarization.

The export derives requirements from the captured graph/registry and compares them to cold verified History's bag context. Selected objectives require verified input-independent objective evaluation context and the actual saved standalone helper. Optional optimizer, scheduler, initialization, gradient, and trainability helper branches are retained. The anchored hash-pinned artifact reads, storage reservation, private staging, and publication path are unchanged. Exported source/checkpoint metadata remains subject to the existing disclosure; source observations, prepared rows, CSVs, partition membership, and captured derivative artifacts are not added to the package.

The portable inference interface accepts whole-group records: `indices` plus `per_sample_weights` only when enabled. Fixed groups have exactly the configured capacity; packed groups permit zero through capacity items. The helper constructs offsets rather than accepting caller-supplied coordinate metadata. No source preparation or training loss is performed by prediction.

## Test evidence

Root's actual semantic RED is `.local/bag-final-gates-red.log`: the two real saved-run export tests reached the unsupported dense input and missing dense objective context respectively; the packet contract positive reached the throwing scaffold; the invalid mapping negative passed. Four export methods executed, with three unexpected failures. The actual CPU run fixtures completed before hitting export refusal (binary 6.125 seconds, selected objective 5.186 seconds).

Implementation is frozen for root's serialized Swift GREEN; no Swift command was run by this agent. GREEN evidence must be appended from the actual root terminal, not inferred from source or independent Python packages.

Tests cover all eight fixed/packed × weighted/unweighted × binary/objective contract variants, actual named coordinate roles, mapping/vocabulary, absence of invented dense axes, mismatched mapping controls and overlapping target refusal. Two actual runtime tests stage native saved exports (binary and cross-entropy objective), check schema/helper/privacy inventory, publish a portable copy, corrupt the original checkpoint to prove restaging refuses its stale pin, delete original run files and staging, and execute the relocated package using clean Python import settings. Predictions compare batch sizes 1 and 7, objective values against verified captured outputs, and binary class indices against verified captured predictions. Binary captured rows contain no logits, so that route does not claim a captured-logit numerical comparison.

Limit: these actual saved-run integration cases use the public fixed unweighted example on CPU. Packed/weighted/empty-group inference combinations and CPU/MPS numerical cases are covered in the companion Python suite, not claimed as native saved-package cases here. Existing ordinary trained-export regression tests remain required in root's closure replay.

First root GREEN attempt (`.local/bag-final-gates-green.log`) ran all four export methods: packet geometry/negative contract methods passed; both actual saved export cases failed the new worker-identity guard. Diagnosis: a workflow receipt ID and a model worker run ID are intentionally distinct. The bag context belongs to the verified worker; comparing it to the workflow ID was incorrect. The guard now compares the captured context to `outcome.workerRunID`, already independently tied to the trained model manifest by History loading. Both actual integration cases additionally assert the distinct IDs so this boundary remains explicit. Root's replay is pending.

## Final root GREEN

Actual serialized root replay `.local/bag-final-gates-replay-green.log` passed **10 App methods, zero failures and zero skips, 49.080 seconds**. This includes all four `BagHistoryExportTests` methods (**12.502 seconds**): actual native saved binary export and relocated prediction comparisons (**6.119 seconds**), actual selected-objective helper export and relocated numerical prediction comparisons (**6.282 seconds**), all eight packet contract variants, and invalid controls/target overlap refusal. The same replay passed existing gradient, initialization, and loss History/export coverage.

The first mixed replay's separate ordinary gradient test refused a changing worker source between workflow phases; that refusal was retained in its original log, not treated as a portable export defect. A subsequent compile attempt exposed that the new identity assertion had referenced the live terminal outcome type rather than the archived History outcome; root corrected the assertion to `cold.outcome`. The final GREEN used the actual archived identity assertion, the repaired worker-ID guard, and the final frozen inference helper. No Swift tests or builds were launched by this agent.

Companion Python inference evidence is independently recorded in `TORCH_BAG_NATIVE_INFERENCE_20261009.md`: 75 checks including real trained fixed/packed, weighted/unweighted, binary/objective portable packages, with ordinary dense/objective regressions. These tests supplement rather than replace the final native saved-package replay above.
