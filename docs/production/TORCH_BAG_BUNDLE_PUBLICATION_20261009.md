# EmbeddingBag authored bundle publication — 2026-10-09

Status: authored bundle, deterministic generated definitions and approved icon resources are implemented and Python verified. Root subsequently verified native declarations, discovery gating and public example execution as recorded below. Six managed lesson closure and UI acceptance remain separate gates. This receipt does not claim full bundle admission or native UAT.

## Finite change

`components/embeddings/bundle.json` is bundle version 2 with the complete `bag_input`/`embedding_bag` pair. Its original `embedding` and `index_input` rows retain their exact parsed values and configuration version 2; fixed SHA256 assertions protect both complete legacy rows. The new source/operator use configuration version 1 and their independently reviewed four/twelve controls. Their runtime contracts are separate from the legacy embedding descriptor and retain honest whole-bag/empty/weighted/MPS policies.

The embedding validator dispatches new rows through a narrow exact-definition/runtime/source-authority validator. It accepts either the original legacy pair or all four members; partial pairs, duplicates, mismatched publication flags, incorrect versions, fields/defaults/bounds/choices, role metadata, runtime policies, unsafe/missing references and changed lesson/example identities refuse. Configured roles remain the responsibility of actual typed source/operator adapters; `outputConditions` stays empty. Offsets are not misrepresented as dependent on the terminal-offset switch.

The generated new declarations call `EmbeddingBagCapabilityBridge.declaration`, and new examples call the actual `WorkflowExamples.preparedEmbeddingBagClassifier(device:)` extension. New learning metadata points to its real source `EmbeddingBagWorkflowExample.swift`, source resource `bag-observations`, and the exact six managed assessment IDs. `Learning.lessonIDs` is optional to retain decoding of legacy rows. Existing two factory and declaration mappings are unchanged.

Approved generated icons were copied byte-for-byte into App resources, with no regeneration or replacement artwork:

- `bag-input-v1.png`: `ec079864d36b0285c4ff214b87931a1acfaddd4ae22993750ee0fae00280b09f`.
- `embedding-bag-v1.png`: `ad5de9056c16b1c14d412d41c37f85984b52b61132e4c25b44fea35a62a74d78`.

Canonical generation changed exactly `BundledEmbeddingComponents.generated.swift` and `Resources/components.json`; all other generated output bytes were unchanged. Registry membership is now 88 total entries, including 85 model-family entries and the three separately tested control recipes. The general ordering test now explicitly excludes initialization, gradient policy and trainability when checking model-family ordering.

Two additionally leased Python fixture helpers (`test_bag_source_graph.py`, `test_bag_cli.py`) remove any shipped bag definitions before appending their independent descriptor literals. They do not substitute generated definitions for independent expectations; no semantic assertion changed. Root separately owns the equivalent Swift fixture changes.

## Actual evidence

1. Semantic authoring RED: `.local/embedding-bag-bundle-authoring-red.log`, **1 failed / 26 passed**, 0.09 seconds. The exact proposed four-row positive failed at `Unknown embedding operation` before validator/render implementation.
2. After implementation, the positive initially refused the genuinely missing public CSV resource. That check was preserved until root supplied the real resource; no placeholder resource or path bypass was added.
3. Final authoring/generator/legacy/general GREEN: `.local/embedding-bag-publication-authoring-green-2.log`, **70 passed**, 0.57 seconds. Command: `.venv/bin/python -m pytest scripts/tests/test_embedding_bag_bundle_authoring.py scripts/tests/test_embedding_bundle_authoring.py scripts/tests/test_component_bundles.py -q`.
4. Legacy and independent source/CLI fixture GREEN: `.local/embedding-bag-publication-fixture-green.log`, **92 passed**, 20.82 seconds. Command: `.venv/bin/python -m pytest scripts/tests/test_embedding_bundle_authoring.py runner/tests/test_bag_source_graph.py runner/tests/test_bag_cli.py -q`.
5. Canonical `scripts/component_bundles.py` generation completed; `--check` passed afterward. `git diff --check` passed for the leased modified files.

The new authoring tests retain a checked-in reviewed proposal snapshot rather than depending on ignored `.local` files. They compare independent runtime producers to authored expectations, verify approved icon bytes, exact distinct factory/declaration dispatch, unchanged legacy rows, precise refusal controls and current generated registry/Swift output. Runtime producers were authored independently by the runtime worker, not generated from bundle JSON.

No Swift build or native UI action was executed by this worker. Generated compiled sources were written only after root released the semantic-RED freeze and are now frozen for root's serialized native verification.

## Subsequent root-executed native evidence

Root's `.local/bag-published-workflow-green.log` completed **33 tests with zero failures**: 22 Core and 11 App tests. This evidence is separate from the Python authoring checks above.

- All four `EmbeddingBagPublicationTests` passed, covering the published generated pair/defaults and declaration behavior plus the real public example's whole-bag mapping, category-ID treatment, and invalid identity/device refusals.
- Runtime capability tests admitted the exact independent pair and rejected changed/unknown fields, types, versions, budgets and absent/mismatched pair discovery. Palette/direct-placement and selected-workflow gates were verified.
- `PublishedBagWorkflowTests.testActualPublishedExampleDiscoversRunsAndReopensWholeObservations` passed in **4.146 seconds**, using the actual public factory, bundled CSV, runtime discovery, training and reopened History whole-observation route.
- Root closed physical coordinate specialization in the Core bridge after a meaningful positive RED exposed invalid specialized shapes. Generic unspecialized bag coordinates and wrong ports continue to refuse; concrete typed source/operator role contracts remain authoritative.

These are native XCTest integration results, not a claim that computer-use UAT or all six lessons have completed. This worker did not execute Swift or change source during this receipt update.
