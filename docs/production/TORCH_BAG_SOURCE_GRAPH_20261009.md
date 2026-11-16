# Bag source graph and shared semantics · 2026-10-09

Status: bounded Core/runtime graph integration verified. No registry/catalog publication or full EmbeddingBag bundle acceptance. Source preparation authority, composition, portable export, training/capture and lessons remain open. Accepted coverage83/88 is unchanged.

## Observable behavior

One graph2 `bag_input` counts as the sole model source. Its exact four controls select fixed/packed layout, 1..128 maximum items, optional floating item weights and terminal-offset policy. Conditional named ports carry real indices, packed-only offsets and optional per_sample_weights. Core BagPortContract and Python BagCoordinateLayout preserve source identity, role and source controls without fabricating an ordinary packed sample layout.

EmbeddingBag requires each enabled role exactly once from the same coherent source, matching layout/weight/terminal controls. Its result is an ordinary float32 observation sample with the configured embedding width. Existing aggregate2M parameter and trainable-parameter checks apply; initial freeze is counted. Ordinary tensor consumers explicitly refuse unreduced bag coordinates. Saved wires to disabled ports remain visible as actionable validation errors. Graph1 refuses both new types; historical graph1/2 behavior is preserved.

Runtime source admission validates real packet geometry and per-bag capacity without detaching or copying its tensors. CheckedEmbeddingBag executes native reductions and actual table/floating input-weight derivatives. Shared graph_semantics now owns bag configuration normalization and identity; the temporary engine-local duplicate was removed. All meaningful bag source/operator controls affect semantic identity, while cosmetic names, coordinates, edge IDs and node/edge ordering do not.

Bag-bearing composition and export currently refuse explicitly rather than silently flattening multiple source roles. No generated registry entry exposes this incomplete bundle.

## Actual RED/GREEN

Core original semantic RED: seven methods,22 failed assertions/two unexpected unwrap failures; `.local/embedding-bag-core-graph-red.log` (SHA256 `4ea7a83a49deb365fe9adde9a09ec068566869276e01c9a88952e58a24f0429f`). First implementation replay:15 methods/eight failures/two unexpected, `.local/embedding-bag-core-graph-green.log`. The independent injected registry fixture incorrectly declared zero fields; existing shared field validation correctly refused it. Corrected only the fixture to literal four/twelve field descriptors. Production validation and positive assertions were not weakened.

Root repair replay:18 Core methods pass, zero failures/skips, `.local/embedding-bag-core-graph-green-2.log`; filter `BagSourceGraphTests|EmbeddingBagConfigurationTests|EmbeddingConfigurationTests|GraphValidationTests|GraphValidatorTests|TypedConfigurationTests|ConfiguredComponentPortsTests`. The existing prepared workflow class also matches this filter. Adjacent graph/typed/port replay:97 Core methods pass, zero failures/skips, `.local/embedding-bag-core-graph-adjacent.log`; filter `GraphTests|TypedGraphConfigurationTests|TypedTensorContractTests|CanvasPortDescriptorTests|CanvasWorkflowPortPresentationTests|TensorRoutingPortIntegrationTests`. App target matches zero methods; do not claim native UI acceptance. Filters overlap; counts are not summed.

Runtime original graph RED25 failed/five passed; source/graph implementation GREEN30 passed. Additional schema1/registry-ceiling RED six failed/36 passed; final scoped/adjacent union195 passed with one existing ordinary Embedding MPS-renorm warning. New bag checks have no fallback. Logs `.local/bag-source-graph-red.log`, `.local/bag-source-schema-registry-red.log`, `.local/bag-source-schema-registry-final-green.log`.

Shared semantic consolidation RED15 failed/nine passed; final focused/adjacent158 passed with the same existing ordinary Embedding warning. Logs `.local/bag-source-semantics-red.log`, `.local/bag-source-semantics-final-green.log`. An independent HEAD comparison against6494ac49bee4d5bcee30db567211cae2e4745559 confirms ordinary graph1Input+Linear and graph2IndexInput+Embedding retain exact semantic graphs, export bytes, state keys, seeded parameters and forwards (`.local/bag-source-semantics-final-pins.json`). This is representative compatibility evidence, not every prior capability's refreshed acceptance binding.

Root combined graph/input/preparation/checkpoint replay253 passed, zero skips/warnings,4.14s; `.local/bag-graph-input-root-green.log`, SHA256 `db86a0a96f3c497a1761b29389a2c2b52bbcacb28ab46d886edcf11fcbcf6b6f`. Exact command: `.venv/bin/python -m pytest runner/tests/test_bag_source_graph.py runner/tests/test_bag_source_semantics.py runner/tests/test_bag_checkpoint_identity.py runner/tests/test_model_input.py runner/tests/test_embedding_bag_packet.py runner/tests/test_prepared_bag_data.py runner/tests/test_checkpoint_lifecycle.py runner/tests/test_checkpoint_lifecycle_modes.py -q`. Actual available CPU/MPS numerical and weight-derivative checks run in the graph matrix.

## Remaining finite path

BAG.3–4 still require independent composition/source consumers, standalone export, authorized completed preparation inventory and paired partition checks, new mapping transport/version, whole-observation training/evaluation/preview and permitted capture/History integration. BAG.5–6 require authentic paired lessons, generated icon registration, real source-to-export lifecycle and native/affected-prior acceptance. [Full contract](TORCH_EMBEDDING_BAG_SOURCE_CONTRACT_20261009.md).

## Verified source pins

| File | SHA256 |
|---|---|
| Sources/KwipMLCore/BagSourceConfiguration.swift | a8fb20886cccb6dd562ef4c1d9ffd0131637230932db68beadd8df0a8d71f8be |
| Sources/KwipMLCore/Graph.swift | 43e17200bb63dd4da16b6ada8a572e060fc41bda19b50ff091ffaac1c5c7edef |
| Sources/KwipMLCore/GraphValidator.swift | aa22920bbc6b58e42b165d98c2fddcac30266f51a0bcb0f046aa667a3f564ed6 |
| Sources/KwipMLCore/ConfiguredComponentPorts.swift | f4445970e729e999ea26ed21eb181eee274a598cf95a3be6e38bf92821b9899a |
| Sources/KwipMLCore/TypedConfiguration.swift | 381a03c3561a27dff4f6c79452b06c7e953294f2cb36e865acf430c46249f6e5 |
| Tests/KwipMLCoreTests/BagSourceGraphTests.swift | 28a31a445065ae09ebb60095236451ca2f550d1150734d44fecf2e2f7db8eb5b |
| runner/engine.py | ce7e29c8a868405106b0e6ca983b34ec52fc56165f34f7fcd4e069b7e275fa55 |
| runner/graph_semantics.py | 9bbd5be1b3fd9564fe90dce1af12ec6874a92c88f6d5ae62b69cf68bdae89164 |
| runner/bag_source.py | 421b1eface7ce8aff8020042b1ca21627d576a9b46468b3e4d5c2277d19206b9 |
| runner/tests/test_bag_source_graph.py | f839f67378d23adf37e46eea2ee4e8fc0f72064bdc2e6104f15eb344401397c3 |
| runner/tests/test_bag_source_semantics.py | 904475a44b1dc768860ec32e87bdb7be8f57c13b23c39bdfb6edcb574ff31aac |
