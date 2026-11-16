# Paired prepared bag loader — 2026-10-09

Status: **bounded completed-preparation loader implemented and tested; not engine/Workflow transport or EmbeddingBag bundle acceptance**. The four original capability modes, complete controls, execution/export, lessons/artwork and final coverage admission remain required.

## Ownership and API

Only new `runner/prepared_bag_loader.py`, `runner/tests/test_prepared_bag_loader.py` and this receipt were authored in this task. No existing loader, engine, Swift, registry or catalog files edited; no builds, native UI or installs.

```python
load_prepared_bags(manifest_path, training_node, training_port,
                   validation_node, validation_port, *,
                   source_config, mapping, num_embeddings, embedding_dim, mode,
                   expected_manifest_sha256, expected_graph_digest,
                   target_column=None, objective_mapping=None,
                   objective_recipe=None, prediction_shape=None)
# -> ((train_packet, train_targets, val_packet, val_targets), provenance, files)
```

Source configuration and mapping are exactly the contracts in [prepared row conversion](TORCH_PREPARED_BAG_DATA_20261009.md). Actual table-to-packet/target conversion delegates to `prepare_bag_table`. No second source, fake ordinary sample shape, lesson-only weights or new target framework is introduced.

## Admission and lineage

- Both manifest and source-graph pins are mandatory exact lowercase SHA-256 strings. `dataflow_storage.completed_inventory` validates completed status, regular bounded files, inventory uniqueness and captured artifact bytes/size/hash before conversion. Failed, cancelled or running preparation is refused even with freshly recomputed outer hashes.
- The captured `dataflow.json` must match the graph pin and pass actual `dataflow.validate`. A valid hash alone does not admit malformed operators/connections.
- Every table/state inventory entry must identify a declared graph node/output with matching state-versus-table kind. Duplicate node/port entries are refused. Selected training/validation endpoints must be actual declared table outputs with the correct role.
- Public `table_from_artifact` validates selected schema, row identities, membership/data version and fitted-state metadata before tensor conversion. Paired tables must have identical schema, source digest, split identity and ordered fitted-state chain, and disjoint observation IDs.
- Actual fitted-state artifacts are parsed through existing `dataflow.load_state` (including training-role/membership/content validation), must belong to the captured graph, and every referenced state must share source/split identity and have fit observation IDs contained in the selected training IDs. Missing state and rehashed heldout fit membership are refused.
- Bag IDs/categories may legitimately repeat across partitions; no false categorical-value disjointness is imposed. Observation-ID isolation and the established source/split/fitted-state authorities prevent partition leakage.

The loader consumes only its captured inventory; it does not scan another run, execute archived preparation programs or independently authorize a different model source. Caller-owned model/workflow transport and execution authority remain unchanged and require later integration.

## Provenance

`dataset` is `prepared-dataflow-bags-v1`. Existing compatible provenance fields retain sourceHash, splitHash, training/validation/sample counts, ordered trainingRowIDs/validationRowIDs, prepared-partition/no-resplitting disclosure and ordered featureColumns. Binary targets retain targetColumn; explicit objective output instead retains the existing objectiveMapping/objectiveRecipe.

`preparation` retains exact manifestSHA256, graphDigest, registryDigest, selected node/ports, training/validation data versions, fitted-state digests and artifactDirectory.

`modelInput` is an explicit packet identity: schemaVersion1, kind `embeddingBagPacket`, layout, maximumItemsPerBag, perSampleWeights, includeLastOffset, indicesDType, offsetsDType (packed only), weightDType (weighted only) and ordered mapping. It deliberately contains **no ordinary sampleShape** for packed storage.

`bagSources.training` and `.validation` retain the full conversion identities: canonical captured table pin, ordered observation IDs, source/split/data/membership/fitted-state identities, source flags, mapping and target selection. These provide concrete checkpoint/preparation binding material; this task does not claim the checkpoint consumer already uses them.

## Actual TDD and corrected fixture evidence

First invocation against an importable no-op loader:

```text
.venv/bin/python -m pytest -q runner/tests/test_prepared_bag_loader.py
```

`.local/prepared-bag-loader-red.log` reported17 failed, exit1: **15 semantic failures and two fixture setup errors**. The fixture incorrectly named a nonexistent `standardScalerApply` operator. The registered application operator is `kwip.data.transform`. This was a test-fixture defect, not product evidence for those two cases.

The first implementation run retained the same fixture mistake:15 passed/two failed, exit1, `.local/prepared-bag-loader-green.log`. After correcting the fixture, an explicit no-op control on the corrected actual completed-dataflow fixtures was run:

```python
from runner import prepared_bag_loader
prepared_bag_loader.load_prepared_bags = lambda *a, **k: (None, {}, {})
pytest.main(['-q', 'runner/tests/test_prepared_bag_loader.py'])
```

That verification control produced **17 semantic failures**, exit1, `.local/prepared-bag-loader-semantic-red.log`. It is a post-implementation no-op control, not a falsely claimed earlier RED cycle. The real unchanged implementation then passed **17 tests, zero skips**, exit0, `.local/prepared-bag-loader-green-2.log`.

Supplementary fixed/nonterminal/unweighted/objective/artifact checks were added; final joint command:

```text
.venv/bin/python -m pytest -q runner/tests/test_prepared_bag_loader.py runner/tests/test_prepared_bag_data.py runner/tests/test_embedding_bag_packet.py
```

**160 passed, zero skips, exit0**, `.local/prepared-bag-loader-green-3.log`:24 paired-loader cases,62 row-conversion cases and74 packet cases. Additional assertions are supplementary verification, not separately claimed RED cycles.

Paired tests create actual CSV→Cast→stratified Split completed dataflow artifacts. Fitted cases additionally execute real training-only StandardScaler Fit and Transform on both partitions. Independent literal row calculations verify actual packed IDs/weights/offsets and empty-bag target/ID alignment; fixed/unweighted/nonterminal variants preserve observation counts. Continuous objective targets use the existing MSE target loader without binary coercion.

Negatives include wrong/malformed/missing pins before tensor allocation, partial run status, rehashed schema/source/split/observation overlap/fitted-chain changes, rehashed invalid source graph, invented graph endpoints, a genuine fitted state rehashed with a heldout fit observation, duplicate ports, missing referenced state, changed artifact bytes and manifest/artifact symlinks. The heldout-state failure matches the loader's nontraining-observation refusal after existing state parsing.

Final SHA256:

- `runner/prepared_bag_loader.py`: `935669cedcc7d25636d0a5f8b622ca6411bc6aae9157620255d42fa3ce6fbe8c`
- `runner/tests/test_prepared_bag_loader.py`: `3ff4bd87ec58d2952f916736a57e729a2f3a96b7f232baa85e574601d461717f`
- Corrected semantic no-op control log: `9058c5d385b046f402f5364a20cad84a107518efb148f9054a2fc3093a1d6bd8`
- Final joint GREEN log: `8e9b34143518e5ac9f0b102a1e981a84899cb7385115f24c3c8b19c44f7e5899`

## Remaining work

Root owns captured mapping transport, public source configuration/graph shape domains, actual training slice/device/capture/checkpoint/export consumers and UI/History/lessons. These tests establish actual prepared-source admission and conversion, not actual model execution, numerical layer gradients, checkpoint replay, standalone inference or native consumer acceptance. No EmbeddingBag coverage credit is claimed.
