# Whole-observation bag input and checkpoint identity · 2026-10-09

Status: bounded shared-input/checkpoint foundations verified. Not prepared-inventory admission, integrated training, export, History or complete EmbeddingBag bundle acceptance. Accepted coverage remains83/88.

## Implemented contract

`runner/model_input.py` exposes observation_count, select_observations, move_to_device, canonical_input_tensors and model_inputs_digest for ordinary tensors or strict EmbeddingBagPacket. Packed observation count comes from bag boundaries, excluding the optional terminal boundary; item count cannot stand in for training sample count. Whole-bag selection preserves empty bags, rebases offsets and retains floating weight derivatives and the requested observation order. CPU/MPS movement preserves all packet fields; unsupported placement refuses without fallback. Integer coordinates never claim differentiability.

Generic packet admission here establishes bounded source geometry, not a connected layer's vocabulary, allocation or device-operation support. CheckedEmbeddingBag and the graph source retain those independent checks. No packet pretends to expose Tensor shape, slicing or detach.

`checkpoint_lifecycle.tensor_digest` now routes typed sources through an explicitly framed identity. It binds source position, packet schema/layout/terminal policy/weight presence, named tensor roles, dtype, shape and actual ordered bytes. Altered offsets, coordinates, weights, observation order or targets change identity. CPU and MPS copies share identity. Ordinary tensor-only collections, generators and the empty collection retain the exact historical byte protocol; no ordinary checkpoint rehash/migration is required.

Mapping/provenance/observation-ID binding still belongs to the prepared-data and checkpoint contract integration. This input digest does not authorize a captured source, complete epoch continuation or accept a partial materialization.

## Actual verification

- Shared-helper semantic RED:21 failed; helper/packet GREEN95 passed, `.local/embedding-bag-model-input-green.log`.
- Actual checkpoint consumer RED:3 failed, one passed (`EmbeddingBagPacket` had no detach), `.local/bag-checkpoint-identity-red.log`; log SHA256 `7b6c382932687c00e5b043590348b82b5e2ab9556ff616be2af07974744134e0`.
- Actual checkpoint consumer plus helper, packet and existing lifecycle/mode regression GREEN:125 passed, zero skips, `.local/bag-checkpoint-identity-green.log`.
- Root combined replay adds the prepared-table converter:187 passed, zero skips,3.54s, `.local/bag-input-foundations-root-green.log`, SHA256 `8d44d521db378455c5ed4d20c918f9e6cd6a50d0aae025e7e4307a8a2de5ed7b`.

The combined command is `.venv/bin/python -m pytest runner/tests/test_bag_checkpoint_identity.py runner/tests/test_model_input.py runner/tests/test_embedding_bag_packet.py runner/tests/test_prepared_bag_data.py runner/tests/test_checkpoint_lifecycle.py runner/tests/test_checkpoint_lifecycle_modes.py -q`. Counts overlap; do not sum them as unique coverage. Actual available MPS movement/identity/weight derivatives were exercised. No native application observation is claimed here.

Source SHA256 at verification:

| File | SHA256 |
|---|---|
| runner/model_input.py | 87e26c5689ad3528e5e42dbd6e0acfd65b0261c24255c267f6f3d19f2088ef87 |
| runner/tests/test_model_input.py | 8fe68897eba7b3f986bbc59a849fe1e6e54f02cb07b38145a2c070c62c569f3e |
| runner/checkpoint_lifecycle.py | 494024b7818577c80a089cf0b82172b44cdc449263a5cca1961e6b338040cad2 |
| runner/tests/test_bag_checkpoint_identity.py | df767fb6dfaaeb90ec1ddac351b9802d472ab38d753558e2cd5b2c8ca5f0b57d |

Remaining BAG.3–6: consolidate graph source semantics, source/plan/export admission, real completed-inventory and paired partition authority, versioned configurable mapping, training/preview/capture integration, paired lessons/art registration and final native/affected-regression acceptance. The finite [source contract](TORCH_EMBEDDING_BAG_SOURCE_CONTRACT_20261009.md) remains authoritative.
