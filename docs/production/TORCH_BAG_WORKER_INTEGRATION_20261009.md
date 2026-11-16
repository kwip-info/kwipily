# EmbeddingBag worker integration · 2026-10-09

Status: actual prepared-data/training/checkpoint/export paths verified at worker level. Native workflow mapping/source admission, composition, requested-derivative and History readers, lessons/art registration and final bundle acceptance remain open. Coverage remains83/88; no discovery/catalog publication.

## Working path

Actual CSV→Cast→stratified Split artifacts pass the new completed-inventory bag loader. Selected ID/weight/length columns produce real fixed/packed packets with ordered targets, source/split/fitted-state identities and table pins. ProductExecutionAdapter validates packet sources and captured named ports without inventing a packed sample shape, and retains explicit CPU/MPS placement and real floating weight gradients. [Loader](TORCH_PREPARED_BAG_LOADER_20261009.md) · [Product adapter](TORCH_BAG_PRODUCT_EXECUTION_20261009.md).

The worker counts observations once per partition. Training/accumulation windows, validation batches, representative capture, metrics and checkpoint schedule use bags rather than packed items. Whole-observation slicing preserves empty bags and their targets. Ordinary tensor slices and legacy prediction metadata remain unchanged. Default Adam refuses sparse table gradients explicitly; supported native sparse execution remains independently verified, not a silent dense conversion.

CPU batches1 and3 match an independent native Adam loop exactly for trained weights and epoch losses. Epoch-boundary CPU continuation matches uninterrupted weights/history; changing bag boundaries refuses continuation. Checkpoint input identity now additionally copies the prepared model mapping and ordered bagSources provenance; semantic contracts bind EmbeddingBag parameter registration order. Actual source authority belongs to the prepared loader, not these identity fields.

Real prepared training and saved weights execute on available CPU/MPS. Final validation is checked against direct row-by-row weighted table arithmetic. The saved model.py and learned state then reproduce those outputs in a fresh `python -I` process outside the checkout, with no runner imports. Self-contained export preserves real IDs/offsets/weights, native controls/state keys and input-weight derivatives. [Export receipt](TORCH_BAG_EXPORT_20261009.md).

Bag prediction artifacts explicitly use schema3 and a bagPacket source descriptor; each row contains its actual indices/optional floating weights, including empty arrays for a real empty bag. Binary targets remain labels; explicit objectives retain real continuous target/output/raw-loss rows and do not invent classification accuracy. Prediction/activation capture-off keeps values omitted. schema3 is **not yet admitted by native History/source consumers**. Ordinary artifacts keep their existing schemas. Requested bag-derivative capture has a clear temporary refusal pending its typed consumer; direct native floating derivatives are verified separately. These unresolved consumers prevent publication/full-bundle credit.

## Actual evidence

- Training RED:3 failures on packet `len`, `.local/bag-training-red.log`.
- First actual training GREEN:3 passes, `.local/bag-training-green-attempt.log`.
- Added real CSV, CPU/MPS isolated export and explicit objective/capture checks. First expanded replay:five passes/one incorrect test assumption that nonclassification results include `accuracy=None`; existing behavior correctly omits the key. Corrected assertion to require absence; no product change for that failure. Final training plus checkpoint-identity12 passes, `.local/bag-training-integrated-green.log`.
- Checkpoint provenance/order RED:two failures/four passes, `.local/bag-checkpoint-provenance-red-2.log`; actual absent mapping/order fields. After binding, focused lifecycle/training35 passes, `.local/bag-checkpoint-provenance-green.log`.
- Initial combined worker replay:521 passes/one failure, `.local/bag-worker-integration-root-green.log`. In-process CPU training tests leaked their deterministic-algorithms setting into a later MPS index-accumulation backward test. Production runs own isolated workers. Added test-local restoration of the prior deterministic/warn-only flags; did not disable runtime checks or loosen MPS numerics. Final identical combined replay: **522 passed, zero skips, three existing warnings,14.42s**, `.local/bag-worker-integration-root-green-2.log`, SHA256 `042a64bb0c3abae72965e85fe66d31b5c330421ce46baf26767ad930c2729ba6`. Failed log SHA256 `b645652c327233e1a1c160a6e7f9003dc20e6863e14e6e400e5176bd42102bce`.

Exact final command: `.venv/bin/python -m pytest runner/tests/test_bag_training.py runner/tests/test_bag_export.py runner/tests/test_bag_product_execution.py runner/tests/test_prepared_bag_loader.py runner/tests/test_prepared_bag_data.py runner/tests/test_bag_source_graph.py runner/tests/test_bag_source_semantics.py runner/tests/test_bag_checkpoint_identity.py runner/tests/test_model_input.py runner/tests/test_embedding_bag_packet.py runner/tests/test_checkpoint_lifecycle.py runner/tests/test_checkpoint_lifecycle_modes.py runner/tests/test_product_execution.py runner/tests/test_runner.py runner/tests/test_training_sample_capture.py runner/tests/test_universal_sample_capture.py runner/tests/test_gradient_policy.py -q`.

Warnings are one existing nested-Tensor prototype notice and two existing KL mean-reduction notices. Scoped component/loader/export receipts record their own genuine RED/GREEN and narrower counts; these overlap and are not added as unique coverage. No native application observation is claimed by Python execution.

## Next finite closure

1. Freeze and implement versioned workflow bag mapping, native configuration and independently checked source transport, preserving legacy envelopes.
2. Admit actual named-role bag sources in eligible native containers without weakening unary Sequential policy.
3. Integrate requested floating weight derivatives and schema3 prediction/History readers with permissions/budgets/partial-failure behavior.
4. Pair two approachable assessed no-code lessons and the prepared example; register existing generated original art through authored bundle.
5. Package/native journey, affected prior83 replay and evidence/catalog admission; only then close the bundle.

## Root verified pins

| File | SHA256 |
|---|---|
| runner/engine.py | 49ff50cef05b6ec484647443be221b8eb5bf9136c4c52d2723adf0a58bbb67ad |
| runner/product_execution.py | e4c9190d4fe20b6c2c28b99a154c7b4ff82c13ea74b83ce6316683501d078a8d |
| runner/bag_training.py | ed75cdd412a3522aca585b22ff3354f7b5e497f6b3145f3675972ea6ba68266a |
| runner/prepared_bag_loader.py | 935669cedcc7d25636d0a5f8b622ca6411bc6aae9157620255d42fa3ce6fbe8c |
| runner/bag_export.py | 791b87ceef3e73f4425e50ebc90a2d64b749b7ff0112342c3d97721e00313759 |
| runner/checkpoint_lifecycle.py | a972388aca7da23407de9813bc9fecaf7a40b5c588a75e6800bcbdec0a5ce3c0 |
| runner/tests/test_bag_training.py | c04cba5237b79df0d6ec86420f8c745b7b4025391bb062caefabc32004ae3206 |
| runner/tests/test_bag_checkpoint_identity.py | 95bd48daba88e436477de4be162787534666f614572e8de4cfffcce2d505129f |
