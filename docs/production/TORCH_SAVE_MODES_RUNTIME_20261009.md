# Checkpoint lifecycle: coherent current-source mode matrix

This additive verification covers the five frozen TORCH-LIFE.1 modes on the actual current worker. It changes only the new test and this receipt. The loader, training engine, frozen-model reader, registry, previous receipts and acceptance catalog remain unchanged. This is host automated evidence, not native UI or acceptance credit.

## Real construction and bounded checks

`runner/tests/test_checkpoint_lifecycle_modes.py` uses the existing genuine CSV preparation helper: CSV source → typed cast → seeded partition → `load_prepared_objective`, with its actual manifest and graph pins. Sixteen observations become twelve training and four held-out rows. The model is Input → tracked BatchNorm1d → Dropout(0.25) → learned affine output; training uses explicit MSE, SGD(momentum 0.5, decay 0.01) and StepLR(step 1, gamma 0.5). Three batches per epoch avoid a singleton normalization training batch. No trusted provenance is fabricated.

- **Weights save/read:** actual completed CPU and MPS runs produce portable model source and saved model state. A fresh exported model loads only that state, evaluates all four held-out rows together and one at a time, and matches a separate plain Python normalization/affine scalar oracle. Dropout is correctly inactive in evaluation. CPU RNG is unchanged by this read/evaluation. The completed checkpoint also contains continuation state; this does not claim a separate minimal weights-only file format.
- **Continuation save/restore:** four uninterrupted CPU epochs equal two saved plus two resumed epochs bit for bit for learned parameters, running buffers, optimizer state and update counts, RNG, explicit objective state, scheduler state, metrics and rate trace. A real epoch callback consumes RNG to exercise restoration. The BatchNorm counter reaches twelve training calls. Exact lineage restores every applicable state family.
- **Fresh fine-tuning:** both CPU and MPS load actual learned parameters and tracked buffers. An observation hook captures the real `GraphModel.load_state_dict` result immediately after restoration, before training, and compares every entry to the parent. Zero learning rate isolates parameter transfer while three fresh training batches update normalization buffers; the counter advances six → nine. Fresh optimizer counts are three rather than the parent's six, momentum buffers are finite and differ from the parent's continuation buffers, RNG/epoch restart with the new seed, and the new rate trace begins at tick zero. Lineage promises weights-only fine-tuning, not MPS exact continuation.
- **Incompatible refusal:** actual parent artifacts reject changed seed, changed objective reduction, changed scheduler recipe, exact MPS continuation, rehashed incompatible buffer dtype and unpinned changed checkpoint bytes. Every failure leaves the successful parent byte-identical and produces no completed child manifest.

Every successful restore/read also checks immutable parent artifact hashes. The matrix exercises one representative objective with no fixed objective buffers (`objectiveState == {}`), not a duplicate of the eight-loss matrix or every checkpoint option. Existing mode/security receipts supply bounded deserialization and prior objective/buffer families separately.

## Actual execution

Host `.venv`, Torch **2.14.1**, macOS 26.6 arm64; **MPS available and exercised**, zero skips.

Command:

```sh
.venv/bin/python -m pytest -q runner/tests/test_checkpoint_lifecycle_modes.py
```

The first execution passed 11 checks in 2.08 seconds. After adding explicit fresh RNG/momentum-buffer assertions, the final execution passed **11 checks, zero failures, zero skips in 1.56 seconds**. This is regression evidence against already implemented behavior; there was no new production repair and no invented RED. Earlier descriptor/deserialization TDD remains recorded in `TORCH_SAVE_RUNTIME_20261009.md`.

Final log `.local/checkpoint-lifecycle-modes-final-green.log` SHA256:
`9b2209b412b576ca4f0a2560d8abf2195e9683f4ed1ed3d99155ea54a8b26604`.

Ignored machine-readable fixture/runtime/source/log pins: `.local/checkpoint-lifecycle-modes-pins.json`.

## Final source identities

| Source | SHA256 |
|---|---|
| New mode tests | `1d51c225528f8e9b7788c30f2ddff900d5a29b7ccfec6b70ba41964ac5625ebe` |
| Frozen checkpoint lifecycle | `66c5cd27da00fe4e3a794c6daa798284fd26f734af051dca414d3aee9c988ab5` |
| Frozen engine | `f62d9be7898c03b3242104e12092f2cce1864c82156635b52ab6b5c491f16f91` |
| Frozen model reader | `b4bc5f685de6358bc1d0b3cfa48d2f5575d8e48abdadad2f941bde9f39909b76` |
| Genuine sealed preparation fixture | `29be334a77b13a9b2b78603cc27d38675f1a9cb3826b3697ab6a29a34e0cc292` |
| Typed objective preparation loader | `7e2e147737f75bc3a77151330606148b78e95aece6148ef73860547ba8f94be3` |
| Dataflow CLI preparation | `0d3a330f501314cd970da208a45a406e03c78cfc0a0358e15a2c2142dda6c40f` |
