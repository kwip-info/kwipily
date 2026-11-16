# Prepared row-to-bag conversion — 2026-10-09

Status: **bounded preparation helper implemented; not engine/UI integration or EmbeddingBag bundle acceptance**. All four original capability modes and reviewed controls remain required. Root owns integration and final coverage admission.

## Owned files and API

This task authored only new `runner/prepared_bag_data.py`, `runner/tests/test_prepared_bag_data.py` and this receipt. Existing prepared loaders, engine, Swift, workflow contracts, registry and catalog were not edited. No builds, native UI, installs or source publication.

```python
prepare_bag_table(item, *, role, source_config, mapping,
                  num_embeddings, embedding_dim, mode,
                  expected_table_digest, target_column=None,
                  objective_mapping=None, objective_recipe=None,
                  prediction_shape=None,
                  require_both_binary_classes=True)
# -> (EmbeddingBagPacket, targets, identity)
```

`source_config` has exactly four keys: `layout` (`fixed` or `packed`), `maximum_items_per_bag` (exact integer1..128), `per_sample_weights` (boolean), `include_last_offset` (boolean). This frozen source contract is validated locally without depending on another agent's unpublished source module.

`mapping` is version1 with exact variant-specific keys: always `schemaVersion`, ordered distinct `indicesColumns`; `weightColumns` iff weights are configured; `lengthColumn` iff packed. ID and weight lists each exactly match the source row capacity. Names are nonblank strings up to200 characters. All ID/weight/length source columns must be present and distinct, and separate from every selected target-role column. Objective inputColumns must match the ID column order exactly; conflicting independent input order is refused.

## Authority and preserved identity

The required `expected_table_digest` is a lowercase SHA-256 canonical `dataflow.digest(item)` integrity pin from the captured artifact. **It is not execution permission.** The caller must first verify completed_inventory, the captured manifest/artifact/source graph and appropriate partition authority. A self-consistent/rehashed table does not establish that authority.

The helper invokes public `dataflow_cache.table_from_artifact` for schema/typed rows, unique observation IDs, membership/data-version identity, bounded fitted-state chain and diagnostics checks. It additionally requires the exact requested training/validation/test role and lowercase source/partition hash identities. The helper handles one table only: same-source/split/fitted-state matching across paired partitions remains root's existing-loader integration responsibility. Failed/cancelled capture admission is neither bypassed nor newly established here.

Returned identity copies the ordered observation IDs (including empty bags), source/partition/data-version/membership digests, fitted-state chain, complete lineage, source configuration, ordered mapping, target mapping, selected target columns and canonical table pin. A mapping digest binds source controls, ordered source mapping and target selection. Mutable returned metadata does not alias the input artifact/mapping.

## Conversion semantics

One prepared row remains one bag/observation. Every mapped ID is an exact nonnegative int64 value below the selected vocabulary size. Every mapped importance cell is finite numeric data convertible to float32 without overflow or nonzero underflow; booleans are refused. These checks apply even to empty bags and unused suffix cells before packing, so invalid coordinates/importance cannot disappear through length selection.

- Fixed layout retains every ordered ID/weight cell as `[B,L]`, has no offsets, and retains the terminal flag without applying it.
- Packed layout requires explicit int64 lengths0..L, selects the first length cells in each row and creates actual `[N]` arrays plus offsets `[B]` or `[B+1]`. Length0 preserves the observation/target; equal offsets and all-empty packed partitions are supported.
- Optional weights are actual float32 tensors read from user-mapped columns, never configuration constants. They require sum reduction.
- Binary target policy retains explicit int64 schema, exact0/1 values and both-class presence by default; an explicit false policy supports existing test-partition behavior.
- Explicit objective roles are validated/converted by the existing `validate_target_mapping` and `table_objective_tensors` implementation, preserving its configured dtype/shape/objective rules. This helper introduces no loss interpretation, dummy labels or binary coercion of continuous targets.

Prepared bags admit1..4096 rows, row capacity1..128 and output B×D at most2M elements. Static table dimensions remain bounded100M; actual model allocation/initialization limits remain downstream. Packet validation is reused after construction. This helper creates CPU preparation tensors; selected-device movement/batching belongs to the execution boundary. It does not make table-derived weights learned model parameters or claim their gradients were captured by a production run.

## Actual TDD and verification

The first run used an importable no-op helper returning `(None,None,{})`, so observed failures were semantic rather than missing import/compiler failures:

```text
.venv/bin/python -m pytest -q runner/tests/test_prepared_bag_data.py
```

- **RED:**42 failed, exit1; `.local/prepared-bag-data-red.log`.
- **GREEN after implementation:**42 passed, zero skips, exit0; `.local/prepared-bag-data-green.log`.
- Added supplementary exact-variant/pin, unused-cell, detached identity, objective mismatch and preallocation-limit checks. Joint command:

```text
.venv/bin/python -m pytest -q runner/tests/test_prepared_bag_data.py runner/tests/test_embedding_bag_packet.py
```

**136 passed, zero skips, exit0**, `.local/prepared-bag-data-green-2.log`:62 prepared-helper cases plus74 previously implemented packet cases. New assertions are supplementary verification, not a separately claimed RED cycle. The joint packet matrix includes actual CPU/MPS source collation; prepared conversion itself is intentionally CPU.

Independent literal fixtures assert exact ordered ID/weight rows, packed offsets with/without terminal, zero-length preservation, binary and continuous target rows, source/fitted-state identities and mapping-order changes. Negatives cover exact config/schema/types/key sets, duplicates/absent/overlapping columns, malformed IDs/lengths/weights even in unused positions, table-pin/lineage/fitted-state/role tampering, binary class loss, conflicting objective input order, objective target dtype/output geometry, missing/nonfinite values, detached metadata and rejection before tensor allocation. These are in-memory pinned table tests, **not a completed-inventory integration, actual training, export, History or native lesson proof**.

Final source SHA256:

- `runner/prepared_bag_data.py`: `09cd9a84de5e715ed8bcb54ecb7302c13d4370c1664b584bbb529accf7c96ccc`
- `runner/tests/test_prepared_bag_data.py`: `0459cda15f2ef17bb1375fec03a0b63def41663ed452ba1bb6f4989e7fae49ce`

Local log SHA256:

- RED: `cffd00b27112bcacfa25a44b18a626b9e121e9933bcd1555703a8cded324dbbb`
- Final GREEN: `7a9e79cfd99e6ae7001b1b8a4f3f7f405f96bc4be7ff0f80891cb1806033816e`

Remaining: root must freeze actual captured mapping transport, verify real completed preparation inventory and paired partition lineage, integrate source/packet shape domains and per-bag slicing/device movement, then actual layer execution/gradients/training/checkpoints/export and configurable native examples/lessons. This helper alone confers no capability coverage credit.
