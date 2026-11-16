# EmbeddingBag configuration foundation — 2026-10-09

Scoped configuration proof only; no bundle acceptance, native device execution, table allocation, runtime numerical proof, state/export proof or catalog credit. The original EmbeddingBag four modes remain pending their separate integration evidence. Root froze the helper contract before implementation; field names were coordinated with the packet owner.

Files: NEW `runner/embedding_bag_configuration.py`, `runner/tests/test_embedding_bag_configuration.py`, and this receipt only. No engine, registry, generator, Swift, catalog or old receipt edits.

## Exact interface and configuration

Component identity proposed/frozen for integration: `embedding_bag`, configuration1. The helper validates an exact complete configuration object; it does not create an alternative recipe envelope or advertise a capability. Callers provide defaults during authoring, then pass all fields. `normalize_embedding_bag(config)` returns a fresh canonical dictionary without mutating its input. `configured_input_ports(config)` first validates that same configuration and returns an ordered tuple of real port names.

| Field | Default | Admission |
|---|---|---|
| num_embeddings | 16 | integral numeric 1…100000000; Boolean refused; canonical int |
| embedding_dim | 4 | integral numeric 1…floor(100000000/num_embeddings); canonical int |
| mode | mean | exact mean/sum/max |
| padding_idx | null | null or integral -num_embeddings…num_embeddings-1; negative maps to native nonnegative row |
| max_norm | null | null or positive finite value that remains nonzero/finite in float32 |
| norm_type | 2.0 | positive finite value that remains nonzero/finite in float32 |
| scale_grad_by_freq | false | exact Boolean |
| sparse | false | exact Boolean |
| include_last_offset | false | exact Boolean |
| freeze | false | exact Boolean |
| layout | fixed | exact fixed/packed |
| per_sample_weights | false | exact Boolean, enabling actual floating input |

Numeric positive maximum is3.4028234663852886e38. NaN, infinity, nonpositive numbers, bool-as-number and values rounding to float32 zero refuse. Numeric controls retain their finite Python float value after admission, rather than silently rounding caller controls. Defaults are not silently inserted by the strict normalizer. Unknown or missing fields refuse.

Inputs: fixed unweighted `(indices,)`; fixed weighted `(indices,per_sample_weights)`; packed unweighted `(indices,offsets)`; packed weighted `(indices,offsets,per_sample_weights)`. The configuration Boolean `per_sample_weights` is distinct from the actual packet tensor under the same field name. Packet schemaVersion1 identity, real dtype/rank/value checks, shape/bag/item/output budgets and observed weights-presence matching belong to the separate packet/runtime seam. Output `out` is the intended integration interface; no forward adapter is implemented here.

Max mode refuses sparse or frequency-scaled gradients. Per-item weights require sum. Sparse plus frequency scaling refuses because its native backward is unsupported. These combination checks remain strict even when freeze is true; the helper never declares a backward-invalid training configuration supported by relying on a dormant freeze switch. Valid sparse sum/mean configuration is retained, without granting an optimizer or device. Fixed layout retains include_last_offset=true in canonical configuration; installed native fixed-row forward ignores that flag. It is not silently rewritten or treated as a packed offset port.

The100M table-product bound is existing static shape algebra only, checked without allocating a table. **Shared aggregate2000000 model-parameter admission is required before native table construction**, including all other model parameters. Selected initialization separately imposes its200000 whole-model-parameter limit; it is not the universal graph limit. Packet output limits are separate. No huge allocation probe was performed.

## Actual TDD

Host command for both runs: `.venv/bin/python -m pytest -q runner/tests/test_embedding_bag_configuration.py`.

The initial importable implementation copied the supplied dictionary unchanged and exposed only indices. Actual RED: **63 failed,18 passed in0.12s**, log `.local/embedding-bag-configuration-red.log`. Failures measured missing normalization, strict key/type/range/interaction refusal and packed/weighted input availability; this was not a missing-import failure.

Minimal Torch-free implementation then passed **81 tests in0.04s**, zero skips/warnings, terminal success, log `.local/embedding-bag-configuration-green.log`. Tests cover both layouts/three reductions, all four input variants, negative/positive padding canonicalization, dimensions/norm numeric normalization, every required field, unknown object/keys, finite float32 bounds, static product boundary without allocation, exact switches and native-invalid combinations. Input immutability and valid dormant terminal/freeze/sparse settings are checked.

The helper imports only stdlib math/struct. Existing Embedding configuration served as the validation pattern; its native execution/device/fallback claims were not inherited. Installed PyTorch functional source was read to corroborate fixed-row terminal-ignore, max sparse/frequency refusal and sum-only weighting. Native forward/backward was not executed in this slice.

## Frozen pins

| File/log | SHA256 |
|---|---|
| runner/embedding_bag_configuration.py | d101c2b882c7435cb8a761fec16b626de532dde2fbded7ce97d439097e89f1d6 |
| runner/tests/test_embedding_bag_configuration.py | 846aa671184988c109ba30a2a3c361b1ba8c7b378432f31915384f931548404d |
| .local/embedding-bag-configuration-red.log | b4e3b0d18b45cadaa00ba2f912d54a6f935a9a8ae925118a7eeef41a3512a903 |
| .local/embedding-bag-configuration-green.log | cf1ecf0678da6a06242176e837d30c83990958fa91d23dc7c94a2d871164e644 |

Remaining: actual native adapter/device/control matrix; packet/prepared-data integration; genuine per-item-weight derivatives; sparse optimizer policy; parameter/state/initialization/trainability/gradient consumers; portable export and continuation; authored descriptor/bridge/discovery; lessons/examples/art/native checks and prior regression. This finite helper closure does not imply any of those gates passed.

Root current-source correction: engine general model limit is2000000; selected initialization limit is200000. The helper docstring and this receipt were corrected after GREEN without changing executable configuration logic or tests. The81-test result applies to identical executable logic.
