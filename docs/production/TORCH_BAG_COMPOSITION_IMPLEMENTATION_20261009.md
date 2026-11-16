# Bag composition worker integration — 2026-10-09

Scoped worker implementation, not bundle acceptance. Owned production changes: only engine composition validation/forward/container/export regions, module_composition_plan.py and module_composition_source.py. Root owns engine.run and transport changes; those current bytes are included in the whole-engine hash. NEW test_bag_composition.py and this receipt are owned. Core/App/main, discovery, registry, catalog and old acceptance receipts were not edited. The review's version clarification now crosslinks [transport review](TORCH_BAG_TRANSPORT_REVIEW_20261009.md): source3/expansion2/Graph2/plan1/policy1 composition wire is unchanged; new bag mapping uses Workflow12/compiler11/capture7 while legacy11/10/6 remains supported.

## Actual boundary and native execution

Bag input is the single root source sentinel and is excluded from native compute registration. Nested source sentinels are explicitly refused by independent source expansion. The existing root plan input keeps actual sourceID and all outgoing target endpoints. Its value is the real named-role bundle from execute_bag_source. Only this root boundary may contain distinct source ports; every address must reference an enabled role of the same actual source. Every nested boundary retains the original single external (node,port) identity rule.

GraphModel executes the source once and retains its real dictionary in the invocation-local frame. The root binds each target through its actual flat edge.sourcePort; it never broadcasts the dictionary into a tensor port. Source indices/offsets/item weights retain object/autograd identity and configured ordering. Native Graph/List/Map children receive their exact named role tensors. Real CheckedEmbeddingBag then validates/reconstructs its coherent packet, reduces coordinates to normal float32 features and owns the sole native `.operation.weight` path. No aliases, duplicate parameter registration or additional table allocation were introduced. native_ownership.py is unchanged.

Frozen unary Sequential semantics are unchanged. Fixed unweighted EmbeddingBag can participate directly as a unary child. Packed/weighted bags use real multiport Graph/List/Map boundaries, then a genuine native Sequential on reduced ordinary values. Direct multi-input Sequential and nested combined-role boundary aliases refuse before allocation. No blanket bag-composition refusal remains.

## Portable hierarchy

Composition export reuses the bounded standalone bag helpers and real native container constructor. It recognizes the reviewed bag forward signature and exposes raw `(indices, offsets, per_sample_weights)` arguments, or its own generated packet, with normal ambiguity refusal. It coerces once before source execution, preserves named inspection and exact native state keys, and requires no installed runner imports. Legacy ordinary exports remove the bag-only dispatch source before emission, retaining their exact existing text. No change to ordinary forward arguments or hierarchy keys follows from the new bag path.

## Actual RED/GREEN and finite proof

Actual initial semantic RED: **16 failed,8 passed in1.07s**, `.local/bag-composition-red.log`. Real constructor/execution/export/source-reexpansion positives hit the existing explicit refusal; malformed cases already refused. No missing-import or fabricated native evidence was used.

The first implementation passed24 new checks, then two private-table fanout cases passed (26 total). The final28 new cases additionally check exported full/microbatch values, all exported table/head/source-weight derivatives, and frozen table input influence on CPU/MPS. Literal table/output/whole-gradient arithmetic is independent of runtime adapters. A real named source fans out to two distinct tables, registration order differs from DAG order, and a native Sequential hook proves the downstream chain executes exactly once. All saved state keys contain one `.operation` boundary and strict reload preserves actual predictions. A `python -I` subprocess outside the checkout loads only actual emitted source and saved weights, checks literal outputs and verifies runner modules were not imported.

Final command (recorded exact executable/arguments in `.local/bag-composition-final-pins.json`):

```sh
.venv/bin/python -m pytest -q runner/tests/test_bag_composition.py runner/tests/test_bag_export.py runner/tests/test_bag_source_graph.py runner/tests/test_module_composition_execution.py runner/tests/test_module_composition_export.py runner/tests/test_module_composition_lifecycle.py runner/tests/test_module_composition_source.py runner/tests/test_composite_source.py runner/tests/test_composite_fields.py runner/tests/test_composite_integer_products.py
```

Actual terminal **331 passed,0 deselected,0 skipped,0 warnings in11.96s**, exit0. Before/after snapshots for all selected source/tests: **sourceDrift=[]**. The explicitly leased old graph assertion was corrected after its actual **1 failed,41 deselected in0.60s** RED (`.local/bag-composition-old-refusal-red.log`): malformed compositionPlan={} still refuses, now with exact `Composition plan has missing or unknown fields.` rather than the removed temporary blanket-refusal message. Its real compiled export/inference positive is retained. No supported path is deselected. All eight new allocation-protected malformed boundary cases run, including malformed source roles, missing/duplicated root targets, internal source registration, nested role combination and packed multi-input Sequential.

Independent literal source3 documents exercise graph/list/map re-expansion using a real registry file copied from shipped metadata with explicit injected reviewed bag declarations (publication remains pending). Rehashed edited plan targets and nested bag_input sources refuse. These are worker-owned literal source fixtures; **genuine Core-emitted source capture replay is still pending root** and is not claimed by these checks. Existing CPU exact/MPS weights-only composition lifecycle tests are in the union; new bag-specific full-engine continuation remains root's typed prepared transport evidence, not claimed as a new end-to-end proof here.

Separate local proof `.local/bag-composition-legacy-proof.json` loads actual engine source from baseline5cb847a0322950fe3a3ebe05550888cab3b2c788 in a distinct file so source inspection uses genuine baseline declarations. All four ordinary native container kinds retain byte-identical generated exports, seeded state keys/values and forward results. No legacy acceptance SHA refresh was performed.

| Frozen current worker/test source | SHA256 |
|---|---|
| `runner/engine.py` | `c95567c85894a43f00857469b798eb8a5f416c55acc8c4d723b94fbee1a4a4fb` |
| `runner/module_composition_plan.py` | `9df3635aa4e86374506bc79c6b533437a59a2d37db2aaa569e0a61d061efac3f` |
| `runner/module_composition_source.py` | `75ca33f624c6a56f2ee81d5425534bb9122edb118a2fd451f16a114ef9cf91a9` |
| `runner/tests/test_bag_composition.py` | `f0a83a54908fa100074db66ca93429a239e8cf414f1fbceea3a7c56b2aafda5a` |

| Actual evidence | SHA256 |
|---|---|
| `.local/bag-composition-red.log` | `6dc0cf99a045d165ff2156a72df48779f83d88d906bb501f2dda996cb36b8819` |
| `.local/bag-composition-final-green.log` | `517daebabba6748dcff465986fc4c1c7ee3a75c0438ee63de54e834ccd6eed56` |
| `.local/bag-composition-legacy-proof.json` | `81d525dc992668c38e8adf366b03cf9b112aaf31dc1939274f3f599240a5b16c` |

Source and numerical helper admission limits, CPU-only maximum norm and trainable-frequency MPS refusal remain unchanged. No new optimizer/sparse backend parity, native UI, package or final bundle acceptance is claimed. Root's matching Core expansion, immutable transport/native owner verification and genuine captured-source proof are remaining consumer gates.
