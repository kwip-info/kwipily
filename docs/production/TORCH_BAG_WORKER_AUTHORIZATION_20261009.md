# Managed bag mapping admission — 2026-10-09

Status: bounded Python authority implementation verified. This does not accept
the complete EmbeddingBag bundle, publish registry entries or establish native
application/UAT acceptance. Frozen transport contract:
`TORCH_BAG_TRANSPORT_REVIEW_20261009.md`; preceding CLI receipt:
`TORCH_BAG_CLI_20261009.md`.

## Ownership and finite result

Initial lease: `runner/worker_authorization.py`, `runner/worker_input_contract.py`,
new `runner/tests/test_bag_worker_authorization.py` and this receipt. Parent
explicitly extended it to `runner/model_training_operation_contract.py`, a narrow
phase-verifier call in `runner/operation_authorization.py`, and a bag-only
captured/runtime registry parity check in `runner/worker_bootstrap.py`. No engine,
main, Core, native App, archive or authorization schema files were edited here.

Managed training admits exactly one bounded canonical bag mapping flag. Its
schema, UTF-8 encoding, duplicate-key refusal, source variants, capacities,
ordered/disjoint columns, binary/objective target distinction and preparation
pins match the post-guard mapping contract. Ordinary feature/CSV/split controls
cannot accompany it. Preclaim mapping validation is framework-free in
`validate_bag_arguments`; explicit fixed/packed × weighted/unweighted parity
checks compare it against `bag_input_mapping.decode_mapping_json`.

`verify_bag_graph_inputs` verifies bounded captured graph/registry identities,
one actual bag source, exact four source controls and every actual consumer's
configured role wires/controls. It keeps each consumer's own configuration and
allows different vocabulary sizes, widths and valid reduction modes. It checks
the captured source expanded graph when supplied. It deliberately does **not**
claim to re-expand composite source, infer complete tensor semantics or allocate
a model before the lifetime guard. Those remain the independent main/engine
checks after the guard and before table conversion/training.

## Two distinct managed input routes

**Workflow phases** retain their existing `workflow-runs/<project>/<directory>`
layout. Both experiment `LaunchAuthority` and ordinary phase
`OperationAuthority` call the shared `verify_phase_bag_inputs` after their
existing retained path/owner checks. Actual bag graphs with an omitted mapping
also refuse; ordinary absent-flag routes retain their prior diagnostics and
payload semantics.

The phase verifier binds:

- `training/request.json` to `capture.json.plan.modelGraph`;
- capture workflow12/compiler11/format7 and project/directory/run/request IDs to
  the retained phase authority;
- canonical mapping to `plan.bagInput`, selected preparation endpoints to
  `plan.training`/`validation`, and binary target or existing objective
  mapping/recipe/scalarization to the corresponding plan fields;
- capture-pinned root `model-registry.json` and `runtime-evidence.json` to the
  actual `--registry` bytes and `workerFiles['components.json']` pin;
- selected `training/composite-source.json` to the CLI hash and captured
  `plan.modelProvenance`;
- `preparation/artifacts/manifest.json` to the exact manifest/graph pins,
  completed status and private bounded regular artifact inventory.

Capture identity remains fixed through authority rechecks. Shared pinned-file
checks retain descriptor/signature/hash protections. Runtime registry reads
allow root-owned installed files while still requiring bounded regular,
non-writable, stable bytes and their captured runtime hash. Capture ownership
rules remain strict for owner-published inputs. Full native archive decoding and
compiler plan recomputation are separate parent-owned gates; the test capture
fixtures intentionally contain the finite fields this Python seam consumes.

**Standalone modelTraining operations** retain `input/graph.json`,
`input/preparation/*`, optional `input/composite/source.json` and `artifacts`.
Bag operations additionally require captured `input/model-registry.json` in the
exact pinned input inventory. They use the existing prepared partition selection
and objective route rather than dense feature mapping. All input pins are
verified before bag role checks; completed inventory remains verified before
execution. No arbitrary `--registry` argument is admitted into the standalone
request. The existing trusted bootstrap injects the verified runtime registry.

Before that injection, the narrow bag-only `worker_bootstrap._arguments` hook
requires the captured registry to match `RuntimeContext.registry`. Formatting
and equivalent integer/decimal representations can differ; booleans cannot
impersonate numbers. Missing/foreign runtime registries refuse before model
allocation. Legacy standalone and phase argument bytes/routes are unchanged.

## Actual evidence and corrections

Runtime: Python3.13.6/Torch2.14.1, CPU and MPS available. Meaningful initial RED
was run before implementation:

```sh
.venv/bin/python -m pytest -q runner/tests/test_bag_worker_authorization.py
```

**3 failed, 15 passed in 0.94s**, exit1. Both actual retained workflow phase
claim cases and fresh-process basic argument admission failed because the bag
option was absent. The15 negative cases initially passed via generic flag
refusal; they are not claimed as independent semantic RED evidence.

First implementation replay: **17 passed, 1 failed in 3.57s**, exit1. It exposed
an existing unconditional `validate_gradient_arguments` import of
`trainability.py`, which imports Torch even with no selected controls. An
approved no-controls early return now avoids that unused dependency while
preserving every selected-control validation. Corrected replay: **18 passed in
0.88s**, exit0; a fresh subprocess proves basic argument validation does not
import Torch.

Additional standalone/objective/source-ordering verification initially reported
**28 passed, 4 failed in 1.13s** because the CLI-derived standalone fixture lacked
the mandatory explicit seed and learning rate; main's defaults are insufficient
for managed requests. Correcting that fixture produced **32 passed in1.10s**,
exit0. These later assertions are verification, not new initial RED claims.

An overly broad regression selection included existing native watchdog fixture
consumers. It compiled temporary C test dylibs and was intentionally stopped:
**174 passed before KeyboardInterrupt in16.02s**, exit2. This is not a completed
GREEN receipt. No Swift/native App/UI build or edits occurred. The replacement
Python-only selection completed **169 passed,3 deselected in21.47s**, exit0.
Final registry type-parity controls then exposed one test collection typo; the
collection failed, exit2, and was corrected before the sealed replay.

Final frozen-source command:

```sh
.venv/bin/python -m pytest -q runner/tests/test_bag_worker_authorization.py runner/tests/test_worker_authorization.py runner/tests/test_operation_authorization.py runner/tests/test_model_training_operation.py runner/tests/test_bag_cli.py -k 'not actual_guarded'
```

**175 passed, 3 deselected in24.74s**, exit0. The3 deliberately deselected tests
are the existing actual guarded training tests requiring native fixture builds;
no test skipped or failed in this final selection. Owned-file `git diff --check`
also passed. This includes39 scoped bag authority checks plus the preceding
real CPU/MPS CLI training suite and existing retained authority/training cases.

The actual operation/experiment claims exercise valid inputs and rechecks;
wrong roles, missing/changed mappings, wrong source controls, foreign pins,
registry changes, capture version/graph differences and target overlap refuse
before claim/publication. Existing objective phase recipe/mapping/scalarization
is bound to capture. Standalone claims verify exact inventories and bootstrap
registry parity. A rehashed forged source body with truthful byte/expanded-graph
identity is intentionally accepted by the limited preclaim checks, then actual
main independently refuses it with the source-expansion mismatch diagnostic.
Table-loader and training spies prove neither was entered; no output directory
was published. That test restores process signal handlers afterward.

The registry fixture uses published entries when available and adds only absent
reviewed bag roles. It does not substitute for parent-owned publication or a
full native managed execution/capture acceptance run.

## Frozen files

| File | SHA-256 |
| --- | --- |
| `runner/worker_authorization.py` | `0d615deffbc570df5a1264d23a65d4d33e6fd259970b8133a3c10ee7355b934e` |
| `runner/worker_input_contract.py` | `f5c74f1a9187063064f8d39fd7e73825cb3e9bf6e1061ccd3e8b07bc6d57113e` |
| `runner/model_training_operation_contract.py` | `fa6b19f5e8e97e4baf21237f02d981a518c83710bb73ad16f1c31abe66843ae0` |
| `runner/operation_authorization.py` | `5aeda1db7190106a41e722074e1817a9feb6cdb4aae2cd3c0b80389dd9aab287` |
| `runner/worker_bootstrap.py` | `01720cca94e1207ca19c91618d09fef1dfd0deed4a6fad0191eb5dc505d7a2ff` |
| `runner/tests/test_bag_worker_authorization.py` | `3a5cfe8ede8f751962a4655c20e607c6ade08a038ba6c3764f7ef49debd3e8f5` |
| `.local/bag-worker-red.log` | `639ec5c617d7169c6f08ac9b763d188211b1dc4f48706717068f3eb0109ce53b` |
| `.local/bag-worker-green-sealed-2.log` | `1a5c4660fe7e7574f2dd6bef98cb2b7fa39c484d00bb6aa13f42807f5d4bc80e` |

## Remaining boundaries

Root serialized replay subsequently included the three native watchdog fixture
cases with the same five-file test selection and no `-k` exclusion:
**178 passed in25.98s**, exit0 (`.local/bag-worker-authority-root-full.log`).
This closes the previously excluded automated authority checks; it does not
establish an application-window run or complete bundle acceptance.

Parent-owned native12/11/7 capture/writer/argument/readers, complete archive
recomputation, actual native managed execution and final bundle acceptance remain
separate. The pre-existing selected-trainability validation path can still import
Torch before guard; this slice only fixes the no-controls import and does not
claim a wider trainability dependency refactor. No completion percentage or
capability credit changes are made by this receipt.
