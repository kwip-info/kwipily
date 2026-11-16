# Bag requested-gradient worker seam — 2026-10-09

Status: implemented and Python verified; engine launch, native recipes/readers and product admission remain separate integration work. This receipt makes no native UAT or complete-bundle claim.

## Contract

`analyze` and `analyze_requested_gradients` retain their call signatures. Native bag analysis derives the actual source contract through `ProductExecutionAdapter`; it does not infer an observation-first sample shape for coordinates. Ordinary artifact schema1 is unchanged. Bag artifact schema2 adds `modelInput` with the adapter's exact schema1 bag packet contract, and each request sample adds `bagItemCounts` for its selected whole observations.

`preflight_requested_gradients(model, prepared_inputs, policy, *, partition_row_ids, device, derivative_capture=None)` returns a list of `(request, selected_batch, selected_row_ids)` tuples. It performs no forward. It verifies full partition source controls/roles and device, whole-bag selection and rebased offsets, row-ID count/range, actual ordinary float32 output and upstream geometry, native parameter devices, genuine source target roles, aliases and aggregate element/byte budgets. The final analysis calls this same seam. Root can call it before training with the selected-device prepared partitions.

Integer IDs/offsets report `integerInput`; actual floating per-sample weights receive autograd derivatives. Packed all-empty weights have actual shape `[0]`, observed empty values `[]`, and integer coordinates remain unavailable. Fixed coordinates retain `[bags, capacity]`; packed offsets retain `[bags + terminal]`. Native parameters retain existing owner resolution and limits, including the existing 200000-parameter inventory ceiling and 65536 returned-element ceiling. Existing isolation restores parameter values, original grad references/content, buffers, module modes and CPU/MPS RNG.

`gradient_policy_code(..., bag_input_contract=None)` accepts the actual adapter contract for bag exports. It validates exact source configuration/role metadata and imports the exported sibling `model.py` source validator, thereby using that sibling's packet class. It injects only frozen metadata, never fake model nodes. Ordinary helper source bodies remain retained under private legacy names and are emitted with their original names. All three emitted legacy bodies were independently compared byte-for-byte against HEAD; the ordinary selected-policy whole-helper SHA256 regression is `833e4541c0d594bb89f8e692e4d9ac0eda33e0db9b10d4be4a20ab8229ac9c27`.

## Actual evidence

- Semantic RED: `.local/bag-requested-gradients-red.log`, 5 failures before implementation, including CPU/MPS packet input, artifact binding and absent preflight.
- Final GREEN: `.local/bag-requested-gradients-final-green-3.log`, **155 passed**, 3 existing PyTorch warnings, 4.16 seconds. Both CPU and MPS are available and executed; no device skips in this replay.
- Command: `.venv/bin/python -m pytest runner/tests/test_bag_requested_gradients.py runner/tests/test_requested_gradients.py runner/tests/test_gradient_export.py runner/tests/test_bag_product_execution.py -q`.
- New tests include independent literal chain-rule results and a centered finite difference, fixed/packed signed upstreams with both terminal-offset modes, all-empty packed CPU/MPS derivatives, whole-bag ranges and hashed row selection, pre-forward role/geometry/device refusals, frozen/disconnected/recording distinctions, parameter budgets, exact state restoration on success/errors/cancellation, source-inspection forgery refusal, strict exported contracts, an isolated relocated sibling-model derivative subprocess, and ordinary helper byte parity.
- `git diff --check` passed for the owned implementation/test files.

An intermediate portable test failed because its exported fixture omitted `gradient_policy_recipe`, so the export correctly did not publish derivative owner metadata. The fixture now requests the actual gradient-enabled export; no synthetic `layer_keys` or packet class is supplied by the test. No engine, Core, App, registry or catalog files were changed for this worker slice.
