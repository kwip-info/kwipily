# Remaining activation runtime — 2026-10-08

Finite runtime scope: PReLU, LogSoftmax and Dropout under `TORCH_REMAINING_ACTIVATION_CLOSURE.md`. This receipt supplies worker evidence, not component acceptance or framework coverage credit. Root owns Core/App lessons, native verification, builds, catalogs and acceptance; authoring owns actual generated metadata; learning owns the independent canonical mask resource.

## Contracts and integration

New helper `runner/remaining_activation_configuration.py` provides `normalize_remaining_activation(operation,config)`, `resolve_remaining_activation(operation,config,layout)` and `capability(operation)`. Resolver returns canonical `config`, preserved `layout` and actual `parameterCount` (PReLU slope count, otherwise zero). Configuration accepts declared defaults only, refuses unknown fields, Boolean numbers, fractions for integer fields, nonfinite/out-of-range coefficients and invalid rank/observation axes. Integral JSON numeric counts/directions canonicalize to integers. All three operators require graph schema 2 and actual dense finite float32 rank 2–8 with exact authored nonempty sample layout; the logical 100M-element budget is checked before scans/allocation.

PReLU resolves one shared slope or one slope per first sample direction/channel, not trailing column. Every authored node owns its registered `nn.PReLU` module; fanout uses the same node's state. Actual created slopes count toward the existing two-million-parameter limit before module construction. Collision-safe ModuleDict keys are preserved through export/checkpoint. Input influence at zero is the slope, parameter influence at zero is zero, matching the installed CPU/MPS convention. Learned slopes must remain finite and device/dtype-compatible with input. Result overflow refuses visibly.

LogSoftmax normalizes a selected sample direction and returns stable logarithmic probabilities, with no new learned parameters or mutation. Observation direction and rank-invalid aliases refuse. Finite inputs that nevertheless produce nonfinite float32 arithmetic refuse.

Dropout is a registered `nn.Dropout`, so parent train/eval controls both worker and exported model. Training draws fresh masks and scales surviving values; p=0 and p=1 retain actual endpoint behavior. Evaluation is identity and does not consume random draws. Execution never reseeds in forward. Reproducibility is same-runtime/device/dtype/shape/seed/call schedule only; CPU/MPS masks are not claimed equal.

Portable source includes the bounded execution guard and actual module constructors, with strict state-key compatibility and no runner imports. Discovery, authored runtime contracts and generated callable capabilities match exactly for all three IDs; descriptor version is 1.2.0. Existing activation descriptors remain separate.

## Checkpoint behavior

Only graphs containing PReLU add actual parameter-node registration order to their checkpoint schedule; this includes every existing learned family and the new slopes. Graphs containing Dropout additionally bind actual stochastic-node execution order, independent of sorted semantic topology. Actual reordered two-branch PReLU/Dropout graphs preserve semantic graph digest but refuse exact resume when their registration/draw order differs.

CPU epoch-boundary resume restores actual optimizer, model and random-generator state; MPS remains explicitly `weightsOnly`. Fine-tuning restores learned state and resets optimizer/random/epoch history under the existing lineage policy. No old contract fields were rewritten. A separately executed old-graph serialization comparison against `HEAD:runner/checkpoint_lifecycle.py` produced identical bytes, SHA-256 `e10259d340720fa09dd0c50b0a44191f0ed6aae54fcbf6a0d7e31625a697f4ad` on the recorded host fixture. Local comparison: `.local/remaining-activation-legacy-contract-comparison.json`.

## Meaningful RED/GREEN

Actual host Python 3.13.6 / PyTorch 2.14.1 / macOS 26.6 arm64, MPS available. Initial real graph tests before implementation: **7 failed / 9 passed** (the passes are expected strict-refusal tests); local log `.local/remaining-activation-runtime-red.log`. No fake registry metadata was used. Expanded graph checks while actual generation was pending failed on absent new operators and were not treated as acceptance.

After real production registry42 generation, first full run was **95 passed / 2 failed**: both failures were zero-initial-slope expected tensors inferred as int64. Only those independent expected tensors were explicitly typed float32; no production behavior or acceptance tolerance changed.

Final actual command:

```text
.venv/bin/python -m pytest runner/tests/test_remaining_activation_runtime.py -q
```

Result: **97 passed in 1.20s**, zero skips/failures/warnings. Actual terminal log `.local/remaining-activation-runtime-green.log`. Scoped diff whitespace check passed.

Evidence includes:

- Independent scalar PReLU outputs and mixed-sign weighted input/slope influence for shared/channel slopes, negative/zero initial slopes, exact-zero inputs, noncontiguous arrays and actual rank-eight execution.
- Independent stable scalar LogSoftmax values and weighted influence on signed sample directions with large shifts and noncontiguous arrays.
- All strict configuration, shape/schema/dtype/sparse/empty/nonfinite/resource negatives; count4096 boundary; actual PReLU slopes crossing the total model parameter budget refuse before construction.
- CPU/MPS actual parent train/eval, seeded repeat and successive-call advancement, p=0/.5/1 endpoints and output/weighted influence. Upstream ones masks are generated separately from measured adapter results for device-specific endpoint/call-schedule checks.
- Independent frozen canonical mask bank, captured directly from upstream CPU float64 Dropout, verifies actual CPU float32 adapter output and scalar weighted influence for seeds 7/41/173/911 and observation counts 2/3/4. The expected masks are never inferred from measured adapter output. Resource: `runner/fixtures/remaining_activation_dropout_masks_v1.json`, SHA-256 `5a96044ae7f6286b354d6cb43cd6bec4e040a15b55bffc01ecaa57bbc1056538`. This proves parity on the recorded runtime, not across versions/devices.
- Actual portable full graph CPU/MPS train/eval value/state parity; strict wrong-slope-count state refusal; fresh initialization, independent-node/fanout ownership and collision-safe keys.
- Actual CPU uninterrupted versus resumed training for PReLU and Dropout: model/optimizer/RNG states and metric bytes match exactly. Actual MPS saved contracts remain weights-only. Parameter gradient capture includes the PReLU weight inventory; CPU/MPS fine-tuning resets epoch/seed/optimizer and preserves finite state.
- Exact descriptor equality across authored bundle, generated table and runtime discovery.

Applicable prior runtime plus composition regression: **1022 passed**, zero skips, one existing MPS `embedding_renorm_` CPU-fallback warning. Local log `.local/remaining-activation-prior-runtime-green.log`. This covers runner/layout/multi-output/checkpoint/preview/frozen/embedding/configurable-activation/pooling and the two accepted composition test files. Root separately owns any full accepted-catalog-bound replay and source-binding refresh.

## Explicit limits

No scalar, sparse/ragged, integer or float64 production execution; no observation-axis normalization, in-place switch, per-node seed/device/mode override or cross-device mask guarantee. Learning CPU float64 measurement remains a separate path. Native UI/lesson state ownership, private/cold receipt behavior, generated artwork, prepared examples, clean-machine distribution and complete accepted-unit replay are not established by these worker tests. This runtime slice has no known blocker to its finite checks; broader release acceptance remains root-owned.

## Scoped App runtime admission — root-serialized checks

A later root assignment added App runtime admission only in `CapabilityActions.swift`, the new `WorkspaceStore.remainingActivationRuntimeCapabilities` property and the selected-model/queued gate in `WorkflowWorkspaceSession.swift`, plus `RemainingActivationRuntimeAdmissionTests.swift`. New optional argument `remainingActivationDescriptors` defaults to an empty dictionary. Absent or mismatched descriptors never grant support from palette presence; component counts include only exact verified members. Every refresh, failed installation and discovery failure revokes prior new-activation support. Actual worker discovery parses only the three selected descriptor identities. Existing operator defaults/support remain separate.

Tests author operator descriptors as independent literal JSON, rather than encoding the Core bridge or bundle runtime contract under test. The selected compiled graph is checked before source capture and again before queued dispatch; unrelated disconnected canvas components do not block that graph. Queue revocation releases reserved storage/staged run sources and prevents launch, including experiments.

The first setup attempt failed before admission because experiment configuration was added before migrating prepared workflow fixtures to version 4; the fixture was corrected to the existing control schema. A subsequent attempt exposed a real Core descriptor lifecycle issue before admission; root added the existing training-partition-only boundary to its owned new descriptor bridge. Neither setup outcome is counted as meaningful admission RED, and no production admission rule was weakened.

After those setup boundaries, actual semantic RED in `.local/remaining-activation-admission-boundary-red.log` completed **one method with 21 failed assertions, zero unexpected failures**: all three new operators incorrectly admitted training/experiments without optional descriptors. Implementation followed that recorded RED.

Root then executed the shared scoped checks serially. Actual saved `.local/remaining-activation-shared-green.log` shows **19 Core + 12 App = 31 passed**, zero failures/skips, including all **seven new App admission methods** (2.466 seconds). These verify missing/mismatched/exact descriptors; training/experiment capture; actual queued revocation and cleanup; unrelated disconnected nodes; partial discovery/counts; default-helper and failed-install revocation; and actual worker discovery. This agent inspected the terminal evidence but did not run Swift/build/native tools. Root-reported terminal completion is GREEN.

Shared GREEN log SHA-256: `d8a0519666b20154bf5480bef082e04a774ef322ea6a0277446fad77ccd2211c`.

No additional source edits were needed after that GREEN. App admission is a finite shared slice, not native lesson/UI or release acceptance.
