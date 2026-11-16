# TORCH-01 · Bilinear delivery contract

Status: scoped local delivery accepted 2026-10-07. Strict configuration/layout, runtime/discovery/authoring, numerical gradients/state/export/lifecycle, bundled prepared example, both interactive lesson modes and native journeys verified. Owner: Codex/root. Broader PyTorch coverage and production gates remain open.

## User-visible operation

A configurable learned interaction between two tensors: y[o] = sum(i,j, left[i] * weight[o,i,j] * right[j]) + optional bias[o]. Two named float32 input ports (`a`, `b`), one float32 output. User labels remain distinct from type names. Parameters: positive integer left width, right width, output width; Boolean bias. Preserve observation batch and any matching leading sample dimensions. Both inputs must have identical leading dimensions; no implicit broadcasting, flattening or cross-observation multiplication.

Bias-on and bias-off modes need shape, execution, gradient, state/export and UI evidence. Reject mismatched widths/prefixes, unsupported dtypes and parameter/output budgets before allocation. Parameter count is outputWidth × leftWidth × rightWidth plus optional outputWidth bias. Values should match an independent contraction and gradients must not couple observations. Noncontiguous inputs require a real fixture. CPU and MPS support must be tested and disclosed, not inferred from registry availability.

## Integration acceptance

- Shared registry/configuration and Swift/Python shape rules agree; malformed/missing/extra fields fail precisely. Ports expose each independent feature width.
- Native trainability validation accepts a Bilinear-only parameterized graph; the former Linear-only heuristic has been replaced without permitting parameterless training.
- Runner construction, generated transferable source, model state registration and checkpoint compatibility agree. Historical Linear checkpoint ordering/identity must not change. Audit the existing `linearNodeOrder` field before extending parameter schedules.
- Real universal prepared training, inference, checkpoint continuation/fine-tuning as supported, archived inspection and portable export/reload match accepted behavior.
- Bundle a two-input interaction example and learning material explaining the weight tensor, contraction, parameter growth, biases and gradients. Keep no-broadcast behavior explicit.
- Native palette/inspector/wiring/configuration/run/result verification follows automated acceptance. Record unsupported modes honestly; do not count the broader family complete from one unit.

## Accepted scope

The dated evidence below closes this bounded local component delivery. It does not establish full framework coverage, clean-machine distribution or the remaining production gate matrix.

## Foundation evidence · 2026-10-07

Python RED:12 failures/9 passes for missing configuration/layout support. Swift RED: missing BilinearConfiguration API. Implemented strict schema2 Python normalization and matching Swift configuration plus batch-preserving layout algebra. Independent width fields accept integral JSON numbers but reject Booleans/fractions; exact2M parameter boundary and output-size bounds covered. Python45tests pass including existing matrix multiplication regressions (`.local/bilinear-contract-green.log`). Swift results in `.local/bilinear-swift-green.log`. Initial Swift test string literal required explicit JSONValue.string; corrected before successful verification. No claim of running Bilinear models or exposing a palette entry yet.

## Graph and execution evidence · 2026-10-07

Private-registry numerical integration verifies explicit contraction values/gradients for both bias modes, independent widths, noncontiguous inputs, shared-source inputs, observation isolation, higher-rank prefixes, safe ModuleDict identities and standalone source/state reload. CPU/MPS forward-backward parity passes on this host. Frozen parameter state now remains within the total parameter ceiling but does not satisfy trainability; native and runner rules agree.

Actual CPU training verifies exact resume model/optimizer/RNG/metrics, rank3 gradient inventory, source/checkpoint reload, cached inference, incompatible-bias refusal and fine-tuning while retaining the parent. Existing checkpoint schema and linearNodeOrder remain unchanged; semantic architecture identity and deterministic model registration already cover the Bilinear operation. Existing lifecycle regressions pass. These checks do not establish packaged discovery, universal example or palette/UI delivery.

## Runtime discovery evidence · 2026-10-07

Worker advertises an exact versioned Bilinear descriptor including schema, dtype, independent feature bounds, parameter ceiling, sample-rank/element limits, bias modes and no-broadcast policy. Native decoding is bounded and exact; missing or incompatible optional support refuses Bilinear graphs without invalidating legacy support. Every runtime reset/error path clears cached Bilinear availability. Actual worker refresh installs the descriptor. Shared registry/palette remains unchanged pending native component/bridge, example and lesson integration.

## Native authoring source integration · 2026-10-07

Shared registry now includes Bilinear under Layers, with independently named input widths, output width and bias controls. The strict capability bridge describes named trailing feature dimensions, learned state and the matching adapter. Empty lesson/example reference lists intentionally remain until those assets ship. Connection commands reject known width/type/prefix mismatches; unresolved draft shapes remain visible. Theme follows the rose Layers family with a two-input interaction glyph and explanatory help. Source tests pass; actual rendered native acceptance remains required.

## Example and independent teaching oracle · 2026-10-07

PreparedWorkflowExample.bilinear creates a real universal preparation/training flow using bundled observations and independent3/5-feature branches. Learning library includes an Open bilinear example action and parameter-count explanation. Actual host integration generates nn.Bilinear source before execution, trains, verifies saved history and inspects captured data. Layout tests confirm separated initial cards and forward edges; existing persisted layouts are untouched.

BilinearLessonReference is a bounded scalar oracle independent of PyTorch and learner output. It computes outputs and gradients for left/right/weights/optional bias, summing shared parameter uses without mixing observation input gradients. Hand-calculated66output and exact gradients, higher-rank prefix/zero upstream isolation, malformed shapes/bias and no-broadcast rejection pass. The oracle is not yet a user-facing interactive assessment; task/submission/worker/UI integration remains required.

## Assessment plan foundation · 2026-10-07

Both bias modes now have seeded, versioned assessment plans with four independent fixtures and separate output/input/parameter gradient criteria. Fixture dimensions vary and preserve exact sample prefixes. Central finite differences validate every scalar gradient of the teaching reference. Five targeted Swift tests pass with no skips or failures (`.local/bilinear-assessment-green.log`); missing-API RED is recorded separately. Strict learner submission, measured worker execution and interactive UI remain outstanding.

## Measured learner graph and persisted replay · 2026-10-07

Strict submissions now execute the actual authored graph through float64 PyTorch Bilinear/autograd, including optional protected-axis layout transforms. Gradients refer to original fixtures even when an authored transform changes routing. Bias policy, exact prefixes, bounds, cancellation and malformed graphs are checked before allocation where applicable. Shared drafts recover incomplete graphs; canonical receipts bind submission bytes, task/version, seed, fixtures and measured gradients. Real Swift-to-Python grading passes both modes and multiple seeds, with saved receipt reload. 36Python and37Swift targeted/regression tests pass with no skips/failures. Interactive canvas/readout integration and packaged native verification remain outstanding.

## Interactive lesson source integration · 2026-10-07

Both bias modes appear in Learn and use the shared canvas, draft history and measured-session lifecycle. A dedicated cached shape projection keeps parameter tensors unbatched while rendering observation axes symbolically. Shared weights/bias and all gradient categories are inspectable in formative mode; private assessment fixtures remain hidden. Runtime absence refuses submission with an actionable message. 44targetedSwift tests pass0skip/fail, including actual grading, persisted reopening and matrix regressions. Final UI build passes. Packaged rendered/native acceptance remains open; source tests do not close it.

## Packaged bias-free native journey · 2026-10-07

Candidate a4b5b78 built from current verified worker payload. Palette-added and pointer-wired bias-free construction passed actual packaged assessment and saved its result. Bottom Results displayed shared[2,2,3]weight gradients without observation slicing, supplied weights and all contributing input/upstream tensors. Assessment mode hid formative values. Native-found unconnected weight-port label corrected with a failing/passing regression; rebuild/replay pending. Bias-enabled lesson and production example/run/export native acceptance remain.

## Scoped native closure · 2026-10-07

Candidate4f92b34 replay confirms unconnected weight/bias ports declare sharedparameters without a batch axis. Manually palette-added and pointer-wired the bias-enabled6node construction; all5criteria passed with zero error, including10biasgradient values; verified result saved. The earlier bias-free journey covers the other mode and formative/privacy readouts.

Native bundled example generated nn.Bilinear(3,5,2,bias=True) before running. Prepared workflow C5DE0BD0-DFE2-4AF9-9670-F09A58BFC7FD completed3epochs, validationloss0.57887/75% fixture accuracy,29archived artifacts; History verified captured evidence. Native export dialog saved `.local/KwipML-model-C5DE0BD0`; its standalone inference helper loaded the trained checkpoint outside the app and produced2finite logit rows/classindices. Native Learn bias toggle changed current generated source to bias=False; Undo restoredTrue and Save retained the restored example. Candidate quit.

Bridge now references both delivered lesson IDs and prepared-bilinear-classifier; missing-reference RED and4testGREEN in `.local/bilinear-links-{red,green}.log`. This metadata change postdates the accepted candidate. Bilinear is closed as a scoped local unit; remaining linear-family LazyLinear and wider PyTorch inventory remain open.
