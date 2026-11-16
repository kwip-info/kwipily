# MaxPool1d delivery

Status: scoped MaxPool1d delivery accepted locally in candidate 3d86660 on 2026-10-07. Automated geometry/routing/training/export evidence and packaged native controls, Results, History, restart and child capture-status replay passed. Owner ART, scored pooling lessons and broad production/distribution acceptance remain separate.

## Contract and acceptance

First unit in the existing pooling inventory. Expose kernel size, nullable stride (null follows kernel), padding, dilation, ceiling mode and return indices. Preserve channels and use batched channels-first sample geometry [C,L]. No learned parameters. Native limits follow current convolution axis/channel bounds: C1–512, L/kernel/stride/dilation1–4096; padding0–floor(kernel/2). These are app limits, not claims about PyTorch limits.

Use [PyTorch2.14 MaxPool1d documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool1d.html) and installed2.14.1 behavior. Ceiling output must omit windows starting wholly in right padding. Signed floor division matters when the effective window exceeds the input. Padding behaves as negative infinity; gradients select winning input locations, not learned weights. Indices are int64, nondifferentiable input-axis locations.

1. Strict native/worker settings and shape parity, including null stride, dilation, floor/ceiling and invalid geometry.
2. Typed value/indices output routing through validation, execution, composites, inspection and code export. No silently discarded return_indices option or unsupported catalog coverage claim.
3. Independent window/argmax oracle, ties and gradient accumulation, actual CPU/MPS behavior, noncontiguous inputs and finite-output diagnostics. Test padded windows with no sampled input explicitly; do not confuse valid output geometry with finite values.
4. Actual classifier training, checkpoint/resume, cache, export/inference/reopen; parameterless layer still passes gradients to upstream learned layers.
5. Thematic component/controls, representative workflow and learning explanation of selection, receptive fields, shape arithmetic, information loss and piecewise gradients. Native editable controls, code-before-run, run/artifacts/restart.

## Original integration audit (resolved by evidence below)

runner/engine.py GraphModel currently stores values by nodeID and execute_node reads incoming nodeIDs. Although edges preserve sourcePort in the serialized graph, execution needs a shared per-port representation to carry both values and indices safely. Audit native GraphValidator and model projection, export, inspections and checkpoints together before exposing indices. This is a reusable model-output capability, not a MaxPool-specific tuple workaround.

## 2026-10-07 foundation evidence

- Added MaxPool1dConfiguration with six strict settings, canonical nullable stride, bounded channels/length, correct signed-floor and ceiling correction. Optional indices layout shares value geometry but does not claim measured argmax locations.
- MissingAPI RED captured in .local/maxpool1d-config-red.log. GREEN8Coretests (4pooling/4existingConv1d), no failures/skips: .local/maxpool1d-config-green.log. Covers partial windows, right-padding correction, invalid types/keys/bounds, channel preservation and index-layout presence.
- Read-only installedTorch2.14.1 probes confirmed L4/K3/S2/ceil→2 outputs; L1/K2/S2/P1/ceil→1 output; L1/K2/S1/no padding→invalid; dilated L2/K2/D2/P1→2 outputs. These probes guide tests, not full backend acceptance.
- No public component or worker implementation yet. Last accepted7ae5d69 candidate remains quit. Next: shared typed output-routing contract plus worker pooling geometry/runtime tests.


## 2026-10-07 — Worker pooling geometry and output-routing audit

- Added allocation-free worker normalize_maxpool1d/resolve_maxpool1d mirroring the native six-setting contract and nullable stride. Exact keys, Boolean/numeric distinctions, app bounds, floor/ceil/right-padding rules and optional indices geometry checked. Not registered as an executable component yet.
- Missingmodule RED (.local/maxpool1d-worker-red.log) →70Pythonchecks passed with Conv1d regression (.local/maxpool1d-worker-green.log), no skips. One test exhaustively compares2,112small configurations to actualTorch2.14.1 output geometry or failure; includes dilation gaps and negative numerators. Only pre-existing even-kernel convolution warning.
- Detailed routing audit: engine.validate and GraphModel.incoming discard sourcePort; native GraphValidator.contract reads tensorContracts[sourceNode]; export_source aliases tensors by node; ProductExecutionAdapter.validate_stage and preview activation/cache serialization assume one Tensor per node. Therefore updating only forward would leave validation and cached inspection incorrect.
- Next contract: carry an explicit node/port reference for edges; per-port typed layouts and values are authoritative. Keep existing node-level out views for old single-output callers and old archive compatibility, without treating an arbitrary missing port as out. Validate and export each selected output; add typed indices selection and finite-float checks without pretending indices carry gradients. Version new multi-output preview-cache payloads and preserve reading old single-output caches. Cover old graphs first, then MaxPool1d two-output execution and native contracts. No publication or capability coverage credit yet.


## 2026-10-07 — Native output-port tensor authority

- ValidationResult now owns typed tensors by Hashable GraphPortAddress(nodeID,portID). Legacy tensorContracts/tensorLayouts remain default-out projections; changing an out contract preserves independent named outputs. GraphValidator writes directly to the authoritative map and reads each incoming sourcePort, refusing unresolved named outputs instead of substituting out. Canvas tensor presentation likewise follows exact output addresses. No serialized graph/schema change.
- Four tests cover distinct float/indices contracts, compatibility mutation, absent-port refusal, exact input presentation and existing graph output publication. MissingAPI RED .local/tensor-output-routing-red.log (corrected missing try in test fixture before implementation). Broader GREEN exposed a stale GraphTests expected registry list missing nine already delivered layers; updated the explicit list, without changing registry/product scope.
- Final87Swiftchecks passed (70Core/17App), zero failures/skips: .local/tensor-output-routing-final.log. Includes graph families, layouts, model-origin presentation, composite derived bindings and workflow/canvas regressions. Earlier stale-list failure retained in .local/tensor-output-routing-green.log.
- Native typed authority prerequisite implemented; worker per-port values/export/cache and actual MaxPool component remain next. No new packaged candidate/native acceptance. Last accepted7ae5d69unchanged/quit; all test handles terminal.


## 2026-10-07 — Runner explicit output addressing

- Added immutable OutputAddress(node,port) and shared select_output for tensor/layout/type/code-local selection. A single value represents only out; named outputs require explicit mappings and missing outputs refuse. Selection preserves object identity, dtype and autograd. Engine validation/incoming edges now retain sourcePort; runtime and generated-code input lookups use the shared selector. Existing state keys and exported local naming remain unchanged.
- Missingmodule RED .local/runner-output-routing-red.log. Fixed the new test's registry fixture to use the engine's existing registry-file interface before executing tests. First broader run108passed/1failed due to pre-existing test demanding the word Linear in the general trainable-parameter refusal; updated only that stale assertion.
- Final154Pythonchecks passed, no skips (.local/runner-output-routing-final.log): explicit named-output refusal, source address retention, actual forward/gradient/export equivalence, conv/lazy graph behavior, CLI/training, ordinary and embedding preview/resume and product tensor validation. Three expected framework warnings.
- This closes default-flow migration to explicit addresses, not multiple-output execution acceptance. Next: per-output runtime validation and versioned preview/cache payloads, then MaxPool registration with indices and native contracts. Candidate7ae5d69unchanged; fresh runtime payload will be required after worker additions. Test processes terminal; production gates unchanged.


## 2026-10-07 — Per-output runtime validation and preview cache

- ProductExecutionAdapter validates exact named-output sets and each tensor's declared shape/dtype/device/row count/finite values. Preserves original tensors and autograd; unknown/missing/extra ports, mismatched contract keys, tuple substitution, sparse outputs and nested arbitrary mappings refuse. Ordinary tensor stages retain their existing contract.
- Eleven failing multi-output validation tests reproduced AttributeError before implementation (.local/multi-output-validation-red.log). GREEN59Pythonchecks (.local/multi-output-validation-green.log), no skips, including existing product/preview/embedding paths.
- Preview uses schema3 envelopes only when model layouts declare named outputs; old ordinary schema1 and forward-mutating embedding schema2 remain unchanged. Each output is copied to CPU, summarized separately and validated before cache reuse. Schema3 retains the activations/mutatedWeights envelope, including an empty weights map when unused.
- Two cache tests first failed (.local/multi-output-preview-red.log). Real torch.sort in a test-only GraphModel adapter supplies values/indices to test infrastructure; this is not a registered component or MaxPool runtime claim. A subsequent GREEN attempt exposed the remaining single-tensor detach on reuse; corrected to per-output transfer. Final65Pythonchecks passed (.local/multi-output-preview-final.log), no skips: resume matches every output summary, exact executed suffix, int64 indices persist, and a mistyped tensor with recomputed integrity hash still refuses. Existing preparation/cancellation/embedding cache regressions pass. Two expected framework warnings.
- Next: MaxPool registration, executable per-output layout/type publication and exported tuple assignment; native preview consumers must render named outputs before public acceptance. Current7ae5d69candidate unchanged/quit. All handles terminal; production gates remain open.


## 2026-10-07 — MaxPool1d worker execution/export/training

- Added schema2 MaxPool1d semantic normalization, shape/type validation and runtime execution behind a private test registry. Explicit return_indices emits out(float32)/indices(int64); disabled indices cannot be wired and integer indices cannot silently enter float-only layers. Runtime checks the exact authored sample shape. No public registry entry yet.
- Export emits torch.nn.functional.max_pool1d with all six authored settings and collision-safe named tuple locals when indices are enabled. Downstream aliases resolve the selected output. Parameterless pooling adds no checkpoint weights.
- Tests-first4failures in .local/maxpool1d-graph-red.log →32passed initial graph/geometry/routing/export checks. Independent explicit-window oracle verifies negative inputs, overlapping gradient accumulation, first-winner ties and actual indices. Raw CPUfloat64 graph/export outputs and input gradients match exactly. Actual pool preview/cache resume retains both outputs.
- Training test then exposed final capture's tensor-only shape access (.local/maxpool1d-training-red.log:1failed/4passed). Shared map_stage/summarize_stage now serves inference and training, including activation-value capture policy for every output.
- Final60Pythonchecks passed without warnings/skips (.local/maxpool1d-graph-final.log), including actual3epoch CPU training vs1+resume3 with exact model/optimizer/RNG equality, omitted values AND indices under capture policy, existing convolution lifecycle and capture/privacy regressions. No MPS/full native acceptance claim yet.
- Next: public/native registry/typed validation and discovery, named-output inspection, MPS/shape/export boundary checks, example/Learn and packaged native run/export/restart. Candidate7ae5d69unchanged/quit. All processes terminal; broad production gates unchanged.


## 2026-10-07 — Native pooling contract and actual MPS checks

- Added six-field MaxPool1dComponent (including nullable stride and optional indices), reviewed bridge declaration and strict schema2 configuration migration. Native validation produces separate float32 values/int64 indices; disabled indices have no tensor contract, and indices wired to float-only operations refuse without aliasing values. Component is tested through an appended private registry, not yet exposed in shipped registry/ribbon.
- MissingAPI RED .local/maxpool1d-native-red.log →70Corechecks passed, zero failures/skips (.local/maxpool1d-native-green.log), including pooling geometry/graph/bridge, explicit output routing and existing graph-family regressions.
- Actual MPS test runs for both return-indices modes compare full-model forward/input gradients to CPU and exact winning indices. Full focused worker graph suite8passed, no skips/warnings (.local/maxpool1d-mps-green.log); includes previous CPU training/resume/privacy. This is measured device support for those fixtures, not full supported-hardware acceptance.
- Next: runtime discovery, shipped registry parity, disabled-output presentation and named-output inspection before public ribbon availability. Preserve independent shape/type contracts across preview and History; symbolic capability declarations alone do not mean a disabled output is connectable. Example/Learn/export boundary/native acceptance remain pending. Last accepted7ae5d69candidate unchanged/quit; all verification handles terminal.


## 2026-10-07 — Native named training-capture validation

- Found native WorkflowModelTrainingExecution.verifyPresentation rejected the worker's named-output envelope. It now validates the exact out/indices set for reviewed MaxPool1d return-indices mode, matching shapes, float32 values and int64 indices, with existing bounded samples and capture-status policy. Disabled indices cannot submit a named envelope. Flat outputs retain their existing presentation contract; integer samples now require exact JSON integers rather than accepting fractional numbers.
- Corrected explicit JSONValue string construction in the test fixture, then observed2genuine failing acceptance assertions across3tests (.local/named-tensor-capture-red.log). Final15Appchecks passed, zero failures/skips (.local/named-tensor-capture-green.log), including training tool/result and capture regressions. Tests cover exact integer9007199254740993, missing/extra/wrong-type/fractional/disabled output refusals and omission policy across both outputs.
- This is decoding/verification, not UI display acceptance. Named-output controls/rendering, runtime discovery/public registry, examples and packaged acceptance remain next. Last accepted7ae5d69candidate unchanged/quit. All handles terminal.


## 2026-10-07 — Named tensor display foundation

- Added shared named-output formatting and SwiftUI preview, preserving exact int64 samples and inherited capture policy. Legacy workspace node inspection displays each captured output; edge inspection resolves its exact source port and typed contract, with no fallback to another output.
- Missing-API RED in .local/named-tensor-display-red.log. An intermediate edit accidentally removed trailing workspace view declarations; restored them before verification. Final focused build/tests passed 7 App checks with zero failures in .local/named-tensor-display-green.log. This verifies formatting/capture contracts and compilation, not native visual acceptance.
- Audit found the universal workflow bottom Results dock still exposes preparation-data snapshots, not model activation captures. Next: connect verified model captures with origin and stale/current provenance there; do not claim legacy workspace rendering closes universal canvas inspection. Public registry/discovery, disabled-port presentation, example/Learn and packaged acceptance remain open. Last accepted candidate 7ae5d69 remains unchanged and quit.


## 2026-10-07 — Universal Results model-capture path

- Added immutable WorkflowVerifiedModelCaptures: verifies the existing bounded training presentation contract before retaining typed samples, requires exact graph/origin coverage, and resolves full instance-path/local-node identities without aliases. Named ports inherit capture policy and preserve exact integers.
- Execution publishes model captures only after completed training artifacts and live predictions verify. Legacy receipts without samples remain valid but publish none. Kept model captures separate from preparation data. The active workspace supplies run/request/revision/binding provenance with project/session/retirement checks and completed-phase gating.
- Universal bottom Results now renders captured top-level model nodes and exact source-port connections, displays stale canvas/binding warnings, and states current external files are not revalidated. Uses authored component names and the shared tensor renderer. Selection of an unavailable named output does not fall back to out.
- MissingAPI RED .local/model-capture-routing-red.log; final28Apptests,27passed/1explicit benchmark skip/0failures (.local/model-capture-routing-final.log). Includes real preparation-to-training execution with per-node verified capture assertions, origin isolation, exact integers, omission policy, formatting, session and existing data-output regressions. The skipped near-limit source archive benchmark is unrelated to this change.
- Remaining: Block boundary/child display integration, saved History activation replay, disabled-output presentation, discovery/public registry, example/Learn and packaged native acceptance. No visual acceptance claimed from compilation or tests. Last accepted candidate7ae5d69 unchanged/quit. All test handles terminal.


## 2026-10-07 — Block output capture routing and child Results

- Capture routing now retains the immutable captured composite source and follows renamed/nested output boundaries to exact primitive output ports. It uses captured topology, full instance paths, bounded recursion and unique definition/instance matches; missing or disabled outputs are absent, never substituted. Workflow block node/edge inspection uses this resolver.
- Workflow-owned child windows receive a guarded capture provider tied to the active parent session/root. Added a resizable bottom Results readout for selected child components/connections. Draft/stale warnings distinguish captured applied-model values from edited proposals. Retired/replaced child owners cannot retrieve captures. Legacy model windows do not claim this workflow provider.
- Missing-API RED .local/block-capture-routing-red.log. Initial UI compilation exposed private selection access; added a read-only filtered edge-selection projection. Final30Appchecks:29passed/1unrelated source-archive benchmark skip/0failures (.local/block-capture-routing-final.log). Covers renamed nested indices/value routing, missing paths/ports, exact integers, actual training capture, child lifetime and session regressions. All handles terminal.
- This implements scoped rendering/routing but does not close native visual acceptance. Next: saved History activation artifact/replay, runtime discovery/public registry and disabled-port display, then example/Learn and packaged pooling acceptance. Last accepted7ae5d69 candidate remains unchanged/quit; production gates remain open.


## 2026-10-07 — Saved activation artifacts and History replay

- Worker now atomically stores bounded activations.json before the completion manifest, with run/graph identity, first-validation-row sampling policy, exact typed per-port previews and accepted capture policy. Manifest pins its bytes/SHA256. Omitted values/indices remain omitted in durable output. Added2MB to worker/native storage reservations and archive allowlist.
- Native independent decoder verifies file identity, schema, exact keys, hash/size, run/graph/policy and bounded tensor contracts. Extracted shared activation validation from the existing presentation verifier. Both workflow and captured training paths compare live previews to the verified saved copy. Captured-tool policy is parsed before artifact verification and passed explicitly.
- History reader retains verified samples; History UI exposes expandable node/port readouts. Old runs without the artifact explicitly report no saved samples. No runtime or active editor is needed to decode archived samples. This is archive replay, not a claim of selectable historical overlays on the current canvas.
- Tests-first Python2failures (.local/activation-artifact-red.log) and native missingAPI RED (.local/activation-artifact-native-red.log). Worker8poolingchecks plus57capture/storage/lifecycle regressions passed (.local/activation-artifact-worker-green.log and -worker-regression.log), zero skips. Native initial runs exposed stale exact-capacity fixtures after the added2MB allowance; corrected expected bounds and retained failed logs. Final42Appchecks:41passed/1unrelated archive benchmark skip/0failures (.local/activation-artifact-native-final2.log). Includes actual training, History reopening and corruption refusal, policy-leak refusal, storage phase transitions, hash/run/disabled-output refusal and captured training scenarios.
- Next: MaxPool1d runtime discovery/public registry, disabled-output presentation, example/Learn, export boundary audit and packaged native acceptance. Last accepted7ae5d69 candidate unchanged/quit. All handles terminal; production gates unchanged.


## 2026-10-07 — Public pooling registry, runtime discovery and configured outputs

- Registered the reviewed MaxPool1d definition in the shared registry. Native and Python graph tests now use the actual shipped registry instead of appended private fixtures. Added exact runtime discovery for bounded windows/geometry, nullable stride, floor/ceil modes and float32/int64 named outputs; native admission rejects missing/altered descriptors and clears stale support on runtime changes.
- Added configured output availability shared by workflow projection and graph connection admission. With indices disabled, the universal canvas omits that unconnected port. If a saved wire references it, existing projection retains the wire with an unresolved endpoint for explicit repair. Legacy/child port descriptions mark disabled outputs unknown and connection admission refuses them; registry declarations still enumerate supported modes.
- Tests-first Python discovery failure (.local/maxpool-discovery-worker-red.log), native missingAPI discovery/ports RED (.local/maxpool-discovery-native-red.log, .local/maxpool-ports-red.log). Discovery/pooling Python9passed, no skips (.local/maxpool-discovery-worker-green.log). Native discovery/graph initial69passed; broader configured-port run97checks:96passed/1opt-in benchmark skip/0failures (.local/maxpool-ports-green.log). Includes real worker probing, registry parity, descriptor mutation/refusal/reset, graph/command/canvas regressions. Additional workflow projection test verifies disabled→enabled and retained broken wire transitions (.local/maxpool-port-projection-final.log).
- Next: example/Learn, explicit export sample-shape boundary audit, themed component presentation and packaged native pooling acceptance. No native visual acceptance claimed yet. Current candidate7ae5d69unchanged/quit; verification handles terminal. Public registry source changes require a freshly built runtime/candidate before UAT.


## 2026-10-07 — Pooling example, explanation and export boundary

- Added a prepared MaxPool1d classifier and Learn entry: two measurements→14 learned features→[N,2,7]→pool[2,4]→ReLU→flatten8→two class scores;60parameters, optional indices enabled by default. Same source binding/preparation/train launch path as other bundled examples. Registry capability links the example. Added a teal pooling glyph/purpose/short label within the current vector icon system; owner ART remains separate.
- Learning text explains all six settings, signed floor/ceiling/right-padding geometry, negative-infinity padding, indexed-window numerical example, nondifferentiable indices/ties, sparse selection y=Ax and gradient Aᵀg, overlap accumulation, information loss and all-padding dilation diagnostics. It explicitly labels feature positions as constructed tabular features. This is explanatory material and a runnable exercise, not a scored assessment or guided curriculum completion claim.
- Export RED exposed accepting an empty batch (.local/maxpool-export-boundary-red.log). Generated MaxPool1d now enforces the same exact authored sample/nonempty batch as runtime;9Python graph/export/CPU/MPS/training checks passed (.local/maxpool-export-boundary-green.log).
- MissingAPI example RED (.local/maxpool-example-red.log). Initial UI compile used an unavailable theme member; replaced with the established Divider. Final15Swiftchecks passed without skips (.local/maxpool-example-final.log): example geometry/both indices modes, layout separation, workflow compilation/roundtrip, configuration/graph and actual discovery.
- Actual bundled-example integration first exposed the helper fixture's missing flat-model classification for the new example; corrected that expectation. Final1integrationtest passed without skips (.local/maxpool-example-integration-final.log): code-before-run, real3epoch preparation/training, verified History, matching live/saved int64 indices, binding/construction staleness and retired-host refusal.
- Next: refresh packaged runtime, build candidate, then native controls/connector toggle, Results/History, code/export/inference and reopen walkthrough. Full pooling/native acceptance still open; last accepted7ae5d69 remains quit. All test handles terminal.


## 2026-10-07 — Packaged pooling run/export and native capture-label finding

- Built candidate e2e84bb with refreshed .local/maxpool1d-runtime-20261007,79current worker files verified. Initial refresh hit an existing copied runner directory; replaced only that new build-owned copy. Initial package verification also rejected watchdog bytes built with a different selected developer toolchain; rebuilt the payload watchdog with the same Xcode DEVELOPER_DIR, refreshed integrity and passed current-source/signature checks. Package log .local/maxpool-package.log.
- Native1440×900: Learn expansion settled with full pooling explanation readable; opened example, inspected all six controls, toggled indices off/on and observed canvas port labels in/out versus in/out/indices. Code before training rendered syntax highlighting, named tuple outputs and authored-shape guard.
- Actual native run0F05692C-7B12-4B1C-AF85-5373DDF5AAAE, requestb324dbbaddb0331d1d208c018fd63068b6ecfc75c45ff19e1065346151869ce8, source revision2, started2026-10-07T13:50:52.937Z, finished13:50:58.202Z. Verified History:3epochs,30artifacts,6computed/0reused. Bottom Results showed indices[8,2,4]int64 and out[8,2,4]float32; selecting values wire showed only out. History expansion matched both saved samples.
- Exported through native review/save to .local/KwipML-model-0F05692C. Standalone inference.py --example --batch-size2 exited0 with logits[[0.1546837687,0.1519083679],[-0.2325543463,0.7007339597]],classIndices[0,1]. This is synthetic prepared-input inference, not raw preprocessing export.
- Native finding: canvas nodes/wires still said Not evaluated while Results showed verified model captures. Root cause was data-only status admission/provider dispatch. Added actual-training status/staleness assertions: RED2failures (.local/maxpool-capture-label-red.log), then model-qualified snapshot dispatch and canvas acceptance. GREEN6Appchecks passed,no skips (.local/maxpool-capture-label-green.log), including existing data identity/lifetime regressions. Native replay of the fix remains required.
- Candidate quit; process absence verified. No pooling acceptance closure yet. Next rebuild corrected candidate, replay Captured/Stale statuses and project/History reopen, plus remaining field/Block inspection checks. Last fully accepted scoped candidate remains7ae5d69.


## 2026-10-07 — Pooling native replay and child capture status

- Packaged e67e698 passed root Captured/Stale labels after actual pooling run 1153C0B0-0B3B-4867-B6E3-066BDBE54844. Restarted History retained run 0F05692C-7B12-4B1C-AF85-5373DDF5AAAE and exact values/indices. Nullable stride off/on restored the default and accepted an explicit 2. Ceiling-mode edits marked results stale. Padding 2 with kernel 3 was refused before execution; restoring padding 1 allowed run A990D785-EBE2-47A3-9E62-85E137AD8C7F. Diagnostics repeated the cause with downstream contract failures; broader diagnostics polish remains open.
- Actual separable Block run 5AC85E78-1ECB-48EE-BDEB-2AFD602BAC04 exposed the verified boundary [8,4,4,5] tensor. Child bottom Results distinguished depthwise [8,6,4,5] from pointwise [8,4,4,5], with an applied-model warning for the editing draft. Native inspection found child canvas labels still hard-coded to Not evaluated.
- Moved the guarded capture provider into ModelCanvasAdapter so labels and bottom Results use the same project/session/node-qualified snapshot. Drafts report Stale; invalidated editors clear the provider. Tests-first missing-API RED (.local/child-capture-status-red.log); final 14 App checks passed, zero failures (.local/child-capture-status-green.log). Packaged child-label replay still required. Candidate quit before edits/tests; no scoped closure or broad production acceptance claimed.


## 2026-10-07 — MaxPool1d scoped acceptance

- Candidate 3d86660 packaged successfully (.local/maxpool-child-status-package.log). Actual native separable regression run EEEA2FBC-F223-43A0-85A3-B4EE25EA4A36 completed. Parent boundary was Captured; opened draft children and their connection correctly showed Stale. Depthwise [8,6,4,5] and pointwise [8,4,4,5] samples matched their respective earlier runs. Screenshot inspection confirmed the flat child canvas and independent bottom Results panel. Discarded unchanged draft, quit candidate, verified process absence.
- Acceptance audit added explicit noncontiguous pooling values/indices/gradient parity and valid all-padding dilated-window cases, in both indices modes. The latter confirms raw PyTorch negative infinity while product capture refuses it with component-qualified finite-value guidance. These are evidence tests for existing behavior, not a claimed failing-test implementation cycle. Final pooling contract/graph/discovery run: 33 passed, zero skips (.local/maxpool-closure-audit.log).
- Acceptance mapping: (1) strict settings/geometry: native configuration tests and exhaustive worker geometry; (2) exact typed outputs: graph/command/output-routing, named capture, composite boundary and export tests; (3) numerical behavior: independent indexed oracle, tie/overlap gradients, real CPU/MPS, noncontiguous and all-padding checks; (4) lifecycle: exact CPU checkpoint resume, preview cache, native training/History restart, reviewed export and standalone inference; (5) product: six editable controls, configured connectors, code-before-run, bundled example and math explanation, native readout and label replay.
- Scoped component delivery closes. This does not close TORCH-01, any broad production gate, owner ART, or a guided/scored pooling course. Next inventory unit: MaxPool2d, reusing per-output infrastructure and shared geometry where appropriate. Keep one active workstream.
